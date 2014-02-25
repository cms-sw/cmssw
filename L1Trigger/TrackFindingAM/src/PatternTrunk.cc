#include "../interface/PatternTrunk.h"

PatternTrunk::PatternTrunk(Pattern* p){
  lowDefPattern = new GradedPattern(*p);
}

PatternTrunk::PatternTrunk(){
  lowDefPattern = new GradedPattern();
}

PatternTrunk::~PatternTrunk(){
  for(map<string, GradedPattern*>::iterator itr = fullDefPatterns.begin(); itr != fullDefPatterns.end(); ++itr){
    delete itr->second;
  }
  delete lowDefPattern;
}

void PatternTrunk::addFDPattern(Pattern* p){
  lowDefPattern->increment();
  if(p!=NULL){
    string key=p->getKey();
    map<string, GradedPattern*>::iterator it = fullDefPatterns.find(key);
    if(it==fullDefPatterns.end()){//not found
      GradedPattern* gp = new GradedPattern(*p);
      gp->increment();
      fullDefPatterns[key]=gp;
    }
    else{
      (it->second)->increment();
    }
  }
}

void PatternTrunk::addFDPattern(Pattern* p, float pt){
  lowDefPattern->increment(pt);
  if(p!=NULL){
    string key=p->getKey();
    map<string, GradedPattern*>::iterator it = fullDefPatterns.find(key);
    if(it==fullDefPatterns.end()){//not found
      GradedPattern* gp = new GradedPattern(*p);
      gp->increment(pt);
      fullDefPatterns[key]=gp;
    }
    else{
      (it->second)->increment(pt);
    }
  }
}

vector<GradedPattern*> PatternTrunk::getFDPatterns(){
  vector<GradedPattern*> res;
  for(map<string, GradedPattern*>::iterator itr = fullDefPatterns.begin(); itr != fullDefPatterns.end(); ++itr){
    res.push_back(new GradedPattern(*(itr->second)));
  }
  return res;
}

GradedPattern* PatternTrunk::getLDPattern(){
  return new GradedPattern(*lowDefPattern);
}

float PatternTrunk::getLDPatternPT(){
  return lowDefPattern->getAveragePt();
}

int PatternTrunk::getFDPatternNumber(){
  return fullDefPatterns.size();
}

void PatternTrunk::link(Detector& d, const vector< vector<int> >& sec, const vector<map<int, vector<int> > >& modules){
  lowDefPattern->link(d, sec, modules);
}

GradedPattern* PatternTrunk::getActivePattern(int active_threshold){
  if(lowDefPattern->isActive(active_threshold)){
    return new GradedPattern(*lowDefPattern);
  }
  return NULL;
}

void PatternTrunk::deleteFDPatterns(){
  for(map<string, GradedPattern*>::iterator itr = fullDefPatterns.begin(); itr != fullDefPatterns.end(); ++itr){
    delete itr->second;
  }
  fullDefPatterns.clear();
}

/*
  This method is recursive. We know what to do with arrays with only two elements. 
  If the size is above 2, we split the array in 2 smaller arrays and call the method on both parts.
  As we are using gray code, it's a bit tricky to know if we need to put a 0 or a 1. If you are in
  the first half array -> 0 left branch and 1 right branch. If you are in the second half it's the opposite.
  The 'reverse' parameter is used to know in which half array we are.
 */
void PatternTrunk::computeDCBits(vector<int> &v, bool* values, int size, int reverse){

  
  if(size==2){
    if(values[0] && values[1]){
      v.push_back(2);
      return;
    }
    if(values[0]){
      v.push_back(reverse);
      return;
    }
    else{
      v.push_back(1-reverse);
      return;
    }
  }

  bool half1=0;
  bool half2=0;

  for(int i=0;i<size/2;i++){
    half1 |= values[i];
  }
  for(int i=size/2;i<size;i++){
    half2 |= values[i];
  }

  if(half1 && half2){
    vector<int> v1;
    vector<int> v2;
    computeDCBits(v1, values, size/2, 0);
    computeDCBits(v2, values+size/2, size/2, 1);
    v.push_back(2);
    for(unsigned int i=0;i<v1.size();i++){
      if(v1[i]==v2[i])
	v.push_back(v1[i]);
      else
	v.push_back(2);
    }
    return;
  }
  if(half1){
    vector<int> v1;
    computeDCBits(v1, values, size/2, 0);
    v.push_back(reverse);
    for(unsigned int i=0;i<v1.size();i++){
      v.push_back(v1[i]);
    }
    return;
  }
  if(half2){
    vector<int> v1;
    computeDCBits(v1, values+size/2, size/2, 1);
    v.push_back(1-reverse);
    for(unsigned int i=0;i<v1.size();i++){
      v.push_back(v1[i]);
    }
    return;
  }

}

void PatternTrunk::computeAdaptativePattern(short r){
  int size = (int)pow(2.0,r);
  bool strips[size];
  int nb_layers = lowDefPattern->getNbLayers();
  for(int i=0;i<nb_layers;i++){
    memset(strips,false,size*sizeof(bool));
    PatternLayer* pl = lowDefPattern->getLayerStrip(i);
    int ld_position = pl->getStrip();
    
    for(map<string, GradedPattern*>::iterator itr = fullDefPatterns.begin(); itr != fullDefPatterns.end(); ++itr){
      PatternLayer* fd_pl = itr->second->getLayerStrip(i);
      int index = fd_pl->getStrip()-size*ld_position;
      strips[index]=true;
    }
   
    vector<int> bits;
    computeDCBits(bits,strips,size,0);

    for(unsigned int j=0;j<bits.size();j++){
      pl->setDC(j,bits[j]);
    }
  }
  deleteFDPatterns();
}

void PatternTrunk::updateDCBits(GradedPattern* p){
  if(lowDefPattern->getKey().compare(p->getKey())==0){//the pattern layers are similar -> we update the DC bits
    int nb_layers = lowDefPattern->getNbLayers();
    int nb_dc1 = lowDefPattern->getLayerStrip(0)->getDCBitsNumber();
    int nb_dc2 = p->getLayerStrip(0)->getDCBitsNumber();
    int max_nb_dc;
    if(nb_dc1>nb_dc2)
      max_nb_dc = nb_dc1;
    else
      max_nb_dc = nb_dc2;
    if(max_nb_dc==0)
      return;
    int size = (int)pow(2.0,max_nb_dc);
    bool strips[size];

    for(int i=0;i<nb_layers;i++){
      memset(strips,false,size*sizeof(bool));
      PatternLayer* pl = lowDefPattern->getLayerStrip(i);

      vector<string> positions=pl->getPositionsFromDC();
      for(unsigned int j=0;j<positions.size();j++){
	for(int k=0;k<=max_nb_dc-nb_dc1;k++){
	  strips[PatternLayer::GRAY_POSITIONS[positions[j]]*(max_nb_dc-nb_dc1+1)+k]=true;
	}
      }
   
      positions.clear();
      PatternLayer* pl2 = p->getLayerStrip(i);
      positions=pl2->getPositionsFromDC();
      for(unsigned int j=0;j<positions.size();j++){
	for(int k=0;k<=max_nb_dc-nb_dc2;k++){
	  strips[PatternLayer::GRAY_POSITIONS[positions[j]]*(max_nb_dc-nb_dc2+1)+k]=true;
	}
      }
      
      vector<int> bits;
      computeDCBits(bits,strips,size,0);
      
      for(unsigned int j=0;j<bits.size();j++){
	pl->setDC(j,bits[j]);
      }
    }
  }
}

bool PatternTrunk::checkPattern(Pattern* hp){
  if(hp==NULL)
    return false;
  if(fullDefPatterns.size()!=0){
    string key=hp->getKey();
    map<string, GradedPattern*>::iterator it = fullDefPatterns.find(key);
    if(it==fullDefPatterns.end())//not found
      return false;
    else
      return true;
  }
  else{
    return lowDefPattern->contains(hp);
  }
}
