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

#ifdef IPNL_USE_CUDA
void PatternTrunk::linkCuda(patternBank* p, deviceDetector* d, int pattern_index, const vector< vector<int> >& sec, const vector<map<int, vector<int> > >& modules, vector<int> layers, unsigned int* cache){
  lowDefPattern->linkCuda(p,d,pattern_index, sec, modules, layers, cache);
}
#endif

GradedPattern* PatternTrunk::getActivePattern(int active_threshold){
  if(lowDefPattern->isActive(active_threshold)){
    return new GradedPattern(*lowDefPattern);
  }
  return NULL;
}

GradedPattern* PatternTrunk::getActivePatternUsingMissingHit(int max_nb_missing_hit, int active_threshold){
  if(lowDefPattern->isActiveUsingMissingHit(max_nb_missing_hit, active_threshold)){
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

void PatternTrunk::computeAdaptativePattern(short r){
  int nb_layers = lowDefPattern->getNbLayers();
  for(int i=0;i<nb_layers;i++){
    
    PatternLayer* pl = lowDefPattern->getLayerStrip(i);
    int last_bits=0;
    vector<int> bits(r,0);

    for(map<string, GradedPattern*>::iterator itr = fullDefPatterns.begin(); itr != fullDefPatterns.end(); ++itr){
      PatternLayer* fd_pl = itr->second->getLayerStrip(i);
      last_bits = fd_pl->getStripCode();
      if(itr==fullDefPatterns.begin()){//first pattern, we simply copy the last bits
	for(int j=0;j<r;j++){
	  bits[j]=((last_bits>>(r-j-1)))&(0x1);
	}
      }
      else{//this is not the first pattern: if we have a different bit we set the DC value to don't care
	for(int j=0;j<r;j++){
	  if(bits[j]!=((last_bits>>(r-j-1))&(0x1)))
	    bits[j]=2;
	}
      }
    }

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

    for(int i=0;i<nb_layers;i++){
      PatternLayer* pl = lowDefPattern->getLayerStrip(i);
      PatternLayer* pl2 = p->getLayerStrip(i);
      vector<int> bits(max_nb_dc,0);

      for(int j=max_nb_dc-1;j>=0;j--){
	char pat1=pl->getDC(j);
	char pat2=pl2->getDC(j);

	if(pat1==3)
	  bits[j]=pat2;
	else if(pat2==3)
	  bits[j]=pat1;
	else if(pat1==pat2)
	  bits[j]=pat1;
	else
	  bits[j]=2;
	
      }

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
