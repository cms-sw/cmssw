#include "../interface/PatternTree.h"

PatternTree::PatternTree(){
}

PatternTree::~PatternTree(){
  for(map<string, PatternTrunk*>::iterator itr = patterns.begin(); itr != patterns.end(); ++itr){
    delete (itr->second);
  }
}

void PatternTree::addPattern(Pattern* ldp, Pattern* fdp){
  string key = ldp->getKey();
  map<string, PatternTrunk*>::iterator it = patterns.find(key);
  if(it==patterns.end()){//not found
    PatternTrunk* pt = new PatternTrunk(ldp);
    pt->addFDPattern(fdp);
    patterns[key]=pt;
  }
  else{
    (it->second)->addFDPattern(fdp);
  }
}

void PatternTree::addPattern(Pattern* ldp, Pattern* fdp, float new_pt){
  string key = ldp->getKey();
  map<string, PatternTrunk*>::iterator it = patterns.find(key);
  if(it==patterns.end()){//not found
    PatternTrunk* pt = new PatternTrunk(ldp);
    pt->addFDPattern(fdp, new_pt);
    patterns[key]=pt;
  }
  else{
    (it->second)->addFDPattern(fdp, new_pt);
  }
}

vector<GradedPattern*> PatternTree::getFDPatterns(){
  vector<GradedPattern*> res;
  for(map<string, PatternTrunk*>::iterator itr = patterns.begin(); itr != patterns.end(); ++itr){
    vector<GradedPattern*> fdp = itr->second->getFDPatterns();
    res.insert(res.end(), fdp.begin(), fdp.end());
  }
  return res;
}

vector<GradedPattern*> PatternTree::getLDPatterns(){
  vector<GradedPattern*> res;
  for(map<string, PatternTrunk*>::iterator itr = patterns.begin(); itr != patterns.end(); ++itr){
    GradedPattern* ldp = itr->second->getLDPattern();
    res.push_back(ldp);
  }
  return res;
}

vector<int> PatternTree::getPTHisto(){
  vector<int> h;
  for(int i=0;i<150;i++){
    h.push_back(0);
  }
  for(map<string, PatternTrunk*>::iterator itr = patterns.begin(); itr != patterns.end(); ++itr){
    float pt = itr->second->getLDPatternPT();
    if(pt>149)
      pt=149;
    if(pt<0)
      pt=0;
    h[(int)pt]=h[(int)pt]+1;
  }
  return h;
}

int PatternTree::getFDPatternNumber(){
  vector<GradedPattern*> res;
  int num=0;
  for(map<string, PatternTrunk*>::iterator itr = patterns.begin(); itr != patterns.end(); ++itr){
    num += itr->second->getFDPatternNumber();
  }
  return num;
}

int PatternTree::getLDPatternNumber(){
  return patterns.size();
}

void PatternTree::computeAdaptativePatterns(short r){
  for(map<string, PatternTrunk*>::iterator itr = patterns.begin(); itr != patterns.end(); ++itr){
    itr->second->computeAdaptativePattern(r);
  }
}

void PatternTree::link(Detector& d, const vector< vector<int> >& sec, const vector<map<int, vector<int> > >& modules){
  for(map<string, PatternTrunk*>::iterator itr = patterns.begin(); itr != patterns.end(); ++itr){
    itr->second->link(d,sec, modules);
  }
}

void PatternTree::getActivePatterns(int active_threshold, vector<GradedPattern*>& active_patterns){
  for(map<string, PatternTrunk*>::iterator itr = patterns.begin(); itr != patterns.end(); ++itr){
    GradedPattern* p = itr->second->getActivePattern(active_threshold);
    if(p!=NULL)
      active_patterns.push_back(p);
  }
}

void PatternTree::addPatternsFromTree(PatternTree* p){
  vector<GradedPattern*> ld = p->getLDPatterns();
  for(unsigned int i=0;i<ld.size();i++){
    GradedPattern* patt = ld[i];

    addPatternForMerging(patt);

    delete patt;
  }
}


void PatternTree::addPatternForMerging(GradedPattern* ldp){
  string key = ldp->getKey();
  map<string, PatternTrunk*>::iterator it = patterns.find(key);
  if(it==patterns.end()){//not found
    PatternTrunk* pt = new PatternTrunk(ldp);
    for(int i=0;i<ldp->getGrade();i++){
      pt->addFDPattern(NULL, ldp->getAveragePt());
    }
    patterns[key]=pt;
  }
  else{
    (it->second)->updateDCBits(ldp);
     for(int i=0;i<ldp->getGrade();i++){
       (it->second)->addFDPattern(NULL, ldp->getAveragePt());
     }
  }
}

bool PatternTree::checkPattern(Pattern* lp, Pattern* hp){
  if(lp==NULL || hp==NULL)
    return false;
  string key = lp->getKey();
  map<string, PatternTrunk*>::iterator it = patterns.find(key);
  if(it==patterns.end())//not found
    return false;
  else{
    return (it->second)->checkPattern(hp);
  }
}
