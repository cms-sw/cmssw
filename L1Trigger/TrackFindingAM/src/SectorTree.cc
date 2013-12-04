#include "../interface/SectorTree.h"

SectorTree::SectorTree(){
  srand ( time(NULL) );
  superStripSize=-1;
}

SectorTree::~SectorTree(){
  for(unsigned int i=0;i<sector_list.size();i++){
    delete sector_list[i];
  }
}

Sector* SectorTree::getSector(vector<int> ladders, vector<int> modules){
  pair<multimap<string,Sector*>::iterator,multimap<string,Sector*>::iterator> ret;
  multimap<string,Sector*>::iterator first;

  ostringstream oss;
  oss<<std::setfill('0');
  for(unsigned int j=0;j<ladders.size();j++){
    oss<<setw(2)<<ladders[j];
  }

  ret = sectors.equal_range(oss.str());

  if(ret.first==ret.second){//NOT FOUND
    return NULL;
  }

  first=ret.first;  
  while(first!=ret.second){
    Sector* test = first->second;
    
    bool found=false;
    for(unsigned int i=0;i<ladders.size();i++){
      vector<int> mods = test->getModules(i,ladders[i]);
      found = false;
      for(unsigned int j=0;j<mods.size();j++){
	if(modules[i]==mods[j]){
	  found=true;
	  break;
	}
      }
      if(!found)
	break;
    }
    if(found)//this one is ok-> we take it
      return test;
    first++;
  }
  //none of the selected sectors where ok for modules...
  return NULL;
}

Sector* SectorTree::getSector(const Hit& h){
  for(unsigned int i=0;i<sector_list.size();i++){
    if(sector_list[i]->contains(h))
      return sector_list[i];
  }
  return NULL;
}

void SectorTree::addSector(Sector s){
  Sector* ns = new Sector(s);
  sector_list.push_back(ns);
  vector<string> keys = ns->getKeys();
  for(unsigned int i=0;i<keys.size();i++){
    sectors.insert(pair<string, Sector*>(keys[i],ns));
  }
}

void SectorTree::updateSectorMap(){
  sectors.clear();
  for(unsigned int j=0;j<sector_list.size();j++){
    Sector* ns = sector_list[j];
    vector<string> keys = ns->getKeys();
    for(unsigned int i=0;i<keys.size();i++){
      sectors.insert(pair<string, Sector*>(keys[i],ns));
    }
  }
}

vector<Sector*> SectorTree::getAllSectors(){
  return sector_list;
}

int SectorTree::getLDPatternNumber(){
  int nb = 0;
  for(unsigned int i=0;i<sector_list.size();i++){
    nb+=sector_list[i]->getLDPatternNumber();
  }
  return nb;
}

int SectorTree::getFDPatternNumber(){
  int nb = 0;
  for(unsigned int i=0;i<sector_list.size();i++){
    nb+=sector_list[i]->getFDPatternNumber();
  }
  return nb;  
}

void SectorTree::computeAdaptativePatterns(short r){
  for(unsigned int i=0;i<sector_list.size();i++){
    Sector* s=sector_list[i];
    s->computeAdaptativePatterns(r);
  }
}

void SectorTree::link(Detector& d){
  for(unsigned int i=0;i<sector_list.size();i++){
    sector_list[i]->link(d);
  }
}

vector<Sector*> SectorTree::getActivePatternsPerSector(int active_threshold){
  vector<Sector*> list;
  for(unsigned int i=0;i<sector_list.size();i++){
    Sector* copy = new Sector(*sector_list[i]);
    vector<GradedPattern*> active_patterns = sector_list[i]->getActivePatterns(active_threshold);
    for(unsigned int j=0;j<active_patterns.size();j++){
      copy->getPatternTree()->addPattern(active_patterns[j], NULL);
      delete active_patterns[j];
    }
    list.push_back(copy);
  }
  return list;
}

int SectorTree::getSuperStripSize(){
  return superStripSize;
}

void SectorTree::setSuperStripSize(int s){
  if(s>0)
    superStripSize=s;
}

int SectorTree::getNbSectors(){
  return sector_list.size();
}
