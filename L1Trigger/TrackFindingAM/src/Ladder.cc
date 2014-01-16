#include "../interface/Ladder.h"

Ladder::Ladder(int nbMod, int segmentSize, int sstripSize){
  for(int i=0;i<nbMod;i++){
    Module* m = new Module(segmentSize, sstripSize);
    modules.push_back(m);
  }
}

Ladder::~Ladder(){
  for(unsigned int i=0;i<modules.size();i++){
    delete modules[i];
  }
  modules.clear();
}

Module* Ladder::getModule(int zPos){
  if(zPos>-1 && zPos<(int)modules.size())
    return modules[zPos];
  return NULL;
}

void Ladder::clear(){
  for(unsigned int i=0;i<modules.size();i++){
    modules[i]->clear();
  }
}
