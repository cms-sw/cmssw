#include "../interface/Layer.h"

Layer::Layer(int nbLad, int nbMod, int segmentSize, int sstripSize){
  for(int i=0;i<nbLad;i++){
    Ladder* l = new Ladder(nbMod, segmentSize, sstripSize);
    ladders.push_back(l);
  }
}

Layer::~Layer(){
  for(unsigned int i=0;i<ladders.size();i++){
    delete ladders[i];
  }
  ladders.clear();
}

Ladder* Layer::getLadder(int pos){
  if(pos>=0 && pos<(int)ladders.size())
    return ladders[pos];//ladders numbered from 0 to n-1
  cout<<"ladder "<<pos<<" not found in layer!"<<endl;
  return NULL;
}

void Layer::clear(){
  for(unsigned int i=0;i<ladders.size();i++){
    ladders[i]->clear();
  }
}
