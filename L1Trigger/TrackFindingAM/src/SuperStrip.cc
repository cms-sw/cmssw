#include "../interface/SuperStrip.h"

SuperStrip::SuperStrip(int s){
  hit=false;
  size=s;
}

SuperStrip::~SuperStrip(){
  clear();
}

short SuperStrip::getSize(){
  return size;
}

bool SuperStrip::isHit(){
  return hit;
}

vector<Hit*>& SuperStrip::getHits(){
  return hits;
}

void SuperStrip::clear(){
  hit=false;
  for(unsigned int i=0;i<hits.size();i++){
    delete(hits[i]);
  }
  hits.clear();
}

void SuperStrip::touch(const Hit* h){
  hit=true;
  Hit* copy = new Hit(*h);
  hits.push_back(copy);
}
