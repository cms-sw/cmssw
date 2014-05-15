#include "../interface/Segment.h"

Segment::Segment(int stripNumber, int sstripSize){
  int nbSStrips=stripNumber/sstripSize;
  sStripSize=sstripSize;
  for(int i=0;i<nbSStrips;i++){
    SuperStrip* ss = new SuperStrip(sstripSize);
    strips.push_back(ss);
  }
}

Segment::~Segment(){
  for(unsigned int i=0;i<strips.size();i++){
    delete strips[i];
  }
  strips.clear();
}

SuperStrip* Segment::getSuperStrip(int stripNumber){
  int sstripPosition = stripNumber/sStripSize;
  if(sstripPosition<(int)strips.size())
    return strips[sstripPosition];
  return NULL;
}

SuperStrip* Segment::getSuperStripFromIndex(int sstripNumber){
  if(sstripNumber<(int)strips.size())
    return strips[sstripNumber];
  return NULL;
}

void Segment::clear(){
  for(unsigned int i=0;i<strips.size();i++){
    strips[i]->clear();
  }
}
