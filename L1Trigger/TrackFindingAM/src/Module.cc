#include "../interface/Module.h"

Module::Module(int segmentSize, int sstripSize){
  segments[0]=new Segment(segmentSize, sstripSize);
  segments[1]=new Segment(segmentSize, sstripSize);
}

Module::~Module(){
  delete segments[0];
  delete segments[1];
}

Segment* Module::getSegment(int n){
  if(n>-1 && n<2)
    return segments[n];
  return NULL;
}

void Module::clear(){
  segments[0]->clear();
  segments[1]->clear();
}
