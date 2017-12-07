// Sushil Dubey, Shashi Dugad, TIFR, December 2017
#ifndef CablingMap_H
#define CablingMap_H

struct CablingMap{
  unsigned int size;
  unsigned int *fed;
  unsigned int *link;
  unsigned int *roc;
  unsigned int *RawId;
  unsigned int *rocInDet;
  unsigned int *moduleId;
  bool *modToUnp;
  bool *badRocs;
};

#endif
