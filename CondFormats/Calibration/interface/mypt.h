#ifndef MYPT_H
#define MYPT_H
#include "CondFormats/Common/interface/Serializable.h"

#include "CondFormats/Calibration/interface/fixedArray.h"
class mypt{
 public:
  mypt(){}
  void fill();
  fixedArray<unsigned short,2097> pt;

 COND_SERIALIZABLE;
};
#endif
