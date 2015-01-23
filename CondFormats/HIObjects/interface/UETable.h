#ifndef __UETable_h__
#define __UETable_h__

#include "CondFormats/Serialization/interface/Serializable.h"
#include <vector>

class UETable{
 public:
  UETable(){};
  float get(int i){return values[i];}
  void push(float v){values.push_back(v);}
  std::vector<float> values;

  COND_SERIALIZABLE;
};

#endif
