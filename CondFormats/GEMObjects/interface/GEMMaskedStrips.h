#ifndef GEMMaskedStrips_h
#define GEMMaskedStrips_h

#include "CondFormats/Serialization/interface/Serializable.h"
#include<vector>
#include<iostream>
#include<boost/cstdint.hpp>

class GEMMaskedStrips {

 public:
  struct MaskItem {
    int rawId;
    int strip;
    COND_SERIALIZABLE;
  };
  
  GEMMaskedStrips(){}
  ~GEMMaskedStrips(){}

  std::vector<MaskItem> const & getMaskVec() const {return MaskVec;}
  std::vector<MaskItem> MaskVec;

  COND_SERIALIZABLE;
};
#endif
