#ifndef GEMDeadStrips_h
#define GEMDeadStrips_h

#include "CondFormats/Serialization/interface/Serializable.h"
#include<vector>
#include<iostream>
#include<boost/cstdint.hpp>

class GEMDeadStrips {

 public:
  struct DeadItem {
    int rawId;
    int strip;  
    COND_SERIALIZABLE;
  };
  
  GEMDeadStrips(){}
  ~GEMDeadStrips(){}

  std::vector<DeadItem> const & getDeadVec() const {return DeadVec;}
  std::vector<DeadItem> DeadVec;

  COND_SERIALIZABLE;
};
#endif
