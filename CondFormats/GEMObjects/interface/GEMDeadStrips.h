#ifndef CondFormats_GEMObjects_GEMDeadStrips_h
#define CondFormats_GEMObjects_GEMDeadStrips_h

#include "CondFormats/Serialization/interface/Serializable.h"
#include <vector>
#include <iostream>

class GEMDeadStrips {

 public:
  struct DeadItem {
    int rawId;
    int strip;  
    COND_SERIALIZABLE;
  };
  
  GEMDeadStrips(){}
  ~GEMDeadStrips(){}

  std::vector<DeadItem> const & getDeadVec() const {return deadVec_;}

 private:
  std::vector<DeadItem> deadVec_;

  COND_SERIALIZABLE;
};
#endif
