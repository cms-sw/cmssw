#ifndef RPCDeadStrips_h
#define RPCDeadStrips_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include<vector>
#include<iostream>
#include<boost/cstdint.hpp>


class RPCDeadStrips {

 public:

  struct DeadItem {
    int rawId;
    int strip;
  
  COND_SERIALIZABLE;
};
  
  RPCDeadStrips(){}

  ~RPCDeadStrips(){}

  std::vector<DeadItem> const & getDeadVec() const {return DeadVec;}

  std::vector<DeadItem> DeadVec;


 COND_SERIALIZABLE;
};

#endif
