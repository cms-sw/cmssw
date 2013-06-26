#ifndef RPCDeadStrips_h
#define RPCDeadStrips_h

#include<vector>
#include<iostream>
#include<boost/cstdint.hpp>


class RPCDeadStrips {

 public:

  struct DeadItem {
    int rawId;
    int strip;
  };
  
  RPCDeadStrips(){}

  ~RPCDeadStrips(){}

  std::vector<DeadItem> const & getDeadVec() const {return DeadVec;}

  std::vector<DeadItem> DeadVec;

};

#endif
