#ifndef RPCStripNoises_h
#define RPCStripNoises_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include<vector>
#include<map>
#include<iostream>
#include<boost/cstdint.hpp>

class RPCStripNoises {

 public:

  struct NoiseItem {
    int dpid;
    float noise;
    float eff;
    float time;
  
  COND_SERIALIZABLE;
};
  
  RPCStripNoises(){}
  ~RPCStripNoises(){}
  
  std::vector<NoiseItem>  const & getVNoise() const {return v_noises;}
  std::vector<float>  const & getCls() const {return v_cls;}

  std::vector<NoiseItem>  v_noises; 
  std::vector<float>  v_cls; 

 COND_SERIALIZABLE;
};

#endif
