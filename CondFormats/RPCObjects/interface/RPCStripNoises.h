#ifndef RPCStripNoises_h
#define RPCStripNoises_h

#include<vector>
#include<map>
#include<iostream>
#include<boost/cstdint.hpp>


class RPCStripNoises {

 public:

  struct NoiseItem {
    int dpid;
    float noise[96];
    float eff[96];
    float time;
  };
  
  RPCStripNoises(const RPCStripNoises& );
  RPCStripNoises(){};
  ~RPCStripNoises(){};
  
  std::vector<NoiseItem>  getVNoise() const {return v_noises;}
  std::vector<double>  getCls() const {return v_cls;}

  std::vector<NoiseItem>  v_noises; 
  std::vector<double>  v_cls; 
};

#endif
