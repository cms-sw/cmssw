#ifndef RPCNoiseObject_h
#define RPCNoiseObject_h

#include<vector>
#include<map>
#include<iostream>
#include<boost/cstdint.hpp>

class RPCNoiseObject {

 public:

  int version;
  int run;

  //structure suitable for cluster size
  struct NoiseObjectItem {
    int dpid;
    float deadStrips;
    float maskedStrips;
    float stripsToUnmask;
    float stripsToMask;
    float rate;
    float weight;
  };
  
  RPCNoiseObject(){}
  ~RPCNoiseObject(){}
 
  RPCNoiseObject* Fake_RPCNoiseObject();
 
  std::vector<NoiseObjectItem>  const & getCls() const {return v_cls;}
  std::vector<NoiseObjectItem>  v_cls; 

 private:

};

#endif  //RPCNoiseObject_h
