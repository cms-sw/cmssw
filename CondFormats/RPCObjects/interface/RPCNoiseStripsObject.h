#ifndef RPCNoiseStripsObject_h
#define RPCNoiseStripsObject_h

#include<vector>
#include<map>
#include<iostream>
#include<boost/cstdint.hpp>

class RPCNoiseStripsObject {

 public:

  int version;
  int run;

  //structure suitable for cluster size
  struct NoiseStripsObjectItem {
    int dpid;
    float channelNumber;
    float stripNumber;
    float isDead;
    float isMasked;
    float rate;
    float weight;
  };
  
  RPCNoiseStripsObject(){}
  ~RPCNoiseStripsObject(){}
 
  std::vector<NoiseStripsObjectItem>  const & getCls() const;
  std::vector<NoiseStripsObjectItem>  v_cls; 

 private:

};

#endif  //RPCNoiseStripsObject_h
