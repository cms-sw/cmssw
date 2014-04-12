#ifndef RPCMaskedStrips_h
#define RPCMaskedStrips_h

#include<vector>
#include<iostream>
#include<boost/cstdint.hpp>


class RPCMaskedStrips {

 public:

  struct MaskItem {
    int rawId;
    int strip;
  };
  
  RPCMaskedStrips(){}

  ~RPCMaskedStrips(){}

  std::vector<MaskItem> const & getMaskVec() const {return MaskVec;}

  std::vector<MaskItem> MaskVec;

};

#endif
