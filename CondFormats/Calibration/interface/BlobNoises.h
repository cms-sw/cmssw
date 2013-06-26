#ifndef BLOBNOISES_H
#define BLOBNOISES_H
#include<vector>
//#include<boost/cstdint.hpp>
#include <stdint.h>

class BlobNoises {
public:
  BlobNoises();
  void fill(unsigned int id,short bsize);
  virtual ~BlobNoises();
  struct DetRegistry{
    uint32_t detid;
    uint32_t ibegin;
    uint32_t iend;
    bool operator==(const DetRegistry& rhs) const {
      if(detid!=rhs.detid)return false;
      if(ibegin!=rhs.ibegin)return false;
      if(iend!=rhs.iend)return false;
      return true;
    }
    bool operator!=(const DetRegistry& rhs) const {
      return !operator==(rhs);
    }
  };
  bool operator==(const BlobNoises& rhs) const {
    if(v_noises!=rhs.v_noises){
      return false;
    }
    if(indexes!=rhs.indexes){
      return false;
    }
    return true;
  }
  bool operator!=(const BlobNoises& rhs) const {
    return !operator==(rhs);
  }
  
  //std::vector<int16_t>  v_noises; //dictionary problem with this
  std::vector<short> v_noises;
  std::vector<DetRegistry> indexes;
};
#endif
