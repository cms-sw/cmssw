#ifndef EventFilter_L1TRawToDigi_Omtf_RpcDataWord64_H
#define EventFilter_L1TRawToDigi_Omtf_RpcDataWord64_H

#include<iostream>
#include "EventFilter/L1TRawToDigi/interface/OmtfDataWord64.h"

namespace omtf {

class RpcDataWord64 {
public:
  RpcDataWord64(Word64 data) : rawData(data) {}
  RpcDataWord64() : rawData(Word64(DataWord64::rpc)<<60) {}
  unsigned int frame1() const { return frame1_;}
  unsigned int frame2() const { return frame2_;}
  unsigned int frame3() const { return frame3_;}
  unsigned int empty() const { return empty_;}
  unsigned int linkNum() const { return linkNum_;}
  unsigned int bxNum() const { return bxNum_; }
  unsigned int type() const { return type_;}
  friend class OmtfPacker;
  friend class RpcPacker;
  friend std::ostream & operator<< (std::ostream &out, const RpcDataWord64 &o);

private:
  union {
    uint64_t rawData;
    struct {
      uint64_t frame3_  : 16;
      uint64_t frame2_  : 16;
      uint64_t frame1_  : 16;
      uint64_t empty_   : 4;
      uint64_t linkNum_ : 5;
      uint64_t bxNum_   : 3;
      uint64_t type_    : 4;
    };
  };
};

} //namespace omtf

#endif
