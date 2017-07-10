#ifndef EventFilter_L1TRawToDigi_Omtf_MuonDataWord64_H
#define EventFilter_L1TRawToDigi_Omtf_MuonDataWord64_H

#include <iostream>
#include "EventFilter/L1TRawToDigi/interface/OmtfDataWord64.h"

namespace omtf {
class MuonDataWord64 {
public:
  MuonDataWord64(Word64 data=0) : rawData(data) {}
  unsigned int weight_lowBits() const { return weight_; }
  unsigned int layers() const { return layers_; }
  unsigned int ch() const { return ch_; }
  unsigned int vch() const { return vch_; }
           int phi() const { return phi_; }
           int eta() const { return eta_; }
  unsigned int pT() const { return pT_; }
  unsigned int quality() const { return quality_; }
  unsigned int bxNum() const { return bxNum_; }
  unsigned int type() const { return type_;}
  friend std::ostream & operator<< (std::ostream &out, const MuonDataWord64 &o);

private:
  union {
    uint64_t rawData;
    struct {
      uint64_t pT_ : 9;
      uint64_t quality_ : 4;
       int64_t eta_ : 9;
      uint64_t empty_   : 1; //not used, orig h/f
       int64_t phi_ : 8;
      uint64_t bc0_  : 1;
      uint64_t ch_ : 1;
      uint64_t vch_ : 1;
      uint64_t layers_ : 18;
      uint64_t weight_ : 5;
      uint64_t bxNum_   : 3;
      uint64_t type_    : 4;
    };
  };
};

} //namespace omtf
#endif

