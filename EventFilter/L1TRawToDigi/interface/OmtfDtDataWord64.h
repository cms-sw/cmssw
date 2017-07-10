#ifndef EventFilter_L1TRawToDigi_Omtf_DtDataWord64_H
#define EventFilter_L1TRawToDigi_Omtf_DtDataWord64_H

#include<iostream>
#include "EventFilter/L1TRawToDigi/interface/OmtfDataWord64.h"

namespace omtf {

class DtDataWord64 {
public:
  DtDataWord64(Word64 data) : rawData(data) {}
  DtDataWord64() : rawData(Word64(DataWord64::dt)<<60) {}
  int phi() const { return st_phi_; }
  int phiB() const { return st_phib_; }
  unsigned int quality() const { return st_q_; }
  unsigned int eta() const { return eta_hit_; }
  unsigned int etaQuality() const { return eta_qbit_; }
  unsigned int bcnt_st() const { return bcnt_st_; }
  unsigned int bcnt_e0() const { return bcnt_e0_; }
  unsigned int bcnt_e1() const { return bcnt_e1_; }
  unsigned int valid() const { return valid_; }
  unsigned int station() const { return st_; }
  unsigned int fiber() const { return fiber_; }
  unsigned int sector() const { return sector_; }
  unsigned int bxNum() const { return bxNum_; }
  unsigned int type() const { return type_;}
  friend class OmtfPacker;
  friend class DtPacker;
  friend std::ostream & operator<< (std::ostream &out, const DtDataWord64 &o);

private:
  union {
    uint64_t rawData;
    struct {
      int64_t st_phi_ : 12;
      int64_t st_phib_ : 10;
      uint64_t st_q_     : 5;
      uint64_t st_cal_   : 1;
      uint64_t eta_qbit_ : 7;
      uint64_t eta_hit_ : 7;
      uint64_t dummy1_  : 1;
      uint64_t bcnt_st_ : 2;
      uint64_t bcnt_e0_ : 2;
      uint64_t bcnt_e1_ : 2;
      uint64_t valid_   : 3;
      uint64_t st_      : 2;
      uint64_t fiber_   : 1;
      uint64_t sector_  : 2;
      uint64_t bxNum_   : 3;
      uint64_t type_    : 4;
    };
  };
};

} //namespace Omtf
#endif


