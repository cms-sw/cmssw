#ifndef DataFormats_L1TParticleFlow_sums_h
#define DataFormats_L1TParticleFlow_sums_h

#include "DataFormats/L1TParticleFlow/interface/datatypes.h"
#include "DataFormats/L1TParticleFlow/interface/gt_datatypes.h"
#include "DataFormats/L1TParticleFlow/interface/bit_encoding.h"

namespace l1ct {

  struct Sum {
    pt_t hwPt;
    glbphi_t hwPhi;
    pt_t hwSumPt;

    inline bool operator==(const Sum &other) const {
      return hwPt == other.hwPt && hwPhi == other.hwPhi && hwSumPt == other.hwSumPt;
    }

    inline void clear() {
      hwPt = 0;
      hwPhi = 0;
      hwSumPt = 0;
    }

    int intPt() const { return Scales::intPt(hwPt); }
    int intPhi() const { return hwPhi.to_int(); }
    int intSumPt() const { return Scales::intPt(hwSumPt); }
    float floatPt() const { return Scales::floatPt(hwPt); }
    float floatPhi() const { return Scales::floatPhi(hwPhi); }
    float floatSumPt() const { return Scales::floatPt(hwSumPt); }

    static const int BITWIDTH = pt_t::width + glbphi_t::width + pt_t::width;
    inline ap_uint<BITWIDTH> pack() const {
      ap_uint<BITWIDTH> ret;
      unsigned int start = 0;
      pack_into_bits(ret, start, hwPt);
      pack_into_bits(ret, start, hwPhi);
      pack_into_bits(ret, start, hwSumPt);
      return ret;
    }
    inline static Sum unpack(const ap_uint<BITWIDTH> &src) {
      Sum ret;
      unsigned int start = 0;
      unpack_from_bits(src, start, ret.hwPt);
      unpack_from_bits(src, start, ret.hwPhi);
      unpack_from_bits(src, start, ret.hwSumPt);
      return ret;
    }

    l1gt::Sum toGT() const {
      l1gt::Sum sum;
      sum.valid = (hwPt != 0) || (hwSumPt != 0);
      sum.vector_pt = CTtoGT_pt(hwPt);
      sum.vector_phi = CTtoGT_phi(hwPhi);
      sum.scalar_pt = CTtoGT_pt(hwSumPt);
      return sum;
    }
  };

  inline void clear(Sum &c) { c.clear(); }

}  // namespace l1ct

#endif
