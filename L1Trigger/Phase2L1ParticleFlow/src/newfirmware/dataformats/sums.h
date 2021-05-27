#ifndef FIRMWARE_dataformats_sums_h
#define FIRMWARE_dataformats_sums_h

#include "datatypes.h"
#include "bit_encoding.h"

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
      _pack_into_bits(ret, start, hwPt);
      _pack_into_bits(ret, start, hwPhi);
      _pack_into_bits(ret, start, hwSumPt);
      return ret;
    }
    inline static Sum unpack(const ap_uint<BITWIDTH> &src) {
      Sum ret;
      unsigned int start = 0;
      _unpack_from_bits(src, start, ret.hwPt);
      _unpack_from_bits(src, start, ret.hwPhi);
      _unpack_from_bits(src, start, ret.hwSumPt);
      return ret;
    }
  };

  inline void clear(Sum &c) { c.clear(); }

}  // namespace l1ct

#endif
