#ifndef FIRMWARE_dataformats_jets_h
#define FIRMWARE_dataformats_jets_h

#include "datatypes.h"
#include "bit_encoding.h"

namespace l1ct {

  struct Jet {
    pt_t hwPt;
    glbeta_t hwEta;
    glbphi_t hwPhi;

    inline bool operator==(const Jet &other) const {
      return hwPt == other.hwPt && hwEta == other.hwEta && hwPhi == other.hwPhi;
    }

    inline bool operator>(const Jet &other) const { return hwPt > other.hwPt; }
    inline bool operator<(const Jet &other) const { return hwPt < other.hwPt; }

    inline void clear() {
      hwPt = 0;
      hwEta = 0;
      hwPhi = 0;
    }

    int intPt() const { return Scales::intPt(hwPt); }
    int intEta() const { return hwEta.to_int(); }
    int intPhi() const { return hwPhi.to_int(); }
    float floatPt() const { return Scales::floatPt(hwPt); }
    float floatEta() const { return Scales::floatEta(hwEta); }
    float floatPhi() const { return Scales::floatPhi(hwPhi); }

    static const int BITWIDTH = pt_t::width + glbeta_t::width + glbphi_t::width;
    inline ap_uint<BITWIDTH> pack() const {
      ap_uint<BITWIDTH> ret;
      unsigned int start = 0;
      _pack_into_bits(ret, start, hwPt);
      _pack_into_bits(ret, start, hwEta);
      _pack_into_bits(ret, start, hwPhi);
      return ret;
    }
    inline static Jet unpack(const ap_uint<BITWIDTH> &src) {
      Jet ret;
      unsigned int start = 0;
      _unpack_from_bits(src, start, ret.hwPt);
      _unpack_from_bits(src, start, ret.hwEta);
      _unpack_from_bits(src, start, ret.hwPhi);
      return ret;
    }
  };

  inline void clear(Jet &c) { c.clear(); }

}  // namespace l1ct

#endif
