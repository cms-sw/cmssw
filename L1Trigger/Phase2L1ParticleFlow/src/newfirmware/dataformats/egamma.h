#ifndef FIRMWARE_dataformats_egamma_h
#define FIRMWARE_dataformats_egamma_h


#include "datatypes.h"
#include "bit_encoding.h"

namespace l1ct {
  struct EGIsoObj {
    pt_t hwPt;
    eta_t hwEta;  // at calo face
    phi_t hwPhi;
    // FIXME: add quality and isolation
    // uint16_t hwQual;
    // uint16_t hwIso;
    
    int intPt() const { return Scales::intPt(hwPt); }
    int intEta() const { return hwEta.to_int(); }
    int intPhi() const { return hwPhi.to_int(); }
    float floatPt() const { return Scales::floatPt(hwPt); }
    float floatEta() const { return Scales::floatEta(hwEta); }
    float floatPhi() const { return Scales::floatPhi(hwPhi); }

    inline bool operator==(const EGIsoObj & other) const {
      // FIXME: add quality and isolation
      return hwPt == other.hwPt && 
             hwEta == other.hwEta &&
             hwPhi == other.hwPhi;
    }

    inline bool operator>(const EGIsoObj &other) const { 
        return hwPt > other.hwPt; 
    }
    inline bool operator<(const EGIsoObj &other) const { 
        return hwPt < other.hwPt; 
    }

    inline void clear() {
      // FIXME: add quality and isolation
      hwPt = 0;
      hwEta = 0;
      hwPhi = 0;
    }

    // FIXME: fix bit width once object fully defined
    static const int BITWIDTH = pt_t::width + eta_t::width + phi_t::width;
    inline ap_uint<BITWIDTH> pack() const {
          ap_uint<BITWIDTH> ret; unsigned int start = 0;
          _pack_into_bits(ret, start, hwPt);
          _pack_into_bits(ret, start, hwEta);
          _pack_into_bits(ret, start, hwPhi);
          return ret;
    }
    inline static EGIsoObj unpack(const ap_uint<BITWIDTH> & src) {
          EGIsoObj ret; unsigned int start = 0;
          _unpack_from_bits(src, start, ret.hwPt);
          _unpack_from_bits(src, start, ret.hwEta);
          _unpack_from_bits(src, start, ret.hwPhi);
          return ret;
    }
  };
  
  
  inline void clear(EGIsoObj &c) {
    c.clear();
   }

  
  
  struct EGIsoEleObj : public EGIsoObj {
    // WARNING: for whatever reason, maybe connected with datamember alignment,
    //          in 2019.2 synthesis fails if DEta & DPhi are put before Z0 & Dxy
    z0_t hwZ0;
    tkdeta_t hwDEta; // relative to the region center, at calo
    tkdphi_t hwDPhi; // relative to the region center, at calo
    bool hwCharge;

    phi_t hwVtxPhi() const {
      // FIXME: address charge
      // return hwId.chargeOrNull() ? hwPhi + hwDPhi : hwPhi - hwDPhi;
      return hwPhi - hwDPhi;
    }
    eta_t hwVtxEta() const { return hwEta + hwDEta; }

    inline bool operator==(const EGIsoEleObj & other) const {
      // FIXME: add quality and isolation
      return hwPt == other.hwPt &&
             hwEta == other.hwEta &&
             hwPhi == other.hwPhi &&
             hwDEta == other.hwDEta &&
             hwDPhi == other.hwDPhi &&
             hwZ0 == other.hwZ0 &&
             hwCharge == other.hwCharge;
    }
    
    inline bool operator>(const EGIsoEleObj &other) const { 
        return hwPt > other.hwPt; 
    }
    inline bool operator<(const EGIsoEleObj &other) const { 
        return hwPt < other.hwPt; 
    }


    inline void clear() {
      // FIXME: add quality and isolation
      hwPt = 0;
      hwEta = 0;
      hwPhi = 0;
      hwDEta = 0;
      hwDPhi = 0;
      hwZ0 = 0;
      hwCharge = false;
    }

    int intCharge() const { return hwCharge ? +1 : -1; }
    float floatDEta() const { return Scales::floatEta(hwDEta); }
    float floatDPhi() const { return Scales::floatPhi(hwDPhi); }
    float floatVtxEta() const { return Scales::floatEta(hwVtxEta()); }
    float floatVtxPhi() const { return Scales::floatPhi(hwVtxPhi()); }
    float floatZ0() const { return Scales::floatZ0(hwZ0); }

    static const int BITWIDTH = EGIsoObj::BITWIDTH + tkdeta_t::width + tkdphi_t::width + z0_t::width + 1;
    inline ap_uint<BITWIDTH> pack() const {
          ap_uint<BITWIDTH> ret; unsigned int start = 0;
          _pack_into_bits(ret, start, hwPt);
          _pack_into_bits(ret, start, hwEta);
          _pack_into_bits(ret, start, hwPhi);
          _pack_into_bits(ret, start, hwDEta);
          _pack_into_bits(ret, start, hwDPhi);
          _pack_into_bits(ret, start, hwZ0);
          _pack_bool_into_bits(ret, start, hwCharge);
          return ret;
    }
    inline static EGIsoEleObj unpack(const ap_uint<BITWIDTH> & src) {
          EGIsoEleObj ret; unsigned int start = 0;
          _unpack_from_bits(src, start, ret.hwPt);
          _unpack_from_bits(src, start, ret.hwEta);
          _unpack_from_bits(src, start, ret.hwPhi);
          _unpack_from_bits(src, start, ret.hwDEta);
          _unpack_from_bits(src, start, ret.hwDPhi);
          _unpack_from_bits(src, start, ret.hwZ0);
          _unpack_bool_from_bits(src, start, ret.hwCharge);
          return ret;
    }

    
    
  };

  inline void clear(EGIsoEleObj & c) {
    c.clear();
  }
}
#endif
