#ifndef FIRMWARE_dataformats_pf_h
#define FIRMWARE_dataformats_pf_h

#include "datatypes.h"
#include "bit_encoding.h"

namespace l1ct {

struct PFCommonObj {
  pt_t hwPt;
  eta_t hwEta; // relative to the region center, at calo
  phi_t hwPhi; // relative to the region center, at calo
  ParticleID hwId;

  int intPt() const { return Scales::intPt(hwPt); }
  int intEta() const { return hwEta.to_int(); }
  int intPhi() const { return hwPhi.to_int(); }
  float floatPt() const { return Scales::floatPt(hwPt); }
  float floatEta() const { return Scales::floatEta(hwEta); }
  float floatPhi() const { return Scales::floatPhi(hwPhi); }
  int intId() const { return hwId.rawId(); }
  int oldId() const { return hwPt > 0 ? hwId.oldId() : 0; }
  int pdgId() const { return hwId.pdgId(); }

  static const int _PFCOMMON_BITWIDTH = pt_t::width + eta_t::width + phi_t::width + 3;
  template<typename U>
  inline void _pack_common(U & out, unsigned int & start) const {
        _pack_into_bits(out, start, hwPt);
        _pack_into_bits(out, start, hwEta);
        _pack_into_bits(out, start, hwPhi);
        _pack_into_bits(out, start, hwId.bits);
  }
  template<typename U>
  inline void _unpack_common(const U & src, unsigned int & start) {
        _unpack_from_bits(src, start, hwPt);
        _unpack_from_bits(src, start, hwEta);
        _unpack_from_bits(src, start, hwPhi);
        _unpack_from_bits(src, start, hwId.bits);
  }

};

struct PFChargedObj : public PFCommonObj {
  // WARNING: for whatever reason, maybe connected with datamember alignment,
  //          in 2019.2 synthesis fails if DEta & DPhi are put before Z0 & Dxy
  z0_t hwZ0;
  dxy_t hwDxy;
  tkdeta_t hwDEta; // relative to the region center, at calo
  tkdphi_t hwDPhi; // relative to the region center, at calo
  tkquality_t hwTkQuality;

  phi_t hwVtxPhi() const {
    return hwId.chargeOrNull() ? hwPhi + hwDPhi : hwPhi - hwDPhi;
  }
  eta_t hwVtxEta() const { return hwEta + hwDEta; }

  inline bool operator==(const PFChargedObj & other) const {
    return hwPt == other.hwPt &&
      hwEta == other.hwEta &&
      hwPhi == other.hwPhi &&
      hwId == other.hwId &&
      hwDEta == other.hwDEta &&
      hwDPhi == other.hwDPhi &&
      hwZ0 == other.hwZ0 &&
      hwDxy == other.hwDxy &&
      hwTkQuality == other.hwTkQuality;
  }

  inline bool operator>(const PFChargedObj &other) const { 
      return hwPt > other.hwPt; 
  }
  inline bool operator<(const PFChargedObj &other) const { 
      return hwPt < other.hwPt; 
  }


  inline void clear() {
    hwPt = 0;
    hwEta = 0;
    hwPhi = 0;
    hwId.clear();
    hwDEta = 0;
    hwDPhi = 0;
    hwZ0 = 0;
    hwDxy = 0;
    hwTkQuality = 0;
  }

  float floatDEta() const { return Scales::floatEta(hwDEta); }
  float floatDPhi() const { return Scales::floatPhi(hwDPhi); }
  float floatVtxEta() const { return Scales::floatEta(hwVtxEta()); }
  float floatVtxPhi() const { return Scales::floatPhi(hwVtxPhi()); }
  float floatZ0() const { return Scales::floatZ0(hwZ0); }
  float floatDxy() const { return Scales::floatDxy(hwDxy); }

  static const int BITWIDTH = _PFCOMMON_BITWIDTH + tkdeta_t::width + tkdphi_t::width + z0_t::width + dxy_t::width + tkquality_t::width;
  inline ap_uint<BITWIDTH> pack() const {
        ap_uint<BITWIDTH> ret; unsigned int start = 0;
        _pack_common(ret, start);
        _pack_into_bits(ret, start, hwDEta);
        _pack_into_bits(ret, start, hwDPhi);
        _pack_into_bits(ret, start, hwZ0);
        _pack_into_bits(ret, start, hwDxy);
        _pack_into_bits(ret, start, hwTkQuality);
        return ret;
  }
  inline static PFChargedObj unpack(const ap_uint<BITWIDTH> & src) {
        PFChargedObj ret; unsigned int start = 0;
        ret._unpack_common(src, start);
        _unpack_from_bits(src, start, ret.hwDEta);
        _unpack_from_bits(src, start, ret.hwDPhi);
        _unpack_from_bits(src, start, ret.hwZ0);
        _unpack_from_bits(src, start, ret.hwDxy);
        _unpack_from_bits(src, start, ret.hwTkQuality);
        return ret;
  }

};
inline void clear(PFChargedObj &c) {
  c.clear();
 }

struct PFNeutralObj : public PFCommonObj {
  pt_t hwEmPt;
  ap_uint<6> hwEmID;
  ap_uint<6> hwPUID;

  inline bool operator==(const PFNeutralObj & other) const {
    return hwPt == other.hwPt &&
      hwEta == other.hwEta &&
      hwPhi == other.hwPhi &&
      hwId == other.hwId &&
      hwEmPt == other.hwEmPt &&
      hwEmID == other.hwEmID &&
      hwPUID == other.hwPUID;
  }

  inline bool operator>(const PFNeutralObj &other) const { 
      return hwPt > other.hwPt; 
  }
  inline bool operator<(const PFNeutralObj &other) const { 
      return hwPt < other.hwPt; 
  }
  
  inline void clear() {
    hwPt = 0;
    hwEta = 0;
    hwPhi = 0;
    hwId.clear();
    hwEmPt = 0;
    hwEmID = 0;
    hwPUID = 0;
  }


  int intEmPt() const { return Scales::intPt(hwEmPt); }
  float floatEmPt() const { return Scales::floatPt(hwEmPt); }

  static const int BITWIDTH = _PFCOMMON_BITWIDTH + pt_t::width + ap_uint<6>::width + ap_uint<6>::width;
  inline ap_uint<BITWIDTH> pack() const {
        ap_uint<BITWIDTH> ret; unsigned int start = 0;
        _pack_common(ret, start);
        _pack_into_bits(ret, start, hwEmPt);
        _pack_into_bits(ret, start, hwEmID);
        _pack_into_bits(ret, start, hwPUID);
        return ret;
  }
  inline static PFNeutralObj unpack(const ap_uint<BITWIDTH> & src) {
        PFNeutralObj ret; unsigned int start = 0;
        ret._unpack_common(src, start);
        _unpack_from_bits(src, start, ret.hwEmPt);
        _unpack_from_bits(src, start, ret.hwEmID);
        _unpack_from_bits(src, start, ret.hwPUID);
        return ret;
  }

};

inline void clear(PFNeutralObj &c) {
  c.clear();
}

struct PFRegion {
    glbeta_t hwEtaCenter;

    inline glbeta_t hwGlbEta(eta_t hwEta) const {
        return hwEtaCenter + hwEta;
    }

    inline float floatEtaCenter() const {
        return Scales::floatEta(hwEtaCenter);
    }
    inline float floatGlbEta(eta_t hwEta) const {
        return Scales::floatEta(hwGlbEta(hwEta));
    }

    static const int BITWIDTH = glbeta_t::width;
    inline ap_uint<BITWIDTH> pack() const {
        ap_uint<BITWIDTH> ret; unsigned int start = 0;
        _pack_into_bits(ret, start, hwEtaCenter);
        return ret;
    }
    inline static PFRegion unpack(const ap_uint<BITWIDTH> & src) {
        PFRegion ret; unsigned int start = 0;
        _unpack_from_bits(src, start, ret.hwEtaCenter);
        return ret;
    }


};

} // namespace

#endif
