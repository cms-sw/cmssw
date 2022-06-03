#ifndef DataFormats_L1TParticleFlow_pf_h
#define DataFormats_L1TParticleFlow_pf_h

#include "DataFormats/L1TParticleFlow/interface/datatypes.h"
#include "DataFormats/L1TParticleFlow/interface/bit_encoding.h"

namespace l1ct {

  struct PFCommonObj {
    pt_t hwPt;
    eta_t hwEta;  // relative to the region center, at calo
    phi_t hwPhi;  // relative to the region center, at calo
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
    int intCharge() const { return hwId.intCharge(); }

    static const int _PFCOMMON_BITWIDTH = pt_t::width + eta_t::width + phi_t::width + 3;
    template <typename U>
    inline void pack_common(U &out, unsigned int &start) const {
      pack_into_bits(out, start, hwPt);
      pack_into_bits(out, start, hwEta);
      pack_into_bits(out, start, hwPhi);
      pack_into_bits(out, start, hwId.bits);
    }
    template <typename U>
    inline void unpack_common(const U &src, unsigned int &start) {
      unpack_from_bits(src, start, hwPt);
      unpack_from_bits(src, start, hwEta);
      unpack_from_bits(src, start, hwPhi);
      unpack_from_bits(src, start, hwId.bits);
    }
  };

  struct PFChargedObj : public PFCommonObj {
    // WARNING: for whatever reason, maybe connected with datamember alignment,
    //          in 2019.2 synthesis fails if DEta & DPhi are put before Z0 & Dxy
    z0_t hwZ0;
    dxy_t hwDxy;
    tkdeta_t hwDEta;  // relative to the region center, at calo
    tkdphi_t hwDPhi;  // relative to the region center, at calo
    tkquality_t hwTkQuality;

    phi_t hwVtxPhi() const { return hwId.chargeOrNull() ? hwPhi + hwDPhi : hwPhi - hwDPhi; }
    eta_t hwVtxEta() const { return hwEta + hwDEta; }

    inline bool operator==(const PFChargedObj &other) const {
      return hwPt == other.hwPt && hwEta == other.hwEta && hwPhi == other.hwPhi && hwId == other.hwId &&
             hwDEta == other.hwDEta && hwDPhi == other.hwDPhi && hwZ0 == other.hwZ0 && hwDxy == other.hwDxy &&
             hwTkQuality == other.hwTkQuality;
    }

    inline bool operator>(const PFChargedObj &other) const { return hwPt > other.hwPt; }
    inline bool operator<(const PFChargedObj &other) const { return hwPt < other.hwPt; }

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

    int intVtxEta() const { return hwVtxEta().to_int(); }
    int intVtxPhi() const { return hwVtxPhi().to_int(); }
    float floatDEta() const { return Scales::floatEta(hwDEta); }
    float floatDPhi() const { return Scales::floatPhi(hwDPhi); }
    float floatVtxEta() const { return Scales::floatEta(hwVtxEta()); }
    float floatVtxPhi() const { return Scales::floatPhi(hwVtxPhi()); }
    float floatZ0() const { return Scales::floatZ0(hwZ0); }
    float floatDxy() const { return Scales::floatDxy(hwDxy); }

    static const int BITWIDTH =
        _PFCOMMON_BITWIDTH + tkdeta_t::width + tkdphi_t::width + z0_t::width + dxy_t::width + tkquality_t::width;
    inline ap_uint<BITWIDTH> pack() const {
      ap_uint<BITWIDTH> ret;
      unsigned int start = 0;
      pack_common(ret, start);
      pack_into_bits(ret, start, hwDEta);
      pack_into_bits(ret, start, hwDPhi);
      pack_into_bits(ret, start, hwZ0);
      pack_into_bits(ret, start, hwDxy);
      pack_into_bits(ret, start, hwTkQuality);
      return ret;
    }
    inline static PFChargedObj unpack(const ap_uint<BITWIDTH> &src) {
      PFChargedObj ret;
      unsigned int start = 0;
      ret.unpack_common(src, start);
      unpack_from_bits(src, start, ret.hwDEta);
      unpack_from_bits(src, start, ret.hwDPhi);
      unpack_from_bits(src, start, ret.hwZ0);
      unpack_from_bits(src, start, ret.hwDxy);
      unpack_from_bits(src, start, ret.hwTkQuality);
      return ret;
    }
  };
  inline void clear(PFChargedObj &c) { c.clear(); }

  struct PFNeutralObj : public PFCommonObj {
    pt_t hwEmPt;
    emid_t hwEmID;
    ap_uint<6> hwPUID;

    inline bool operator==(const PFNeutralObj &other) const {
      return hwPt == other.hwPt && hwEta == other.hwEta && hwPhi == other.hwPhi && hwId == other.hwId &&
             hwEmPt == other.hwEmPt && hwEmID == other.hwEmID && hwPUID == other.hwPUID;
    }

    inline bool operator>(const PFNeutralObj &other) const { return hwPt > other.hwPt; }
    inline bool operator<(const PFNeutralObj &other) const { return hwPt < other.hwPt; }

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
      ap_uint<BITWIDTH> ret;
      unsigned int start = 0;
      pack_common(ret, start);
      pack_into_bits(ret, start, hwEmPt);
      pack_into_bits(ret, start, hwEmID);
      pack_into_bits(ret, start, hwPUID);
      return ret;
    }
    inline static PFNeutralObj unpack(const ap_uint<BITWIDTH> &src) {
      PFNeutralObj ret;
      unsigned int start = 0;
      ret.unpack_common(src, start);
      unpack_from_bits(src, start, ret.hwEmPt);
      unpack_from_bits(src, start, ret.hwEmID);
      unpack_from_bits(src, start, ret.hwPUID);
      return ret;
    }
  };

  inline void clear(PFNeutralObj &c) { c.clear(); }

  struct PFRegion {
    glbeta_t hwEtaCenter;
    glbphi_t hwPhiCenter;
    eta_t hwEtaHalfWidth;
    phi_t hwPhiHalfWidth;
    eta_t hwEtaExtra;
    phi_t hwPhiExtra;

    inline int intEtaCenter() const { return hwEtaCenter.to_int(); }
    inline int intPhiCenter() const { return hwPhiCenter.to_int(); }
    inline float floatEtaCenter() const { return Scales::floatEta(hwEtaCenter); }
    inline float floatPhiCenter() const { return Scales::floatPhi(hwPhiCenter); }
    inline float floatEtaHalfWidth() const { return Scales::floatEta(hwEtaHalfWidth); }
    inline float floatPhiHalfWidth() const { return Scales::floatPhi(hwPhiHalfWidth); }
    inline float floatEtaExtra() const { return Scales::floatEta(hwEtaExtra); }
    inline float floatPhiExtra() const { return Scales::floatPhi(hwPhiExtra); }
    inline float floatPhiHalfWidthExtra() const { return floatPhiHalfWidth() + floatPhiExtra(); }
    inline float floatEtaMin() const { return Scales::floatEta(glbeta_t(hwEtaCenter - hwEtaHalfWidth)); }
    inline float floatEtaMax() const { return Scales::floatEta(glbeta_t(hwEtaCenter + hwEtaHalfWidth)); }
    inline float floatEtaMinExtra() const {
      return Scales::floatEta(glbeta_t(hwEtaCenter - hwEtaHalfWidth - hwEtaExtra));
    }
    inline float floatEtaMaxExtra() const {
      return Scales::floatEta(glbeta_t(hwEtaCenter + hwEtaHalfWidth + hwEtaExtra));
    }

    inline glbeta_t hwGlbEta(eta_t hwEta) const { return hwEtaCenter + hwEta; }
    inline glbeta_t hwGlbEta(glbeta_t hwEta) const { return hwEtaCenter + hwEta; }
    inline glbphi_t hwGlbPhi(glbphi_t hwPhi) const {
      ap_int<glbphi_t::width + 1> ret = hwPhiCenter + hwPhi;
      if (ret > Scales::INTPHI_PI)
        return ret - Scales::INTPHI_TWOPI;
      else if (ret <= -Scales::INTPHI_PI)
        return ret + Scales::INTPHI_TWOPI;
      else
        return ret;
    }

    template <typename T>
    inline glbeta_t hwGlbEtaOf(const T &t) const {
      return hwGlbEta(t.hwEta);
    }
    template <typename T>
    inline glbphi_t hwGlbPhiOf(const T &t) const {
      return hwGlbPhi(t.hwPhi);
    }

    inline float floatGlbEta(eta_t hwEta) const { return Scales::floatEta(hwGlbEta(hwEta)); }
    inline float floatGlbPhi(phi_t hwPhi) const { return Scales::floatPhi(hwGlbPhi(hwPhi)); }
    inline float floatGlbEta(glbeta_t hwEta) const { return Scales::floatEta(hwGlbEta(hwEta)); }
    inline float floatGlbPhi(glbphi_t hwPhi) const { return Scales::floatPhi(hwGlbPhi(hwPhi)); }

    template <typename T>
    inline float floatGlbEtaOf(const T &t) const {
      return floatGlbEta(t.hwEta);
    }
    template <typename T>
    inline float floatGlbPhiOf(const T &t) const {
      return floatGlbPhi(t.hwPhi);
    }

    inline bool isFiducial(eta_t hwEta, phi_t hwPhi) const {
      return hwEta <= hwEtaHalfWidth && hwEta > -hwEtaHalfWidth && hwPhi <= hwPhiHalfWidth && hwPhi > -hwPhiHalfWidth;
    }
    template <typename ET, typename PT>  // forcing down to eta_t and phi_t may have overflows & crops
    inline bool isInside(ET hwEta, PT hwPhi) const {
      return hwEta <= hwEtaHalfWidth + hwEtaExtra && hwEta >= -hwEtaHalfWidth - hwEtaExtra &&
             hwPhi <= hwPhiHalfWidth + hwPhiExtra && hwPhi >= -hwPhiHalfWidth - hwPhiExtra;
    }

    template <typename T>
    inline bool isFiducial(const T &t) const {
      return isFiducial(t.hwEta, t.hwPhi);
    }
    template <typename T>
    inline bool isInside(const T &t) const {
      return isInside(t.hwEta, t.hwPhi);
    }

    static const int BITWIDTH = glbeta_t::width + glbphi_t::width + 2 * eta_t::width + 2 * phi_t::width;
    inline ap_uint<BITWIDTH> pack() const {
      ap_uint<BITWIDTH> ret;
      unsigned int start = 0;
      pack_into_bits(ret, start, hwEtaCenter);
      pack_into_bits(ret, start, hwPhiCenter);
      pack_into_bits(ret, start, hwEtaHalfWidth);
      pack_into_bits(ret, start, hwPhiHalfWidth);
      pack_into_bits(ret, start, hwEtaExtra);
      pack_into_bits(ret, start, hwPhiExtra);
      return ret;
    }
    inline static PFRegion unpack(const ap_uint<BITWIDTH> &src) {
      PFRegion ret;
      unsigned int start = 0;
      unpack_from_bits(src, start, ret.hwEtaCenter);
      unpack_from_bits(src, start, ret.hwPhiCenter);
      unpack_from_bits(src, start, ret.hwEtaHalfWidth);
      unpack_from_bits(src, start, ret.hwPhiHalfWidth);
      unpack_from_bits(src, start, ret.hwEtaExtra);
      unpack_from_bits(src, start, ret.hwPhiExtra);
      return ret;
    }
  };

}  // namespace l1ct

#endif
