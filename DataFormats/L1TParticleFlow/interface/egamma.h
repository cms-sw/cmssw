#ifndef DataFormats_L1TParticleFlow_egamma_h
#define DataFormats_L1TParticleFlow_egamma_h

#include "DataFormats/L1TParticleFlow/interface/datatypes.h"
#include "DataFormats/L1TParticleFlow/interface/gt_datatypes.h"
#include "DataFormats/L1TParticleFlow/interface/bit_encoding.h"

namespace l1ct {

  struct EGIsoObj {
    pt_t hwPt;
    glbeta_t hwEta;  // at calo face
    glbphi_t hwPhi;
    egquality_t hwQual;
    iso_t hwIso;

    int intPt() const { return Scales::intPt(hwPt); }
    int intEta() const { return hwEta.to_int(); }
    int intPhi() const { return hwPhi.to_int(); }
    int intQual() const { return hwQual.to_int(); }
    int intIso() const { return hwIso.to_int(); }

    float floatPt() const { return Scales::floatPt(hwPt); }
    float floatEta() const { return Scales::floatEta(hwEta); }
    float floatPhi() const { return Scales::floatPhi(hwPhi); }
    float floatIso() const { return Scales::floatIso(hwIso); }

    inline bool operator==(const EGIsoObj &other) const {
      return hwPt == other.hwPt && hwEta == other.hwEta && hwPhi == other.hwPhi && hwQual == other.hwQual &&
             hwIso == other.hwIso;
    }

    inline bool operator>(const EGIsoObj &other) const { return hwPt > other.hwPt; }
    inline bool operator<(const EGIsoObj &other) const { return hwPt < other.hwPt; }

    inline void clear() {
      hwPt = 0;
      hwEta = 0;
      hwPhi = 0;
      hwQual = 0;
      hwIso = 0;
    }

    static const int BITWIDTH = pt_t::width + glbeta_t::width + glbphi_t::width + egquality_t::width + iso_t::width;
    inline ap_uint<BITWIDTH> pack() const {
      ap_uint<BITWIDTH> ret;
      unsigned int start = 0;
      pack_into_bits(ret, start, hwPt);
      pack_into_bits(ret, start, hwEta);
      pack_into_bits(ret, start, hwPhi);
      pack_into_bits(ret, start, hwQual);
      pack_into_bits(ret, start, hwIso);
      return ret;
    }
    inline static EGIsoObj unpack(const ap_uint<BITWIDTH> &src) {
      EGIsoObj ret;
      ret.initFromBits(src);
      return ret;
    }

    inline void initFromBits(const ap_uint<BITWIDTH> &src) {
      unsigned int start = 0;
      unpack_from_bits(src, start, hwPt);
      unpack_from_bits(src, start, hwEta);
      unpack_from_bits(src, start, hwPhi);
      unpack_from_bits(src, start, hwQual);
      unpack_from_bits(src, start, hwIso);
    }

    l1gt::Photon toGT() const {
      l1gt::Photon pho;
      pho.valid = hwPt != 0;
      pho.v3.pt = CTtoGT_pt(hwPt);
      pho.v3.phi = CTtoGT_phi(hwPhi);
      pho.v3.eta = CTtoGT_eta(hwEta);
      pho.quality = hwQual;
      pho.isolation = hwIso;
      return pho;
    }
  };

  inline void clear(EGIsoObj &c) { c.clear(); }

  struct EGIsoEleObj : public EGIsoObj {
    // WARNING: for whatever reason, maybe connected with datamember alignment,
    //          in 2019.2 synthesis fails if DEta & DPhi are put before Z0 & Dxy
    z0_t hwZ0;
    tkdeta_t hwDEta;  // relative to the region center, at calo
    tkdphi_t hwDPhi;  // relative to the region center, at calo
    bool hwCharge;

    phi_t hwVtxPhi() const { return hwCharge ? hwPhi + hwDPhi : hwPhi - hwDPhi; }
    eta_t hwVtxEta() const { return hwEta + hwDEta; }

    inline bool operator==(const EGIsoEleObj &other) const {
      return hwPt == other.hwPt && hwEta == other.hwEta && hwPhi == other.hwPhi && hwQual == other.hwQual &&
             hwIso == other.hwIso && hwDEta == other.hwDEta && hwDPhi == other.hwDPhi && hwZ0 == other.hwZ0 &&
             hwCharge == other.hwCharge;
    }

    inline bool operator>(const EGIsoEleObj &other) const { return hwPt > other.hwPt; }
    inline bool operator<(const EGIsoEleObj &other) const { return hwPt < other.hwPt; }

    inline void clear() {
      hwPt = 0;
      hwEta = 0;
      hwPhi = 0;
      hwQual = 0;
      hwIso = 0;
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
      ap_uint<BITWIDTH> ret;
      unsigned int start = 0;
      pack_into_bits(ret, start, hwPt);
      pack_into_bits(ret, start, hwEta);
      pack_into_bits(ret, start, hwPhi);
      pack_into_bits(ret, start, hwQual);
      pack_into_bits(ret, start, hwIso);
      pack_into_bits(ret, start, hwDEta);
      pack_into_bits(ret, start, hwDPhi);
      pack_into_bits(ret, start, hwZ0);
      pack_bool_into_bits(ret, start, hwCharge);
      return ret;
    }
    inline static EGIsoEleObj unpack(const ap_uint<BITWIDTH> &src) {
      EGIsoEleObj ret;
      ret.initFromBits(src);
      return ret;
    }

    inline void initFromBits(const ap_uint<BITWIDTH> &src) {
      unsigned int start = 0;
      unpack_from_bits(src, start, hwPt);
      unpack_from_bits(src, start, hwEta);
      unpack_from_bits(src, start, hwPhi);
      unpack_from_bits(src, start, hwQual);
      unpack_from_bits(src, start, hwIso);
      unpack_from_bits(src, start, hwDEta);
      unpack_from_bits(src, start, hwDPhi);
      unpack_from_bits(src, start, hwZ0);
      unpack_bool_from_bits(src, start, hwCharge);
    }

    l1gt::Electron toGT() const {
      l1gt::Electron ele;
      ele.valid = hwPt != 0;
      ele.v3.pt = CTtoGT_pt(hwPt);
      ele.v3.phi = CTtoGT_phi(hwPhi);
      ele.v3.eta = CTtoGT_eta(hwEta);
      ele.quality = hwQual;
      ele.charge = hwCharge;
      ele.z0(l1ct::z0_t::width - 1, 0) = hwZ0(l1ct::z0_t::width - 1, 0);
      ele.isolation = hwIso;
      return ele;
    }
  };

  inline void clear(EGIsoEleObj &c) { c.clear(); }
}  // namespace l1ct
#endif
