#ifndef DataFormats_L1TParticleFlow_layer1_objs_h
#define DataFormats_L1TParticleFlow_layer1_objs_h

#include "DataFormats/L1TParticleFlow/interface/datatypes.h"
#include "DataFormats/L1TParticleFlow/interface/bit_encoding.h"

namespace l1ct {

  struct HadCaloObj {
    pt_t hwPt;
    eta_t hwEta;  // relative to the region center, at calo
    phi_t hwPhi;  // relative to the region center, at calo
    pt_t hwEmPt;
    emid_t hwEmID;

    inline bool operator==(const HadCaloObj &other) const {
      return hwPt == other.hwPt && hwEta == other.hwEta && hwPhi == other.hwPhi && hwEmPt == other.hwEmPt &&
             hwEmID == other.hwEmID;
    }

    inline bool operator>(const HadCaloObj &other) const { return hwPt > other.hwPt; }
    inline bool operator<(const HadCaloObj &other) const { return hwPt < other.hwPt; }

    inline void clear() {
      hwPt = 0;
      hwEta = 0;
      hwPhi = 0;
      hwEmPt = 0;
      hwEmID = 0;
    }

    int intPt() const { return Scales::intPt(hwPt); }
    int intEmPt() const { return Scales::intPt(hwEmPt); }
    int intEta() const { return hwEta.to_int(); }
    int intPhi() const { return hwPhi.to_int(); }
    float floatPt() const { return Scales::floatPt(hwPt); }
    float floatEmPt() const { return Scales::floatPt(hwEmPt); }
    float floatEta() const { return Scales::floatEta(hwEta); }
    float floatPhi() const { return Scales::floatPhi(hwPhi); }

    bool hwIsEM() const { return hwEmID != 0; }

    static const int BITWIDTH = pt_t::width + eta_t::width + phi_t::width + pt_t::width + emid_t::width;
    inline ap_uint<BITWIDTH> pack() const {
      ap_uint<BITWIDTH> ret;
      unsigned int start = 0;
      pack_into_bits(ret, start, hwPt);
      pack_into_bits(ret, start, hwEta);
      pack_into_bits(ret, start, hwPhi);
      pack_into_bits(ret, start, hwEmPt);
      pack_into_bits(ret, start, hwEmID);
      return ret;
    }
    inline static HadCaloObj unpack(const ap_uint<BITWIDTH> &src) {
      HadCaloObj ret;
      unsigned int start = 0;
      unpack_from_bits(src, start, ret.hwPt);
      unpack_from_bits(src, start, ret.hwEta);
      unpack_from_bits(src, start, ret.hwPhi);
      unpack_from_bits(src, start, ret.hwEmPt);
      unpack_from_bits(src, start, ret.hwEmID);
      return ret;
    }
  };

  inline void clear(HadCaloObj &c) { c.clear(); }

  struct EmCaloObj {
    pt_t hwPt, hwPtErr;
    eta_t hwEta;  // relative to the region center, at calo
    phi_t hwPhi;  // relative to the region center, at calo
    emid_t hwEmID;

    inline bool operator==(const EmCaloObj &other) const {
      return hwPt == other.hwPt && hwEta == other.hwEta && hwPhi == other.hwPhi && hwPtErr == other.hwPtErr &&
             hwEmID == other.hwEmID;
    }

    inline bool operator>(const EmCaloObj &other) const { return hwPt > other.hwPt; }
    inline bool operator<(const EmCaloObj &other) const { return hwPt < other.hwPt; }

    inline void clear() {
      hwPt = 0;
      hwPtErr = 0;
      hwEta = 0;
      hwPhi = 0;
      hwEmID = 0;
    }

    int intPt() const { return Scales::intPt(hwPt); }
    int intPtErr() const { return Scales::intPt(hwPtErr); }
    int intEta() const { return hwEta.to_int(); }
    int intPhi() const { return hwPhi.to_int(); }
    float floatPt() const { return Scales::floatPt(hwPt); }
    float floatPtErr() const { return Scales::floatPt(hwPtErr); }
    float floatEta() const { return Scales::floatEta(hwEta); }
    float floatPhi() const { return Scales::floatPhi(hwPhi); }

    static const int BITWIDTH = pt_t::width + eta_t::width + phi_t::width + pt_t::width + emid_t::width;
    inline ap_uint<BITWIDTH> pack() const {
      ap_uint<BITWIDTH> ret;
      unsigned int start = 0;
      pack_into_bits(ret, start, hwPt);
      pack_into_bits(ret, start, hwEta);
      pack_into_bits(ret, start, hwPhi);
      pack_into_bits(ret, start, hwPtErr);
      pack_into_bits(ret, start, hwEmID);
      return ret;
    }
    inline static EmCaloObj unpack(const ap_uint<BITWIDTH> &src) {
      EmCaloObj ret;
      unsigned int start = 0;
      unpack_from_bits(src, start, ret.hwPt);
      unpack_from_bits(src, start, ret.hwEta);
      unpack_from_bits(src, start, ret.hwPhi);
      unpack_from_bits(src, start, ret.hwPtErr);
      unpack_from_bits(src, start, ret.hwEmID);
      return ret;
    }
  };
  inline void clear(EmCaloObj &c) { c.clear(); }

  struct TkObj {
    pt_t hwPt;
    eta_t hwEta;      // relative to the region center, at calo
    phi_t hwPhi;      // relative to the region center, at calo
    tkdeta_t hwDEta;  //  vtx - calo
    tkdphi_t hwDPhi;  // |vtx - calo| (sign is derived by the charge)
    bool hwCharge;    // 1 = positive, 0 = negative
    z0_t hwZ0;
    dxy_t hwDxy;
    tkquality_t hwQuality;

    enum TkQuality { PFLOOSE = 1, PFTIGHT = 2 };
    bool isPFLoose() const { return hwQuality[0]; }
    bool isPFTight() const { return hwQuality[1]; }
    phi_t hwVtxPhi() const { return hwCharge ? hwPhi + hwDPhi : hwPhi - hwDPhi; }
    eta_t hwVtxEta() const { return hwEta + hwDEta; }

    inline bool operator==(const TkObj &other) const {
      return hwPt == other.hwPt && hwEta == other.hwEta && hwPhi == other.hwPhi && hwDEta == other.hwDEta &&
             hwDPhi == other.hwDPhi && hwZ0 == other.hwZ0 && hwDxy == other.hwDxy && hwCharge == other.hwCharge &&
             hwQuality == other.hwQuality;
    }

    inline bool operator>(const TkObj &other) const { return hwPt > other.hwPt; }
    inline bool operator<(const TkObj &other) const { return hwPt < other.hwPt; }

    inline void clear() {
      hwPt = 0;
      hwEta = 0;
      hwPhi = 0;
      hwDEta = 0;
      hwDPhi = 0;
      hwZ0 = 0;
      hwDxy = 0;
      hwCharge = false;
      hwQuality = 0;
    }

    int intPt() const { return Scales::intPt(hwPt); }
    int intEta() const { return hwEta.to_int(); }
    int intPhi() const { return hwPhi.to_int(); }
    int intVtxEta() const { return hwVtxEta().to_int(); }
    int intVtxPhi() const { return hwVtxPhi().to_int(); }
    int intCharge() const { return hwCharge ? +1 : -1; }
    float floatPt() const { return Scales::floatPt(hwPt); }
    float floatEta() const { return Scales::floatEta(hwEta); }
    float floatPhi() const { return Scales::floatPhi(hwPhi); }
    float floatDEta() const { return Scales::floatEta(hwDEta); }
    float floatDPhi() const { return Scales::floatPhi(hwDPhi); }
    float floatVtxEta() const { return Scales::floatEta(hwVtxEta()); }
    float floatVtxPhi() const { return Scales::floatPhi(hwVtxPhi()); }
    float floatZ0() const { return Scales::floatZ0(hwZ0); }
    float floatDxy() const { return Scales::floatDxy(hwDxy); }

    static const int BITWIDTH = pt_t::width + eta_t::width + phi_t::width + tkdeta_t::width + tkdphi_t::width + 1 +
                                z0_t::width + dxy_t::width + tkquality_t::width;
    inline ap_uint<BITWIDTH> pack() const {
      ap_uint<BITWIDTH> ret;
      unsigned int start = 0;
      pack_into_bits(ret, start, hwPt);
      pack_into_bits(ret, start, hwEta);
      pack_into_bits(ret, start, hwPhi);
      pack_into_bits(ret, start, hwDEta);
      pack_into_bits(ret, start, hwDPhi);
      pack_bool_into_bits(ret, start, hwCharge);
      pack_into_bits(ret, start, hwZ0);
      pack_into_bits(ret, start, hwDxy);
      pack_into_bits(ret, start, hwQuality);
      return ret;
    }
    inline static TkObj unpack(const ap_uint<BITWIDTH> &src) {
      TkObj ret;
      unsigned int start = 0;
      unpack_from_bits(src, start, ret.hwPt);
      unpack_from_bits(src, start, ret.hwEta);
      unpack_from_bits(src, start, ret.hwPhi);
      unpack_from_bits(src, start, ret.hwDEta);
      unpack_from_bits(src, start, ret.hwDPhi);
      unpack_bool_from_bits(src, start, ret.hwCharge);
      unpack_from_bits(src, start, ret.hwZ0);
      unpack_from_bits(src, start, ret.hwDxy);
      unpack_from_bits(src, start, ret.hwQuality);
      return ret;
    }
  };
  inline void clear(TkObj &c) { c.clear(); }

  struct MuObj {
    pt_t hwPt;
    glbeta_t hwEta;   // relative to the region center, at calo
    glbphi_t hwPhi;   // relative to the region center, at calo
    tkdeta_t hwDEta;  //  vtx - calo
    tkdphi_t hwDPhi;  // |vtx - calo| (sign is derived by the charge)
    bool hwCharge;    // 1 = positive, 0 = negative
    z0_t hwZ0;
    dxy_t hwDxy;
    ap_uint<3> hwQuality;
    glbphi_t hwVtxPhi() const { return hwCharge ? hwPhi + hwDPhi : hwPhi - hwDPhi; }
    glbeta_t hwVtxEta() const { return hwEta + hwDEta; }

    inline bool operator==(const MuObj &other) const {
      return hwPt == other.hwPt && hwEta == other.hwEta && hwPhi == other.hwPhi && hwDEta == other.hwDEta &&
             hwDPhi == other.hwDPhi && hwZ0 == other.hwZ0 && hwDxy == other.hwDxy && hwCharge == other.hwCharge &&
             hwQuality == other.hwQuality;
    }

    inline bool operator>(const MuObj &other) const { return hwPt > other.hwPt; }
    inline bool operator<(const MuObj &other) const { return hwPt < other.hwPt; }

    inline void clear() {
      hwPt = 0;
      hwEta = 0;
      hwPhi = 0;
      hwDEta = 0;
      hwDPhi = 0;
      hwZ0 = 0;
      hwDxy = 0;
      hwCharge = false;
      hwQuality = 0;
    }

    int intPt() const { return Scales::intPt(hwPt); }
    int intEta() const { return hwEta.to_int(); }
    int intPhi() const { return hwPhi.to_int(); }
    int intVtxEta() const { return hwVtxEta().to_int(); }
    int intVtxPhi() const { return hwVtxPhi().to_int(); }
    int intCharge() const { return hwCharge ? +1 : -1; }
    float floatPt() const { return Scales::floatPt(hwPt); }
    float floatEta() const { return Scales::floatEta(hwEta); }
    float floatPhi() const { return Scales::floatPhi(hwPhi); }
    float floatDEta() const { return Scales::floatEta(hwDEta); }
    float floatDPhi() const { return Scales::floatPhi(hwDPhi); }
    float floatVtxEta() const { return Scales::floatEta(hwVtxEta()); }
    float floatVtxPhi() const { return Scales::floatPhi(hwVtxPhi()); }
    float floatZ0() const { return Scales::floatZ0(hwZ0); }
    float floatDxy() const { return Scales::floatDxy(hwDxy); }

    static const int BITWIDTH = pt_t::width + glbeta_t::width + glbphi_t::width + tkdeta_t::width + tkdphi_t::width +
                                1 + z0_t::width + dxy_t::width + ap_uint<3>::width;
    inline ap_uint<BITWIDTH> pack() const {
      ap_uint<BITWIDTH> ret;
      unsigned int start = 0;
      pack_into_bits(ret, start, hwPt);
      pack_into_bits(ret, start, hwEta);
      pack_into_bits(ret, start, hwPhi);
      pack_into_bits(ret, start, hwDEta);
      pack_into_bits(ret, start, hwDPhi);
      pack_bool_into_bits(ret, start, hwCharge);
      pack_into_bits(ret, start, hwZ0);
      pack_into_bits(ret, start, hwDxy);
      pack_into_bits(ret, start, hwQuality);
      return ret;
    }
    inline static MuObj unpack(const ap_uint<BITWIDTH> &src) {
      MuObj ret;
      unsigned int start = 0;
      unpack_from_bits(src, start, ret.hwPt);
      unpack_from_bits(src, start, ret.hwEta);
      unpack_from_bits(src, start, ret.hwPhi);
      unpack_from_bits(src, start, ret.hwDEta);
      unpack_from_bits(src, start, ret.hwDPhi);
      unpack_bool_from_bits(src, start, ret.hwCharge);
      unpack_from_bits(src, start, ret.hwZ0);
      unpack_from_bits(src, start, ret.hwDxy);
      unpack_from_bits(src, start, ret.hwQuality);
      return ret;
    }
  };
  inline void clear(MuObj &c) { c.clear(); }

  struct PVObj {
    z0_t hwZ0;

    inline bool operator==(const PVObj &other) const { return hwZ0 == other.hwZ0; }

    inline void clear() { hwZ0 = 0; }

    float floatZ0() const { return Scales::floatZ0(hwZ0); }

    static const int BITWIDTH = z0_t::width;
    inline ap_uint<BITWIDTH> pack() const {
      ap_uint<BITWIDTH> ret;
      unsigned int start = 0;
      pack_into_bits(ret, start, hwZ0);
      return ret;
    }
    inline static PVObj unpack(const ap_uint<BITWIDTH> &src) {
      PVObj ret;
      unsigned int start = 0;
      unpack_from_bits(src, start, ret.hwZ0);
      return ret;
    }
  };
  inline void clear(PVObj &c) { c.clear(); }

}  // namespace l1ct

#endif
