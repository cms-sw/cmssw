#ifndef DataFormats_L1TParticleFlow_taus_h
#define DataFormats_L1TParticleFlow_taus_h

#include "DataFormats/L1TParticleFlow/interface/datatypes.h"
#include "DataFormats/L1TParticleFlow/interface/bit_encoding.h"
#include "DataFormats/L1TParticleFlow/interface/gt_datatypes.h"

namespace l1ct {

  struct Tau {
    typedef ap_uint<2> type_t;
    typedef ap_uint<10> rawid_t;
    typedef ap_uint<2> lepid_t;

    pt_t hwPt;
    glbeta_t hwEta;
    glbphi_t hwPhi;
    pt_t hwSeedPt;
    z0_t hwSeedZ0;
    bool hwCharge;
    type_t hwType;
    rawid_t hwRawId;  // will contain isolation or MVA output
    lepid_t hwIdVsMu;
    lepid_t hwIdVsEle;
    rawid_t hwIsoOrMVA;

    inline bool operator==(const Tau &other) const {
      return hwPt == other.hwPt && hwEta == other.hwEta && hwPhi == other.hwPhi && hwSeedPt == other.hwSeedPt &&
             hwSeedZ0 == other.hwSeedZ0 && hwCharge == other.hwCharge && hwType == other.hwType &&
             hwIsoOrMVA == other.hwIsoOrMVA && hwIdVsMu == other.hwIdVsMu && hwIdVsEle == other.hwIdVsEle;
    }

    inline bool operator>(const Tau &other) const { return hwPt > other.hwPt; }
    inline bool operator<(const Tau &other) const { return hwPt < other.hwPt; }

    inline pt_t hwAbsIso() const {
      pt10_t ret;
      ret(9, 0) = hwRawId(9, 0);
      return ret;
    }

    inline void setAbsIso(pt10_t absIso) { hwRawId(9, 0) = absIso(9, 0); }

    inline void clear() {
      hwPt = 0;
      hwEta = 0;
      hwPhi = 0;
      hwSeedPt = 0;
      hwSeedZ0 = 0;
      hwCharge = false;
      hwType = 0;
      hwIsoOrMVA = 0;
      hwIdVsMu = 0;
      hwIdVsEle = 0;
      hwIsoOrMVA = 0;
    }

    int intPt() const { return Scales::intPt(hwPt); }
    int intEta() const { return hwEta.to_int(); }
    int intPhi() const { return hwPhi.to_int(); }
    int intSeedPt() const { return Scales::intPt(hwSeedPt); }
    float floatPt() const { return Scales::floatPt(hwPt); }
    float floatEta() const { return Scales::floatEta(hwEta); }
    float floatPhi() const { return Scales::floatPhi(hwPhi); }
    float floatSeedPt() const { return Scales::floatPt(hwSeedPt); }
    float floatSeedZ0() const { return Scales::floatZ0(hwSeedZ0); }
    int intCharge() const { return hwCharge ? +1 : -1; }
    int pdgId() const { return -15 * intCharge(); }
    int intType() const { return hwType.to_int(); }

    float floatAbsIso() const { return Scales::floatPt(hwAbsIso()); }

    static const int BITWIDTH = pt_t::width + glbeta_t::width + glbphi_t::width + pt10_t::width + z0_t::width + 1 +
                                type_t::width + rawid_t::width + 2 * lepid_t::width;
    inline ap_uint<BITWIDTH> pack() const {
      ap_uint<BITWIDTH> ret;
      unsigned int start = 0;
      pack_into_bits(ret, start, hwPt);
      pack_into_bits(ret, start, hwEta);
      pack_into_bits(ret, start, hwPhi);
      pack_into_bits(ret, start, hwSeedPt);
      pack_into_bits(ret, start, hwSeedZ0);
      pack_bool_into_bits(ret, start, hwCharge);
      pack_into_bits(ret, start, hwType);
      pack_into_bits(ret, start, hwRawId);
      pack_into_bits(ret, start, hwIdVsMu);
      pack_into_bits(ret, start, hwIdVsEle);
      pack_into_bits(ret, start, hwIsoOrMVA);
      return ret;
    }

    inline static Tau unpack(const ap_uint<BITWIDTH> &src) {
      Tau ret;
      unsigned int start = 0;
      unpack_from_bits(src, start, ret.hwPt);
      unpack_from_bits(src, start, ret.hwEta);
      unpack_from_bits(src, start, ret.hwPhi);
      unpack_from_bits(src, start, ret.hwSeedPt);
      unpack_from_bits(src, start, ret.hwSeedZ0);
      unpack_from_bits(src, start, ret.hwType);
      unpack_from_bits(src, start, ret.hwRawId);
      unpack_from_bits(src, start, ret.hwIdVsMu);
      unpack_from_bits(src, start, ret.hwIdVsEle);
      unpack_from_bits(src, start, ret.hwIsoOrMVA);
      return ret;
    }

    l1gt::Tau toGT() const {
      l1gt::Tau t;
      t.valid = hwPt != 0;

      t.v3.pt = CTtoGT_pt(hwPt);
      t.v3.phi = CTtoGT_phi(hwPhi);
      t.v3.eta = CTtoGT_eta(hwEta);

      t.seed_pt = hwSeedPt;
      t.seed_z0 = hwSeedZ0;
      t.charge = hwCharge;

      t.type = hwType;
      t.isolation = hwRawId;
      return t;
    }
  };

  inline void clear(Tau &c) { c.clear(); }

}  // namespace l1ct

#endif
