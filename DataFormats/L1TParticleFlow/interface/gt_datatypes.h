#ifndef DataFormats_L1TParticleFlow_gt_datatypes_h
#define DataFormats_L1TParticleFlow_gt_datatypes_h

#if (!defined(__CLANG__)) && defined(__GNUC__) && defined(CMSSW_GIT_HASH)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuninitialized"
#endif
#include <ap_int.h>
#if (!defined(__CLANG__)) && defined(__GNUC__) && defined(CMSSW_GIT_HASH)
#pragma GCC diagnostic pop
#endif

#include "DataFormats/L1TParticleFlow/interface/bit_encoding.h"
#include <array>
#include <cstdint>

namespace l1gt {
  // Using rounding & saturation modes to avoid unnecessary rounding errors
  // Don't saturate phi since -π to +π periodicity is handled by numerical wrap around
  // Rounding and saturation settings are lost when sending the data over the link
  // Unless the receiving end uses the same data types

  // Common fields
  typedef ap_ufixed<16, 11, AP_RND_CONV, AP_SAT> pt_t;
  typedef ap_fixed<13, 13, AP_RND_CONV> phi_t;
  typedef ap_fixed<14, 14, AP_RND_CONV, AP_SAT> eta_t;
  // While bitwise identical to the l1ct::z0_t value, we store z0 in mm to profit of ap_fixed goodies
  typedef ap_fixed<10, 9, AP_RND_CONV, AP_SAT> z0_t;  // NOTE: mm instead of cm!!!
  typedef ap_uint<1> valid_t;

  // E/gamma fields
  typedef ap_ufixed<11, 9> iso_t;
  typedef ap_uint<4> egquality_t;

  // tau fields
  typedef ap_ufixed<10, 8> tauseed_pt_t;
  typedef ap_uint<10> tau_rawid_t;
  typedef std::array<uint64_t, 2> PackedTau;

  namespace Scales {
    const int INTPHI_PI = 1 << (phi_t::width - 1);
    const float INTPT_LSB = 1.0 / (1 << (pt_t::width - pt_t::iwidth));
    const int INTPHI_TWOPI = 2 * INTPHI_PI;
    constexpr float ETAPHI_LSB = M_PI / INTPHI_PI;
    constexpr float Z0_UNITS = 0.1;  // 1 L1 unit is 1 mm, while CMS standard units are cm
    inline float floatPt(pt_t pt) { return pt.to_float(); }
    inline float floatEta(eta_t eta) { return eta.to_float() * ETAPHI_LSB; }
    inline float floatPhi(phi_t phi) { return phi.to_float() * ETAPHI_LSB; }
    inline float floatZ0(z0_t z0) { return z0.to_float() * Z0_UNITS; }
  }  // namespace Scales

  struct ThreeVector {
    pt_t pt;
    phi_t phi;
    eta_t eta;

    inline bool operator==(const ThreeVector &other) const {
      return pt == other.pt && phi == other.phi && eta == other.eta;
    }

    static const int BITWIDTH = pt_t::width + phi_t::width + eta_t::width;
    inline ap_uint<BITWIDTH> pack() const {
      ap_uint<BITWIDTH> ret;
      unsigned int start = 0;
      pack_into_bits(ret, start, pt);
      pack_into_bits(ret, start, phi);
      pack_into_bits(ret, start, eta);
      return ret;
    }

    inline static ThreeVector unpack_ap(const ap_uint<BITWIDTH> &src) {
      ThreeVector ret;
      ret.initFromBits(src);
      return ret;
    }

    inline void initFromBits(const ap_uint<BITWIDTH> &src) {
      unsigned int start = 0;
      unpack_from_bits(src, start, pt);
      unpack_from_bits(src, start, phi);
      unpack_from_bits(src, start, eta);
    }
  };

  struct Jet {
    valid_t valid;
    ThreeVector v3;
    z0_t z0;

    inline bool operator==(const Jet &other) const { return valid == other.valid && z0 == other.z0 && v3 == other.v3; }

    static const int BITWIDTH = 128;
    inline ap_uint<BITWIDTH> pack_ap() const {
      ap_uint<BITWIDTH> ret = 0;
      unsigned int start = 0;
      pack_into_bits(ret, start, valid);
      pack_into_bits(ret, start, v3.pack());
      pack_into_bits(ret, start, z0);
      return ret;
    }

    inline std::array<uint64_t, 2> pack() const {
      std::array<uint64_t, 2> packed;
      ap_uint<BITWIDTH> bits = this->pack_ap();
      packed[0] = bits(63, 0);
      packed[1] = bits(127, 64);
      return packed;
    }

    inline static Jet unpack_ap(const ap_uint<BITWIDTH> &src) {
      Jet ret;
      ret.initFromBits(src);
      return ret;
    }

    inline void initFromBits(const ap_uint<BITWIDTH> &src) {
      unsigned int start = 0;
      unpack_from_bits(src, start, valid);
      unpack_from_bits(src, start, v3.pt);
      unpack_from_bits(src, start, v3.phi);
      unpack_from_bits(src, start, v3.eta);
      unpack_from_bits(src, start, z0);
    }

    inline static Jet unpack(const std::array<uint64_t, 2> &src) {
      ap_uint<BITWIDTH> bits;
      bits(63, 0) = src[0];
      bits(127, 64) = src[1];
      return unpack_ap(bits);
    }

    inline static Jet unpack(long long unsigned int &src) {
      // unpack from single 64b int
      ap_uint<BITWIDTH> bits = src;
      return unpack_ap(bits);
    }

  };  // struct Jet

  struct Sum {
    valid_t valid;
    pt_t vector_pt;
    phi_t vector_phi;
    pt_t scalar_pt;

    inline bool operator==(const Sum &other) const {
      return valid == other.valid && vector_pt == other.vector_pt && vector_phi == other.vector_phi &&
             scalar_pt == other.scalar_pt;
    }

    inline void clear() {
      valid = 0;
      vector_pt = 0;
      vector_phi = 0;
      scalar_pt = 0;
    }

    static const int BITWIDTH = 64;
    inline ap_uint<BITWIDTH> pack() const {
      ap_uint<BITWIDTH> ret;
      unsigned int start = 0;
      pack_into_bits(ret, start, valid);
      pack_into_bits(ret, start, vector_pt);
      pack_into_bits(ret, start, vector_phi);
      pack_into_bits(ret, start, scalar_pt);
      return ret;
    }

    inline static Sum unpack_ap(const ap_uint<BITWIDTH> &src) {
      Sum ret;
      ret.initFromBits(src);
      return ret;
    }

    inline void initFromBits(const ap_uint<BITWIDTH> &src) {
      unsigned int start = 0;
      unpack_from_bits(src, start, valid);
      unpack_from_bits(src, start, vector_pt);
      unpack_from_bits(src, start, vector_phi);
      unpack_from_bits(src, start, scalar_pt);
    }
  };  // struct Sum

  struct Tau {
    valid_t valid;
    ThreeVector v3;
    tauseed_pt_t seed_pt;
    z0_t seed_z0;
    ap_uint<1> charge;
    ap_uint<2> type;
    tau_rawid_t isolation;
    ap_uint<2> id0;
    ap_uint<2> id1;

    static const int BITWIDTH = 128;
    inline ap_uint<BITWIDTH> pack_ap() const {
      ap_uint<BITWIDTH> ret;
      unsigned int start = 0;
      pack_into_bits(ret, start, valid);
      pack_into_bits(ret, start, v3.pack());
      pack_into_bits(ret, start, seed_pt);
      pack_into_bits(ret, start, seed_z0);
      pack_into_bits(ret, start, charge);
      pack_into_bits(ret, start, type);
      pack_into_bits(ret, start, isolation);
      pack_into_bits(ret, start, id0);
      pack_into_bits(ret, start, id1);
      return ret;
    }

    inline PackedTau pack() const {
      PackedTau packed;
      ap_uint<BITWIDTH> bits = this->pack_ap();
      packed[0] = bits(63, 0);
      packed[1] = bits(127, 64);
      return packed;
    }

    inline static Tau unpack_ap(const ap_uint<BITWIDTH> &src) {
      Tau ret;
      ret.initFromBits(src);
      return ret;
    }

    inline static Tau unpack(const PackedTau &src) {
      ap_uint<BITWIDTH> bits;
      bits(63, 0) = src[0];
      bits(127, 64) = src[1];

      return unpack_ap(bits);
    }

    inline void initFromBits(const ap_uint<BITWIDTH> &src) {
      unsigned int start = 0;
      unpack_from_bits(src, start, valid);
      unpack_from_bits(src, start, v3.pt);
      unpack_from_bits(src, start, v3.phi);
      unpack_from_bits(src, start, v3.eta);
      unpack_from_bits(src, start, seed_pt);
      unpack_from_bits(src, start, seed_z0);
      unpack_from_bits(src, start, charge);
      unpack_from_bits(src, start, type);
      unpack_from_bits(src, start, isolation);
      unpack_from_bits(src, start, id0);
      unpack_from_bits(src, start, id1);
    }

  };  // struct Tau

  struct Electron {
    valid_t valid;
    ThreeVector v3;
    egquality_t quality;
    ap_uint<1> charge;
    z0_t z0;
    iso_t isolation;

    static const int BITWIDTH = 96;
    inline ap_uint<BITWIDTH> pack() const {
      ap_uint<BITWIDTH> ret(0);
      unsigned int start = 0;
      pack_into_bits(ret, start, valid);
      pack_into_bits(ret, start, v3.pack());
      pack_into_bits(ret, start, quality);
      pack_into_bits(ret, start, isolation);
      pack_into_bits(ret, start, charge);
      pack_into_bits(ret, start, z0);
      return ret;
    }

    inline void initFromBits(const ap_uint<BITWIDTH> &src) {
      unsigned int start = 0;
      unpack_from_bits(src, start, valid);
      unpack_from_bits(src, start, v3.pt);
      unpack_from_bits(src, start, v3.phi);
      unpack_from_bits(src, start, v3.eta);
      unpack_from_bits(src, start, quality);
      unpack_from_bits(src, start, isolation);
      unpack_from_bits(src, start, charge);
      unpack_from_bits(src, start, z0);
    }

    inline static Electron unpack_ap(const ap_uint<BITWIDTH> &src) {
      Electron ret;
      ret.initFromBits(src);
      return ret;
    }

    inline static Electron unpack(const std::array<uint64_t, 2> &src, int parity) {
      ap_uint<BITWIDTH> bits;
      if (parity == 0) {
        bits(63, 0) = src[0];
        bits(95, 64) = src[1];
      } else {
        bits(63, 0) = src[1];
        bits(95, 64) = (src[0] >> 32);
      }
      return unpack_ap(bits);
    }
  };

  struct Photon {
    valid_t valid;
    ThreeVector v3;
    egquality_t quality;
    iso_t isolation;

    static const int BITWIDTH = 96;
    inline ap_uint<BITWIDTH> pack() const {
      ap_uint<96> ret(0);
      unsigned int start = 0;
      pack_into_bits(ret, start, valid);
      pack_into_bits(ret, start, v3.pack());
      pack_into_bits(ret, start, quality);
      pack_into_bits(ret, start, isolation);
      return ret;
    }

    inline void initFromBits(const ap_uint<BITWIDTH> &src) {
      unsigned int start = 0;
      unpack_from_bits(src, start, valid);
      unpack_from_bits(src, start, v3.pt);
      unpack_from_bits(src, start, v3.phi);
      unpack_from_bits(src, start, v3.eta);
      unpack_from_bits(src, start, quality);
      unpack_from_bits(src, start, isolation);
    }

    inline static Photon unpack_ap(const ap_uint<BITWIDTH> &src) {
      Photon ret;
      ret.initFromBits(src);
      return ret;
    }

    inline static Photon unpack(const std::array<uint64_t, 2> &src, int parity) {
      ap_uint<BITWIDTH> bits;
      if (parity == 0) {
        bits(63, 0) = src[0];
        bits(95, 64) = src[1];
      } else {
        bits(63, 0) = src[1];
        bits(95, 64) = (src[0] >> 32);
      }
      return unpack_ap(bits);
    }
  };

}  // namespace l1gt

namespace l1ct {

  typedef ap_fixed<18, 5, AP_RND_CONV, AP_SAT> etaphi_sf_t;  // use a DSP input width

  namespace Scales {
    const etaphi_sf_t ETAPHI_CTtoGT_SCALE = (Scales::ETAPHI_LSB / l1gt::Scales::ETAPHI_LSB);
  }

  inline l1gt::pt_t CTtoGT_pt(pt_t x) {
    // the CT & GT pT are both ap_fixed with different power-of-2 LSBs
    // -> conversion is just a cast
    return (l1gt::pt_t)x;
  }

  inline l1gt::eta_t CTtoGT_eta(glbeta_t x) {
    // rescale the eta into the GT coordinates
    return x * Scales::ETAPHI_CTtoGT_SCALE;
  }

  inline l1gt::phi_t CTtoGT_phi(glbphi_t x) {
    // rescale the phi into the GT coordinates
    return x * Scales::ETAPHI_CTtoGT_SCALE;
  }

}  // namespace l1ct

#endif
