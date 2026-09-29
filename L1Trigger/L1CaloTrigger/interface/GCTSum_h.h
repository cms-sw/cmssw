//------------------------------------
// Header and data formats for Phase2L1GCTSumEmulator
// (based heavily on algo_top.h in GCT Sum firmware repo)
//------------------------------------
#ifndef L1Trigger_L1CaloTrigger_GCTSum_h
#define L1Trigger_L1CaloTrigger_GCTSum_h

#include <ap_int.h>
#include <algorithm>
#include <cstdint>
#include <iostream>

namespace p2gctsum {

static constexpr int N_INPUT_LINKS = 12;
static constexpr int N_OUTPUT_LINKS = 3;
static constexpr int N_GCT_CONNECTED = 4;
static constexpr int N_GCT_OBJECTS_PER_SOURCE = 6;
static constexpr int N_GCT_OBJECTS = 24;
static constexpr int N_GCT_OBJECTS_SORT = 32;
static constexpr int N_GCT_SUMS = 1;

typedef ap_uint<10> loop;
typedef ap_uint<48> xvar;

class GCTvar {
public:
  ap_uint<12> ET;
  ap_uint<10> Eta;
  ap_uint<9> Phi;
  ap_uint<4> PtClusterSeed;
  ap_uint<1> isBarrel;

  GCTvar() { ET = 0; Eta = 0; Phi = 0; PtClusterSeed = 0; isBarrel = 0; }

  GCTvar(const GCTvar& rhs) {
    ET = rhs.ET;
    Eta = rhs.Eta;
    Phi = rhs.Phi;
    PtClusterSeed = rhs.PtClusterSeed;
    isBarrel = rhs.isBarrel;
  }

  GCTvar& operator=(const GCTvar& rhs) {
    this->ET = rhs.ET;
    this->Eta = rhs.Eta;
    this->Phi = rhs.Phi;
    this->PtClusterSeed = rhs.PtClusterSeed;
    this->isBarrel = rhs.isBarrel;
    return *this;
  }

  void unpack(ap_uint<48> i, bool hasSeed) {
    this->ET = i.range(11, 0);
    this->isBarrel = i.range(47, 47);

    if (hasSeed) {
      this->Eta = i.range(17, 12);
      this->Phi = i.range(26, 18);
      this->PtClusterSeed = i.range(30, 27);
    } else {
      this->Eta = i.range(18, 12);
      this->Phi = i.range(27, 19);
      this->PtClusterSeed = 0;
    }
  }

  void getGCTvarBarrelGammas(ap_uint<48> i, ap_uint<9> phiOffset = 0) {
    this->ET = i.range(11, 0);
    this->Eta = i.range(18, 12);
    ap_uint<9> raw_phi = i.range(25, 19);
    this->Phi = (raw_phi + phiOffset) & 0x1FF;
    this->isBarrel = 1;
    this->PtClusterSeed = 0;
  }

  void getGCTvarEndcapGammas(ap_uint<48> i, ap_uint<9> phiOffset = 0) {
    this->ET = i.range(11, 0);
    this->Eta = (ap_uint<10>)i.range(18, 12) + 85;
    ap_uint<9> raw_phi = i.range(27, 19);
    this->Phi = (raw_phi + phiOffset) & 0x1FF;
    this->isBarrel = 0;
    this->PtClusterSeed = 0;
  }

  void getGCTvarJetsTaus(ap_uint<48> i, ap_uint<9> phiOffset, bool barrelFlag) {
    this->ET = i.range(11, 0);
    ap_uint<6> raw_eta = i.range(17, 12);
    ap_uint<10> eta = barrelFlag ? (ap_uint<10>)raw_eta : (ap_uint<10>)(raw_eta + 6);
    this->Eta = eta;
    ap_uint<9> raw_phi = i.range(26, 18);
    this->Phi = (raw_phi + phiOffset) & 0x1FF;
    this->PtClusterSeed = i.range(30, 27);
    this->isBarrel = (barrelFlag ? 1 : 0);
  }

  ap_uint<48> pack() const {
    ap_uint<48> out = 0;

    if (isBarrel) {
      bool isGammaFormat = (Eta > 0x3F);

      if (isGammaFormat) {
        out = ((ap_uint<48>)isBarrel << 47) |
              ((ap_uint<48>)(Phi & 0x7F) << 19) |
              ((ap_uint<48>)Eta << 12) |
              (ap_uint<48>)ET;
      } else {
        out = ((ap_uint<48>)isBarrel << 47) |
              ((ap_uint<48>)PtClusterSeed << 27) |
              ((ap_uint<48>)Phi << 18) |
              ((ap_uint<48>)Eta << 12) |
              (ap_uint<48>)ET;
      }
    } else {
      if (PtClusterSeed == 0 && Eta > 0x3F) {
        out = ((ap_uint<48>)isBarrel << 47) |
              ((ap_uint<48>)Phi << 19) |
              ((ap_uint<48>)Eta << 12) |
              (ap_uint<48>)ET;
      } else {
        out = ((ap_uint<48>)isBarrel << 47) |
              ((ap_uint<48>)PtClusterSeed << 27) |
              ((ap_uint<48>)Phi << 18) |
              ((ap_uint<48>)Eta << 12) |
              (ap_uint<48>)ET;
      }
    }

    return out;
  }
};

class GCTsum {
public:
  ap_int<16> Ex, Ey, Htx, Hty;
  ap_uint<16> SumET, Ht, NObj;

  GCTsum() : Ex(0), Ey(0), Htx(0), Hty(0), SumET(0), Ht(0), NObj(0) {}

  // Sums ABI v2: source vector components are already in the global phi
  // frame. Four 12-bit source lanes fit losslessly in these 16-bit lanes.
  void addSource(const ap_uint<576>& input) {
    Ex += (ap_int<12>)input.range(11, 0);
    Ey += (ap_int<12>)input.range(23, 12);
    Htx += (ap_int<12>)input.range(35, 24);
    Hty += (ap_int<12>)input.range(47, 36);
    SumET += (ap_uint<12>)input.range(75, 64);
    Ht += (ap_uint<12>)input.range(87, 76);
    NObj += (ap_uint<12>)input.range(99, 88);
  }

  // Internal SUM -> TO_GT transport: four vector lanes in word 0 and three
  // scalar lanes in word 1. All remaining bits are reserved zero.
  ap_uint<576> pack() const {
    ap_uint<576> out = 0;
    out.range(15, 0) = Ex;
    out.range(31, 16) = Ey;
    out.range(47, 32) = Htx;
    out.range(63, 48) = Hty;
    out.range(79, 64) = SumET;
    out.range(95, 80) = Ht;
    out.range(111, 96) = NObj;
    return out;
  }

  void unpack(const ap_uint<576>& input) {
    Ex = (ap_int<16>)input.range(15, 0);
    Ey = (ap_int<16>)input.range(31, 16);
    Htx = (ap_int<16>)input.range(47, 32);
    Hty = (ap_int<16>)input.range(63, 48);
    SumET = (ap_uint<16>)input.range(79, 64);
    Ht = (ap_uint<16>)input.range(95, 80);
    NObj = (ap_uint<16>)input.range(111, 96);
  }
};

namespace gcts {

// Fixed-iteration restoring square root: exact floor(sqrt(n)).
inline ap_uint<24> isqrt48(ap_uint<48> n) {
  ap_uint<50> remainder = 0;
  ap_uint<24> root = 0;
  for (int i = 23; i >= 0; --i) {
    remainder = (remainder << 2) | ((n >> (2 * i)) & 3);
    root <<= 1;
    const ap_uint<26> trial = ((ap_uint<26>)root << 1) | 1;
    if (remainder >= trial) {
      remainder -= trial;
      root += 1;
    }
  }
  return root;
}

// CORDIC vectoring in global coordinates. Output is two's-complement phi,
// LSB = pi/4096, nearest code; +pi wraps to -pi. Zero has phi=0.
inline ap_int<13> vectorPhi(ap_int<18> inputX, ap_int<18> inputY) {
  static const long long angles[24] = {1073741824LL,
                                      633866811LL,
                                      334917815LL,
                                      170009512LL,
                                      85334662LL,
                                      42708931LL,
                                      21359677LL,
                                      10680490LL,
                                      5340327LL,
                                      2670173LL,
                                      1335088LL,
                                      667544LL,
                                      333772LL,
                                      166886LL,
                                      83443LL,
                                      41722LL,
                                      20861LL,
                                      10430LL,
                                      5215LL,
                                      2608LL,
                                      1304LL,
                                      652LL,
                                      326LL,
                                      163LL};
  ap_int<48> x = (ap_int<48>)inputX << 24;
  ap_int<48> y = (ap_int<48>)inputY << 24;
  ap_int<35> angle = 0;
  if (x < 0) {
    angle = y < 0 ? -4294967296LL : 4294967296LL;
    x = -x;
    y = -y;
  }
  for (int i = 0; i < 24; ++i) {
    const ap_int<48> oldX = x;
    if (y > 0) {
      x += y >> i;
      y -= oldX >> i;
      angle += angles[i];
    } else if (y < 0) {
      x -= y >> i;
      y += oldX >> i;
      angle -= angles[i];
    }
  }
  const ap_int<35> absoluteAngle = angle < 0 ? (ap_int<35>)(-angle) : angle;
  ap_int<15> code = (absoluteAngle + 524288) >> 20;
  if (angle < 0)
    code = -code;
  return (ap_int<13>)code;
}

inline ap_uint<16> saturate16(ap_uint<24> value) {
  return value > 65535 ? ap_uint<16>(65535) : ap_uint<16>(value);
}

// Components are combined before taking the negative vector. Magnitude and
// scalar values are converted from 0.5 GeV to the GT 1/32 GeV convention.
inline ap_uint<64> packVectorSum(ap_int<17> x, ap_int<17> y, ap_uint<17> scalar) {
  const ap_int<34> x2 = (ap_int<34>)x * (ap_int<34>)x;
  const ap_int<34> y2 = (ap_int<34>)y * (ap_int<34>)y;
  const ap_uint<35> square = (ap_uint<35>)x2 + (ap_uint<35>)y2;
  const ap_uint<48> scaledSquare = (ap_uint<48>)square << 8;
  ap_uint<64> out = 0;
  out[0] = 1;
  out.range(16, 1) = saturate16(isqrt48(scaledSquare));
  out.range(29, 17) = vectorPhi(-(ap_int<18>)x, -(ap_int<18>)y);
  out.range(45, 30) = saturate16((ap_uint<24>)scalar << 4);
  return out;
}

}  // namespace gcts

static const ap_uint<10> BARREL_GAMMA_BOUNDARY_ETA = 84;
static const ap_uint<10> ENDCAP_GAMMA_BOUNDARY_ETA = 85;
static const ap_uint<10> BARREL_HAD_BOUNDARY_ETA = 5;
static const ap_uint<10> ENDCAP_HAD_BOUNDARY_ETA = 6;

inline bool match_dphi_gamma(ap_uint<9> ephi, ap_uint<9> bphi) {
  ap_int<10> dphi = (ap_int<10>)ephi - (ap_int<10>)bphi;
  if (dphi > 180)
    dphi -= 360;
  if (dphi < -180)
    dphi += 360;
  return (dphi >= -1 && dphi <= 1);
}

inline bool match_dphi_had(ap_uint<9> ephi, ap_uint<9> bphi) {
  ap_int<10> dphi = (ap_int<10>)ephi - (ap_int<10>)bphi;
  if (dphi > 12)
    dphi -= 24;
  if (dphi < -12)
    dphi += 24;
  return (dphi >= -1 && dphi <= 1);
}

void algo_top(ap_uint<576> link_in[N_INPUT_LINKS], ap_uint<576> link_out[N_OUTPUT_LINKS]);

}  // namespace p2gctsum

#endif
