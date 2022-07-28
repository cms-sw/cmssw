#ifndef L1Trigger_Phase2L1ParticleFlow_HTMHT_h
#define L1Trigger_Phase2L1ParticleFlow_HTMHT_h

#include "DataFormats/L1TParticleFlow/interface/jets.h"
#include "DataFormats/L1TParticleFlow/interface/sums.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/dbgPrintf.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/jetmet/L1SeedConePFJetEmulator.h"

#ifndef CMSSW_GIT_HASH
#include "hls_math.h"
#endif

#include <vector>
#include <numeric>
#include <algorithm>
#include "ap_int.h"
#include "ap_fixed.h"

namespace P2L1HTMHTEmu {
  typedef l1ct::pt_t pt_t;          // Type for pt/ht 1 unit = 0.25 GeV; max = 16 TeV
  typedef l1ct::glbeta_t etaphi_t;  // Type for eta & phi

  typedef ap_fixed<12, 3> radians_t;
  typedef ap_fixed<9, 2> cossin_t;
  typedef ap_fixed<16, 13> pxy_t;

  static constexpr int N_TABLE = 2048;

  // Class for intermediate variables
  class PtPxPy {
  public:
    pt_t pt = 0.;
    pxy_t px = 0.;
    pxy_t py = 0.;

    PtPxPy operator+(const PtPxPy& b) const {
      PtPxPy c;
      c.pt = this->pt + b.pt;
      c.px = this->px + b.px;
      c.py = this->py + b.py;
      return c;
    }
  };

  namespace Scales {
    const ap_fixed<12, -4> scale_degToRad = M_PI / 180.;
  };  // namespace Scales

  template <class data_T, class table_T, int N>
  void init_sinphi_table(table_T table_out[N]) {
    for (int i = 0; i < N; i++) {
      float x = i * (M_PI / 180.) / 2.;
      table_T sin_x = std::sin(x);
      table_out[i] = sin_x;
    }
  }
  template <class in_t, class table_t, int N>
  table_t sine_with_conversion(etaphi_t hwPhi) {
    table_t sin_table[N];
    init_sinphi_table<in_t, table_t, N>(sin_table);
    table_t out = sin_table[hwPhi];
    return out;
  }

  inline etaphi_t phi_cordic(pxy_t y, pxy_t x) {
#ifdef CMSSW_GIT_HASH
    ap_fixed<12, 3> phi = atan2(y.to_double(), x.to_double());  // hls_math.h not available yet in CMSSW
#else
    ap_fixed<12, 3> phi = hls::atan2(y, x);
#endif
    ap_fixed<16, 9> etaphiscale = (float)l1ct::Scales::INTPHI_PI / M_PI;  // radians to hwPhi
    return phi * etaphiscale;
  }

  inline PtPxPy mht_compute(l1ct::Jet jet) {
    // Add an extra bit to px/py for the sign, and one additional bit to improve precision (pt_t is ap_ufixed<14, 12>)
    PtPxPy v_pxpy;

    //Initialize table once
    cossin_t sin_table[N_TABLE];
    init_sinphi_table<etaphi_t, cossin_t, N_TABLE>(sin_table);

    cossin_t sinphi;
    cossin_t cosphi;
    bool sign = jet.hwPhi.sign();

    etaphi_t hwphi = jet.hwPhi;

    // Reduce precision of hwPhi
    ap_int<10> phi;
    phi.V = hwphi(11, 1);
    phi = (phi > 0) ? phi : (ap_int<10>)-phi;  //Only store values for positive phi, pick up sign later

    sinphi = sin_table[phi];

    sinphi = (sign > 0) ? (cossin_t)(-sign * sinphi) : sinphi;  // Change sign bit if hwPt is negative, sin(-x)=-sin(x)
    cosphi = sin_table[phi + 90 * 2];  //cos(x)=sin(x+90). Do nothing with sign, cos(-θ) = cos θ,

    v_pxpy.pt = jet.hwPt;
    v_pxpy.py = jet.hwPt * sinphi;
    v_pxpy.px = jet.hwPt * cosphi;

    return v_pxpy;
  }
}  // namespace P2L1HTMHTEmu

//TODO replace with l1ct::Jet
inline l1ct::Sum htmht(std::vector<l1ct::Jet> jets) {
  // compute jet px, py
  std::vector<P2L1HTMHTEmu::PtPxPy> ptpxpy;
  ptpxpy.resize(jets.size());
  std::transform(
      jets.begin(), jets.end(), ptpxpy.begin(), [](const l1ct::Jet& jet) { return P2L1HTMHTEmu::mht_compute(jet); });

  // Sum pt, px, py over jets
  P2L1HTMHTEmu::PtPxPy hthxhy = std::accumulate(ptpxpy.begin(), ptpxpy.end(), P2L1HTMHTEmu::PtPxPy());

  // Compute the MHT magnitude and direction
  l1ct::Sum ht;
  ht.hwSumPt = hthxhy.pt;
#ifdef CMSSW_GIT_HASH
  ht.hwPt =
      sqrt(((hthxhy.px * hthxhy.px) + (hthxhy.py * hthxhy.py)).to_double());  // hls_math.h not available yet in CMSSW
#else
  ht.hwPt = hls::sqrt(((hthxhy.px * hthxhy.px) + (hthxhy.py * hthxhy.py)));
#endif
  ht.hwPhi = P2L1HTMHTEmu::phi_cordic(hthxhy.py, hthxhy.px);
  return ht;
}

#endif
