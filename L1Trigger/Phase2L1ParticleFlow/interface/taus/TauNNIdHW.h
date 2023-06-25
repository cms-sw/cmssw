#ifndef L1Trigger_Phase2L1ParticleFlow_TAUNNIDHW_H_
#define L1Trigger_Phase2L1ParticleFlow_TAUNNIDHW_H_

#include "DataFormats/L1TParticleFlow/interface/layer1_emulator.h"

#include <cstdio>
#include <complex>
#include "ap_int.h"
#include "ap_fixed.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/taus/tau_parameters.h"

#include "DataFormats/L1TParticleFlow/interface/PFCandidate.h"

#include "L1Trigger/Phase2L1ParticleFlow/interface/common/nnet_layer.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/common/nnet_activation.h"

//hls-fpga-machine-learning insert weights
#include "L1Trigger/Phase2L1ParticleFlow/interface/taus/weights/w1.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/taus/weights/b1.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/taus/weights/w2.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/taus/weights/b2.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/taus/weights/w3.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/taus/weights/b3.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/taus/weights/w4.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/taus/weights/b4.h"

typedef ap_ufixed<16, 14> pt_t;
typedef ap_fixed<10, 4> etaphi_t;

namespace L1TauEmu {
  // Data types and constants used in the FPGA and FPGA-optimized functions
  //etaphi_base maps physical eta phi units onto bits
  //This way, the least significant bit of etaphi_t is exactly 0.01
  //Even though 0.01 is not a power of 2
  static constexpr float etaphi_base = 100. / 64;
  typedef ap_ufixed<16, 14> pt_t;        // 1 unit = 0.25 GeV;
  typedef ap_fixed<10, 4> etaphi_t;      // 1 unit = 0.01;
  typedef ap_fixed<12, 6> detaphi_t;     // type for the difference between etas or phis
  typedef ap_fixed<18, 9> detaphi2_t;    // type for detaphi_t squared
  typedef ap_fixed<22, 16> pt_etaphi_t;  // type for product of pt with deta or phi
  typedef ap_int<8> dxy_t;
  typedef ap_int<10> z0_t;
  typedef ap_uint<5> count_t;  // type for multiplicity
  typedef ap_uint<5> id_t;     // type for multiplicity

  // constants for the axis update
  typedef ap_ufixed<18, -2> inv_pt_t;
  static constexpr int N_table_inv_pt = 1024;
  static const detaphi_t TWOPI = 3.14159 * 2. * etaphi_base;
  static const detaphi_t PI = 3.14159 * etaphi_base;
  static const detaphi_t HALFPI = 3.14159 / 2 * etaphi_base;
  static const detaphi_t RCONE = 0.4 * 100 / 128;
  static const detaphi_t R2CONE = RCONE * RCONE;
  //
  static const etaphi_t FIDUCIAL_ETA_PHI = 5.11 * etaphi_base;

  constexpr int ceillog2(int x) { return (x <= 2) ? 1 : 1 + ceillog2((x + 1) / 2); }
  constexpr int floorlog2(int x) { return (x < 2) ? 0 : 1 + floorlog2(x / 2); }
  constexpr int pow2(int x) { return x == 0 ? 1 : 2 * pow2(x - 1); }

  template <class data_T, int N>
  inline float real_val_from_idx(unsigned i) {
    // Treat the index as the top N bits
    static constexpr int NB = ceillog2(N);  // number of address bits for table
    data_T x(0);
    // The MSB of 1 is implicit in the table
    x[x.width - 1] = 1;
    // So we can use the next NB bits for real data
    x(x.width - 2, x.width - NB - 1) = i;
    return (float)x;
  }

  template <class data_T, int N>
  inline unsigned idx_from_real_val(data_T x) {
    // Slice the top N bits to get an index into the table
    static constexpr int NB = ceillog2(N);  // number of address bits for table
    // Slice the top-1 NB bits of the value
    // the MSB of '1' is implicit, so only slice below that
    ap_uint<NB> y = x(x.width - 2, x.width - NB - 1);
    return (unsigned)y(NB - 1, 0);
  }

  template <class data_T, class table_T, int N>
  void init_invert_table(table_T table_out[N]) {
    // The template data_T is the data type used to address the table
    for (unsigned i = 0; i < N; i++) {
      float x = real_val_from_idx<data_T, N>(i);
      table_T inv_x = 1 / x;
      table_out[i] = inv_x;
    }
  }

  template <class in_t, class table_t, int N>
  table_t invert_with_shift(in_t in, bool debug = false) {
    table_t inv_table[N];
    init_invert_table<in_t, table_t, N>(inv_table);

    // find the first '1' in the denominator
    int msb = 0;
    for (int b = 0; b < in.width; b++) {
      if (in[b])
        msb = b;
    }
    // shift up the denominator such that the left-most bit (msb) is '1'
    in_t in_shifted = in << (in.width - msb - 1);
    // lookup the inverse of the shifted input
    int idx = idx_from_real_val<in_t, N>(in_shifted);
    table_t inv_in = inv_table[idx];
    // shift the output back
    table_t out = inv_in << (in.width - msb - 1);

    return out;
  }

  inline detaphi_t deltaPhi(l1t::PFCandidate a, l1t::PFCandidate b) {
    // scale the particle eta, phi to hardware units
    etaphi_t aphi = etaphi_t(a.phi() * etaphi_base);
    etaphi_t bphi = etaphi_t(b.phi() * etaphi_base);
    detaphi_t dphi = detaphi_t(aphi) - detaphi_t(bphi);
    // phi wrap
    detaphi_t dphi0 =
        dphi > detaphi_t(l1ct::Scales::INTPHI_PI) ? detaphi_t(l1ct::Scales::INTPHI_TWOPI - dphi) : detaphi_t(dphi);
    detaphi_t dphi1 =
        dphi < detaphi_t(-l1ct::Scales::INTPHI_PI) ? detaphi_t(l1ct::Scales::INTPHI_TWOPI + dphi) : detaphi_t(dphi);
    //dphi > PI ? detaphi_t(TWOPI - dphi) : detaphi_t(dphi);
    //dphi < -PI ? detaphi_t(TWOPI + dphi) : detaphi_t(dphi);
    detaphi_t dphiw = dphi > detaphi_t(0) ? dphi0 : dphi1;
    return dphiw;
  }

  inline bool inCone(l1t::PFCandidate seed, l1t::PFCandidate part, detaphi_t cone2) {
    // scale the particle eta, phi to hardware units
    etaphi_t seta = etaphi_t(seed.eta() * etaphi_base);
    etaphi_t peta = etaphi_t(part.eta() * etaphi_base);
    detaphi_t deta = detaphi_t(seta) - detaphi_t(peta);
    detaphi_t dphi = deltaPhi(seed, part);
    bool ret = (deta * deta + dphi * dphi) < cone2;
    return ret;
  }

};  // namespace L1TauEmu

class TauNNIdHW {
public:
  TauNNIdHW();
  ~TauNNIdHW();

  void initialize(const std::string &iName, int iNParticles);
  void SetNNVectorVar();
  input_t *NNVectorVar() { return NNvectorVar_.data(); }
  result_t EvaluateNN();
  result_t compute(const l1t::PFCandidate &iSeed, std::vector<l1t::PFCandidate> &iParts);
  //void print();

  std::string fInput_;
  unsigned fNParticles_;
  unique_ptr<pt_t[]> fPt_;
  unique_ptr<etaphi_t[]> fEta_;
  unique_ptr<etaphi_t[]> fPhi_;
  unique_ptr<id_t[]> fId_;
  //FILE *file_;

private:
  std::vector<input_t> NNvectorVar_;
};

#endif
