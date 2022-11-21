#ifndef L1Trigger_Phase2L1ParticleFlow_L1SeedConePFJetEmulator_h
#define L1Trigger_Phase2L1ParticleFlow_L1SeedConePFJetEmulator_h

#include "DataFormats/L1TParticleFlow/interface/layer1_emulator.h"
#include "DataFormats/L1TParticleFlow/interface/jets.h"

#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>

#include "L1Trigger/Phase2L1ParticleFlow/interface/dbgPrintf.h"

class L1SCJetEmu {
public:
  // Data types and constants used in the FPGA and FPGA-optimized functions
  // This header file is also for use in the standalone FPGA-tools simulation
  // and thus contains no CMSSW/EDM specific content
  typedef l1ct::pt_t pt_t;
  typedef l1ct::glbeta_t etaphi_t;       // Type for eta & phi
  typedef ap_int<13> detaphi_t;          // Type for deta & dphi
  typedef ap_fixed<18, 23> detaphi2_t;   // Type for deta^2 & dphi^2
  typedef ap_fixed<22, 22> pt_etaphi_t;  // Type for product of pt with deta & dphi
  typedef l1ct::PuppiObjEmu Particle;

  class Jet : public l1ct::Jet {
  public:
    std::vector<l1ct::PuppiObjEmu> constituents;
  };

  L1SCJetEmu(bool debug, float coneSize, unsigned nJets);

  std::vector<Jet> emulateEvent(std::vector<Particle>& parts) const;

private:
  // Configuration settings
  bool debug_;
  float coneSize_;
  unsigned nJets_;
  detaphi2_t rCone2_;

  // constants for the axis update
  typedef ap_ufixed<18, -2> inv_pt_t;
  static constexpr int N_table_inv_pt = 1024;
  inv_pt_t inv_pt_table_[N_table_inv_pt];

  static constexpr int ceillog2(int x) { return (x <= 2) ? 1 : 1 + ceillog2((x + 1) / 2); }

  static constexpr int floorlog2(int x) { return (x < 2) ? 0 : 1 + floorlog2(x / 2); }

  template <int B>
  static constexpr int pow(int x) {
    return x == 0 ? 1 : B * pow<B>(x - 1);
  }

  static constexpr int pow2(int x) { return pow<2>(x); }

  /* ---
  * Balanced tree reduce implementation.
  * Reduces an array of inputs to a single value using the template binary operator 'Op',
  * for example summing all elements with Op_add, or finding the maximum with Op_max
  * Use only when the input array is fully unrolled. Or, slice out a fully unrolled section
  * before applying and accumulate the result over the rolled dimension.
  * Required for emulation to guarantee equality of ordering.
  * --- */
  template <class T, class Op>
  static T reduce(std::vector<T> x, Op op) {
    int N = x.size();
    int leftN = pow2(floorlog2(N - 1)) > 0 ? pow2(floorlog2(N - 1)) : 0;
    //static constexpr int rightN = N - leftN > 0 ? N - leftN : 0;
    if (N == 1) {
      return x.at(0);
    } else if (N == 2) {
      return op(x.at(0), x.at(1));
    } else {
      std::vector<T> left(x.begin(), x.begin() + leftN);
      std::vector<T> right(x.begin() + leftN, x.end());
      return op(reduce<T, Op>(left, op), reduce<T, Op>(right, op));
    }
  }

  class OpPuppiObjMax {
  public:
    Particle const& operator()(Particle const& a, Particle const& b) const { return a.hwPt >= b.hwPt ? a : b; }
  };

  static constexpr OpPuppiObjMax op_max{};

  template <class data_T, int N>
  static inline float real_val_from_idx(unsigned i) {
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
  static inline unsigned idx_from_real_val(data_T x) {
    // Slice the top N bits to get an index into the table
    static constexpr int NB = ceillog2(N);  // number of address bits for table
    // Slice the top-1 NB bits of the value
    // the MSB of '1' is implicit, so only slice below that
    ap_uint<NB> y = x(x.width - 2, x.width - NB - 1);
    return (unsigned)y(NB - 1, 0);
  }

  template <class data_T, class table_T, int N>
  static void init_invert_table(table_T table_out[N]) {
    // The template data_T is the data type used to address the table
    for (unsigned i = 0; i < N; i++) {
      float x = real_val_from_idx<data_T, N>(i);
      table_T inv_x = 1 / x;
      table_out[i] = inv_x;
    }
  }

  template <class in_t, class table_t, int N>
  static table_t invert_with_shift(const in_t in, const table_t inv_table[N], bool debug = false) {
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
    if (debug) {
      dbgCout() << "           x " << in << ", msb = " << msb << ", shift = " << (in.width - msb) << ", idx = " << idx
                << std::endl;
      dbgCout() << "     pre 1 / " << in_shifted << " = " << inv_in << "(" << 1 / (float)in_shifted << ")" << std::endl;
      dbgCout() << "    post 1 / " << in << " = " << out << "(" << 1 / (float)in << ")" << std::endl;
    }
    return out;
  }

  static detaphi_t deltaPhi(Particle a, Particle b);
  bool inCone(Particle seed, Particle part) const;
  Jet makeJet_HW(const std::vector<Particle>& parts) const;

};  // class L1SCJetEmu

#endif
