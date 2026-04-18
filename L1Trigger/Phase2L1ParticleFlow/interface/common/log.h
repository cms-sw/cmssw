#ifndef CC_LOG_H__
#define CC_LOG_H__

#include "L1Trigger/Phase2L1ParticleFlow/interface/common/inversion.h"

namespace l1ct {

  // LUT function for log(x)
  template <class data_T, class table_T, int N>
  table_T log_table(int idx) {
    float x = real_val_from_idx<data_T, N>(idx);
    table_T log_x = std::log(x);
    return log_x;
  }

  // LUT function for log(2^x) used for shifting
  template <class data_T, class table_T, int N>
  table_T log_power_table(int idx) {
    table_T log_pow_x = std::log(std::pow(2, (idx)));
    return log_pow_x;
  }

  template <class in_t, class table_t, int N>
  table_t log_with_shift(in_t in) {
    // shift up the denominator such that the left-most bit (msb) is '1'
    int msb = 0;
    for (int b = 0; b < in.width; b++) {
      // #pragma HLS unroll
      if (in[b])
        msb = b;
    }
    // shift up the denominator such that the left-most bit (msb) is '1'
    in_t in_shifted = in << (in.width - msb - 1);
    // lookup the log of the shifted input
    int idx = idx_from_real_val<in_t, N>(in_shifted);
    table_t log_in = log_table<in_t, table_t, N>(idx);
    // lookup the shift needed to get back to original basis
    // log(A/B) = log(A) - log(B), A is pT shifted, B is the shift
    table_t log_shift = log_power_table<in_t, table_t, N>(in.width - msb - 1);
    table_t out = log_in - log_shift;
    return out;
  }

}  // namespace l1ct
#endif
