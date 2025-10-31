#ifndef CC_LOG_H__
#define CC_LOG_H__

#include "L1Trigger/Phase2L1ParticleFlow/interface/common/inversion.h"

namespace l1ct {

  template <class data_T, class table_T, int N>
  void init_log_table(table_T table_out[N]) {
    // The template data_T is the data type used to address the table
    for (unsigned i = 0; i < N; i++) {
      float x = real_val_from_idx<data_T, N>(i);
      table_T log_x = std::log(x);
      table_out[i] = log_x;
    }
  }

  template <class data_T, class table_T, int N>
  void init_little_log_table(table_T table_out[N]) {
    // The template data_T is the data type used to address the table
    for (unsigned i = 0; i < N; i++) {
      table_T log_pow_x = std::log(std::pow(2, (i)));
      table_out[i] = log_pow_x;
    }
  }

  template <class in_t, class table_t, int N>
  table_t log_with_shift(in_t in) {
    table_t log_table[N];
    init_log_table<in_t, table_t, N>(log_table);

    table_t log_pow_table[in.width];
    init_little_log_table<in_t, table_t, in.width>(log_pow_table);

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
    table_t log_in = log_table[idx];
    // lookup the shift needed to get back to original basis
    // log(A/B) = log(A) - log(B), A is pT shifted, B is the shift
    table_t log_shift = log_pow_table[in.width - msb - 1];
    table_t out = log_in - log_shift;
    return out;
  }

}  // namespace l1ct
#endif