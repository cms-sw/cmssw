#ifndef L1TRIGGER_PHASE2L1PARTICLE_FLOW_HGCAL_CONV_NNET_ACTIVATION_H_
#define L1TRIGGER_PHASE2L1PARTICLE_FLOW_HGCAL_CONV_NNET_ACTIVATION_H_

#include "ap_fixed.h"
#include "nnet_common.h"
#include <cmath>

namespace nnet {

  inline float exp_fcn_float(float input) { return std::exp(input); }

  template <class data_T, typename CONFIG_T>
  inline unsigned softmax_idx_from_real_val(data_T x) {
    // Slice the top N bits to get an index into the table
    static constexpr int N = ceillog2(CONFIG_T::table_size);  // number of address bits for table
    ap_uint<N> y = x(x.width - 1, x.width - N);               // slice the top N bits of input
    return (unsigned)y(N - 1, 0);
  }

  template <class data_T, typename CONFIG_T>
  inline float softmax_real_val_from_idx(unsigned i) {
    // Treat the index as the top N bits
    static constexpr int N = ceillog2(CONFIG_T::table_size);  // number of address bits for table
    data_T x(0);
    x(x.width - 1, x.width - N) = i;
    return (float)x;
  }

  template <class data_T, typename CONFIG_T>
  void init_exp_table(typename CONFIG_T::exp_table_t table_out[CONFIG_T::table_size]) {
    // The template data_T is the data type used to address the table
    for (unsigned i = 0; i < CONFIG_T::table_size; i++) {
      // Slicing bits for address is going to round towards 0, so take the central value
      float x = softmax_real_val_from_idx<data_T, CONFIG_T>(i);
      typename CONFIG_T::exp_table_t exp_x = exp_fcn_float(x);
      table_out[i] = exp_x;
    }
  }

  template <class data_T, typename CONFIG_T>
  void init_invert_table(typename CONFIG_T::inv_table_t table_out[CONFIG_T::table_size]) {
    // The template data_T is the data type used to address the table
    for (unsigned i = 0; i < CONFIG_T::table_size; i++) {
      float x = softmax_real_val_from_idx<data_T, CONFIG_T>(i);
      typename CONFIG_T::inv_table_t inv_x = 1 / x;
      table_out[i] = inv_x;
    }
  }

  template <class data_T, class res_T, typename CONFIG_T>
  void softmax_stable(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    // Initialize the lookup tables
#ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::exp_table_t exp_table[CONFIG_T::table_size];
    typename CONFIG_T::inv_table_t invert_table[CONFIG_T::table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::exp_table_t exp_table[CONFIG_T::table_size];
    static typename CONFIG_T::inv_table_t invert_table[CONFIG_T::table_size];

#endif
    if (!initialized) {
      // Note we are exponentiating the inputs, which have type data_T
      init_exp_table<data_T, CONFIG_T>(exp_table);
      // Note we are inverting the exponentials, which have type exp_table_t
      init_invert_table<typename CONFIG_T::exp_table_t, CONFIG_T>(invert_table);
      initialized = true;
    }

    // Find the max and compute all delta(x_i, x_max)
    Op_max<data_T> op_max;
    data_T x_max = reduce<data_T, CONFIG_T::n_in, Op_max<data_T>>(data, op_max);

    // For the diffs, use the same type as the input but force rounding and saturation
    ap_fixed<data_T::width, data_T::iwidth, AP_RND, AP_SAT> d_xi_xmax[CONFIG_T::n_in];
    for (unsigned i = 0; i < CONFIG_T::n_in; i++) {
      d_xi_xmax[i] = data[i] - x_max;
    }

    // Calculate all the e^x's
    typename CONFIG_T::exp_table_t exp_res[CONFIG_T::n_in];
    typename CONFIG_T::exp_table_t exp_sum(0);
    for (unsigned i = 0; i < CONFIG_T::n_in; i++) {
      unsigned x = softmax_idx_from_real_val<data_T, CONFIG_T>(d_xi_xmax[i]);
      exp_res[i] = exp_table[x];
    }

    // Explicitly sum the results with an adder tree.
    // Rounding & Saturation mode, which improve accuracy, prevent Vivado from expression balancing
    Op_add<typename CONFIG_T::exp_table_t> op_add;
    exp_sum =
        reduce<typename CONFIG_T::exp_table_t, CONFIG_T::n_in, Op_add<typename CONFIG_T::exp_table_t>>(exp_res, op_add);

    typename CONFIG_T::inv_table_t inv_exp_sum =
        invert_table[softmax_idx_from_real_val<typename CONFIG_T::exp_table_t, CONFIG_T>(exp_sum)];
    for (unsigned i = 0; i < CONFIG_T::n_in; i++) {
      res[i] = exp_res[i] * inv_exp_sum;
    }
  }

}  // namespace nnet
#endif