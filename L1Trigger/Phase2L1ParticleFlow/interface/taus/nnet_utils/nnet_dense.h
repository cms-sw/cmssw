#ifndef NNET_DENSE_H_
#define NNET_DENSE_H_

#include "nnet_common.h"
#include "nnet_mult.h"
#include <math.h>

namespace nnet {

  struct dense_config {
    // Internal data type definitions
    typedef float bias_t;
    typedef float weight_t;
    typedef float accum_t;

    // Layer Sizes
    static const unsigned n_in = 10;
    static const unsigned n_out = 10;

    // Resource reuse info
    int io_type = io_parallel;
    int strategy = latency;
    int reuse_factor = 1;
    static const bool store_weights_in_bram = false;
    int n_zeros = 0;
    // partitioning arrays cyclically to go with roll factors?
    // Product function to use
    template <class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
  };

  template <class data_T, class res_T, typename CONFIG_T>
  void dense(data_T data[CONFIG_T::n_in],
             res_T res[CONFIG_T::n_out],
             typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
             typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {
    data_T cache;
    typename CONFIG_T::accum_t mult[CONFIG_T::n_in * CONFIG_T::n_out];
    typename CONFIG_T::accum_t acc[CONFIG_T::n_out];

    // Do the matrix-multiply
    for (unsigned ii = 0; ii < CONFIG_T::n_in; ii++) {
      cache = data[ii];
      for (unsigned jj = 0; jj < CONFIG_T::n_out; jj++) {
        unsigned index = ii * CONFIG_T::n_out + jj;
        mult[index] = CONFIG_T::template product<data_T, typename CONFIG_T::weight_t>::product(cache, weights[index]);
      }
    }

    // Initialize accumulator with input biases
    for (unsigned iacc = 0; iacc < CONFIG_T::n_out; iacc++) {
      acc[iacc] = (typename CONFIG_T::accum_t)biases[iacc];
    }

    // Accumulate multiplication result
    for (unsigned ii = 0; ii < CONFIG_T::n_in; ii++) {
      for (unsigned jj = 0; jj < CONFIG_T::n_out; jj++) {
        unsigned index = ii * CONFIG_T::n_out + jj;
        acc[jj] += mult[index];
      }
    }

    // Cast to "res_t" type
    for (unsigned ires = 0; ires < CONFIG_T::n_out; ires++) {
      // res[ires] = (res_T) (acc[ires]);
      res[ires] = cast<data_T, res_T, CONFIG_T>(acc[ires]);
    }
  }

}  // namespace nnet

#endif
