//
//    rfnoc-hls-neuralnet: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2017 EJ Kreinar
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//

#ifndef NNET_LAYER_H_
#define NNET_LAYER_H_

#include "nnet_common.h"
#include <cmath>

namespace nnet {

  struct layer_config {
    // Internal data type definitions
    typedef float bias_t;
    typedef float weight_t;
    typedef float accum_t;

    // Layer Sizes
    static const unsigned n_in = 10;
    static const unsigned n_out = 10;

    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
    static const unsigned n_zeros = 0;
    static const bool use_lowlatency = true;
    // partitioning arrays cyclically to go with roll factors?
  };

#define DIV_ROUNDUP(n, d) ((n + d - 1) / d)

  template <class data_T, class res_T, typename CONFIG_T>
  void compute_layer(data_T data[CONFIG_T::n_in],
                     res_T res[CONFIG_T::n_out],
                     typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
                     typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {
    unsigned cycle_factor = DIV_ROUNDUP(CONFIG_T::n_in * CONFIG_T::n_out, CONFIG_T::reuse_factor);
    typename CONFIG_T::weight_t mult[CONFIG_T::n_in * CONFIG_T::n_out];
    /*
    if(CONFIG_T::use_lowlatency) { 
      int multiplier_limit  = ceil(float(CONFIG_T::n_in*CONFIG_T::n_out) / float(CONFIG_T::reuse_factor)) - floor(float(CONFIG_T::n_zeros) / float(CONFIG_T::reuse_factor));
    } 
    */
    typename CONFIG_T::accum_t acc[CONFIG_T::n_out];
    for (unsigned iacc = 0; iacc < CONFIG_T::n_out; iacc++) {
      acc[iacc] = (typename CONFIG_T::accum_t)biases[iacc];
    }
    unsigned rufactor = CONFIG_T::reuse_factor;
    if (CONFIG_T::use_lowlatency) {
      rufactor = CONFIG_T::n_in;
      cycle_factor = CONFIG_T::n_out;
    }
    data_T cache;
    for (unsigned ii = 0; ii < rufactor; ii++) {
      if (CONFIG_T::use_lowlatency) {
        cache = data[ii];
      }
      for (unsigned jj = 0; jj < cycle_factor; jj++) {
        unsigned windex = ii * cycle_factor + jj;
        unsigned index = windex / CONFIG_T::n_out;
        if (windex > CONFIG_T::n_in * CONFIG_T::n_out - 1)
          continue;
        if (CONFIG_T::use_lowlatency) {
          mult[windex] = cache * (weights[windex]);
        } else {
          int aindex = windex / CONFIG_T::n_in;
          acc[aindex] += data[index] * weights[windex];
        }
      }
    }
    if (CONFIG_T::use_lowlatency) {
      // Accumulate multiplication result
      for (unsigned ii = 0; ii < CONFIG_T::n_in; ii++) {
        for (unsigned jj = 0; jj < CONFIG_T::n_out; jj++) {
          int index = ii * CONFIG_T::n_out + jj;
          acc[jj] += mult[index];
        }
      }
    }
    for (unsigned ires = 0; ires < CONFIG_T::n_out; ires++) {
      res[ires] = (res_T)(acc[ires]);
    }
  }

}  // namespace nnet

#endif
