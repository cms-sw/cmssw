#ifndef NNET_DENSE_H_
#define NNET_DENSE_H_


#include "nnet_common.h"
#include "nnet_dense_latency.h"
#include "nnet_helpers.h"
#include "nnet_mult.h"
#include <math.h>

namespace nnet {

struct dense_config {
    // Internal data type definitions
    typedef float bias_t;
    typedef float weight_t;
    typedef float accum_t;

    // Layer Sizes
    static int n_in;
    static int n_out;

    // Resource reuse info
    int io_type = io_parallel;
    int strategy = latency;
    int reuse_factor = 1;
    static const bool store_weights_in_bram = false;
    int n_zeros = 0;
    // partitioning arrays cyclically to go with roll factors?
    // Product function to use
    template <class x_T, class y_T> using product = nnet::product::mult<x_T, y_T>;
};

int dense_config::n_in=10;
int dense_config::n_out=10;

template <class data_T, class res_T, typename CONFIG_T>
void dense(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_out],
           typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
           typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {
        dense_latency<data_T, res_T, CONFIG_T>(data, res, weights, biases);
}

} // namespace nnet

#endif
