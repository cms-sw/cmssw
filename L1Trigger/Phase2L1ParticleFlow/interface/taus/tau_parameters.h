#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "ap_fixed.h"
#include "ap_int.h"

#include <complex>

// Tau NN components
#include "L1Trigger/Phase2L1ParticleFlow/interface/taus/nnet_utils/nnet_activation.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/taus/nnet_utils/nnet_dense.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/taus/defines.h"

// Load the NN weights
#include "L1Trigger/Phase2L1ParticleFlow/interface/taus/weights/w2.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/taus/weights/b2.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/taus/weights/w5.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/taus/weights/b5.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/taus/weights/w8.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/taus/weights/b8.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/taus/weights/w11.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/taus/weights/b11.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/taus/weights/w14.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/taus/weights/b14.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/taus/weights/w17.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/taus/weights/b17.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/taus/weights/w20.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/taus/weights/b20.h"

// hls-fpga-machine-learning insert layer-config
// Dense_1
struct config2 : nnet::dense_config {
    static int n_in;
    static int n_out;
    int io_type = nnet::io_parallel;
    int strategy = nnet::latency;
    int reuse_factor = 1;
    int n_zeros = 1205;
    int n_nonzeros = 795;
    int multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef model_default_t accum_t;
    typedef bias2_t bias_t;
    typedef weight2_t weight_t;
    typedef layer2_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

int config2::n_in = 80;
int config2::n_out = 25;

// relu_1
struct relu_config4 : nnet::activ_config {
    static int n_in;
    static const int table_size = 1024;
    int io_type = nnet::io_parallel;
    int reuse_factor = 1;
    typedef relu_1_table_t table_t;
};
int relu_config4::n_in = 25;

// Dense_2
struct config5 : nnet::dense_config {
    static int n_in;
    static int n_out;
    int io_type = nnet::io_parallel;
    int strategy = nnet::latency;
    int reuse_factor = 1;
    int n_zeros = 375;
    int n_nonzeros = 250;
    int multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef model_default_t accum_t;
    typedef bias5_t bias_t;
    typedef weight5_t weight_t;
    typedef layer5_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

int config5::n_in = 25;
int config5::n_out = 25;

// relu_2
struct relu_config7 : nnet::activ_config {
    static int n_in;
    static const int table_size = 1024;
    int io_type = nnet::io_parallel;
    int reuse_factor = 1;
    typedef relu_2_table_t table_t;
};

int relu_config7::n_in = 25;

// Dense_3
struct config8 : nnet::dense_config {
    static int n_in;
    static int n_out;
    int io_type = nnet::io_parallel;
    int strategy = nnet::latency;
    int reuse_factor = 1;
    int n_zeros = 225;
    int n_nonzeros = 150;
    int multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef model_default_t accum_t;
    typedef bias8_t bias_t;
    typedef weight8_t weight_t;
    typedef layer8_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

int config8::n_in = 25;
int config8::n_out = 15;

// relu_3
struct relu_config10 : nnet::activ_config {
    static int n_in;
    static const int table_size = 1024;
    int io_type = nnet::io_parallel;
    int reuse_factor = 1;
    typedef relu_3_table_t table_t;
};

int relu_config10::n_in = 15;

// Dense_4
struct config11 : nnet::dense_config {
    static int n_in;
    static int n_out;
    int io_type = nnet::io_parallel;
    int strategy = nnet::latency;
    int reuse_factor = 1;
    int n_zeros = 135;
    int n_nonzeros = 90;
    int multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef model_default_t accum_t;
    typedef bias11_t bias_t;
    typedef weight11_t weight_t;
    typedef layer11_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

int config11::n_in = 15;
int config11::n_out = 15;

// relu_4
struct relu_config13 : nnet::activ_config {
    static int n_in;
    static const int table_size = 1024;
    int io_type = nnet::io_parallel;
    int reuse_factor = 1;
    typedef relu_4_table_t table_t;
};

int relu_config13::n_in = 15;

// Dense_5
struct config14 : nnet::dense_config {
    static int n_in;
    static int n_out;
    int io_type = nnet::io_parallel;
    int strategy = nnet::latency;
    int reuse_factor = 1;
    int n_zeros = 90;
    int n_nonzeros = 60;
    int multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef model_default_t accum_t;
    typedef bias14_t bias_t;
    typedef weight14_t weight_t;
    typedef layer14_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

int config14::n_in=15;
int config14::n_out=10;

// relu_5
struct relu_config16 : nnet::activ_config {
    static int n_in;
    static const int table_size=1024;
    int io_type = nnet::io_parallel;
    int reuse_factor = 1;
    typedef relu_5_table_t table_t;
};

int relu_config16::n_in=10;

// Dense_6
struct config17 : nnet::dense_config {
    static int n_in;
    static int n_out;
    int io_type = nnet::io_parallel;
    int strategy = nnet::latency;
    int reuse_factor = 1;
    int n_zeros = 6;
    int n_nonzeros = 4;
    int multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef model_default_t accum_t;
    typedef bias17_t bias_t;
    typedef weight17_t weight_t;
    typedef layer17_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};
int config17::n_in=10;
int config17::n_out=1;

// jetID_output
struct sigmoid_config19 : nnet::activ_config {
    static int n_in;
    static const int table_sizetable_size=1024;
    int io_type = nnet::io_parallel;
    int reuse_factor = 1;
    typedef jetID_output_table_t table_t;
};

int sigmoid_config19::n_in=1;

// pT_output
struct config20 : nnet::dense_config {
    static int n_in;
    static int n_out;
    int io_type = nnet::io_parallel;
    int strategy = nnet::latency;
    int reuse_factor = 1;
    int n_zeros = 6;
    int n_nonzeros = 4;
    int multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef model_default_t accum_t;
    typedef bias20_t bias_t;
    typedef weight20_t weight_t;
    typedef layer20_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

int config20::n_in = 10;
int config20::n_out = 1;

#endif