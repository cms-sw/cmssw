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
  static const unsigned n_in = 80;
  static const unsigned n_out = 25;
  static const unsigned io_type = nnet::io_parallel;
  static const unsigned strategy = nnet::latency;
  static const unsigned reuse_factor = 1;
  static const unsigned n_zeros = 1205;
  static const unsigned n_nonzeros = 795;
  static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
  static const bool store_weights_in_bram = false;
  typedef model_default_t accum_t;
  typedef bias2_t bias_t;
  typedef weight2_t weight_t;
  typedef layer2_index index_t;
  template <class x_T, class y_T>
  using product = nnet::product::mult<x_T, y_T>;
};

// relu_1
struct relu_config4 : nnet::activ_config {
  static const unsigned n_in = 25;
  static const unsigned table_size = 1024;
  static const unsigned io_type = nnet::io_parallel;
  static const unsigned reuse_factor = 1;
  typedef relu_1_table_t table_t;
};

// Dense_2
struct config5 : nnet::dense_config {
  static const unsigned n_in = 25;
  static const unsigned n_out = 25;
  static const unsigned io_type = nnet::io_parallel;
  static const unsigned strategy = nnet::latency;
  static const unsigned reuse_factor = 1;
  static const unsigned n_zeros = 375;
  static const unsigned n_nonzeros = 250;
  static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
  static const bool store_weights_in_bram = false;
  typedef model_default_t accum_t;
  typedef bias5_t bias_t;
  typedef weight5_t weight_t;
  typedef layer5_index index_t;
  template <class x_T, class y_T>
  using product = nnet::product::mult<x_T, y_T>;
};

// relu_2
struct relu_config7 : nnet::activ_config {
  static const unsigned n_in = 25;
  static const unsigned table_size = 1024;
  static const unsigned io_type = nnet::io_parallel;
  static const unsigned reuse_factor = 1;
  typedef relu_2_table_t table_t;
};

// Dense_3
struct config8 : nnet::dense_config {
  static const unsigned n_in = 25;
  static const unsigned n_out = 15;
  static const unsigned io_type = nnet::io_parallel;
  static const unsigned strategy = nnet::latency;
  static const unsigned reuse_factor = 1;
  static const unsigned n_zeros = 225;
  static const unsigned n_nonzeros = 150;
  static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
  static const bool store_weights_in_bram = false;
  typedef model_default_t accum_t;
  typedef bias8_t bias_t;
  typedef weight8_t weight_t;
  typedef layer8_index index_t;
  template <class x_T, class y_T>
  using product = nnet::product::mult<x_T, y_T>;
};

// relu_3
struct relu_config10 : nnet::activ_config {
  static const unsigned n_in = 15;
  static const unsigned table_size = 1024;
  static const unsigned io_type = nnet::io_parallel;
  static const unsigned reuse_factor = 1;
  typedef relu_3_table_t table_t;
};

// Dense_4
struct config11 : nnet::dense_config {
  static const unsigned n_in = 15;
  static const unsigned n_out = 15;
  static const unsigned io_type = nnet::io_parallel;
  static const unsigned strategy = nnet::latency;
  static const unsigned reuse_factor = 1;
  static const unsigned n_zeros = 135;
  static const unsigned n_nonzeros = 90;
  static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
  static const bool store_weights_in_bram = false;
  typedef model_default_t accum_t;
  typedef bias11_t bias_t;
  typedef weight11_t weight_t;
  typedef layer11_index index_t;
  template <class x_T, class y_T>
  using product = nnet::product::mult<x_T, y_T>;
};

// relu_4
struct relu_config13 : nnet::activ_config {
  static const unsigned n_in = 15;
  static const unsigned table_size = 1024;
  static const unsigned io_type = nnet::io_parallel;
  static const unsigned reuse_factor = 1;
  typedef relu_4_table_t table_t;
};

// Dense_5
struct config14 : nnet::dense_config {
  static const unsigned n_in = 15;
  static const unsigned n_out = 10;
  static const unsigned io_type = nnet::io_parallel;
  static const unsigned strategy = nnet::latency;
  static const unsigned reuse_factor = 1;
  static const unsigned n_zeros = 90;
  static const unsigned n_nonzeros = 60;
  static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
  static const bool store_weights_in_bram = false;
  typedef model_default_t accum_t;
  typedef bias14_t bias_t;
  typedef weight14_t weight_t;
  typedef layer14_index index_t;
  template <class x_T, class y_T>
  using product = nnet::product::mult<x_T, y_T>;
};

// relu_5
struct relu_config16 : nnet::activ_config {
  static const unsigned n_in = 10;
  static const unsigned table_size = 1024;
  static const unsigned io_type = nnet::io_parallel;
  static const unsigned reuse_factor = 1;
  typedef relu_5_table_t table_t;
};

// Dense_6
struct config17 : nnet::dense_config {
  static const unsigned n_in = 10;
  static const unsigned n_out = 1;
  static const unsigned io_type = nnet::io_parallel;
  static const unsigned strategy = nnet::latency;
  static const unsigned reuse_factor = 1;
  static const unsigned n_zeros = 6;
  static const unsigned n_nonzeros = 4;
  static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
  static const bool store_weights_in_bram = false;
  typedef model_default_t accum_t;
  typedef bias17_t bias_t;
  typedef weight17_t weight_t;
  typedef layer17_index index_t;
  template <class x_T, class y_T>
  using product = nnet::product::mult<x_T, y_T>;
};

// jetID_output
struct sigmoid_config19 : nnet::activ_config {
  static const unsigned n_in = 1;
  static const unsigned table_size = 1024;
  static const unsigned io_type = nnet::io_parallel;
  static const unsigned reuse_factor = 1;
  typedef jetID_output_table_t table_t;
};

// pT_output
struct config20 : nnet::dense_config {
  static const unsigned n_in = 10;
  static const unsigned n_out = 1;
  static const unsigned io_type = nnet::io_parallel;
  static const unsigned strategy = nnet::latency;
  static const unsigned reuse_factor = 1;
  static const unsigned n_zeros = 6;
  static const unsigned n_nonzeros = 4;
  static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
  static const bool store_weights_in_bram = false;
  typedef model_default_t accum_t;
  typedef bias20_t bias_t;
  typedef weight20_t weight_t;
  typedef layer20_index index_t;
  template <class x_T, class y_T>
  using product = nnet::product::mult<x_T, y_T>;
};

#endif