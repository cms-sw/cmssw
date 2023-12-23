#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"

#include <cstddef>
#include <cstdio>

// hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 80
#define N_LAYER_2 25
#define N_LAYER_2 25
#define N_LAYER_5 25
#define N_LAYER_5 25
#define N_LAYER_8 15
#define N_LAYER_8 15
#define N_LAYER_11 15
#define N_LAYER_11 15
#define N_LAYER_14 10
#define N_LAYER_14 10
#define N_LAYER_17 1
#define N_LAYER_17 1
#define N_LAYER_20 1

// hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<16, 6> input_t;
typedef ap_fixed<24, 12> input2_t;
typedef ap_fixed<16, 6> model_default_t;
typedef ap_fixed<16, 6> layer2_t;
typedef ap_fixed<9, 3> weight2_t;
typedef ap_fixed<9, 3> bias2_t;
typedef ap_uint<1> layer2_index;
typedef ap_ufixed<9, 0, AP_RND_CONV, AP_SAT> layer4_t;
typedef ap_fixed<18, 8> relu_1_table_t;
typedef ap_fixed<16, 6> layer5_t;
typedef ap_fixed<9, 3> weight5_t;
typedef ap_fixed<9, 3> bias5_t;
typedef ap_uint<1> layer5_index;
typedef ap_ufixed<9, 0, AP_RND_CONV, AP_SAT> layer7_t;
typedef ap_fixed<18, 8> relu_2_table_t;
typedef ap_fixed<16, 6> layer8_t;
typedef ap_fixed<9, 3> weight8_t;
typedef ap_fixed<9, 3> bias8_t;
typedef ap_uint<1> layer8_index;
typedef ap_ufixed<9, 0, AP_RND_CONV, AP_SAT> layer10_t;
typedef ap_fixed<18, 8> relu_3_table_t;
typedef ap_fixed<16, 6> layer11_t;
typedef ap_fixed<9, 3> weight11_t;
typedef ap_fixed<9, 3> bias11_t;
typedef ap_uint<1> layer11_index;
typedef ap_ufixed<9, 0, AP_RND_CONV, AP_SAT> layer13_t;
typedef ap_fixed<18, 8> relu_4_table_t;
typedef ap_fixed<16, 6> layer14_t;
typedef ap_fixed<9, 3> weight14_t;
typedef ap_fixed<9, 3> bias14_t;
typedef ap_uint<1> layer14_index;
typedef ap_ufixed<9, 0, AP_RND_CONV, AP_SAT> layer16_t;
typedef ap_fixed<18, 8> relu_5_table_t;
typedef ap_fixed<16, 6> layer17_t;
typedef ap_fixed<16, 7> weight17_t;
typedef ap_fixed<16, 7> bias17_t;
typedef ap_uint<1> layer17_index;
typedef ap_fixed<16, 6> result_t;
typedef ap_fixed<18, 8> jetID_output_table_t;
typedef ap_fixed<16, 7> weight20_t;
typedef ap_fixed<16, 7> bias20_t;
typedef ap_uint<1> layer20_index;

#endif