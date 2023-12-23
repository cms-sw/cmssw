#ifndef NNET_ACTIVATION_H_
#define NNET_ACTIVATION_H_

#include <cmath>
#include "ap_fixed.h"
#include "nnet_common.h"

namespace nnet {

  struct activ_config {
    // IO size
    static const unsigned n_in = 10;

    // Internal info
    static const unsigned table_size = 1024;

    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned reuse_factor = 1;

    // Internal data type definitions
    typedef ap_fixed<18, 8> table_t;
  };

  // *************************************************
  //       LINEAR Activation -- See Issue 53
  // *************************************************
  template <class data_T, class res_T, typename CONFIG_T>
  void linear(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    for (unsigned ii = 0; ii < CONFIG_T::n_in; ii++) {
      res[ii] = data[ii];
    }
  }

  // *************************************************
  //       RELU Activation
  // *************************************************
  template <class data_T, class res_T, typename CONFIG_T>
  void relu(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    data_T datareg;
    for (unsigned ii = 0; ii < CONFIG_T::n_in; ii++) {
      datareg = data[ii];
      if (datareg > 0)
        res[ii] = datareg;
      else
        res[ii] = 0;
    }
  }

  // *************************************************
  //       Sigmoid Activation
  // *************************************************
  template <class out_T>
  inline out_T sigmoid_fcn_float(float input) {
    return 1.0 / (1 + exp(-input));
  }

  template <class res_T, typename CONFIG_T, int N_TABLE>
  void init_sigmoid_table(res_T table_out[N_TABLE]) {
    // Default logistic sigmoid function:
    //   result = 1/(1+e^(-x))
    for (unsigned ii = 0; ii < N_TABLE; ii++) {
      // First, convert from table index to X-value (signed 8-bit, range -8 to +8)
      float in_val = 2 * 8.0 * (ii - float(N_TABLE) / 2.0) / float(N_TABLE);
      // Next, compute lookup table function
      res_T real_val = sigmoid_fcn_float<res_T>(in_val);
      //std::cout << "Lookup table In Value: " << in_val << " Result: " << real_val << std::endl;
      table_out[ii] = (res_T)real_val;
    }
  }

  template <class data_T, class res_T, typename CONFIG_T>
  void sigmoid(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    // Initialize the lookup table
    res_T sigmoid_table[CONFIG_T::table_size];
    init_sigmoid_table<res_T, CONFIG_T, CONFIG_T::table_size>(sigmoid_table);

    // Index into the lookup table based on data
    int data_round;
    unsigned index;
    for (unsigned ii = 0; ii < CONFIG_T::n_in; ii++) {
      data_round = data[ii] * CONFIG_T::table_size / 16;
      index = data_round + 8 * CONFIG_T::table_size / 16;
      /*if (index < 0)
        index = 0;*/
      if (index > CONFIG_T::table_size - 1)
        index = CONFIG_T::table_size - 1;
      res[ii] = (res_T)sigmoid_table[index];
    }
  }

}  // namespace nnet

#endif
