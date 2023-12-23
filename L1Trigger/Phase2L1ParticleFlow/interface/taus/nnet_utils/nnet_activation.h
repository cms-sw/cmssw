#ifndef NNET_ACTIVATION_H_
#define NNET_ACTIVATION_H_

#include "ap_fixed.h"
#include "nnet_common.h"
#include <cmath>

namespace nnet {

struct activ_config {
    // IO size
    static int n_in;

    // Internal info
    static const int table_size=1024;

    // Resource reuse info
    int io_type = io_parallel;
    int reuse_factor = 1;

    // Internal data type definitions
    typedef ap_fixed<18, 8> table_t;
};

int activ_config::n_in=10;

// *************************************************
//       RELU Activation
// *************************************************
template <class data_T, class res_T, typename CONFIG_T> void relu(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {

    data_T datareg;
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
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
inline float sigmoid_fcn_float(float input) { return 1.0 / (1 + std::exp(-input)); }

template <typename CONFIG_T, int N_TABLE> void init_sigmoid_table(typename CONFIG_T::table_t table_out[N_TABLE]) {
    // Default logistic sigmoid function:
    //   result = 1/(1+e^(-x))
    for (int ii = 0; ii < N_TABLE; ii++) {
        // First, convert from table index to X-value (signed 8-bit, range -8 to +8)
        float in_val = 2 * 8.0 * (ii - float(N_TABLE) / 2.0) / float(N_TABLE);
        // Next, compute lookup table function
        typename CONFIG_T::table_t real_val = sigmoid_fcn_float(in_val);
        // std::cout << "Lookup table In Value: " << in_val << " Result: " << real_val << std::endl;
        table_out[ii] = real_val;
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void sigmoid(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    // Initialize the lookup table
#ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::table_t sigmoid_table[CONFIG_T::table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::table_t sigmoid_table[CONFIG_T::table_size];
#endif
    if (!initialized) {
        init_sigmoid_table<CONFIG_T, CONFIG_T::table_size>(sigmoid_table);
        initialized = true;
    }

    // Index into the lookup table based on data
    int data_round;
    int index;
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        data_round = data[ii] * CONFIG_T::table_size / 16;
        index = data_round + 8 * CONFIG_T::table_size / 16;
        if (index < 0)
            index = 0;
        if (index > CONFIG_T::table_size - 1)
            index = CONFIG_T::table_size - 1;
        res[ii] = (res_T)sigmoid_table[index];
    }
}

} // namespace nnet

#endif
