#ifndef NNET_MULT_H_
#define NNET_MULT_H_

#include "nnet_common.h"
#include <iostream>
#include <math.h>

namespace nnet {

  constexpr int ceillog2(int x) { return (x <= 2) ? 1 : 1 + ceillog2((x + 1) / 2); }

  namespace product {

    /* ---
 * different methods to perform the product of input and weight, depending on the
 * types of each.
 * --- */

    class Product {};

    template <class x_T, class w_T>
    class both_binary : public Product {
    public:
      static x_T product(x_T a, w_T w) { return a == w; }
    };

    template <class x_T, class w_T>
    class weight_binary : public Product {
    public:
      static auto product(x_T a, w_T w) -> decltype(-a) {
        if (w == 0)
          return -a;
        else
          return a;
      }
    };

    template <class x_T, class w_T>
    class data_binary : public Product {
    public:
      static auto product(x_T a, w_T w) -> decltype(-w) {
        if (a == 0)
          return -w;
        else
          return w;
      }
    };

    template <class x_T, class w_T>
    class weight_ternary : public Product {
    public:
      static auto product(x_T a, w_T w) -> decltype(-a) {
        if (w == 0)
          return 0;
        else if (w == -1)
          return -a;
        else
          return a;  // if(w == 1)
      }
    };

    template <class x_T, class w_T>
    class mult : public Product {
    public:
      static auto product(x_T a, w_T w) -> decltype(a * w) { return a * w; }
    };

    template <class x_T, class w_T>
    class weight_exponential : public Product {
    public:
      using r_T =
          ap_fixed<2 * (decltype(w_T::weight)::width + x_T::width), (decltype(w_T::weight)::width + x_T::width)>;
      static r_T product(x_T a, w_T w) {
        // Shift by the exponent. Negative weights shift right
        r_T y = static_cast<r_T>(a) << w.weight;

        // Negate or not depending on weight sign
        return w.sign == 1 ? y : static_cast<r_T>(-y);
      }
    };

  }  // namespace product

  template <class data_T, class res_T, typename CONFIG_T>
  inline typename std::enable_if<std::is_same<data_T, ap_uint<1>>::value &&
                                     std::is_same<typename CONFIG_T::weight_t, ap_uint<1>>::value,
                                 ap_int<nnet::ceillog2(CONFIG_T::n_in) + 2>>::type
  cast(typename CONFIG_T::accum_t x) {
    return (ap_int<nnet::ceillog2(CONFIG_T::n_in) + 2>)(x - CONFIG_T::n_in / 2) * 2;
  }

  template <class data_T, class res_T, typename CONFIG_T>
  inline typename std::enable_if<std::is_same<data_T, ap_uint<1>>::value &&
                                     !std::is_same<typename CONFIG_T::weight_t, ap_uint<1>>::value,
                                 res_T>::type
  cast(typename CONFIG_T::accum_t x) {
    return (res_T)x;
  }

  template <class data_T, class res_T, typename CONFIG_T>
  inline typename std::enable_if<(!std::is_same<data_T, ap_uint<1>>::value), res_T>::type cast(
      typename CONFIG_T::accum_t x) {
    return (res_T)x;
  }

}  // namespace nnet

#endif
