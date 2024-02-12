#ifndef PHYSICSTOOLS_TENSORFLOWAOT_UTIL_H
#define PHYSICSTOOLS_TENSORFLOWAOT_UTIL_H

/*
 * AOT utils and type definitions.
 *
 * Author: Marcel Rieger, Bogdan Wiederspan
 */

#include "tensorflow/compiler/tf2xla/xla_compiled_cpu_function.h"
#include "tensorflow/core/platform/types.h"

namespace tfaot {

  // typedefs
  typedef tensorflow::XlaCompiledCpuFunction::AllocMode AllocMode;
  typedef std::vector<std::vector<bool>> BoolArrays;
  typedef std::vector<std::vector<int32_t>> Int32Arrays;
  typedef std::vector<std::vector<int64_t>> Int64Arrays;
  typedef std::vector<std::vector<float>> FloatArrays;
  typedef std::vector<std::vector<double>> DoubleArrays;

  // helper to create lambdas accepting a function that is called with an index
  template <size_t... Index>
  auto createIndexLooper(std::index_sequence<Index...>) {
    return [](auto&& f) { (f(std::integral_constant<size_t, Index>{}), ...); };
  }

  // helper to create lambdas accepting a function that is called with an index in a range [0, N)
  template <size_t N>
  auto createIndexLooper() {
    return createIndexLooper(std::make_index_sequence<N>{});
  }

}  // namespace tfaot

#endif  // PHYSICSTOOLS_TENSORFLOWAOT_UTIL_H
