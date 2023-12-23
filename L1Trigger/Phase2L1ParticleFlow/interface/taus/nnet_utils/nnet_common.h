#ifndef NNET_COMMON_H_
#define NNET_COMMON_H_

#include "ap_fixed.h"

// This is a substitute for "ceil(n/(float)d)".
#define DIV_ROUNDUP(n, d) ((n + d - 1) / d)
#define MIN(n, d) (n > d ? d : n)
#define MAX(n, d) (n > d ? n : d)

#define STRINGIFY(x) #x
#define EXPAND_STRING(x) STRINGIFY(x)

namespace nnet {

  // Common type definitions
  enum io_type { io_parallel = 0, io_stream };
  enum strategy { latency, resource };

  template <class T>
  class Op_add {
  public:
    T operator()(T a, T b) { return a + b; }
  };

  template <class T>
  class Op_and {
  public:
    T operator()(T a, T b) { return a && b; }
  };

  template <class T>
  class Op_or {
  public:
    T operator()(T a, T b) { return a || b; }
  };

  template <class T>
  class Op_max {
  public:
    T operator()(T a, T b) { return a >= b ? a : b; }
  };

  template <class T>
  class Op_min {
  public:
    T operator()(T a, T b) { return a <= b ? a : b; }
  };

}  // namespace nnet

#endif
