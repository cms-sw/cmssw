#include <Math/VectorUtil.h>
#include <iostream>
#include <limits>
#include "TMath.h"

namespace CommonUtils {
  template<typename T> inline bool isinf(T value)
    {
      value = TMath::Abs(value);
      return std::numeric_limits<T>::has_infinity && value == std::numeric_limits<T>::infinity();
    }
}
