#ifndef L1Trigger_TrackFindingTMTT_Array2D_h
#define L1Trigger_TrackFindingTMTT_Array2D_h

#include "FWCore/Utilities/interface/Exception.h"

#include <vector>
#include <utility>

//=== Generic 2D array class.

// (Replaced boost::numeric::ublas::matrix when boost library
//  became too big).

// Author: Lucas Camolezi

namespace tmtt {

  template <typename T>
  class Array2D {
  public:
    //for a mxn matrix - row major
    Array2D(unsigned int m, unsigned int n) : array2D_(m * n), m_{m}, n_{n} {}

    const T& operator()(unsigned int i, unsigned int j) const {
      if (i >= m_ || j >= n_)
        throw cms::Exception("LogicError")
            << "Array2D: indices out of range " << i << " " << j << " " << m_ << " " << n_;
      return array2D_[i * n_ + j];
    }

    T& operator()(unsigned int i, unsigned int j) {
      // Non-const version of operator, without needing to duplicate code.
      // (Scott Meyers trick).
      return const_cast<T&>(std::as_const(*this)(i, j));
    }

  private:
    std::vector<T> array2D_;
    unsigned int m_, n_;
  };
}  // namespace tmtt

#endif
