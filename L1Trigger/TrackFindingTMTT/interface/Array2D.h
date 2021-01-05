#ifndef L1Trigger_TrackFindingTMTT_Array2D_h
#define L1Trigger_TrackFindingTMTT_Array2D_h

#include <vector>
#include <stdexcept>

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
      checkBounds(i, j);
      return array2D_.at(i * n_ + j);
    }

    T& operator()(unsigned int i, unsigned int j) {
      checkBounds(i, j);
      return array2D_[i * n_ + j];
    }

  private:
    void checkBounds(unsigned int i, unsigned int j) const {
      if (i >= m_ || j >= n_)
        throw std::out_of_range("matrix access out of bounds");
    }

  private:
    std::vector<T> array2D_;
    unsigned int m_, n_;
  };
}  // namespace tmtt

#endif
