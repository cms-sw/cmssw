#ifndef L1Trigger_TrackFindingTMTT_Array2D_h
#define L1Trigger_TrackFindingTMTT_Array2D_h

#include <vector>

//=== Generic 2D array class.

// (Replaced boost::numeric::ublas::matrix when boost library 
//  became too big).

// Author: Lucas Camolezi

namespace tmtt {

  template <typename T>
  class Array2D {
  public:
    //for a mxn matrix - row major
    Array2D(unsigned int m, unsigned int n) : _n{n}, _m{m} { array2D_.resize(m * n); }
    const T& operator()(unsigned int i, unsigned int j) const { return array2D_.at(i * _n + j); }
    T& operator()(unsigned int i, unsigned int j) {
      if (i >= _m || j >= _n)
        throw std::out_of_range("matrix access out of bounds");

      return array2D_[i * _n + j];
    }

  private:
    std::vector<T> array2D_;
    unsigned int _n, _m;
  };
}  // namespace tmtt

#endif
