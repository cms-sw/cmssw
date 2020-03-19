#ifndef __L1TRK_MATRIX_H__
#define __L1TRK_MATRIX_H__
 
#include <vector>
 
namespace TMTT {

template <typename T> class Matrix {
 private:
  std::vector<std::vector<T> > mat;
  unsigned rows;
  unsigned cols;
 
 public:
  Matrix();
  Matrix(unsigned _rows, unsigned _cols, const T& _initial);
  Matrix(const Matrix<T>& rhs);
  virtual ~Matrix();
 
  // Operator overloading, for "standard" mathematical matrix operations                                                                                                                                                          
  Matrix<T>& operator=(const Matrix<T>& rhs);
 
  // Matrix mathematical operations                                                                                                                                                                                              
  Matrix<T> operator+(const Matrix<T>& rhs);
  Matrix<T>& operator+=(const Matrix<T>& rhs);
  Matrix<T> operator-(const Matrix<T>& rhs);
  Matrix<T>& operator-=(const Matrix<T>& rhs);
  Matrix<T> operator*(const Matrix<T>& rhs);
  Matrix<T>& operator*=(const Matrix<T>& rhs);
  Matrix<T> transpose();
  T determinant();
  Matrix<T> cofactor();
  Matrix<T> inverse();
 
  // Matrix/scalar operations                                                                                                                                                                                                    
  Matrix<T> operator+(const T& rhs);
  Matrix<T> operator-(const T& rhs);
  Matrix<T> operator*(const T& rhs);
  Matrix<T> operator/(const T& rhs);
 
  // Matrix/vector operations                                                                                                                                                                                                    
  std::vector<T> operator*(const std::vector<T>& rhs);
  std::vector<T> diag_vec();
 
  // Access the individual elements                                                                                                                                                                                          
  T& operator()(const unsigned& row, const unsigned& col);
  const T& operator()(const unsigned& row, const unsigned& col) const;
 
  // Access the row and column sizes
  unsigned get_rows() const;
  unsigned get_cols() const;
 
  // Print to stdout
  void print();
 
};

}
 
#endif
 
 

