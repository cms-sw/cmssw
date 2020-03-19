/* ---
* Matrix class supporting common operations such as multiplication, inverse, determinant.
* Derived from: https://www.cs.rochester.edu/~brown/Crypto/assts/projects/adj.html
* and https://www.quantstart.com/articles/Matrix-Classes-in-C-The-Source-File
* --- */
 
#include "L1Trigger/TrackFindingTMTT/interface/Matrix.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <iostream>
#include <cassert>
 
namespace TMTT {

// Parameter Constructor                                                                                                                                                      
template<typename T>
Matrix<T>::Matrix(){
    mat.resize(0);
    rows = 0;
    cols = 0;
}
template<typename T>
Matrix<T>::Matrix(unsigned _rows, unsigned _cols, const T& _initial) {
  mat.resize(_rows);
  for (unsigned i=0; i<mat.size(); i++) {
    mat[i].resize(_cols, _initial);
  }
  rows = _rows;
  cols = _cols;
}
 
// Copy Constructor                                                                                                                                                          
template<typename T>
Matrix<T>::Matrix(const Matrix<T>& rhs) {
  mat = rhs.mat;
  rows = rhs.get_rows();
  cols = rhs.get_cols();
}
 
// (Virtual) Destructor                                                                                                                                                      
template<typename T>
Matrix<T>::~Matrix() {}
 
// Assignment Operator                                                                                                                                                        
template<typename T>
Matrix<T>& Matrix<T>::operator=(const Matrix<T>& rhs) {
  if (&rhs == this)
    return *this;
 
  unsigned new_rows = rhs.get_rows();
  unsigned new_cols = rhs.get_cols();
 
  mat.resize(new_rows);
  for (unsigned i=0; i<mat.size(); i++) {
    mat[i].resize(new_cols);
  }
 
  for (unsigned i=0; i<new_rows; i++) {
    for (unsigned j=0; j<new_cols; j++) {
      mat[i][j] = rhs(i, j);
    }
  }
  rows = new_rows;
  cols = new_cols;
 
  return *this;
}
 
// Addition of two matrices                                                                                                                                                  
template<typename T>
Matrix<T> Matrix<T>::operator+(const Matrix<T>& rhs) {
  Matrix result(rows, cols, 0.0);
 
  for (unsigned i=0; i<rows; i++) {
    for (unsigned j=0; j<cols; j++) {
      result(i,j) = this->mat[i][j] + rhs(i,j);
    }
  }
 
  return result;
}
 
// Cumulative addition of this matrix and another                                                                                                                            
template<typename T>
Matrix<T>& Matrix<T>::operator+=(const Matrix<T>& rhs) {
  unsigned rows = rhs.get_rows();
  unsigned cols = rhs.get_cols();
 
  for (unsigned i=0; i<rows; i++) {
    for (unsigned j=0; j<cols; j++) {
      this->mat[i][j] += rhs(i,j);
    }
  }
 
  return *this;
}
 
// Subtraction of this matrix and another                                                                                                                                    
template<typename T>
Matrix<T> Matrix<T>::operator-(const Matrix<T>& rhs) {
  unsigned rows = rhs.get_rows();
  unsigned cols = rhs.get_cols();
  Matrix result(rows, cols, 0.0);
 
  for (unsigned i=0; i<rows; i++) {
    for (unsigned j=0; j<cols; j++) {
      result(i,j) = this->mat[i][j] - rhs(i,j);
    }
  }
 
  return result;
}
 
// Cumulative subtraction of this matrix and another                                                                                                                          
template<typename T>
Matrix<T>& Matrix<T>::operator-=(const Matrix<T>& rhs) {
  unsigned rows = rhs.get_rows();
  unsigned cols = rhs.get_cols();
 
  for (unsigned i=0; i<rows; i++) {
    for (unsigned j=0; j<cols; j++) {
      this->mat[i][j] -= rhs(i,j);
    }
  }
 
  return *this;
}
 
// Left multiplication of this matrix and another                                                                                                                              
template<typename T>
Matrix<T> Matrix<T>::operator*(const Matrix<T>& rhs) {
  assert(this->cols == rhs.get_rows());
  unsigned rRows = rhs.get_rows();
  unsigned rCols = rhs.get_cols();
  unsigned lRows = this->rows;
  unsigned lCols = this->cols;
  Matrix result(lRows, rCols, 0.0);
  for(unsigned i=0; i < lRows; i++){
    for(unsigned j = 0; j < rCols; j++){
      for(unsigned k = 0; k < lCols; k++){
        result(i,j) += this->mat[i][k] * rhs.mat[k][j];
      }
    }
  }
  return result;
}
 
// Cumulative left multiplication of this matrix and another                                                                                                                  
template<typename T>
Matrix<T>& Matrix<T>::operator*=(const Matrix<T>& rhs) {
  Matrix result = (*this) * rhs;
  (*this) = result;
  return *this;
}
 
// Calculate a transpose of this matrix                                                                                                                                      
template<typename T>
Matrix<T> Matrix<T>::transpose() {
  Matrix result(cols, rows, 0.0);
 
  for (unsigned i=0; i<cols; i++) {
    for (unsigned j=0; j<rows; j++) {
      result(i,j) = this->mat[j][i];
    }
  }
 
  return result;
}
 
// Recursively calculate the determinant
template<typename T>
T Matrix<T>::determinant(){
  T det = 0;
  if(rows < 1){
    std::cerr << "Can't have determinant of matrix with " << rows << " rows." << std::endl;
  } else if (rows == 1){
    det = this->mat[0][0];
  } else if (rows == 2){
    det = this->mat[0][0] * this->mat[1][1] - this->mat[1][0] * this->mat[0][1];
  }else{
    for(unsigned i = 0; i < rows; i++){
      Matrix m(rows-1, cols-1, 0);
      for(unsigned j = 1; j < rows; j++){
        unsigned k = 0;
        for(unsigned l =0; l < rows; l++){
          if(l == i)
            continue;
          m(j-1,k) = this->mat[j][l];
          k++;
        }
      }
      T sign;
      if ((i+2)%2 == 0)
        sign = 1;
      else
        sign = -1;
      det += sign * this->mat[0][i] * m.determinant();
    }
  }
  return(det);
}
 
template<typename T>
Matrix<T> Matrix<T>::cofactor(){
  Matrix result(rows, cols, 0);
  Matrix c(rows-1, cols-1, 0);
  if(rows != cols){
    std::cerr << "Can only compute cofactor of square matrix." << std::endl;
    return result;
  }
  for(unsigned j = 0; j < rows; j++){
    for(unsigned i = 0; i < rows; i++){
      unsigned i1 = 0;
      for(unsigned ii = 0; ii < rows; ii++){
        if(ii==i)
          continue;
        unsigned j1 = 0;
        for(unsigned jj = 0; jj< rows; jj++){
          if(jj == j)
            continue;
          c(i1,j1) = this->mat[ii][jj];
          j1++;
        }
        i1++;
      }
 
      T det = c.determinant();
      T sign;
      if ((i+j+2)%2 == 0)
        sign = 1;
      else
        sign = -1;
      result(i,j) = sign * det;
    }
  }
  return result;
}
 
template<typename T>
Matrix<T> Matrix<T>::inverse(){
  if(rows != cols){
    std::cerr << "Matrix: cannot invert matrix with " << rows << " rows and " << cols << "cols" << std::endl;
  }
  if(this->determinant() == 0)
    std::cerr << "Matrix with 0 determinant has no inverse." << std::endl;
  Matrix result = this->cofactor().transpose() / this->determinant();
  return result;
}
 
// Matrix/scalar addition                                                                                                                                                    
template<typename T>
Matrix<T> Matrix<T>::operator+(const T& rhs) {
  Matrix result(rows, cols, 0.0);
 
  for (unsigned i=0; i<rows; i++) {
    for (unsigned j=0; j<cols; j++) {
      result(i,j) = this->mat[i][j] + rhs;
    }
  }
 
  return result;
}
 
// Matrix/scalar subtraction                                                                                                                                                  
template<typename T>
Matrix<T> Matrix<T>::operator-(const T& rhs) {
  Matrix result(rows, cols, 0.0);

  for (unsigned i=0; i<rows; i++) {
    for (unsigned j=0; j<cols; j++) {
      result(i,j) = this->mat[i][j] - rhs;
    }
  }
 
  return result;
}
 
// Matrix/scalar multiplication                                                                                                                                              
template<typename T>
Matrix<T> Matrix<T>::operator*(const T& rhs) {
  Matrix result(rows, cols, 0.0);
 
  for (unsigned i=0; i<rows; i++) {
    for (unsigned j=0; j<cols; j++) {
      result(i,j) = this->mat[i][j] * rhs;
    }
  }
 
  return result;
}
 
// Matrix/scalar division                                                                                                                                                    
template<typename T>
Matrix<T> Matrix<T>::operator/(const T& rhs) {
  Matrix result(rows, cols, 0.0);
 
  if ( rhs == 0 ) throw cms::Exception("Matrix.cc: Trying to divide (matrix/scalar) by zero");

  for (unsigned i=0; i<rows; i++) {
    for (unsigned j=0; j<cols; j++) {
      result(i,j) = this->mat[i][j] / rhs;
    }
  }
 
  return result;
}
 
// Multiply a matrix with a vector                                                                                                                                            
template<typename T>
std::vector<T> Matrix<T>::operator*(const std::vector<T>& rhs) {
  std::vector<T> result(this->rows, 0.0);
 
  for (unsigned i=0; i<rows; i++) {
    for (unsigned j=0; j<cols; j++) {
      result[i] += this->mat[i][j] * rhs[j];
    }
  }
 
  return result;
}
 
// Obtain a vector of the diagonal elements                                                                                                                                  
template<typename T>
std::vector<T> Matrix<T>::diag_vec() {
  std::vector<T> result(rows, 0.0);
 
  for (unsigned i=0; i<rows; i++) {
    result[i] = this->mat[i][i];
  }
 
  return result;
}
 
// Access the individual elements                                                                                                                                            
template<typename T>
T& Matrix<T>::operator()(const unsigned& row, const unsigned& col) {
  return this->mat[row][col];
}
 
// Access the individual elements (const)                                                                                                                                    
template<typename T>
const T& Matrix<T>::operator()(const unsigned& row, const unsigned& col) const {
  return this->mat[row][col];
}
 
// Get the number of rows of the matrix                                                                                                                                      
template<typename T>
unsigned Matrix<T>::get_rows() const {
  return this->rows;
}
 
// Get the number of columns of the matrix                                                                                                                                    
template<typename T>
unsigned Matrix<T>::get_cols() const {
  return this->cols;
}
 
template<typename T>
void Matrix<T>::print(){
    for(unsigned i=0; i < this->rows; i++){
        for(unsigned j=0; j < this->cols; j++){
            std::cout << this->mat[i][j] << ", ";
        }
        std::cout << std::endl;
    }
}
 
template class Matrix<double>;
template class Matrix<float>;
template class Matrix<int>;
 
}
