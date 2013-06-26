//-------------------------------------------------
//
/** \class L1MuGMTMatrix
 *  Matrix.
 *
 *  general matrix
*/
//
//   $Date: 2007/03/23 18:51:35 $
//   $Revision: 1.4 $
//
//   Author :
//   N. Neumeister            CERN EP 
//
//   Migrated to CMSSW:
//   I. Mikulec
//
//--------------------------------------------------
#ifndef L1TriggerGlobalMuonTrigger_L1MuGMTMatrix_h
#define L1TriggerGlobalMuonTrigger_L1MuGMTMatrix_h

//---------------
// C++ Headers --
//---------------

#include <iostream>
#include <cassert>
#include <string>
#include <sstream>


//----------------------
// Base Class Headers --
//----------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------


#include "FWCore/MessageLogger/interface/MessageLogger.h"


//              ---------------------
//              -- Class Interface --
//              ---------------------

template<class T>
class L1MuGMTMatrix {

  public:

     /// constructor
     L1MuGMTMatrix(int r, int c);

     /// copy constructor
     L1MuGMTMatrix(const L1MuGMTMatrix<T>&);

     /// destructor
     virtual ~L1MuGMTMatrix();

     ///
     T& operator()(int r, int c);

     ///
     const T& operator()(int r, int c) const;

     /// initialize matrix
     void init(T v=0);

     /// set matrix element
     void set(int r, int c, T v);
   
     /// assignment operator
     L1MuGMTMatrix& operator=(const L1MuGMTMatrix& m);

     /// matrix addition
     L1MuGMTMatrix& operator+=(const L1MuGMTMatrix& m);

     /// scalar addition
     L1MuGMTMatrix& operator+=(const T& s);

     /// scalar multiplication
     L1MuGMTMatrix& operator*=(const T& s);

     /// is the element (r,c) the max. entry in its row and column?
     bool isMax(int r, int c) const; 

     /// is the element (r,c) the min. entry in its row and column?
     bool isMin(int r, int c) const;

     /// is any element in column c > 0 ? return index or -1 
     int colAny(int c) const;

     /// is any element in row r > 0 ? return index or -1 
     int rowAny(int r) const;
          
     /// print matrix
     void print() const;

  private:

     int r_size, c_size;

     T **p;

};


template<class T>
L1MuGMTMatrix<T>::L1MuGMTMatrix(int r, int c) : r_size(r), c_size(c) {

   p = new T*[r];

   assert(p != 0);

   for (int i = 0; i < r; i++) {
     p[i] = new T[c];
     assert(p[i] != 0);
   }

}

// copy constructor


template<class T>
L1MuGMTMatrix<T>::L1MuGMTMatrix(const L1MuGMTMatrix<T>& mat) : r_size(mat.r_size), c_size(mat.c_size) {

   p = new T*[r_size];

   assert(p != 0);

   for (int i = 0; i < r_size; i++) {
     p[i] = new T[c_size];
     assert(p[i] != 0);
     for (int j = 0; j < c_size; j++) p[i][j] = mat.p[i][j];
   }

}


//--------------
// Destructor --
//--------------
template<class T>
L1MuGMTMatrix<T>::~L1MuGMTMatrix() {

   for (int i = 0; i < r_size; i++) {
     delete [] p[i];
   }

   delete [] p;

}


//
//
//
template<class T>
T& L1MuGMTMatrix<T>::operator()(int r, int c) {

   assert( r >= 0 && r < r_size && c >= 0 && c < c_size );

   return p[r][c];

}


//
//
//
template<class T>
const T& L1MuGMTMatrix<T>::operator()(int r, int c) const {

   assert( r >= 0 && r < r_size && c >= 0 && c < c_size );

   return p[r][c];

}


//
// initialize matrix
//
template<class T>
void L1MuGMTMatrix<T>::init(T v) {

   for (int r = 0; r < r_size; r++) {
     for (int c = 0; c < c_size; c++) p[r][c] = v;
   }

}


//
// set matrix element
//
template<class T>
void L1MuGMTMatrix<T>::set(int r, int c, T v) {

   assert( r >= 0 && r < r_size && c >= 0 && c < c_size );

   p[r][c] = v;

}


//
// assignment operator
//
template<class T>
L1MuGMTMatrix<T>& L1MuGMTMatrix<T>::operator=(const L1MuGMTMatrix& m) {

  if ( this != &m ) {
    assert(m.c_size == c_size && m.r_size == r_size);

    for (int r = 0; r < r_size; r++) {
      for (int c = 0; c < c_size; c++) p[r][c] = m.p[r][c];
    }
  }

  return (*this);

}


//
// matrix addition
//
template<class T>
L1MuGMTMatrix<T>& L1MuGMTMatrix<T>::operator+=(const L1MuGMTMatrix& m) {

   assert(m.c_size == c_size && m.r_size == r_size);

   for (int r = 0; r < r_size; r++) {
     for (int c = 0; c < c_size; c++) {
       p[r][c] += m.p[r][c];
     }
   }

   return *this;

}


//
// scalar addition
//
template<class T>
L1MuGMTMatrix<T>& L1MuGMTMatrix<T>::operator+=(const T& s) {

   for (int r = 0; r < r_size; r++) {
     for (int c = 0; c < c_size; c++) p[r][c] += s;
   }

   return *this;

}


//
// scalar multiplication
//
template<class T>
L1MuGMTMatrix<T>& L1MuGMTMatrix<T>::operator*=(const T& s) {

   for (int r = 0; r < r_size; r++) {
     for (int c = 0; c < c_size; c++) p[r][c] *= s;
   }

   return *this;

}


//
// is the element (r,c) the max. entry in its row and column?
//
template<class T>
bool L1MuGMTMatrix<T>::isMax(int r, int c) const {

  bool max = true;

  for (int i = 0; i < c; i++) {
    max = max && ( this->p[r][c] > this->p[r][i]);
  }    
  for (int i = c+1; i < c_size; i++) {
    max = max && ( this->p[r][c] >= this->p[r][i]);
  }    

  for (int j = 0; j < r; j++) {
    max = max && ( this->p[r][c] > this->p[j][c]);
  }
  for (int j = r+1; j < r_size; j++) {
    max = max && ( this->p[r][c] >= this->p[j][c]);
  }
  
  return max;
  
}


//
// is the element (r,c) the min. entry in its row and column?
//
template<class T>
bool L1MuGMTMatrix<T>::isMin(int r, int c) const {

  bool min = true;
  for (int i = 0; i < c_size; i++) {
    min = min && ( this->p[r][c] <= this->p[r][i]);
  }    
  for (int j = 0; j < r_size; j++) {
    min = min && ( this->p[r][c] <= this->p[j][c]);
  }
  
  return min;
  
} 


//
// is any element in column c > 0 ? return index or -1 
//
template<class T>
int L1MuGMTMatrix<T>::colAny(int c) const {

  int stat = -1;
  for (int i = 0; i < r_size; i++) {
    if ( this->p[i][c] > 0 ) stat = i;
  }
 
  return stat;
 
}


//
// is any element in row r > 0 ? return index or -1  
//
template<class T>
int L1MuGMTMatrix<T>::rowAny(int r) const {

  int stat = -1;
  for (int i = 0; i < c_size; i++) {
    if ( this->p[r][i] > 0 ) stat = i;
  } 

  return stat;

}


//
// print matrix
// 
template<class T>
void L1MuGMTMatrix<T>::print() const {

  for (int r = 0; r < r_size; r++) {
    std::stringstream output;
    for (int c = 0; c < c_size; c++) output << p[r][c] << "\t";
    edm::LogVerbatim("GMTMatrix_print") << output.str();
  }
  edm::LogVerbatim("GMTMatrix_print");

}

#endif
