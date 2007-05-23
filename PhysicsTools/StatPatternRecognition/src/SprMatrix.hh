// -*- C++ -*-
// CLASSDOC OFF
// $Id: SprMatrix.hh,v 1.3 2006/11/13 19:09:40 narsky Exp $
// ---------------------------------------------------------------------------
// CLASSDOC ON
//
// This file is a part of the CLHEP - a Class Library for High Energy Physics.
// 
// This is the definition of the SprMatrix class.
// 
// Copyright Cornell University 1993, 1996, All Rights Reserved.
// 
// This software written by Nobu Katayama and Mike Smyth, Cornell University.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
// 1. Redistributions of source code must retain the above copyright
//    notice and author attribution, this list of conditions and the
//    following disclaimer. 
// 2. Redistributions in binary form must reproduce the above copyright
//    notice and author attribution, this list of conditions and the
//    following disclaimer in the documentation and/or other materials
//    provided with the distribution.
// 3. Neither the name of the University nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
// 
// Creation of derivative forms of this software for commercial
// utilization may be subject to restriction; written permission may be
// obtained from Cornell University.
// 
// CORNELL MAKES NO REPRESENTATIONS OR WARRANTIES, EXPRESS OR IMPLIED.  By way
// of example, but not limitation, CORNELL MAKES NO REPRESENTATIONS OR
// WARRANTIES OF MERCANTABILITY OR FITNESS FOR ANY PARTICULAR PURPOSE OR THAT
// THE USE OF THIS SOFTWARE OR DOCUMENTATION WILL NOT INFRINGE ANY PATENTS,
// COPYRIGHTS, TRADEMARKS, OR OTHER RIGHTS.  Cornell University shall not be
// held liable for any liability with respect to any claim by the user or any
// other party arising from use of the program.
//
//
// .SS Usage
// The Matrix class does all the obvious things, in all the obvious ways.
// You declare a Matrix by saying
//
// .ft B
//       SprMatrix m1(n, m);
//
//  To declare a Matrix as a copy of a Matrix m2, say
//
// .ft B
//       SprMatrix m1(m2);
// or
// .ft B
//       SprMatrix m1 = m2;
// 
// You can declare initilizations of a Matrix, by giving it a third
// integer argument, either 0 or 1. 0 means initialize to 0, one means
// the identity matrix. If you do not give a third argument the memory
// is not initialized.
//
// .ft B
//       SprMatrix m1(n, m, 1);
//
// ./"This code has been written by Mike Smyth, and the algorithms used are
// ./"described in the thesis "A Tracking Library for a Silicon Vertex Detector"
// ./"(Mike Smyth, Cornell University, June 1993).
// ./"Copyright (C) Cornell University 1993. Permission is granted to copy and 
// ./"distribute this code, provided this copyright is not changed or deleted.
// ./"You may modify your copy, providing that you cause the modified file to
// ./"carry prominent notices stating that you changed the files, and the date
// ./"of any change. This code may not be sold, nor may it be contained in
// ./"programs that are to be sold without the written permission of the author.
// ./"You may, however, charge a fee for the physical act of transferring a
// ./"copy of this code. The code is offered "as is" without warranty of any 
// ./"kind, either expressed or implied.  By copying, distributing, or 
// ./"modifying this code you indicate your acceptance of this license to
// ./"do so, and all its terms and conditions.
// ./"This is file contains C++ stuff for doing things with Matrices.
// 
//  To find the number of rows or number of columns, say 
//
// .ft B
// nr = m1.num_row();
//
// or
//
// .ft B
// nc = m1.num_col();
//
// You can print a Matrix by
//
// .ft B
// cout << m1;
//
//  You can add,
//  subtract, and multiply two Matrices.  You can multiply a Matrix by a
//  scalar, on the left or the right.  +=, *=, -= act in the obvious way.
//  m1 *= m2 is as m1 = m1*m2. You can also divide by a scalar on the
//  right, or use /= to do the same thing.  
// 
//  You can read or write a Matrix element by saying
//
// .ft B
//  m(row, col) = blah. (1 <= row <= num_row(), 1 <= col <= num_col())
//
// .ft B
//  blah = m(row, col) ...
//
//  m(row, col) is inline, and by default does not do bound checking. 
// 
//  You can also access the element using C subscripting operator []
//
// .ft B
//  m[row][col] = blah. (0 <= row < num_row(), 0 <= col < num_col())
//
// .ft B
//  blah = m[row][col] ...
//
//  m[row][col] is inline, and by default does not do bound checking. 
//
// .SS Comments on precision
//
//  The user would normally use "Matrix" class whose precision is the same
//  as the other classes of CLHEP (ThreeVec, for example). However, he/she
//  can explicitly choose Matrix (double) or MatrixF (float) if he/she wishes.
//  In the former case, include "Matrix.h." In the latter case, include either
//  "Matrix.h," or "MatrixF.h," or both. The only operators that can talk
//  to each other are the copy constructor and assignment operator.
//
// .ft B
//  Matrix d(3,4,HEP_MATRIX_IDENTITY);
//
// .ft B
//  MatrixF f;
//
// .ft B
//  f = d;
//
// .ft B
//  MatrixF g(d);
//
//  will convert from one to the other.
//
// .SS Other functions
//
// .ft B
//  mt = m.T();
//
//  returns the transpose of m. 
//
// .ft B
//  ms = m2.sub(row_min, row_max, col_min, col_max);
//
//  returns the subMatrix.
//  m2(row_min:row_max, col_min:col_max) in matlab terminology.
//  If instead you say
//
// .ft B
//  m2.sub(row, col, m1);
//
//  then the subMatrix
//  m2(row:row+m1.num_row()-1, col:col+m1.num_col()-1) is replaced with m1.
//
// .ft B
// md = dsum(m1,m2);
//
// returns the direct sum of the two matrices.
//
// Operations that do not have to be member functions or friends are listed
// towards the end of this man page.
//
//
// To invert a matrix, say
//
// .ft B
// minv = m.inverse(ierr);
//
// ierr will be non-zero if the matrix is singular.
//
// If you can overwrite the matrix, you can use the invert function to avoid
// two extra copies. Use
//
// .ft B
// m.invert(ierr);
//
// Many basic linear algebra functions such as QR decomposition are defined.
// In order to keep the header file a reasonable size, the functions are
// defined in MatrixLinear.h.
//
// 
// .ft B 
//  Note that Matrices run from (1,1) to (n, m), and [0][0] to
//  [n-1][m-1]. The users of the latter form should be careful about sub
//  functions.
//
// ./" The program that this class was orginally used with lots of small
// ./" Matrices.  It was very costly to malloc and free array space every
// ./" time a Matrix is needed.  So, a number of previously freed arrays are
// ./" kept around, and when needed again one of these array is used.  Right
// ./" now, a set of piles of arrays with row <= row_max and col <= col_max
// ./" are kept around.  The piles of kept Matrices can be easily changed.
// ./" Array mallocing and freeing are done only in new_m() and delete_m(),
// ./" so these are the only routines that need to be rewritten.
// 
//  You can do things thinking of a Matrix as a list of numbers.  
//
// .ft B
//  m = m1.apply(HEP_MATRIX_ELEMENT (*f)(HEP_MATRIX_ELEMENT, int r, int c));
// 
//  applies f to every element of m1 and places it in m.
//
// .SS See Also:
// SymMatrix[DF].h, GenMatrix[DF].h, DiagMatrix[DF].h Vector[DF].h
// MatrixLinear[DF].h 

#ifndef _SprMatrix_HH
#define _SprMatrix_HH

#include <vector>

#include "SprGenMatrix.hh"

class SprSymMatrix;
class SprVector;

/**
 * @author
 * @ingroup matrix
 */
class SprMatrix : public SprGenMatrix {
public:
   inline SprMatrix();
   // Default constructor. Gives 0 x 0 matrix. Another Matrix can be
   // assigned to it.

   SprMatrix(int p, int q);
   // Constructor. Gives an unitialized p x q matrix.
   SprMatrix(int p, int q, int i);
   // Constructor. Gives an initialized p x q matrix. 
   // If i=0, it is initialized to all 0. If i=1, the diagonal elements
   // are set to 1.0.

   SprMatrix(const SprMatrix &m1);
   // Copy constructor.

   SprMatrix(const SprSymMatrix &m1);
   SprMatrix(const SprVector &m1);
   // Constructors from SymMatrix, DiagMatrix and Vector.

   virtual ~SprMatrix();
   // Destructor.

   inline virtual int num_row() const;
   // Returns the number of rows.

   inline virtual int num_col() const;
   // Returns the number of columns.

   inline virtual const double & operator()(int row, int col) const;
   inline virtual double & operator()(int row, int col);
   // Read or write a matrix element. 
   // ** Note that the indexing starts from (1,1). **

   SprMatrix & operator *= (double t);
   // Multiply a Matrix by a floating number.

   SprMatrix & operator /= (double t); 
   // Divide a Matrix by a floating number.

   SprMatrix & operator += ( const SprMatrix &m2);
   SprMatrix & operator += ( const SprSymMatrix &m2);
   SprMatrix & operator += ( const SprVector &m2);
   SprMatrix & operator -= ( const SprMatrix &m2);
   SprMatrix & operator -= ( const SprSymMatrix &m2);
   SprMatrix & operator -= ( const SprVector &m2);
   // Add or subtract a Matrix. 
   // When adding/subtracting Vector, Matrix must have num_col of one.

   SprMatrix & operator = ( const SprMatrix &m2);
   SprMatrix & operator = ( const SprSymMatrix &m2);
   SprMatrix & operator = ( const SprVector &m2);
   // Assignment operators.

   SprMatrix operator- () const;
   // unary minus, ie. flip the sign of each element.

   SprMatrix apply(double (*f)(double, int, int)) const;
   // Apply a function to all elements of the matrix.

   SprMatrix T() const;
   // Returns the transpose of a Matrix.

   SprMatrix sub(int min_row, int max_row, int min_col, int max_col) const;
   // Returns a sub matrix of a Matrix.
   // WARNING: rows and columns are numbered from 1
   void sub(int row, int col, const SprMatrix &m1);
   // Sub matrix of this Matrix is replaced with m1.
   // WARNING: rows and columns are numbered from 1

   friend inline void swap(SprMatrix &m1, SprMatrix &m2);
   // Swap m1 with m2.

  //   inline SprMatrix inverse(int& ierr) const;
   // Invert a Matrix. Matrix must be square and is not changed.
   // Returns ierr = 0 (zero) when successful, otherwise non-zero.

  // I removed Matrix invert functionality -Xuan Luo 18 Jul 2006
  //   virtual void invert(int& ierr);
   // Invert a Matrix. Matrix must be square.
   // N.B. the contents of the matrix are replaced by the inverse.
   // Returns ierr = 0 (zero) when successful, otherwise non-zero. 
   // This method has less overhead then inverse().

   double determinant() const;
   // calculate the determinant of the matrix.

   double trace() const;
   // calculate the trace of the matrix (sum of diagonal elements).

   class SprMatrix_row {
   public:
      inline SprMatrix_row(SprMatrix&,int);
      double & operator[](int);
   private:
      SprMatrix& a_;
      int r_;
   };
   class SprMatrix_row_const {
   public:
      inline SprMatrix_row_const (const SprMatrix&,int);
      const double & operator[](int) const;
   private:
      const SprMatrix& a_;
      int r_;
   };
   // helper classes for implementing m[i][j]

   inline SprMatrix_row operator[] (int);
   inline const SprMatrix_row_const operator[] (int) const;
   // Read or write a matrix element.
   // While it may not look like it, you simply do m[i][j] to get an
   // element. 
   // ** Note that the indexing starts from [0][0]. **

protected:
   virtual inline int num_size() const;

private:
   friend class SprMatrix_row;
   friend class SprMatrix_row_const;
   friend class SprVector;
   friend class SprSymMatrix;
   // Friend classes.

   friend SprMatrix operator+(const SprMatrix &m1, const SprMatrix &m2);
   friend SprMatrix operator-(const SprMatrix &m1, const SprMatrix &m2);
   friend SprMatrix operator*(const SprMatrix &m1, const SprMatrix &m2);
   friend SprMatrix operator*(const SprMatrix &m1, const SprSymMatrix &m2);
   friend SprMatrix operator*(const SprSymMatrix &m1, const SprMatrix &m2);
   friend SprMatrix operator*(const SprVector &m1, const SprMatrix &m2);
   friend SprVector operator*(const SprMatrix &m1, const SprVector &m2);
   friend SprMatrix operator*(const SprSymMatrix &m1, const SprSymMatrix &m2);
   // Multiply a Matrix by a Matrix or Vector.

   friend SprVector solve(const SprMatrix &, const SprVector &);
   // solve the system of linear eq

   int dfact_matrix(double &det, int *ir);
   // factorize the matrix. If successful, the return code is 0. On
   // return, det is the determinant and ir[] is row-interchange
   // matrix. See CERNLIB's DFACT routine.

   int dfinv_matrix(int *ir);
   // invert the matrix. See CERNLIB DFINV.

   std::vector<double > m;
   int nrow, ncol;
   int size;
};

// Operations other than member functions for Matrix
// implemented in Matrix.cc and Matrix.icc (inline).

SprMatrix operator*(const SprMatrix &m1, const SprMatrix &m2);
SprMatrix operator*(double t, const SprMatrix &m1);
SprMatrix operator*(const SprMatrix &m1, double t);
// Multiplication operators
// Note that m *= m1 is always faster than m = m * m1.

SprMatrix operator/(const SprMatrix &m1, double t);
// m = m1 / t. (m /= t is faster if you can use it.)

SprMatrix operator+(const SprMatrix &m1, const SprMatrix &m2);
// m = m1 + m2;
// Note that m += m1 is always faster than m = m + m1.

SprMatrix operator-(const SprMatrix &m1, const SprMatrix &m2);
// m = m1 - m2;
// Note that m -= m1 is always faster than m = m - m1.

SprMatrix dsum(const SprMatrix&, const SprMatrix&);
// Direct sum of two matrices. The direct sum of A and B is the matrix 
//        A 0
//        0 B

SprVector solve(const SprMatrix &, const SprVector &);
// solve the system of linear equations using LU decomposition.

std::ostream& operator<<(std::ostream &s, const SprMatrix &q);
// Read in, write out Matrix into a stream.
 
// -*- C++ -*-
// $Id: SprMatrix.hh,v 1.3 2006/11/13 19:09:40 narsky Exp $
// ---------------------------------------------------------------------------
//

inline SprMatrix::SprMatrix()
  : m(0), nrow(0), ncol(0), size(0) {}

inline int SprMatrix::num_row() const { return nrow;}

inline int SprMatrix::num_col() const  { return ncol;}

inline int SprMatrix::num_size() const { return size;}

inline double & SprMatrix::operator()(int row, int col)
{
  return *(m.begin()+(row-1)*ncol+col-1);
}

inline const double & SprMatrix::operator()(int row, int col) const 
{
  return *(m.begin()+(row-1)*ncol+col-1);
}

inline SprMatrix::SprMatrix_row SprMatrix::operator[] (int r)
{
  SprMatrix_row b(*this,r);
  return b;
}

inline const SprMatrix::SprMatrix_row_const SprMatrix::operator[] (int r) const
{
  SprMatrix_row_const b(*this,r);
  return b;
}

inline double &SprMatrix::SprMatrix_row::operator[](int c) {
  return *(a_.m.begin()+r_*a_.ncol+c);
}

inline const double &SprMatrix::SprMatrix_row_const::operator[](int c) const
{
  return *(a_.m.begin()+r_*a_.ncol+c);
}

inline SprMatrix::SprMatrix_row::SprMatrix_row(SprMatrix&a,int r) 
: a_(a) {
  r_ = r;
}

inline SprMatrix::SprMatrix_row_const::SprMatrix_row_const 
(const SprMatrix&a, int r) 
   : a_(a) 
{
  r_ = r;
}

// This function swaps two Matrices without doing a full copy.
inline void swap(SprMatrix &m1,SprMatrix &m2) {
  SprGenMatrix::swap(m1.m,m2.m);
/*** commented
  SprGenMatrix::swap(m1.nrow,m2.nrow);
  SprGenMatrix::swap(m1.ncol,m2.ncol);
  SprGenMatrix::swap(m1.size,m2.size);
*/
}

//  /*-ap inline */ SprMatrix SprMatrix::inverse(int &ierr) const
/*{
  SprMatrix mTmp(*this);
  mTmp.invert(ierr);
  return mTmp;
  }*/


#endif /*_Matrix_H*/
