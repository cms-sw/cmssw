// -*- C++ -*-
// CLASSDOC OFF
// $Id: SprGenMatrix.hh,v 1.3 2006/11/13 19:09:40 narsky Exp $
// ---------------------------------------------------------------------------
// CLASSDOC ON
//
// This file is a part of the CLHEP - a Class Library for High Energy Physics.
//
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
// This is the definition of the SprGenMatrix, base class for SprMatrix,
// SprSymMatrix and HepDiagMatrix. This is an abstract cless.
// See definitions in Matrix.h, SymMatrix.h, DiagMatrix.h and Vector.h

#ifndef _SprGenMatrix_HH
#define _SprGenMatrix_HH

#include <vector>

#include <iostream>

class SprGenMatrix_row;
class SprGenMatrix_row_const;
class SprGenMatrix;

/**
 * @author
 * @ingroup matrix
 */
class SprGenMatrix {
 
public:
   virtual ~SprGenMatrix() {}



   typedef std::vector<double >::iterator mIter;
   typedef std::vector<double >::const_iterator mcIter;

   virtual int num_row() const = 0;
   virtual int num_col() const = 0;

   virtual const double & operator()(int row, int col) const =0;
   virtual double & operator()(int row, int col) =0;
   // Read or write a matrix element. 
   // ** Note that the indexing starts from (1,1). **

   class SprGenMatrix_row {
   public:
      inline SprGenMatrix_row(SprGenMatrix&,int);
      double & operator[](int);
   private:
      SprGenMatrix& a_;
      int r_;
   };
   class SprGenMatrix_row_const {
   public:
      inline SprGenMatrix_row_const (const SprGenMatrix&,int);
      const double & operator[](int) const;
   private:
      const SprGenMatrix& a_;
      int r_;
   };
   // helper classes to implement m[i][j]

   inline SprGenMatrix_row operator[] (int);
   inline const SprGenMatrix_row_const operator[] (int) const;
   // Read or write a matrix element.
   // While it may not look like it, you simply do m[i][j] to get an
   // element. 
   // ** Note that the indexing starts from [0][0]. **

   inline static void swap(int&,int&);
   inline static void swap(std::vector<double >&, std::vector<double >&);

   virtual bool operator== ( const SprGenMatrix& ) const;
   // equality operator for matrices (BaBar)

   static void error(const char *s);

protected:
   virtual int num_size() const = 0;
   void delete_m(int size, double*);
   double* new_m(int size);

public:
   enum{size_max = 25};
   // This is not the maximum size of the Matrix. It is the maximum length of
   // the array (1D) which can be put on the pile.
   //
   // This enum used to be private, but it then is not accessible
   // in the definition of array_pile in the .cc file for Sun CC 4.0.1.
   // efrank@upenn5.hep.upenn.edu
 
private:
   void operator=(const SprGenMatrix &) {}
   // Remove default operator for SprGenMatrix.

   friend class SprGenMatrix_row;
   friend class SprGenMatrix_row_const;

   //-ap: removed this as it is taken over by the std::vector<double>
   //-ap  double data_array[size_max];  
};

//double norm(const SprGenMatrix &m);
double norm1(const SprGenMatrix &m);
double norm_infinity(const SprGenMatrix &m);
// 2, 1 or infinity-norm of a matrix.


// -*- C++ -*-
// $Id: SprGenMatrix.hh,v 1.3 2006/11/13 19:09:40 narsky Exp $
// ---------------------------------------------------------------------------
//

// swap
//
inline void SprGenMatrix::swap(int &i,int &j) {int t=i;i=j;j=t;}
inline void SprGenMatrix::swap(std::vector<double >& i, std::vector<double >& j) {
  std::vector<double > t=i;i=j;j=t;
}

//
// operator [] (I cannot make it virtual because return types are different.)
// Therefore I will have to use the virtual operator (,).
//
inline double &SprGenMatrix::SprGenMatrix_row::operator[](int c) {
  return a_(r_+1,c+1);
}

inline const double &SprGenMatrix::SprGenMatrix_row_const::
operator[](int c) const {
  return a_(r_+1,c+1);
}

inline SprGenMatrix::SprGenMatrix_row SprGenMatrix::operator[](int r) {
  SprGenMatrix_row b(*this,r); 
  return b;
}

inline const SprGenMatrix::SprGenMatrix_row_const SprGenMatrix::
operator[](int r) const{
  SprGenMatrix_row_const b(*this,r); 
  return b;
}

inline SprGenMatrix::SprGenMatrix_row::SprGenMatrix_row(SprGenMatrix&a,int r) 
: a_(a) {
  r_ = r;
}

inline SprGenMatrix::SprGenMatrix_row_const::
SprGenMatrix_row_const (const SprGenMatrix&a, int r) 
   : a_(a) {
  r_ = r;
}


#endif
