// -*- C++ -*-
// CLASSDOC OFF
// $Id: SprVector.hh,v 1.2 2006/10/19 21:27:52 narsky Exp $
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
//   Although Vector and Matrix class are very much related, I like the typing
//   information I get by making them different types. It is usually an error
//   to use a Matrix where a Vector is expected, except in the case that the
//   Matrix is a single column.  But this case can be taken care of by using
//   constructors as conversions.  For this same reason, I don't want to make
//   a Vector a derived class of Matrix.
//

#ifndef _SprVector_HH
#define _SprVector_HH

#include "SprGenMatrix.hh"

class SprMatrix;
class SprSymMatrix;

/**
 * @author
 * @ingroup matrix
 */
class SprVector : public SprGenMatrix {
public:
   inline SprVector();
   // Default constructor. Gives vector of length 0.
   // Another Vector can be assigned to it.

   explicit SprVector(int p);
   SprVector(int p, int);
   // Constructor. Gives vector of length p.

   SprVector(const SprVector &v);
   SprVector(const SprMatrix &m);
   // Copy constructors.
   // Note that there is an assignment operator for v = Hep3Vector.

   virtual ~SprVector();
   // Destructor.

   inline const double & operator()(int row) const;
   inline double & operator()(int row);
   // Read or write a matrix element. 
   // ** Note that the indexing starts from (1). **
   
   inline const double & operator[](int row) const;
   inline double & operator[](int row);
   // Read and write an element of a Vector.
   // ** Note that the indexing starts from [0]. **

   inline virtual const double & operator()(int row, int col) const;
   inline virtual double & operator()(int row, int col);
   // Read or write a matrix element. 
   // ** Note that the indexing starts from (1,1). **
   // Allows accessing Vector using GenMatrix

   SprVector & operator*=(double t);
   // Multiply a Vector by a floating number. 

   SprVector & operator/=(double t); 
   // Divide a Vector by a floating number.

   SprVector & operator+=( const SprMatrix &v2);
   SprVector & operator+=( const SprVector &v2);
   SprVector & operator-=( const SprMatrix &v2);
   SprVector & operator-=( const SprVector &v2);
   // Add or subtract a Vector.

   SprVector & operator=( const SprVector &m2);
   // Assignment operators.

   SprVector& operator=(const SprMatrix &);
   // assignment operators from other classes.

   SprVector operator- () const;
   // unary minus, ie. flip the sign of each element.

   SprVector apply(double (*f)(double, int)) const;
   // Apply a function to all elements.

   SprVector sub(int min_row, int max_row) const;
   // Returns a sub vector.
   SprVector sub(int min_row, int max_row);
   // SGI CC bug. I have to have both with/without const. I should not need
   // one without const.

   void sub(int row, const SprVector &v1);
   // Replaces a sub vector of a Vector with v1.

   inline double normsq() const;
   // Returns norm squared.

   inline double norm() const;
   // Returns norm.

   inline virtual int num_row() const;
   // Returns number of rows.

   inline virtual int num_col() const;
   // Number of columns. Always returns 1. Provided for compatibility with
   // GenMatrix. 

   SprMatrix T() const;
   // Returns the transpose of a Vector. Note that the returning type is
   // Matrix.

   friend inline void swap(SprVector &v1, SprVector &v2);
   // Swaps two vectors.

protected:
   virtual inline int num_size() const;

private:
   virtual void invert(int&);
   // produces an error. Demanded by GenMatrix

   friend class SprSymMatrix;
   friend class SprMatrix;
   // friend classes

   friend double dot(const SprVector &v1, const SprVector &v2);
   // f = v1 * v2;

   friend SprVector operator+(const SprVector &v1, const SprVector &v2);
   friend SprVector operator-(const SprVector &v1, const SprVector &v2);
   friend SprVector operator*(const SprSymMatrix &m1, const SprVector &m2);
   friend SprMatrix operator*(const SprVector &m1, const SprMatrix &m2);
   friend SprVector operator*(const SprMatrix &m1, const SprVector &m2);

   friend SprVector solve(const SprMatrix &a, const SprVector &v);
   friend SprSymMatrix vT_times_v(const SprVector &v);

   std::vector<double > m;
   int nrow;
};

//
// Operations other than member functions
//

std::ostream& operator<<(std::ostream &s, const SprVector &v);
// Write out Matrix, SymMatrix, DiagMatrix and Vector into ostream.

SprVector operator*(const SprMatrix &m1, const SprVector &m2);
SprVector operator*(double t, const SprVector &v1);
SprVector operator*(const SprVector &v1, double t);
// Multiplication operators.
// Note that m *= x is always faster than m = m * x.

SprVector operator/(const SprVector &v1, double t);
// Divide by a real number.

SprVector operator+(const SprMatrix &m1, const SprVector &v2);
SprVector operator+(const SprVector &v1, const SprMatrix &m2);
SprVector operator+(const SprVector &v1, const SprVector &v2);
// Addition operators

SprVector operator-(const SprMatrix &m1, const SprVector &v2);
SprVector operator-(const SprVector &v1, const SprMatrix &m2);
SprVector operator-(const SprVector &v1, const SprVector &v2);
// subtraction operators

SprVector dsum(const SprVector &s1, const SprVector &s2);
// Direct sum of two vectors;

// -*- C++ -*-
// $Id: SprVector.hh,v 1.2 2006/10/19 21:27:52 narsky Exp $
// ---------------------------------------------------------------------------
//

#include <cmath>
#include <cstdlib>

// Swap two vectors without doing a full copy.
inline void swap(SprVector &v1,SprVector &v2) {
  SprGenMatrix::swap(v1.m,v2.m);
  SprGenMatrix::swap(v1.nrow,v2.nrow);
}

inline SprVector::SprVector()
   : m(0), nrow(0)
{}

inline double SprVector::normsq() const {return dot((*this),(*this));}
inline double SprVector::norm() const {return sqrt(normsq());}
inline int SprVector::num_row() const {return nrow;} 
inline int SprVector::num_size() const {return nrow;} 
inline int SprVector::num_col() const { return 1; }

inline double & SprVector::operator()(int row)
{

  return *(m.begin()+row-1);
}
inline const double & SprVector::operator()(int row) const 
{

  return *(m.begin()+row-1);
}
inline double & SprVector::operator[](int row)
{

  return *(m.begin()+row);
}
inline const double & SprVector::operator[](int row) const 
{

  return *(m.begin()+row);
}

inline double & SprVector::operator()(int row, int)
{

  return *(m.begin()+(row-1));
}

inline const double & SprVector::operator()(int row, int) const 
{

  return *(m.begin()+(row-1));
}

#endif /*!_Vector_H*/
