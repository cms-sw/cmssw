// -*- C++ -*-
// $Id: SprVector.cc,v 1.2 2006/10/19 21:27:52 narsky Exp $
// ---------------------------------------------------------------------------
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

#include <cstring>

#include "SprVector.hh"
#include "SprMatrix.hh"

// Simple operation for all elements

#define SIMPLE_UOP(OPER)          \
   register SprGenMatrix::mIter a=m.begin();            \
   register SprGenMatrix::mIter e=m.begin()+num_size(); \
   for(;a<e; a++) (*a) OPER t;

#define SIMPLE_BOP(OPER)          \
   register mIter a=m.begin();            \
   register mcIter b=m2.m.begin();               \
   register mcIter e=m.begin()+num_size(); \
   for(;a<e; a++, b++) (*a) OPER (*b);

#define SIMPLE_TOP(OPER)          \
   register SprGenMatrix::mcIter a=m1.m.begin();            \
   register SprGenMatrix::mcIter b=m2.m.begin();         \
   register SprGenMatrix::mIter t=mret.m.begin();         \
   register SprGenMatrix::mcIter e=m1.m.begin()+m1.num_size(); \
   for( ;a<e; a++, b++, t++) (*t) = (*a) OPER (*b);

#define CHK_DIM_2(r1,r2,c1,c2,fun) \
   if (r1!=r2 || c1!=c2)  { \
     SprGenMatrix::error("Range error in Vector function " #fun "(1)."); \
   }

#define CHK_DIM_1(c1,r2,fun) \
   if (c1!=r2) { \
     SprGenMatrix::error("Range error in Vector function " #fun "(2)."); \
   }

// Constructors. (Default constructors are inlined and in .icc file)

SprVector::SprVector(int p)
   : m(p), nrow(p)
{
}

SprVector::SprVector(int p, int init)
   : m(p), nrow(p)
{
   switch (init)
   {
   case 0:
      m.assign(p,0);
      break;
      
   case 1:
      {
	 mIter e = m.begin() + nrow;
	 for (mIter i=m.begin(); i<e; i++) *i = 1.0;
	 break;
      }
      
   default:
      error("Vector: initialization must be either 0 or 1.");
   }
}

//
// Destructor
//
SprVector::~SprVector() {
}

SprVector::SprVector(const SprVector &m1)
   : m(m1.nrow), nrow(m1.nrow)
{
   m = m1.m;
}

//
// Copy constructor from the class of other precision
//


SprVector::SprVector(const SprMatrix &m1)
   : m(m1.nrow), nrow(m1.nrow)
{
   if (m1.num_col() != 1)
      error("Vector::Vector(Matrix) : Matrix is not Nx1");
   
   m = m1.m;
}

// Sub matrix

SprVector SprVector::sub(int min_row, int max_row) const
{
  SprVector vret(max_row-min_row+1);
  if(max_row > num_row())
    error("SprVector::sub: Index out of range");
  SprGenMatrix::mIter a = vret.m.begin();
  SprGenMatrix::mcIter b = m.begin() + min_row - 1;
  SprGenMatrix::mIter e = vret.m.begin() + vret.num_row();
  for(;a<e;) *(a++) = *(b++);
  return vret;
}

SprVector SprVector::sub(int min_row, int max_row)
{
  SprVector vret(max_row-min_row+1);
  if(max_row > num_row())
    error("SprVector::sub: Index out of range");
  SprGenMatrix::mIter a = vret.m.begin();
  SprGenMatrix::mIter b = m.begin() + min_row - 1;
  SprGenMatrix::mIter e = vret.m.begin() + vret.num_row();
  for(;a<e;) *(a++) = *(b++);
  return vret;
}

void SprVector::sub(int row,const SprVector &v1)
{
  if(row <1 || row+v1.num_row()-1 > num_row())
    error("SprVector::sub: Index out of range");
  SprGenMatrix::mcIter a = v1.m.begin();
  SprGenMatrix::mIter b = m.begin() + row - 1;
  SprGenMatrix::mcIter e = v1.m.begin() + v1.num_row();
  for(;a<e;) *(b++) = *(a++);
}

//
// Direct sum of two matricies
//

SprVector dsum(const SprVector &m1,
				     const SprVector &m2)
{
  SprVector mret(m1.num_row() + m2.num_row(),
				       0);
  mret.sub(1,m1);
  mret.sub(m1.num_row()+1,m2);
  return mret;
}

/* -----------------------------------------------------------------------
   This section contains support routines for matrix.h. This section contains
   The two argument functions +,-. They call the copy constructor and +=,-=.
   ----------------------------------------------------------------------- */
SprVector SprVector::operator- () const 
{
   SprVector m2(nrow);
   register SprGenMatrix::mcIter a=m.begin();
   register SprGenMatrix::mIter b=m2.m.begin();
   register SprGenMatrix::mcIter e=m.begin()+num_size();
   for(;a<e; a++, b++) (*b) = -(*a);
   return m2;
}

   

SprVector operator+(const SprMatrix &m1,const SprVector &m2)
{
  SprVector mret(m2);
  CHK_DIM_2(m1.num_row(),m2.num_row(),m1.num_col(),1,+);
  mret += m1;
  return mret;
}

SprVector operator+(const SprVector &m1,const SprMatrix &m2)
{
  SprVector mret(m1);
  CHK_DIM_2(m1.num_row(),m2.num_row(),1,m2.num_col(),+);
  mret += m2;
  return mret;
}

SprVector operator+(const SprVector &m1,const SprVector &m2)
{
  SprVector mret(m1.num_row());
  CHK_DIM_1(m1.num_row(),m2.num_row(),+);
  SIMPLE_TOP(+)
  return mret;
}

//
// operator -
//

SprVector operator-(const SprMatrix &m1,const SprVector &m2)
{
  SprVector mret;
  CHK_DIM_2(m1.num_row(),m2.num_row(),m1.num_col(),1,-);
  mret = m1;
  mret -= m2;
  return mret;
}

SprVector operator-(const SprVector &m1,const SprMatrix &m2)
{
  SprVector mret(m1);
  CHK_DIM_2(m1.num_row(),m2.num_row(),1,m2.num_col(),-);
  mret -= m2;
  return mret;
}

SprVector operator-(const SprVector &m1,const SprVector &m2)
{
  SprVector mret(m1.num_row());
  CHK_DIM_1(m1.num_row(),m2.num_row(),-);
  SIMPLE_TOP(-)
  return mret;
}

/* -----------------------------------------------------------------------
   This section contains support routines for matrix.h. This file contains
   The two argument functions *,/. They call copy constructor and then /=,*=.
   ----------------------------------------------------------------------- */

SprVector operator/(
const SprVector &m1,double t)
{
  SprVector mret(m1);
  mret /= t;
  return mret;
}

SprVector operator*(const SprVector &m1,double t)
{
  SprVector mret(m1);
  mret *= t;
  return mret;
}

SprVector operator*(double t,const SprVector &m1)
{
  SprVector mret(m1);
  mret *= t;
  return mret;
}

SprVector operator*(const SprMatrix &m1,const SprVector &m2)
{
  SprVector mret(m1.num_row());
  CHK_DIM_1(m1.num_col(),m2.num_row(),*);
  SprGenMatrix::mcIter m1p,m2p,vp;
  SprGenMatrix::mIter m3p;
  double temp;
  m3p=mret.m.begin();
  for(m1p=m1.m.begin();m1p<m1.m.begin()+m1.num_row()*m1.num_col();m1p=m2p)
    {
      temp=0;
      vp=m2.m.begin();
      m2p=m1p;
      while(m2p<m1p+m1.num_col())
	temp+=(*(m2p++))*(*(vp++));
      *(m3p++)=temp;
    }
  return mret;
}

SprMatrix operator*(const SprVector &m1,const SprMatrix &m2)
{
  SprMatrix mret(m1.num_row(),m2.num_col());
  CHK_DIM_1(1,m2.num_row(),*);
  SprGenMatrix::mcIter m1p;
  SprMatrix::mcIter m2p;
  SprMatrix::mIter mrp=mret.m.begin();
  for(m1p=m1.m.begin();m1p<m1.m.begin()+m1.num_row();m1p++)
    for(m2p=m2.m.begin();m2p<m2.m.begin()+m2.num_col();m2p++)
      *(mrp++)=*m1p*(*m2p);
  return mret;
}

/* -----------------------------------------------------------------------
   This section contains the assignment and inplace operators =,+=,-=,*=,/=.
   ----------------------------------------------------------------------- */

SprMatrix & SprMatrix::operator+=(const SprVector &m2)
{
  CHK_DIM_2(num_row(),m2.num_row(),num_col(),1,+=);
  SIMPLE_BOP(+=)
  return (*this);
}

SprVector & SprVector::operator+=(const SprMatrix &m2)
{
  CHK_DIM_2(num_row(),m2.num_row(),1,m2.num_col(),+=);
  SIMPLE_BOP(+=)
  return (*this);
}

SprVector & SprVector::operator+=(const SprVector &m2)
{
  CHK_DIM_1(num_row(),m2.num_row(),+=);
  SIMPLE_BOP(+=)
  return (*this);
}

SprMatrix &  SprMatrix::operator-=(const SprVector &m2)
{
  CHK_DIM_2(num_row(),m2.num_row(),num_col(),1,-=);
  SIMPLE_BOP(-=)
  return (*this);
}

SprVector & SprVector::operator-=(const SprMatrix &m2)
{
  CHK_DIM_2(num_row(),m2.num_row(),1,m2.num_col(),-=);
  SIMPLE_BOP(-=)
  return (*this);
}

SprVector & SprVector::operator-=(const SprVector &m2)
{
  CHK_DIM_1(num_row(),m2.num_row(),-=);
  SIMPLE_BOP(-=)
  return (*this);
}

SprVector & SprVector::operator/=(double t)
{
  SIMPLE_UOP(/=)
  return (*this);
}

SprVector & SprVector::operator*=(double t)
{
  SIMPLE_UOP(*=)
  return (*this);
}

SprMatrix & SprMatrix::operator=(const SprVector &m1)
{
   if(m1.nrow != size)
   {
      size = m1.nrow;
      m.resize(size);
   }
   nrow = m1.nrow;
   ncol = 1;
   m = m1.m;
   return (*this);
}

SprVector & SprVector::operator=(const SprVector &m1)
{
   if(m1.nrow != nrow)
   {
      nrow = m1.nrow;
      m.resize(nrow);
   }
   m = m1.m;
   return (*this);
}

SprVector & SprVector::operator=(const SprMatrix &m1)
{
   if (m1.num_col() != 1)
      error("Vector::operator=(Matrix) : Matrix is not Nx1");
   
   if(m1.nrow != nrow)
   {
      nrow = m1.nrow;
      m.resize(nrow);
   }
   m = m1.m;
   return (*this);
}

//
// Copy constructor from the class of other precision
//


// Print the Matrix.

std::ostream& operator<<(std::ostream &s, const SprVector &q)
{
  s << std::endl;
/* Fixed format needs 3 extra characters for field, while scientific needs 7 */
  int width;
  if(s.flags() & std::ios::fixed)
    width = s.precision()+3;
  else
    width = s.precision()+7;
  for(int irow = 1; irow<= q.num_row(); irow++)
    {
      s.width(width);
      s << q(irow) << std::endl;
    }
  return s;
}

SprMatrix SprVector::T() const
{
  SprMatrix mret(1,num_row());
  mret.m = m;
  return mret;
}

double dot(const SprVector &v1,const SprVector &v2)
{
  if(v1.num_row()!=v2.num_row())
     SprGenMatrix::error("v1 and v2 need to be the same size in dot(SprVector, SprVector)");
  double d= 0;
  SprGenMatrix::mcIter a = v1.m.begin();
  SprGenMatrix::mcIter b = v2.m.begin();
  SprGenMatrix::mcIter e = a + v1.num_size();
  for(;a<e;) d += (*(a++)) * (*(b++));
  return d;
}

SprVector SprVector::
apply(double (*f)(double, int)) const
{
  SprVector mret(num_row());
  SprGenMatrix::mcIter a = m.begin();
  SprGenMatrix::mIter b = mret.m.begin();
  for(int ir=1;ir<=num_row();ir++) {
    *(b++) = (*f)(*(a++), ir);
  }
  return mret;
}

void SprVector::invert(int &) {
   error("SprVector::invert: You cannot invert a Vector");
}

SprVector solve(const SprMatrix &a, const SprVector &v)
{
  SprVector vret(v);
  static int max_array = 20;
  static int *ir = new int [max_array+1];

  if(a.ncol != a.nrow)
     SprGenMatrix::error("Matrix::solve Matrix is not NxN");
  if(a.ncol != v.nrow)
     SprGenMatrix::error("Matrix::solve Vector has wrong number of rows");

  int n = a.ncol;
  if (n > max_array) {
    delete [] ir;
    max_array = n;
    ir = new int [max_array+1];
  }
  double det;
  SprMatrix mt(a);
  int i = mt.dfact_matrix(det, ir);
  if (i!=0) {
    for (i=1;i<=n;i++) vret(i) = 0;
    return vret;
  }
  double s21, s22;
  int nxch = ir[n];
  if (nxch!=0) {
    for (int mm=1;mm<=nxch;mm++) {
      int ij = ir[mm];
      i = ij >> 12;
      int j = ij%4096;
      double te = vret(i);
      vret(i) = vret(j);
      vret(j) = te;
    }
  }
  vret(1) = mt(1,1) * vret(1);
  if (n!=1) {
    for (i=2;i<=n;i++) {
      s21 = -vret(i);
      for (int j=1;j<i;j++) {
	s21 += mt(i,j) * vret(j); 
      }
      vret(i) = -mt(i,i)*s21;
    }
    for (i=1;i<n;i++) {
      int nmi = n-i;
      s22 = -vret(nmi);
      for (int j=1;j<=i;j++) {
	s22 += mt(nmi,n-j+1) * vret(n-j+1);
      }
      vret(nmi) = -s22;
    }
  }
  return vret;
}

