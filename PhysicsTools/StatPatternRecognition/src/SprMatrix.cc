// -*- C++ -*-
// $Id: SprMatrix.cc,v 1.2 2006/10/19 21:27:52 narsky Exp $
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

#include <cfloat>        // for DBL_EPSILON
#include <cmath>
#include <cstdlib>

#include "SprMatrix.hh"
#include "SprSymMatrix.hh"
#include "SprVector.hh"

// Simple operation for all elements

#define SIMPLE_UOP(OPER)                            \
   register mIter a=m.begin();                      \
   register mIter e=m.end();                        \
   for(;a!=e; a++) (*a) OPER t;

#define SIMPLE_BOP(OPER)                            \
   register SprMatrix::mIter a=m.begin();                      \
   register SprMatrix::mcIter b=m2.m.begin();                  \
   register SprMatrix::mIter e=m.end();                        \
   for(;a!=e; a++, b++) (*a) OPER (*b);

#define SIMPLE_TOP(OPER)                            \
   register SprMatrix::mcIter a=m1.m.begin();       \
   register SprMatrix::mcIter b=m2.m.begin();       \
   register SprMatrix::mIter t=mret.m.begin();      \
   register SprMatrix::mcIter e=m1.m.end();         \
   for(;a!=e; a++, b++, t++) (*t) = (*a) OPER (*b);

// Static functions.

#define CHK_DIM_2(r1,r2,c1,c2,fun) \
   if (r1!=r2 || c1!=c2)  { \
     SprGenMatrix::error("Range error in Matrix function " #fun "(1)."); \
   }

#define CHK_DIM_1(c1,r2,fun) \
   if (c1!=r2) { \
     SprGenMatrix::error("Range error in Matrix function " #fun "(2)."); \
   }

// Constructors. (Default constructors are inlined and in .icc file)

SprMatrix::SprMatrix(int p,int q)
   : m(p*q), nrow(p), ncol(q)
{
  size = nrow * ncol;
}

SprMatrix::SprMatrix(int p,int q,int init)
   : m(p*q), nrow(p), ncol(q)
{
   size = nrow * ncol;

   if (size > 0) {
      switch(init)
      {
      case 0:
	 break;

      case 1:
	 {
	    if ( ncol == nrow ) {
 	       mIter a = m.begin();
 	       mIter b = m.end();
	       for( ; a<b; a+=(ncol+1)) *a = 1.0;
	    } else {
	       error("Invalid dimension in SprMatrix(int,int,1).");
	    }
	    break;
	 }
      default:
	 error("Matrix: initialization must be either 0 or 1.");
      }
   }
}

//
// Destructor
//
SprMatrix::~SprMatrix() {
}

SprMatrix::SprMatrix(const SprMatrix &m1)
   : m(m1.size), nrow(m1.nrow), ncol(m1.ncol), size(m1.size)
{
   m = m1.m;

}

SprMatrix::SprMatrix(const SprSymMatrix &m1)
   : m(m1.nrow*m1.nrow), nrow(m1.nrow), ncol(m1.nrow)
{
   size = nrow * ncol;

   int n = ncol;
   mcIter sjk = m1.m.begin();
   mIter m1j = m.begin();
   mIter mj  = m.begin();
   // j >= k
   for(int j=1;j<=nrow;j++) {
      mIter mjk = mj;
      mIter mkj = m1j;
      for(int k=1;k<=j;k++) {
	 *(mjk++) = *sjk;
	 if(j!=k) *mkj = *sjk;
	 sjk++;
	 mkj += n;
      }
      mj += n;
      m1j++;
   }
}

SprMatrix::SprMatrix(const SprVector &m1)
   : m(m1.nrow), nrow(m1.nrow), ncol(1)
{

   size = nrow;
   m = m1.m;
}


//
//
// Sub matrix
//
//

SprMatrix SprMatrix::sub(int min_row, int max_row,
			 int min_col,int max_col) const
{
  SprMatrix mret(max_row-min_row+1,max_col-min_col+1);
  if(max_row > num_row() || max_col >num_col())
    error("SprMatrix::sub: Index out of range");
  mIter a = mret.m.begin();
  int nc = num_col();
  mcIter b1 = m.begin() + (min_row - 1) * nc + min_col - 1;
  
  for(int irow=1; irow<=mret.num_row(); irow++) {
    mcIter brc = b1;
    for(int icol=1; icol<=mret.num_col(); icol++) {
      *(a++) = *(brc++);
    }
    b1 += nc;
  }
  return mret;
}

void SprMatrix::sub(int row,int col,const SprMatrix &m1)
{
  if(row <1 || row+m1.num_row()-1 > num_row() || 
     col <1 || col+m1.num_col()-1 > num_col()   )
    error("SprMatrix::sub: Index out of range");
  mcIter a = m1.m.begin();
  int nc = num_col();
  mIter b1 = m.begin() + (row - 1) * nc + col - 1;
  
  for(int irow=1; irow<=m1.num_row(); irow++) {
    mIter brc = b1;
    for(int icol=1; icol<=m1.num_col(); icol++) {
      *(brc++) = *(a++);
    }
    b1 += nc;
  }
}

//
// Direct sum of two matricies
//

SprMatrix dsum(const SprMatrix &m1, const SprMatrix &m2)
{
  SprMatrix mret(m1.num_row() + m2.num_row(), m1.num_col() + m2.num_col(),
		 0);
  mret.sub(1,1,m1);
  mret.sub(m1.num_row()+1,m1.num_col()+1,m2);
  return mret;
}

/* -----------------------------------------------------------------------
   This section contains support routines for matrix.h. This section contains
   The two argument functions +,-. They call the copy constructor and +=,-=.
   ----------------------------------------------------------------------- */
SprMatrix SprMatrix::operator- () const 
{
   SprMatrix m2(nrow, ncol);
   register mcIter a=m.begin();
   register mIter b=m2.m.begin();
   register mcIter e=m.end();
   for(;a<e; a++, b++) (*b) = -(*a);
   return m2;
}

   

SprMatrix operator+(const SprMatrix &m1,const SprMatrix &m2)
{
  SprMatrix mret(m1.nrow, m1.ncol);
  CHK_DIM_2(m1.num_row(),m2.num_row(), m1.num_col(),m2.num_col(),+);
  SIMPLE_TOP(+)
  return mret;
}

//
// operator -
//

SprMatrix operator-(const SprMatrix &m1,const SprMatrix &m2)
{
  SprMatrix mret(m1.num_row(), m1.num_col());
  CHK_DIM_2(m1.num_row(),m2.num_row(),
			 m1.num_col(),m2.num_col(),-);
  SIMPLE_TOP(-)
  return mret;
}

/* -----------------------------------------------------------------------
   This section contains support routines for matrix.h. This file contains
   The two argument functions *,/. They call copy constructor and then /=,*=.
   ----------------------------------------------------------------------- */

SprMatrix operator/(
const SprMatrix &m1,double t)
{
  SprMatrix mret(m1);
  mret /= t;
  return mret;
}

SprMatrix operator*(const SprMatrix &m1,double t)
{
  SprMatrix mret(m1);
  mret *= t;
  return mret;
}

SprMatrix operator*(double t,const SprMatrix &m1)
{
  SprMatrix mret(m1);
  mret *= t;
  return mret;
}

SprMatrix operator*(const SprMatrix &m1,const SprMatrix &m2)
{
  // initialize matrix to 0.0
  SprMatrix mret(m1.nrow,m2.ncol,0);
  CHK_DIM_1(m1.ncol,m2.nrow,*);

  int m1cols = m1.ncol;
  int m2cols = m2.ncol;

  for (int i=0; i<m1.nrow; i++)
  {
     for (int j=0; j<m1cols; j++) 
     {
	register double temp = m1.m[i*m1cols+j];
	register SprMatrix::mIter pt = mret.m.begin() + i*m2cols;
	
	// Loop over k (the column index in matrix m2)
	register SprMatrix::mcIter pb = m2.m.begin() + m2cols*j;
	const SprMatrix::mcIter pblast = pb + m2cols;
	while (pb < pblast)
	{
	   (*pt) += temp * (*pb);
	   pb++;
	   pt++;
	}
     }
  }

  return mret;
}

/* -----------------------------------------------------------------------
   This section contains the assignment and inplace operators =,+=,-=,*=,/=.
   ----------------------------------------------------------------------- */

SprMatrix & SprMatrix::operator+=(const SprMatrix &m2)
{
  CHK_DIM_2(num_row(),m2.num_row(),num_col(),m2.num_col(),+=);
  SIMPLE_BOP(+=)
  return (*this);
}

SprMatrix & SprMatrix::operator-=(const SprMatrix &m2)
{
  CHK_DIM_2(num_row(),m2.num_row(),num_col(),m2.num_col(),-=);
  SIMPLE_BOP(-=)
  return (*this);
}

SprMatrix & SprMatrix::operator/=(double t)
{
  SIMPLE_UOP(/=)
  return (*this);
}

SprMatrix & SprMatrix::operator*=(double t)
{
  SIMPLE_UOP(*=)
  return (*this);
}

SprMatrix & SprMatrix::operator=(const SprMatrix &m1)
{
   if(m1.nrow*m1.ncol != size) //??fixme?? m1.size != size
   {
      size = m1.nrow * m1.ncol;
      m.resize(size); //??fixme?? if (size < m1.size) m.resize(m1.size);
   }
   nrow = m1.nrow;
   ncol = m1.ncol;
   m = m1.m;
   return (*this);
}

// SprMatrix & SprMatrix::operator=(const HepRotation &m2) 
// is now in Matrix=Rotation.cc

// Print the Matrix.

std::ostream& operator<<(std::ostream &s, const SprMatrix &q)
{
  s << "\n";
/* Fixed format needs 3 extra characters for field, while scientific needs 7 */
  int width;
  if(s.flags() & std::ios::fixed)
    width = s.precision()+3;
  else
    width = s.precision()+7;
  for(int irow = 1; irow<= q.num_row(); irow++)
    {
      for(int icol = 1; icol <= q.num_col(); icol++)
	{
	  s.width(width);
	  s << q(irow,icol) << " ";
	}
      s << std::endl;
    }
  return s;
}

SprMatrix SprMatrix::T() const
{
   SprMatrix mret(ncol,nrow);
   register mcIter pl = m.end();
   register mcIter pme = m.begin();
   register mIter pt = mret.m.begin();
   register mIter ptl = mret.m.end();
   for (; pme < pl; pme++, pt+=nrow)
   {
      if (pt >= ptl) pt -= (size-1);
      (*pt) = (*pme);
   }
   return mret;
}

SprMatrix SprMatrix::apply(double (*f)(double, int, int)) const
{
  SprMatrix mret(num_row(),num_col());
  mcIter a = m.begin();
  mIter b = mret.m.begin();
  for(int ir=1;ir<=num_row();ir++) {
    for(int ic=1;ic<=num_col();ic++) {
      *(b++) = (*f)(*(a++), ir, ic);
    }
  }
  return mret;
}

int SprMatrix::dfinv_matrix(int *ir) {
  if (num_col()!=num_row())
    error("dfinv_matrix: Matrix is not NxN");
  register int n = num_col();
  if (n==1) return 0;

  double s31, s32;
  register double s33, s34;

  mIter m11 = m.begin();
  mIter m12 = m11 + 1;
  mIter m21 = m11 + n;
  mIter m22 = m12 + n;
  *m21 = -(*m22) * (*m11) * (*m21);
  *m12 = -(*m12);
  if (n>2) {
    mIter mi = m.begin() + 2 * n;
    mIter mii= m.begin() + 2 * n + 2;
    mIter mimim = m.begin() + n + 1;
    for (int i=3;i<=n;i++) {
      int im2 = i - 2;
      mIter mj = m.begin();
      mIter mji = mj + i - 1;
      mIter mij = mi;
      for (int j=1;j<=im2;j++) { 
	s31 = 0.0;
	s32 = *mji;
	mIter mkj = mj + j - 1;
	mIter mik = mi + j - 1;
	mIter mjkp = mj + j;
	mIter mkpi = mj + n + i - 1;
	for (int k=j;k<=im2;k++) {
	  s31 += (*mkj) * (*(mik++));
	  s32 += (*(mjkp++)) * (*mkpi);
	  mkj += n;
	  mkpi += n;
	}
	*mij = -(*mii) * (((*(mij-n)))*( (*(mii-1)))+(s31));
	*mji = -s32;
	mj += n;
	mji += n;
	mij++;
      }
      *(mii-1) = -(*mii) * (*mimim) * (*(mii-1));
      *(mimim+1) = -(*(mimim+1));
      mi += n;
      mimim += (n+1);
      mii += (n+1);
    }
  }
  mIter mi = m.begin();
  mIter mii = m.begin();
  for (int i=1;i<n;i++) {
    int ni = n - i;
    mIter mij = mi;
    int j;
    for (j=1; j<=i;j++) {
      s33 = *mij;
      register mIter mikj = mi + n + j - 1;
      register mIter miik = mii + 1;
      mIter min_end = mi + n;
      for (;miik<min_end;) {
	s33 += (*mikj) * (*(miik++));
	mikj += n;
      }
      *(mij++) = s33;
    }
    for (j=1;j<=ni;j++) {
      s34 = 0.0;
      mIter miik = mii + j;
      mIter mikij = mii + j * n + j;
      for (int k=j;k<=ni;k++) {
	s34 += *mikij * (*(miik++));
	mikij += n;
      }
      *(mii+j) = s34;
    }
    mi += n;
    mii += (n+1);
  }
  int nxch = ir[n];
  if (nxch==0) return 0;
  for (int mm=1;mm<=nxch;mm++) {
    int k = nxch - mm + 1;
    int ij = ir[k];
    int i = ij >> 12;
    int j = ij%4096;
    mIter mki = m.begin() + i - 1;
    mIter mkj = m.begin() + j - 1;
    for (k=1; k<=n;k++) {
      // 2/24/05 David Sachs fix of improper swap bug that was present
      // for many years:
      double ti = *mki; // 2/24/05
      *mki = *mkj;
      *mkj = ti;	// 2/24/05
      mki += n;
      mkj += n;
    }
  }
  return 0;
}

int SprMatrix::dfact_matrix(double &det, int *ir) {
  if (ncol!=nrow)
     error("dfact_matrix: Matrix is not NxN");

  int ifail, jfail;
  register int n = ncol;

  double tf;
  double g1 = 1.0e-19, g2 = 1.0e19;

  double p, q, t;
  double s11, s12;

  double epsilon = 8*DBL_EPSILON;
  // could be set to zero (like it was before)
  // but then the algorithm often doesn't detect
  // that a matrix is singular

  int normal = 0, imposs = -1;
  int jrange = 0, jover = 1, junder = -1;
  ifail = normal;
  jfail = jrange;
  int nxch = 0;
  det = 1.0;
  mIter mj = m.begin();
  mIter mjj = mj;
  for (int j=1;j<=n;j++) {
    int k = j;
    p = (fabs(*mjj));
    if (j!=n) {
      mIter mij = mj + n + j - 1; 
      for (int i=j+1;i<=n;i++) {
	q = (fabs(*(mij)));
	if (q > p) {
	  k = i;
	  p = q;
	}
	mij += n;
      }
      if (k==j) {
	if (p <= epsilon) {
	  det = 0;
	  ifail = imposs;
	  jfail = jrange;
	  return ifail;
	}
	det = -det; // in this case the sign of the determinant
	            // must not change. So I change it twice. 
      }
      mIter mjl = mj;
      mIter mkl = m.begin() + (k-1)*n;
      for (int l=1;l<=n;l++) {
        tf = *mjl;
        *(mjl++) = *mkl;
        *(mkl++) = tf;
      }
      nxch = nxch + 1;  // this makes the determinant change its sign
      ir[nxch] = (((j)<<12)+(k));
    } else {
      if (p <= epsilon) {
	det = 0.0;
	ifail = imposs;
	jfail = jrange;
	return ifail;
      }
    }
    det *= *mjj;
    *mjj = 1.0 / *mjj;
    t = (fabs(det));
    if (t < g1) {
      det = 0.0;
      if (jfail == jrange) jfail = junder;
    } else if (t > g2) {
      det = 1.0;
      if (jfail==jrange) jfail = jover;
    }
    if (j!=n) {
      mIter mk = mj + n;
      mIter mkjp = mk + j;
      mIter mjk = mj + j;
      for (k=j+1;k<=n;k++) {
	s11 = - (*mjk);
	s12 = - (*mkjp);
	if (j!=1) {
	  mIter mik = m.begin() + k - 1;
	  mIter mijp = m.begin() + j;
	  mIter mki = mk;
	  mIter mji = mj;
	  for (int i=1;i<j;i++) {
	    s11 += (*mik) * (*(mji++));
	    s12 += (*mijp) * (*(mki++));
	    mik += n;
	    mijp += n;
	  }
	}
	*(mjk++) = -s11 * (*mjj);
	*(mkjp) = -(((*(mjj+1)))*((*(mkjp-1)))+(s12));
	mk += n;
	mkjp += n;
      }
    }
    mj += n;
    mjj += (n+1);
  }
  if (nxch%2==1) det = -det;
  if (jfail !=jrange) det = 0.0;
  ir[n] = nxch;
  return 0;
}

// I removed Matrix invert functionality -Xuan Luo 18 Jul 2006
// void SprMatrix::invert(int &) {}

double SprMatrix::determinant() const {
  static int max_array = 20;
  static int *ir = new int [max_array+1];
  if(ncol != nrow)
    error("SprMatrix::determinant: Matrix is not NxN");
  if (ncol > max_array) {
    delete [] ir;
    max_array = nrow;
    ir = new int [max_array+1];
  }
  double det;
  SprMatrix mt(*this);
  int i = mt.dfact_matrix(det, ir);
  if(i==0) return det;
  return 0;
}

double SprMatrix::trace() const {
   double t = 0.0;
   for (mcIter d = m.begin(); d < m.end(); d += (ncol+1) )
      t += *d;
   return t;
}

