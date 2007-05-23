// -*- C++ -*-
// $Id: SprGenMatrix.cc,v 1.2 2006/10/19 21:27:52 narsky Exp $
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
// This is the implementation of the SprGenMatrix class.
//

#include <cstring>
#include <cmath>
#include <cstdlib>

#include "SprGenMatrix.hh"
#include "SprSymMatrix.hh"
#include "SprMatrix.hh"

double norm_infinity(const SprGenMatrix &m) {
  double max=0,sum;
  for(int r=1;r<=m.num_row();r++) {
    sum=0;
    for(int c=1;c<=m.num_col();c++) {
      sum+=fabs(m(r,c));
    }
    if(sum>max) max=sum;
  }
  return max;
}

double norm1(const SprGenMatrix &m) {
  double max=0,sum;
  for(int c=1;c<=m.num_col();c++) {
    sum=0;
    for(int r=1;r<=m.num_row();r++)
      sum+=fabs(m(r,c));
    if(sum>max) max=sum;
  }
  return max;
}

void SprGenMatrix::error(const char *s)
{
  std::cerr << s << std::endl;
  std::cerr << "---Exiting to System." << std::endl;
  abort();
}

bool SprGenMatrix::operator== ( const SprGenMatrix& o) const {
  if(o.num_row()!=num_row() || o.num_col()!=num_col()) return false;
  for (int k1=1; k1<=num_row(); k1++)
    for (int k2=1; k2<=num_col(); k2++)
      if(o(k1,k2) != (*this)(k1,k2)) return false;
  return true;
}

// implementation using pre-allocated data array
// -----------------------------------------------------------------

void SprGenMatrix::delete_m(int size, double* m)
{
   if (m)
   {
     if(size > size_max)
       delete [] m;
   }
}

double* SprGenMatrix::new_m(int size)
{
  /*-ap: data_array is replaced by the std::vector<double>,
   *     so we simply return 0 here
   * 
   *   if (size == 0) return 0;
   *   else {
   *     if ( size <= size_max ) {
   *       memset(data_array, 0, size * sizeof(double));
   *       return data_array;
   *     } else {
   *       double * nnn = new double[size];
   *       memset(nnn, 0, size * sizeof(double));
   *       return nnn;
   *     }
   *   }
   *-ap end 
   */
  return 0;
}

