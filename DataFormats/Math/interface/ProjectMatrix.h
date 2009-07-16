#ifndef DataFormat_Math_ProjectMatrix_H
#define DataFormat_Math_ProjectMatrix_H

#include "Math/SMatrix.h"



template<typename T, unsigned int N, unsigned int D>
struct ProjectMatrix{
  typedef  ROOT::Math::SMatrix<T,D,D,ROOT::Math::MatRepSym<T,D> > SMatDD;
  typedef  ROOT::Math::SMatrix<T,N,N,ROOT::Math::MatRepSym<T,N> > SMatNN;
  typedef  ROOT::Math::SMatrix<T,N,D > SMatND;

  // no constructor
  
  // H*S
  SMatND project(SMatDD const & s) {
    SMatND r;
    for (unsigned int i=0; i<D; i++)
      for (unsigned int j=0; j<D; j++)
	r(index(i),j) = s(i,j);
    return r;
  }

  // K*H
  SMatNN project(SMatND const & k) {
    SMatNN s;
    for (unsigned int i=0; i<N; i++)
      for (unsigned int j=0; j<D; j++)
	s(i,index(j)) = k(i,j);
    return s;
  }

  // only H(i,index(i))=1.
  unsigned int index[D];

}


#endif
