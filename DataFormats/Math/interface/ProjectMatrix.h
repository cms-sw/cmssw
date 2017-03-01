#ifndef DataFormat_Math_ProjectMatrix_H
#define DataFormat_Math_ProjectMatrix_H

#define SMATRIX_USE_CONSTEXPR
#include "Math/SMatrix.h"



template<typename T, unsigned int N, unsigned int D>
struct ProjectMatrix{
  typedef  ROOT::Math::SMatrix<T,D,D,ROOT::Math::MatRepSym<T,D> > SMatDD;
  typedef  ROOT::Math::SMatrix<T,N,N > SMatNN;
  typedef  ROOT::Math::SMatrix<T,N,D > SMatND;
  typedef  ROOT::Math::SMatrix<T,D,N > SMatDN;
  

  // no constructor
  
  // H
  SMatDN matrix() const {
    SMatDN r;
    for (unsigned int i=0; i<D; i++)
        r(i,index[i]) = T(1.);
    return r;
  }

  // constuct given H
  void fromH(SMatDN const & H) {
    for (unsigned int i=0; i<D; i++) {
      index[i]=N;
      for (unsigned int j=0; j<N; j++)
        if ( H(i,j)>0 ) { index[i]=j; break;}
   }
  }

  // H*S
  SMatND project(SMatDD const & s) const {
    SMatND r;
    for (unsigned int i=0; i<D; i++)
      for (unsigned int j=0; j<D; j++)
	r(index[i],j) = s(i,j);
    return r;
  }
  
  // K*H
  SMatNN project(SMatND const & k) const {
    SMatNN s;
    for (unsigned int i=0; i<N; i++)
      for (unsigned int j=0; j<D; j++)
	s(i,index[j]) = k(i,j);
    return s;
  }

  // S-K*H
  void projectAndSubtractFrom(SMatNN & __restrict__ s, SMatND const & __restrict__ k) const {
    for (unsigned int i=0; i<N; i++)
      for (unsigned int j=0; j<D; j++)
	s(i,index[j]) -= k(i,j);
  }

  // only H(i,index[i])=1.
  unsigned int index[D];

};


#endif
