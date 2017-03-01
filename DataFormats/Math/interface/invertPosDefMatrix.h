#ifndef DataFormat_Math_invertPosDefMatrix_H
#define DataFormat_Math_invertPosDefMatrix_H

#define SMATRIX_USE_CONSTEXPR
#include "Math/SMatrix.h"
#include "Math/CholeskyDecomp.h"

template<typename T,unsigned int N>
inline bool invertPosDefMatrix(ROOT::Math::SMatrix<T,N,N,ROOT::Math::MatRepSym<T,N> > & m) {
  
  ROOT::Math::CholeskyDecomp<T,N> decomp(m);
  if (!decomp) {
    return m.Invert();
  } else 
    decomp.Invert(m);
  return true;

}

template<>
inline bool invertPosDefMatrix<double,1>(ROOT::Math::SMatrix<double,1,1,ROOT::Math::MatRepSym<double,1> > & m) {
  m(0,0) = 1./m(0,0); 
  return true;
}
template<>
inline bool invertPosDefMatrix<float,1>(ROOT::Math::SMatrix<float,1,1,ROOT::Math::MatRepSym<float,1> > & m) {
  m(0,0) = 1.f/m(0,0);
  return true;
}

template<typename T,unsigned int N>
inline bool invertPosDefMatrix(ROOT::Math::SMatrix<T,N,N,ROOT::Math::MatRepSym<T,N> > const & mIn,
			ROOT::Math::SMatrix<T,N,N,ROOT::Math::MatRepSym<T,N> > & mOut) {
  
  ROOT::Math::CholeskyDecomp<T,N> decomp(mIn);
  if (!decomp) {
    mOut=mIn;
    return mOut.Invert();
  } else 
    decomp.Invert(mOut);
  return true;

}

#endif
