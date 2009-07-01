#ifndef DataFormat_Math_invertPosDefMatrix_H
#define DataFormat_Math_invertPosDefMatrix_H


#include "DataFormats/Math/interface/CholeskyDecomp.h"

template<typename T,unsigned int N>
bool invertPosDefMatrix(ROOT::Math::SMatrix<T,N,N,ROOT::Math::MatRepSym<T,N> > & m) {
  
  ROOT::Math::CholeskyDecomp<T,N> decomp(m);
  if (!decomp) {
    return m.Invert();
  } else 
    decomp.Invert(m);
  return true;

}

template<typename T,unsigned int N>
bool invertPosDefMatrix(ROOT::Math::SMatrix<T,N,N,ROOT::Math::MatRepSym<T,N> > const & mIn,
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
