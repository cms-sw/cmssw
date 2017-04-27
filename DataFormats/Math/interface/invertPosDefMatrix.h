#ifndef DataFormat_Math_invertPosDefMatrix_H
#define DataFormat_Math_invertPosDefMatrix_H

#define SMATRIX_USE_CONSTEXPR
#include "Math/SMatrix.h"
#include "Math/CholeskyDecomp.h"
#include<type_traits>

template<typename T,unsigned int N>
inline bool invertPosDefMatrix(ROOT::Math::SMatrix<T,N,N,ROOT::Math::MatRepSym<T,N> > & m) {
  
  ROOT::Math::CholeskyDecomp<T,N> decomp(m);
  if (!decomp) {
    return m.Invert();
  } else 
    decomp.Invert(m);
  return true;

}

template<typename PDM2>
void fastInvertPDM2(PDM2&mm) {
  auto m = mm.Array();

  constexpr typename std::remove_reference<decltype(m[0])>::type one = 1.;
  auto c0 = one/m[0];
  auto c1 = m[1]*m[1]* c0;
  auto c2 = one/(m[2] - c1);

  auto li21 = c1 * c0 * c2;
  m[0] = li21 + c0;
  m[1] = - m[1]*c0*c2;
  m[2] = c2;
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


template<>
inline bool invertPosDefMatrix<double,2>(ROOT::Math::SMatrix<double,2,2,ROOT::Math::MatRepSym<double,2> > & m) {
  fastInvertPDM2(m); 
  return true;
}
template<>
inline bool invertPosDefMatrix<float,2>(ROOT::Math::SMatrix<float,2,2,ROOT::Math::MatRepSym<float,2> > & m) {
  fastInvertPDM2(m); 
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
