#ifndef DataFormat_Math_invertPosDefMatrix_H
#define DataFormat_Math_invertPosDefMatrix_H

#include "Math/SMatrix.h"
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

// here for a test
#if defined(__SSE3__)
#include <pmmintrin.h>

namespace MathSSE {
  struct M2 {
    union {
      __m128d r[2];
      double m[4];
    };
    
    double & operator[](int i) { return m[i];}
    __m128d & r0() { return r[0]; }
    __m128d & r1() { return r[1]; }
    
    double  operator[](int i) const { return m[i];}
    __m128d const & r0() const { return r[0]; }
    __m128d const & r1() const { return r[1]; }
    
    
    // assume second row is already shuffle
    void invert() {
      //  load 2-3 as 3-2
      // __m128d tmp = _mm_shuffle_pd(r1(),r1(),1);
      __m128d tmp = r1();
      // mult and sub
      __m128d det  = _mm_mul_pd(r0(),tmp);
      __m128d det2 = _mm_shuffle_pd(det,det,1);
      // det  and -det 
      det = _mm_sub_pd(det,det2);
      // m0 /det, m1/-det -> m3, m2
      r1() = _mm_div_pd(r0(),det);
      //back in order
      // r1() = _mm_shuffle_pd(r1(),r1(),1);
      // m3/det, m2/-det -> m0 m1
      r0() = _mm_div_pd(tmp,det);
    } 
  
  }  __attribute__ ((aligned (16))) ;
}

template<>
bool invertPosDefMatrix<double,2>(ROOT::Math::SMatrix<double,2,2,ROOT::Math::MatRepSym<double,2> > & m) {
  // load shuffled
  MathSSE::M2 mm = { m.Array()[0], m.Array()[1], m.Array()[2], m.Array()[1]  };

  mm.invert();
  m.Array()[0] = mm[0];
  m.Array()[1] = mm[1];
  m.Array()[2] = mm[2];

  return true;
}

template<>
bool invertPosDefMatrix<double,2>(ROOT::Math::SMatrix<double,2,2,ROOT::Math::MatRepSym<double,2> > const & mIn,
				  ROOT::Math::SMatrix<double,2,2,ROOT::Math::MatRepSym<double,2> > & mOut) {
 
  MathSSE::M2 mm = { mIn.Array()[0], mIn.Array()[1], mIn.Array()[2], mIn.Array()[1]  };

  mm.invert();
  mOut.Array()[0] = mm[0];
  mOut.Array()[1] = mm[1];
  mOut.Array()[2] = mm[2];

  return true;
}

#endif


#endif
