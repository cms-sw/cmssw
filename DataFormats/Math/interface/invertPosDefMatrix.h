#ifndef DataFormat_Math_invertPosDefMatrix_H
#define DataFormat_Math_invertPosDefMatrix_H

#include "Math/SMatrix.h"
#include "Math/CholeskyDecomp.h"
// #include "DataFormats/Math/interface/CholeskyDecomp.h"

template<typename T,unsigned int N>
inline bool invertPosDefMatrix(ROOT::Math::SMatrix<T,N,N,ROOT::Math::MatRepSym<T,N> > & m) {
  
  ROOT::Math::CholeskyDecomp<T,N> decomp(m);
  if (!decomp) {
    return m.Invert();
  } else 
    decomp.Invert(m);
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

// here for a test
#if defined(__SSE3__)
#include <pmmintrin.h>

namespace mathSSE {
  struct M2 {
    // the matrix is shuffed
    struct M {double m00,m01,m11,m10;};
    union {
      double m[4];
      __m128d r[2];
      M mm;
    };
    

    // load shuffled
    inline M2(double i00, double i01, double i10, double i11) {
      mm.m00=i00; mm.m01=i01; mm.m11=i11; mm.m10=i10; }

    double & operator[](int i) { return m[i];}
    __m128d & r0() { return r[0]; }
    __m128d & r1() { return r[1]; }
    
    double  operator[](int i) const { return m[i];}
    __m128d const & r0() const { return r[0]; }
    __m128d const & r1() const { return r[1]; }
    
    
    // assume second row is already shuffled
    inline bool invert() {
      __m128d tmp = r1();
      // mult and sub
      __m128d det  = _mm_mul_pd(r0(),tmp);
      __m128d det2 = _mm_shuffle_pd(det,det,1);
      // det  and -det 
      det = _mm_sub_pd(det,det2);
      // m0 /det, m1/-det -> m3, m2
      r1() = _mm_div_pd(r0(),det);
      // m3/det, m2/-det -> m0 m1
      r0() = _mm_div_pd(tmp,det);
      double d; _mm_store_sd(&d,det);
      return !(0.==d);
    } 
  
  }  __attribute__ ((aligned (16))) ;
}

template<>
inline bool invertPosDefMatrix<double,2>(ROOT::Math::SMatrix<double,2,2,ROOT::Math::MatRepSym<double,2> > & m) {
  mathSSE::M2 mm(m.Array()[0], m.Array()[1], m.Array()[1], m.Array()[2]);

  bool ok = mm.invert();
  if (ok) {
    m.Array()[0] = mm[0];
    m.Array()[1] = mm[1];
    m.Array()[2] = mm[2];
  }
  return ok;
}

template<>
inline bool invertPosDefMatrix<double,2>(ROOT::Math::SMatrix<double,2,2,ROOT::Math::MatRepSym<double,2> > const & mIn,
				  ROOT::Math::SMatrix<double,2,2,ROOT::Math::MatRepSym<double,2> > & mOut) {
 
  mathSSE::M2 mm(mIn.Array()[0], mIn.Array()[1], mIn.Array()[1], mIn.Array()[2]);

  bool ok = mm.invert();
  mOut.Array()[0] = mm[0];
  mOut.Array()[1] = mm[1];
  mOut.Array()[2] = mm[2];

  return ok;
}

#endif


#endif
