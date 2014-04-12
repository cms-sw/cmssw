#ifndef DataFormat_Math_invertPosDefMatrix_H
#define DataFormat_Math_invertPosDefMatrix_H

#include "Math/SMatrix.h"
#include "Math/CholeskyDecomp.h"
// #include "DataFormats/Math/interface/CholeskyDecomp.h"
#include "SIMDVec.h"

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
#ifdef NEVER
// #if defined(USE_EXTVECT)

namespace mathSSE {
  struct M2 {
    using Vec=Vec4<double>;
    // the matrix is shuffed
    Vec mm[2];

    double m00() const { return mm[0][0];}
    double m01() const { return mm[0][1];}
    double m11() const { return mm[1][0];}
    double m10() const { return mm[1][1];}

    double & m00() { return mm[0][0];}
    double & m01() { return mm[0][1];}
    double & m11() { return mm[1][0];}
    double & m10() { return mm[1][1];}
    

    // load shuffled
    inline M2(double i00, double i01, double i10, double i11) {
      m00()=i00; m01()=i01; m11()=i11; m10()=i10; }

    Vec & r0() { return mm[0]; }
    Vec & r1() { return mm[1]; }
    
    Vec const & r0() const { return mm[0]; }
    Vec const & r1() const { return mm[1]; }
    
    
    // assume second row is already shuffled
    inline bool invert() {
      Vec tmp = r1();
      // mult and sub
      Vec det  = r0()*tmp;
      Vec det2{det[1],det[0]};
      // det  and -det 
      det -= det2;
      // m0 /det, m1/-det -> m3, m2
      r1() = r0()/det;
      // m3/det, m2/-det -> m0 m1
      r0() = tmp/det;
      return (0.!=det[0]);
    } 
  
  };
}

template<>
inline bool invertPosDefMatrix<double,2>(ROOT::Math::SMatrix<double,2,2,ROOT::Math::MatRepSym<double,2> > & m) {
  mathSSE::M2 mm(m.Array()[0], m.Array()[1], m.Array()[1], m.Array()[2]);

  bool ok = mm.invert();
  if (ok) {
    m.Array()[0] = mm.m00();
    m.Array()[1] = mm.m01();
    m.Array()[2] = mm.m11();
  }
  return ok;
}

template<>
inline bool invertPosDefMatrix<double,2>(ROOT::Math::SMatrix<double,2,2,ROOT::Math::MatRepSym<double,2> > const & mIn,
				  ROOT::Math::SMatrix<double,2,2,ROOT::Math::MatRepSym<double,2> > & mOut) {
 
  mathSSE::M2 mm(mIn.Array()[0], mIn.Array()[1], mIn.Array()[1], mIn.Array()[2]);

  bool ok = mm.invert();
  mOut.Array()[0] = mm.m00();
  mOut.Array()[1] = mm.m01();
  mOut.Array()[2] = mm.m11();

  return ok;
}




#endif


#endif
