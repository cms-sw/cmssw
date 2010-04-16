#ifndef DataFormat_Math_SSERot_H
#define DataFormat_Math_SSERot_H


#include "DataFormats/Math/interface/SSEVec.h"

namespace mathSSE {

  template<typename T>
  struct OldRot { 
    T R11, R12, R13;
    T R21, R22, R23;
    T R31, R32, R33;
  }  __attribute__ ((aligned (16)));
  

  template<typename T>
  struct Rot3 {
    Vec3<T>  axis[3];

    Rot3() {
      axis[0].arr[0]=1;
      axis[1].arr[1]=1;
      axis[2].arr[2]=1;
    }
    
    Rot3( T xx, T xy, T xz, T yx, T yy, T yz, T zx, T zy, T zz) {
      axis[0].set(xx,xy,xz);
      axis[1].set(yx,yy,yz);
      axis[2].set(zx,zy,zz);
    }

    Rot3 transpose() const {
      return Rot3( axis[0].arr[0], axis[1].arr[0], axis[2].arr[0],
		   axis[0].arr[1], axis[1].arr[1], axis[2].arr[1],
		   axis[0].arr[2], axis[1].arr[2], axis[2].arr[2]
		   );
    }

    Vec3<T> rotate(Vec3<T> v) const {
      return transpose().rotateBack(v);
    }

    Vec3<T> rotateBack(Vec3<T> v) const {
      return v.get1(0)*axis[0] +  v.get1(1)*axis[1] + v.get1(2)*axis[2];
    }

  };
  
}

#endif //  DataFormat_Math_SSERot_H
