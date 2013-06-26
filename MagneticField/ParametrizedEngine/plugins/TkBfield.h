#ifndef ParametrizedEngine_TkBfield_h
#define ParametrizedEngine_TkBfield_h

/** \class magfieldparam::TkBfield
 *
 *
 *    B-field in Tracker volume - based on the TOSCA computation version 1103l
 *    (tuned on MTCC measured field (fall 2006))
 *    
 *     In:   x[3]: coordinates (m)
 *    Out:   B[3]: Bx,By,Bz    (T)    (getBxyz)
 *    Out:   B[3]: Br,Bf,Bz    (T)    (getBrfz)
 *
 *    Valid for r<1.15m and |z|<2.80m
 *
 *  \author V.Karimaki 080228, 080407
 *  new float version V.I. October 2012
 */


#include "BCyl.h"
#include <string>


namespace magfieldparam {
  class TkBfield {
  public:

    TkBfield (std::string T="3_8T");  


    /// B out in cartesian
    void getBxyz(float const  * __restrict__ x, float * __restrict__ Bxyz) const; 
    /// B out in cylindrical
    void getBrfz(float const  * __restrict__ x, float * __restrict__ Brfz) const;

  private:

    BCycl<float> bcyl;

  };
}

#endif
