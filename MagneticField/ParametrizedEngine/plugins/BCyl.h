#ifndef ParametrizedEngine_BCyl_h
#define ParametrizedEngine_BCyl_h
/** 
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

#include<cmath>

#include "DataFormats/Math/interface/approx_exp.h"

namespace magfieldparam {
  template<typename T>
  struct BCylParam {
    //  constexpr BCylParam(std::initializer_list<T> init) :
    template<typename ... Args>
    constexpr BCylParam(Args... init) :
      prm{std::forward<Args>(init)...},
    //prm(std::forward<Args>(init)...), 
      ap2(4*prm[0]*prm[0]/(prm[1]*prm[1])), 
      hb0(0.5*prm[2]*std::sqrt(1.0+ap2)),
      hlova(1/std::sqrt(ap2)),
      ainv(2*hlova/prm[1]),
      coeff(1/(prm[8]*prm[8])){}
    
    T prm[9];
    T ap2, hb0, hlova, ainv,coeff;
    
    
  };
  
  namespace bcylDetails{
    
    template<typename T>
    inline void ffunkti(T u, T * __restrict__ ff) __attribute__((always_inline));
    
    template<typename T>
    inline void ffunkti(T u, T * __restrict__ ff) {
      // Function and its 3 derivatives
      T a,b,a2,u2;
      u2=u*u; 
      a=T(1)/(T(1)+u2);
      a2=-T(3)*a*a;
      b=std::sqrt(a);
      ff[0]=u*b;
      ff[1]=a*b;
      ff[2]=a2*ff[0];
      ff[3]=a2*ff[1]*(T(1)-4*u2);
    }
    
    inline double myExp(double x) { return std::exp(x);}
    inline float myExp(float x) { return unsafe_expf<3>(x);}
    
  }
  
  
  template<typename T>
  class BCycl  {
  public:
    BCycl(BCylParam<T> const & ipar) : pars(ipar) {} 
    
    void operator()(T r2, T z, T& Br, T& Bz) const {
      compute(r2,z,Br,Bz);
    }
    
    
  // in meters and T  (Br needs to be multiplied by r)
    void compute(T r2, T z, T& Br, T& Bz) const {
      using namespace  bcylDetails;
      //  if (r<1.15&&fabs(z)<2.8) // NOTE: check omitted, is done already by the wrapper! (NA)
      z-=pars.prm[3];                    // max Bz point is shifted in z
      T az=std::abs(z);
      T zainv=z*pars.ainv;
      T u=pars.hlova-zainv;
      T v=pars.hlova+zainv;
      T fu[4],gv[4];
      ffunkti(u,fu);
      ffunkti(v,gv);
      T rat=T(0.5)*pars.ainv;
      T rat2=rat*rat*r2;
      Br=pars.hb0*rat*(fu[1]-gv[1]-(fu[3]-gv[3])*rat2*T(0.5));
      Bz=pars.hb0*(fu[0]+gv[0]-(fu[2]+gv[2])*rat2);
      
      T corBr= pars.prm[4]*z*(az-pars.prm[5])*(az-pars.prm[5]);
      T corBz=-pars.prm[6]*(
			    myExp(-(z-pars.prm[7])*(z-pars.prm[7])*pars.coeff) +
			    myExp(-(z+pars.prm[7])*(z+pars.prm[7])*pars.coeff)
			    ); // double Gaussian
      Br+=corBr;
      Bz+=corBz;
    }
    
  private:
    BCylParam<T> pars;
    
  };
}

#endif
