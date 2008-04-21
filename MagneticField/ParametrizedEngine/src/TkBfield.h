#ifndef ParametrizedEngine_TkBfield_h
#define ParametrizedEngine_TkBfield_h

/** \class TkBfield
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
 *  $Date: 2008/04/01 08:32:41 $
 *  $Revision: 1.2 $
 *  \author V.Karimaki 080228, 080407
 */


#include <vector>

namespace magfieldparam {
  class TkBfield 
  {
  public:

    TkBfield (std::string T="3_8T");  

    ~TkBfield() {}

    /// B out in cartesian
    void getBxyz(const double* x, double* B); 
    /// B out in cylindrical
    void getBrfz(const double* x, double* B);    

    /// Define validity region (r, z in m)
    bool isDefined(double r, double z);

  private:
    double prm[9];
    double Bw[3];
    double ap2, hb0, hlova, ainv;
    void Bcyl(const double *x);
    void ffunkti(const double u, double* ff);    
  };
}

#endif
