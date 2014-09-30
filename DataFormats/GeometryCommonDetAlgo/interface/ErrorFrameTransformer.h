#ifndef ErrorFrameTransformer_H
#define ErrorFrameTransformer_H

#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/LocalError.h"
#include "DataFormats/Math/interface/AlgebraicROOTObjects.h"


#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"


struct   ErrorFrameTransformer {
  
  typedef Surface::Scalar         Scalar;
 
  //old version 1 
  static GlobalError transform(const LocalError& le, const Surface& surf) {
    // the GlobalError is a sym matrix, initialisation takes only
    // 6 T because GlobalError is stored as a lower triangular matrix.
    Scalar cxx = le.xx();
    Scalar cxy = le.xy();
    Scalar cyy = le.yy();
    
    Surface::RotationType r=surf.rotation();
   
    return GlobalError( r.xx()*(r.xx()*cxx+r.yx()*cxy) + r.yx()*(r.xx()*cxy+r.yx()*cyy) ,
                        r.xx()*(r.xy()*cxx+r.yy()*cxy) + r.yx()*(r.xy()*cxy+r.yy()*cyy) ,
                        r.xy()*(r.xy()*cxx+r.yy()*cxy) + r.yy()*(r.xy()*cxy+r.yy()*cyy) ,
                        r.xx()*(r.xz()*cxx+r.yz()*cxy) + r.yx()*(r.xz()*cxy+r.yz()*cyy) ,
                        r.xy()*(r.xz()*cxx+r.yz()*cxy) + r.yy()*(r.xz()*cxy+r.yz()*cyy) ,
                        r.xz()*(r.xz()*cxx+r.yz()*cxy) + r.yz()*(r.xz()*cxy+r.yz()*cyy) );
  }
 
  //old version 2 
  static LocalError transform(const LocalErrorExtended& ge, const Surface& surf) {
    LocalError newError(ge.matrix()[0][0], ge.matrix()[0][1], ge.matrix()[1][1]);
    return newError;
  }

  //for validation 1
  static GlobalError transform33(const GlobalError& ge, const Surface& surf, bool fromGlobalToLocal) {

    AlgebraicSymMatrix as(3,0);
    as[0][0] = ge.cxx();
    as[1][0] = ge.cyx(); as[1][1] = ge.cyy();
    as[2][0] = ge.czx(); as[2][1] = ge.czy(); as[2][2] = ge.czz();

    Surface::RotationType r=surf.rotation();

    AlgebraicMatrix am(3,3);
    am[0][0] = r.xx(); am[0][1] = r.xy(); am[0][2] = r.xz();
    am[1][0] = r.yx(); am[1][1] = r.yy(); am[1][2] = r.yz();
    am[2][0] = r.zx(); am[2][1] = r.zy(); am[2][2] = r.zz();

    if (fromGlobalToLocal) { 
       as = as.similarityT(am);
    } else {
       as = as.similarity(am);
    }
    GlobalError newError(as[0][0], as[1][0], as[1][1], as[2][0], as[2][1], as[2][2]);

    return newError;
  }

  static LocalError transform(const GlobalError& ge, const Surface& surf) {
    Scalar cxx = ge.cxx(); Scalar cyx = ge.cyx(); Scalar cyy = ge.cyy();
    Scalar czx = ge.czx(); Scalar czy = ge.czy(); Scalar czz = ge.czz();
    
    Surface::RotationType r=surf.rotation();
    
    Scalar l11 
      = r.xx()*(r.xx()*cxx + r.xy()*cyx + r.xz()*czx)
      + r.xy()*(r.xx()*cyx + r.xy()*cyy + r.xz()*czy)
      + r.xz()*(r.xx()*czx + r.xy()*czy + r.xz()*czz);
    Scalar l12 
      = r.yx()*(r.xx()*cxx + r.xy()*cyx + r.xz()*czx)
      + r.yy()*(r.xx()*cyx + r.xy()*cyy + r.xz()*czy)
      + r.yz()*(r.xx()*czx + r.xy()*czy + r.xz()*czz);
    Scalar l22
      = r.yx()*(r.yx()*cxx + r.yy()*cyx + r.yz()*czx)
      + r.yy()*(r.yx()*cyx + r.yy()*cyy + r.yz()*czy)
      + r.yz()*(r.yx()*czx + r.yy()*czy + r.yz()*czz);
    
    return LocalError( l11, l12, l22);
  }

  //Jacobian used in the GeometryAligner
  static LocalErrorExtended transform46(const GlobalErrorExtended& ge, const AlgebraicVector& shifts, const AlgebraicVector& angles) {

    AlgebraicSymMatrix66 as(ge.matrix());

    AlgebraicMatrix46 jacobian46;
    jacobian46[0][0] = 1.;
    jacobian46[0][1] = 0.;
    jacobian46[0][2] = -angles(0);
    jacobian46[0][3] = -shifts(1)*angles(0);
    jacobian46[0][4] = shifts(0)*angles(0);
    jacobian46[0][5] = -shifts(1);

    jacobian46[1][0] = 0.;
    jacobian46[1][1] = 1.;
    jacobian46[1][2] = -angles(1);
    jacobian46[1][3] = -shifts(1)*angles(1);
    jacobian46[1][4] = shifts(0)*angles(1);
    jacobian46[1][5] = shifts(0);

    jacobian46[2][0] = 0.;
    jacobian46[2][1] = 0.;
    jacobian46[2][2] = 0.;
    jacobian46[2][3] = -angles(1)*angles(0);
    jacobian46[2][4] = 1.+angles(0)*angles(0);
    jacobian46[2][5] = -angles(1);

    jacobian46[3][0] = 0.;
    jacobian46[3][1] = 0.;
    jacobian46[3][2] = 0.;
    jacobian46[3][3] = -1.-angles(1)*angles(1);
    jacobian46[3][4] = angles(0)*angles(1);
    jacobian46[3][5] = angles(0);

    //protect from umerical instability - multiplying small numbers...
    jacobian46 *= 100000000.;
    AlgebraicSymMatrix44 out = ROOT::Math::Similarity(jacobian46,as); 
    out /= 100000000.*100000000.;

    LocalErrorExtended newError(out);

    return newError;
  }


};

#endif
