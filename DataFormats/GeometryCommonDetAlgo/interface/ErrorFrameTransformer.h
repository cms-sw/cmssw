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

  //new Jacobian for 6x6 APE in muon code
  static LocalErrorExtended transform46(const GlobalErrorExtended& ge, const AlgebraicVector& positions, const AlgebraicVector& directions) {

    AlgebraicSymMatrix66 as(ge.matrix());

    AlgebraicMatrix46 jacobian46;
    jacobian46[0][0] = 1.;
    jacobian46[0][1] = 0.;
    jacobian46[0][2] = -directions[0];
    jacobian46[0][3] = -positions[1]*directions[0];
    jacobian46[0][4] = positions[0]*directions[0];
    jacobian46[0][5] = -positions[1];

    jacobian46[1][0] = 0.;
    jacobian46[1][1] = 1.;
    jacobian46[1][2] = -directions[1];
    jacobian46[1][3] = -positions[1]*directions[1];
    jacobian46[1][4] = positions[0]*directions[1];
    jacobian46[1][5] = positions[0];

    jacobian46[2][0] = 0.;
    jacobian46[2][1] = 0.;
    jacobian46[2][2] = 0.;
    jacobian46[2][3] = -directions[1]*directions[0];
    jacobian46[2][4] = 1.+directions[0]*directions[0];
    jacobian46[2][5] = -directions[1];

    jacobian46[3][0] = 0.;
    jacobian46[3][1] = 0.;
    jacobian46[3][2] = 0.;
    jacobian46[3][3] = -1.-directions[1]*directions[1];
    jacobian46[3][4] = directions[0]*directions[1];
    jacobian46[3][5] = directions[0];

    AlgebraicSymMatrix44 out = ROOT::Math::Similarity(jacobian46,as); 

    LocalErrorExtended newError(out);

    return newError;
  }


};

#endif
