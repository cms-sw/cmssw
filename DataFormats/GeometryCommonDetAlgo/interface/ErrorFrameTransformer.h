#ifndef ErrorFrameTransformer_H
#define ErrorFrameTransformer_H

#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/LocalError.h"

struct   ErrorFrameTransformer {
  
  typedef Surface::Scalar         Scalar;
  
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

};

#endif
