#include "DataFormats/TrackingRecHit/interface/AlignmentPositionError.h"

AlignmentPositionError::AlignmentPositionError( float xx, float yy, float zz, float phixphix, float phiyphiy, float phizphiz){
  theGlobalError = GlobalErrorExtended(xx,0.,0.,0.,0.,0.,yy,0.,0.,0.,0.,zz,0.,0.,0.,phixphix,0.,0.,phiyphiy,0.,phizphiz);
}


AlignmentPositionError::AlignmentPositionError(const GlobalError& ge) {
  theGlobalError = GlobalErrorExtended(ge.cxx(),ge.cyx(),ge.czx(),0.,0.,0.,ge.cyy(),ge.czy(),0.,0.,0.,ge.czz(),0.,0.,0.,0.,0.,0.,0.,0.,0.);
}
