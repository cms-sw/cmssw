#include "DataFormats/TrackingRecHit/interface/AlignmentPositionError.h"

AlignmentPositionError::AlignmentPositionError( float xx, float yy, float phixphix, float phiyphiy){
  theLocalError = LocalErrorExtended(xx,0.,0.,0.,yy,0.,0.,phixphix,0.,phiyphiy);
}
