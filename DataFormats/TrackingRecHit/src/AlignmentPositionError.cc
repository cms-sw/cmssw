#include "DataFormats/TrackingRecHit/interface/AlignmentPositionError.h"

AlignmentPositionError::AlignmentPositionError( float dx, float dy, float dz){
  theGlobalError = GlobalError(double(dx*dx),           
		               0., double(dy*dy),       
		               0., 0., double(dz*dz) ) ;
}
