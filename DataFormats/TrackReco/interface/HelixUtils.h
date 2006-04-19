#ifndef TrackReco_HelixUtils_h
#define TrackReco_HelixUtils_h
/* 
 * helper functions to transform helix parameters to and
 * from cartesian coordinates
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: HelixParameters.h,v 1.10 2006/04/03 11:59:29 llista Exp $
 *
 */
#include "DataFormats/TrackReco/interface/HelixParameters.h"
#include "DataFormats/TrackReco/interface/HelixCovariance.h"

namespace reco {
  namespace helix {
    /// convert from cartesian coordinates to 5-helix parameters.
    /// The point passed must be the point of closest approach to the beamline
    void setFromCartesian( int q, const Point &, const Vector &, 
			   const PosMomError & ,
			   Parameters &, Covariance & ); 
    
    /// compute position-momentum 6x6 degenerate covariance matrix
    /// from 5 parameters and 5x5 covariance matrix
    PosMomError posMomError( const Parameters &, const Covariance & );
  }
}

#endif
