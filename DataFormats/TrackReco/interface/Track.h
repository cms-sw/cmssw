#ifndef TrackReco_Track_h
#define TrackReco_Track_h
/** \class reco::Track
 *
 * Reconstructed Track. It is ment to be stored
 * in the AOD, with a reference to an extension
 * object stored in the RECO
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: Track.h,v 1.12 2006/03/01 12:23:40 llista Exp $
 *
 */
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtension.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

namespace reco {

  class Track : public TrackBase, public TrackExtension<TrackExtraRef> {
  public:
    /// default constructor
    Track() { }
    /// constructor from fit parameters and error matrix
    Track( float chi2, unsigned short ndof, int found, int invalid, int lost,
	   const Parameters &, const Covariance & );
    /// constructor from cartesian coordinates and covariance matrix.
    /// notice that the vertex passed must be 
    /// the point of closest approch to the beamline.    
    Track( float chi2, unsigned short ndof, int found, int invalid, int lost,
	   int q, const Point & v, const Vector & p, 
	   const PosMomError & err );
  };

}

#endif
