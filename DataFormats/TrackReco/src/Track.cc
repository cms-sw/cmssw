#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
using namespace reco;

Track::Track( float chi2, unsigned short ndof,  
	      int found, int lost, int invalid, 
	      const helix::Parameters & par, const helix::Covariance & cov  ) : 
  TrackBase( chi2, ndof, found, lost, invalid, par, cov ) {
}

Track::Track( float chi2, unsigned short ndof, int found, int invalid, int lost,
	      int q, const Point & v, const Vector & p, 
	      const PosMomError & err ) :
  TrackBase( chi2, ndof, found, lost, invalid, q, v, p, err ) {
}
