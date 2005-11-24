#include "DataFormats/TrackReco/interface/Track.h"
using namespace reco;

Track::Track( float chi2, unsigned short ndof,  
	      int found, int lost, int invalid, 
	      const helix::Parameters & par, const helix::Covariance & cov  ) : 
  chi2_( chi2 ), ndof_( ndof ), 
  found_( found ), lost_( lost ), invalid_( invalid ),
  par_( par ), cov_( cov ) {
}

Track::Track( float chi2, unsigned short ndof, int found, int invalid, int lost,
	      int q, const Point & v, const Vector & p, 
	      const PosMomError & err ) :
  chi2_( chi2 ), ndof_( ndof ), 
  found_( found ), lost_( lost ), invalid_( invalid ),
  par_( ), cov_( ) {
  helix::setFromCartesian( q, v, p, err, par_, cov_ );
}



