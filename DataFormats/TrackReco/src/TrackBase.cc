#include "DataFormats/TrackReco/interface/TrackBase.h"
using namespace reco;

TrackBase::TrackBase( double chi2, double ndof,  
		      int found, int lost, int invalid, 
		      const helix::Parameters & par, const helix::Covariance & cov  ) : 
  chi2_( chi2 ), ndof_( ndof ), 
  found_( found ), lost_( lost ), invalid_( invalid ),
  par_( par ), cov_( cov ) {
}

TrackBase::TrackBase( double chi2, double ndof, int found, int invalid, int lost,
	      int q, const Point & v, const Vector & p, 
	      const PosMomError & err ) :
  chi2_( chi2 ), ndof_( ndof ), 
  found_( found ), lost_( lost ), invalid_( invalid ),
  par_( ), cov_( ) {
  helix::setFromCartesian( q, v, p, err, par_, cov_ );
}


