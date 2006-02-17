#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
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

const Track::Point & Track::outerPosition() const { 
  return extra_->outerPosition(); 
}

const Track::Vector & Track::outerMomentum() const { 
  return extra_->outerMomentum(); 
}

bool Track::outerOk() const { 
  return extra_->outerOk(); 
}

recHit_iterator Track::recHitsBegin() const { 
  return extra_->recHitsBegin(); 
}

recHit_iterator Track::recHitsEnd() const { 
  return extra_->recHitsEnd(); 
}

size_t Track::recHitsSize() const { 
  return extra_->recHitsSize(); 
}

double Track::outerPx() const { 
  return extra_->outerPx(); 
}

double Track::outerPy() const { 
  return extra_->outerPy(); 
}

double Track::outerPz() const { 
  return extra_->outerPz(); 
}

double Track::outerX() const { 
  return extra_->outerX(); 
}

double Track::outerY() const { 
  return extra_->outerY(); 
}

double Track::outerZ() const { 
  return extra_->outerZ(); 
}

double Track::outerP() const { 
  return extra_->outerP(); 
}

double Track::outerPt() const { 
  return extra_->outerPt(); 
}

double Track::outerPhi() const { 
  return extra_->outerPhi(); 
}

double Track::outerEta() const { 
  return extra_->outerEta(); 
}

double Track::outerTheta() const { 
  return extra_->outerTheta(); 
}

double Track::outerRadius() const { 
  return extra_->outerRadius(); 
}


