#include "DataFormats/TrackerRecHit2D/interface/SiStripGaussianSmearingRecHit2D.h"


SiStripGaussianSmearingRecHit2D::SiStripGaussianSmearingRecHit2D( const LocalPoint& pos, const LocalError& err,
								  const DetId& id,
								  const unsigned int& simtrackId,
								  const PSimHit& simhit):
  BaseSiTrackerRecHit2DLocalPos(pos,err,id) ,
  simtrackId_ (simtrackId) ,
  simhit_(simhit) {}
