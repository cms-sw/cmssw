#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"

SiStripRecHit1D::SiStripRecHit1D(const SiStripRecHit2D* hit2D) :
TrackerSingleRecHit(hit2D->localPosition(),
		    LocalError(hit2D->localPositionError().xx(),0.,DBL_MAX),
		    hit2D->geographicalId(), hit2D->omniCluster()
		    ),
  sigmaPitch_(-1){}
}
