#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include <limits>

SiStripRecHit1D::SiStripRecHit1D(const SiStripRecHit2D* hit2D) :
TrackerSingleRecHit(hit2D->localPosition(),
		    LocalError(hit2D->localPositionError().xx(),0.f,std::numeric_limits<float>::max()),
		    *hit2D->det(), hit2D->omniCluster()
		    ){}

