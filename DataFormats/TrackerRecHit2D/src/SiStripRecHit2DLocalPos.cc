#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DLocalPos.h"


SiStripRecHit2DLocalPos::SiStripRecHit2DLocalPos( const LocalPoint& pos, const LocalError& err,
						  const GeomDet* det, const DetId& id,
						  const std::vector<const SiStripCluster*>& cluster):
  pos_(pos), err_(err), det_(det), id_(id), cluster_(cluster) {}
