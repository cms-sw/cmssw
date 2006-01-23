#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DLocalPos.h"


SiStripRecHit2DLocalPos::SiStripRecHit2DLocalPos( const LocalPoint& pos, const LocalError& err,
						  const DetId& id,
						  const std::vector<const SiStripCluster*>& cluster): BaseSiStripRecHit2DLocalPos(pos,err,id),
  cluster_(cluster) {}
