#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"


SiPixelRecHit::SiPixelRecHit( const LocalPoint& pos, const LocalError& err,
			      const DetId& id,
			      const SiPixelCluster * cluster): 
  BaseSiTrackerRecHit2DLocalPos(pos,err,id),
  cluster_(cluster) 
{
}
