#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"


SiPixelRecHit::SiPixelRecHit( const LocalPoint& pos, const LocalError& err,
						  const DetId& id,
						  const std::vector<const SiStripCluster*>& cluster): 
  BaseSiStripRecHit2DLocalPos(pos,err,id),
  cluster_(cluster) 
{
}
