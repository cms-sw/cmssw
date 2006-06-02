#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"

SiPixelRecHit::SiPixelRecHit( const LocalPoint& pos, const LocalError& err,
			      const DetId& id,
			      edm::Ref< edm::DetSetVector<SiPixelCluster>, SiPixelCluster> const&  cluster): 
  BaseSiTrackerRecHit2DLocalPos(pos,err,id),
  cluster_(cluster) 

{
}
