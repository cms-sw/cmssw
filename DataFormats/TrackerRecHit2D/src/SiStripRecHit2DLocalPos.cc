#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DLocalPos.h"


SiStripRecHit2DLocalPos::SiStripRecHit2DLocalPos( const LocalPoint& pos, const LocalError& err,
						  const DetId& id,
						  edm::Ref<edm::DetSetVector<SiStripCluster>,SiStripCluster, edm::refhelper::FindForDetSetVector<SiStripCluster>  > const & cluster): BaseSiTrackerRecHit2DLocalPos(pos,err,id),
  cluster_(cluster) {}
