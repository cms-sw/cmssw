#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"

SiPixelRecHit::SiPixelRecHit( const LocalPoint& pos, const LocalError& err,
			      const DetId& id,
			      edm::Ref< edm::DetSetVector<SiPixelCluster>, SiPixelCluster> const&  cluster): 
  BaseSiTrackerRecHit2DLocalPos(pos,err,id),
  cluster_(cluster) 

{
}

bool SiPixelRecHit::sharesInput( const TrackingRecHit* other, 
				 SharedInputType what) const
{
  if (geographicalId() != other->geographicalId()) return false;

  const SiPixelRecHit* otherCast = dynamic_cast<const SiPixelRecHit*>(other);
  if ( otherCast == 0 )  return false;

  return cluster() == otherCast->cluster();
}
