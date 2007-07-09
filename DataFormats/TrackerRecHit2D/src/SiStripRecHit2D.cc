#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"


SiStripRecHit2D::SiStripRecHit2D( const LocalPoint& pos, const LocalError& err,
				  const DetId& id,
				  edm::Ref<edm::DetSetVector<SiStripCluster>,SiStripCluster, edm::refhelper::FindForDetSetVector<SiStripCluster>  > const & cluster):
  
  BaseSiTrackerRecHit2DLocalPos(pos,err,id), 
  cluster_(cluster),
  clusterRegional_(),
  sigmaPitch_(-1.)
 {}


SiStripRecHit2D::SiStripRecHit2D( const LocalPoint& pos, const LocalError& err,
				  const DetId& id,
				  edm::SiStripRefGetter<SiStripCluster>::value_ref const& cluster): 
  BaseSiTrackerRecHit2DLocalPos(pos,err,id), 
  cluster_(),
  clusterRegional_(cluster),
  sigmaPitch_(-1.) {}


bool 
SiStripRecHit2D::sharesInput( const TrackingRecHit* other, 
			      SharedInputType what) const
{
  if (trackerId() != other->geographicalId()) return false;
  if(! other->isValid()) return false;

  const SiStripRecHit2D* otherCast = static_cast<const SiStripRecHit2D*>(other);

  return cluster_ == otherCast->cluster();
}

