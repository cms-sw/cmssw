#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"


SiStripRecHit2D::SiStripRecHit2D( const LocalPoint& pos, const LocalError& err,
				  const DetId& id,
				  ClusterRef const & cluster):
  
  BaseSiTrackerRecHit2DLocalPos(pos,err,id), 
  cluster_(cluster),
  clusterRegional_(),
  sigmaPitch_(-1.)
 {}


SiStripRecHit2D::SiStripRecHit2D( const LocalPoint& pos, const LocalError& err,
				  const DetId& id,
				  ClusterRegionalRef const& cluster): 
  BaseSiTrackerRecHit2DLocalPos(pos,err,id), 
  cluster_(),
  clusterRegional_(cluster),
  sigmaPitch_(-1.) {}


bool 
SiStripRecHit2D::sharesInput( const TrackingRecHit* other, 
			      SharedInputType what) const
{
  if (geographicalId() != other->geographicalId()) return false;

  const SiStripRecHit2D* otherCast = static_cast<const SiStripRecHit2D*>(other);
  // as 'null == null' is true, we can't just "or" the two equality tests: one of the two refs is always null! (gpetrucc)
  if (cluster_.isNonnull()) {
    return (cluster_ == otherCast->cluster());
  } else {
    return (clusterRegional_ == otherCast->cluster_regional());
  }
}

