#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"


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
  //here we exclude non si-strip subdetectors
  if( ((geographicalId().rawId()) >> (DetId::kSubdetOffset) ) != ( (other->geographicalId().rawId())>> (DetId::kSubdetOffset)) ) return false;

  //Protection against invalid hits
  if(! other->isValid()) return false;

  const std::type_info & otherType = typeid(*other);
  if (otherType == typeid(SiStripRecHit2D)) {
    const SiStripRecHit2D* otherCast = static_cast<const SiStripRecHit2D*>(other);
    // as 'null == null' is true, we can't just "or" the two equality tests: one of the two refs is always null! (gpetrucc)
    if (cluster_.isNonnull()) {
      return (cluster_ == otherCast->cluster());
    } else {
      return (clusterRegional_ == otherCast->cluster_regional());
    }
  } else if (otherType == typeid(SiStripRecHit1D)) {
    const SiStripRecHit1D* otherCast = static_cast<const SiStripRecHit1D*>(other);
    // as 'null == null' is true, we can't just "or" the two equality tests: one of the two refs is always null! (gpetrucc)
    if (cluster_.isNonnull()) {
      return (cluster_ == otherCast->cluster());
    } else {
      return (clusterRegional_ == otherCast->cluster_regional());
    }
  } else if (otherType == typeid(ProjectedSiStripRecHit2D)) {
    const SiStripRecHit2D* otherCast = & (static_cast<const ProjectedSiStripRecHit2D*>(other)->originalHit());
    // as 'null == null' is true, we can't just "or" the two equality tests: one of the two refs is always null! (gpetrucc)
    if (cluster_.isNonnull()) {
      return (cluster_ == otherCast->cluster());
    } else {
      return (clusterRegional_ == otherCast->cluster_regional());
    }
  } else if ((otherType == typeid(SiStripMatchedRecHit2D)) && (what == all)) {
    return false; 
  } else {
    // last resort, recur to 'recHits()', even if it returns a vector by value
    std::vector<const TrackingRecHit*> otherHits = other->recHits();
    int ncomponents=otherHits.size();
    if(ncomponents==0)return false;
    else if(ncomponents==1)return sharesInput(otherHits.front(),what);
    else if (ncomponents>1){
      if(what == all )return false;
      else{
	for(int i=0;i<ncomponents;i++){
	  if(sharesInput(otherHits[i],what))return true;
	}
	return false;
      }
    }
    return false;
  }
}

