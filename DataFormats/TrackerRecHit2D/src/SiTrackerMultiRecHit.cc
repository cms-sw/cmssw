#include "DataFormats/TrackerRecHit2D/interface/SiTrackerMultiRecHit.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;
using namespace edm;

SiTrackerMultiRecHit::SiTrackerMultiRecHit(const LocalPoint& pos, const LocalError& err, GeomDet const & idet,
					   const std::vector< std::pair<const TrackingRecHit*, float> >& aHitMap):
  BaseTrackerRecHit(pos,err, idet,trackerHitRTTI::multi)	
{
  for(std::vector<std::pair<const TrackingRecHit*, float> >::const_iterator ihit = aHitMap.begin(); ihit != aHitMap.end(); ihit++){
    theHits.push_back(ihit->first->clone());
    theWeights.push_back(ihit->second);
  }
}


bool SiTrackerMultiRecHit::sharesInput(const TrackingRecHit* other,
				       SharedInputType what) const
{
  if(geographicalId() != other->geographicalId()&& what==all ) return false;
  vector<const TrackingRecHit*> otherhits=other->recHits();
  if(what==all){
    if(theHits.size()!=other->recHits().size())return false;
    for(vector<const TrackingRecHit*>::iterator otherhit=otherhits.begin();otherhit!=otherhits.end();++otherhit){
      bool found=false;
      for(OwnVector<TrackingRecHit>::const_iterator hit=theHits.begin();hit!=theHits.end();++hit){
	if((hit)->sharesInput(*otherhit,all)){
	  found=true;
	  break;
	}
      }
      if(found==false){
	return false;
      }
    }
    return true;
  }
  else{
    for(OwnVector<TrackingRecHit>::const_iterator hit=theHits.begin();hit!=theHits.end();++hit){
      if(otherhits.size()!=0){ 
	for(vector<const TrackingRecHit*>::iterator otherhit=otherhits.begin();otherhit!=otherhits.end();++otherhit){
	  if((hit)->sharesInput(*otherhit,some))return true;
	}
      }
      else{//otherwise it should be a single rechit
	if((hit)->sharesInput(other,some))return true;
      } 
    }
    return false;
  }
}


vector<const TrackingRecHit*> SiTrackerMultiRecHit::recHits() const{
  vector<const TrackingRecHit*> myhits;
  for(edm::OwnVector<TrackingRecHit>::const_iterator ihit = theHits.begin(); ihit != theHits.end(); ihit++) {
    myhits.push_back(&*ihit);
  }
  return myhits;
}

vector<TrackingRecHit*> SiTrackerMultiRecHit::recHits() {
  //        vector<TrackingRecHit*> myhits;
  //         for(edm::OwnVector<TrackingRecHit>::const_iterator ihit = theHits.begin(); ihit != theHits.end(); ihit++) {
  //                 const TrackingRecHit* ahit = &(*ihit);
  //                 myhits.push_back(const_cast<TrackingRecHit*>(ahit));
  //         }
  return theHits.data();
}
