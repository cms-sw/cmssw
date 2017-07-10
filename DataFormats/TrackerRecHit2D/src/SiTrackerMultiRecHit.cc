#include "DataFormats/TrackerRecHit2D/interface/SiTrackerMultiRecHit.h"

using namespace std;
using namespace edm;

SiTrackerMultiRecHit::SiTrackerMultiRecHit(const LocalPoint& pos, const LocalError& err, GeomDet const & idet,
					   const std::vector< std::pair<const TrackingRecHit*, float> >& aHitMap, double annealing):
  BaseTrackerRecHit(pos,err, idet,trackerHitRTTI::multi)	
{
  for(const auto & ihit : aHitMap){
    theHits.push_back(ihit.first->clone());
    theWeights.push_back(ihit.second);
  }
  annealing_ = annealing;
}


bool SiTrackerMultiRecHit::sharesInput(const TrackingRecHit* other,
				       SharedInputType what) const
{
  if(geographicalId() != other->geographicalId()&& what==all ) return false;
  vector<const TrackingRecHit*> otherhits=other->recHits();
  if(what==all){
    if(theHits.size()!=other->recHits().size())return false;
    for(auto & otherhit : otherhits){
      bool found=false;
      for(const auto & theHit : theHits){
	if(theHit.->sharesInput(otherhit,all)){
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
    for(const auto & theHit : theHits){
      if(otherhits.size()!=0){ 
	for(auto & otherhit : otherhits){
	  if(theHit.->sharesInput(otherhit,some))return true;
	}
      }
      else{//otherwise it should be a single rechit
	if(theHit.->sharesInput(other,some))return true;
      } 
    }
    return false;
  }
}


vector<const TrackingRecHit*> SiTrackerMultiRecHit::recHits() const{
  vector<const TrackingRecHit*> myhits;
  for(const auto & theHit : theHits) {
    myhits.push_back(&theHit);
  }
  return myhits;
}

vector<TrackingRecHit*> SiTrackerMultiRecHit::recHits() {
  return theHits.data();
}


int SiTrackerMultiRecHit::dimension() const{
  //supposing all the hits inside of a MRH have the same id == same type
  int randomComponent = 0; 
  if(theHits[randomComponent].dimension() == 1 ){  return 1;  }
  else if(theHits[randomComponent].dimension() == 2 ){  return 2;  }
  else {  return 0;  }
}

void SiTrackerMultiRecHit::getKfComponents( KfComponentsHolder & holder ) const { 
  if (dimension() == 1)  getKfComponents1D(holder); 
  if (dimension() == 2)  getKfComponents2D(holder); 
} 
