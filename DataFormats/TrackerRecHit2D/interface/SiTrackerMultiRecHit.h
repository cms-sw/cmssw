#ifndef SiTrackerMultiRecHit_H
#define SiTrackerMultiRecHit_H

#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include <vector>
#include <map>
/*
A rechit type suitable for tracking algorithm that use soft hit-to-track assignement, 
such as the Deterministic Annealing Filter (DAF) or the Multi Track Filter (MTF).
it contains an OwnVector with the component rechits and a vector of weights	
*/
class SiTrackerMultiRecHit : public BaseTrackerRecHit
{
public:
  SiTrackerMultiRecHit():
    theHits(),
    theWeights(){}
  virtual ~SiTrackerMultiRecHit(){}	
  
  
  SiTrackerMultiRecHit(const LocalPoint&, const LocalError&, const DetId&, 
		       const std::vector< std::pair<const TrackingRecHit*, float> >&);
  
  virtual SiTrackerMultiRecHit* clone() const {return new SiTrackerMultiRecHit(*this);}
  
  virtual int dimension() const {return 2;}
  virtual void getKfComponents( KfComponentsHolder & holder ) const { getKfComponents2D(holder); }


  
  //vector of component rechits
  virtual std::vector<const TrackingRecHit*> recHits() const;
  
  virtual std::vector<TrackingRecHit*> recHits() ;
  
  //vector of weights
  std::vector<float> weights() const {return theWeights;}
  
  //returns the weight for the i component
  float  weight(unsigned int i) const ;
	
  bool sharesInput(const TrackingRecHit* other,
		   SharedInputType what) const;
private:
  
  edm::OwnVector<TrackingRecHit> theHits;
  std::vector<float> theWeights;
	

};

// Comparison operators
inline bool operator<( const SiTrackerMultiRecHit& one, const SiTrackerMultiRecHit& other) {
  if ( one.geographicalId() < other.geographicalId() ) {
    return true;
  } else {
    return false;
  }
}

#endif
