#ifndef DataFormats_TrackerRecHit2D_MTDTrackingRecHit_h
#define DataFormats_TrackerRecHit2D_MTDTrackingRecHit_h

/// A 2D TrackerRecHit with time and time error information

#include <cassert>
#include "DataFormats/TrackerRecHit2D/interface/TrackerSingleRecHit.h"
#include "DataFormats/FTLRecHit/interface/FTLRecHitCollections.h"

class MTDTrackingRecHit : public TrackerSingleRecHit {
    public:
        
        MTDTrackingRecHit() : TrackerSingleRecHit() {}
        
        MTDTrackingRecHit(const LocalPoint& p, const LocalError& e,
			  const GeomDet& idet, const FTLClusterRef& objref) :
	  TrackerSingleRecHit(p, e, idet, trackerHitRTTI::mipTiming, objref)
	  {}
        
        MTDTrackingRecHit* clone() const override { return new MTDTrackingRecHit(*this); }
	
	// things to specialize from BaseTrackerRecHit
	bool isPhase2() const final { return true; }
	void getKfComponents(KfComponentsHolder& holder) const final;
		
	int dimension() const final { return 2; }
	
	//specific timing stuff
        float energy() const { return omniCluster().mtdCluster().energy(); }
	float time() const { return omniCluster().mtdCluster().time(); }
	float timeError() const { return omniCluster().mtdCluster().timeError(); }
};


// Instantiations and specializations for FTLRecHitRef and reco::CaloClusterPtr
#include "DataFormats/Common/interface/DetSetVector.h" 
#include "DataFormats/Common/interface/OwnVector.h"
typedef edmNew::DetSetVector<MTDTrackingRecHit> MTDTrackingDetSetVector;
typedef edm::OwnVector<MTDTrackingRecHit> MTDTrackingOwnVector;

#endif
