#ifndef RecoMuon_MuonTransientTrackingRecHit_MuonTransientTrackingRecHitBuilder_h
#define RecoMuon_MuonTransientTrackingRecHit_MuonTransientTrackingRecHitBuilder_h

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "FWCore/Framework/interface/ESHandle.h"

class MuonTransientTrackingRecHitBuilder {
  
 public:
  
  typedef TransientTrackingRecHit::RecHitPointer          RecHitPointer;
  
  MuonTransientTrackingRecHitBuilder(){}

  /// Call the MuonTransientTrackingRecHit::specificBuild
  RecHitPointer build (const TrackingRecHit *p, 
		       edm::ESHandle<GlobalTrackingGeometry> trackingGeometry) const ;
  
 private:
};

#endif
