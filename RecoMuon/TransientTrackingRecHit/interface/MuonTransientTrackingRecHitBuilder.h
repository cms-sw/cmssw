#ifndef RecoMuon_MuonTransientTrackingRecHit_MuonTransientTrackingRecHitBuilder_h
#define RecoMuon_MuonTransientTrackingRecHit_MuonTransientTrackingRecHitBuilder_h

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"

#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "FWCore/Framework/interface/ESHandle.h"

class MuonTransientTrackingRecHitBuilder: public TransientTrackingRecHitBuilder {
  
 public:
  
  typedef TransientTrackingRecHit::RecHitPointer          RecHitPointer;
  
  MuonTransientTrackingRecHitBuilder(edm::ESHandle<GlobalTrackingGeometry> trackingGeometry = 0);

  /// Call the MuonTransientTrackingRecHit::specificBuild
  RecHitPointer build(const TrackingRecHit *p, 
		      edm::ESHandle<GlobalTrackingGeometry> trackingGeometry) const ;
  
  virtual RecHitPointer build(const TrackingRecHit * p) const;
  
 private:
  edm::ESHandle<GlobalTrackingGeometry> theTrackingGeometry;

};

#endif
