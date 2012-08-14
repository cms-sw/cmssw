#ifndef RecoMuon_MuonTransientTrackingRecHit_MuonTransientTrackingRecHitBuilder_h
#define RecoMuon_MuonTransientTrackingRecHit_MuonTransientTrackingRecHitBuilder_h

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"

#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "FWCore/Framework/interface/ESHandle.h"

class MuonTransientTrackingRecHitBuilder: public TransientTrackingRecHitBuilder {
  
 public:
  
  typedef TransientTrackingRecHit::RecHitPointer RecHitPointer;
  typedef TransientTrackingRecHit::ConstRecHitContainer ConstRecHitContainer;   

  MuonTransientTrackingRecHitBuilder(edm::ESHandle<GlobalTrackingGeometry> trackingGeometry = 0);

  virtual ~MuonTransientTrackingRecHitBuilder() {} ;

  /// Call the MuonTransientTrackingRecHit::specificBuild
  RecHitPointer build(const TrackingRecHit *p, 
		      edm::ESHandle<GlobalTrackingGeometry> trackingGeometry) const ;
  
  RecHitPointer build(const TrackingRecHit * p) const;
  
  ConstRecHitContainer build(const trackingRecHit_iterator start, const trackingRecHit_iterator stop) const;
  
 private:
  edm::ESHandle<GlobalTrackingGeometry> theTrackingGeometry;

};

#endif
