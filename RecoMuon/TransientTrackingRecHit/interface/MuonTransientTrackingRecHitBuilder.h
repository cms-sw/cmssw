#ifndef MuonTransientTrackingRecHitBuilder_h
#define MuonTransientTrackingRecHitBuilder_h

#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"

class MuonTransientTrackingRecHitBuilder : public TransientTrackingRecHitBuilder {
  
  public:
  
    typedef TransientTrackingRecHit::RecHitPointer          RecHitPointer;
    typedef MuonTransientTrackingRecHit::MuonRecHitPointer  MuonRecHitPointer;
  
    MuonTransientTrackingRecHitBuilder( const edm::ParameterSet& );
    RecHitPointer build (const TrackingRecHit * p) const ;
    void setES(const edm::EventSetup&);

  private:

   edm::ESHandle<GlobalTrackingGeometry> theTrackingGeometry;
  
};

#endif
