#ifndef GoodSeedProducer_H
#define GoodSeedProducer_H
// system include files
#include <memory>

// user include files

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/PatternTools/interface/TrajectoryFitter.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h" 
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"  
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "RecoTracker/TrackProducer/interface/TrackProducerAlgorithm.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrackFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFTrajectoryPoint.h"
#include "RecoParticleFlow/PFTracking/interface/PFTrackTransformer.h"
#include "DataFormats/ParticleFlowReco/interface/PFResolutionMap.h"
namespace reco {
  class PFResolutionMap;
}
class GoodSeedProducer : public edm::EDProducer {
  typedef TrajectoryStateOnSurface TSOS;
   public:
      explicit GoodSeedProducer(const edm::ParameterSet&);
      ~GoodSeedProducer();
  
   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob(){}
      int findIndex(reco::TrackCollection  hj, 
		    AlgoProduct ap);

      int getBin(float,float);

      edm::ParameterSet conf_;
      std::string recTrackCandidateModuleLabel_;
      std::string recTrackCollectionLabel_;
      edm::InputTag pfCLusTagPSLabel_;
      edm::InputTag pfCLusTagECLabel_;
      std::string fitterName_;
      std::string propagatorName_;
      std::string builderName_;
      std::string preidckf_;
      std::string preidgsf_;
      edm::ESHandle<TrackerGeometry> tracker;
      TrackProducerAlgorithm trackAlgo_;
      const MagneticField * magField;
      edm::ESHandle<MagneticField> theMF;
      edm::ESHandle<TrackerGeometry> theG;
      edm::ESHandle<Propagator> thePropagator;
      edm::ESHandle<TransientTrackingRecHitBuilder> theBuilder;
      edm::ESHandle<TrajectoryFitter> theFitter;


      //      AnalyticalPropagator bkwdPropagator(magField, oppositeToMomentum);
      
      // ----------member data ---------------------------
   
  
      TrajectorySeed Seed;
      float pt_threshold;
      const TransientTrackingRecHitBuilder *RHBuilder;
      std::vector<reco::PFRecTrack> pftracks;
      PFTrackTransformer *PFTransformer;
      bool produceCkfseed,produceCkfPFT;
      int index;
      int side;
      static reco::PFResolutionMap* resMapEtaECAL_;
      static reco::PFResolutionMap* resMapPhiECAL_;
      float thr[105];
};
#endif
