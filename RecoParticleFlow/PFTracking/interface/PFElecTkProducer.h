#ifndef PFElecTkProducer_H
#define PFElecTkProducer_H
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoTracker/TrackProducer/interface/TrackProducerAlgorithm.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h" 
#include "TrackingTools/Records/interface/TransientRecHitRecord.h" 
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrackFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFTrajectoryPoint.h"
#include "RecoParticleFlow/PFTracking/interface/PFTrackTransformer.h"

typedef std::pair<Trajectory*, reco::GsfTrack*> AlGsfProduct; 

class PFElecTkProducer : public edm::EDProducer {
   public:
      explicit PFElecTkProducer(const edm::ParameterSet&);
      ~PFElecTkProducer();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
 
      // ----------member data ---------------------------
  
      edm::ParameterSet conf_;
      TrackProducerAlgorithm trackAlgo_;
      std::string gsfTrackModule_;
      std::string gsfTrackCandidateModule_;
      std::string fitterName_;
      std::string propagatorName_;
      std::string builderName_;
      edm::ESHandle<MagneticField> theMF;
      edm::ESHandle<TrackerGeometry> theG;
      edm::ESHandle<Propagator> thePropagator;
      edm::ESHandle<TransientTrackingRecHitBuilder> theBuilder;
      edm::ESHandle<TrajectoryFitter> theFitter;
      const MagneticField * magField;
      std::vector<reco::PFRecTrack> pftracks;
      PFTrackTransformer *PFTransformer; 
      bool trajinev_;
};
#endif
