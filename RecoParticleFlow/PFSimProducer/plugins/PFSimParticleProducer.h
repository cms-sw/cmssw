#ifndef RecoParticleFlow_PFProducer_PFSimParticleProducer_h_
#define RecoParticleFlow_PFProducer_PFSimParticleProducer_h_

// system include files
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "RecoParticleFlow/PFTracking/interface/PFGeometry.h"
#include "DataFormats/ParticleFlowReco/interface/PFSimParticle.h"
#include "DataFormats/ParticleFlowReco/interface/PFSimParticleFwd.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrackFwd.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "FastSimulation/Event/interface/FSimVertex.h"

#include "RecoParticleFlow/PFProducer/interface/PFBlockAlgo.h"

#include "FastSimulation/Particle/interface/ParticleTable.h"

/**\class PFSimParticleProducer 
\brief Producer for PFRecTracks and PFSimParticles

\todo Remove the PFRecTrack part, which is now handled by PFTracking
\author Colin Bernet
\date   April 2007
*/

class FSimEvent;



class PFSimParticleProducer : public edm::stream::EDProducer<> {
 public:

  explicit PFSimParticleProducer(const edm::ParameterSet&);

  ~PFSimParticleProducer() override;
  
  void produce(edm::Event&, const edm::EventSetup&) override;

  typedef edm::Handle<reco::PFRecTrackCollection> TrackHandle;
  void getSimIDs( const TrackHandle& trackh,
		  std::vector<unsigned>& recTrackSimID );

 private:
    

  /// module label for retrieving input simtrack and simvertex
  edm::InputTag    inputTagSim_;
  edm::EDGetTokenT<std::vector<SimTrack> >  tokenSim_;  
  edm::EDGetTokenT<std::vector<SimVertex> >  tokenSimVertices_;  

  //MC Truth Matching 
  //modif-beg
  bool mctruthMatchingInfo_;
  edm::InputTag    inputTagFamosSimHits_;
  edm::EDGetTokenT<edm::PCaloHitContainer>    tokenFamosSimHits_;
  //modif-end

  edm::InputTag   inputTagRecTracks_;
  edm::EDGetTokenT<reco::PFRecTrackCollection>   tokenRecTracks_;
  edm::InputTag    inputTagEcalRecHitsEB_;
  edm::EDGetTokenT<EcalRecHitCollection>    tokenEcalRecHitsEB_;
  edm::InputTag    inputTagEcalRecHitsEE_;
  edm::EDGetTokenT<EcalRecHitCollection>    tokenEcalRecHitsEE_;

  // parameters for retrieving true particles information --

  edm::ParameterSet particleFilter_;
  FSimEvent* mySimEvent;

  // flags for the various tasks ---------------------------

  /// process particles on/off
  bool   processParticles_;

  /// verbose ?
  bool   verbose_;

};  

#endif
