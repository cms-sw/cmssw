// Original Author:     Ryan Kelley (UCSD)  
// Created:             Mon Feb 25 19:25:11 PST 2008

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"

// Tracking Specific Includes
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"
#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

// Track Association Methods
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/PatternTools/interface/TSCBLBuilderNoMaterial.h"

// physics tools
#include "DataFormats/Math/interface/LorentzVector.h"

// Producer objects
#include "AnalysisDataFormats/TrackInfo/interface/TPtoRecoTrack.h"
#include "AnalysisDataFormats/TrackInfo/interface/TPtoRecoTrackCollection.h"
#include "AnalysisDataFormats/TrackInfo/interface/RecoTracktoTP.h"
#include "AnalysisDataFormats/TrackInfo/interface/RecoTracktoTPCollection.h"

#include <string>
#include <vector>
#include <TMath.h>





class TrackAlgoCompareUtil : public edm::global::EDProducer<>
{
 public:
   
  explicit TrackAlgoCompareUtil(const edm::ParameterSet&);
  ~TrackAlgoCompareUtil();
  
 private:
  
  virtual void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  
  void SetTrackingParticleD0Dz(TrackingParticleRef tp, const reco::BeamSpot &bs, const MagneticField *bf, TPtoRecoTrack& TPRT) const;
  void SetTrackingParticleD0Dz(TrackingParticleRef tp, const reco::BeamSpot &bs, const MagneticField *bf, RecoTracktoTP& RTTP) const;
      
  // ----------member data ---------------------------
  edm::EDGetTokenT<edm::View<reco::Track>> trackLabel_algoA;
  edm::EDGetTokenT<edm::View<reco::Track>> trackLabel_algoB;
  edm::EDGetTokenT<TrackingParticleCollection> trackingParticleLabel_fakes;
  edm::EDGetTokenT<TrackingParticleCollection> trackingParticleLabel_effic;
  edm::EDGetTokenT<reco::VertexCollection> vertexLabel_algoA;
  edm::EDGetTokenT<reco::VertexCollection> vertexLabel_algoB;
  edm::EDGetTokenT<reco::BeamSpot> beamSpotLabel;
  edm::EDGetTokenT<reco::RecoToSimCollection> associatormap_algoA_recoToSim;
  edm::EDGetTokenT<reco::RecoToSimCollection> associatormap_algoB_recoToSim;
  edm::EDGetTokenT<reco::SimToRecoCollection> associatormap_algoA_simToReco;
  edm::EDGetTokenT<reco::SimToRecoCollection> associatormap_algoB_simToReco;
  edm::EDGetTokenT<reco::TrackToTrackingParticleAssociator> assocLabel_algoA;
  edm::EDGetTokenT<reco::TrackToTrackingParticleAssociator> assocLabel_algoB;
  const bool UseAssociators;
  const bool UseVertex;
  
};


//define this as a plug-in
DEFINE_FWK_MODULE(TrackAlgoCompareUtil);
