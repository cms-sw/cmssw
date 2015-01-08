// Original Author:     Ryan Kelley (UCSD)  
// Created:             Mon Feb 25 19:25:11 PST 2008

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
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
#include "CommonTools/RecoAlgos/interface/RecoTrackSelector.h"
//#include "CommonTools/RecoAlgos/interface/TrackingParticleSelector.h"
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





class TrackAlgoCompareUtil : public edm::EDProducer 
{
 public:
   
  explicit TrackAlgoCompareUtil(const edm::ParameterSet&);
  ~TrackAlgoCompareUtil();
  
 private:
  
  virtual void beginJob();
  virtual void produce(edm::Event&, const edm::EventSetup&) override;
  virtual void endJob();
  
  void SetTrackingParticleD0Dz(TrackingParticleRef tp, const reco::BeamSpot &bs, const MagneticField *bf, TPtoRecoTrack& TPRT);
  void SetTrackingParticleD0Dz(TrackingParticleRef tp, const reco::BeamSpot &bs, const MagneticField *bf, RecoTracktoTP& RTTP);
      
  // ----------member data ---------------------------
  edm::InputTag trackLabel_algoA;
  edm::InputTag trackLabel_algoB;
  edm::InputTag trackingParticleLabel_fakes;
  edm::InputTag trackingParticleLabel_effic;
  edm::InputTag vertexLabel_algoA;
  edm::InputTag vertexLabel_algoB;
  edm::InputTag trackingVertexLabel;
  edm::InputTag beamSpotLabel;
  edm::InputTag associatormap_algoA;
  edm::InputTag associatormap_algoB;
  bool UseAssociators;
  bool UseVertex;
  std::string assocLabel_algoA;     
  std::string assocLabel_algoB;     
  
};


//define this as a plug-in
DEFINE_FWK_MODULE(TrackAlgoCompareUtil);
