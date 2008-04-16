// Original Author:		Ryan Kelley (UCSD)  
// Created:  			Mon Feb 25 19:25:11 PST 2008

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
#include "FWCore/ParameterSet/interface/InputTag.h"

// Tracking Specific Includes
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"
#include "PhysicsTools/RecoAlgos/interface/RecoTrackSelector.h"
#include "PhysicsTools/RecoAlgos/interface/TrackingParticleSelector.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

// Track Association Methods
#include "SimTracker/TrackAssociation/interface/TrackAssociatorByHits.h"
#include "SimTracker/TrackAssociation/interface/TrackAssociatorByChi2.h"
#include "SimTracker/Records/interface/TrackAssociatorRecord.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/PatternTools/interface/TrajectoryStateClosestToBeamLineBuilder.h"

// physics tools
//#include "Math/ProbFuncMathMore.h"
#include "DataFormats/Math/interface/LorentzVector.h"

// Producer object
#include "AnalysisDataFormats/TrackInfo/interface/TPtoRecoTrack.h"
#include "AnalysisDataFormats/TrackInfo/interface/TPtoRecoTrackCollection.h"

#include <string>
#include <vector>
#include <TMath.h>

using namespace std;
using namespace edm;


class TrackAlgoCompareUtil : public edm::EDProducer 
{
 public:
   
  explicit TrackAlgoCompareUtil(const edm::ParameterSet&);
  ~TrackAlgoCompareUtil();
  
 private:
  
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  
  void SetTrackingParticleD0Dz(TrackingParticleRef tp, const reco::BeamSpot &bs, const MagneticField *bf, TPtoRecoTrack& TPRT);
	  
  // ----------member data ---------------------------
  edm::InputTag trackLabel_algoA;
  edm::InputTag trackLabel_algoB;
  edm::InputTag trackingParticleLabel_fakes;
  edm::InputTag trackingParticleLabel_effic;
  edm::InputTag vertexLabel_algoA;
  edm::InputTag vertexLabel_algoB;
  edm::InputTag trackingVertexLabel;
  edm::InputTag beamSpotLabel;
  std::string assocLabel;     
  
};


//define this as a plug-in
DEFINE_FWK_MODULE(TrackAlgoCompareUtil);
