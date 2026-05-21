#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/PackedGenParticle.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonTrackLinks.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "SimDataFormats/Associations/interface/TrackAssociation.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "TrackingTools/PatternTools/interface/ClosestApproachInRPhi.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"

#include "HepMC/GenEvent.h"
#include "HepMC/GenParticle.h"
#include "HepMC/GenVertex.h"
#include "HepMC/HeavyIon.h"

#include <TTree.h>

class MuonAnalyzer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  MuonAnalyzer(const edm::ParameterSet&);
  ~MuonAnalyzer() override;

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  // handles to collections of objects
  edm::EDGetTokenT<reco::VertexCollection> vertexToken_;
  edm::EDGetTokenT<edm::View<pat::PackedGenParticle>> genToken_;
  edm::EDGetTokenT<edm::View<pat::Muon>> muonToken_;
  edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> trackBuilderToken_;

  bool doGen_;
  bool doReco_;

  TTree* tree_;

  // event
  UInt_t run_;
  ULong64_t event_;
  UInt_t lumi_;

  // gen
  int nGen_;

  std::vector<float> genVtx_x_;
  std::vector<float> genVtx_y_;
  std::vector<float> genVtx_z_;

  std::vector<int> genPID_;
  std::vector<int> genStatus_;
  std::vector<float> genP_;
  std::vector<float> genPt_;
  std::vector<float> genEta_;
  std::vector<float> genPhi_;

  std::vector<int> genMotherID_;

  // reconstructed muons
  int nReco_;

  std::vector<float> recoP_;
  std::vector<float> recoPt_;
  std::vector<float> recoEta_;
  std::vector<float> recoPhi_;
  std::vector<float> recoL1Eta_;
  std::vector<float> recoL1Phi_;
  std::vector<int> recoCharge_;
  std::vector<int> recoType_;
  std::vector<bool> recoIsGood_;

  std::vector<bool> recoIsGlobal_;
  std::vector<bool> recoIsTracker_;
  std::vector<bool> recoIsPF_;
  std::vector<bool> recoIsSTA_;

  std::vector<float> recoDxy_;
  std::vector<float> recoDz_;
  std::vector<float> recoDxyErr_;
  std::vector<float> recoDzErr_;

  std::vector<float> recoIP3D_;
  std::vector<float> recoIP3DErr_;

  std::vector<int> recoNMatchedStations_;
  std::vector<float> recoIsoTrk_;
  std::vector<float> recoPFChIso_;
  std::vector<float> recoPFPhoIso_;
  std::vector<float> recoPFNeuIso_;
  std::vector<float> recoPFPUIso_;

  std::vector<float> recoMVAIso_;
  std::map<std::string, std::vector<bool>> recoMVAIsoWP_{{{"WP95",{}}, {"WP90",{}}, {"WP85",{}}, {"WP80",{}}}};

  std::vector<bool> recoIDHybridSoft_;
  std::vector<bool> recoIDSoft_;
  std::vector<bool> recoIDLoose_;
  std::vector<bool> recoIDMedium_;
  std::vector<bool> recoIDMediumPrompt_;
  std::vector<bool> recoIDTight_;
  std::vector<bool> recoIDGlobalHighPt_;
  std::vector<bool> recoIDTrkHighPt_;
  std::vector<bool> recoIDInTime_;

  std::vector<bool> recoMVAIDSoft_;
  std::vector<bool> recoMVAIDLoose_;
  std::vector<bool> recoMVAIDMedium_;
  std::vector<bool> recoMVAIDTight_;
  std::vector<bool> recoMVAIDLooseLowPt_;
  std::vector<bool> recoMVAIDMediumLowPt_;

  // inner tracker
  int nInner_;

  std::vector<float> innerDxy_;
  std::vector<float> innerDz_;
  std::vector<float> innerDxyErr_;
  std::vector<float> innerDzErr_;
  std::vector<float> innerP_;
  std::vector<float> innerPt_;
  std::vector<float> innerPtErr_;
  std::vector<float> innerEta_;

  std::vector<int> innerTrkLayers_;
  std::vector<int> innerNTrkHits_;
  std::vector<int> innerPixelLayers_;
  std::vector<int> innerNPixelHits_;
  std::vector<bool> innerIsHighPurityTrack_;

  std::vector<float> innerNormChi2_;

  // global
  int nGlobal_;

  std::vector<float> globalP_;
  std::vector<float> globalPt_;
  std::vector<float> globalPtErr_;
  std::vector<float> globalEta_;

  std::vector<bool> globalIsArbitrated_;

  std::vector<float> globalDxy_;
  std::vector<float> globalDz_;
  std::vector<float> globalDxyErr_;
  std::vector<float> globalDzErr_;

  std::vector<float> globalNormChi2_;
  std::vector<int> globalNMuonHits_;
};
