// -*- C++ -*-
//
// Package:    LhcTrackAnalyzer
// Class:      LhcTrackAnalyzer
//
/**\class LhcTrackAnalyzer LhcTrackAnalyzer.cc Alignment/HIPAlignmentAlgorithm/plugins/LhcTrackAnalyzer.cc

   Originally written by M.Musich
   Expanded by A. Bonato

   Description: Ntuplizer for collision tracks
*/
//

// system include files
#include <memory>
#include <string>
#include <vector>

// user include files
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

// ROOT includes
#include "TFile.h"
#include "TTree.h"

//
// class decleration
//

class LhcTrackAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit LhcTrackAnalyzer(const edm::ParameterSet&);
  ~LhcTrackAnalyzer() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  // ----------member data ---------------------------
  edm::InputTag TrackCollectionTag_;
  edm::InputTag PVtxCollectionTag_;
  bool debug_;
  std::vector<unsigned int> acceptedBX_;

  edm::EDGetTokenT<reco::TrackCollection> theTrackCollectionToken;
  edm::EDGetTokenT<reco::VertexCollection> theVertexCollectionToken;

  // Output
  std::string filename_;
  TFile* rootFile_;
  TTree* rootTree_;

  // Root-Tuple variables :
  //=======================
  void SetVarToZero();

  static const int nMaxtracks_ = 3000;
  int nTracks_;
  int run_;
  int event_;
  double pt_[nMaxtracks_];
  double eta_[nMaxtracks_];
  double phi_[nMaxtracks_];
  double chi2_[nMaxtracks_];
  double chi2ndof_[nMaxtracks_];
  int charge_[nMaxtracks_];
  double qoverp_[nMaxtracks_];
  double dz_[nMaxtracks_];
  double dxy_[nMaxtracks_];
  double xPCA_[nMaxtracks_];
  double yPCA_[nMaxtracks_];
  double zPCA_[nMaxtracks_];
  int trkAlgo_[nMaxtracks_];
  int trkQuality_[nMaxtracks_];
  int isHighPurity_[nMaxtracks_];
  int validhits_[nMaxtracks_][7];
  bool goodbx_;
  bool goodvtx_;
};

// Constructor

LhcTrackAnalyzer::LhcTrackAnalyzer(const edm::ParameterSet& iConfig)
    : TrackCollectionTag_(iConfig.getParameter<edm::InputTag>("TrackCollectionTag")),
      PVtxCollectionTag_(iConfig.getParameter<edm::InputTag>("PVtxCollectionTag")),
      debug_(iConfig.getParameter<bool>("Debug")),
      acceptedBX_(iConfig.getParameter<std::vector<unsigned int>>("acceptedBX")),
      filename_(iConfig.getParameter<std::string>("OutputFileName")) {
  //now do what ever initialization is needed
  theTrackCollectionToken = consumes<reco::TrackCollection>(TrackCollectionTag_);
  theVertexCollectionToken = consumes<reco::VertexCollection>(PVtxCollectionTag_);
}

//
// member functions
//

/*****************************************************************************/
void LhcTrackAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
/*****************************************************************************/
{
  using namespace edm;
  using namespace reco;
  using namespace std;

  //=======================================================
  // Initialize Root-tuple variables
  //=======================================================

  SetVarToZero();

  //=======================================================
  // Retrieve the Track information
  //=======================================================

  const auto& vertices = iEvent.get(theVertexCollectionToken);
  const auto& vtx = vertices.front();
  if (vtx.isFake()) {
    goodvtx_ = false;
  } else {
    goodvtx_ = true;
  }

  int bx = iEvent.bunchCrossing();
  if (acceptedBX_.empty()) {
    goodbx_ = true;
  } else {
    if (std::find(acceptedBX_.begin(), acceptedBX_.end(), bx) != acceptedBX_.end()) {
      goodbx_ = true;
    }
  }

  run_ = iEvent.id().run();
  event_ = iEvent.id().event();

  const auto& tracks = iEvent.get(theTrackCollectionToken);
  if (debug_) {
    edm::LogInfo("LhcTrackAnalyzer") << "LhcTrackAnalyzer::analyze() looping over " << tracks.size() << "tracks."
                                     << endl;
  }

  for (const auto& track : tracks) {
    if (nTracks_ >= nMaxtracks_) {
      edm::LogWarning("LhcTrackAnalyzer")
          << " LhcTrackAnalyzer::analyze() : Warning - Run " << run_ << " Event " << event_
          << "\tNumber of tracks: " << tracks.size() << " , greater than " << nMaxtracks_ << std::endl;
      continue;
    }
    pt_[nTracks_] = track.pt();
    eta_[nTracks_] = track.eta();
    phi_[nTracks_] = track.phi();
    chi2_[nTracks_] = track.chi2();
    chi2ndof_[nTracks_] = track.normalizedChi2();
    charge_[nTracks_] = track.charge();
    qoverp_[nTracks_] = track.qoverp();
    dz_[nTracks_] = track.dz();
    dxy_[nTracks_] = track.dxy();
    xPCA_[nTracks_] = track.vertex().x();
    yPCA_[nTracks_] = track.vertex().y();
    zPCA_[nTracks_] = track.vertex().z();
    validhits_[nTracks_][0] = track.numberOfValidHits();
    validhits_[nTracks_][1] = track.hitPattern().numberOfValidPixelBarrelHits();
    validhits_[nTracks_][2] = track.hitPattern().numberOfValidPixelEndcapHits();
    validhits_[nTracks_][3] = track.hitPattern().numberOfValidStripTIBHits();
    validhits_[nTracks_][4] = track.hitPattern().numberOfValidStripTIDHits();
    validhits_[nTracks_][5] = track.hitPattern().numberOfValidStripTOBHits();
    validhits_[nTracks_][6] = track.hitPattern().numberOfValidStripTECHits();

    int myalgo = -88;
    if (track.algo() == reco::TrackBase::undefAlgorithm) {
      myalgo = 0;
    } else if (track.algo() == reco::TrackBase::ctf) {
      myalgo = 1;
    } else if (track.algo() == reco::TrackBase::duplicateMerge) {
      myalgo = 2;
    } else if (track.algo() == reco::TrackBase::cosmics) {
      myalgo = 3;
    } else if (track.algo() == reco::TrackBase::initialStep) {
      myalgo = 4;
    } else if (track.algo() == reco::TrackBase::lowPtTripletStep) {
      myalgo = 5;
    } else if (track.algo() == reco::TrackBase::pixelPairStep) {
      myalgo = 6;
    } else if (track.algo() == reco::TrackBase::detachedTripletStep) {
      myalgo = 7;
    } else if (track.algo() == reco::TrackBase::mixedTripletStep) {
      myalgo = 8;
    } else if (track.algo() == reco::TrackBase::pixelLessStep) {
      myalgo = 9;
    } else if (track.algo() == reco::TrackBase::tobTecStep) {
      myalgo = 10;
    } else if (track.algo() == reco::TrackBase::jetCoreRegionalStep) {
      myalgo = 11;
    } else if (track.algo() == reco::TrackBase::muonSeededStepInOut) {
      myalgo = 13;
    } else if (track.algo() == reco::TrackBase::muonSeededStepOutIn) {
      myalgo = 14;
    } else if (track.algo() == reco::TrackBase::highPtTripletStep) {
      myalgo = 22;
    } else if (track.algo() == reco::TrackBase::lowPtQuadStep) {
      myalgo = 23;
    } else if (track.algo() == reco::TrackBase::detachedQuadStep) {
      myalgo = 24;
    } else {
      myalgo = 25;
      edm::LogWarning("LhcTrackAnalyzer")
          << "LhcTrackAnalyzer does not support all types of tracks, encountered one from algo "
          << reco::TrackBase::algoName(track.algo());
    }
    trkAlgo_[nTracks_] = myalgo;

    int myquality = -99;
    if (track.quality(reco::TrackBase::undefQuality))
      myquality = -1;
    if (track.quality(reco::TrackBase::loose))
      myquality = 0;
    if (track.quality(reco::TrackBase::tight))
      myquality = 1;
    if (track.quality(reco::TrackBase::highPurity))
      myquality = 2;
    trkQuality_[nTracks_] = myquality;

    if (track.quality(reco::TrackBase::highPurity))
      isHighPurity_[nTracks_] = 1;
    else
      isHighPurity_[nTracks_] = 0;
    nTracks_++;

  }  //end loop on tracks

  for (int d = 0; d < nTracks_; ++d) {
    if (abs(trkQuality_[d]) > 5)
      edm::LogInfo("LhcTrackAnalyzer") << "MYQUALITY!!! " << trkQuality_[d] << " at track # " << d << "/" << nTracks_
                                       << endl;
  }

  rootTree_->Fill();
}

/*****************************************************************************/
void LhcTrackAnalyzer::beginJob()
/*****************************************************************************/
{
  edm::LogInfo("beginJob") << "Begin Job" << std::endl;
  // Define TTree for output
  rootFile_ = new TFile(filename_.c_str(), "recreate");
  rootTree_ = new TTree("tree", "Lhc Track tree");

  // Track Paramters
  rootTree_->Branch("run", &run_, "run/I");
  rootTree_->Branch("event", &event_, "event/I");
  rootTree_->Branch("goodbx", &goodbx_, "goodbx/O");
  rootTree_->Branch("goodvtx", &goodvtx_, "goodvtx/O");
  rootTree_->Branch("nTracks", &nTracks_, "nTracks/I");
  rootTree_->Branch("pt", &pt_, "pt[nTracks]/D");
  rootTree_->Branch("eta", &eta_, "eta[nTracks]/D");
  rootTree_->Branch("phi", &phi_, "phi[nTracks]/D");
  rootTree_->Branch("chi2", &chi2_, "chi2[nTracks]/D");
  rootTree_->Branch("chi2ndof", &chi2ndof_, "chi2ndof[nTracks]/D");
  rootTree_->Branch("charge", &charge_, "charge[nTracks]/I");
  rootTree_->Branch("qoverp", &qoverp_, "qoverp[nTracks]/D");
  rootTree_->Branch("dz", &dz_, "dz[nTracks]/D");
  rootTree_->Branch("dxy", &dxy_, "dxy[nTracks]/D");
  rootTree_->Branch("xPCA", &xPCA_, "xPCA[nTracks]/D");
  rootTree_->Branch("yPCA", &yPCA_, "yPCA[nTracks]/D");
  rootTree_->Branch("zPCA", &zPCA_, "zPCA[nTracks]/D");
  rootTree_->Branch("isHighPurity", &isHighPurity_, "isHighPurity[nTracks]/I");
  rootTree_->Branch("trkQuality", &trkQuality_, "trkQuality[nTracks]/I");
  rootTree_->Branch("trkAlgo", &trkAlgo_, "trkAlgo[nTracks]/I");
  rootTree_->Branch("nValidHits", &validhits_, "nValidHits[nTracks][7]/I");
}

/*****************************************************************************/
void LhcTrackAnalyzer::endJob()
/*****************************************************************************/
{
  if (rootFile_) {
    rootFile_->Write();
    rootFile_->Close();
  }
}

/*****************************************************************************/
void LhcTrackAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions)
/*****************************************************************************/
{
  edm::ParameterSetDescription desc;
  desc.setComment("Ntuplizer for LHC tracks");
  desc.add<edm::InputTag>("TrackCollectionTag", edm::InputTag("ALCARECOTkAlMinBias"));
  desc.add<edm::InputTag>("PVtxCollectionTag", edm::InputTag("offlinePrimaryVertices"));
  desc.add<bool>("Debug", false);
  desc.add<std::vector<unsigned int>>("acceptedBX", {});
  desc.add<std::string>("OutputFileName", "LhcTrackAnalyzer_Output_default.root");
  descriptions.addWithDefaultLabel(desc);
}

/*****************************************************************************/
void LhcTrackAnalyzer::SetVarToZero()
/*****************************************************************************/
{
  run_ = -1;
  event_ = -99;
  nTracks_ = 0;
  for (int i = 0; i < nMaxtracks_; ++i) {
    pt_[i] = 0;
    eta_[i] = 0;
    phi_[i] = 0;
    chi2_[i] = 0;
    chi2ndof_[i] = 0;
    charge_[i] = 0;
    qoverp_[i] = 0;
    dz_[i] = 0;
    dxy_[i] = 0;
    xPCA_[i] = 0;
    yPCA_[i] = 0;
    zPCA_[i] = 0;
    trkQuality_[i] = 0;
    trkAlgo_[i] = -1;
    isHighPurity_[i] = -3;
    for (int j = 0; j < 7; j++) {
      validhits_[nTracks_][j] = -1 * j;
    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(LhcTrackAnalyzer);
