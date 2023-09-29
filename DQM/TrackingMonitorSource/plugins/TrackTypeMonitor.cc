// system includes
#include <string>
#include <vector>
#include <map>
#include <tuple>
#include <set>

// user includes
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

// ROOT includes
#include "TFile.h"
#include "TH1.h"
#include "TMath.h"
#include "TPRegexp.h"

class TrackTypeMonitor : public DQMEDAnalyzer {
public:
  TrackTypeMonitor(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

private:
  void fillHistograms(const reco::Track& track, int indx);

  const edm::ParameterSet parameters_;

  const std::string moduleName_;
  const std::string folderName_;
  const bool verbose_;

  const edm::InputTag muonTag_;
  const edm::InputTag electronTag_;
  const edm::InputTag trackTag_;
  const edm::InputTag bsTag_;
  const edm::InputTag vertexTag_;

  const edm::EDGetTokenT<reco::GsfElectronCollection> electronToken_;
  const edm::EDGetTokenT<reco::MuonCollection> muonToken_;
  const edm::EDGetTokenT<reco::TrackCollection> trackToken_;
  const edm::EDGetTokenT<reco::BeamSpot> bsToken_;
  const edm::EDGetTokenT<reco::VertexCollection> vertexToken_;

  const std::string trackQuality_;

  std::vector<MonitorElement*> trackEtaHList_;
  std::vector<MonitorElement*> trackPhiHList_;
  std::vector<MonitorElement*> trackPHList_;
  std::vector<MonitorElement*> trackPtHList_;
  std::vector<MonitorElement*> trackPterrHList_;
  std::vector<MonitorElement*> trackqOverpHList_;
  std::vector<MonitorElement*> trackChi2bynDOFHList_;
  std::vector<MonitorElement*> nTracksHList_;
  std::vector<MonitorElement*> trackdzHList_;

  MonitorElement* hcounterH_;
  MonitorElement* dphiH_;
  MonitorElement* drH_;

  unsigned long long m_cacheID_;
};

void TrackTypeMonitor::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.addUntracked<std::string>("ModuleName", "TrackTypeMonitor");
  desc.addUntracked<std::string>("FolderName", "highPurityTracks");
  desc.addUntracked<bool>("verbose", false);
  desc.addUntracked<edm::InputTag>("muonInputTag", edm::InputTag("muons"));
  desc.addUntracked<edm::InputTag>("electronInputTag", edm::InputTag("gedGsfElectrons"));
  desc.addUntracked<edm::InputTag>("trackInputTag", edm::InputTag("generalTracks"));
  desc.addUntracked<edm::InputTag>("offlineBeamSpot", edm::InputTag("offlineBeamSpot"));
  desc.addUntracked<edm::InputTag>("vertexTag", edm::InputTag("offlinePrimaryVertices"));
  desc.addUntracked<std::string>("trackQuality", "highPurity");

  std::map<std::string, std::tuple<int, double, double> > listOfPS = {{"TrackEtaPar", {60, -3.0, 3.0}},
                                                                      {"TrackPhiPar", {100, -4.0, 4.0}},
                                                                      {"TrackPPar", {100, 0., 100.}},
                                                                      {"TrackPtPar", {100, 0., 100.}},
                                                                      {"TrackPterrPar", {100, 0., 100.}},
                                                                      {"TrackqOverpPar", {100, -10., 10.}},
                                                                      {"TrackdzPar", {100, -100., 100.}},
                                                                      {"TrackChi2bynDOFPar", {100, 0.0, 10.0}},
                                                                      {"nTracksPar", {100, -0.5, 99.5}}};

  for (const auto& [key, pSet] : listOfPS) {
    edm::ParameterSetDescription PSetToAdd;
    PSetToAdd.add<int>("Xbins", std::get<0>(pSet));
    PSetToAdd.add<double>("Xmin", std::get<1>(pSet));
    PSetToAdd.add<double>("Xmax", std::get<2>(pSet));
    desc.add<edm::ParameterSetDescription>(key, PSetToAdd);
  }

  descriptions.addWithDefaultLabel(desc);
}

template <class T>
class PtComparator {
public:
  bool operator()(const T* a, const T* b) const { return (a->pt() > b->pt()); }
};
// -----------------------------
// constructors and destructor
// -----------------------------
TrackTypeMonitor::TrackTypeMonitor(const edm::ParameterSet& ps)
    : parameters_(ps),
      moduleName_(parameters_.getUntrackedParameter<std::string>("ModuleName", "TrackTypeMonitor")),
      folderName_(parameters_.getUntrackedParameter<std::string>("FolderName", "highPurityTracks")),
      verbose_(parameters_.getUntrackedParameter<bool>("verbose", false)),
      muonTag_(ps.getUntrackedParameter<edm::InputTag>("muonInputTag", edm::InputTag("muons"))),
      electronTag_(ps.getUntrackedParameter<edm::InputTag>("electronInputTag", edm::InputTag("gedGsfElectrons"))),
      trackTag_(parameters_.getUntrackedParameter<edm::InputTag>("trackInputTag", edm::InputTag("generalTracks"))),
      bsTag_(parameters_.getUntrackedParameter<edm::InputTag>("offlineBeamSpot", edm::InputTag("offlineBeamSpot"))),
      vertexTag_(
          parameters_.getUntrackedParameter<edm::InputTag>("vertexTag", edm::InputTag("offlinePrimaryVertices"))),
      electronToken_(consumes<reco::GsfElectronCollection>(electronTag_)),
      muonToken_(consumes<reco::MuonCollection>(muonTag_)),
      trackToken_(consumes<reco::TrackCollection>(trackTag_)),
      bsToken_(consumes<reco::BeamSpot>(bsTag_)),
      vertexToken_(consumes<reco::VertexCollection>(vertexTag_)),
      trackQuality_(parameters_.getUntrackedParameter<std::string>("trackQuality", "highPurity")) {
  trackEtaHList_.clear();
  trackPhiHList_.clear();
  trackPHList_.clear();
  trackPtHList_.clear();
  trackPterrHList_.clear();
  trackqOverpHList_.clear();
  trackChi2bynDOFHList_.clear();
  nTracksHList_.clear();
}
void TrackTypeMonitor::bookHistograms(DQMStore::IBooker& iBook, edm::Run const& iRun, edm::EventSetup const& iSetup) {
  edm::ParameterSet TrackEtaHistoPar = parameters_.getParameter<edm::ParameterSet>("TrackEtaPar");
  edm::ParameterSet TrackPhiHistoPar = parameters_.getParameter<edm::ParameterSet>("TrackPhiPar");
  edm::ParameterSet TrackPHistoPar = parameters_.getParameter<edm::ParameterSet>("TrackPPar");
  edm::ParameterSet TrackPtHistoPar = parameters_.getParameter<edm::ParameterSet>("TrackPtPar");
  edm::ParameterSet TrackPterrHistoPar = parameters_.getParameter<edm::ParameterSet>("TrackPterrPar");
  edm::ParameterSet TrackqOverpHistoPar = parameters_.getParameter<edm::ParameterSet>("TrackqOverpPar");
  edm::ParameterSet TrackdzHistoPar = parameters_.getParameter<edm::ParameterSet>("TrackdzPar");
  edm::ParameterSet TrackChi2bynDOFHistoPar = parameters_.getParameter<edm::ParameterSet>("TrackChi2bynDOFPar");
  edm::ParameterSet nTracksHistoPar = parameters_.getParameter<edm::ParameterSet>("nTracksPar");

  std::string currentFolder = moduleName_ + "/" + folderName_;
  iBook.setCurrentFolder(currentFolder);

  trackEtaHList_.push_back(iBook.book1D("trackEtaIso",
                                        "Isolated Track Eta",
                                        TrackEtaHistoPar.getParameter<int32_t>("Xbins"),
                                        TrackEtaHistoPar.getParameter<double>("Xmin"),
                                        TrackEtaHistoPar.getParameter<double>("Xmax")));
  trackEtaHList_.push_back(iBook.book1D("trackEtaNoIso",
                                        "NonIsolated Track Eta",
                                        TrackEtaHistoPar.getParameter<int32_t>("Xbins"),
                                        TrackEtaHistoPar.getParameter<double>("Xmin"),
                                        TrackEtaHistoPar.getParameter<double>("Xmax")));
  trackEtaHList_.push_back(iBook.book1D("trackEtaUL",
                                        "Underlying Track Eta",
                                        TrackEtaHistoPar.getParameter<int32_t>("Xbins"),
                                        TrackEtaHistoPar.getParameter<double>("Xmin"),
                                        TrackEtaHistoPar.getParameter<double>("Xmax")));
  trackEtaHList_.push_back(iBook.book1D("trackEtaALL",
                                        "All Track Eta",
                                        TrackEtaHistoPar.getParameter<int32_t>("Xbins"),
                                        TrackEtaHistoPar.getParameter<double>("Xmin"),
                                        TrackEtaHistoPar.getParameter<double>("Xmax")));

  trackPhiHList_.push_back(iBook.book1D("trackPhiIso",
                                        "Isolated Track Phi",
                                        TrackPhiHistoPar.getParameter<int32_t>("Xbins"),
                                        TrackPhiHistoPar.getParameter<double>("Xmin"),
                                        TrackPhiHistoPar.getParameter<double>("Xmax")));
  trackPhiHList_.push_back(iBook.book1D("trackPhiNonIso",
                                        "NonIsolated Track Phi",
                                        TrackPhiHistoPar.getParameter<int32_t>("Xbins"),
                                        TrackPhiHistoPar.getParameter<double>("Xmin"),
                                        TrackPhiHistoPar.getParameter<double>("Xmax")));
  trackPhiHList_.push_back(iBook.book1D("trackPhiUL",
                                        "Underlying Track Phi",
                                        TrackPhiHistoPar.getParameter<int32_t>("Xbins"),
                                        TrackPhiHistoPar.getParameter<double>("Xmin"),
                                        TrackPhiHistoPar.getParameter<double>("Xmax")));
  trackPhiHList_.push_back(iBook.book1D("trackPhiALL",
                                        "All Track Phi",
                                        TrackPhiHistoPar.getParameter<int32_t>("Xbins"),
                                        TrackPhiHistoPar.getParameter<double>("Xmin"),
                                        TrackPhiHistoPar.getParameter<double>("Xmax")));

  trackPHList_.push_back(iBook.book1D("trackPIso",
                                      "Isolated Track P",
                                      TrackPHistoPar.getParameter<int32_t>("Xbins"),
                                      TrackPHistoPar.getParameter<double>("Xmin"),
                                      TrackPHistoPar.getParameter<double>("Xmax")));
  trackPHList_.push_back(iBook.book1D("trackPNonIso",
                                      "NonIsolated Track P",
                                      TrackPHistoPar.getParameter<int32_t>("Xbins"),
                                      TrackPHistoPar.getParameter<double>("Xmin"),
                                      TrackPHistoPar.getParameter<double>("Xmax")));
  trackPHList_.push_back(iBook.book1D("trackPUL",
                                      "Underlying Track P",
                                      TrackPHistoPar.getParameter<int32_t>("Xbins"),
                                      TrackPHistoPar.getParameter<double>("Xmin"),
                                      TrackPHistoPar.getParameter<double>("Xmax")));
  trackPHList_.push_back(iBook.book1D("trackPALL",
                                      "All Track P",
                                      TrackPHistoPar.getParameter<int32_t>("Xbins"),
                                      TrackPHistoPar.getParameter<double>("Xmin"),
                                      TrackPHistoPar.getParameter<double>("Xmax")));

  trackPtHList_.push_back(iBook.book1D("trackPtIsolated",
                                       "Isolated Track Pt",
                                       TrackPtHistoPar.getParameter<int32_t>("Xbins"),
                                       TrackPtHistoPar.getParameter<double>("Xmin"),
                                       TrackPtHistoPar.getParameter<double>("Xmax")));
  trackPtHList_.push_back(iBook.book1D("trackPtNonIsolated",
                                       "NonIsolated Track Pt",
                                       TrackPtHistoPar.getParameter<int32_t>("Xbins"),
                                       TrackPtHistoPar.getParameter<double>("Xmin"),
                                       TrackPtHistoPar.getParameter<double>("Xmax")));
  trackPtHList_.push_back(iBook.book1D("trackPtUL",
                                       "Underlying Track Pt",
                                       TrackPtHistoPar.getParameter<int32_t>("Xbins"),
                                       TrackPtHistoPar.getParameter<double>("Xmin"),
                                       TrackPtHistoPar.getParameter<double>("Xmax")));
  trackPtHList_.push_back(iBook.book1D("trackPtALL",
                                       "All Track Pt",
                                       TrackPtHistoPar.getParameter<int32_t>("Xbins"),
                                       TrackPtHistoPar.getParameter<double>("Xmin"),
                                       TrackPtHistoPar.getParameter<double>("Xmax")));

  trackPterrHList_.push_back(iBook.book1D("trackPterrIsolated",
                                          "Isolated Track Pterr",
                                          TrackPterrHistoPar.getParameter<int32_t>("Xbins"),
                                          TrackPterrHistoPar.getParameter<double>("Xmin"),
                                          TrackPterrHistoPar.getParameter<double>("Xmax")));
  trackPterrHList_.push_back(iBook.book1D("trackPterrNonIsolated",
                                          "NonIsolated Track Pterr",
                                          TrackPterrHistoPar.getParameter<int32_t>("Xbins"),
                                          TrackPterrHistoPar.getParameter<double>("Xmin"),
                                          TrackPterrHistoPar.getParameter<double>("Xmax")));
  trackPterrHList_.push_back(iBook.book1D("trackPterrUL",
                                          "Underlying Track Pterr",
                                          TrackPterrHistoPar.getParameter<int32_t>("Xbins"),
                                          TrackPterrHistoPar.getParameter<double>("Xmin"),
                                          TrackPterrHistoPar.getParameter<double>("Xmax")));
  trackPterrHList_.push_back(iBook.book1D("trackPterrALL",
                                          "All Track Pterr",
                                          TrackPterrHistoPar.getParameter<int32_t>("Xbins"),
                                          TrackPterrHistoPar.getParameter<double>("Xmin"),
                                          TrackPterrHistoPar.getParameter<double>("Xmax")));

  trackqOverpHList_.push_back(iBook.book1D("trackqOverpIsolated",
                                           "Isolated Track qOverp",
                                           TrackqOverpHistoPar.getParameter<int32_t>("Xbins"),
                                           TrackqOverpHistoPar.getParameter<double>("Xmin"),
                                           TrackqOverpHistoPar.getParameter<double>("Xmax")));
  trackqOverpHList_.push_back(iBook.book1D("trackqOverpNonIsolated",
                                           "NonIsolated Track qOverp",
                                           TrackqOverpHistoPar.getParameter<int32_t>("Xbins"),
                                           TrackqOverpHistoPar.getParameter<double>("Xmin"),
                                           TrackqOverpHistoPar.getParameter<double>("Xmax")));
  trackqOverpHList_.push_back(iBook.book1D("trackqOverpUL",
                                           "Underlying Track qOverp",
                                           TrackqOverpHistoPar.getParameter<int32_t>("Xbins"),
                                           TrackqOverpHistoPar.getParameter<double>("Xmin"),
                                           TrackqOverpHistoPar.getParameter<double>("Xmax")));
  trackqOverpHList_.push_back(iBook.book1D("trackqOverpALL",
                                           "All Track qOverp",
                                           TrackqOverpHistoPar.getParameter<int32_t>("Xbins"),
                                           TrackqOverpHistoPar.getParameter<double>("Xmin"),
                                           TrackqOverpHistoPar.getParameter<double>("Xmax")));

  trackdzHList_.push_back(iBook.book1D("trackdzIsolated",
                                       "Isolated Track dz",
                                       TrackdzHistoPar.getParameter<int32_t>("Xbins"),
                                       TrackdzHistoPar.getParameter<double>("Xmin"),
                                       TrackdzHistoPar.getParameter<double>("Xmax")));
  trackdzHList_.push_back(iBook.book1D("trackdzNonIsolated",
                                       "NonIsolated Track dz",
                                       TrackdzHistoPar.getParameter<int32_t>("Xbins"),
                                       TrackdzHistoPar.getParameter<double>("Xmin"),
                                       TrackdzHistoPar.getParameter<double>("Xmax")));
  trackdzHList_.push_back(iBook.book1D("trackdzUL",
                                       "Underlying Track dz",
                                       TrackdzHistoPar.getParameter<int32_t>("Xbins"),
                                       TrackdzHistoPar.getParameter<double>("Xmin"),
                                       TrackdzHistoPar.getParameter<double>("Xmax")));
  trackdzHList_.push_back(iBook.book1D("trackdzALL",
                                       "All Track dz",
                                       TrackdzHistoPar.getParameter<int32_t>("Xbins"),
                                       TrackdzHistoPar.getParameter<double>("Xmin"),
                                       TrackdzHistoPar.getParameter<double>("Xmax")));

  trackChi2bynDOFHList_.push_back(iBook.book1D("trackChi2bynDOFIsolated",
                                               "Isolated Track Chi2bynDOF",
                                               TrackChi2bynDOFHistoPar.getParameter<int32_t>("Xbins"),
                                               TrackChi2bynDOFHistoPar.getParameter<double>("Xmin"),
                                               TrackChi2bynDOFHistoPar.getParameter<double>("Xmax")));
  trackChi2bynDOFHList_.push_back(iBook.book1D("trackChi2bynDOFNonIsolated",
                                               "NonIsolated Track Chi2bynDOF",
                                               TrackChi2bynDOFHistoPar.getParameter<int32_t>("Xbins"),
                                               TrackChi2bynDOFHistoPar.getParameter<double>("Xmin"),
                                               TrackChi2bynDOFHistoPar.getParameter<double>("Xmax")));
  trackChi2bynDOFHList_.push_back(iBook.book1D("trackChi2bynDOFUL",
                                               "Underlying Track Chi2bynDOF",
                                               TrackChi2bynDOFHistoPar.getParameter<int32_t>("Xbins"),
                                               TrackChi2bynDOFHistoPar.getParameter<double>("Xmin"),
                                               TrackChi2bynDOFHistoPar.getParameter<double>("Xmax")));
  trackChi2bynDOFHList_.push_back(iBook.book1D("trackChi2bynDOFAll",
                                               "All Track Chi2bynDOF",
                                               TrackChi2bynDOFHistoPar.getParameter<int32_t>("Xbins"),
                                               TrackChi2bynDOFHistoPar.getParameter<double>("Xmin"),
                                               TrackChi2bynDOFHistoPar.getParameter<double>("Xmax")));

  nTracksHList_.push_back(iBook.book1D("nTracksIsolated",
                                       "Isolated Track nTracks",
                                       nTracksHistoPar.getParameter<int32_t>("Xbins"),
                                       nTracksHistoPar.getParameter<double>("Xmin"),
                                       nTracksHistoPar.getParameter<double>("Xmax")));
  nTracksHList_.push_back(iBook.book1D("nTracksNonIsolated",
                                       "NonIsolated Track nTracks",
                                       nTracksHistoPar.getParameter<int32_t>("Xbins"),
                                       nTracksHistoPar.getParameter<double>("Xmin"),
                                       nTracksHistoPar.getParameter<double>("Xmax")));
  nTracksHList_.push_back(iBook.book1D("nTracksUL",
                                       "Underlying Track nTracks",
                                       nTracksHistoPar.getParameter<int32_t>("Xbins"),
                                       nTracksHistoPar.getParameter<double>("Xmin"),
                                       nTracksHistoPar.getParameter<double>("Xmax")));
  nTracksHList_.push_back(iBook.book1D("nTracksAll",
                                       "All Track nTracks",
                                       nTracksHistoPar.getParameter<int32_t>("Xbins"),
                                       nTracksHistoPar.getParameter<double>("Xmin"),
                                       nTracksHistoPar.getParameter<double>("Xmax")));

  hcounterH_ = iBook.book1D("hcounter", "hcounter", 7, -0.5, 6.5);
  dphiH_ = iBook.book1D("dphi", "dphi", 100, 0, 7);
  drH_ = iBook.book1D("dr", "dr", 100, 0, 6);
}
void TrackTypeMonitor::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) {
  // read the beam spot
  edm::Handle<reco::BeamSpot> beamSpot;
  iEvent.getByToken(bsToken_, beamSpot);

  std::vector<const reco::Track*> isoTrkList;

  // muons
  edm::Handle<reco::MuonCollection> muonColl;
  iEvent.getByToken(muonToken_, muonColl);

  if (muonColl.isValid()) {
    for (auto const& muo : *muonColl) {
      if (muo.isGlobalMuon() && muo.isPFMuon() && std::abs(muo.eta()) <= 2.1 && muo.pt() > 1) {
        reco::TrackRef gtrkref = muo.globalTrack();
        if (!gtrkref.isNonnull())
          continue;
        const reco::Track* gtk = &(*gtrkref);
        double chi2 = gtk->chi2();
        double ndof = gtk->ndof();
        double chbyndof = (ndof > 0) ? chi2 / ndof : 0;

        const reco::HitPattern& hitp = gtk->hitPattern();
        int nPixelHits = hitp.numberOfValidPixelHits();
        int nStripHits = hitp.numberOfValidStripHits();

        reco::TrackRef itrkref = muo.innerTrack();  // tracker segment only
        if (!itrkref.isNonnull())
          continue;
        const reco::Track* tk = &(*itrkref);
        double trkd0 = tk->d0();
        double trkdz = tk->dz();
        if (beamSpot.isValid()) {
          trkd0 = -(tk->dxy(beamSpot->position()));
          trkdz = tk->dz(beamSpot->position());
        }

        // Hits/section in the muon chamber
        int nChambers = muo.numberOfChambers();
        int nMatches = muo.numberOfMatches();
        int nMatchedStations = muo.numberOfMatchedStations();

        // PF Isolation
        const reco::MuonPFIsolation& pfIso04 = muo.pfIsolationR04();
        double absiso = pfIso04.sumChargedParticlePt +
                        std::max(0.0, pfIso04.sumNeutralHadronEt + pfIso04.sumPhotonEt - 0.5 * pfIso04.sumPUPt);
        if (chbyndof < 10 && std::abs(trkd0) < 0.02 && std::abs(trkdz) < 20 && nPixelHits > 1 && nStripHits > 8 &&
            nChambers > 2 && nMatches > 2 && nMatchedStations > 2 && absiso / muo.pt() < 0.3) {
          isoTrkList.push_back(gtk);
          fillHistograms(*gtk, 0);
        }
      }
    }
  }

  // electrons
  edm::Handle<reco::GsfElectronCollection> electronColl;
  iEvent.getByToken(electronToken_, electronColl);

  if (electronColl.isValid()) {
    for (auto const& ele : *electronColl) {
      if (!ele.ecalDriven())
        continue;
      if (ele.pt() < 5)
        continue;
      hcounterH_->Fill(0);

      double hOverE = ele.hadronicOverEm();
      double sigmaee = ele.sigmaIetaIeta();
      double deltaPhiIn = ele.deltaPhiSuperClusterTrackAtVtx();
      double deltaEtaIn = ele.deltaEtaSuperClusterTrackAtVtx();

      if (ele.isEB()) {
        if (std::abs(deltaPhiIn) >= .15 && std::abs(deltaEtaIn) >= .007 && hOverE >= .12 && sigmaee >= .01)
          continue;
      } else if (ele.isEE()) {
        if (std::abs(deltaPhiIn) >= .10 && std::abs(deltaEtaIn) >= .009 && hOverE >= .10 && sigmaee >= .03)
          continue;
      }
      hcounterH_->Fill(1);

      reco::GsfTrackRef gsftrk = ele.gsfTrack();
      if (!gsftrk.isNonnull())
        continue;
      const reco::GsfTrack* trk = &(*gsftrk);
      double trkd0 = trk->d0();
      double trkdz = trk->dz();
      if (beamSpot.isValid()) {
        trkd0 = -(trk->dxy(beamSpot->position()));
        trkdz = trk->dz(beamSpot->position());
      }
      double chi2 = trk->chi2();
      double ndof = trk->ndof();
      double chbyndof = (ndof > 0) ? chi2 / ndof : 0;
      if (chbyndof >= 10)
        continue;
      hcounterH_->Fill(2);

      if (std::abs(trkd0) >= 0.02 || std::abs(trkdz) >= 20)
        continue;
      hcounterH_->Fill(3);

      const reco::HitPattern& hitp = trk->hitPattern();
      int nPixelHits = hitp.numberOfValidPixelHits();
      if (nPixelHits < 1)
        continue;
      hcounterH_->Fill(4);

      int nStripHits = hitp.numberOfValidStripHits();
      if (nStripHits < 8)
        continue;
      hcounterH_->Fill(5);

      reco::GsfElectron::PflowIsolationVariables pfIso = ele.pfIsolationVariables();
      float absiso =
          pfIso.sumChargedHadronPt + std::max(0.0, pfIso.sumNeutralHadronEt + pfIso.sumPhotonEt - 0.5 * pfIso.sumPUPt);
      float eiso = absiso / ele.pt();
      if (eiso > 0.2)
        continue;
      hcounterH_->Fill(6);

      isoTrkList.push_back(trk);
      fillHistograms(*trk, 0);
    }
  }
  nTracksHList_.at(0)->Fill(isoTrkList.size());

  // Read track collection
  edm::Handle<reco::TrackCollection> tracks;
  iEvent.getByToken(trackToken_, tracks);

  // Read vertex collection
  edm::Handle<reco::VertexCollection> vertexColl;
  iEvent.getByToken(vertexToken_, vertexColl);

  if (tracks.isValid()) {
    std::vector<const reco::Track*> nisoTrkList;
    const reco::Vertex& vit = vertexColl->front();  // Highest sumPt vertex
    reco::Track::TrackQuality quality = reco::Track::qualityByName(trackQuality_);

    int nmatch = 0;
    int nTracks = 0, nMBTracks = 0;
    for (auto const& track : *tracks) {
      if (!track.quality(quality))
        continue;
      ++nTracks;
      fillHistograms(track, 3);

      // now classify primary and underlying tracks using vertex information
      double dxy = track.dxy(vit.position());
      double dz = track.dz(vit.position());
      if (std::abs(dxy) < 0.02 && std::abs(dz) < 20) {  // primary tracks
                                                        // remove tracks associated to the identified particles
        double drmin = 999;
        for (auto const& tk : isoTrkList) {
          double dr = deltaR(track.eta(), track.phi(), tk->eta(), tk->phi());
          if (dr < drmin)
            drmin = dr;
        }
        if (drmin < 0.01) {
          if (verbose_) {
            edm::LogInfo("TrackTypeMonitor") << " Match: " << ++nmatch << " drmin = " << drmin;
            edm::LogInfo("TrackTypeMonitor")
                << " Track Pt: " << track.pt() << " Eta: " << track.eta() << " Phi: " << track.phi();
            for (auto const& isotk : isoTrkList) {
              edm::LogInfo("TrackTypeMonitor")
                  << " Lepton Pt: " << isotk->pt() << " Eta: " << isotk->eta() << " Phi: " << isotk->phi();
            }
          }
          continue;
        }

        fillHistograms(track, 1);
        nisoTrkList.push_back(&track);
      } else {
        ++nMBTracks;
        fillHistograms(track, 2);  //non-primary tracks
      }
    }
    nTracksHList_.at(1)->Fill(nisoTrkList.size());
    nTracksHList_.at(2)->Fill(nMBTracks);
    nTracksHList_.at(3)->Fill(nTracks);

    std::sort(nisoTrkList.begin(), nisoTrkList.end(), PtComparator<reco::Track>());
    const reco::Track* isoTrk = isoTrkList.at(0);
    for (auto const& obj : nisoTrkList) {
      if (obj->pt() > 5) {
        double dphi = deltaPhi(isoTrk->phi(), obj->phi());
        dphiH_->Fill(std::abs(dphi));

        double dr = deltaR(isoTrk->eta(), isoTrk->phi(), obj->eta(), obj->phi());
        drH_->Fill(dr);
      }
    }
  }
}
void TrackTypeMonitor::fillHistograms(const reco::Track& track, int indx) {
  if (indx >= 0 && indx < static_cast<int>(trackEtaHList_.size()))
    trackEtaHList_.at(indx)->Fill(track.eta());

  if (indx >= 0 && indx < static_cast<int>(trackPhiHList_.size()))
    trackPhiHList_.at(indx)->Fill(track.phi());

  if (indx >= 0 && indx < static_cast<int>(trackPHList_.size()))
    trackPHList_.at(indx)->Fill(track.p());

  if (indx >= 0 && indx < static_cast<int>(trackPtHList_.size()))
    trackPtHList_.at(indx)->Fill(track.pt());

  if (indx >= 0 || indx < static_cast<int>(trackPterrHList_.size()))
    trackPterrHList_.at(indx)->Fill(track.ptError());

  if (indx >= 0 || indx < static_cast<int>(trackqOverpHList_.size()))
    trackqOverpHList_.at(indx)->Fill(track.qoverp());

  if (indx >= 0 || indx < static_cast<int>(trackqOverpHList_.size()))
    trackdzHList_.at(indx)->Fill(track.dz());

  double chi2 = track.chi2();
  double ndof = track.ndof();
  double chbyndof = (ndof > 0) ? chi2 / ndof : 0;
  trackChi2bynDOFHList_.at(indx)->Fill(chbyndof);
}
// Define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TrackTypeMonitor);
