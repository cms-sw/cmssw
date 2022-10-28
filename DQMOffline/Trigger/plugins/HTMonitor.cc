#include <string>
#include <vector>

#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMOffline/Trigger/plugins/TriggerDQMBase.h"
#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/JetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

class HTMonitor : public DQMEDAnalyzer, public TriggerDQMBase {
public:
  typedef dqm::reco::MonitorElement MonitorElement;
  typedef dqm::reco::DQMStore DQMStore;

  HTMonitor(const edm::ParameterSet&);
  ~HTMonitor() throw() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) override;

private:
  const std::string folderName_;

  const bool requireValidHLTPaths_;
  bool hltPathsAreValid_;
  const bool enableFullMonitoring_;

  edm::InputTag metInputTag_;
  edm::InputTag jetInputTag_;
  edm::InputTag eleInputTag_;
  edm::InputTag muoInputTag_;
  edm::InputTag vtxInputTag_;

  edm::EDGetTokenT<reco::PFMETCollection> metToken_;
  edm::EDGetTokenT<reco::JetView> jetToken_;
  edm::EDGetTokenT<reco::GsfElectronCollection> eleToken_;
  edm::EDGetTokenT<reco::MuonCollection> muoToken_;
  edm::EDGetTokenT<reco::VertexCollection> vtxToken_;

  std::vector<double> ht_variable_binning_;
  std::vector<double> jetptBinning_;
  MEbinning ht_binning_;
  MEbinning ls_binning_;

  ObjME qME_variableBinning_;
  ObjME htVsLS_;
  ObjME METPhiME_;
  ObjME jetpt1ME_;
  ObjME jetpt2ME_;
  ObjME phij1ME_;
  ObjME phij2ME_;
  ObjME etaj1ME_;
  ObjME etaj2ME_;
  ObjME nJetsME_;    //number of jets passing jet selection (Pt>0)
  ObjME nJetsHTME_;  //number of jets passing jet selection HT (Pt>30 & eta<2.5)

  std::unique_ptr<GenericTriggerEventFlag> num_genTriggerEventFlag_;
  std::unique_ptr<GenericTriggerEventFlag> den_genTriggerEventFlag_;

  StringCutObjectSelector<reco::MET, true> metSelection_;
  StringCutObjectSelector<reco::Jet, true> jetSelection_;
  StringCutObjectSelector<reco::GsfElectron, true> eleSelection_;
  StringCutObjectSelector<reco::Muon, true> muoSelection_;
  StringCutObjectSelector<reco::Jet, true> jetSelection_HT_;
  unsigned njets_;
  unsigned nelectrons_;
  unsigned nmuons_;
  double dEtaCut_;

  static constexpr double MAXedge_PHI = 3.2;
  static constexpr int Nbin_PHI = 64;
  static constexpr MEbinning phi_binning_{Nbin_PHI, -MAXedge_PHI, MAXedge_PHI};
  // Define Eta Bining
  static constexpr double MAX_ETA = 5.0;
  static constexpr int N_ETA = 20;
  static constexpr MEbinning eta_binning{N_ETA, -MAX_ETA, MAX_ETA};
  //Define nJets Binning HT selection Pt>30 && eta<2.4
  static constexpr int MIN_NJETS_HT = 0;
  static constexpr int MAX_NJETS_HT = 30;
  static constexpr int N_BIN_NJETS_HT = 30;
  static constexpr MEbinning nJets_HT_binning{N_BIN_NJETS_HT, MIN_NJETS_HT, MAX_NJETS_HT};
  //Define nJets Binning general selection Pt>0
  static constexpr int MIN_NJETS = 0;
  static constexpr int MAX_NJETS = 200;
  static constexpr int N_BIN_NJETS = 200;
  static constexpr MEbinning nJets_binning{N_BIN_NJETS, MIN_NJETS, MAX_NJETS};

  std::string quantity_;
  bool warningWasPrinted_;
};

HTMonitor::HTMonitor(const edm::ParameterSet& iConfig)
    : folderName_(iConfig.getParameter<std::string>("FolderName")),
      requireValidHLTPaths_(iConfig.getParameter<bool>("requireValidHLTPaths")),
      hltPathsAreValid_(false),
      enableFullMonitoring_(iConfig.getParameter<bool>("enableFullMonitoring")),
      metInputTag_(iConfig.getParameter<edm::InputTag>("met")),
      jetInputTag_(iConfig.getParameter<edm::InputTag>("jets")),
      eleInputTag_(iConfig.getParameter<edm::InputTag>("electrons")),
      muoInputTag_(iConfig.getParameter<edm::InputTag>("muons")),
      vtxInputTag_(iConfig.getParameter<edm::InputTag>("vertices")),
      metToken_(consumes<reco::PFMETCollection>(metInputTag_)),
      jetToken_(mayConsume<reco::JetView>(jetInputTag_)),
      eleToken_(mayConsume<reco::GsfElectronCollection>(eleInputTag_)),
      muoToken_(mayConsume<reco::MuonCollection>(muoInputTag_)),
      vtxToken_(mayConsume<reco::VertexCollection>(vtxInputTag_)),
      ht_variable_binning_(
          iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("htBinning")),
      jetptBinning_(
          iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("jetptBinning")),
      ht_binning_(
          getHistoPSet(iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("htPSet"))),
      ls_binning_(
          getHistoPSet(iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("lsPSet"))),
      num_genTriggerEventFlag_(new GenericTriggerEventFlag(
          iConfig.getParameter<edm::ParameterSet>("numGenericTriggerEventPSet"), consumesCollector(), *this)),
      den_genTriggerEventFlag_(new GenericTriggerEventFlag(
          iConfig.getParameter<edm::ParameterSet>("denGenericTriggerEventPSet"), consumesCollector(), *this)),
      metSelection_(iConfig.getParameter<std::string>("metSelection")),
      jetSelection_(iConfig.getParameter<std::string>("jetSelection")),
      eleSelection_(iConfig.getParameter<std::string>("eleSelection")),
      muoSelection_(iConfig.getParameter<std::string>("muoSelection")),
      jetSelection_HT_(iConfig.getParameter<std::string>("jetSelection_HT")),
      njets_(iConfig.getParameter<unsigned>("njets")),
      nelectrons_(iConfig.getParameter<unsigned>("nelectrons")),
      nmuons_(iConfig.getParameter<unsigned>("nmuons")),
      dEtaCut_(iConfig.getParameter<double>("dEtaCut")),
      quantity_(iConfig.getParameter<std::string>("quantity")),
      warningWasPrinted_(false) {}

HTMonitor::~HTMonitor() throw() {
  if (num_genTriggerEventFlag_) {
    num_genTriggerEventFlag_.reset();
  }
  if (den_genTriggerEventFlag_) {
    den_genTriggerEventFlag_.reset();
  }
}

void HTMonitor::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& iRun, edm::EventSetup const& iSetup) {
  // Initialize the GenericTriggerEventFlag
  if (num_genTriggerEventFlag_ && num_genTriggerEventFlag_->on()) {
    num_genTriggerEventFlag_->initRun(iRun, iSetup);
  }
  if (den_genTriggerEventFlag_ && den_genTriggerEventFlag_->on()) {
    den_genTriggerEventFlag_->initRun(iRun, iSetup);
  }

  // check if every HLT path specified in numerator and denominator has a valid match in the HLT Menu
  hltPathsAreValid_ = (num_genTriggerEventFlag_ && den_genTriggerEventFlag_ && num_genTriggerEventFlag_->on() &&
                       den_genTriggerEventFlag_->on() && num_genTriggerEventFlag_->allHLTPathsAreValid() &&
                       den_genTriggerEventFlag_->allHLTPathsAreValid());

  // if valid HLT paths are required,
  // create DQM outputs only if all paths are valid
  if (requireValidHLTPaths_ and (not hltPathsAreValid_)) {
    return;
  }

  std::string histname, histtitle;

  std::string currentFolder = folderName_;
  ibooker.setCurrentFolder(currentFolder);
  if (quantity_ == "HT") {
    histname = "ht_variable";
    histtitle = "HT";
    bookME(ibooker, qME_variableBinning_, histname, histtitle, ht_variable_binning_);
    setMETitle(qME_variableBinning_, "HT [GeV]", "events / [GeV]");

    histname = "htVsLS";
    histtitle = "HT vs LS";
    bookME(ibooker,
           htVsLS_,
           histname,
           histtitle,
           ls_binning_.nbins,
           ls_binning_.xmin,
           ls_binning_.xmax,
           ht_binning_.xmin,
           ht_binning_.xmax);
    setMETitle(htVsLS_, "LS", "HT [GeV]");

    if (!enableFullMonitoring_) {
      return;
    }

    histname = "METPhi";
    histtitle = "METPhi";
    bookME(ibooker, METPhiME_, histname, histtitle, phi_binning_.nbins, phi_binning_.xmin, phi_binning_.xmax);
    setMETitle(METPhiME_, "MET phi", "events / 0.1 rad");

    histname = "jetPt1";
    histtitle = "leading Jet Pt";
    bookME(ibooker, jetpt1ME_, histname, histtitle, jetptBinning_);
    setMETitle(jetpt1ME_, "Pt_1 [GeV]", "events");

    histname = "jetPt2";
    histtitle = "second leading Jet Pt";
    bookME(ibooker, jetpt2ME_, histname, histtitle, jetptBinning_);
    setMETitle(jetpt2ME_, "Pt_2 [GeV]", "events");

    histname = "jetEta1";
    histtitle = "leading Jet eta";
    bookME(ibooker, etaj1ME_, histname, histtitle, eta_binning.nbins, eta_binning.xmin, eta_binning.xmax);
    setMETitle(etaj1ME_, "Jet_1 #eta", "events");

    histname = "jetEta2";
    histtitle = "subleading Jet eta";
    bookME(ibooker, etaj2ME_, histname, histtitle, eta_binning.nbins, eta_binning.xmin, eta_binning.xmax);
    setMETitle(etaj2ME_, "Jet_2 #eta", "events");

    histname = "jetPhi1";
    histtitle = "leading Jet phi";
    bookME(ibooker, phij1ME_, histname, histtitle, phi_binning_.nbins, phi_binning_.xmin, phi_binning_.xmax);
    setMETitle(phij1ME_, "Jet_1 #phi", "events / 0.1 rad");

    histname = "jetPhi2";
    histtitle = "subleading Jet phi";
    bookME(ibooker, phij2ME_, histname, histtitle, phi_binning_.nbins, phi_binning_.xmin, phi_binning_.xmax);
    setMETitle(phij2ME_, "Jet_2 #phi", "events / 0.1 rad");

    histname = "nJets";
    histtitle = "number of Jets";
    bookME(ibooker, nJetsME_, histname, histtitle, nJets_binning.nbins, nJets_binning.xmin, nJets_binning.xmax);
    setMETitle(nJetsME_, "number of Jets", "events");

    histname = "nJetsHT";
    histtitle = "number of Jets HT";
    bookME(
        ibooker, nJetsHTME_, histname, histtitle, nJets_HT_binning.nbins, nJets_HT_binning.xmin, nJets_HT_binning.xmax);
    setMETitle(nJetsHTME_, "number of Jets HT", "events");
  }  //end if --  HT quantity
  else if (quantity_ == "Mjj") {
    histname = "mjj_variable";
    histtitle = "Mjj";
    bookME(ibooker, qME_variableBinning_, histname, histtitle, ht_variable_binning_);
    setMETitle(qME_variableBinning_, "Mjj [GeV]", "events / [GeV]");
  }  //end if -- Mjj quantity
  else if (quantity_ == "softdrop") {
    histname = "softdrop_variable";
    histtitle = "softdropmass";
    bookME(ibooker, qME_variableBinning_, histname, histtitle, ht_variable_binning_);
    setMETitle(qME_variableBinning_, "leading jet softdropmass [GeV]", "events / [GeV]");
  }  //end if -- softdrop quantity
}

void HTMonitor::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) {
  // if valid HLT paths are required,
  // analyze event only if all paths are valid
  if (requireValidHLTPaths_ and (not hltPathsAreValid_)) {
    return;
  }

  // Filter out events if Trigger Filtering is requested
  if (den_genTriggerEventFlag_->on() && !den_genTriggerEventFlag_->accept(iEvent, iSetup))
    return;

  edm::Handle<reco::PFMETCollection> metHandle;
  iEvent.getByToken(metToken_, metHandle);
  if (not metHandle.isValid()) {
    if (not warningWasPrinted_) {
      edm::LogWarning("HTMonitor") << "skipping events because the collection " << metInputTag_.label().c_str()
                                   << " is not available";
      warningWasPrinted_ = true;
    }
    return;
  }
  reco::PFMET pfmet = metHandle->front();
  if (!metSelection_(pfmet))
    return;

  edm::Handle<reco::JetView> jetHandle;  //add a configurable jet collection & jet pt selection
  iEvent.getByToken(jetToken_, jetHandle);
  if (!jetHandle.isValid()) {
    if (not warningWasPrinted_) {
      edm::LogWarning("HTMonitor") << "skipping events because the collection " << jetInputTag_.label().c_str()
                                   << " is not available";
      warningWasPrinted_ = true;
    }
    return;
  }
  std::vector<reco::Jet> jets;
  if (jetHandle->size() < njets_)
    return;
  for (auto const& j : *jetHandle) {
    if (jetSelection_(j)) {
      jets.push_back(j);
    }
  }

  if (jets.size() < njets_)
    return;

  const float metphi = pfmet.phi();
  const int nJetsSel = jets.size();  //jetSelection: PT>0
  const float Pt_J1 = !jets.empty() ? jets[0].pt() : -10.0;
  const float Pt_J2 = jets.size() >= 2 ? jets[1].pt() : -10.0;
  const float Phi_J1 = !jets.empty() ? jets[0].phi() : -10.0;
  const float Phi_J2 = jets.size() >= 2 ? jets[1].phi() : -10.0;
  const float Eta_J1 = !jets.empty() ? jets[0].p4().eta() : -10.0;
  const float Eta_J2 = jets.size() >= 2 ? jets[1].p4().eta() : -10.0;

  std::vector<reco::GsfElectron> electrons;
  edm::Handle<reco::GsfElectronCollection> eleHandle;
  iEvent.getByToken(eleToken_, eleHandle);
  if (eleHandle.isValid()) {
    if (eleHandle->size() < nelectrons_)
      return;
    for (auto const& e : *eleHandle) {
      if (eleSelection_(e))
        electrons.push_back(e);
    }
    if (electrons.size() < nelectrons_)
      return;
  } else {
    if (not warningWasPrinted_) {
      if (eleInputTag_.label().empty())
        edm::LogWarning("HTMonitor") << "GsfElectronCollection not set";
      else
        edm::LogWarning("HTMonitor") << "skipping events because the collection " << eleInputTag_.label().c_str()
                                     << " is not available";

      warningWasPrinted_ = true;
    }
    if (!eleInputTag_.label().empty())
      return;
  }

  reco::Vertex vtx;
  edm::Handle<reco::VertexCollection> vtxHandle;
  iEvent.getByToken(vtxToken_, vtxHandle);
  if (vtxHandle.isValid()) {
    for (auto const& v : *vtxHandle) {
      bool isFake = v.isFake();

      if (!isFake) {
        vtx = v;
        break;
      }
    }
  } else {
    if (not warningWasPrinted_) {
      if (vtxInputTag_.label().empty())
        edm::LogWarning("HTMonitor") << "VertexCollection not set";
      else
        edm::LogWarning("HTMonitor") << "skipping events because the collection " << vtxInputTag_.label().c_str()
                                     << " is not available";

      warningWasPrinted_ = true;
    }
    if (!vtxInputTag_.label().empty())
      return;
  }

  std::vector<reco::Muon> muons;
  edm::Handle<reco::MuonCollection> muoHandle;
  iEvent.getByToken(muoToken_, muoHandle);
  if (muoHandle.isValid()) {
    if (muoHandle->size() < nmuons_)
      return;
    for (auto const& m : *muoHandle) {
      if (muoSelection_(m) && m.isGlobalMuon() && m.isPFMuon() && m.globalTrack()->normalizedChi2() < 10. &&
          m.globalTrack()->hitPattern().numberOfValidMuonHits() > 0 && m.numberOfMatchedStations() > 1 &&
          fabs(m.muonBestTrack()->dxy(vtx.position())) < 0.2 && fabs(m.muonBestTrack()->dz(vtx.position())) < 0.5 &&
          m.innerTrack()->hitPattern().numberOfValidPixelHits() > 0 &&
          m.innerTrack()->hitPattern().trackerLayersWithMeasurement() > 5)
        muons.push_back(m);
    }
    if (muons.size() < nmuons_)
      return;
  } else {
    if (not warningWasPrinted_) {
      if (muoInputTag_.label().empty())
        edm::LogWarning("HTMonitor") << "MuonCollection not set";
      else
        edm::LogWarning("HTMonitor") << "skipping events because the collection " << muoInputTag_.label().c_str()
                                     << " is not available";

      warningWasPrinted_ = true;
    }
    if (!muoInputTag_.label().empty())
      return;
  }

  // fill histograms
  if (quantity_ == "HT") {
    float ht = 0.0;
    int nJets_HT = 0;
    for (auto const& j : *jetHandle) {
      if (jetSelection_HT_(j)) {
        ht += j.pt();
        nJets_HT = nJets_HT + 1;  //===== jetSelection HT: Pt>30 & eta<2.5
      }
    }

    // filling histograms (denominator)
    qME_variableBinning_.denominator->Fill(ht);

    int ls = iEvent.id().luminosityBlock();
    htVsLS_.denominator->Fill(ls, ht);

    if (enableFullMonitoring_) {  //===check the flag

      METPhiME_.denominator->Fill(metphi);
      jetpt1ME_.denominator->Fill(Pt_J1);
      jetpt2ME_.denominator->Fill(Pt_J2);
      phij1ME_.denominator->Fill(Phi_J1);
      phij2ME_.denominator->Fill(Phi_J2);
      etaj1ME_.denominator->Fill(Eta_J1);
      etaj2ME_.denominator->Fill(Eta_J2);
      nJetsME_.denominator->Fill(nJetsSel);
      nJetsHTME_.denominator->Fill(nJets_HT);
    }

    // applying selection for numerator
    if (num_genTriggerEventFlag_->on() && !num_genTriggerEventFlag_->accept(iEvent, iSetup))
      return;

    // filling histograms (num_genTriggerEventFlag_)
    qME_variableBinning_.numerator->Fill(ht);

    htVsLS_.numerator->Fill(ls, ht);

    if (enableFullMonitoring_) {  //===check the flag
      METPhiME_.numerator->Fill(metphi);
      jetpt1ME_.numerator->Fill(Pt_J1);
      jetpt2ME_.numerator->Fill(Pt_J2);
      phij1ME_.numerator->Fill(Phi_J1);
      phij2ME_.numerator->Fill(Phi_J2);
      etaj1ME_.numerator->Fill(Eta_J1);
      etaj2ME_.numerator->Fill(Eta_J2);
      nJetsME_.numerator->Fill(nJetsSel);
      nJetsHTME_.numerator->Fill(nJets_HT);
    }
  }  //end if --  HT
  else if (quantity_ == "Mjj") {
    if (jets.size() < 2)
      return;

    // deltaEta cut
    if (fabs(jets[0].p4().Eta() - jets[1].p4().Eta()) >= dEtaCut_)
      return;
    float mjj = (jets[0].p4() + jets[1].p4()).M();

    qME_variableBinning_.denominator->Fill(mjj);

    // applying selection for numerator
    if (num_genTriggerEventFlag_->on() && !num_genTriggerEventFlag_->accept(iEvent, iSetup))
      return;

    qME_variableBinning_.numerator->Fill(mjj);
  }  // end if -- Mjj
  else if (quantity_ == "softdrop") {
    if (jets.size() < 2)
      return;

    // deltaEta cut
    if (fabs(jets[0].p4().Eta() - jets[1].p4().Eta()) >= dEtaCut_)
      return;

    float softdrop = jets[0].p4().M();

    qME_variableBinning_.denominator->Fill(softdrop);

    // applying selection for numerator
    if (num_genTriggerEventFlag_->on() && !num_genTriggerEventFlag_->accept(iEvent, iSetup))
      return;

    qME_variableBinning_.numerator->Fill(softdrop);
  }  //end  if -- softdrop
}

void HTMonitor::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("FolderName", "HLT/HT");
  desc.add<bool>("requireValidHLTPaths", true);
  desc.add<bool>("enableFullMonitoring", false);
  desc.add<edm::InputTag>("met", edm::InputTag("pfMet"));
  desc.add<edm::InputTag>("jets", edm::InputTag("ak4PFJetsCHS"));
  desc.add<edm::InputTag>("electrons", edm::InputTag("gedGsfElectrons"));
  desc.add<edm::InputTag>("muons", edm::InputTag("muons"));
  desc.add<edm::InputTag>("vertices", edm::InputTag("offlinePrimaryVertices"));
  desc.add<std::string>("metSelection", "pt > 0");
  desc.add<std::string>("jetSelection", "pt > 0");
  desc.add<std::string>("eleSelection", "pt > 0");
  desc.add<std::string>("muoSelection", "pt > 0");
  desc.add<std::string>("jetSelection_HT", "pt > 30 && eta < 2.5");
  desc.add<unsigned>("njets", 0);
  desc.add<unsigned>("nelectrons", 0);
  desc.add<unsigned>("nmuons", 0);
  desc.add<double>("dEtaCut", 1.3);

  edm::ParameterSetDescription genericTriggerEventPSet;
  GenericTriggerEventFlag::fillPSetDescription(genericTriggerEventPSet);

  desc.add<edm::ParameterSetDescription>("numGenericTriggerEventPSet", genericTriggerEventPSet);
  desc.add<edm::ParameterSetDescription>("denGenericTriggerEventPSet", genericTriggerEventPSet);

  edm::ParameterSetDescription histoPSet;
  edm::ParameterSetDescription htPSet;
  fillHistoPSetDescription(htPSet);
  histoPSet.add<edm::ParameterSetDescription>("htPSet", htPSet);
  std::vector<double> bins = {0.,   20.,  40.,   60.,   80.,   90.,   100.,  110.,  120.,  130.,  140.,
                              150., 160., 170.,  180.,  190.,  200.,  220.,  240.,  260.,  280.,  300.,
                              350., 400., 450.,  500.,  550.,  600.,  650.,  700.,  750.,  800.,  850.,
                              900., 950., 1000., 1050., 1100., 1200., 1300., 1400., 1500., 2000., 2500.};
  histoPSet.add<std::vector<double> >("htBinning", bins);
  std::vector<double> bins_2 = {
      0.,   20.,  40.,  60.,  80.,  90.,  100., 110., 120., 130., 140., 150., 160.,
      170., 180., 190., 200., 220., 240., 260., 280., 300., 350., 400., 450., 1000.};  // Jet pT Binning
  histoPSet.add<std::vector<double> >("jetptBinning", bins_2);
  edm::ParameterSetDescription lsPSet;
  fillHistoLSPSetDescription(lsPSet);
  histoPSet.add<edm::ParameterSetDescription>("lsPSet", lsPSet);

  desc.add<edm::ParameterSetDescription>("histoPSet", histoPSet);

  desc.add<std::string>("quantity", "HT");

  descriptions.add("htMonitoring", desc);
}

DEFINE_FWK_MODULE(HTMonitor);
