#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"
#include "CommonTools/CandUtils/interface/AddFourMomenta.h"

#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include "DataFormats/Candidate/interface/CompositeCandidateFwd.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/BTauReco/interface/ShallowTagInfo.h"
#include "DataFormats/Math/interface/LorentzVector.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMOffline/Trigger/plugins/TriggerDQMBase.h"

#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/Utilities/interface/transform.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "JetMETCorrections/JetCorrector/interface/JetCorrector.h"

#include <string>
#include <vector>
#include <memory>
#include <map>

class ParticleNetJetTagMonitor : public DQMEDAnalyzer, public TriggerDQMBase {
public:
  typedef dqm::reco::MonitorElement MonitorElement;
  typedef dqm::reco::DQMStore DQMStore;

  ParticleNetJetTagMonitor(const edm::ParameterSet&);
  ~ParticleNetJetTagMonitor() throw() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) override;

private:
  // folder for output histograms
  const std::string folderName_;
  // validity of HLT paths required for the monitoring element
  const bool requireValidHLTPaths_;
  const bool requireHLTOfflineJetMatching_;
  bool denHLTPathsAreValid_;
  bool numHLTPathsAreValid_;
  // input vertex collection
  const edm::EDGetTokenT<reco::VertexCollection> vertexToken_;
  // input muon collection
  const edm::EDGetTokenT<reco::MuonCollection> muonToken_;
  // input electron collection and IDs
  const edm::EDGetTokenT<reco::GsfElectronCollection> electronToken_;
  const edm::EDGetTokenT<edm::ValueMap<bool>> tagElectronIDToken_;
  const edm::EDGetTokenT<edm::ValueMap<bool>> vetoElectronIDToken_;
  // input jet collection
  const edm::EDGetTokenT<reco::PFJetCollection> jetToken_;
  // PNET score for offline and online jets
  const edm::EDGetTokenT<reco::JetTagCollection> jetPNETScoreToken_;
  const edm::EDGetTokenT<reco::JetTagCollection> jetPNETScoreHLTToken_;
  // Collection and PNET score for ak4 b-tagging and HT if needed
  const edm::EDGetTokenT<reco::PFJetCollection> jetForHTandBTagToken_;
  const edm::EDGetTokenT<reco::JetTagCollection> jetPNETScoreForHTandBTagToken_;
  // Jet soft drop value map
  const edm::EDGetTokenT<edm::ValueMap<float>> jetSoftDropMassToken_;
  // input MET collection
  const edm::EDGetTokenT<reco::PFMETCollection> metToken_;
  // JEC corrections
  const edm::EDGetTokenT<reco::JetCorrector> jecMCToken_;
  const edm::EDGetTokenT<reco::JetCorrector> jecDataToken_;
  // trigger conditions for numerator and denominator
  std::unique_ptr<GenericTriggerEventFlag> numGenericTriggerEvent_;
  std::unique_ptr<GenericTriggerEventFlag> denGenericTriggerEvent_;
  // Selectors for jets, electrons, muons, and lepton pairs
  StringCutObjectSelector<reco::PFJet, true> jetSelection_;
  StringCutObjectSelector<reco::PFJet, true> jetSelectionForHTandBTag_;
  StringCutObjectSelector<reco::Muon, true> tagMuonSelection_;
  StringCutObjectSelector<reco::GsfElectron, true> tagElectronSelection_;
  StringCutObjectSelector<reco::Muon, true> vetoMuonSelection_;
  StringCutObjectSelector<reco::GsfElectron, true> vetoElectronSelection_;
  StringCutObjectSelector<reco::Vertex, true> vertexSelection_;
  StringCutObjectSelector<reco::CompositeCandidate, true> dileptonSelection_;
  StringCutObjectSelector<reco::PFMET, true> metSelection_;
  // Number of objects used in the event selection
  const int njets_;
  const int nbjets_;
  const int ntagleptons_;
  const int ntagmuons_;
  const int ntagelectrons_;
  const int nvetoleptons_;
  const int nvetomuons_;
  const int nvetoelectrons_;
  const int nemupairs_;
  const unsigned int ntrigobjecttomatch_;
  // delta-R for cleaning and other parameters for the event selection
  const double lepJetDeltaRmin_;
  const double lepJetDeltaRminForHTandBTag_;
  const double hltRecoDeltaRmax_;
  const double maxLeptonDxyCut_;
  const double maxLeptonDzCut_;
  const double minPNETScoreCut_;
  const double minPNETBTagCut_;
  const double minSoftDropMassCut_;
  const double maxSoftDropMassCut_;
  // binning for efficiency histograms (up to two jets in the final state)
  std::vector<double> leptonPtBinning;
  std::vector<double> leptonEtaBinning;
  std::vector<double> diLeptonPtBinning;
  std::vector<double> diLeptonMassBinning;
  std::vector<double> HTBinning;
  std::vector<double> NjetBinning;
  std::vector<double> jet1PtBinning;
  std::vector<double> jet2PtBinning;
  std::vector<double> jet1EtaBinning;
  std::vector<double> jet2EtaBinning;
  std::vector<double> jet1PNETscoreBinning;
  std::vector<double> jet2PNETscoreBinning;
  std::vector<double> jet1PNETscoreTransBinning;
  std::vector<double> jet2PNETscoreTransBinning;
  std::vector<double> jet1PtBinning2d;
  std::vector<double> jet2PtBinning2d;
  std::vector<double> jet1EtaBinning2d;
  std::vector<double> jet2EtaBinning2d;
  std::vector<double> jet1PNETscoreBinning2d;
  std::vector<double> jet2PNETscoreBinning2d;
  std::vector<double> jet1PNETscoreTransBinning2d;
  std::vector<double> jet2PNETscoreTransBinning2d;

  // Selections imposed
  MonitorElement* selectionFlow = nullptr;
  // Efficiencies
  ObjME muon_pt;
  ObjME electron_pt;
  ObjME muon_eta;
  ObjME electron_eta;
  ObjME dilepton_pt;
  ObjME dilepton_mass;
  ObjME njets;
  ObjME nbjets;
  ObjME ht;
  ObjME jet1_pt;
  ObjME jet2_pt;
  ObjME jet1_eta;
  ObjME jet2_eta;
  ObjME jet1_pnetscore;
  ObjME jet2_pnetscore;
  ObjME jet1_pnetscore_trans;
  ObjME jet2_pnetscore_trans;
  ObjME mean_2j_pnetscore;
  ObjME mean_2j_pnetscore_trans;

  ObjME jet1_pt_eta;
  ObjME jet2_pt_eta;
  ObjME jet1_pt_pnetscore1;
  ObjME jet2_pt_pnetscore1;
  ObjME jet1_pt_pnetscore2;
  ObjME jet2_pt_pnetscore2;
  ObjME jet1_pt_pnetscore1_trans;
  ObjME jet2_pt_pnetscore1_trans;
  ObjME jet1_pt_pnetscore2_trans;
  ObjME jet2_pt_pnetscore2_trans;
  ObjME jet1_pt_mean2pnetscore;
  ObjME jet2_pt_mean2pnetscore;
  ObjME jet1_pt_mean2pnetscore_trans;
  ObjME jet2_pt_mean2pnetscore_trans;
};

ParticleNetJetTagMonitor::ParticleNetJetTagMonitor(const edm::ParameterSet& iConfig)
    : folderName_(iConfig.getParameter<std::string>("FolderName")),
      requireValidHLTPaths_(iConfig.getParameter<bool>("requireValidHLTPaths")),
      requireHLTOfflineJetMatching_(iConfig.getParameter<bool>("requireHLTOfflineJetMatching")),
      denHLTPathsAreValid_(false),
      numHLTPathsAreValid_(false),
      vertexToken_(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertices"))),
      muonToken_(consumes<reco::MuonCollection>(iConfig.getParameter<edm::InputTag>("muons"))),
      electronToken_(consumes<reco::GsfElectronCollection>(iConfig.getParameter<edm::InputTag>("electrons"))),
      tagElectronIDToken_(consumes<edm::ValueMap<bool>>(iConfig.getParameter<edm::InputTag>("tagElectronID"))),
      vetoElectronIDToken_(consumes<edm::ValueMap<bool>>(iConfig.getParameter<edm::InputTag>("vetoElectronID"))),
      jetToken_(consumes<reco::PFJetCollection>(iConfig.getParameter<edm::InputTag>("jets"))),
      jetPNETScoreToken_(consumes<reco::JetTagCollection>(iConfig.getParameter<edm::InputTag>("jetPNETScore"))),
      jetPNETScoreHLTToken_(mayConsume<reco::JetTagCollection>(iConfig.getParameter<edm::InputTag>("jetPNETScoreHLT"))),
      jetForHTandBTagToken_(mayConsume<reco::PFJetCollection>(iConfig.getParameter<edm::InputTag>("jetsForHTandBTag"))),
      jetPNETScoreForHTandBTagToken_(
          mayConsume<reco::JetTagCollection>(iConfig.getParameter<edm::InputTag>("jetPNETScoreForHTandBTag"))),
      jetSoftDropMassToken_(mayConsume<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("jetSoftDropMass"))),
      metToken_(consumes<reco::PFMETCollection>(iConfig.getParameter<edm::InputTag>("met"))),
      jecMCToken_(mayConsume<reco::JetCorrector>(iConfig.getParameter<edm::InputTag>("jecForMC"))),
      jecDataToken_(mayConsume<reco::JetCorrector>(iConfig.getParameter<edm::InputTag>("jecForData"))),
      numGenericTriggerEvent_(new GenericTriggerEventFlag(
          iConfig.getParameter<edm::ParameterSet>("numGenericTriggerEvent"), consumesCollector(), *this)),
      denGenericTriggerEvent_(new GenericTriggerEventFlag(
          iConfig.getParameter<edm::ParameterSet>("denGenericTriggerEvent"), consumesCollector(), *this)),
      jetSelection_(iConfig.getParameter<std::string>("jetSelection")),
      jetSelectionForHTandBTag_(iConfig.existsAs<std::string>("jetSelectionForHTandBTag")
                                    ? iConfig.getParameter<std::string>("jetSelectionForHTandBTag")
                                    : ""),
      tagMuonSelection_(iConfig.getParameter<std::string>("tagMuonSelection")),
      tagElectronSelection_(iConfig.getParameter<std::string>("tagElectronSelection")),
      vetoMuonSelection_(iConfig.getParameter<std::string>("vetoMuonSelection")),
      vetoElectronSelection_(iConfig.getParameter<std::string>("vetoElectronSelection")),
      vertexSelection_(iConfig.getParameter<std::string>("vertexSelection")),
      dileptonSelection_(iConfig.getParameter<std::string>("dileptonSelection")),
      metSelection_(iConfig.getParameter<std::string>("metSelection")),
      njets_(iConfig.getParameter<int>("njets")),
      nbjets_(iConfig.getParameter<int>("nbjets")),
      ntagleptons_(iConfig.getParameter<int>("ntagleptons")),
      ntagmuons_(iConfig.getParameter<int>("ntagmuons")),
      ntagelectrons_(iConfig.getParameter<int>("ntagelectrons")),
      nvetoleptons_(iConfig.getParameter<int>("nvetoleptons")),
      nvetomuons_(iConfig.getParameter<int>("nvetomuons")),
      nvetoelectrons_(iConfig.getParameter<int>("nvetoelectrons")),
      nemupairs_(iConfig.getParameter<int>("nemupairs")),
      ntrigobjecttomatch_(iConfig.getParameter<unsigned int>("ntrigobjecttomatch")),
      lepJetDeltaRmin_(iConfig.getParameter<double>("lepJetDeltaRmin")),
      lepJetDeltaRminForHTandBTag_(iConfig.getParameter<double>("lepJetDeltaRminForHTandBTag")),
      hltRecoDeltaRmax_(iConfig.getParameter<double>("hltRecoDeltaRmax")),
      maxLeptonDxyCut_(iConfig.getParameter<double>("maxLeptonDxyCut")),
      maxLeptonDzCut_(iConfig.getParameter<double>("maxLeptonDzCut")),
      minPNETScoreCut_(iConfig.getParameter<double>("minPNETScoreCut")),
      minPNETBTagCut_(iConfig.getParameter<double>("minPNETBTagCut")),
      minSoftDropMassCut_(iConfig.getParameter<double>("minSoftDropMassCut")),
      maxSoftDropMassCut_(iConfig.getParameter<double>("maxSoftDropMassCut")),
      leptonPtBinning(iConfig.getParameter<std::vector<double>>("leptonPtBinning")),
      leptonEtaBinning(iConfig.getParameter<std::vector<double>>("leptonEtaBinning")),
      diLeptonMassBinning(iConfig.getParameter<std::vector<double>>("diLeptonMassBinning")),
      HTBinning(iConfig.getParameter<std::vector<double>>("HTBinning")),
      NjetBinning(iConfig.getParameter<std::vector<double>>("NjetBinning")),
      jet1PtBinning(iConfig.getParameter<std::vector<double>>("jet1PtBinning")),
      jet2PtBinning(iConfig.getParameter<std::vector<double>>("jet2PtBinning")),
      jet1EtaBinning(iConfig.getParameter<std::vector<double>>("jet1EtaBinning")),
      jet2EtaBinning(iConfig.getParameter<std::vector<double>>("jet2EtaBinning")),
      jet1PNETscoreBinning(iConfig.getParameter<std::vector<double>>("jet1PNETscoreBinning")),
      jet2PNETscoreBinning(iConfig.getParameter<std::vector<double>>("jet2PNETscoreBinning")),
      jet1PNETscoreTransBinning(iConfig.getParameter<std::vector<double>>("jet1PNETscoreTransBinning")),
      jet2PNETscoreTransBinning(iConfig.getParameter<std::vector<double>>("jet2PNETscoreTransBinning")),
      jet1PtBinning2d(iConfig.getParameter<std::vector<double>>("jet1PtBinning2d")),
      jet2PtBinning2d(iConfig.getParameter<std::vector<double>>("jet2PtBinning2d")),
      jet1EtaBinning2d(iConfig.getParameter<std::vector<double>>("jet1EtaBinning2d")),
      jet2EtaBinning2d(iConfig.getParameter<std::vector<double>>("jet2EtaBinning2d")),
      jet1PNETscoreBinning2d(iConfig.getParameter<std::vector<double>>("jet1PNETscoreBinning2d")),
      jet2PNETscoreBinning2d(iConfig.getParameter<std::vector<double>>("jet2PNETscoreBinning2d")),
      jet1PNETscoreTransBinning2d(iConfig.getParameter<std::vector<double>>("jet1PNETscoreTransBinning2d")),
      jet2PNETscoreTransBinning2d(iConfig.getParameter<std::vector<double>>("jet2PNETscoreTransBinning2d")) {}

ParticleNetJetTagMonitor::~ParticleNetJetTagMonitor() throw() {
  if (numGenericTriggerEvent_)
    denGenericTriggerEvent_.reset();
  if (numGenericTriggerEvent_)
    denGenericTriggerEvent_.reset();
}

void ParticleNetJetTagMonitor::bookHistograms(DQMStore::IBooker& ibooker,
                                              edm::Run const& iRun,
                                              edm::EventSetup const& iSetup) {
  // Initialize the GenericTriggerEventFlag
  if (denGenericTriggerEvent_ and denGenericTriggerEvent_->on())
    denGenericTriggerEvent_->initRun(iRun, iSetup);
  if (numGenericTriggerEvent_ and numGenericTriggerEvent_->on())
    numGenericTriggerEvent_->initRun(iRun, iSetup);

  denHLTPathsAreValid_ =
      (denGenericTriggerEvent_ && denGenericTriggerEvent_->on() && denGenericTriggerEvent_->allHLTPathsAreValid());
  numHLTPathsAreValid_ =
      (numGenericTriggerEvent_ && numGenericTriggerEvent_->on() && numGenericTriggerEvent_->allHLTPathsAreValid());

  if (requireValidHLTPaths_ && (!denHLTPathsAreValid_))
    return;
  if (requireValidHLTPaths_ && (!numHLTPathsAreValid_))
    return;

  ibooker.setCurrentFolder(folderName_);

  selectionFlow = ibooker.book1D("selectionFlow", "selectionFlow", 17, 0, 17);
  selectionFlow->setBinLabel(1, "all");
  selectionFlow->setBinLabel(2, "denTrigSel");
  selectionFlow->setBinLabel(3, "collecttionSel");
  selectionFlow->setBinLabel(4, "vtxSel");
  selectionFlow->setBinLabel(5, "tagMuSel");
  selectionFlow->setBinLabel(6, "vetoMuSel");
  selectionFlow->setBinLabel(7, "tagEleSel");
  selectionFlow->setBinLabel(8, "vetoEleSel");
  selectionFlow->setBinLabel(9, "tagLepSel");
  selectionFlow->setBinLabel(10, "vetoLepSel");
  selectionFlow->setBinLabel(11, "diLepSel");
  selectionFlow->setBinLabel(12, "btagSel");
  selectionFlow->setBinLabel(13, "METSel");
  selectionFlow->setBinLabel(14, "jetSel");
  selectionFlow->setBinLabel(15, "pnetScoreSel");
  selectionFlow->setBinLabel(16, "trigMatchSel");
  selectionFlow->setBinLabel(17, "numTrigSel");

  if (!NjetBinning.empty()) {
    bookME(ibooker, njets, "njets", "n-jets", NjetBinning);
    setMETitle(njets, "number of jets", "Entries");
    bookME(ibooker, nbjets, "nbjets", "b-jets", NjetBinning);
    setMETitle(nbjets, "number of b-jets", "Entries");
  }
  if (!HTBinning.empty()) {
    bookME(ibooker, ht, "ht", "HT", HTBinning);
    setMETitle(ht, "H_{T}", "Entries");
  }

  if (!leptonPtBinning.empty()) {
    bookME(ibooker, muon_pt, "muon_pt", "muon p_{T}", leptonPtBinning);
    setMETitle(muon_pt, "p_{T}(#mu)", "Entries");
    bookME(ibooker, electron_pt, "electron_pt", "electron p_{T}", leptonPtBinning);
    setMETitle(electron_pt, "p_{T}(ele)", "Entries");
  }
  if (!leptonEtaBinning.empty()) {
    bookME(ibooker, muon_eta, "muon_eta", "muon #eta", leptonEtaBinning);
    setMETitle(muon_eta, "#eta(#mu)", "Entries");
    bookME(ibooker, electron_eta, "electron_eta", "electron #eta", leptonEtaBinning);
    setMETitle(electron_eta, "#eta(ele)", "Entries");
  }
  if (!diLeptonPtBinning.empty()) {
    bookME(ibooker, dilepton_pt, "dilepton_pt", "dilepton p_{T}", diLeptonPtBinning);
    setMETitle(dilepton_pt, "p_{T}(ll)", "Entries");
  }
  if (!diLeptonMassBinning.empty()) {
    bookME(ibooker, dilepton_mass, "dilepton_mass", "dilepton mass", diLeptonMassBinning);
    setMETitle(dilepton_mass, "m(ll)", "Entries");
  }

  if (!jet1PtBinning.empty()) {
    bookME(ibooker, jet1_pt, "jet1_pt", "jet1 p_{T}", jet1PtBinning);
    setMETitle(jet1_pt, "p_{T}(j1)", "Entries");
  }
  if (!jet2PtBinning.empty()) {
    bookME(ibooker, jet2_pt, "jet2_pt", "jet2 p_{T}", jet2PtBinning);
    setMETitle(jet2_pt, "p_{T}(j2)", "Entries");
  }
  if (!jet1EtaBinning.empty()) {
    bookME(ibooker, jet1_eta, "jet1_eta", "jet1 #eta", jet1EtaBinning);
    setMETitle(jet1_eta, "#eta(j1)", "Entries");
  }
  if (!jet2EtaBinning.empty()) {
    bookME(ibooker, jet2_eta, "jet2_eta", "jet2 #eta", jet2EtaBinning);
    setMETitle(jet2_eta, "#eta(j2)", "Entries");
  }
  if (!jet1PNETscoreBinning.empty()) {
    bookME(ibooker, jet1_pnetscore, "jet1_pnetscore", "jet lead PNET-score", jet1PNETscoreBinning);
    setMETitle(jet1_pnetscore, "Lead PNET-score", "Entries");
  }

  if (!jet2PNETscoreBinning.empty()) {
    bookME(ibooker, jet2_pnetscore, "jet2_pnetscore", "jet train PNET-score", jet2PNETscoreBinning);
    setMETitle(jet2_pnetscore, "Trail PNET-score", "Entries");
  }
  if (!jet1PNETscoreBinning.empty() and !jet2PNETscoreBinning.empty()) {
    bookME(ibooker, mean_2j_pnetscore, "mean_2j_pnetscore", "mean 2jet PNET-score", jet1PNETscoreBinning);
    setMETitle(mean_2j_pnetscore, "Mean(PNET-score)", "Entries");
  }

  if (!jet1PNETscoreTransBinning.empty()) {
    bookME(ibooker,
           jet1_pnetscore_trans,
           "jet1_pnetscore_trans",
           "jet lead PNET-score transformed",
           jet1PNETscoreTransBinning);
    setMETitle(jet1_pnetscore_trans, "Lead atanh(PNET-score)", "Entries");
  }
  if (!jet2PNETscoreTransBinning.empty()) {
    bookME(ibooker,
           jet2_pnetscore_trans,
           "jet2_pnetscore_trans",
           "jet trail PNET-score transformed",
           jet2PNETscoreTransBinning);
    setMETitle(jet2_pnetscore_trans, "Trail atanh(PNET-score)", "Entries");
  }
  if (!jet1PNETscoreTransBinning.empty() and !jet2PNETscoreTransBinning.empty()) {
    bookME(ibooker,
           mean_2j_pnetscore_trans,
           "mean_2j_pnetscore_trans",
           "mean 2jet PNET-score transformed",
           jet1PNETscoreTransBinning);
    setMETitle(mean_2j_pnetscore_trans, "atanh(Mean(PNET-score))", "Entries");
  }

  // 2D efficiencies
  if (!jet1PtBinning2d.empty() and !jet1EtaBinning2d.empty()) {
    bookME(ibooker, jet1_pt_eta, "jet1_pt_eta", "jet1 p_{T} vs #eta", jet1PtBinning2d, jet1EtaBinning2d);
    setMETitle(jet1_pt_eta, "p_{T}(j1)", "#eta(j1)");
  }
  if (!jet2PtBinning2d.empty() and !jet2EtaBinning2d.empty()) {
    bookME(ibooker, jet2_pt_eta, "jet2_pt_eta", "jet2 p_{T} vs #eta", jet2PtBinning2d, jet2EtaBinning2d);
    setMETitle(jet2_pt_eta, "p_{T}(j2)", "#eta(j2)");
  }

  if (!jet1PtBinning2d.empty() and !jet1PNETscoreBinning2d.empty()) {
    bookME(ibooker,
           jet1_pt_pnetscore1,
           "jet1_pt_pnetscore1",
           "jet1 p{T} vs lead PNET-score",
           jet1PtBinning2d,
           jet1PNETscoreBinning2d);
    setMETitle(jet1_pt_pnetscore1, "p_{T}(j1)", "Lead PNET-score");
  }
  if (!jet1PtBinning2d.empty() and !jet2PNETscoreBinning2d.empty()) {
    bookME(ibooker,
           jet1_pt_pnetscore2,
           "jet1_pt_pnetscore2",
           "jet1 p_{T} vs trail PNET-score",
           jet1PtBinning2d,
           jet2PNETscoreBinning2d);
    setMETitle(jet1_pt_pnetscore2, "p_{T}(j1)", "Trail PNET-score");
  }
  if (!jet1PtBinning2d.empty() and !jet1PNETscoreBinning2d.empty() and !jet2PNETscoreBinning2d.empty()) {
    bookME(ibooker,
           jet1_pt_mean2pnetscore,
           "jet1_pt_mean2pnetscore",
           "jet1 p_{T} vs mean 2jet PNET-score",
           jet1PtBinning2d,
           jet1PNETscoreBinning2d);
    setMETitle(jet1_pt_mean2pnetscore, "p_{T}(j1)", "Mean(PNET-score)");
  }

  if (!jet2PtBinning2d.empty() and !jet1PNETscoreBinning2d.empty()) {
    bookME(ibooker,
           jet2_pt_pnetscore1,
           "jet2_pt_pnetscore1",
           "jet2 p_{T} vs lead PNET-score",
           jet2PtBinning2d,
           jet1PNETscoreBinning2d);
    setMETitle(jet2_pt_pnetscore1, "p_{T}(j2)", "Lead PNET-score");
  }
  if (!jet2PtBinning2d.empty() and !jet2PNETscoreBinning2d.empty()) {
    bookME(ibooker,
           jet2_pt_pnetscore2,
           "jet2_pt_pnetscore2",
           "jet2 p_{T} vs trail PNET-score",
           jet2PtBinning2d,
           jet2PNETscoreBinning2d);
    setMETitle(jet2_pt_pnetscore2, "p_{T}(j2)", "Trail PNET-score");
  }
  if (!jet2PtBinning2d.empty() and !jet1PNETscoreBinning2d.empty() and !jet2PNETscoreBinning2d.empty()) {
    bookME(ibooker,
           jet2_pt_mean2pnetscore,
           "jet2_pt_mean2pnetscore",
           "jet2 p_{T} vs mean 2jet PNET-score",
           jet2PtBinning2d,
           jet1PNETscoreBinning2d);
    setMETitle(jet2_pt_mean2pnetscore, "p_{T}(j2)", "Mean(PNET-score)");
  }

  if (!jet1PtBinning2d.empty() and !jet1PNETscoreTransBinning2d.empty()) {
    bookME(ibooker,
           jet1_pt_pnetscore1_trans,
           "jet1_pt_pnetscore1_trans",
           "jet1 p_{T} vs lead PNET-score transformed",
           jet1PtBinning2d,
           jet1PNETscoreTransBinning2d);
    setMETitle(jet1_pt_pnetscore1_trans, "p_{T}(j1)", "Lead atanh(PNET-score)");
  }
  if (!jet1PtBinning2d.empty() and !jet2PNETscoreTransBinning2d.empty()) {
    bookME(ibooker,
           jet1_pt_pnetscore2_trans,
           "jet1_pt_pnetscore2_trans",
           "jet1 p_{T} vs trail PNET-score transformed",
           jet1PtBinning2d,
           jet2PNETscoreTransBinning2d);
    setMETitle(jet1_pt_pnetscore2_trans, "p_{T}(j1)", "Trail atanh(PNET-score)");
  }
  if (!jet1PtBinning2d.empty() and !jet1PNETscoreTransBinning2d.empty() and !jet2PNETscoreTransBinning2d.empty()) {
    bookME(ibooker,
           jet1_pt_mean2pnetscore_trans,
           "jet1_pt_mean2pnetscore_trans",
           "jet1 p_{T} vs mean 2jet PNET-score transformed",
           jet1PtBinning2d,
           jet1PNETscoreBinning2d);
    setMETitle(jet1_pt_mean2pnetscore_trans, "p_{T}(j1)", "atanh(Mean(PNET-score))");
  }

  if (!jet2PtBinning2d.empty() and !jet1PNETscoreTransBinning2d.empty()) {
    bookME(ibooker,
           jet2_pt_pnetscore1_trans,
           "jet2_pt_pnetscore1_trans",
           "jet2 p_{T} vs lead PNET-score transformed",
           jet2PtBinning2d,
           jet1PNETscoreTransBinning2d);
    setMETitle(jet2_pt_pnetscore1_trans, "p_{T}(j2)", "Lead atanh(PNET-score)");
  }
  if (!jet2PtBinning2d.empty() and !jet2PNETscoreTransBinning2d.empty()) {
    bookME(ibooker,
           jet2_pt_pnetscore2_trans,
           "jet2_pt_pnetscore2_trans",
           "jet2 p_{T} vs trail PNET-score transformed",
           jet2PtBinning2d,
           jet2PNETscoreTransBinning2d);
    setMETitle(jet2_pt_pnetscore2_trans, "p_{T}(j2)", "Trail atanh(PNET-score)");
  }
  if (!jet2PtBinning2d.empty() and !jet1PNETscoreTransBinning2d.empty() and !jet2PNETscoreTransBinning2d.empty()) {
    bookME(ibooker,
           jet2_pt_mean2pnetscore_trans,
           "jet2_pt_mean2pnetscore_trans",
           "jet2 p_{T} vs mean 2jet PNET-score transformed",
           jet2PtBinning2d,
           jet1PNETscoreBinning2d);
    setMETitle(jet2_pt_mean2pnetscore_trans, "p_{T}(j2)", "atanh(Mean(PNET-score))");
  }
}

void ParticleNetJetTagMonitor::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) {
  // abort if triggers are not valid
  if (requireValidHLTPaths_ and (!denHLTPathsAreValid_ or !numHLTPathsAreValid_))
    return;

  int selectionFlowStatus = 0;
  selectionFlow->Fill(selectionFlowStatus);

  // Filter out events that don't pass the denominator trigger condition
  if (denGenericTriggerEvent_->on() and !denGenericTriggerEvent_->accept(iEvent, iSetup))
    return;

  selectionFlowStatus++;
  selectionFlow->Fill(selectionFlowStatus);

  // get all input collections
  edm::Handle<reco::VertexCollection> primaryVertices;
  iEvent.getByToken(vertexToken_, primaryVertices);
  if (!primaryVertices.isValid()) {
    edm::LogWarning("ParticleNetJetTagMonitor") << "Invalid primary vertex collection, will skip the event";
    return;
  }

  edm::Handle<reco::MuonCollection> muonHandle;
  iEvent.getByToken(muonToken_, muonHandle);
  if (!muonHandle.isValid()) {
    edm::LogWarning("ParticleNetJetTagMonitor") << "Muon collection not valid, will skip the event \n";
    return;
  }

  edm::Handle<reco::GsfElectronCollection> electronHandle;
  iEvent.getByToken(electronToken_, electronHandle);
  if (!electronHandle.isValid()) {
    edm::LogWarning("ParticleNetJetTagMonitor") << "Electron collection not valid, will skip the event \n";
    return;
  }

  edm::Handle<edm::ValueMap<bool>> tagEleIDHandle;
  iEvent.getByToken(tagElectronIDToken_, tagEleIDHandle);
  if (!tagEleIDHandle.isValid()) {
    edm::LogWarning("ParticleNetJetTagMonitor") << "Electron ID for tag not valid, will skip the event \n";
    return;
  }

  edm::Handle<edm::ValueMap<bool>> vetoEleIDHandle;
  iEvent.getByToken(vetoElectronIDToken_, vetoEleIDHandle);
  if (!vetoEleIDHandle.isValid()) {
    edm::LogWarning("ParticleNetJetTagMonitor") << "Electron ID for veto not valid, will skip the event \n";
    return;
  }

  edm::Handle<reco::PFJetCollection> jetHandle;
  iEvent.getByToken(jetToken_, jetHandle);
  if (!jetHandle.isValid()) {
    edm::LogWarning("ParticleNetJetTagMonitor") << "Jet collection not valid, will skip the event \n";
    return;
  }

  edm::Handle<reco::PFMETCollection> metHandle;
  iEvent.getByToken(metToken_, metHandle);
  if (!metHandle.isValid()) {
    edm::LogWarning("ParticleNetJetTagMonitor") << "MET collection not valid, will skip the event \n";
    return;
  }

  edm::Handle<reco::JetTagCollection> jetPNETScoreHandle;
  iEvent.getByToken(jetPNETScoreToken_, jetPNETScoreHandle);
  if (!jetPNETScoreHandle.isValid()) {
    edm::LogWarning("ParticleNetJetTagMonitor") << "Jet PNET score collection not valid, will skip event \n";
    return;
  }
  const reco::JetTagCollection& jetPNETScore = *(jetPNETScoreHandle.product());

  // Collections that are only imported when necessary and their validity determins the selection applied (different for ak4/ak8 workflows)
  edm::Handle<edm::ValueMap<float>> jetSoftDropMassHandle;
  iEvent.getByToken(jetSoftDropMassToken_, jetSoftDropMassHandle);
  edm::Handle<reco::PFJetCollection> jetForHTandBTagHandle;
  iEvent.getByToken(jetForHTandBTagToken_, jetForHTandBTagHandle);
  edm::Handle<reco::JetTagCollection> jetPNETScoreForHTandBTagHandle;
  iEvent.getByToken(jetPNETScoreForHTandBTagToken_, jetPNETScoreForHTandBTagHandle);
  edm::Handle<reco::JetCorrector> jecHandle;
  if (iEvent.isRealData())
    iEvent.getByToken(jecDataToken_, jecHandle);
  else
    iEvent.getByToken(jecMCToken_, jecHandle);

  // Start the selection part
  selectionFlowStatus++;
  selectionFlow->Fill(selectionFlowStatus);

  // primary vertex selection
  const reco::Vertex* pv = nullptr;
  for (auto const& v : *primaryVertices) {
    if (not vertexSelection_(v))
      continue;
    pv = &v;
    break;
  }

  if (pv == nullptr) {
    edm::LogWarning("ParticleNetJetTagMonitor") << "No good vertex found in the event --> skipped";
    return;
  }

  selectionFlowStatus++;
  selectionFlow->Fill(selectionFlowStatus);

  // Muon selection
  std::vector<reco::Muon> tagMuons;
  std::vector<reco::Muon> vetoMuons;
  for (auto const& m : *muonHandle) {
    if (tagMuonSelection_(m) and std::fabs(m.muonBestTrack()->dxy(pv->position())) <= maxLeptonDxyCut_ and
        std::fabs(m.muonBestTrack()->dz(pv->position())) <= maxLeptonDzCut_)
      tagMuons.push_back(m);
    if (vetoMuonSelection_(m) and std::fabs(m.muonBestTrack()->dxy(pv->position())) <= maxLeptonDxyCut_ and
        std::fabs(m.muonBestTrack()->dz(pv->position())) <= maxLeptonDzCut_)
      vetoMuons.push_back(m);
  }

  if (ntagmuons_ >= 0 and int(tagMuons.size()) != ntagmuons_)
    return;
  selectionFlowStatus++;
  selectionFlow->Fill(selectionFlowStatus);

  if (nvetomuons_ >= 0 and int(vetoMuons.size()) != nvetomuons_)
    return;
  selectionFlowStatus++;
  selectionFlow->Fill(selectionFlowStatus);

  // electron selection
  std::vector<reco::GsfElectron> tagElectrons;
  std::vector<reco::GsfElectron> vetoElectrons;
  for (size_t index = 0; index < electronHandle->size(); index++) {
    const auto e = electronHandle->at(index);
    if (tagElectronSelection_(e) and (*tagEleIDHandle)[reco::GsfElectronRef(electronHandle, index)] and
        std::fabs(e.gsfTrack()->dxy(pv->position())) <= maxLeptonDxyCut_ and
        std::fabs(e.gsfTrack()->dz(pv->position())) <= maxLeptonDzCut_)
      tagElectrons.push_back(e);
    if (vetoElectronSelection_(e) and (*vetoEleIDHandle)[reco::GsfElectronRef(electronHandle, index)] and
        std::fabs(e.gsfTrack()->dxy(pv->position())) <= maxLeptonDxyCut_ and
        std::fabs(e.gsfTrack()->dz(pv->position())) <= maxLeptonDzCut_)
      vetoElectrons.push_back(e);
  }

  if (ntagelectrons_ >= 0 and int(tagElectrons.size()) != ntagelectrons_)
    return;
  selectionFlowStatus++;
  selectionFlow->Fill(selectionFlowStatus);

  if (nvetoelectrons_ >= 0 and int(vetoElectrons.size()) != nvetoelectrons_)
    return;
  selectionFlowStatus++;
  selectionFlow->Fill(selectionFlowStatus);

  // Overall number of leptons for flavor composition
  if (ntagleptons_ >= 0 and int(tagElectrons.size() + tagMuons.size()) != ntagleptons_)
    return;
  selectionFlowStatus++;
  selectionFlow->Fill(selectionFlowStatus);

  if (nvetoleptons_ >= 0 and int(vetoElectrons.size() + vetoMuons.size()) != nvetoleptons_)
    return;
  selectionFlowStatus++;
  selectionFlow->Fill(selectionFlowStatus);

  // Dilepton pairs
  std::vector<reco::CompositeCandidate> emuPairs;
  for (auto const& muon : tagMuons) {
    for (auto const& electron : tagElectrons) {
      reco::CompositeCandidate emuPair("emPair");
      emuPair.addDaughter(*dynamic_cast<const reco::Candidate*>(&muon), "lep1");
      emuPair.addDaughter(*dynamic_cast<const reco::Candidate*>(&electron), "lep2");
      AddFourMomenta addp4;
      addp4.set(emuPair);
      if (dileptonSelection_(emuPair))
        emuPairs.push_back(emuPair);
    }
  }

  if (nemupairs_ >= 0 and int(emuPairs.size()) != nemupairs_)
    return;
  selectionFlowStatus++;
  selectionFlow->Fill(selectionFlowStatus);

  // For b-tagging requriements / content used in AK8 PNET efficiency measurement in semi-leptonic ttbar
  float hT = 0;
  std::vector<math::XYZTLorentzVector> jetsBTagged;
  if (jetForHTandBTagHandle.isValid()) {
    const reco::JetTagCollection& jetPNETScoreForHTandBTag = *(jetPNETScoreForHTandBTagHandle.product());
    for (auto const& j : *jetForHTandBTagHandle) {
      if (not jetSelectionForHTandBTag_(j))
        continue;
      float minDR_jm = 1000;
      for (size_t imu = 0; imu < vetoMuons.size(); imu++) {
        float dR = reco::deltaR(vetoMuons.at(imu).p4(), j.p4());
        if (dR < minDR_jm)
          minDR_jm = dR;
      }
      if (minDR_jm < lepJetDeltaRminForHTandBTag_)
        continue;
      float minDR_je = 1000;
      for (size_t iel = 0; iel < vetoElectrons.size(); iel++) {
        float dR = reco::deltaR(vetoElectrons.at(iel).p4(), j.p4());
        if (dR < minDR_je)
          minDR_je = dR;
      }
      if (minDR_je < lepJetDeltaRminForHTandBTag_)
        continue;
      hT += j.pt();
      auto const& jref = reco::JetBaseRef(reco::PFJetRef(jetForHTandBTagHandle, &j - &(*jetForHTandBTagHandle)[0]));
      if (jetPNETScoreForHTandBTag[jref] < minPNETBTagCut_)
        continue;
      jetsBTagged.push_back(j.p4());
    }
    if (int(jetsBTagged.size()) < nbjets_)
      return;
  }
  selectionFlowStatus++;
  selectionFlow->Fill(selectionFlowStatus);

  // MET selectiona
  reco::PFMET pfMet = metHandle->front();
  if (!metSelection_(pfMet))
    return;
  selectionFlowStatus++;
  selectionFlow->Fill(selectionFlowStatus);

  // Jet selection
  std::vector<reco::PFJet> selectedJets;
  std::vector<float> jetPtCorrectedValues;
  std::vector<float> jetPNETScoreValues;

  for (auto const& j : *jetHandle) {
    // apply or not jecs
    float jec = 1;
    if (jecHandle.isValid())
      jec = jecHandle->correction(j);
    auto jet = *(j.clone());
    jet.setP4(j.p4() * jec);

    // Basic selection
    if (not jetSelection_(jet))
      continue;
    // SoftDrop mass
    if (jetSoftDropMassHandle.isValid()) {
      auto const& massSD = (*jetSoftDropMassHandle)[reco::PFJetRef(jetHandle, &j - &(*jetHandle)[0])];
      if (massSD < minSoftDropMassCut_ or massSD > maxSoftDropMassCut_)
        continue;
    }
    // cleaning with leptons
    float minDR_jm = 1000;
    for (size_t imu = 0; imu < vetoMuons.size(); imu++) {
      float dR = reco::deltaR(vetoMuons.at(imu).p4(), jet.p4());
      if (dR < minDR_jm)
        minDR_jm = dR;
    }
    if (minDR_jm < lepJetDeltaRmin_)
      continue;
    float minDR_je = 1000;
    for (size_t iel = 0; iel < vetoElectrons.size(); iel++) {
      float dR = reco::deltaR(vetoElectrons.at(iel).p4(), jet.p4());
      if (dR < minDR_je)
        minDR_je = dR;
    }
    if (minDR_je < lepJetDeltaRmin_)
      continue;
    // Compute HT
    if (not jetForHTandBTagHandle.isValid())
      hT += jet.pt();
    // selected jets, pT values (post-correction), and PNET score
    selectedJets.push_back(jet);
    jetPtCorrectedValues.push_back(jet.pt());
    auto jref = reco::JetBaseRef(reco::PFJetRef(jetHandle, &j - &(*jetHandle)[0]));
    if (jetPNETScore[jref])
      jetPNETScoreValues.push_back(jetPNETScore[jref]);
    else
      jetPNETScoreValues.push_back(0);
  }
  if (njets_ >= 0 and int(selectedJets.size()) < njets_)
    return;
  selectionFlowStatus++;
  selectionFlow->Fill(selectionFlowStatus);

  // count number of jets over minPNETScoreCut
  if (std::count_if(jetPNETScoreValues.begin(), jetPNETScoreValues.end(), [&](float score) {
        return score > minPNETScoreCut_;
      }) < njets_)
    return;
  selectionFlowStatus++;
  selectionFlow->Fill(selectionFlowStatus);

  // sort descending corrected pt order only if jecHandle is valid otherwise keep the current ordering
  std::vector<size_t> jetPtSortedIndices(jetPtCorrectedValues.size());
  std::iota(jetPtSortedIndices.begin(), jetPtSortedIndices.end(), 0);
  if (jecHandle.isValid()) {
    std::sort(jetPtSortedIndices.begin(), jetPtSortedIndices.end(), [&](const size_t& i1, const size_t& i2) {
      return jetPtCorrectedValues[i1] > jetPtCorrectedValues[i2];
    });
  }

  // sort descending PNET score order
  std::vector<size_t> jetPNETScoreSortedIndices(jetPNETScoreValues.size());
  std::iota(jetPNETScoreSortedIndices.begin(), jetPNETScoreSortedIndices.end(), 0);
  std::sort(jetPNETScoreSortedIndices.begin(),
            jetPNETScoreSortedIndices.end(),
            [&](const size_t& i1, const size_t& i2) { return jetPNETScoreValues[i1] > jetPNETScoreValues[i2]; });

  // trigger object matching (for jets)
  if (requireHLTOfflineJetMatching_) {
    edm::Handle<reco::JetTagCollection> jetPNETScoreHLTHandle;
    iEvent.getByToken(jetPNETScoreHLTToken_, jetPNETScoreHLTHandle);
    if (!jetPNETScoreHLTHandle.isValid()) {
      edm::LogWarning("ParticleNetJetTagMonitor") << "HLT Jet tags collection not valid, will skip event \n";
      return;
    }

    std::vector<float> jetPNETScoreValuesHLT;
    std::vector<reco::JetBaseRef> jetHLTRefs;

    // protect for wrong event content
    if (not jetPNETScoreHLTHandle->keyProduct().isAvailable()) {
      edm::LogWarning("ParticleNetJetTagMonitor")
          << "Collection used as a key by HLT Jet tags collection is not available, will skip event";
      return;
    }

    for (const auto& jtag : *jetPNETScoreHLTHandle) {
      jetPNETScoreValuesHLT.push_back(jtag.second);
      jetHLTRefs.push_back(jtag.first);
    }

    // sort in PNET score
    std::vector<size_t> jetPNETScoreSortedIndicesHLT(jetPNETScoreValuesHLT.size());
    std::iota(jetPNETScoreSortedIndicesHLT.begin(), jetPNETScoreSortedIndicesHLT.end(), 0);
    std::sort(
        jetPNETScoreSortedIndicesHLT.begin(),
        jetPNETScoreSortedIndicesHLT.end(),
        [&](const size_t& i1, const size_t& i2) { return jetPNETScoreValuesHLT[i1] > jetPNETScoreValuesHLT[i2]; });

    // match reco and hlt objects considering only the first ntrigobjecttomatch jets for both reco and HLT. Each of them must be matched
    std::vector<int> matched_obj;
    for (size_t jreco = 0; jreco < ntrigobjecttomatch_; jreco++) {
      if (jreco >= jetPNETScoreSortedIndices.size())
        break;
      float minDR = 1000;
      int match_index = -1;
      for (size_t jhlt = 0; jhlt < ntrigobjecttomatch_; jhlt++) {
        if (jhlt >= jetPNETScoreSortedIndicesHLT.size())
          break;
        if (std::find(matched_obj.begin(), matched_obj.end(), jhlt) != matched_obj.end())
          continue;
        float dR = reco::deltaR(selectedJets[jetPNETScoreSortedIndices.at(jreco)].p4(),
                                jetHLTRefs.at(jetPNETScoreSortedIndicesHLT.at(jhlt))->p4());
        if (dR < hltRecoDeltaRmax_ and dR < minDR) {
          match_index = jhlt;
          minDR = dR;
        }
      }
      if (match_index >= 0)
        matched_obj.push_back(match_index);
    }
    if (matched_obj.size() != ntrigobjecttomatch_)
      return;
  }
  selectionFlowStatus++;
  selectionFlow->Fill(selectionFlowStatus);

  // numerator condition
  const bool trg_passed = (numGenericTriggerEvent_->on() and numGenericTriggerEvent_->accept(iEvent, iSetup));
  if (trg_passed) {
    selectionFlowStatus++;
    selectionFlow->Fill(selectionFlowStatus);
  }

  // Fill histograms for efficiency
  if (njets.numerator != nullptr)
    njets.fill(trg_passed, selectedJets.size());
  if (nbjets.numerator != nullptr)
    nbjets.fill(trg_passed, jetsBTagged.size());
  if (ht.numerator != nullptr)
    ht.fill(trg_passed, hT);
  if (muon_pt.numerator != nullptr)
    muon_pt.fill(trg_passed, (!tagMuons.empty()) ? tagMuons.front().pt() : 0);
  if (muon_eta.numerator != nullptr)
    muon_eta.fill(trg_passed, (!tagMuons.empty()) ? tagMuons.front().eta() : 0);
  if (electron_pt.numerator != nullptr)
    electron_pt.fill(trg_passed, (!tagElectrons.empty()) ? tagElectrons.front().pt() : -100);
  if (electron_eta.numerator != nullptr)
    electron_eta.fill(trg_passed, (!tagElectrons.empty()) ? tagElectrons.front().eta() : -100);
  if (dilepton_pt.numerator != nullptr)
    dilepton_pt.fill(trg_passed, (!emuPairs.empty()) ? emuPairs.front().pt() : 0);
  if (dilepton_mass.numerator != nullptr)
    dilepton_mass.fill(trg_passed, (!emuPairs.empty()) ? emuPairs.front().mass() : 0);

  if (jet1_pt.numerator != nullptr)
    jet1_pt.fill(trg_passed, (!selectedJets.empty()) ? selectedJets.at(jetPtSortedIndices.at(0)).pt() : 0);
  if (jet2_pt.numerator != nullptr)
    jet2_pt.fill(trg_passed, (selectedJets.size() > 1) ? selectedJets.at(jetPtSortedIndices.at(1)).pt() : 0);
  if (jet1_eta.numerator != nullptr)
    jet1_eta.fill(trg_passed, (!selectedJets.empty()) ? selectedJets.at(jetPtSortedIndices.at(0)).eta() : -100);
  if (jet2_eta.numerator != nullptr)
    jet2_eta.fill(trg_passed, (selectedJets.size() > 1) ? selectedJets.at(jetPtSortedIndices.at(1)).eta() : -100);
  if (jet1_pnetscore.numerator != nullptr)
    jet1_pnetscore.fill(trg_passed,
                        (!selectedJets.empty()) ? jetPNETScoreValues.at(jetPNETScoreSortedIndices.at(0)) : -1);
  if (jet2_pnetscore.numerator != nullptr)
    jet2_pnetscore.fill(trg_passed,
                        (selectedJets.size() > 1) ? jetPNETScoreValues.at(jetPNETScoreSortedIndices.at(1)) : -1);
  if (jet1_pnetscore_trans.numerator != nullptr)
    jet1_pnetscore_trans.fill(
        trg_passed, (!selectedJets.empty()) ? atanh(jetPNETScoreValues.at(jetPNETScoreSortedIndices.at(0))) : -1);
  if (jet2_pnetscore_trans.numerator != nullptr)
    jet2_pnetscore_trans.fill(
        trg_passed, (selectedJets.size() > 1) ? atanh(jetPNETScoreValues.at(jetPNETScoreSortedIndices.at(1))) : -1);
  if (mean_2j_pnetscore.numerator != nullptr)
    mean_2j_pnetscore.fill(trg_passed,
                           (selectedJets.size() > 1) ? (jetPNETScoreValues.at(jetPNETScoreSortedIndices.at(0)) +
                                                        jetPNETScoreValues.at(jetPNETScoreSortedIndices.at(1))) /
                                                           2.
                                                     : -1);
  if (mean_2j_pnetscore_trans.numerator != nullptr)
    mean_2j_pnetscore_trans.fill(trg_passed,
                                 (selectedJets.size() > 1)
                                     ? atanh((jetPNETScoreValues.at(jetPNETScoreSortedIndices.at(0)) +
                                              jetPNETScoreValues.at(jetPNETScoreSortedIndices.at(1))) /
                                             2.)
                                     : -1);

  if (jet1_pt_eta.numerator != nullptr)
    jet1_pt_eta.fill(trg_passed,
                     (!selectedJets.empty()) ? selectedJets.at(jetPtSortedIndices.at(0)).pt() : 0,
                     (!selectedJets.empty()) ? selectedJets.at(jetPtSortedIndices.at(0)).eta() : -1);
  if (jet2_pt_eta.numerator != nullptr)
    jet2_pt_eta.fill(trg_passed,
                     (selectedJets.size() > 1) ? selectedJets.at(jetPtSortedIndices.at(1)).pt() : 0,
                     (selectedJets.size() > 1) ? selectedJets.at(jetPtSortedIndices.at(1)).eta() : -1);

  if (jet1_pt_pnetscore1.numerator != nullptr)
    jet1_pt_pnetscore1.fill(trg_passed,
                            (!selectedJets.empty()) ? selectedJets.at(jetPtSortedIndices.at(0)).pt() : 0,
                            (!selectedJets.empty()) ? jetPNETScoreValues.at(jetPNETScoreSortedIndices.at(0)) : -1);
  if (jet2_pt_pnetscore1.numerator != nullptr)
    jet2_pt_pnetscore1.fill(trg_passed,
                            (selectedJets.size() > 1) ? selectedJets.at(jetPtSortedIndices.at(1)).pt() : 0,
                            (selectedJets.size() > 1) ? jetPNETScoreValues.at(jetPNETScoreSortedIndices.at(0)) : -1);
  if (jet1_pt_pnetscore2.numerator != nullptr)
    jet1_pt_pnetscore2.fill(trg_passed,
                            (selectedJets.size() > 1) ? selectedJets.at(jetPtSortedIndices.at(0)).pt() : 0,
                            (selectedJets.size() > 1) ? jetPNETScoreValues.at(jetPNETScoreSortedIndices.at(1)) : -1);
  if (jet2_pt_pnetscore2.numerator != nullptr)
    jet2_pt_pnetscore2.fill(trg_passed,
                            (selectedJets.size() > 1) ? selectedJets.at(jetPtSortedIndices.at(1)).pt() : 0,
                            (selectedJets.size() > 1) ? jetPNETScoreValues.at(jetPNETScoreSortedIndices.at(1)) : -1);

  if (jet1_pt_pnetscore1_trans.numerator != nullptr)
    jet1_pt_pnetscore1_trans.fill(
        trg_passed,
        (!selectedJets.empty()) ? selectedJets.at(jetPtSortedIndices.at(0)).pt() : 0,
        (!selectedJets.empty()) ? atanh(jetPNETScoreValues.at(jetPNETScoreSortedIndices.at(0))) : -1);
  if (jet2_pt_pnetscore1_trans.numerator != nullptr)
    jet2_pt_pnetscore1_trans.fill(
        trg_passed,
        (selectedJets.size() > 1) ? selectedJets.at(jetPtSortedIndices.at(1)).pt() : 0,
        (selectedJets.size() > 1) ? atanh(jetPNETScoreValues.at(jetPNETScoreSortedIndices.at(0))) : -1);
  if (jet1_pt_pnetscore2_trans.numerator != nullptr)
    jet1_pt_pnetscore2_trans.fill(
        trg_passed,
        (selectedJets.size() > 1) ? selectedJets.at(jetPtSortedIndices.at(0)).pt() : 0,
        (selectedJets.size() > 1) ? atanh(jetPNETScoreValues.at(jetPNETScoreSortedIndices.at(1))) : -1);
  if (jet2_pt_pnetscore2_trans.numerator != nullptr)
    jet2_pt_pnetscore2_trans.fill(
        trg_passed,
        (selectedJets.size() > 1) ? selectedJets.at(jetPtSortedIndices.at(1)).pt() : 0,
        (selectedJets.size() > 1) ? atanh(jetPNETScoreValues.at(jetPNETScoreSortedIndices.at(1))) : -1);

  if (jet1_pt_mean2pnetscore.numerator != nullptr)
    jet1_pt_mean2pnetscore.fill(trg_passed,
                                (selectedJets.size() > 1) ? selectedJets.at(jetPtSortedIndices.at(0)).pt() : 0,
                                (selectedJets.size() > 1) ? (jetPNETScoreValues.at(jetPNETScoreSortedIndices.at(0)) +
                                                             jetPNETScoreValues.at(jetPNETScoreSortedIndices.at(1))) /
                                                                2
                                                          : -1);
  if (jet2_pt_mean2pnetscore.numerator != nullptr)
    jet2_pt_mean2pnetscore.fill(trg_passed,
                                (selectedJets.size() > 1) ? selectedJets.at(jetPtSortedIndices.at(1)).pt() : 0,
                                (selectedJets.size() > 1)
                                    ? atanh((jetPNETScoreValues.at(jetPNETScoreSortedIndices.at(0)) +
                                             jetPNETScoreValues.at(jetPNETScoreSortedIndices.at(1))) /
                                            2)
                                    : -1);

  if (jet1_pt_mean2pnetscore_trans.numerator != nullptr)
    jet1_pt_mean2pnetscore_trans.fill(trg_passed,
                                      (selectedJets.size() > 1) ? selectedJets.at(jetPtSortedIndices.at(0)).pt() : 0,
                                      (selectedJets.size() > 1)
                                          ? atanh((jetPNETScoreValues.at(jetPNETScoreSortedIndices.at(0)) +
                                                   jetPNETScoreValues.at(jetPNETScoreSortedIndices.at(1))) /
                                                  2)
                                          : -1);
  if (jet2_pt_mean2pnetscore_trans.numerator != nullptr)
    jet2_pt_mean2pnetscore_trans.fill(trg_passed,
                                      (selectedJets.size() > 1) ? selectedJets.at(jetPtSortedIndices.at(1)).pt() : 0,
                                      (selectedJets.size() > 1)
                                          ? atanh((jetPNETScoreValues.at(jetPNETScoreSortedIndices.at(0)) +
                                                   jetPNETScoreValues.at(jetPNETScoreSortedIndices.at(1))) /
                                                  2)
                                          : -1);
}

void ParticleNetJetTagMonitor::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("FolderName", "HLT/Higgs");
  desc.add<bool>("requireValidHLTPaths", true);
  desc.add<bool>("requireHLTOfflineJetMatching", true);
  desc.add<edm::InputTag>("vertices", edm::InputTag("offlinePrimaryVertices"));
  desc.add<edm::InputTag>("muons", edm::InputTag("muons"));
  desc.add<edm::InputTag>("electrons", edm::InputTag("gedGsfElectrons"));
  desc.add<edm::InputTag>("tagElectronID",
                          edm::InputTag("egmGsfElectronIDsForDQM:cutBasedElectronID-RunIIIWinter22-V1-tight"));
  desc.add<edm::InputTag>("vetoElectronID",
                          edm::InputTag("egmGsfElectronIDsForDQM:cutBasedElectronID-RunIIIWinter22-V1-loose"));
  desc.add<edm::InputTag>("jets", edm::InputTag("ak4PFJetsCHS"));
  desc.add<edm::InputTag>("jetPNETScore", edm::InputTag("pfParticleNetAK4DiscriminatorsJetTags", "BvsAll"));
  desc.add<edm::InputTag>("jetPNETScoreHLT", edm::InputTag("hltParticleNetDiscriminatorsJetTags", "BvsAll"));
  desc.add<edm::InputTag>("jetsForHTandBTag", edm::InputTag(""));
  desc.add<edm::InputTag>("jetPNETScoreForHTandBTag", edm::InputTag(""));
  desc.add<edm::InputTag>("jetSoftDropMass", edm::InputTag(""));
  desc.add<edm::InputTag>("met", edm::InputTag("pfMetPuppi"));
  desc.add<edm::InputTag>("jecForMC", edm::InputTag("ak4PFCHSL1FastL2L3Corrector"));
  desc.add<edm::InputTag>("jecForData", edm::InputTag("ak4PFCHSL1FastL2L3ResidualCorrector"));
  desc.add<std::string>("tagMuonSelection",
                        "pt > 25 && abs(eta) < 2.4 && passed(CutBasedIdTight) && passed(PFIsoTight)");
  desc.add<std::string>("tagElectronSelection", "pt > 20 && abs(eta) < 2.5");
  desc.add<std::string>("vetoMuonSelection",
                        "pt > 10 && abs(eta) < 2.4 && passed(CutBasedIdLoose) && passed(PFIsoLoose)");
  desc.add<std::string>("vetoElectronSelection", "pt > 10 && abs(eta) < 2.5");
  desc.add<std::string>("jetSelection", "pt > 30 && abs(eta) < 2.5");
  desc.add<std::string>("jetSelectionForHTandBTag", "pt > 30 && abs(eta) < 2.5");
  desc.add<std::string>("vertexSelection", "!isFake && ndof > 4 && abs(z) <= 24 && position.Rho <= 2");
  desc.add<std::string>("dileptonSelection", "((mass > 20 && mass < 75) || mass > 105) && charge == 0");
  desc.add<std::string>("metSelection", "pt > 0");
  desc.add<int>("ntagleptons", 2);
  desc.add<int>("ntagmuons", 1);
  desc.add<int>("ntagelectrons", 1);
  desc.add<int>("nvetoleptons", 2);
  desc.add<int>("nvetomuons", 1);
  desc.add<int>("nvetoelectrons", 1);
  desc.add<int>("nemupairs", 1);
  desc.add<int>("njets", 2);
  desc.add<int>("nbjets", -1);
  desc.add<unsigned int>("ntrigobjecttomatch", 2);
  desc.add<double>("lepJetDeltaRmin", 0.4);
  desc.add<double>("lepJetDeltaRminForHTandBTag", 0.4);
  desc.add<double>("hltRecoDeltaRmax", 0.4);
  desc.add<double>("maxLeptonDxyCut", 0.1);
  desc.add<double>("maxLeptonDzCut", 0.2);
  desc.add<double>("minPNETScoreCut", 0.2);
  desc.add<double>("minPNETBTagCut", 0.5);
  desc.add<double>("minSoftDropMassCut", 50);
  desc.add<double>("maxSoftDropMassCut", 110);
  desc.add<std::vector<double>>("leptonPtBinning", {});
  desc.add<std::vector<double>>("leptonEtaBinning", {});
  desc.add<std::vector<double>>("diLeptonPtBinning", {});
  desc.add<std::vector<double>>("diLeptonMassBinning", {});
  desc.add<std::vector<double>>("HTBinning", {});
  desc.add<std::vector<double>>("NjetBinning", {});
  desc.add<std::vector<double>>("jet1PtBinning", {});
  desc.add<std::vector<double>>("jet2PtBinning", {});
  desc.add<std::vector<double>>("jet1EtaBinning", {});
  desc.add<std::vector<double>>("jet2EtaBinning", {});
  desc.add<std::vector<double>>("jet1PNETscoreBinning", {});
  desc.add<std::vector<double>>("jet2PNETscoreBinning", {});
  desc.add<std::vector<double>>("jet1PNETscoreTransBinning", {});
  desc.add<std::vector<double>>("jet2PNETscoreTransBinning", {});
  desc.add<std::vector<double>>("jet1PtBinning2d", {});
  desc.add<std::vector<double>>("jet2PtBinning2d", {});
  desc.add<std::vector<double>>("jet1EtaBinning2d", {});
  desc.add<std::vector<double>>("jet2EtaBinning2d", {});
  desc.add<std::vector<double>>("jet1PNETscoreBinning2d", {});
  desc.add<std::vector<double>>("jet2PNETscoreBinning2d", {});
  desc.add<std::vector<double>>("jet1PNETscoreTransBinning2d", {});
  desc.add<std::vector<double>>("jet2PNETscoreTransBinning2d", {});
  edm::ParameterSetDescription genericTriggerEventPSet;
  GenericTriggerEventFlag::fillPSetDescription(genericTriggerEventPSet);
  desc.add<edm::ParameterSetDescription>("numGenericTriggerEvent", genericTriggerEventPSet);
  desc.add<edm::ParameterSetDescription>("denGenericTriggerEvent", genericTriggerEventPSet);
  descriptions.add("ParticleNetJetTagMonitor", desc);
}

// Define this as a plug-in
DEFINE_FWK_MODULE(ParticleNetJetTagMonitor);
