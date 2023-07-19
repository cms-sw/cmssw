#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
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
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"
#include "DQMOffline/Trigger/plugins/TriggerDQMBase.h"

#include <string>
#include <vector>
#include <memory>
#include <map>

class TopMonitor : public DQMEDAnalyzer, public TriggerDQMBase {
public:
  typedef dqm::reco::MonitorElement MonitorElement;
  typedef dqm::reco::DQMStore DQMStore;

  TopMonitor(const edm::ParameterSet&);
  ~TopMonitor() throw() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) override;

  struct JetRefCompare {
    inline bool operator()(const edm::RefToBase<reco::Jet>& j1, const edm::RefToBase<reco::Jet>& j2) const {
      return (j1.id() < j2.id()) || ((j1.id() == j2.id()) && (j1.key() < j2.key()));
    }
  };
  typedef std::map<edm::RefToBase<reco::Jet>, float, JetRefCompare> JetTagMap;

private:
  const std::string folderName_;

  const bool requireValidHLTPaths_;
  bool hltPathsAreValid_;

  edm::EDGetTokenT<reco::VertexCollection> vtxToken_;
  edm::EDGetTokenT<reco::MuonCollection> muoToken_;
  edm::EDGetTokenT<edm::View<reco::GsfElectron> > eleToken_;
  edm::EDGetTokenT<edm::ValueMap<bool> > elecIDToken_;
  edm::EDGetTokenT<reco::PhotonCollection> phoToken_;
  edm::EDGetTokenT<reco::PFJetCollection> jetToken_;
  std::vector<edm::EDGetTokenT<reco::JetTagCollection> > jetTagTokens_;
  edm::EDGetTokenT<reco::PFMETCollection> metToken_;

  struct PVcut {
    double dxy;
    double dz;
  };

  MEbinning met_binning_;
  MEbinning ls_binning_;
  MEbinning phi_binning_;
  MEbinning pt_binning_;
  MEbinning eta_binning_;
  MEbinning HT_binning_;
  MEbinning DR_binning_;
  MEbinning csv_binning_;
  MEbinning invMass_mumu_binning_;
  MEbinning MHT_binning_;

  std::vector<double> met_variable_binning_;
  std::vector<double> HT_variable_binning_;
  std::vector<double> jetPt_variable_binning_;
  std::vector<double> muPt_variable_binning_;
  std::vector<double> elePt_variable_binning_;
  std::vector<double> jetEta_variable_binning_;
  std::vector<double> muEta_variable_binning_;
  std::vector<double> eleEta_variable_binning_;
  std::vector<double> invMass_mumu_variable_binning_;
  std::vector<double> MHT_variable_binning_;

  std::vector<double> HT_variable_binning_2D_;
  std::vector<double> jetPt_variable_binning_2D_;
  std::vector<double> muPt_variable_binning_2D_;
  std::vector<double> elePt_variable_binning_2D_;
  std::vector<double> phoPt_variable_binning_2D_;
  std::vector<double> jetEta_variable_binning_2D_;
  std::vector<double> muEta_variable_binning_2D_;
  std::vector<double> eleEta_variable_binning_2D_;
  std::vector<double> phoEta_variable_binning_2D_;
  std::vector<double> phi_variable_binning_2D_;

  ObjME metME_;
  ObjME metME_variableBinning_;
  ObjME metVsLS_;
  ObjME metPhiME_;

  ObjME jetVsLS_;
  ObjME muVsLS_;
  ObjME eleVsLS_;
  ObjME phoVsLS_;
  ObjME bjetVsLS_;
  ObjME htVsLS_;

  ObjME jetEtaPhi_HEP17_;  // for HEP17 monitoring

  ObjME jetMulti_;
  ObjME eleMulti_;
  ObjME muMulti_;
  ObjME phoMulti_;
  ObjME bjetMulti_;

  ObjME elePt_jetPt_;
  ObjME elePt_eventHT_;

  ObjME ele1Pt_ele2Pt_;
  ObjME ele1Eta_ele2Eta_;
  ObjME mu1Pt_mu2Pt_;
  ObjME mu1Eta_mu2Eta_;
  ObjME elePt_muPt_;
  ObjME eleEta_muEta_;
  ObjME invMass_mumu_;
  ObjME eventMHT_;
  ObjME invMass_mumu_variableBinning_;
  ObjME eventMHT_variableBinning_;
  ObjME muPt_phoPt_;
  ObjME muEta_phoEta_;

  ObjME DeltaR_jet_Mu_;

  ObjME eventHT_;
  ObjME eventHT_variableBinning_;

  std::vector<ObjME> muPhi_;
  std::vector<ObjME> muEta_;
  std::vector<ObjME> muPt_;

  std::vector<ObjME> elePhi_;
  std::vector<ObjME> eleEta_;
  std::vector<ObjME> elePt_;

  std::vector<ObjME> jetPhi_;
  std::vector<ObjME> jetEta_;
  std::vector<ObjME> jetPt_;

  std::vector<ObjME> phoPhi_;
  std::vector<ObjME> phoEta_;
  std::vector<ObjME> phoPt_;

  std::vector<ObjME> bjetPhi_;
  std::vector<ObjME> bjetEta_;
  std::vector<ObjME> bjetPt_;
  std::vector<ObjME> bjetCSV_;
  std::vector<ObjME> muPt_variableBinning_;
  std::vector<ObjME> elePt_variableBinning_;
  std::vector<ObjME> jetPt_variableBinning_;
  std::vector<ObjME> bjetPt_variableBinning_;

  std::vector<ObjME> muEta_variableBinning_;
  std::vector<ObjME> eleEta_variableBinning_;
  std::vector<ObjME> jetEta_variableBinning_;
  std::vector<ObjME> bjetEta_variableBinning_;

  // 2D distributions
  std::vector<ObjME> jetPtEta_;
  std::vector<ObjME> jetEtaPhi_;

  std::vector<ObjME> elePtEta_;
  std::vector<ObjME> eleEtaPhi_;

  std::vector<ObjME> muPtEta_;
  std::vector<ObjME> muEtaPhi_;

  std::vector<ObjME> phoPtEta_;
  std::vector<ObjME> phoEtaPhi_;

  std::vector<ObjME> bjetPtEta_;
  std::vector<ObjME> bjetEtaPhi_;
  std::vector<ObjME> bjetCSVHT_;

  std::unique_ptr<GenericTriggerEventFlag> num_genTriggerEventFlag_;
  std::unique_ptr<GenericTriggerEventFlag> den_genTriggerEventFlag_;

  StringCutObjectSelector<reco::PFMET> metSelection_;
  StringCutObjectSelector<reco::PFJet> jetSelection_;
  StringCutObjectSelector<reco::GsfElectron, true> eleSelection_;
  StringCutObjectSelector<reco::Muon> muoSelection_;
  StringCutObjectSelector<reco::Photon> phoSelection_;
  StringCutObjectSelector<reco::PFJet> HTdefinition_;

  StringCutObjectSelector<reco::Vertex> vtxSelection_;

  StringCutObjectSelector<reco::Jet, true> bjetSelection_;

  unsigned int njets_;
  unsigned int nelectrons_;
  unsigned int nmuons_;
  unsigned int nphotons_;
  double leptJetDeltaRmin_;
  double bJetMuDeltaRmax_;
  double bJetDeltaEtaMax_;
  double HTcut_;
  unsigned int nbjets_;
  double workingpoint_;
  std::string btagalgoName_;
  PVcut lepPVcuts_;
  bool applyLeptonPVcuts_;

  bool applyMETcut_ = false;

  double invMassUppercut_;
  double invMassLowercut_;
  bool opsign_;
  StringCutObjectSelector<reco::PFJet> MHTdefinition_;
  double MHTcut_;

  bool invMassCutInAllMuPairs_;

  bool enablePhotonPlot_;
  bool enableMETPlot_;
  bool enable2DPlots_;
};

TopMonitor::TopMonitor(const edm::ParameterSet& iConfig)
    : folderName_(iConfig.getParameter<std::string>("FolderName")),
      requireValidHLTPaths_(iConfig.getParameter<bool>("requireValidHLTPaths")),
      hltPathsAreValid_(false),
      vtxToken_(mayConsume<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertices"))),
      muoToken_(mayConsume<reco::MuonCollection>(iConfig.getParameter<edm::InputTag>("muons"))),
      eleToken_(mayConsume<edm::View<reco::GsfElectron> >(iConfig.getParameter<edm::InputTag>("electrons"))),
      elecIDToken_(consumes<edm::ValueMap<bool> >(iConfig.getParameter<edm::InputTag>("elecID"))),
      phoToken_(mayConsume<reco::PhotonCollection>(iConfig.getParameter<edm::InputTag>("photons"))),
      jetToken_(mayConsume<reco::PFJetCollection>(iConfig.getParameter<edm::InputTag>("jets"))),
      jetTagTokens_(
          edm::vector_transform(iConfig.getParameter<std::vector<edm::InputTag> >("btagAlgos"),
                                [this](edm::InputTag const& tag) { return mayConsume<reco::JetTagCollection>(tag); })),
      metToken_(consumes<reco::PFMETCollection>(iConfig.getParameter<edm::InputTag>("met"))),
      met_binning_(getHistoPSet(
          iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("metPSet"))),
      ls_binning_(
          getHistoPSet(iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("lsPSet"))),
      phi_binning_(getHistoPSet(
          iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("phiPSet"))),
      pt_binning_(
          getHistoPSet(iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("ptPSet"))),
      eta_binning_(getHistoPSet(
          iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("etaPSet"))),
      HT_binning_(
          getHistoPSet(iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("htPSet"))),
      DR_binning_(
          getHistoPSet(iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("DRPSet"))),
      csv_binning_(getHistoPSet(
          iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("csvPSet"))),
      invMass_mumu_binning_(getHistoPSet(
          iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("invMassPSet"))),
      MHT_binning_(getHistoPSet(
          iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("MHTPSet"))),
      met_variable_binning_(
          iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("metBinning")),
      HT_variable_binning_(
          iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("HTBinning")),
      jetPt_variable_binning_(
          iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("jetPtBinning")),
      muPt_variable_binning_(
          iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("muPtBinning")),
      elePt_variable_binning_(
          iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("elePtBinning")),
      jetEta_variable_binning_(
          iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("jetEtaBinning")),
      muEta_variable_binning_(
          iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("muEtaBinning")),
      eleEta_variable_binning_(
          iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("eleEtaBinning")),
      invMass_mumu_variable_binning_(iConfig.getParameter<edm::ParameterSet>("histoPSet")
                                         .getParameter<std::vector<double> >("invMassVariableBinning")),
      MHT_variable_binning_(
          iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("MHTVariableBinning")),
      HT_variable_binning_2D_(
          iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("HTBinning2D")),
      jetPt_variable_binning_2D_(
          iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("jetPtBinning2D")),
      muPt_variable_binning_2D_(
          iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("muPtBinning2D")),
      elePt_variable_binning_2D_(
          iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("elePtBinning2D")),
      phoPt_variable_binning_2D_(
          iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("phoPtBinning2D")),
      jetEta_variable_binning_2D_(
          iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("jetEtaBinning2D")),
      muEta_variable_binning_2D_(
          iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("muEtaBinning2D")),
      eleEta_variable_binning_2D_(
          iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("eleEtaBinning2D")),
      phoEta_variable_binning_2D_(
          iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("phoEtaBinning2D")),
      phi_variable_binning_2D_(
          iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("phiBinning2D")),
      num_genTriggerEventFlag_(new GenericTriggerEventFlag(
          iConfig.getParameter<edm::ParameterSet>("numGenericTriggerEventPSet"), consumesCollector(), *this)),
      den_genTriggerEventFlag_(new GenericTriggerEventFlag(
          iConfig.getParameter<edm::ParameterSet>("denGenericTriggerEventPSet"), consumesCollector(), *this)),
      metSelection_(iConfig.getParameter<std::string>("metSelection")),
      jetSelection_(iConfig.getParameter<std::string>("jetSelection")),
      eleSelection_(iConfig.getParameter<std::string>("eleSelection")),
      muoSelection_(iConfig.getParameter<std::string>("muoSelection")),
      phoSelection_(iConfig.getParameter<std::string>("phoSelection")),
      HTdefinition_(iConfig.getParameter<std::string>("HTdefinition")),
      vtxSelection_(iConfig.getParameter<std::string>("vertexSelection")),
      bjetSelection_(iConfig.getParameter<std::string>("bjetSelection")),
      njets_(iConfig.getParameter<unsigned int>("njets")),
      nelectrons_(iConfig.getParameter<unsigned int>("nelectrons")),
      nmuons_(iConfig.getParameter<unsigned int>("nmuons")),
      nphotons_(iConfig.getParameter<unsigned int>("nphotons")),
      leptJetDeltaRmin_(iConfig.getParameter<double>("leptJetDeltaRmin")),
      bJetMuDeltaRmax_(iConfig.getParameter<double>("bJetMuDeltaRmax")),
      bJetDeltaEtaMax_(iConfig.getParameter<double>("bJetDeltaEtaMax")),
      HTcut_(iConfig.getParameter<double>("HTcut")),
      nbjets_(iConfig.getParameter<unsigned int>("nbjets")),
      workingpoint_(iConfig.getParameter<double>("workingpoint")),
      applyLeptonPVcuts_(iConfig.getParameter<bool>("applyLeptonPVcuts")),
      invMassUppercut_(iConfig.getParameter<double>("invMassUppercut")),
      invMassLowercut_(iConfig.getParameter<double>("invMassLowercut")),
      opsign_(iConfig.getParameter<bool>("oppositeSignMuons")),
      MHTdefinition_(iConfig.getParameter<std::string>("MHTdefinition")),
      MHTcut_(iConfig.getParameter<double>("MHTcut")),
      invMassCutInAllMuPairs_(iConfig.getParameter<bool>("invMassCutInAllMuPairs")),
      enablePhotonPlot_(iConfig.getParameter<bool>("enablePhotonPlot")),
      enableMETPlot_(iConfig.getParameter<bool>("enableMETPlot")),
      enable2DPlots_(iConfig.getParameter<bool>("enable2DPlots")) {
  ObjME empty;

  muPhi_ = std::vector<ObjME>(nmuons_, empty);
  muEta_ = std::vector<ObjME>(nmuons_, empty);
  muPt_ = std::vector<ObjME>(nmuons_, empty);
  muEta_variableBinning_ = std::vector<ObjME>(nmuons_, empty);
  muPt_variableBinning_ = std::vector<ObjME>(nmuons_, empty);
  muPtEta_ = std::vector<ObjME>(nmuons_, empty);
  muEtaPhi_ = std::vector<ObjME>(nmuons_, empty);

  elePhi_ = std::vector<ObjME>(nelectrons_, empty);
  eleEta_ = std::vector<ObjME>(nelectrons_, empty);
  elePt_ = std::vector<ObjME>(nelectrons_, empty);
  eleEta_variableBinning_ = std::vector<ObjME>(nelectrons_, empty);
  elePt_variableBinning_ = std::vector<ObjME>(nelectrons_, empty);
  elePtEta_ = std::vector<ObjME>(nelectrons_, empty);
  eleEtaPhi_ = std::vector<ObjME>(nelectrons_, empty);

  jetPhi_ = std::vector<ObjME>(njets_, empty);
  jetEta_ = std::vector<ObjME>(njets_, empty);
  jetPt_ = std::vector<ObjME>(njets_, empty);
  jetEta_variableBinning_ = std::vector<ObjME>(njets_, empty);
  jetPt_variableBinning_ = std::vector<ObjME>(njets_, empty);
  jetPtEta_ = std::vector<ObjME>(njets_, empty);
  jetEtaPhi_ = std::vector<ObjME>(njets_, empty);

  //Menglei Sun
  phoPhi_ = std::vector<ObjME>(nphotons_, empty);
  phoEta_ = std::vector<ObjME>(nphotons_, empty);
  phoPt_ = std::vector<ObjME>(nphotons_, empty);
  phoPtEta_ = std::vector<ObjME>(nphotons_, empty);
  phoEtaPhi_ = std::vector<ObjME>(nphotons_, empty);

  // Marina
  bjetPhi_ = std::vector<ObjME>(nbjets_, empty);
  bjetEta_ = std::vector<ObjME>(nbjets_, empty);
  bjetPt_ = std::vector<ObjME>(nbjets_, empty);
  bjetCSV_ = std::vector<ObjME>(nbjets_, empty);
  bjetEta_variableBinning_ = std::vector<ObjME>(nbjets_, empty);
  bjetPt_variableBinning_ = std::vector<ObjME>(nbjets_, empty);
  bjetPtEta_ = std::vector<ObjME>(nbjets_, empty);
  bjetEtaPhi_ = std::vector<ObjME>(nbjets_, empty);
  bjetCSVHT_ = std::vector<ObjME>(nbjets_, empty);
  //Suvankar
  lepPVcuts_.dxy = (iConfig.getParameter<edm::ParameterSet>("leptonPVcuts")).getParameter<double>("dxy");
  lepPVcuts_.dz = (iConfig.getParameter<edm::ParameterSet>("leptonPVcuts")).getParameter<double>("dz");
}

TopMonitor::~TopMonitor() throw() {
  if (num_genTriggerEventFlag_)
    num_genTriggerEventFlag_.reset();
  if (den_genTriggerEventFlag_)
    den_genTriggerEventFlag_.reset();
}

void TopMonitor::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& iRun, edm::EventSetup const& iSetup) {
  // Initialize the GenericTriggerEventFlag
  if (num_genTriggerEventFlag_ && num_genTriggerEventFlag_->on())
    num_genTriggerEventFlag_->initRun(iRun, iSetup);
  if (den_genTriggerEventFlag_ && den_genTriggerEventFlag_->on())
    den_genTriggerEventFlag_->initRun(iRun, iSetup);

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

  if (enableMETPlot_) {
    histname = "met";
    histtitle = "PFMET";
    bookME(ibooker, metME_, histname, histtitle, met_binning_.nbins, met_binning_.xmin, met_binning_.xmax);
    setMETitle(metME_, "PF MET [GeV]", "events / [GeV]");

    histname = "metPhi";
    histtitle = "PFMET phi";
    bookME(ibooker, metPhiME_, histname, histtitle, phi_binning_.nbins, phi_binning_.xmin, phi_binning_.xmax);
    setMETitle(metPhiME_, "PF MET #phi", "events / 0.1 rad");

    histname = "met_variable";
    histtitle = "PFMET";
    bookME(ibooker, metME_variableBinning_, histname, histtitle, met_variable_binning_);
    setMETitle(metME_variableBinning_, "PF MET [GeV]", "events / [GeV]");

    histname = "metVsLS";
    histtitle = "PFMET vs LS";
    bookME(ibooker,
           metVsLS_,
           histname,
           histtitle,
           ls_binning_.nbins,
           ls_binning_.xmin,
           ls_binning_.xmax,
           met_binning_.xmin,
           met_binning_.xmax);
    setMETitle(metVsLS_, "LS", "PF MET [GeV]");
  }

  if (njets_ > 0) {
    histname = "jetVsLS";
    histtitle = "jet pt vs LS";
    bookME(ibooker,
           jetVsLS_,
           histname,
           histtitle,
           ls_binning_.nbins,
           ls_binning_.xmin,
           ls_binning_.xmax,
           pt_binning_.xmin,
           pt_binning_.xmax);
    setMETitle(jetVsLS_, "LS", "jet pt [GeV]");

    histname = "jetEtaPhi_HEP17";
    histtitle = "jet #eta-#phi for HEP17";
    bookME(ibooker, jetEtaPhi_HEP17_, histname, histtitle, 10, -2.5, 2.5, 18, -3.1415, 3.1415);  // for HEP17 monitoring
    setMETitle(jetEtaPhi_HEP17_, "jet #eta", "jet #phi");

    histname = "jetMulti";
    histtitle = "jet multiplicity";
    bookME(ibooker, jetMulti_, histname, histtitle, 11, -.5, 10.5);
    setMETitle(jetMulti_, "jet multiplicity", "events");
  }

  if (nmuons_ > 0) {
    histname = "muVsLS";
    histtitle = "muon pt vs LS";
    bookME(ibooker,
           muVsLS_,
           histname,
           histtitle,
           ls_binning_.nbins,
           ls_binning_.xmin,
           ls_binning_.xmax,
           pt_binning_.xmin,
           pt_binning_.xmax);
    setMETitle(muVsLS_, "LS", "muon pt [GeV]");

    histname = "muMulti";
    histtitle = "muon multiplicity";
    bookME(ibooker, muMulti_, histname, histtitle, 6, -.5, 5.5);
    setMETitle(muMulti_, "muon multiplicity", "events");

    if (njets_ > 0) {
      histname = "DeltaR_jet_Mu";
      histtitle = "#DeltaR(jet,mu)";
      bookME(ibooker, DeltaR_jet_Mu_, histname, histtitle, DR_binning_.nbins, DR_binning_.xmin, DR_binning_.xmax);
      setMETitle(DeltaR_jet_Mu_, "#DeltaR(jet,mu)", "events");
    }
  }

  if (nelectrons_ > 0) {
    histname = "eleVsLS";
    histtitle = "electron pt vs LS";
    bookME(ibooker,
           eleVsLS_,
           histname,
           histtitle,
           ls_binning_.nbins,
           ls_binning_.xmin,
           ls_binning_.xmax,
           pt_binning_.xmin,
           pt_binning_.xmax);
    setMETitle(eleVsLS_, "LS", "electron pt [GeV]");

    histname = "eleMulti";
    histtitle = "electron multiplicity";
    bookME(ibooker, eleMulti_, histname, histtitle, 6, -.5, 5.5);
    setMETitle(eleMulti_, "electron multiplicity", "events");

    if (njets_ > 0 && enable2DPlots_) {
      histname = "elePt_jetPt";
      histtitle = "electron pt vs jet pt";
      bookME(ibooker, elePt_jetPt_, histname, histtitle, elePt_variable_binning_2D_, jetPt_variable_binning_2D_);
      setMETitle(elePt_jetPt_, "leading electron pt", "leading jet pt");
    }

    if (nmuons_ > 0 && enable2DPlots_) {
      histname = "elePt_muPt";
      histtitle = "electron pt vs muon pt";
      bookME(ibooker, elePt_muPt_, histname, histtitle, elePt_variable_binning_2D_, muPt_variable_binning_2D_);
      setMETitle(elePt_muPt_, "electron pt [GeV]", "muon pt [GeV]");

      histname = "eleEta_muEta";
      histtitle = "electron #eta vs muon #eta";
      bookME(ibooker, eleEta_muEta_, histname, histtitle, eleEta_variable_binning_2D_, muEta_variable_binning_2D_);
      setMETitle(eleEta_muEta_, "electron #eta", "muon #eta");
    }
  }

  //Menglei
  if (enablePhotonPlot_) {
    if (nphotons_ > 0) {
      histname = "photonVsLS";
      histtitle = "photon pt vs LS";
      bookME(ibooker,
             phoVsLS_,
             histname,
             histtitle,
             ls_binning_.nbins,
             ls_binning_.xmin,
             ls_binning_.xmax,
             pt_binning_.xmin,
             pt_binning_.xmax);
      setMETitle(phoVsLS_, "LS", "photon pt [GeV]");
    }
  }

  // Marina
  if (nbjets_ > 0) {
    histname = "bjetVsLS";
    histtitle = "b-jet pt vs LS";
    bookME(ibooker,
           bjetVsLS_,
           histname,
           histtitle,
           ls_binning_.nbins,
           ls_binning_.xmin,
           ls_binning_.xmax,
           pt_binning_.xmin,
           pt_binning_.xmax);
    setMETitle(bjetVsLS_, "LS", "b-jet pt [GeV]");

    histname = "bjetMulti";
    histtitle = "b-jet multiplicity";
    bookME(ibooker, bjetMulti_, histname, histtitle, 6, -.5, 5.5);
    setMETitle(bjetMulti_, "b-jet multiplicity", "events");
  }

  if (nelectrons_ > 1 && enable2DPlots_) {
    histname = "ele1Pt_ele2Pt";
    histtitle = "electron-1 pt vs electron-2 pt";
    bookME(ibooker, ele1Pt_ele2Pt_, histname, histtitle, elePt_variable_binning_2D_, elePt_variable_binning_2D_);
    setMETitle(ele1Pt_ele2Pt_, "electron-1 pt [GeV]", "electron-2 pt [GeV]");

    histname = "ele1Eta_ele2Eta";
    histtitle = "electron-1 #eta vs electron-2 #eta";
    bookME(ibooker, ele1Eta_ele2Eta_, histname, histtitle, eleEta_variable_binning_2D_, eleEta_variable_binning_2D_);
    setMETitle(ele1Eta_ele2Eta_, "electron-1 #eta", "electron-2 #eta");
  }

  if (nmuons_ > 1) {
    if (enable2DPlots_) {
      histname = "mu1Pt_mu2Pt";
      histtitle = "muon-1 pt vs muon-2 pt";
      bookME(ibooker, mu1Pt_mu2Pt_, histname, histtitle, muPt_variable_binning_2D_, muPt_variable_binning_2D_);
      setMETitle(mu1Pt_mu2Pt_, "muon-1 pt [GeV]", "muon-2 pt [GeV]");

      histname = "mu1Eta_mu2Eta";
      histtitle = "muon-1 #eta vs muon-2 #eta";
      bookME(ibooker, mu1Eta_mu2Eta_, histname, histtitle, muEta_variable_binning_2D_, muEta_variable_binning_2D_);
      setMETitle(mu1Eta_mu2Eta_, "muon-1 #eta", "muon-2 #eta");
    }
    //george
    histname = "invMass";
    histtitle = "M mu1 mu2";
    bookME(ibooker,
           invMass_mumu_,
           histname,
           histtitle,
           invMass_mumu_binning_.nbins,
           invMass_mumu_binning_.xmin,
           invMass_mumu_binning_.xmax);
    setMETitle(invMass_mumu_, "M(mu1,mu2) [GeV]", "events");
    histname = "invMass_variable";
    histtitle = "M mu1 mu2 variable";
    bookME(ibooker, invMass_mumu_variableBinning_, histname, histtitle, invMass_mumu_variable_binning_);
    setMETitle(invMass_mumu_variableBinning_, "M(mu1,mu2) [GeV]", "events / [GeV]");
  }

  if (HTcut_ > 0) {
    histname = "htVsLS";
    histtitle = "event HT vs LS";
    bookME(ibooker,
           htVsLS_,
           histname,
           histtitle,
           ls_binning_.nbins,
           ls_binning_.xmin,
           ls_binning_.xmax,
           pt_binning_.xmin,
           pt_binning_.xmax);
    setMETitle(htVsLS_, "LS", "event HT [GeV]");

    histname = "eventHT";
    histtitle = "event HT";
    bookME(ibooker, eventHT_, histname, histtitle, HT_binning_.nbins, HT_binning_.xmin, HT_binning_.xmax);
    setMETitle(eventHT_, " event HT [GeV]", "events");
    histname.append("_variableBinning");
    bookME(ibooker, eventHT_variableBinning_, histname, histtitle, HT_variable_binning_);
    setMETitle(eventHT_variableBinning_, "event HT [GeV]", "events");

    if (nelectrons_ > 0 && enable2DPlots_) {
      histname = "elePt_eventHT";
      histtitle = "electron pt vs event HT";
      bookME(ibooker, elePt_eventHT_, histname, histtitle, elePt_variable_binning_2D_, HT_variable_binning_2D_);
      setMETitle(elePt_eventHT_, "leading electron pt", "event HT");
    }
  }

  if (MHTcut_ > 0) {
    //george
    histname = "eventMHT";
    histtitle = "event MHT";
    bookME(ibooker, eventMHT_, histname, histtitle, MHT_binning_.nbins, MHT_binning_.xmin, MHT_binning_.xmax);
    setMETitle(eventMHT_, " event MHT [GeV]", "events");

    histname = "eventMHT_variable";
    histtitle = "event MHT variable";
    bookME(ibooker, eventMHT_variableBinning_, histname, histtitle, MHT_variable_binning_);
    setMETitle(eventMHT_variableBinning_, "event MHT [GeV]", "events / [GeV]");
  }

  //Menglei
  if (enablePhotonPlot_) {
    if ((nmuons_ > 0) && (nphotons_ > 0)) {
      histname = "muPt_phoPt", histtitle = "muon pt vs photon pt";
      bookME(ibooker, muPt_phoPt_, histname, histtitle, muPt_variable_binning_2D_, phoPt_variable_binning_2D_);
      setMETitle(muPt_phoPt_, "muon pt [GeV]", "photon pt [GeV]");

      histname = "muEta_phoEta", histtitle = "muon #eta vs photon #eta";
      bookME(ibooker, muEta_phoEta_, histname, histtitle, muEta_variable_binning_2D_, phoEta_variable_binning_2D_);
      setMETitle(muEta_phoEta_, "muon #eta", "photon #eta");
    }
  }

  for (unsigned int iMu = 0; iMu < nmuons_; ++iMu) {
    std::string index = std::to_string(iMu + 1);

    histname = "muPt_";
    histtitle = "muon p_{T} - ";
    histname.append(index);
    histtitle.append(index);
    bookME(ibooker, muPt_.at(iMu), histname, histtitle, pt_binning_.nbins, pt_binning_.xmin, pt_binning_.xmax);
    setMETitle(muPt_.at(iMu), "muon p_{T} [GeV]", "events");
    histname.append("_variableBinning");
    bookME(ibooker, muPt_variableBinning_.at(iMu), histname, histtitle, muPt_variable_binning_);
    setMETitle(muPt_variableBinning_.at(iMu), "muon p_{T} [GeV]", "events");

    histname = "muEta_";
    histtitle = "muon #eta - ";
    histname.append(index);
    histtitle.append(index);
    bookME(ibooker, muEta_.at(iMu), histname, histtitle, eta_binning_.nbins, eta_binning_.xmin, eta_binning_.xmax);
    setMETitle(muEta_.at(iMu), " muon #eta", "events");
    histname.append("_variableBinning");
    bookME(ibooker, muEta_variableBinning_.at(iMu), histname, histtitle, muEta_variable_binning_);
    setMETitle(muEta_variableBinning_.at(iMu), " muon #eta", "events");

    histname = "muPhi_";
    histtitle = "muon #phi - ";
    histname.append(index);
    histtitle.append(index);
    bookME(ibooker, muPhi_.at(iMu), histname, histtitle, phi_binning_.nbins, phi_binning_.xmin, phi_binning_.xmax);
    setMETitle(muPhi_.at(iMu), " muon #phi", "events");

    if (enable2DPlots_) {
      histname = "muPtEta_";
      histtitle = "muon p_{T} - #eta - ";
      histname.append(index);
      histtitle.append(index);
      bookME(ibooker, muPtEta_.at(iMu), histname, histtitle, muPt_variable_binning_2D_, muEta_variable_binning_2D_);
      setMETitle(muPtEta_.at(iMu), "muon p_{T} [GeV]", "muon #eta");

      histname = "muEtaPhi_";
      histtitle = "muon #eta - #phi - ";
      histname.append(index);
      histtitle.append(index);
      bookME(ibooker, muEtaPhi_.at(iMu), histname, histtitle, muEta_variable_binning_2D_, phi_variable_binning_2D_);
      setMETitle(muEtaPhi_.at(iMu), "muon #eta", "muon #phi");
    }
  }

  for (unsigned int iEle = 0; iEle < nelectrons_; ++iEle) {
    std::string index = std::to_string(iEle + 1);

    histname = "elePt_";
    histtitle = "electron p_{T} - ";
    histname.append(index);
    histtitle.append(index);
    bookME(ibooker, elePt_.at(iEle), histname, histtitle, pt_binning_.nbins, pt_binning_.xmin, pt_binning_.xmax);
    setMETitle(elePt_.at(iEle), "electron p_{T} [GeV]", "events");
    histname.append("_variableBinning");
    bookME(ibooker, elePt_variableBinning_.at(iEle), histname, histtitle, elePt_variable_binning_);
    setMETitle(elePt_variableBinning_.at(iEle), "electron p_{T} [GeV]", "events");

    histname = "eleEta_";
    histtitle = "electron #eta - ";
    histname.append(index);
    histtitle.append(index);
    bookME(ibooker, eleEta_.at(iEle), histname, histtitle, eta_binning_.nbins, eta_binning_.xmin, eta_binning_.xmax);
    setMETitle(eleEta_.at(iEle), " electron #eta", "events");
    histname.append("_variableBinning");
    bookME(ibooker, eleEta_variableBinning_.at(iEle), histname, histtitle, eleEta_variable_binning_);
    setMETitle(eleEta_variableBinning_.at(iEle), "electron #eta", "events");

    histname = "elePhi_";
    histtitle = "electron #phi - ";
    histname.append(index);
    histtitle.append(index);
    bookME(ibooker, elePhi_.at(iEle), histname, histtitle, phi_binning_.nbins, phi_binning_.xmin, phi_binning_.xmax);
    setMETitle(elePhi_.at(iEle), " electron #phi", "events");

    if (enable2DPlots_) {
      histname = "elePtEta_";
      histtitle = "electron p_{T} - #eta - ";
      histname.append(index);
      histtitle.append(index);
      bookME(ibooker, elePtEta_.at(iEle), histname, histtitle, elePt_variable_binning_2D_, eleEta_variable_binning_2D_);
      setMETitle(elePtEta_.at(iEle), "electron p_{T} [GeV]", "electron #eta");

      histname = "eleEtaPhi_";
      histtitle = "electron #eta - #phi - ";
      histname.append(index);
      histtitle.append(index);
      bookME(ibooker, eleEtaPhi_.at(iEle), histname, histtitle, eleEta_variable_binning_2D_, phi_variable_binning_2D_);
      setMETitle(eleEtaPhi_.at(iEle), "electron #eta", "electron #phi");
    }
  }

  //Menglei
  if (enablePhotonPlot_) {
    for (unsigned int iPho(0); iPho < nphotons_; iPho++) {
      std::string index = std::to_string(iPho + 1);

      histname = "phoPt_";
      histtitle = "photon p_{T} - ";
      histname.append(index);
      histtitle.append(index);
      bookME(ibooker, phoPt_[iPho], histname, histtitle, pt_binning_.nbins, pt_binning_.xmin, pt_binning_.xmax);
      setMETitle(phoPt_[iPho], "photon p_{T} [GeV]", "events");

      histname = "phoEta_";
      histtitle = "photon #eta - ";
      histname.append(index);
      histtitle.append(index);
      bookME(ibooker, phoEta_[iPho], histname, histtitle, eta_binning_.nbins, eta_binning_.xmin, eta_binning_.xmax);
      setMETitle(phoEta_[iPho], "photon #eta", "events");

      histname = "phoPhi_";
      histtitle = "photon #phi - ";
      histname.append(index);
      histtitle.append(index);
      bookME(ibooker, phoPhi_[iPho], histname, histtitle, phi_binning_.nbins, phi_binning_.xmin, phi_binning_.xmax);
      setMETitle(phoPhi_[iPho], "photon #phi", "events");

      histname = "phoPtEta_";
      histtitle = "photon p_{T} - #eta - ";
      histname.append(index);
      histtitle.append(index);
      bookME(ibooker, phoPtEta_[iPho], histname, histtitle, phoPt_variable_binning_2D_, phoEta_variable_binning_2D_);
      setMETitle(phoPtEta_[iPho], "photon p_{T} [GeV]", "photon #eta");

      histname = "phoEtaPhi_";
      histtitle = "photon #eta - #phi - ";
      histname.append(index);
      histtitle.append(index);
      bookME(ibooker, phoEtaPhi_[iPho], histname, histtitle, phoEta_variable_binning_2D_, phi_variable_binning_2D_);
      setMETitle(phoEtaPhi_[iPho], "photon #eta", "photon #phi");
    }
  }

  for (unsigned int iJet = 0; iJet < njets_; ++iJet) {
    std::string index = std::to_string(iJet + 1);

    histname = "jetPt_";
    histtitle = "jet p_{T} - ";
    histname.append(index);
    histtitle.append(index);
    bookME(ibooker, jetPt_.at(iJet), histname, histtitle, pt_binning_.nbins, pt_binning_.xmin, pt_binning_.xmax);
    setMETitle(jetPt_.at(iJet), "jet p_{T} [GeV]", "events");
    histname.append("_variableBinning");
    bookME(ibooker, jetPt_variableBinning_.at(iJet), histname, histtitle, jetPt_variable_binning_);
    setMETitle(jetPt_variableBinning_.at(iJet), "jet p_{T} [GeV]", "events");

    histname = "jetEta_";
    histtitle = "jet #eta - ";
    histname.append(index);
    histtitle.append(index);
    bookME(ibooker, jetEta_.at(iJet), histname, histtitle, eta_binning_.nbins, eta_binning_.xmin, eta_binning_.xmax);
    setMETitle(jetEta_.at(iJet), "jet #eta", "events");
    histname.append("_variableBinning");
    bookME(ibooker, jetEta_variableBinning_.at(iJet), histname, histtitle, jetEta_variable_binning_);
    setMETitle(jetEta_variableBinning_.at(iJet), "jet #eta", "events");

    histname = "jetPhi_";
    histtitle = "jet #phi - ";
    histname.append(index);
    histtitle.append(index);
    bookME(ibooker, jetPhi_.at(iJet), histname, histtitle, phi_binning_.nbins, phi_binning_.xmin, phi_binning_.xmax);
    setMETitle(jetPhi_.at(iJet), "jet #phi", "events");

    if (enable2DPlots_) {
      histname = "jetPtEta_";
      histtitle = "jet p_{T} - #eta - ";
      histname.append(index);
      histtitle.append(index);
      bookME(ibooker, jetPtEta_.at(iJet), histname, histtitle, jetPt_variable_binning_2D_, jetEta_variable_binning_2D_);
      setMETitle(jetPtEta_.at(iJet), "jet p_{T} [GeV]", "jet #eta");

      histname = "jetEtaPhi_";
      histtitle = "jet #eta - #phi - ";
      histname.append(index);
      histtitle.append(index);
      bookME(ibooker, jetEtaPhi_.at(iJet), histname, histtitle, jetEta_variable_binning_2D_, phi_variable_binning_2D_);
      setMETitle(jetEtaPhi_.at(iJet), "jet #eta", "jet #phi");
    }
  }

  // Marina
  for (unsigned int iBJet = 0; iBJet < nbjets_; ++iBJet) {
    std::string index = std::to_string(iBJet + 1);

    histname = "bjetPt_";
    histtitle = "b-jet p_{T} - ";
    histname.append(index);
    histtitle.append(index);
    bookME(ibooker, bjetPt_.at(iBJet), histname, histtitle, pt_binning_.nbins, pt_binning_.xmin, pt_binning_.xmax);
    setMETitle(bjetPt_.at(iBJet), "b-jet p_{T} [GeV]", "events");
    histname.append("_variableBinning");
    bookME(ibooker, bjetPt_variableBinning_.at(iBJet), histname, histtitle, jetPt_variable_binning_);
    setMETitle(bjetPt_variableBinning_.at(iBJet), "b-jet p_{T} [GeV]", "events");

    histname = "bjetEta_";
    histtitle = "b-jet #eta - ";
    histname.append(index);
    histtitle.append(index);
    bookME(ibooker, bjetEta_.at(iBJet), histname, histtitle, eta_binning_.nbins, eta_binning_.xmin, eta_binning_.xmax);
    setMETitle(bjetEta_.at(iBJet), "b-jet #eta", "events");
    histname.append("_variableBinning");
    bookME(ibooker, bjetEta_variableBinning_.at(iBJet), histname, histtitle, jetEta_variable_binning_);
    setMETitle(bjetEta_variableBinning_.at(iBJet), "b-jet #eta", "events");

    histname = "bjetPhi_";
    histtitle = "b-jet #phi - ";
    histname.append(index);
    histtitle.append(index);
    bookME(ibooker, bjetPhi_.at(iBJet), histname, histtitle, phi_binning_.nbins, phi_binning_.xmin, phi_binning_.xmax);
    setMETitle(bjetPhi_.at(iBJet), "b-jet #phi", "events");

    histname = "bjetCSV_";
    histtitle = "b-jet CSV - ";
    histname.append(index);
    histtitle.append(index);
    bookME(ibooker, bjetCSV_.at(iBJet), histname, histtitle, csv_binning_.nbins, csv_binning_.xmin, csv_binning_.xmax);
    setMETitle(bjetCSV_.at(iBJet), "b-jet CSV", "events");

    if (enable2DPlots_) {
      histname = "bjetPtEta_";
      histtitle = "b-jet p_{T} - #eta - ";
      histname.append(index);
      histtitle.append(index);
      bookME(
          ibooker, bjetPtEta_.at(iBJet), histname, histtitle, jetPt_variable_binning_2D_, jetEta_variable_binning_2D_);
      setMETitle(bjetPtEta_.at(iBJet), "b-jet p_{T} [GeV]", "b-jet #eta");

      histname = "bjetEtaPhi_";
      histtitle = "b-jet #eta - #phi - ";
      histname.append(index);
      histtitle.append(index);
      bookME(
          ibooker, bjetEtaPhi_.at(iBJet), histname, histtitle, jetEta_variable_binning_2D_, phi_variable_binning_2D_);
      setMETitle(bjetEtaPhi_.at(iBJet), "b-jet #eta", "b-jet #phi");
    }

    histname = "bjetCSVHT_";
    histtitle = "HT - b-jet CSV - ";
    histname.append(index);
    histtitle.append(index);
    bookME(ibooker,
           bjetCSVHT_.at(iBJet),
           histname,
           histtitle,
           csv_binning_.nbins,
           csv_binning_.xmin,
           csv_binning_.xmax,
           HT_binning_.nbins,
           HT_binning_.xmin,
           HT_binning_.xmax);
    setMETitle(bjetCSVHT_.at(iBJet), "b-jet CSV", "event HT [GeV]");
  }
}

void TopMonitor::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) {
  // if valid HLT paths are required,
  // analyze event only if all paths are valid
  if (requireValidHLTPaths_ and (not hltPathsAreValid_)) {
    return;
  }

  // Filter out events if Trigger Filtering is requested
  if (den_genTriggerEventFlag_->on() && !den_genTriggerEventFlag_->accept(iEvent, iSetup)) {
    return;
  }

  edm::Handle<reco::VertexCollection> primaryVertices;
  iEvent.getByToken(vtxToken_, primaryVertices);
  //Primary Vertex selection
  const reco::Vertex* pv = nullptr;
  for (auto const& v : *primaryVertices) {
    if (!vtxSelection_(v)) {
      continue;
    }
    pv = &v;
    break;
  }
  if (applyLeptonPVcuts_ && (pv == nullptr)) {
    edm::LogWarning("TopMonitor") << "Invalid handle to reco::VertexCollection, event will be skipped";
    return;
  }

  edm::Handle<reco::PFMETCollection> metHandle;
  iEvent.getByToken(metToken_, metHandle);
  if ((not metHandle.isValid()) && enableMETPlot_) {
    edm::LogWarning("TopMonitor") << "MET handle not valid \n";
    return;
  }

  double met_pt(-99.);
  double met_phi(-99.);

  if (enableMETPlot_) {
    const reco::PFMET& pfmet = metHandle->front();

    if (!metSelection_(pfmet)) {
      return;
    }

    met_pt = pfmet.pt();
    met_phi = pfmet.phi();
  }

  edm::Handle<edm::View<reco::GsfElectron> > eleHandle;
  iEvent.getByToken(eleToken_, eleHandle);
  if (!eleHandle.isValid() && nelectrons_ > 0) {
    edm::LogWarning("TopMonitor") << "Electron handle not valid \n";
    return;
  }

  edm::Handle<edm::ValueMap<bool> > eleIDHandle;
  iEvent.getByToken(elecIDToken_, eleIDHandle);
  if (!eleIDHandle.isValid() && nelectrons_ > 0) {
    edm::LogWarning("TopMonitor") << "Electron ID handle not valid \n";
    return;
  }

  std::vector<reco::GsfElectron> electrons;
  if (nelectrons_ > 0) {
    if (eleHandle->size() < nelectrons_) {
      return;
    }

    for (size_t index = 0; index < eleHandle->size(); index++) {
      const auto e = eleHandle->at(index);
      const auto el = eleHandle->ptrAt(index);

      bool pass_id = (*eleIDHandle)[el];

      if (eleSelection_(e) && pass_id) {
        electrons.push_back(e);
      }

      if (applyLeptonPVcuts_ && ((std::fabs(e.gsfTrack()->dxy(pv->position())) >= lepPVcuts_.dxy) ||
                                 (std::fabs(e.gsfTrack()->dz(pv->position())) >= lepPVcuts_.dz))) {
        continue;
      }
    }

    if (electrons.size() < nelectrons_) {
      return;
    }
  }

  edm::Handle<reco::MuonCollection> muoHandle;
  iEvent.getByToken(muoToken_, muoHandle);
  if (!muoHandle.isValid() && nmuons_ > 0) {
    edm::LogWarning("TopMonitor") << "Muon handle not valid \n";
    return;
  }

  if (muoHandle->size() < nmuons_) {
    return;
  }

  std::vector<reco::Muon> muons;
  if (nmuons_ > 0) {
    for (auto const& m : *muoHandle) {
      if (muoSelection_(m)) {
        muons.push_back(m);
      }

      if (applyLeptonPVcuts_ && ((std::fabs(m.muonBestTrack()->dxy(pv->position())) >= lepPVcuts_.dxy) ||
                                 (std::fabs(m.muonBestTrack()->dz(pv->position())) >= lepPVcuts_.dz))) {
        continue;
      }
    }

    if (muons.size() < nmuons_) {
      return;
    }
  }

  double mll(-2);
  if (nmuons_ > 1) {
    mll = (muons[0].p4() + muons[1].p4()).M();

    if ((invMassUppercut_ > -1) && (invMassLowercut_ > -1) && ((mll > invMassUppercut_) || (mll < invMassLowercut_))) {
      return;
    }
    if (opsign_ && (muons[0].charge() == muons[1].charge())) {
      return;
    }
  }

  edm::Handle<reco::PhotonCollection> phoHandle;
  iEvent.getByToken(phoToken_, phoHandle);
  if (!phoHandle.isValid()) {
    edm::LogWarning("TopMonitor") << "Photon handle not valid \n";
    return;
  }
  if (phoHandle->size() < nphotons_) {
    return;
  }

  std::vector<reco::Photon> photons;
  for (auto const& p : *phoHandle) {
    if (phoSelection_(p)) {
      photons.push_back(p);
    }
  }
  if (photons.size() < nphotons_) {
    return;
  }

  double eventHT(0.);
  math::XYZTLorentzVector eventMHT(0., 0., 0., 0.);

  edm::Handle<reco::PFJetCollection> jetHandle;
  iEvent.getByToken(jetToken_, jetHandle);
  if (!jetHandle.isValid() && njets_ > 0) {
    edm::LogWarning("TopMonitor") << "Jet handle not valid \n";
    return;
  }
  std::vector<reco::PFJet> jets;
  if (njets_ > 0) {
    if (jetHandle->size() < njets_)
      return;
    for (auto const& j : *jetHandle) {
      if (HTdefinition_(j)) {
        eventHT += j.pt();
      }
      if (MHTdefinition_(j)) {
        eventMHT += j.p4();
      }
      if (jetSelection_(j)) {
        bool isJetOverlappedWithLepton = false;
        if (nmuons_ > 0) {
          for (auto const& m : muons) {
            if (deltaR(j, m) < leptJetDeltaRmin_) {
              isJetOverlappedWithLepton = true;
              break;
            }
          }
        }
        if (isJetOverlappedWithLepton)
          continue;
        if (nelectrons_ > 0) {
          for (auto const& e : electrons) {
            if (deltaR(j, e) < leptJetDeltaRmin_) {
              isJetOverlappedWithLepton = true;
              break;
            }
          }
        }
        if (isJetOverlappedWithLepton)
          continue;
        jets.push_back(j);
      }
    }
    if (jets.size() < njets_)
      return;
  }

  if (eventHT < HTcut_) {
    return;
  }

  if ((MHTcut_ > 0) && (eventMHT.pt() < MHTcut_)) {
    return;
  }

  bool allpairs = false;
  if (nmuons_ > 2) {
    double mumu_mass;
    for (unsigned int idx = 0; idx < muons.size(); idx++) {
      for (unsigned int idx2 = idx + 1; idx2 < muons.size(); idx2++) {
        //compute inv mass of two different leptons
        mumu_mass = (muons[idx2].p4() + muons[idx2].p4()).M();
        if (mumu_mass < invMassLowercut_ || mumu_mass > invMassUppercut_)
          allpairs = true;
      }
    }
  }
  //cut only if enabled and the event has a pair that failed the mll range
  if (allpairs && invMassCutInAllMuPairs_) {
    return;
  }

  JetTagMap bjets;

  if (nbjets_ > 0) {
    // map of Jet,btagValues (for all jets passing bJetSelection_)
    //  - btagValue of each jet is calculated as sum of values from InputTags in jetTagTokens_
    JetTagMap allJetBTagVals;

    for (const auto& jetTagToken : jetTagTokens_) {
      edm::Handle<reco::JetTagCollection> bjetHandle;
      iEvent.getByToken(jetTagToken, bjetHandle);
      if (not bjetHandle.isValid()) {
        edm::LogWarning("TopMonitor") << "B-Jet handle not valid, will skip event \n";
        return;
      }

      const reco::JetTagCollection& bTags = *(bjetHandle.product());

      for (const auto& i_jetTag : bTags) {
        const auto& jetRef = i_jetTag.first;

        if (not bjetSelection_(*(jetRef.get()))) {
          continue;
        }

        const auto btagVal = i_jetTag.second;

        if (not std::isfinite(btagVal)) {
          continue;
        }

        if (allJetBTagVals.find(jetRef) != allJetBTagVals.end()) {
          allJetBTagVals.at(jetRef) += btagVal;
        } else {
          allJetBTagVals.insert(JetTagMap::value_type(jetRef, btagVal));
        }
      }
    }

    for (const auto& jetBTagVal : allJetBTagVals) {
      if (jetBTagVal.second < workingpoint_) {
        continue;
      }

      bjets.insert(JetTagMap::value_type(jetBTagVal.first, jetBTagVal.second));
    }

    if (bjets.size() < nbjets_) {
      return;
    }
  }

  if (nbjets_ > 1) {
    double deltaEta = std::abs(bjets.begin()->first->eta() - (++bjets.begin())->first->eta());
    if (deltaEta > bJetDeltaEtaMax_)
      return;
  }

  if ((nbjets_ > 0) && (nmuons_ > 0)) {
    bool foundMuonInsideJet = false;
    for (auto const& bjet : bjets) {
      for (auto const& mu : muons) {
        double dR = deltaR(*bjet.first, mu);
        if (dR < bJetMuDeltaRmax_) {
          foundMuonInsideJet = true;
          break;
        }
      }
      if (foundMuonInsideJet)
        break;
    }

    if (!foundMuonInsideJet)
      return;
  }

  const int ls = iEvent.id().luminosityBlock();

  // numerator condition
  const bool trg_passed = (num_genTriggerEventFlag_->on() && num_genTriggerEventFlag_->accept(iEvent, iSetup));

  if (enableMETPlot_) {
    metME_.fill(trg_passed, met_pt);
    metME_variableBinning_.fill(trg_passed, met_pt);
    metPhiME_.fill(trg_passed, met_phi);
    metVsLS_.fill(trg_passed, ls, met_pt);
  }
  if (HTcut_ > 0) {
    eventHT_.fill(trg_passed, eventHT);
    eventHT_variableBinning_.fill(trg_passed, eventHT);
    htVsLS_.fill(trg_passed, ls, eventHT);
  }
  //george
  if (MHTcut_ > 0) {
    eventMHT_.fill(trg_passed, eventMHT.pt());
    eventMHT_variableBinning_.fill(trg_passed, eventMHT.pt());
  }

  if (njets_ > 0) {
    jetMulti_.fill(trg_passed, jets.size());
    jetEtaPhi_HEP17_.fill(trg_passed, jets.at(0).eta(), jets.at(0).phi());  // for HEP17 monitorning
    jetVsLS_.fill(trg_passed, ls, jets.at(0).pt());
  }

  if (enablePhotonPlot_) {
    phoMulti_.fill(trg_passed, photons.size());
  }

  // Marina
  if (nbjets_ > 0) {
    bjetMulti_.fill(trg_passed, bjets.size());
    bjetVsLS_.fill(trg_passed, ls, bjets.begin()->first->pt());
  }

  if (nmuons_ > 0) {
    muMulti_.fill(trg_passed, muons.size());
    muVsLS_.fill(trg_passed, ls, muons.at(0).pt());
    if (nmuons_ > 1) {
      mu1Pt_mu2Pt_.fill(trg_passed, muons.at(0).pt(), muons.at(1).pt());
      mu1Eta_mu2Eta_.fill(trg_passed, muons.at(0).eta(), muons.at(1).eta());
      invMass_mumu_.fill(trg_passed, mll);
      invMass_mumu_variableBinning_.fill(trg_passed, mll);
    }
    if (njets_ > 0) {
      DeltaR_jet_Mu_.fill(trg_passed, deltaR(jets.at(0), muons.at(0)));
    }
  }

  if (nelectrons_ > 0) {
    eleMulti_.fill(trg_passed, electrons.size());
    eleVsLS_.fill(trg_passed, ls, electrons.at(0).pt());
    if (HTcut_ > 0)
      elePt_eventHT_.fill(trg_passed, electrons.at(0).pt(), eventHT);
    if (njets_ > 0)
      elePt_jetPt_.fill(trg_passed, electrons.at(0).pt(), jets.at(0).pt());
    if (nmuons_ > 0) {
      elePt_muPt_.fill(trg_passed, electrons.at(0).pt(), muons.at(0).pt());
      eleEta_muEta_.fill(trg_passed, electrons.at(0).eta(), muons.at(0).eta());
    }
    if (nelectrons_ > 1) {
      ele1Pt_ele2Pt_.fill(trg_passed, electrons.at(0).pt(), electrons.at(1).pt());
      ele1Eta_ele2Eta_.fill(trg_passed, electrons.at(0).eta(), electrons.at(1).eta());
    }
  }

  if (enablePhotonPlot_) {
    if (nphotons_ > 0) {
      phoVsLS_.fill(trg_passed, ls, photons.at(0).pt());
      if (nmuons_ > 0) {
        muPt_phoPt_.fill(trg_passed, muons.at(0).pt(), photons.at(0).pt());
        muEta_phoEta_.fill(trg_passed, muons.at(0).eta(), photons.at(0).eta());
      }
    }
  }

  for (unsigned int iMu = 0; iMu < muons.size(); ++iMu) {
    if (iMu >= nmuons_)
      break;
    muPhi_.at(iMu).fill(trg_passed, muons.at(iMu).phi());
    muEta_.at(iMu).fill(trg_passed, muons.at(iMu).eta());
    muPt_.at(iMu).fill(trg_passed, muons.at(iMu).pt());
    muEta_variableBinning_.at(iMu).fill(trg_passed, muons.at(iMu).eta());
    muPt_variableBinning_.at(iMu).fill(trg_passed, muons.at(iMu).pt());
    muPtEta_.at(iMu).fill(trg_passed, muons.at(iMu).pt(), muons.at(iMu).eta());
    muEtaPhi_.at(iMu).fill(trg_passed, muons.at(iMu).eta(), muons.at(iMu).phi());
  }
  for (unsigned int iEle = 0; iEle < electrons.size(); ++iEle) {
    if (iEle >= nelectrons_)
      break;
    elePhi_.at(iEle).fill(trg_passed, electrons.at(iEle).phi());
    eleEta_.at(iEle).fill(trg_passed, electrons.at(iEle).eta());
    elePt_.at(iEle).fill(trg_passed, electrons.at(iEle).pt());
    eleEta_variableBinning_.at(iEle).fill(trg_passed, electrons.at(iEle).eta());
    elePt_variableBinning_.at(iEle).fill(trg_passed, electrons.at(iEle).pt());
    elePtEta_.at(iEle).fill(trg_passed, electrons.at(iEle).pt(), electrons.at(iEle).eta());
    eleEtaPhi_.at(iEle).fill(trg_passed, electrons.at(iEle).eta(), electrons.at(iEle).phi());
  }
  //Menglei
  if (enablePhotonPlot_) {
    for (unsigned int iPho = 0; iPho < photons.size(); ++iPho) {
      if (iPho >= nphotons_)
        break;
      phoPhi_[iPho].fill(trg_passed, photons[iPho].phi());
      phoEta_[iPho].fill(trg_passed, photons[iPho].eta());
      phoPt_[iPho].fill(trg_passed, photons[iPho].pt());
      phoPtEta_[iPho].fill(trg_passed, photons[iPho].pt(), photons[iPho].eta());
      phoEtaPhi_[iPho].fill(trg_passed, photons[iPho].eta(), photons[iPho].phi());
    }
  }

  for (unsigned int iJet = 0; iJet < jets.size(); ++iJet) {
    if (iJet >= njets_)
      break;
    jetPhi_.at(iJet).fill(trg_passed, jets.at(iJet).phi());
    jetEta_.at(iJet).fill(trg_passed, jets.at(iJet).eta());
    jetPt_.at(iJet).fill(trg_passed, jets.at(iJet).pt());
    jetEta_variableBinning_.at(iJet).fill(trg_passed, jets.at(iJet).eta());
    jetPt_variableBinning_.at(iJet).fill(trg_passed, jets.at(iJet).pt());
    jetPtEta_.at(iJet).fill(trg_passed, jets.at(iJet).pt(), jets.at(iJet).eta());
    jetEtaPhi_.at(iJet).fill(trg_passed, jets.at(iJet).eta(), jets.at(iJet).phi());
  }

  // Marina
  unsigned int iBJet = 0;
  for (auto& bjet : bjets) {
    if (iBJet >= nbjets_)
      break;

    bjetPhi_.at(iBJet).fill(trg_passed, bjet.first->phi());
    bjetEta_.at(iBJet).fill(trg_passed, bjet.first->eta());
    bjetPt_.at(iBJet).fill(trg_passed, bjet.first->pt());
    bjetCSV_.at(iBJet).fill(trg_passed, std::fmax(0.0, bjet.second));
    bjetEta_variableBinning_.at(iBJet).fill(trg_passed, bjet.first->eta());
    bjetPt_variableBinning_.at(iBJet).fill(trg_passed, bjet.first->pt());
    bjetPtEta_.at(iBJet).fill(trg_passed, bjet.first->pt(), bjet.first->eta());
    bjetEtaPhi_.at(iBJet).fill(trg_passed, bjet.first->eta(), bjet.first->phi());
    bjetCSVHT_.at(iBJet).fill(trg_passed, std::fmax(0.0, bjet.second), eventHT);

    iBJet++;
  }
}

void TopMonitor::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("FolderName", "HLT/TOP");

  desc.add<bool>("requireValidHLTPaths", true);

  desc.add<edm::InputTag>("vertices", edm::InputTag("offlinePrimaryVertices"));
  desc.add<edm::InputTag>("muons", edm::InputTag("muons"));
  desc.add<edm::InputTag>("electrons", edm::InputTag("gedGsfElectrons"));
  desc.add<edm::InputTag>("elecID",
                          edm::InputTag("egmGsfElectronIDsForDQM:cutBasedElectronID-RunIIIWinter22-V1-tight"));
  desc.add<edm::InputTag>("photons", edm::InputTag("photons"));
  desc.add<edm::InputTag>("jets", edm::InputTag("ak4PFJetsCHS"));
  desc.add<std::vector<edm::InputTag> >(
      "btagAlgos", {edm::InputTag("pfDeepCSVJetTags:probb"), edm::InputTag("pfDeepCSVJetTags:probbb")});
  desc.add<edm::InputTag>("met", edm::InputTag("pfMet"));

  desc.add<std::string>("metSelection", "pt > 0");
  desc.add<std::string>("jetSelection", "pt > 0");
  desc.add<std::string>("eleSelection", "pt > 0");
  desc.add<std::string>("muoSelection", "pt > 0");
  desc.add<std::string>("phoSelection", "pt > 0");
  desc.add<std::string>("HTdefinition", "pt > 0");
  desc.add<std::string>("vertexSelection", "!isFake");
  desc.add<std::string>("bjetSelection", "pt > 0");
  desc.add<unsigned int>("njets", 0);
  desc.add<unsigned int>("nelectrons", 0);
  desc.add<unsigned int>("nmuons", 0);
  desc.add<unsigned int>("nphotons", 0);
  desc.add<double>("leptJetDeltaRmin", 0);
  desc.add<double>("bJetMuDeltaRmax", 9999.);
  desc.add<double>("bJetDeltaEtaMax", 9999.);
  desc.add<double>("HTcut", 0);

  desc.add<unsigned int>("nbjets", 0);
  desc.add<double>("workingpoint", 0.4941);  // DeepCSV Medium wp
  desc.add<bool>("applyLeptonPVcuts", false);
  desc.add<double>("invMassUppercut", -1.0);
  desc.add<double>("invMassLowercut", -1.0);
  desc.add<bool>("oppositeSignMuons", false);
  desc.add<std::string>("MHTdefinition", "pt > 0");
  desc.add<double>("MHTcut", -1);
  desc.add<bool>("invMassCutInAllMuPairs", false);
  desc.add<bool>("enablePhotonPlot", false);
  desc.add<bool>("enableMETPlot", false);
  desc.add<bool>("enable2DPlots", true);

  edm::ParameterSetDescription genericTriggerEventPSet;
  GenericTriggerEventFlag::fillPSetDescription(genericTriggerEventPSet);
  desc.add<edm::ParameterSetDescription>("numGenericTriggerEventPSet", genericTriggerEventPSet);
  desc.add<edm::ParameterSetDescription>("denGenericTriggerEventPSet", genericTriggerEventPSet);

  edm::ParameterSetDescription histoPSet;
  edm::ParameterSetDescription metPSet;
  edm::ParameterSetDescription phiPSet;
  edm::ParameterSetDescription etaPSet;
  edm::ParameterSetDescription ptPSet;
  edm::ParameterSetDescription htPSet;
  edm::ParameterSetDescription DRPSet;
  edm::ParameterSetDescription csvPSet;
  edm::ParameterSetDescription invMassPSet;
  edm::ParameterSetDescription MHTPSet;
  fillHistoPSetDescription(metPSet);
  fillHistoPSetDescription(phiPSet);
  fillHistoPSetDescription(ptPSet);
  fillHistoPSetDescription(etaPSet);
  fillHistoPSetDescription(htPSet);
  fillHistoPSetDescription(DRPSet);
  fillHistoPSetDescription(csvPSet);
  fillHistoPSetDescription(MHTPSet);
  fillHistoPSetDescription(invMassPSet);
  histoPSet.add<edm::ParameterSetDescription>("metPSet", metPSet);
  histoPSet.add<edm::ParameterSetDescription>("etaPSet", etaPSet);
  histoPSet.add<edm::ParameterSetDescription>("phiPSet", phiPSet);
  histoPSet.add<edm::ParameterSetDescription>("ptPSet", ptPSet);
  histoPSet.add<edm::ParameterSetDescription>("htPSet", htPSet);
  histoPSet.add<edm::ParameterSetDescription>("DRPSet", DRPSet);
  histoPSet.add<edm::ParameterSetDescription>("csvPSet", csvPSet);
  histoPSet.add<edm::ParameterSetDescription>("invMassPSet", invMassPSet);
  histoPSet.add<edm::ParameterSetDescription>("MHTPSet", MHTPSet);

  std::vector<double> bins = {0.,   20.,  40.,  60.,  80.,  90.,  100., 110., 120., 130., 140., 150., 160.,
                              170., 180., 190., 200., 220., 240., 260., 280., 300., 350., 400., 450., 1000.};
  std::vector<double> eta_bins = {-3., -2.5, -2., -1.5, -1., -.5, 0., .5, 1., 1.5, 2., 2.5, 3.};
  histoPSet.add<std::vector<double> >("metBinning", bins);
  histoPSet.add<std::vector<double> >("HTBinning", bins);
  histoPSet.add<std::vector<double> >("jetPtBinning", bins);
  histoPSet.add<std::vector<double> >("elePtBinning", bins);
  histoPSet.add<std::vector<double> >("muPtBinning", bins);
  histoPSet.add<std::vector<double> >("jetEtaBinning", eta_bins);
  histoPSet.add<std::vector<double> >("eleEtaBinning", eta_bins);
  histoPSet.add<std::vector<double> >("muEtaBinning", eta_bins);
  histoPSet.add<std::vector<double> >("invMassVariableBinning", bins);
  histoPSet.add<std::vector<double> >("MHTVariableBinning", bins);

  std::vector<double> bins_2D = {0., 40., 80., 100., 120., 140., 160., 180., 200., 240., 280., 350., 450., 1000.};
  std::vector<double> eta_bins_2D = {-3., -2., -1., 0., 1., 2., 3.};
  std::vector<double> phi_bins_2D = {
      -3.1415, -2.5132, -1.8849, -1.2566, -0.6283, 0, 0.6283, 1.2566, 1.8849, 2.5132, 3.1415};
  histoPSet.add<std::vector<double> >("HTBinning2D", bins_2D);
  histoPSet.add<std::vector<double> >("jetPtBinning2D", bins_2D);
  histoPSet.add<std::vector<double> >("elePtBinning2D", bins_2D);
  histoPSet.add<std::vector<double> >("muPtBinning2D", bins_2D);
  histoPSet.add<std::vector<double> >("phoPtBinning2D", bins_2D);
  histoPSet.add<std::vector<double> >("jetEtaBinning2D", eta_bins_2D);
  histoPSet.add<std::vector<double> >("eleEtaBinning2D", eta_bins_2D);
  histoPSet.add<std::vector<double> >("muEtaBinning2D", eta_bins_2D);
  histoPSet.add<std::vector<double> >("phoEtaBinning2D", eta_bins_2D);
  histoPSet.add<std::vector<double> >("phiBinning2D", phi_bins_2D);

  edm::ParameterSetDescription lsPSet;
  fillHistoLSPSetDescription(lsPSet);
  histoPSet.add<edm::ParameterSetDescription>("lsPSet", lsPSet);

  desc.add<edm::ParameterSetDescription>("histoPSet", histoPSet);

  edm::ParameterSetDescription lPVcutPSet;
  lPVcutPSet.add<double>("dxy", 9999.);
  lPVcutPSet.add<double>("dz", 9999.);
  desc.add<edm::ParameterSetDescription>("leptonPVcuts", lPVcutPSet);

  descriptions.add("topMonitoring", desc);
}

// Define this as a plug-in
DEFINE_FWK_MODULE(TopMonitor);
