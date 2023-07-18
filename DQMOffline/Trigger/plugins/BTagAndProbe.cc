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

//Tagging variables
#include "DataFormats/BTauReco/interface/ShallowTagInfo.h"

#include <string>
#include <vector>
#include <memory>
#include <map>

class BTagAndProbe : public DQMEDAnalyzer, public TriggerDQMBase {
public:
  typedef dqm::reco::MonitorElement MonitorElement;
  typedef dqm::reco::DQMStore DQMStore;

  BTagAndProbe(const edm::ParameterSet&);
  ~BTagAndProbe() throw() override;
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

  const edm::EDGetTokenT<reco::VertexCollection> vtxToken_;
  const edm::EDGetTokenT<reco::MuonCollection> muoToken_;
  const edm::EDGetTokenT<edm::View<reco::GsfElectron> > eleToken_;
  const edm::EDGetTokenT<edm::ValueMap<bool> > elecIDToken_;
  const std::vector<edm::EDGetTokenT<reco::JetTagCollection> > jetTagTokens_;

  //Tag info
  const edm::EDGetTokenT<std::vector<reco::ShallowTagInfo> > shallowTagInfosToken_;
  struct PVcut {
    double dxy;
    double dz;

    PVcut(double dxy_, double dz_) {
      dxy = dxy_;
      dz = dz_;
    }
  };

  // for the tag and probe
  MonitorElement* h_nElectrons1 = nullptr;
  MonitorElement* h_nElectrons2 = nullptr;
  MonitorElement* h_nElectrons3 = nullptr;
  MonitorElement* h_nElectrons4 = nullptr;
  MonitorElement* h_nElectrons5 = nullptr;
  MonitorElement* h_nElectrons6 = nullptr;
  MonitorElement* h_nElectrons7 = nullptr;
  MonitorElement* h_nElectrons8 = nullptr;

  MonitorElement* h_nMuons1 = nullptr;
  MonitorElement* h_nMuons2 = nullptr;
  MonitorElement* h_nMuons3 = nullptr;
  MonitorElement* h_nMuons4 = nullptr;
  MonitorElement* h_nMuons5 = nullptr;
  MonitorElement* h_nMuons6 = nullptr;
  MonitorElement* h_nMuons7 = nullptr;

  MonitorElement* h_nJets = nullptr;
  MonitorElement* h_btagVal = nullptr;
  MonitorElement* h_btagVal2 = nullptr;
  MonitorElement* h_btagVal_pp = nullptr;
  MonitorElement* h_btagVal_pf = nullptr;
  MonitorElement* h_btagVal_pa = nullptr;

  MonitorElement* h_nJets1 = nullptr;
  MonitorElement* h_nJets2 = nullptr;
  MonitorElement* h_nJets3 = nullptr;
  MonitorElement* h_nJets4 = nullptr;
  MonitorElement* h_nJets5 = nullptr;
  MonitorElement* h_nJets6 = nullptr;
  MonitorElement* h_nJets7 = nullptr;
  MonitorElement* h_nJets8 = nullptr;
  MonitorElement* h_nJets9 = nullptr;
  MonitorElement* h_nJets10 = nullptr;
  MonitorElement* h_nJets11 = nullptr;
  MonitorElement* h_nJets12 = nullptr;

  //muon pt
  MonitorElement* h_Muons1_pt = nullptr;
  MonitorElement* h_Muons2_pt = nullptr;
  MonitorElement* h_Muons3_pt = nullptr;
  MonitorElement* h_Muons4_pt = nullptr;
  MonitorElement* h_Muons5_pt = nullptr;

  //muon eta
  MonitorElement* h_Muons1_eta = nullptr;
  MonitorElement* h_Muons2_eta = nullptr;
  MonitorElement* h_Muons3_eta = nullptr;
  MonitorElement* h_Muons4_eta = nullptr;
  MonitorElement* h_Muons5_eta = nullptr;

  //electron pt
  MonitorElement* h_Electrons1_pt = nullptr;
  MonitorElement* h_Electrons2_pt = nullptr;
  MonitorElement* h_Electrons3_pt = nullptr;
  MonitorElement* h_Electrons4_pt = nullptr;

  //electron eta
  MonitorElement* h_Electrons1_eta = nullptr;
  MonitorElement* h_Electrons2_eta = nullptr;
  MonitorElement* h_Electrons3_eta = nullptr;
  MonitorElement* h_Electrons4_eta = nullptr;

  MonitorElement* cutFlow = nullptr;

  // new for tnp
  ObjME jetNSecondaryVertices_;
  ObjME jet_tagVal_;
  ObjME jet_pt_;
  ObjME jet_eta_;
  ObjME trackSumJetEtRatio_;
  ObjME trackSip2dValAboveCharm_;
  ObjME trackSip2dSigAboveCharm_;
  ObjME trackSip3dValAboveCharm_;
  ObjME trackSip3dSigAboveCharm_;
  ObjME jetNTracksEtaRel_;
  ObjME jetNSelectedTracks_;
  ObjME vertexCategory_;
  ObjME trackSumJetDeltaR_;

  ObjME trackJetDistVal_;
  ObjME trackPtRel_;
  ObjME trackDeltaR_;
  ObjME trackPtRatio_;
  ObjME trackSip3dSig_;
  ObjME trackSip2dSig_;
  ObjME trackDecayLenVal_;
  ObjME trackEtaRel_;

  ObjME vertexMass_;
  ObjME vertexNTracks_;
  ObjME vertexEnergyRatio_;
  ObjME vertexJetDeltaR_;
  ObjME flightDistance2dVal_;
  ObjME flightDistance3dVal_;
  ObjME flightDistance2dSig_;
  ObjME flightDistance3dSig_;

  std::unique_ptr<GenericTriggerEventFlag> genTriggerEventFlag_;

  StringCutObjectSelector<reco::PFJet, true> jetSelection_;
  StringCutObjectSelector<reco::GsfElectron, true> eleSelection_;
  StringCutObjectSelector<reco::Muon, true> muoSelection_;

  StringCutObjectSelector<reco::Vertex, true> vtxSelection_;

  StringCutObjectSelector<reco::Jet, true> bjetSelection_;

  const unsigned int nelectrons_;
  const unsigned int nmuons_;
  const double leptJetDeltaRmin_;
  const double bJetDeltaEtaMax_;
  const unsigned int nbjets_;
  const double workingpoint_;
  const std::string btagalgoName_;
  const PVcut lepPVcuts_;
  const bool applyLeptonPVcuts_;
  const bool debug_;

  const bool applyMETcut_ = false;
};

BTagAndProbe::BTagAndProbe(const edm::ParameterSet& iConfig)
    : folderName_(iConfig.getParameter<std::string>("FolderName")),
      requireValidHLTPaths_(iConfig.getParameter<bool>("requireValidHLTPaths")),
      hltPathsAreValid_(false),
      vtxToken_(mayConsume<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertices"))),
      muoToken_(mayConsume<reco::MuonCollection>(iConfig.getParameter<edm::InputTag>("muons"))),
      eleToken_(mayConsume<edm::View<reco::GsfElectron> >(iConfig.getParameter<edm::InputTag>("electrons"))),
      elecIDToken_(consumes<edm::ValueMap<bool> >(iConfig.getParameter<edm::InputTag>("elecID"))),
      jetTagTokens_(
          edm::vector_transform(iConfig.getParameter<std::vector<edm::InputTag> >("btagAlgos"),
                                [this](edm::InputTag const& tag) { return mayConsume<reco::JetTagCollection>(tag); })),
      shallowTagInfosToken_(
          consumes<std::vector<reco::ShallowTagInfo> >(edm::InputTag("hltDeepCombinedSecondaryVertexBJetTagsInfos"))),
      genTriggerEventFlag_(new GenericTriggerEventFlag(
          iConfig.getParameter<edm::ParameterSet>("genericTriggerEventPSet"), consumesCollector(), *this)),
      jetSelection_(iConfig.getParameter<std::string>("jetSelection")),
      eleSelection_(iConfig.getParameter<std::string>("eleSelection")),
      muoSelection_(iConfig.getParameter<std::string>("muoSelection")),
      vtxSelection_(iConfig.getParameter<std::string>("vertexSelection")),
      bjetSelection_(iConfig.getParameter<std::string>("bjetSelection")),
      nelectrons_(iConfig.getParameter<unsigned int>("nelectrons")),
      nmuons_(iConfig.getParameter<unsigned int>("nmuons")),
      leptJetDeltaRmin_(iConfig.getParameter<double>("leptJetDeltaRmin")),
      bJetDeltaEtaMax_(iConfig.getParameter<double>("bJetDeltaEtaMax")),
      nbjets_(iConfig.getParameter<unsigned int>("nbjets")),
      workingpoint_(iConfig.getParameter<double>("workingpoint")),
      lepPVcuts_(((iConfig.getParameter<edm::ParameterSet>("leptonPVcuts")).getParameter<double>("dxy")),
                 ((iConfig.getParameter<edm::ParameterSet>("leptonPVcuts")).getParameter<double>("dz"))),
      applyLeptonPVcuts_(iConfig.getParameter<bool>("applyLeptonPVcuts")),
      debug_(iConfig.getParameter<bool>("debug")) {
  ObjME empty;
}

BTagAndProbe::~BTagAndProbe() throw() {
  if (genTriggerEventFlag_)
    genTriggerEventFlag_.reset();
}

void BTagAndProbe::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& iRun, edm::EventSetup const& iSetup) {
  std::string histname, histtitle;
  std::string title;
  std::string currentFolder = folderName_;
  ibooker.setCurrentFolder(currentFolder);

  histname = "cutFlow";
  title = "cutFlow";
  cutFlow = ibooker.book1D(histname.c_str(), title.c_str(), 20, 1, 21);
  cutFlow->setBinLabel(1, "all");

  // Initialize the GenericTriggerEventFlag
  if (genTriggerEventFlag_ && genTriggerEventFlag_->on())
    genTriggerEventFlag_->initRun(iRun, iSetup);

  // check if every HLT path specified in numerator and denominator has a valid match in the HLT Menu
  hltPathsAreValid_ =
      (genTriggerEventFlag_ && genTriggerEventFlag_->on() && genTriggerEventFlag_->allHLTPathsAreValid());

  // if valid HLT paths are required,
  // create DQM outputs only if all paths are valid
  if (requireValidHLTPaths_ && (!hltPathsAreValid_)) {
    return;
  }

  if (debug_) {
    histname = "nElectrons1";
    title = "number of electrons1";
    h_nElectrons1 = ibooker.book1D(histname.c_str(), title.c_str(), 10, 0, 10);

    histname = "nElectrons2";
    title = "number of electrons2";
    h_nElectrons2 = ibooker.book1D(histname.c_str(), title.c_str(), 10, 0, 10);

    histname = "nElectrons3";
    title = "number of electrons3";
    h_nElectrons3 = ibooker.book1D(histname.c_str(), title.c_str(), 10, 0, 10);

    histname = "nElectrons4";
    title = "number of electrons4";
    h_nElectrons4 = ibooker.book1D(histname.c_str(), title.c_str(), 10, 0, 10);

    histname = "nElectrons5";
    title = "number of electrons5";
    h_nElectrons5 = ibooker.book1D(histname.c_str(), title.c_str(), 10, 0, 10);

    histname = "nElectrons6";
    title = "number of electrons6";
    h_nElectrons6 = ibooker.book1D(histname.c_str(), title.c_str(), 10, 0, 10);

    histname = "nElectrons7";
    title = "number of electrons7";
    h_nElectrons7 = ibooker.book1D(histname.c_str(), title.c_str(), 10, 0, 10);

    histname = "nElectrons8";
    title = "number of electrons8";
    h_nElectrons8 = ibooker.book1D(histname.c_str(), title.c_str(), 10, 0, 10);

    histname = "nMuons1";
    title = "number of muons1";
    h_nMuons1 = ibooker.book1D(histname.c_str(), title.c_str(), 10, 0, 10);

    histname = "nMuons2";
    title = "number of muons2";
    h_nMuons2 = ibooker.book1D(histname.c_str(), title.c_str(), 10, 0, 10);

    histname = "nMuons3";
    title = "number of muons3";
    h_nMuons3 = ibooker.book1D(histname.c_str(), title.c_str(), 10, 0, 10);

    histname = "nMuons4";
    title = "number of muons4";
    h_nMuons4 = ibooker.book1D(histname.c_str(), title.c_str(), 10, 0, 10);

    histname = "nMuons5";
    title = "number of muons5";
    h_nMuons5 = ibooker.book1D(histname.c_str(), title.c_str(), 10, 0, 10);

    histname = "nMuons6";
    title = "number of muons6";
    h_nMuons6 = ibooker.book1D(histname.c_str(), title.c_str(), 10, 0, 10);

    histname = "nMuons7";
    title = "number of muons7";
    h_nMuons7 = ibooker.book1D(histname.c_str(), title.c_str(), 10, 0, 10);

    //muon pt
    histname = "Muons1_pt";
    title = "muons1 pt";
    h_Muons1_pt = ibooker.book1D(histname.c_str(), title.c_str(), 50, 0, 100);

    histname = "Muons2_pt";
    title = "muons2 pt";
    h_Muons2_pt = ibooker.book1D(histname.c_str(), title.c_str(), 50, 0, 100);

    histname = "Muons3_pt";
    title = "muons3 pt";
    h_Muons3_pt = ibooker.book1D(histname.c_str(), title.c_str(), 50, 0, 100);

    histname = "Muons4_pt";
    title = "muons4 pt";
    h_Muons4_pt = ibooker.book1D(histname.c_str(), title.c_str(), 50, 0, 50);

    histname = "Muons5_pt";
    title = "muons5 pt";
    h_Muons5_pt = ibooker.book1D(histname.c_str(), title.c_str(), 50, 0, 50);

    //muon eta
    histname = "Muons1_eta";
    title = "muons1 eta";
    h_Muons1_eta = ibooker.book1D(histname.c_str(), title.c_str(), 10, -5, 5);

    histname = "Muons2_eta";
    title = "muons2 eta";
    h_Muons2_eta = ibooker.book1D(histname.c_str(), title.c_str(), 10, -5, 5);

    histname = "Muons3_eta";
    title = "muons3 eta";
    h_Muons3_eta = ibooker.book1D(histname.c_str(), title.c_str(), 10, -5, 5);

    histname = "Muons4_eta";
    title = "muons4 eta";
    h_Muons4_eta = ibooker.book1D(histname.c_str(), title.c_str(), 10, -2.50, 2.50);

    histname = "Muons5_eta";
    title = "muons5 eta";
    h_Muons5_eta = ibooker.book1D(histname.c_str(), title.c_str(), 10, -2.50, 2.50);

    //electron pt
    histname = "Electrons1_pt";
    title = "electrons1 pt";
    h_Electrons1_pt = ibooker.book1D(histname.c_str(), title.c_str(), 50, 0, 100);

    histname = "Electrons2_pt";
    title = "Electrons2 pt";
    h_Electrons2_pt = ibooker.book1D(histname.c_str(), title.c_str(), 50, 0, 100);

    histname = "Electrons3_pt";
    title = "Electrons3 pt";
    h_Electrons3_pt = ibooker.book1D(histname.c_str(), title.c_str(), 50, 0, 100);

    histname = "Electrons4_pt";
    title = "Electrons4 pt";
    h_Electrons4_pt = ibooker.book1D(histname.c_str(), title.c_str(), 50, 0, 100);

    //electron eta
    histname = "Electrons1_eta";
    title = "Electrons1 eta";
    h_Electrons1_eta = ibooker.book1D(histname.c_str(), title.c_str(), 10, -2.50, 2.50);

    histname = "Electrons2_eta";
    title = "Electrons2 eta";
    h_Electrons2_eta = ibooker.book1D(histname.c_str(), title.c_str(), 10, -2.50, 2.50);

    histname = "Electrons3_eta";
    title = "Electrons3 eta";
    h_Electrons3_eta = ibooker.book1D(histname.c_str(), title.c_str(), 10, -2.50, 2.50);

    histname = "Electrons4_eta";
    title = "Electrons4 eta";
    h_Electrons4_eta = ibooker.book1D(histname.c_str(), title.c_str(), 10, -2.50, 2.50);

    //nJets
    histname = "nJets1";
    title = "number of jets1";
    h_nJets1 = ibooker.book1D(histname.c_str(), title.c_str(), 20, 0, 20);

    histname = "nJets2";
    title = "number of jets2";
    h_nJets2 = ibooker.book1D(histname.c_str(), title.c_str(), 20, 0, 20);

    histname = "nJets3";
    title = "number of jets3";
    h_nJets3 = ibooker.book1D(histname.c_str(), title.c_str(), 20, 0, 20);

    histname = "nJets4";
    title = "number of jets4";
    h_nJets4 = ibooker.book1D(histname.c_str(), title.c_str(), 20, 0, 20);

    histname = "nJets5";
    title = "number of jets5";
    h_nJets5 = ibooker.book1D(histname.c_str(), title.c_str(), 20, 0, 20);

    histname = "nJets6";
    title = "number of jets6";
    h_nJets6 = ibooker.book1D(histname.c_str(), title.c_str(), 20, 0, 20);

    histname = "nJets7";
    title = "number of jets7";
    h_nJets7 = ibooker.book1D(histname.c_str(), title.c_str(), 20, 0, 20);

    histname = "nJets8";
    title = "number of jets8";
    h_nJets8 = ibooker.book1D(histname.c_str(), title.c_str(), 20, 0, 20);

    histname = "nJets9";
    title = "number of jets9";
    h_nJets9 = ibooker.book1D(histname.c_str(), title.c_str(), 20, 0, 20);

    histname = "nJets10";
    title = "number of jets10";
    h_nJets10 = ibooker.book1D(histname.c_str(), title.c_str(), 20, 0, 20);

    histname = "nJets11";
    title = "number of jets11";
    h_nJets11 = ibooker.book1D(histname.c_str(), title.c_str(), 20, 0, 20);

    histname = "nJets12";
    title = "number of jets12";
    h_nJets12 = ibooker.book1D(histname.c_str(), title.c_str(), 20, 0, 20);
  }

  histname = "btagVal";
  title = "btagVal";
  h_btagVal = ibooker.book1D(histname.c_str(), title.c_str(), 10, 0, 1);

  histname = "btagVal2";
  title = "btagVal";
  h_btagVal2 = ibooker.book1D(histname.c_str(), title.c_str(), 10, 0, 1);

  histname = "btagVal_probe_pass";
  title = "btagVal";
  h_btagVal_pp = ibooker.book1D(histname.c_str(), title.c_str(), 10, 0, 1);

  histname = "btagVal_probe_fail";
  title = "btagVal";
  h_btagVal_pf = ibooker.book1D(histname.c_str(), title.c_str(), 10, 0, 1);

  histname = "btagVal_probe_all";
  title = "btagVal";
  h_btagVal_pa = ibooker.book1D(histname.c_str(), title.c_str(), 10, 0, 1);

  histname = "jetNSecondaryVertices";
  title = "jetNSecondaryVertices";
  bookME(ibooker, jetNSecondaryVertices_, histname, title, 10, -0.5, 9.5);
  setMETitle(jetNSecondaryVertices_, "jetNSecondaryVertices", "Entries");

  histname = "jet_bTagVal";
  title = "jet bTag Val";
  bookME(ibooker, jet_tagVal_, histname, title, 10, 0., 1.);
  setMETitle(jet_tagVal_, "jet bTag Val", "Entries");

  histname = "jet_pt";
  title = "jet p_{T}";
  bookME(ibooker, jet_pt_, histname, title, 100, -0.1, 400.);
  setMETitle(jet_pt_, "jet pt", "Entries");

  histname = "jet_eta";
  title = "jet #eta";
  bookME(ibooker, jet_eta_, histname, title, 20, -2.5, 2.5);
  setMETitle(jet_eta_, "#eta", "Entries");

  histname = "jetNSecondaryVertices";
  title = "jet NSecondary Vertices";
  bookME(ibooker, jetNSecondaryVertices_, histname, title, 10, -0.5, 9.5);
  setMETitle(jetNSecondaryVertices_, "N Secondary Vertices", "Entries");

  histname = "trackSumJetEtRatio";
  title = "trackSumJetEtRatio";
  bookME(ibooker, trackSumJetEtRatio_, histname, title, 100, -.1, 1.5);
  setMETitle(trackSumJetEtRatio_, "trackSumJetEtRatio", "Entries");

  histname = "trackSumJetDeltaR";
  title = "trackSumJetDeltaR";
  bookME(ibooker, trackSumJetDeltaR_, histname, title, 100, -0.1, 0.35);
  setMETitle(trackSumJetDeltaR_, "trackSumJetDeltaR", "Entries");

  histname = "vertexCategory";
  title = "vertexCategory";
  bookME(ibooker, vertexCategory_, histname, title, 4, -1.5, 2.5);
  setMETitle(vertexCategory_, "vertexCategory", "Entries");

  histname = "trackSip2dValAboveCharm";
  title = "trackSip2dValAboveCharm";
  bookME(ibooker, trackSip2dValAboveCharm_, histname, title, 100, -0.2, 0.2);
  setMETitle(trackSip2dValAboveCharm_, "trackSip2dValAboveCharm", "Entries");

  histname = "trackSip2dSigAboveCharm";
  title = "trackSip2dSigAboveCharm";
  bookME(ibooker, trackSip2dSigAboveCharm_, histname, title, 100, -50, 50);
  setMETitle(trackSip2dSigAboveCharm_, "trackSip2dSigAboveCharm", "Entries");

  histname = "trackSip3dValAboveCharm";
  title = "trackSip3dValAboveCharm";
  bookME(ibooker, trackSip3dValAboveCharm_, histname, title, 100, -0.2, 0.2);
  setMETitle(trackSip3dValAboveCharm_, "trackSip3dValAboveCharm", "Entries");

  histname = "trackSip3dSigAboveCharm";
  title = "trackSip3dSigAboveCharm";
  bookME(ibooker, trackSip3dSigAboveCharm_, histname, title, 100, -50, 50);
  setMETitle(trackSip3dSigAboveCharm_, "trackSip3dSigAboveCharm", "Entries");

  histname = "jetNSelectedTracks";
  title = "jetNSelectedTracks";
  bookME(ibooker, jetNSelectedTracks_, histname, title, 42, -1.5, 40.5);
  setMETitle(jetNSelectedTracks_, "jetNSelectedTracks", "Entries");

  histname = "jetNTracksEtaRel";
  title = "jetNTracksEtaRel";
  bookME(ibooker, jetNTracksEtaRel_, histname, title, 42, -1.5, 40.5);
  setMETitle(jetNTracksEtaRel_, "jetNTracksEtaRel", "Entries");

  histname = "trackJetDistVal";
  title = "trackJetDistVal";
  bookME(ibooker, trackJetDistVal_, histname, title, 100, -1, 0.01);
  setMETitle(trackJetDistVal_, "trackJetDistVal", "Entries");

  histname = "trackPtRel";
  title = "trackPtRel";
  bookME(ibooker, trackPtRel_, histname, title, 100, -0.1, 7);
  setMETitle(trackPtRel_, "trackPtRel", "Entries");

  histname = "trackDeltaR";
  title = "trackDeltaR";
  bookME(ibooker, trackDeltaR_, histname, title, 160, -0.05, 0.47);
  setMETitle(trackDeltaR_, "trackDeltaR", "Entries");

  histname = "trackPtRatio";
  title = "trackPtRatio";
  bookME(ibooker, trackPtRatio_, histname, title, 100, -0.01, 0.3);
  setMETitle(trackPtRatio_, "trackPtRatio", "Entries");

  histname = "trackSip3dSig";
  title = "trackSip3dSig";
  bookME(ibooker, trackSip3dSig_, histname, title, 40, -40, 40);
  setMETitle(trackSip3dSig_, "trackSip3dSig", "Entries");

  histname = "trackSip2dSig";
  title = "trackSip2dSig";
  bookME(ibooker, trackSip2dSig_, histname, title, 100, -50, 50.);
  setMETitle(trackSip2dSig_, "trackSip2dSig", "Entries");

  histname = "trackDecayLenVal";
  title = "trackDecayLenVal";
  bookME(ibooker, trackDecayLenVal_, histname, title, 100, -0.1, 22);
  setMETitle(trackDecayLenVal_, "trackDecayLenVal", "Entries");

  histname = "trackEtaRel";
  title = "trackEtaRel";
  bookME(ibooker, trackEtaRel_, histname, title, 31, -0.1, 30);
  setMETitle(trackEtaRel_, "trackEtaRel", "Entries");

  histname = "vertexMass";
  title = "vertexMass";
  bookME(ibooker, vertexMass_, histname, title, 20, 0, 10);
  setMETitle(vertexMass_, "vertexMass", "Entries");

  histname = "vertexNTracks";
  title = "vertexNTracks";
  bookME(ibooker, vertexNTracks_, histname, title, 20, -0.5, 19.5);
  setMETitle(vertexNTracks_, "vertexNTracks", "Entries");

  histname = "vertexEnergyRatio";
  title = "vertexEnergyRatio";
  bookME(ibooker, vertexEnergyRatio_, histname, title, 100, -0.1, 3);
  setMETitle(vertexEnergyRatio_, "vertexEnergyRatio", "Entries");

  histname = "vertexJetDeltaR";
  title = "vertexJetDeltaR";
  bookME(ibooker, vertexJetDeltaR_, histname, title, 100, -0.01, .4);
  setMETitle(vertexJetDeltaR_, "vertexJetDeltaR", "Entries");

  histname = "flightDistance2dVal";
  title = "flightDistance2dVal";
  bookME(ibooker, flightDistance2dVal_, histname, title, 100, -0.1, 5);
  setMETitle(flightDistance2dVal_, "flightDistance2dVal", "Entries");

  histname = "flightDistance2dSig";
  title = "flightDistance2dSig";
  bookME(ibooker, flightDistance2dSig_, histname, title, 100, -10, 150.);
  setMETitle(flightDistance2dSig_, "flightDistance2dSig", "Entries");

  histname = "flightDistance3dVal";
  title = "flightDistance3dVal";
  bookME(ibooker, flightDistance3dVal_, histname, title, 100, -0.1, 5);
  setMETitle(flightDistance3dVal_, "flightDistance3dVal", "Entries");

  histname = "flightDistance3dSig";
  title = "flightDistance3dSig";
  bookME(ibooker, flightDistance3dSig_, histname, title, 100, -10, 150.);
  setMETitle(flightDistance3dSig_, "flightDistance3dSig", "Entries");
}

void BTagAndProbe::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) {
  //vector definitions
  std::vector<reco::GsfElectron> electrons;
  std::vector<reco::Muon> muons;

  //clear vectors
  electrons.clear();
  muons.clear();

  //jets map definition
  // map of Jet,btagValues (for all jets passing bJetSelection_)
  //  - btagValue of each jet is calculated as sum of values from InputTags in jetTagTokens_
  JetTagMap allJetBTagVals;

  JetTagMap bjets;

  allJetBTagVals.clear();
  bjets.clear();

  int cutFlowStatus = 1;
  cutFlow->Fill(cutFlowStatus);

  cutFlowStatus++;
  cutFlow->setBinLabel(cutFlowStatus, "allValid");
  // if valid HLT paths are required,
  // analyze event only if all paths are valid
  if (requireValidHLTPaths_ && (!hltPathsAreValid_))
    return;

  // electron Handle valid
  edm::Handle<edm::View<reco::GsfElectron> > eleHandle;
  iEvent.getByToken(eleToken_, eleHandle);
  if (!eleHandle.isValid() && nelectrons_ > 0) {
    edm::LogWarning("BTagAndProbe") << "Electron handle not valid \n";
    return;
  }

  //electron ID Handle valid
  edm::Handle<edm::ValueMap<bool> > eleIDHandle;
  iEvent.getByToken(elecIDToken_, eleIDHandle);
  if (!eleIDHandle.isValid() && nelectrons_ > 0) {
    edm::LogWarning("BTagAndProbe") << "Electron ID handle not valid \n";
    return;
  }

  //muon handle valid
  edm::Handle<reco::MuonCollection> muoHandle;
  iEvent.getByToken(muoToken_, muoHandle);
  if (!muoHandle.isValid() && nmuons_ > 0) {
    edm::LogWarning("BTagAndProbe") << "Muon handle not valid \n";
    return;
  }

  for (const auto& jetTagToken : jetTagTokens_) {
    edm::Handle<reco::JetTagCollection> bjetHandle;
    iEvent.getByToken(jetTagToken, bjetHandle);
    if (!bjetHandle.isValid() && nbjets_ > 0) {
      edm::LogWarning("BTagAndProbe") << "B-Jet handle not valid, will skip event \n";
      return;
    }
  }

  //tag info
  edm::Handle<std::vector<reco::ShallowTagInfo> > shallowTagInfos;
  iEvent.getByToken(shallowTagInfosToken_, shallowTagInfos);
  if (!shallowTagInfos.isValid()) {
    edm::LogWarning("BTagAndProbe") << "shallow tag handle not valid, will skip event \n";
    return;
  }

  cutFlow->Fill(cutFlowStatus);

  // Filter out events if Trigger Filtering is requested
  if (genTriggerEventFlag_->on() && !genTriggerEventFlag_->accept(iEvent, iSetup))
    return;
  cutFlowStatus++;
  cutFlow->setBinLabel(cutFlowStatus, "passTrigger");
  cutFlow->Fill(cutFlowStatus);

  edm::Handle<reco::VertexCollection> primaryVertices;
  iEvent.getByToken(vtxToken_, primaryVertices);
  //Primary Vertex selection
  const reco::Vertex* pv = nullptr;
  for (auto const& v : *primaryVertices) {
    if (!vtxSelection_(v))
      continue;
    pv = &v;
    break;
  }

  cutFlowStatus++;
  cutFlow->setBinLabel(cutFlowStatus, "LeptonPVcuts");
  if (applyLeptonPVcuts_ && (pv == nullptr)) {
    edm::LogWarning("BTagAndProbe") << "Invalid handle to reco::VertexCollection, event will be skipped";
    return;
  }
  cutFlow->Fill(cutFlowStatus);

  unsigned int nElectrons = 0;
  if (nelectrons_ > 0) {
    if (debug_) {
      h_nElectrons1->Fill(eleHandle->size());
      h_nElectrons2->Fill(eleIDHandle->size());
    }

    for (size_t index = 0; index < eleHandle->size(); index++) {
      const auto e = eleHandle->at(index);
      const auto el = eleHandle->ptrAt(index);

      bool pass_id = (*eleIDHandle)[el];

      if (debug_) {
        h_Electrons1_pt->Fill(e.pt());
        h_Electrons1_eta->Fill(e.eta());
      }

      if (eleSelection_(e) && pass_id) {
        if (debug_) {
          h_Electrons2_pt->Fill(e.pt());
          h_Electrons2_eta->Fill(e.eta());
        }

        if (applyLeptonPVcuts_ && ((std::fabs(e.gsfTrack()->dxy(pv->position())) >= lepPVcuts_.dxy) ||
                                   (std::fabs(e.gsfTrack()->dz(pv->position())) >= lepPVcuts_.dz))) {
          continue;
        }
        electrons.push_back(e);

        if (debug_) {
          h_Electrons3_pt->Fill(e.pt());
          h_Electrons3_eta->Fill(e.eta());
        }
      }
    }
    nElectrons = electrons.size();
    if (debug_)
      h_nElectrons3->Fill(nElectrons);
  }

  if (debug_)
    h_nMuons1->Fill(muoHandle->size());

  unsigned int nMuons = 0;
  if (nmuons_ > 0) {  // need nmuons_ at least be '1'
    for (auto const& m : *muoHandle) {
      if (debug_) {
        h_Muons1_pt->Fill(m.pt());
        h_Muons1_eta->Fill(m.eta());
      }
      if (muoSelection_(m)) {
        if (debug_) {
          h_Muons2_pt->Fill(m.pt());
          h_Muons2_eta->Fill(m.eta());
        }

        if (applyLeptonPVcuts_ && ((std::fabs(m.muonBestTrack()->dxy(pv->position())) >= lepPVcuts_.dxy) ||
                                   (std::fabs(m.muonBestTrack()->dz(pv->position())) >= lepPVcuts_.dz))) {
          continue;
        }
        muons.push_back(m);

        if (debug_) {
          h_Muons3_pt->Fill(m.pt());
          h_Muons3_eta->Fill(m.eta());
        }
      }
    }
    if (debug_)
      h_nMuons2->Fill(muons.size());

    nMuons = muons.size();
    if (nMuons < nmuons_)
      return;
    if (debug_)
      h_nMuons3->Fill(nMuons);
  }

  int nbjets1 = 0;
  int nbjets2 = 0;
  int nbjets3 = 0;

  for (const auto& jetTagToken : jetTagTokens_) {
    edm::Handle<reco::JetTagCollection> bjetHandle;
    iEvent.getByToken(jetTagToken, bjetHandle);

    const reco::JetTagCollection& bTags = *(bjetHandle.product());

    for (const auto& i_jetTag : bTags) {
      const auto& jetRef = i_jetTag.first;  // where jet that is tagged exists
      nbjets1++;
      if (not bjetSelection_(*dynamic_cast<const reco::Jet*>(jetRef.get())))
        continue;
      nbjets2++;
      const auto btagVal = i_jetTag.second;  // bTagVal exists
      h_btagVal->Fill(btagVal);

      if (not std::isfinite(btagVal))
        continue;  // checks bTagVal exists
      nbjets3++;
      if (allJetBTagVals.find(jetRef) != allJetBTagVals.end()) {
        allJetBTagVals.at(jetRef) += btagVal;  // add bjet tagVal to map
      } else {
        allJetBTagVals.insert(JetTagMap::value_type(jetRef, btagVal));
      }
    }
  }

  if (debug_) {
    h_nJets1->Fill(nbjets1);
    h_nJets2->Fill(nbjets2);
    h_nJets3->Fill(nbjets3);
  }

  int nbjets4 = 0;
  int nbjets5 = 0;
  int nbjets6 = 0;
  for (const auto& jetBTagVal : allJetBTagVals) {
    bool isJetOverlappedWithLepton = false;
    nbjets4++;
    if (nmuons_ > 0) {
      for (auto const& m : muons) {
        if (deltaR2(*jetBTagVal.first, m) < leptJetDeltaRmin_ * leptJetDeltaRmin_) {
          isJetOverlappedWithLepton = true;
          break;
        }
        if (debug_) {
          h_Muons4_pt->Fill(m.pt());
          h_Muons4_eta->Fill(m.eta());
        }
      }
    }
    if (isJetOverlappedWithLepton)
      continue;
    nbjets5++;

    isJetOverlappedWithLepton = false;
    if (nelectrons_ > 0) {
      for (auto const& e : electrons) {
        if (deltaR2(*jetBTagVal.first, e) < leptJetDeltaRmin_ * leptJetDeltaRmin_) {
          isJetOverlappedWithLepton = true;
          break;
        }
      }
    }
    if (isJetOverlappedWithLepton)
      continue;
    nbjets6++;

    bjets.insert(JetTagMap::value_type(jetBTagVal.first, jetBTagVal.second));
  }
  if (debug_) {
    h_nJets4->Fill(nbjets4);
    h_nJets5->Fill(nbjets5);
    h_nJets6->Fill(nbjets6);

    h_nJets7->Fill(bjets.size());
    h_nElectrons4->Fill(nElectrons);
    h_nMuons4->Fill(nMuons);

    h_nJets8->Fill(bjets.size());
  }

  unsigned int nbJets = bjets.size();

  cutFlowStatus++;
  cutFlow->setBinLabel(cutFlowStatus, "PassJetDeltaEta");
  if (bjets.size() > 1) {
    double deltaEta = std::abs(bjets.begin()->first->eta() - (++bjets.begin())->first->eta());
    if (deltaEta > bJetDeltaEtaMax_)
      return;
  }
  cutFlow->Fill(cutFlowStatus);
  if (debug_)
    h_nJets9->Fill(bjets.size());

  // Event selection //
  if (debug_)
    h_nElectrons5->Fill(nElectrons);
  cutFlowStatus++;
  cutFlow->setBinLabel(cutFlowStatus, "reqNumElectrons " + std::to_string(nelectrons_));
  if (nElectrons < nelectrons_)
    return;
  cutFlow->Fill(cutFlowStatus);
  if (debug_) {
    h_nJets10->Fill(bjets.size());
    h_nElectrons6->Fill(nElectrons);
    h_nMuons5->Fill(nMuons);
  }
  cutFlowStatus++;
  cutFlow->setBinLabel(cutFlowStatus, "reqNumMuons " + std::to_string(nmuons_));
  if (nMuons < nmuons_)
    return;
  cutFlow->Fill(cutFlowStatus);
  if (debug_) {
    h_nJets11->Fill(bjets.size());
    h_nElectrons7->Fill(nElectrons);
    h_nMuons6->Fill(nMuons);
  }
  cutFlowStatus++;
  cutFlow->setBinLabel(cutFlowStatus, "twoOrMoreJets");

  if (nbJets < 2)
    return;

  cutFlow->Fill(cutFlowStatus);
  if (debug_)
    h_nJets12->Fill(nbJets);

  //loop electron, muon distributions
  if (debug_) {
    for (auto const& m : muons) {
      h_Muons5_pt->Fill(m.pt());
      h_Muons5_eta->Fill(m.eta());
    }

    for (auto const& e : electrons) {
      h_Electrons4_pt->Fill(e.pt());
      h_Electrons4_eta->Fill(e.eta());
    }
    h_nElectrons8->Fill(nElectrons);  //Fill electron counter
    h_nMuons7->Fill(nMuons);          //Fill muon counter
  }

  bool isProbe;
  bool passProbe;
  for (auto& jet1 : bjets) {
    isProbe = false;
    for (auto& jet2 : bjets) {
      if (deltaR2(*jet1.first, *jet2.first) < 0.3 * 0.3)
        continue;                          // check if same jet
      if (jet2.second >= workingpoint_) {  // check if passing btag
        isProbe = true;
        break;
      }
    }

    if (isProbe) {
      h_btagVal_pa->Fill(jet1.second);
      passProbe = false;

      if (jet1.second >= workingpoint_) {
        h_btagVal_pp->Fill(jet1.second);
        passProbe = true;
      } else
        h_btagVal_pf->Fill(jet1.second);

      for (const auto& shallowTagInfo : *shallowTagInfos) {
        const auto tagJet = shallowTagInfo.jet();
        const auto& tagVars = shallowTagInfo.taggingVariables();

        if (deltaR2(jet1.first->eta(), jet1.first->phi(), tagJet->eta(), tagJet->phi()) > 0.3 * 0.3)
          continue;

        jet_pt_.fill(passProbe, tagJet->pt());
        jet_eta_.fill(passProbe, tagJet->eta());

        for (const auto& tagVar : tagVars.getList(reco::btau::jetNSecondaryVertices, false)) {
          jetNSecondaryVertices_.fill(passProbe, tagVar);
        }

        for (const auto& tagVar : tagVars.getList(reco::btau::trackSumJetEtRatio, false)) {
          trackSumJetEtRatio_.fill(passProbe, tagVar);
        }

        for (const auto& tagVar : tagVars.getList(reco::btau::trackSumJetDeltaR, false)) {
          trackSumJetDeltaR_.fill(passProbe, tagVar);
        }

        for (const auto& tagVar : tagVars.getList(reco::btau::vertexCategory, false)) {
          vertexCategory_.fill(passProbe, tagVar);
        }

        for (const auto& tagVar : tagVars.getList(reco::btau::trackSip2dValAboveCharm, false)) {
          trackSip2dValAboveCharm_.fill(passProbe, tagVar);
        }

        for (const auto& tagVar : tagVars.getList(reco::btau::trackSip2dSigAboveCharm, false)) {
          trackSip2dSigAboveCharm_.fill(passProbe, tagVar);
        }

        for (const auto& tagVar : tagVars.getList(reco::btau::trackSip3dValAboveCharm, false)) {
          trackSip3dValAboveCharm_.fill(passProbe, tagVar);
        }

        for (const auto& tagVar : tagVars.getList(reco::btau::trackSip3dSigAboveCharm, false)) {
          trackSip3dSigAboveCharm_.fill(passProbe, tagVar);
        }

        for (const auto& tagVar : tagVars.getList(reco::btau::jetNSelectedTracks, false)) {
          jetNSelectedTracks_.fill(passProbe, tagVar);
        }

        for (const auto& tagVar : tagVars.getList(reco::btau::jetNTracksEtaRel, false)) {
          jetNTracksEtaRel_.fill(passProbe, tagVar);
        }

        for (const auto& tagVar : tagVars.getList(reco::btau::trackJetDistVal, false)) {
          trackJetDistVal_.fill(passProbe, tagVar);
        }

        for (const auto& tagVar : tagVars.getList(reco::btau::trackPtRel, false)) {
          trackPtRel_.fill(passProbe, tagVar);
        }

        for (const auto& tagVar : tagVars.getList(reco::btau::trackDeltaR, false)) {
          trackDeltaR_.fill(passProbe, tagVar);
        }

        for (const auto& tagVar : tagVars.getList(reco::btau::trackPtRatio, false)) {
          trackPtRatio_.fill(passProbe, tagVar);
        }

        for (const auto& tagVar : tagVars.getList(reco::btau::trackSip3dSig, false)) {
          trackSip3dSig_.fill(passProbe, tagVar);
        }

        for (const auto& tagVar : tagVars.getList(reco::btau::trackSip2dSig, false)) {
          trackSip2dSig_.fill(passProbe, tagVar);
        }

        for (const auto& tagVar : tagVars.getList(reco::btau::trackDecayLenVal, false)) {
          trackDecayLenVal_.fill(passProbe, tagVar);
        }

        for (const auto& tagVar : tagVars.getList(reco::btau::trackEtaRel, false)) {
          trackEtaRel_.fill(passProbe, tagVar);
        }

        for (const auto& tagVar : tagVars.getList(reco::btau::vertexMass, false)) {
          vertexMass_.fill(passProbe, tagVar);
        }

        for (const auto& tagVar : tagVars.getList(reco::btau::vertexNTracks, false)) {
          vertexNTracks_.fill(passProbe, tagVar);
        }

        for (const auto& tagVar : tagVars.getList(reco::btau::vertexEnergyRatio, false)) {
          vertexEnergyRatio_.fill(passProbe, tagVar);
        }

        for (const auto& tagVar : tagVars.getList(reco::btau::vertexJetDeltaR, false)) {
          vertexJetDeltaR_.fill(passProbe, tagVar);
        }

        for (const auto& tagVar : tagVars.getList(reco::btau::flightDistance2dVal, false)) {
          flightDistance2dVal_.fill(passProbe, tagVar);
        }

        for (const auto& tagVar : tagVars.getList(reco::btau::flightDistance2dSig, false)) {
          flightDistance2dSig_.fill(passProbe, tagVar);
        }

        for (const auto& tagVar : tagVars.getList(reco::btau::flightDistance3dVal, false)) {
          flightDistance3dVal_.fill(passProbe, tagVar);
        }

        for (const auto& tagVar : tagVars.getList(reco::btau::flightDistance3dSig, false)) {
          flightDistance3dSig_.fill(passProbe, tagVar);
        }
      }
    }
  }
}

void BTagAndProbe::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("FolderName", "HLT/BTV");

  desc.add<bool>("requireValidHLTPaths", true);

  desc.add<edm::InputTag>("vertices", edm::InputTag("offlinePrimaryVertices"));
  desc.add<edm::InputTag>("muons", edm::InputTag("muons"));
  desc.add<edm::InputTag>("electrons", edm::InputTag("gedGsfElectrons"));
  desc.add<edm::InputTag>("elecID",
                          edm::InputTag("egmGsfElectronIDsForDQM:cutBasedElectronID-RunIIIWinter22-V1-tight"));
  desc.add<std::vector<edm::InputTag> >(
      "btagAlgos", {edm::InputTag("pfDeepCSVJetTags:probb"), edm::InputTag("pfDeepCSVJetTags:probbb")});

  desc.add<std::string>("jetSelection", "pt > 30");
  desc.add<std::string>("eleSelection", "pt > 0 && abs(eta) < 2.5");
  desc.add<std::string>("muoSelection", "pt > 6 && abs(eta) < 2.4");
  desc.add<std::string>("vertexSelection", "!isFake");
  desc.add<std::string>("bjetSelection", "pt > 30");
  desc.add<unsigned int>("nelectrons", 0);
  desc.add<unsigned int>("nmuons", 0);
  desc.add<double>("leptJetDeltaRmin", 0);
  desc.add<double>("bJetMuDeltaRmax", 9999.);
  desc.add<double>("bJetDeltaEtaMax", 9999.);

  desc.add<unsigned int>("nbjets", 0);
  desc.add<double>("workingpoint", 0.4941);  // DeepCSV Medium wp
  desc.add<bool>("applyLeptonPVcuts", false);
  desc.add<bool>("debug", false);

  edm::ParameterSetDescription genericTriggerEventPSet;
  GenericTriggerEventFlag::fillPSetDescription(genericTriggerEventPSet);

  desc.add<edm::ParameterSetDescription>("genericTriggerEventPSet", genericTriggerEventPSet);

  edm::ParameterSetDescription lPVcutPSet;
  lPVcutPSet.add<double>("dxy", 9999.);
  lPVcutPSet.add<double>("dz", 9999.);
  desc.add<edm::ParameterSetDescription>("leptonPVcuts", lPVcutPSet);

  descriptions.add("BTagAndProbeMonitoring", desc);
}

// Define this as a plug-in
DEFINE_FWK_MODULE(BTagAndProbe);
