#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMOffline/Trigger/plugins/TriggerDQMBase.h"
#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include <string>
#include <vector>

class NoBPTXMonitor : public DQMEDAnalyzer, public TriggerDQMBase {
public:
  typedef dqm::reco::MonitorElement MonitorElement;
  typedef dqm::reco::DQMStore DQMStore;

  NoBPTXMonitor(const edm::ParameterSet&);
  ~NoBPTXMonitor() throw() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) override;

private:
  const std::string folderName_;

  const bool requireValidHLTPaths_;
  bool hltPathsAreValid_;

  edm::EDGetTokenT<reco::CaloJetCollection> jetToken_;
  edm::EDGetTokenT<reco::TrackCollection> muonToken_;

  std::vector<double> jetE_variable_binning_;
  MEbinning jetE_binning_;
  MEbinning jetEta_binning_;
  MEbinning jetPhi_binning_;
  std::vector<double> muonPt_variable_binning_;
  MEbinning muonPt_binning_;
  MEbinning muonEta_binning_;
  MEbinning muonPhi_binning_;
  MEbinning ls_binning_;
  MEbinning bx_binning_;

  ObjME jetENoBPTX_;
  ObjME jetENoBPTX_variableBinning_;
  ObjME jetEVsLS_;
  ObjME jetEVsBX_;
  ObjME jetEtaNoBPTX_;
  ObjME jetEtaVsLS_;
  ObjME jetEtaVsBX_;
  ObjME jetPhiNoBPTX_;
  ObjME jetPhiVsLS_;
  ObjME jetPhiVsBX_;
  ObjME muonPtNoBPTX_;
  ObjME muonPtNoBPTX_variableBinning_;
  ObjME muonPtVsLS_;
  ObjME muonPtVsBX_;
  ObjME muonEtaNoBPTX_;
  ObjME muonEtaVsLS_;
  ObjME muonEtaVsBX_;
  ObjME muonPhiNoBPTX_;
  ObjME muonPhiVsLS_;
  ObjME muonPhiVsBX_;

  std::unique_ptr<GenericTriggerEventFlag> num_genTriggerEventFlag_;
  std::unique_ptr<GenericTriggerEventFlag> den_genTriggerEventFlag_;

  StringCutObjectSelector<reco::CaloJet, true> jetSelection_;
  StringCutObjectSelector<reco::Track, true> muonSelection_;

  unsigned int njets_;
  unsigned int nmuons_;
};

NoBPTXMonitor::NoBPTXMonitor(const edm::ParameterSet& iConfig)
    : folderName_(iConfig.getParameter<std::string>("FolderName")),
      requireValidHLTPaths_(iConfig.getParameter<bool>("requireValidHLTPaths")),
      hltPathsAreValid_(false),
      jetToken_(consumes<reco::CaloJetCollection>(iConfig.getParameter<edm::InputTag>("jets"))),
      muonToken_(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("muons"))),
      jetE_variable_binning_(
          iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("jetEBinning")),
      jetE_binning_(getHistoPSet(
          iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("jetEPSet"))),
      jetEta_binning_(getHistoPSet(
          iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("jetEtaPSet"))),
      jetPhi_binning_(getHistoPSet(
          iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("jetPhiPSet"))),
      muonPt_variable_binning_(
          iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("muonPtBinning")),
      muonPt_binning_(getHistoPSet(
          iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("muonPtPSet"))),
      muonEta_binning_(getHistoPSet(
          iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("muonEtaPSet"))),
      muonPhi_binning_(getHistoPSet(
          iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("muonPhiPSet"))),
      ls_binning_(
          getHistoPSet(iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("lsPSet"))),
      bx_binning_(getHistoLSPSet(
          iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("bxPSet"))),
      num_genTriggerEventFlag_(new GenericTriggerEventFlag(
          iConfig.getParameter<edm::ParameterSet>("numGenericTriggerEventPSet"), consumesCollector(), *this)),
      den_genTriggerEventFlag_(new GenericTriggerEventFlag(
          iConfig.getParameter<edm::ParameterSet>("denGenericTriggerEventPSet"), consumesCollector(), *this)),
      jetSelection_(iConfig.getParameter<std::string>("jetSelection")),
      muonSelection_(iConfig.getParameter<std::string>("muonSelection")),
      njets_(iConfig.getParameter<unsigned int>("njets")),
      nmuons_(iConfig.getParameter<unsigned int>("nmuons")) {}

NoBPTXMonitor::~NoBPTXMonitor() throw() {
  if (num_genTriggerEventFlag_) {
    num_genTriggerEventFlag_.reset();
  }
  if (den_genTriggerEventFlag_) {
    den_genTriggerEventFlag_.reset();
  }
}

void NoBPTXMonitor::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& iRun, edm::EventSetup const& iSetup) {
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

  histname = "jetE";
  histtitle = "jetE";
  bookME(ibooker, jetENoBPTX_, histname, histtitle, jetE_binning_.nbins, jetE_binning_.xmin, jetE_binning_.xmax);
  setMETitle(jetENoBPTX_, "Jet E [GeV]", "Events / [GeV]");

  histname = "jetE_variable";
  histtitle = "jetE";
  bookME(ibooker, jetENoBPTX_variableBinning_, histname, histtitle, jetE_variable_binning_);
  setMETitle(jetENoBPTX_variableBinning_, "Jet E [GeV]", "Events / [GeV]");

  histname = "jetEVsLS";
  histtitle = "jetE vs LS";
  bookME(ibooker,
         jetEVsLS_,
         histname,
         histtitle,
         ls_binning_.nbins,
         ls_binning_.xmin,
         ls_binning_.xmax,
         jetE_binning_.xmin,
         jetE_binning_.xmax);
  setMETitle(jetEVsLS_, "LS", "Jet E [GeV]");

  histname = "jetEVsBX";
  histtitle = "jetE vs BX";
  bookME(ibooker,
         jetEVsBX_,
         histname,
         histtitle,
         bx_binning_.nbins,
         bx_binning_.xmin,
         bx_binning_.xmax,
         jetE_binning_.xmin,
         jetE_binning_.xmax,
         false);
  setMETitle(jetEVsBX_, "BX", "Jet E [GeV]");

  histname = "jetEta";
  histtitle = "jetEta";
  bookME(
      ibooker, jetEtaNoBPTX_, histname, histtitle, jetEta_binning_.nbins, jetEta_binning_.xmin, jetEta_binning_.xmax);
  setMETitle(jetEtaNoBPTX_, "Jet #eta", "Events");

  histname = "jetEtaVsLS";
  histtitle = "jetEta vs LS";
  bookME(ibooker,
         jetEtaVsLS_,
         histname,
         histtitle,
         ls_binning_.nbins,
         ls_binning_.xmin,
         ls_binning_.xmax,
         jetEta_binning_.xmin,
         jetEta_binning_.xmax,
         false);
  setMETitle(jetEtaVsLS_, "LS", "Jet #eta");

  histname = "jetEtaVsBX";
  histtitle = "jetEta vs BX";
  bookME(ibooker,
         jetEtaVsBX_,
         histname,
         histtitle,
         bx_binning_.nbins,
         bx_binning_.xmin,
         bx_binning_.xmax,
         jetEta_binning_.xmin,
         jetEta_binning_.xmax,
         false);
  setMETitle(jetEtaVsBX_, "BX", "Jet #eta");

  histname = "jetPhi";
  histtitle = "jetPhi";
  bookME(
      ibooker, jetPhiNoBPTX_, histname, histtitle, jetPhi_binning_.nbins, jetPhi_binning_.xmin, jetPhi_binning_.xmax);
  setMETitle(jetPhiNoBPTX_, "Jet #phi", "Events");

  histname = "jetPhiVsLS";
  histtitle = "jetPhi vs LS";
  bookME(ibooker,
         jetPhiVsLS_,
         histname,
         histtitle,
         ls_binning_.nbins,
         ls_binning_.xmin,
         ls_binning_.xmax,
         jetPhi_binning_.xmin,
         jetPhi_binning_.xmax,
         false);
  setMETitle(jetPhiVsLS_, "LS", "Jet #phi");

  histname = "jetPhiVsBX";
  histtitle = "jetPhi vs BX";
  bookME(ibooker,
         jetPhiVsBX_,
         histname,
         histtitle,
         bx_binning_.nbins,
         bx_binning_.xmin,
         bx_binning_.xmax,
         jetPhi_binning_.xmin,
         jetPhi_binning_.xmax,
         false);
  setMETitle(jetPhiVsBX_, "BX", "Jet #phi");

  histname = "muonPt";
  histtitle = "muonPt";
  bookME(
      ibooker, muonPtNoBPTX_, histname, histtitle, muonPt_binning_.nbins, muonPt_binning_.xmin, muonPt_binning_.xmax);
  setMETitle(muonPtNoBPTX_, "DisplacedStandAlone Muon p_{T} [GeV]", "Events / [GeV]");

  histname = "muonPt_variable";
  histtitle = "muonPt";
  bookME(ibooker, muonPtNoBPTX_variableBinning_, histname, histtitle, muonPt_variable_binning_);
  setMETitle(muonPtNoBPTX_variableBinning_, "DisplacedStandAlone Muon p_{T} [GeV]", "Events / [GeV]");

  histname = "muonPtVsLS";
  histtitle = "muonPt vs LS";
  bookME(ibooker,
         muonPtVsLS_,
         histname,
         histtitle,
         ls_binning_.nbins,
         ls_binning_.xmin,
         ls_binning_.xmax,
         muonPt_binning_.xmin,
         muonPt_binning_.xmax,
         false);
  setMETitle(muonPtVsLS_, "LS", "DisplacedStandAlone Muon p_{T} [GeV]");

  histname = "muonPtVsBX";
  histtitle = "muonPt vs BX";
  bookME(ibooker,
         muonPtVsBX_,
         histname,
         histtitle,
         bx_binning_.nbins,
         bx_binning_.xmin,
         bx_binning_.xmax,
         muonPt_binning_.xmin,
         muonPt_binning_.xmax,
         false);
  setMETitle(muonPtVsBX_, "BX", "DisplacedStandAlone Muon p_{T} [GeV]");

  histname = "muonEta";
  histtitle = "muonEta";
  bookME(ibooker,
         muonEtaNoBPTX_,
         histname,
         histtitle,
         muonEta_binning_.nbins,
         muonEta_binning_.xmin,
         muonEta_binning_.xmax);
  setMETitle(muonEtaNoBPTX_, "DisplacedStandAlone Muon #eta", "Events");

  histname = "muonEtaVsLS";
  histtitle = "muonEta vs LS";
  bookME(ibooker,
         muonEtaVsLS_,
         histname,
         histtitle,
         ls_binning_.nbins,
         ls_binning_.xmin,
         ls_binning_.xmax,
         muonEta_binning_.xmin,
         muonEta_binning_.xmax,
         false);
  setMETitle(muonEtaVsLS_, "LS", "DisplacedStandAlone Muon #eta");

  histname = "muonEtaVsBX";
  histtitle = "muonEta vs BX";
  bookME(ibooker,
         muonEtaVsBX_,
         histname,
         histtitle,
         bx_binning_.nbins,
         bx_binning_.xmin,
         bx_binning_.xmax,
         muonEta_binning_.xmin,
         muonEta_binning_.xmax,
         false);
  setMETitle(muonEtaVsBX_, "BX", "DisplacedStandAlone Muon #eta");

  histname = "muonPhi";
  histtitle = "muonPhi";
  bookME(ibooker,
         muonPhiNoBPTX_,
         histname,
         histtitle,
         muonPhi_binning_.nbins,
         muonPhi_binning_.xmin,
         muonPhi_binning_.xmax);
  setMETitle(muonPhiNoBPTX_, "DisplacedStandAlone Muon #phi", "Events");

  histname = "muonPhiVsLS";
  histtitle = "muonPhi vs LS";
  bookME(ibooker,
         muonPhiVsLS_,
         histname,
         histtitle,
         ls_binning_.nbins,
         ls_binning_.xmin,
         ls_binning_.xmax,
         muonPhi_binning_.xmin,
         muonPhi_binning_.xmax,
         false);
  setMETitle(muonPhiVsLS_, "LS", "DisplacedStandAlone Muon #phi");

  histname = "muonPhiVsBX";
  histtitle = "muonPhi vs BX";
  bookME(ibooker,
         muonPhiVsBX_,
         histname,
         histtitle,
         bx_binning_.nbins,
         bx_binning_.xmin,
         bx_binning_.xmax,
         muonPhi_binning_.xmin,
         muonPhi_binning_.xmax,
         false);
  setMETitle(muonPhiVsBX_, "BX", "DisplacedStandAlone Muon #phi");
}

void NoBPTXMonitor::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) {
  // if valid HLT paths are required,
  // analyze event only if all paths are valid
  if (requireValidHLTPaths_ and (not hltPathsAreValid_)) {
    return;
  }

  // Filter out events if Trigger Filtering is requested
  if (den_genTriggerEventFlag_->on() && !den_genTriggerEventFlag_->accept(iEvent, iSetup)) {
    return;
  }

  const int ls = iEvent.id().luminosityBlock();
  const int bx = iEvent.bunchCrossing();

  edm::Handle<reco::CaloJetCollection> jetHandle;
  iEvent.getByToken(jetToken_, jetHandle);

  if ((unsigned int)(jetHandle->size()) < njets_)
    return;

  std::vector<reco::CaloJet> jets;
  for (auto const& j : *jetHandle) {
    if (jetSelection_(j))
      jets.push_back(j);
  }

  if ((unsigned int)(jets.size()) < njets_)
    return;

  double jetE = -999;
  double jetEta = -999;
  double jetPhi = -999;
  if (!jets.empty()) {
    jetE = jets[0].energy();
    jetEta = jets[0].eta();
    jetPhi = jets[0].phi();
  }

  edm::Handle<reco::TrackCollection> DSAHandle;
  iEvent.getByToken(muonToken_, DSAHandle);

  if ((unsigned int)(DSAHandle->size()) < nmuons_)
    return;

  std::vector<reco::Track> muons;
  for (auto const& m : *DSAHandle) {
    if (muonSelection_(m))
      muons.push_back(m);
  }

  if ((unsigned int)(muons.size()) < nmuons_)
    return;

  double muonPt = -999;
  double muonEta = -999;
  double muonPhi = -999;
  if (!muons.empty()) {
    muonPt = muons[0].pt();
    muonEta = muons[0].eta();
    muonPhi = muons[0].phi();
  }

  // passes numerator-trigger (fill-numerator flag)
  const bool trg_passed = (num_genTriggerEventFlag_->on() && num_genTriggerEventFlag_->accept(iEvent, iSetup));

  // filling histograms
  jetENoBPTX_.fill(trg_passed, jetE);
  jetENoBPTX_variableBinning_.fill(trg_passed, jetE);
  jetEtaNoBPTX_.fill(trg_passed, jetEta);
  jetPhiNoBPTX_.fill(trg_passed, jetPhi);
  muonPtNoBPTX_.fill(trg_passed, muonPt);
  muonPtNoBPTX_variableBinning_.fill(trg_passed, muonPt);
  muonEtaNoBPTX_.fill(trg_passed, muonEta);
  muonPhiNoBPTX_.fill(trg_passed, muonPhi);

  jetEVsLS_.fill(trg_passed, ls, jetE);

  if (trg_passed) {
    jetEVsBX_.numerator->Fill(bx, jetE);
    jetEtaVsLS_.numerator->Fill(ls, jetEta);
    jetEtaVsBX_.numerator->Fill(bx, jetEta);
    jetPhiVsLS_.numerator->Fill(ls, jetPhi);
    jetPhiVsBX_.numerator->Fill(bx, jetPhi);
    muonPtVsLS_.numerator->Fill(ls, muonPt);
    muonPtVsBX_.numerator->Fill(bx, muonPt);
    muonEtaVsLS_.numerator->Fill(ls, muonEta);
    muonEtaVsBX_.numerator->Fill(bx, muonEta);
    muonPhiVsLS_.numerator->Fill(ls, muonPhi);
    muonPhiVsBX_.numerator->Fill(bx, muonPhi);
  }
}

void NoBPTXMonitor::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("FolderName", "HLT/NoBPTX");
  desc.add<bool>("requireValidHLTPaths", true);

  desc.add<edm::InputTag>("jets", edm::InputTag("ak4CaloJets"));
  desc.add<edm::InputTag>("muons", edm::InputTag("displacedStandAloneMuons"));
  desc.add<std::string>("jetSelection", "pt > 0");
  desc.add<std::string>("muonSelection", "pt > 0");
  desc.add<unsigned int>("njets", 0);
  desc.add<unsigned int>("nmuons", 0);

  edm::ParameterSetDescription genericTriggerEventPSet;
  genericTriggerEventPSet.add<bool>("andOr");
  genericTriggerEventPSet.add<edm::InputTag>("dcsInputTag", edm::InputTag("scalersRawToDigi"));
  genericTriggerEventPSet.add<std::vector<int> >("dcsPartitions", {});
  genericTriggerEventPSet.add<bool>("andOrDcs", false);
  genericTriggerEventPSet.add<bool>("errorReplyDcs", true);
  genericTriggerEventPSet.add<std::string>("dbLabel", "");
  genericTriggerEventPSet.add<bool>("andOrHlt", true);
  genericTriggerEventPSet.add<edm::InputTag>("hltInputTag", edm::InputTag("TriggerResults::HLT"));
  genericTriggerEventPSet.add<std::vector<std::string> >("hltPaths", {});
  genericTriggerEventPSet.add<std::string>("hltDBKey", "");
  genericTriggerEventPSet.add<bool>("errorReplyHlt", false);
  genericTriggerEventPSet.add<unsigned int>("verbosityLevel", 1);

  desc.add<edm::ParameterSetDescription>("numGenericTriggerEventPSet", genericTriggerEventPSet);
  desc.add<edm::ParameterSetDescription>("denGenericTriggerEventPSet", genericTriggerEventPSet);

  edm::ParameterSetDescription histoPSet;
  edm::ParameterSetDescription jetEPSet;
  edm::ParameterSetDescription jetEtaPSet;
  edm::ParameterSetDescription jetPhiPSet;
  edm::ParameterSetDescription muonPtPSet;
  edm::ParameterSetDescription muonEtaPSet;
  edm::ParameterSetDescription muonPhiPSet;
  edm::ParameterSetDescription lsPSet;
  edm::ParameterSetDescription bxPSet;
  fillHistoPSetDescription(jetEPSet);
  fillHistoPSetDescription(jetEtaPSet);
  fillHistoPSetDescription(jetPhiPSet);
  fillHistoPSetDescription(muonPtPSet);
  fillHistoPSetDescription(muonEtaPSet);
  fillHistoPSetDescription(muonPhiPSet);
  fillHistoPSetDescription(lsPSet);
  fillHistoLSPSetDescription(bxPSet);
  histoPSet.add<edm::ParameterSetDescription>("jetEPSet", jetEPSet);
  histoPSet.add<edm::ParameterSetDescription>("jetEtaPSet", jetEtaPSet);
  histoPSet.add<edm::ParameterSetDescription>("jetPhiPSet", jetPhiPSet);
  histoPSet.add<edm::ParameterSetDescription>("muonPtPSet", muonPtPSet);
  histoPSet.add<edm::ParameterSetDescription>("muonEtaPSet", muonEtaPSet);
  histoPSet.add<edm::ParameterSetDescription>("muonPhiPSet", muonPhiPSet);
  histoPSet.add<edm::ParameterSetDescription>("lsPSet", lsPSet);
  histoPSet.add<edm::ParameterSetDescription>("bxPSet", bxPSet);
  std::vector<double> bins = {0.,   20.,  40.,  60.,  80.,  90.,  100., 110., 120., 130., 140., 150., 160.,
                              170., 180., 190., 200., 220., 240., 260., 280., 300., 350., 400., 450., 1000.};
  histoPSet.add<std::vector<double> >("jetEBinning", bins);
  histoPSet.add<std::vector<double> >("muonPtBinning", bins);

  desc.add<edm::ParameterSetDescription>("histoPSet", histoPSet);

  descriptions.add("NoBPTXMonitoring", desc);
}

// Define this as a plug-in
DEFINE_FWK_MODULE(NoBPTXMonitor);
