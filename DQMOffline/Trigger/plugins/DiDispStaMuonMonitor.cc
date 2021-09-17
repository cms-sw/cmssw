#include <string>
#include <vector>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMOffline/Trigger/plugins/TriggerDQMBase.h"
#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

class DiDispStaMuonMonitor : public DQMEDAnalyzer, public TriggerDQMBase {
public:
  typedef dqm::reco::MonitorElement MonitorElement;
  typedef dqm::reco::DQMStore DQMStore;

  DiDispStaMuonMonitor(const edm::ParameterSet&);
  ~DiDispStaMuonMonitor() throw() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) override;

private:
  const std::string folderName_;

  const bool requireValidHLTPaths_;
  bool hltPathsAreValid_;

  edm::EDGetTokenT<reco::TrackCollection> muonToken_;

  std::vector<double> muonPt_variable_binning_;
  MEbinning muonPt_binning_;
  MEbinning muonEta_binning_;
  MEbinning muonPhi_binning_;
  MEbinning muonDxy_binning_;
  MEbinning ls_binning_;

  ObjME muonPtME_;
  ObjME muonPtNoDxyCutME_;
  ObjME muonPtME_variableBinning_;
  ObjME muonPtVsLS_;
  ObjME muonEtaME_;
  ObjME muonPhiME_;
  ObjME muonDxyME_;
  ObjME subMuonPtME_;
  ObjME subMuonPtME_variableBinning_;
  ObjME subMuonEtaME_;
  ObjME subMuonPhiME_;
  ObjME subMuonDxyME_;

  std::unique_ptr<GenericTriggerEventFlag> num_genTriggerEventFlag_;
  std::unique_ptr<GenericTriggerEventFlag> den_genTriggerEventFlag_;

  StringCutObjectSelector<reco::Track, true> muonSelectionGeneral_;
  StringCutObjectSelector<reco::Track, true> muonSelectionPt_;
  StringCutObjectSelector<reco::Track, true> muonSelectionDxy_;

  unsigned int nmuons_;
};

DiDispStaMuonMonitor::DiDispStaMuonMonitor(const edm::ParameterSet& iConfig)
    : folderName_(iConfig.getParameter<std::string>("FolderName")),
      requireValidHLTPaths_(iConfig.getParameter<bool>("requireValidHLTPaths")),
      hltPathsAreValid_(false),
      muonToken_(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("muons"))),
      muonPt_variable_binning_(
          iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double>>("muonPtBinning")),
      muonPt_binning_(getHistoPSet(
          iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("muonPtPSet"))),
      muonEta_binning_(getHistoPSet(
          iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("muonEtaPSet"))),
      muonPhi_binning_(getHistoPSet(
          iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("muonPhiPSet"))),
      muonDxy_binning_(getHistoPSet(
          iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("muonDxyPSet"))),
      ls_binning_(
          getHistoPSet(iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("lsPSet"))),
      num_genTriggerEventFlag_(new GenericTriggerEventFlag(
          iConfig.getParameter<edm::ParameterSet>("numGenericTriggerEventPSet"), consumesCollector(), *this)),
      den_genTriggerEventFlag_(new GenericTriggerEventFlag(
          iConfig.getParameter<edm::ParameterSet>("denGenericTriggerEventPSet"), consumesCollector(), *this)),
      muonSelectionGeneral_(
          iConfig.getParameter<edm::ParameterSet>("muonSelection").getParameter<std::string>("general")),
      muonSelectionPt_(iConfig.getParameter<edm::ParameterSet>("muonSelection").getParameter<std::string>("pt")),
      muonSelectionDxy_(iConfig.getParameter<edm::ParameterSet>("muonSelection").getParameter<std::string>("dxy")),
      nmuons_(iConfig.getParameter<unsigned int>("nmuons")) {}

DiDispStaMuonMonitor::~DiDispStaMuonMonitor() throw() {
  if (num_genTriggerEventFlag_) {
    num_genTriggerEventFlag_.reset();
  }
  if (den_genTriggerEventFlag_) {
    den_genTriggerEventFlag_.reset();
  }
}

void DiDispStaMuonMonitor::bookHistograms(DQMStore::IBooker& ibooker,
                                          edm::Run const& iRun,
                                          edm::EventSetup const& iSetup) {
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

  histname = "muonPt";
  histtitle = "muonPt";

  bookME(ibooker, muonPtME_, histname, histtitle, muonPt_binning_.nbins, muonPt_binning_.xmin, muonPt_binning_.xmax);
  setMETitle(muonPtME_, "DisplacedStandAlone Muon p_{T} [GeV]", "Events / [GeV]");

  histname = "muonPtNoDxyCut";
  histtitle = "muonPtNoDxyCut";
  bookME(ibooker,
         muonPtNoDxyCutME_,
         histname,
         histtitle,
         muonPt_binning_.nbins,
         muonPt_binning_.xmin,
         muonPt_binning_.xmax);
  setMETitle(muonPtNoDxyCutME_, "DisplacedStandAlone Muon p_{T} [GeV] without Dxy cut", "Events / [GeV]");

  histname = "muonPt_variable";
  histtitle = "muonPt";
  bookME(ibooker, muonPtME_variableBinning_, histname, histtitle, muonPt_variable_binning_);
  setMETitle(muonPtME_variableBinning_, "DisplacedStandAlone Muon p_{T} [GeV]", "Events / [GeV]");

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
         muonPt_binning_.xmax);
  setMETitle(muonPtVsLS_, "LS", "DisplacedStandAlone Muon p_{T} [GeV]");

  histname = "muonEta";
  histtitle = "muonEta";
  bookME(
      ibooker, muonEtaME_, histname, histtitle, muonEta_binning_.nbins, muonEta_binning_.xmin, muonEta_binning_.xmax);
  setMETitle(muonEtaME_, "DisplacedStandAlone Muon #eta", "Events");

  histname = "muonPhi";
  histtitle = "muonPhi";
  bookME(
      ibooker, muonPhiME_, histname, histtitle, muonPhi_binning_.nbins, muonPhi_binning_.xmin, muonPhi_binning_.xmax);
  setMETitle(muonPhiME_, "DisplacedStandAlone Muon #phi", "Events");

  histname = "muonDxy";
  histtitle = "muonDxy";
  bookME(
      ibooker, muonDxyME_, histname, histtitle, muonDxy_binning_.nbins, muonDxy_binning_.xmin, muonDxy_binning_.xmax);
  setMETitle(muonDxyME_, "DisplacedStandAlone Muon #dxy", "Events");

  if (nmuons_ > 1) {
    histname = "subMuonPt";
    histtitle = "subMuonPt";
    bookME(
        ibooker, subMuonPtME_, histname, histtitle, muonPt_binning_.nbins, muonPt_binning_.xmin, muonPt_binning_.xmax);
    setMETitle(subMuonPtME_, "Subleading DisplacedStandAlone Muon p_{T} [GeV]", "Events / [GeV]");

    histname = "subMuonPt_variable";
    histtitle = "subMuonPt";
    bookME(ibooker, subMuonPtME_variableBinning_, histname, histtitle, muonPt_variable_binning_);
    setMETitle(subMuonPtME_variableBinning_, "Subleading DisplacedStandAlone Muon p_{T} [GeV]", "Events / [GeV]");

    histname = "subMuonEta";
    histtitle = "subMuonEta";
    bookME(ibooker,
           subMuonEtaME_,
           histname,
           histtitle,
           muonEta_binning_.nbins,
           muonEta_binning_.xmin,
           muonEta_binning_.xmax);
    setMETitle(subMuonEtaME_, "Subleading DisplacedStandAlone Muon #eta", "Events");

    histname = "subMuonPhi";
    histtitle = "subMuonPhi";
    bookME(ibooker,
           subMuonPhiME_,
           histname,
           histtitle,
           muonPhi_binning_.nbins,
           muonPhi_binning_.xmin,
           muonPhi_binning_.xmax);
    setMETitle(subMuonPhiME_, "Subleading DisplacedStandAlone Muon #phi", "Events");

    histname = "subMuonDxy";
    histtitle = "subMuonDxy";
    bookME(ibooker,
           subMuonDxyME_,
           histname,
           histtitle,
           muonDxy_binning_.nbins,
           muonDxy_binning_.xmin,
           muonDxy_binning_.xmax);
    setMETitle(subMuonDxyME_, "Subleading DisplacedStandAlone Muon #dxy", "Events");
  }
}

void DiDispStaMuonMonitor::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) {
  // if valid HLT paths are required,
  // analyze event only if all paths are valid
  if (requireValidHLTPaths_ and (not hltPathsAreValid_)) {
    return;
  }

  // Filter out events if Trigger Filtering is requested
  if (den_genTriggerEventFlag_->on() && !den_genTriggerEventFlag_->accept(iEvent, iSetup))
    return;

  const int ls = iEvent.id().luminosityBlock();

  edm::Handle<reco::TrackCollection> DSAHandle;
  iEvent.getByToken(muonToken_, DSAHandle);
  if ((unsigned int)(DSAHandle->size()) < nmuons_)
    return;
  std::vector<edm::Ptr<reco::Track>> dsaMuonPtrs_{};  // = DSAHandle->ptrs();
  for (size_t i(0); i != DSAHandle->size(); ++i) {
    dsaMuonPtrs_.emplace_back(DSAHandle, i);
  }
  std::vector<edm::Ptr<reco::Track>> muons{}, muonsCutOnPt{}, muonsCutOnDxy{}, muonsCutOnPtAndDxy{};

  // general selection
  auto selectGeneral_([this](edm::Ptr<reco::Track> const& m) -> bool { return muonSelectionGeneral_(*m); });
  std::copy_if(dsaMuonPtrs_.begin(), dsaMuonPtrs_.end(), back_inserter(muons), selectGeneral_);
  if ((unsigned int)(muons.size()) < nmuons_)
    return;

  // sort by pt
  auto ptSorter_ = [](edm::Ptr<reco::Track> const& lhs, edm::Ptr<reco::Track> const& rhs) -> bool {
    return lhs->pt() > rhs->pt();
  };
  std::sort(muons.begin(), muons.end(), ptSorter_);

  // cut on pt
  auto selectOnPt_([this](edm::Ptr<reco::Track> const& m) -> bool { return muonSelectionPt_(*m); });
  std::copy_if(muons.begin(), muons.end(), back_inserter(muonsCutOnPt), selectOnPt_);
  // cut on dxy
  auto selectOnDxy_([this](edm::Ptr<reco::Track> const& m) -> bool { return muonSelectionDxy_(*m); });
  std::copy_if(muons.begin(), muons.end(), back_inserter(muonsCutOnDxy), selectOnDxy_);
  // cut on pt and dxy
  auto selectOnPtAndDxy_(
      [this](edm::Ptr<reco::Track> const& m) -> bool { return muonSelectionPt_(*m) && muonSelectionDxy_(*m); });
  std::copy_if(muons.begin(), muons.end(), back_inserter(muonsCutOnPtAndDxy), selectOnPtAndDxy_);

  std::sort(muonsCutOnPt.begin(), muonsCutOnPt.end(), ptSorter_);
  std::sort(muonsCutOnDxy.begin(), muonsCutOnDxy.end(), ptSorter_);
  std::sort(muonsCutOnPtAndDxy.begin(), muonsCutOnPtAndDxy.end(), ptSorter_);

  // --------------------------------
  // filling histograms (denominator)
  // --------------------------------
  if (muonsCutOnDxy.size() >= nmuons_) {
    // pt has cut on dxy
    muonPtME_.denominator->Fill(muonsCutOnDxy[0]->pt());
    muonPtNoDxyCutME_.denominator->Fill(muons[0]->pt());
    muonPtME_variableBinning_.denominator->Fill(muonsCutOnDxy[0]->pt());
    muonPtVsLS_.denominator->Fill(ls, muonsCutOnDxy[0]->pt());
    if (nmuons_ > 1) {
      subMuonPtME_.denominator->Fill(muonsCutOnDxy[1]->pt());
      subMuonPtME_variableBinning_.denominator->Fill(muonsCutOnDxy[1]->pt());
    }
  }
  if (muonsCutOnPtAndDxy.size() >= nmuons_) {
    // eta, phi have cut on pt and dxy
    muonEtaME_.denominator->Fill(muonsCutOnPtAndDxy[0]->eta());
    muonPhiME_.denominator->Fill(muonsCutOnPtAndDxy[0]->phi());
    if (nmuons_ > 1) {
      subMuonEtaME_.denominator->Fill(muonsCutOnPtAndDxy[1]->eta());
      subMuonPhiME_.denominator->Fill(muonsCutOnPtAndDxy[1]->phi());
    }
  }
  if (muonsCutOnPt.size() >= nmuons_) {
    // dxy has cut on pt
    muonDxyME_.denominator->Fill(muonsCutOnPt[0]->dxy());
    if (nmuons_ > 1) {
      subMuonDxyME_.denominator->Fill(muonsCutOnPt[1]->dxy());
    }
  }

  // --------------------------------
  // filling histograms (numerator)
  // --------------------------------
  if (num_genTriggerEventFlag_->on() && !num_genTriggerEventFlag_->accept(iEvent, iSetup))
    return;

  if (muonsCutOnDxy.size() >= nmuons_) {
    // pt has cut on dxy
    muonPtME_.numerator->Fill(muonsCutOnDxy[0]->pt());
    muonPtNoDxyCutME_.numerator->Fill(muons[0]->pt());
    muonPtME_variableBinning_.numerator->Fill(muonsCutOnDxy[0]->pt());
    muonPtVsLS_.numerator->Fill(ls, muonsCutOnDxy[0]->pt());
    if (nmuons_ > 1) {
      subMuonPtME_.numerator->Fill(muonsCutOnDxy[1]->pt());
      subMuonPtME_variableBinning_.numerator->Fill(muonsCutOnDxy[1]->pt());
    }
  }
  if (muonsCutOnPtAndDxy.size() >= nmuons_) {
    // eta, phi have cut on pt and dxy
    muonEtaME_.numerator->Fill(muonsCutOnPtAndDxy[0]->eta());
    muonPhiME_.numerator->Fill(muonsCutOnPtAndDxy[0]->phi());
    if (nmuons_ > 1) {
      subMuonEtaME_.numerator->Fill(muonsCutOnPtAndDxy[1]->eta());
      subMuonPhiME_.numerator->Fill(muonsCutOnPtAndDxy[1]->phi());
    }
  }
  if (muonsCutOnPt.size() >= nmuons_) {
    // dxy has cut on pt
    muonDxyME_.numerator->Fill(muonsCutOnPt[0]->dxy());
    if (nmuons_ > 1) {
      subMuonDxyME_.numerator->Fill(muonsCutOnPt[1]->dxy());
    }
  }
}

void DiDispStaMuonMonitor::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("FolderName", "HLT/EXO/DiDispStaMuon");
  desc.add<bool>("requireValidHLTPaths", true);

  desc.add<edm::InputTag>("muons", edm::InputTag("displacedStandAloneMuons"));
  desc.add<unsigned int>("nmuons", 2);

  edm::ParameterSetDescription muonSelection;
  muonSelection.add<std::string>("general", "pt > 0");
  muonSelection.add<std::string>("pt", "");
  muonSelection.add<std::string>("dxy", "pt > 0");
  desc.add<edm::ParameterSetDescription>("muonSelection", muonSelection);

  edm::ParameterSetDescription genericTriggerEventPSet;
  genericTriggerEventPSet.add<bool>("andOr");
  genericTriggerEventPSet.add<edm::InputTag>("dcsInputTag", edm::InputTag("scalersRawToDigi"));
  genericTriggerEventPSet.add<std::vector<int>>("dcsPartitions", {});
  genericTriggerEventPSet.add<bool>("andOrDcs", false);
  genericTriggerEventPSet.add<bool>("errorReplyDcs", true);
  genericTriggerEventPSet.add<std::string>("dbLabel", "");
  genericTriggerEventPSet.add<bool>("andOrHlt", true);
  genericTriggerEventPSet.add<edm::InputTag>("hltInputTag", edm::InputTag("TriggerResults::HLT"));
  genericTriggerEventPSet.add<std::vector<std::string>>("hltPaths", {});
  genericTriggerEventPSet.add<std::string>("hltDBKey", "");
  genericTriggerEventPSet.add<bool>("errorReplyHlt", false);
  genericTriggerEventPSet.add<unsigned int>("verbosityLevel", 1);

  desc.add<edm::ParameterSetDescription>("numGenericTriggerEventPSet", genericTriggerEventPSet);
  desc.add<edm::ParameterSetDescription>("denGenericTriggerEventPSet", genericTriggerEventPSet);

  edm::ParameterSetDescription histoPSet;
  edm::ParameterSetDescription muonPtPSet;
  edm::ParameterSetDescription muonEtaPSet;
  edm::ParameterSetDescription muonPhiPSet;
  edm::ParameterSetDescription muonDxyPSet;
  edm::ParameterSetDescription lsPSet;
  fillHistoPSetDescription(muonPtPSet);
  fillHistoPSetDescription(muonEtaPSet);
  fillHistoPSetDescription(muonPhiPSet);
  fillHistoPSetDescription(muonDxyPSet);
  fillHistoPSetDescription(lsPSet);
  histoPSet.add<edm::ParameterSetDescription>("muonPtPSet", muonPtPSet);
  histoPSet.add<edm::ParameterSetDescription>("muonEtaPSet", muonEtaPSet);
  histoPSet.add<edm::ParameterSetDescription>("muonPhiPSet", muonPhiPSet);
  histoPSet.add<edm::ParameterSetDescription>("muonDxyPSet", muonDxyPSet);
  histoPSet.add<edm::ParameterSetDescription>("lsPSet", lsPSet);
  std::vector<double> bins = {0.,   20.,  40.,  60.,  80.,  90.,  100., 110., 120., 130., 140., 150., 160.,
                              170., 180., 190., 200., 220., 240., 260., 280., 300., 350., 400., 450., 1000.};
  histoPSet.add<std::vector<double>>("muonPtBinning", bins);

  desc.add<edm::ParameterSetDescription>("histoPSet", histoPSet);

  descriptions.add("DiDispStaMuonMonitoring", desc);
}

DEFINE_FWK_MODULE(DiDispStaMuonMonitor);
