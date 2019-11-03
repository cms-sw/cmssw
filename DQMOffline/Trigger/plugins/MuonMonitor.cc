#include "DQMOffline/Trigger/plugins/MuonMonitor.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"

MuonMonitor::MuonMonitor(const edm::ParameterSet& iConfig)
  : folderName_(iConfig.getParameter<std::string>("FolderName"))
  , requireValidHLTPaths_(iConfig.getParameter<bool>("requireValidHLTPaths"))
  , hltPathsAreValid_(false)
  , metToken_(consumes<reco::PFMETCollection>(iConfig.getParameter<edm::InputTag>("met")))
  , muonToken_(mayConsume<reco::MuonCollection>(iConfig.getParameter<edm::InputTag>("muons")))
  , vtxToken_(mayConsume<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertices")))
  , eleToken_(mayConsume<edm::View<reco::GsfElectron> >(iConfig.getParameter<edm::InputTag>("electrons")))
  , muon_variable_binning_(iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("muonBinning"))
  , muoneta_variable_binning_(iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("muonetaBinning"))
  , muon_binning_(getHistoPSet(iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("muonPSet")))
  , ls_binning_(getHistoPSet(iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("lsPSet")))
  , muPt_variable_binning_2D_(iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("muPtBinning2D"))
  , elePt_variable_binning_2D_(iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("elePtBinning2D"))
  , muEta_variable_binning_2D_(iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("muEtaBinning2D"))
  , eleEta_variable_binning_2D_(iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("eleEtaBinning2D"))
  , num_genTriggerEventFlag_(new GenericTriggerEventFlag(iConfig.getParameter<edm::ParameterSet>("numGenericTriggerEventPSet"), consumesCollector(), *this))
  , den_genTriggerEventFlag_(new GenericTriggerEventFlag(iConfig.getParameter<edm::ParameterSet>("denGenericTriggerEventPSet"), consumesCollector(), *this))
  , metSelection_(iConfig.getParameter<std::string>("metSelection"))
  , muonSelection_(iConfig.getParameter<std::string>("muonSelection"))
  , eleSelection_(iConfig.getParameter<std::string>("eleSelection"))
  , nmuons_(iConfig.getParameter<unsigned int>("nmuons"))
  , nelectrons_(iConfig.getParameter<unsigned int>("nelectrons")) {
}

MuonMonitor::~MuonMonitor() throw() {

  if(num_genTriggerEventFlag_){ num_genTriggerEventFlag_.reset(); }
  if(den_genTriggerEventFlag_){ den_genTriggerEventFlag_.reset(); }
}

void MuonMonitor::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& iRun, edm::EventSetup const& iSetup) {

  // Initialize the GenericTriggerEventFlag
  if(num_genTriggerEventFlag_ && num_genTriggerEventFlag_->on()){ num_genTriggerEventFlag_->initRun(iRun, iSetup); }
  if(den_genTriggerEventFlag_ && den_genTriggerEventFlag_->on()){ den_genTriggerEventFlag_->initRun(iRun, iSetup); }

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

  ibooker.setCurrentFolder(folderName_);

  histname = "muon_pt";
  histtitle = "muon PT";
  bookME(ibooker, muonME_, histname, histtitle, muon_binning_.nbins, muon_binning_.xmin, muon_binning_.xmax);
  setMETitle(muonME_, "Muon pT [GeV]", "events / [GeV]");

  histname = "muon_pt_variable";
  histtitle = "muon PT";
  bookME(ibooker, muonME_variableBinning_, histname, histtitle, muon_variable_binning_);
  setMETitle(muonME_variableBinning_, "Muon pT [GeV]", "events / [GeV]");

  histname = "muonVsLS";
  histtitle = "muon pt vs LS";
  bookME(ibooker, muonVsLS_, histname, histtitle, ls_binning_.nbins, ls_binning_.xmin, ls_binning_.xmax, muon_binning_.xmin, muon_binning_.xmax);
  setMETitle(muonVsLS_, "LS", "Muon pT [GeV]");

  histname = "muon_phi";
  histtitle = "Muon phi";
  bookME(ibooker, muonPhiME_, histname, histtitle, phi_binning_.nbins, phi_binning_.xmin, phi_binning_.xmax);
  setMETitle(muonPhiME_, "Muon #phi", "events / 0.1 rad");

  histname = "muon_eta";
  histtitle = "Muon eta";
  bookME(ibooker, muonEtaME_, histname, histtitle, eta_binning_.nbins, eta_binning_.xmin, eta_binning_.xmax);
  setMETitle(muonEtaME_, "Muon #eta", "events");

  histname = "muon_eta_variablebinning";
  histtitle = "Muon eta";
  bookME(ibooker, muonEtaME_variableBinning_, histname, histtitle, muoneta_variable_binning_);
  setMETitle(muonEtaME_variableBinning_, "Muon #eta", "events");

  histname = "muon_dxy";
  histtitle = "Muon dxy";
  bookME(ibooker, muondxy_, histname, histtitle, dxy_binning_.nbins, dxy_binning_.xmin, dxy_binning_.xmax);
  setMETitle(muondxy_, "Muon #dxy", "events");

  histname = "muon_dz";
  histtitle = "Muon dz";
  bookME(ibooker, muondz_, histname, histtitle, dxy_binning_.nbins, dxy_binning_.xmin, dxy_binning_.xmax);
  setMETitle(muondz_, "Muon #dz", "events");

  histname = "muon_etaphi";
  histtitle = "Muon eta-phi";
  bookME(ibooker, muonEtaPhiME_, histname, histtitle, eta_binning_.nbins, eta_binning_.xmin, eta_binning_.xmax, phi_binning_.nbins, phi_binning_.xmin, phi_binning_.xmax);
  setMETitle(muonEtaPhiME_, "#eta", "#phi");

  histname = "electron_pt_variable";
  histtitle = "electron PT";
  bookME(ibooker, eleME_variableBinning_, histname, histtitle, muon_variable_binning_);
  setMETitle(eleME_variableBinning_, "Electron pT [GeV]", "events / [GeV]");

  histname = "electron_eta";
  histtitle = "electron eta";
  bookME(ibooker, eleEtaME_, histname, histtitle, eta_binning_.nbins, eta_binning_.xmin, eta_binning_.xmax);
  setMETitle(eleEtaME_, "Electron #eta", "events");

  histname = "elePt_muPt";
  histtitle = "electron pt vs muon pt";
  bookME(ibooker, elePt_muPt_, histname, histtitle, elePt_variable_binning_2D_, muPt_variable_binning_2D_);
  setMETitle(elePt_muPt_, "electron pt [GeV]", "muon pt [GeV]");

  histname = "eleEta_muEta";
  histtitle = "electron #eta vs muon #eta";
  bookME(ibooker, eleEta_muEta_, histname, histtitle, eleEta_variable_binning_2D_, muEta_variable_binning_2D_);
  setMETitle(eleEta_muEta_, "electron #eta", "muon #eta");
}

void MuonMonitor::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) {

  // if valid HLT paths are required,
  // analyze event only if all paths are valid
  if (requireValidHLTPaths_ and (not hltPathsAreValid_)) {
    return;
  }

  // Filter out events if Trigger Filtering is requested
  if (den_genTriggerEventFlag_->on() && !den_genTriggerEventFlag_->accept(iEvent, iSetup)){
    return;
  }

  edm::Handle<reco::PFMETCollection> metHandle;
  iEvent.getByToken(metToken_, metHandle);
  reco::PFMET pfmet = metHandle->front();
  if (!metSelection_(pfmet)){
    return;
  }

  edm::Handle<reco::VertexCollection> vtxHandle;
  iEvent.getByToken(vtxToken_, vtxHandle);

  math::XYZPoint pv(0, 0, 0);
  for (reco::Vertex const& v : *vtxHandle) {
    if (not v.isFake()) {
      pv.SetXYZ(v.x(), v.y(), v.z());
      break;
    }
  }

  edm::Handle<reco::MuonCollection> muonHandle;
  iEvent.getByToken(muonToken_, muonHandle);
  std::vector<reco::Muon> muons;
  if (muonHandle->size() < nmuons_)
    return;
  for (auto const& p : *muonHandle) {
    if (muonSelection_(p))
      muons.push_back(p);
  }
  if (muons.size() < nmuons_)
    return;

  edm::Handle<edm::View<reco::GsfElectron> > eleHandle;
  iEvent.getByToken(eleToken_, eleHandle);
  std::vector<reco::GsfElectron> electrons;
  if (eleHandle->size() < nelectrons_)
    return;
  for (auto const& e : *eleHandle) {
    if (eleSelection_(e))
      electrons.push_back(e);
  }
  if (electrons.size() < nelectrons_)
    return;

  // filling histograms (denominator)
  const int ls = iEvent.id().luminosityBlock();

  if (!muons.empty()){
    muonME_.denominator->Fill(muons[0].pt());
    muonME_variableBinning_.denominator->Fill(muons[0].pt());
    muonPhiME_.denominator->Fill(muons[0].phi());
    muonEtaME_.denominator->Fill(muons[0].eta());
    muonVsLS_.denominator->Fill(ls, muons[0].pt());
    muonEtaPhiME_.denominator->Fill(muons[0].eta(), muons[0].phi());
    muondxy_.denominator->Fill(muons[0].muonBestTrack()->dxy(pv));
    muondz_.denominator->Fill(muons[0].muonBestTrack()->dz(pv));
    if (!electrons.empty()) {
      eleME_variableBinning_.denominator->Fill(electrons[0].pt());
      eleEtaME_.denominator->Fill(electrons[0].eta());
      eleEta_muEta_.denominator->Fill(electrons[0].eta(), muons[0].eta());
      elePt_muPt_.denominator->Fill(electrons[0].pt(), muons[0].pt());
    }
  }

  // applying selection for numerator
  if (num_genTriggerEventFlag_->on() && !num_genTriggerEventFlag_->accept(iEvent, iSetup))
    return;

  // filling histograms (num_genTriggerEventFlag_)
  if (!muons.empty()) {
    muonME_.numerator->Fill(muons[0].pt());
    muonME_variableBinning_.numerator->Fill(muons[0].pt());
    muonPhiME_.numerator->Fill(muons[0].phi());
    muonEtaME_.numerator->Fill(muons[0].eta());
    muonVsLS_.numerator->Fill(ls, muons[0].pt());
    muonEtaPhiME_.numerator->Fill(muons[0].eta(), muons[0].phi());
    muondxy_.numerator->Fill(muons[0].muonBestTrack()->dxy(pv));
    muondz_.numerator->Fill(muons[0].muonBestTrack()->dz(pv));
    if (!electrons.empty()) {
      eleME_variableBinning_.numerator->Fill(electrons[0].pt());
      eleEtaME_.numerator->Fill(electrons[0].eta());
      eleEta_muEta_.numerator->Fill(electrons[0].eta(), muons[0].eta());
      elePt_muPt_.numerator->Fill(electrons[0].pt(), muons[0].pt());
    }
  }
}

void MuonMonitor::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {

  edm::ParameterSetDescription desc;
  desc.add<std::string>("FolderName", "HLT/Muon");
  desc.add<bool>("requireValidHLTPaths", false);

  desc.add<edm::InputTag>("met", edm::InputTag("pfMet"));
  desc.add<edm::InputTag>("muons", edm::InputTag("muons"));
  desc.add<edm::InputTag>("vertices", edm::InputTag("offlinePrimaryVertices"));
  desc.add<edm::InputTag>("electrons", edm::InputTag("gedGsfElectrons"));
  desc.add<std::string>("metSelection", "pt > 0");
  desc.add<std::string>("muonSelection", "pt > 6 && eta<2.4");
  desc.add<std::string>("eleSelection", "pt > 0");
  desc.add<unsigned int>("nmuons", 0);
  desc.add<unsigned int>("nelectrons", 0);

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
  genericTriggerEventPSet.add<bool>("errorReplyL1", false);
  genericTriggerEventPSet.add<unsigned int>("verbosityLevel", 1);
  genericTriggerEventPSet.add<bool>("andOrL1", false);
  genericTriggerEventPSet.add<bool>("l1BeforeMask", false);
  genericTriggerEventPSet.add<std::vector<std::string> >("l1Algorithms", {});

  desc.add<edm::ParameterSetDescription>("numGenericTriggerEventPSet", genericTriggerEventPSet);
  desc.add<edm::ParameterSetDescription>("denGenericTriggerEventPSet", genericTriggerEventPSet);

  edm::ParameterSetDescription histoPSet;
  edm::ParameterSetDescription metPSet;
  fillHistoPSetDescription(metPSet);
  histoPSet.add<edm::ParameterSetDescription>("muonPSet", metPSet);
  std::vector<double> bins = {0.,   20.,  40.,  60.,  80.,  90.,  100., 110., 120., 130., 140., 150., 160.,
                              170., 180., 190., 200., 220., 240., 260., 280., 300., 350., 400., 450., 1000.};
  histoPSet.add<std::vector<double> >("muonBinning", bins);

  std::vector<double> etabins = {-3., -2.5, -2., -1.5, -1., -.5, 0., .5, 1., 1.5, 2., 2.5, 3.};
  histoPSet.add<std::vector<double> >("muonetaBinning", etabins);

  std::vector<double> bins_2D = {0., 40., 80., 100., 120., 140., 160., 180., 200., 240., 280., 350., 450., 1000.};
  std::vector<double> eta_bins_2D = {-3., -2., -1., 0., 1., 2., 3.};
  std::vector<double> phi_bins_2D = {-3.1415, -2.5132, -1.8849, -1.2566, -0.6283, 0, 0.6283, 1.2566, 1.8849, 2.5132, 3.1415};
  histoPSet.add<std::vector<double> >("elePtBinning2D", bins_2D);
  histoPSet.add<std::vector<double> >("muPtBinning2D", bins_2D);
  histoPSet.add<std::vector<double> >("eleEtaBinning2D", eta_bins_2D);
  histoPSet.add<std::vector<double> >("muEtaBinning2D", eta_bins_2D);

  edm::ParameterSetDescription lsPSet;
  fillHistoLSPSetDescription(lsPSet);
  histoPSet.add<edm::ParameterSetDescription>("lsPSet", lsPSet);

  desc.add<edm::ParameterSetDescription>("histoPSet", histoPSet);

  descriptions.add("muonMonitoring", desc);
}

DEFINE_FWK_MODULE(MuonMonitor);
