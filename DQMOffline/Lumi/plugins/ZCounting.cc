#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Utilities/interface/RegexMatch.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/Math/interface/deltaR.h"

#include <TLorentzVector.h>

#include <memory>

#include "DQMOffline/Lumi/plugins/ZCounting.h"

//
// -------------------------------------- Constructor --------------------------------------------
//
ZCounting::ZCounting(const edm::ParameterSet& iConfig)
    : triggerResultsInputTag_(iConfig.getParameter<edm::InputTag>("TriggerResults")),
      fPVName_token(consumes<reco::VertexCollection>(
          iConfig.getUntrackedParameter<std::string>("edmPVName", "offlinePrimaryVertices"))),
      fMuonName_token(consumes<reco::MuonCollection>(iConfig.getUntrackedParameter<std::string>("edmName", "muons"))),
      fStandaloneRegName_token(consumes<reco::TrackCollection>(
          iConfig.getUntrackedParameter<std::string>("StandaloneReg", "standAloneMuons"))),
      fStandaloneUpdName_token(consumes<reco::TrackCollection>(
          iConfig.getUntrackedParameter<std::string>("StandaloneUpd", "standAloneMuons:UpdatedAtVtx"))),
      fTrackName_token(
          consumes<reco::TrackCollection>(iConfig.getUntrackedParameter<std::string>("edmTrackName", "generalTracks"))),

      PtCutL1_(iConfig.getUntrackedParameter<double>("PtCutL1")),
      PtCutL2_(iConfig.getUntrackedParameter<double>("PtCutL2")),
      EtaCutL1_(iConfig.getUntrackedParameter<double>("EtaCutL1")),
      EtaCutL2_(iConfig.getUntrackedParameter<double>("EtaCutL2")),

      MassBin_(iConfig.getUntrackedParameter<int>("MassBin")),
      MassMin_(iConfig.getUntrackedParameter<double>("MassMin")),
      MassMax_(iConfig.getUntrackedParameter<double>("MassMax")),

      LumiBin_(iConfig.getUntrackedParameter<int>("LumiBin")),
      LumiMin_(iConfig.getUntrackedParameter<double>("LumiMin")),
      LumiMax_(iConfig.getUntrackedParameter<double>("LumiMax")),

      PVBin_(iConfig.getUntrackedParameter<int>("PVBin")),
      PVMin_(iConfig.getUntrackedParameter<double>("PVMin")),
      PVMax_(iConfig.getUntrackedParameter<double>("PVMax")),

      VtxNTracksFitCut_(iConfig.getUntrackedParameter<double>("VtxNTracksFitMin")),
      VtxNdofCut_(iConfig.getUntrackedParameter<double>("VtxNdofMin")),
      VtxAbsZCut_(iConfig.getUntrackedParameter<double>("VtxAbsZMax")),
      VtxRhoCut_(iConfig.getUntrackedParameter<double>("VtxRhoMax")),

      IDTypestr_(iConfig.getUntrackedParameter<std::string>("IDType")),
      IsoTypestr_(iConfig.getUntrackedParameter<std::string>("IsoType")),
      IsoCut_(iConfig.getUntrackedParameter<double>("IsoCut")) {
  edm::LogInfo("ZCounting") << "Constructor  ZCounting::ZCounting " << std::endl;

  // Trigger settings
  triggers = new TriggerTools();
  triggers->setTriggerResultsToken(consumes<edm::TriggerResults>(triggerResultsInputTag_));
  triggers->setTriggerEventToken(consumes<trigger::TriggerEvent>(iConfig.getParameter<edm::InputTag>("TriggerEvent")));
  triggers->setDRMAX(DRMAX_HLT);

  edm::LogVerbatim("ZCounting") << "ZCounting::ZCounting set trigger names";
  const std::vector<std::string> patterns_ = iConfig.getParameter<std::vector<std::string>>("MuonTriggerNames");
  for (const std::string& pattern_ : patterns_) {
    triggers->addTriggerRecord(pattern_);
  }

  if (IDTypestr_ == "Loose")
    IDType_ = LooseID;
  else if (IDTypestr_ == "Medium")
    IDType_ = MediumID;
  else if (IDTypestr_ == "Tight")
    IDType_ = TightID;
  else if (IDTypestr_ == "CustomTight")
    IDType_ = CustomTightID;
  else
    IDType_ = NoneID;

  if (IsoTypestr_ == "Tracker-based")
    IsoType_ = TrackerIso;
  else if (IsoTypestr_ == "PF-based")
    IsoType_ = PFIso;
  else
    IsoType_ = NoneIso;
}

//
//  -------------------------------------- Destructor --------------------------------------------
//
ZCounting::~ZCounting() { edm::LogInfo("ZCounting") << "Destructor ZCounting::~ZCounting " << std::endl; }

//
// -------------------------------------- beginRun --------------------------------------------
//
void ZCounting::dqmBeginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {
  edm::LogInfo("ZCounting") << "ZCounting::beginRun" << std::endl;

  // initialize triggers

  edm::LogVerbatim("ZCounting") << "ZCounting::dqmBeginRun now at " << iRun.id();
  bool hltChanged_ = true;
  if (hltConfigProvider_.init(iRun, iSetup, triggerResultsInputTag_.process(), hltChanged_)) {
    edm::LogVerbatim("ZCounting") << "ZCounting::dqmBeginRun [TriggerObjMatchValueMapsProducer::beginRun] "
                                     "HLTConfigProvider initialized [processName() = \""
                                  << hltConfigProvider_.processName() << "\", tableName() = \""
                                  << hltConfigProvider_.tableName() << "\", size() = " << hltConfigProvider_.size()
                                  << "]";
  } else {
    edm::LogError("ZCounting") << "ZCounting::dqmBeginRun Initialization of HLTConfigProvider failed for Run="
                               << iRun.id() << " (process=\"" << triggerResultsInputTag_.process()
                               << "\") -> plugin will not produce outputs for this Run";
    return;
  }

  triggers->initHLTObjects(hltConfigProvider_);
}

//
// -------------------------------------- bookHistos --------------------------------------------
//
void ZCounting::bookHistograms(DQMStore::IBooker& ibooker_, edm::Run const&, edm::EventSetup const&) {
  edm::LogInfo("ZCounting") << "ZCounting::bookHistograms" << std::endl;
  ibooker_.cd();
  ibooker_.setCurrentFolder("ZCounting/Histograms");

  // Muon histograms
  h_mass_2HLT_BB = ibooker_.book2D("h_mass_2HLT_BB",
                                   "Both muon pass HLT in barrel-barrel",
                                   LumiBin_,
                                   LumiMin_,
                                   LumiMax_,
                                   MassBin_,
                                   MassMin_,
                                   MassMax_);
  h_mass_2HLT_BE = ibooker_.book2D("h_mass_2HLT_BE",
                                   "Both muon pass HLT passing in barrel-endcap",
                                   LumiBin_,
                                   LumiMin_,
                                   LumiMax_,
                                   MassBin_,
                                   MassMin_,
                                   MassMax_);
  h_mass_2HLT_EE = ibooker_.book2D("h_mass_2HLT_EE",
                                   "Both muon pass HLT passing in endcap-endcap",
                                   LumiBin_,
                                   LumiMin_,
                                   LumiMax_,
                                   MassBin_,
                                   MassMin_,
                                   MassMax_);
  h_mass_1HLT_BB = ibooker_.book2D("h_mass_1HLT_BB",
                                   "One muon pass HLT in barrel-barrel",
                                   LumiBin_,
                                   LumiMin_,
                                   LumiMax_,
                                   MassBin_,
                                   MassMin_,
                                   MassMax_);
  h_mass_1HLT_BE = ibooker_.book2D("h_mass_1HLT_BE",
                                   "One muon pass HLT passing in barrel-endcap",
                                   LumiBin_,
                                   LumiMin_,
                                   LumiMax_,
                                   MassBin_,
                                   MassMin_,
                                   MassMax_);
  h_mass_1HLT_EE = ibooker_.book2D("h_mass_1HLT_EE",
                                   "One muon pass HLT passing in endcap-endcap",
                                   LumiBin_,
                                   LumiMin_,
                                   LumiMax_,
                                   MassBin_,
                                   MassMin_,
                                   MassMax_);

  h_mass_ID_fail_BB = ibooker_.book2D(
      "h_mass_ID_fail_BB", "Muon ID failing barrel-barrel", LumiBin_, LumiMin_, LumiMax_, MassBin_, MassMin_, MassMax_);
  h_mass_ID_fail_BE = ibooker_.book2D(
      "h_mass_ID_fail_BE", "Muon ID failing barrel-endcap", LumiBin_, LumiMin_, LumiMax_, MassBin_, MassMin_, MassMax_);

  h_mass_ID_fail_EE = ibooker_.book2D(
      "h_mass_ID_fail_EE", "Muon ID failing endcap-endcap", LumiBin_, LumiMin_, LumiMax_, MassBin_, MassMin_, MassMax_);

  h_mass_Glo_pass_BB = ibooker_.book2D("h_mass_Glo_pass_BB",
                                       "Muon Glo passing barrel-barrel",
                                       LumiBin_,
                                       LumiMin_,
                                       LumiMax_,
                                       MassBin_,
                                       MassMin_,
                                       MassMax_);
  h_mass_Glo_pass_BE = ibooker_.book2D("h_mass_Glo_pass_BE",
                                       "Muon Glo passing barrel-endcap",
                                       LumiBin_,
                                       LumiMin_,
                                       LumiMax_,
                                       MassBin_,
                                       MassMin_,
                                       MassMax_);

  h_mass_Glo_pass_EE = ibooker_.book2D("h_mass_Glo_pass_EE",
                                       "Muon Glo passing endcap-endcap",
                                       LumiBin_,
                                       LumiMin_,
                                       LumiMax_,
                                       MassBin_,
                                       MassMin_,
                                       MassMax_);

  h_mass_Glo_fail_BB = ibooker_.book2D("h_mass_Glo_fail_BB",
                                       "Muon Glo failing barrel-barrel",
                                       LumiBin_,
                                       LumiMin_,
                                       LumiMax_,
                                       MassBin_,
                                       MassMin_,
                                       MassMax_);
  h_mass_Glo_fail_BE = ibooker_.book2D("h_mass_Glo_fail_BE",
                                       "Muon Glo failing barrel-endcap",
                                       LumiBin_,
                                       LumiMin_,
                                       LumiMax_,
                                       MassBin_,
                                       MassMin_,
                                       MassMax_);

  h_mass_Glo_fail_EE = ibooker_.book2D("h_mass_Glo_fail_EE",
                                       "Muon Glo failing endcap-endcap",
                                       LumiBin_,
                                       LumiMin_,
                                       LumiMax_,
                                       MassBin_,
                                       MassMin_,
                                       MassMax_);

  h_mass_Sta_pass_BB = ibooker_.book2D("h_mass_Sta_pass_BB",
                                       "Muon Sta passing barrel-barrel",
                                       LumiBin_,
                                       LumiMin_,
                                       LumiMax_,
                                       MassBin_,
                                       MassMin_,
                                       MassMax_);
  h_mass_Sta_pass_BE = ibooker_.book2D("h_mass_Sta_pass_BE",
                                       "Muon Sta passing barrel-endcap",
                                       LumiBin_,
                                       LumiMin_,
                                       LumiMax_,
                                       MassBin_,
                                       MassMin_,
                                       MassMax_);

  h_mass_Sta_pass_EE = ibooker_.book2D("h_mass_Sta_pass_EE",
                                       "Muon Sta passing endcap-endcap",
                                       LumiBin_,
                                       LumiMin_,
                                       LumiMax_,
                                       MassBin_,
                                       MassMin_,
                                       MassMax_);

  h_mass_Sta_fail_BB = ibooker_.book2D("h_mass_Sta_fail_BB",
                                       "Muon Sta failing barrel-barrel",
                                       LumiBin_,
                                       LumiMin_,
                                       LumiMax_,
                                       MassBin_,
                                       MassMin_,
                                       MassMax_);
  h_mass_Sta_fail_BE = ibooker_.book2D("h_mass_Sta_fail_BE",
                                       "Muon Sta failing barrel-endcap",
                                       LumiBin_,
                                       LumiMin_,
                                       LumiMax_,
                                       MassBin_,
                                       MassMin_,
                                       MassMax_);

  h_mass_Sta_fail_EE = ibooker_.book2D("h_mass_Sta_fail_EE",
                                       "Muon Sta failing endcap-endcap",
                                       LumiBin_,
                                       LumiMin_,
                                       LumiMax_,
                                       MassBin_,
                                       MassMin_,
                                       MassMax_);

  h_npv = ibooker_.book2D(
      "h_npv", "Events with valid primary vertex", LumiBin_, LumiMin_, LumiMax_, PVBin_, PVMin_, PVMax_);

  // Axis titles
  h_mass_2HLT_BB->setAxisTitle("luminosity section", 1);
  h_mass_2HLT_BE->setAxisTitle("luminosity section", 1);
  h_mass_2HLT_EE->setAxisTitle("luminosity section", 1);
  h_mass_1HLT_BB->setAxisTitle("luminosity section", 1);
  h_mass_1HLT_BE->setAxisTitle("luminosity section", 1);
  h_mass_1HLT_EE->setAxisTitle("luminosity section", 1);
  h_mass_ID_fail_BB->setAxisTitle("luminosity section", 1);
  h_mass_ID_fail_BE->setAxisTitle("luminosity section", 1);
  h_mass_ID_fail_EE->setAxisTitle("luminosity section", 1);
  h_mass_Glo_pass_BB->setAxisTitle("luminosity section", 1);
  h_mass_Glo_pass_BE->setAxisTitle("luminosity section", 1);
  h_mass_Glo_pass_EE->setAxisTitle("luminosity section", 1);
  h_mass_Glo_fail_BB->setAxisTitle("luminosity section", 1);
  h_mass_Glo_fail_BE->setAxisTitle("luminosity section", 1);
  h_mass_Glo_fail_EE->setAxisTitle("luminosity section", 1);
  h_mass_Sta_pass_BB->setAxisTitle("luminosity section", 1);
  h_mass_Sta_pass_BE->setAxisTitle("luminosity section", 1);
  h_mass_Sta_pass_EE->setAxisTitle("luminosity section", 1);
  h_mass_Sta_fail_BB->setAxisTitle("luminosity section", 1);
  h_mass_Sta_fail_BE->setAxisTitle("luminosity section", 1);
  h_mass_Sta_fail_EE->setAxisTitle("luminosity section", 1);
  h_mass_2HLT_BB->setAxisTitle("tag and probe mass", 2);
  h_mass_2HLT_BE->setAxisTitle("tag and probe mass", 2);
  h_mass_2HLT_EE->setAxisTitle("tag and probe mass", 2);
  h_mass_1HLT_BB->setAxisTitle("tag and probe mass", 2);
  h_mass_1HLT_BE->setAxisTitle("tag and probe mass", 2);
  h_mass_1HLT_EE->setAxisTitle("tag and probe mass", 2);
  h_mass_ID_fail_BB->setAxisTitle("tag and probe mass", 2);
  h_mass_ID_fail_BE->setAxisTitle("tag and probe mass", 2);
  h_mass_ID_fail_EE->setAxisTitle("tag and probe mass", 2);
  h_mass_Glo_pass_BB->setAxisTitle("tag and probe mass", 2);
  h_mass_Glo_pass_BE->setAxisTitle("tag and probe mass", 2);
  h_mass_Glo_pass_EE->setAxisTitle("tag and probe mass", 2);
  h_mass_Glo_fail_BB->setAxisTitle("tag and probe mass", 2);
  h_mass_Glo_fail_BE->setAxisTitle("tag and probe mass", 2);
  h_mass_Glo_fail_EE->setAxisTitle("tag and probe mass", 2);
  h_mass_Sta_pass_BB->setAxisTitle("tag and probe mass", 2);
  h_mass_Sta_pass_BE->setAxisTitle("tag and probe mass", 2);
  h_mass_Sta_pass_EE->setAxisTitle("tag and probe mass", 2);
  h_mass_Sta_fail_BB->setAxisTitle("tag and probe mass", 2);
  h_mass_Sta_fail_BE->setAxisTitle("tag and probe mass", 2);
  h_mass_Sta_fail_EE->setAxisTitle("tag and probe mass", 2);
  h_npv->setAxisTitle("luminosity section", 1);
  h_npv->setAxisTitle("number of primary vertices", 2);
}

//
// -------------------------------------- Analyze --------------------------------------------
//
//--------------------------------------------------------------------------------------------------
void ZCounting::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {  // Fill event tree on the fly
  edm::LogInfo("ZCounting") << "ZCounting::analyze" << std::endl;

  //-------------------------------
  //--- Vertex
  //-------------------------------
  edm::Handle<reco::VertexCollection> hVertexProduct;
  iEvent.getByToken(fPVName_token, hVertexProduct);
  if (!hVertexProduct.isValid()) {
    edm::LogWarning("ZCounting") << "ZCounting::analyze - no valid primary vertex product found" << std::endl;
    return;
  }

  const reco::Vertex* pv = nullptr;
  int nvtx = 0;

  for (auto const& itVtx : *hVertexProduct) {
    if (itVtx.isFake())
      continue;
    if (itVtx.tracksSize() < VtxNTracksFitCut_)
      continue;
    if (itVtx.ndof() < VtxNdofCut_)
      continue;
    if (fabs(itVtx.z()) > VtxAbsZCut_)
      continue;
    if (itVtx.position().Rho() > VtxRhoCut_)
      continue;

    if (nvtx == 0) {
      pv = &itVtx;
    }
    nvtx++;
  }

  h_npv->Fill(iEvent.luminosityBlock(), nvtx);

  //-------------------------------
  //--- Trigger
  //-------------------------------
  triggers->readEvent(iEvent);

  // Trigger requirement
  if (!triggers->pass())
    return;

  //-------------------------------
  //--- Muon and Track collections
  //-------------------------------
  edm::Handle<reco::MuonCollection> hMuonProduct;
  iEvent.getByToken(fMuonName_token, hMuonProduct);
  if (!hMuonProduct.isValid()) {
    edm::LogWarning("ZCounting") << "ZCounting::analyze - no valid hMuonProduct found" << std::endl;
    return;
  }

  edm::Handle<reco::TrackCollection> hTrackProduct;
  iEvent.getByToken(fTrackName_token, hTrackProduct);
  if (!hTrackProduct.isValid()) {
    edm::LogWarning("ZCounting") << "ZCounting::analyze - no valid hTrackProduct found" << std::endl;
    return;
  }

  //-------------------------------
  //--- Merged standalone muon collections
  //--- The muon collection contains duplicates (from standAloneMuons and standAloneMuons:UpdatedAtVtx collections) and missing standAloneMuons
  //--- We need to produce a merged standalone muon collection to reproduce the decision in the global muon producer
  //-------------------------------
  edm::Handle<reco::TrackCollection> tracksStandAlone;
  iEvent.getByToken(fStandaloneRegName_token, tracksStandAlone);
  if (!tracksStandAlone.isValid()) {
    edm::LogWarning("ZCounting") << "ZCounting::analyze - no valid tracksStandAlone found" << std::endl;
    return;
  }

  edm::Handle<reco::TrackCollection> tracksStandAloneUpdatedAtVtx;
  iEvent.getByToken(fStandaloneUpdName_token, tracksStandAloneUpdatedAtVtx);
  if (!tracksStandAloneUpdatedAtVtx.isValid()) {
    edm::LogWarning("ZCounting") << "ZCounting::analyze - no valid tracksStandAloneUpdatedAtVtx found" << std::endl;
    return;
  }

  std::vector<const reco::Track*> hStandaloneProduct;
  std::vector<bool> passGlobalMuonMap;

  for (auto const& standAlone : *tracksStandAlone) {
    auto const extraIdx = standAlone.extra().key();

    const reco::Track* track = &standAlone;

    // replicate logic in GlobalMuonProducer, take the updatedAtVtx track if it exists and has
    // the same eta sign as the original, otherwise take the original
    for (auto const& standAloneUpdatedAtVtx : *tracksStandAloneUpdatedAtVtx) {
      if (standAloneUpdatedAtVtx.extra().key() == extraIdx) {
        const bool etaFlip1 = (standAloneUpdatedAtVtx.eta() * standAlone.eta()) >= 0;
        if (etaFlip1) {
          track = &standAloneUpdatedAtVtx;
        }
        break;
      }
    }

    // kinematic cuts
    if (track->pt() < MIN_PT_STA)
      continue;
    if (fabs(track->eta()) > MAX_ETA_STA)
      continue;
    // require minimum number of valid hits (mainly to reduce background)
    if (track->numberOfValidHits() < N_STA_HITS)
      continue;

    // look for corresponding muon object to check if the standalone muon is global
    bool isGlobalMuon = false;
    for (auto const& itMu2 : *hMuonProduct) {
      if (itMu2.standAloneMuon().isNull())
        continue;

      auto const& muonStandAlone = *itMu2.standAloneMuon();

      if (track->extra().key() == muonStandAlone.extra().key()) {
        // we found a corresponding muon object
        if (muonStandAlone.pt() == track->pt() && muonStandAlone.eta() == track->eta() &&
            muonStandAlone.phi() == track->phi()) {
          // the corresponding muon object uses the same standalone muon track
          // check if is a global muon
          isGlobalMuon = passGlobalMuon(itMu2);
        }
        break;
      }
    }

    passGlobalMuonMap.push_back(isGlobalMuon);
    hStandaloneProduct.push_back(track);
  }

  TLorentzVector vTag(0., 0., 0., 0.);
  TLorentzVector vProbe(0., 0., 0., 0.);
  TLorentzVector vTrack(0., 0., 0., 0.);

  // Tag loop
  for (auto const& itMu1 : *hMuonProduct) {
    const float pt1 = itMu1.muonBestTrack()->pt();
    const float eta1 = itMu1.muonBestTrack()->eta();
    const float phi1 = itMu1.muonBestTrack()->phi();
    const float q1 = itMu1.muonBestTrack()->charge();

    // Tag selection: kinematic cuts, lepton selection and trigger matching
    if (pt1 < PtCutL1_)
      continue;
    if (fabs(eta1) > EtaCutL1_)
      continue;
    if (!(passGlobalMuon(itMu1) && passMuonID(itMu1, pv) && passMuonIso(itMu1)))
      continue;
    if (!triggers->passObj(eta1, phi1))
      continue;

    vTag.SetPtEtaPhiM(pt1, eta1, phi1, MUON_MASS);

    bool isTagCentral = false;
    if (fabs(eta1) < MUON_BOUND)
      isTagCentral = true;

    // Probe loop over muons
    for (auto const& itMu2 : *hMuonProduct) {
      if (&itMu2 == &itMu1)
        continue;

      const float pt2 = itMu2.muonBestTrack()->pt();
      const float eta2 = itMu2.muonBestTrack()->eta();
      const float phi2 = itMu2.muonBestTrack()->phi();
      const float q2 = itMu2.muonBestTrack()->charge();

      // Probe selection: kinematic cuts and opposite charge requirement
      if (pt2 < PtCutL2_)
        continue;
      if (fabs(eta2) > EtaCutL2_)
        continue;
      if (q1 == q2)
        continue;

      vProbe.SetPtEtaPhiM(pt2, eta2, phi2, MUON_MASS);

      // Mass window
      TLorentzVector vDilep = vTag + vProbe;
      float dilepMass = vDilep.M();
      if ((dilepMass < MassMin_) || (dilepMass > MassMax_))
        continue;

      bool isProbeCentral = fabs(eta2) < MUON_BOUND;

      // Determine event category for efficiency calculation
      if (passGlobalMuon(itMu2) && passMuonID(itMu2, pv) && passMuonIso(itMu2)) {
        if (triggers->passObj(eta2, phi2)) {
          // category 2HLT: both muons passing trigger requirements
          if (&itMu1 > &itMu2)
            continue;  // make sure we don't double count MuMu2HLT category

          if (isTagCentral && isProbeCentral) {
            h_mass_2HLT_BB->Fill(iEvent.luminosityBlock(), dilepMass);
          } else if (!isTagCentral && !isProbeCentral) {
            h_mass_2HLT_EE->Fill(iEvent.luminosityBlock(), dilepMass);
          } else {
            h_mass_2HLT_BE->Fill(iEvent.luminosityBlock(), dilepMass);
          }
        } else {
          // category 1HLT: only one muon passes trigger
          if (isTagCentral && isProbeCentral) {
            h_mass_1HLT_BB->Fill(iEvent.luminosityBlock(), dilepMass);
          } else if (!isTagCentral && !isProbeCentral) {
            h_mass_1HLT_EE->Fill(iEvent.luminosityBlock(), dilepMass);
          } else {
            h_mass_1HLT_BE->Fill(iEvent.luminosityBlock(), dilepMass);
          }
        }
      } else if (passGlobalMuon(itMu2)) {
        // category Glo: probe is a Global muon but failing selection
        if (isTagCentral && isProbeCentral) {
          h_mass_ID_fail_BB->Fill(iEvent.luminosityBlock(), dilepMass);
        } else if (!isTagCentral && !isProbeCentral) {
          h_mass_ID_fail_EE->Fill(iEvent.luminosityBlock(), dilepMass);
        } else {
          h_mass_ID_fail_BE->Fill(iEvent.luminosityBlock(), dilepMass);
        }
      }
    }  // End of probe loop over muons

    // Probe loop over standalone muons, for global muon efficiency calculation
    for (std::vector<reco::Track>::size_type idx = 0; idx < hStandaloneProduct.size(); idx++) {
      const reco::Track* itSta = hStandaloneProduct[idx];

      // standalone muon kinematics
      const float pt2 = itSta->pt();
      const float eta2 = itSta->eta();
      const float phi2 = itSta->phi();

      // kinematic cuts
      if (pt2 < PtCutL2_)
        continue;
      if (fabs(eta2) > EtaCutL2_)
        continue;

      vProbe.SetPtEtaPhiM(pt2, eta2, phi2, MUON_MASS);

      // Mass window
      TLorentzVector vDilep = vTag + vProbe;
      float dilepMass = vDilep.M();
      if ((dilepMass < MassMin_) || (dilepMass > MassMax_))
        continue;

      const bool isProbeCentral = fabs(eta2) < MUON_BOUND;

      if (passGlobalMuonMap[idx]) {
        if (isTagCentral && isProbeCentral) {
          h_mass_Glo_pass_BB->Fill(iEvent.luminosityBlock(), dilepMass);
        } else if (!isTagCentral && !isProbeCentral) {
          h_mass_Glo_pass_EE->Fill(iEvent.luminosityBlock(), dilepMass);
        } else {
          h_mass_Glo_pass_BE->Fill(iEvent.luminosityBlock(), dilepMass);
        }
      } else {
        if (isTagCentral && isProbeCentral) {
          h_mass_Glo_fail_BB->Fill(iEvent.luminosityBlock(), dilepMass);
        } else if (!isTagCentral && !isProbeCentral) {
          h_mass_Glo_fail_EE->Fill(iEvent.luminosityBlock(), dilepMass);
        } else {
          h_mass_Glo_fail_BE->Fill(iEvent.luminosityBlock(), dilepMass);
        }
      }
    }

    // Probe loop over tracks, only for standalone efficiency calculation
    for (auto const& itTrk : *hTrackProduct) {
      const float pt2 = itTrk.pt();
      const float eta2 = itTrk.eta();
      const float phi2 = itTrk.phi();
      const float q2 = itTrk.charge();

      // Probe selection:  kinematic cuts and opposite charge requirement
      if (pt2 < PtCutL2_)
        continue;
      if (fabs(eta2) > EtaCutL2_)
        continue;
      if (q1 == q2)
        continue;
      if (!passTrack(itTrk))
        continue;

      vTrack.SetPtEtaPhiM(pt2, eta2, phi2, MUON_MASS);

      TLorentzVector vDilep = vTag + vTrack;
      float dilepMass = vDilep.M();
      if ((dilepMass < MassMin_) || (dilepMass > MassMax_))
        continue;

      // check if track is matched to standalone muon
      bool isStandalone = false;
      for (const reco::Track* itSta : hStandaloneProduct) {
        if (reco::deltaR2(itSta->eta(), itSta->phi(), eta2, phi2) < DRMAX_IO) {
          isStandalone = true;
          break;
        }
      }

      const bool isTrackCentral = fabs(eta2) < MUON_BOUND;

      if (isStandalone) {
        if (isTagCentral && isTrackCentral) {
          h_mass_Sta_pass_BB->Fill(iEvent.luminosityBlock(), dilepMass);
        } else if (!isTagCentral && !isTrackCentral) {
          h_mass_Sta_pass_EE->Fill(iEvent.luminosityBlock(), dilepMass);
        } else {
          h_mass_Sta_pass_BE->Fill(iEvent.luminosityBlock(), dilepMass);
        }
      } else {
        if (isTagCentral && isTrackCentral) {
          h_mass_Sta_fail_BB->Fill(iEvent.luminosityBlock(), dilepMass);
        } else if (!isTagCentral && !isTrackCentral) {
          h_mass_Sta_fail_EE->Fill(iEvent.luminosityBlock(), dilepMass);
        } else {
          h_mass_Sta_fail_BE->Fill(iEvent.luminosityBlock(), dilepMass);
        }
      }
    }  //End of probe loop over tracks
  }    //End of tag loop
}

//
// -------------------------------------- functions --------------------------------------------
//

//--------------------------------------------------------------------------------------------------
// Definition of the CustomTightID function
bool ZCounting::isCustomTightMuon(const reco::Muon& muon) {
  // tight POG cut based ID w/o impact parameter cuts
  return muon.isGlobalMuon() && muon.isPFMuon() && muon.globalTrack()->normalizedChi2() < 10. &&
         muon.globalTrack()->hitPattern().numberOfValidMuonHits() > 0 && muon.numberOfMatchedStations() > 1 &&
         muon.innerTrack()->hitPattern().trackerLayersWithMeasurement() > 5 &&
         muon.innerTrack()->hitPattern().numberOfValidPixelHits() > 0;
}

//--------------------------------------------------------------------------------------------------
bool ZCounting::passMuonID(const reco::Muon& muon, const reco::Vertex* vtx) {
  // Muon ID selection, using internal function "DataFormats/MuonReco/src/MuonSelectors.cc
  switch (IDType_) {
    case LooseID:
      return muon::isLooseMuon(muon);
    case MediumID:
      return muon::isMediumMuon(muon);
    case CustomTightID:
      return isCustomTightMuon(muon);
    case TightID:
      return vtx != nullptr && muon::isTightMuon(muon, *vtx);
    case NoneID:
      return true;
  }
  return false;
}

//--------------------------------------------------------------------------------------------------
bool ZCounting::passGlobalMuon(const reco::Muon& muon) {
  // Global muon selection:
  // - standard global muon criterium,
  // - requirements on inner and outer track pT>15 and |eta|
  // - requirements on deltaR(inner track, outer track)

  return muon.isGlobalMuon() && muon.outerTrack()->numberOfValidHits() >= N_STA_HITS &&
         muon.innerTrack()->pt() > MIN_PT_TRK && std::abs(muon.innerTrack()->eta()) < MAX_ETA_TRK &&
         muon.outerTrack()->pt() > MIN_PT_STA && std::abs(muon.outerTrack()->eta()) < MAX_ETA_STA &&
         reco::deltaR2(
             muon.outerTrack()->eta(), muon.outerTrack()->phi(), muon.innerTrack()->eta(), muon.innerTrack()->phi()) <
             DRMAX_IO;
}

//--------------------------------------------------------------------------------------------------
bool ZCounting::passTrack(const reco::Track& track) {
  return track.hitPattern().trackerLayersWithMeasurement() >= 6 && track.hitPattern().numberOfValidPixelHits() >= 1 &&
         track.originalAlgo() != 13      // reject muon seeded tracks - InOut
         && track.originalAlgo() != 14;  // reject muon seeded tracks - OutIn
}

//--------------------------------------------------------------------------------------------------
bool ZCounting::passMuonIso(const reco::Muon& muon) {
  //Muon isolation selection, up-to-date with MUO POG recommendation
  switch (IsoType_) {
    case TrackerIso:
      return muon.isolationR03().sumPt < IsoCut_;
    case PFIso:
      return muon.pfIsolationR04().sumChargedHadronPt +
                 std::max(0.,
                          muon.pfIsolationR04().sumNeutralHadronEt + muon.pfIsolationR04().sumPhotonEt -
                              0.5 * muon.pfIsolationR04().sumPUPt) <
             IsoCut_;
    case NoneIso:
      return true;
  }

  return false;
}

DEFINE_FWK_MODULE(ZCounting);
