#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Utilities/interface/RegexMatch.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"

#include "DQMOffline/Lumi/interface/TriggerDefs.h"
#include "DQMOffline/Lumi/interface/TTrigger.h"
#include "DQMOffline/Lumi/interface/TriggerTools.h"

#include <TLorentzVector.h>

#include <memory>

#include "DQMOffline/Lumi/plugins/ZCounting.h"

using namespace ZCountingTrigger;

//
// -------------------------------------- Constructor --------------------------------------------
//
ZCounting::ZCounting(const edm::ParameterSet& iConfig)
    : fHLTObjTag(iConfig.getParameter<edm::InputTag>("TriggerEvent")),
      fHLTTag(iConfig.getParameter<edm::InputTag>("TriggerResults")),
      fPVName(iConfig.getUntrackedParameter<std::string>("edmPVName", "offlinePrimaryVertices")),
      fMuonName(iConfig.getUntrackedParameter<std::string>("edmName", "muons")),
      fTrackName(iConfig.getUntrackedParameter<std::string>("edmTrackName", "generalTracks")),

      // Electron-specific Parameters
      fElectronName(iConfig.getUntrackedParameter<std::string>("edmGsfEleName", "gedGsfElectrons")),
      fSCName(iConfig.getUntrackedParameter<std::string>("edmSCName", "particleFlowEGamma")),

      // Electron-specific Tags
      fRhoTag(iConfig.getParameter<edm::InputTag>("rhoname")),
      fBeamspotTag(iConfig.getParameter<edm::InputTag>("beamspotName")),
      fConversionTag(iConfig.getParameter<edm::InputTag>("conversionsName")),

      // Electron-specific Cuts
      ELE_PT_CUT_TAG(iConfig.getUntrackedParameter<double>("PtCutEleTag")),
      ELE_PT_CUT_PROBE(iConfig.getUntrackedParameter<double>("PtCutEleProbe")),
      ELE_ETA_CUT_TAG(iConfig.getUntrackedParameter<double>("EtaCutEleTag")),
      ELE_ETA_CUT_PROBE(iConfig.getUntrackedParameter<double>("EtaCutEleProbe")),

      ELE_MASS_CUT_LOW(iConfig.getUntrackedParameter<double>("MassCutEleLow")),
      ELE_MASS_CUT_HIGH(iConfig.getUntrackedParameter<double>("MassCutEleHigh")),

      ELE_ID_WP(iConfig.getUntrackedParameter<std::string>("ElectronIDType", "TIGHT")),
      EleID_(ElectronIdentifier(iConfig)) {
  edm::LogInfo("ZCounting") << "Constructor  ZCounting::ZCounting " << std::endl;

  //Get parameters from configuration file
  fHLTTag_token = consumes<edm::TriggerResults>(fHLTTag);
  fHLTObjTag_token = consumes<trigger::TriggerEvent>(fHLTObjTag);
  fPVName_token = consumes<reco::VertexCollection>(fPVName);
  fMuonName_token = consumes<reco::MuonCollection>(fMuonName);
  fTrackName_token = consumes<reco::TrackCollection>(fTrackName);

  // Trigger-specific Parameters
  fMuonHLTNames = iConfig.getParameter<std::vector<std::string>>("MuonTriggerNames");
  fMuonHLTObjectNames = iConfig.getParameter<std::vector<std::string>>("MuonTriggerObjectNames");
  if (fMuonHLTNames.size() != fMuonHLTObjectNames.size()) {
    edm::LogError("ZCounting") << "List of MuonTriggerNames and MuonTriggerObjectNames has to be the same length"
                               << std::endl;
  }

  // Electron-specific parameters
  fGsfElectronName_token = consumes<edm::View<reco::GsfElectron>>(fElectronName);
  fSCName_token = consumes<edm::View<reco::SuperCluster>>(fSCName);
  fRhoToken = consumes<double>(fRhoTag);
  fBeamspotToken = consumes<reco::BeamSpot>(fBeamspotTag);
  fConversionToken = consumes<reco::ConversionCollection>(fConversionTag);

  // Muon-specific Cuts
  IDTypestr_ = iConfig.getUntrackedParameter<std::string>("IDType");
  IsoTypestr_ = iConfig.getUntrackedParameter<std::string>("IsoType");
  IsoCut_ = iConfig.getUntrackedParameter<double>("IsoCut");

  if (IDTypestr_ == "Loose")
    IDType_ = LooseID;
  else if (IDTypestr_ == "Medium")
    IDType_ = MediumID;
  else if (IDTypestr_ == "Tight")
    IDType_ = TightID;
  else
    IDType_ = NoneID;

  if (IsoTypestr_ == "Tracker-based")
    IsoType_ = TrackerIso;
  else if (IsoTypestr_ == "PF-based")
    IsoType_ = PFIso;
  else
    IsoType_ = NoneIso;

  PtCutL1_ = iConfig.getUntrackedParameter<double>("PtCutL1");
  PtCutL2_ = iConfig.getUntrackedParameter<double>("PtCutL2");
  EtaCutL1_ = iConfig.getUntrackedParameter<double>("EtaCutL1");
  EtaCutL2_ = iConfig.getUntrackedParameter<double>("EtaCutL2");

  MassBin_ = iConfig.getUntrackedParameter<int>("MassBin");
  MassMin_ = iConfig.getUntrackedParameter<double>("MassMin");
  MassMax_ = iConfig.getUntrackedParameter<double>("MassMax");

  LumiBin_ = iConfig.getUntrackedParameter<int>("LumiBin");
  LumiMin_ = iConfig.getUntrackedParameter<double>("LumiMin");
  LumiMax_ = iConfig.getUntrackedParameter<double>("LumiMax");

  PVBin_ = iConfig.getUntrackedParameter<int>("PVBin");
  PVMin_ = iConfig.getUntrackedParameter<double>("PVMin");
  PVMax_ = iConfig.getUntrackedParameter<double>("PVMax");

  VtxNTracksFitCut_ = iConfig.getUntrackedParameter<double>("VtxNTracksFitMin");
  VtxNdofCut_ = iConfig.getUntrackedParameter<double>("VtxNdofMin");
  VtxAbsZCut_ = iConfig.getUntrackedParameter<double>("VtxAbsZMax");
  VtxRhoCut_ = iConfig.getUntrackedParameter<double>("VtxRhoMax");

  EleID_.setID(ELE_ID_WP);
}

//
//  -------------------------------------- Destructor --------------------------------------------
//
ZCounting::~ZCounting() { edm::LogInfo("ZCounting") << "Destructor ZCounting::~ZCounting " << std::endl; }

//
// -------------------------------------- beginRun --------------------------------------------
//
void ZCounting::dqmBeginRun(edm::Run const&, edm::EventSetup const&) {
  edm::LogInfo("ZCounting") << "ZCounting::beginRun" << std::endl;

  // Triggers
  fTrigger = std::make_unique<ZCountingTrigger::TTrigger>(fMuonHLTNames, fMuonHLTObjectNames);
}
//
// -------------------------------------- bookHistos --------------------------------------------
//
void ZCounting::bookHistograms(DQMStore::IBooker& ibooker_, edm::Run const&, edm::EventSetup const&) {
  edm::LogInfo("ZCounting") << "ZCounting::bookHistograms" << std::endl;
  ibooker_.cd();
  ibooker_.setCurrentFolder("ZCounting/Histograms");

  // Muon histograms
  h_mass_HLT_pass_central = ibooker_.book2D("h_mass_HLT_pass_central",
                                            "Muon HLT passing probes central",
                                            LumiBin_,
                                            LumiMin_,
                                            LumiMax_,
                                            MassBin_,
                                            MassMin_,
                                            MassMax_);
  h_mass_HLT_pass_forward = ibooker_.book2D("h_mass_HLT_pass_forward",
                                            "Muon HLT passing probes forward",
                                            LumiBin_,
                                            LumiMin_,
                                            LumiMax_,
                                            MassBin_,
                                            MassMin_,
                                            MassMax_);
  h_mass_HLT_fail_central = ibooker_.book2D("h_mass_HLT_fail_central",
                                            "Muon HLT failing probes central",
                                            LumiBin_,
                                            LumiMin_,
                                            LumiMax_,
                                            MassBin_,
                                            MassMin_,
                                            MassMax_);
  h_mass_HLT_fail_forward = ibooker_.book2D("h_mass_HLT_fail_forward",
                                            "Muon HLT failing probes forward",
                                            LumiBin_,
                                            LumiMin_,
                                            LumiMax_,
                                            MassBin_,
                                            MassMin_,
                                            MassMax_);

  h_mass_SIT_pass_central = ibooker_.book2D("h_mass_SIT_pass_central",
                                            "Muon SIT passing probes central",
                                            LumiBin_,
                                            LumiMin_,
                                            LumiMax_,
                                            MassBin_,
                                            MassMin_,
                                            MassMax_);
  h_mass_SIT_pass_forward = ibooker_.book2D("h_mass_SIT_pass_forward",
                                            "Muon SIT passing probes forward",
                                            LumiBin_,
                                            LumiMin_,
                                            LumiMax_,
                                            MassBin_,
                                            MassMin_,
                                            MassMax_);
  h_mass_SIT_fail_central = ibooker_.book2D("h_mass_SIT_fail_central",
                                            "Muon SIT_failing probes central",
                                            LumiBin_,
                                            LumiMin_,
                                            LumiMax_,
                                            MassBin_,
                                            MassMin_,
                                            MassMax_);
  h_mass_SIT_fail_forward = ibooker_.book2D("h_mass_SIT_fail_forward",
                                            "Muon SIT failing probes forward",
                                            LumiBin_,
                                            LumiMin_,
                                            LumiMax_,
                                            MassBin_,
                                            MassMin_,
                                            MassMax_);

  h_mass_Glo_pass_central = ibooker_.book2D("h_mass_Glo_pass_central",
                                            "Muon Glo passing probes central",
                                            LumiBin_,
                                            LumiMin_,
                                            LumiMax_,
                                            MassBin_,
                                            MassMin_,
                                            MassMax_);
  h_mass_Glo_pass_forward = ibooker_.book2D("h_mass_Glo_pass_forward",
                                            "Muon Glo passing probes forward",
                                            LumiBin_,
                                            LumiMin_,
                                            LumiMax_,
                                            MassBin_,
                                            MassMin_,
                                            MassMax_);
  h_mass_Glo_fail_central = ibooker_.book2D("h_mass_Glo_fail_central",
                                            "Muon Glo failing probes central",
                                            LumiBin_,
                                            LumiMin_,
                                            LumiMax_,
                                            MassBin_,
                                            MassMin_,
                                            MassMax_);
  h_mass_Glo_fail_forward = ibooker_.book2D("h_mass_Glo_fail_forward",
                                            "Muon Glo failing probes forward",
                                            LumiBin_,
                                            LumiMin_,
                                            LumiMax_,
                                            MassBin_,
                                            MassMin_,
                                            MassMax_);

  h_npv = ibooker_.book2D(
      "h_npv", "Events with valid primary vertex", LumiBin_, LumiMin_, LumiMax_, PVBin_, PVMin_, PVMax_);
  h_mass_yield_Z = ibooker_.book2D(
      "h_mass_yield_Z", "reconstructed Z bosons", LumiBin_, LumiMin_, LumiMax_, MassBin_, MassMin_, MassMax_);
  h_npv_yield_Z =
      ibooker_.book2D("h_npv_yield_Z", "reconstructed Z bosons", LumiBin_, LumiMin_, LumiMax_, PVBin_, PVMin_, PVMax_);
  h_yieldBB_Z = ibooker_.book1D("h_yieldBB_Z", "reconstructed Z bosons in barrel", LumiBin_, LumiMin_, LumiMax_);
  h_yieldEE_Z = ibooker_.book1D("h_yieldEE_Z", "reconstructed Z bosons in endcap", LumiBin_, LumiMin_, LumiMax_);

  // Axis titles
  h_mass_HLT_pass_central->setAxisTitle("luminosiry section", 1);
  h_mass_HLT_pass_forward->setAxisTitle("luminosiry section", 1);
  h_mass_HLT_fail_central->setAxisTitle("luminosiry section", 1);
  h_mass_HLT_fail_forward->setAxisTitle("luminosiry section", 1);
  h_mass_SIT_pass_central->setAxisTitle("luminosiry section", 1);
  h_mass_SIT_pass_forward->setAxisTitle("luminosiry section", 1);
  h_mass_SIT_fail_central->setAxisTitle("luminosiry section", 1);
  h_mass_SIT_fail_forward->setAxisTitle("luminosiry section", 1);
  h_mass_Glo_pass_central->setAxisTitle("luminosiry section", 1);
  h_mass_Glo_pass_forward->setAxisTitle("luminosiry section", 1);
  h_mass_Glo_fail_central->setAxisTitle("luminosiry section", 1);
  h_mass_Glo_fail_forward->setAxisTitle("luminosiry section", 1);
  h_mass_HLT_pass_central->setAxisTitle("tag and probe mass", 2);
  h_mass_HLT_pass_forward->setAxisTitle("tag and probe mass", 2);
  h_mass_HLT_fail_central->setAxisTitle("tag and probe mass", 2);
  h_mass_HLT_fail_forward->setAxisTitle("tag and probe mass", 2);
  h_mass_SIT_pass_central->setAxisTitle("tag and probe mass", 2);
  h_mass_SIT_pass_forward->setAxisTitle("tag and probe mass", 2);
  h_mass_SIT_fail_central->setAxisTitle("tag and probe mass", 2);
  h_mass_SIT_fail_forward->setAxisTitle("tag and probe mass", 2);
  h_mass_Glo_pass_central->setAxisTitle("tag and probe mass", 2);
  h_mass_Glo_pass_forward->setAxisTitle("tag and probe mass", 2);
  h_mass_Glo_fail_central->setAxisTitle("tag and probe mass", 2);
  h_mass_Glo_fail_forward->setAxisTitle("tag and probe mass", 2);
  h_npv->setAxisTitle("luminosity section", 1);
  h_npv->setAxisTitle("number of primary vertices", 2);
  h_npv_yield_Z->setAxisTitle("luminosiry section", 1);
  h_npv_yield_Z->setAxisTitle("number of primary vertices", 2);
  h_mass_yield_Z->setAxisTitle("luminosiry section", 1);
  h_mass_yield_Z->setAxisTitle("tag and probe mass", 2);

  h_yieldBB_Z->setAxisTitle("luminosiry section", 1);
  h_yieldEE_Z->setAxisTitle("luminosiry section", 1);

  /*
  // Electron histograms
  h_ee_mass_id_pass_central  = ibooker_.book2D("h_ee_mass_id_pass_central", "h_ee_mass_id_pass_central", LumiBin_, LumiMin_, LumiMax_, MassBin_, MassMin_, MassMax_);
  h_ee_mass_id_fail_central  = ibooker_.book2D("h_ee_mass_id_fail_central", "h_ee_mass_id_fail_central", LumiBin_, LumiMin_, LumiMax_, MassBin_, MassMin_, MassMax_);
  h_ee_mass_id_pass_forward  = ibooker_.book2D("h_ee_mass_id_pass_forward", "h_ee_mass_id_pass_forward", LumiBin_, LumiMin_, LumiMax_, MassBin_, MassMin_, MassMax_);
  h_ee_mass_id_fail_forward  = ibooker_.book2D("h_ee_mass_id_fail_forward", "h_ee_mass_id_fail_forward", LumiBin_, LumiMin_, LumiMax_, MassBin_, MassMin_, MassMax_);

  h_ee_mass_HLT_pass_central = ibooker_.book2D("h_ee_mass_HLT_pass_central", "h_ee_mass_HLT_pass_central", LumiBin_, LumiMin_, LumiMax_, MassBin_, MassMin_, MassMax_);
  h_ee_mass_HLT_fail_central = ibooker_.book2D("h_ee_mass_HLT_fail_central", "h_ee_mass_HLT_fail_central", LumiBin_, LumiMin_, LumiMax_, MassBin_, MassMin_, MassMax_);
  h_ee_mass_HLT_pass_forward = ibooker_.book2D("h_ee_mass_HLT_pass_forward", "h_ee_mass_HLT_pass_forward", LumiBin_, LumiMin_, LumiMax_, MassBin_, MassMin_, MassMax_);
  h_ee_mass_HLT_fail_forward = ibooker_.book2D("h_ee_mass_HLT_fail_forward", "h_ee_mass_HLT_fail_forward", LumiBin_, LumiMin_, LumiMax_, MassBin_, MassMin_, MassMax_);

  h_ee_yield_Z_ebeb       = ibooker_.book1D("h_ee_yield_Z_ebeb", "h_ee_yield_Z_ebeb", LumiBin_, LumiMin_, LumiMax_);
  h_ee_yield_Z_ebee       = ibooker_.book1D("h_ee_yield_Z_ebee", "h_ee_yield_Z_ebee", LumiBin_, LumiMin_, LumiMax_);
  h_ee_yield_Z_eeee       = ibooker_.book1D("h_ee_yield_Z_eeee", "h_ee_yield_Z_eeee", LumiBin_, LumiMin_, LumiMax_);*/
}

//
// -------------------------------------- Analyze --------------------------------------------
//
//--------------------------------------------------------------------------------------------------
void ZCounting::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {  // Fill event tree on the fly
  edm::LogInfo("ZCounting") << "ZCounting::analyze" << std::endl;
  analyzeMuons(iEvent, iSetup);
  //analyzeElectrons(iEvent, iSetup);
}

void ZCounting::analyzeMuons(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::LogInfo("ZCounting") << "ZCounting::analyzeMuons" << std::endl;
  //-------------------------------
  //--- Vertex
  //-------------------------------
  edm::Handle<reco::VertexCollection> hVertexProduct;
  iEvent.getByToken(fPVName_token, hVertexProduct);
  if (!hVertexProduct.isValid()) {
    edm::LogWarning("ZCounting") << "ZCounting::analyzeMuons - no valid primary vertex product found" << std::endl;
    return;
  }
  const reco::VertexCollection* pvCol = hVertexProduct.product();
  const reco::Vertex* pv = &(*pvCol->begin());
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
  edm::Handle<edm::TriggerResults> hTrgRes;
  iEvent.getByToken(fHLTTag_token, hTrgRes);
  if (!hTrgRes.isValid())
    return;

  edm::Handle<trigger::TriggerEvent> hTrgEvt;
  iEvent.getByToken(fHLTObjTag_token, hTrgEvt);

  const edm::TriggerNames& triggerNames = iEvent.triggerNames(*hTrgRes);
  bool config_changed = false;
  if (fTriggerNamesID != triggerNames.parameterSetID()) {
    fTriggerNamesID = triggerNames.parameterSetID();
    config_changed = true;
  }
  if (config_changed) {
    initHLT(*hTrgRes, triggerNames);
  }

  TriggerBits triggerBits;
  for (unsigned int irec = 0; irec < fTrigger->fRecords.size(); irec++) {
    if (fTrigger->fRecords[irec].hltPathIndex == (unsigned int)-1)
      continue;
    if (hTrgRes->accept(fTrigger->fRecords[irec].hltPathIndex)) {
      triggerBits[fTrigger->fRecords[irec].baconTrigBit] = true;
    }
  }
  //if(fSkipOnHLTFail && triggerBits == 0) return;

  // Trigger requirement
  if (!isMuonTrigger(*fTrigger, triggerBits))
    return;

  //-------------------------------
  //--- Muons and Tracks
  //-------------------------------
  edm::Handle<reco::MuonCollection> hMuonProduct;
  iEvent.getByToken(fMuonName_token, hMuonProduct);
  if (!hMuonProduct.isValid())
    return;

  edm::Handle<reco::TrackCollection> hTrackProduct;
  iEvent.getByToken(fTrackName_token, hTrackProduct);
  if (!hTrackProduct.isValid())
    return;

  TLorentzVector vTag(0., 0., 0., 0.);
  TLorentzVector vProbe(0., 0., 0., 0.);
  TLorentzVector vTrack(0., 0., 0., 0.);

  // Tag loop
  for (auto const& itMu1 : *hMuonProduct) {
    float pt1 = itMu1.muonBestTrack()->pt();
    float eta1 = itMu1.muonBestTrack()->eta();
    float phi1 = itMu1.muonBestTrack()->phi();
    float q1 = itMu1.muonBestTrack()->charge();

    // Tag selection: kinematic cuts, lepton selection and trigger matching
    if (pt1 < PtCutL1_)
      continue;
    if (fabs(eta1) > EtaCutL1_)
      continue;
    if (!(passMuonID(itMu1, *pv, IDType_) && passMuonIso(itMu1, IsoType_, IsoCut_)))
      continue;
    if (!isMuonTriggerObj(*fTrigger, TriggerTools::matchHLT(eta1, phi1, fTrigger->fRecords, *hTrgEvt)))
      continue;

    vTag.SetPtEtaPhiM(pt1, eta1, phi1, MUON_MASS);

    // Probe loop over muons
    for (auto const& itMu2 : *hMuonProduct) {
      if (&itMu2 == &itMu1)
        continue;

      float pt2 = itMu2.muonBestTrack()->pt();
      float eta2 = itMu2.muonBestTrack()->eta();
      float phi2 = itMu2.muonBestTrack()->phi();
      float q2 = itMu2.muonBestTrack()->charge();

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

      bool isTagCentral = false;
      bool isProbeCentral = false;
      if (fabs(eta1) < MUON_BOUND)
        isTagCentral = true;
      if (fabs(eta2) < MUON_BOUND)
        isProbeCentral = true;

      // Determine event category for efficiency calculation
      if (passMuonID(itMu2, *pv, IDType_) && passMuonIso(itMu2, IsoType_, IsoCut_)) {
        if (isMuonTriggerObj(*fTrigger, TriggerTools::matchHLT(eta2, phi2, fTrigger->fRecords, *hTrgEvt))) {
          // category 2HLT: both muons passing trigger requirements
          if (&itMu1 > &itMu2)
            continue;  // make sure we don't double count MuMu2HLT category

          // Fill twice for each event, since both muons pass trigger
          if (isTagCentral) {
            h_mass_HLT_pass_central->Fill(iEvent.luminosityBlock(), dilepMass);
            h_mass_SIT_pass_central->Fill(iEvent.luminosityBlock(), dilepMass);
            h_mass_Glo_pass_central->Fill(iEvent.luminosityBlock(), dilepMass);
          } else {
            h_mass_HLT_pass_forward->Fill(iEvent.luminosityBlock(), dilepMass);
            h_mass_SIT_pass_forward->Fill(iEvent.luminosityBlock(), dilepMass);
            h_mass_Glo_pass_forward->Fill(iEvent.luminosityBlock(), dilepMass);
          }

          if (isProbeCentral) {
            h_mass_HLT_pass_central->Fill(iEvent.luminosityBlock(), dilepMass);
            h_mass_SIT_pass_central->Fill(iEvent.luminosityBlock(), dilepMass);
            h_mass_Glo_pass_central->Fill(iEvent.luminosityBlock(), dilepMass);
          } else {
            h_mass_HLT_pass_forward->Fill(iEvent.luminosityBlock(), dilepMass);
            h_mass_SIT_pass_forward->Fill(iEvent.luminosityBlock(), dilepMass);
            h_mass_Glo_pass_forward->Fill(iEvent.luminosityBlock(), dilepMass);
          }

        } else {
          // category 1HLT: probe passing selection but not trigger
          if (isProbeCentral) {
            h_mass_HLT_fail_central->Fill(iEvent.luminosityBlock(), dilepMass);
            h_mass_SIT_pass_central->Fill(iEvent.luminosityBlock(), dilepMass);
            h_mass_Glo_pass_central->Fill(iEvent.luminosityBlock(), dilepMass);
          } else {
            h_mass_HLT_fail_forward->Fill(iEvent.luminosityBlock(), dilepMass);
            h_mass_SIT_pass_forward->Fill(iEvent.luminosityBlock(), dilepMass);
            h_mass_Glo_pass_forward->Fill(iEvent.luminosityBlock(), dilepMass);
          }
        }
        // category 2HLT + 1HLT: Fill once for Z yield
        h_npv_yield_Z->Fill(iEvent.luminosityBlock(), nvtx);
        h_mass_yield_Z->Fill(iEvent.luminosityBlock(), dilepMass);
        if (isTagCentral && isProbeCentral)
          h_yieldBB_Z->Fill(iEvent.luminosityBlock());
        else if (!isTagCentral && !isProbeCentral)
          h_yieldEE_Z->Fill(iEvent.luminosityBlock());
      } else if (itMu2.isGlobalMuon()) {
        // category Glo: probe is a Global muon but failing selection
        if (isProbeCentral) {
          h_mass_SIT_fail_central->Fill(iEvent.luminosityBlock(), dilepMass);
          h_mass_Glo_pass_central->Fill(iEvent.luminosityBlock(), dilepMass);
        } else {
          h_mass_SIT_fail_forward->Fill(iEvent.luminosityBlock(), dilepMass);
          h_mass_Glo_pass_forward->Fill(iEvent.luminosityBlock(), dilepMass);
        }
      } else if (itMu2.isStandAloneMuon()) {
        // category Sta: probe is a Standalone muon
        if (isProbeCentral) {
          h_mass_Glo_fail_central->Fill(iEvent.luminosityBlock(), dilepMass);
        } else {
          h_mass_Glo_fail_forward->Fill(iEvent.luminosityBlock(), dilepMass);
        }
      } else if (itMu2.innerTrack()->hitPattern().trackerLayersWithMeasurement() >= 6 &&
                 itMu2.innerTrack()->hitPattern().numberOfValidPixelHits() >= 1) {
        // cateogry Trk: probe is a tracker track
        if (isProbeCentral) {
          h_mass_Glo_fail_central->Fill(iEvent.luminosityBlock(), dilepMass);
        } else {
          h_mass_Glo_fail_forward->Fill(iEvent.luminosityBlock(), dilepMass);
        }
      }

    }  // End of probe loop over muons

    // Probe loop over tracks, only for standalone efficiency calculation
    for (auto const& itTrk : *hTrackProduct) {
      // Check track is not a muon
      bool isMuon = false;
      for (auto const& itMu : *hMuonProduct) {
        if (itMu.innerTrack().isNonnull() && itMu.innerTrack().get() == &itTrk) {
          isMuon = true;
          break;
        }
      }
      if (isMuon)
        continue;

      float pt2 = itTrk.pt();
      float eta2 = itTrk.eta();
      float phi2 = itTrk.phi();
      float q2 = itTrk.charge();

      // Probe selection:  kinematic cuts and opposite charge requirement
      if (pt2 < PtCutL2_)
        continue;
      if (fabs(eta2) > EtaCutL2_)
        continue;
      if (q1 == q2)
        continue;

      vTrack.SetPtEtaPhiM(pt2, eta2, phi2, MUON_MASS);

      TLorentzVector vDilep = vTag + vTrack;
      float dilepMass = vDilep.M();
      if ((dilepMass < MassMin_) || (dilepMass > MassMax_))
        continue;

      bool isTrackCentral = false;
      if (fabs(eta2) < MUON_BOUND)
        isTrackCentral = true;

      if (itTrk.hitPattern().trackerLayersWithMeasurement() >= 6 && itTrk.hitPattern().numberOfValidPixelHits() >= 1) {
        if (isTrackCentral)
          h_mass_Glo_fail_central->Fill(iEvent.luminosityBlock(), dilepMass);
        else
          h_mass_Glo_fail_forward->Fill(iEvent.luminosityBlock(), dilepMass);
      }

    }  //End of probe loop over tracks

  }  //End of tag loop
}
void ZCounting::analyzeElectrons(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::LogInfo("ZCounting") << "ZCounting::analyzeElectrons" << std::endl;

  //-------------------------------
  //--- Vertex
  //-------------------------------
  edm::Handle<reco::VertexCollection> hVertexProduct;
  iEvent.getByToken(fPVName_token, hVertexProduct);
  if (!hVertexProduct.isValid())
    return;

  const reco::VertexCollection* pvCol = hVertexProduct.product();
  int nvtx = 0;

  for (auto const& vtx : *pvCol) {
    if (vtx.isFake())
      continue;
    if (vtx.tracksSize() < VtxNTracksFitCut_)
      continue;
    if (vtx.ndof() < VtxNdofCut_)
      continue;
    if (fabs(vtx.z()) > VtxAbsZCut_)
      continue;
    if (vtx.position().Rho() > VtxRhoCut_)
      continue;

    nvtx++;
  }

  // Good vertex requirement
  if (nvtx == 0)
    return;

  //-------------------------------
  //--- Trigger
  //-------------------------------
  edm::Handle<edm::TriggerResults> hTrgRes;
  iEvent.getByToken(fHLTTag_token, hTrgRes);
  if (!hTrgRes.isValid())
    return;

  edm::Handle<trigger::TriggerEvent> hTrgEvt;
  iEvent.getByToken(fHLTObjTag_token, hTrgEvt);

  const edm::TriggerNames& triggerNames = iEvent.triggerNames(*hTrgRes);
  Bool_t config_changed = false;
  if (fTriggerNamesID != triggerNames.parameterSetID()) {
    fTriggerNamesID = triggerNames.parameterSetID();
    config_changed = true;
  }
  if (config_changed) {
    initHLT(*hTrgRes, triggerNames);
  }

  TriggerBits triggerBits;
  for (unsigned int irec = 0; irec < fTrigger->fRecords.size(); irec++) {
    if (fTrigger->fRecords[irec].hltPathIndex == (unsigned int)-1)
      continue;
    if (hTrgRes->accept(fTrigger->fRecords[irec].hltPathIndex)) {
      triggerBits[fTrigger->fRecords[irec].baconTrigBit] = true;
    }
  }

  // Trigger requirement
  if (!isElectronTrigger(*fTrigger, triggerBits))
    return;

  // Get Electrons
  edm::Handle<edm::View<reco::GsfElectron>> electrons;
  iEvent.getByToken(fGsfElectronName_token, electrons);

  // Get SuperClusters
  edm::Handle<edm::View<reco::SuperCluster>> superclusters;
  iEvent.getByToken(fSCName_token, superclusters);

  // Get Rho
  edm::Handle<double> rhoHandle;
  iEvent.getByToken(fRhoToken, rhoHandle);
  EleID_.setRho(*rhoHandle);

  // Get beamspot
  edm::Handle<reco::BeamSpot> beamspotHandle;
  iEvent.getByToken(fBeamspotToken, beamspotHandle);

  // Conversions
  edm::Handle<reco::ConversionCollection> conversionsHandle;
  iEvent.getByToken(fConversionToken, conversionsHandle);

  edm::Ptr<reco::GsfElectron> eleProbe;
  enum { eEleEle2HLT = 1, eEleEle1HLT1L1, eEleEle1HLT, eEleEleNoSel, eEleSC };  // event category enum

  // Loop over Tags
  for (size_t itag = 0; itag < electrons->size(); ++itag) {
    const auto el1 = electrons->ptrAt(itag);
    if (not EleID_.passID(el1, beamspotHandle, conversionsHandle))
      continue;

    float pt1 = el1->pt();
    float eta1 = el1->eta();
    float phi1 = el1->phi();

    if (!isElectronTriggerObj(*fTrigger, TriggerTools::matchHLT(eta1, phi1, fTrigger->fRecords, *hTrgEvt)))
      continue;
    TLorentzVector vTag(0., 0., 0., 0.);
    vTag.SetPtEtaPhiM(pt1, eta1, phi1, ELECTRON_MASS);

    // Tag selection: kinematic cuts, lepton selection and trigger matching
    double tag_pt = vTag.Pt();
    double tag_abseta = fabs(vTag.Eta());

    bool tag_is_valid_tag = ele_tag_selection(tag_pt, tag_abseta);
    bool tag_is_valid_probe = ele_probe_selection(tag_pt, tag_abseta);

    if (not(tag_is_valid_tag or tag_is_valid_probe))
      continue;

    // Loop over probes
    for (size_t iprobe = 0; iprobe < superclusters->size(); ++iprobe) {
      // Initialize probe
      const auto sc = superclusters->ptrAt(iprobe);
      if (*sc == *(el1->superCluster())) {
        continue;
      }

      // Find matching electron
      for (size_t iele = 0; iele < electrons->size(); ++iele) {
        if (iele == itag)
          continue;
        const auto ele = electrons->ptrAt(iele);
        if (*sc == *(ele->superCluster())) {
          eleProbe = ele;
          break;
        }
      }

      // Assign final probe 4-vector
      TLorentzVector vProbe(0., 0., 0., 0.);
      if (eleProbe.isNonnull()) {
        vProbe.SetPtEtaPhiM(eleProbe->pt(), eleProbe->eta(), eleProbe->phi(), ELECTRON_MASS);
      } else {
        double pt = sc->energy() * sqrt(1 - pow(tanh(sc->eta()), 2));
        vProbe.SetPtEtaPhiM(pt, sc->eta(), sc->phi(), ELECTRON_MASS);
      }

      // Probe Selection
      double probe_pt = vProbe.Pt();
      double probe_abseta = fabs(sc->eta());
      bool probe_is_valid_probe = ele_probe_selection(probe_pt, probe_abseta);
      if (!probe_is_valid_probe)
        continue;

      // Good Probe found!

      // Require good Z
      TLorentzVector vDilep = vTag + vProbe;

      if ((vDilep.M() < ELE_MASS_CUT_LOW) || (vDilep.M() > ELE_MASS_CUT_HIGH))
        continue;
      if (eleProbe.isNonnull() and (eleProbe->charge() != -el1->charge()))
        continue;

      // Good Z found!

      long ls = iEvent.luminosityBlock();
      bool probe_pass_trigger = isElectronTriggerObj(
          *fTrigger, TriggerTools::matchHLT(vProbe.Eta(), vProbe.Phi(), fTrigger->fRecords, *hTrgEvt));
      bool probe_pass_id = eleProbe.isNonnull() and EleID_.passID(eleProbe, beamspotHandle, conversionsHandle);

      //// Fill for yields
      bool probe_is_forward = probe_abseta > ELE_ETA_CRACK_LOW;
      bool tag_is_forward = tag_abseta > ELE_ETA_CRACK_LOW;

      if (probe_pass_id) {
        if (probe_is_forward and tag_is_forward) {
          h_ee_yield_Z_eeee->Fill(ls);
        } else if (!probe_is_forward and !tag_is_forward) {
          h_ee_yield_Z_ebeb->Fill(ls);
        } else {
          h_ee_yield_Z_ebee->Fill(ls);
        }
      }

      if (!tag_is_valid_tag)
        continue;

      /// Fill for ID efficiency
      if (probe_pass_id) {
        if (probe_is_forward) {
          h_ee_mass_id_pass_forward->Fill(ls, vDilep.M());
        } else {
          h_ee_mass_id_pass_central->Fill(ls, vDilep.M());
        }
      } else {
        if (probe_is_forward) {
          h_ee_mass_id_fail_forward->Fill(ls, vDilep.M());
        } else {
          h_ee_mass_id_fail_central->Fill(ls, vDilep.M());
        }
      }

      /// Fill for HLT efficiency
      if (probe_pass_id and probe_pass_trigger) {
        if (probe_is_forward) {
          h_ee_mass_HLT_pass_forward->Fill(ls, vDilep.M());
        } else {
          h_ee_mass_HLT_pass_central->Fill(ls, vDilep.M());
        }
      } else if (probe_pass_id) {
        if (probe_is_forward) {
          h_ee_mass_HLT_fail_forward->Fill(ls, vDilep.M());
        } else {
          h_ee_mass_HLT_fail_central->Fill(ls, vDilep.M());
        }
      }
    }  // End of probe loop
  }    //End of tag loop
}

bool ZCounting::ele_probe_selection(double pt, double abseta) {
  if (pt < ELE_PT_CUT_PROBE)
    return false;
  if (abseta > ELE_ETA_CUT_PROBE)
    return false;
  if ((abseta > ELE_ETA_CRACK_LOW) and (abseta < ELE_ETA_CRACK_HIGH))
    return false;
  return true;
}
bool ZCounting::ele_tag_selection(double pt, double abseta) {
  if (pt < ELE_PT_CUT_TAG)
    return false;
  if (abseta > ELE_ETA_CUT_TAG)
    return false;
  if ((abseta > ELE_ETA_CRACK_LOW) and (abseta < ELE_ETA_CRACK_HIGH))
    return false;
  return true;
}
//
// -------------------------------------- functions --------------------------------------------
//

void ZCounting::initHLT(const edm::TriggerResults& result, const edm::TriggerNames& triggerNames) {
  for (unsigned int irec = 0; irec < fTrigger->fRecords.size(); irec++) {
    fTrigger->fRecords[irec].hltPathName = "";
    fTrigger->fRecords[irec].hltPathIndex = (unsigned int)-1;
    const std::string pattern = fTrigger->fRecords[irec].hltPattern;
    if (edm::is_glob(pattern)) {  // handle pattern with wildcards (*,?)
      std::vector<std::vector<std::string>::const_iterator> matches =
          edm::regexMatch(triggerNames.triggerNames(), pattern);
      if (matches.empty()) {
        edm::LogWarning("ZCounting") << "requested pattern [" << pattern << "] does not match any HLT paths"
                                     << std::endl;
      } else {
        for (auto const& match : matches) {
          fTrigger->fRecords[irec].hltPathName = *match;
        }
      }
    } else {  // take full HLT path name given
      fTrigger->fRecords[irec].hltPathName = pattern;
    }
    // Retrieve index in trigger menu corresponding to HLT path
    unsigned int index = triggerNames.triggerIndex(fTrigger->fRecords[irec].hltPathName);
    if (index < result.size()) {  // check for valid index
      fTrigger->fRecords[irec].hltPathIndex = index;
    }
  }
}

//--------------------------------------------------------------------------------------------------
bool ZCounting::isMuonTrigger(const ZCountingTrigger::TTrigger& triggerMenu, const TriggerBits& hltBits) {
  for (unsigned int i = 0; i < fMuonHLTNames.size(); ++i) {
    if (triggerMenu.pass(fMuonHLTNames.at(i), hltBits))
      return true;
  }
  return false;
}

//--------------------------------------------------------------------------------------------------
bool ZCounting::isMuonTriggerObj(const ZCountingTrigger::TTrigger& triggerMenu, const TriggerObjects& hltMatchBits) {
  for (unsigned int i = 0; i < fMuonHLTNames.size(); ++i) {
    if (triggerMenu.passObj(fMuonHLTNames.at(i), fMuonHLTObjectNames.at(i), hltMatchBits))
      return true;
  }
  return false;
}

//--------------------------------------------------------------------------------------------------
// Definition of the CustomTightID function
bool ZCounting::isCustomTightMuon(const reco::Muon& muon) {
  if (!muon.isPFMuon() || !muon.isGlobalMuon())
    return false;

  bool muID = isGoodMuon(muon, muon::GlobalMuonPromptTight) && (muon.numberOfMatchedStations() > 1);

  bool muIdAndHits = muID && muon.innerTrack()->hitPattern().trackerLayersWithMeasurement() > 5 &&
                     muon.innerTrack()->hitPattern().numberOfValidPixelHits() > 0;

  return muIdAndHits;
}

//--------------------------------------------------------------------------------------------------
bool ZCounting::passMuonID(
    const reco::Muon& muon,
    const reco::Vertex& vtx,
    const MuonIDTypes& idType) {  //Muon ID selection, using internal function "DataFormats/MuonReco/src/MuonSelectors.cc

  if (idType == LooseID && muon::isLooseMuon(muon))
    return true;
  else if (idType == MediumID && muon::isMediumMuon(muon))
    return true;
  else if (idType == TightID && muon::isTightMuon(muon, vtx))
    return true;
  else if (idType == CustomTightID && isCustomTightMuon(muon))
    return true;
  else if (idType == NoneID)
    return true;
  else
    return false;
}
//--------------------------------------------------------------------------------------------------
bool ZCounting::passMuonIso(const reco::Muon& muon,
                            const MuonIsoTypes& isoType,
                            const float isoCut) {  //Muon isolation selection, up-to-date with MUO POG recommendation

  if (isoType == TrackerIso && muon.isolationR03().sumPt < isoCut)
    return true;
  else if (isoType == PFIso &&
           muon.pfIsolationR04().sumChargedHadronPt +
                   std::max(0.,
                            muon.pfIsolationR04().sumNeutralHadronEt + muon.pfIsolationR04().sumPhotonEt -
                                0.5 * muon.pfIsolationR04().sumPUPt) <
               isoCut)
    return true;
  else if (isoType == NoneIso)
    return true;
  else
    return false;
}

//--------------------------------------------------------------------------------------------------
bool ZCounting::isElectronTrigger(ZCountingTrigger::TTrigger triggerMenu, TriggerBits hltBits) {
  return triggerMenu.pass("HLT_Ele35_WPTight_Gsf_v*", hltBits);
}
//--------------------------------------------------------------------------------------------------
bool ZCounting::isElectronTriggerObj(ZCountingTrigger::TTrigger triggerMenu, TriggerObjects hltMatchBits) {
  return triggerMenu.passObj("HLT_Ele35_WPTight_Gsf_v*", "hltEle35noerWPTightGsfTrackIsoFilter", hltMatchBits);
}
DEFINE_FWK_MODULE(ZCounting);
