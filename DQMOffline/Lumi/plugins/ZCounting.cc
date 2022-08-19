#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Utilities/interface/RegexMatch.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"

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
  triggers->setDRMAX(DRMAX);

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

  h_mass_SIT_fail_BB = ibooker_.book2D("h_mass_SIT_fail_BB",
                                       "Muon SIT failing barrel-barrel",
                                       LumiBin_,
                                       LumiMin_,
                                       LumiMax_,
                                       MassBin_,
                                       MassMin_,
                                       MassMax_);
  h_mass_SIT_fail_BE = ibooker_.book2D("h_mass_SIT_fail_BE",
                                       "Muon SIT failing barrel-endcap",
                                       LumiBin_,
                                       LumiMin_,
                                       LumiMax_,
                                       MassBin_,
                                       MassMin_,
                                       MassMax_);

  h_mass_SIT_fail_EE = ibooker_.book2D("h_mass_SIT_fail_EE",
                                       "Muon SIT failing endcap-endcap",
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

  h_npv = ibooker_.book2D(
      "h_npv", "Events with valid primary vertex", LumiBin_, LumiMin_, LumiMax_, PVBin_, PVMin_, PVMax_);

  // Axis titles
  h_mass_2HLT_BB->setAxisTitle("luminosity section", 1);
  h_mass_2HLT_BE->setAxisTitle("luminosity section", 1);
  h_mass_2HLT_EE->setAxisTitle("luminosity section", 1);
  h_mass_1HLT_BB->setAxisTitle("luminosity section", 1);
  h_mass_1HLT_BE->setAxisTitle("luminosity section", 1);
  h_mass_1HLT_EE->setAxisTitle("luminosity section", 1);
  h_mass_SIT_fail_BB->setAxisTitle("luminosity section", 1);
  h_mass_SIT_fail_BE->setAxisTitle("luminosity section", 1);
  h_mass_SIT_fail_EE->setAxisTitle("luminosity section", 1);
  h_mass_Glo_fail_BB->setAxisTitle("luminosity section", 1);
  h_mass_Glo_fail_BE->setAxisTitle("luminosity section", 1);
  h_mass_Glo_fail_EE->setAxisTitle("luminosity section", 1);
  h_mass_2HLT_BB->setAxisTitle("tag and probe mass", 2);
  h_mass_2HLT_BE->setAxisTitle("tag and probe mass", 2);
  h_mass_2HLT_EE->setAxisTitle("tag and probe mass", 2);
  h_mass_1HLT_BB->setAxisTitle("tag and probe mass", 2);
  h_mass_1HLT_BE->setAxisTitle("tag and probe mass", 2);
  h_mass_1HLT_EE->setAxisTitle("tag and probe mass", 2);
  h_mass_SIT_fail_BB->setAxisTitle("tag and probe mass", 2);
  h_mass_SIT_fail_BE->setAxisTitle("tag and probe mass", 2);
  h_mass_SIT_fail_EE->setAxisTitle("tag and probe mass", 2);
  h_mass_Glo_fail_BB->setAxisTitle("tag and probe mass", 2);
  h_mass_Glo_fail_BE->setAxisTitle("tag and probe mass", 2);
  h_mass_Glo_fail_EE->setAxisTitle("tag and probe mass", 2);
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
    const float pt1 = itMu1.muonBestTrack()->pt();
    const float eta1 = itMu1.muonBestTrack()->eta();
    const float phi1 = itMu1.muonBestTrack()->phi();
    const float q1 = itMu1.muonBestTrack()->charge();

    // Tag selection: kinematic cuts, lepton selection and trigger matching
    if (pt1 < PtCutL1_)
      continue;
    if (fabs(eta1) > EtaCutL1_)
      continue;
    if (!(passMuonID(itMu1, pv) && passMuonIso(itMu1)))
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

      bool isProbeCentral = false;
      if (fabs(eta2) < MUON_BOUND)
        isProbeCentral = true;

      // Determine event category for efficiency calculation
      if (passMuonID(itMu2, pv) && passMuonIso(itMu2)) {
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
      } else if (itMu2.isGlobalMuon()) {
        // category Glo: probe is a Global muon but failing selection
        if (isTagCentral && isProbeCentral) {
          h_mass_SIT_fail_BB->Fill(iEvent.luminosityBlock(), dilepMass);
        } else if (!isTagCentral && !isProbeCentral) {
          h_mass_SIT_fail_EE->Fill(iEvent.luminosityBlock(), dilepMass);
        } else {
          h_mass_SIT_fail_BE->Fill(iEvent.luminosityBlock(), dilepMass);
        }
      } else if (itMu2.isStandAloneMuon()) {
        // category Sta: probe is a Standalone muon
        if (isTagCentral && isProbeCentral) {
          h_mass_Glo_fail_BB->Fill(iEvent.luminosityBlock(), dilepMass);
        } else if (!isTagCentral && !isProbeCentral) {
          h_mass_Glo_fail_EE->Fill(iEvent.luminosityBlock(), dilepMass);
        } else {
          h_mass_Glo_fail_BE->Fill(iEvent.luminosityBlock(), dilepMass);
        }
      } else if (itMu2.innerTrack()->hitPattern().trackerLayersWithMeasurement() >= 6 &&
                 itMu2.innerTrack()->hitPattern().numberOfValidPixelHits() >= 1) {
        // cateogry Trk: probe is a tracker track
        if (isTagCentral && isProbeCentral) {
          h_mass_Glo_fail_BB->Fill(iEvent.luminosityBlock(), dilepMass);
        } else if (!isTagCentral && !isProbeCentral) {
          h_mass_Glo_fail_EE->Fill(iEvent.luminosityBlock(), dilepMass);
        } else {
          h_mass_Glo_fail_BE->Fill(iEvent.luminosityBlock(), dilepMass);
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

      vTrack.SetPtEtaPhiM(pt2, eta2, phi2, MUON_MASS);

      TLorentzVector vDilep = vTag + vTrack;
      float dilepMass = vDilep.M();
      if ((dilepMass < MassMin_) || (dilepMass > MassMax_))
        continue;

      bool isTrackCentral = false;
      if (fabs(eta2) < MUON_BOUND)
        isTrackCentral = true;

      if (itTrk.hitPattern().trackerLayersWithMeasurement() >= 6 && itTrk.hitPattern().numberOfValidPixelHits() >= 1) {
        if (isTagCentral && isTrackCentral) {
          h_mass_Glo_fail_BB->Fill(iEvent.luminosityBlock(), dilepMass);
        } else if (!isTagCentral && !isTrackCentral) {
          h_mass_Glo_fail_EE->Fill(iEvent.luminosityBlock(), dilepMass);
        } else {
          h_mass_Glo_fail_BE->Fill(iEvent.luminosityBlock(), dilepMass);
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
  if (!muon.isPFMuon() || !muon.isGlobalMuon())
    return false;

  bool muID = isGoodMuon(muon, muon::GlobalMuonPromptTight) && (muon.numberOfMatchedStations() > 1);

  bool muIdAndHits = muID && muon.innerTrack()->hitPattern().trackerLayersWithMeasurement() > 5 &&
                     muon.innerTrack()->hitPattern().numberOfValidPixelHits() > 0;

  return muIdAndHits;
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
