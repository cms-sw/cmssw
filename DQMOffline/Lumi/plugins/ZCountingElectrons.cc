#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Utilities/interface/RegexMatch.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include <TLorentzVector.h>

#include <memory>

#include "DQMOffline/Lumi/plugins/ZCountingElectrons.h"

//
// -------------------------------------- Constructor --------------------------------------------
//
ZCountingElectrons::ZCountingElectrons(const edm::ParameterSet& iConfig)
    : triggerResultsInputTag_(iConfig.getParameter<edm::InputTag>("TriggerResults")),
      fPVName_token(consumes<reco::VertexCollection>(
          iConfig.getUntrackedParameter<std::string>("edmPVName", "offlinePrimaryVertices"))),

      // Electron-specific Parameters
      fGsfElectronName_token(consumes<edm::View<reco::GsfElectron>>(
          iConfig.getUntrackedParameter<std::string>("edmGsfEleName", "gedGsfElectrons"))),
      fSCName_token(consumes<edm::View<reco::SuperCluster>>(
          iConfig.getUntrackedParameter<std::string>("edmSCName", "particleFlowEGamma"))),

      // Electron-specific Tags
      fRhoToken(consumes<double>(iConfig.getParameter<edm::InputTag>("rhoname"))),
      fBeamspotToken(consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamspotName"))),
      fConversionToken(consumes<reco::ConversionCollection>(iConfig.getParameter<edm::InputTag>("conversionsName"))),

      // Electron-specific Cuts
      PtCutL1_(iConfig.getUntrackedParameter<double>("PtCutL1")),
      PtCutL2_(iConfig.getUntrackedParameter<double>("PtCutL2")),
      EtaCutL1_(iConfig.getUntrackedParameter<double>("EtaCutL1")),
      EtaCutL2_(iConfig.getUntrackedParameter<double>("EtaCutL2")),

      MassBin_(iConfig.getUntrackedParameter<double>("MassBin")),
      MassMin_(iConfig.getUntrackedParameter<double>("MassMin")),
      MassMax_(iConfig.getUntrackedParameter<double>("MassMax")),

      LumiBin_(iConfig.getUntrackedParameter<double>("LumiBin")),
      LumiMin_(iConfig.getUntrackedParameter<double>("LumiMin")),
      LumiMax_(iConfig.getUntrackedParameter<double>("LumiMax")),

      PVBin_(iConfig.getUntrackedParameter<int>("PVBin")),
      PVMin_(iConfig.getUntrackedParameter<double>("PVMin")),
      PVMax_(iConfig.getUntrackedParameter<double>("PVMax")),

      VtxNTracksFitCut_(iConfig.getUntrackedParameter<double>("VtxNTracksFitMin")),
      VtxNdofCut_(iConfig.getUntrackedParameter<double>("VtxNdofMin")),
      VtxAbsZCut_(iConfig.getUntrackedParameter<double>("VtxAbsZMax")),
      VtxRhoCut_(iConfig.getUntrackedParameter<double>("VtxRhoMax")),

      ELE_ID_WP(iConfig.getUntrackedParameter<std::string>("ElectronIDType", "TIGHT")),
      EleID_(ElectronIdentifier(iConfig)) {
  edm::LogInfo("ZCounting") << "Constructor  ZCountingElectrons::ZCounting " << std::endl;

  // Trigger settings
  triggers = new TriggerTools();
  triggers->setTriggerResultsToken(consumes<edm::TriggerResults>(triggerResultsInputTag_));
  triggers->setTriggerEventToken(consumes<trigger::TriggerEvent>(iConfig.getParameter<edm::InputTag>("TriggerEvent")));
  triggers->setDRMAX(DRMAX);

  edm::LogVerbatim("ZCounting") << "ZCounting::ZCounting set trigger names";
  const std::vector<std::string> patterns_ = iConfig.getParameter<std::vector<std::string>>("ElectronTriggerNames");
  for (const std::string& pattern_ : patterns_) {
    triggers->addTriggerRecord(pattern_);
  }

  EleID_.setID(ELE_ID_WP);
}

//
//  -------------------------------------- Destructor --------------------------------------------
//
ZCountingElectrons::~ZCountingElectrons() {
  edm::LogInfo("ZCountingElectrons") << "Destructor ZCountingElectrons::~ZCountingElectrons " << std::endl;
}

//
// -------------------------------------- beginRun --------------------------------------------
//
void ZCountingElectrons::dqmBeginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {
  edm::LogInfo("ZCountingElectrons") << "ZCountingElectrons::beginRun" << std::endl;

  // initialize triggers

  edm::LogVerbatim("ZCountingElectrons") << "ZCountingElectrons::dqmBeginRun now at " << iRun.id();

  bool hltChanged_ = true;
  if (hltConfigProvider_.init(iRun, iSetup, triggerResultsInputTag_.process(), hltChanged_)) {
    edm::LogVerbatim("ZCountingElectrons")
        << "ZCountingElectrons::dqmBeginRun [TriggerObjMatchValueMapsProducer::beginRun] "
           "HLTConfigProvider initialized [processName() = \""
        << hltConfigProvider_.processName() << "\", tableName() = \"" << hltConfigProvider_.tableName()
        << "\", size() = " << hltConfigProvider_.size() << "]";
  } else {
    edm::LogError("ZCountingElectrons")
        << "ZCountingElectrons::dqmBeginRun Initialization of HLTConfigProvider failed for Run=" << iRun.id()
        << " (process=\"" << triggerResultsInputTag_.process() << "\") -> plugin will not produce outputs for this Run";
    return;
  }

  triggers->initHLTObjects(hltConfigProvider_);
}
//
// -------------------------------------- bookHistos --------------------------------------------
//
void ZCountingElectrons::bookHistograms(DQMStore::IBooker& ibooker_, edm::Run const&, edm::EventSetup const&) {
  edm::LogInfo("ZCountingElectrons") << "ZCountingElectrons::bookHistograms" << std::endl;
  ibooker_.cd();
  ibooker_.setCurrentFolder("ZCounting/Histograms");

  h_npv = ibooker_.book2D(
      "h_npv", "Events with valid primary vertex", LumiBin_, LumiMin_, LumiMax_, PVBin_, PVMin_, PVMax_);

  h_npv->setAxisTitle("luminosity section", 1);
  h_npv->setAxisTitle("number of primary vertices", 2);

  // Electron histograms
  h_ee_mass_id_pass_central = ibooker_.book2D("h_ee_mass_id_pass_central",
                                              "h_ee_mass_id_pass_central",
                                              LumiBin_,
                                              LumiMin_,
                                              LumiMax_,
                                              MassBin_,
                                              MassMin_,
                                              MassMax_);
  h_ee_mass_id_fail_central = ibooker_.book2D("h_ee_mass_id_fail_central",
                                              "h_ee_mass_id_fail_central",
                                              LumiBin_,
                                              LumiMin_,
                                              LumiMax_,
                                              MassBin_,
                                              MassMin_,
                                              MassMax_);
  h_ee_mass_id_pass_forward = ibooker_.book2D("h_ee_mass_id_pass_forward",
                                              "h_ee_mass_id_pass_forward",
                                              LumiBin_,
                                              LumiMin_,
                                              LumiMax_,
                                              MassBin_,
                                              MassMin_,
                                              MassMax_);
  h_ee_mass_id_fail_forward = ibooker_.book2D("h_ee_mass_id_fail_forward",
                                              "h_ee_mass_id_fail_forward",
                                              LumiBin_,
                                              LumiMin_,
                                              LumiMax_,
                                              MassBin_,
                                              MassMin_,
                                              MassMax_);

  h_ee_mass_HLT_pass_central = ibooker_.book2D("h_ee_mass_HLT_pass_central",
                                               "h_ee_mass_HLT_pass_central",
                                               LumiBin_,
                                               LumiMin_,
                                               LumiMax_,
                                               MassBin_,
                                               MassMin_,
                                               MassMax_);
  h_ee_mass_HLT_fail_central = ibooker_.book2D("h_ee_mass_HLT_fail_central",
                                               "h_ee_mass_HLT_fail_central",
                                               LumiBin_,
                                               LumiMin_,
                                               LumiMax_,
                                               MassBin_,
                                               MassMin_,
                                               MassMax_);
  h_ee_mass_HLT_pass_forward = ibooker_.book2D("h_ee_mass_HLT_pass_forward",
                                               "h_ee_mass_HLT_pass_forward",
                                               LumiBin_,
                                               LumiMin_,
                                               LumiMax_,
                                               MassBin_,
                                               MassMin_,
                                               MassMax_);
  h_ee_mass_HLT_fail_forward = ibooker_.book2D("h_ee_mass_HLT_fail_forward",
                                               "h_ee_mass_HLT_fail_forward",
                                               LumiBin_,
                                               LumiMin_,
                                               LumiMax_,
                                               MassBin_,
                                               MassMin_,
                                               MassMax_);

  h_ee_yield_Z_ebeb = ibooker_.book1D("h_ee_yield_Z_ebeb", "h_ee_yield_Z_ebeb", LumiBin_, LumiMin_, LumiMax_);
  h_ee_yield_Z_ebee = ibooker_.book1D("h_ee_yield_Z_ebee", "h_ee_yield_Z_ebee", LumiBin_, LumiMin_, LumiMax_);
  h_ee_yield_Z_eeee = ibooker_.book1D("h_ee_yield_Z_eeee", "h_ee_yield_Z_eeee", LumiBin_, LumiMin_, LumiMax_);
}

//
// -------------------------------------- Analyze --------------------------------------------
//
//--------------------------------------------------------------------------------------------------
void ZCountingElectrons::analyze(const edm::Event& iEvent,
                                 const edm::EventSetup& iSetup) {  // Fill event tree on the fly
  edm::LogInfo("ZCountingElectrons") << "ZCountingElectrons::analyze" << std::endl;

  //-------------------------------
  //--- Vertex
  //-------------------------------
  edm::Handle<reco::VertexCollection> hVertexProduct;
  iEvent.getByToken(fPVName_token, hVertexProduct);
  if (!hVertexProduct.isValid()) {
    edm::LogWarning("ZCounting") << "ZCountingElectrons::analyze - no valid primary vertex product found" << std::endl;
    return;
  }
  // const reco::VertexCollection* pvCol = hVertexProduct.product();
  // const reco::Vertex* pv = &(*pvCol->begin());
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

    // if (nvtx == 0) {
    //   pv = &itVtx;
    // }
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

    if (!triggers->passObj(eta1, phi1))
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

      if ((vDilep.M() < MassMin_) || (vDilep.M() > MassMax_))
        continue;
      if (eleProbe.isNonnull() and (eleProbe->charge() != -el1->charge()))
        continue;

      // Good Z found!
      long ls = iEvent.luminosityBlock();
      bool probe_pass_trigger = triggers->passObj(vProbe.Eta(), vProbe.Phi());
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

//
// -------------------------------------- functions --------------------------------------------
//

bool ZCountingElectrons::ele_probe_selection(double pt, double abseta) {
  if (pt < PtCutL2_)
    return false;
  if (abseta > EtaCutL2_)
    return false;
  if ((abseta > ELE_ETA_CRACK_LOW) and (abseta < ELE_ETA_CRACK_HIGH))
    return false;
  return true;
}

bool ZCountingElectrons::ele_tag_selection(double pt, double abseta) {
  if (pt < PtCutL1_)
    return false;
  if (abseta > EtaCutL1_)
    return false;
  if ((abseta > ELE_ETA_CRACK_LOW) and (abseta < ELE_ETA_CRACK_HIGH))
    return false;
  return true;
}

DEFINE_FWK_MODULE(ZCountingElectrons);
