#include "DQMOffline/L1Trigger/interface/L1TEGammaOffline.h"
#include "DQMOffline/L1Trigger/interface/L1TFillWithinLimits.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"
// Geometry
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "TLorentzVector.h"

#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <string>
#include <sstream>
#include <math.h>
#include <algorithm>

//
// -------------------------------------- Constructor --------------------------------------------
//
L1TEGammaOffline::L1TEGammaOffline(const edm::ParameterSet& ps) :
        theGsfElectronCollection_(
            consumes < reco::GsfElectronCollection > (ps.getParameter < edm::InputTag > ("electronCollection"))),
        thePhotonCollection_(
            consumes < std::vector<reco::Photon> > (ps.getParameter < edm::InputTag > ("photonCollection"))),
        thePVCollection_(consumes < reco::VertexCollection > (ps.getParameter < edm::InputTag > ("PVCollection"))),
        theBSCollection_(consumes < reco::BeamSpot > (ps.getParameter < edm::InputTag > ("beamSpotCollection"))),
        triggerEvent_(consumes < trigger::TriggerEvent > (ps.getParameter < edm::InputTag > ("TriggerEvent"))),
        triggerResults_(consumes < edm::TriggerResults > (ps.getParameter < edm::InputTag > ("TriggerResults"))),
        triggerFilter_(ps.getParameter < edm::InputTag > ("TriggerFilter")),
        triggerPath_(ps.getParameter < std::string > ("TriggerPath")),
        histFolder_(ps.getParameter < std::string > ("histFolder")),
        efficiencyFolder_(histFolder_ + "/efficiency_raw"),
        stage2CaloLayer2EGammaToken_(
            consumes < l1t::EGammaBxCollection > (ps.getParameter < edm::InputTag > ("stage2CaloLayer2EGammaSource"))),
        electronEfficiencyThresholds_(ps.getParameter < std::vector<double> > ("electronEfficiencyThresholds")),
        electronEfficiencyBins_(ps.getParameter < std::vector<double> > ("electronEfficiencyBins"))
{
  edm::LogInfo("L1TEGammaOffline") << "Constructor " << "L1TEGammaOffline::L1TEGammaOffline " << std::endl;
}

//
// -- Destructor
//
L1TEGammaOffline::~L1TEGammaOffline()
{
  edm::LogInfo("L1TEGammaOffline") << "Destructor L1TEGammaOffline::~L1TEGammaOffline " << std::endl;
}

//
// -------------------------------------- beginRun --------------------------------------------
//
void L1TEGammaOffline::dqmBeginRun(edm::Run const &, edm::EventSetup const &)
{
  edm::LogInfo("L1TEGammaOffline") << "L1TEGammaOffline::beginRun" << std::endl;
}
//
// -------------------------------------- bookHistos --------------------------------------------
//
void L1TEGammaOffline::bookHistograms(DQMStore::IBooker & ibooker, edm::Run const &, edm::EventSetup const &)
{
  edm::LogInfo("L1TEGammaOffline") << "L1TEGammaOffline::bookHistograms" << std::endl;

  //book at beginRun
  bookElectronHistos(ibooker);
  bookPhotonHistos(ibooker);
}
//
// -------------------------------------- beginLuminosityBlock --------------------------------------------
//
void L1TEGammaOffline::beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& context)
{
  edm::LogInfo("L1TEGammaOffline") << "L1TEGammaOffline::beginLuminosityBlock" << std::endl;
}

//
// -------------------------------------- Analyze --------------------------------------------
//
void L1TEGammaOffline::analyze(edm::Event const& e, edm::EventSetup const& eSetup)
{
  edm::LogInfo("L1TEGammaOffline") << "L1TEGammaOffline::analyze" << std::endl;

  edm::Handle<reco::VertexCollection> vertexHandle;
  e.getByToken(thePVCollection_, vertexHandle);
  if (!vertexHandle.isValid()) {
    edm::LogError("L1TEGammaOffline") << "invalid collection: vertex " << std::endl;
    return;
  }

  unsigned int nVertex = vertexHandle->size();
  dqmoffline::l1t::fillWithinLimits(h_nVertex_, nVertex);

  // L1T
  fillElectrons(e, nVertex);
//      fillPhotons(e, nVertex);
}

void L1TEGammaOffline::fillElectrons(edm::Event const& e, const unsigned int nVertex)
{
  edm::Handle<l1t::EGammaBxCollection> l1EGamma;
  e.getByToken(stage2CaloLayer2EGammaToken_, l1EGamma);

  edm::Handle<reco::GsfElectronCollection> gsfElectrons;
  e.getByToken(theGsfElectronCollection_, gsfElectrons);

  if (!gsfElectrons.isValid()) {
    edm::LogError("L1TEGammaOffline") << "invalid collection: GSF electrons " << std::endl;
    return;
  }
  if (gsfElectrons->size() == 0) {
    edm::LogError("L1TEGammaOffline") << "empty collection: GSF electrons " << std::endl;
    return;
  }
  if (!l1EGamma.isValid()) {
    edm::LogError("L1TEGammaOffline") << "invalid collection: L1 EGamma " << std::endl;
    return;
  }

  auto leadingGsfElectron = gsfElectrons->front();

  // find corresponding L1 EG
  double minDeltaR = 0.3;
  l1t::EGamma closestL1EGamma;
  bool foundMatch = false;

  int bunchCrossing = 0;
  for (auto egamma = l1EGamma->begin(bunchCrossing); egamma != l1EGamma->end(bunchCrossing); ++egamma) {
    double currentDeltaR = deltaR(egamma->eta(), egamma->phi(), leadingGsfElectron.eta(), leadingGsfElectron.phi());
    if (currentDeltaR > minDeltaR) {
      continue;
    } else {
      minDeltaR = currentDeltaR;
      closestL1EGamma = *egamma;
      foundMatch = true;
    }

  }
//  }

  if (!foundMatch) {
    edm::LogError("L1TEGammaOffline") << "Could not find a matching L1 EGamma " << std::endl;
    return;
  }

  double recoEt = leadingGsfElectron.et();
  double recoEta = leadingGsfElectron.eta();
  double recoPhi = leadingGsfElectron.phi();

  double l1Et = closestL1EGamma.et();
  double l1Eta = closestL1EGamma.eta();
  double l1Phi = closestL1EGamma.phi();

  // if no reco value, relative resolution does not make sense -> sort to overflow
  double outOfBounds = 9999;
  double resolutionEt = recoEt > 0 ? (l1Et - recoEt) / recoEt : outOfBounds;
  double resolutionEta = abs(recoEta) > 0 ? (l1Eta - recoEta) / recoEta : outOfBounds;
  double resolutionPhi = abs(recoPhi) > 0 ? (l1Phi - recoPhi) / recoPhi : outOfBounds;

  using namespace dqmoffline::l1t;
  // eta
  fill2DWithinLimits(h_L1EGammaEtavsElectronEta_, recoEta, l1Eta);
  fillWithinLimits(h_resolutionElectronEta_, resolutionEta);

  if (abs(recoEta) <= 1.479) { // barrel
    // et
    fill2DWithinLimits(h_L1EGammaETvsElectronET_EB_, recoEt, l1Et);
    fill2DWithinLimits(h_L1EGammaETvsElectronET_EB_EE_, recoEt, l1Et);
    //resolution
    fillWithinLimits(h_resolutionElectronET_EB_, resolutionEt);
    fillWithinLimits(h_resolutionElectronET_EB_EE_, resolutionEt);
    // phi
    fill2DWithinLimits(h_L1EGammaPhivsElectronPhi_EB_, recoPhi, l1Phi);
    fill2DWithinLimits(h_L1EGammaPhivsElectronPhi_EB_EE_, recoPhi, l1Phi);
    // resolution
    fillWithinLimits(h_resolutionElectronPhi_EB_, resolutionPhi);
    fillWithinLimits(h_resolutionElectronPhi_EB_EE_, resolutionPhi);

    // turn-ons

    for (auto threshold : electronEfficiencyThresholds_) {
      fillWithinLimits(h_efficiencyElectronET_EB_total_[threshold], recoEt);
      fillWithinLimits(h_efficiencyElectronET_EB_EE_total_[threshold], recoEt);
      if (l1Et > threshold) {
        fillWithinLimits(h_efficiencyElectronET_EB_pass_[threshold], recoEt);
        fillWithinLimits(h_efficiencyElectronET_EB_EE_pass_[threshold], recoEt);
      }
    }

  } else { // end-cap
    // et
    fill2DWithinLimits(h_L1EGammaETvsElectronET_EE_, recoEt, l1Et);
    fill2DWithinLimits(h_L1EGammaETvsElectronET_EB_EE_, recoEt, l1Et);
    //resolution
    fillWithinLimits(h_resolutionElectronET_EE_, resolutionEt);
    fillWithinLimits(h_resolutionElectronET_EB_EE_, resolutionEt);
    // phi
    fill2DWithinLimits(h_L1EGammaPhivsElectronPhi_EE_, recoPhi, l1Phi);
    fill2DWithinLimits(h_L1EGammaPhivsElectronPhi_EB_EE_, recoPhi, l1Phi);
    // resolution
    fillWithinLimits(h_resolutionElectronPhi_EE_, resolutionPhi);
    fillWithinLimits(h_resolutionElectronPhi_EB_EE_, resolutionPhi);

    // turn-ons
    for (auto threshold : electronEfficiencyThresholds_) {
      fillWithinLimits(h_efficiencyElectronET_EE_total_[threshold], recoEt);
      fillWithinLimits(h_efficiencyElectronET_EB_EE_total_[threshold], recoEt);
      if (l1Et > threshold) {
        fillWithinLimits(h_efficiencyElectronET_EE_pass_[threshold], recoEt);
        fillWithinLimits(h_efficiencyElectronET_EB_EE_pass_[threshold], recoEt);
      }
    }
  }
}

void L1TEGammaOffline::fillPhotons(edm::Event const& e, const unsigned int nVertex)
{
  edm::Handle<l1t::EGammaBxCollection> l1EGamma;
  e.getByToken(stage2CaloLayer2EGammaToken_, l1EGamma);

  edm::Handle<reco::Photon> photons;
  e.getByToken(thePhotonCollection_, photons);

  if (!photons.isValid()) {
    edm::LogError("L1TEGammaOffline") << "invalid collection: reco::Photons " << std::endl;
    return;
  }
  if (!l1EGamma.isValid()) {
    edm::LogError("L1TEGammaOffline") << "invalid collection: L1 EGamma " << std::endl;
    return;
  }

  int bunchCrossing = 0;

  for (auto egamma = l1EGamma->begin(bunchCrossing); egamma != l1EGamma->end(bunchCrossing); ++egamma) {
//    double et = egamma->et();
//    double phi = egamma->phi();
//    double eta = egamma->eta();
  }
}

//
// -------------------------------------- endLuminosityBlock --------------------------------------------
//
void L1TEGammaOffline::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup)
{
  edm::LogInfo("L1TEGammaOffline") << "L1TEGammaOffline::endLuminosityBlock" << std::endl;
}

//
// -------------------------------------- endRun --------------------------------------------
//
void L1TEGammaOffline::endRun(edm::Run const& run, edm::EventSetup const& eSetup)
{
  edm::LogInfo("L1TEGammaOffline") << "L1TEGammaOffline::endRun" << std::endl;
}

//
// -------------------------------------- book histograms --------------------------------------------
//
void L1TEGammaOffline::bookElectronHistos(DQMStore::IBooker & ibooker)
{
  ibooker.cd();
  ibooker.setCurrentFolder(histFolder_.c_str());
  h_nVertex_ = ibooker.book1D("nVertex", "Number of event vertices in collection", 40, -0.5, 39.5);
  // electron reco vs L1
  h_L1EGammaETvsElectronET_EB_ = ibooker.book2D("L1EGammaETvsElectronET_HB",
      "L1 EGamma E_{T} vs GSF Electron E_{T} (HB); GSF Electron E_{T} (GeV); L1 EGamma E_{T} (GeV)", 300, 0, 300, 300,
      0, 300);
  h_L1EGammaETvsElectronET_EE_ = ibooker.book2D("L1EGammaETvsElectronET_HE",
      "L1 EGamma E_{T} vs GSF Electron E_{T} (HE); GSF Electron E_{T} (GeV); L1 EGamma E_{T} (GeV)", 300, 0, 300, 300,
      0, 300);
  h_L1EGammaETvsElectronET_EB_EE_ = ibooker.book2D("L1EGammaETvsElectronET_EB_HE",
      "L1 EGamma E_{T} vs GSF Electron E_{T} (HB+HE); GSF Electron E_{T} (GeV); L1 EGamma E_{T} (GeV)", 300, 0, 300,
      300, 0, 300);

  h_L1EGammaPhivsElectronPhi_EB_ = ibooker.book2D("L1EGammaPhivsElectronET_HB",
      "#phi_{electron}^{L1} vs #phi_{electron}^{offline} (HB); #phi_{electron}^{offline}; #phi_{electron}^{L1}", 100,
      -4, 4, 100, -4, 4);
  h_L1EGammaPhivsElectronPhi_EE_ = ibooker.book2D("L1EGammaPhivsElectronET_HE",
      "#phi_{electron}^{L1} vs #phi_{electron}^{offline} (HE); #phi_{electron}^{offline}; #phi_{electron}^{L1}", 100,
      -4, 4, 100, -4, 4);
  h_L1EGammaPhivsElectronPhi_EB_EE_ = ibooker.book2D("L1EGammaPhivsElectronET_EB_HE",
      "#phi_{electron}^{L1} vs #phi_{electron}^{offline} (HB+HE); #phi_{electron}^{offline}; #phi_{electron}^{L1}", 100,
      -4, 4, 100, -4, 4);

  h_L1EGammaEtavsElectronEta_ = ibooker.book2D("L1EGammaEtavsElectronEta",
      "L1 EGamma #eta vs GSF Electron #eta; GSF Electron #eta; L1 EGamma #eta", 100, -10, 10, 100, -10, 10);

  // electron resolutions
  h_resolutionElectronET_EB_ = ibooker.book1D("resolutionElectronET_HB",
      "electron ET resolution (HB); (L1 EGamma E_{T} - GSF Electron E_{T})/GSF Electron E_{T}; events", 50, -1, 1.5);
  h_resolutionElectronET_EE_ = ibooker.book1D("resolutionElectronET_HE",
      "electron ET resolution (HE); (L1 EGamma E_{T} - GSF Electron E_{T})/GSF Electron E_{T}; events", 50, -1, 1.5);
  h_resolutionElectronET_EB_EE_ = ibooker.book1D("resolutionElectronET_EB_HE",
      "electron ET resolution (HB+HE); (L1 EGamma E_{T} - GSF Electron E_{T})/GSF Electron E_{T}; events", 50, -1, 1.5);

  h_resolutionElectronPhi_EB_ =
      ibooker.book1D("resolutionElectronPhi_HB",
          "#phi_{electron} resolution (HB); (#phi_{electron}^{L1} - #phi_{electron}^{offline})/#phi_{electron}^{offline}; events",
          120, -0.3, 0.3);
  h_resolutionElectronPhi_EE_ =
      ibooker.book1D("resolutionElectronPhi_HE",
          "electron #phi resolution (HE); (#phi_{electron}^{L1} - #phi_{electron}^{offline})/#phi_{electron}^{offline}; events",
          120, -0.3, 0.3);
  h_resolutionElectronPhi_EB_EE_ =
      ibooker.book1D("resolutionElectronPhi_EB_HE",
          "electron #phi resolution (HB+HE); (#phi_{electron}^{L1} - #phi_{electron}^{offline})/#phi_{electron}^{offline}; events",
          120, -0.3, 0.3);

  h_resolutionElectronEta_ = ibooker.book1D("resolutionElectronEta",
      "electron #eta resolution  (HB); (L1 EGamma #eta - GSF Electron #eta)/GSF Electron #eta; events", 120, -0.3, 0.3);

  // electron turn-ons
  ibooker.setCurrentFolder(efficiencyFolder_.c_str());
  std::vector<float> electronBins(electronEfficiencyBins_.begin(), electronEfficiencyBins_.end());
  int nBins = electronBins.size() - 1;
  float* electronBinArray = &(electronBins[0]);

  for (auto threshold : electronEfficiencyThresholds_) {
    std::string str_threshold = std::to_string(int(threshold));
    h_efficiencyElectronET_EB_pass_[threshold] = ibooker.book1D(
        "efficiencyElectronET_EB_threshold_" + str_threshold + "_Num",
        "electron efficiency (HB); GSF Electron E_{T} (GeV); events", nBins, electronBinArray);
    h_efficiencyElectronET_EE_pass_[threshold] = ibooker.book1D(
        "efficiencyElectronET_EE_threshold_" + str_threshold + "_Num",
        "electron efficiency (HE); GSF Electron E_{T} (GeV); events", nBins, electronBinArray);
    h_efficiencyElectronET_EB_EE_pass_[threshold] = ibooker.book1D(
        "efficiencyElectronET_EB_EE_threshold_" + str_threshold + "_Num",
        "electron efficiency (HB+HE); GSF Electron E_{T} (GeV); events", nBins, electronBinArray);

    h_efficiencyElectronET_EB_total_[threshold] = ibooker.book1D(
        "efficiencyElectronET_EB_threshold_" + str_threshold + "_Den",
        "electron efficiency (HB); GSF Electron E_{T} (GeV); events", nBins, electronBinArray);
    h_efficiencyElectronET_EE_total_[threshold] = ibooker.book1D(
        "efficiencyElectronET_EE_threshold_" + str_threshold + "_Den",
        "electron efficiency (HE); GSF Electron E_{T} (GeV); events", nBins, electronBinArray);
    h_efficiencyElectronET_EB_EE_total_[threshold] = ibooker.book1D(
        "efficiencyElectronET_EB_EE_threshold_" + str_threshold + "_Den",
        "electron efficiency (HB+HE); GSF Electron E_{T} (GeV); events", nBins, electronBinArray);
  }

  ibooker.cd();
}

void L1TEGammaOffline::bookPhotonHistos(DQMStore::IBooker & ibooker)
{
  ibooker.cd();
  ibooker.setCurrentFolder(histFolder_.c_str());
  ibooker.cd();
}

//define this as a plug-in
DEFINE_FWK_MODULE (L1TEGammaOffline);
