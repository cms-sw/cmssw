#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "HLTriggerOffline/SUSYBSM/interface/SUSY_HLT_PhotonMET.h"

SUSY_HLT_PhotonMET::SUSY_HLT_PhotonMET(const edm::ParameterSet &ps) {
  edm::LogInfo("SUSY_HLT_PhotonMET") << "Constructor SUSY_HLT_PhotonMET::SUSY_HLT_PhotonMET " << std::endl;
  // Get parameters from configuration file
  thePfMETCollection_ = consumes<reco::PFMETCollection>(ps.getParameter<edm::InputTag>("pfMETCollection"));
  thePhotonCollection_ = consumes<reco::PhotonCollection>(ps.getParameter<edm::InputTag>("photonCollection"));
  triggerResults_ = consumes<edm::TriggerResults>(ps.getParameter<edm::InputTag>("TriggerResults"));
  triggerPath_ = ps.getParameter<std::string>("TriggerPath");
  triggerPathBase_ = ps.getParameter<std::string>("TriggerPathBase");
  ptThrOffline_ = ps.getUntrackedParameter<double>("ptThrOffline");
  metThrOffline_ = ps.getUntrackedParameter<double>("metThrOffline");
}

SUSY_HLT_PhotonMET::~SUSY_HLT_PhotonMET() {
  edm::LogInfo("SUSY_HLT_PhotonMET") << "Destructor SUSY_HLT_PhotonMET::~SUSY_HLT_PhotonMET " << std::endl;
}

void SUSY_HLT_PhotonMET::bookHistograms(DQMStore::IBooker &ibooker_, edm::Run const &, edm::EventSetup const &) {
  edm::LogInfo("SUSY_HLT_PhotonMET") << "SUSY_HLT_PhotonMET::bookHistograms" << std::endl;
  // book at beginRun
  bookHistos(ibooker_);
}

void SUSY_HLT_PhotonMET::analyze(edm::Event const &e, edm::EventSetup const &eSetup) {
  edm::LogInfo("SUSY_HLT_PhotonMET") << "SUSY_HLT_PhotonMET::analyze" << std::endl;

  //-------------------------------
  //--- MET
  //-------------------------------
  edm::Handle<reco::PFMETCollection> pfMETCollection;
  e.getByToken(thePfMETCollection_, pfMETCollection);
  if (!pfMETCollection.isValid()) {
    edm::LogError("SUSY_HLT_PhotonMET") << "invalid met collection"
                                        << "\n";
    return;
  }
  //-------------------------------
  //--- Photon
  //-------------------------------
  edm::Handle<reco::PhotonCollection> photonCollection;
  e.getByToken(thePhotonCollection_, photonCollection);
  if (!photonCollection.isValid()) {
    edm::LogError("SUSY_HLT_PhotonMET") << "invalid egamma collection"
                                        << "\n";
    return;
  }

  // check what is in the menu
  edm::Handle<edm::TriggerResults> hltresults;
  e.getByToken(triggerResults_, hltresults);
  if (!hltresults.isValid()) {
    edm::LogError("SUSY_HLT_PhotonMET") << "invalid collection: TriggerResults"
                                        << "\n";
    return;
  }

  // use only events with leading photon in barrel
  if (photonCollection->empty() || abs(photonCollection->begin()->superCluster()->eta()) > 1.4442)
    return;

  // get reco photon and met
  float const recoPhotonPt = !photonCollection->empty() ? photonCollection->begin()->et() : 0;
  float const recoMET = !pfMETCollection->empty() ? pfMETCollection->begin()->et() : 0;
  h_recoPhotonPt->Fill(recoPhotonPt);
  h_recoMet->Fill(recoMET);

  // the actual trigger efficiencies
  bool hasFired = false, hasFiredBaseTrigger = false;
  edm::TriggerNames const &trigNames = e.triggerNames(*hltresults);
  unsigned int const numTriggers = trigNames.size();
  for (unsigned int hltIndex = 0; hltIndex < numTriggers; ++hltIndex) {
    if (trigNames.triggerName(hltIndex).find(triggerPath_) != std::string::npos && hltresults->wasrun(hltIndex) &&
        hltresults->accept(hltIndex))
      hasFired = true;
    if (trigNames.triggerName(hltIndex).find(triggerPathBase_) != std::string::npos && hltresults->wasrun(hltIndex) &&
        hltresults->accept(hltIndex))
      hasFiredBaseTrigger = true;
  }

  if (hasFiredBaseTrigger || !e.isRealData()) {
    // passed base trigger
    if (recoPhotonPt > ptThrOffline_)
      h_metTurnOn_den->Fill(recoMET);
    if (recoMET > metThrOffline_)
      h_photonTurnOn_den->Fill(recoPhotonPt);
    if (hasFired) {
      // passed base and signal trigger
      if (recoPhotonPt > ptThrOffline_)
        h_metTurnOn_num->Fill(recoMET);
      if (recoMET > metThrOffline_)
        h_photonTurnOn_num->Fill(recoPhotonPt);
    }
  }
}

void SUSY_HLT_PhotonMET::bookHistos(DQMStore::IBooker &ibooker_) {
  ibooker_.cd();
  ibooker_.setCurrentFolder("HLT/SUSYBSM/" + triggerPath_);

  // offline quantities
  h_recoPhotonPt = ibooker_.book1D("recoPhotonPt", "reco Photon transverse momentum; p_{T} (GeV)", 20, 0, 1000);
  h_recoMet = ibooker_.book1D("recoMet", "reco Missing transverse energy;E_{T}^{miss} (GeV)", 20, 0, 1000);
  h_metTurnOn_num = ibooker_.book1D("pfMetTurnOn_num", "PF MET Turn On Numerator", 20, 0, 500);
  h_metTurnOn_den = ibooker_.book1D("pfMetTurnOn_den", "PF MET Turn On Denominator", 20, 0, 500);
  h_photonTurnOn_num = ibooker_.book1D("photonTurnOn_num", "Photon Turn On Numerator", 20, 0, 1000);
  h_photonTurnOn_den = ibooker_.book1D("photonTurnOn_den", "Photon Turn On Denominator", 20, 0, 1000);

  ibooker_.cd();
}

// define this as a plug-in
DEFINE_FWK_MODULE(SUSY_HLT_PhotonMET);
