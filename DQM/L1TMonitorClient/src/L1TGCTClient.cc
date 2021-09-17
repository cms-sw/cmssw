#include "DQM/L1TMonitorClient/interface/L1TGCTClient.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/Core/interface/DQMStore.h"

using namespace edm;
using namespace std;

// Define statics for bins etc.
const unsigned int ETABINS = 22;
const float ETAMIN = -0.5;
const float ETAMAX = 21.5;

const unsigned int PHIBINS = 18;
const float PHIMIN = -0.5;
const float PHIMAX = 17.5;

L1TGCTClient::L1TGCTClient(const edm::ParameterSet& ps)
    : monitorDir_(ps.getUntrackedParameter<string>("monitorDir", "")),
      counterLS_(0),
      counterEvt_(0),
      prescaleLS_(ps.getUntrackedParameter<int>("prescaleLS", -1)),
      prescaleEvt_(ps.getUntrackedParameter<int>("prescaleEvt", -1)),
      m_runInEventLoop(ps.getUntrackedParameter<bool>("runInEventLoop", false)),
      m_runInEndLumi(ps.getUntrackedParameter<bool>("runInEndLumi", false)),
      m_runInEndRun(ps.getUntrackedParameter<bool>("runInEndRun", false)),
      m_runInEndJob(ps.getUntrackedParameter<bool>("runInEndJob", false)),
      m_stage1_layer2_(ps.getUntrackedParameter<bool>("stage1_layer2_"))

{}

L1TGCTClient::~L1TGCTClient() {}

void L1TGCTClient::book(DQMStore::IBooker& ibooker) {
  // Set to directory with ME in
  ibooker.setCurrentFolder(monitorDir_);

  l1GctIsoEmOccEta_ = ibooker.book1D("IsoEmOccEta", "ISO EM  #eta OCCUPANCY", ETABINS, ETAMIN, ETAMAX);
  l1GctIsoEmOccPhi_ = ibooker.book1D("IsoEmOccPhi", "ISO EM  #phi OCCUPANCY", PHIBINS, PHIMIN, PHIMAX);
  l1GctNonIsoEmOccEta_ = ibooker.book1D("NonIsoEmOccEta", "NON-ISO EM  #eta OCCUPANCY", ETABINS, ETAMIN, ETAMAX);
  l1GctNonIsoEmOccPhi_ = ibooker.book1D("NonIsoEmOccPhi", "NON-ISO EM  #phi OCCUPANCY", PHIBINS, PHIMIN, PHIMAX);
  l1GctAllJetsOccEta_ =
      ibooker.book1D("AllJetsOccEta", "CENTRAL AND FORWARD JET  #eta OCCUPANCY", ETABINS, ETAMIN, ETAMAX);
  l1GctAllJetsOccPhi_ =
      ibooker.book1D("AllJetsOccPhi", "CENTRAL AND FORWARD JET  #phi OCCUPANCY", PHIBINS, PHIMIN, PHIMAX);
  l1GctCenJetsOccEta_ = ibooker.book1D("CenJetsOccEta", "CENTRAL JET  #eta OCCUPANCY", ETABINS, ETAMIN, ETAMAX);
  l1GctCenJetsOccPhi_ = ibooker.book1D("CenJetsOccPhi", "CENTRAL JET  #phi OCCUPANCY", PHIBINS, PHIMIN, PHIMAX);
  l1GctForJetsOccEta_ = ibooker.book1D("ForJetsOccEta", "FORWARD JET  #eta OCCUPANCY", ETABINS, ETAMIN, ETAMAX);
  l1GctForJetsOccPhi_ = ibooker.book1D("ForJetsOccPhi", "FORWARD JET  #phi OCCUPANCY", PHIBINS, PHIMIN, PHIMAX);
  l1GctTauJetsOccEta_ = ibooker.book1D("TauJetsOccEta", "TAU JET  #eta OCCUPANCY", ETABINS, ETAMIN, ETAMAX);
  l1GctTauJetsOccPhi_ = ibooker.book1D("TauJetsOccPhi", "TAU JET  #phi OCCUPANCY", PHIBINS, PHIMIN, PHIMAX);
  if (m_stage1_layer2_ == true) {
    l1GctIsoTauJetsOccEta_ = ibooker.book1D("IsoTauJetsOccEta", "TAU JET #eta OCCUPANCY", ETABINS, ETAMIN, ETAMAX);
    l1GctIsoTauJetsOccPhi_ = ibooker.book1D("IsoTauJetsOccPhi", "TAU JET #phi OCCUPANCY", PHIBINS, PHIMIN, PHIMAX);
  }
}

void L1TGCTClient::dqmEndLuminosityBlock(DQMStore::IBooker& ibooker,
                                         DQMStore::IGetter& igetter,
                                         const edm::LuminosityBlock& lumiSeg,
                                         const edm::EventSetup& c) {
  if (m_runInEndLumi) {
    book(ibooker);
    processHistograms(igetter);
  }
}

void L1TGCTClient::dqmEndJob(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) {
  if (m_runInEndRun) {
    book(ibooker);
    processHistograms(igetter);
  }
}

void L1TGCTClient::processHistograms(DQMStore::IGetter& igetter) {
  MonitorElement* Input;

  if (m_stage1_layer2_ == false) {
    InputDir = "L1T/L1TGCT/";
  } else {
    InputDir = "L1T/L1TStage1Layer2/";
  }

  Input = igetter.get(InputDir + "IsoEmOccEtaPhi");
  if (Input != nullptr) {
    makeXProjection(Input->getTH2F(), l1GctIsoEmOccEta_);
    makeYProjection(Input->getTH2F(), l1GctIsoEmOccPhi_);
  }

  Input = igetter.get(InputDir + "NonIsoEmOccEtaPhi");
  if (Input != nullptr) {
    makeXProjection(Input->getTH2F(), l1GctNonIsoEmOccEta_);
    makeYProjection(Input->getTH2F(), l1GctNonIsoEmOccPhi_);
  }

  Input = igetter.get(InputDir + "AllJetsOccEtaPhi");
  if (Input != nullptr) {
    makeXProjection(Input->getTH2F(), l1GctAllJetsOccEta_);
    makeYProjection(Input->getTH2F(), l1GctAllJetsOccPhi_);
  }

  Input = igetter.get(InputDir + "CenJetsOccEtaPhi");
  if (Input != nullptr) {
    makeXProjection(Input->getTH2F(), l1GctCenJetsOccEta_);
    makeYProjection(Input->getTH2F(), l1GctCenJetsOccPhi_);
  }

  Input = igetter.get(InputDir + "ForJetsOccEtaPhi");
  if (Input != nullptr) {
    makeXProjection(Input->getTH2F(), l1GctForJetsOccEta_);
    makeYProjection(Input->getTH2F(), l1GctForJetsOccPhi_);
  }

  Input = igetter.get(InputDir + "TauJetsOccEtaPhi");
  if (Input != nullptr) {
    makeXProjection(Input->getTH2F(), l1GctTauJetsOccEta_);
    makeYProjection(Input->getTH2F(), l1GctTauJetsOccPhi_);
  }

  if (m_stage1_layer2_ == true) {
    Input = igetter.get(InputDir + "IsoTauJetsOccEtaPhi");
    if (Input != nullptr) {
      makeXProjection(Input->getTH2F(), l1GctIsoTauJetsOccEta_);
      makeYProjection(Input->getTH2F(), l1GctIsoTauJetsOccPhi_);
    }
  }
}

void L1TGCTClient::makeXProjection(TH2F* input, MonitorElement* output) {
  // Are the provided input and output consistent
  if (input->GetNbinsX() != output->getNbinsX())
    return;

  // Make the projection
  TH1D* projX = input->ProjectionX();

  for (Int_t i = 0; i < projX->GetNbinsX(); i++) {
    output->setBinContent(i + 1, projX->GetBinContent(i + 1));
  }
  delete projX;
}

void L1TGCTClient::makeYProjection(TH2F* input, MonitorElement* output) {
  // Are the provided input and output consistent
  if (input->GetNbinsY() != output->getNbinsX())
    return;

  // Make the projection
  TH1D* projY = input->ProjectionY();

  for (Int_t i = 0; i < projY->GetNbinsX(); i++) {
    output->setBinContent(i + 1, projY->GetBinContent(i + 1));
  }
  delete projY;
}
