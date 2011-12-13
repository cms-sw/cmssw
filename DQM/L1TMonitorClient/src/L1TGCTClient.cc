#include "DQM/L1TMonitorClient/interface/L1TGCTClient.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/Core/interface/QReport.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

using namespace edm;
using namespace std;

// Define statics for bins etc.
const unsigned int ETABINS = 22;
const float ETAMIN = -0.5;
const float ETAMAX = 21.5;

const unsigned int PHIBINS = 18;
const float PHIMIN = -0.5;
const float PHIMAX = 17.5;

L1TGCTClient::L1TGCTClient(const edm::ParameterSet& ps):
  monitorDir_(ps.getUntrackedParameter<string>("monitorDir","")),
  counterLS_(0), 
  counterEvt_(0), 
  prescaleLS_(ps.getUntrackedParameter<int>("prescaleLS", -1)),
  prescaleEvt_(ps.getUntrackedParameter<int>("prescaleEvt", -1))
{
}

L1TGCTClient::~L1TGCTClient(){}

void L1TGCTClient::beginJob(void)
{
  // Get backendinterface  
  dbe_ = Service<DQMStore>().operator->();

  // Set to directory with ME in
  dbe_->setCurrentFolder(monitorDir_);
  
  l1GctIsoEmOccEta_ = dbe_->book1D("IsoEmOccEta","ISO EM  #eta OCCUPANCY", ETABINS, ETAMIN, ETAMAX);
  l1GctIsoEmOccPhi_ = dbe_->book1D("IsoEmOccPhi","ISO EM  #phi OCCUPANCY", PHIBINS, PHIMIN, PHIMAX);
  l1GctNonIsoEmOccEta_ = dbe_->book1D("NonIsoEmOccEta","NON-ISO EM  #eta OCCUPANCY", ETABINS, ETAMIN, ETAMAX);
  l1GctNonIsoEmOccPhi_ = dbe_->book1D("NonIsoEmOccPhi","NON-ISO EM  #phi OCCUPANCY", PHIBINS, PHIMIN, PHIMAX);
  l1GctAllJetsOccEta_ = dbe_->book1D("AllJetsOccEta","CENTRAL AND FORWARD JET  #eta OCCUPANCY", ETABINS, ETAMIN, ETAMAX);
  l1GctAllJetsOccPhi_ = dbe_->book1D("AllJetsOccPhi","CENTRAL AND FORWARD JET  #phi OCCUPANCY", PHIBINS, PHIMIN, PHIMAX);
  l1GctCenJetsOccEta_ = dbe_->book1D("CenJetsOccEta","CENTRAL JET  #eta OCCUPANCY", ETABINS, ETAMIN, ETAMAX);
  l1GctCenJetsOccPhi_ = dbe_->book1D("CenJetsOccPhi","CENTRAL JET  #phi OCCUPANCY", PHIBINS, PHIMIN, PHIMAX);
  l1GctForJetsOccEta_ = dbe_->book1D("ForJetsOccEta","FORWARD JET  #eta OCCUPANCY", ETABINS, ETAMIN, ETAMAX);
  l1GctForJetsOccPhi_ = dbe_->book1D("ForJetsOccPhi","FORWARD JET  #phi OCCUPANCY", PHIBINS, PHIMIN, PHIMAX);
  l1GctTauJetsOccEta_ = dbe_->book1D("TauJetsOccEta","TAU JET  #eta OCCUPANCY", ETABINS, ETAMIN, ETAMAX);
  l1GctTauJetsOccPhi_ = dbe_->book1D("TauJetsOccPhi","TAU JET  #phi OCCUPANCY", PHIBINS, PHIMIN, PHIMAX);
}

void L1TGCTClient::beginRun(const Run& r, const EventSetup& context) {}

void L1TGCTClient::beginLuminosityBlock(const LuminosityBlock& lumiSeg, const EventSetup& context) {}

void L1TGCTClient::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& c) 
{
  if (dbe_->get("L1T/L1TGCT/IsoEmOccEtaPhi")!=NULL){
    makeXProjection(dbe_->get("L1T/L1TGCT/IsoEmOccEtaPhi")->getTH2F(),l1GctIsoEmOccEta_);
    makeYProjection(dbe_->get("L1T/L1TGCT/IsoEmOccEtaPhi")->getTH2F(),l1GctIsoEmOccPhi_);
  }

  if (dbe_->get("L1T/L1TGCT/NonIsoEmOccEtaPhi")!=NULL){
    makeXProjection(dbe_->get("L1T/L1TGCT/NonIsoEmOccEtaPhi")->getTH2F(),l1GctNonIsoEmOccEta_);
    makeYProjection(dbe_->get("L1T/L1TGCT/NonIsoEmOccEtaPhi")->getTH2F(),l1GctNonIsoEmOccPhi_);
  }

  if (dbe_->get("L1T/L1TGCT/AllJetsOccEtaPhi")!=NULL){
    makeXProjection(dbe_->get("L1T/L1TGCT/AllJetsOccEtaPhi")->getTH2F(),l1GctAllJetsOccEta_);
    makeYProjection(dbe_->get("L1T/L1TGCT/AllJetsOccEtaPhi")->getTH2F(),l1GctAllJetsOccPhi_);
  }

  if (dbe_->get("L1T/L1TGCT/CenJetsOccEtaPhi")!=NULL){
    makeXProjection(dbe_->get("L1T/L1TGCT/CenJetsOccEtaPhi")->getTH2F(),l1GctCenJetsOccEta_);
    makeYProjection(dbe_->get("L1T/L1TGCT/CenJetsOccEtaPhi")->getTH2F(),l1GctCenJetsOccPhi_);
  }

  if (dbe_->get("L1T/L1TGCT/ForJetsOccEtaPhi")!=NULL){
    makeXProjection(dbe_->get("L1T/L1TGCT/ForJetsOccEtaPhi")->getTH2F(),l1GctForJetsOccEta_);
    makeYProjection(dbe_->get("L1T/L1TGCT/ForJetsOccEtaPhi")->getTH2F(),l1GctForJetsOccPhi_);
  }
  
  if (dbe_->get("L1T/L1TGCT/TauJetsOccEtaPhi")!=NULL){
    makeXProjection(dbe_->get("L1T/L1TGCT/TauJetsOccEtaPhi")->getTH2F(),l1GctTauJetsOccEta_);
    makeYProjection(dbe_->get("L1T/L1TGCT/TauJetsOccEtaPhi")->getTH2F(),l1GctTauJetsOccPhi_);
  }
}

void L1TGCTClient::analyze(const Event& e, const EventSetup& context){}

void L1TGCTClient::endRun(const Run& r, const EventSetup& context)
{
  if (dbe_->get("L1T/L1TGCT/IsoEmOccEtaPhi")!=NULL){
    makeXProjection(dbe_->get("L1T/L1TGCT/IsoEmOccEtaPhi")->getTH2F(),l1GctIsoEmOccEta_);
    makeYProjection(dbe_->get("L1T/L1TGCT/IsoEmOccEtaPhi")->getTH2F(),l1GctIsoEmOccPhi_);
  }

  if (dbe_->get("L1T/L1TGCT/NonIsoEmOccEtaPhi")!=NULL){
    makeXProjection(dbe_->get("L1T/L1TGCT/NonIsoEmOccEtaPhi")->getTH2F(),l1GctNonIsoEmOccEta_);
    makeYProjection(dbe_->get("L1T/L1TGCT/NonIsoEmOccEtaPhi")->getTH2F(),l1GctNonIsoEmOccPhi_);
  }

  if (dbe_->get("L1T/L1TGCT/AllJetsOccEtaPhi")!=NULL){
    makeXProjection(dbe_->get("L1T/L1TGCT/AllJetsOccEtaPhi")->getTH2F(),l1GctAllJetsOccEta_);
    makeYProjection(dbe_->get("L1T/L1TGCT/AllJetsOccEtaPhi")->getTH2F(),l1GctAllJetsOccPhi_);
  }

  if (dbe_->get("L1T/L1TGCT/CenJetsOccEtaPhi")!=NULL){
    makeXProjection(dbe_->get("L1T/L1TGCT/CenJetsOccEtaPhi")->getTH2F(),l1GctCenJetsOccEta_);
    makeYProjection(dbe_->get("L1T/L1TGCT/CenJetsOccEtaPhi")->getTH2F(),l1GctCenJetsOccPhi_);
  }

  if (dbe_->get("L1T/L1TGCT/ForJetsOccEtaPhi")!=NULL){
    makeXProjection(dbe_->get("L1T/L1TGCT/ForJetsOccEtaPhi")->getTH2F(),l1GctForJetsOccEta_);
    makeYProjection(dbe_->get("L1T/L1TGCT/ForJetsOccEtaPhi")->getTH2F(),l1GctForJetsOccPhi_);
  }
  
  if (dbe_->get("L1T/L1TGCT/TauJetsOccEtaPhi")!=NULL){
    makeXProjection(dbe_->get("L1T/L1TGCT/TauJetsOccEtaPhi")->getTH2F(),l1GctTauJetsOccEta_);
    makeYProjection(dbe_->get("L1T/L1TGCT/TauJetsOccEtaPhi")->getTH2F(),l1GctTauJetsOccPhi_);
  }
}

void L1TGCTClient::endJob(){}

void L1TGCTClient::makeXProjection(TH2F* input, MonitorElement* output)
{
  // Are the provided input and output consistent
  if (input->GetNbinsX() != output->getNbinsX()) return;
  
  // Make the projection
  TH1D* projX = input->ProjectionX();
  
  for (Int_t i=0; i<projX->GetNbinsX(); i++) {
    output->setBinContent(i+1,projX->GetBinContent(i+1));
  }
  delete projX;
}

void L1TGCTClient::makeYProjection(TH2F* input, MonitorElement* output)
{
  // Are the provided input and output consistent
  if (input->GetNbinsY() != output->getNbinsX()) return;
  
  // Make the projection
  TH1D* projY = input->ProjectionY();
  
  for (Int_t i=0; i<projY->GetNbinsX(); i++) {
    output->setBinContent(i+1,projY->GetBinContent(i+1));
  }
  delete projY;
}


