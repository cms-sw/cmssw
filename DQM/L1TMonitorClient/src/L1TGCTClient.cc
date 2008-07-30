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

void L1TGCTClient::beginJob(const EventSetup& context)
{
  // Get backendinterface  
  dbe_ = Service<DQMStore>().operator->();

  // Set to directory with ME in
  dbe_->setCurrentFolder(monitorDir_);
  
  l1GctIsoEmHotChannelEtaMap_     = dbe_->book1D("IsoEmHotChannelEtaMap","ISO EM HOT ETA CHANNELS",
                                                ETABINS, ETAMIN, ETAMAX);
  l1GctIsoEmHotChannelPhiMap_     = dbe_->book1D("IsoEmHotChannelPhiMap","ISO EM HOT PHI CHANNELS",
                                                 PHIBINS, PHIMIN, PHIMAX); 		    
  l1GctIsoEmDeadChannelEtaMap_    = dbe_->book1D("IsoEmDeadChannelEtaMap","ISO EM DEAD ETA CHANNELS",
                                                 ETABINS, ETAMIN, ETAMAX);
  l1GctIsoEmDeadChannelPhiMap_    = dbe_->book1D("IsoEmDeadChannelPhiMap","ISO EM DEAD PHI CHANNELS",
                                                 PHIBINS, PHIMIN, PHIMAX); 		    
  l1GctNonIsoEmHotChannelEtaMap_  = dbe_->book1D("NonIsoEmHotChannelEtaMap","NON-ISO EM HOT ETA CHANNELS",
                                                 ETABINS, ETAMIN, ETAMAX);
  l1GctNonIsoEmHotChannelPhiMap_  = dbe_->book1D("NonIsoEmHotChannelPhiMap","NON-ISO EM HOT PHI CHANNELS",
                                                 PHIBINS, PHIMIN, PHIMAX); 		    
  l1GctNonIsoEmDeadChannelEtaMap_ = dbe_->book1D("NonIsoEmDeadChannelEtaMap","NON-ISO EM DEAD ETA CHANNELS",
                                                 ETABINS, ETAMIN, ETAMAX);
  l1GctNonIsoEmDeadChannelPhiMap_ = dbe_->book1D("NonIsoEmDeadChannelPhiMap","NON-ISO EM DEAD PHI CHANNELS",
                                                 PHIBINS, PHIMIN, PHIMAX); 		    
  l1GctForJetsHotChannelEtaMap_   = dbe_->book1D("ForJetsHotChannelEtaMap","FOR JETS HOT ETA CHANNELS",
                                                 ETABINS, ETAMIN, ETAMAX);
  l1GctForJetsHotChannelPhiMap_   = dbe_->book1D("ForJetsHotChannelPhiMap","FOR JETS HOT PHI CHANNELS",
                                                 PHIBINS, PHIMIN, PHIMAX); 		    
  l1GctForJetsDeadChannelEtaMap_  = dbe_->book1D("ForJetsDeadChannelEtaMap","FOR JETS DEAD ETA CHANNELS",
                                                 ETABINS, ETAMIN, ETAMAX);
  l1GctForJetsDeadChannelPhiMap_  = dbe_->book1D("ForJetsDeadChannelPhiMap","FOR JETS DEAD PHI CHANNELS",
                                                 PHIBINS, PHIMIN, PHIMAX); 		    
  l1GctCenJetsHotChannelEtaMap_   = dbe_->book1D("CenJetsHotChannelEtaMap","CEN JETS HOT ETA CHANNELS",
                                                 ETABINS, ETAMIN, ETAMAX);
  l1GctCenJetsHotChannelPhiMap_   = dbe_->book1D("CenJetsHotChannelPhiMap","CEN JETS HOT PHI CHANNELS",
                                                 PHIBINS, PHIMIN, PHIMAX); 		    
  l1GctCenJetsDeadChannelEtaMap_  = dbe_->book1D("CenJetsDeadChannelEtaMap","CEN JETS DEAD ETA CHANNELS",
                                                 ETABINS, ETAMIN, ETAMAX);
  l1GctCenJetsDeadChannelPhiMap_  = dbe_->book1D("CenJetsDeadChannelPhiMap","CEN JETS DEAD PHI CHANNELS",
                                                 PHIBINS, PHIMIN, PHIMAX); 		
  l1GctTauJetsHotChannelEtaMap_   = dbe_->book1D("TauJetsHotChannelEtaMap","TAU JETS HOT ETA CHANNELS",
                                                 ETABINS, ETAMIN, ETAMAX);
  l1GctTauJetsHotChannelPhiMap_   = dbe_->book1D("TauJetsHotChannelPhiMap","TAU JETS HOT PHI CHANNELS",
                                                 PHIBINS, PHIMIN, PHIMAX); 		    
  l1GctTauJetsDeadChannelEtaMap_  = dbe_->book1D("TauJetsDeadChannelEtaMap","TAU JETS DEAD ETA CHANNELS",
                                                 ETABINS, ETAMIN, ETAMAX);
  l1GctTauJetsDeadChannelPhiMap_  = dbe_->book1D("TauJetsDeadChannelPhiMap","TAU JETS DEAD PHI CHANNELS",
                                                 PHIBINS, PHIMIN, PHIMAX); 		    
}

void L1TGCTClient::beginRun(const Run& r, const EventSetup& context) {}

void L1TGCTClient::beginLuminosityBlock(const LuminosityBlock& lumiSeg, const EventSetup& context) {}

void L1TGCTClient::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& c)
{

  // Get results of Q tests.
  // Will code this in a more elegant way in the future!

  // Iso EM
  MonitorElement *IsoEmHotEtaChannels = dbe_->get("L1T/L1TGCT/IsoEmOccEta");
  if (IsoEmHotEtaChannels){
    const QReport *IsoEmHotEtaQReport = IsoEmHotEtaChannels->getQReport("HotChannels");
    if (IsoEmHotEtaQReport) {
      vector<dqm::me_util::Channel> badChannels = IsoEmHotEtaQReport->getBadChannels();
      for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); 
	   channel != badChannels.end(); channel++) {
	l1GctIsoEmHotChannelEtaMap_->setBinContent((*channel).getBin(),(*channel).getContents());
      }
    } 
  }

  MonitorElement *IsoEmHotPhiChannels = dbe_->get("L1T/L1TGCT/IsoEmOccPhi");
  if (IsoEmHotPhiChannels){
    const QReport *IsoEmHotPhiQReport = IsoEmHotPhiChannels->getQReport("HotChannels");
    if (IsoEmHotPhiQReport) {
      vector<dqm::me_util::Channel> badChannels = IsoEmHotPhiQReport->getBadChannels();
      for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); 
	   channel != badChannels.end(); channel++) {
	l1GctIsoEmHotChannelPhiMap_->setBinContent((*channel).getBin(),(*channel).getContents());
      }
    } 
  }

  MonitorElement *IsoEmDeadEtaChannels = dbe_->get("L1T/L1TGCT/IsoEmOccEta");
  if (IsoEmDeadEtaChannels){
    const QReport *IsoEmDeadEtaQReport = IsoEmDeadEtaChannels->getQReport("DeadChannels");
    if (IsoEmDeadEtaQReport) {
      vector<dqm::me_util::Channel> badChannels = IsoEmDeadEtaQReport->getBadChannels();
      for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); 
	   channel != badChannels.end(); channel++) {
	l1GctIsoEmDeadChannelEtaMap_->setBinContent((*channel).getBin(),(*channel).getContents());
      }
    } 
  }

  MonitorElement *IsoEmDeadPhiChannels = dbe_->get("L1T/L1TGCT/IsoEmOccPhi");
  if (IsoEmDeadPhiChannels){
    const QReport *IsoEmDeadPhiQReport = IsoEmDeadPhiChannels->getQReport("DeadChannels");
    if (IsoEmDeadPhiQReport) {
      vector<dqm::me_util::Channel> badChannels = IsoEmDeadPhiQReport->getBadChannels();
      for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); 
	   channel != badChannels.end(); channel++) {
	l1GctIsoEmDeadChannelPhiMap_->setBinContent((*channel).getBin(),(*channel).getContents());
      }
    } 
  }
  
  // Non-iso EM
  MonitorElement *NonIsoEmHotEtaChannels = dbe_->get("L1T/L1TGCT/NonIsoEmOccEta");
  if (NonIsoEmHotEtaChannels){
    const QReport *NonIsoEmHotEtaQReport = NonIsoEmHotEtaChannels->getQReport("HotChannels");
    if (NonIsoEmHotEtaQReport) {
      vector<dqm::me_util::Channel> badChannels = NonIsoEmHotEtaQReport->getBadChannels();
      for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); 
	   channel != badChannels.end(); channel++) {
	l1GctNonIsoEmHotChannelEtaMap_->setBinContent((*channel).getBin(),(*channel).getContents());
      }
    } 
  }

  MonitorElement *NonIsoEmHotPhiChannels = dbe_->get("L1T/L1TGCT/NonIsoEmOccPhi");
  if (NonIsoEmHotPhiChannels){
    const QReport *NonIsoEmHotPhiQReport = NonIsoEmHotPhiChannels->getQReport("HotChannels");
    if (NonIsoEmHotPhiQReport) {
      vector<dqm::me_util::Channel> badChannels = NonIsoEmHotPhiQReport->getBadChannels();
      for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); 
	   channel != badChannels.end(); channel++) {
	l1GctNonIsoEmHotChannelPhiMap_->setBinContent((*channel).getBin(),(*channel).getContents());
      }
    } 
  }

  MonitorElement *NonIsoEmDeadEtaChannels = dbe_->get("L1T/L1TGCT/NonIsoEmOccEta");
  if (NonIsoEmDeadEtaChannels){
    const QReport *NonIsoEmDeadEtaQReport = NonIsoEmDeadEtaChannels->getQReport("DeadChannels");
    if (NonIsoEmDeadEtaQReport) {
      vector<dqm::me_util::Channel> badChannels = NonIsoEmDeadEtaQReport->getBadChannels();
      for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); 
	   channel != badChannels.end(); channel++) {
	l1GctNonIsoEmDeadChannelEtaMap_->setBinContent((*channel).getBin(),(*channel).getContents());
      }
    } 
  }

  MonitorElement *NonIsoEmDeadPhiChannels = dbe_->get("L1T/L1TGCT/NonIsoEmOccPhi");
  if (NonIsoEmDeadPhiChannels){
    const QReport *NonIsoEmDeadPhiQReport = NonIsoEmDeadPhiChannels->getQReport("DeadChannels");
    if (NonIsoEmDeadPhiQReport) {
      vector<dqm::me_util::Channel> badChannels = NonIsoEmDeadPhiQReport->getBadChannels();
      for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); 
	   channel != badChannels.end(); channel++) {
	l1GctNonIsoEmDeadChannelPhiMap_->setBinContent((*channel).getBin(),(*channel).getContents());
      }
    } 
  }
  
  // Forward Jets
  MonitorElement *ForJetsHotEtaChannels = dbe_->get("L1T/L1TGCT/ForJetsOccEta");
  if (ForJetsHotEtaChannels){
    const QReport *ForJetsHotEtaQReport = ForJetsHotEtaChannels->getQReport("HotChannels");
	if (ForJetsHotEtaQReport) {
	vector<dqm::me_util::Channel> badChannels = ForJetsHotEtaQReport->getBadChannels();
	for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); 
	  channel != badChannels.end(); channel++ ) {
	l1GctForJetsHotChannelEtaMap_->setBinContent((*channel).getBin(),(*channel).getContents());
	  }
	}
  }

  MonitorElement *ForJetsHotPhiChannels = dbe_->get("L1T/L1TGCT/ForJetsOccPhi");
  if (ForJetsHotPhiChannels){
    const QReport *ForJetsHotPhiQReport = ForJetsHotPhiChannels->getQReport("HotChannels");
	if (ForJetsHotPhiQReport) {
	vector<dqm::me_util::Channel> badChannels = ForJetsHotPhiQReport->getBadChannels();
	for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); 
	  channel != badChannels.end(); channel++ ) {
	l1GctForJetsHotChannelPhiMap_->setBinContent((*channel).getBin(),(*channel).getContents());
	  }
	}
  }

  MonitorElement *ForJetsDeadEtaChannels = dbe_->get("L1T/L1TGCT/ForJetsOccEta");
  if (ForJetsDeadEtaChannels){
    const QReport *ForJetsDeadEtaQReport = ForJetsDeadEtaChannels->getQReport("DeadChannels");
	if (ForJetsDeadEtaQReport) {
	vector<dqm::me_util::Channel> badChannels = ForJetsDeadEtaQReport->getBadChannels();
	for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); 
	  channel != badChannels.end(); channel++ ) {
	l1GctForJetsDeadChannelEtaMap_->setBinContent((*channel).getBin(),(*channel).getContents());
	  }
	}
  }

  MonitorElement *ForJetsDeadPhiChannels = dbe_->get("L1T/L1TGCT/ForJetsOccPhi");
  if (ForJetsDeadPhiChannels){
    const QReport *ForJetsDeadPhiQReport = ForJetsDeadPhiChannels->getQReport("DeadChannels");
	if (ForJetsDeadPhiQReport) {
	vector<dqm::me_util::Channel> badChannels = ForJetsDeadPhiQReport->getBadChannels();
	for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); 
	  channel != badChannels.end(); channel++ ) {
	l1GctForJetsDeadChannelPhiMap_->setBinContent((*channel).getBin(),(*channel).getContents());
	  }
	}
  }

  //Central Jets
  MonitorElement *CenJetsHotEtaChannels = dbe_->get("L1T/L1TGCT/CenJetsOccEta");
  if (CenJetsHotEtaChannels){
    const QReport *CenJetsHotEtaQReport = CenJetsHotEtaChannels->getQReport("HotChannels");
	if (CenJetsHotEtaQReport) {
	vector<dqm::me_util::Channel> badChannels = CenJetsHotEtaQReport->getBadChannels();
	for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); 
	  channel != badChannels.end(); channel++ ) {
	l1GctCenJetsHotChannelEtaMap_->setBinContent((*channel).getBin(),(*channel).getContents());
	  }
	}
  }

  MonitorElement *CenJetsHotPhiChannels = dbe_->get("L1T/L1TGCT/CenJetsOccPhi");
  if (CenJetsHotPhiChannels){
    const QReport *CenJetsHotPhiQReport = CenJetsHotPhiChannels->getQReport("HotChannels");
	if (CenJetsHotPhiQReport) {
	vector<dqm::me_util::Channel> badChannels = CenJetsHotPhiQReport->getBadChannels();
	for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); 
	  channel != badChannels.end(); channel++ ) {
	l1GctCenJetsHotChannelPhiMap_->setBinContent((*channel).getBin(),(*channel).getContents());
	  }
	}
  }

  MonitorElement *CenJetsDeadEtaChannels = dbe_->get("L1T/L1TGCT/CenJetsOccEta");
  if (CenJetsDeadEtaChannels){
    const QReport *CenJetsDeadEtaQReport = CenJetsDeadEtaChannels->getQReport("DeadChannels");
	if (CenJetsDeadEtaQReport) {
	vector<dqm::me_util::Channel> badChannels = CenJetsDeadEtaQReport->getBadChannels();
	for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); 
	  channel != badChannels.end(); channel++ ) {
	l1GctCenJetsDeadChannelEtaMap_->setBinContent((*channel).getBin(),(*channel).getContents());
	  }
	}
  }

  MonitorElement *CenJetsDeadPhiChannels = dbe_->get("L1T/L1TGCT/CenJetsOccPhi");
  if (CenJetsDeadPhiChannels){
    const QReport *CenJetsDeadPhiQReport = CenJetsDeadPhiChannels->getQReport("DeadChannels");
	if (CenJetsDeadPhiQReport) {
	vector<dqm::me_util::Channel> badChannels = CenJetsDeadPhiQReport->getBadChannels();
	for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); 
	  channel != badChannels.end(); channel++ ) {
	l1GctCenJetsDeadChannelPhiMap_->setBinContent((*channel).getBin(),(*channel).getContents());
	  }
	}
  }

  //Tau Jets
  MonitorElement *TauJetsHotEtaChannels = dbe_->get("L1T/L1TGCT/TauJetsOccEta");
  if (TauJetsHotEtaChannels){
    const QReport *TauJetsHotEtaQReport = TauJetsHotEtaChannels->getQReport("HotChannels");
	if (TauJetsHotEtaQReport) {
	vector<dqm::me_util::Channel> badChannels = TauJetsHotEtaQReport->getBadChannels();
	for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); 
	  channel != badChannels.end(); channel++ ) {
	l1GctTauJetsHotChannelEtaMap_->setBinContent((*channel).getBin(),(*channel).getContents());
	  }
	}
  }

  MonitorElement *TauJetsHotPhiChannels = dbe_->get("L1T/L1TGCT/TauJetsOccPhi");
  if (TauJetsHotPhiChannels){
    const QReport *TauJetsHotPhiQReport = TauJetsHotPhiChannels->getQReport("HotChannels");
	if (TauJetsHotPhiQReport) {
	vector<dqm::me_util::Channel> badChannels = TauJetsHotPhiQReport->getBadChannels();
	for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); 
	  channel != badChannels.end(); channel++ ) {
	l1GctTauJetsHotChannelPhiMap_->setBinContent((*channel).getBin(),(*channel).getContents());
	  }
	}
  }

  MonitorElement *TauJetsDeadEtaChannels = dbe_->get("L1T/L1TGCT/TauJetsOccEta");
  if (TauJetsDeadEtaChannels){
    const QReport *TauJetsDeadEtaQReport = TauJetsDeadEtaChannels->getQReport("DeadChannels");
	if (TauJetsDeadEtaQReport) {
	vector<dqm::me_util::Channel> badChannels = TauJetsDeadEtaQReport->getBadChannels();
	for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); 
	  channel != badChannels.end(); channel++ ) {
	l1GctTauJetsDeadChannelEtaMap_->setBinContent((*channel).getBin(),(*channel).getContents());
	  }
	}
  }

  MonitorElement *TauJetsDeadPhiChannels = dbe_->get("L1T/L1TGCT/TauJetsOccPhi");
  if (TauJetsDeadPhiChannels){
    const QReport *TauJetsDeadPhiQReport = TauJetsDeadPhiChannels->getQReport("DeadChannels");
	if (TauJetsDeadPhiQReport) {
	vector<dqm::me_util::Channel> badChannels = TauJetsDeadPhiQReport->getBadChannels();
	for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); 
	  channel != badChannels.end(); channel++ ) {
	l1GctTauJetsDeadChannelPhiMap_->setBinContent((*channel).getBin(),(*channel).getContents());
	  }
	}
  }

}

void L1TGCTClient::analyze(const Event& e, const EventSetup& context){}

void L1TGCTClient::endRun(const Run& r, const EventSetup& context){}

void L1TGCTClient::endJob(){}
