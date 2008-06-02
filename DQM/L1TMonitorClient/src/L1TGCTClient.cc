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
}

void L1TGCTClient::analyze(const Event& e, const EventSetup& context){}

void L1TGCTClient::endRun(const Run& r, const EventSetup& context){}

void L1TGCTClient::endJob(){}









