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
  
  l1GctIsoEmHotChannelEtaPhiMap_     = dbe_->book2D("IsoEmHotChannelEtaMap","ISO EM HOT CHANNELS",
                                                    ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);
  l1GctIsoEmDeadChannelEtaPhiMap_    = dbe_->book2D("IsoEmDeadChannelEtaMap","ISO EM DEAD CHANNELS",
                                                 ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);
  l1GctNonIsoEmHotChannelEtaPhiMap_  = dbe_->book2D("NonIsoEmHotChannelEtaMap","NON-ISO EM HOT CHANNELS",
                                                 ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);
  l1GctNonIsoEmDeadChannelEtaPhiMap_ = dbe_->book2D("NonIsoEmDeadChannelEtaMap","NON-ISO EM DEAD CHANNELS",
                                                 ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);
  l1GctAllJetsHotChannelEtaPhiMap_   = dbe_->book2D("ForJetsHotChannelEtaMap","FOR JETS HOT CHANNELS",
                                                 ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);
  l1GctAllJetsDeadChannelEtaPhiMap_  = dbe_->book2D("ForJetsDeadChannelEtaMap","FOR JETS DEAD CHANNELS",
                                                 ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);
  l1GctTauJetsHotChannelEtaPhiMap_   = dbe_->book2D("TauJetsHotChannelEtaMap","TAU JETS HOT CHANNELS",
                                                 ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);
  l1GctTauJetsDeadChannelEtaPhiMap_  = dbe_->book2D("TauJetsDeadChannelEtaMap","TAU JETS DEAD CHANNELS",
                                                 ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);
}

void L1TGCTClient::beginRun(const Run& r, const EventSetup& context) {}

void L1TGCTClient::beginLuminosityBlock(const LuminosityBlock& lumiSeg, const EventSetup& context) {}

void L1TGCTClient::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& c)
{

  // Get results of Q tests.
  // Will code this in a more elegant way in the future!

  // Iso EM
  MonitorElement *IsoEmHotEtaPhiChannels = dbe_->get("L1T/L1TGCT/IsoEmRankEtaPhi");
  if (IsoEmHotEtaPhiChannels){
    const QReport *IsoEmHotEtaPhiQReport = IsoEmHotEtaPhiChannels->getQReport("HotChannels_GCT_2D");
    if (IsoEmHotEtaPhiQReport) {
      vector<dqm::me_util::Channel> badChannels = IsoEmHotEtaPhiQReport->getBadChannels();
      for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); 
	   channel != badChannels.end(); channel++) {
	l1GctIsoEmHotChannelEtaPhiMap_->setBinContent((*channel).getBin(),(*channel).getContents());
      }
    } 
  }

  MonitorElement *IsoEmDeadEtaPhiChannels = dbe_->get("L1T/L1TGCT/IsoEmRankEtaPhi");
  if (IsoEmDeadEtaPhiChannels){
    const QReport *IsoEmDeadEtaPhiQReport = IsoEmDeadEtaPhiChannels->getQReport("DeadChannels_GCT_2D_loose");
    if (IsoEmDeadEtaPhiQReport) {
      vector<dqm::me_util::Channel> badChannels = IsoEmDeadEtaPhiQReport->getBadChannels();
      for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); 
	   channel != badChannels.end(); channel++) {
	l1GctIsoEmDeadChannelEtaPhiMap_->setBinContent((*channel).getBin(),(*channel).getContents());
      }
    } 
  }

  // Non-Iso EM
  MonitorElement *NonIsoEmHotEtaPhiChannels = dbe_->get("L1T/L1TGCT/NonIsoEmRankEtaPhi");
  if (NonIsoEmHotEtaPhiChannels){
    const QReport *NonIsoEmHotEtaPhiQReport = NonIsoEmHotEtaPhiChannels->getQReport("HotChannels_GCT_2D");
    if (NonIsoEmHotEtaPhiQReport) {
      vector<dqm::me_util::Channel> badChannels = NonIsoEmHotEtaPhiQReport->getBadChannels();
      for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); 
	   channel != badChannels.end(); channel++) {
	l1GctNonIsoEmHotChannelEtaPhiMap_->setBinContent((*channel).getBin(),(*channel).getContents());
      }
    } 
  }

  MonitorElement *NonIsoEmDeadEtaPhiChannels = dbe_->get("L1T/L1TGCT/NonIsoEmRankEtaPhi");
  if (NonIsoEmDeadEtaPhiChannels){
    const QReport *NonIsoEmDeadEtaPhiQReport = NonIsoEmDeadEtaPhiChannels->getQReport("DeadChannels_GCT_2D_loose");
    if (NonIsoEmDeadEtaPhiQReport) {
      vector<dqm::me_util::Channel> badChannels = NonIsoEmDeadEtaPhiQReport->getBadChannels();
      for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); 
	   channel != badChannels.end(); channel++) {
	l1GctNonIsoEmDeadChannelEtaPhiMap_->setBinContent((*channel).getBin(),(*channel).getContents());
      }
    } 
  }

  // Tau jets
  MonitorElement *TauJetsHotEtaPhiChannels = dbe_->get("L1T/L1TGCT/TauJetsEtEtaPhi");
  if (TauJetsHotEtaPhiChannels){
    const QReport *TauJetsHotEtaPhiQReport = TauJetsHotEtaPhiChannels->getQReport("HotChannels_GCT_2D");
    if (TauJetsHotEtaPhiQReport) {
      vector<dqm::me_util::Channel> badChannels = TauJetsHotEtaPhiQReport->getBadChannels();
      for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); 
	   channel != badChannels.end(); channel++) {
	l1GctTauJetsHotChannelEtaPhiMap_->setBinContent((*channel).getBin(),(*channel).getContents());
      }
    } 
  }

  MonitorElement *TauJetsDeadEtaPhiChannels = dbe_->get("L1T/L1TGCT/TauJetsEtEtaPhi");
  if (TauJetsDeadEtaPhiChannels){
    const QReport *TauJetsDeadEtaPhiQReport = TauJetsDeadEtaPhiChannels->getQReport("DeadChannels_GCT_2D_loose");
    if (TauJetsDeadEtaPhiQReport) {
      vector<dqm::me_util::Channel> badChannels = TauJetsDeadEtaPhiQReport->getBadChannels();
      for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); 
	   channel != badChannels.end(); channel++) {
	l1GctTauJetsDeadChannelEtaPhiMap_->setBinContent((*channel).getBin(),(*channel).getContents());
      }
    } 
  }

  // All jets
  MonitorElement *AllJetsHotEtaPhiChannels = dbe_->get("L1T/L1TGCT/AllJetsEtEtaPhi");
  if (AllJetsHotEtaPhiChannels){
    const QReport *AllJetsHotEtaPhiQReport = AllJetsHotEtaPhiChannels->getQReport("HotChannels_GCT_2D");
    if (AllJetsHotEtaPhiQReport) {
      vector<dqm::me_util::Channel> badChannels = AllJetsHotEtaPhiQReport->getBadChannels();
      for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); 
	   channel != badChannels.end(); channel++) {
	l1GctAllJetsHotChannelEtaPhiMap_->setBinContent((*channel).getBin(),(*channel).getContents());
      }
    } 
  }

  MonitorElement *AllJetsDeadEtaPhiChannels = dbe_->get("L1T/L1TGCT/AllJetsEtEtaPhi");
  if (AllJetsDeadEtaPhiChannels){
    const QReport *AllJetsDeadEtaPhiQReport = AllJetsDeadEtaPhiChannels->getQReport("DeadChannels_GCT_2D_loose");
    if (AllJetsDeadEtaPhiQReport) {
      vector<dqm::me_util::Channel> badChannels = AllJetsDeadEtaPhiQReport->getBadChannels();
      for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); 
	   channel != badChannels.end(); channel++) {
	l1GctAllJetsDeadChannelEtaPhiMap_->setBinContent((*channel).getBin(),(*channel).getContents());
      }
    } 
  }


}

void L1TGCTClient::analyze(const Event& e, const EventSetup& context){}

void L1TGCTClient::endRun(const Run& r, const EventSetup& context){}

void L1TGCTClient::endJob(){}
