#include "DQM/RPCMonitorClient/interface/RPCMonitorLinkSynchro.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondFormats/RPCObjects/interface/RPCEMap.h"
#include "CondFormats/DataRecord/interface/RPCEMapRcd.h"
#include "CondFormats/RPCObjects/interface/RPCReadOutMapping.h"

#include "DQM/RPCMonitorClient/interface/RPCLinkSynchroHistoMaker.h"


using namespace edm;

RPCMonitorLinkSynchro::RPCMonitorLinkSynchro( const edm::ParameterSet& cfg) 
    : theConfig(cfg), theCabling(0), 
      me_delaySummary(0), me_delaySpread(0)
//      ,me_linksLowStat(0), me_linksBadSynchro(0), me_linksMostNoisy(0) 
{ }

RPCMonitorLinkSynchro::~RPCMonitorLinkSynchro()
{ 
  LogTrace("") << "RPCMonitorLinkSynchro destructor"; 
}

void RPCMonitorLinkSynchro::endLuminosityBlock(const LuminosityBlock& ls, const EventSetup& es)
{
  LogTrace("") << "RPCMonitorLinkSynchro END OF LUMI BLOCK CALLED!";

  if (theCablingWatcher.check(es)) {
    delete theCabling;
    ESHandle<RPCEMap> readoutMapping;
    LogTrace("") << "record has CHANGED!!, initialise readout map!";
    es.get<RPCEMapRcd>().get(readoutMapping);
    theCabling = readoutMapping->convert();
    LogTrace("") <<" READOUT MAP VERSION: " << theCabling->version();
  }
  RPCLinkSynchroHistoMaker hm(theSynchro,theCabling);
  hm.fillDelaySpreadHisto(me_delaySpread->getTH2F());
  me_delaySpread->update();
  
//  hm.fillLinksBadSynchro(me_linksBadSynchro->getTH2F());
//  me_linksBadSynchro->update();
//  hm.fillLinksLowStat(me_linksLowStat->getTH2F());
//  me_linksLowStat->update();
//  hm.fillLinksMostNoisy(me_linksMostNoisy->getTH2F());
//  me_linksMostNoisy->update();
}

void RPCMonitorLinkSynchro::beginJob(const edm::EventSetup&)
{
  DQMStore* dmbe = edm::Service<DQMStore>().operator->();
  dmbe->setCurrentFolder("RPC/FEDIntegrity/");
  RPCLinkSynchroHistoMaker hm(theSynchro);

  me_delaySummary = dmbe->book1D("delaySummary","LinkDelaySummary",8,-3.5, 4.5);
  me_delaySummary->getTH1F()->SetStats(111);

  me_delaySpread = dmbe->book2D("delaySpread","LinkDelaySpread",80,-3.5, 4.5,40,0.,4.);
  me_delaySpread->getTH2F()->SetStats(0);

//  me_linksBadSynchro = dmbe->book2D("linksBadSynchro","LinkBadSynchro",8,0., 8.,10,0.,10.);
//  me_linksBadSynchro->getTH2F()->SetStats(0);
//  me_linksLowStat = dmbe->book2D("linksLowStat","LinkLowStat",8,0., 8.,10,0.,10.);
//  me_linksLowStat->getTH2F()->SetStats(0);
//  me_linksMostNoisy = dmbe->book2D("linksMostNoisy","LinkMostNoisy",8,0., 8.,10,0.,10.);
//  me_linksMostNoisy->getTH2F()->SetStats(0);
  
}

void RPCMonitorLinkSynchro::endJob()
{
  if (theConfig.getUntrackedParameter<bool>("dumpDelays"))
    LogInfo("RPCMonitorLinkSynchro DumpDelays") << RPCLinkSynchroHistoMaker(theSynchro,theCabling).dumpDelays(); 
}

void RPCMonitorLinkSynchro::analyze(const edm::Event& ev, const edm::EventSetup& es)
{
  edm::Handle<RPCRawSynchro::ProdItem> synchroCounts;
  ev.getByType(synchroCounts);
  theSynchro.add( *synchroCounts.product());

  typedef RPCRawSynchro::ProdItem::const_iterator IT;
  for (IT it=synchroCounts->begin(); it != synchroCounts->end(); ++it) me_delaySummary->Fill( it->second-3);
  me_delaySummary->update();
}
