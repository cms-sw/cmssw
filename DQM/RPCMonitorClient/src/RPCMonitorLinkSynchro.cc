#include "DQM/RPCMonitorClient/interface/RPCMonitorLinkSynchro.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondFormats/RPCObjects/interface/RPCEMap.h"
#include "CondFormats/DataRecord/interface/RPCEMapRcd.h"
#include "CondFormats/RPCObjects/interface/RPCReadOutMapping.h"

#include "CondFormats/DataRecord/interface/RPCEMapRcd.h"

#include "DQM/RPCMonitorClient/interface/RPCLinkSynchroHistoMaker.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TFile.h"

//#include "UserCode/konec/test/R2DTimerObserver.h"



RPCMonitorLinkSynchro::RPCMonitorLinkSynchro( const edm::ParameterSet& cfg) 
    : theConfig(cfg),
      theSynchroStat(RPCLinkSynchroStat(theConfig.getUntrackedParameter<bool>("useFirstHitOnly", false))),
      rpcRawSynchroProdItemTag_(cfg.getParameter<edm::InputTag>("rpcRawSynchroProdItemTag"))
{ 
}

RPCMonitorLinkSynchro::~RPCMonitorLinkSynchro()
{ 
}

void RPCMonitorLinkSynchro::beginRun(const edm::Run&, const edm::EventSetup& es)
{
  if (theCablingWatcher.check(es)) {
    edm::ESTransientHandle<RPCEMap> readoutMapping;
    es.get<RPCEMapRcd>().get(readoutMapping);
    RPCReadOutMapping * cabling = readoutMapping->convert();
    edm::LogInfo("RPCMonitorLinkSynchro") << "RPCMonitorLinkSynchro - record has CHANGED!!, read map, VERSION: " << cabling->version();
    theSynchroStat.init(cabling, theConfig.getUntrackedParameter<bool>("dumpDelays"));
    delete cabling;
  }
}

void RPCMonitorLinkSynchro::endLuminosityBlock(const edm::LuminosityBlock& ls, const edm::EventSetup& es)
{

  RPCLinkSynchroHistoMaker hm(theSynchroStat);
  hm.fill(me_delaySummary->getTH1F(), me_delaySpread->getTH2F(), me_topOccup->getTH2F(), me_topSpread->getTH2F());
}

void RPCMonitorLinkSynchro::beginJob()
{
  DQMStore* dmbe = edm::Service<DQMStore>().operator->();
  dmbe->setCurrentFolder("RPC/LinkMonitor/");

  me_delaySummary = dmbe->book1D("delaySummary","LinkDelaySummary",8,-3.5, 4.5);
  me_delaySummary->getTH1F()->SetStats(111);

  me_delaySpread = dmbe->book2D("delaySpread","LinkDelaySpread",71,-3.05, 4.05, 31,-0.05,3.05);
  me_delaySpread->getTH2F()->SetStats(0);

  me_notComplete[0] = dmbe->book2D("notComplete790","FED790: not All Paths hit",36,-0.5,35.5,18,-0.5,17.5);
  me_notComplete[1] = dmbe->book2D("notComplete791","FED791: not All Paths hit",36,-0.5,35.5,18,-0.5,17.5);
  me_notComplete[2] = dmbe->book2D("notComplete792","FED792: not All Paths hit",36,-0.5,35.5,18,-0.5,17.5);
  for (unsigned int i=0;i<3;++i) {
    me_notComplete[i]->getTH2F()->GetXaxis()->SetNdivisions(512);
    me_notComplete[i]->getTH2F()->GetYaxis()->SetNdivisions(505);
    me_notComplete[i]->getTH2F()->SetXTitle("rmb");
    me_notComplete[i]->getTH2F()->SetYTitle("link");
    me_notComplete[i]->getTH2F()->SetStats(0);
  }
  me_topOccup  = dmbe->book2D("topOccup","Top10 LinkBoard occupancy",8,-0.5,7.5, 10,0.,10.);
  me_topSpread = dmbe->book2D("topSpread","Top10 LinkBoard delay spread",8,-0.5,7.5, 10,0.,10.);
  me_topOccup->getTH2F()->GetXaxis()->SetNdivisions(110);
  me_topSpread->getTH2F()->GetXaxis()->SetNdivisions(110);
  me_topOccup->getTH2F()->SetStats(0);
  me_topSpread->getTH2F()->SetStats(0);

}

TObjArray RPCMonitorLinkSynchro::histos() const
{
  TObjArray result;
  result.Add(me_delaySummary->getTH1F());
  result.Add(me_delaySpread->getTH2F());
  result.Add(me_topOccup->getTH2F());
  result.Add(me_topSpread->getTH2F());
  for (unsigned int i=0;i<=2;i++) result.Add(me_notComplete[i]->getTH2F());
  return result;
}


void RPCMonitorLinkSynchro::endJob()
{
  if (theConfig.getUntrackedParameter<bool>("dumpDelays")) edm::LogInfo("RPCMonitorLinkSynchro DumpDelays") <<  theSynchroStat.dumpDelays();
}

void RPCMonitorLinkSynchro::analyze(const edm::Event& ev, const edm::EventSetup& es)
{
  edm::Handle<RPCRawSynchro::ProdItem> synchroCounts;
  ev.getByLabel(rpcRawSynchroProdItemTag_, synchroCounts);
  std::vector<LinkBoardElectronicIndex> problems;
  const RPCRawSynchro::ProdItem &vItem = select(*synchroCounts.product(), ev,es);
  theSynchroStat.add(vItem, problems);

  for (std::vector<LinkBoardElectronicIndex>::const_iterator it=problems.begin(); it != problems.end(); ++it) {
    me_notComplete[it->dccId-790]->Fill(it->dccInputChannelNum,it->tbLinkInputNum);    
  }
}
