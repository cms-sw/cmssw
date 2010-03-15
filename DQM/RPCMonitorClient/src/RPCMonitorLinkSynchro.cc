#include "DQM/RPCMonitorClient/interface/RPCMonitorLinkSynchro.h"
#include "FWCore/Utilities/interface/InputTag.h"
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

#include "CondFormats/DataRecord/interface/RPCEMapRcd.h"

#include "DQM/RPCMonitorClient/interface/RPCLinkSynchroHistoMaker.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TFile.h"

//#include "UserCode/konec/test/R2DTimerObserver.h"

using namespace edm;

//R2DTimerObserver theTimer("**** MY TIMING REPORT ***");
//TH1D hCPU("hCPU","hCPU",100,0.,1.50);

RPCMonitorLinkSynchro::RPCMonitorLinkSynchro( const edm::ParameterSet& cfg) 
    : theConfig(cfg),
      theSynchroStat(RPCLinkSynchroStat(theConfig.getUntrackedParameter<bool>("useFirstHitOnly", false))) 
{ 
}

RPCMonitorLinkSynchro::~RPCMonitorLinkSynchro()
{ 
// LogTrace("") << "RPCMonitorLinkSynchro destructor"; 
}

void RPCMonitorLinkSynchro::beginRun(const edm::Run&, const edm::EventSetup& es)
{
// LogTrace("") << "RPCMonitorLinkSynchro::beginRun !!!!" << std::endl;
 if (theCablingWatcher.check(es)) {
    ESHandle<RPCEMap> readoutMapping;
    es.get<RPCEMapRcd>().get(readoutMapping);
    RPCReadOutMapping * cabling = readoutMapping->convert();
    LogTrace("") << "RPCMonitorLinkSynchro - record has CHANGED!!, read map, VERSION: " << cabling->version();
    theSynchroStat.init(cabling, theConfig.getUntrackedParameter<bool>("dumpDelays"));
    delete cabling;
  }
}

void RPCMonitorLinkSynchro::endLuminosityBlock(const LuminosityBlock& ls, const EventSetup& es)
{

//  LogTrace("") << "RPCMonitorLinkSynchro END OF LUMI BLOCK CALLED!";

  RPCLinkSynchroHistoMaker hm(theSynchroStat);
  hm.fill(me_delaySummary->getTH1F(), me_delaySpread->getTH2F(), me_topOccup->getTH2F(), me_topSpread->getTH2F());

//  hm.fillDelaySpreadHisto(me_delaySpread->getTH2F());
//  hm.fillDelayHisto(me_delaySummary->getTH1F());

//  me_delaySummary->update();
//  me_delaySpread->update();
//  for (unsigned int i=0;i<3;++i)me_notComplete[i]->update();
}

void RPCMonitorLinkSynchro::beginJob()
{
  DQMStore* dmbe = edm::Service<DQMStore>().operator->();
  dmbe->setCurrentFolder("RPC/FEDIntegrity/");

  me_delaySummary = dmbe->book1D("delaySummary","LinkDelaySummary",8,-3.5, 4.5);
  me_delaySummary->getTH1F()->SetStats(111);

  me_delaySpread = dmbe->book2D("delaySpread","LinkDelaySpread",71,-3.05, 4.05, 31,-0.05,3.05);
  me_delaySpread->getTH2F()->SetStats(0);

  me_notComplete[0] = dmbe->book2D("notComplete790","FED790: not All Paths hit",36,-0.5,35.5,18,-0.5,17.5);
  me_notComplete[1] = dmbe->book2D("notComplete791","FED791: not All Paths hit",36,-0.5,35.5,18,-0.5,17.5);
  me_notComplete[2] = dmbe->book2D("notComplete792","FED792: not All Paths hit",36,-0.5,35.5,18,-0.5,17.5);
  for (unsigned int i=0;i<3;++i) {
    me_notComplete[0]->getTH2F()->GetXaxis()->SetNdivisions(512);
    me_notComplete[0]->getTH2F()->GetYaxis()->SetNdivisions(505);
    me_notComplete[0]->getTH2F()->SetStats(0);
  }
  me_topOccup  = dmbe->book2D("topOccup","Top10 LinkBoard occupancy",8,-0.5,7.5, 10,0.,10.);
  me_topSpread = dmbe->book2D("topSpread","Top10 LinkBoard delay spread",8,-0.5,7.5, 10,0.,10.);
  me_topOccup->getTH2F()->GetXaxis()->SetNdivisions(110);
  me_topSpread->getTH2F()->GetXaxis()->SetNdivisions(110);
  me_topOccup->getTH2F()->SetStats(0);
  me_topSpread->getTH2F()->SetStats(0);

}


void RPCMonitorLinkSynchro::endJob()
{
//  RPCLinkSynchroHistoMaker hm(theSynchroStat);
//  hm.fill();
  bool writeHistos = theConfig.getUntrackedParameter<bool>("writeHistograms", false);
  if (writeHistos) {
    std::string histoFile = theConfig.getUntrackedParameter<std::string>("histoFileName");
    TFile f(histoFile.c_str(),"RECREATE");
    me_delaySummary->getTH1F()->Write();
    me_delaySpread->getTH2F()->Write();
    me_topOccup->getTH2F()->Write();
    me_topSpread->getTH2F()->Write();
    for (unsigned int i=0;i<=2;i++) me_notComplete[i]->getTH2F()->Write();
//    hCPU.Write();
    edm::LogInfo(" END JOB, histos saved!");
    f.Close();
  }

  if (theConfig.getUntrackedParameter<bool>("dumpDelays")) LogInfo("RPCMonitorLinkSynchro DumpDelays") <<  theSynchroStat.dumpDelays();

}

void RPCMonitorLinkSynchro::analyze(const edm::Event& ev, const edm::EventSetup& es)
{
  edm::Handle<RPCRawSynchro::ProdItem> synchroCounts;
  ev.getByType(synchroCounts);
//  theTimer.start();
  std::vector<LinkBoardElectronicIndex> problems;
  theSynchroStat.add(*synchroCounts.product(), problems);
//  theTimer.stop();

  for (std::vector<LinkBoardElectronicIndex>::const_iterator it=problems.begin(); it != problems.end(); ++it) {
    me_notComplete[it->dccId-790]->Fill(it->dccInputChannelNum,it->tbLinkInputNum);    
  }

//  std::cout << "TIMING IS: (real)" << theTimer.lastMeasurement().real() << std::endl;
//  hCPU.Fill(theTimer.lastMeasurement().real());
}
