#include <memory>

#include "DQM/RPCMonitorClient/interface/RPCMonitorLinkSynchro.h"
#include "DQM/RPCMonitorClient/interface/RPCLinkSynchroHistoMaker.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"

#include "CondFormats/RPCObjects/interface/RPCEMap.h"
#include "CondFormats/RPCObjects/interface/RPCReadOutMapping.h"

RPCMonitorLinkSynchro::RPCMonitorLinkSynchro(const edm::ParameterSet& cfg)
    : theConfig(cfg),
      theSynchroStat(RPCLinkSynchroStat(theConfig.getUntrackedParameter<bool>("useFirstHitOnly", false)))

{
  rpcRawSynchroProdItemTag_ =
      consumes<RPCRawSynchro::ProdItem>(cfg.getParameter<edm::InputTag>("rpcRawSynchroProdItemTag"));
}

RPCMonitorLinkSynchro::~RPCMonitorLinkSynchro() {}

void RPCMonitorLinkSynchro::endLuminosityBlock(const edm::LuminosityBlock& ls, const edm::EventSetup& es) {
  RPCLinkSynchroHistoMaker hm(theSynchroStat);
  hm.fill(me_delaySummary->getTH1F(), me_delaySpread->getTH2F(), me_topOccup->getTH2F(), me_topSpread->getTH2F());
}

void RPCMonitorLinkSynchro::dqmBeginRun(const edm::Run& r, const edm::EventSetup& es) {
  if (theCablingWatcher.check(es)) {
    edm::ESTransientHandle<RPCEMap> readoutMapping;
    es.get<RPCEMapRcd>().get(readoutMapping);
    std::unique_ptr<RPCReadOutMapping const> cabling{readoutMapping->convert()};
    edm::LogInfo("RPCMonitorLinkSynchro")
        << "RPCMonitorLinkSynchro - record has CHANGED!!, read map, VERSION: " << cabling->version();
    theSynchroStat.init(cabling.get(), theConfig.getUntrackedParameter<bool>("dumpDelays"));
  }
}

void RPCMonitorLinkSynchro::bookHistograms(DQMStore::IBooker& ibooker,
                                           edm::Run const& iRun,
                                           edm::EventSetup const& es) {
  ibooker.cd();
  ibooker.setCurrentFolder("RPC/LinkMonitor/");

  me_delaySummary = ibooker.book1D("delaySummary", "LinkDelaySummary", 8, -3.5, 4.5);
  me_delaySummary->getTH1F()->SetStats(true);

  me_delaySpread = ibooker.book2D("delaySpread", "LinkDelaySpread", 71, -3.05, 4.05, 31, -0.05, 3.05);
  me_delaySpread->getTH2F()->SetStats(false);

  me_notComplete[0] = ibooker.book2D("notComplete790", "FED790: not All Paths hit", 36, -0.5, 35.5, 18, -0.5, 17.5);
  me_notComplete[1] = ibooker.book2D("notComplete791", "FED791: not All Paths hit", 36, -0.5, 35.5, 18, -0.5, 17.5);
  me_notComplete[2] = ibooker.book2D("notComplete792", "FED792: not All Paths hit", 36, -0.5, 35.5, 18, -0.5, 17.5);
  for (unsigned int i = 0; i < 3; ++i) {
    me_notComplete[i]->getTH2F()->GetXaxis()->SetNdivisions(512);
    me_notComplete[i]->getTH2F()->GetYaxis()->SetNdivisions(505);
    me_notComplete[i]->setAxisTitle("rmb");
    me_notComplete[i]->getTH2F()->SetYTitle("link");
    me_notComplete[i]->getTH2F()->SetStats(false);
  }
  me_topOccup = ibooker.book2D("topOccup", "Top10 LinkBoard occupancy", 8, -0.5, 7.5, 10, 0., 10.);
  me_topSpread = ibooker.book2D("topSpread", "Top10 LinkBoard delay spread", 8, -0.5, 7.5, 10, 0., 10.);
  me_topOccup->getTH2F()->GetXaxis()->SetNdivisions(110);
  me_topSpread->getTH2F()->GetXaxis()->SetNdivisions(110);
  me_topOccup->getTH2F()->SetStats(false);
  me_topSpread->getTH2F()->SetStats(false);
}

void RPCMonitorLinkSynchro::analyze(const edm::Event& ev, const edm::EventSetup& es) {
  edm::Handle<RPCRawSynchro::ProdItem> synchroCounts;
  ev.getByToken(rpcRawSynchroProdItemTag_, synchroCounts);
  std::vector<LinkBoardElectronicIndex> problems;
  const RPCRawSynchro::ProdItem& vItem = select(*synchroCounts.product(), ev, es);
  theSynchroStat.add(vItem, problems);

  for (std::vector<LinkBoardElectronicIndex>::const_iterator it = problems.begin(); it != problems.end(); ++it) {
    me_notComplete[it->dccId - 790]->Fill(it->dccInputChannelNum, it->tbLinkInputNum);
  }
}
