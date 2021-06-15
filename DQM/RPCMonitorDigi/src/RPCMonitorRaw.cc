#include "DQM/RPCMonitorDigi/interface/RPCMonitorRaw.h"
#include "DQM/RPCMonitorDigi/interface/RPCRawDataCountsHistoMaker.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondFormats/RPCObjects/interface/LinkBoardElectronicIndex.h"
#include "DataFormats/RPCDigi/interface/ReadoutError.h"

RPCMonitorRaw::RPCMonitorRaw(const edm::ParameterSet& cfg) : theConfig(cfg) {
  rpcRawDataCountsTag_ = consumes<RPCRawDataCounts>(cfg.getParameter<edm::InputTag>("rpcRawDataCountsTag"));

  for (unsigned int i = 0; i < 10; i++)
    theWatchedErrorHistoPos[i] = 0;
  std::vector<int> algos = cfg.getUntrackedParameter<std::vector<int> >("watchedErrors");
  for (std::vector<int>::const_iterator it = algos.begin(); it != algos.end(); ++it) {
    unsigned int ialgo = *it;
    if (ialgo < 10)
      theWatchedErrorHistoPos[ialgo] = 1;  // real position initialisain is in begin job. here mark just switched on.
  }
}

RPCMonitorRaw::~RPCMonitorRaw() { LogTrace("") << "RPCMonitorRaw destructor"; }

void RPCMonitorRaw::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& iRun, edm::EventSetup const& iSetup) {
  ibooker.cd();
  ibooker.setCurrentFolder("RPC/LinkMonitor");

  me_t[0] = ibooker.book1D("recordType_790", RPCRawDataCountsHistoMaker::emptyRecordTypeHisto(790));
  me_t[1] = ibooker.book1D("recordType_791", RPCRawDataCountsHistoMaker::emptyRecordTypeHisto(791));
  me_t[2] = ibooker.book1D("recordType_792", RPCRawDataCountsHistoMaker::emptyRecordTypeHisto(792));
  for (int i = 0; i < 3; ++i)
    me_t[i]->getTH1F()->SetStats(false);

  me_e[0] = ibooker.book1D("readoutErrors_790", RPCRawDataCountsHistoMaker::emptyReadoutErrorHisto(790));
  me_e[1] = ibooker.book1D("readoutErrors_791", RPCRawDataCountsHistoMaker::emptyReadoutErrorHisto(791));
  me_e[2] = ibooker.book1D("readoutErrors_792", RPCRawDataCountsHistoMaker::emptyReadoutErrorHisto(792));
  for (int i = 0; i < 3; ++i)
    me_e[i]->getTH1F()->SetStats(false);

  me_mapGoodEvents = ibooker.book2D("mapGoodRecords", "mapGoodRecords", 36, -0.5, 35.5, 3, 789.5, 792.5);
  me_mapGoodEvents->getTH2F()->SetNdivisions(3, "y");
  me_mapGoodEvents->setAxisTitle("rmb");
  me_mapGoodEvents->getTH2F()->SetYTitle("fed");
  me_mapGoodEvents->getTH2F()->SetStats(false);
  me_mapBadEvents = ibooker.book2D("mapErrorRecords", "mapErrorRecords", 36, -0.5, 35.5, 3, 789.5, 792.5);
  me_mapBadEvents->setAxisTitle("fed");
  me_mapBadEvents->getTH2F()->SetYTitle("rmb");
  me_mapBadEvents->getTH2F()->SetNdivisions(3, "y");
  me_mapBadEvents->getTH2F()->SetStats(false);

  for (unsigned int i = 0; i <= 9; ++i) {
    if (theWatchedErrorHistoPos[i]) {
      for (unsigned int fed = 790; fed <= 792; ++fed) {
        TH2F* histo = RPCRawDataCountsHistoMaker::emptyReadoutErrorMapHisto(fed, i);
        MonitorElement* watched = ibooker.book2D(histo->GetName(), histo);
        theWatchedErrorHistos[fed - 790].push_back(watched);
        theWatchedErrorHistoPos[i] = theWatchedErrorHistos[fed - 790].size();
      }
    }
  }
}

void RPCMonitorRaw::analyze(const edm::Event& ev, const edm::EventSetup& es) {
  edm::Handle<RPCRawDataCounts> rawCounts;
  ev.getByToken(rpcRawDataCountsTag_, rawCounts);
  const RPCRawDataCounts& counts = *rawCounts.product();

  // record type
  for (auto cnt : counts.theRecordTypes)
    me_t[cnt.first.first - 790]->Fill(cnt.first.second, cnt.second);

  // good events topology
  for (auto cnt : counts.theGoodEvents)
    me_mapGoodEvents->Fill(cnt.first.second, cnt.first.first, cnt.second);

  // bad events topology
  for (auto cnt : counts.theBadEvents)
    me_mapBadEvents->Fill(cnt.first.second, cnt.first.first, cnt.second);

  // readout errors
  for (auto cnt : counts.theReadoutErrors) {
    rpcrawtodigi::ReadoutError error(cnt.first.second);
    LinkBoardElectronicIndex ele = error.where();
    rpcrawtodigi::ReadoutError::ReadoutErrorType type = error.type();

    int fed = cnt.first.first;
    me_e[fed - 790]->Fill(type, cnt.second);

    // in addition fill location map for selected errors
    int idx = theWatchedErrorHistoPos[type] - 1;
    if (idx >= 0) {
      std::vector<MonitorElement*>& wh = theWatchedErrorHistos[fed - 790];
      MonitorElement* me = wh[idx];
      me->Fill(ele.dccInputChannelNum, ele.tbLinkInputNum, cnt.second);
    }
  }
}
