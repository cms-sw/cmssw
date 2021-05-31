#ifndef DQM_RPCMonitorDigi_RPCMonitorLinkSynchro_H
#define DQM_RPCMonitorDigi_RPCMonitorLinkSynchro_H

/** \class RPCMonitorLinkSynchro
 ** Monitor and anlyse synchro counts () produced by R2D. 
 **/
#include "DQMServices/Core/interface/DQMOneEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "CondFormats/RPCObjects/interface/RPCEMap.h"
#include "CondFormats/DataRecord/interface/RPCEMapRcd.h"
#include "DQM/RPCMonitorDigi/interface/RPCLinkSynchroStat.h"

class RPCMonitorLinkSynchro : public DQMOneEDAnalyzer<edm::one::WatchLuminosityBlocks> {
public:
  explicit RPCMonitorLinkSynchro(const edm::ParameterSet& cfg);
  ~RPCMonitorLinkSynchro() override = default;

  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void dqmBeginRun(const edm::Run& r, const edm::EventSetup& c) override;
  void beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&) final {}
  void endLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  virtual const RPCRawSynchro::ProdItem& select(const RPCRawSynchro::ProdItem& v,
                                                const edm::Event&,
                                                const edm::EventSetup&) {
    return v;
  };

protected:
  edm::ParameterSet theConfig;
  edm::ESWatcher<RPCEMapRcd> theCablingWatcher;
  edm::ESGetToken<RPCEMap, RPCEMapRcd> rpcEMapToken_;
  RPCLinkSynchroStat theSynchroStat;

  MonitorElement* me_delaySummary;
  MonitorElement* me_delaySpread;
  MonitorElement* me_topOccup;
  MonitorElement* me_topSpread;
  MonitorElement* me_notComplete[3];

private:
  edm::EDGetTokenT<RPCRawSynchro::ProdItem> rpcRawSynchroProdItemTag_;
};

#endif
