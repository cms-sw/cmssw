#ifndef DQM_RPCMonitorClient_RPCMonitorLinkSynchro_H
#define DQM_RPCMonitorClient_RPCMonitorLinkSynchro_H

/** \class RPCMonitorLinkSynchro
 ** Monitor and anlyse synchro counts () produced by R2D. 
 **/
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "CondFormats/DataRecord/interface/RPCEMapRcd.h"

#include "DQM/RPCMonitorClient/interface/RPCLinkSynchroStat.h"
#include "DQMServices/Core/interface/DQMOneEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

namespace edm {
  class Event;
  class EventSetup;
  class Run;
}  // namespace edm

class RPCMonitorLinkSynchro : public DQMOneEDAnalyzer<edm::one::WatchLuminosityBlocks> {
public:
  explicit RPCMonitorLinkSynchro(const edm::ParameterSet& cfg);
  ~RPCMonitorLinkSynchro() override;

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
