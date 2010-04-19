#ifndef DQM_RPCMonitorClient_RPCMonitorLinkSynchroMerger_H
#define DQM_RPCMonitorClient_RPCMonitorLinkSynchroMerger_H

#include "DQM/RPCMonitorClient/interface/RPCMonitorLinkSynchro.h"

class RPCMonitorLinkSynchroMerger : public RPCMonitorLinkSynchro {
public:
  RPCMonitorLinkSynchroMerger(const edm::ParameterSet& cfg);
  virtual ~RPCMonitorLinkSynchroMerger(){}
  virtual void beginRun(const edm::Run&, const edm::EventSetup& es);
  virtual void analyze(const edm::Event&, const edm::EventSetup&){}
private:
  void preFillFromFile(const std::string & file);
  bool isInitialised;
};
#endif
