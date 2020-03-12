#ifndef DQMSERVICES_CORE_DQM_SERVICE_H
#define DQMSERVICES_CORE_DQM_SERVICE_H

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

class DQMBasicNet;

/** A bridge to udpate the DQM network layer at the end of every event.  */
class DQMService {
public:
  typedef dqm::legacy::DQMStore DQMStore;
  typedef dqm::legacy::MonitorElement MonitorElement;

  DQMService(const edm::ParameterSet &pset, edm::ActivityRegistry &ar);
  ~DQMService();

public:
  void flush(edm::StreamContext const &sc);

private:
  void shutdown();

  DQMStore *store_;
  DQMBasicNet *net_;
  double lastFlush_;
  double publishFrequency_;

public:
  void flushStandalone();
};

#endif  // DQMSERVICES_CORE_DQM_SERVICE_H
