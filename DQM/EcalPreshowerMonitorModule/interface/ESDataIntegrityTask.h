#ifndef ESDataIntegrityTask_H
#define ESDataIntegrityTask_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "TBDataFormats/ESTBRawData/interface/ESDCCHeaderBlock.h"
#include "TBDataFormats/ESTBRawData/interface/ESKCHIPBlock.h"
#include "TBDataFormats/ESTBRawData/interface/ESLocalRawDataCollections.h"
#include "TBDataFormats/ESTBRawData/interface/ESRawDataCollections.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <fstream>
#include <vector>

using namespace std;
using namespace edm;

class ESDataIntegrityTask: public EDAnalyzer{

 public:

  ESDataIntegrityTask(const ParameterSet& ps);
  virtual ~ESDataIntegrityTask();

 protected:

  void analyze(const Event& e, const EventSetup& c);
  void beginJob(const EventSetup& c);
  void endJob(void);
  void setup(void);
  void cleanup(void);

 private:

  int ievt_;
  int detType_;
  
  DaqMonitorBEInterface* dbe_;

  MonitorElement* meCRCError_;
  MonitorElement* meDCCError_;
  MonitorElement* meGlbBC_;
  MonitorElement* meGlbEC_;
  MonitorElement* meKchipBC_;
  MonitorElement* meKchipEC_;
  MonitorElement* meFlag1_;
  MonitorElement* meFlag2_;
  MonitorElement* meEvtLen_;

  MonitorElement* fedIds_;
  MonitorElement* DCCfedId1_;
  MonitorElement* DCCfedId4_;
  MonitorElement* DCCfedId10_;
  MonitorElement* DCCfedId40_;
  MonitorElement* Kchip_;

  string label_;
  string instanceName_;

  bool sta_;
  bool init_;

};

#endif
