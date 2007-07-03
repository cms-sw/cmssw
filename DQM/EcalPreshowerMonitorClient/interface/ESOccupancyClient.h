#ifndef ESOCCUPANCYCLIENT_H
#define ESOCCUPANCYCLIENT_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <DataFormats/Common/interface/Handle.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Core/interface/MonitorElementBaseT.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <memory>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>

using namespace edm;
using namespace std;

class ESOccupancyClient: public EDAnalyzer{

 public:
  
  ESOccupancyClient(const ParameterSet& ps);
  virtual ~ESOccupancyClient();
  
 protected:
  
  void beginJob(const EventSetup& c);
  void analyze(const Event& e, const EventSetup& c);
  void endJob();

  string getMEName(const int & plane);

 private:
  
  bool writeHisto_;
  string outputFile_;
  string rootFolder_;

  DaqMonitorBEInterface* dbe_;

};

#endif

