#ifndef ESPEDESTALCMCTCLIENT_H
#define ESPEDESTALCMCTCLIENT_H

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

class ESPedestalCMCTClient: public EDAnalyzer{

 public:
  
  ESPedestalCMCTClient(const ParameterSet& ps);
  virtual ~ESPedestalCMCTClient();
  
 protected:
  
  void beginJob(const EventSetup& c);
  void analyze(const Event& e, const EventSetup& c);
  void endJob();
  void setup();
  void cleanup();
  void doQT();

  string getMEName(const int & zside, const int & plane, const int & row, const int & col, const int & strip, const int & type);
  void htmlOutput(int run, string htmlDir, string htmlName);

 private:
  
  bool writeHisto_;
  bool writeHTML_;
  int dumpRate_;
  string outputFile_;
  string outputFileName_;
  string rootFolder_;
  string htmlDir_;
  string htmlName_;
  double cmnThreshold_;
  bool sta_;
  int count_;
  int run_;
  bool init_;

  DaqMonitorBEInterface* dbe_;

  MonitorElement* meMean_;
  MonitorElement* meRMS_;
  MonitorElement* meCMCol_[2][6];

};

#endif

