#ifndef ESPEDESTALTBCLIENT_H
#define ESPEDESTALTBCLIENT_H

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

#include "TF1.h"

#include <memory>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>

using namespace edm;
using namespace std;

class ESPedestalTBClient: public EDAnalyzer{

 public:
  
  ESPedestalTBClient(const ParameterSet& ps);
  virtual ~ESPedestalTBClient();
  
 protected:
  
  void beginJob(const EventSetup& c);
  void analyze(const Event& e, const EventSetup& c);
  void endJob();
  void setup();
  void cleanup();
  void doQT();

  string getMEName(const int & plane, const int & row, const int & col, const int & strip);
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
  int count_;
  int run_;
  bool sta_;
  bool init_;
  TF1 *fg;

  DaqMonitorBEInterface* dbe_;

  MonitorElement* meMean_;
  MonitorElement* meRMS_;
  MonitorElement* meFitMean_;
  MonitorElement* meFitRMS_;
  MonitorElement* mePedCol_[2];
  MonitorElement* mePedMeanRMS_[2][4][4];
  MonitorElement* mePedRMS_[2][4][4];
  MonitorElement* mePedFitMeanRMS_[2][4][4];
  MonitorElement* mePedFitRMS_[2][4][4];

};

#endif

