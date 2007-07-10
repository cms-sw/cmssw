#ifndef ESTDCCTCLIENT_H
#define ESTDCCTCLIENT_H

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

#include "TH1F.h"
#include "TH2F.h"

using namespace edm;
using namespace std;

class ESTDCCTClient: public EDAnalyzer{

 public:
  
  ESTDCCTClient(const ParameterSet& ps);
  virtual ~ESTDCCTClient();
  
 protected:
  
  void beginJob(const EventSetup& c);
  void analyze(const Event& e, const EventSetup& c);
  void endJob();
  void setup();
  void cleanup();
  void doQT();

  string getMEName(const string & meName);
  string getMEName(const int & zside, const int & plane, const int & slot, const int & type);

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

  DaqMonitorBEInterface* dbe_;

  TH1F *hTDC_;
  TH2F *hTDCADC_[2][6][3];
  TH2F *hTDCADCT_[2][6];

};

#endif

