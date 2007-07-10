#ifndef ESTDCCTTask_H
#define ESTDCCTTask_H

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

#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "DataFormats/EcalDigi/interface/ESDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <fstream>
#include <vector>

#include "TH1F.h"
#include "TFile.h"

using namespace std;
using namespace edm;

class ESTDCCTTask: public EDAnalyzer{

 public:

  ESTDCCTTask(const ParameterSet& ps);
  virtual ~ESTDCCTTask();

 protected:

  void analyze(const Event& e, const EventSetup& c);
  void beginJob(const EventSetup& c);
  void endJob(void);
  void setup(void);
  void cleanup(void);

 private:

  int ievt_;

  DaqMonitorBEInterface* dbe_;

  MonitorElement* meTDC_;
  MonitorElement* meTDCADC_[2][6][3];
  MonitorElement* meTDCADCT_[2][6];

  string label_;
  string instanceName_;
  string pedestalFile_;

  bool sta_;
  bool init_;

  TFile *ped_;
  TH1F* hist_[2][6][2][5];

};

#endif
