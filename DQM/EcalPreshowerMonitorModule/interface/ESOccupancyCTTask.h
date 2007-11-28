#ifndef ESOccupancyCTTask_H
#define ESOccupancyCTTask_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "DataFormats/EcalDigi/interface/ESDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

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

class ESOccupancyCTTask: public EDAnalyzer{

 public:

  ESOccupancyCTTask(const ParameterSet& ps);
  virtual ~ESOccupancyCTTask();

 protected:

  void analyze(const Event& e, const EventSetup& c);
  void beginJob(const EventSetup& c);
  void endJob(void);
  void setup(void);
  void cleanup(void);

 private:

  int ievt_;
  DaqMonitorBEInterface* dbe_;

  MonitorElement* meEnergy_[2][6];
  MonitorElement* meOccupancy1D_[2][6];
  MonitorElement* meOccupancy2D_[2][6];

  string digilabel_;
  string rechitlabel_;
  string instanceName_;
  int gain_;
  bool sta_;

  bool init_;

  MonitorElement *hitStrips1B_;
  MonitorElement *hitStrips2B_;
  MonitorElement *hitSensors1B_;
  MonitorElement *hitSensors2B_;

};

#endif
