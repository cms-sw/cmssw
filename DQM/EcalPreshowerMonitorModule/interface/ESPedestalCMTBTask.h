#ifndef ESPedestalCMTBTask_H
#define ESPedestalCMTBTask_H

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

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <TROOT.h>
#include <TFile.h>
#include <TTree.h>
#include <TF1.h>
#include <TH1.h>
#include <TH2.h>

#include <iostream>
#include <fstream>
#include <vector>

using namespace std;
using namespace edm;

class ESPedestalCMTBTask: public EDAnalyzer{

 public:

  ESPedestalCMTBTask(const ParameterSet& ps);
  virtual ~ESPedestalCMTBTask();

 protected:

  void analyze(const Event& e, const EventSetup& c);
  void beginJob(const EventSetup& c);
  void endJob(void);
  void setup(void);
  void cleanup(void);
  void DoCommonModeItr(float data[], float *cm);
  void DoCommonMode32(float det_data[], float *cm1);
  void DoCommonMode(float det_data[], float *cm1, float *cm2);

 private:

  int ievt_;

  DaqMonitorBEInterface* dbe_;

  MonitorElement* mePedestalCM_S0_[2][4][4][32];
  MonitorElement* mePedestalCM_S1_[2][4][4][32];
  MonitorElement* mePedestalCM_S2_[2][4][4][32];

  MonitorElement* meSensorCM_S0_[2][4][4];
  MonitorElement* meSensorCM_S1_[2][4][4];
  MonitorElement* meSensorCM_S2_[2][4][4];

  MonitorElement* meADC_[2][3];
  MonitorElement* meADCZS_[2][3];
  MonitorElement* meOccupancy2D_[2][3];

  TH1F* hist_[2][4][4];

  string label_;
  string instanceName_;
  string pedestalFile_;

  TFile* ped_;

//  TFile* test_;
//  TTree* tree_;

  bool doCM_;
  bool sta_;
  bool init_;

  int gain_;
  int cmMethod_;

  double zs_;
  double sigma_;

};

#endif
