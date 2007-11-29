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

  void DoTracking(float local_event[2][6][10][32], int zbox);

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
  double zs_N_sigmas_;

  MonitorElement *hitStrips1B_;
  MonitorElement *hitStrips2B_;
  MonitorElement *hitSensors1B_;
  MonitorElement *hitSensors2B_;

  MonitorElement *meTrack_Npoints_;
  MonitorElement *me_Nhits_lad0_;
  MonitorElement *me_Nhits_lad1_;
  MonitorElement *me_Nhits_lad2_;
  MonitorElement *me_Nhits_lad3_;
  MonitorElement *me_Nhits_lad4_;
  MonitorElement *me_Nhits_lad5_;
  MonitorElement *me_hit_x_;
  MonitorElement *me_hit_y_;
  MonitorElement *meTrack_hit0_;
  MonitorElement *meTrack_hit1_;
  MonitorElement *meTrack_hit2_;
  MonitorElement *meTrack_hit3_;
  MonitorElement *meTrack_hit4_;
  MonitorElement *meTrack_hit5_;
  MonitorElement *meTrack_Px0_;
  MonitorElement *meTrack_Px1_;
  MonitorElement *meTrack_Px2_;
  MonitorElement *meTrack_Px3_;
  MonitorElement *meTrack_Px4_;
  MonitorElement *meTrack_Px5_;
  MonitorElement *meTrack_Py0_;
  MonitorElement *meTrack_Py1_;
  MonitorElement *meTrack_Py2_;
  MonitorElement *meTrack_Py3_;
  MonitorElement *meTrack_Py4_;
  MonitorElement *meTrack_Py5_;
  MonitorElement *meTrack_Pz0_;
  MonitorElement *meTrack_Pz1_;
  MonitorElement *meTrack_Pz2_;
  MonitorElement *meTrack_Pz3_;
  MonitorElement *meTrack_Pz4_;
  MonitorElement *meTrack_Pz5_;
  MonitorElement *meTrack_par0_;
  MonitorElement *meTrack_par1_;
  MonitorElement *meTrack_par2_;
  MonitorElement *meTrack_par3_;
  MonitorElement *meTrack_par4_;
  MonitorElement *meTrack_par5_;
};

#endif
