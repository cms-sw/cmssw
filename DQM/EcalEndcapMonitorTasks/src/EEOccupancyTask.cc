/*
 * \file EEOccupancyTask.cc
 *
 * $Date: 2012/04/27 13:46:15 $
 * $Revision: 1.94 $
 * \author G. Della Ricca
 * \author G. Franzoni
 *
*/

#include <iostream>
#include <fstream>
#include <vector>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DQMServices/Core/interface/MonitorElement.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgoRcd.h"

#include "DQM/EcalCommon/interface/Numbers.h"

#include "DQM/EcalEndcapMonitorTasks/interface/EEOccupancyTask.h"

EEOccupancyTask::EEOccupancyTask(const edm::ParameterSet& ps){

  init_ = false;

  initCaloGeometry_ = false;

  dqmStore_ = edm::Service<DQMStore>().operator->();

  prefixME_ = ps.getUntrackedParameter<std::string>("prefixME", "");

  subfolder_ = ps.getUntrackedParameter<std::string>("subfolder", "");

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  mergeRuns_ = ps.getUntrackedParameter<bool>("mergeRuns", false);

  EcalRawDataCollection_ = ps.getParameter<edm::InputTag>("EcalRawDataCollection");
  EEDigiCollection_ = ps.getParameter<edm::InputTag>("EEDigiCollection");
  EcalPnDiodeDigiCollection_ = ps.getParameter<edm::InputTag>("EcalPnDiodeDigiCollection");
  EcalRecHitCollection_ = ps.getParameter<edm::InputTag>("EcalRecHitCollection");
  EcalTrigPrimDigiCollection_ = ps.getParameter<edm::InputTag>("EcalTrigPrimDigiCollection");

  for (int i = 0; i < 18; i++) {
    meOccupancy_[i]    = 0;
    meOccupancyMem_[i] = 0;
    meEERecHitEnergy_[i] = 0;
    meSpectrum_[i] = 0;
  }

  meEERecHitSpectrum_[0] = 0;
  meEERecHitSpectrum_[1] = 0;

  meEEDigiOccupancy_[0] = 0;
  meEEDigiOccupancyProEta_[0] = 0;
  meEEDigiOccupancyProPhi_[0] = 0;
  meEEDigiOccupancy_[1] = 0;
  meEEDigiOccupancyProEta_[1] = 0;
  meEEDigiOccupancyProPhi_[1] = 0;

  meEERecHitOccupancy_[0] = 0;
  meEERecHitOccupancyProEta_[0] = 0;
  meEERecHitOccupancyProPhi_[0] = 0;
  meEERecHitOccupancy_[1] = 0;
  meEERecHitOccupancyProEta_[1] = 0;
  meEERecHitOccupancyProPhi_[1] = 0;

  meEERecHitOccupancyThr_[0] = 0;
  meEERecHitOccupancyProEtaThr_[0] = 0;
  meEERecHitOccupancyProPhiThr_[0] = 0;
  meEERecHitOccupancyThr_[1] = 0;
  meEERecHitOccupancyProEtaThr_[1] = 0;
  meEERecHitOccupancyProPhiThr_[1] = 0;

  meEETrigPrimDigiOccupancy_[0] = 0;
  meEETrigPrimDigiOccupancyProEta_[0] = 0;
  meEETrigPrimDigiOccupancyProPhi_[0] = 0;
  meEETrigPrimDigiOccupancy_[1] = 0;
  meEETrigPrimDigiOccupancyProEta_[1] = 0;
  meEETrigPrimDigiOccupancyProPhi_[1] = 0;

  meEETrigPrimDigiOccupancyThr_[0] = 0;
  meEETrigPrimDigiOccupancyProEtaThr_[0] = 0;
  meEETrigPrimDigiOccupancyProPhiThr_[0] = 0;
  meEETrigPrimDigiOccupancyThr_[1] = 0;
  meEETrigPrimDigiOccupancyProEtaThr_[1] = 0;
  meEETrigPrimDigiOccupancyProPhiThr_[1] = 0;

  meEETestPulseDigiOccupancy_[0] = 0;
  meEETestPulseDigiOccupancy_[1] = 0;

  meEELaserDigiOccupancy_[0] = 0;
  meEELaserDigiOccupancy_[1] = 0;

  meEELedDigiOccupancy_[0] = 0;
  meEELedDigiOccupancy_[1] = 0;

  meEEPedestalDigiOccupancy_[0] = 0;
  meEEPedestalDigiOccupancy_[1] = 0;

  recHitEnergyMin_ = 0.500; // GeV
  trigPrimEtMin_ = 4.; // 4 ADCs == 1 GeV

  for (int i = 0; i < EEDetId::kSizeForDenseIndexing; i++) {
    geometryEE[i][0] = 0;
    geometryEE[i][1] = 0;
  }

}

EEOccupancyTask::~EEOccupancyTask(){

}

void EEOccupancyTask::beginJob(void){

  ievt_ = 0;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EEOccupancyTask");
    if(subfolder_.size())
      dqmStore_->setCurrentFolder(prefixME_ + "/EEOccupancyTask/" + subfolder_);
    dqmStore_->rmdir(prefixME_ + "/EEOccupancyTask");
  }

}

void EEOccupancyTask::beginRun(const edm::Run& r, const edm::EventSetup& c) {

  Numbers::initGeometry(c, false);

  if( !initCaloGeometry_ ) {
    c.get<CaloGeometryRecord>().get(pGeometry_);
    initCaloGeometry_ = true;
  }

  if ( ! mergeRuns_ ) this->reset();

}

void EEOccupancyTask::endRun(const edm::Run& r, const edm::EventSetup& c) {

}

void EEOccupancyTask::reset(void) {

  for (int i = 0; i < 18; i++) {
    if ( meOccupancy_[i] ) meOccupancy_[i]->Reset();
    if ( meOccupancyMem_[i] ) meOccupancyMem_[i]->Reset();
    if ( meEERecHitEnergy_[i] ) meEERecHitEnergy_[i]->Reset();
    if ( meSpectrum_[i] ) meSpectrum_[i]->Reset();
  }

  if ( meEERecHitSpectrum_[0] ) meEERecHitSpectrum_[0]->Reset();
  if ( meEERecHitSpectrum_[1] ) meEERecHitSpectrum_[1]->Reset();

  if ( meEEDigiOccupancy_[0] ) meEEDigiOccupancy_[0]->Reset();
  if ( meEEDigiOccupancyProEta_[0] ) meEEDigiOccupancyProEta_[0]->Reset();
  if ( meEEDigiOccupancyProPhi_[0] ) meEEDigiOccupancyProPhi_[0]->Reset();
  if ( meEEDigiOccupancy_[1] ) meEEDigiOccupancy_[1]->Reset();
  if ( meEEDigiOccupancyProEta_[1] ) meEEDigiOccupancyProEta_[1]->Reset();
  if ( meEEDigiOccupancyProPhi_[1] ) meEEDigiOccupancyProPhi_[1]->Reset();

  if ( meEERecHitOccupancy_[0] ) meEERecHitOccupancy_[0]->Reset();
  if ( meEERecHitOccupancyProEta_[0] ) meEERecHitOccupancyProEta_[0]->Reset();
  if ( meEERecHitOccupancyProPhi_[0] ) meEERecHitOccupancyProPhi_[0]->Reset();
  if ( meEERecHitOccupancy_[1] ) meEERecHitOccupancy_[1]->Reset();
  if ( meEERecHitOccupancyProEta_[1] ) meEERecHitOccupancyProEta_[1]->Reset();
  if ( meEERecHitOccupancyProPhi_[1] ) meEERecHitOccupancyProPhi_[1]->Reset();

  if ( meEERecHitOccupancyThr_[0] ) meEERecHitOccupancyThr_[0]->Reset();
  if ( meEERecHitOccupancyProEtaThr_[0] ) meEERecHitOccupancyProEtaThr_[0]->Reset();
  if ( meEERecHitOccupancyProPhiThr_[0] ) meEERecHitOccupancyProPhiThr_[0]->Reset();
  if ( meEERecHitOccupancyThr_[1] ) meEERecHitOccupancyThr_[1]->Reset();
  if ( meEERecHitOccupancyProEtaThr_[1] ) meEERecHitOccupancyProEtaThr_[1]->Reset();
  if ( meEERecHitOccupancyProPhiThr_[1] ) meEERecHitOccupancyProPhiThr_[1]->Reset();

  if ( meEETrigPrimDigiOccupancy_[0] ) meEETrigPrimDigiOccupancy_[0]->Reset();
  if ( meEETrigPrimDigiOccupancyProEta_[0] ) meEETrigPrimDigiOccupancyProEta_[0]->Reset();
  if ( meEETrigPrimDigiOccupancyProPhi_[0] ) meEETrigPrimDigiOccupancyProPhi_[0]->Reset();
  if ( meEETrigPrimDigiOccupancy_[1] ) meEETrigPrimDigiOccupancy_[1]->Reset();
  if ( meEETrigPrimDigiOccupancyProEta_[1] ) meEETrigPrimDigiOccupancyProEta_[1]->Reset();
  if ( meEETrigPrimDigiOccupancyProPhi_[1] ) meEETrigPrimDigiOccupancyProPhi_[1]->Reset();

  if ( meEETrigPrimDigiOccupancyThr_[0] ) meEETrigPrimDigiOccupancyThr_[0]->Reset();
  if ( meEETrigPrimDigiOccupancyProEtaThr_[0] ) meEETrigPrimDigiOccupancyProEtaThr_[0]->Reset();
  if ( meEETrigPrimDigiOccupancyProPhiThr_[0] ) meEETrigPrimDigiOccupancyProPhiThr_[0]->Reset();
  if ( meEETrigPrimDigiOccupancyThr_[1] ) meEETrigPrimDigiOccupancyThr_[1]->Reset();
  if ( meEETrigPrimDigiOccupancyProEtaThr_[1] ) meEETrigPrimDigiOccupancyProEtaThr_[1]->Reset();
  if ( meEETrigPrimDigiOccupancyProPhiThr_[1] ) meEETrigPrimDigiOccupancyProPhiThr_[1]->Reset();

  if ( meEETestPulseDigiOccupancy_[0] ) meEETestPulseDigiOccupancy_[0]->Reset();
  if ( meEETestPulseDigiOccupancy_[1] ) meEETestPulseDigiOccupancy_[1]->Reset();

  if ( meEELaserDigiOccupancy_[0] ) meEELaserDigiOccupancy_[0]->Reset();
  if ( meEELaserDigiOccupancy_[1] ) meEELaserDigiOccupancy_[1]->Reset();

  if ( meEELedDigiOccupancy_[0] ) meEELedDigiOccupancy_[0]->Reset();
  if ( meEELedDigiOccupancy_[1] ) meEELedDigiOccupancy_[1]->Reset();

  if ( meEEPedestalDigiOccupancy_[0] ) meEEPedestalDigiOccupancy_[0]->Reset();
  if ( meEEPedestalDigiOccupancy_[1] ) meEEPedestalDigiOccupancy_[1]->Reset();

}

void EEOccupancyTask::setup(void){

  init_ = true;

  std::string name;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EEOccupancyTask");
    if(subfolder_.size())
      dqmStore_->setCurrentFolder(prefixME_ + "/EEOccupancyTask/" + subfolder_);

    for (int i = 0; i < 18; i++) {
      name = "EEOT digi occupancy " + Numbers::sEE(i+1);
      meOccupancy_[i] = dqmStore_->book2D(name, name, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50.);
      meOccupancy_[i]->setAxisTitle("ix", 1);
      if ( i+1 >= 1 && i+1 <= 9 ) meOccupancy_[i]->setAxisTitle("101-ix", 1);
      meOccupancy_[i]->setAxisTitle("iy", 2);
      dqmStore_->tag(meOccupancy_[i], i+1);

      name = "EEOT MEM digi occupancy " + Numbers::sEE(i+1);
      meOccupancyMem_[i] = dqmStore_->book2D(name, name, 10, 0., 10., 5, 0., 5.);
      meOccupancyMem_[i]->setAxisTitle("pseudo-strip", 1);
      meOccupancyMem_[i]->setAxisTitle("channel", 2);
      dqmStore_->tag(meOccupancyMem_[i], i+1);

      name = "EEOT rec hit energy " + Numbers::sEE(i+1);
      meEERecHitEnergy_[i] = dqmStore_->bookProfile2D(name, name, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 4096, 0., 4096., "s");
      meEERecHitEnergy_[i]->setAxisTitle("ix", 1);
      if ( i+1 >= 1 && i+1 <= 9 ) meEERecHitEnergy_[i]->setAxisTitle("101-ix", 1);
      meEERecHitEnergy_[i]->setAxisTitle("iy", 2);
      meEERecHitEnergy_[i]->setAxisTitle("energy (GeV)", 3);
      dqmStore_->tag(meEERecHitEnergy_[i], i+1);

      name = "EEOT energy spectrum " + Numbers::sEE(i+1);
      meSpectrum_[i] = dqmStore_->book1D(name, name, 100, 0., 1.5);
      meSpectrum_[i]->setAxisTitle("energy (GeV)", 1);
      dqmStore_->tag(meSpectrum_[i], i+1);
    }

    name = "EEOT rec hit spectrum EE -";
    meEERecHitSpectrum_[0] = dqmStore_->book1D(name, name, 100, 0., 10.);
    meEERecHitSpectrum_[0]->setAxisTitle("energy (GeV)", 1);

    name = "EEOT rec hit spectrum EE +";
    meEERecHitSpectrum_[1] = dqmStore_->book1D(name, name, 100, 0., 10.);
    meEERecHitSpectrum_[1]->setAxisTitle("energy (GeV)", 1);

    name = "EEOT digi occupancy EE -";
    meEEDigiOccupancy_[0] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
    meEEDigiOccupancy_[0]->setAxisTitle("jx", 1);
    meEEDigiOccupancy_[0]->setAxisTitle("jy", 2);
    name = "EEOT digi occupancy EE - projection eta";
    meEEDigiOccupancyProEta_[0] = dqmStore_->book1DD(name, name, 22, -3.0, -1.479);
    meEEDigiOccupancyProEta_[0]->setAxisTitle("eta", 1);
    meEEDigiOccupancyProEta_[0]->setAxisTitle("number of digis", 2);
    name = "EEOT digi occupancy EE - projection phi";
    meEEDigiOccupancyProPhi_[0] = dqmStore_->book1DD(name, name, 50, -M_PI, M_PI);
    meEEDigiOccupancyProPhi_[0]->setAxisTitle("phi", 1);
    meEEDigiOccupancyProPhi_[0]->setAxisTitle("number of digis", 2);

    name = "EEOT digi occupancy EE +";
    meEEDigiOccupancy_[1] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
    meEEDigiOccupancy_[1]->setAxisTitle("jx", 1);
    meEEDigiOccupancy_[1]->setAxisTitle("jy", 2);
    name = "EEOT digi occupancy EE + projection eta";
    meEEDigiOccupancyProEta_[1] = dqmStore_->book1DD(name, name, 22, 1.479, 3.0);
    meEEDigiOccupancyProEta_[1]->setAxisTitle("eta", 1);
    meEEDigiOccupancyProEta_[1]->setAxisTitle("number of digis", 2);
    name = "EEOT digi occupancy EE + projection phi";
    meEEDigiOccupancyProPhi_[1] = dqmStore_->book1DD(name, name, 50, -M_PI, M_PI);
    meEEDigiOccupancyProPhi_[1]->setAxisTitle("phi", 1);
    meEEDigiOccupancyProPhi_[1]->setAxisTitle("number of digis", 2);

    name = "EEOT rec hit occupancy EE -";
    meEERecHitOccupancy_[0] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
    meEERecHitOccupancy_[0]->setAxisTitle("jx", 1);
    meEERecHitOccupancy_[0]->setAxisTitle("jy", 2);
    name = "EEOT rec hit occupancy EE - projection eta";
    meEERecHitOccupancyProEta_[0] = dqmStore_->book1DD(name, name, 22, -3.0, -1.479);
    meEERecHitOccupancyProEta_[0]->setAxisTitle("eta", 1);
    meEERecHitOccupancyProEta_[0]->setAxisTitle("number of hits", 2);
    name = "EEOT rec hit occupancy EE - projection phi";
    meEERecHitOccupancyProPhi_[0] = dqmStore_->book1DD(name, name, 50, -M_PI, M_PI);
    meEERecHitOccupancyProPhi_[0]->setAxisTitle("phi", 1);
    meEERecHitOccupancyProPhi_[0]->setAxisTitle("number of hits", 2);

    name = "EEOT rec hit occupancy EE +";
    meEERecHitOccupancy_[1] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
    meEERecHitOccupancy_[1]->setAxisTitle("jx", 1);
    meEERecHitOccupancy_[1]->setAxisTitle("jy", 2);
    name = "EEOT rec hit occupancy EE + projection eta";
    meEERecHitOccupancyProEta_[1] = dqmStore_->book1DD(name, name, 22, 1.479, 3.0);
    meEERecHitOccupancyProEta_[1]->setAxisTitle("eta", 1);
    meEERecHitOccupancyProEta_[1]->setAxisTitle("number of hits", 2);
    name = "EEOT rec hit occupancy EE + projection phi";
    meEERecHitOccupancyProPhi_[1] = dqmStore_->book1DD(name, name, 50, -M_PI, M_PI);
    meEERecHitOccupancyProPhi_[1]->setAxisTitle("phi", 1);
    meEERecHitOccupancyProPhi_[1]->setAxisTitle("number of hits", 2);

    name = "EEOT rec hit thr occupancy EE -";
    meEERecHitOccupancyThr_[0] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
    meEERecHitOccupancyThr_[0]->setAxisTitle("jx", 1);
    meEERecHitOccupancyThr_[0]->setAxisTitle("jy", 2);
    name = "EEOT rec hit thr occupancy EE - projection eta";
    meEERecHitOccupancyProEtaThr_[0] = dqmStore_->book1DD(name, name, 22, -3.0, -1.479);
    meEERecHitOccupancyProEtaThr_[0]->setAxisTitle("eta", 1);
    meEERecHitOccupancyProEtaThr_[0]->setAxisTitle("number of hits", 2);
    name = "EEOT rec hit thr occupancy EE - projection phi";
    meEERecHitOccupancyProPhiThr_[0] = dqmStore_->book1DD(name, name, 50, -M_PI, M_PI);
    meEERecHitOccupancyProPhiThr_[0]->setAxisTitle("phi", 1);
    meEERecHitOccupancyProPhiThr_[0]->setAxisTitle("number of hits", 2);

    name = "EEOT rec hit thr occupancy EE +";
    meEERecHitOccupancyThr_[1] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
    meEERecHitOccupancyThr_[1]->setAxisTitle("jx", 1);
    meEERecHitOccupancyThr_[1]->setAxisTitle("jy", 2);
    name = "EEOT rec hit thr occupancy EE + projection eta";
    meEERecHitOccupancyProEtaThr_[1] = dqmStore_->book1DD(name, name, 22, 1.479, 3.0);
    meEERecHitOccupancyProEtaThr_[1]->setAxisTitle("eta", 1);
    meEERecHitOccupancyProEtaThr_[1]->setAxisTitle("number of hits", 2);
    name = "EEOT rec hit thr occupancy EE + projection phi";
    meEERecHitOccupancyProPhiThr_[1] = dqmStore_->book1DD(name, name, 50, -M_PI, M_PI);
    meEERecHitOccupancyProPhiThr_[1]->setAxisTitle("phi", 1);
    meEERecHitOccupancyProPhiThr_[1]->setAxisTitle("number of hits", 2);

    name = "EEOT TP digi occupancy EE -";
    meEETrigPrimDigiOccupancy_[0] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
    meEETrigPrimDigiOccupancy_[0]->setAxisTitle("jx", 1);
    meEETrigPrimDigiOccupancy_[0]->setAxisTitle("jy", 2);
    name = "EEOT TP digi occupancy EE - projection eta";
    meEETrigPrimDigiOccupancyProEta_[0] = dqmStore_->book1DD(name, name, 22, -3.0, -1.479);
    meEETrigPrimDigiOccupancyProEta_[0]->setAxisTitle("eta", 1);
    meEETrigPrimDigiOccupancyProEta_[0]->setAxisTitle("number of TP digis", 2);
    name = "EEOT TP digi occupancy EE - projection phi";
    meEETrigPrimDigiOccupancyProPhi_[0] = dqmStore_->book1DD(name, name, 50, -M_PI, M_PI);
    meEETrigPrimDigiOccupancyProPhi_[0]->setAxisTitle("phi", 1);
    meEETrigPrimDigiOccupancyProPhi_[0]->setAxisTitle("number of TP digis", 2);

    name = "EEOT TP digi occupancy EE +";
    meEETrigPrimDigiOccupancy_[1] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
    meEETrigPrimDigiOccupancy_[1]->setAxisTitle("jx", 1);
    meEETrigPrimDigiOccupancy_[1]->setAxisTitle("jy", 2);
    name = "EEOT TP digi occupancy EE + projection eta";
    meEETrigPrimDigiOccupancyProEta_[1] = dqmStore_->book1DD(name, name, 22, 1.479, 3.0);
    meEETrigPrimDigiOccupancyProEta_[1]->setAxisTitle("eta", 1);
    meEETrigPrimDigiOccupancyProEta_[1]->setAxisTitle("number of TP digis", 2);
    name = "EEOT TP digi occupancy EE + projection phi";
    meEETrigPrimDigiOccupancyProPhi_[1] = dqmStore_->book1DD(name, name, 50, -M_PI, M_PI);
    meEETrigPrimDigiOccupancyProPhi_[1]->setAxisTitle("phi", 1);
    meEETrigPrimDigiOccupancyProPhi_[1]->setAxisTitle("number of TP digis", 2);

    name = "EEOT TP digi thr occupancy EE -";
    meEETrigPrimDigiOccupancyThr_[0] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
    meEETrigPrimDigiOccupancyThr_[0]->setAxisTitle("jx", 1);
    meEETrigPrimDigiOccupancyThr_[0]->setAxisTitle("jy", 2);
    name = "EEOT TP digi thr occupancy EE - projection eta";
    meEETrigPrimDigiOccupancyProEtaThr_[0] = dqmStore_->book1DD(name, name, 22, -3.0, -1.479);
    meEETrigPrimDigiOccupancyProEtaThr_[0]->setAxisTitle("eta", 1);
    meEETrigPrimDigiOccupancyProEtaThr_[0]->setAxisTitle("number of TP digis", 2);
    name = "EEOT TP digi thr occupancy EE - projection phi";
    meEETrigPrimDigiOccupancyProPhiThr_[0] = dqmStore_->book1DD(name, name, 50, -M_PI, M_PI);
    meEETrigPrimDigiOccupancyProPhiThr_[0]->setAxisTitle("phi", 1);
    meEETrigPrimDigiOccupancyProPhiThr_[0]->setAxisTitle("number of TP digis", 2);

    name = "EEOT TP digi thr occupancy EE +";
    meEETrigPrimDigiOccupancyThr_[1] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
    meEETrigPrimDigiOccupancyThr_[1]->setAxisTitle("jx", 1);
    meEETrigPrimDigiOccupancyThr_[1]->setAxisTitle("jy", 2);
    name = "EEOT TP digi thr occupancy EE + projection eta";
    meEETrigPrimDigiOccupancyProEtaThr_[1] = dqmStore_->book1DD(name, name, 22, 1.479, 3.0);
    meEETrigPrimDigiOccupancyProEtaThr_[1]->setAxisTitle("eta", 1);
    meEETrigPrimDigiOccupancyProEtaThr_[1]->setAxisTitle("number of TP digis", 2);
    name = "EEOT TP digi thr occupancy EE + projection phi";
    meEETrigPrimDigiOccupancyProPhiThr_[1] = dqmStore_->book1DD(name, name, 50, -M_PI, M_PI);
    meEETrigPrimDigiOccupancyProPhiThr_[1]->setAxisTitle("phi", 1);
    meEETrigPrimDigiOccupancyProPhiThr_[1]->setAxisTitle("number of TP digis", 2);

    name = "EEOT test pulse digi occupancy EE -";
    meEETestPulseDigiOccupancy_[0] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
    meEETestPulseDigiOccupancy_[0]->setAxisTitle("jx", 1);
    meEETestPulseDigiOccupancy_[0]->setAxisTitle("jy", 2);

    name = "EEOT test pulse digi occupancy EE +";
    meEETestPulseDigiOccupancy_[1] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
    meEETestPulseDigiOccupancy_[1]->setAxisTitle("jx", 1);
    meEETestPulseDigiOccupancy_[1]->setAxisTitle("jy", 2);

    name = "EEOT led digi occupancy EE -";
    meEELedDigiOccupancy_[0] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
    meEELedDigiOccupancy_[0]->setAxisTitle("jx", 1);
    meEELedDigiOccupancy_[0]->setAxisTitle("jy", 2);

    name = "EEOT led digi occupancy EE +";
    meEELedDigiOccupancy_[1] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
    meEELedDigiOccupancy_[1]->setAxisTitle("jx", 1);
    meEELedDigiOccupancy_[1]->setAxisTitle("jy", 2);

    name = "EEOT laser digi occupancy EE -";
    meEELaserDigiOccupancy_[0] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
    meEELaserDigiOccupancy_[0]->setAxisTitle("jx", 1);
    meEELaserDigiOccupancy_[0]->setAxisTitle("jy", 2);

    name = "EEOT laser digi occupancy EE +";
    meEELaserDigiOccupancy_[1] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
    meEELaserDigiOccupancy_[1]->setAxisTitle("jx", 1);
    meEELaserDigiOccupancy_[1]->setAxisTitle("jy", 2);

    name = "EEOT pedestal digi occupancy EE -";
    meEEPedestalDigiOccupancy_[0] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
    meEEPedestalDigiOccupancy_[0]->setAxisTitle("jx", 1);
    meEEPedestalDigiOccupancy_[0]->setAxisTitle("jy", 2);

    name = "EEOT pedestal digi occupancy EE +";
    meEEPedestalDigiOccupancy_[1] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
    meEEPedestalDigiOccupancy_[1]->setAxisTitle("jx", 1);
    meEEPedestalDigiOccupancy_[1]->setAxisTitle("jy", 2);

  }

}

void EEOccupancyTask::cleanup(void){

  if ( ! init_ ) return;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EEOccupancyTask");
    if(subfolder_.size())
      dqmStore_->setCurrentFolder(prefixME_ + "/EEOccupancyTask/" + subfolder_);

    for (int i = 0; i < 18; i++) {
      if ( meOccupancy_[i] ) dqmStore_->removeElement( meOccupancy_[i]->getName() );
      meOccupancy_[i] = 0;
      if ( meOccupancyMem_[i] ) dqmStore_->removeElement( meOccupancyMem_[i]->getName() );
      meOccupancyMem_[i] = 0;
      if ( meEERecHitEnergy_[i] ) dqmStore_->removeElement( meEERecHitEnergy_[i]->getName() );
      meEERecHitEnergy_[i] = 0;
      if ( meSpectrum_[i] ) dqmStore_->removeElement( meSpectrum_[i]->getName() );
      meSpectrum_[i] = 0;
    }

    if ( meEERecHitSpectrum_[0] ) dqmStore_->removeElement( meEERecHitSpectrum_[0]->getName() );
    meEERecHitSpectrum_[0] = 0;
    if ( meEERecHitSpectrum_[1] ) dqmStore_->removeElement( meEERecHitSpectrum_[1]->getName() );
    meEERecHitSpectrum_[1] = 0;

    if ( meEEDigiOccupancy_[0] ) dqmStore_->removeElement( meEEDigiOccupancy_[0]->getName() );
    meEEDigiOccupancy_[0] = 0;
    if ( meEEDigiOccupancyProEta_[0] ) dqmStore_->removeElement( meEEDigiOccupancyProEta_[0]->getName() );
    meEEDigiOccupancyProEta_[0] = 0;
    if ( meEEDigiOccupancyProPhi_[0] ) dqmStore_->removeElement( meEEDigiOccupancyProPhi_[0]->getName() );
    meEEDigiOccupancyProPhi_[0] = 0;

    if ( meEEDigiOccupancy_[1] ) dqmStore_->removeElement( meEEDigiOccupancy_[1]->getName() );
    meEEDigiOccupancy_[1] = 0;
    if ( meEEDigiOccupancyProEta_[1] ) dqmStore_->removeElement( meEEDigiOccupancyProEta_[1]->getName() );
    meEEDigiOccupancyProEta_[1] = 0;
    if ( meEEDigiOccupancyProPhi_[1] ) dqmStore_->removeElement( meEEDigiOccupancyProPhi_[1]->getName() );
    meEEDigiOccupancyProPhi_[1] = 0;

    if ( meEERecHitOccupancy_[0] ) dqmStore_->removeElement( meEERecHitOccupancy_[0]->getName() );
    meEERecHitOccupancy_[0] = 0;
    if ( meEERecHitOccupancyProEta_[0] ) dqmStore_->removeElement( meEERecHitOccupancyProEta_[0]->getName() );
    meEERecHitOccupancyProEta_[0] = 0;
    if ( meEERecHitOccupancyProPhi_[0] ) dqmStore_->removeElement( meEERecHitOccupancyProPhi_[0]->getName() );
    meEERecHitOccupancyProPhi_[0] = 0;

    if ( meEERecHitOccupancy_[1] ) dqmStore_->removeElement( meEERecHitOccupancy_[1]->getName() );
    meEERecHitOccupancy_[1] = 0;
    if ( meEERecHitOccupancyProEta_[1] ) dqmStore_->removeElement( meEERecHitOccupancyProEta_[1]->getName() );
    meEERecHitOccupancyProEta_[1] = 0;
    if ( meEERecHitOccupancyProPhi_[1] ) dqmStore_->removeElement( meEERecHitOccupancyProPhi_[1]->getName() );
    meEERecHitOccupancyProPhi_[1] = 0;

    if ( meEERecHitOccupancyThr_[0] ) dqmStore_->removeElement( meEERecHitOccupancyThr_[0]->getName() );
    meEERecHitOccupancyThr_[0] = 0;
    if ( meEERecHitOccupancyProEtaThr_[0] ) dqmStore_->removeElement( meEERecHitOccupancyProEtaThr_[0]->getName() );
    meEERecHitOccupancyProEtaThr_[0] = 0;
    if ( meEERecHitOccupancyProPhiThr_[0] ) dqmStore_->removeElement( meEERecHitOccupancyProPhiThr_[0]->getName() );
    meEERecHitOccupancyProPhiThr_[0] = 0;

    if ( meEERecHitOccupancyThr_[1] ) dqmStore_->removeElement( meEERecHitOccupancyThr_[1]->getName() );
    meEERecHitOccupancyThr_[1] = 0;
    if ( meEERecHitOccupancyProEtaThr_[1] ) dqmStore_->removeElement( meEERecHitOccupancyProEtaThr_[1]->getName() );
    meEERecHitOccupancyProEtaThr_[1] = 0;
    if ( meEERecHitOccupancyProPhiThr_[1] ) dqmStore_->removeElement( meEERecHitOccupancyProPhiThr_[1]->getName() );
    meEERecHitOccupancyProPhiThr_[1] = 0;

    if ( meEETrigPrimDigiOccupancy_[0] ) dqmStore_->removeElement( meEETrigPrimDigiOccupancy_[0]->getName() );
    meEETrigPrimDigiOccupancy_[0] = 0;
    if ( meEETrigPrimDigiOccupancyProEta_[0] ) dqmStore_->removeElement( meEETrigPrimDigiOccupancyProEta_[0]->getName() );
    meEETrigPrimDigiOccupancyProEta_[0] = 0;
    if ( meEETrigPrimDigiOccupancyProPhi_[0] ) dqmStore_->removeElement( meEETrigPrimDigiOccupancyProPhi_[0]->getName() );
    meEETrigPrimDigiOccupancyProPhi_[0] = 0;

    if ( meEETrigPrimDigiOccupancy_[1] ) dqmStore_->removeElement( meEETrigPrimDigiOccupancy_[1]->getName() );
    meEETrigPrimDigiOccupancy_[1] = 0;
    if ( meEETrigPrimDigiOccupancyProEta_[1] ) dqmStore_->removeElement( meEETrigPrimDigiOccupancyProEta_[1]->getName() );
    meEETrigPrimDigiOccupancyProEta_[1] = 0;
    if ( meEETrigPrimDigiOccupancyProPhi_[1] ) dqmStore_->removeElement( meEETrigPrimDigiOccupancyProPhi_[1]->getName() );
    meEETrigPrimDigiOccupancyProPhi_[1] = 0;

    if ( meEETrigPrimDigiOccupancyThr_[0] ) dqmStore_->removeElement( meEETrigPrimDigiOccupancyThr_[0]->getName() );
    meEETrigPrimDigiOccupancyThr_[0] = 0;
    if ( meEETrigPrimDigiOccupancyProEtaThr_[0] ) dqmStore_->removeElement( meEETrigPrimDigiOccupancyProEtaThr_[0]->getName() );
    meEETrigPrimDigiOccupancyProEtaThr_[0] = 0;
    if ( meEETrigPrimDigiOccupancyProPhiThr_[0] ) dqmStore_->removeElement( meEETrigPrimDigiOccupancyProPhiThr_[0]->getName() );
    meEETrigPrimDigiOccupancyProPhiThr_[0] = 0;

    if ( meEETrigPrimDigiOccupancyThr_[1] ) dqmStore_->removeElement( meEETrigPrimDigiOccupancyThr_[1]->getName() );
    meEETrigPrimDigiOccupancyThr_[1] = 0;
    if ( meEETrigPrimDigiOccupancyProEtaThr_[1] ) dqmStore_->removeElement( meEETrigPrimDigiOccupancyProEtaThr_[1]->getName() );
    meEETrigPrimDigiOccupancyProEtaThr_[1] = 0;
    if ( meEETrigPrimDigiOccupancyProPhiThr_[1] ) dqmStore_->removeElement( meEETrigPrimDigiOccupancyProPhiThr_[1]->getName() );
    meEETrigPrimDigiOccupancyProPhiThr_[1] = 0;

    if ( meEETestPulseDigiOccupancy_[0] ) dqmStore_->removeElement( meEETestPulseDigiOccupancy_[0]->getName() );
    meEETestPulseDigiOccupancy_[0] = 0;
    if ( meEETestPulseDigiOccupancy_[1] ) dqmStore_->removeElement( meEETestPulseDigiOccupancy_[1]->getName() );
    meEETestPulseDigiOccupancy_[1] = 0;

    if ( meEELaserDigiOccupancy_[0] ) dqmStore_->removeElement( meEELaserDigiOccupancy_[0]->getName() );
    meEELaserDigiOccupancy_[0] = 0;
    if ( meEELaserDigiOccupancy_[1] ) dqmStore_->removeElement( meEELaserDigiOccupancy_[1]->getName() );
    meEELaserDigiOccupancy_[1] = 0;

    if ( meEELedDigiOccupancy_[0] ) dqmStore_->removeElement( meEELedDigiOccupancy_[0]->getName() );
    meEELedDigiOccupancy_[0] = 0;
    if ( meEELedDigiOccupancy_[1] ) dqmStore_->removeElement( meEELedDigiOccupancy_[1]->getName() );
    meEELedDigiOccupancy_[1] = 0;

    if ( meEEPedestalDigiOccupancy_[0] ) dqmStore_->removeElement( meEEPedestalDigiOccupancy_[0]->getName() );
    meEEPedestalDigiOccupancy_[0] = 0;
    if ( meEEPedestalDigiOccupancy_[1] ) dqmStore_->removeElement( meEEPedestalDigiOccupancy_[1]->getName() );
    meEEPedestalDigiOccupancy_[1] = 0;

  }

  init_ = false;

}

void EEOccupancyTask::endJob(void) {

  edm::LogInfo("EEOccupancyTask") << "analyzed " << ievt_ << " events";

  if ( enableCleanup_ ) this->cleanup();

}

void EEOccupancyTask::analyze(const edm::Event& e, const edm::EventSetup& c){

  if ( ! init_ ) this->setup();

  ievt_++;

  int runType[18] = { notdata };

  edm::Handle<EcalRawDataCollection> dcchs;

  if ( e.getByLabel(EcalRawDataCollection_, dcchs) ) {

    for ( EcalRawDataCollection::const_iterator dcchItr = dcchs->begin(); dcchItr != dcchs->end(); ++dcchItr ) {

      if ( Numbers::subDet( *dcchItr ) != EcalEndcap ) continue;

      int ism = Numbers::iSM( *dcchItr, EcalEndcap );

      int runtype = dcchItr->getRunType();

      if ( runtype == EcalDCCHeaderBlock::COSMIC ||
           runtype == EcalDCCHeaderBlock::MTCC ||
           runtype == EcalDCCHeaderBlock::COSMICS_GLOBAL ||
           runtype == EcalDCCHeaderBlock::PHYSICS_GLOBAL ||
           runtype == EcalDCCHeaderBlock::COSMICS_LOCAL ||
           runtype == EcalDCCHeaderBlock::PHYSICS_LOCAL ) runType[ism-1] = physics;
      if ( runtype == EcalDCCHeaderBlock::TESTPULSE_MGPA ||
           runtype == EcalDCCHeaderBlock::TESTPULSE_GAP ) runType[ism-1] = testpulse;
      if ( runtype == EcalDCCHeaderBlock::LASER_STD ||
           runtype == EcalDCCHeaderBlock::LASER_GAP ) runType[ism-1] = laser;
      if ( runtype == EcalDCCHeaderBlock::LED_STD ||
           runtype == EcalDCCHeaderBlock::LED_GAP ) runType[ism-1] = led;
      if ( runtype == EcalDCCHeaderBlock::PEDESTAL_STD ||
           runtype == EcalDCCHeaderBlock::PEDESTAL_GAP ) runType[ism-1] = pedestal;

    }

  } else {
    edm::LogWarning("EEOccupancyTask") << EcalRawDataCollection_ << " not available";
  }

  edm::Handle<EEDigiCollection> digis;

  if ( e.getByLabel(EEDigiCollection_, digis) ) {

    int need = digis->size();
    LogDebug("EEOccupancyTask") << "event " << ievt_ << " digi collection size " << need;

    for ( EEDigiCollection::const_iterator digiItr = digis->begin(); digiItr != digis->end(); ++digiItr ) {

      EEDetId id = digiItr->id();

      int ix = id.ix();
      int iy = id.iy();

      int hi = id.hashedIndex();

      if ( geometryEE[hi][0] == 0 ) {
        const GlobalPoint& pos = pGeometry_->getGeometry(id)->getPosition();
        geometryEE[hi][0] = pos.eta();
        geometryEE[hi][1] = pos.phi();
      }

      float eta = geometryEE[hi][0];
      float phi = geometryEE[hi][1];

      int ism = Numbers::iSM( id );

      if ( ism >= 1 && ism <= 9 ) ix = 101 - ix;

      float xix = ix - 0.5;
      float xiy = iy - 0.5;

      if ( xix <= 0. || xix >= 100. || xiy <= 0. || xiy >= 100. ) {
        edm::LogWarning("EEOccupancyTask") << " det id = " << id;
        edm::LogWarning("EEOccupancyTask") << " sm, ix, iw " << ism << " " << ix << " " << iy;
        edm::LogWarning("EEOccupancyTask") << " xix, xiy " << xix << " " << xiy;
      }

      if ( meOccupancy_[ism-1] ) meOccupancy_[ism-1]->Fill( xix, xiy );

      int eex = id.ix();
      int eey = id.iy();

      float xeex = eex - 0.5;
      float xeey = eey - 0.5;

      if ( runType[ism-1] == physics || runType[ism-1] == notdata ) {

        if ( ism >=1 && ism <= 9 ) {
          if ( meEEDigiOccupancy_[0] ) meEEDigiOccupancy_[0]->Fill( xeex, xeey );
          if ( meEEDigiOccupancyProEta_[0] ) meEEDigiOccupancyProEta_[0]->Fill( eta );
          if ( meEEDigiOccupancyProPhi_[0] ) meEEDigiOccupancyProPhi_[0]->Fill( phi );
        } else {
          if ( meEEDigiOccupancy_[1] ) meEEDigiOccupancy_[1]->Fill( xeex, xeey );
          if ( meEEDigiOccupancyProEta_[1] ) meEEDigiOccupancyProEta_[1]->Fill( eta );
          if ( meEEDigiOccupancyProPhi_[1] ) meEEDigiOccupancyProPhi_[1]->Fill( phi );
        }

      }

      if ( runType[ism-1] == testpulse ) {

        if ( ism >=1 && ism <= 9 ) {
          if ( meEETestPulseDigiOccupancy_[0] ) meEETestPulseDigiOccupancy_[0]->Fill( xeex, xeey );
        } else {
          if ( meEETestPulseDigiOccupancy_[1] ) meEETestPulseDigiOccupancy_[1]->Fill( xeex, xeey );
        }

      }

      if ( runType[ism-1] == laser ) {

        if ( ism >=1 && ism <= 9 ) {
          if ( meEELaserDigiOccupancy_[0] ) meEELaserDigiOccupancy_[0]->Fill( xeex, xeey );
        } else {
          if ( meEELaserDigiOccupancy_[1] ) meEELaserDigiOccupancy_[1]->Fill( xeex, xeey );
        }

      }

      if ( runType[ism-1] == led ) {

        if ( ism >=1 && ism <= 9 ) {
          if ( meEELedDigiOccupancy_[0] ) meEELedDigiOccupancy_[0]->Fill( xeex, xeey );
        } else {
          if ( meEELedDigiOccupancy_[1] ) meEELedDigiOccupancy_[1]->Fill( xeex, xeey );
        }

      }

      if ( runType[ism-1] == pedestal ) {

        if ( ism >=1 && ism <= 9 ) {
          if ( meEEPedestalDigiOccupancy_[0] ) meEEPedestalDigiOccupancy_[0]->Fill( xeex, xeey );
        } else {
          if ( meEEPedestalDigiOccupancy_[1] ) meEEPedestalDigiOccupancy_[1]->Fill( xeex, xeey );
        }

      }

    }

  } else {

    edm::LogWarning("EEOccupancyTask") << EEDigiCollection_ << " not available";

  }

  edm::Handle<EcalPnDiodeDigiCollection> PNs;

  if ( e.getByLabel(EcalPnDiodeDigiCollection_, PNs) ) {

    // filling mem occupancy only for the 5 channels belonging
    // to a fully reconstructed PN's

    for ( EcalPnDiodeDigiCollection::const_iterator pnItr = PNs->begin(); pnItr != PNs->end(); ++pnItr ) {

      if ( Numbers::subDet( pnItr->id() ) != EcalEndcap ) continue;

      int   ism   = Numbers::iSM( pnItr->id() );

      float PnId  = pnItr->id().iPnId();

      PnId        = PnId - 0.5;
      float st    = 0.0;

      for (int chInStrip = 1; chInStrip <= 5; chInStrip++){
        if ( meOccupancyMem_[ism-1] ) {
          st = chInStrip - 0.5;
          meOccupancyMem_[ism-1]->Fill(PnId, st);
        }
      }

    }

  } else {

    edm::LogWarning("EEOccupancyTask") << EcalPnDiodeDigiCollection_ << " not available";

  }

  edm::ESHandle<EcalSeverityLevelAlgo> sevlv;
  c.get<EcalSeverityLevelAlgoRcd>().get(sevlv);

  edm::Handle<EcalRecHitCollection> rechits;

  if ( e.getByLabel(EcalRecHitCollection_, rechits) ) {

    int nebrh = rechits->size();
    LogDebug("EEOccupancyTask") << "event " << ievt_ << " rec hits collection size " << nebrh;

    for ( EcalRecHitCollection::const_iterator rechitItr = rechits->begin(); rechitItr != rechits->end(); ++rechitItr ) {

      EEDetId id = rechitItr->id();

      int eex = id.ix();
      int eey = id.iy();

      int hi = id.hashedIndex();

      if ( geometryEE[hi][0] == 0 ) {
        const GlobalPoint& pos = pGeometry_->getGeometry(id)->getPosition();
        geometryEE[hi][0] = pos.eta();
        geometryEE[hi][1] = pos.phi();
      }

      float eta = geometryEE[hi][0];
      float phi = geometryEE[hi][1];

      int ism = Numbers::iSM( id );

      // sector view (from electronics)
      float xix = ( ism >= 1 && ism <= 9 ) ? 101 - eex - 0.5 : eex - 0.5;
      float xiy = eey - 0.5;

      // physics view (from IP)
      float xeex = eex - 0.5;
      float xeey = eey - 0.5;

      if ( runType[ism-1] == physics || runType[ism-1] == notdata ) {

        if ( ism >= 1 && ism <= 9 ) {
          if ( meEERecHitOccupancy_[0] ) meEERecHitOccupancy_[0]->Fill( xeex, xeey );
          if ( meEERecHitOccupancyProEta_[0] ) meEERecHitOccupancyProEta_[0]->Fill( eta );
          if ( meEERecHitOccupancyProPhi_[0] ) meEERecHitOccupancyProPhi_[0]->Fill( phi );
        } else {
          if ( meEERecHitOccupancy_[1] ) meEERecHitOccupancy_[1]->Fill( xeex, xeey );
          if ( meEERecHitOccupancyProEta_[1] ) meEERecHitOccupancyProEta_[1]->Fill( eta );
          if ( meEERecHitOccupancyProPhi_[1] ) meEERecHitOccupancyProPhi_[1]->Fill( phi );
        }

        uint32_t flag = rechitItr->recoFlag();

        uint32_t sev = sevlv->severityLevel(id, *rechits);

        if ( rechitItr->energy() > recHitEnergyMin_ && flag == EcalRecHit::kGood && sev == EcalSeverityLevel::kGood ) {

          if ( ism >= 1 && ism <= 9 ) {
            if ( meEERecHitOccupancyThr_[0] ) meEERecHitOccupancyThr_[0]->Fill( xeex, xeey );
            if ( meEERecHitOccupancyProEtaThr_[0] ) meEERecHitOccupancyProEtaThr_[0]->Fill( eta );
            if ( meEERecHitOccupancyProPhiThr_[0] ) meEERecHitOccupancyProPhiThr_[0]->Fill( phi );
          } else {
            if ( meEERecHitOccupancyThr_[1] ) meEERecHitOccupancyThr_[1]->Fill( xeex, xeey );
            if ( meEERecHitOccupancyProEtaThr_[1] ) meEERecHitOccupancyProEtaThr_[1]->Fill( eta );
            if ( meEERecHitOccupancyProPhiThr_[1] ) meEERecHitOccupancyProPhiThr_[1]->Fill( phi );
          }

        }

        if ( flag == EcalRecHit::kGood && sev == EcalSeverityLevel::kGood ) {
          if ( meEERecHitEnergy_[ism-1] ) meEERecHitEnergy_[ism-1]->Fill( xix, xiy, rechitItr->energy() );
          if ( meSpectrum_[ism-1] ) meSpectrum_[ism-1]->Fill( rechitItr->energy() );
          if (  ism >= 1 && ism <= 9  ) meEERecHitSpectrum_[0]->Fill( rechitItr->energy() );
          else meEERecHitSpectrum_[1]->Fill( rechitItr->energy() );
        }

      }
    }

  } else {

    edm::LogWarning("EEOccupancyTask") << EcalRecHitCollection_ << " not available";

  }

  edm::Handle<EcalTrigPrimDigiCollection> trigPrimDigis;

  if ( e.getByLabel(EcalTrigPrimDigiCollection_, trigPrimDigis) ) {

    int nebtpg = trigPrimDigis->size();
    LogDebug("EEOccupancyTask") << "event " << ievt_ << " trigger primitives digis collection size " << nebtpg;

    for ( EcalTrigPrimDigiCollection::const_iterator tpdigiItr = trigPrimDigis->begin(); tpdigiItr != trigPrimDigis->end(); ++tpdigiItr ) {

      if ( Numbers::subDet( tpdigiItr->id() ) != EcalEndcap ) continue;

      int ism = Numbers::iSM( tpdigiItr->id() );

      std::vector<DetId>* crystals = Numbers::crystals( tpdigiItr->id() );

      for ( unsigned int i=0; i<crystals->size(); i++ ) {

        EEDetId id = (*crystals)[i];

        int eex = id.ix();
        int eey = id.iy();

        int hi = id.hashedIndex();

        if ( geometryEE[hi][0] == 0 ) {
          const GlobalPoint& pos = pGeometry_->getGeometry(id)->getPosition();
          geometryEE[hi][0] = pos.eta();
          geometryEE[hi][1] = pos.phi();
        }

        float eta = geometryEE[hi][0];
        float phi = geometryEE[hi][1];

        float xeex = eex - 0.5;
        float xeey = eey - 0.5;

        if ( runType[ism-1] == physics || runType[ism-1] == notdata ) {

          if ( ism >= 1 && ism <= 9 ) {
            if ( meEETrigPrimDigiOccupancy_[0] ) meEETrigPrimDigiOccupancy_[0]->Fill( xeex, xeey );
            if ( meEETrigPrimDigiOccupancyProEta_[0] ) meEETrigPrimDigiOccupancyProEta_[0]->Fill( eta );
            if ( meEETrigPrimDigiOccupancyProPhi_[0] ) meEETrigPrimDigiOccupancyProPhi_[0]->Fill( phi );
          } else {
            if ( meEETrigPrimDigiOccupancy_[1] ) meEETrigPrimDigiOccupancy_[1]->Fill( xeex, xeey );
            if ( meEETrigPrimDigiOccupancyProEta_[1] ) meEETrigPrimDigiOccupancyProEta_[1]->Fill( eta );
            if ( meEETrigPrimDigiOccupancyProPhi_[1] ) meEETrigPrimDigiOccupancyProPhi_[1]->Fill( phi );
          }

          if ( tpdigiItr->compressedEt() > trigPrimEtMin_ ) {

            if ( ism >= 1 && ism <= 9 ) {
              if ( meEETrigPrimDigiOccupancyThr_[0] ) meEETrigPrimDigiOccupancyThr_[0]->Fill( xeex, xeey );
              if ( meEETrigPrimDigiOccupancyProEtaThr_[0] ) meEETrigPrimDigiOccupancyProEtaThr_[0]->Fill( eta );
              if ( meEETrigPrimDigiOccupancyProPhiThr_[0] ) meEETrigPrimDigiOccupancyProPhiThr_[0]->Fill( phi );
            } else {
              if ( meEETrigPrimDigiOccupancyThr_[1] ) meEETrigPrimDigiOccupancyThr_[1]->Fill( xeex, xeey );
              if ( meEETrigPrimDigiOccupancyProEtaThr_[1] ) meEETrigPrimDigiOccupancyProEtaThr_[1]->Fill( eta );
              if ( meEETrigPrimDigiOccupancyProPhiThr_[1] ) meEETrigPrimDigiOccupancyProPhiThr_[1]->Fill( phi );
            }

          }

        }
      }
    }

  } else {

    edm::LogWarning("EEOccupancyTask") << EcalTrigPrimDigiCollection_ << " not available";

  }

}

