/*
 * \file EBLaserTask.cc
 *
 * $Date: 2012/04/27 13:46:02 $
 * $Revision: 1.139 $
 * \author G. Della Ricca
 *
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/MonitorElement.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "DQM/EcalCommon/interface/Numbers.h"

#include "DQM/EcalBarrelMonitorTasks/interface/EBLaserTask.h"

EBLaserTask::EBLaserTask(const edm::ParameterSet& ps){

  init_ = false;

  dqmStore_ = edm::Service<DQMStore>().operator->();

  prefixME_ = ps.getUntrackedParameter<std::string>("prefixME", "");

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  mergeRuns_ = ps.getUntrackedParameter<bool>("mergeRuns", false);

  EcalRawDataCollection_ = ps.getParameter<edm::InputTag>("EcalRawDataCollection");
  EBDigiCollection_ = ps.getParameter<edm::InputTag>("EBDigiCollection");
  EcalPnDiodeDigiCollection_ = ps.getParameter<edm::InputTag>("EcalPnDiodeDigiCollection");
  EcalUncalibratedRecHitCollection_ = ps.getParameter<edm::InputTag>("EcalUncalibratedRecHitCollection");

  // vector of enabled wavelengths (Default to all 4)
  laserWavelengths_.reserve(4);
  for ( unsigned int i = 1; i <= 4; i++ ) laserWavelengths_.push_back(i);
  laserWavelengths_ = ps.getUntrackedParameter<std::vector<int> >("laserWavelengths", laserWavelengths_);

  for (int i = 0; i < 36; i++) {
    meShapeMapL1_[i] = 0;
    meAmplMapL1_[i] = 0;
    meTimeMapL1_[i] = 0;
    meAmplPNMapL1_[i] = 0;
    mePnAmplMapG01L1_[i] = 0;
    mePnPedMapG01L1_[i] = 0;
    mePnAmplMapG16L1_[i] = 0;
    mePnPedMapG16L1_[i] = 0;

    meShapeMapL2_[i] = 0;
    meAmplMapL2_[i] = 0;
    meTimeMapL2_[i] = 0;
    meAmplPNMapL2_[i] = 0;
    mePnAmplMapG01L2_[i] = 0;
    mePnPedMapG01L2_[i] = 0;
    mePnAmplMapG16L2_[i] = 0;
    mePnPedMapG16L2_[i] = 0;

    meShapeMapL3_[i] = 0;
    meAmplMapL3_[i] = 0;
    meTimeMapL3_[i] = 0;
    meAmplPNMapL3_[i] = 0;
    mePnAmplMapG01L3_[i] = 0;
    mePnPedMapG01L3_[i] = 0;
    mePnAmplMapG16L3_[i] = 0;
    mePnPedMapG16L3_[i] = 0;

    meShapeMapL4_[i] = 0;
    meAmplMapL4_[i] = 0;
    meTimeMapL4_[i] = 0;
    meAmplPNMapL4_[i] = 0;
    mePnAmplMapG01L4_[i] = 0;
    mePnPedMapG01L4_[i] = 0;
    mePnAmplMapG16L4_[i] = 0;
    mePnPedMapG16L4_[i] = 0;
  }

  meAmplSummaryMapL1_ = 0;
  meAmplSummaryMapL2_ = 0;
  meAmplSummaryMapL3_ = 0;
  meAmplSummaryMapL4_ = 0;

}

EBLaserTask::~EBLaserTask(){

}

void EBLaserTask::beginJob(void){

  ievt_ = 0;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EBLaserTask");
    dqmStore_->rmdir(prefixME_ + "/EBLaserTask");
  }

}

void EBLaserTask::beginRun(const edm::Run& r, const edm::EventSetup& c) {

  Numbers::initGeometry(c, false);

  if ( ! mergeRuns_ ) this->reset();

}

void EBLaserTask::endRun(const edm::Run& r, const edm::EventSetup& c) {

}

void EBLaserTask::reset(void) {

  for (int i = 0; i < 36; i++) {
    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 1) != laserWavelengths_.end() ) {
      if ( meShapeMapL1_[i] )  meShapeMapL1_[i]->Reset();
      if ( meAmplMapL1_[i] ) meAmplMapL1_[i]->Reset();
      if ( meTimeMapL1_[i] ) meTimeMapL1_[i]->Reset();
      if ( meAmplPNMapL1_[i] ) meAmplPNMapL1_[i]->Reset();
    }

    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 2) != laserWavelengths_.end() ) {
      if ( meShapeMapL2_[i] )  meShapeMapL2_[i]->Reset();
      if ( meAmplMapL2_[i] ) meAmplMapL2_[i]->Reset();
      if ( meTimeMapL2_[i] ) meTimeMapL2_[i]->Reset();
      if ( meAmplPNMapL2_[i] ) meAmplPNMapL2_[i]->Reset();
    }

    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 3) != laserWavelengths_.end() ) {
      if ( meShapeMapL3_[i] )  meShapeMapL3_[i]->Reset();
      if ( meAmplMapL3_[i] ) meAmplMapL3_[i]->Reset();
      if ( meTimeMapL3_[i] ) meTimeMapL3_[i]->Reset();
      if ( meAmplPNMapL3_[i] ) meAmplPNMapL3_[i]->Reset();
    }

    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 4) != laserWavelengths_.end() ) {
      if ( meShapeMapL4_[i] )  meShapeMapL4_[i]->Reset();
      if ( meAmplMapL4_[i] ) meAmplMapL4_[i]->Reset();
      if ( meTimeMapL4_[i] ) meTimeMapL4_[i]->Reset();
      if ( meAmplPNMapL4_[i] ) meAmplPNMapL4_[i]->Reset();
    }

    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 1) != laserWavelengths_.end() ) {
      if ( mePnAmplMapG01L1_[i] ) mePnAmplMapG01L1_[i]->Reset();
      if ( mePnPedMapG01L1_[i] ) mePnPedMapG01L1_[i]->Reset();

      if ( mePnAmplMapG16L1_[i] ) mePnAmplMapG16L1_[i]->Reset();
      if ( mePnPedMapG16L1_[i] ) mePnPedMapG16L1_[i]->Reset();
    }

    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 2) != laserWavelengths_.end() ) {
      if ( mePnAmplMapG01L2_[i] ) mePnAmplMapG01L2_[i]->Reset();
      if ( mePnPedMapG01L2_[i] ) mePnPedMapG01L2_[i]->Reset();

      if ( mePnAmplMapG16L2_[i] ) mePnAmplMapG16L2_[i]->Reset();
      if ( mePnPedMapG16L2_[i] ) mePnPedMapG16L2_[i]->Reset();
    }

    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 3) != laserWavelengths_.end() ) {
      if ( mePnAmplMapG01L3_[i] ) mePnAmplMapG01L3_[i]->Reset();
      if ( mePnPedMapG01L3_[i] ) mePnPedMapG01L3_[i]->Reset();

      if ( mePnAmplMapG16L3_[i] ) mePnAmplMapG16L3_[i]->Reset();
      if ( mePnPedMapG16L3_[i] ) mePnPedMapG16L3_[i]->Reset();
    }

    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 4) != laserWavelengths_.end() ) {
      if ( mePnAmplMapG01L4_[i] ) mePnAmplMapG01L4_[i]->Reset();
      if ( mePnPedMapG01L4_[i] ) mePnPedMapG01L4_[i]->Reset();

      if ( mePnAmplMapG16L4_[i] ) mePnAmplMapG16L4_[i]->Reset();
      if ( mePnPedMapG16L4_[i] ) mePnPedMapG16L4_[i]->Reset();
    }
  }

  if( meAmplSummaryMapL1_ ) meAmplSummaryMapL1_->Reset();
  if( meAmplSummaryMapL2_ ) meAmplSummaryMapL2_->Reset();
  if( meAmplSummaryMapL3_ ) meAmplSummaryMapL3_->Reset();
  if( meAmplSummaryMapL4_ ) meAmplSummaryMapL4_->Reset();

}

void EBLaserTask::setup(void){

  init_ = true;

  std::string name;
  std::stringstream LaserN, LN;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EBLaserTask");

    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 1) != laserWavelengths_.end() ) {

      LaserN.str("");
      LaserN << "Laser" << 1;
      LN.str("");
      LN << "L" << 1;

      dqmStore_->setCurrentFolder(prefixME_ + "/EBLaserTask/" + LaserN.str());
      for (int i = 0; i < 36; i++) {
        name = "EBLT shape " + Numbers::sEB(i+1) + " " + LN.str();
        meShapeMapL1_[i] = dqmStore_->bookProfile2D(name, name, 1700, 0., 1700., 10, 0., 10., 4096, 0., 4096., "s");
        meShapeMapL1_[i]->setAxisTitle("channel", 1);
        meShapeMapL1_[i]->setAxisTitle("sample", 2);
        meShapeMapL1_[i]->setAxisTitle("amplitude", 3);
        dqmStore_->tag(meShapeMapL1_[i], i+1);
        name = "EBLT amplitude " + Numbers::sEB(i+1) + " " + LN.str();
        meAmplMapL1_[i] = dqmStore_->bookProfile2D(name, name, 85, 0., 85., 20, 0., 20., 4096, 0., 4096.*12., "s");
        meAmplMapL1_[i]->setAxisTitle("ieta", 1);
        meAmplMapL1_[i]->setAxisTitle("iphi", 2);
        dqmStore_->tag(meAmplMapL1_[i], i+1);
        name = "EBLT timing " + Numbers::sEB(i+1) + " " + LN.str();
        meTimeMapL1_[i] = dqmStore_->bookProfile2D(name, name, 85, 0., 85., 20, 0., 20., 250, 0., 10., "s");
        meTimeMapL1_[i]->setAxisTitle("ieta", 1);
        meTimeMapL1_[i]->setAxisTitle("iphi", 2);
        dqmStore_->tag(meTimeMapL1_[i], i+1);
        name = "EBLT amplitude over PN " + Numbers::sEB(i+1) + " " + LN.str();
        meAmplPNMapL1_[i] = dqmStore_->bookProfile2D(name, name, 85, 0., 85., 20, 0., 20., 4096, 0., 4096.*12., "s");
        meAmplPNMapL1_[i]->setAxisTitle("ieta", 1);
        meAmplPNMapL1_[i]->setAxisTitle("iphi", 2);
        dqmStore_->tag(meAmplPNMapL1_[i], i+1);
      }

      name = "EBLT amplitude map " + LN.str();
      meAmplSummaryMapL1_ = dqmStore_->bookProfile2D(name, name, 72, 0., 360., 34, -85., 85., 0., 4096.);
      meAmplSummaryMapL1_->setAxisTitle("jphi", 1);
      meAmplSummaryMapL1_->setAxisTitle("jeta", 2);

      dqmStore_->setCurrentFolder(prefixME_ + "/EBLaserTask/" + LaserN.str() + "/PN");

      dqmStore_->setCurrentFolder(prefixME_ + "/EBLaserTask/" + LaserN.str() + "/PN/Gain01");
      for (int i = 0; i < 36; i++) {
        name = "EBLT PNs amplitude " + Numbers::sEB(i+1) + " G01 " + LN.str(); 
        mePnAmplMapG01L1_[i] = dqmStore_->bookProfile(name, name, 10, 0., 10., 4096, 0., 4096., "s");
        mePnAmplMapG01L1_[i]->setAxisTitle("channel", 1);
        mePnAmplMapG01L1_[i]->setAxisTitle("amplitude", 2);
        dqmStore_->tag(mePnAmplMapG01L1_[i], i+1);

	name = "EBLT PNs pedestal " + Numbers::sEB(i+1) + " G01 " + LN.str(); 
        mePnPedMapG01L1_[i] = dqmStore_->bookProfile(name, name, 10, 0., 10., 4096, 0., 4096., "s");
        mePnPedMapG01L1_[i]->setAxisTitle("channel", 1);
        mePnPedMapG01L1_[i]->setAxisTitle("pedestal", 2);
        dqmStore_->tag(mePnPedMapG01L1_[i], i+1);
      }

      dqmStore_->setCurrentFolder(prefixME_ + "/EBLaserTask/" + LaserN.str() + "/PN/Gain16");
      for (int i = 0; i < 36; i++) {
	name = "EBLT PNs amplitude " + Numbers::sEB(i+1) + " G16 " + LN.str(), 
        mePnAmplMapG16L1_[i] = dqmStore_->bookProfile(name, name, 10, 0., 10., 4096, 0., 4096., "s");
        mePnAmplMapG16L1_[i]->setAxisTitle("channel", 1);
        mePnAmplMapG16L1_[i]->setAxisTitle("amplitude", 2);
        dqmStore_->tag(mePnAmplMapG16L1_[i], i+1);

	name = "EBLT PNs pedestal " + Numbers::sEB(i+1) + " G16 " + LN.str(); 
        mePnPedMapG16L1_[i] = dqmStore_->bookProfile(name, name, 10, 0., 10., 4096, 0., 4096., "s");
        mePnPedMapG16L1_[i]->setAxisTitle("channel", 1);
        mePnPedMapG16L1_[i]->setAxisTitle("pedestal", 2);
        dqmStore_->tag(mePnPedMapG16L1_[i], i+1);
      }


    }

    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 2) != laserWavelengths_.end() ) {

      LaserN.str("");
      LaserN << "Laser" << 2;
      LN.str("");
      LN << "L" << 2;

      dqmStore_->setCurrentFolder(prefixME_ + "/EBLaserTask/" + LaserN.str());
      for (int i = 0; i < 36; i++) {
        name = "EBLT shape " + Numbers::sEB(i+1) + " " + LN.str();
        meShapeMapL2_[i] = dqmStore_->bookProfile2D(name, name, 1700, 0., 1700., 10, 0., 10., 4096, 0., 4096., "s");
        meShapeMapL2_[i]->setAxisTitle("channel", 1);
        meShapeMapL2_[i]->setAxisTitle("sample", 2);
        meShapeMapL2_[i]->setAxisTitle("amplitude", 3);
        dqmStore_->tag(meShapeMapL2_[i], i+1);
        name = "EBLT amplitude " + Numbers::sEB(i+1) + " " + LN.str();
        meAmplMapL2_[i] = dqmStore_->bookProfile2D(name, name, 85, 0., 85., 20, 0., 20., 4096, 0., 4096.*12., "s");
        meAmplMapL2_[i]->setAxisTitle("ieta", 1);
        meAmplMapL2_[i]->setAxisTitle("iphi", 2);
        dqmStore_->tag(meAmplMapL2_[i], i+1);
        name = "EBLT timing " + Numbers::sEB(i+1) + " " + LN.str();
        meTimeMapL2_[i] = dqmStore_->bookProfile2D(name, name, 85, 0., 85., 20, 0., 20., 250, 0., 10., "s");
        meTimeMapL2_[i]->setAxisTitle("ieta", 1);
        meTimeMapL2_[i]->setAxisTitle("iphi", 2);
        dqmStore_->tag(meTimeMapL2_[i], i+1);
        name = "EBLT amplitude over PN " + Numbers::sEB(i+1) + " " + LN.str();
        meAmplPNMapL2_[i] = dqmStore_->bookProfile2D(name, name, 85, 0., 85., 20, 0., 20., 4096, 0., 4096.*12., "s");
        meAmplPNMapL2_[i]->setAxisTitle("ieta", 1);
        meAmplPNMapL2_[i]->setAxisTitle("iphi", 2);
        dqmStore_->tag(meAmplPNMapL2_[i], i+1);
      }

      name = "EBLT amplitude map " + LN.str();
      meAmplSummaryMapL2_ = dqmStore_->bookProfile2D(name, name, 72, 0., 360., 34, -85., 85., 0., 4096.);
      meAmplSummaryMapL2_->setAxisTitle("jphi", 1);
      meAmplSummaryMapL2_->setAxisTitle("jeta", 2);

      dqmStore_->setCurrentFolder(prefixME_ + "/EBLaserTask/" + LaserN.str() + "/PN");

      dqmStore_->setCurrentFolder(prefixME_ + "/EBLaserTask/" + LaserN.str() + "/PN/Gain01");
      for (int i = 0; i < 36; i++) {
        name = "EBLT PNs amplitude " + Numbers::sEB(i+1) + " G01 " + LN.str(); 
        mePnAmplMapG01L2_[i] = dqmStore_->bookProfile(name, name, 10, 0., 10., 4096, 0., 4096., "s");
        mePnAmplMapG01L2_[i]->setAxisTitle("channel", 1);
        mePnAmplMapG01L2_[i]->setAxisTitle("amplitude", 2);
        dqmStore_->tag(mePnAmplMapG01L2_[i], i+1);

	name = "EBLT PNs pedestal " + Numbers::sEB(i+1) + " G01 " + LN.str(); 
        mePnPedMapG01L2_[i] = dqmStore_->bookProfile(name, name, 10, 0., 10., 4096, 0., 4096., "s");
        mePnPedMapG01L2_[i]->setAxisTitle("channel", 1);
        mePnPedMapG01L2_[i]->setAxisTitle("pedestal", 2);
        dqmStore_->tag(mePnPedMapG01L2_[i], i+1);
      }

      dqmStore_->setCurrentFolder(prefixME_ + "/EBLaserTask/" + LaserN.str() + "/PN/Gain16");
      for (int i = 0; i < 36; i++) {
	name = "EBLT PNs amplitude " + Numbers::sEB(i+1) + " G16 " + LN.str(), 
        mePnAmplMapG16L2_[i] = dqmStore_->bookProfile(name, name, 10, 0., 10., 4096, 0., 4096., "s");
        mePnAmplMapG16L2_[i]->setAxisTitle("channel", 1);
        mePnAmplMapG16L2_[i]->setAxisTitle("amplitude", 2);
        dqmStore_->tag(mePnAmplMapG16L2_[i], i+1);

	name = "EBLT PNs pedestal " + Numbers::sEB(i+1) + " G16 " + LN.str(); 
        mePnPedMapG16L2_[i] = dqmStore_->bookProfile(name, name, 10, 0., 10., 4096, 0., 4096., "s");
        mePnPedMapG16L2_[i]->setAxisTitle("channel", 1);
        mePnPedMapG16L2_[i]->setAxisTitle("pedestal", 2);
        dqmStore_->tag(mePnPedMapG16L2_[i], i+1);
      }


    }

    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 3) != laserWavelengths_.end() ) {

      LaserN.str("");
      LaserN << "Laser" << 3;
      LN.str("");
      LN << "L" << 3;

      dqmStore_->setCurrentFolder(prefixME_ + "/EBLaserTask/" + LaserN.str());
      for (int i = 0; i < 36; i++) {
        name = "EBLT shape " + Numbers::sEB(i+1) + " " + LN.str();
        meShapeMapL3_[i] = dqmStore_->bookProfile2D(name, name, 1700, 0., 1700., 10, 0., 10., 4096, 0., 4096., "s");
        meShapeMapL3_[i]->setAxisTitle("channel", 1);
        meShapeMapL3_[i]->setAxisTitle("sample", 2);
        meShapeMapL3_[i]->setAxisTitle("amplitude", 3);
        dqmStore_->tag(meShapeMapL3_[i], i+1);
        name = "EBLT amplitude " + Numbers::sEB(i+1) + " " + LN.str();
        meAmplMapL3_[i] = dqmStore_->bookProfile2D(name, name, 85, 0., 85., 20, 0., 20., 4096, 0., 4096.*12., "s");
        meAmplMapL3_[i]->setAxisTitle("ieta", 1);
        meAmplMapL3_[i]->setAxisTitle("iphi", 2);
        dqmStore_->tag(meAmplMapL3_[i], i+1);
        name = "EBLT timing " + Numbers::sEB(i+1) + " " + LN.str();
        meTimeMapL3_[i] = dqmStore_->bookProfile2D(name, name, 85, 0., 85., 20, 0., 20., 250, 0., 10., "s");
        meTimeMapL3_[i]->setAxisTitle("ieta", 1);
        meTimeMapL3_[i]->setAxisTitle("iphi", 2);
        dqmStore_->tag(meTimeMapL3_[i], i+1);
        name = "EBLT amplitude over PN " + Numbers::sEB(i+1) + " " + LN.str();
        meAmplPNMapL3_[i] = dqmStore_->bookProfile2D(name, name, 85, 0., 85., 20, 0., 20., 4096, 0., 4096.*12., "s");
        meAmplPNMapL3_[i]->setAxisTitle("ieta", 1);
        meAmplPNMapL3_[i]->setAxisTitle("iphi", 2);
        dqmStore_->tag(meAmplPNMapL3_[i], i+1);
      }

      name = "EBLT amplitude map " + LN.str();
      meAmplSummaryMapL3_ = dqmStore_->bookProfile2D(name, name, 72, 0., 360., 34, -85., 85., 0., 4096.);
      meAmplSummaryMapL3_->setAxisTitle("jphi", 1);
      meAmplSummaryMapL3_->setAxisTitle("jeta", 2);

      dqmStore_->setCurrentFolder(prefixME_ + "/EBLaserTask/" + LaserN.str() + "/PN");

      dqmStore_->setCurrentFolder(prefixME_ + "/EBLaserTask/" + LaserN.str() + "/PN/Gain01");
      for (int i = 0; i < 36; i++) {
        name = "EBLT PNs amplitude " + Numbers::sEB(i+1) + " G01 " + LN.str(); 
        mePnAmplMapG01L3_[i] = dqmStore_->bookProfile(name, name, 10, 0., 10., 4096, 0., 4096., "s");
        mePnAmplMapG01L3_[i]->setAxisTitle("channel", 1);
        mePnAmplMapG01L3_[i]->setAxisTitle("amplitude", 2);
        dqmStore_->tag(mePnAmplMapG01L3_[i], i+1);

	name = "EBLT PNs pedestal " + Numbers::sEB(i+1) + " G01 " + LN.str(); 
        mePnPedMapG01L3_[i] = dqmStore_->bookProfile(name, name, 10, 0., 10., 4096, 0., 4096., "s");
        mePnPedMapG01L3_[i]->setAxisTitle("channel", 1);
        mePnPedMapG01L3_[i]->setAxisTitle("pedestal", 2);
        dqmStore_->tag(mePnPedMapG01L3_[i], i+1);
      }

      dqmStore_->setCurrentFolder(prefixME_ + "/EBLaserTask/" + LaserN.str() + "/PN/Gain16");
      for (int i = 0; i < 36; i++) {
	name = "EBLT PNs amplitude " + Numbers::sEB(i+1) + " G16 " + LN.str(), 
        mePnAmplMapG16L3_[i] = dqmStore_->bookProfile(name, name, 10, 0., 10., 4096, 0., 4096., "s");
        mePnAmplMapG16L3_[i]->setAxisTitle("channel", 1);
        mePnAmplMapG16L3_[i]->setAxisTitle("amplitude", 2);
        dqmStore_->tag(mePnAmplMapG16L3_[i], i+1);

	name = "EBLT PNs pedestal " + Numbers::sEB(i+1) + " G16 " + LN.str(); 
        mePnPedMapG16L3_[i] = dqmStore_->bookProfile(name, name, 10, 0., 10., 4096, 0., 4096., "s");
        mePnPedMapG16L3_[i]->setAxisTitle("channel", 1);
        mePnPedMapG16L3_[i]->setAxisTitle("pedestal", 2);
        dqmStore_->tag(mePnPedMapG16L3_[i], i+1);
      }

    }

    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 4) != laserWavelengths_.end() ) {

      LaserN.str("");
      LaserN << "Laser" << 4;
      LN.str("");
      LN << "L" << 4;

      dqmStore_->setCurrentFolder(prefixME_ + "/EBLaserTask/" + LaserN.str());
      for (int i = 0; i < 36; i++) {
        name = "EBLT shape " + Numbers::sEB(i+1) + " " + LN.str();
        meShapeMapL4_[i] = dqmStore_->bookProfile2D(name, name, 1700, 0., 1700., 10, 0., 10., 4096, 0., 4096., "s");
        meShapeMapL4_[i]->setAxisTitle("channel", 1);
        meShapeMapL4_[i]->setAxisTitle("sample", 2);
        meShapeMapL4_[i]->setAxisTitle("amplitude", 3);
        dqmStore_->tag(meShapeMapL4_[i], i+1);
        name = "EBLT amplitude " + Numbers::sEB(i+1) + " " + LN.str();
        meAmplMapL4_[i] = dqmStore_->bookProfile2D(name, name, 85, 0., 85., 20, 0., 20., 4096, 0., 4096.*12., "s");
        meAmplMapL4_[i]->setAxisTitle("ieta", 1);
        meAmplMapL4_[i]->setAxisTitle("iphi", 2);
        dqmStore_->tag(meAmplMapL4_[i], i+1);
        name = "EBLT timing " + Numbers::sEB(i+1) + " " + LN.str();
        meTimeMapL4_[i] = dqmStore_->bookProfile2D(name, name, 85, 0., 85., 20, 0., 20., 250, 0., 10., "s");
        meTimeMapL4_[i]->setAxisTitle("ieta", 1);
        meTimeMapL4_[i]->setAxisTitle("iphi", 2);
        dqmStore_->tag(meTimeMapL4_[i], i+1);
        name = "EBLT amplitude over PN " + Numbers::sEB(i+1) + " " + LN.str();
        meAmplPNMapL4_[i] = dqmStore_->bookProfile2D(name, name, 85, 0., 85., 20, 0., 20., 4096, 0., 4096.*12., "s");
        meAmplPNMapL4_[i]->setAxisTitle("ieta", 1);
        meAmplPNMapL4_[i]->setAxisTitle("iphi", 2);
        dqmStore_->tag(meAmplPNMapL4_[i], i+1);
      }

      name = "EBLT amplitude map " + LN.str();
      meAmplSummaryMapL4_ = dqmStore_->bookProfile2D(name, name, 72, 0., 360., 34, -85., 85., 0., 4096.);
      meAmplSummaryMapL4_->setAxisTitle("jphi", 1);
      meAmplSummaryMapL4_->setAxisTitle("jeta", 2);

      dqmStore_->setCurrentFolder(prefixME_ + "/EBLaserTask/" + LaserN.str() + "/PN");

      dqmStore_->setCurrentFolder(prefixME_ + "/EBLaserTask/" + LaserN.str() + "/PN/Gain01");
      for (int i = 0; i < 36; i++) {
        name = "EBLT PNs amplitude " + Numbers::sEB(i+1) + " G01 " + LN.str(); 
        mePnAmplMapG01L4_[i] = dqmStore_->bookProfile(name, name, 10, 0., 10., 4096, 0., 4096., "s");
        mePnAmplMapG01L4_[i]->setAxisTitle("channel", 1);
        mePnAmplMapG01L4_[i]->setAxisTitle("amplitude", 2);
        dqmStore_->tag(mePnAmplMapG01L4_[i], i+1);

	name = "EBLT PNs pedestal " + Numbers::sEB(i+1) + " G01 " + LN.str(); 
        mePnPedMapG01L4_[i] = dqmStore_->bookProfile(name, name, 10, 0., 10., 4096, 0., 4096., "s");
        mePnPedMapG01L4_[i]->setAxisTitle("channel", 1);
        mePnPedMapG01L4_[i]->setAxisTitle("pedestal", 2);
        dqmStore_->tag(mePnPedMapG01L4_[i], i+1);
      }

      dqmStore_->setCurrentFolder(prefixME_ + "/EBLaserTask/" + LaserN.str() + "/PN/Gain16");
      for (int i = 0; i < 36; i++) {
	name = "EBLT PNs amplitude " + Numbers::sEB(i+1) + " G16 " + LN.str(), 
        mePnAmplMapG16L4_[i] = dqmStore_->bookProfile(name, name, 10, 0., 10., 4096, 0., 4096., "s");
        mePnAmplMapG16L4_[i]->setAxisTitle("channel", 1);
        mePnAmplMapG16L4_[i]->setAxisTitle("amplitude", 2);
        dqmStore_->tag(mePnAmplMapG16L4_[i], i+1);

	name = "EBLT PNs pedestal " + Numbers::sEB(i+1) + " G16 " + LN.str(); 
        mePnPedMapG16L4_[i] = dqmStore_->bookProfile(name, name, 10, 0., 10., 4096, 0., 4096., "s");
        mePnPedMapG16L4_[i]->setAxisTitle("channel", 1);
        mePnPedMapG16L4_[i]->setAxisTitle("pedestal", 2);
        dqmStore_->tag(mePnPedMapG16L4_[i], i+1);
      }

    }

  }

}

void EBLaserTask::cleanup(void){

  if ( ! init_ ) return;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EBLaserTask");

    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 1) != laserWavelengths_.end() ) {
      dqmStore_->setCurrentFolder(prefixME_ + "/EBLaserTask/Laser1");
      for (int i = 0; i < 36; i++) {
        if ( meShapeMapL1_[i] )  dqmStore_->removeElement( meShapeMapL1_[i]->getName() );
        meShapeMapL1_[i] = 0;
        if ( meAmplMapL1_[i] ) dqmStore_->removeElement( meAmplMapL1_[i]->getName() );
        meAmplMapL1_[i] = 0;
        if ( meTimeMapL1_[i] ) dqmStore_->removeElement( meTimeMapL1_[i]->getName() );
        meTimeMapL1_[i] = 0;
        if ( meAmplPNMapL1_[i] ) dqmStore_->removeElement( meAmplPNMapL1_[i]->getName() );
        meAmplPNMapL1_[i] = 0;
      }
      if( meAmplSummaryMapL1_ ) dqmStore_->removeElement( meAmplSummaryMapL1_->getName() );
      meAmplSummaryMapL1_ = 0;
    }

    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 2) != laserWavelengths_.end() ) {
      dqmStore_->setCurrentFolder(prefixME_ + "/EBLaserTask/Laser2");
      for (int i = 0; i < 36; i++) {
        if ( meShapeMapL2_[i] )  dqmStore_->removeElement( meShapeMapL2_[i]->getName() );
        meShapeMapL2_[i] = 0;
        if ( meAmplMapL2_[i] ) dqmStore_->removeElement( meAmplMapL2_[i]->getName() );
        meAmplMapL2_[i] = 0;
        if ( meTimeMapL2_[i] ) dqmStore_->removeElement( meTimeMapL2_[i]->getName() );
        meTimeMapL2_[i] = 0;
        if ( meAmplPNMapL2_[i] ) dqmStore_->removeElement( meAmplPNMapL2_[i]->getName() );
        meAmplPNMapL2_[i] = 0;
      }
      if( meAmplSummaryMapL2_ ) dqmStore_->removeElement( meAmplSummaryMapL2_->getName() );
      meAmplSummaryMapL2_ = 0;
    }

    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 3) != laserWavelengths_.end() ) {
      dqmStore_->setCurrentFolder(prefixME_ + "/EBLaserTask/Laser3");
      for (int i = 0; i < 36; i++) {
        if ( meShapeMapL3_[i] )  dqmStore_->removeElement( meShapeMapL3_[i]->getName() );
        meShapeMapL3_[i] = 0;
        if ( meAmplMapL3_[i] ) dqmStore_->removeElement( meAmplMapL3_[i]->getName() );
        meAmplMapL3_[i] = 0;
        if ( meTimeMapL3_[i] ) dqmStore_->removeElement( meTimeMapL3_[i]->getName() );
        meTimeMapL3_[i] = 0;
        if ( meAmplPNMapL3_[i] ) dqmStore_->removeElement( meAmplPNMapL3_[i]->getName() );
        meAmplPNMapL3_[i] = 0;
      }
      if( meAmplSummaryMapL3_ ) dqmStore_->removeElement( meAmplSummaryMapL3_->getName() );
      meAmplSummaryMapL3_ = 0;
    }

    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 4) != laserWavelengths_.end() ) {
      dqmStore_->setCurrentFolder(prefixME_ + "/EBLaserTask/Laser4");
      for (int i = 0; i < 36; i++) {
        if ( meShapeMapL4_[i] )  dqmStore_->removeElement( meShapeMapL4_[i]->getName() );
        meShapeMapL4_[i] = 0;
        if ( meAmplMapL4_[i] ) dqmStore_->removeElement( meAmplMapL4_[i]->getName() );
        meAmplMapL4_[i] = 0;
        if ( meTimeMapL4_[i] ) dqmStore_->removeElement( meTimeMapL4_[i]->getName() );
        meTimeMapL4_[i] = 0;
        if ( meAmplPNMapL4_[i] ) dqmStore_->removeElement( meAmplPNMapL4_[i]->getName() );
        meAmplPNMapL4_[i] = 0;
      }
      if( meAmplSummaryMapL4_ ) dqmStore_->removeElement( meAmplSummaryMapL4_->getName() );
      meAmplSummaryMapL4_ = 0;
    }

    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 1) != laserWavelengths_.end() ) {
      dqmStore_->setCurrentFolder(prefixME_ + "/EBLaserTask/Laser1/PN");

      dqmStore_->setCurrentFolder(prefixME_ + "/EBLaserTask/Laser1/PN/Gain01");
      for (int i = 0; i < 36; i++) {
        if ( mePnAmplMapG01L1_[i] ) dqmStore_->removeElement( mePnAmplMapG01L1_[i]->getName() );
        mePnAmplMapG01L1_[i] = 0;
        if ( mePnPedMapG01L1_[i] ) dqmStore_->removeElement( mePnPedMapG01L1_[i]->getName() );
        mePnPedMapG01L1_[i] = 0;
      }

      dqmStore_->setCurrentFolder(prefixME_ + "/EBLaserTask/Laser1/PN/Gain16");
      for (int i = 0; i < 36; i++) {
        if ( mePnAmplMapG16L1_[i] ) dqmStore_->removeElement( mePnAmplMapG16L1_[i]->getName() );
        mePnAmplMapG16L1_[i] = 0;
        if ( mePnPedMapG16L1_[i] ) dqmStore_->removeElement( mePnPedMapG16L1_[i]->getName() );
        mePnPedMapG16L1_[i] = 0;
      }
    }

    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 2) != laserWavelengths_.end() ) {
      dqmStore_->setCurrentFolder(prefixME_ + "/EBLaserTask/Laser2/PN");

      dqmStore_->setCurrentFolder(prefixME_ + "/EBLaserTask/Laser2/PN/Gain01");
      for (int i = 0; i < 36; i++) {
        if ( mePnAmplMapG01L2_[i] ) dqmStore_->removeElement( mePnAmplMapG01L2_[i]->getName() );
        mePnAmplMapG01L2_[i] = 0;
        if ( mePnPedMapG01L2_[i] ) dqmStore_->removeElement( mePnPedMapG01L2_[i]->getName() );
        mePnPedMapG01L2_[i] = 0;
      }

      dqmStore_->setCurrentFolder(prefixME_ + "/EBLaserTask/Laser2/PN/Gain16");
      for (int i = 0; i < 36; i++) {
        if ( mePnAmplMapG16L2_[i] ) dqmStore_->removeElement( mePnAmplMapG16L2_[i]->getName() );
        mePnAmplMapG16L2_[i] = 0;
        if ( mePnPedMapG16L2_[i] ) dqmStore_->removeElement( mePnPedMapG16L2_[i]->getName() );
        mePnPedMapG16L2_[i] = 0;
      }
    }

    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 3) != laserWavelengths_.end() ) {
      dqmStore_->setCurrentFolder(prefixME_ + "/EBLaserTask/Laser3/PN");

      dqmStore_->setCurrentFolder(prefixME_ + "/EBLaserTask/Laser3/PN/Gain01");
      for (int i = 0; i < 36; i++) {
        if ( mePnAmplMapG01L3_[i] ) dqmStore_->removeElement( mePnAmplMapG01L3_[i]->getName() );
        mePnAmplMapG01L3_[i] = 0;
        if ( mePnPedMapG01L3_[i] ) dqmStore_->removeElement( mePnPedMapG01L3_[i]->getName() );
        mePnPedMapG01L3_[i] = 0;
      }

      dqmStore_->setCurrentFolder(prefixME_ + "/EBLaserTask/Laser3/PN/Gain16");
      for (int i = 0; i < 36; i++) {
        if ( mePnAmplMapG16L3_[i] ) dqmStore_->removeElement( mePnAmplMapG16L3_[i]->getName() );
        mePnAmplMapG16L3_[i] = 0;
        if ( mePnPedMapG16L3_[i] ) dqmStore_->removeElement( mePnPedMapG16L3_[i]->getName() );
        mePnPedMapG16L3_[i] = 0;
      }
    }

    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 4) != laserWavelengths_.end() ) {
      dqmStore_->setCurrentFolder(prefixME_ + "/EBLaserTask/Laser4/PN");

      dqmStore_->setCurrentFolder(prefixME_ + "/EBLaserTask/Laser4/PN/Gain01");
      for (int i = 0; i < 36; i++) {
        if ( mePnAmplMapG01L4_[i] ) dqmStore_->removeElement( mePnAmplMapG01L4_[i]->getName() );
        mePnAmplMapG01L4_[i] = 0;
        if ( mePnPedMapG01L4_[i] ) dqmStore_->removeElement( mePnPedMapG01L4_[i]->getName() );
        mePnPedMapG01L4_[i] = 0;
      }

      dqmStore_->setCurrentFolder(prefixME_ + "/EBLaserTask/Laser4/PN/Gain16");
      for (int i = 0; i < 36; i++) {
        if ( mePnAmplMapG16L4_[i] ) dqmStore_->removeElement( mePnAmplMapG16L4_[i]->getName() );
        mePnAmplMapG16L4_[i] = 0;
        if ( mePnPedMapG16L4_[i] ) dqmStore_->removeElement( mePnPedMapG16L4_[i]->getName() );
        mePnPedMapG16L4_[i] = 0;
      }

    }

  }

  init_ = false;

}

void EBLaserTask::endJob(void){

  edm::LogInfo("EBLaserTask") << "analyzed " << ievt_ << " events";

  if ( enableCleanup_ ) this->cleanup();

}

void EBLaserTask::analyze(const edm::Event& e, const edm::EventSetup& c){

  bool enable = false;
  int runType[36];
  for (int i=0; i<36; i++) runType[i] = -1;
  unsigned rtHalf[36];
  for (int i=0; i<36; i++) rtHalf[i] = -1;
  int waveLength[36];
  for (int i=0; i<36; i++) waveLength[i] = -1;

  edm::Handle<EcalRawDataCollection> dcchs;

  if ( e.getByLabel(EcalRawDataCollection_, dcchs) ) {

    for ( EcalRawDataCollection::const_iterator dcchItr = dcchs->begin(); dcchItr != dcchs->end(); ++dcchItr ) {

      if ( Numbers::subDet( *dcchItr ) != EcalBarrel ) continue;

      int ism = Numbers::iSM( *dcchItr, EcalBarrel );

      runType[ism-1] = dcchItr->getRunType();
      rtHalf[ism-1] = dcchItr->getRtHalf();
      waveLength[ism-1] = dcchItr->getEventSettings().wavelength;

      if ( dcchItr->getRunType() == EcalDCCHeaderBlock::LASER_STD ||
           dcchItr->getRunType() == EcalDCCHeaderBlock::LASER_GAP ) enable = true;

    }

  } else {

    edm::LogWarning("EBLaserTask") << EcalRawDataCollection_ << " not available";

  }

  if ( ! enable ) return;

  if ( ! init_ ) this->setup();

  ievt_++;

  edm::Handle<EBDigiCollection> digis;

  if ( e.getByLabel(EBDigiCollection_, digis) ) {

    int maxpos[10];
    for(int i(0); i < 10; i++)
      maxpos[i] = 0;

    int nReadouts(0);

    for ( EBDigiCollection::const_iterator digiItr = digis->begin(); digiItr != digis->end(); ++digiItr ) {

      EBDetId id = digiItr->id();

      int ism = Numbers::iSM( id );

      if ( ! ( runType[ism-1] == EcalDCCHeaderBlock::LASER_STD ||
               runType[ism-1] == EcalDCCHeaderBlock::LASER_GAP ) ) continue;

      if ( rtHalf[ism-1] != Numbers::RtHalf(id) ) continue;

      nReadouts++;

      EBDataFrame dataframe = (*digiItr);

      int iMax(-1);
      float max(0.);
      float min(4096.);
      for (int i = 0; i < 10; i++) {
        int adc = dataframe.sample(i).adc();
	if(adc > max){
	  max = adc;
	  iMax = i;
	}
	if(adc < min)
	  min = adc;
      }
      if(iMax >= 0 && max - min > 20.)
	maxpos[iMax] += 1;

    }

    int threshold(nReadouts / 2);
    enable = false;
    for(int i(0); i < 10; i++){
      if(maxpos[i] > threshold){
	enable = true;
	break;
      }
    }

    if(!enable) return;

    int nebd = digis->size();
    LogDebug("EBLaserTask") << "event " << ievt_ << " digi collection size " << nebd;

    for ( EBDigiCollection::const_iterator digiItr = digis->begin(); digiItr != digis->end(); ++digiItr ) {

      EBDetId id = digiItr->id();

      int ic = id.ic();

      int ism = Numbers::iSM( id );

      if ( ! ( runType[ism-1] == EcalDCCHeaderBlock::LASER_STD ||
               runType[ism-1] == EcalDCCHeaderBlock::LASER_GAP ) ) continue;

      if ( rtHalf[ism-1] != Numbers::RtHalf(id) ) continue;

      EBDataFrame dataframe = (*digiItr);

      for (int i = 0; i < 10; i++) {

        int adc = dataframe.sample(i).adc();

        MonitorElement* meShapeMap = 0;

        if ( rtHalf[ism-1] == 0 || rtHalf[ism-1] == 1 ) {

          if ( waveLength[ism-1] == 0 ) meShapeMap = meShapeMapL1_[ism-1];
          if ( waveLength[ism-1] == 1 ) meShapeMap = meShapeMapL2_[ism-1];
          if ( waveLength[ism-1] == 2 ) meShapeMap = meShapeMapL3_[ism-1];
          if ( waveLength[ism-1] == 3 ) meShapeMap = meShapeMapL4_[ism-1];

        } else {

          edm::LogWarning("EBLaserTask") << " RtHalf = " << rtHalf[ism-1];

        }

        float xval = float(adc);

        if ( meShapeMap ) meShapeMap->Fill(ic - 0.5, i + 0.5, xval);

      }

    }

  } else {

    edm::LogWarning("EBLaserTask") << EBDigiCollection_ << " not available";

  }

  float adcA[36];
  float adcB[36];

  for ( int i = 0; i < 36; i++ ) {
    adcA[i] = 0.;
    adcB[i] = 0.;
  }

  edm::Handle<EcalPnDiodeDigiCollection> pns;

  if ( e.getByLabel(EcalPnDiodeDigiCollection_, pns) ) {

    int nep = pns->size();
    LogDebug("EBLaserTask") << "event " << ievt_ << " pns collection size " << nep;

    for ( EcalPnDiodeDigiCollection::const_iterator pnItr = pns->begin(); pnItr != pns->end(); ++pnItr ) {

      if ( Numbers::subDet( pnItr->id() ) != EcalBarrel ) continue;

      int ism = Numbers::iSM( pnItr->id() );

      int num = pnItr->id().iPnId();

      if ( ! ( runType[ism-1] == EcalDCCHeaderBlock::LASER_STD ||
               runType[ism-1] == EcalDCCHeaderBlock::LASER_GAP ) ) continue;

      float xvalped = 0.;

      for (int i = 0; i < 4; i++) {

        int adc = pnItr->sample(i).adc();

        MonitorElement* mePNPed = 0;

        if ( pnItr->sample(i).gainId() == 0 ) {
          if ( waveLength[ism-1] == 0 ) mePNPed = mePnPedMapG01L1_[ism-1];
          if ( waveLength[ism-1] == 1 ) mePNPed = mePnPedMapG01L2_[ism-1];
          if ( waveLength[ism-1] == 2 ) mePNPed = mePnPedMapG01L3_[ism-1];
          if ( waveLength[ism-1] == 3 ) mePNPed = mePnPedMapG01L4_[ism-1];
        }
        if ( pnItr->sample(i).gainId() == 1 ) {
          if ( waveLength[ism-1] == 0 ) mePNPed = mePnPedMapG16L1_[ism-1];
          if ( waveLength[ism-1] == 1 ) mePNPed = mePnPedMapG16L2_[ism-1];
          if ( waveLength[ism-1] == 2 ) mePNPed = mePnPedMapG16L3_[ism-1];
          if ( waveLength[ism-1] == 3 ) mePNPed = mePnPedMapG16L4_[ism-1];
        }

        float xval = float(adc);

        if ( mePNPed ) mePNPed->Fill(num - 0.5, xval);

        xvalped = xvalped + xval;

      }

      xvalped = xvalped / 4;

      float xvalmax = 0.;

      MonitorElement* mePN = 0;

      for (int i = 0; i < 50; i++) {

        int adc = pnItr->sample(i).adc();

        float xval = float(adc);

        if ( xval >= xvalmax ) xvalmax = xval;

      }

      xvalmax = xvalmax - xvalped;

      if ( pnItr->sample(0).gainId() == 0 ) {
        if ( waveLength[ism-1] == 0 ) mePN = mePnAmplMapG01L1_[ism-1];
        if ( waveLength[ism-1] == 1 ) mePN = mePnAmplMapG01L2_[ism-1];
        if ( waveLength[ism-1] == 2 ) mePN = mePnAmplMapG01L3_[ism-1];
        if ( waveLength[ism-1] == 3 ) mePN = mePnAmplMapG01L4_[ism-1];
      }
      if ( pnItr->sample(0).gainId() == 1 ) {
        if ( waveLength[ism-1] == 0 ) mePN = mePnAmplMapG16L1_[ism-1];
        if ( waveLength[ism-1] == 1 ) mePN = mePnAmplMapG16L2_[ism-1];
        if ( waveLength[ism-1] == 2 ) mePN = mePnAmplMapG16L3_[ism-1];
        if ( waveLength[ism-1] == 3 ) mePN = mePnAmplMapG16L4_[ism-1];
      }

      if ( mePN ) mePN->Fill(num - 0.5, xvalmax);

      if ( num == 1 ) adcA[ism-1] = xvalmax;
      if ( num == 6 ) adcB[ism-1] = xvalmax;

    }

  } else {

    edm::LogWarning("EBLaserTask") << EcalPnDiodeDigiCollection_ << " not available";

  }

  edm::Handle<EcalUncalibratedRecHitCollection> hits;

  if ( e.getByLabel(EcalUncalibratedRecHitCollection_, hits) ) {

    int neh = hits->size();
    LogDebug("EBLaserTask") << "event " << ievt_ << " hits collection size " << neh;

    for ( EcalUncalibratedRecHitCollection::const_iterator hitItr = hits->begin(); hitItr != hits->end(); ++hitItr ) {

      EBDetId id = hitItr->id();

      int ic = id.ic();
      int ie = (ic-1)/20 + 1;
      int ip = (ic-1)%20 + 1;

      int ism = Numbers::iSM( id );

      float xie = ie - 0.5;
      float xip = ip - 0.5;

      if ( ! ( runType[ism-1] == EcalDCCHeaderBlock::LASER_STD ||
               runType[ism-1] == EcalDCCHeaderBlock::LASER_GAP ) ) continue;

      if ( rtHalf[ism-1] != Numbers::RtHalf(id) ) continue;

      MonitorElement* meAmplMap = 0;
      MonitorElement* meTimeMap = 0;
      MonitorElement* meAmplPNMap = 0;
      MonitorElement* meAmplSummaryMap = 0;

      if ( rtHalf[ism-1] == 0 || rtHalf[ism-1] == 1 ) {

        if ( waveLength[ism-1] == 0 ) {
          meAmplMap = meAmplMapL1_[ism-1];
          meTimeMap = meTimeMapL1_[ism-1];
          meAmplPNMap = meAmplPNMapL1_[ism-1];
	  meAmplSummaryMap = meAmplSummaryMapL1_;
        }
        if ( waveLength[ism-1] == 1 ) {
          meAmplMap = meAmplMapL2_[ism-1];
          meTimeMap = meTimeMapL2_[ism-1];
          meAmplPNMap = meAmplPNMapL2_[ism-1];
	  meAmplSummaryMap = meAmplSummaryMapL2_;
        }
        if ( waveLength[ism-1] == 2 ) {
          meAmplMap = meAmplMapL3_[ism-1];
          meTimeMap = meTimeMapL3_[ism-1];
          meAmplPNMap = meAmplPNMapL3_[ism-1];
	  meAmplSummaryMap = meAmplSummaryMapL3_;
        }
        if ( waveLength[ism-1] == 3 ) {
          meAmplMap = meAmplMapL4_[ism-1];
          meTimeMap = meTimeMapL4_[ism-1];
          meAmplPNMap = meAmplPNMapL4_[ism-1];
	  meAmplSummaryMap = meAmplSummaryMapL4_;
        }

      } else {

        edm::LogWarning("EBLaserTask") << " RtHalf = " << rtHalf[ism-1];

      }

      float xval = hitItr->amplitude();
      if ( xval <= 0. ) xval = 0.0;
      float yval = hitItr->jitter() + 5.0;
      if ( yval <= 0. ) yval = 0.0;
      float zval = hitItr->pedestal();
      if ( zval <= 0. ) zval = 0.0;

      if ( meAmplMap ) meAmplMap->Fill(xie, xip, xval);

      if ( xval > 12. ) {
        if ( meTimeMap ) meTimeMap->Fill(xie, xip, yval);
      }

      float wval = 0.;

      if ( rtHalf[ism-1] == 0 ) {

        if ( adcA[ism-1] != 0. ) wval = xval / adcA[ism-1];

      } else if ( rtHalf[ism-1] == 1 ) {

        if ( adcB[ism-1] != 0. ) wval = xval / adcB[ism-1];

      } else {

        edm::LogWarning("EBLaserTask") << " RtHalf = " << rtHalf[ism-1];

      }

      if ( meAmplPNMap ) meAmplPNMap->Fill(xie, xip, wval);

      float xjp = id.iphi() - 0.5;
      float xje = id.ieta() - 0.5 * id.zside();

      if( meAmplSummaryMap ) meAmplSummaryMap->Fill(xjp, xje, xval);

    }

  } else {

    edm::LogWarning("EBLaserTask") << EcalUncalibratedRecHitCollection_ << " not available";

  }

}

