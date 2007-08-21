/*
 * \file EELaserTask.cc
 *
 * $Date: 2007/08/16 14:26:08 $
 * $Revision: 1.17 $
 * \author G. Della Ricca
 *
*/

#include <iostream>
#include <fstream>
#include <vector>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"

#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include <DQM/EcalCommon/interface/Numbers.h>

#include <DQM/EcalEndcapMonitorTasks/interface/EELaserTask.h>

using namespace cms;
using namespace edm;
using namespace std;

EELaserTask::EELaserTask(const ParameterSet& ps){

  init_ = false;

  // get hold of back-end interface
  dbe_ = Service<DaqMonitorBEInterface>().operator->();

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", true);

  EcalRawDataCollection_ = ps.getParameter<edm::InputTag>("EcalRawDataCollection");
  EEDigiCollection_ = ps.getParameter<edm::InputTag>("EEDigiCollection");
  EcalPnDiodeDigiCollection_ = ps.getParameter<edm::InputTag>("EcalPnDiodeDigiCollection");
  EcalUncalibratedRecHitCollection_ = ps.getParameter<edm::InputTag>("EcalUncalibratedRecHitCollection");

  for (int i = 0; i < 18 ; i++) {
    meShapeMapL1A_[i] = 0;
    meAmplMapL1A_[i] = 0;
    meTimeMapL1A_[i] = 0;
    meAmplPNMapL1A_[i] = 0;
    meShapeMapL1B_[i] = 0;
    meAmplMapL1B_[i] = 0;
    meTimeMapL1B_[i] = 0;
    meAmplPNMapL1B_[i] = 0;
    mePnAmplMapG01L1_[i] = 0;
    mePnPedMapG01L1_[i] = 0;
    mePnAmplMapG16L1_[i] = 0;
    mePnPedMapG16L1_[i] = 0;

    meShapeMapL2A_[i] = 0;
    meAmplMapL2A_[i] = 0;
    meTimeMapL2A_[i] = 0;
    meAmplPNMapL2A_[i] = 0;
    meShapeMapL2B_[i] = 0;
    meAmplMapL2B_[i] = 0;
    meTimeMapL2B_[i] = 0;
    meAmplPNMapL2B_[i] = 0;
    mePnAmplMapG01L2_[i] = 0;
    mePnPedMapG01L2_[i] = 0;
    mePnAmplMapG16L2_[i] = 0;
    mePnPedMapG16L2_[i] = 0;

    meShapeMapL3A_[i] = 0;
    meAmplMapL3A_[i] = 0;
    meTimeMapL3A_[i] = 0;
    meAmplPNMapL3A_[i] = 0;
    meShapeMapL3B_[i] = 0;
    meAmplMapL3B_[i] = 0;
    meTimeMapL3B_[i] = 0;
    meAmplPNMapL3B_[i] = 0;
    mePnAmplMapG01L3_[i] = 0;
    mePnPedMapG01L3_[i] = 0;
    mePnAmplMapG16L3_[i] = 0;
    mePnPedMapG16L3_[i] = 0;

    meShapeMapL4A_[i] = 0;
    meAmplMapL4A_[i] = 0;
    meTimeMapL4A_[i] = 0;
    meAmplPNMapL4A_[i] = 0;
    meShapeMapL4B_[i] = 0;
    meAmplMapL4B_[i] = 0;
    meTimeMapL4B_[i] = 0;
    meAmplPNMapL4B_[i] = 0;
    mePnAmplMapG01L4_[i] = 0;
    mePnPedMapG01L4_[i] = 0;
    mePnAmplMapG16L4_[i] = 0;
    mePnPedMapG16L4_[i] = 0;
  }

}

EELaserTask::~EELaserTask(){

}

void EELaserTask::beginJob(const EventSetup& c){

  ievt_ = 0;

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalEndcap/EELaserTask");
    dbe_->rmdir("EcalEndcap/EELaserTask");
  }

}

void EELaserTask::setup(void){

  init_ = true;

  Char_t histo[200];

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalEndcap/EELaserTask");

    dbe_->setCurrentFolder("EcalEndcap/EELaserTask/Laser1");
    for (int i = 0; i < 18 ; i++) {
      sprintf(histo, "EELT shape %s L1A", Numbers::sEE(i+1).c_str());
      meShapeMapL1A_[i] = dbe_->bookProfile2D(histo, histo, 1700, 0., 1700., 10, 0., 10., 4096, 0., 4096., "s");
      dbe_->tag(meShapeMapL1A_[i], i+1);
      sprintf(histo, "EELT amplitude %s L1A", Numbers::sEE(i+1).c_str());
      meAmplMapL1A_[i] = dbe_->bookProfile2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 4096, 0., 4096.*12., "s");
      dbe_->tag(meAmplMapL1A_[i], i+1);
      sprintf(histo, "EELT timing %s L1A", Numbers::sEE(i+1).c_str());
      meTimeMapL1A_[i] = dbe_->bookProfile2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 250, 0., 10., "s");
      dbe_->tag(meTimeMapL1A_[i], i+1);
      sprintf(histo, "EELT amplitude over PN %s L1A", Numbers::sEE(i+1).c_str());
      meAmplPNMapL1A_[i] = dbe_->bookProfile2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 4096, 0., 4096.*12., "s");
      dbe_->tag(meAmplPNMapL1A_[i], i+1);

      sprintf(histo, "EELT shape %s L1B", Numbers::sEE(i+1).c_str());
      meShapeMapL1B_[i] = dbe_->bookProfile2D(histo, histo, 1700, 0., 1700., 10, 0., 10., 4096, 0., 4096., "s");
      dbe_->tag(meShapeMapL1B_[i], i+1);
      sprintf(histo, "EELT amplitude %s L1B", Numbers::sEE(i+1).c_str());
      meAmplMapL1B_[i] = dbe_->bookProfile2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 4096, 0., 4096.*12., "s");
      dbe_->tag(meAmplMapL1B_[i], i+1);
      sprintf(histo, "EELT timing %s L1B", Numbers::sEE(i+1).c_str());
      meTimeMapL1B_[i] = dbe_->bookProfile2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 250, 0., 10., "s");
      dbe_->tag(meTimeMapL1B_[i], i+1);
      sprintf(histo, "EELT amplitude over PN %s L1B", Numbers::sEE(i+1).c_str());
      meAmplPNMapL1B_[i] = dbe_->bookProfile2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 4096, 0., 4096.*12., "s");
      dbe_->tag(meAmplPNMapL1B_[i], i+1);
    }

    dbe_->setCurrentFolder("EcalEndcap/EELaserTask/Laser2");
    for (int i = 0; i < 18 ; i++) {
      sprintf(histo, "EELT shape %s L2A", Numbers::sEE(i+1).c_str());
      meShapeMapL2A_[i] = dbe_->bookProfile2D(histo, histo, 1700, 0., 1700., 10, 0., 10., 4096, 0., 4096., "s");
      dbe_->tag(meShapeMapL2A_[i], i+1);
      sprintf(histo, "EELT amplitude %s L2A", Numbers::sEE(i+1).c_str());
      meAmplMapL2A_[i] = dbe_->bookProfile2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 4096, 0., 4096.*12., "s");
      dbe_->tag(meAmplMapL2A_[i], i+1);
      sprintf(histo, "EELT timing %s L2A", Numbers::sEE(i+1).c_str());
      meTimeMapL2A_[i] = dbe_->bookProfile2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 250, 0., 10., "s");
      dbe_->tag(meTimeMapL2A_[i], i+1);
      sprintf(histo, "EELT amplitude over PN %s L2A", Numbers::sEE(i+1).c_str());
      meAmplPNMapL2A_[i] = dbe_->bookProfile2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 4096, 0., 4096.*12., "s");
      dbe_->tag(meAmplPNMapL2A_[i], i+1);

      sprintf(histo, "EELT shape %s L2B", Numbers::sEE(i+1).c_str());
      meShapeMapL2B_[i] = dbe_->bookProfile2D(histo, histo, 1700, 0., 1700., 10, 0., 10., 4096, 0., 4096., "s");
      dbe_->tag(meShapeMapL2B_[i], i+1);
      sprintf(histo, "EELT amplitude %s L2B", Numbers::sEE(i+1).c_str());
      meAmplMapL2B_[i] = dbe_->bookProfile2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 4096, 0., 4096.*12., "s");
      dbe_->tag(meAmplMapL2B_[i], i+1);
      sprintf(histo, "EELT timing %s L2B", Numbers::sEE(i+1).c_str());
      meTimeMapL2B_[i] = dbe_->bookProfile2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 250, 0., 10., "s");
      dbe_->tag(meTimeMapL2B_[i], i+1);
      sprintf(histo, "EELT amplitude over PN %s L2B", Numbers::sEE(i+1).c_str());
      meAmplPNMapL2B_[i] = dbe_->bookProfile2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 4096, 0., 4096.*12., "s");
      dbe_->tag(meAmplPNMapL2B_[i], i+1);
    }

    dbe_->setCurrentFolder("EcalEndcap/EELaserTask/Laser3");
    for (int i = 0; i < 18 ; i++) {
      sprintf(histo, "EELT shape %s L3A", Numbers::sEE(i+1).c_str());
      meShapeMapL3A_[i] = dbe_->bookProfile2D(histo, histo, 1700, 0., 1700., 10, 0., 10., 4096, 0., 4096., "s");
      dbe_->tag(meShapeMapL3A_[i], i+1);
      sprintf(histo, "EELT amplitude %s L3A", Numbers::sEE(i+1).c_str());
      meAmplMapL3A_[i] = dbe_->bookProfile2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 4096, 0., 4096.*12., "s");
      dbe_->tag(meAmplMapL3A_[i], i+1);
      sprintf(histo, "EELT timing %s L3A", Numbers::sEE(i+1).c_str());
      meTimeMapL3A_[i] = dbe_->bookProfile2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 250, 0., 10., "s");
      dbe_->tag(meTimeMapL3A_[i], i+1);
      sprintf(histo, "EELT amplitude over PN %s L3A", Numbers::sEE(i+1).c_str());
      meAmplPNMapL3A_[i] = dbe_->bookProfile2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 4096, 0., 4096.*12., "s");
      dbe_->tag(meAmplPNMapL3A_[i], i+1);

      sprintf(histo, "EELT shape %s L3B", Numbers::sEE(i+1).c_str());
      meShapeMapL3B_[i] = dbe_->bookProfile2D(histo, histo, 1700, 0., 1700., 10, 0., 10., 4096, 0., 4096., "s");
      dbe_->tag(meShapeMapL3B_[i], i+1);
      sprintf(histo, "EELT amplitude %s L3B", Numbers::sEE(i+1).c_str());
      meAmplMapL3B_[i] = dbe_->bookProfile2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 4096, 0., 4096.*12., "s");
      dbe_->tag(meAmplMapL3B_[i], i+1);
      sprintf(histo, "EELT timing %s L3B", Numbers::sEE(i+1).c_str());
      meTimeMapL3B_[i] = dbe_->bookProfile2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 250, 0., 10., "s");
      dbe_->tag(meTimeMapL3B_[i], i+1);
      sprintf(histo, "EELT amplitude over PN %s L3B", Numbers::sEE(i+1).c_str());
      meAmplPNMapL3B_[i] = dbe_->bookProfile2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 4096, 0., 4096.*12., "s");
      dbe_->tag(meAmplPNMapL3B_[i], i+1);
    }

    dbe_->setCurrentFolder("EcalEndcap/EELaserTask/Laser4");
    for (int i = 0; i < 18 ; i++) {
      sprintf(histo, "EELT shape %s L4A", Numbers::sEE(i+1).c_str());
      meShapeMapL4A_[i] = dbe_->bookProfile2D(histo, histo, 1700, 0., 1700., 10, 0., 10., 4096, 0., 4096., "s");
      dbe_->tag(meShapeMapL4A_[i], i+1);
      sprintf(histo, "EELT amplitude %s L4A", Numbers::sEE(i+1).c_str());
      meAmplMapL4A_[i] = dbe_->bookProfile2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 4096, 0., 4096.*12., "s");
      dbe_->tag(meAmplMapL4A_[i], i+1);
      sprintf(histo, "EELT timing %s L4A", Numbers::sEE(i+1).c_str());
      meTimeMapL4A_[i] = dbe_->bookProfile2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 250, 0., 10., "s");
      dbe_->tag(meTimeMapL4A_[i], i+1);
      sprintf(histo, "EELT amplitude over PN %s L4A", Numbers::sEE(i+1).c_str());
      meAmplPNMapL4A_[i] = dbe_->bookProfile2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 4096, 0., 4096.*12., "s");
      dbe_->tag(meAmplPNMapL4A_[i], i+1);

      sprintf(histo, "EELT shape %s L4B", Numbers::sEE(i+1).c_str());
      meShapeMapL4B_[i] = dbe_->bookProfile2D(histo, histo, 1700, 0., 1700., 10, 0., 10., 4096, 0., 4096., "s");
      dbe_->tag(meShapeMapL4B_[i], i+1);
      sprintf(histo, "EELT amplitude %s L4B", Numbers::sEE(i+1).c_str());
      meAmplMapL4B_[i] = dbe_->bookProfile2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 4096, 0., 4096.*12., "s");
      dbe_->tag(meAmplMapL4B_[i], i+1);
      sprintf(histo, "EELT timing %s L4B", Numbers::sEE(i+1).c_str());
      meTimeMapL4B_[i] = dbe_->bookProfile2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 250, 0., 10., "s");
      dbe_->tag(meTimeMapL4B_[i], i+1);
      sprintf(histo, "EELT amplitude over PN %s L4B", Numbers::sEE(i+1).c_str());
      meAmplPNMapL4B_[i] = dbe_->bookProfile2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 4096, 0., 4096.*12., "s");
      dbe_->tag(meAmplPNMapL4B_[i], i+1);
    }

    dbe_->setCurrentFolder("EcalEndcap/EELaserTask/Laser1/PN");

    dbe_->setCurrentFolder("EcalEndcap/EELaserTask/Laser1/PN/Gain01");
    for (int i = 0; i < 18 ; i++) {
      sprintf(histo, "EEPDT PNs amplitude %s G01 L1", Numbers::sEE(i+1).c_str());
      mePnAmplMapG01L1_[i] = dbe_->bookProfile2D(histo, histo, 1, 0., 1., 10, 0., 10., 4096, 0., 4096., "s");
      dbe_->tag(mePnAmplMapG01L1_[i], i+1);
      sprintf(histo, "EEPDT PNs pedestal %s G01 L1", Numbers::sEE(i+1).c_str());
      mePnPedMapG01L1_[i] = dbe_->bookProfile2D(histo, histo, 1, 0., 1., 10, 0., 10., 4096, 0., 4096., "s");
      dbe_->tag(mePnPedMapG01L1_[i], i+1);
    }

    dbe_->setCurrentFolder("EcalEndcap/EELaserTask/Laser1/PN/Gain16");
    for (int i = 0; i < 18 ; i++) {
      sprintf(histo, "EEPDT PNs amplitude %s G16 L1", Numbers::sEE(i+1).c_str());
      mePnAmplMapG16L1_[i] = dbe_->bookProfile2D(histo, histo, 1, 0., 1., 10, 0., 10., 4096, 0., 4096., "s");
      dbe_->tag(mePnAmplMapG16L1_[i], i+1);
      sprintf(histo, "EEPDT PNs pedestal %s G16 L1", Numbers::sEE(i+1).c_str());
      mePnPedMapG16L1_[i] = dbe_->bookProfile2D(histo, histo, 1, 0., 1., 10, 0., 10., 4096, 0., 4096., "s");
      dbe_->tag(mePnPedMapG16L1_[i], i+1);
    }

    dbe_->setCurrentFolder("EcalEndcap/EELaserTask/Laser2/PN");

    dbe_->setCurrentFolder("EcalEndcap/EELaserTask/Laser2/PN/Gain01");
    for (int i = 0; i < 18 ; i++) {
      sprintf(histo, "EEPDT PNs amplitude %s G01 L2", Numbers::sEE(i+1).c_str());
      mePnAmplMapG01L2_[i] = dbe_->bookProfile2D(histo, histo, 1, 0., 1., 10, 0., 10., 4096, 0., 4096., "s");
      dbe_->tag(mePnAmplMapG01L2_[i], i+1);
      sprintf(histo, "EEPDT PNs pedestal %s G01 L2", Numbers::sEE(i+1).c_str());
      mePnPedMapG01L2_[i] = dbe_->bookProfile2D(histo, histo, 1, 0., 1., 10, 0., 10., 4096, 0., 4096., "s");
      dbe_->tag(mePnPedMapG01L2_[i], i+1);
    }

    dbe_->setCurrentFolder("EcalEndcap/EELaserTask/Laser2/PN/Gain16");
    for (int i = 0; i < 18 ; i++) {
      sprintf(histo, "EEPDT PNs amplitude %s G16 L2", Numbers::sEE(i+1).c_str());
      mePnAmplMapG16L2_[i] = dbe_->bookProfile2D(histo, histo, 1, 0., 1., 10, 0., 10., 4096, 0., 4096., "s");
      dbe_->tag(mePnAmplMapG16L2_[i], i+1);
      sprintf(histo, "EEPDT PNs pedestal %s G16 L2", Numbers::sEE(i+1).c_str());
      mePnPedMapG16L2_[i] = dbe_->bookProfile2D(histo, histo, 1, 0., 1., 10, 0., 10., 4096, 0., 4096., "s");
      dbe_->tag(mePnPedMapG16L2_[i], i+1);
    }

    dbe_->setCurrentFolder("EcalEndcap/EELaserTask/Laser3/PN");

    dbe_->setCurrentFolder("EcalEndcap/EELaserTask/Laser3/PN/Gain01");
    for (int i = 0; i < 18 ; i++) {
      sprintf(histo, "EEPDT PNs amplitude %s G01 L3", Numbers::sEE(i+1).c_str());
      mePnAmplMapG01L3_[i] = dbe_->bookProfile2D(histo, histo, 1, 0., 1., 10, 0., 10., 4096, 0., 4096., "s");
      dbe_->tag(mePnAmplMapG01L3_[i], i+1);
      sprintf(histo, "EEPDT PNs pedestal %s G01 L3", Numbers::sEE(i+1).c_str());
      mePnPedMapG01L3_[i] = dbe_->bookProfile2D(histo, histo, 1, 0., 1., 10, 0., 10., 4096, 0., 4096., "s");
      dbe_->tag(mePnPedMapG01L3_[i], i+1);
    }

    dbe_->setCurrentFolder("EcalEndcap/EELaserTask/Laser3/PN/Gain16");
    for (int i = 0; i < 18 ; i++) {
      sprintf(histo, "EEPDT PNs amplitude %s G16 L3", Numbers::sEE(i+1).c_str());
      mePnAmplMapG16L3_[i] = dbe_->bookProfile2D(histo, histo, 1, 0., 1., 10, 0., 10., 4096, 0., 4096., "s");
      dbe_->tag(mePnAmplMapG16L3_[i], i+1);
      sprintf(histo, "EEPDT PNs pedestal %s G16 L3", Numbers::sEE(i+1).c_str());
      mePnPedMapG16L3_[i] = dbe_->bookProfile2D(histo, histo, 1, 0., 1., 10, 0., 10., 4096, 0., 4096., "s");
      dbe_->tag(mePnPedMapG16L3_[i], i+1);
    }

    dbe_->setCurrentFolder("EcalEndcap/EELaserTask/Laser4/PN");

    dbe_->setCurrentFolder("EcalEndcap/EELaserTask/Laser4/PN/Gain01");
    for (int i = 0; i < 18 ; i++) {
      sprintf(histo, "EEPDT PNs amplitude %s G01 L4", Numbers::sEE(i+1).c_str());
      mePnAmplMapG01L4_[i] = dbe_->bookProfile2D(histo, histo, 1, 0., 1., 10, 0., 10., 4096, 0., 4096., "s");
      dbe_->tag(mePnAmplMapG01L4_[i], i+1);
      sprintf(histo, "EEPDT PNs pedestal %s G01 L4", Numbers::sEE(i+1).c_str());
      mePnPedMapG01L4_[i] = dbe_->bookProfile2D(histo, histo, 1, 0., 1., 10, 0., 10., 4096, 0., 4096., "s");
      dbe_->tag(mePnPedMapG01L4_[i], i+1);
    }

    dbe_->setCurrentFolder("EcalEndcap/EELaserTask/Laser4/PN/Gain16");
    for (int i = 0; i < 18 ; i++) {
      sprintf(histo, "EEPDT PNs amplitude %s G16 L4", Numbers::sEE(i+1).c_str());
      mePnAmplMapG16L4_[i] = dbe_->bookProfile2D(histo, histo, 1, 0., 1., 10, 0., 10., 4096, 0., 4096., "s");
      dbe_->tag(mePnAmplMapG16L4_[i], i+1);
      sprintf(histo, "EEPDT PNs pedestal %s G16 L4", Numbers::sEE(i+1).c_str());
      mePnPedMapG16L4_[i] = dbe_->bookProfile2D(histo, histo, 1, 0., 1., 10, 0., 10., 4096, 0., 4096., "s");
      dbe_->tag(mePnPedMapG16L4_[i], i+1);
    }

  }

}

void EELaserTask::cleanup(void){

  if ( ! enableCleanup_ ) return;

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalEndcap/EELaserTask");

    dbe_->setCurrentFolder("EcalEndcap/EELaserTask/Laser1");
    for (int i = 0; i < 18 ; i++) {
      if ( meShapeMapL1A_[i] )  dbe_->removeElement( meShapeMapL1A_[i]->getName() );
      meShapeMapL1A_[i] = 0;
      if ( meAmplMapL1A_[i] ) dbe_->removeElement( meAmplMapL1A_[i]->getName() );
      meAmplMapL1A_[i] = 0;
      if ( meTimeMapL1A_[i] ) dbe_->removeElement( meTimeMapL1A_[i]->getName() );
      meTimeMapL1A_[i] = 0;
      if ( meAmplPNMapL1A_[i] ) dbe_->removeElement( meAmplPNMapL1A_[i]->getName() );
      meAmplPNMapL1A_[i] = 0;

      if ( meShapeMapL1B_[i] )  dbe_->removeElement( meShapeMapL1B_[i]->getName() );
      meShapeMapL1B_[i] = 0;
      if ( meAmplMapL1B_[i] ) dbe_->removeElement( meAmplMapL1B_[i]->getName() );
      meAmplMapL1B_[i] = 0;
      if ( meTimeMapL1B_[i] ) dbe_->removeElement( meTimeMapL1B_[i]->getName() );
      meTimeMapL1B_[i] = 0;
      if ( meAmplPNMapL1B_[i] ) dbe_->removeElement( meAmplPNMapL1B_[i]->getName() );
      meAmplPNMapL1B_[i] = 0;
    }

    dbe_->setCurrentFolder("EcalEndcap/EELaserTask/Laser2");
    for (int i = 0; i < 18 ; i++) {
      if ( meShapeMapL2A_[i] )  dbe_->removeElement( meShapeMapL2A_[i]->getName() );
      meShapeMapL2A_[i] = 0;
      if ( meAmplMapL2A_[i] ) dbe_->removeElement( meAmplMapL2A_[i]->getName() );
      meAmplMapL2A_[i] = 0;
      if ( meTimeMapL2A_[i] ) dbe_->removeElement( meTimeMapL2A_[i]->getName() );
      meTimeMapL2A_[i] = 0;
      if ( meAmplPNMapL2A_[i] ) dbe_->removeElement( meAmplPNMapL2A_[i]->getName() );
      meAmplPNMapL2A_[i] = 0;

      if ( meShapeMapL2B_[i] )  dbe_->removeElement( meShapeMapL2B_[i]->getName() );
      meShapeMapL2B_[i] = 0;
      if ( meAmplMapL2B_[i] ) dbe_->removeElement( meAmplMapL2B_[i]->getName() );
      meAmplMapL2B_[i] = 0;
      if ( meTimeMapL2B_[i] ) dbe_->removeElement( meTimeMapL2B_[i]->getName() );
      meTimeMapL2B_[i] = 0;
      if ( meAmplPNMapL2B_[i] ) dbe_->removeElement( meAmplPNMapL2B_[i]->getName() );
      meAmplPNMapL2B_[i] = 0;
    }

    dbe_->setCurrentFolder("EcalEndcap/EELaserTask/Laser3");
    for (int i = 0; i < 18 ; i++) {
      if ( meShapeMapL3A_[i] )  dbe_->removeElement( meShapeMapL3A_[i]->getName() );
      meShapeMapL3A_[i] = 0;
      if ( meAmplMapL3A_[i] ) dbe_->removeElement( meAmplMapL3A_[i]->getName() );
      meAmplMapL3A_[i] = 0;
      if ( meTimeMapL3A_[i] ) dbe_->removeElement( meTimeMapL3A_[i]->getName() );
      meTimeMapL3A_[i] = 0;
      if ( meAmplPNMapL3A_[i] ) dbe_->removeElement( meAmplPNMapL3A_[i]->getName() );
      meAmplPNMapL3A_[i] = 0;

      if ( meShapeMapL3B_[i] )  dbe_->removeElement( meShapeMapL3B_[i]->getName() );
      meShapeMapL3B_[i] = 0;
      if ( meAmplMapL3B_[i] ) dbe_->removeElement( meAmplMapL3B_[i]->getName() );
      meAmplMapL3B_[i] = 0;
      if ( meTimeMapL3B_[i] ) dbe_->removeElement( meTimeMapL3B_[i]->getName() );
      meTimeMapL3B_[i] = 0;
      if ( meAmplPNMapL3B_[i] ) dbe_->removeElement( meAmplPNMapL3B_[i]->getName() );
      meAmplPNMapL3B_[i] = 0;
    }

    dbe_->setCurrentFolder("EcalEndcap/EELaserTask/Laser4");
    for (int i = 0; i < 18 ; i++) {
      if ( meShapeMapL4A_[i] )  dbe_->removeElement( meShapeMapL4A_[i]->getName() );
      meShapeMapL4A_[i] = 0;
      if ( meAmplMapL4A_[i] ) dbe_->removeElement( meAmplMapL4A_[i]->getName() );
      meAmplMapL4A_[i] = 0;
      if ( meTimeMapL4A_[i] ) dbe_->removeElement( meTimeMapL4A_[i]->getName() );
      meTimeMapL4A_[i] = 0;
      if ( meAmplPNMapL4A_[i] ) dbe_->removeElement( meAmplPNMapL4A_[i]->getName() );
      meAmplPNMapL4A_[i] = 0;

      if ( meShapeMapL4B_[i] )  dbe_->removeElement( meShapeMapL4B_[i]->getName() );
      meShapeMapL4B_[i] = 0;
      if ( meAmplMapL4B_[i] ) dbe_->removeElement( meAmplMapL4B_[i]->getName() );
      meAmplMapL4B_[i] = 0;
      if ( meTimeMapL4B_[i] ) dbe_->removeElement( meTimeMapL4B_[i]->getName() );
      meTimeMapL4B_[i] = 0;
      if ( meAmplPNMapL4B_[i] ) dbe_->removeElement( meAmplPNMapL4B_[i]->getName() );
      meAmplPNMapL4B_[i] = 0;
    }

    dbe_->setCurrentFolder("EcalEndcap/EELaserTask/Laser1/PN");

    dbe_->setCurrentFolder("EcalEndcap/EELaserTask/Laser1/PN/Gain01");
    for (int i = 0; i < 18 ; i++) {
      if ( mePnAmplMapG01L1_[i] ) dbe_->removeElement( mePnAmplMapG01L1_[i]->getName() );
      mePnAmplMapG01L1_[i] = 0;
      if ( mePnPedMapG01L1_[i] ) dbe_->removeElement( mePnPedMapG01L1_[i]->getName() );
      mePnPedMapG01L1_[i] = 0;
    }

    dbe_->setCurrentFolder("EcalEndcap/EELaserTask/Laser1/PN/Gain16");
    for (int i = 0; i < 18 ; i++) {
      if ( mePnAmplMapG16L1_[i] ) dbe_->removeElement( mePnAmplMapG16L1_[i]->getName() );
      mePnAmplMapG16L1_[i] = 0;
      if ( mePnPedMapG16L1_[i] ) dbe_->removeElement( mePnPedMapG16L1_[i]->getName() );
      mePnPedMapG16L1_[i] = 0;
    }

    dbe_->setCurrentFolder("EcalEndcap/EELaserTask/Laser2/PN");

    dbe_->setCurrentFolder("EcalEndcap/EELaserTask/Laser2/PN/Gain01");
    for (int i = 0; i < 18 ; i++) {
      if ( mePnAmplMapG01L2_[i] ) dbe_->removeElement( mePnAmplMapG01L2_[i]->getName() );
      mePnAmplMapG01L2_[i] = 0;
      if ( mePnPedMapG01L2_[i] ) dbe_->removeElement( mePnPedMapG01L2_[i]->getName() );
      mePnPedMapG01L2_[i] = 0;
    }

    dbe_->setCurrentFolder("EcalEndcap/EELaserTask/Laser2/PN/Gain16");
    for (int i = 0; i < 18 ; i++) {
      if ( mePnAmplMapG16L2_[i] ) dbe_->removeElement( mePnAmplMapG16L2_[i]->getName() );
      mePnAmplMapG16L2_[i] = 0;
      if ( mePnPedMapG16L2_[i] ) dbe_->removeElement( mePnPedMapG16L2_[i]->getName() );
      mePnPedMapG16L2_[i] = 0;
    }

    dbe_->setCurrentFolder("EcalEndcap/EELaserTask/Laser3/PN");

    dbe_->setCurrentFolder("EcalEndcap/EELaserTask/Laser3/PN/Gain01");
    for (int i = 0; i < 18 ; i++) {
      if ( mePnAmplMapG01L3_[i] ) dbe_->removeElement( mePnAmplMapG01L3_[i]->getName() );
      mePnAmplMapG01L3_[i] = 0;
      if ( mePnPedMapG01L3_[i] ) dbe_->removeElement( mePnPedMapG01L3_[i]->getName() );
      mePnPedMapG01L3_[i] = 0;
    }

    dbe_->setCurrentFolder("EcalEndcap/EELaserTask/Laser3/PN/Gain16");
    for (int i = 0; i < 18 ; i++) {
      if ( mePnAmplMapG16L3_[i] ) dbe_->removeElement( mePnAmplMapG16L3_[i]->getName() );
      mePnAmplMapG16L3_[i] = 0;
      if ( mePnPedMapG16L3_[i] ) dbe_->removeElement( mePnPedMapG16L3_[i]->getName() );
      mePnPedMapG16L3_[i] = 0;
    }

    dbe_->setCurrentFolder("EcalEndcap/EELaserTask/Laser4/PN");

    dbe_->setCurrentFolder("EcalEndcap/EELaserTask/Laser4/PN/Gain01");
    for (int i = 0; i < 18 ; i++) {
      if ( mePnAmplMapG01L4_[i] ) dbe_->removeElement( mePnAmplMapG01L4_[i]->getName() );
      mePnAmplMapG01L4_[i] = 0;
      if ( mePnPedMapG01L4_[i] ) dbe_->removeElement( mePnPedMapG01L4_[i]->getName() );
      mePnPedMapG01L4_[i] = 0;
    }

    dbe_->setCurrentFolder("EcalEndcap/EELaserTask/Laser4/PN/Gain16");
    for (int i = 0; i < 18 ; i++) {
      if ( mePnAmplMapG16L4_[i] ) dbe_->removeElement( mePnAmplMapG16L4_[i]->getName() );
      mePnAmplMapG16L4_[i] = 0;
      if ( mePnPedMapG16L4_[i] ) dbe_->removeElement( mePnPedMapG16L4_[i]->getName() );
      mePnPedMapG16L4_[i] = 0;
    }

  }

  init_ = false;

}

void EELaserTask::endJob(void){

  LogInfo("EELaserTask") << "analyzed " << ievt_ << " events";

  if ( init_ ) this->cleanup();

}

void EELaserTask::analyze(const Event& e, const EventSetup& c){

  Numbers::initGeometry(c);

  bool enable = false;
  map<int, EcalDCCHeaderBlock> dccMap;

  try {

    Handle<EcalRawDataCollection> dcchs;
    e.getByLabel(EcalRawDataCollection_, dcchs);

    for ( EcalRawDataCollection::const_iterator dcchItr = dcchs->begin(); dcchItr != dcchs->end(); ++dcchItr ) {

      EcalDCCHeaderBlock dcch = (*dcchItr);

      int ism = Numbers::iSM( dcch, EcalEndcap );

      map<int, EcalDCCHeaderBlock>::iterator i = dccMap.find( ism );
      if ( i != dccMap.end() ) continue;

      dccMap[ ism ] = dcch;

      if ( dcch.getRunType() == EcalDCCHeaderBlock::LASER_STD ||
           dcch.getRunType() == EcalDCCHeaderBlock::LASER_GAP ) enable = true;

    }

  } catch ( exception& ex) {

    LogWarning("EELaserTask") << EcalRawDataCollection_ << " not available";

  }

  if ( ! enable ) return;

  if ( ! init_ ) this->setup();

  ievt_++;

  try {

    Handle<EEDigiCollection> digis;
    e.getByLabel(EEDigiCollection_, digis);

    int need = digis->size();
    LogDebug("EELaserTask") << "event " << ievt_ << " digi collection size " << need;

    for ( EEDigiCollection::const_iterator digiItr = digis->begin(); digiItr != digis->end(); ++digiItr ) {

      EEDataFrame dataframe = (*digiItr);
      EEDetId id = dataframe.id();

      int ix = id.ix();
      int iy = id.iy();

      int ism = Numbers::iSM( id );

      if ( ism >= 1 && ism <= 9 ) ix = 101 - ix;

      map<int, EcalDCCHeaderBlock>::iterator i = dccMap.find(ism);
      if ( i == dccMap.end() ) continue;

      if ( ! ( dccMap[ism].getRunType() == EcalDCCHeaderBlock::LASER_STD ||
               dccMap[ism].getRunType() == EcalDCCHeaderBlock::LASER_GAP ) ) continue;

      LogDebug("EELaserTask") << " det id = " << id;
      LogDebug("EELaserTask") << " sm, ix, iy " << ism << " " << ix << " " << iy;

      int ic = Numbers::icEE(ism, ix, iy);

      for (int i = 0; i < 10; i++) {

        EcalMGPASample sample = dataframe.sample(i);
        int adc = sample.adc();
        float gain = 1.;

        MonitorElement* meShapeMap = 0;

        if ( sample.gainId() == 1 ) gain = 1./12.;
        if ( sample.gainId() == 2 ) gain = 1./ 6.;
        if ( sample.gainId() == 3 ) gain = 1./ 1.;

        if ( ix < 6 || iy > 10 ) {

          if ( dccMap[ism].getEventSettings().wavelength == 0 ) meShapeMap = meShapeMapL1A_[ism-1];
          if ( dccMap[ism].getEventSettings().wavelength == 1 ) meShapeMap = meShapeMapL2A_[ism-1];
          if ( dccMap[ism].getEventSettings().wavelength == 2 ) meShapeMap = meShapeMapL3A_[ism-1];
          if ( dccMap[ism].getEventSettings().wavelength == 3 ) meShapeMap = meShapeMapL4A_[ism-1];

        } else {

          if ( dccMap[ism].getEventSettings().wavelength == 0 ) meShapeMap = meShapeMapL1B_[ism-1];
          if ( dccMap[ism].getEventSettings().wavelength == 1 ) meShapeMap = meShapeMapL2B_[ism-1];
          if ( dccMap[ism].getEventSettings().wavelength == 2 ) meShapeMap = meShapeMapL3B_[ism-1];
          if ( dccMap[ism].getEventSettings().wavelength == 3 ) meShapeMap = meShapeMapL4B_[ism-1];

        }

//        float xval = float(adc) * gain;
        float xval = float(adc);

        if ( meShapeMap ) meShapeMap->Fill(ic - 0.5, i + 0.5, xval);

      }

    }

  } catch ( exception& ex) {

    LogWarning("EELaserTask") << EEDigiCollection_ << " not available";

  }

  float adcA[18];
  float adcB[18];

  for ( int i = 0; i < 18; i++ ) {
    adcA[i] = 0.;
    adcB[i] = 0.;
  }

  try {

    Handle<EcalPnDiodeDigiCollection> pns;
    e.getByLabel(EcalPnDiodeDigiCollection_, pns);

    int nep = pns->size();
    LogDebug("EELaserTask") << "event " << ievt_ << " pns collection size " << nep;

    for ( EcalPnDiodeDigiCollection::const_iterator pnItr = pns->begin(); pnItr != pns->end(); ++pnItr ) {

      EcalPnDiodeDigi pn = (*pnItr);
      EcalPnDiodeDetId id = pn.id();

      int ism = Numbers::iSM( id );

      int num = id.iPnId();

      map<int, EcalDCCHeaderBlock>::iterator i = dccMap.find(ism);
      if ( i == dccMap.end() ) continue;

      if ( ! ( dccMap[ism].getRunType() == EcalDCCHeaderBlock::LASER_STD ||
               dccMap[ism].getRunType() == EcalDCCHeaderBlock::LASER_GAP ) ) continue;

      LogDebug("EELaserTask") << " det id = " << id;
      LogDebug("EELaserTask") << " sm, num " << ism << " " << num;

      float xvalped = 0.;

      for (int i = 0; i < 4; i++) {

        EcalFEMSample sample = pn.sample(i);
        int adc = sample.adc();

        MonitorElement* mePNPed = 0;

        if ( sample.gainId() == 0 ) {
          if ( dccMap[ism].getEventSettings().wavelength == 0 ) mePNPed = mePnPedMapG01L1_[ism-1];
          if ( dccMap[ism].getEventSettings().wavelength == 1 ) mePNPed = mePnPedMapG01L2_[ism-1];
          if ( dccMap[ism].getEventSettings().wavelength == 2 ) mePNPed = mePnPedMapG01L3_[ism-1];
          if ( dccMap[ism].getEventSettings().wavelength == 3 ) mePNPed = mePnPedMapG01L4_[ism-1];
        }
        if ( sample.gainId() == 1 ) {
          if ( dccMap[ism].getEventSettings().wavelength == 0 ) mePNPed = mePnPedMapG16L1_[ism-1];
          if ( dccMap[ism].getEventSettings().wavelength == 1 ) mePNPed = mePnPedMapG16L2_[ism-1];
          if ( dccMap[ism].getEventSettings().wavelength == 2 ) mePNPed = mePnPedMapG16L3_[ism-1];
          if ( dccMap[ism].getEventSettings().wavelength == 3 ) mePNPed = mePnPedMapG16L4_[ism-1];
        }

        float xval = float(adc);

        if ( mePNPed ) mePNPed->Fill(0.5, num - 0.5, xval);

        xvalped = xvalped + xval;

      }

      xvalped = xvalped / 4;

      float xvalmax = 0.;

      MonitorElement* mePN = 0;

      for (int i = 0; i < 50; i++) {

        EcalFEMSample sample = pn.sample(i);
        int adc = sample.adc();

        float xval = float(adc);

        if ( xval >= xvalmax ) xvalmax = xval;

      }

      xvalmax = xvalmax - xvalped;

      if ( pn.sample(0).gainId() == 0 ) {
        if ( dccMap[ism].getEventSettings().wavelength == 0 ) mePN = mePnAmplMapG01L1_[ism-1];
        if ( dccMap[ism].getEventSettings().wavelength == 1 ) mePN = mePnAmplMapG01L2_[ism-1];
        if ( dccMap[ism].getEventSettings().wavelength == 2 ) mePN = mePnAmplMapG01L3_[ism-1];
        if ( dccMap[ism].getEventSettings().wavelength == 3 ) mePN = mePnAmplMapG01L4_[ism-1];
      }
      if ( pn.sample(0).gainId() == 1 ) {
        if ( dccMap[ism].getEventSettings().wavelength == 0 ) mePN = mePnAmplMapG16L1_[ism-1];
        if ( dccMap[ism].getEventSettings().wavelength == 1 ) mePN = mePnAmplMapG16L2_[ism-1];
        if ( dccMap[ism].getEventSettings().wavelength == 2 ) mePN = mePnAmplMapG16L3_[ism-1];
        if ( dccMap[ism].getEventSettings().wavelength == 3 ) mePN = mePnAmplMapG16L4_[ism-1];
      }

      if ( mePN ) mePN->Fill(0.5, num - 0.5, xvalmax);

      if ( num == 1 ) adcA[ism-1] = xvalmax;
      if ( num == 6 ) adcB[ism-1] = xvalmax;

    }

  } catch ( exception& ex) {

    LogWarning("EELaserTask") << EcalPnDiodeDigiCollection_ << " not available";

  }

  try {

    Handle<EcalUncalibratedRecHitCollection> hits;
    e.getByLabel(EcalUncalibratedRecHitCollection_, hits);

    int neh = hits->size();
    LogDebug("EELaserTask") << "event " << ievt_ << " hits collection size " << neh;

    for ( EcalUncalibratedRecHitCollection::const_iterator hitItr = hits->begin(); hitItr != hits->end(); ++hitItr ) {

      EcalUncalibratedRecHit hit = (*hitItr);
      EEDetId id = hit.id();

      int ix = id.ix();
      int iy = id.iy();

      int ism = Numbers::iSM( id );

      if ( ism >= 1 && ism <= 9 ) ix = 101 - ix;

      float xix = ix - 0.5;
      float xiy = iy - 0.5;

      map<int, EcalDCCHeaderBlock>::iterator i = dccMap.find(ism);
      if ( i == dccMap.end() ) continue;

      if ( ! ( dccMap[ism].getRunType() == EcalDCCHeaderBlock::LASER_STD ||
               dccMap[ism].getRunType() == EcalDCCHeaderBlock::LASER_GAP ) ) continue;

      LogDebug("EELaserTask") << " det id = " << id;
      LogDebug("EELaserTask") << " sm, ix, iy " << ism << " " << ix << " " << iy;

      MonitorElement* meAmplMap = 0;
      MonitorElement* meTimeMap = 0;
      MonitorElement* meAmplPNMap = 0;

      if ( ix < 6 || iy > 10 ) {

        if ( dccMap[ism].getEventSettings().wavelength == 0 ) {
          meAmplMap = meAmplMapL1A_[ism-1];
          meTimeMap = meTimeMapL1A_[ism-1];
          meAmplPNMap = meAmplPNMapL1A_[ism-1];
        }
        if ( dccMap[ism].getEventSettings().wavelength == 1 ) {
          meAmplMap = meAmplMapL2A_[ism-1];
          meTimeMap = meTimeMapL2A_[ism-1];
          meAmplPNMap = meAmplPNMapL2A_[ism-1];
        }
        if ( dccMap[ism].getEventSettings().wavelength == 2 ) {
          meAmplMap = meAmplMapL3A_[ism-1];
          meTimeMap = meTimeMapL3A_[ism-1];
          meAmplPNMap = meAmplPNMapL3A_[ism-1];
        }
        if ( dccMap[ism].getEventSettings().wavelength == 3 ) {
          meAmplMap = meAmplMapL4A_[ism-1];
          meTimeMap = meTimeMapL4A_[ism-1];
          meAmplPNMap = meAmplPNMapL4A_[ism-1];
        }

      } else {

        if ( dccMap[ism].getEventSettings().wavelength == 0 ) {
          meAmplMap = meAmplMapL1B_[ism-1];
          meTimeMap = meTimeMapL1B_[ism-1];
          meAmplPNMap = meAmplPNMapL1B_[ism-1];
        }
        if ( dccMap[ism].getEventSettings().wavelength == 1 ) {
          meAmplMap = meAmplMapL2B_[ism-1];
          meTimeMap = meTimeMapL2B_[ism-1];
          meAmplPNMap = meAmplPNMapL2B_[ism-1];
        }
        if ( dccMap[ism].getEventSettings().wavelength == 2 ) {
          meAmplMap = meAmplMapL3B_[ism-1];
          meTimeMap = meTimeMapL3B_[ism-1];
          meAmplPNMap = meAmplPNMapL3B_[ism-1];
        }
        if ( dccMap[ism].getEventSettings().wavelength == 3 ) {
          meAmplMap = meAmplMapL4B_[ism-1];
          meTimeMap = meTimeMapL4B_[ism-1];
          meAmplPNMap = meAmplPNMapL4B_[ism-1];
        }

      }

      float xval = hit.amplitude();
      if ( xval <= 0. ) xval = 0.0;
      float yval = hit.jitter() + 6.0;
      if ( yval <= 0. ) yval = 0.0;
      float zval = hit.pedestal();
      if ( zval <= 0. ) zval = 0.0;

      LogDebug("EELaserTask") << " hit amplitude " << xval;
      LogDebug("EELaserTask") << " hit jitter " << yval;
      LogDebug("EELaserTask") << " hit pedestal " << zval;

      if ( meAmplMap ) meAmplMap->Fill(xix, xiy, xval);

      if ( meTimeMap ) meTimeMap->Fill(xix, xiy, yval);

      float wval = 0.;

      if ( ix < 6 || iy > 10 ) {

        if ( adcA[ism-1] != 0. ) wval = xval / adcA[ism-1];

      } else {

        if ( adcB[ism-1] != 0. ) wval = xval / adcB[ism-1];

      }

      LogDebug("EELaserTask") << " hit amplitude over PN " << wval;

      if ( meAmplPNMap ) meAmplPNMap->Fill(xix, xiy, wval);

    }

  } catch ( exception& ex) {

    LogWarning("EELaserTask") << EcalUncalibratedRecHitCollection_ << " not available";

  }

}

