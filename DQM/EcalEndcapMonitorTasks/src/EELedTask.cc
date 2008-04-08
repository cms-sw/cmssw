/*
 * \file EELedTask.cc
 *
 * $Date: 2008/04/07 11:30:25 $
 * $Revision: 1.33 $
 * \author G. Della Ricca
 *
*/

#include <iostream>
#include <fstream>
#include <vector>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/MonitorElement.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include <DQM/EcalCommon/interface/Numbers.h>

#include <DQM/EcalEndcapMonitorTasks/interface/EELedTask.h>

using namespace cms;
using namespace edm;
using namespace std;

EELedTask::EELedTask(const ParameterSet& ps){

  init_ = false;

  // get hold of back-end interface
  dqmStore_ = Service<DQMStore>().operator->();

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  EcalRawDataCollection_ = ps.getParameter<edm::InputTag>("EcalRawDataCollection");
  EEDigiCollection_ = ps.getParameter<edm::InputTag>("EEDigiCollection");
  EcalPnDiodeDigiCollection_ = ps.getParameter<edm::InputTag>("EcalPnDiodeDigiCollection");
  EcalUncalibratedRecHitCollection_ = ps.getParameter<edm::InputTag>("EcalUncalibratedRecHitCollection");

  for (int i = 0; i < 18; i++) {
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
  }

}

EELedTask::~EELedTask(){

}

void EELedTask::beginJob(const EventSetup& c){

  ievt_ = 0;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder("EcalEndcap/EELedTask");
    dqmStore_->rmdir("EcalEndcap/EELedTask");
  }

  Numbers::initGeometry(c, false);

}

void EELedTask::setup(void){

  init_ = true;

  char histo[200];

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder("EcalEndcap/EELedTask");

    dqmStore_->setCurrentFolder("EcalEndcap/EELedTask/Led1");
    for (int i = 0; i < 18; i++) {
      sprintf(histo, "EELDT shape %s L1A", Numbers::sEE(i+1).c_str());
      meShapeMapL1A_[i] = dqmStore_->bookProfile2D(histo, histo, 850, 0., 850., 10, 0., 10., 4096, 0., 4096., "s");
      meShapeMapL1A_[i]->setAxisTitle("channel", 1);
      meShapeMapL1A_[i]->setAxisTitle("sample", 2);
      meShapeMapL1A_[i]->setAxisTitle("amplitude", 3);
      dqmStore_->tag(meShapeMapL1A_[i], i+1);
      sprintf(histo, "EELDT amplitude %s L1A", Numbers::sEE(i+1).c_str());
      meAmplMapL1A_[i] = dqmStore_->bookProfile2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 4096, 0., 4096.*12., "s");
      meAmplMapL1A_[i]->setAxisTitle("jx", 1);
      meAmplMapL1A_[i]->setAxisTitle("jy", 2);
      dqmStore_->tag(meAmplMapL1A_[i], i+1);
      sprintf(histo, "EELDT timing %s L1A", Numbers::sEE(i+1).c_str());
      meTimeMapL1A_[i] = dqmStore_->bookProfile2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 250, 0., 10., "s");
      meTimeMapL1A_[i]->setAxisTitle("jx", 1);
      meTimeMapL1A_[i]->setAxisTitle("jy", 2);
      dqmStore_->tag(meTimeMapL1A_[i], i+1);
      sprintf(histo, "EELDT amplitude over PN %s L1A", Numbers::sEE(i+1).c_str());
      meAmplPNMapL1A_[i] = dqmStore_->bookProfile2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 4096, 0., 4096.*12., "s");
      meAmplPNMapL1A_[i]->setAxisTitle("jx", 1);
      meAmplPNMapL1A_[i]->setAxisTitle("jy", 2);
      dqmStore_->tag(meAmplPNMapL1A_[i], i+1);

      sprintf(histo, "EELDT shape %s L1B", Numbers::sEE(i+1).c_str());
      meShapeMapL1B_[i] = dqmStore_->bookProfile2D(histo, histo, 850, 0., 850., 10, 0., 10., 4096, 0., 4096., "s");
      meShapeMapL1B_[i]->setAxisTitle("channel", 1);
      meShapeMapL1B_[i]->setAxisTitle("sample", 2);
      meShapeMapL1B_[i]->setAxisTitle("amplitude", 3);
      dqmStore_->tag(meShapeMapL1B_[i], i+1);
      sprintf(histo, "EELDT amplitude %s L1B", Numbers::sEE(i+1).c_str());
      meAmplMapL1B_[i] = dqmStore_->bookProfile2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 4096, 0., 4096.*12., "s");
      meAmplMapL1B_[i]->setAxisTitle("jx", 1);
      meAmplMapL1B_[i]->setAxisTitle("jy", 2);
      dqmStore_->tag(meAmplMapL1B_[i], i+1);
      sprintf(histo, "EELDT timing %s L1B", Numbers::sEE(i+1).c_str());
      meTimeMapL1B_[i] = dqmStore_->bookProfile2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 250, 0., 10., "s");
      meTimeMapL1B_[i]->setAxisTitle("jx", 1);
      meTimeMapL1B_[i]->setAxisTitle("jy", 2);
      dqmStore_->tag(meTimeMapL1B_[i], i+1);
      sprintf(histo, "EELDT amplitude over PN %s L1B", Numbers::sEE(i+1).c_str());
      meAmplPNMapL1B_[i] = dqmStore_->bookProfile2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 4096, 0., 4096.*12., "s");
      meAmplPNMapL1B_[i]->setAxisTitle("jx", 1);
      meAmplPNMapL1B_[i]->setAxisTitle("jy", 2);
      dqmStore_->tag(meAmplPNMapL1B_[i], i+1);
    }

    dqmStore_->setCurrentFolder("EcalEndcap/EELedTask/Led2");
    for (int i = 0; i < 18; i++) {
      sprintf(histo, "EELDT shape %s L2A", Numbers::sEE(i+1).c_str());
      meShapeMapL2A_[i] = dqmStore_->bookProfile2D(histo, histo, 850, 0., 850., 10, 0., 10., 4096, 0., 4096., "s");
      meShapeMapL2A_[i]->setAxisTitle("channel", 1);
      meShapeMapL2A_[i]->setAxisTitle("sample", 2);
      meShapeMapL2A_[i]->setAxisTitle("amplitude", 3);
      dqmStore_->tag(meShapeMapL2A_[i], i+1);
      sprintf(histo, "EELDT amplitude %s L2A", Numbers::sEE(i+1).c_str());
      meAmplMapL2A_[i] = dqmStore_->bookProfile2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 4096, 0., 4096.*12., "s");
      meAmplMapL2A_[i]->setAxisTitle("jx", 1);
      meAmplMapL2A_[i]->setAxisTitle("jy", 2);
      dqmStore_->tag(meAmplMapL2A_[i], i+1);
      sprintf(histo, "EELDT timing %s L2A", Numbers::sEE(i+1).c_str());
      meTimeMapL2A_[i] = dqmStore_->bookProfile2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 250, 0., 10., "s");
      meTimeMapL2A_[i]->setAxisTitle("jx", 1);
      meTimeMapL2A_[i]->setAxisTitle("jy", 2);
      dqmStore_->tag(meTimeMapL2A_[i], i+1);
      sprintf(histo, "EELDT amplitude over PN %s L2A", Numbers::sEE(i+1).c_str());
      meAmplPNMapL2A_[i] = dqmStore_->bookProfile2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 4096, 0., 4096.*12., "s");
      meAmplPNMapL2A_[i]->setAxisTitle("jx", 1);
      meAmplPNMapL2A_[i]->setAxisTitle("jy", 2);
      dqmStore_->tag(meAmplPNMapL2A_[i], i+1);

      sprintf(histo, "EELDT shape %s L2B", Numbers::sEE(i+1).c_str());
      meShapeMapL2B_[i] = dqmStore_->bookProfile2D(histo, histo, 850, 0., 850., 10, 0., 10., 4096, 0., 4096., "s");
      meShapeMapL2B_[i]->setAxisTitle("channel", 1);
      meShapeMapL2B_[i]->setAxisTitle("sample", 2);
      meShapeMapL2B_[i]->setAxisTitle("amplitude", 3);
      dqmStore_->tag(meShapeMapL2B_[i], i+1);
      sprintf(histo, "EELDT amplitude %s L2B", Numbers::sEE(i+1).c_str());
      meAmplMapL2B_[i] = dqmStore_->bookProfile2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 4096, 0., 4096.*12., "s");
      meAmplMapL2B_[i]->setAxisTitle("jx", 1);
      meAmplMapL2B_[i]->setAxisTitle("jy", 2);
      dqmStore_->tag(meAmplMapL2B_[i], i+1);
      sprintf(histo, "EELDT timing %s L2B", Numbers::sEE(i+1).c_str());
      meTimeMapL2B_[i] = dqmStore_->bookProfile2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 250, 0., 10., "s");
      meTimeMapL2B_[i]->setAxisTitle("jx", 1);
      meTimeMapL2B_[i]->setAxisTitle("jy", 2);
      dqmStore_->tag(meTimeMapL2B_[i], i+1);
      sprintf(histo, "EELDT amplitude over PN %s L2B", Numbers::sEE(i+1).c_str());
      meAmplPNMapL2B_[i] = dqmStore_->bookProfile2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 4096, 0., 4096.*12., "s");
      meAmplPNMapL2B_[i]->setAxisTitle("jx", 1);
      meAmplPNMapL2B_[i]->setAxisTitle("jy", 2);
      dqmStore_->tag(meAmplPNMapL2B_[i], i+1);
    }

    dqmStore_->setCurrentFolder("EcalEndcap/EELedTask/Led1/PN");

    dqmStore_->setCurrentFolder("EcalEndcap/EELedTask/Led1/PN/Gain01");
    for (int i = 0; i < 18; i++) {
      sprintf(histo, "EEPDT PNs amplitude %s G01 L1", Numbers::sEE(i+1).c_str());
      mePnAmplMapG01L1_[i] = dqmStore_->bookProfile(histo, histo, 10, 0., 10., 4096, 0., 4096., "s");
      mePnAmplMapG01L1_[i]->setAxisTitle("channel", 1);
      mePnAmplMapG01L1_[i]->setAxisTitle("amplitude", 2);
      dqmStore_->tag(mePnAmplMapG01L1_[i], i+1);
      sprintf(histo, "EEPDT PNs pedestal %s G01 L1", Numbers::sEE(i+1).c_str());
      mePnPedMapG01L1_[i] = dqmStore_->bookProfile(histo, histo, 10, 0., 10., 4096, 0., 4096., "s");
      mePnPedMapG01L1_[i]->setAxisTitle("channel", 1);
      mePnPedMapG01L1_[i]->setAxisTitle("pedestal", 2);
      dqmStore_->tag(mePnPedMapG01L1_[i], i+1);
    }

    dqmStore_->setCurrentFolder("EcalEndcap/EELedTask/Led1/PN/Gain16");
    for (int i = 0; i < 18; i++) {
      sprintf(histo, "EEPDT PNs amplitude %s G16 L1", Numbers::sEE(i+1).c_str());
      mePnAmplMapG16L1_[i] = dqmStore_->bookProfile(histo, histo, 10, 0., 10., 4096, 0., 4096., "s");
      mePnAmplMapG16L1_[i]->setAxisTitle("channel", 1);
      mePnAmplMapG16L1_[i]->setAxisTitle("amplitude", 2);
      dqmStore_->tag(mePnAmplMapG16L1_[i], i+1);
      sprintf(histo, "EEPDT PNs pedestal %s G16 L1", Numbers::sEE(i+1).c_str());
      mePnPedMapG16L1_[i] = dqmStore_->bookProfile(histo, histo, 10, 0., 10., 4096, 0., 4096., "s");
      mePnPedMapG16L1_[i]->setAxisTitle("channel", 1);
      mePnPedMapG16L1_[i]->setAxisTitle("pedestal", 2); 
      dqmStore_->tag(mePnPedMapG16L1_[i], i+1);
    }

    dqmStore_->setCurrentFolder("EcalEndcap/EELedTask/Led2/PN");

    dqmStore_->setCurrentFolder("EcalEndcap/EELedTask/Led2/PN/Gain01");
    for (int i = 0; i < 18; i++) {
      sprintf(histo, "EEPDT PNs amplitude %s G01 L2", Numbers::sEE(i+1).c_str());
      mePnAmplMapG01L2_[i] = dqmStore_->bookProfile(histo, histo, 10, 0., 10., 4096, 0., 4096., "s");
      mePnAmplMapG01L2_[i]->setAxisTitle("amplitude", 2);
      mePnAmplMapG01L2_[i]->setAxisTitle("channel", 1);
      dqmStore_->tag(mePnAmplMapG01L2_[i], i+1);
      sprintf(histo, "EEPDT PNs pedestal %s G01 L2", Numbers::sEE(i+1).c_str());
      mePnPedMapG01L2_[i] = dqmStore_->bookProfile(histo, histo, 10, 0., 10., 4096, 0., 4096., "s");
      mePnPedMapG01L2_[i]->setAxisTitle("channel", 1);
      mePnPedMapG01L2_[i]->setAxisTitle("pedestal", 2);
      dqmStore_->tag(mePnPedMapG01L2_[i], i+1);
    }

    dqmStore_->setCurrentFolder("EcalEndcap/EELedTask/Led2/PN/Gain16");
    for (int i = 0; i < 18; i++) {
      sprintf(histo, "EEPDT PNs amplitude %s G16 L2", Numbers::sEE(i+1).c_str());
      mePnAmplMapG16L2_[i] = dqmStore_->bookProfile(histo, histo, 10, 0., 10., 4096, 0., 4096., "s");
      mePnAmplMapG16L2_[i]->setAxisTitle("channel", 1);
      mePnAmplMapG16L2_[i]->setAxisTitle("amplitude", 2);
      dqmStore_->tag(mePnAmplMapG16L2_[i], i+1);
      sprintf(histo, "EEPDT PNs pedestal %s G16 L2", Numbers::sEE(i+1).c_str());
      mePnPedMapG16L2_[i] = dqmStore_->bookProfile(histo, histo, 10, 0., 10., 4096, 0., 4096., "s");
      mePnPedMapG16L2_[i]->setAxisTitle("channel", 1);
      mePnPedMapG16L2_[i]->setAxisTitle("pedestal", 2); 
      dqmStore_->tag(mePnPedMapG16L2_[i], i+1);
    }

  }

}

void EELedTask::cleanup(void){

  if ( ! enableCleanup_ ) return;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder("EcalEndcap/EELedTask");

    dqmStore_->setCurrentFolder("EcalEndcap/EELedTask/Led1");
    for (int i = 0; i < 18; i++) {
      if ( meShapeMapL1A_[i] )  dqmStore_->removeElement( meShapeMapL1A_[i]->getName() );
      meShapeMapL1A_[i] = 0;
      if ( meAmplMapL1A_[i] ) dqmStore_->removeElement( meAmplMapL1A_[i]->getName() );
      meAmplMapL1A_[i] = 0;
      if ( meTimeMapL1A_[i] ) dqmStore_->removeElement( meTimeMapL1A_[i]->getName() );
      meTimeMapL1A_[i] = 0;
      if ( meAmplPNMapL1A_[i] ) dqmStore_->removeElement( meAmplPNMapL1A_[i]->getName() );
      meAmplPNMapL1A_[i] = 0;

      if ( meShapeMapL1B_[i] )  dqmStore_->removeElement( meShapeMapL1B_[i]->getName() );
      meShapeMapL1B_[i] = 0;
      if ( meAmplMapL1B_[i] ) dqmStore_->removeElement( meAmplMapL1B_[i]->getName() );
      meAmplMapL1B_[i] = 0;
      if ( meTimeMapL1B_[i] ) dqmStore_->removeElement( meTimeMapL1B_[i]->getName() );
      meTimeMapL1B_[i] = 0;
      if ( meAmplPNMapL1B_[i] ) dqmStore_->removeElement( meAmplPNMapL1B_[i]->getName() );
      meAmplPNMapL1B_[i] = 0;
    }

    dqmStore_->setCurrentFolder("EcalEndcap/EELedTask/Led2");
    for (int i = 0; i < 18; i++) {
      if ( meShapeMapL2A_[i] )  dqmStore_->removeElement( meShapeMapL2A_[i]->getName() );
      meShapeMapL2A_[i] = 0;
      if ( meAmplMapL2A_[i] ) dqmStore_->removeElement( meAmplMapL2A_[i]->getName() );
      meAmplMapL2A_[i] = 0;
      if ( meTimeMapL2A_[i] ) dqmStore_->removeElement( meTimeMapL2A_[i]->getName() );
      meTimeMapL2A_[i] = 0;
      if ( meAmplPNMapL2A_[i] ) dqmStore_->removeElement( meAmplPNMapL2A_[i]->getName() );
      meAmplPNMapL2A_[i] = 0;

      if ( meShapeMapL2B_[i] )  dqmStore_->removeElement( meShapeMapL2B_[i]->getName() );
      meShapeMapL2B_[i] = 0;
      if ( meAmplMapL2B_[i] ) dqmStore_->removeElement( meAmplMapL2B_[i]->getName() );
      meAmplMapL2B_[i] = 0;
      if ( meTimeMapL2B_[i] ) dqmStore_->removeElement( meTimeMapL2B_[i]->getName() );
      meTimeMapL2B_[i] = 0;
      if ( meAmplPNMapL2B_[i] ) dqmStore_->removeElement( meAmplPNMapL2B_[i]->getName() );
      meAmplPNMapL2B_[i] = 0;
    }

    dqmStore_->setCurrentFolder("EcalEndcap/EELedTask/Led1/PN");

    dqmStore_->setCurrentFolder("EcalEndcap/EELedTask/Led1/PN/Gain01");
    for (int i = 0; i < 18; i++) {
      if ( mePnAmplMapG01L1_[i] ) dqmStore_->removeElement( mePnAmplMapG01L1_[i]->getName() );
      mePnAmplMapG01L1_[i] = 0;
      if ( mePnPedMapG01L1_[i] ) dqmStore_->removeElement( mePnPedMapG01L1_[i]->getName() );
      mePnPedMapG01L1_[i] = 0;
    }

    dqmStore_->setCurrentFolder("EcalEndcap/EELedTask/Led1/PN/Gain16");
    for (int i = 0; i < 18; i++) {
      if ( mePnAmplMapG16L1_[i] ) dqmStore_->removeElement( mePnAmplMapG16L1_[i]->getName() );
      mePnAmplMapG16L1_[i] = 0;
      if ( mePnPedMapG16L1_[i] ) dqmStore_->removeElement( mePnPedMapG16L1_[i]->getName() );
      mePnPedMapG16L1_[i] = 0;
    }

    dqmStore_->setCurrentFolder("EcalEndcap/EELedTask/Led2/PN");

    dqmStore_->setCurrentFolder("EcalEndcap/EELedTask/Led2/PN/Gain01");
    for (int i = 0; i < 18; i++) {
      if ( mePnAmplMapG01L2_[i] ) dqmStore_->removeElement( mePnAmplMapG01L2_[i]->getName() );
      mePnAmplMapG01L2_[i] = 0;
      if ( mePnPedMapG01L2_[i] ) dqmStore_->removeElement( mePnPedMapG01L2_[i]->getName() );
      mePnPedMapG01L2_[i] = 0;
    }

    dqmStore_->setCurrentFolder("EcalEndcap/EELedTask/Led2/PN/Gain16");
    for (int i = 0; i < 18; i++) {
      if ( mePnAmplMapG16L2_[i] ) dqmStore_->removeElement( mePnAmplMapG16L2_[i]->getName() );
      mePnAmplMapG16L2_[i] = 0;
      if ( mePnPedMapG16L2_[i] ) dqmStore_->removeElement( mePnPedMapG16L2_[i]->getName() );
      mePnPedMapG16L2_[i] = 0;
    }

  }

  init_ = false;

}

void EELedTask::endJob(void){

  LogInfo("EELedTask") << "analyzed " << ievt_ << " events";

  if ( init_ ) this->cleanup();

}

void EELedTask::analyze(const Event& e, const EventSetup& c){

  bool enable = false;
  map<int, EcalDCCHeaderBlock> dccMap;

  Handle<EcalRawDataCollection> dcchs;

  if ( e.getByLabel(EcalRawDataCollection_, dcchs) ) {

    for ( EcalRawDataCollection::const_iterator dcchItr = dcchs->begin(); dcchItr != dcchs->end(); ++dcchItr ) {

      EcalDCCHeaderBlock dcch = (*dcchItr);

      if ( Numbers::subDet( dcch ) != EcalEndcap ) continue;

      int ism = Numbers::iSM( dcch, EcalEndcap );

      map<int, EcalDCCHeaderBlock>::iterator i = dccMap.find( ism );
      if ( i != dccMap.end() ) continue;

      dccMap[ ism ] = dcch;

      if ( dcch.getRunType() == EcalDCCHeaderBlock::LED_STD ||
           dcch.getRunType() == EcalDCCHeaderBlock::LED_GAP ) enable = true;

    }

  } else {

    LogWarning("EELedTask") << EcalRawDataCollection_ << " not available";

  }

  if ( ! enable ) return;

  if ( ! init_ ) this->setup();

  ievt_++;

  Handle<EEDigiCollection> digis;

  if ( e.getByLabel(EEDigiCollection_, digis) ) {

    int need = digis->size();
    LogDebug("EELedTask") << "event " << ievt_ << " digi collection size " << need;

    for ( EEDigiCollection::const_iterator digiItr = digis->begin(); digiItr != digis->end(); ++digiItr ) {

      EEDataFrame dataframe = (*digiItr);
      EEDetId id = dataframe.id();

      int ix = id.ix();
      int iy = id.iy();

      int ism = Numbers::iSM( id );

      map<int, EcalDCCHeaderBlock>::iterator i = dccMap.find(ism);
      if ( i == dccMap.end() ) continue;

      if ( ! ( dccMap[ism].getRunType() == EcalDCCHeaderBlock::LED_STD ||
               dccMap[ism].getRunType() == EcalDCCHeaderBlock::LED_GAP ) ) continue;

      if ( dccMap[ism].getRtHalf() != 3 &&
           dccMap[ism].getRtHalf() != Numbers::RtHalf(id) ) continue;

      LogDebug("EELedTask") << " det id = " << id;
      LogDebug("EELedTask") << " sm, ix, iy " << ism << " " << ix << " " << iy;

      int ic = Numbers::icEE(ism, ix, iy);

      for (int i = 0; i < 10; i++) {

        EcalMGPASample sample = dataframe.sample(i);
        int adc = sample.adc();
        float gain = 1.;

        MonitorElement* meShapeMap = 0;

        if ( sample.gainId() == 1 ) gain = 1./12.;
        if ( sample.gainId() == 2 ) gain = 1./ 6.;
        if ( sample.gainId() == 3 ) gain = 1./ 1.;

        if ( dccMap[ism].getRtHalf() == 1 || ( dccMap[ism].getRtHalf() == 3 && Numbers::RtHalf(id) == 1 ) ) {

          if ( dccMap[ism].getEventSettings().wavelength == 0 ) meShapeMap = meShapeMapL1A_[ism-1];
          if ( dccMap[ism].getEventSettings().wavelength == 1 ) meShapeMap = meShapeMapL2A_[ism-1];

        } else if ( dccMap[ism].getRtHalf() == 2 || ( dccMap[ism].getRtHalf() == 3 && Numbers::RtHalf(id) == 2 ) ) {

          if ( dccMap[ism].getEventSettings().wavelength == 0 ) meShapeMap = meShapeMapL1B_[ism-1];
          if ( dccMap[ism].getEventSettings().wavelength == 1 ) meShapeMap = meShapeMapL2B_[ism-1];

        } else {

          LogWarning("EELedTask") << " RtHalf = " << dccMap[ism].getRtHalf();

        }

//        float xval = float(adc) * gain;
        float xval = float(adc);

        if ( meShapeMap ) meShapeMap->Fill(ic - 0.5, i + 0.5, xval);

      }

    }

  } else {

    LogWarning("EELedTask") << EEDigiCollection_ << " not available";

  }

  float adcA[18];
  float adcB[18];

  for ( int i = 0; i < 18; i++ ) {
    adcA[i] = 0.;
    adcB[i] = 0.;
  }

  Handle<EcalPnDiodeDigiCollection> pns;

  if ( e.getByLabel(EcalPnDiodeDigiCollection_, pns) ) {

    int nep = pns->size();
    LogDebug("EELedTask") << "event " << ievt_ << " pns collection size " << nep;

    for ( EcalPnDiodeDigiCollection::const_iterator pnItr = pns->begin(); pnItr != pns->end(); ++pnItr ) {

      EcalPnDiodeDigi pn = (*pnItr);
      EcalPnDiodeDetId id = pn.id();

      if ( Numbers::subDet( id ) != EcalEndcap ) continue;

      int ism = Numbers::iSM( id );

      int num = id.iPnId();

      map<int, EcalDCCHeaderBlock>::iterator i = dccMap.find(ism);
      if ( i == dccMap.end() ) continue;

      if ( ! ( dccMap[ism].getRunType() == EcalDCCHeaderBlock::LED_STD ||
               dccMap[ism].getRunType() == EcalDCCHeaderBlock::LED_GAP ) ) continue;

      LogDebug("EELedTask") << " det id = " << id;
      LogDebug("EELedTask") << " sm, num " << ism << " " << num;

      float xvalped = 0.;

      for (int i = 0; i < 4; i++) {

        EcalFEMSample sample = pn.sample(i);
        int adc = sample.adc();

        MonitorElement* mePNPed = 0;

        if ( sample.gainId() == 0 ) {
          if ( dccMap[ism].getEventSettings().wavelength == 0 ) mePNPed = mePnPedMapG01L1_[ism-1];
          if ( dccMap[ism].getEventSettings().wavelength == 1 ) mePNPed = mePnPedMapG01L2_[ism-1];
        }
        if ( sample.gainId() == 1 ) {
          if ( dccMap[ism].getEventSettings().wavelength == 0 ) mePNPed = mePnPedMapG16L1_[ism-1];
          if ( dccMap[ism].getEventSettings().wavelength == 1 ) mePNPed = mePnPedMapG16L2_[ism-1];
        }

        float xval = float(adc);

        if ( mePNPed ) mePNPed->Fill(num - 0.5, xval);

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
      }
      if ( pn.sample(0).gainId() == 1 ) {
        if ( dccMap[ism].getEventSettings().wavelength == 0 ) mePN = mePnAmplMapG16L1_[ism-1];
        if ( dccMap[ism].getEventSettings().wavelength == 1 ) mePN = mePnAmplMapG16L2_[ism-1];
      }

      if ( mePN ) mePN->Fill(num - 0.5, xvalmax);

      if ( num == 1 ) adcA[ism-1] = xvalmax;
      if ( num == 6 ) adcB[ism-1] = xvalmax;

    }

  } else {

    LogWarning("EELedTask") << EcalPnDiodeDigiCollection_ << " not available";

  }

  Handle<EcalUncalibratedRecHitCollection> hits;

  if ( e.getByLabel(EcalUncalibratedRecHitCollection_, hits) ) {

    int neh = hits->size();
    LogDebug("EELedTask") << "event " << ievt_ << " hits collection size " << neh;

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

      if ( ! ( dccMap[ism].getRunType() == EcalDCCHeaderBlock::LED_STD ||
               dccMap[ism].getRunType() == EcalDCCHeaderBlock::LED_GAP ) ) continue;

      if ( dccMap[ism].getRtHalf() != 3 &&
           dccMap[ism].getRtHalf() != Numbers::RtHalf(id) ) continue;

      LogDebug("EELedTask") << " det id = " << id;
      LogDebug("EELedTask") << " sm, ix, iy " << ism << " " << ix << " " << iy;

      MonitorElement* meAmplMap = 0;
      MonitorElement* meTimeMap = 0;
      MonitorElement* meAmplPNMap = 0;

      if ( dccMap[ism].getRtHalf() == 1 || ( dccMap[ism].getRtHalf() == 3 && Numbers::RtHalf(id) == 1 ) ) {

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

      } else if ( dccMap[ism].getRtHalf() == 2 || ( dccMap[ism].getRtHalf() == 3 && Numbers::RtHalf(id) == 2 ) ) { 

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

      } else {

        LogWarning("EELedTask") << " RtHalf = " << dccMap[ism].getRtHalf();

      }

      float xval = hit.amplitude();
      if ( xval <= 0. ) xval = 0.0;
      float yval = hit.jitter() + 6.0;
      if ( yval <= 0. ) yval = 0.0;
      float zval = hit.pedestal();
      if ( zval <= 0. ) zval = 0.0;

      LogDebug("EELedTask") << " hit amplitude " << xval;
      LogDebug("EELedTask") << " hit jitter " << yval;
      LogDebug("EELedTask") << " hit pedestal " << zval;

      if ( meAmplMap ) meAmplMap->Fill(xix, xiy, xval);

      if ( meTimeMap ) meTimeMap->Fill(xix, xiy, yval);

      float wval = 0.;

      if ( dccMap[ism].getRtHalf() == 1 || ( dccMap[ism].getRtHalf() == 3 && Numbers::RtHalf(id) == 1 ) ) {

        if ( adcA[ism-1] != 0. ) wval = xval / adcA[ism-1];

      } else if ( dccMap[ism].getRtHalf() == 2 || ( dccMap[ism].getRtHalf() == 3 && Numbers::RtHalf(id) == 2 ) ) {

        if ( adcB[ism-1] != 0. ) wval = xval / adcB[ism-1];

      } else {

        LogWarning("EELedTask") << " RtHalf = " << dccMap[ism].getRtHalf();

      }

      LogDebug("EELedTask") << " hit amplitude over PN " << wval;

      if ( meAmplPNMap ) meAmplPNMap->Fill(xix, xiy, wval);

    }

  } else {

    LogWarning("EELedTask") << EcalUncalibratedRecHitCollection_ << " not available";

  }

}

