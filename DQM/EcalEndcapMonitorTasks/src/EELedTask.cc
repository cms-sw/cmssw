/*
 * \file EELedTask.cc
 *
 * $Date: 2012/05/15 11:12:27 $
 * $Revision: 1.73 $
 * \author G. Della Ricca
 *
*/

#include <iostream>
#include <sstream>
#include <vector>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/MonitorElement.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"

#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EcalElectronicsId.h"
#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "DQM/EcalCommon/interface/Numbers.h"
#include "DQM/EcalCommon/interface/NumbersPn.h"

#include "DQM/EcalEndcapMonitorTasks/interface/EELedTask.h"

EELedTask::EELedTask(const edm::ParameterSet& ps){

  init_ = false;

  dqmStore_ = edm::Service<DQMStore>().operator->();

  prefixME_ = ps.getUntrackedParameter<std::string>("prefixME", "");

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  mergeRuns_ = ps.getUntrackedParameter<bool>("mergeRuns", false);

  EcalRawDataCollection_ = ps.getParameter<edm::InputTag>("EcalRawDataCollection");
  EEDigiCollection_ = ps.getParameter<edm::InputTag>("EEDigiCollection");
  EcalPnDiodeDigiCollection_ = ps.getParameter<edm::InputTag>("EcalPnDiodeDigiCollection");
  EcalUncalibratedRecHitCollection_ = ps.getParameter<edm::InputTag>("EcalUncalibratedRecHitCollection");

  // vector of enabled wavelengths (Default to all 2)
  ledWavelengths_.reserve(2);
  for ( unsigned int i = 1; i <= 2; i++ ) ledWavelengths_.push_back(i);
  ledWavelengths_ = ps.getUntrackedParameter<std::vector<int> >("ledWavelengths", ledWavelengths_);

  for (int i = 0; i < 18; i++) {
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
  }

}

EELedTask::~EELedTask(){

}

void EELedTask::beginJob(void){

  ievt_ = 0;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EELedTask");
    dqmStore_->rmdir(prefixME_ + "/EELedTask");
  }

}

void EELedTask::beginRun(const edm::Run& r, const edm::EventSetup& c) {

  Numbers::initGeometry(c, false);

  if ( ! mergeRuns_ ) this->reset();

}

void EELedTask::endRun(const edm::Run& r, const edm::EventSetup& c) {

  for (int i = 0; i < 18; i++) {
    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 1) != ledWavelengths_.end() ) {
      if ( meShapeMapL1_[i] )  meShapeMapL1_[i]->Reset();
      if ( meAmplMapL1_[i] ) meAmplMapL1_[i]->Reset();
      if ( meTimeMapL1_[i] ) meTimeMapL1_[i]->Reset();
      if ( meAmplPNMapL1_[i] ) meAmplPNMapL1_[i]->Reset();
    }

    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 2) != ledWavelengths_.end() ) {
      if ( meShapeMapL2_[i] )  meShapeMapL2_[i]->Reset();
      if ( meAmplMapL2_[i] ) meAmplMapL2_[i]->Reset();
      if ( meTimeMapL2_[i] ) meTimeMapL2_[i]->Reset();
      if ( meAmplPNMapL2_[i] ) meAmplPNMapL2_[i]->Reset();
    }

    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 1) != ledWavelengths_.end() ) {
      if ( mePnAmplMapG01L1_[i] ) mePnAmplMapG01L1_[i]->Reset();
      if ( mePnPedMapG01L1_[i] ) mePnPedMapG01L1_[i]->Reset();

      if ( mePnAmplMapG16L1_[i] ) mePnAmplMapG16L1_[i]->Reset();
      if ( mePnPedMapG16L1_[i] ) mePnPedMapG16L1_[i]->Reset();
    }

    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 2) != ledWavelengths_.end() ) {
      if ( mePnAmplMapG01L2_[i] ) mePnAmplMapG01L2_[i]->Reset();
      if ( mePnPedMapG01L2_[i] ) mePnPedMapG01L2_[i]->Reset();

      if ( mePnAmplMapG16L2_[i] ) mePnAmplMapG16L2_[i]->Reset();
      if ( mePnPedMapG16L2_[i] ) mePnPedMapG16L2_[i]->Reset();
    }
  }

}

void EELedTask::reset(void) {

}

void EELedTask::setup(void){

  init_ = true;

  std::string name;
  std::stringstream LedN, LN;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EELedTask");

    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 1) != ledWavelengths_.end() ) {

      LedN.str("");
      LedN << "Led" << 1;
      LN.str("");
      LN << "L" << 1;

      dqmStore_->setCurrentFolder(prefixME_ + "/EELedTask/" + LedN.str());
      for (int i = 0; i < 18; i++) {
	name = "EELDT shape " + Numbers::sEE(i+1) + " " + LN.str();
        meShapeMapL1_[i] = dqmStore_->bookProfile2D(name, name, 850, 0., 850., 10, 0., 10., 4096, 0., 4096., "s");
        meShapeMapL1_[i]->setAxisTitle("channel", 1);
        meShapeMapL1_[i]->setAxisTitle("sample", 2);
        meShapeMapL1_[i]->setAxisTitle("amplitude", 3);
        dqmStore_->tag(meShapeMapL1_[i], i+1);

	name = "EELDT amplitude " + Numbers::sEE(i+1) + " " + LN.str(); 
        meAmplMapL1_[i] = dqmStore_->bookProfile2D(name, name, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 4096, 0., 4096.*12., "s");
        meAmplMapL1_[i]->setAxisTitle("ix", 1);
        if ( i+1 >= 1 && i+1 <= 9 ) meAmplMapL1_[i]->setAxisTitle("101-ix", 1);
        meAmplMapL1_[i]->setAxisTitle("iy", 2);
        dqmStore_->tag(meAmplMapL1_[i], i+1);

	name = "EELDT timing " + Numbers::sEE(i+1) + " " + LN.str(); 
        meTimeMapL1_[i] = dqmStore_->bookProfile2D(name, name, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 250, 0., 10., "s");
        meTimeMapL1_[i]->setAxisTitle("ix", 1);
        if ( i+1 >= 1 && i+1 <= 9 ) meTimeMapL1_[i]->setAxisTitle("101-ix", 1);
        meTimeMapL1_[i]->setAxisTitle("iy", 2);
        dqmStore_->tag(meTimeMapL1_[i], i+1);

	name = "EELDT amplitude over PN " + Numbers::sEE(i+1) + " " + LN.str(); 
        meAmplPNMapL1_[i] = dqmStore_->bookProfile2D(name, name, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 4096, 0., 4096.*12., "s");
        meAmplPNMapL1_[i]->setAxisTitle("ix", 1);
        if ( i+1 >= 1 && i+1 <= 9 ) meAmplPNMapL1_[i]->setAxisTitle("101-ix", 1);
        meAmplPNMapL1_[i]->setAxisTitle("iy", 2);
        dqmStore_->tag(meAmplPNMapL1_[i], i+1);
      }

      dqmStore_->setCurrentFolder(prefixME_ + "/EELedTask/" + LedN.str() + "/PN");

      dqmStore_->setCurrentFolder(prefixME_ + "/EELedTask/" + LedN.str() + "/PN/Gain01");
      for (int i = 0; i < 18; i++) {
	name = "EELDT PNs amplitude " + Numbers::sEE(i+1) + " G01 " + LN.str();
        mePnAmplMapG01L1_[i] = dqmStore_->bookProfile(name, name, 10, 0., 10., 4096, 0., 4096., "s");
        mePnAmplMapG01L1_[i]->setAxisTitle("channel", 1);
        mePnAmplMapG01L1_[i]->setAxisTitle("amplitude", 2);
        dqmStore_->tag(mePnAmplMapG01L1_[i], i+1);

	name = "EELDT PNs pedestal " + Numbers::sEE(i+1) + " G01 " + LN.str();
        mePnPedMapG01L1_[i] = dqmStore_->bookProfile(name, name, 10, 0., 10., 4096, 0., 4096., "s");
        mePnPedMapG01L1_[i]->setAxisTitle("channel", 1);
        mePnPedMapG01L1_[i]->setAxisTitle("pedestal", 2);
        dqmStore_->tag(mePnPedMapG01L1_[i], i+1);
      }

      dqmStore_->setCurrentFolder(prefixME_ + "/EELedTask/" + LedN.str() + "/PN/Gain16");

      for (int i = 0; i < 18; i++) {
	name = "EELDT PNs amplitude " + Numbers::sEE(i+1) + " G16 " + LN.str();
        mePnAmplMapG16L1_[i] = dqmStore_->bookProfile(name, name, 10, 0., 10., 4096, 0., 4096., "s");
        mePnAmplMapG16L1_[i]->setAxisTitle("channel", 1);
        mePnAmplMapG16L1_[i]->setAxisTitle("amplitude", 2);
        dqmStore_->tag(mePnAmplMapG16L1_[i], i+1);

	name = "EELDT PNs pedestal " + Numbers::sEE(i+1) + " G16 " + LN.str();
        mePnPedMapG16L1_[i] = dqmStore_->bookProfile(name, name, 10, 0., 10., 4096, 0., 4096., "s");
        mePnPedMapG16L1_[i]->setAxisTitle("channel", 1);
        mePnPedMapG16L1_[i]->setAxisTitle("pedestal", 2);
        dqmStore_->tag(mePnPedMapG16L1_[i], i+1);
      }

    }

    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 2) != ledWavelengths_.end() ) {

      LedN.str("");
      LedN << "Led" << 2;
      LN.str("");
      LN << "L" << 2;

      dqmStore_->setCurrentFolder(prefixME_ + "/EELedTask/" + LedN.str());
      for (int i = 0; i < 18; i++) {
	name = "EELDT shape " + Numbers::sEE(i+1) + " " + LN.str();
        meShapeMapL2_[i] = dqmStore_->bookProfile2D(name, name, 850, 0., 850., 10, 0., 10., 4096, 0., 4096., "s");
        meShapeMapL2_[i]->setAxisTitle("channel", 1);
        meShapeMapL2_[i]->setAxisTitle("sample", 2);
        meShapeMapL2_[i]->setAxisTitle("amplitude", 3);
        dqmStore_->tag(meShapeMapL2_[i], i+1);

	name = "EELDT amplitude " + Numbers::sEE(i+1) + " " + LN.str(); 
        meAmplMapL2_[i] = dqmStore_->bookProfile2D(name, name, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 4096, 0., 4096.*12., "s");
        meAmplMapL2_[i]->setAxisTitle("ix", 1);
        if ( i+1 >= 1 && i+1 <= 9 ) meAmplMapL2_[i]->setAxisTitle("101-ix", 1);
        meAmplMapL2_[i]->setAxisTitle("iy", 2);
        dqmStore_->tag(meAmplMapL2_[i], i+1);

	name = "EELDT timing " + Numbers::sEE(i+1) + " " + LN.str(); 
        meTimeMapL2_[i] = dqmStore_->bookProfile2D(name, name, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 250, 0., 10., "s");
        meTimeMapL2_[i]->setAxisTitle("ix", 1);
        if ( i+1 >= 1 && i+1 <= 9 ) meTimeMapL2_[i]->setAxisTitle("101-ix", 1);
        meTimeMapL2_[i]->setAxisTitle("iy", 2);
        dqmStore_->tag(meTimeMapL2_[i], i+1);

	name = "EELDT amplitude over PN " + Numbers::sEE(i+1) + " " + LN.str(); 
        meAmplPNMapL2_[i] = dqmStore_->bookProfile2D(name, name, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 4096, 0., 4096.*12., "s");
        meAmplPNMapL2_[i]->setAxisTitle("ix", 1);
        if ( i+1 >= 1 && i+1 <= 9 ) meAmplPNMapL2_[i]->setAxisTitle("101-ix", 1);
        meAmplPNMapL2_[i]->setAxisTitle("iy", 2);
        dqmStore_->tag(meAmplPNMapL2_[i], i+1);
      }

      dqmStore_->setCurrentFolder(prefixME_ + "/EELedTask/" + LedN.str() + "/PN");

      dqmStore_->setCurrentFolder(prefixME_ + "/EELedTask/" + LedN.str() + "/PN/Gain01");
      for (int i = 0; i < 18; i++) {
	name = "EELDT PNs amplitude " + Numbers::sEE(i+1) + " G01 " + LN.str();
        mePnAmplMapG01L2_[i] = dqmStore_->bookProfile(name, name, 10, 0., 10., 4096, 0., 4096., "s");
        mePnAmplMapG01L2_[i]->setAxisTitle("channel", 1);
        mePnAmplMapG01L2_[i]->setAxisTitle("amplitude", 2);
        dqmStore_->tag(mePnAmplMapG01L2_[i], i+1);

	name = "EELDT PNs pedestal " + Numbers::sEE(i+1) + " G01 " + LN.str();
        mePnPedMapG01L2_[i] = dqmStore_->bookProfile(name, name, 10, 0., 10., 4096, 0., 4096., "s");
        mePnPedMapG01L2_[i]->setAxisTitle("channel", 1);
        mePnPedMapG01L2_[i]->setAxisTitle("pedestal", 2);
        dqmStore_->tag(mePnPedMapG01L2_[i], i+1);
      }

      dqmStore_->setCurrentFolder(prefixME_ + "/EELedTask/" + LedN.str() + "/PN/Gain16");

      for (int i = 0; i < 18; i++) {
	name = "EELDT PNs amplitude " + Numbers::sEE(i+1) + " G16 " + LN.str();
        mePnAmplMapG16L2_[i] = dqmStore_->bookProfile(name, name, 10, 0., 10., 4096, 0., 4096., "s");
        mePnAmplMapG16L2_[i]->setAxisTitle("channel", 1);
        mePnAmplMapG16L2_[i]->setAxisTitle("amplitude", 2);
        dqmStore_->tag(mePnAmplMapG16L2_[i], i+1);

	name = "EELDT PNs pedestal " + Numbers::sEE(i+1) + " G16 " + LN.str();
        mePnPedMapG16L2_[i] = dqmStore_->bookProfile(name, name, 10, 0., 10., 4096, 0., 4096., "s");
        mePnPedMapG16L2_[i]->setAxisTitle("channel", 1);
        mePnPedMapG16L2_[i]->setAxisTitle("pedestal", 2);
        dqmStore_->tag(mePnPedMapG16L2_[i], i+1);
      }

    }

  }

}

void EELedTask::cleanup(void){

  if ( ! init_ ) return;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EELedTask");

    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 1) != ledWavelengths_.end() ) {

      dqmStore_->setCurrentFolder(prefixME_ + "/EELedTask/Led1");
      for (int i = 0; i < 18; i++) {
        if ( meShapeMapL1_[i] )  dqmStore_->removeElement( meShapeMapL1_[i]->getName() );
        meShapeMapL1_[i] = 0;
        if ( meAmplMapL1_[i] ) dqmStore_->removeElement( meAmplMapL1_[i]->getName() );
        meAmplMapL1_[i] = 0;
        if ( meTimeMapL1_[i] ) dqmStore_->removeElement( meTimeMapL1_[i]->getName() );
        meTimeMapL1_[i] = 0;
        if ( meAmplPNMapL1_[i] ) dqmStore_->removeElement( meAmplPNMapL1_[i]->getName() );
        meAmplPNMapL1_[i] = 0;
      }

    }

    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 2) != ledWavelengths_.end() ) {

      dqmStore_->setCurrentFolder(prefixME_ + "/EELedTask/Led2");
      for (int i = 0; i < 18; i++) {
        if ( meShapeMapL2_[i] )  dqmStore_->removeElement( meShapeMapL2_[i]->getName() );
        meShapeMapL2_[i] = 0;
        if ( meAmplMapL2_[i] ) dqmStore_->removeElement( meAmplMapL2_[i]->getName() );
        meAmplMapL2_[i] = 0;
        if ( meTimeMapL2_[i] ) dqmStore_->removeElement( meTimeMapL2_[i]->getName() );
        meTimeMapL2_[i] = 0;
        if ( meAmplPNMapL2_[i] ) dqmStore_->removeElement( meAmplPNMapL2_[i]->getName() );
        meAmplPNMapL2_[i] = 0;
      }

    }

    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 1) != ledWavelengths_.end() ) {

      dqmStore_->setCurrentFolder(prefixME_ + "/EELedTask/Led1/PN");

      dqmStore_->setCurrentFolder(prefixME_ + "/EELedTask/Led1/PN/Gain01");
      for (int i = 0; i < 18; i++) {
        if ( mePnAmplMapG01L1_[i] ) dqmStore_->removeElement( mePnAmplMapG01L1_[i]->getName() );
        mePnAmplMapG01L1_[i] = 0;
        if ( mePnPedMapG01L1_[i] ) dqmStore_->removeElement( mePnPedMapG01L1_[i]->getName() );
        mePnPedMapG01L1_[i] = 0;
      }

      dqmStore_->setCurrentFolder(prefixME_ + "/EELedTask/Led1/PN/Gain16");
      for (int i = 0; i < 18; i++) {
        if ( mePnAmplMapG16L1_[i] ) dqmStore_->removeElement( mePnAmplMapG16L1_[i]->getName() );
        mePnAmplMapG16L1_[i] = 0;
        if ( mePnPedMapG16L1_[i] ) dqmStore_->removeElement( mePnPedMapG16L1_[i]->getName() );
        mePnPedMapG16L1_[i] = 0;
      }

    }

    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 2) != ledWavelengths_.end() ) {

      dqmStore_->setCurrentFolder(prefixME_ + "/EELedTask/Led2/PN");

      dqmStore_->setCurrentFolder(prefixME_ + "/EELedTask/Led2/PN/Gain01");
      for (int i = 0; i < 18; i++) {
        if ( mePnAmplMapG01L2_[i] ) dqmStore_->removeElement( mePnAmplMapG01L2_[i]->getName() );
        mePnAmplMapG01L2_[i] = 0;
        if ( mePnPedMapG01L2_[i] ) dqmStore_->removeElement( mePnPedMapG01L2_[i]->getName() );
        mePnPedMapG01L2_[i] = 0;
      }

      dqmStore_->setCurrentFolder(prefixME_ + "/EELedTask/Led2/PN/Gain16");
      for (int i = 0; i < 18; i++) {
        if ( mePnAmplMapG16L2_[i] ) dqmStore_->removeElement( mePnAmplMapG16L2_[i]->getName() );
        mePnAmplMapG16L2_[i] = 0;
        if ( mePnPedMapG16L2_[i] ) dqmStore_->removeElement( mePnPedMapG16L2_[i]->getName() );
        mePnPedMapG16L2_[i] = 0;
      }

    }

  }

  init_ = false;

}

void EELedTask::endJob(void){

  edm::LogInfo("EELedTask") << "analyzed " << ievt_ << " events";

  if ( enableCleanup_ ) this->cleanup();

}

void EELedTask::analyze(const edm::Event& e, const edm::EventSetup& c){

  bool enable = false;
  int runType[18];
  for (int i=0; i<18; i++) runType[i] = -1;
  unsigned rtHalf[18];
  for (int i=0; i<18; i++) rtHalf[i] = -1;
  int waveLength[18];
  for (int i=0; i<18; i++) waveLength[i] = -1;

  edm::Handle<EcalRawDataCollection> dcchs;

  if ( e.getByLabel(EcalRawDataCollection_, dcchs) ) {

    for ( EcalRawDataCollection::const_iterator dcchItr = dcchs->begin(); dcchItr != dcchs->end(); ++dcchItr ) {

      if ( Numbers::subDet( *dcchItr ) != EcalEndcap ) continue;

      int ism = Numbers::iSM( *dcchItr, EcalEndcap );

      runType[ism-1] = dcchItr->getRunType();
      rtHalf[ism-1] = dcchItr->getRtHalf();
      waveLength[ism-1] = dcchItr->getEventSettings().wavelength;

      if ( dcchItr->getRunType() == EcalDCCHeaderBlock::LED_STD ||
           dcchItr->getRunType() == EcalDCCHeaderBlock::LED_GAP ) enable = true;

    }

  } else {

    edm::LogWarning("EELedTask") << EcalRawDataCollection_ << " not available";

  }

  if ( ! enable ) return;

  if ( ! init_ ) this->setup();

  ievt_++;

  bool numPN[80];
  float adcPN[80];
  for ( int i = 0; i < 80; i++ ) {
    numPN[i] = false;
    adcPN[i] = 0.;
  }

  std::vector<int> PNs;
  PNs.reserve(12);

  edm::Handle<EEDigiCollection> digis;

  if ( e.getByLabel(EEDigiCollection_, digis) ) {

    int maxpos[10];
    for(int i(0); i < 10; i++)
      maxpos[i] = 0;
    int nReadouts(0);

    for ( EEDigiCollection::const_iterator digiItr = digis->begin(); digiItr != digis->end(); ++digiItr ) {

      EEDetId id = digiItr->id();

      int ism = Numbers::iSM( id );

      if ( ! ( runType[ism-1] == EcalDCCHeaderBlock::LED_STD ||
               runType[ism-1] == EcalDCCHeaderBlock::LED_GAP ) ) continue;

      if ( rtHalf[ism-1] != Numbers::RtHalf(id) ) continue;

      nReadouts++;

      EEDataFrame dataframe = (*digiItr);

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

    int need = digis->size();
    LogDebug("EELedTask") << "event " << ievt_ << " digi collection size " << need;

    for ( EEDigiCollection::const_iterator digiItr = digis->begin(); digiItr != digis->end(); ++digiItr ) {

      EEDetId id = digiItr->id();

      int ix = id.ix();
      int iy = id.iy();

      int ism = Numbers::iSM( id );

      if ( ! ( runType[ism-1] == EcalDCCHeaderBlock::LED_STD ||
               runType[ism-1] == EcalDCCHeaderBlock::LED_GAP ) ) continue;

      if ( runType[ism-1] == EcalDCCHeaderBlock::LED_GAP &&
           rtHalf[ism-1] != Numbers::RtHalf(id) ) continue;

      int ic = Numbers::icEE(ism, ix, iy);

      EEDataFrame dataframe = (*digiItr);

      for (int i = 0; i < 10; i++) {

        int adc = dataframe.sample(i).adc();

        MonitorElement* meShapeMap = 0;

        if ( Numbers::RtHalf(id) == 0 || Numbers::RtHalf(id) == 1 ) {

          if ( waveLength[ism-1] == 0 ) meShapeMap = meShapeMapL1_[ism-1];
          if ( waveLength[ism-1] == 2 ) meShapeMap = meShapeMapL2_[ism-1];

        } else {

          edm::LogWarning("EELedTask") << " RtHalf = " << Numbers::RtHalf(id);

        }

        float xval = float(adc);

        if ( meShapeMap ) meShapeMap->Fill(ic - 0.5, i + 0.5, xval);

      }

      NumbersPn::getPNs( ism, ix, iy, PNs );

      for (unsigned int i=0; i<PNs.size(); i++) {
        int ipn = PNs[i];
        if ( ipn >= 0 && ipn < 80 ) numPN[ipn] = true;
      }

    }

  } else {

    edm::LogWarning("EELedTask") << EEDigiCollection_ << " not available";

  }

  edm::Handle<EcalPnDiodeDigiCollection> pns;

  if ( e.getByLabel(EcalPnDiodeDigiCollection_, pns) ) {

    int nep = pns->size();
    LogDebug("EELedTask") << "event " << ievt_ << " pns collection size " << nep;

    for ( EcalPnDiodeDigiCollection::const_iterator pnItr = pns->begin(); pnItr != pns->end(); ++pnItr ) {

      if ( Numbers::subDet( pnItr->id() ) != EcalEndcap ) continue;

      int ism = Numbers::iSM( pnItr->id() );

      int num = pnItr->id().iPnId();

      if ( ! ( runType[ism-1] == EcalDCCHeaderBlock::LED_STD ||
               runType[ism-1] == EcalDCCHeaderBlock::LED_GAP ) ) continue;

      int ipn = NumbersPn::ipnEE( ism, num );

      if ( ipn >= 0 && ipn < 80 && numPN[ipn] == false ) continue;

      float xvalped = 0.;

      for (int i = 0; i < 4; i++) {

        int adc = pnItr->sample(i).adc();

        MonitorElement* mePNPed = 0;

        if ( pnItr->sample(i).gainId() == 0 ) {
          if ( waveLength[ism-1] == 0 ) mePNPed = mePnPedMapG01L1_[ism-1];
          if ( waveLength[ism-1] == 2 ) mePNPed = mePnPedMapG01L2_[ism-1];
        }
        if ( pnItr->sample(i).gainId() == 1 ) {
          if ( waveLength[ism-1] == 0 ) mePNPed = mePnPedMapG16L1_[ism-1];
          if ( waveLength[ism-1] == 2 ) mePNPed = mePnPedMapG16L2_[ism-1];
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
        if ( waveLength[ism-1] == 2 ) mePN = mePnAmplMapG01L2_[ism-1];
      }
      if ( pnItr->sample(0).gainId() == 1 ) {
        if ( waveLength[ism-1] == 0 ) mePN = mePnAmplMapG16L1_[ism-1];
        if ( waveLength[ism-1] == 2 ) mePN = mePnAmplMapG16L2_[ism-1];
      }

      if ( mePN ) mePN->Fill(num - 0.5, xvalmax);

      if ( ipn >= 0 && ipn < 80 ) adcPN[ipn] = xvalmax;

    }

  } else {

    edm::LogWarning("EELedTask") << EcalPnDiodeDigiCollection_ << " not available";

  }

  edm::Handle<EcalUncalibratedRecHitCollection> hits;

  if ( e.getByLabel(EcalUncalibratedRecHitCollection_, hits) ) {

    int neh = hits->size();
    LogDebug("EELedTask") << "event " << ievt_ << " hits collection size " << neh;

    for ( EcalUncalibratedRecHitCollection::const_iterator hitItr = hits->begin(); hitItr != hits->end(); ++hitItr ) {

      EEDetId id = hitItr->id();

      int ix = id.ix();
      int iy = id.iy();

      int ism = Numbers::iSM( id );

      if ( ism >= 1 && ism <= 9 ) ix = 101 - ix;

      float xix = ix - 0.5;
      float xiy = iy - 0.5;

      if ( ! ( runType[ism-1] == EcalDCCHeaderBlock::LED_STD ||
               runType[ism-1] == EcalDCCHeaderBlock::LED_GAP ) ) continue;

      if ( runType[ism-1] == EcalDCCHeaderBlock::LED_GAP &&
           rtHalf[ism-1] != Numbers::RtHalf(id) ) continue;

      MonitorElement* meAmplMap = 0;
      MonitorElement* meTimeMap = 0;
      MonitorElement* meAmplPNMap = 0;

      // Temporary measure to remove broken LED boxes for L1
      if(waveLength[ism - 1] == 0){
	if(ism == 14){
	  EcalElectronicsId eid(Numbers::getElectronicsMapping()->getElectronicsId(id));
	  int tower(eid.towerId());
	  if(tower == 1 || tower == 2 || tower == 3 || tower == 4 || tower == 5 || tower == 6 || tower == 9 || tower == 15) continue;
	}
	else if(ism == 15){
	  EcalElectronicsId eid(Numbers::getElectronicsMapping()->getElectronicsId(id));
	  int tower(eid.towerId());
	  if(tower == 3 || tower == 4 || tower == 10 || tower == 11 || tower == 12 || tower == 18 || tower == 19 || tower == 25) continue;
	}
      }

      if ( Numbers::RtHalf(id) == 0 || Numbers::RtHalf(id) == 1 ) {

        if ( waveLength[ism-1] == 0 ) {
          meAmplMap = meAmplMapL1_[ism-1];
          meTimeMap = meTimeMapL1_[ism-1];
          meAmplPNMap = meAmplPNMapL1_[ism-1];
        }
        if ( waveLength[ism-1] == 2 ) {
          meAmplMap = meAmplMapL2_[ism-1];
          meTimeMap = meTimeMapL2_[ism-1];
          meAmplPNMap = meAmplPNMapL2_[ism-1];
        }

      } else {

        edm::LogWarning("EELedTask") << " RtHalf = " << Numbers::RtHalf(id);

      }

      float xval = hitItr->amplitude();
      if ( xval <= 0. ) xval = 0.0;
      float yval = hitItr->jitter() + 6.0;
      if ( yval <= 0. ) yval = 0.0;
      float zval = hitItr->pedestal();
      if ( zval <= 0. ) zval = 0.0;

      if ( meAmplMap ) meAmplMap->Fill(xix, xiy, xval);

      if ( xval > 16. ) {
        if ( meTimeMap ) meTimeMap->Fill(xix, xiy, yval);
      }

      float wval = 0.;

      NumbersPn::getPNs( ism, ix, iy, PNs );

      if ( PNs.size() > 0 ) {
        int ipn = PNs[0];
        if ( ipn >= 0 && ipn < 80 ) {
          if ( adcPN[ipn] != 0. ) wval = xval / adcPN[ipn];
        }
      }

      if ( meAmplPNMap ) meAmplPNMap->Fill(xix, xiy, wval);

    }

  } else {

    edm::LogWarning("EELedTask") << EcalUncalibratedRecHitCollection_ << " not available";

  }

}

