/*
 * \file EELedTask.cc
 *
 * $Date: 2007/08/14 17:44:47 $
 * $Revision: 1.5 $
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

#include <DQM/EcalEndcapMonitorTasks/interface/EELedTask.h>

using namespace cms;
using namespace edm;
using namespace std;

EELedTask::EELedTask(const ParameterSet& ps){

  init_ = false;

  // get hold of back-end interface
  dbe_ = Service<DaqMonitorBEInterface>().operator->();

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", true);

  EcalRawDataCollection_ = ps.getParameter<edm::InputTag>("EcalRawDataCollection");
  EEDigiCollection_ = ps.getParameter<edm::InputTag>("EEDigiCollection");
  EcalPnDiodeDigiCollection_ = ps.getParameter<edm::InputTag>("EcalPnDiodeDigiCollection");
  EcalUncalibratedRecHitCollection_ = ps.getParameter<edm::InputTag>("EcalUncalibratedRecHitCollection");

  for (int i = 0; i < 18 ; i++) {
    meShapeMapA_[i] = 0;
    meAmplMapA_[i] = 0;
    meTimeMapA_[i] = 0;
    meAmplPNMapA_[i] = 0;
    meShapeMapB_[i] = 0;
    meAmplMapB_[i] = 0;
    meTimeMapB_[i] = 0;
    meAmplPNMapB_[i] = 0;
    mePnAmplMapG01_[i] = 0;
    mePnPedMapG01_[i] = 0;
    mePnAmplMapG16_[i] = 0;
    mePnPedMapG16_[i] = 0;
  }

}

EELedTask::~EELedTask(){

}

void EELedTask::beginJob(const EventSetup& c){

  ievt_ = 0;

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalEndcap/EELedTask");
    dbe_->rmdir("EcalEndcap/EELedTask");
  }

}

void EELedTask::setup(void){

  init_ = true;

  Char_t histo[200];

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalEndcap/EELedTask");

    for (int i = 0; i < 18 ; i++) {
      sprintf(histo, "EELDT shape %s A", Numbers::sEE(i+1).c_str());
      meShapeMapA_[i] = dbe_->bookProfile2D(histo, histo, 1700, 0., 1700., 10, 0., 10., 4096, 0., 4096., "s");
      dbe_->tag(meShapeMapA_[i], i+1);
      sprintf(histo, "EELDT amplitude %s A", Numbers::sEE(i+1).c_str());
      meAmplMapA_[i] = dbe_->bookProfile2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 4096, 0., 4096.*12., "s");
      dbe_->tag(meAmplMapA_[i], i+1);
      sprintf(histo, "EELDT timing %s A", Numbers::sEE(i+1).c_str());
      meTimeMapA_[i] = dbe_->bookProfile2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 250, 0., 10., "s");
      dbe_->tag(meTimeMapA_[i], i+1);
      sprintf(histo, "EELDT amplitude over PN %s A", Numbers::sEE(i+1).c_str());
      meAmplPNMapA_[i] = dbe_->bookProfile2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 4096, 0., 4096.*12., "s");
      dbe_->tag(meAmplPNMapA_[i], i+1);

      sprintf(histo, "EELDT shape %s B", Numbers::sEE(i+1).c_str());
      meShapeMapB_[i] = dbe_->bookProfile2D(histo, histo, 1700, 0., 1700., 10, 0., 10., 4096, 0., 4096., "s");
      dbe_->tag(meShapeMapB_[i], i+1);
      sprintf(histo, "EELDT amplitude %s B", Numbers::sEE(i+1).c_str());
      meAmplMapB_[i] = dbe_->bookProfile2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 4096, 0., 4096.*12., "s");
      dbe_->tag(meAmplMapB_[i], i+1);
      sprintf(histo, "EELDT timing %s B", Numbers::sEE(i+1).c_str());
      meTimeMapB_[i] = dbe_->bookProfile2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 250, 0., 10., "s");
      dbe_->tag(meTimeMapB_[i], i+1);
      sprintf(histo, "EELDT amplitude over PN %s B", Numbers::sEE(i+1).c_str());
      meAmplPNMapB_[i] = dbe_->bookProfile2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 4096, 0., 4096.*12., "s");
      dbe_->tag(meAmplPNMapB_[i], i+1);
    }

    dbe_->setCurrentFolder("EcalEndcap/EELedTask/PN/Gain01");
    for (int i = 0; i < 18 ; i++) {
      sprintf(histo, "EEPDT PNs amplitude %s G01", Numbers::sEE(i+1).c_str());
      mePnAmplMapG01_[i] = dbe_->bookProfile2D(histo, histo, 1, 0., 1., 10, 0., 10., 4096, 0., 4096., "s");
      dbe_->tag(mePnAmplMapG01_[i], i+1);
      sprintf(histo, "EEPDT PNs pedestal %s G01", Numbers::sEE(i+1).c_str());
      mePnPedMapG01_[i] = dbe_->bookProfile2D(histo, histo, 1, 0., 1., 10, 0., 10., 4096, 0., 4096., "s");
      dbe_->tag(mePnPedMapG01_[i], i+1);
    }

    dbe_->setCurrentFolder("EcalEndcap/EELedTask/PN/Gain16");
    for (int i = 0; i < 18 ; i++) {
      sprintf(histo, "EEPDT PNs amplitude %s G16", Numbers::sEE(i+1).c_str());
      mePnAmplMapG16_[i] = dbe_->bookProfile2D(histo, histo, 1, 0., 1., 10, 0., 10., 4096, 0., 4096., "s");
      dbe_->tag(mePnAmplMapG16_[i], i+1);
      sprintf(histo, "EEPDT PNs pedestal %s G16", Numbers::sEE(i+1).c_str());
      mePnPedMapG16_[i] = dbe_->bookProfile2D(histo, histo, 1, 0., 1., 10, 0., 10., 4096, 0., 4096., "s");
      dbe_->tag(mePnPedMapG16_[i], i+1);
    }

  }

}

void EELedTask::cleanup(void){

  if ( ! enableCleanup_ ) return;

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalEndcap/EELedTask");

    for (int i = 0; i < 18 ; i++) {
      if ( meShapeMapA_[i] )  dbe_->removeElement( meShapeMapA_[i]->getName() );
      meShapeMapA_[i] = 0;
      if ( meAmplMapA_[i] ) dbe_->removeElement( meAmplMapA_[i]->getName() );
      meAmplMapA_[i] = 0;
      if ( meTimeMapA_[i] ) dbe_->removeElement( meTimeMapA_[i]->getName() );
      meTimeMapA_[i] = 0;
      if ( meAmplPNMapA_[i] ) dbe_->removeElement( meAmplPNMapA_[i]->getName() );
      meAmplPNMapA_[i] = 0;

      if ( meShapeMapB_[i] )  dbe_->removeElement( meShapeMapB_[i]->getName() );
      meShapeMapB_[i] = 0;
      if ( meAmplMapB_[i] ) dbe_->removeElement( meAmplMapB_[i]->getName() );
      meAmplMapB_[i] = 0;
      if ( meTimeMapB_[i] ) dbe_->removeElement( meTimeMapB_[i]->getName() );
      meTimeMapB_[i] = 0;
      if ( meAmplPNMapB_[i] ) dbe_->removeElement( meAmplPNMapB_[i]->getName() );
      meAmplPNMapB_[i] = 0;
    }

    dbe_->setCurrentFolder("EcalEndcap/EELedTask/PN/Gain01");
    for (int i = 0; i < 18 ; i++) {
      if ( mePnAmplMapG01_[i] ) dbe_->removeElement( mePnAmplMapG01_[i]->getName() );
      mePnAmplMapG01_[i] = 0;
      if ( mePnPedMapG01_[i] ) dbe_->removeElement( mePnPedMapG01_[i]->getName() );
      mePnPedMapG01_[i] = 0;
    }

    dbe_->setCurrentFolder("EcalEndcap/EELedTask/PN/Gain16");
    for (int i = 0; i < 18 ; i++) {
      if ( mePnAmplMapG16_[i] ) dbe_->removeElement( mePnAmplMapG16_[i]->getName() );
      mePnAmplMapG16_[i] = 0;
      if ( mePnPedMapG16_[i] ) dbe_->removeElement( mePnPedMapG16_[i]->getName() );
      mePnPedMapG16_[i] = 0;
    }

  }

  init_ = false;

}

void EELedTask::endJob(void){

  LogInfo("EELedTask") << "analyzed " << ievt_ << " events";

  if ( init_ ) this->cleanup();

}

void EELedTask::analyze(const Event& e, const EventSetup& c){

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

      if ( dcch.getRunType() == EcalDCCHeaderBlock::LED_STD ||
           dcch.getRunType() == EcalDCCHeaderBlock::LED_GAP ) enable = true;

    }

  } catch ( exception& ex) {

    LogWarning("EELedTask") << EcalRawDataCollection_ << " not available";

  }

  if ( ! enable ) return;

  if ( ! init_ ) this->setup();

  ievt_++;

  try {

    Handle<EEDigiCollection> digis;
    e.getByLabel(EEDigiCollection_, digis);

    int need = digis->size();
    LogDebug("EELedTask") << "event " << ievt_ << " digi collection size " << need;

    for ( EEDigiCollection::const_iterator digiItr = digis->begin(); digiItr != digis->end(); ++digiItr ) {

      EEDataFrame dataframe = (*digiItr);
      EEDetId id = dataframe.id();

      int ix = 101 - id.ix();
      int iy = id.iy();
  
      int ism = Numbers::iSM( id );
  
      map<int, EcalDCCHeaderBlock>::iterator i = dccMap.find(ism);
      if ( i == dccMap.end() ) continue;

      if ( ! ( dccMap[ism].getRunType() == EcalDCCHeaderBlock::LED_STD ||
               dccMap[ism].getRunType() == EcalDCCHeaderBlock::LED_GAP ) ) continue;

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

        if ( ix < 6 || iy > 10 ) {

          meShapeMap = meShapeMapA_[ism-1];

        } else {

          meShapeMap = meShapeMapB_[ism-1];

        }

//        float xval = float(adc) * gain;
        float xval = float(adc);

        if ( meShapeMap ) meShapeMap->Fill(ic - 0.5, i + 0.5, xval);

      }

    }

  } catch ( exception& ex) {

    LogWarning("EELedTask") << EEDigiCollection_ << " not available";

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
    LogDebug("EELedTask") << "event " << ievt_ << " pns collection size " << nep;

    for ( EcalPnDiodeDigiCollection::const_iterator pnItr = pns->begin(); pnItr != pns->end(); ++pnItr ) {

      EcalPnDiodeDigi pn = (*pnItr);
      EcalPnDiodeDetId id = pn.id();

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
          mePNPed = mePnPedMapG01_[ism-1];
        }
        if ( sample.gainId() == 1 ) {
          mePNPed = mePnPedMapG16_[ism-1];
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
        mePN = mePnAmplMapG01_[ism-1];
      }
      if ( pn.sample(0).gainId() == 1 ) {
        mePN = mePnAmplMapG16_[ism-1];
      }

      if ( mePN ) mePN->Fill(0.5, num - 0.5, xvalmax);

      if ( num == 1 ) adcA[ism-1] = xvalmax;
      if ( num == 6 ) adcB[ism-1] = xvalmax;

    }

  } catch ( exception& ex) {

    LogWarning("EELedTask") << EcalPnDiodeDigiCollection_ << " not available";

  }

  try {

    Handle<EcalUncalibratedRecHitCollection> hits;
    e.getByLabel(EcalUncalibratedRecHitCollection_, hits);

    int neh = hits->size();
    LogDebug("EELedTask") << "event " << ievt_ << " hits collection size " << neh;

    for ( EcalUncalibratedRecHitCollection::const_iterator hitItr = hits->begin(); hitItr != hits->end(); ++hitItr ) {

      EcalUncalibratedRecHit hit = (*hitItr);
      EEDetId id = hit.id();

      int ix = 101 - id.ix();
      int iy = id.iy();

      int ism = Numbers::iSM( id );

      float xix = ix - 0.5;
      float xiy = iy - 0.5;

      map<int, EcalDCCHeaderBlock>::iterator i = dccMap.find(ism);
      if ( i == dccMap.end() ) continue;

      if ( ! ( dccMap[ism].getRunType() == EcalDCCHeaderBlock::LED_STD ||
               dccMap[ism].getRunType() == EcalDCCHeaderBlock::LED_GAP ) ) continue;

      LogDebug("EELedTask") << " det id = " << id;
      LogDebug("EELedTask") << " sm, ix, iy " << ism << " " << ix << " " << iy;

      MonitorElement* meAmplMap = 0;
      MonitorElement* meTimeMap = 0;
      MonitorElement* meAmplPNMap = 0;

      if ( ix < 6 || iy > 10 ) {

        meAmplMap = meAmplMapA_[ism-1];
        meTimeMap = meTimeMapA_[ism-1];
        meAmplPNMap = meAmplPNMapA_[ism-1];

      } else {

        meAmplMap = meAmplMapB_[ism-1];
        meTimeMap = meTimeMapB_[ism-1];
        meAmplPNMap = meAmplPNMapB_[ism-1];

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

      if ( ix < 6 || iy > 10 ) {

        if ( adcA[ism-1] != 0. ) wval = xval / adcA[ism-1];

      } else {

        if ( adcB[ism-1] != 0. ) wval = xval / adcB[ism-1];

      }

      LogDebug("EELedTask") << " hit amplitude over PN " << wval;

      if ( meAmplPNMap ) meAmplPNMap->Fill(xix, xiy, wval);

    }

  } catch ( exception& ex) {

    LogWarning("EELedTask") << EcalUncalibratedRecHitCollection_ << " not available";

  }

}

