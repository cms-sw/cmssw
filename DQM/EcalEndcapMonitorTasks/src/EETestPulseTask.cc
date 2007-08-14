/*
 * \file EETestPulseTask.cc
 *
 * $Date: 2007/07/21 10:13:26 $
 * $Revision: 1.14 $
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

#include <DQM/EcalEndcapMonitorTasks/interface/EETestPulseTask.h>

using namespace cms;
using namespace edm;
using namespace std;

EETestPulseTask::EETestPulseTask(const ParameterSet& ps){

  init_ = false;

  // get hold of back-end interface
  dbe_ = Service<DaqMonitorBEInterface>().operator->();

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", true);

  EcalRawDataCollection_ = ps.getParameter<edm::InputTag>("EcalRawDataCollection");
  EEDigiCollection_ = ps.getParameter<edm::InputTag>("EEDigiCollection");
  EcalPnDiodeDigiCollection_ = ps.getParameter<edm::InputTag>("EcalPnDiodeDigiCollection");
  EcalUncalibratedRecHitCollection_ = ps.getParameter<edm::InputTag>("EcalUncalibratedRecHitCollection");

  for (int i = 0; i < 18 ; i++) {
    meShapeMapG01_[i] = 0;
    meAmplMapG01_[i] = 0;
    meShapeMapG06_[i] = 0;
    meAmplMapG06_[i] = 0;
    meShapeMapG12_[i] = 0;
    meAmplMapG12_[i] = 0;
    mePnAmplMapG01_[i] = 0;
    mePnPedMapG01_[i] = 0;
    mePnAmplMapG16_[i] = 0;
    mePnPedMapG16_[i] = 0;
  }

}

EETestPulseTask::~EETestPulseTask(){

}

void EETestPulseTask::beginJob(const EventSetup& c){

  ievt_ = 0;

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalEndcap/EETestPulseTask");
    dbe_->rmdir("EcalEndcap/EETestPulseTask");
  }

}

void EETestPulseTask::setup(void){

  init_ = true;

  Char_t histo[200];

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalEndcap/EETestPulseTask");

    dbe_->setCurrentFolder("EcalEndcap/EETestPulseTask/Gain01");
    for (int i = 0; i < 18 ; i++) {
      sprintf(histo, "EETPT shape %s G01", Numbers::sEE(i+1).c_str());
      meShapeMapG01_[i] = dbe_->bookProfile2D(histo, histo, 1700, 0., 1700., 10, 0., 10., 4096, 0., 4096., "s");
      dbe_->tag(meShapeMapG01_[i], i+1);
      sprintf(histo, "EETPT amplitude %s G01", Numbers::sEE(i+1).c_str());
      meAmplMapG01_[i] = dbe_->bookProfile2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 4096, 0., 4096.*12., "s");
      dbe_->tag(meAmplMapG01_[i], i+1);
    }

    dbe_->setCurrentFolder("EcalEndcap/EETestPulseTask/Gain06");
    for (int i = 0; i < 18 ; i++) {
      sprintf(histo, "EETPT shape %s G06", Numbers::sEE(i+1).c_str());
      meShapeMapG06_[i] = dbe_->bookProfile2D(histo, histo, 1700, 0., 1700., 10, 0., 10., 4096, 0., 4096., "s");
      dbe_->tag(meShapeMapG06_[i], i+1);
      sprintf(histo, "EETPT amplitude %s G06", Numbers::sEE(i+1).c_str());
      meAmplMapG06_[i] = dbe_->bookProfile2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 4096, 0., 4096.*12., "s");
      dbe_->tag(meAmplMapG06_[i], i+1);
    }

    dbe_->setCurrentFolder("EcalEndcap/EETestPulseTask/Gain12");
    for (int i = 0; i < 18 ; i++) {
      sprintf(histo, "EETPT shape %s G12", Numbers::sEE(i+1).c_str());
      meShapeMapG12_[i] = dbe_->bookProfile2D(histo, histo, 1700, 0., 1700., 10, 0., 10., 4096, 0., 4096., "s");
      dbe_->tag(meShapeMapG12_[i], i+1);
      sprintf(histo, "EETPT amplitude %s G12", Numbers::sEE(i+1).c_str());
      meAmplMapG12_[i] = dbe_->bookProfile2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 4096, 0., 4096.*12., "s");
      dbe_->tag(meAmplMapG12_[i], i+1);
   }

    dbe_->setCurrentFolder("EcalEndcap/EETestPulseTask/PN");

    dbe_->setCurrentFolder("EcalEndcap/EETestPulseTask/PN/Gain01");
    for (int i = 0; i < 18 ; i++) {
      sprintf(histo, "EEPDT PNs amplitude %s G01", Numbers::sEE(i+1).c_str());
      mePnAmplMapG01_[i] = dbe_->bookProfile2D(histo, histo, 1, 0., 1., 10, 0., 10., 4096, 0., 4096., "s");
      dbe_->tag(mePnAmplMapG01_[i], i+1);
      sprintf(histo, "EEPDT PNs pedestal %s G01", Numbers::sEE(i+1).c_str());
      mePnPedMapG01_[i] =  dbe_->bookProfile2D(histo, histo, 1, 0., 1., 10, 0., 10., 4096, 0., 4096., "s");
      dbe_->tag(mePnPedMapG01_[i], i+1);
    }

    dbe_->setCurrentFolder("EcalEndcap/EETestPulseTask/PN/Gain16");
    for (int i = 0; i < 18 ; i++) {
      sprintf(histo, "EEPDT PNs amplitude %s G16", Numbers::sEE(i+1).c_str());
      mePnAmplMapG16_[i] = dbe_->bookProfile2D(histo, histo, 1, 0., 1., 10, 0., 10., 4096, 0., 4096., "s");
      dbe_->tag(mePnAmplMapG16_[i], i+1);
      sprintf(histo, "EEPDT PNs pedestal %s G16", Numbers::sEE(i+1).c_str());
      mePnPedMapG16_[i] =  dbe_->bookProfile2D(histo, histo, 1, 0., 1., 10, 0., 10., 4096, 0., 4096., "s");
      dbe_->tag(mePnPedMapG16_[i], i+1);
    }

  }

}

void EETestPulseTask::cleanup(void){

  if ( ! enableCleanup_ ) return;

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalEndcap/EETestPulseTask");

    dbe_->setCurrentFolder("EcalEndcap/EETestPulseTask/Gain01");
    for (int i = 0; i < 18 ; i++) {
      if ( meShapeMapG01_[i] ) dbe_->removeElement( meShapeMapG01_[i]->getName() );
      meShapeMapG01_[i] = 0;
      if ( meAmplMapG01_[i] ) dbe_->removeElement( meAmplMapG01_[i]->getName() );
      meAmplMapG01_[i] = 0;
    }

    dbe_->setCurrentFolder("EcalEndcap/EETestPulseTask/Gain06");
    for (int i = 0; i < 18 ; i++) {
      if ( meShapeMapG06_[i] ) dbe_->removeElement( meShapeMapG06_[i]->getName() );
      meShapeMapG06_[i] = 0;
      if ( meAmplMapG06_[i] ) dbe_->removeElement( meAmplMapG06_[i]->getName() );
      meAmplMapG06_[i] = 0;
    }

    dbe_->setCurrentFolder("EcalEndcap/EETestPulseTask/Gain12");
    for (int i = 0; i < 18 ; i++) {
      if ( meShapeMapG12_[i] ) dbe_->removeElement( meShapeMapG12_[i]->getName() );
      meShapeMapG12_[i] = 0;
      if ( meAmplMapG12_[i] ) dbe_->removeElement( meAmplMapG12_[i]->getName() );
      meAmplMapG12_[i] = 0;
    }

    dbe_->setCurrentFolder("EcalEndcap/EETestPulseTask/PN");

    dbe_->setCurrentFolder("EcalEndcap/EETestPulseTask/PN/Gain01");
    for (int i = 0; i < 18 ; i++) {
      if ( mePnAmplMapG01_[i] ) dbe_->removeElement( mePnAmplMapG01_[i]->getName() );
      mePnAmplMapG01_[i] = 0;
      if ( mePnPedMapG01_[i] ) dbe_->removeElement( mePnPedMapG01_[i]->getName() );
      mePnPedMapG01_[i] = 0;
    }

    dbe_->setCurrentFolder("EcalEndcap/EETestPulseTask/PN/Gain16");
    for (int i = 0; i < 18 ; i++) {
      if ( mePnAmplMapG16_[i] ) dbe_->removeElement( mePnAmplMapG16_[i]->getName() );
      mePnAmplMapG16_[i] = 0;
      if ( mePnPedMapG16_[i] ) dbe_->removeElement( mePnPedMapG16_[i]->getName() );
      mePnPedMapG16_[i] = 0;
    }

  }

  init_ = false;

}

void EETestPulseTask::endJob(void){

  LogInfo("EETestPulseTask") << "analyzed " << ievt_ << " events";

  if ( init_ ) this->cleanup();

}

void EETestPulseTask::analyze(const Event& e, const EventSetup& c){

  Numbers::initGeometry(c);

  bool enable = false;
  map<int, EcalDCCHeaderBlock> dccMap;

  try {

    Handle<EcalRawDataCollection> dcchs;
    e.getByLabel(EcalRawDataCollection_, dcchs);

    for ( EcalRawDataCollection::const_iterator dcchItr = dcchs->begin(); dcchItr != dcchs->end(); ++dcchItr ) {

      EcalDCCHeaderBlock dcch = (*dcchItr);

      int ism = Numbers::iSM( dcch );

      map<int, EcalDCCHeaderBlock>::iterator i = dccMap.find( ism );
      if ( i != dccMap.end() ) continue;

      dccMap[ ism ] = dcch;

      if ( dcch.getRunType() == EcalDCCHeaderBlock::TESTPULSE_MGPA ||
           dcch.getRunType() == EcalDCCHeaderBlock::TESTPULSE_GAP ) enable = true;

    }

  } catch ( exception& ex) {

    LogWarning("EETestPulseTask") << EcalRawDataCollection_ << " not available";

  }

  if ( ! enable ) return;

  if ( ! init_ ) this->setup();

  ievt_++;

  try {

    Handle<EEDigiCollection> digis;
    e.getByLabel(EEDigiCollection_, digis);

    int need = digis->size();
    LogDebug("EETestPulseTask") << "event " << ievt_ << " digi collection size " << need;

    for ( EEDigiCollection::const_iterator digiItr = digis->begin(); digiItr != digis->end(); ++digiItr ) {

      EEDataFrame dataframe = (*digiItr);
      EEDetId id = dataframe.id();

      int ix = 101 - id.ix();
      int iy = id.iy();

      int ism = Numbers::iSM( id );
  
      map<int, EcalDCCHeaderBlock>::iterator i = dccMap.find(ism);
      if ( i == dccMap.end() ) continue;

      if ( ! ( dccMap[ism].getRunType() != EcalDCCHeaderBlock::TESTPULSE_MGPA ||
               dccMap[ism].getRunType() != EcalDCCHeaderBlock::TESTPULSE_GAP ) ) continue;

      LogDebug("EETestPulseTask") << " det id = " << id;
      LogDebug("EETestPulseTask") << " sm, ix, iy " << ism << " " << ix << " " << iy;

      int ic = Numbers::icEE(ism, ix, iy);

      for (int i = 0; i < 10; i++) {

        EcalMGPASample sample = dataframe.sample(i);
        int adc = sample.adc();
        float gain = 1.;

        MonitorElement* meShapeMap = 0;

        if ( sample.gainId() == 1 ) gain = 1./12.;
        if ( sample.gainId() == 2 ) gain = 1./ 6.;
        if ( sample.gainId() == 3 ) gain = 1./ 1.;

        if ( dccMap[ism].getMgpaGain() == 3 ) meShapeMap = meShapeMapG01_[ism-1];
        if ( dccMap[ism].getMgpaGain() == 2 ) meShapeMap = meShapeMapG06_[ism-1];
        if ( dccMap[ism].getMgpaGain() == 1 ) meShapeMap = meShapeMapG12_[ism-1];

//        float xval = float(adc) * gain;
        float xval = float(adc);

        if ( meShapeMap ) meShapeMap->Fill(ic - 0.5, i + 0.5, xval);

      }

    }

  } catch ( exception& ex) {

    LogWarning("EETestPulseTask") << EEDigiCollection_ << " not available";

  }

  try {

    Handle<EcalUncalibratedRecHitCollection> hits;
    e.getByLabel(EcalUncalibratedRecHitCollection_, hits);

    int neh = hits->size();
    LogDebug("EETestPulseTask") << "event " << ievt_ << " hits collection size " << neh;

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

      if ( ! ( dccMap[ism].getRunType() != EcalDCCHeaderBlock::TESTPULSE_MGPA ||
               dccMap[ism].getRunType() != EcalDCCHeaderBlock::TESTPULSE_GAP ) ) continue;

      LogDebug("EETestPulseTask") << " det id = " << id;
      LogDebug("EETestPulseTask") << " sm, ix, iy " << ism << " " << ix << " " << iy;

      MonitorElement* meAmplMap = 0;

      if ( dccMap[ism].getMgpaGain() == 3 ) meAmplMap = meAmplMapG01_[ism-1];
      if ( dccMap[ism].getMgpaGain() == 2 ) meAmplMap = meAmplMapG06_[ism-1];
      if ( dccMap[ism].getMgpaGain() == 1 ) meAmplMap = meAmplMapG12_[ism-1];

      float xval = hit.amplitude();
      if ( xval <= 0. ) xval = 0.0;

//      if ( dccMap[ism].getMgpaGain() == 3 ) xval = xval * 1./12.;
//      if ( dccMap[ism].getMgpaGain() == 2 ) xval = xval * 1./ 2.;
//      if ( dccMap[ism].getMgpaGain() == 1 ) xval = xval * 1./ 1.;

      LogDebug("EETestPulseTask") << " hit amplitude " << xval;

      if ( meAmplMap ) meAmplMap->Fill(xix, xiy, xval);

      LogDebug("EETestPulseTask") << "Crystal " << ix << " " << iy << " Amplitude = " << xval;

    }

  } catch ( exception& ex) {

    LogWarning("EETestPulseTask") << EcalUncalibratedRecHitCollection_ << " not available";

  }

  try {

    Handle<EcalPnDiodeDigiCollection> pns;
    e.getByLabel(EcalPnDiodeDigiCollection_, pns);

    int nep = pns->size();
    LogDebug("EETestPulseTask") << "event " << ievt_ << " pns collection size " << nep;

    for ( EcalPnDiodeDigiCollection::const_iterator pnItr = pns->begin(); pnItr != pns->end(); ++pnItr ) {

      EcalPnDiodeDigi pn = (*pnItr);
      EcalPnDiodeDetId id = pn.id();

      int ism = Numbers::iSM( id );

      int num = id.iPnId();

      map<int, EcalDCCHeaderBlock>::iterator i = dccMap.find(ism);
      if ( i == dccMap.end() ) continue;

      if ( ! ( dccMap[ism].getRunType() != EcalDCCHeaderBlock::TESTPULSE_MGPA ||
               dccMap[ism].getRunType() != EcalDCCHeaderBlock::TESTPULSE_GAP ) ) continue;

      LogDebug("EETestPulseTask") << " det id = " << id;
      LogDebug("EETestPulseTask") << " sm, num " << ism << " " << num;

      float xvalped = 0.;

      for (int i = 0; i < 4; i++) {

        EcalFEMSample sample = pn.sample(i);
        int adc = sample.adc();

        MonitorElement* mePNPed = 0;

        if ( sample.gainId() == 0 ) mePNPed = mePnPedMapG01_[ism-1];
        if ( sample.gainId() == 1 ) mePNPed = mePnPedMapG16_[ism-1];

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

      if ( pn.sample(0).gainId() == 0 ) mePN = mePnAmplMapG01_[ism-1];
      if ( pn.sample(0).gainId() == 1 ) mePN = mePnAmplMapG16_[ism-1];

      if ( mePN ) mePN->Fill(0.5, num - 0.5, xvalmax);

    }

  } catch ( exception& ex) {

    LogWarning("EETestPulseTask") << EcalPnDiodeDigiCollection_ << " not available";

  }

}

