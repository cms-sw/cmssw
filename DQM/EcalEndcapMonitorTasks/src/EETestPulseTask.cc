/*
 * \file EETestPulseTask.cc
 *
 * $Date: 2008/04/08 18:11:28 $
 * $Revision: 1.41 $
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

#include <DQM/EcalEndcapMonitorTasks/interface/EETestPulseTask.h>

using namespace cms;
using namespace edm;
using namespace std;

EETestPulseTask::EETestPulseTask(const ParameterSet& ps){

  init_ = false;

  dqmStore_ = Service<DQMStore>().operator->();

  prefixME_ = ps.getUntrackedParameter<string>("prefixME", "");

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  mergeRuns_ = ps.getUntrackedParameter<bool>("mergeRuns", false);

  EcalRawDataCollection_ = ps.getParameter<edm::InputTag>("EcalRawDataCollection");
  EEDigiCollection_ = ps.getParameter<edm::InputTag>("EEDigiCollection");
  EcalPnDiodeDigiCollection_ = ps.getParameter<edm::InputTag>("EcalPnDiodeDigiCollection");
  EcalUncalibratedRecHitCollection_ = ps.getParameter<edm::InputTag>("EcalUncalibratedRecHitCollection");

  for (int i = 0; i < 18; i++) {
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

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EETestPulseTask");
    dqmStore_->rmdir(prefixME_ + "/EETestPulseTask");
  }

  Numbers::initGeometry(c, false);

}

void EETestPulseTask::beginRun(const Run& r, const EventSetup& c) {

  if ( ! mergeRuns_ ) this->reset();

}

void EETestPulseTask::endRun(const Run& r, const EventSetup& c) {

}

void EETestPulseTask::reset(void) {

  for (int i = 0; i < 18; i++) {
    if ( meShapeMapG01_[i] ) meShapeMapG01_[i]->Reset();
    if ( meAmplMapG01_[i] ) meAmplMapG01_[i]->Reset();
    if ( meShapeMapG06_[i] ) meShapeMapG06_[i]->Reset();
    if ( meAmplMapG06_[i] ) meAmplMapG06_[i]->Reset();
    if ( meShapeMapG12_[i] ) meShapeMapG12_[i]->Reset();
    if ( meAmplMapG12_[i] ) meAmplMapG12_[i]->Reset();
    if ( mePnAmplMapG01_[i] ) mePnAmplMapG01_[i]->Reset();
    if ( mePnPedMapG01_[i] ) mePnPedMapG01_[i]->Reset();
    if ( mePnAmplMapG16_[i] ) mePnAmplMapG16_[i]->Reset();
    if ( mePnPedMapG16_[i] ) mePnPedMapG16_[i]->Reset();
  }

}

void EETestPulseTask::setup(void){

  init_ = true;

  char histo[200];

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EETestPulseTask");

    dqmStore_->setCurrentFolder(prefixME_ + "/EETestPulseTask/Gain01");
    for (int i = 0; i < 18; i++) {
      sprintf(histo, "EETPT shape %s G01", Numbers::sEE(i+1).c_str());
      meShapeMapG01_[i] = dqmStore_->bookProfile2D(histo, histo, 850, 0., 850., 10, 0., 10., 4096, 0., 4096., "s");
      meShapeMapG01_[i]->setAxisTitle("channel", 1);
      meShapeMapG01_[i]->setAxisTitle("sample", 2);
      meShapeMapG01_[i]->setAxisTitle("amplitude", 3);
      dqmStore_->tag(meShapeMapG01_[i], i+1);
      sprintf(histo, "EETPT amplitude %s G01", Numbers::sEE(i+1).c_str());
      meAmplMapG01_[i] = dqmStore_->bookProfile2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 4096, 0., 4096.*12., "s");
      meAmplMapG01_[i]->setAxisTitle("jx", 1);
      meAmplMapG01_[i]->setAxisTitle("jy", 2);
      dqmStore_->tag(meAmplMapG01_[i], i+1);
    }

    dqmStore_->setCurrentFolder(prefixME_ + "/EETestPulseTask/Gain06");
    for (int i = 0; i < 18; i++) {
      sprintf(histo, "EETPT shape %s G06", Numbers::sEE(i+1).c_str());
      meShapeMapG06_[i] = dqmStore_->bookProfile2D(histo, histo, 850, 0., 850., 10, 0., 10., 4096, 0., 4096., "s");
      meShapeMapG06_[i]->setAxisTitle("channel", 1);
      meShapeMapG06_[i]->setAxisTitle("sample", 2);
      meShapeMapG06_[i]->setAxisTitle("amplitude", 3);
      dqmStore_->tag(meShapeMapG06_[i], i+1);
      sprintf(histo, "EETPT amplitude %s G06", Numbers::sEE(i+1).c_str());
      meAmplMapG06_[i] = dqmStore_->bookProfile2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 4096, 0., 4096.*12., "s");
      meAmplMapG06_[i]->setAxisTitle("jx", 1);
      meAmplMapG06_[i]->setAxisTitle("jy", 2);
      dqmStore_->tag(meAmplMapG06_[i], i+1);
    }

    dqmStore_->setCurrentFolder(prefixME_ + "/EETestPulseTask/Gain12");
    for (int i = 0; i < 18; i++) {
      sprintf(histo, "EETPT shape %s G12", Numbers::sEE(i+1).c_str());
      meShapeMapG12_[i] = dqmStore_->bookProfile2D(histo, histo, 850, 0., 850., 10, 0., 10., 4096, 0., 4096., "s");
      meShapeMapG12_[i]->setAxisTitle("channel", 1);
      meShapeMapG12_[i]->setAxisTitle("sample", 2);
      meShapeMapG12_[i]->setAxisTitle("amplitude", 3);
      dqmStore_->tag(meShapeMapG12_[i], i+1);
      sprintf(histo, "EETPT amplitude %s G12", Numbers::sEE(i+1).c_str());
      meAmplMapG12_[i] = dqmStore_->bookProfile2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 4096, 0., 4096.*12., "s");
      meAmplMapG12_[i]->setAxisTitle("jx", 1);
      meAmplMapG12_[i]->setAxisTitle("jy", 2);
      dqmStore_->tag(meAmplMapG12_[i], i+1);
   }

    dqmStore_->setCurrentFolder(prefixME_ + "/EETestPulseTask/PN");

    dqmStore_->setCurrentFolder(prefixME_ + "/EETestPulseTask/PN/Gain01");
    for (int i = 0; i < 18; i++) {
      sprintf(histo, "EEPDT PNs amplitude %s G01", Numbers::sEE(i+1).c_str());
      mePnAmplMapG01_[i] = dqmStore_->bookProfile(histo, histo, 10, 0., 10., 4096, 0., 4096., "s");
      mePnAmplMapG01_[i]->setAxisTitle("channel", 1);
      mePnAmplMapG01_[i]->setAxisTitle("amplitude", 2);
      dqmStore_->tag(mePnAmplMapG01_[i], i+1);
      sprintf(histo, "EEPDT PNs pedestal %s G01", Numbers::sEE(i+1).c_str());
      mePnPedMapG01_[i] =  dqmStore_->bookProfile(histo, histo, 10, 0., 10., 4096, 0., 4096., "s");
      mePnPedMapG01_[i]->setAxisTitle("channel", 1);
      mePnPedMapG01_[i]->setAxisTitle("pedestal", 2);
      dqmStore_->tag(mePnPedMapG01_[i], i+1);
    }

    dqmStore_->setCurrentFolder(prefixME_ + "/EETestPulseTask/PN/Gain16");
    for (int i = 0; i < 18; i++) {
      sprintf(histo, "EEPDT PNs amplitude %s G16", Numbers::sEE(i+1).c_str());
      mePnAmplMapG16_[i] = dqmStore_->bookProfile(histo, histo, 10, 0., 10., 4096, 0., 4096., "s");
      mePnAmplMapG16_[i]->setAxisTitle("channel", 1);
      mePnAmplMapG16_[i]->setAxisTitle("amplitude", 2);
      dqmStore_->tag(mePnAmplMapG16_[i], i+1);
      sprintf(histo, "EEPDT PNs pedestal %s G16", Numbers::sEE(i+1).c_str());
      mePnPedMapG16_[i] =  dqmStore_->bookProfile(histo, histo, 10, 0., 10., 4096, 0., 4096., "s");
      mePnPedMapG16_[i]->setAxisTitle("channel", 1);
      mePnPedMapG16_[i]->setAxisTitle("pedestal", 2);
      dqmStore_->tag(mePnPedMapG16_[i], i+1);
    }

  }

}

void EETestPulseTask::cleanup(void){

  if ( ! init_ ) return;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EETestPulseTask");

    dqmStore_->setCurrentFolder(prefixME_ + "/EETestPulseTask/Gain01");
    for (int i = 0; i < 18; i++) {
      if ( meShapeMapG01_[i] ) dqmStore_->removeElement( meShapeMapG01_[i]->getName() );
      meShapeMapG01_[i] = 0;
      if ( meAmplMapG01_[i] ) dqmStore_->removeElement( meAmplMapG01_[i]->getName() );
      meAmplMapG01_[i] = 0;
    }

    dqmStore_->setCurrentFolder(prefixME_ + "/EETestPulseTask/Gain06");
    for (int i = 0; i < 18; i++) {
      if ( meShapeMapG06_[i] ) dqmStore_->removeElement( meShapeMapG06_[i]->getName() );
      meShapeMapG06_[i] = 0;
      if ( meAmplMapG06_[i] ) dqmStore_->removeElement( meAmplMapG06_[i]->getName() );
      meAmplMapG06_[i] = 0;
    }

    dqmStore_->setCurrentFolder(prefixME_ + "/EETestPulseTask/Gain12");
    for (int i = 0; i < 18; i++) {
      if ( meShapeMapG12_[i] ) dqmStore_->removeElement( meShapeMapG12_[i]->getName() );
      meShapeMapG12_[i] = 0;
      if ( meAmplMapG12_[i] ) dqmStore_->removeElement( meAmplMapG12_[i]->getName() );
      meAmplMapG12_[i] = 0;
    }

    dqmStore_->setCurrentFolder(prefixME_ + "/EETestPulseTask/PN");

    dqmStore_->setCurrentFolder(prefixME_ + "/EETestPulseTask/PN/Gain01");
    for (int i = 0; i < 18; i++) {
      if ( mePnAmplMapG01_[i] ) dqmStore_->removeElement( mePnAmplMapG01_[i]->getName() );
      mePnAmplMapG01_[i] = 0;
      if ( mePnPedMapG01_[i] ) dqmStore_->removeElement( mePnPedMapG01_[i]->getName() );
      mePnPedMapG01_[i] = 0;
    }

    dqmStore_->setCurrentFolder(prefixME_ + "/EETestPulseTask/PN/Gain16");
    for (int i = 0; i < 18; i++) {
      if ( mePnAmplMapG16_[i] ) dqmStore_->removeElement( mePnAmplMapG16_[i]->getName() );
      mePnAmplMapG16_[i] = 0;
      if ( mePnPedMapG16_[i] ) dqmStore_->removeElement( mePnPedMapG16_[i]->getName() );
      mePnPedMapG16_[i] = 0;
    }

  }

  init_ = false;

}

void EETestPulseTask::endJob(void){

  LogInfo("EETestPulseTask") << "analyzed " << ievt_ << " events";

  if ( enableCleanup_ ) this->cleanup();

}

void EETestPulseTask::analyze(const Event& e, const EventSetup& c){

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

      if ( dcch.getRunType() == EcalDCCHeaderBlock::TESTPULSE_MGPA ||
           dcch.getRunType() == EcalDCCHeaderBlock::TESTPULSE_GAP ) enable = true;

    }

  } else {

    LogWarning("EETestPulseTask") << EcalRawDataCollection_ << " not available";

  }

  if ( ! enable ) return;

  if ( ! init_ ) this->setup();

  ievt_++;

  Handle<EEDigiCollection> digis;

  if ( e.getByLabel(EEDigiCollection_, digis) ) {

    int need = digis->size();
    LogDebug("EETestPulseTask") << "event " << ievt_ << " digi collection size " << need;

    for ( EEDigiCollection::const_iterator digiItr = digis->begin(); digiItr != digis->end(); ++digiItr ) {

      EEDataFrame dataframe = (*digiItr);
      EEDetId id = dataframe.id();

      int ix = id.ix();
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

  } else {

    LogWarning("EETestPulseTask") << EEDigiCollection_ << " not available";

  }

  Handle<EcalUncalibratedRecHitCollection> hits;

  if ( e.getByLabel(EcalUncalibratedRecHitCollection_, hits) ) {

    int neh = hits->size();
    LogDebug("EETestPulseTask") << "event " << ievt_ << " hits collection size " << neh;

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

  } else {

    LogWarning("EETestPulseTask") << EcalUncalibratedRecHitCollection_ << " not available";

  }

  Handle<EcalPnDiodeDigiCollection> pns;

  if ( e.getByLabel(EcalPnDiodeDigiCollection_, pns) ) {

    int nep = pns->size();
    LogDebug("EETestPulseTask") << "event " << ievt_ << " pns collection size " << nep;

    for ( EcalPnDiodeDigiCollection::const_iterator pnItr = pns->begin(); pnItr != pns->end(); ++pnItr ) {

      EcalPnDiodeDigi pn = (*pnItr);
      EcalPnDiodeDetId id = pn.id();

      if ( Numbers::subDet( id ) != EcalEndcap ) continue;

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

      if ( pn.sample(0).gainId() == 0 ) mePN = mePnAmplMapG01_[ism-1];
      if ( pn.sample(0).gainId() == 1 ) mePN = mePnAmplMapG16_[ism-1];

      if ( mePN ) mePN->Fill(num - 0.5, xvalmax);

    }

  } else {

    LogWarning("EETestPulseTask") << EcalPnDiodeDigiCollection_ << " not available";

  }

}

