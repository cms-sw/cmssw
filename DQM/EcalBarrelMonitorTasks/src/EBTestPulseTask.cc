/*
 * \file EBTestPulseTask.cc
 *
 * $Date: 2008/02/29 15:04:43 $
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

#include "DQMServices/Core/interface/MonitorElement.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include <DQM/EcalCommon/interface/Numbers.h>

#include <DQM/EcalBarrelMonitorTasks/interface/EBTestPulseTask.h>

using namespace cms;
using namespace edm;
using namespace std;

EBTestPulseTask::EBTestPulseTask(const ParameterSet& ps){

  init_ = false;

  // get hold of back-end interface
  dbe_ = Service<DQMStore>().operator->();

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  EcalRawDataCollection_ = ps.getParameter<edm::InputTag>("EcalRawDataCollection");
  EBDigiCollection_ = ps.getParameter<edm::InputTag>("EBDigiCollection");
  EcalPnDiodeDigiCollection_ = ps.getParameter<edm::InputTag>("EcalPnDiodeDigiCollection");
  EcalUncalibratedRecHitCollection_ = ps.getParameter<edm::InputTag>("EcalUncalibratedRecHitCollection");

  for (int i = 0; i < 36; i++) {
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

EBTestPulseTask::~EBTestPulseTask(){

}

void EBTestPulseTask::beginJob(const EventSetup& c){

  ievt_ = 0;

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalBarrel/EBTestPulseTask");
    dbe_->rmdir("EcalBarrel/EBTestPulseTask");
  }

  Numbers::initGeometry(c, false);

}

void EBTestPulseTask::setup(void){

  init_ = true;

  char histo[200];

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalBarrel/EBTestPulseTask");

    dbe_->setCurrentFolder("EcalBarrel/EBTestPulseTask/Gain01");
    for (int i = 0; i < 36; i++) {
      sprintf(histo, "EBTPT shape %s G01", Numbers::sEB(i+1).c_str());
      meShapeMapG01_[i] = dbe_->bookProfile2D(histo, histo, 1700, 0., 1700., 10, 0., 10., 4096, 0., 4096., "s");
      meShapeMapG01_[i]->setAxisTitle("channel", 1);
      meShapeMapG01_[i]->setAxisTitle("sample", 2);
      meShapeMapG01_[i]->setAxisTitle("amplitude", 3);
      dbe_->tag(meShapeMapG01_[i], i+1);
      sprintf(histo, "EBTPT amplitude %s G01", Numbers::sEB(i+1).c_str());
      meAmplMapG01_[i] = dbe_->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096.*12., "s");
      meAmplMapG01_[i]->setAxisTitle("ieta", 1);
      meAmplMapG01_[i]->setAxisTitle("iphi", 2);
      dbe_->tag(meAmplMapG01_[i], i+1);
    }

    dbe_->setCurrentFolder("EcalBarrel/EBTestPulseTask/Gain06");
    for (int i = 0; i < 36; i++) {
      sprintf(histo, "EBTPT shape %s G06", Numbers::sEB(i+1).c_str());
      meShapeMapG06_[i] = dbe_->bookProfile2D(histo, histo, 1700, 0., 1700., 10, 0., 10., 4096, 0., 4096., "s");
      meShapeMapG06_[i]->setAxisTitle("channel", 1);
      meShapeMapG06_[i]->setAxisTitle("sample", 2);
      meShapeMapG06_[i]->setAxisTitle("amplitude", 3);
      dbe_->tag(meShapeMapG06_[i], i+1);
      sprintf(histo, "EBTPT amplitude %s G06", Numbers::sEB(i+1).c_str());
      meAmplMapG06_[i] = dbe_->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096.*12., "s");
      meAmplMapG06_[i]->setAxisTitle("ieta", 1);
      meAmplMapG06_[i]->setAxisTitle("iphi", 2);
      dbe_->tag(meAmplMapG06_[i], i+1);
    }

    dbe_->setCurrentFolder("EcalBarrel/EBTestPulseTask/Gain12");
    for (int i = 0; i < 36; i++) {
      sprintf(histo, "EBTPT shape %s G12", Numbers::sEB(i+1).c_str());
      meShapeMapG12_[i] = dbe_->bookProfile2D(histo, histo, 1700, 0., 1700., 10, 0., 10., 4096, 0., 4096., "s");
      meShapeMapG12_[i]->setAxisTitle("channel", 1);
      meShapeMapG12_[i]->setAxisTitle("sample", 2);
      meShapeMapG12_[i]->setAxisTitle("amplitude", 3);
      dbe_->tag(meShapeMapG12_[i], i+1);
      sprintf(histo, "EBTPT amplitude %s G12", Numbers::sEB(i+1).c_str());
      meAmplMapG12_[i] = dbe_->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096.*12., "s");
      meAmplMapG12_[i]->setAxisTitle("ieta", 1);
      meAmplMapG12_[i]->setAxisTitle("iphi", 2);
      dbe_->tag(meAmplMapG12_[i], i+1);
   }

    dbe_->setCurrentFolder("EcalBarrel/EBTestPulseTask/PN");

    dbe_->setCurrentFolder("EcalBarrel/EBTestPulseTask/PN/Gain01");
    for (int i = 0; i < 36; i++) {
      sprintf(histo, "EBPDT PNs amplitude %s G01", Numbers::sEB(i+1).c_str());
      mePnAmplMapG01_[i] = dbe_->bookProfile(histo, histo, 10, 0., 10., 4096, 0., 4096., "s");
      mePnAmplMapG01_[i]->setAxisTitle("channel", 1);
      mePnAmplMapG01_[i]->setAxisTitle("amplitude", 2);
      dbe_->tag(mePnAmplMapG01_[i], i+1);
      sprintf(histo, "EBPDT PNs pedestal %s G01", Numbers::sEB(i+1).c_str());
      mePnPedMapG01_[i] =  dbe_->bookProfile(histo, histo, 10, 0., 10., 4096, 0., 4096., "s");
      mePnPedMapG01_[i]->setAxisTitle("channel", 1);
      mePnPedMapG01_[i]->setAxisTitle("pedestal", 2);
      dbe_->tag(mePnPedMapG01_[i], i+1);
    }

    dbe_->setCurrentFolder("EcalBarrel/EBTestPulseTask/PN/Gain16");
    for (int i = 0; i < 36; i++) {
      sprintf(histo, "EBPDT PNs amplitude %s G16", Numbers::sEB(i+1).c_str());
      mePnAmplMapG16_[i] = dbe_->bookProfile(histo, histo, 10, 0., 10., 4096, 0., 4096., "s");
      mePnAmplMapG16_[i]->setAxisTitle("channel", 1);
      mePnAmplMapG16_[i]->setAxisTitle("amplitude", 2);
      dbe_->tag(mePnAmplMapG16_[i], i+1);
      sprintf(histo, "EBPDT PNs pedestal %s G16", Numbers::sEB(i+1).c_str());
      mePnPedMapG16_[i] =  dbe_->bookProfile(histo, histo, 10, 0., 10., 4096, 0., 4096., "s");
      mePnPedMapG16_[i]->setAxisTitle("channel", 1);
      mePnPedMapG16_[i]->setAxisTitle("pedestal", 2);
      dbe_->tag(mePnPedMapG16_[i], i+1);
    }

  }

}

void EBTestPulseTask::cleanup(void){

  if ( ! enableCleanup_ ) return;

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalBarrel/EBTestPulseTask");

    dbe_->setCurrentFolder("EcalBarrel/EBTestPulseTask/Gain01");
    for (int i = 0; i < 36; i++) {
      if ( meShapeMapG01_[i] ) dbe_->removeElement( meShapeMapG01_[i]->getName() );
      meShapeMapG01_[i] = 0;
      if ( meAmplMapG01_[i] ) dbe_->removeElement( meAmplMapG01_[i]->getName() );
      meAmplMapG01_[i] = 0;
    }

    dbe_->setCurrentFolder("EcalBarrel/EBTestPulseTask/Gain06");
    for (int i = 0; i < 36; i++) {
      if ( meShapeMapG06_[i] ) dbe_->removeElement( meShapeMapG06_[i]->getName() );
      meShapeMapG06_[i] = 0;
      if ( meAmplMapG06_[i] ) dbe_->removeElement( meAmplMapG06_[i]->getName() );
      meAmplMapG06_[i] = 0;
    }

    dbe_->setCurrentFolder("EcalBarrel/EBTestPulseTask/Gain12");
    for (int i = 0; i < 36; i++) {
      if ( meShapeMapG12_[i] ) dbe_->removeElement( meShapeMapG12_[i]->getName() );
      meShapeMapG12_[i] = 0;
      if ( meAmplMapG12_[i] ) dbe_->removeElement( meAmplMapG12_[i]->getName() );
      meAmplMapG12_[i] = 0;
    }

    dbe_->setCurrentFolder("EcalBarrel/EBTestPulseTask/PN");

    dbe_->setCurrentFolder("EcalBarrel/EBTestPulseTask/PN/Gain01");
    for (int i = 0; i < 36; i++) {
      if ( mePnAmplMapG01_[i] ) dbe_->removeElement( mePnAmplMapG01_[i]->getName() );
      mePnAmplMapG01_[i] = 0;
      if ( mePnPedMapG01_[i] ) dbe_->removeElement( mePnPedMapG01_[i]->getName() );
      mePnPedMapG01_[i] = 0;
    }

    dbe_->setCurrentFolder("EcalBarrel/EBTestPulseTask/PN/Gain16");
    for (int i = 0; i < 36; i++) {
      if ( mePnAmplMapG16_[i] ) dbe_->removeElement( mePnAmplMapG16_[i]->getName() );
      mePnAmplMapG16_[i] = 0;
      if ( mePnPedMapG16_[i] ) dbe_->removeElement( mePnPedMapG16_[i]->getName() );
      mePnPedMapG16_[i] = 0;
    }

  }

  init_ = false;

}

void EBTestPulseTask::endJob(void){

  LogInfo("EBTestPulseTask") << "analyzed " << ievt_ << " events";

  if ( init_ ) this->cleanup();

}

void EBTestPulseTask::analyze(const Event& e, const EventSetup& c){

  bool enable = false;
  map<int, EcalDCCHeaderBlock> dccMap;

  Handle<EcalRawDataCollection> dcchs;

  if ( e.getByLabel(EcalRawDataCollection_, dcchs) ) {

    for ( EcalRawDataCollection::const_iterator dcchItr = dcchs->begin(); dcchItr != dcchs->end(); ++dcchItr ) {

      EcalDCCHeaderBlock dcch = (*dcchItr);

      if ( Numbers::subDet( dcch ) != EcalBarrel ) continue;

      int ism = Numbers::iSM( dcch, EcalBarrel );

      map<int, EcalDCCHeaderBlock>::iterator i = dccMap.find( ism );
      if ( i != dccMap.end() ) continue;

      dccMap[ ism ] = dcch;

      if ( dcch.getRunType() == EcalDCCHeaderBlock::TESTPULSE_MGPA ||
           dcch.getRunType() == EcalDCCHeaderBlock::TESTPULSE_GAP ) enable = true;

    }

  } else {

    LogWarning("EBTestPulseTask") << EcalRawDataCollection_ << " not available";

  }

  if ( ! enable ) return;

  if ( ! init_ ) this->setup();

  ievt_++;

  Handle<EBDigiCollection> digis;

  if ( e.getByLabel(EBDigiCollection_, digis) ) {

    int nebd = digis->size();
    LogDebug("EBTestPulseTask") << "event " << ievt_ << " digi collection size " << nebd;

    for ( EBDigiCollection::const_iterator digiItr = digis->begin(); digiItr != digis->end(); ++digiItr ) {

      EBDataFrame dataframe = (*digiItr);
      EBDetId id = dataframe.id();

      int ic = id.ic();
      int ie = (ic-1)/20 + 1;
      int ip = (ic-1)%20 + 1;

      int ism = Numbers::iSM( id );

      map<int, EcalDCCHeaderBlock>::iterator i = dccMap.find(ism);
      if ( i == dccMap.end() ) continue;

      if ( ! ( dccMap[ism].getRunType() != EcalDCCHeaderBlock::TESTPULSE_MGPA ||
               dccMap[ism].getRunType() != EcalDCCHeaderBlock::TESTPULSE_GAP ) ) continue;

      LogDebug("EBTestPulseTask") << " det id = " << id;
      LogDebug("EBTestPulseTask") << " sm, ieta, iphi " << ism << " " << ie << " " << ip;

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

    LogWarning("EBTestPulseTask") << EBDigiCollection_ << " not available";

  }

  Handle<EcalUncalibratedRecHitCollection> hits;

  if ( e.getByLabel(EcalUncalibratedRecHitCollection_, hits) ) {

    int neh = hits->size();
    LogDebug("EBTestPulseTask") << "event " << ievt_ << " hits collection size " << neh;

    for ( EcalUncalibratedRecHitCollection::const_iterator hitItr = hits->begin(); hitItr != hits->end(); ++hitItr ) {

      EcalUncalibratedRecHit hit = (*hitItr);
      EBDetId id = hit.id();

      int ic = id.ic();
      int ie = (ic-1)/20 + 1;
      int ip = (ic-1)%20 + 1;

      int ism = Numbers::iSM( id );

      float xie = ie - 0.5;
      float xip = ip - 0.5;

      map<int, EcalDCCHeaderBlock>::iterator i = dccMap.find(ism);
      if ( i == dccMap.end() ) continue;

      if ( ! ( dccMap[ism].getRunType() != EcalDCCHeaderBlock::TESTPULSE_MGPA ||
               dccMap[ism].getRunType() != EcalDCCHeaderBlock::TESTPULSE_GAP ) ) continue;

      LogDebug("EBTestPulseTask") << " det id = " << id;
      LogDebug("EBTestPulseTask") << " sm, ieta, iphi " << ism << " " << ie << " " << ip;

      MonitorElement* meAmplMap = 0;

      if ( dccMap[ism].getMgpaGain() == 3 ) meAmplMap = meAmplMapG01_[ism-1];
      if ( dccMap[ism].getMgpaGain() == 2 ) meAmplMap = meAmplMapG06_[ism-1];
      if ( dccMap[ism].getMgpaGain() == 1 ) meAmplMap = meAmplMapG12_[ism-1];

      float xval = hit.amplitude();
      if ( xval <= 0. ) xval = 0.0;

//      if ( dccMap[ism].getMgpaGain() == 3 ) xval = xval * 1./12.;
//      if ( dccMap[ism].getMgpaGain() == 2 ) xval = xval * 1./ 2.;
//      if ( dccMap[ism].getMgpaGain() == 1 ) xval = xval * 1./ 1.;

      LogDebug("EBTestPulseTask") << " hit amplitude " << xval;

      if ( meAmplMap ) meAmplMap->Fill(xie, xip, xval);

      LogDebug("EBTestPulseTask") << "Crystal " << ie << " " << ip << " Amplitude = " << xval;

    }

  } else {

    LogWarning("EBTestPulseTask") << EcalUncalibratedRecHitCollection_ << " not available";

  }

  Handle<EcalPnDiodeDigiCollection> pns;

  if ( e.getByLabel(EcalPnDiodeDigiCollection_, pns) ) {

    int nep = pns->size();
    LogDebug("EBTestPulseTask") << "event " << ievt_ << " pns collection size " << nep;

    for ( EcalPnDiodeDigiCollection::const_iterator pnItr = pns->begin(); pnItr != pns->end(); ++pnItr ) {

      EcalPnDiodeDigi pn = (*pnItr);
      EcalPnDiodeDetId id = pn.id();

      if ( Numbers::subDet( id ) != EcalBarrel ) continue;

      int ism = Numbers::iSM( id );

      int num = id.iPnId();

      map<int, EcalDCCHeaderBlock>::iterator i = dccMap.find(ism);
      if ( i == dccMap.end() ) continue;

      if ( ! ( dccMap[ism].getRunType() != EcalDCCHeaderBlock::TESTPULSE_MGPA ||
               dccMap[ism].getRunType() != EcalDCCHeaderBlock::TESTPULSE_GAP ) ) continue;

      LogDebug("EBTestPulseTask") << " det id = " << id;
      LogDebug("EBTestPulseTask") << " sm, num " << ism << " " << num;

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

    LogWarning("EBTestPulseTask") << EcalPnDiodeDigiCollection_ << " not available";

  }

}

