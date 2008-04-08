/*
 * \file EEPedestalTask.cc
 *
 * $Date: 2008/04/07 11:30:25 $
 * $Revision: 1.34 $
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

#include <DQM/EcalCommon/interface/Numbers.h>

#include <DQM/EcalEndcapMonitorTasks/interface/EEPedestalTask.h>

using namespace cms;
using namespace edm;
using namespace std;

EEPedestalTask::EEPedestalTask(const ParameterSet& ps){

  init_ = false;

  // get hold of back-end interface
  dqmStore_ = Service<DQMStore>().operator->();

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  EcalRawDataCollection_ = ps.getParameter<edm::InputTag>("EcalRawDataCollection");
  EEDigiCollection_ = ps.getParameter<edm::InputTag>("EEDigiCollection");
  EcalPnDiodeDigiCollection_ = ps.getParameter<edm::InputTag>("EcalPnDiodeDigiCollection");

  for (int i = 0; i < 18; i++) {
    mePedMapG01_[i] = 0;
    mePedMapG06_[i] = 0;
    mePedMapG12_[i] = 0;
    mePed3SumMapG01_[i] = 0;
    mePed3SumMapG06_[i] = 0;
    mePed3SumMapG12_[i] = 0;
    mePed5SumMapG01_[i] = 0;
    mePed5SumMapG06_[i] = 0;
    mePed5SumMapG12_[i] = 0;
    mePnPedMapG01_[i] = 0;
    mePnPedMapG16_[i] = 0;
  }

}

EEPedestalTask::~EEPedestalTask(){

}

void EEPedestalTask::beginJob(const EventSetup& c){

  ievt_ = 0;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder("EcalEndcap/EEPedestalTask");
    dqmStore_->rmdir("EcalEndcap/EEPedestalTask");
  }

  Numbers::initGeometry(c, false);

}

void EEPedestalTask::setup(void){

  init_ = true;

  char histo[200];

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder("EcalEndcap/EEPedestalTask");

    dqmStore_->setCurrentFolder("EcalEndcap/EEPedestalTask/Gain01");
    for (int i = 0; i < 18; i++) {
      sprintf(histo, "EEPT pedestal %s G01", Numbers::sEE(i+1).c_str());
      mePedMapG01_[i] = dqmStore_->bookProfile2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 4096, 0., 4096., "s");
      mePedMapG01_[i]->setAxisTitle("jx", 1);
      mePedMapG01_[i]->setAxisTitle("jy", 2);
      dqmStore_->tag(mePedMapG01_[i], i+1);
      sprintf(histo, "EEPT pedestal 3sum %s G01", Numbers::sEE(i+1).c_str());
      mePed3SumMapG01_[i] = dqmStore_->bookProfile2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 4096, 0., 4096., "s");
      mePed3SumMapG01_[i]->setAxisTitle("jx", 1);
      mePed3SumMapG01_[i]->setAxisTitle("jy", 2);
      dqmStore_->tag(mePed3SumMapG01_[i], i+1);
      sprintf(histo, "EEPT pedestal 5sum %s G01", Numbers::sEE(i+1).c_str());
      mePed5SumMapG01_[i] = dqmStore_->bookProfile2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 4096, 0., 4096., "s");
      mePed5SumMapG01_[i]->setAxisTitle("jx", 1);
      mePed5SumMapG01_[i]->setAxisTitle("jy", 2);
      dqmStore_->tag(mePed5SumMapG01_[i], i+1);
    }

    dqmStore_->setCurrentFolder("EcalEndcap/EEPedestalTask/Gain06");
    for (int i = 0; i < 18; i++) {
      sprintf(histo, "EEPT pedestal %s G06", Numbers::sEE(i+1).c_str());
      mePedMapG06_[i] = dqmStore_->bookProfile2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 4096, 0., 4096., "s");
      mePedMapG06_[i]->setAxisTitle("jx", 1);
      mePedMapG06_[i]->setAxisTitle("jy", 2);
      dqmStore_->tag(mePedMapG06_[i], i+1);
      sprintf(histo, "EEPT pedestal 3sum %s G06", Numbers::sEE(i+1).c_str());
      mePed3SumMapG06_[i] = dqmStore_->bookProfile2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 4096, 0., 4096., "s");
      mePed3SumMapG06_[i]->setAxisTitle("jx", 1);
      mePed3SumMapG06_[i]->setAxisTitle("jy", 2);
      dqmStore_->tag(mePed3SumMapG06_[i], i+1);
      sprintf(histo, "EEPT pedestal 5sum %s G06", Numbers::sEE(i+1).c_str());
      mePed5SumMapG06_[i] = dqmStore_->bookProfile2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 4096, 0., 4096., "s");
      mePed5SumMapG06_[i]->setAxisTitle("jx", 1);
      mePed5SumMapG06_[i]->setAxisTitle("jy", 2);
      dqmStore_->tag(mePed5SumMapG06_[i], i+1);
    }

    dqmStore_->setCurrentFolder("EcalEndcap/EEPedestalTask/Gain12");
    for (int i = 0; i < 18; i++) {
      sprintf(histo, "EEPT pedestal %s G12", Numbers::sEE(i+1).c_str());
      mePedMapG12_[i] = dqmStore_->bookProfile2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 4096, 0., 4096., "s");
      mePedMapG12_[i]->setAxisTitle("jx", 1);
      mePedMapG12_[i]->setAxisTitle("jy", 2);
      dqmStore_->tag(mePedMapG12_[i], i+1);
      sprintf(histo, "EEPT pedestal 3sum %s G12", Numbers::sEE(i+1).c_str());
      mePed3SumMapG12_[i] = dqmStore_->bookProfile2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 4096, 0., 4096., "s");
      mePed3SumMapG12_[i]->setAxisTitle("jx", 1);
      mePed3SumMapG12_[i]->setAxisTitle("jy", 2);
      dqmStore_->tag(mePed3SumMapG12_[i], i+1);
      sprintf(histo, "EEPT pedestal 5sum %s G12", Numbers::sEE(i+1).c_str());
      mePed5SumMapG12_[i] = dqmStore_->bookProfile2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 4096, 0., 4096., "s");
      mePed5SumMapG12_[i]->setAxisTitle("jx", 1);
      mePed5SumMapG12_[i]->setAxisTitle("jy", 2);
      dqmStore_->tag(mePed5SumMapG12_[i], i+1);
    }

    dqmStore_->setCurrentFolder("EcalEndcap/EEPedestalTask/PN");

    dqmStore_->setCurrentFolder("EcalEndcap/EEPedestalTask/PN/Gain01");
    for (int i = 0; i < 18; i++) {
      sprintf(histo, "EEPDT PNs pedestal %s G01", Numbers::sEE(i+1).c_str());
      mePnPedMapG01_[i] =  dqmStore_->bookProfile(histo, histo, 10, 0., 10., 4096, 0., 4096., "s");
      mePnPedMapG01_[i]->setAxisTitle("channel", 1);
      mePnPedMapG01_[i]->setAxisTitle("pedestal", 2);
      dqmStore_->tag(mePnPedMapG01_[i], i+1);
    }

    dqmStore_->setCurrentFolder("EcalEndcap/EEPedestalTask/PN/Gain16");
    for (int i = 0; i < 18; i++) {
      sprintf(histo, "EEPDT PNs pedestal %s G16", Numbers::sEE(i+1).c_str());
      mePnPedMapG16_[i] =  dqmStore_->bookProfile(histo, histo, 10, 0., 10., 4096, 0., 4096., "s");
      mePnPedMapG16_[i]->setAxisTitle("channel", 1);
      mePnPedMapG16_[i]->setAxisTitle("pedestal", 2);
      dqmStore_->tag(mePnPedMapG16_[i], i+1);
    }

  }

}

void EEPedestalTask::cleanup(void){

  if ( ! enableCleanup_ ) return;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder("EcalEndcap/EEPedestalTask");

    dqmStore_->setCurrentFolder("EcalEndcap/EEPedestalTask/Gain01");
    for ( int i = 0; i < 18; i++ ) {
      if ( mePedMapG01_[i] ) dqmStore_->removeElement( mePedMapG01_[i]->getName() );
      mePedMapG01_[i] = 0;
      if ( mePed3SumMapG01_[i] ) dqmStore_->removeElement( mePed3SumMapG01_[i]->getName() );
      mePed3SumMapG01_[i] = 0;
      if ( mePed5SumMapG01_[i] ) dqmStore_->removeElement( mePed5SumMapG01_[i]->getName() );
      mePed5SumMapG01_[i] = 0;
    }

    dqmStore_->setCurrentFolder("EcalEndcap/EEPedestalTask/Gain06");
    for ( int i = 0; i < 18; i++ ) {
      if ( mePedMapG06_[i] ) dqmStore_->removeElement( mePedMapG06_[i]->getName() );
      mePedMapG06_[i] = 0;
      if ( mePed3SumMapG06_[i] ) dqmStore_->removeElement( mePed3SumMapG06_[i]->getName() );
      mePed3SumMapG06_[i] = 0;
      if ( mePed5SumMapG06_[i] ) dqmStore_->removeElement( mePed5SumMapG06_[i]->getName() );
      mePed5SumMapG06_[i] = 0;
    }

    dqmStore_->setCurrentFolder("EcalEndcap/EEPedestalTask/Gain12");
    for ( int i = 0; i < 18; i++ ) {
      if ( mePedMapG12_[i] ) dqmStore_->removeElement( mePedMapG12_[i]->getName() );
      mePedMapG12_[i] = 0;
      if ( mePed3SumMapG12_[i] ) dqmStore_->removeElement( mePed3SumMapG12_[i]->getName() );
      mePed3SumMapG12_[i] = 0;
      if ( mePed5SumMapG12_[i] ) dqmStore_->removeElement( mePed5SumMapG12_[i]->getName() );
      mePed5SumMapG12_[i] = 0;
    }

    dqmStore_->setCurrentFolder("EcalEndcap/EEPedestalTask/PN");

    dqmStore_->setCurrentFolder("EcalEndcap/EEPedestalTask/PN/Gain01");
    for ( int i = 0; i < 18; i++ ) {
      if ( mePnPedMapG01_[i]) dqmStore_->removeElement( mePnPedMapG01_[i]->getName() );
      mePnPedMapG01_[i] = 0;
    }

    dqmStore_->setCurrentFolder("EcalEndcap/EEPedestalTask/PN/Gain16");
    for ( int i = 0; i < 18; i++ ) {
      if ( mePnPedMapG16_[i]) dqmStore_->removeElement( mePnPedMapG16_[i]->getName() );
      mePnPedMapG16_[i] = 0;
    }

  }

  init_ = false;

}

void EEPedestalTask::endJob(void){

  LogInfo("EEPedestalTask") << "analyzed " << ievt_ << " events";

  if ( init_ ) this->cleanup();

}

void EEPedestalTask::analyze(const Event& e, const EventSetup& c){

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

      if ( dcch.getRunType() == EcalDCCHeaderBlock::PEDESTAL_STD ||
           dcch.getRunType() == EcalDCCHeaderBlock::PEDESTAL_GAP ) enable = true;

    }

  } else {

    LogWarning("EEPedestalTask") << EcalRawDataCollection_ << " not available";

  }

  if ( ! enable ) return;

  if ( ! init_ ) this->setup();

  ievt_++;

  Handle<EEDigiCollection> digis;

  if ( e.getByLabel(EEDigiCollection_, digis) ) {

    int need = digis->size();
    LogDebug("EEPedestalTask") << "event " << ievt_ << " digi collection size " << need;

    float xmap01[18][50][50];
    float xmap06[18][50][50];
    float xmap12[18][50][50];

    for ( int ism = 1; ism <= 18; ism++ ) {
      for ( int ix = 1; ix <= 50; ix++ ) {
        for ( int iy = 1; iy <= 50; iy++ ) {

          xmap01[ism-1][ix-1][iy-1] = 0.;
          xmap06[ism-1][ix-1][iy-1] = 0.;
          xmap12[ism-1][ix-1][iy-1] = 0.;

        }
      }
    }

    for ( EEDigiCollection::const_iterator digiItr = digis->begin(); digiItr != digis->end(); ++digiItr ) {

      EEDataFrame dataframe = (*digiItr);
      EEDetId id = dataframe.id();

      int ix = id.ix();
      int iy = id.iy();

      int ism = Numbers::iSM( id );

      if ( ism >= 1 && ism <= 9 ) ix = 101 - ix;

      float xix = ix - 0.5;
      float xiy = iy - 0.5;

      map<int, EcalDCCHeaderBlock>::iterator i = dccMap.find(ism);
      if ( i == dccMap.end() ) continue;

      if ( ! ( dccMap[ism].getRunType() == EcalDCCHeaderBlock::PEDESTAL_STD ||
               dccMap[ism].getRunType() == EcalDCCHeaderBlock::PEDESTAL_GAP ) ) continue;

      LogDebug("EEPedestalTask") << " det id = " << id;
      LogDebug("EEPedestalTask") << " sm, ix, iy " << ism << " " << ix << " " << iy;

      for (int i = 0; i < 10; i++) {

        EcalMGPASample sample = dataframe.sample(i);
        int adc = sample.adc();

        MonitorElement* mePedMap = 0;

        if ( sample.gainId() == 1 ) mePedMap = mePedMapG12_[ism-1];
        if ( sample.gainId() == 2 ) mePedMap = mePedMapG06_[ism-1];
        if ( sample.gainId() == 3 ) mePedMap = mePedMapG01_[ism-1];

        float xval = float(adc);

        if ( mePedMap ) mePedMap->Fill(xix, xiy, xval);

        if ( sample.gainId() == 1 ) xmap12[ism-1][ix-1-Numbers::ix0EE(ism)][iy-1-Numbers::iy0EE(ism)] = xmap12[ism-1][ix-1-Numbers::ix0EE(ism)][iy-1-Numbers::iy0EE(ism)] + xval;
        if ( sample.gainId() == 2 ) xmap06[ism-1][ix-1-Numbers::ix0EE(ism)][iy-1-Numbers::iy0EE(ism)] = xmap06[ism-1][ix-1-Numbers::ix0EE(ism)][iy-1-Numbers::iy0EE(ism)] + xval;
        if ( sample.gainId() == 3 ) xmap01[ism-1][ix-1-Numbers::ix0EE(ism)][iy-1-Numbers::iy0EE(ism)] = xmap01[ism-1][ix-1-Numbers::ix0EE(ism)][iy-1-Numbers::iy0EE(ism)] + xval;

      }

      xmap12[ism-1][ix-1-Numbers::ix0EE(ism)][iy-1-Numbers::iy0EE(ism)]=xmap12[ism-1][ix-1-Numbers::ix0EE(ism)][iy-1-Numbers::iy0EE(ism)]/10.;
      xmap06[ism-1][ix-1-Numbers::ix0EE(ism)][iy-1-Numbers::iy0EE(ism)]=xmap06[ism-1][ix-1-Numbers::ix0EE(ism)][iy-1-Numbers::iy0EE(ism)]/10.;
      xmap01[ism-1][ix-1-Numbers::ix0EE(ism)][iy-1-Numbers::iy0EE(ism)]=xmap01[ism-1][ix-1-Numbers::ix0EE(ism)][iy-1-Numbers::iy0EE(ism)]/10.;

    }

    // to be re-done using the 3x3 & 5x5 Selectors (if faster)

    for ( int ism = 1; ism <= 18; ism++ ) {
      for ( int ix = 1; ix <= 50; ix++ ) {
        for ( int iy = 1; iy <= 50; iy++ ) {

          int jx = ix + Numbers::ix0EE(ism);
          int jy = iy + Numbers::iy0EE(ism);

          if ( ism >= 1 && ism <= 9 ) jx = 101 - jx;

          if ( ! Numbers::validEE(ism, jx, jy) ) continue;

          float xix = ix - 0.5;
          float xiy = iy - 0.5;

          float x3val01;
          float x3val06;
          float x3val12;

          if ( ix >= 2 && ix <= 49 && iy >= 2 && iy <= 49 ) {

            x3val01 = 0.;
            x3val06 = 0.;
            x3val12 = 0.;
            for ( int i = -1; i <= +1; i++ ) {
              for ( int j = -1; j <= +1; j++ ) {

                x3val01 = x3val01 + xmap01[ism-1][ix-1+i][iy-1+j];
                x3val06 = x3val06 + xmap06[ism-1][ix-1+i][iy-1+j];
                x3val12 = x3val12 + xmap12[ism-1][ix-1+i][iy-1+j];

              }
            }
            x3val01 = x3val01 / 9.;
            x3val06 = x3val06 / 9.;
            x3val12 = x3val12 / 9.;
            if ( mePed3SumMapG01_[ism-1] && x3val01 != 0. ) mePed3SumMapG01_[ism-1]->Fill(xix+Numbers::ix0EE(ism), xiy+Numbers::iy0EE(ism), x3val01);
            if ( mePed3SumMapG06_[ism-1] && x3val06 != 0. ) mePed3SumMapG06_[ism-1]->Fill(xix+Numbers::ix0EE(ism), xiy+Numbers::iy0EE(ism), x3val06);
            if ( mePed3SumMapG12_[ism-1] && x3val12 != 0. ) mePed3SumMapG12_[ism-1]->Fill(xix+Numbers::ix0EE(ism), xiy+Numbers::iy0EE(ism), x3val12);

          }

          float x5val01;
          float x5val06;
          float x5val12;

          if ( ix >= 3 && ix <= 48 && iy >= 3 && iy <= 48 ) {

            x5val01 = 0.;
            x5val06 = 0.;
            x5val12 = 0.;
            for ( int i = -2; i <= +2; i++ ) {
              for ( int j = -2; j <= +2; j++ ) {

                x5val01 = x5val01 + xmap01[ism-1][ix-1+i][iy-1+j];
                x5val06 = x5val06 + xmap06[ism-1][ix-1+i][iy-1+j];
                x5val12 = x5val12 + xmap12[ism-1][ix-1+i][iy-1+j];

              }
            }
            x5val01 = x5val01 / 25.;
            x5val06 = x5val06 / 25.;
            x5val12 = x5val12 / 25.;
            if ( mePed5SumMapG01_[ism-1] && x5val01 != 0. ) mePed5SumMapG01_[ism-1]->Fill(xix+Numbers::ix0EE(ism), xiy+Numbers::iy0EE(ism), x5val01);
            if ( mePed5SumMapG06_[ism-1] && x5val06 != 0. ) mePed5SumMapG06_[ism-1]->Fill(xix+Numbers::ix0EE(ism), xiy+Numbers::iy0EE(ism), x5val06);
            if ( mePed5SumMapG12_[ism-1] && x5val12 != 0. ) mePed5SumMapG12_[ism-1]->Fill(xix+Numbers::ix0EE(ism), xiy+Numbers::iy0EE(ism), x5val12);

          }

        }
      }
    }

  } else {

    LogWarning("EEPedestalTask") << EEDigiCollection_ << " not available";

  }

  Handle<EcalPnDiodeDigiCollection> pns;

  if ( e.getByLabel(EcalPnDiodeDigiCollection_, pns) ) {

    int nep = pns->size();
    LogDebug("EEPedestalTask") << "event " << ievt_ << " pns collection size " << nep;

    for ( EcalPnDiodeDigiCollection::const_iterator pnItr = pns->begin(); pnItr != pns->end(); ++pnItr ) {

      EcalPnDiodeDigi pn = (*pnItr);
      EcalPnDiodeDetId id = pn.id();

      if ( Numbers::subDet( id ) != EcalEndcap ) continue;

      int ism = Numbers::iSM( id );

      int num = id.iPnId();

      map<int, EcalDCCHeaderBlock>::iterator i = dccMap.find(ism);
      if ( i == dccMap.end() ) continue;

      if ( ! ( dccMap[ism].getRunType() == EcalDCCHeaderBlock::PEDESTAL_STD ||
               dccMap[ism].getRunType() == EcalDCCHeaderBlock::PEDESTAL_GAP ) ) continue;

      LogDebug("EEPedestalTask") << " det id = " << id;
      LogDebug("EEPedestalTask") << " sm, num " << ism << " " << num;

      for (int i = 0; i < 50; i++) {

        EcalFEMSample sample = pn.sample(i);
        int adc = sample.adc();

        MonitorElement* mePNPed = 0;

        if ( sample.gainId() == 0 ) mePNPed = mePnPedMapG01_[ism-1];
        if ( sample.gainId() == 1 ) mePNPed = mePnPedMapG16_[ism-1];

        float xval = float(adc);

        if ( mePNPed ) mePNPed->Fill(num - 0.5, xval);

      }

    }

  } else {

    LogWarning("EEPedestalTask") << EcalPnDiodeDigiCollection_ << " not available";

  }

}

