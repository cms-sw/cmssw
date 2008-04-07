/*
 * \file EEStatusFlagsTask.cc
 *
 * $Date: 2008/03/16 14:36:19 $
 * $Revision: 1.8 $
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
#include "DataFormats/EcalDetId/interface/EcalDetIdCollections.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include <DQM/EcalCommon/interface/Numbers.h>

#include <DQM/EcalEndcapMonitorTasks/interface/EEStatusFlagsTask.h>

using namespace cms;
using namespace edm;
using namespace std;

EEStatusFlagsTask::EEStatusFlagsTask(const ParameterSet& ps){

  init_ = false;

  // get hold of back-end interface
  dbe_ = Service<DQMStore>().operator->();

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", true);

  EcalRawDataCollection_ = ps.getParameter<edm::InputTag>("EcalRawDataCollection");

  for (int i = 0; i < 18; i++) {
    meEvtType_[i] = 0;

    meFEchErrors_[i][0] = 0;
    meFEchErrors_[i][1] = 0;
  }

}

EEStatusFlagsTask::~EEStatusFlagsTask(){

}

void EEStatusFlagsTask::beginJob(const EventSetup& c){

  ievt_ = 0;

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalEndcap/EEStatusFlagsTask");
    dbe_->rmdir("EcalEndcap/EEStatusFlagsTask");
  }

  Numbers::initGeometry(c, false);

}

void EEStatusFlagsTask::setup(void){

  init_ = true;

  char histo[200];

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalEndcap/EEStatusFlagsTask");

    dbe_->setCurrentFolder("EcalEndcap/EEStatusFlagsTask/EvtType");
    for (int i = 0; i < 18; i++) {
      sprintf(histo, "EESFT EVTTYPE %s", Numbers::sEE(i+1).c_str());
      meEvtType_[i] = dbe_->book1D(histo, histo, 31, -1., 30.);
      meEvtType_[i]->setBinLabel(1, "UNKNOWN", 1);
      meEvtType_[i]->setBinLabel(2+EcalDCCHeaderBlock::COSMIC, "COSMIC", 1);
      meEvtType_[i]->setBinLabel(2+EcalDCCHeaderBlock::BEAMH4, "BEAMH4", 1);
      meEvtType_[i]->setBinLabel(2+EcalDCCHeaderBlock::BEAMH2, "BEAMH2", 1);
      meEvtType_[i]->setBinLabel(2+EcalDCCHeaderBlock::MTCC, "MTCC", 1);
      meEvtType_[i]->setBinLabel(2+EcalDCCHeaderBlock::LASER_STD, "LASER_STD", 1);
      meEvtType_[i]->setBinLabel(2+EcalDCCHeaderBlock::LASER_POWER_SCAN, "LASER_POWER_SCAN", 1);
      meEvtType_[i]->setBinLabel(2+EcalDCCHeaderBlock::LASER_DELAY_SCAN, "LASER_DELAY_SCAN", 1);
      meEvtType_[i]->setBinLabel(2+EcalDCCHeaderBlock::TESTPULSE_SCAN_MEM, "TESTPULSE_SCAN_MEM", 1);
      meEvtType_[i]->setBinLabel(2+EcalDCCHeaderBlock::TESTPULSE_MGPA, "TESTPULSE_MGPA", 1);
      meEvtType_[i]->setBinLabel(2+EcalDCCHeaderBlock::PEDESTAL_STD, "PEDESTAL_STD", 1);
      meEvtType_[i]->setBinLabel(2+EcalDCCHeaderBlock::PEDESTAL_OFFSET_SCAN, "PEDESTAL_OFFSET_SCAN", 1);
      meEvtType_[i]->setBinLabel(2+EcalDCCHeaderBlock::PEDESTAL_25NS_SCAN, "PEDESTAL_25NS_SCAN", 1);
      meEvtType_[i]->setBinLabel(2+EcalDCCHeaderBlock::LED_STD, "LED_STD", 1);
      meEvtType_[i]->setBinLabel(2+EcalDCCHeaderBlock::PHYSICS_GLOBAL, "PHYSICS_GLOBAL", 1);
      meEvtType_[i]->setBinLabel(2+EcalDCCHeaderBlock::COSMICS_GLOBAL, "COSMICS_GLOBAL", 1);
      meEvtType_[i]->setBinLabel(2+EcalDCCHeaderBlock::HALO_GLOBAL, "HALO_GLOBAL", 1);
      meEvtType_[i]->setBinLabel(2+EcalDCCHeaderBlock::LASER_GAP, "LASER_GAP", 1);
      meEvtType_[i]->setBinLabel(2+EcalDCCHeaderBlock::TESTPULSE_GAP, "TESTPULSE_GAP");
      meEvtType_[i]->setBinLabel(2+EcalDCCHeaderBlock::PEDESTAL_GAP, "PEDESTAL_GAP");
      meEvtType_[i]->setBinLabel(2+EcalDCCHeaderBlock::LED_GAP, "LED_GAP", 1);
      meEvtType_[i]->setBinLabel(2+EcalDCCHeaderBlock::PHYSICS_LOCAL, "PHYSICS_LOCAL", 1);
      meEvtType_[i]->setBinLabel(2+EcalDCCHeaderBlock::COSMICS_LOCAL, "COSMICS_LOCAL", 1);
      meEvtType_[i]->setBinLabel(2+EcalDCCHeaderBlock::HALO_LOCAL, "HALO_LOCAL", 1);
      dbe_->tag(meEvtType_[i], i+1);
    }

    dbe_->setCurrentFolder("EcalEndcap/EEStatusFlagsTask/FEStatus");
    for (int i = 0; i < 18; i++) {
      sprintf(histo, "EESFT front-end status %s", Numbers::sEE(i+1).c_str());
      meFEchErrors_[i][0] = dbe_->book2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50.);
      meFEchErrors_[i][0]->setAxisTitle("jx", 1);
      meFEchErrors_[i][0]->setAxisTitle("jy", 2);
      dbe_->tag(meFEchErrors_[i][0], i+1);

      for ( int ix = 1; ix <= 50; ix++ ) {
        for ( int iy = 1; iy <= 50; iy++ ) {
          meFEchErrors_[i][0]->setBinContent( ix, iy, -1. );
        }
      }
      meFEchErrors_[i][0]->setEntries( 0 );

      sprintf(histo, "EESFT front-end status bits %s", Numbers::sEE(i+1).c_str());
      meFEchErrors_[i][1] = dbe_->book1D(histo, histo, 16, 0., 16.);
      meFEchErrors_[i][1]->setBinLabel(1+0, "ACTIVE", 1);
      meFEchErrors_[i][1]->setBinLabel(1+1, "DISABLED", 1);
      meFEchErrors_[i][1]->setBinLabel(1+2, "TIMEOUT", 1);
      meFEchErrors_[i][1]->setBinLabel(1+3, "HEADER", 1);
      meFEchErrors_[i][1]->setBinLabel(1+4, "CHANNEL ID", 1);
      meFEchErrors_[i][1]->setBinLabel(1+5, "LINK", 1);
      meFEchErrors_[i][1]->setBinLabel(1+6, "BLOCKSIZE", 1);
      meFEchErrors_[i][1]->setBinLabel(1+7, "SUPPRESSED", 1);
      meFEchErrors_[i][1]->setBinLabel(1+8, "FIFO FULL", 1);
      meFEchErrors_[i][1]->setBinLabel(1+9, "L1A SYNC", 1);
      meFEchErrors_[i][1]->setBinLabel(1+10, "BX SYNC", 1);
      meFEchErrors_[i][1]->setBinLabel(1+11, "L1A+BX SYNC", 1);
      meFEchErrors_[i][1]->setBinLabel(1+12, "FIFO+L1A", 1);
      meFEchErrors_[i][1]->setBinLabel(1+13, "H PARITY", 1);
      meFEchErrors_[i][1]->setBinLabel(1+14, "V PARITY", 1);
      meFEchErrors_[i][1]->setBinLabel(1+15, "H+V PARITY", 1);
      dbe_->tag(meFEchErrors_[i][1], i+1);
    }

  }

}

void EEStatusFlagsTask::cleanup(void){

  if ( ! enableCleanup_ ) return;

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalEndcap/EEStatusFlagsTask");

    dbe_->setCurrentFolder("EcalEndcap/EEStatusFlagsTask/EvtType");
    for (int i = 0; i < 18; i++) {
      if ( meEvtType_[i] ) dbe_->removeElement( meEvtType_[i]->getName() );
      meEvtType_[i] = 0;
    }

    dbe_->setCurrentFolder("EcalEndcap/EEStatusFlagsTask/FEStatus");
    for (int i = 0; i < 18; i++) {
      if ( meFEchErrors_[i][0] ) dbe_->removeElement( meFEchErrors_[i][0]->getName() );
      meFEchErrors_[i][0] = 0;
      if ( meFEchErrors_[i][1] ) dbe_->removeElement( meFEchErrors_[i][1]->getName() );
      meFEchErrors_[i][1] = 0;
    }

  }

  init_ = false;

}

void EEStatusFlagsTask::endJob(void){

  LogInfo("EEStatusFlagsTask") << "analyzed " << ievt_ << " events";

  if ( init_ ) this->cleanup();

}

void EEStatusFlagsTask::analyze(const Event& e, const EventSetup& c){

  if ( ! init_ ) this->setup();

  ievt_++;

  Handle<EcalRawDataCollection> dcchs;

  if ( e.getByLabel(EcalRawDataCollection_, dcchs) ) {

    for ( EcalRawDataCollection::const_iterator dcchItr = dcchs->begin(); dcchItr != dcchs->end(); ++dcchItr ) {

      EcalDCCHeaderBlock dcch = (*dcchItr);

      if ( Numbers::subDet( dcch ) != EcalEndcap ) continue;

      int ism = Numbers::iSM( dcch, EcalEndcap );

      if ( meEvtType_[ism-1] ) meEvtType_[ism-1]->Fill(dcch.getRunType()+0.5);

      vector<short> status = dcch.getFEStatus();

      for ( unsigned int itt=1; itt<=status.size(); itt++ ) {

        if ( itt > 34 ) continue;

        vector<DetId> crystals = Numbers::crystals( EcalElectronicsId(dcch.id(), itt, 1, 1) );

        for ( unsigned int i=0; i<crystals.size(); i++ ) {

        EEDetId id = crystals[i];

        int ix = id.ix();
        int iy = id.iy();

        if ( ism >= 1 && ism <= 9 ) ix = 101 - ix;

        float xix = ix - 0.5;
        float xiy = iy - 0.5;

        if ( meFEchErrors_[ism-1][0] ) {
          if ( meFEchErrors_[ism-1][0]->getBinContent(ix-Numbers::ix0EE(ism), iy-Numbers::iy0EE(ism)) == -1 ) {
            meFEchErrors_[ism-1][0]->setBinContent(ix-Numbers::ix0EE(ism), iy-Numbers::iy0EE(ism), 0);
          }
        }

        if ( ! ( status[itt-1] == 0 || status[itt-1] == 1 || status[itt-1] == 7 ) ) {
          if ( meFEchErrors_[ism-1][0] ) meFEchErrors_[ism-1][0]->Fill(xix, xiy);
        }

        }

        if ( meFEchErrors_[ism-1][1] ) meFEchErrors_[ism-1][1]->Fill(status[itt-1]+0.5);

      }

    }

  } else {

    LogWarning("EEStatusFlagsTask") << EcalRawDataCollection_ << " not available";

  }

}

