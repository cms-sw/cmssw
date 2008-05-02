/*
 * \file EBStatusFlagsTask.cc
 *
 * $Date: 2008/04/08 15:35:12 $
 * $Revision: 1.12 $
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

#include <DQM/EcalCommon/interface/Numbers.h>

#include <DQM/EcalBarrelMonitorTasks/interface/EBStatusFlagsTask.h>

using namespace cms;
using namespace edm;
using namespace std;

EBStatusFlagsTask::EBStatusFlagsTask(const ParameterSet& ps){

  init_ = false;

  dqmStore_ = Service<DQMStore>().operator->();

  prefixME_ = ps.getUntrackedParameter<string>("prefixME", "");

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  EcalRawDataCollection_ = ps.getParameter<edm::InputTag>("EcalRawDataCollection");

  for (int i = 0; i < 36; i++) {
    meEvtType_[i] = 0;

    meFEchErrors_[i][0] = 0;
    meFEchErrors_[i][1] = 0;
  }

}

EBStatusFlagsTask::~EBStatusFlagsTask(){

}

void EBStatusFlagsTask::beginJob(const EventSetup& c){

  ievt_ = 0;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EBStatusFlagsTask");
    dqmStore_->rmdir(prefixME_ + "/EBStatusFlagsTask");
  }

  Numbers::initGeometry(c, false);

}

void EBStatusFlagsTask::setup(void){

  init_ = true;

  char histo[200];

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EBStatusFlagsTask");

    dqmStore_->setCurrentFolder(prefixME_ + "/EBStatusFlagsTask/EvtType");
    for (int i = 0; i < 36; i++) {
      sprintf(histo, "EBSFT EVTTYPE %s", Numbers::sEB(i+1).c_str());
      meEvtType_[i] = dqmStore_->book1D(histo, histo, 31, -1., 30.);
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
      dqmStore_->tag(meEvtType_[i], i+1);
    }

    dqmStore_->setCurrentFolder(prefixME_ + "/EBStatusFlagsTask/FEStatus");
    for (int i = 0; i < 36; i++) {
      sprintf(histo, "EBSFT front-end status %s", Numbers::sEB(i+1).c_str());
      meFEchErrors_[i][0] = dqmStore_->book2D(histo, histo, 17, 0., 17., 4, 0., 4.);
      meFEchErrors_[i][0]->setAxisTitle("ieta'", 1);
      meFEchErrors_[i][0]->setAxisTitle("iphi'", 2);
      dqmStore_->tag(meFEchErrors_[i][0], i+1);

      for ( int ie = 1; ie <= 17; ie++ ) {
        for ( int ip = 1; ip <= 4; ip++ ) {
          meFEchErrors_[i][0]->setBinContent( ie, ip, -1. );
        }
      }
      meFEchErrors_[i][0]->setEntries( 0 );

      sprintf(histo, "EBSFT front-end status bits %s", Numbers::sEB(i+1).c_str());
      meFEchErrors_[i][1] = dqmStore_->book1D(histo, histo, 16, 0., 16.);
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
      dqmStore_->tag(meFEchErrors_[i][1], i+1);
    }

  }

}

void EBStatusFlagsTask::cleanup(void){

  if ( ! enableCleanup_ ) return;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EBStatusFlagsTask");

    dqmStore_->setCurrentFolder(prefixME_ + "/EBStatusFlagsTask/EvtType");
    for (int i = 0; i < 36; i++) {
      if ( meEvtType_[i] ) dqmStore_->removeElement( meEvtType_[i]->getName() );
      meEvtType_[i] = 0;
    }

    dqmStore_->setCurrentFolder(prefixME_ + "/EBStatusFlagsTask/FEStatus");
    for (int i = 0; i < 36; i++) {
      if ( meFEchErrors_[i][0] ) dqmStore_->removeElement( meFEchErrors_[i][0]->getName() );
      meFEchErrors_[i][0] = 0;
      if ( meFEchErrors_[i][1] ) dqmStore_->removeElement( meFEchErrors_[i][1]->getName() );
      meFEchErrors_[i][1] = 0;
    }

  }

  init_ = false;

}

void EBStatusFlagsTask::endJob(void){

  LogInfo("EBStatusFlagsTask") << "analyzed " << ievt_ << " events";

  if ( init_ ) this->cleanup();

}

void EBStatusFlagsTask::analyze(const Event& e, const EventSetup& c){

  if ( ! init_ ) this->setup();

  ievt_++;

  Handle<EcalRawDataCollection> dcchs;

  if ( e.getByLabel(EcalRawDataCollection_, dcchs) ) {

    for ( EcalRawDataCollection::const_iterator dcchItr = dcchs->begin(); dcchItr != dcchs->end(); ++dcchItr ) {

      EcalDCCHeaderBlock dcch = (*dcchItr);

      if ( Numbers::subDet( dcch ) != EcalBarrel ) continue;

      int ism = Numbers::iSM( dcch, EcalBarrel );

      if ( meEvtType_[ism-1] ) meEvtType_[ism-1]->Fill(dcch.getRunType()+0.5);

      vector<short> status = dcch.getFEStatus();

      for ( unsigned int itt=1; itt<=status.size(); itt++ ) {

        if ( itt > 68 ) continue;

        int iet = (itt-1)/4 + 1;
        int ipt = (itt-1)%4 + 1;

        float xiet = iet - 0.5;
        float xipt = ipt - 0.5;

        if ( meFEchErrors_[ism-1][0] ) {
          if ( meFEchErrors_[ism-1][0]->getBinContent(iet, ipt) == -1 ) {
            meFEchErrors_[ism-1][0]->setBinContent(iet, ipt, 0);
          }
        }

        if ( ! ( status[itt-1] == 0 || status[itt-1] == 1 || status[itt-1] == 7 ) ) {
          if ( meFEchErrors_[ism-1][0] ) meFEchErrors_[ism-1][0]->Fill(xiet, xipt);
        }

        if ( meFEchErrors_[ism-1][1] ) meFEchErrors_[ism-1][1]->Fill(status[itt-1]+0.5); 

      }

    }

  } else {

    LogWarning("EBStatusFlagsTask") << EcalRawDataCollection_ << " not available";

  }

}

