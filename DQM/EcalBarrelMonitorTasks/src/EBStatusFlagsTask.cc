/*
 * \file EBStatusFlagsTask.cc
 *
 * $Date: 2011/08/30 09:30:33 $
 * $Revision: 1.34 $
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

#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"

#include "DQM/EcalCommon/interface/Numbers.h"

#include "DQM/EcalBarrelMonitorTasks/interface/EBStatusFlagsTask.h"

EBStatusFlagsTask::EBStatusFlagsTask(const edm::ParameterSet& ps){

  init_ = false;

  dqmStore_ = edm::Service<DQMStore>().operator->();

  prefixME_ = ps.getUntrackedParameter<std::string>("prefixME", "");

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  mergeRuns_ = ps.getUntrackedParameter<bool>("mergeRuns", false);

  EcalRawDataCollection_ = ps.getParameter<edm::InputTag>("EcalRawDataCollection");

  for (int i = 0; i < 36; i++) {

    meFEchErrors_[i][0] = 0;
    meFEchErrors_[i][1] = 0;
    meFEchErrors_[i][2] = 0;
  }

  meFEchErrorsByLumi_ = 0;

  ievt_ = 0;
}

EBStatusFlagsTask::~EBStatusFlagsTask(){

}

void EBStatusFlagsTask::beginJob(void){

  ievt_ = 0;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/FEStatus");
    dqmStore_->rmdir(prefixME_ + "/FEStatus");
  }

}

void EBStatusFlagsTask::beginLuminosityBlock(const edm::LuminosityBlock& lumiBlock, const  edm::EventSetup& iSetup) {

  if ( meFEchErrorsByLumi_ ) meFEchErrorsByLumi_->Reset();

}

void EBStatusFlagsTask::endLuminosityBlock(const edm::LuminosityBlock&  lumiBlock, const  edm::EventSetup& iSetup) {
}

void EBStatusFlagsTask::beginRun(const edm::Run& r, const edm::EventSetup& c) {

  Numbers::initGeometry(c, false);

  if ( ! mergeRuns_ ) this->reset();

}

void EBStatusFlagsTask::endRun(const edm::Run& r, const edm::EventSetup& c) {

}

void EBStatusFlagsTask::reset(void) {

  for (int i = 0; i < 36; i++) {

    if ( meFEchErrors_[i][0] ) meFEchErrors_[i][0]->Reset();
    if ( meFEchErrors_[i][1] ) meFEchErrors_[i][1]->Reset();
    if ( meFEchErrors_[i][2] ) meFEchErrors_[i][2]->Reset();
  }
  if ( meFEchErrorsByLumi_ ) meFEchErrorsByLumi_->Reset();
}

void EBStatusFlagsTask::setup(void){

  init_ = true;

  std::string name;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/FEStatus");

    dqmStore_->setCurrentFolder(prefixME_ + "/FEStatus/Flags");
    for (int i = 0; i < 36; i++) {
      name = "FEStatusTask front-end status bits " + Numbers::sEB(i+1);
      meFEchErrors_[i][2] = dqmStore_->book1D(name, name, 16, 0., 16.);
      meFEchErrors_[i][2]->setBinLabel(1+0, "ACTIVE", 1);
      meFEchErrors_[i][2]->setBinLabel(1+1, "DISABLED", 1);
      meFEchErrors_[i][2]->setBinLabel(1+2, "TIMEOUT", 1);
      meFEchErrors_[i][2]->setBinLabel(1+3, "HEADER", 1);
      meFEchErrors_[i][2]->setBinLabel(1+4, "CHANNEL ID", 1);
      meFEchErrors_[i][2]->setBinLabel(1+5, "LINK", 1);
      meFEchErrors_[i][2]->setBinLabel(1+6, "BLOCKSIZE", 1);
      meFEchErrors_[i][2]->setBinLabel(1+7, "SUPPRESSED", 1);
      meFEchErrors_[i][2]->setBinLabel(1+8, "FORCED FS", 1);
      meFEchErrors_[i][2]->setBinLabel(1+9, "L1A SYNC", 1);
      meFEchErrors_[i][2]->setBinLabel(1+10, "BX SYNC", 1);
      meFEchErrors_[i][2]->setBinLabel(1+11, "L1A+BX SYNC", 1);
      meFEchErrors_[i][2]->setBinLabel(1+12, "FIFO FULL+L1A", 1);
      meFEchErrors_[i][2]->setBinLabel(1+13, "H PARITY", 1);
      meFEchErrors_[i][2]->setBinLabel(1+14, "V PARITY", 1);
      meFEchErrors_[i][2]->setBinLabel(1+15, "FORCED ZS", 1);
      dqmStore_->tag(meFEchErrors_[i][2], i+1);
    }


    dqmStore_->setCurrentFolder(prefixME_ + "/FEStatus");
    // checking the number of front-end errors in each DCC for each lumi
    // tower error is weighted by 1/68
    // bin 0 contains the number of processed events in the lumi (for normalization)
    name = "FEStatusTask errors by lumi EB";
    meFEchErrorsByLumi_ = dqmStore_->book1D(name, name, 36, 1., 37.);
    meFEchErrorsByLumi_->setLumiFlag();
    for (int i = 0; i < 36; i++) {
      meFEchErrorsByLumi_->setBinLabel(i+1, Numbers::sEB(i+1), 1);
    }

    dqmStore_->setCurrentFolder(prefixME_ + "/FEStatus/Errors");

  }

}

void EBStatusFlagsTask::cleanup(void){

  if ( ! init_ ) return;

  if ( dqmStore_ ) {

    for (int i = 0; i < 36; i++) {
      if ( meFEchErrors_[i][0] ) dqmStore_->removeElement( meFEchErrors_[i][0]->getFullname() );
      meFEchErrors_[i][0] = 0;
      if ( meFEchErrors_[i][1] ) dqmStore_->removeElement( meFEchErrors_[i][1]->getFullname() );
      meFEchErrors_[i][1] = 0;
      if ( meFEchErrors_[i][2] ) dqmStore_->removeElement( meFEchErrors_[i][2]->getFullname() );
      meFEchErrors_[i][2] = 0;
    }

    if ( meFEchErrorsByLumi_ ) dqmStore_->removeElement( meFEchErrorsByLumi_->getFullname() );
    meFEchErrorsByLumi_ = 0;

  }

  init_ = false;

}

void EBStatusFlagsTask::endJob(void){

  edm::LogInfo("EBStatusFlagsTask") << "analyzed " << ievt_ << " events";

  if ( enableCleanup_ ) this->cleanup();

}

void EBStatusFlagsTask::analyze(const edm::Event& e, const edm::EventSetup& c){

  if ( ! init_ ) this->setup();

  ievt_++;

  // fill bin 0 with number of events in the lumi
  if ( meFEchErrorsByLumi_ ) meFEchErrorsByLumi_->Fill(0.);

  edm::Handle<EcalRawDataCollection> dcchs;

  std::string dir, name;
  std::stringstream ss;
  MonitorElement *me(0);

  if ( e.getByLabel(EcalRawDataCollection_, dcchs) ) {

    for ( EcalRawDataCollection::const_iterator dcchItr = dcchs->begin(); dcchItr != dcchs->end(); ++dcchItr ) {

      if ( Numbers::subDet( *dcchItr ) != EcalBarrel ) continue;

      int ism = Numbers::iSM( *dcchItr, EcalBarrel );
      float xism = ism + 0.5;

      const std::vector<short> status = dcchItr->getFEStatus();

      for ( unsigned int itt=1; itt<=status.size(); itt++ ) {

        if ( itt > 70 ) continue;

	if ( ! ( status[itt-1] == 0 || status[itt-1] == 1 || status[itt-1] == 7 || status[itt-1] == 8 || status[itt-1] == 12 || status[itt-1] == 15 ) ) {

	  dir = prefixME_ + "/FEStatus/Errors/";

	  ss.str("");
	  ss << dcchItr->id() << " " << itt;
	  name = "FEStatusTask Error FE " + ss.str();
	  me = dqmStore_->get(dir + name);
	  if (!me) {
	    dqmStore_->setCurrentFolder(dir);
	    me = dqmStore_->book1D(name, name, 1, 0., 1.);
	  }
	  if(me) me->Fill(0.5);

	  if ( itt >= 1 && itt <= 68 ) {
            if ( meFEchErrorsByLumi_ ) meFEchErrorsByLumi_->Fill(xism);
          }
	}

        if ( meFEchErrors_[ism-1][2] ) meFEchErrors_[ism-1][2]->Fill(status[itt-1]+0.5);

      }

    }

  } else {

    edm::LogWarning("EBStatusFlagsTask") << EcalRawDataCollection_ << " not available";

  }

}

