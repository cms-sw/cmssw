/*
 * \file EcalBarrelMonitorModule.cc
 *
 * $Date: 2011/09/02 13:55:02 $
 * $Revision: 1.204 $
 * \author G. Della Ricca
 * \author G. Franzoni
 *
*/

#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "DQMServices/Core/interface/MonitorElement.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DQM/EcalCommon/interface/Numbers.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <memory>
#include <iostream>
#include <fstream>

#include "DQM/EcalBarrelMonitorModule/interface/EcalBarrelMonitorModule.h"

EcalBarrelMonitorModule::EcalBarrelMonitorModule(const edm::ParameterSet& ps){

  // verbose switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

  if ( verbose_ ) {
    std::cout << std::endl;
    std::cout << " *** Ecal Barrel Generic Monitor ***" << std::endl;
    std::cout << std::endl;
  }

  init_ = false;

  EcalRawDataCollection_ = ps.getParameter<edm::InputTag>("EcalRawDataCollection");

  // this should come from the event header
  runNumber_ = ps.getUntrackedParameter<int>("runNumber", 0);

  fixedRunNumber_ = false;
  if ( runNumber_ != 0 ) fixedRunNumber_ = true;

  if ( fixedRunNumber_ ) {
    if ( verbose_ ) {
      std::cout << " fixed Run Number = " << runNumber_ << std::endl;
    }
  }

  // this should come from the event header
  evtNumber_ = 0;

  // this should come from the EcalBarrel event header
  runType_ = ps.getUntrackedParameter<int>("runType", -1);
  evtType_ = runType_;

  fixedRunType_ = false;
  if ( runType_ != -1 ) fixedRunType_ = true;

  if ( fixedRunType_) {
    if ( verbose_ ) {
      std::cout << " fixed Run Type = " << runType_ << std::endl;
    }
  }

  // debug switch
  debug_ = ps.getUntrackedParameter<bool>("debug", false);

  if ( debug_ ) {
    if ( verbose_ ) {
      std::cout << " debug switch is ON" << std::endl;
    }
  } else {
    if ( verbose_ ) {
      std::cout << " debug switch is OFF" << std::endl;
    }
  }

  // prefixME path
  prefixME_ = ps.getUntrackedParameter<std::string>("prefixME", "");

  // enableCleanup switch
  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  // mergeRuns switch
  mergeRuns_ = ps.getUntrackedParameter<bool>("mergeRuns", false);

  if ( enableCleanup_ ) {
    if ( verbose_ ) {
      std::cout << " enableCleanup switch is ON" << std::endl;
    }
  } else {
    if ( verbose_ ) {
      std::cout << " enableCleanup switch is OFF" << std::endl;
    }
  }

  meStatus_ = 0;
  meRun_ = 0;
  meEvt_ = 0;
  meRunType_ = 0;

  ievt_ = 0;
  dqmStore_ = 0;

}

EcalBarrelMonitorModule::~EcalBarrelMonitorModule(){

}

void EcalBarrelMonitorModule::beginJob(void){

  if ( debug_ ) std::cout << "EcalBarrelMonitorModule: beginJob" << std::endl;

  ievt_ = 0;

  dqmStore_ = edm::Service<DQMStore>().operator->();

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EcalInfo");
    dqmStore_->rmdir(prefixME_ + "/EcalInfo");
  }

}

void EcalBarrelMonitorModule::beginRun(const edm::Run& r, const edm::EventSetup& c) {

  if ( debug_ ) std::cout << "EcalBarrelMonitorModule: beginRun" << std::endl;

  if ( ! mergeRuns_ ) this->reset();

}

void EcalBarrelMonitorModule::endRun(const edm::Run& r, const edm::EventSetup& c) {

  if ( debug_ ) std::cout << "EcalBarrelMonitorModule: endRun" << std::endl;

  // end-of-run
  if ( meStatus_ ) meStatus_->Fill(2);

  if ( meRun_ ) meRun_->Fill(runNumber_);
  if ( meEvt_ ) meEvt_->Fill(evtNumber_);

}

void EcalBarrelMonitorModule::reset(void) {


}

void EcalBarrelMonitorModule::setup(void){

  init_ = true;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EcalInfo");

    meStatus_ = dqmStore_->bookInt("STATUS");

    meRun_ = dqmStore_->bookInt("RUN");
    meEvt_ = dqmStore_->bookInt("EVT");

    meRunType_ = dqmStore_->bookInt("RUNTYPE");
  }

  // unknown
  if ( meStatus_ ) meStatus_->Fill(-1);

  if ( meRun_ ) meRun_->Fill(-1);
  if ( meEvt_ ) meEvt_->Fill(-1);

  if ( meRunType_ ) meRunType_->Fill(-1);

  std::string name;

}

void EcalBarrelMonitorModule::cleanup(void){

  if ( ! enableCleanup_ ) return;

  if ( dqmStore_ ) {

    dqmStore_->setCurrentFolder(prefixME_ + "/EcalInfo");

    if ( meStatus_ ) dqmStore_->removeElement( meStatus_->getName() );
    meStatus_ = 0;

    if ( meRun_ ) dqmStore_->removeElement( meRun_->getName() );
    meRun_ = 0;

    if ( meEvt_ ) dqmStore_->removeElement( meEvt_->getName() );
    meEvt_ = 0;

    if ( meRunType_ ) dqmStore_->removeElement( meRunType_->getName() );
    meRunType_ = 0;

  }

  init_ = false;

}

void EcalBarrelMonitorModule::endJob(void) {

  if ( debug_ ) std::cout << "EcalBarrelMonitorModule: endJob, ievt = " << ievt_ << std::endl;

  if ( dqmStore_ ) {
    meStatus_ = dqmStore_->get(prefixME_ + "/EcalInfo/STATUS");
    meRun_ = dqmStore_->get(prefixME_ + "/EcalInfo/RUN");
    meEvt_ = dqmStore_->get(prefixME_ + "/EcalInfo/EVT");
  }

  // end-of-run
  if ( meStatus_ ) meStatus_->Fill(2);

  if ( meRun_ ) meRun_->Fill(runNumber_);
  if ( meEvt_ ) meEvt_->Fill(evtNumber_);

  if ( init_ ) this->cleanup();

}

void EcalBarrelMonitorModule::analyze(const edm::Event& e, const edm::EventSetup& c){

  Numbers::initGeometry(c, verbose_);

  if ( ! init_ ) this->setup();

  ievt_++;

  LogDebug("EcalBarrelMonitorModule") << "processing event " << ievt_;

  if ( ! fixedRunNumber_ ) runNumber_ = e.id().run();

  evtNumber_ = e.id().event();

  edm::Handle<EcalRawDataCollection> dcchs;

  if ( e.getByLabel(EcalRawDataCollection_, dcchs) ) {

    if ( dcchs->size() == 0 ) {
      LogDebug("EcalBarrelMonitorModule") << EcalRawDataCollection_ << " is empty";
      return;
    }

    int nebc = 0;

    for ( EcalRawDataCollection::const_iterator dcchItr = dcchs->begin(); dcchItr != dcchs->end(); ++dcchItr ) {

      if ( Numbers::subDet( *dcchItr ) != EcalBarrel ) continue;

      nebc++;

    }

    for ( EcalRawDataCollection::const_iterator dcchItr = dcchs->begin(); dcchItr != dcchs->end(); ++dcchItr ) {

      if ( Numbers::subDet( *dcchItr ) != EcalBarrel ) continue;

      if ( ! fixedRunNumber_ ) {
        runNumber_ = dcchItr->getRunNumber();
      }

      evtNumber_ = dcchItr->getLV1();

      if ( ! fixedRunType_ ) {
        runType_ = dcchItr->getRunType();
        evtType_ = runType_;
      }

      if ( evtType_ < 0 || evtType_ > 22 ) evtType_ = -1;

    }

    LogDebug("EcalBarrelMonitorModule") << "event: " << ievt_ << " DCC headers collection size: " << nebc;

  } else {

    if ( evtType_ < 0 || evtType_ > 22 ) evtType_ = -1;

    edm::LogWarning("EcalBarrelMonitorModule") << EcalRawDataCollection_ << " not available";

  }

  if ( meRunType_ ) meRunType_->Fill(runType_);

  if ( ievt_ == 1 ) {
    LogDebug("EcalBarrelMonitorModule") << "processing run " << runNumber_;
    // begin-of-run
    if ( meStatus_ ) meStatus_->Fill(0);
  } else {
    // running
    if ( meStatus_ ) meStatus_->Fill(1);
  }

  if ( meRun_ ) meRun_->Fill(runNumber_);
  if ( meEvt_ ) meEvt_->Fill(evtNumber_);

}

