/*
 * \file EBDaqInfoTask.cc
 *
 * $Date: 2012/04/27 13:46:01 $
 * $Revision: 1.15 $
 * \author E. Di Marco
 *
*/

#include <iostream>
#include <vector>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "CondFormats/EcalObjects/interface/EcalDAQTowerStatus.h"
#include "CondFormats/DataRecord/interface/EcalDAQTowerStatusRcd.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#include "DQM/EcalCommon/interface/Numbers.h"

#include "DQM/EcalBarrelMonitorTasks/interface/EBDaqInfoTask.h"

EBDaqInfoTask::EBDaqInfoTask(const edm::ParameterSet& ps) {

  dqmStore_ = edm::Service<DQMStore>().operator->();

  prefixME_ = ps.getUntrackedParameter<std::string>("prefixME", "");

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  mergeRuns_ = ps.getUntrackedParameter<bool>("mergeRuns", false);

  meEBDaqFraction_ = 0;
  meEBDaqActiveMap_ = 0;
  for (int i = 0; i < 36; i++) {
    meEBDaqActive_[i] = 0;
  }

}

EBDaqInfoTask::~EBDaqInfoTask() {

}

void EBDaqInfoTask::beginJob(void){

  std::string name;

  if ( dqmStore_ ) {

    dqmStore_->setCurrentFolder(prefixME_ + "/EventInfo");

    name = "DAQSummary";
    meEBDaqFraction_ = dqmStore_->bookFloat(name);
    meEBDaqFraction_->Fill(0.0);

    name = "DAQSummaryMap";
    meEBDaqActiveMap_ = dqmStore_->book2D(name, name, 72, 0., 72., 34, 0., 34.);
    meEBDaqActiveMap_->setAxisTitle("jphi", 1);
    meEBDaqActiveMap_->setAxisTitle("jeta", 2);

    dqmStore_->setCurrentFolder(prefixME_ + "/EventInfo/DAQContents");

    for (int i = 0; i < 36; i++) {
      name = "EcalBarrel_" + Numbers::sEB(i+1);
      meEBDaqActive_[i] = dqmStore_->bookFloat(name);
      meEBDaqActive_[i]->Fill(0.0);
    }

  }

}

void EBDaqInfoTask::endJob(void) {

  if ( enableCleanup_ ) this->cleanup();

}

void EBDaqInfoTask::beginLuminosityBlock(const edm::LuminosityBlock& lumiBlock, const  edm::EventSetup& iSetup){

  // information is by run, so fill the same for the run and for every lumi section
  for ( int iett = 0; iett < 34; iett++ ) {
    for ( int iptt = 0; iptt < 72; iptt++ ) {
      readyLumi[iptt][iett] = 1;
    }
  }


  if ( !iSetup.find( edm::eventsetup::EventSetupRecordKey::makeKey<EcalDAQTowerStatusRcd>() ) ) {
    edm::LogWarning("EBDaqInfoTask") << "EcalDAQTowerStatus record not found";
    return;
  }

  edm::ESHandle<EcalDAQTowerStatus> pDAQStatus;
  iSetup.get<EcalDAQTowerStatusRcd>().get(pDAQStatus);
    if ( !pDAQStatus.isValid() ) {
    edm::LogWarning("EBDaqInfoTask") << "EcalDAQTowerStatus record not valid";
    return;
  }
  const EcalDAQTowerStatus* daqStatus = pDAQStatus.product();

  for(int iz=-1; iz<=1; iz+=2) {
    for(int iptt=0 ; iptt<72; iptt++) {
      for(int iett=0 ; iett<17; iett++) {
        if (EcalTrigTowerDetId::validDetId(iz,EcalBarrel,iett+1,iptt+1 )){

          EcalTrigTowerDetId ebid(iz,EcalBarrel,iett+1,iptt+1);

          uint16_t dbStatus = 0; // 0 = good
          EcalDAQTowerStatus::const_iterator daqStatusIt = daqStatus->find( ebid.rawId() );
          if ( daqStatusIt != daqStatus->end() ) dbStatus = daqStatusIt->getStatusCode();

          if ( dbStatus > 0 ) {
            int ipttEB = iptt;
            int iettEB = (iz==-1) ? iett : 17+iett;
            readyRun[ipttEB][iettEB] = 0;
            readyLumi[ipttEB][iettEB] = 0;
          }

        }
      }
    }
  }

}



void EBDaqInfoTask::endLuminosityBlock(const edm::LuminosityBlock&  lumiBlock, const  edm::EventSetup& iSetup) {

    this->fillMonitorElements(readyLumi);

}

void EBDaqInfoTask::beginRun(const edm::Run& r, const edm::EventSetup& c) {

  if ( ! mergeRuns_ ) this->reset();

  for ( int iett = 0; iett < 34; iett++ ) {
    for ( int iptt = 0; iptt < 72; iptt++ ) {
      readyRun[iptt][iett] = 1;
    }
  }

}

void EBDaqInfoTask::endRun(const edm::Run& r, const edm::EventSetup& c) {

  this->fillMonitorElements(readyRun);

}

void EBDaqInfoTask::reset(void) {

  if ( meEBDaqFraction_ ) meEBDaqFraction_->Reset();

  for (int i = 0; i < 36; i++) {
    if ( meEBDaqActive_[i] ) meEBDaqActive_[i]->Reset();
  }

  if ( meEBDaqActiveMap_ ) meEBDaqActiveMap_->Reset();

}


void EBDaqInfoTask::cleanup(void){

  if ( dqmStore_ ) {

    dqmStore_->setCurrentFolder(prefixME_ + "/EventInfo");

    if ( meEBDaqFraction_ ) dqmStore_->removeElement( meEBDaqFraction_->getName() );

    if ( meEBDaqActiveMap_ ) dqmStore_->removeElement( meEBDaqActiveMap_->getName() );

    dqmStore_->setCurrentFolder(prefixME_ + "/EventInfo/DAQContents");

    for (int i = 0; i < 36; i++) {
      if ( meEBDaqActive_[i] ) dqmStore_->removeElement( meEBDaqActive_[i]->getName() );
    }

  }

}

void EBDaqInfoTask::fillMonitorElements(int ready[72][34]) {

  float readySum[36];
  for ( int ism = 0; ism < 36; ism++ ) readySum[ism] = 0;
  float readySumTot = 0.;

  for ( int iett = 0; iett < 34; iett++ ) {
    for ( int iptt = 0; iptt < 72; iptt++ ) {

      if(meEBDaqActiveMap_) meEBDaqActiveMap_->setBinContent( iptt+1, iett+1, ready[iptt][iett] );

      int ism = ( iett<17 ) ? iptt/4 : 18+iptt/4;
      if(ready[iptt][iett]) {
        readySum[ism]++;
        readySumTot++;
      }

    }
  }

  for ( int ism = 0; ism < 36; ism++ ) {
    if( meEBDaqActive_[ism] ) meEBDaqActive_[ism]->Fill( readySum[ism]/68. );
  }

  if( meEBDaqFraction_ ) meEBDaqFraction_->Fill(readySumTot/34./72.);

}

void EBDaqInfoTask::analyze(const edm::Event& e, const edm::EventSetup& c){

}
