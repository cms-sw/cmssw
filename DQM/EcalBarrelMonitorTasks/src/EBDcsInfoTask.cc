/*
 * \file EBDcsInfoTask.cc
 *
 * $Date: 2012/04/27 13:46:02 $
 * $Revision: 1.21 $
 * \author E. Di Marco
 *
*/

#include <iostream>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "CondFormats/EcalObjects/interface/EcalDCSTowerStatus.h"
#include "CondFormats/DataRecord/interface/EcalDCSTowerStatusRcd.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DQM/EcalCommon/interface/Numbers.h"

#include "DQM/EcalBarrelMonitorTasks/interface/EBDcsInfoTask.h"

EBDcsInfoTask::EBDcsInfoTask(const edm::ParameterSet& ps) {

  dqmStore_ = edm::Service<DQMStore>().operator->();

  prefixME_ = ps.getUntrackedParameter<std::string>("prefixME", "");

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  mergeRuns_ = ps.getUntrackedParameter<bool>("mergeRuns", false);

  meEBDcsFraction_ = 0;
  meEBDcsActiveMap_ = 0;
  for (int i = 0; i < 36; i++) {
    meEBDcsActive_[i] = 0;
  }

}

EBDcsInfoTask::~EBDcsInfoTask() {

}

void EBDcsInfoTask::beginJob(void){

  std::string name;

  if ( dqmStore_ ) {

    dqmStore_->setCurrentFolder(prefixME_ + "/EventInfo");

    name = "DCSSummary";
    meEBDcsFraction_ = dqmStore_->bookFloat(name);
    meEBDcsFraction_->Fill(0.0);

    name = "DCSSummaryMap";
    meEBDcsActiveMap_ = dqmStore_->book2D(name, name, 72, 0., 72., 34, 0., 34.);
    meEBDcsActiveMap_->setAxisTitle("jphi", 1);
    meEBDcsActiveMap_->setAxisTitle("jeta", 2);

    dqmStore_->setCurrentFolder(prefixME_ + "/EventInfo/DCSContents");

    for (int i = 0; i < 36; i++) {
      name = "EcalBarrel_" + Numbers::sEB(i+1);
      meEBDcsActive_[i] = dqmStore_->bookFloat(name);
      meEBDcsActive_[i]->Fill(-1.0);
    }

  }

}

void EBDcsInfoTask::endJob(void) {

  if ( enableCleanup_ ) this->cleanup();

}

void EBDcsInfoTask::beginLuminosityBlock(const edm::LuminosityBlock& lumiBlock, const  edm::EventSetup& iSetup){

  // information is by run, so fill the same for the run and for every lumi section
  for ( int iett = 0; iett < 34; iett++ ) {
    for ( int iptt = 0; iptt < 72; iptt++ ) {
      readyLumi[iptt][iett] = 1;
    }
  }

  if ( !iSetup.find( edm::eventsetup::EventSetupRecordKey::makeKey<EcalDCSTowerStatusRcd>() ) ) {
    edm::LogWarning("EBDcsInfoTask") << "EcalDCSTowerStatus record not found";
    return;
  }

  edm::ESHandle<EcalDCSTowerStatus> pDCSStatus;
  iSetup.get<EcalDCSTowerStatusRcd>().get(pDCSStatus);
  if ( !pDCSStatus.isValid() ) {
    edm::LogWarning("EBDcsInfoTask") << "EcalDCSTowerStatus record not valid";
    return;
  }
  const EcalDCSTowerStatus* dcsStatus = pDCSStatus.product();

  for(int iz=-1; iz<=1; iz+=2) {
    for(int iptt=0 ; iptt<72; iptt++) {
      for(int iett=0 ; iett<17; iett++) {
        if (EcalTrigTowerDetId::validDetId(iz,EcalBarrel,iett+1,iptt+1 )){

          EcalTrigTowerDetId ebid(iz,EcalBarrel,iett+1,iptt+1);

          uint16_t dbStatus = 0; // 0 = good
          EcalDCSTowerStatus::const_iterator dcsStatusIt = dcsStatus->find( ebid.rawId() );
          if ( dcsStatusIt != dcsStatus->end() ) dbStatus = dcsStatusIt->getStatusCode();

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

void EBDcsInfoTask::endLuminosityBlock(const edm::LuminosityBlock&  lumiBlock, const  edm::EventSetup& iSetup) {

  this->fillMonitorElements(readyLumi);

}

void EBDcsInfoTask::beginRun(const edm::Run& r, const edm::EventSetup& c) {

  if ( ! mergeRuns_ ) this->reset();

  for ( int iett = 0; iett < 34; iett++ ) {
    for ( int iptt = 0; iptt < 72; iptt++ ) {
      readyRun[iptt][iett] = 1;
    }
  }

}

void EBDcsInfoTask::endRun(const edm::Run& r, const edm::EventSetup& c) {

  this->fillMonitorElements(readyRun);

}

void EBDcsInfoTask::reset(void) {

  if ( meEBDcsFraction_ ) meEBDcsFraction_->Reset();

  for (int i = 0; i < 36; i++) {
    if ( meEBDcsActive_[i] ) meEBDcsActive_[i]->Reset();
  }

  if ( meEBDcsActiveMap_ ) meEBDcsActiveMap_->Reset();

}


void EBDcsInfoTask::cleanup(void){

  if ( dqmStore_ ) {

    dqmStore_->setCurrentFolder(prefixME_ + "/EventInfo");

    if ( meEBDcsFraction_ ) dqmStore_->removeElement( meEBDcsFraction_->getName() );

    if ( meEBDcsActiveMap_ ) dqmStore_->removeElement( meEBDcsActiveMap_->getName() );

    dqmStore_->setCurrentFolder(prefixME_ + "/EventInfo/DCSContents");

    for (int i = 0; i < 36; i++) {
      if ( meEBDcsActive_[i] ) dqmStore_->removeElement( meEBDcsActive_[i]->getName() );
    }

  }

}

void EBDcsInfoTask::fillMonitorElements(int ready[72][34]) {

  float readySum[36];
  for ( int ism = 0; ism < 36; ism++ ) readySum[ism] = 0;
  float readySumTot = 0.;

  for ( int iett = 0; iett < 34; iett++ ) {
    for ( int iptt = 0; iptt < 72; iptt++ ) {

      if(meEBDcsActiveMap_) meEBDcsActiveMap_->setBinContent( iptt+1, iett+1, ready[iptt][iett] );

      int ism = ( iett<17 ) ? iptt/4 : 18+iptt/4;
      if(ready[iptt][iett]) {
        readySum[ism]++;
        readySumTot++;
      }

    }
  }

  for ( int ism = 0; ism < 36; ism++ ) {
    if( meEBDcsActive_[ism] ) meEBDcsActive_[ism]->Fill( readySum[ism]/68. );
  }

  if( meEBDcsFraction_ ) meEBDcsFraction_->Fill(readySumTot/34./72.);

}

void EBDcsInfoTask::analyze(const edm::Event& e, const edm::EventSetup& c){

}
