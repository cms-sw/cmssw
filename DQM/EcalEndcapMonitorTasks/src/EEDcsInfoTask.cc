/*
 * \file EEDcsInfoTask.cc
 *
 * $Date: 2010/08/08 08:56:00 $
 * $Revision: 1.16 $
 * \author E. Di Marco
 *
*/

#include <iostream>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "CondFormats/EcalObjects/interface/EcalDCSTowerStatus.h"
#include "CondFormats/DataRecord/interface/EcalDCSTowerStatusRcd.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DQM/EcalCommon/interface/Numbers.h"

#include "DQM/EcalEndcapMonitorTasks/interface/EEDcsInfoTask.h"

EEDcsInfoTask::EEDcsInfoTask(const edm::ParameterSet& ps) {

  dqmStore_ = edm::Service<DQMStore>().operator->();

  prefixME_ = ps.getUntrackedParameter<std::string>("prefixME", "");

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  mergeRuns_ = ps.getUntrackedParameter<bool>("mergeRuns", false);

  meEEDcsFraction_ = 0;
  meEEDcsActiveMap_ = 0;
  for (int i = 0; i < 18; i++) {
    meEEDcsActive_[i] = 0;
  }

}

EEDcsInfoTask::~EEDcsInfoTask() {

}

void EEDcsInfoTask::beginJob(void){

  char histo[200];

  if ( dqmStore_ ) {

    dqmStore_->setCurrentFolder(prefixME_ + "/EventInfo");

    sprintf(histo, "DCSSummary");
    meEEDcsFraction_ = dqmStore_->bookFloat(histo);
    meEEDcsFraction_->Fill(0.0);

    sprintf(histo, "DCSSummaryMap");
    meEEDcsActiveMap_ = dqmStore_->book2D(histo,histo, 200, 0., 200., 100, 0., 100.);
    meEEDcsActiveMap_->setAxisTitle("jx", 1);
    meEEDcsActiveMap_->setAxisTitle("jy", 2);

    dqmStore_->setCurrentFolder(prefixME_ + "/EventInfo/DCSContents");

    for (int i = 0; i < 18; i++) {
      sprintf(histo, "EcalEndcap_%s", Numbers::sEE(i+1).c_str());
      meEEDcsActive_[i] = dqmStore_->bookFloat(histo);
      meEEDcsActive_[i]->Fill(-1.0);
    }

  }

}

void EEDcsInfoTask::endJob(void) {

  if ( enableCleanup_ ) this->cleanup();

}

void EEDcsInfoTask::beginLuminosityBlock(const edm::LuminosityBlock& lumiBlock, const  edm::EventSetup& iSetup){

  // information is by run, so fill the same for the run and for every lumi section
  for ( int itx = 0; itx < 40; itx++ ) {
    for ( int ity = 0; ity < 20; ity++ ) {
      readyLumi[itx][ity] = 1;
    }
  }

  if ( !iSetup.find( edm::eventsetup::EventSetupRecordKey::makeKey<EcalDCSTowerStatusRcd>() ) ) {
    edm::LogWarning("EEDcsInfoTask") << "EcalDAQTowerStatus record not found";
    return;
  }

  edm::ESHandle<EcalDCSTowerStatus> pDCSStatus;
  iSetup.get<EcalDCSTowerStatusRcd>().get(pDCSStatus);
  if ( !pDCSStatus.isValid() ) {
    edm::LogWarning("EEDcsInfoTask") << "EcalDCSTowerStatus record not valid";
    return;
  }
  const EcalDCSTowerStatus* dcsStatus = pDCSStatus.product();

  for(int iz=-1; iz<=1; iz+=2) {
    for(int itx=0 ; itx<20; itx++) {
      for(int ity=0 ; ity<20; ity++) {
        if (EcalScDetId::validDetId(itx+1,ity+1,iz )){

          EcalScDetId eeid(itx+1,ity+1,iz);

          uint16_t dbStatus = 0; // 0 = good
          EcalDCSTowerStatus::const_iterator dcsStatusIt = dcsStatus->find( eeid.rawId() );
          if ( dcsStatusIt != dcsStatus->end() ) dbStatus = dcsStatusIt->getStatusCode();

          if ( dbStatus > 0 ) {
            int offsetSC = (iz > 0) ? 0 : 20;
            readyRun[offsetSC+itx][ity] = 0;
            readyLumi[offsetSC+itx][ity] = 0;
          }

        }
      }
    }
  }

}

void EEDcsInfoTask::endLuminosityBlock(const edm::LuminosityBlock&  lumiBlock, const  edm::EventSetup& iSetup) {

  this->fillMonitorElements(readyLumi);

}

void EEDcsInfoTask::beginRun(const edm::Run& r, const edm::EventSetup& c) {

  if ( ! mergeRuns_ ) this->reset();

  for ( int itx = 0; itx < 40; itx++ ) {
    for ( int ity = 0; ity < 20; ity++ ) {
      readyRun[itx][ity] = 1;
    }
  }

}

void EEDcsInfoTask::endRun(const edm::Run& r, const edm::EventSetup& c) {

  this->fillMonitorElements(readyRun);

}

void EEDcsInfoTask::reset(void) {

  if ( meEEDcsFraction_ ) meEEDcsFraction_->Reset();

  for (int i = 0; i < 18; i++) {
    if ( meEEDcsActive_[i] ) meEEDcsActive_[i]->Reset();
  }

  if ( meEEDcsActiveMap_ ) meEEDcsActiveMap_->Reset();

}


void EEDcsInfoTask::cleanup(void){

  if ( dqmStore_ ) {

    dqmStore_->setCurrentFolder(prefixME_ + "/EventInfo");

    if ( meEEDcsFraction_ ) dqmStore_->removeElement( meEEDcsFraction_->getName() );

    if ( meEEDcsActiveMap_ ) dqmStore_->removeElement( meEEDcsActiveMap_->getName() );

    dqmStore_->setCurrentFolder(prefixME_ + "/EventInfo/DCSContents");

    for (int i = 0; i < 18; i++) {
      if ( meEEDcsActive_[i] ) dqmStore_->removeElement( meEEDcsActive_[i]->getName() );
    }

  }

}

void EEDcsInfoTask::fillMonitorElements(int ready[40][20]) {

  float readySum[18];
  int nValidChannels[18];
  for ( int ism = 0; ism < 18; ism++ ) {
    readySum[ism] = 0;
    nValidChannels[ism] = 0;
  }
  float readySumTot = 0.;
  int nValidChannelsTot = 0;

  for ( int iz = -1; iz < 2; iz+=2 ) {
    for ( int itx = 0; itx < 20; itx++ ) {
      for ( int ity = 0; ity < 20; ity++ ) {
        for ( int h = 0; h < 5; h++ ) {
          for ( int k = 0; k < 5; k++ ) {

            int ix = 5*itx + h;
            int iy = 5*ity + k;

            int offsetSC = (iz > 0) ? 0 : 20;
            int offset = (iz > 0) ? 0 : 100;

            if( EEDetId::validDetId(ix+1, iy+1, iz) ) {

              if(meEEDcsActiveMap_) meEEDcsActiveMap_->setBinContent( offset+ix+1, iy+1, ready[offsetSC+itx][ity] );

              EEDetId id = EEDetId(ix+1, iy+1, iz, EEDetId::XYMODE);

              int ism = Numbers::iSM(id);
              if(ready[offsetSC+itx][ity]) {
                readySum[ism-1]++;
                readySumTot++;
              }

              nValidChannels[ism-1]++;
              nValidChannelsTot++;

            } else {
              if(meEEDcsActiveMap_) meEEDcsActiveMap_->setBinContent( offset+ix+1, iy+1, -1.0 );
            }

          }
        }
      }
    }
  }

  for ( int ism = 0; ism < 18; ism++ ) {
    if( meEEDcsActive_[ism] ) meEEDcsActive_[ism]->Fill( readySum[ism]/float(nValidChannels[ism]) );
  }

  if( meEEDcsFraction_ ) meEEDcsFraction_->Fill( readySumTot/float(nValidChannelsTot) );

}

void EEDcsInfoTask::analyze(const edm::Event& e, const edm::EventSetup& c){

}
