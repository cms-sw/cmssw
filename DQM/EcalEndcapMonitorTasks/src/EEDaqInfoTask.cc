/*
 * \file EEDaqInfoTask.cc
 *
 * $Date: 2010/08/08 08:56:00 $
 * $Revision: 1.12 $
 * \author E. Di Marco
 *
*/

#include <iostream>
#include <vector>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "CondFormats/EcalObjects/interface/EcalDAQTowerStatus.h"
#include "CondFormats/DataRecord/interface/EcalDAQTowerStatusRcd.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#include "DQM/EcalCommon/interface/Numbers.h"

#include "DQM/EcalEndcapMonitorTasks/interface/EEDaqInfoTask.h"

EEDaqInfoTask::EEDaqInfoTask(const edm::ParameterSet& ps) {

  dqmStore_ = edm::Service<DQMStore>().operator->();

  prefixME_ = ps.getUntrackedParameter<std::string>("prefixME", "");

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  mergeRuns_ = ps.getUntrackedParameter<bool>("mergeRuns", false);

  meEEDaqFraction_ = 0;
  meEEDaqActiveMap_ = 0;
  for (int i = 0; i < 18; i++) {
    meEEDaqActive_[i] = 0;
  }

}

EEDaqInfoTask::~EEDaqInfoTask() {

}

void EEDaqInfoTask::beginJob(void){

  char histo[200];

  if ( dqmStore_ ) {

    dqmStore_->setCurrentFolder(prefixME_ + "/EventInfo");

    sprintf(histo, "DAQSummary");
    meEEDaqFraction_ = dqmStore_->bookFloat(histo);
    meEEDaqFraction_->Fill(0.0);

    sprintf(histo, "DAQSummaryMap");
    meEEDaqActiveMap_ = dqmStore_->book2D(histo,histo, 200, 0., 200., 100, 0., 100.);
    meEEDaqActiveMap_->setAxisTitle("jx", 1);
    meEEDaqActiveMap_->setAxisTitle("jy", 2);

    dqmStore_->setCurrentFolder(prefixME_ + "/EventInfo/DAQContents");

    for (int i = 0; i < 18; i++) {
      sprintf(histo, "EcalEndcap_%s", Numbers::sEE(i+1).c_str());
      meEEDaqActive_[i] = dqmStore_->bookFloat(histo);
      meEEDaqActive_[i]->Fill(0.0);
    }

  }

}

void EEDaqInfoTask::endJob(void) {

  if ( enableCleanup_ ) this->cleanup();

}

void EEDaqInfoTask::beginLuminosityBlock(const edm::LuminosityBlock& lumiBlock, const  edm::EventSetup& iSetup){

  // information is by run, so fill the same for the run and for every lumi section
  for ( int itx = 0; itx < 40; itx++ ) {
    for ( int ity = 0; ity < 20; ity++ ) {
      readyLumi[itx][ity] = 1;
    }
  }

  if ( !iSetup.find( edm::eventsetup::EventSetupRecordKey::makeKey<EcalDAQTowerStatusRcd>() ) ) {
    edm::LogWarning("EEDaqInfoTask") << "EcalDAQTowerStatus record not found";
    return;
  }

  edm::ESHandle<EcalDAQTowerStatus> pDAQStatus;
  iSetup.get<EcalDAQTowerStatusRcd>().get(pDAQStatus);
  if ( !pDAQStatus.isValid() ) {
    edm::LogWarning("EEDaqInfoTask") << "EcalDAQTowerStatus record not valid";
    return;
  }
  const EcalDAQTowerStatus* daqStatus = pDAQStatus.product();

  for(int iz=-1; iz<=1; iz+=2) {
    for(int itx=0 ; itx<20; itx++) {
      for(int ity=0 ; ity<20; ity++) {
        if (EcalScDetId::validDetId(itx+1,ity+1,iz )){

          EcalScDetId eeid(itx+1,ity+1,iz);

          uint16_t dbStatus = 0; // 0 = good
          EcalDAQTowerStatus::const_iterator daqStatusIt = daqStatus->find( eeid.rawId() );
          if ( daqStatusIt != daqStatus->end() ) dbStatus = daqStatusIt->getStatusCode();

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

void EEDaqInfoTask::endLuminosityBlock(const edm::LuminosityBlock&  lumiBlock, const  edm::EventSetup& iSetup) {

  this->fillMonitorElements(readyLumi);

}

void EEDaqInfoTask::beginRun(const edm::Run& r, const edm::EventSetup& c) {

  if ( ! mergeRuns_ ) this->reset();

  for ( int itx = 0; itx < 40; itx++ ) {
    for ( int ity = 0; ity < 20; ity++ ) {
      readyRun[itx][ity] = 1;
    }
  }

}

void EEDaqInfoTask::endRun(const edm::Run& r, const edm::EventSetup& c) {

  this->fillMonitorElements(readyRun);

}

void EEDaqInfoTask::reset(void) {

  if ( meEEDaqFraction_ ) meEEDaqFraction_->Reset();

  for (int i = 0; i < 18; i++) {
    if ( meEEDaqActive_[i] ) meEEDaqActive_[i]->Reset();
  }

  if ( meEEDaqActiveMap_ ) meEEDaqActiveMap_->Reset();

}


void EEDaqInfoTask::cleanup(void){

  if ( dqmStore_ ) {

    dqmStore_->setCurrentFolder(prefixME_ + "/EventInfo");

    if ( meEEDaqFraction_ ) dqmStore_->removeElement( meEEDaqFraction_->getName() );

    if ( meEEDaqActiveMap_ ) dqmStore_->removeElement( meEEDaqActiveMap_->getName() );

    dqmStore_->setCurrentFolder(prefixME_ + "/EventInfo/DAQContents");

    for (int i = 0; i < 18; i++) {
      if ( meEEDaqActive_[i] ) dqmStore_->removeElement( meEEDaqActive_[i]->getName() );
    }

  }

}

void EEDaqInfoTask::fillMonitorElements(int ready[40][20]) {

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

              if(meEEDaqActiveMap_) meEEDaqActiveMap_->setBinContent( offset+ix+1, iy+1, ready[offsetSC+itx][ity] );

              EEDetId id = EEDetId(ix+1, iy+1, iz, EEDetId::XYMODE);

              int ism = Numbers::iSM(id);
              if(ready[offsetSC+itx][ity]) {
                readySum[ism-1]++;
                readySumTot++;
              }

              nValidChannels[ism-1]++;
              nValidChannelsTot++;

            } else {
              if(meEEDaqActiveMap_) meEEDaqActiveMap_->setBinContent( offset+ix+1, iy+1, -1.0 );
            }

          }
        }
      }
    }
  }

  for ( int ism = 0; ism < 18; ism++ ) {
    if( meEEDaqActive_[ism] ) meEEDaqActive_[ism]->Fill( readySum[ism]/float(nValidChannels[ism]) );
  }

  if( meEEDaqFraction_ ) meEEDaqFraction_->Fill( readySumTot/float(nValidChannelsTot) );

}

void EEDaqInfoTask::analyze(const edm::Event& e, const edm::EventSetup& c){

}
