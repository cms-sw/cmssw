#include <iostream>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include <DataFormats/EcalDetId/interface/EEDetId.h>

#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"
#include "CondFormats/RunInfo/interface/RunSummary.h"
#include "CondFormats/RunInfo/interface/RunInfo.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include <DQM/EcalCommon/interface/Numbers.h>

#include "DQM/EcalEndcapMonitorTasks/interface/EEDcsInfoTask.h"

using namespace cms;
using namespace edm;
using namespace std;

EEDcsInfoTask::EEDcsInfoTask(const ParameterSet& ps) {

  dqmStore_ = Service<DQMStore>().operator->();

  prefixME_ = ps.getUntrackedParameter<string>("prefixME", "");

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

void EEDcsInfoTask::beginJob(const EventSetup& c){

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
      meEEDcsActive_[i]->Fill(0.0);
    }

  }

}

void EEDcsInfoTask::endJob(void) {

  if ( enableCleanup_ ) this->cleanup();

}

void EEDcsInfoTask::beginLuminosityBlock(const edm::LuminosityBlock& lumiBlock, const  edm::EventSetup& iSetup){

  this->reset();

  int nErrors = 0;
  int nErrorsEE[18];
  int nValidChannels = 0;
  int nValidChannelsEE[18];

  for (int i = 0; i < 18; i++) {
    nErrorsEE[i] = 0;
    nValidChannelsEE[i] = 0;
  }

  for ( int iz = -1; iz < 2; iz+=2 ) {
    for ( int ix = 1; ix <= 100; ix++ ) {
      for ( int iy = 1; iy <= 100; iy++ ) {
        int jx = (iz==1) ? 100 + ix : ix;
        int jy = iy;
        if( EEDetId::validDetId(ix, iy, iz) ) {
          meEEDcsActiveMap_->setBinContent( jx, jy, 0.0 );

          EEDetId id = EEDetId(ix, iy, iz, EEDetId::XYMODE);
          
          int ism = Numbers::iSM(id);

          nValidChannelsEE[ism]++;
          nErrorsEE[ism]++;
          nValidChannels++;
          nErrors++;

        }
        else meEEDcsActiveMap_->setBinContent( jx, jy, -1.0 );
      }
    }
  }

  if( meEEDcsFraction_ ) { 
    if( nValidChannels>0 ) meEEDcsFraction_->Fill( 1.0 - nErrors/nValidChannels );
    else meEEDcsFraction_->Fill( 0.0 );
  }

  for (int i = 0; i < 18; i++) {
    if( meEEDcsActive_[i] ) {
      if( nValidChannelsEE[i]>0 ) meEEDcsActive_[i]->Fill( 1.0 - nErrorsEE[i]/nValidChannelsEE[i] );
      else meEEDcsActive_[i]->Fill( 0.0 );
    }
  }

}

void EEDcsInfoTask::endLuminosityBlock(const edm::LuminosityBlock&  lumiBlock, const  edm::EventSetup& iSetup) {

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

void EEDcsInfoTask::analyze(const Event& e, const EventSetup& c){ 

}
