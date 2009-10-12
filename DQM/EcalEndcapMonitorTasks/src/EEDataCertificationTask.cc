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

#include "DQM/EcalEndcapMonitorTasks/interface/EEDataCertificationTask.h"

using namespace cms;
using namespace edm;
using namespace std;

EEDataCertificationTask::EEDataCertificationTask(const ParameterSet& ps) {

  dqmStore_ = Service<DQMStore>().operator->();

  prefixME_ = ps.getUntrackedParameter<string>("prefixME", "");

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  mergeRuns_ = ps.getUntrackedParameter<bool>("mergeRuns", false);

  meEEDataCertificationSummary_ = 0;
  meEEDataCertificationSummaryMap_ = 0;
  for (int i = 0; i < 18; i++) {
    meEEDataCertification_[i] = 0;
  }

}

EEDataCertificationTask::~EEDataCertificationTask() {

}

void EEDataCertificationTask::beginJob(const EventSetup& c){

  char histo[200];
  
  if ( dqmStore_ ) {

    dqmStore_->setCurrentFolder(prefixME_ + "/EventInfo");
    
    sprintf(histo, "CertificationSummary");
    meEEDataCertificationSummary_ = dqmStore_->bookFloat(histo);
    meEEDataCertificationSummary_->Fill(0.0);

    sprintf(histo, "CertificationSummaryMap");
    meEEDataCertificationSummaryMap_ = dqmStore_->book2D(histo,histo, 200, 0., 200., 100, 0., 100.);
    meEEDataCertificationSummaryMap_->setAxisTitle("jx", 1);
    meEEDataCertificationSummaryMap_->setAxisTitle("jy", 2);
    
    dqmStore_->setCurrentFolder(prefixME_ + "/EventInfo/CertificationContents");

    for (int i = 0; i < 18; i++) {
      sprintf(histo, "EcalEndcap_%s", Numbers::sEE(i+1).c_str());
      meEEDataCertification_[i] = dqmStore_->bookFloat(histo);
      meEEDataCertification_[i]->Fill(0.0);
    }

  }

}

void EEDataCertificationTask::endJob(void) {

  if ( enableCleanup_ ) this->cleanup();

}

void EEDataCertificationTask::beginLuminosityBlock(const edm::LuminosityBlock& lumiBlock, const  edm::EventSetup& iSetup){

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
          meEEDataCertificationSummaryMap_->setBinContent( jx, jy, 1.0 );

          EEDetId id = EEDetId(ix, iy, iz, EEDetId::XYMODE);
          
          int ism = Numbers::iSM(id);

          nValidChannelsEE[ism]++;
          nValidChannels++;

        }
        else meEEDataCertificationSummaryMap_->setBinContent( jx, jy, -1.0 );
      }
    }
  }

  if( meEEDataCertificationSummary_ ) { 
    if( nValidChannels>0 ) meEEDataCertificationSummary_->Fill( 1.0 - nErrors/nValidChannels );
    else meEEDataCertificationSummary_->Fill( 0.0 );
  }

  for (int i = 0; i < 18; i++) {
    if( meEEDataCertification_[i] ) {
      if( nValidChannelsEE[i]>0 ) meEEDataCertification_[i]->Fill( 1.0 - nErrorsEE[i]/nValidChannelsEE[i] );
      else meEEDataCertification_[i]->Fill( 0.0 );
    }
  }

}

void EEDataCertificationTask::endLuminosityBlock(const edm::LuminosityBlock&  lumiBlock, const  edm::EventSetup& iSetup) {

}

void EEDataCertificationTask::reset(void) {

  if ( meEEDataCertificationSummary_ ) meEEDataCertificationSummary_->Reset();

  for (int i = 0; i < 18; i++) {
    if ( meEEDataCertification_[i] ) meEEDataCertification_[i]->Reset();
  }

  if ( meEEDataCertificationSummaryMap_ ) meEEDataCertificationSummaryMap_->Reset();
  
}


void EEDataCertificationTask::cleanup(void){
  
  if ( dqmStore_ ) {

    dqmStore_->setCurrentFolder(prefixME_ + "/EventInfo");
    
    if ( meEEDataCertificationSummary_ ) dqmStore_->removeElement( meEEDataCertificationSummary_->getName() );

    if ( meEEDataCertificationSummaryMap_ ) dqmStore_->removeElement( meEEDataCertificationSummaryMap_->getName() );

    dqmStore_->setCurrentFolder(prefixME_ + "/EventInfo/CertificationContents");

    for (int i = 0; i < 18; i++) {
      if ( meEEDataCertification_[i] ) dqmStore_->removeElement( meEEDataCertification_[i]->getName() );
    }

  }

}

void EEDataCertificationTask::analyze(const Event& e, const EventSetup& c){ 

}
