#include <iostream>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"
#include "CondFormats/RunInfo/interface/RunSummary.h"
#include "CondFormats/RunInfo/interface/RunInfo.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#include <DQM/EcalCommon/interface/Numbers.h>

#include "DQM/EcalBarrelMonitorTasks/interface/EBDataCertificationTask.h"

using namespace cms;
using namespace edm;
using namespace std;

EBDataCertificationTask::EBDataCertificationTask(const ParameterSet& ps) {

  dqmStore_ = Service<DQMStore>().operator->();

  prefixME_ = ps.getUntrackedParameter<string>("prefixME", "");

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  mergeRuns_ = ps.getUntrackedParameter<bool>("mergeRuns", false);

  meEBDataCertificationSummary_ = 0;
  meEBDataCertificationSummaryMap_ = 0;
  for (int i = 0; i < 36; i++) {
    meEBDataCertification_[i] = 0;
  }

}

EBDataCertificationTask::~EBDataCertificationTask() {

}

void EBDataCertificationTask::beginJob(const EventSetup& c){

  char histo[200];
  
  if ( dqmStore_ ) {

    dqmStore_->setCurrentFolder(prefixME_ + "/EventInfo");
    
    sprintf(histo, "CertificationSummary");
    meEBDataCertificationSummary_ = dqmStore_->bookFloat(histo);
    meEBDataCertificationSummary_->Fill(0.0);

    sprintf(histo, "CertificationSummaryMap");
    meEBDataCertificationSummaryMap_ = dqmStore_->book2D(histo,histo, 72, 0., 72., 34, 0., 34.);
    meEBDataCertificationSummaryMap_->setAxisTitle("jphi", 1);
    meEBDataCertificationSummaryMap_->setAxisTitle("jeta", 2);

    dqmStore_->setCurrentFolder(prefixME_ + "/EventInfo/CertificationContents");

    for (int i = 0; i < 36; i++) {
      sprintf(histo, "EcalBarrel_%s", Numbers::sEB(i+1).c_str());
      meEBDataCertification_[i] = dqmStore_->bookFloat(histo);
      meEBDataCertification_[i]->Fill(0.0);
    }

  }

}

void EBDataCertificationTask::endJob(void) {

  if ( enableCleanup_ ) this->cleanup();

}

void EBDataCertificationTask::beginLuminosityBlock(const edm::LuminosityBlock& lumiBlock, const  edm::EventSetup& iSetup){

  this->reset();

  int nErrors = 0;
  int nErrorsEB[36];
  int nValidChannels = 0;
  int nValidChannelsEB[36];

  for (int i = 0; i < 36; i++) {
    nErrorsEB[i] = 0;
    nValidChannelsEB[i] = 0;
  }

  for ( int iett = 0; iett < 34; iett++ ) {
    for ( int iptt = 0; iptt < 72; iptt++ ) {
      meEBDataCertificationSummaryMap_->setBinContent( iptt+1, iett+1, 1.0 );
      int ism = ( iett<17 ) ? iptt/4 : 18+iptt/4;
      // placeholder
      if(1==1) {
        nValidChannelsEB[ism]++;
        nValidChannels++;
      }
    }
  }

  if( meEBDataCertificationSummary_ ) {
    if( nValidChannels>0 ) meEBDataCertificationSummary_->Fill( 1.0 - nErrors/nValidChannels );
    else meEBDataCertificationSummary_->Fill( 0.0 );
  }

  for (int i = 0; i < 36; i++) {
    if( meEBDataCertification_[i] ) {
      if( nValidChannelsEB[i]>0 ) meEBDataCertification_[i]->Fill( 1.0 - nErrorsEB[i]/nValidChannelsEB[i] );
      else meEBDataCertification_[i]->Fill( 0.0 );
    }
  }

}

void EBDataCertificationTask::endLuminosityBlock(const edm::LuminosityBlock&  lumiBlock, const  edm::EventSetup& iSetup) {

}

void EBDataCertificationTask::reset(void) {

  if ( meEBDataCertificationSummary_ ) meEBDataCertificationSummary_->Reset();

  for (int i = 0; i < 36; i++) {
    if ( meEBDataCertification_[i] ) meEBDataCertification_[i]->Reset();
  }

  if ( meEBDataCertificationSummaryMap_ ) meEBDataCertificationSummaryMap_->Reset();
  
}


void EBDataCertificationTask::cleanup(void){
  
  if ( dqmStore_ ) {

    dqmStore_->setCurrentFolder(prefixME_ + "/EventInfo");
    
    if ( meEBDataCertificationSummary_ ) dqmStore_->removeElement( meEBDataCertificationSummary_->getName() );

    if ( meEBDataCertificationSummaryMap_ ) dqmStore_->removeElement( meEBDataCertificationSummaryMap_->getName() );

    dqmStore_->setCurrentFolder(prefixME_ + "/EventInfo/CertificationContents");

    for (int i = 0; i < 36; i++) {
      if ( meEBDataCertification_[i] ) dqmStore_->removeElement( meEBDataCertification_[i]->getName() );
    }

  }

}

void EBDataCertificationTask::analyze(const Event& e, const EventSetup& c){ 

}
