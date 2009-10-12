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

#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#include <DQM/EcalCommon/interface/Numbers.h>

#include "DQM/EcalEndcapMonitorTasks/interface/EEDaqInfoTask.h"

using namespace cms;
using namespace edm;
using namespace std;

EEDaqInfoTask::EEDaqInfoTask(const ParameterSet& ps) {

  dqmStore_ = Service<DQMStore>().operator->();

  prefixME_ = ps.getUntrackedParameter<string>("prefixME", "");

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  mergeRuns_ = ps.getUntrackedParameter<bool>("mergeRuns", false);

  EEMinusFedRangeMin_ = ps.getUntrackedParameter<int>("EEMinusFedRangeMin");
  EEMinusFedRangeMax_ = ps.getUntrackedParameter<int>("EEMinusFedRangeMax");
  EEPlusFedRangeMin_ = ps.getUntrackedParameter<int>("EEPlusFedRangeMin");
  EEPlusFedRangeMax_ = ps.getUntrackedParameter<int>("EEPlusFedRangeMax");

  meEEDaqFraction_ = 0;
  meEEDaqActiveMap_ = 0;
  for (int i = 0; i < 18; i++) {
    meEEDaqActive_[i] = 0;
  }

}

EEDaqInfoTask::~EEDaqInfoTask() {

}

void EEDaqInfoTask::beginJob(const EventSetup& c){

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

  this->reset();
  
  for ( int iz = -1; iz < 2; iz+=2 ) {
    for ( int ix = 1; ix <= 100; ix++ ) {
      for ( int iy = 1; iy <= 100; iy++ ) {
        int jx = (iz==1) ? 100 + ix : ix;
        int jy = iy;
        if( EEDetId::validDetId(ix, iy, iz) ) meEEDaqActiveMap_->setBinContent( jx, jy, 0.0 );
        else meEEDaqActiveMap_->setBinContent( jx, jy, -1.0 );
      }
    }
  }

  edm::eventsetup::EventSetupRecordKey recordKey(edm::eventsetup::EventSetupRecordKey::TypeTag::findType("RunInfoRcd"));

  if( iSetup.find( recordKey ) ) {

    edm::ESHandle<RunInfo> sumFED;
    iSetup.get<RunInfoRcd>().get(sumFED);    
   
    std::vector<int> FedsInIds= sumFED->m_fed_in;   

    float EEFedCount = 0.;

    for( unsigned int fedItr=0; fedItr<FedsInIds.size(); ++fedItr ) {

      int fedID=FedsInIds[fedItr];

      int iside = -1;
      int ism = -1;

      if( fedID >= EEMinusFedRangeMin_ && fedID <= EEMinusFedRangeMax_ ) {
        iside = 0;
        ism = fedID - EEMinusFedRangeMin_ + 1;
      } else if( fedID >= EEPlusFedRangeMin_ && fedID <= EEPlusFedRangeMax_ ) {
        iside = 1;
        ism = fedID - EEPlusFedRangeMin_ + 10;
      }

      if ( iside > -1 ) {

        EEFedCount++;

        if ( meEEDaqActive_[ism-1] ) meEEDaqActive_[ism-1]->Fill(1.0);
        
        if( meEEDaqActiveMap_ ) {

          for( int ix = 1; ix <=100; ix++ ) {
            for( int iy = 1; iy <= 100; iy++ ) {
              int ic = Numbers::icEE( ism, ix, iy );
              int jx = (iside==1) ? 100 + ix : ix;
              int jy = iy;
              if( ic > -1 ) meEEDaqActiveMap_->setBinContent( jx, jy, 1.0 );
            }
          }

        }
        
        if( meEEDaqFraction_ ) meEEDaqFraction_->Fill( EEFedCount/18. );
        
      }

    }
    
  } else {
    
    LogWarning("EEDaqInfoTask") << "Cannot find any RunInfoRcd" << endl;
    
  }

}

void EEDaqInfoTask::endLuminosityBlock(const edm::LuminosityBlock&  lumiBlock, const  edm::EventSetup& iSetup) {

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

void EEDaqInfoTask::analyze(const Event& e, const EventSetup& c){ 

}
