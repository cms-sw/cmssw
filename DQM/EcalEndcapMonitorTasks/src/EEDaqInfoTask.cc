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
    meEEDaqActiveMap_ = dqmStore_->book2D(histo,histo, 40, 0., 40., 20, 0., 20.);
    for ( int jxdcc = 0; jxdcc < 20; jxdcc++ ) {
      for ( int jydcc = 0; jydcc < 20; jydcc++ ) {
        for ( int iside = 0; iside < 2; iside++ ) {
          meEEDaqActiveMap_->setBinContent( 20*iside+jxdcc+1, jydcc+1, 0.0 );
        }
      }
    }
    meEEDaqActiveMap_->setAxisTitle("jx", 1);
    meEEDaqActiveMap_->setAxisTitle("jy", 2);

    dqmStore_->setCurrentFolder(prefixME_ + "/EventInfo/DAQSummaryContents");

    for (int i = 0; i < 18; i++) {
      sprintf(histo, "EcalBarrel_%s", Numbers::sEE(i+1).c_str());
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

  edm::eventsetup::EventSetupRecordKey recordKey(edm::eventsetup::EventSetupRecordKey::TypeTag::findType("RunInfoRcd"));

  if( iSetup.find( recordKey ) ) {

    edm::ESHandle<RunInfo> sumFED;
    iSetup.get<RunInfoRcd>().get(sumFED);    
   
    std::vector<int> FedsInIds= sumFED->m_fed_in;   

    float EEFedCount = 0.;

    // find the coordinates within EE geometry
    bool withinGeometry[100][100];
    for( int jx = 0; jx <100; jx++ ) {
      for( int jy = 0; jy < 100; jy++ ) {
        withinGeometry[jx][jy] = false;
      }
    }

    for( int jx = 0; jx <100; jx++ ) {
      for( int jy = 0; jy < 100; jy++ ) {
        for(int ism = 1; ism<=18; ism++ ) {
          int ic = Numbers::icEE( ism, jx, jy );
          if( ic > -1 ) withinGeometry[jx][jy] = true;
        }
      }
    }

    // make the crystals outside EE geometry out-of-scale
    for( int jx = 0; jx <100; jx++ ) { 
      for( int jy = 0; jy < 100; jy++ ) {
        if( !withinGeometry[jx][jy] ) {
          for( int iside = 0; iside<2; iside++ ) {
            int matrix5x5x = 20*iside + jx/5 + 1;
            int matrix5x5y = jy/5 + 1;
            meEEDaqActiveMap_->setBinContent( matrix5x5x, matrix5x5y, -1.0 );
          }
        }
      }
    }

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

          for( int jx = 0; jx <100; jx++ ) {
            for( int jy = 0; jy < 100; jy++ ) {
              int ic = Numbers::icEE( ism, jx, jy );
              if( ic > -1 ) {

                int matrix5x5x = 20*iside + jx/5 + 1;
                int matrix5x5y = jy/5 + 1;
                
                meEEDaqActiveMap_->setBinContent( matrix5x5x, matrix5x5y, 1.0 );

              }
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

    dqmStore_->setCurrentFolder(prefixME_ + "/EventInfo/DAQSummaryContents");

    for (int i = 0; i < 18; i++) {
      if ( meEEDaqActive_[i] ) dqmStore_->removeElement( meEEDaqActive_[i]->getName() );
    }

  }

}

void EEDaqInfoTask::analyze(const Event& e, const EventSetup& c){ 

}
