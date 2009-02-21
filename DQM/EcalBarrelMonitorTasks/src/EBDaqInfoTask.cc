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

#include "DQM/EcalBarrelMonitorTasks/interface/EBDaqInfoTask.h"

using namespace cms;
using namespace edm;
using namespace std;

EBDaqInfoTask::EBDaqInfoTask(const ParameterSet& ps) {

  dqmStore_ = Service<DQMStore>().operator->();

  prefixME_ = ps.getUntrackedParameter<string>("prefixME", "");

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  mergeRuns_ = ps.getUntrackedParameter<bool>("mergeRuns", false);

  EBFedRangeMin_ = ps.getUntrackedParameter<int>("EBFedRangeMin");
  EBFedRangeMax_ = ps.getUntrackedParameter<int>("EBFedRangeMax");

  meEBDaqFraction_ = 0;
  meEBDaqActiveMap_ = 0;
  for (int i = 0; i < 36; i++) {
    meEBDaqActive_[i] = 0;
  }

}

EBDaqInfoTask::~EBDaqInfoTask() {

}

void EBDaqInfoTask::beginJob(const EventSetup& c){

  char histo[200];
  
  if ( dqmStore_ ) {

    dqmStore_->setCurrentFolder(prefixME_ + "/EventInfo");
    
    sprintf(histo, "DAQSummary");
    meEBDaqFraction_ = dqmStore_->bookFloat(histo);
    meEBDaqFraction_->Fill(0.0);

    sprintf(histo, "DAQSummaryMap");
    meEBDaqActiveMap_ = dqmStore_->book2D(histo,histo, 72, 0., 72., 34, 0., 34.);
    meEBDaqActiveMap_->setAxisTitle("jphi", 1);
    meEBDaqActiveMap_->setAxisTitle("jeta", 2);

    dqmStore_->setCurrentFolder(prefixME_ + "/EventInfo/DAQSummaryContents");

    for (int i = 0; i < 36; i++) {
      sprintf(histo, "EcalBarrel_%s", Numbers::sEB(i+1).c_str());
      meEBDaqActive_[i] = dqmStore_->bookFloat(histo);
      meEBDaqActive_[i]->Fill(0.0);
    }

  }

}

void EBDaqInfoTask::endJob(void) {

  if ( enableCleanup_ ) this->cleanup();

}

void EBDaqInfoTask::beginLuminosityBlock(const edm::LuminosityBlock& lumiBlock, const  edm::EventSetup& iSetup){

  this->reset();

  for ( int iettx = 0; iettx < 34; iettx++ ) {
    for ( int ipttx = 0; ipttx < 72; ipttx++ ) {
      meEBDaqActiveMap_->setBinContent( ipttx+1, iettx+1, 0.0 );
    }
  }

  edm::eventsetup::EventSetupRecordKey recordKey(edm::eventsetup::EventSetupRecordKey::TypeTag::findType("RunInfoRcd"));

  if( iSetup.find( recordKey ) ) {

    edm::ESHandle<RunInfo> sumFED;
    iSetup.get<RunInfoRcd>().get(sumFED);    
   
    std::vector<int> FedsInIds= sumFED->m_fed_in;   

    float EBFedCount = 0.;

    for( unsigned int fedItr=0; fedItr<FedsInIds.size(); ++fedItr ) {

      int fedID=FedsInIds[fedItr];

      if( fedID >= EBFedRangeMin_ && fedID <= EBFedRangeMax_ ) {

        EBFedCount++;
        
        int ism = fedID - EBFedRangeMin_ + 1;
        int iesm = (ism-1) / 18 + 1;
        int ipsm = (ism-1) % 18 + 1;

        if (  meEBDaqActive_[ism-1] ) meEBDaqActive_[ism-1]->Fill(1.0);
        
        if( meEBDaqActiveMap_ ) {

          for( int iett=0; iett<17; iett++ ) {
            for( int iptt=0; iptt<4; iptt++ ) {
              int iettx = (iesm-1)*17 + iett + 1;
              int ipttx = (ipsm-1)*4 + iptt + 1;
              meEBDaqActiveMap_->setBinContent( ipttx, iettx, 1.0);
            }
          }

        }      
    
      }
        
    }

    if( meEBDaqFraction_ ) meEBDaqFraction_->Fill( EBFedCount/36 );

  } else {

    LogWarning("EBDaqInfoTask") << "Cannot find any RunInfoRcd" << endl;

  }

}

void EBDaqInfoTask::endLuminosityBlock(const edm::LuminosityBlock&  lumiBlock, const  edm::EventSetup& iSetup) {

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

    dqmStore_->setCurrentFolder(prefixME_ + "/EventInfo/DAQSummaryContents");

    for (int i = 0; i < 36; i++) {
      if ( meEBDaqActive_[i] ) dqmStore_->removeElement( meEBDaqActive_[i]->getName() );
    }

  }

}

void EBDaqInfoTask::analyze(const Event& e, const EventSetup& c){ 

}
