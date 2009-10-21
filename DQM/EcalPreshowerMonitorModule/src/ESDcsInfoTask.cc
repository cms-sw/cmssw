#include <iostream>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/EcalDetId/interface/ESDetId.h"

#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"
#include "CondFormats/RunInfo/interface/RunSummary.h"
#include "CondFormats/RunInfo/interface/RunInfo.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DQM/EcalPreshowerMonitorModule/interface/ESDcsInfoTask.h"

using namespace cms;
using namespace edm;
using namespace std;

ESDcsInfoTask::ESDcsInfoTask(const ParameterSet& ps) {

  dqmStore_ = Service<DQMStore>().operator->();

  prefixME_ = ps.getUntrackedParameter<string>("prefixME", "");

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  mergeRuns_ = ps.getUntrackedParameter<bool>("mergeRuns", false);

  meESDcsFraction_ = 0;
  meESDcsActiveMap_ = 0;

}

ESDcsInfoTask::~ESDcsInfoTask() {

}

void ESDcsInfoTask::beginJob(const EventSetup& c){

  char histo[200];
  
  if ( dqmStore_ ) {

    dqmStore_->setCurrentFolder(prefixME_ + "/EventInfo");
    
    sprintf(histo, "DCSSummary");
    meESDcsFraction_ = dqmStore_->bookFloat(histo);
    meESDcsFraction_->Fill(0.0);

    sprintf(histo, "DCSSummaryMap");
    meESDcsActiveMap_ = dqmStore_->book2D(histo,histo, 40, 0., 40., 40, 0., 40.);
    meESDcsActiveMap_->setAxisTitle("X", 1);
    meESDcsActiveMap_->setAxisTitle("Y", 2);

  }

}

void ESDcsInfoTask::endJob(void) {

  if ( enableCleanup_ ) this->cleanup();

}

void ESDcsInfoTask::beginLuminosityBlock(const edm::LuminosityBlock& lumiBlock, const  edm::EventSetup& iSetup){

  this->reset();

}

void ESDcsInfoTask::endLuminosityBlock(const edm::LuminosityBlock&  lumiBlock, const  edm::EventSetup& iSetup) {

}

void ESDcsInfoTask::reset(void) {

  if ( meESDcsFraction_ ) meESDcsFraction_->Reset();

  if ( meESDcsActiveMap_ ) meESDcsActiveMap_->Reset();
  
}

void ESDcsInfoTask::cleanup(void){
  
  if ( dqmStore_ ) {

    dqmStore_->setCurrentFolder(prefixME_ + "/EventInfo");
    
    if ( meESDcsFraction_ ) dqmStore_->removeElement( meESDcsFraction_->getName() );

    if ( meESDcsActiveMap_ ) dqmStore_->removeElement( meESDcsActiveMap_->getName() );

  }

}

void ESDcsInfoTask::analyze(const Event& e, const EventSetup& c){ 

}

DEFINE_FWK_MODULE(ESDcsInfoTask);
