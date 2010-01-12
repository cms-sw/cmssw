#include <iostream>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Scalers/interface/DcsStatus.h"
#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"
#include "CondFormats/RunInfo/interface/RunSummary.h"
#include "CondFormats/RunInfo/interface/RunInfo.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include <DQM/EcalCommon/interface/Numbers.h>

#include "DQM/EcalBarrelMonitorTasks/interface/EBDcsInfoTask.h"

using namespace cms;
using namespace edm;
using namespace std;

EBDcsInfoTask::EBDcsInfoTask(const ParameterSet& ps) {

  dqmStore_ = Service<DQMStore>().operator->();

  prefixME_ = ps.getUntrackedParameter<string>("prefixME", "");

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  mergeRuns_ = ps.getUntrackedParameter<bool>("mergeRuns", false);

  dcsStatusCollection_ = ps.getParameter<edm::InputTag>("DcsStatusCollection");

  meEBDcsFraction_ = 0;
  meEBDcsActiveMap_ = 0;
  for (int i = 0; i < 36; i++) {
    meEBDcsActive_[i] = 0;
  }

}

EBDcsInfoTask::~EBDcsInfoTask() {

}

void EBDcsInfoTask::beginJob(void){

  char histo[200];
  
  if ( dqmStore_ ) {

    dqmStore_->setCurrentFolder(prefixME_ + "/EventInfo");
    
    sprintf(histo, "DCSSummary");
    meEBDcsFraction_ = dqmStore_->bookFloat(histo);
    meEBDcsFraction_->Fill(0.0);

    sprintf(histo, "DCSSummaryMap");
    meEBDcsActiveMap_ = dqmStore_->book2D(histo,histo, 72, 0., 72., 34, 0., 34.);
    meEBDcsActiveMap_->setAxisTitle("jphi", 1);
    meEBDcsActiveMap_->setAxisTitle("jeta", 2);

    dqmStore_->setCurrentFolder(prefixME_ + "/EventInfo/DCSContents");

    for (int i = 0; i < 36; i++) {
      sprintf(histo, "EcalBarrel_%s", Numbers::sEB(i+1).c_str());
      meEBDcsActive_[i] = dqmStore_->bookFloat(histo);
      meEBDcsActive_[i]->Fill(-1.0);
    }

  }

}

void EBDcsInfoTask::endJob(void) {

  if ( enableCleanup_ ) this->cleanup();

}

void EBDcsInfoTask::beginLuminosityBlock(const edm::LuminosityBlock& lumiBlock, const  edm::EventSetup& iSetup){

  nTotLumi = 0;
  for ( int iett = 0; iett < 34; iett++ ) {
    for ( int iptt = 0; iptt < 72; iptt++ ) {
      nReadyLumi[iptt][iett] = 0;
    }
  }

}

void EBDcsInfoTask::endLuminosityBlock(const edm::LuminosityBlock&  lumiBlock, const  edm::EventSetup& iSetup) {

  this->fillMonitorElements(nReadyLumi, nTotLumi);

}

void EBDcsInfoTask::beginRun(const Run& r, const EventSetup& c) {

  if ( ! mergeRuns_ ) this->reset();

  nTotRun = 0;
  for ( int iett = 0; iett < 34; iett++ ) {
    for ( int iptt = 0; iptt < 72; iptt++ ) {
      nReadyRun[iptt][iett] = 0;
    }
  }

}

void EBDcsInfoTask::endRun(const Run& r, const EventSetup& c) {

  this->fillMonitorElements(nReadyRun, nTotRun);

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

void EBDcsInfoTask::analyze(const Event& e, const EventSetup& c){ 

  nTotRun++;
  nTotLumi++;

  Handle<DcsStatusCollection> dcsh;

  if ( e.getByLabel(dcsStatusCollection_, dcsh) ) {

    for ( int iett = 0; iett < 34; iett++ ) {
      for ( int iptt = 0; iptt < 72; iptt++ ) {

        bool ready = false;
        
        if ( dcsh->size() > 0 ) ready = (iett < 17) ? (*dcsh)[0].ready(DcsStatus::EBm) : (*dcsh)[0].ready(DcsStatus::EBp);

        if ( ready ) {
          nReadyRun[iptt][iett]++;
          nReadyLumi[iptt][iett]++;
        }

      }
    }
    
  } else {
    LogWarning("EBDcsInfoTask") << dcsStatusCollection_ << " not available";
  }

}

void EBDcsInfoTask::fillMonitorElements(int nReady[72][34], int nTot) {

  float readinessSum = 0.;
  for ( int iett = 0; iett < 34; iett++ ) {
    for ( int iptt = 0; iptt < 72; iptt++ ) {
      
      float readiness = float(nReady[iptt][iett])/float(nTot);

      if(meEBDcsActiveMap_) meEBDcsActiveMap_->setBinContent( iptt+1, iett+1, readiness );

      // take one tower of the FED as a flag
      int ism = ( iett<17 ) ? iptt/4 : 18+iptt/4; 
      if( meEBDcsActive_[ism] ) meEBDcsActive_[ism]->Fill( readiness );

      readinessSum += readiness;

    }
  }      

  if( meEBDcsFraction_ ) meEBDcsFraction_->Fill(readinessSum/34./72.);

}
