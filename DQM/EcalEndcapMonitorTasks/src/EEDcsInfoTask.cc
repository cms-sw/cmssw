#include <iostream>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include <DataFormats/EcalDetId/interface/EEDetId.h>

#include "DataFormats/Scalers/interface/DcsStatus.h"
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

  dcsStatusCollection_ = ps.getParameter<edm::InputTag>("DcsStatusCollection");

  meEEDcsFraction_ = 0;
  meEEDcsActiveMap_ = 0;
  for (int i = 0; i < 18; i++) {
    meEEDcsActive_[i] = 0;
  }
  meDcsErrorsByLumi_ = 0;

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

    // checking the number of DCS errors in each DCC for each lumi
    // tower error is weighted by 1/34
    // bin 0 contains the number of processed events in the lumi (for normalization)
    sprintf(histo, "EE weighted DCS errors by lumi");
    meDcsErrorsByLumi_ = dqmStore_->book1D(histo, histo, 18, 1., 19.);
    meDcsErrorsByLumi_->setLumiFlag();
    for (int i = 0; i < 18; i++) {
      meDcsErrorsByLumi_->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
    }
    
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

  for ( int itx = 0; itx < 40; itx++ ) {
    for ( int ity = 0; ity < 20; ity++ ) {
      readyLumi[itx][ity] = 1;
    }
  }

  if ( meDcsErrorsByLumi_ ) meDcsErrorsByLumi_->Reset();
  eventsInLumi_ =0;

}

void EEDcsInfoTask::endLuminosityBlock(const edm::LuminosityBlock&  lumiBlock, const  edm::EventSetup& iSetup) {

  this->fillMonitorElements(readyLumi);

}

void EEDcsInfoTask::beginRun(const Run& r, const EventSetup& c) {

  if ( ! mergeRuns_ ) this->reset();

  for ( int itx = 0; itx < 40; itx++ ) {
    for ( int ity = 0; ity < 20; ity++ ) {
      readyRun[itx][ity] = 1;
    }
  }

}

void EEDcsInfoTask::endRun(const Run& r, const EventSetup& c) {

  this->fillMonitorElements(readyRun);

}

void EEDcsInfoTask::reset(void) {

  if ( meEEDcsFraction_ ) meEEDcsFraction_->Reset();

  for (int i = 0; i < 18; i++) {
    if ( meEEDcsActive_[i] ) meEEDcsActive_[i]->Reset();
  }

  if ( meEEDcsActiveMap_ ) meEEDcsActiveMap_->Reset();
  if ( meDcsErrorsByLumi_ ) meDcsErrorsByLumi_->Reset();
  eventsInLumi_ = 0;

}


void EEDcsInfoTask::cleanup(void){
  
  if ( dqmStore_ ) {

    dqmStore_->setCurrentFolder(prefixME_ + "/EventInfo");
    
    if ( meEEDcsFraction_ ) dqmStore_->removeElement( meEEDcsFraction_->getName() );

    if ( meEEDcsActiveMap_ ) dqmStore_->removeElement( meEEDcsActiveMap_->getName() );

    if ( meDcsErrorsByLumi_ ) dqmStore_->removeElement( meDcsErrorsByLumi_->getName() );

    dqmStore_->setCurrentFolder(prefixME_ + "/EventInfo/DCSContents");

    for (int i = 0; i < 18; i++) {
      if ( meEEDcsActive_[i] ) dqmStore_->removeElement( meEEDcsActive_[i]->getName() );
    }

  }

}

void EEDcsInfoTask::analyze(const Event& e, const EventSetup& c){ 

  Handle<DcsStatusCollection> dcsh;

  if ( e.getByLabel(dcsStatusCollection_, dcsh) ) {

    for ( int iz = -1; iz < 2; iz+=2 ) {
      for ( int itx = 0; itx < 20; itx++ ) {
        for ( int ity = 0; ity < 20; ity++ ) {
        
          int offsetSC = (iz > 0) ? 0 : 20;
          bool ready = false;
        
          if ( dcsh->size() > 0 ) ready = (iz < 0) ? (*dcsh)[0].ready(DcsStatus::EEm) : (*dcsh)[0].ready(DcsStatus::EEp);
        
          if ( !ready ) {
            readyRun[offsetSC+itx][ity] = 0;
            readyLumi[offsetSC+itx][ity] = 0;
          }
        
        }
      }
    }
    
  } else {
    LogWarning("EEDcsInfoTask") << dcsStatusCollection_ << " not available";
  }

}

void EEDcsInfoTask::fillMonitorElements(int ready[40][20]) {

  // fill bin 0 with number of processed events in the lumi section
  if ( meDcsErrorsByLumi_ ) meDcsErrorsByLumi_->Fill(eventsInLumi_);

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
              float xism = ism + 0.5;
              if(ready[offsetSC+itx][ity]) {
                readySum[ism-1]++;
                readySumTot++;
              } else {
                if ( meDcsErrorsByLumi_ ) meDcsErrorsByLumi_->Fill(xism, 1./34.);
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
