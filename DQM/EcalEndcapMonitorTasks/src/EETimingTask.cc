/*
 * \file EETimingTask.cc
 *
 * $Date: 2009/12/03 14:33:40 $
 * $Revision: 1.46 $
 * \author G. Della Ricca
 *
*/

#include <iostream>
#include <fstream>
#include <vector>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/MonitorElement.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include <DQM/EcalCommon/interface/Numbers.h>

#include <DQM/EcalEndcapMonitorTasks/interface/EETimingTask.h>

using namespace cms;
using namespace edm;
using namespace std;

EETimingTask::EETimingTask(const ParameterSet& ps){

  init_ = false;

  dqmStore_ = Service<DQMStore>().operator->();

  prefixME_ = ps.getUntrackedParameter<string>("prefixME", "");

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  mergeRuns_ = ps.getUntrackedParameter<bool>("mergeRuns", false);

  EcalRawDataCollection_ = ps.getParameter<edm::InputTag>("EcalRawDataCollection");
  EcalUncalibratedRecHitCollection_ = ps.getParameter<edm::InputTag>("EcalUncalibratedRecHitCollection");

  for (int i = 0; i < 18; i++) {
    meTime_[i] = 0;
    meTimeMap_[i] = 0;
    meTimeAmpli_[i] = 0;
  }

  for (int i = 0; i < 2; i++) {
    meTimeSummary1D_[i] = 0;
    meTimeSummaryMap_[i] = 0;
    meTimeSummaryMapProjR_[i] = 0;
    meTimeSummaryMapProjPhi_[i] = 0;
  }

  meTimeDelta_ = 0;

}

EETimingTask::~EETimingTask(){

}

void EETimingTask::beginJob(void){

  ievt_ = 0;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EETimingTask");
    dqmStore_->rmdir(prefixME_ + "/EETimingTask");
  }

}

void EETimingTask::beginRun(const Run& r, const EventSetup& c) {

  Numbers::initGeometry(c, false);

  if ( ! mergeRuns_ ) this->reset();

}

void EETimingTask::endRun(const Run& r, const EventSetup& c) {

}

void EETimingTask::reset(void) {

  for (int i = 0; i < 18; i++) {
    if ( meTime_[i] ) meTime_[i]->Reset();
    if ( meTimeMap_[i] ) meTimeMap_[i]->Reset();
    if ( meTimeAmpli_[i] ) meTimeAmpli_[i]->Reset();
  }

  for (int i = 0; i < 2; i++) {
    if ( meTimeSummary1D_[i] ) meTimeSummary1D_[i]->Reset();
    if ( meTimeSummaryMap_[i] ) meTimeSummaryMap_[i]->Reset();
    if ( meTimeSummaryMapProjR_[i] )  meTimeSummaryMapProjR_[i]->Reset();
    if ( meTimeSummaryMapProjPhi_[i] )  meTimeSummaryMapProjPhi_[i]->Reset();    
  }

  if ( meTimeDelta_ ) meTimeDelta_->Reset();

}

void EETimingTask::setup(void){

  init_ = true;

  char histo[200];

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EETimingTask");

    for (int i = 0; i < 18; i++) {
      sprintf(histo, "EETMT timing 1D %s", Numbers::sEE(i+1).c_str());
      meTime_[i] = dqmStore_->book1D(histo, histo, 50, 0., 10.);
      meTime_[i]->setAxisTitle("jitter (clocks)", 1);
      dqmStore_->tag(meTime_[i], i+1);

      sprintf(histo, "EETMT timing %s", Numbers::sEE(i+1).c_str());
      meTimeMap_[i] = dqmStore_->bookProfile2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 4096, 0., 4096., "s");
      meTimeMap_[i]->setAxisTitle("jx", 1);
      meTimeMap_[i]->setAxisTitle("jy", 2);
      meTimeMap_[i]->setAxisTitle("jitter (clocks)", 3);
      dqmStore_->tag(meTimeMap_[i], i+1);

      sprintf(histo, "EETMT timing vs amplitude %s", Numbers::sEE(i+1).c_str());
      meTimeAmpli_[i] = dqmStore_->book2D(histo, histo, 200, 0., 200., 100, 0., 10.);
      meTimeAmpli_[i]->setAxisTitle("amplitude", 1);
      meTimeAmpli_[i]->setAxisTitle("jitter (clocks)", 2);
      dqmStore_->tag(meTimeAmpli_[i], i+1);
    }

    sprintf(histo, "EETMT timing 1D summary EE -");
    meTimeSummary1D_[0] = dqmStore_->book1D(histo, histo, 50, 0., 10.);
    meTimeSummary1D_[0]->setAxisTitle("jitter (clocks)", 1);

    sprintf(histo, "EETMT timing 1D summary EE +");
    meTimeSummary1D_[1] = dqmStore_->book1D(histo, histo, 50, 0., 10.);
    meTimeSummary1D_[1]->setAxisTitle("jitter (clocks)", 1);

    sprintf(histo, "EETMT timing map EE -");
    meTimeSummaryMap_[0] = dqmStore_->bookProfile2D(histo, histo, 20, 0., 100., 20, 0., 100., 50, 0., 10., "s");
    meTimeSummaryMap_[0]->setAxisTitle("jx", 1);
    meTimeSummaryMap_[0]->setAxisTitle("jy", 2);
    meTimeSummaryMap_[0]->setAxisTitle("jitter (clocks)", 3);

    sprintf(histo, "EETMT timing map EE +");
    meTimeSummaryMap_[1] = dqmStore_->bookProfile2D(histo, histo, 20, 0., 100., 20, 0., 100., 50, 0., 10., "s");
    meTimeSummaryMap_[1]->setAxisTitle("jx", 1);
    meTimeSummaryMap_[1]->setAxisTitle("jy", 2);
    meTimeSummaryMap_[1]->setAxisTitle("jitter (clocks)", 3);
    
    sprintf(histo, "EETMT timing projection R EE -");
    meTimeSummaryMapProjR_[0] = dqmStore_->bookProfile(histo, histo, 20, 0., 100., 50, 0., 10., "s");
    meTimeSummaryMapProjR_[0]->setAxisTitle("R", 1);
    meTimeSummaryMapProjR_[0]->setAxisTitle("jitter (clocks)", 2);

    sprintf(histo, "EETMT timing projection R EE +");
    meTimeSummaryMapProjR_[1] = dqmStore_->bookProfile(histo, histo, 20, 0., 100., 50, 0., 10., "s");
    meTimeSummaryMapProjR_[1]->setAxisTitle("R", 1);
    meTimeSummaryMapProjR_[1]->setAxisTitle("jitter (clocks)", 2);

    sprintf(histo, "EETMT timing projection phi EE -");
    meTimeSummaryMapProjPhi_[0] = dqmStore_->bookProfile(histo, histo, 50, -M_PI, M_PI, 50, 0., 10., "s");
    meTimeSummaryMapProjPhi_[0]->setAxisTitle("phi", 1);
    meTimeSummaryMapProjPhi_[0]->setAxisTitle("jitter (clocks)", 2);

    sprintf(histo, "EETMT timing projection phi EE +");
    meTimeSummaryMapProjPhi_[1] = dqmStore_->bookProfile(histo, histo, 50, -M_PI, M_PI, 50, 0., 10., "s");
    meTimeSummaryMapProjPhi_[1]->setAxisTitle("phi", 1);
    meTimeSummaryMapProjPhi_[1]->setAxisTitle("jitter (clocks)", 2);

    sprintf(histo, "EETMT timing EE+ - EE-");
    meTimeDelta_ = dqmStore_->book1D(histo, histo, 100, -10., 10.);
    meTimeDelta_->setAxisTitle("jitter (clocks)", 1);

  }

}

void EETimingTask::cleanup(void){

  if ( ! init_ ) return;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EETimingTask");

    for ( int i = 0; i < 18; i++ ) {
      if ( meTime_[i] ) dqmStore_->removeElement( meTime_[i]->getName() );
      meTime_[i] = 0;

      if ( meTimeMap_[i] ) dqmStore_->removeElement( meTimeMap_[i]->getName() );
      meTimeMap_[i] = 0;

      if ( meTimeAmpli_[i] ) dqmStore_->removeElement( meTimeAmpli_[i]->getName() );
      meTimeAmpli_[i] = 0;
    }

    for (int i = 0; i < 2; i++) {
      if ( meTimeSummary1D_[i] ) dqmStore_->removeElement( meTimeSummary1D_[i]->getName() );
      meTimeSummary1D_[i] = 0;

      if ( meTimeSummaryMap_[i] ) dqmStore_->removeElement( meTimeSummaryMap_[i]->getName() );
      meTimeSummaryMap_[i] = 0;
      
      if ( meTimeSummaryMapProjR_[i] ) dqmStore_->removeElement( meTimeSummaryMapProjR_[i]->getName() );
      meTimeSummaryMapProjR_[i] = 0;

      if ( meTimeSummaryMapProjPhi_[i] ) dqmStore_->removeElement( meTimeSummaryMapProjPhi_[i]->getName() );
      meTimeSummaryMapProjPhi_[i] = 0;
    }

    if ( meTimeDelta_ ) dqmStore_->removeElement( meTimeDelta_->getName() );
    meTimeDelta_ = 0;

  }

  init_ = false;

}

void EETimingTask::endJob(void){

  LogInfo("EETimingTask") << "analyzed " << ievt_ << " events";

  if ( enableCleanup_ ) this->cleanup();

}

void EETimingTask::analyze(const Event& e, const EventSetup& c){

  bool isData = true;
  bool enable = false;
  int runType[18];
  for (int i=0; i<18; i++) runType[i] = -1;

  Handle<EcalRawDataCollection> dcchs;

  if ( e.getByLabel(EcalRawDataCollection_, dcchs) ) {

    for ( EcalRawDataCollection::const_iterator dcchItr = dcchs->begin(); dcchItr != dcchs->end(); ++dcchItr ) {

      if ( Numbers::subDet( *dcchItr ) != EcalEndcap ) continue;

      int ism = Numbers::iSM( *dcchItr, EcalEndcap );

      runType[ism-1] = dcchItr->getRunType();

      if ( dcchItr->getRunType() == EcalDCCHeaderBlock::COSMIC ||
           dcchItr->getRunType() == EcalDCCHeaderBlock::MTCC ||
           dcchItr->getRunType() == EcalDCCHeaderBlock::COSMICS_GLOBAL ||
           dcchItr->getRunType() == EcalDCCHeaderBlock::PHYSICS_GLOBAL ||
           dcchItr->getRunType() == EcalDCCHeaderBlock::COSMICS_LOCAL ||
           dcchItr->getRunType() == EcalDCCHeaderBlock::PHYSICS_LOCAL ) enable = true;

    }

  } else {

    isData = false; enable = true;
    LogWarning("EETimingTask") << EcalRawDataCollection_ << " not available";

  }

  if ( ! enable ) return;

  if ( ! init_ ) this->setup();

  ievt_++;

  float ievtTimes[2];
  int nGoodRh[2];
  for ( int i=0; i<2; i++ ) {
    ievtTimes[i] = 0;
    nGoodRh[i] = 0;
  }

  Handle<EcalUncalibratedRecHitCollection> hits;

  if ( e.getByLabel(EcalUncalibratedRecHitCollection_, hits) ) {

    int neh = hits->size();
    LogDebug("EETimingTask") << "event " << ievt_ << " hits collection size " << neh;

    for ( EcalUncalibratedRecHitCollection::const_iterator hitItr = hits->begin(); hitItr != hits->end(); ++hitItr ) {

      EEDetId id = hitItr->id();

      int ix = id.ix();
      int iy = id.iy();
      int iz = ( id.positiveZ() ) ? 1 : 0;
      
      int ism = Numbers::iSM( id );

      if ( ism >= 1 && ism <= 9 ) ix = 101 - ix;

      float xix = ix - 0.5;
      float xiy = iy - 0.5;

      if ( isData ) {

        if ( ! ( runType[ism-1] == EcalDCCHeaderBlock::COSMIC ||
                 runType[ism-1] == EcalDCCHeaderBlock::MTCC ||
                 runType[ism-1] == EcalDCCHeaderBlock::COSMICS_GLOBAL ||
                 runType[ism-1] == EcalDCCHeaderBlock::PHYSICS_GLOBAL ||
                 runType[ism-1] == EcalDCCHeaderBlock::COSMICS_LOCAL ||
                 runType[ism-1] == EcalDCCHeaderBlock::PHYSICS_LOCAL ) ) continue;

      }

      LogDebug("EETimingTask") << " det id = " << id;
      LogDebug("EETimingTask") << " sm, ix, iy " << ism << " " << ix << " " << iy;

      MonitorElement* meTime = 0;
      MonitorElement* meTimeMap = 0;
      MonitorElement* meTimeAmpli = 0;

      meTime = meTime_[ism-1];
      meTimeMap = meTimeMap_[ism-1];
      meTimeAmpli = meTimeAmpli_[ism-1];

      float xval = hitItr->amplitude();
      if ( xval <= 0. ) xval = 0.0;
      float yval = hitItr->jitter() + 5.0;
      if ( yval <= 0. ) yval = 0.0;
      float zval = hitItr->pedestal();
      if ( zval <= 0. ) zval = 0.0;

      LogDebug("EETimingTask") << " hit amplitude " << xval;
      LogDebug("EETimingTask") << " hit jitter " << yval;
      LogDebug("EETimingTask") << " hit pedestal " << zval;

      if ( meTimeAmpli ) meTimeAmpli->Fill(xval, yval);

      if ( xval > 8. && hitItr->recoFlag() == EcalUncalibratedRecHit::kGood ) {
        if ( meTimeMap ) meTimeMap->Fill(xix, xiy, yval);

        // exclude the noisiest region around the hole from 1D
        if ( (ix <= 35 || ix >= 65) && (iy <= 35 || iy >= 65) ) {
          if ( meTime ) meTime->Fill(yval);
          if ( meTimeSummary1D_[iz] ) meTimeSummary1D_[iz]->Fill(yval);
          ievtTimes[iz] += yval;
          nGoodRh[iz]++;
        }
      }

      if ( meTimeSummaryMap_[iz] ) meTimeSummaryMap_[iz]->Fill(xix, xiy, yval);
      if ( meTimeSummaryMapProjR_[iz] ) meTimeSummaryMapProjR_[iz]->Fill(sqrt(xix*xix+xiy*xiy), yval);
      if ( meTimeSummaryMapProjPhi_[iz] ) meTimeSummaryMapProjPhi_[iz]->Fill(atan2(xiy-50.,xix-50.), yval);
    }

    float mean[2];
    for ( int i=0; i<2; i++ ) {
      if ( nGoodRh[i] > 0 ) mean[i] = ievtTimes[i] / nGoodRh[i];
    }

    if ( meTimeDelta_ && nGoodRh[0] > 0 && nGoodRh[1] > 0 ) meTimeDelta_->Fill( mean[1] - mean[0] );

  } else {

    LogWarning("EETimingTask") << EcalUncalibratedRecHitCollection_ << " not available";

  }

}

