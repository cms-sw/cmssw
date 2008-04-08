/*
 * \file EETimingTask.cc
 *
 * $Date: 2008/04/08 15:32:10 $
 * $Revision: 1.32 $
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

  EcalRawDataCollection_ = ps.getParameter<edm::InputTag>("EcalRawDataCollection");
  EcalUncalibratedRecHitCollection_ = ps.getParameter<edm::InputTag>("EcalUncalibratedRecHitCollection");

  for (int i = 0; i < 18; i++) {
    meTimeMap_[i] = 0;
    meTimeAmpli_[i] = 0;
  }

}

EETimingTask::~EETimingTask(){

}

void EETimingTask::beginJob(const EventSetup& c){

  ievt_ = 0;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EETimingTask");
    dqmStore_->rmdir(prefixME_ + "/EETimingTask");
  }

  Numbers::initGeometry(c, false);

}

void EETimingTask::setup(void){

  init_ = true;

  char histo[200];

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EETimingTask");

    for (int i = 0; i < 18; i++) {
      sprintf(histo, "EETMT timing %s", Numbers::sEE(i+1).c_str());
      meTimeMap_[i] = dqmStore_->bookProfile2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 4096, 0., 4096., "s");
      meTimeMap_[i]->setAxisTitle("jx", 1);
      meTimeMap_[i]->setAxisTitle("jy", 2);
      dqmStore_->tag(meTimeMap_[i], i+1);

      sprintf(histo, "EETMT timing vs amplitude %s", Numbers::sEE(i+1).c_str());
      meTimeAmpli_[i] = dqmStore_->book2D(histo, histo, 200, 0., 200., 100, 0., 10.);
      meTimeAmpli_[i]->setAxisTitle("amplitude", 1);
      meTimeAmpli_[i]->setAxisTitle("jitter", 2);
      dqmStore_->tag(meTimeAmpli_[i], i+1);
    }

  }

}

void EETimingTask::cleanup(void){

  if ( ! enableCleanup_ ) return;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EETimingTask");

    for ( int i = 0; i < 18; i++ ) {
      if ( meTimeMap_[i] ) dqmStore_->removeElement( meTimeMap_[i]->getName() );
      meTimeMap_[i] = 0;
    }

  }

  init_ = false;

}

void EETimingTask::endJob(void){

  LogInfo("EETimingTask") << "analyzed " << ievt_ << " events";

  if ( init_ ) this->cleanup();

}

void EETimingTask::analyze(const Event& e, const EventSetup& c){

  bool isData = true;
  bool enable = false;
  map<int, EcalDCCHeaderBlock> dccMap;

  Handle<EcalRawDataCollection> dcchs;

  if ( e.getByLabel(EcalRawDataCollection_, dcchs) ) {

    for ( EcalRawDataCollection::const_iterator dcchItr = dcchs->begin(); dcchItr != dcchs->end(); ++dcchItr ) {

      EcalDCCHeaderBlock dcch = (*dcchItr);

      if ( Numbers::subDet( dcch ) != EcalBarrel ) continue;

      int ism = Numbers::iSM( dcch, EcalBarrel );

      map<int, EcalDCCHeaderBlock>::iterator i = dccMap.find( ism );
      if ( i != dccMap.end() ) continue;

      dccMap[ ism ] = dcch;

      if ( dcch.getRunType() == EcalDCCHeaderBlock::COSMIC ||
           dcch.getRunType() == EcalDCCHeaderBlock::MTCC ||
           dcch.getRunType() == EcalDCCHeaderBlock::COSMICS_GLOBAL ||
           dcch.getRunType() == EcalDCCHeaderBlock::PHYSICS_GLOBAL ||
           dcch.getRunType() == EcalDCCHeaderBlock::COSMICS_LOCAL ||
           dcch.getRunType() == EcalDCCHeaderBlock::PHYSICS_LOCAL ) enable = true;

    }

  } else {

    isData = false; enable = true;
    LogWarning("EETimingTask") << EcalRawDataCollection_ << " not available";

  }

  if ( ! enable ) return;

  if ( ! init_ ) this->setup();

  ievt_++;

  Handle<EcalUncalibratedRecHitCollection> hits;

  if ( e.getByLabel(EcalUncalibratedRecHitCollection_, hits) ) {

    int neh = hits->size();
    LogDebug("EETimingTask") << "event " << ievt_ << " hits collection size " << neh;

    for ( EcalUncalibratedRecHitCollection::const_iterator hitItr = hits->begin(); hitItr != hits->end(); ++hitItr ) {

      EcalUncalibratedRecHit hit = (*hitItr);
      EEDetId id = hit.id();

      int ix = id.ix();
      int iy = id.iy();

      int ism = Numbers::iSM( id );

      if ( ism >= 1 && ism <= 9 ) ix = 101 - ix;

      float xix = ix - 0.5;
      float xiy = iy - 0.5;

      if ( isData ) {
      map<int, EcalDCCHeaderBlock>::iterator i = dccMap.find(ism);
      if ( i == dccMap.end() ) continue;

      if ( ! ( dccMap[ism].getRunType() == EcalDCCHeaderBlock::COSMIC ||
               dccMap[ism].getRunType() == EcalDCCHeaderBlock::MTCC ||
               dccMap[ism].getRunType() == EcalDCCHeaderBlock::COSMICS_GLOBAL ||
               dccMap[ism].getRunType() == EcalDCCHeaderBlock::PHYSICS_GLOBAL ||
               dccMap[ism].getRunType() == EcalDCCHeaderBlock::COSMICS_LOCAL ||
               dccMap[ism].getRunType() == EcalDCCHeaderBlock::PHYSICS_LOCAL ) ) continue;
      }

      LogDebug("EETimingTask") << " det id = " << id;
      LogDebug("EETimingTask") << " sm, ix, iy " << ism << " " << ix << " " << iy;

      MonitorElement* meTimeMap = 0;
      MonitorElement* meTimeAmpli = 0;

      meTimeMap = meTimeMap_[ism-1];
      meTimeAmpli = meTimeAmpli_[ism-1];

      float xval = hit.amplitude();
      if ( xval <= 0. ) xval = 0.0;
      float yval = hit.jitter() + 5.0;
      if ( yval <= 0. ) yval = 0.0;
      float zval = hit.pedestal();
      if ( zval <= 0. ) zval = 0.0;

      LogDebug("EETimingTask") << " hit amplitude " << xval;
      LogDebug("EETimingTask") << " hit jitter " << yval;
      LogDebug("EETimingTask") << " hit pedestal " << zval;

      if ( meTimeAmpli ) meTimeAmpli->Fill(xval, yval);

      if ( xval <= 16. ) continue;

      if ( meTimeMap ) meTimeMap->Fill(xix, xiy, yval);

    }

  } else {

    LogWarning("EETimingTask") << EcalUncalibratedRecHitCollection_ << " not available";

  }

}

