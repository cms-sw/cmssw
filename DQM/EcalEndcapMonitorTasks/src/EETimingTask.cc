/*
 * \file EETimingTask.cc
 *
 * $Date: 2008/01/22 19:14:57 $
 * $Revision: 1.23 $
 * \author G. Della Ricca
 *
*/

#include <iostream>
#include <fstream>
#include <vector>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"

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

  // get hold of back-end interface
  dbe_ = Service<DaqMonitorBEInterface>().operator->();

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  EcalUncalibratedRecHitCollection_ = ps.getParameter<edm::InputTag>("EcalUncalibratedRecHitCollection");

  for (int i = 0; i < 18; i++) {
    meTimeMap_[i] = 0;
  }

}

EETimingTask::~EETimingTask(){

}

void EETimingTask::beginJob(const EventSetup& c){

  ievt_ = 0;

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalEndcap/EETimingTask");
    dbe_->rmdir("EcalEndcap/EETimingTask");
  }

  Numbers::initGeometry(c);

}

void EETimingTask::setup(void){

  init_ = true;

  char histo[200];

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalEndcap/EETimingTask");

    for (int i = 0; i < 18; i++) {
      sprintf(histo, "EETMT timing %s", Numbers::sEE(i+1).c_str());
      meTimeMap_[i] = dbe_->bookProfile2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 4096, 0., 4096., "s");
      meTimeMap_[i]->setAxisTitle("jx", 1);
      meTimeMap_[i]->setAxisTitle("jy", 2);
      dbe_->tag(meTimeMap_[i], i+1);
    }

  }

}

void EETimingTask::cleanup(void){

  if ( ! enableCleanup_ ) return;

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalEndcap/EETimingTask");

    for ( int i = 0; i < 18; i++ ) {
      if ( meTimeMap_[i] ) dbe_->removeElement( meTimeMap_[i]->getName() );
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

      LogDebug("EETimingTask") << " det id = " << id;
      LogDebug("EETimingTask") << " sm, ix, iy " << ism << " " << ix << " " << iy;

      MonitorElement* meTimeMap = 0;

      meTimeMap = meTimeMap_[ism-1];

      float xval = hit.amplitude();
      if ( xval <= 0. ) xval = 0.0;
      float yval = hit.jitter() + 6.0;
      if ( yval <= 0. ) yval = 0.0;
      float zval = hit.pedestal();
      if ( zval <= 0. ) zval = 0.0;

      LogDebug("EETimingTask") << " hit amplitude " << xval;
      LogDebug("EETimingTask") << " hit jitter " << yval;
      LogDebug("EETimingTask") << " hit pedestal " << zval;

      if ( xval <= 2. ) continue;

      if ( meTimeMap ) meTimeMap->Fill(xix, xiy, yval);

    }

  } else {

    LogWarning("EETimingTask") << EcalUncalibratedRecHitCollection_ << " not available";

  }

}

