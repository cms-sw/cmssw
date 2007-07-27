/*
 * \file EETimingTask.cc
 *
 * $Date: 2007/05/24 13:26:12 $
 * $Revision: 1.9 $
 * \author G. Della Ricca
 *
*/

#include <iostream>
#include <fstream>
#include <vector>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"

#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include <DQM/EcalCommon/interface/Numbers.h>

#include <DQM/EcalEndcapMonitorTasks/interface/EETimingTask.h>

using namespace cms;
using namespace edm;
using namespace std;

EETimingTask::EETimingTask(const ParameterSet& ps){

  Numbers::maxSM = 18;

  init_ = false;

  // get hold of back-end interface
  dbe_ = Service<DaqMonitorBEInterface>().operator->();

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", true);

  EcalUncalibratedRecHitCollection_ = ps.getParameter<edm::InputTag>("EcalUncalibratedRecHitCollection");

  for (int i = 0; i < 18 ; i++) {
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

}

void EETimingTask::setup(void){

  init_ = true;

  Char_t histo[200];

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalEndcap/EETimingTask");

    for (int i = 0; i < 18 ; i++) {
      sprintf(histo, "EETMT timing %s", Numbers::sEE(i+1).c_str());
      meTimeMap_[i] = dbe_->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 250, -4., 4., "s");
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

  try {

    Handle<EcalUncalibratedRecHitCollection> hits;
    e.getByLabel(EcalUncalibratedRecHitCollection_, hits);

    int neh = hits->size();
    LogDebug("EETimingTask") << "event " << ievt_ << " hits collection size " << neh;

    for ( EcalUncalibratedRecHitCollection::const_iterator hitItr = hits->begin(); hitItr != hits->end(); ++hitItr ) {

      EcalUncalibratedRecHit hit = (*hitItr);
      EBDetId id = hit.id();

      int ic = id.ic();
      int ie = (ic-1)/20 + 1;
      int ip = (ic-1)%20 + 1;

      int ism = Numbers::iSM( id ); if ( ism > 18 ) continue;

      float xie = ie - 0.5;
      float xip = ip - 0.5;

      LogDebug("EETimingTask") << " det id = " << id;
      LogDebug("EETimingTask") << " sm, eta, phi " << ism << " " << ie << " " << ip;

      MonitorElement* meTimeMap = 0;

      meTimeMap = meTimeMap_[ism-1];

      float xval = hit.amplitude();
      float yval = hit.jitter();
      float zval = hit.pedestal();

      LogDebug("EETimingTask") << " hit amplitude " << xval;
      LogDebug("EETimingTask") << " hit jitter " << yval;
      LogDebug("EETimingTask") << " hit pedestal " << zval;

      if ( meTimeMap ) meTimeMap->Fill(xie, xip, yval);

    }

  } catch ( exception& ex) {

    LogWarning("EETimingTask") << EcalUncalibratedRecHitCollection_ << " not available";

  }

}

