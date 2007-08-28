/*
 * \file EBTimingTask.cc
 *
 * $Date: 2007/07/27 16:33:08 $
 * $Revision: 1.19 $
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

#include <DQM/EcalBarrelMonitorTasks/interface/EBTimingTask.h>

using namespace cms;
using namespace edm;
using namespace std;

EBTimingTask::EBTimingTask(const ParameterSet& ps){

  init_ = false;

  // get hold of back-end interface
  dbe_ = Service<DaqMonitorBEInterface>().operator->();

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", true);

  EcalUncalibratedRecHitCollection_ = ps.getParameter<edm::InputTag>("EcalUncalibratedRecHitCollection");

  for (int i = 0; i < 36 ; i++) {
    meTimeMap_[i] = 0;
  }

}

EBTimingTask::~EBTimingTask(){

}

void EBTimingTask::beginJob(const EventSetup& c){

  ievt_ = 0;

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalBarrel/EBTimingTask");
    dbe_->rmdir("EcalBarrel/EBTimingTask");
  }

}

void EBTimingTask::setup(void){

  init_ = true;

  Char_t histo[200];

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalBarrel/EBTimingTask");

    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBTMT timing %s", Numbers::sEB(i+1).c_str());
      meTimeMap_[i] = dbe_->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 250, 0., 10., "s");
      dbe_->tag(meTimeMap_[i], i+1);
    }

  }

}

void EBTimingTask::cleanup(void){

  if ( ! enableCleanup_ ) return;

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalBarrel/EBTimingTask");

    for ( int i = 0; i < 36; i++ ) {
      if ( meTimeMap_[i] ) dbe_->removeElement( meTimeMap_[i]->getName() );
      meTimeMap_[i] = 0;
    }

  }

  init_ = false;

}

void EBTimingTask::endJob(void){

  LogInfo("EBTimingTask") << "analyzed " << ievt_ << " events";

  if ( init_ ) this->cleanup();

}

void EBTimingTask::analyze(const Event& e, const EventSetup& c){

  Numbers::initGeometry(c);

  if ( ! init_ ) this->setup();

  ievt_++;

  try {

    Handle<EcalUncalibratedRecHitCollection> hits;
    e.getByLabel(EcalUncalibratedRecHitCollection_, hits);

    int neh = hits->size();
    LogDebug("EBTimingTask") << "event " << ievt_ << " hits collection size " << neh;

    for ( EcalUncalibratedRecHitCollection::const_iterator hitItr = hits->begin(); hitItr != hits->end(); ++hitItr ) {

      EcalUncalibratedRecHit hit = (*hitItr);
      EBDetId id = hit.id();

      int ic = id.ic();
      int ie = (ic-1)/20 + 1;
      int ip = (ic-1)%20 + 1;

      int ism = Numbers::iSM( id );

      float xie = ie - 0.5;
      float xip = ip - 0.5;

      LogDebug("EBTimingTask") << " det id = " << id;
      LogDebug("EBTimingTask") << " sm, eta, phi " << ism << " " << ie << " " << ip;

      MonitorElement* meTimeMap = 0;

      meTimeMap = meTimeMap_[ism-1];

      float xval = hit.amplitude();
      if ( xval <= 0. ) xval = 0.0;
      float yval = hit.jitter() + 6.0;
      if ( yval <= 0. ) yval = 0.0;
      float zval = hit.pedestal();
      if ( zval <= 0. ) zval = 0.0;

      LogDebug("EBTimingTask") << " hit amplitude " << xval;
      LogDebug("EBTimingTask") << " hit jitter " << yval;
      LogDebug("EBTimingTask") << " hit pedestal " << zval;

      if ( meTimeMap ) meTimeMap->Fill(xie, xip, yval);

    }

  } catch ( exception& ex) {

    LogWarning("EBTimingTask") << EcalUncalibratedRecHitCollection_ << " not available";

  }

}

