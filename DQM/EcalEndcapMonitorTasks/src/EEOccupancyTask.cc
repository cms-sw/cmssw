/*
 * \file EEOccupancyTask.cc
 *
 * $Date: 2007/05/24 13:26:12 $
 * $Revision: 1.10 $
 * \author G. Della Ricca
 * \author G. Franzoni
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
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include <DQM/EcalCommon/interface/Numbers.h>

#include <DQM/EcalEndcapMonitorTasks/interface/EEOccupancyTask.h>

using namespace cms;
using namespace edm;
using namespace std;

EEOccupancyTask::EEOccupancyTask(const ParameterSet& ps){

  init_ = false;

  // get hold of back-end interface
  dbe_ = Service<DaqMonitorBEInterface>().operator->();

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", true);

  EEDigiCollection_ = ps.getParameter<edm::InputTag>("EEDigiCollection");
  EcalPnDiodeDigiCollection_ = ps.getParameter<edm::InputTag>("EcalPnDiodeDigiCollection");

  for (int i = 0; i < 18; i++) {
    meOccupancy_[i]    = 0;
    meOccupancyMem_[i] = 0;
  }

}

EEOccupancyTask::~EEOccupancyTask(){

}

void EEOccupancyTask::beginJob(const EventSetup& c){

  ievt_ = 0;

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalEndcap/EEOccupancyTask");
    dbe_->rmdir("EcalEndcap/EEOccupancyTask");
  }

}

void EEOccupancyTask::setup(void){

  init_ = true;

  Char_t histo[200];

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalEndcap/EEOccupancyTask");

    for (int i = 0; i < 18; i++) {
      sprintf(histo, "EEOT occupancy %s", Numbers::sEE(i+1).c_str());
      meOccupancy_[i] = dbe_->book2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50.);
      dbe_->tag(meOccupancy_[i], i+1);
    }
    for (int i = 0; i < 18; i++) {
      sprintf(histo, "EEOT MEM occupancy %s", Numbers::sEE(i+1).c_str());
      meOccupancyMem_[i] = dbe_->book2D(histo, histo, 10, 0., 10., 5, 0., 5.);
      dbe_->tag(meOccupancyMem_[i], i+1);
    }

  }

}

void EEOccupancyTask::cleanup(void){

  if ( ! enableCleanup_ ) return;

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalEndcap/EEOccupancyTask");

    for (int i = 0; i < 18; i++) {
      if ( meOccupancy_[i] ) dbe_->removeElement( meOccupancy_[i]->getName() );
      meOccupancy_[i] = 0;
      if ( meOccupancyMem_[i] ) dbe_->removeElement( meOccupancyMem_[i]->getName() );
      meOccupancyMem_[i] = 0;
    }

  }

  init_ = false;

}

void EEOccupancyTask::endJob(void) {

  LogInfo("EEOccupancyTask") << "analyzed " << ievt_ << " events";

  if ( init_ ) this->cleanup();

}

void EEOccupancyTask::analyze(const Event& e, const EventSetup& c){

  Numbers::initGeometry(c);

  if ( ! init_ ) this->setup();

  ievt_++;

  try {

    Handle<EEDigiCollection> digis;
    e.getByLabel(EEDigiCollection_, digis);

    int need = digis->size();
    LogDebug("EEOccupancyTask") << "event " << ievt_ << " digi collection size " << need;

    for ( EEDigiCollection::const_iterator digiItr = digis->begin(); digiItr != digis->end(); ++digiItr ) {

      EEDataFrame dataframe = (*digiItr);
      EEDetId id = dataframe.id();

      int ix = 101 - id.ix();
      int iy = id.iy();

      int ism = Numbers::iSM( id );
  
      float xix = ix - 0.5;
      float xiy = iy - 0.5;

      LogDebug("EEOccupancyTask") << " det id = " << id;
      LogDebug("EEOccupancyTask") << " sm, ix, iy " << ism << " " << ix << " " << iy;

      if ( xix <= 0. || xix >= 100. || xiy <= 0. || xiy >= 100. ) {
        LogWarning("EEOccupancyTask") << " det id = " << id;
        LogWarning("EEOccupancyTask") << " sm, ix, iw " << ism << " " << ix << " " << iy;
        LogWarning("EEOccupancyTask") << " xix, xiy " << xix << " " << xiy;
      }

      if ( meOccupancy_[ism-1] ) meOccupancy_[ism-1]->Fill(xix, xiy);

    }

  } catch ( exception& ex) {

    LogWarning("EEOccupancyTask") << EEDigiCollection_ << " not available";

  }

  try {

    Handle<EcalPnDiodeDigiCollection> PNs;
    e.getByLabel(EcalPnDiodeDigiCollection_, PNs);

    // filling mem occupancy only for the 5 channels belonging
    // to a fully reconstructed PN's

    for ( EcalPnDiodeDigiCollection::const_iterator pnItr = PNs->begin(); pnItr != PNs->end(); ++pnItr ) {

      EcalPnDiodeDigi pn = (*pnItr);
      EcalPnDiodeDetId id = pn.id();

      int   ism   = Numbers::iSM( id );

      float PnId  = (*pnItr).id().iPnId();

      PnId        = PnId - 0.5;
      float st    = 0.0;

      for (int chInStrip = 1; chInStrip <= 5; chInStrip++){
        if ( meOccupancyMem_[ism-1] ) {
           st = chInStrip - 0.5;
           meOccupancyMem_[ism-1]->Fill(PnId, st);
        }
      }

    }

  } catch ( exception& ex) {

    LogWarning("EEOccupancyTask") << EcalPnDiodeDigiCollection_ << " not available";

  }

}

