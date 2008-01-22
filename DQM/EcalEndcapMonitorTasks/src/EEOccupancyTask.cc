/*
 * \file EEOccupancyTask.cc
 *
 * $Date: 2008/01/22 19:47:15 $
 * $Revision: 1.22 $
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

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  EEDigiCollection_ = ps.getParameter<edm::InputTag>("EEDigiCollection");
  EcalPnDiodeDigiCollection_ = ps.getParameter<edm::InputTag>("EcalPnDiodeDigiCollection");
  EcalRecHitCollection_ = ps.getParameter<edm::InputTag>("EcalRecHitCollection");
  EcalTrigPrimDigiCollection_ = ps.getParameter<edm::InputTag>("EcalTrigPrimDigiCollection");

  for (int i = 0; i < 18; i++) {
    meOccupancy_[i]    = 0;
    meOccupancyMem_[i] = 0;
  }

  meEEDigiOccupancy_ = 0;
  meEEDigiOccupancyProjX_ = 0;
  meEEDigiOccupancyProjY_ = 0;

  meEERecHitOccupancy_ = 0;
  meEERecHitOccupancyProjX_ = 0;
  meEERecHitOccupancyProjY_ = 0;

  meEETrigPrimDigiOccupancy_ = 0;
  meEETrigPrimDigiOccupancyProjX_ = 0;
  meEETrigPrimDigiOccupancyProjY_ = 0;

}

EEOccupancyTask::~EEOccupancyTask(){

}

void EEOccupancyTask::beginJob(const EventSetup& c){

  ievt_ = 0;

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalEndcap/EEOccupancyTask");
    dbe_->rmdir("EcalEndcap/EEOccupancyTask");
  }

  Numbers::initGeometry(c);

}

void EEOccupancyTask::setup(void){

  init_ = true;

  char histo[200];

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalEndcap/EEOccupancyTask");

    for (int i = 0; i < 18; i++) {
      sprintf(histo, "EEOT digi occupancy %s", Numbers::sEE(i+1).c_str());
      meOccupancy_[i] = dbe_->book2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50.);
      dbe_->tag(meOccupancy_[i], i+1);
    }
    for (int i = 0; i < 18; i++) {
      sprintf(histo, "EEOT MEM digi occupancy %s", Numbers::sEE(i+1).c_str());
      meOccupancyMem_[i] = dbe_->book2D(histo, histo, 10, 0., 10., 5, 0., 5.);
      dbe_->tag(meOccupancyMem_[i], i+1);
    }

    sprintf(histo, "EEOT EE digi occupancy");
    meEEDigiOccupancy_ = dbe_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
    sprintf(histo, "EEOT EE digi occupancy projection x");
    meEEDigiOccupancyProjX_ = dbe_->book1D(histo, histo, 100, 0., 100.);
    sprintf(histo, "EEOT EE digi occupancy projection y");
    meEEDigiOccupancyProjY_ = dbe_->book1D(histo, histo, 100, 0., 100.);

    sprintf(histo, "EEOT EE rec hit occupancy");
    meEERecHitOccupancy_ = dbe_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
    sprintf(histo, "EEOT EE rec hit occupancy projection x");
    meEERecHitOccupancyProjX_ = dbe_->book1D(histo, histo, 100, 0., 100.);
    sprintf(histo, "EEOT EE rec hit occupancy projection y");
    meEERecHitOccupancyProjY_ = dbe_->book1D(histo, histo, 100, 0., 100.);

    sprintf(histo, "EEOT EE trigger primitives digi occupancy");
    meEETrigPrimDigiOccupancy_ = dbe_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
    sprintf(histo, "EEOT EE trigger primitives digi occupancy projection x");
    meEETrigPrimDigiOccupancyProjX_ = dbe_->book1D(histo, histo, 100, 0., 100.);
    sprintf(histo, "EEOT EE trigger primitives digi occupancy projection y");
    meEETrigPrimDigiOccupancyProjY_ = dbe_->book1D(histo, histo, 100, 0., 100.);

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

    if ( meEEDigiOccupancy_ ) dbe_->removeElement( meEEDigiOccupancy_->getName() );
    meEEDigiOccupancy_ = 0;
    if ( meEEDigiOccupancyProjX_ ) dbe_->removeElement( meEEDigiOccupancyProjX_->getName() );
    meEEDigiOccupancyProjX_ = 0;
    if ( meEEDigiOccupancyProjY_ ) dbe_->removeElement( meEEDigiOccupancyProjY_->getName() );
    meEEDigiOccupancyProjY_ = 0;

    if ( meEERecHitOccupancy_ ) dbe_->removeElement( meEERecHitOccupancy_->getName() );
    meEERecHitOccupancy_ = 0;
    if ( meEERecHitOccupancyProjX_ ) dbe_->removeElement( meEERecHitOccupancyProjX_->getName() );
    meEERecHitOccupancyProjX_ = 0;
    if ( meEERecHitOccupancyProjY_ ) dbe_->removeElement( meEERecHitOccupancyProjY_->getName() );
    meEERecHitOccupancyProjY_ = 0;

    if ( meEETrigPrimDigiOccupancy_ ) dbe_->removeElement( meEETrigPrimDigiOccupancy_->getName() );
    meEETrigPrimDigiOccupancy_ = 0;
    if ( meEETrigPrimDigiOccupancyProjX_ ) dbe_->removeElement( meEETrigPrimDigiOccupancyProjX_->getName() );
    meEETrigPrimDigiOccupancyProjX_ = 0;
    if ( meEETrigPrimDigiOccupancyProjY_ ) dbe_->removeElement( meEETrigPrimDigiOccupancyProjY_->getName() );
    meEETrigPrimDigiOccupancyProjY_ = 0;

  }

  init_ = false;

}

void EEOccupancyTask::endJob(void) {

  LogInfo("EEOccupancyTask") << "analyzed " << ievt_ << " events";

  if ( init_ ) this->cleanup();

}

void EEOccupancyTask::analyze(const Event& e, const EventSetup& c){

  if ( ! init_ ) this->setup();

  ievt_++;

  Handle<EEDigiCollection> digis;

  if ( e.getByLabel(EEDigiCollection_, digis) ) {

    int need = digis->size();
    LogDebug("EEOccupancyTask") << "event " << ievt_ << " digi collection size " << need;

    for ( EEDigiCollection::const_iterator digiItr = digis->begin(); digiItr != digis->end(); ++digiItr ) {

      EEDataFrame dataframe = (*digiItr);
      EEDetId id = dataframe.id();

      int ix = id.ix();
      int iy = id.iy();

      int ism = Numbers::iSM( id );

      if ( ism >= 1 && ism <= 9 ) ix = 101 - ix;

      float xix = ix - 0.5;
      float xiy = iy - 0.5;

      LogDebug("EEOccupancyTask") << " det id = " << id;
      LogDebug("EEOccupancyTask") << " sm, ix, iy " << ism << " " << ix << " " << iy;

      if ( xix <= 0. || xix >= 100. || xiy <= 0. || xiy >= 100. ) {
        LogWarning("EEOccupancyTask") << " det id = " << id;
        LogWarning("EEOccupancyTask") << " sm, ix, iw " << ism << " " << ix << " " << iy;
        LogWarning("EEOccupancyTask") << " xix, xiy " << xix << " " << xiy;
      }

      if ( meOccupancy_[ism-1] ) meOccupancy_[ism-1]->Fill( xix, xiy );

      int eex = id.ix();
      int eey = id.iy();

      if ( meEEDigiOccupancy_ ) meEEDigiOccupancy_->Fill( eex, eey );
      if ( meEEDigiOccupancyProjX_ ) meEEDigiOccupancyProjX_->Fill( eex );
      if ( meEEDigiOccupancyProjY_ ) meEEDigiOccupancyProjY_->Fill( eey );

    }

  } else {

    LogWarning("EEOccupancyTask") << EEDigiCollection_ << " not available";

  }

  Handle<EcalPnDiodeDigiCollection> PNs;

  if ( e.getByLabel(EcalPnDiodeDigiCollection_, PNs) ) {

    // filling mem occupancy only for the 5 channels belonging
    // to a fully reconstructed PN's

    for ( EcalPnDiodeDigiCollection::const_iterator pnItr = PNs->begin(); pnItr != PNs->end(); ++pnItr ) {

      EcalPnDiodeDigi pn = (*pnItr);
      EcalPnDiodeDetId id = pn.id();

      if ( Numbers::subDet( id ) != EcalEndcap ) continue;

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

  } else {

    LogWarning("EEOccupancyTask") << EcalPnDiodeDigiCollection_ << " not available";

  }

  Handle<EcalRecHitCollection> rechits;

  if ( e.getByLabel(EcalRecHitCollection_, rechits) ) {

    int nebrh = rechits->size();
    LogDebug("EEOccupancyTask") << "event " << ievt_ << " rec hits collection size " << nebrh;

    for ( EcalRecHitCollection::const_iterator rechitItr = rechits->begin(); rechitItr != rechits->end(); ++rechitItr ) {

      EEDetId id = rechitItr->id();

      int eex = id.ix();
      int eey = id.iy();

      if ( meEERecHitOccupancy_ ) meEERecHitOccupancy_->Fill( eex, eey );
      if ( meEERecHitOccupancyProjX_ ) meEERecHitOccupancyProjX_->Fill( eex );
      if ( meEERecHitOccupancyProjY_ ) meEERecHitOccupancyProjY_->Fill( eey );

    }

  } else {

    LogWarning("EEOccupancyTask") << EcalRecHitCollection_ << " not available";

  }

  Handle<EcalTrigPrimDigiCollection> trigPrimDigis;

  if ( e.getByLabel(EcalTrigPrimDigiCollection_, trigPrimDigis) ) {

    int nebtpg = trigPrimDigis->size();
    LogDebug("EEOccupancyTask") << "event " << ievt_ << " trigger primitives digis collection size " << nebtpg;

    for ( EcalTrigPrimDigiCollection::const_iterator tpdigiItr = trigPrimDigis->begin(); 
	  tpdigiItr != trigPrimDigis->end(); ++tpdigiItr ) {

      EcalTriggerPrimitiveDigi data = (*tpdigiItr);
      EcalTrigTowerDetId idt = data.id();

      if ( Numbers::subDet( idt ) != EcalEndcap ) continue;
      
      int ismt = Numbers::iSM( idt );
      
      vector<DetId> crystals = Numbers::crystals( idt );
      
      for ( unsigned int i=0; i<crystals.size(); i++ ) {
	
	EEDetId id = crystals[i];
	
	int ix = id.ix();
	int iy = id.iy();
	
	if ( ismt >= 1 && ismt <= 9 ) ix = 101 - ix;

	if ( meEETrigPrimDigiOccupancy_ ) meEETrigPrimDigiOccupancy_->Fill( ix, iy );
	if ( meEETrigPrimDigiOccupancyProjX_ ) meEETrigPrimDigiOccupancyProjX_->Fill( ix );
	if ( meEETrigPrimDigiOccupancyProjY_ ) meEETrigPrimDigiOccupancyProjY_->Fill( iy );
      }
    }

  } else {

    LogWarning("EEOccupancyTask") << EcalTrigPrimDigiCollection_ << " not available";

  }

}

