/*
 * \file EBOccupancyTask.cc
 *
 * $Date: 2008/01/20 17:11:38 $
 * $Revision: 1.34 $
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
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include <DQM/EcalCommon/interface/Numbers.h>

#include <DQM/EcalBarrelMonitorTasks/interface/EBOccupancyTask.h>

using namespace cms;
using namespace edm;
using namespace std;

EBOccupancyTask::EBOccupancyTask(const ParameterSet& ps){

  init_ = false;

  // get hold of back-end interface
  dbe_ = Service<DaqMonitorBEInterface>().operator->();

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  EBDigiCollection_ = ps.getParameter<edm::InputTag>("EBDigiCollection");
  EcalPnDiodeDigiCollection_ = ps.getParameter<edm::InputTag>("EcalPnDiodeDigiCollection");
  EcalRecHitCollection_ = ps.getParameter<edm::InputTag>("EcalRecHitCollection");
  EcalTrigPrimDigiCollection_ = ps.getParameter<edm::InputTag>("EcalTrigPrimDigiCollection");

  for (int i = 0; i < 36; i++) {
    meOccupancy_[i]    = 0;
    meOccupancyMem_[i] = 0;
  }

  meEBDigiOccupancy_ = 0;
  meEBDigiOccupancyProjEta_ = 0;
  meEBDigiOccupancyProjPhi_ = 0;

  meEBRecHitOccupancy_ = 0;
  meEBRecHitOccupancyProjEta_ = 0;
  meEBRecHitOccupancyProjPhi_ = 0;

  meEBTrigPrimDigiOccupancy_ = 0;
  meEBTrigPrimDigiOccupancyProjEta_ = 0;
  meEBTrigPrimDigiOccupancyProjPhi_ = 0;

}

EBOccupancyTask::~EBOccupancyTask(){

}

void EBOccupancyTask::beginJob(const EventSetup& c){

  ievt_ = 0;

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalBarrel/EBOccupancyTask");
    dbe_->rmdir("EcalBarrel/EBOccupancyTask");
  }

  Numbers::initGeometry(c);

}

void EBOccupancyTask::setup(void){

  init_ = true;

  Char_t histo[200];

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalBarrel/EBOccupancyTask");

    for (int i = 0; i < 36; i++) {
      sprintf(histo, "EBOT digi occupancy %s", Numbers::sEB(i+1).c_str());
      meOccupancy_[i] = dbe_->book2D(histo, histo, 85, 0., 85., 20, 0., 20.);
      dbe_->tag(meOccupancy_[i], i+1);
    }
    for (int i = 0; i < 36; i++) {
      sprintf(histo, "EBOT MEM digi occupancy %s", Numbers::sEB(i+1).c_str());
      meOccupancyMem_[i] = dbe_->book2D(histo, histo, 10, 0., 10., 5, 0., 5.);
      dbe_->tag(meOccupancyMem_[i], i+1);
    }

    sprintf(histo, "EBOT EB digi occupancy");
    meEBDigiOccupancy_ = dbe_->book2D(histo, histo, 72, 0., 360., 34, -85., 85.);
    sprintf(histo, "EBOT EB digi occupancy projection eta");
    meEBDigiOccupancyProjEta_ = dbe_->book1D(histo, histo, 34, -85., 85.);
    sprintf(histo, "EBOT EB digi occupancy projection phi");
    meEBDigiOccupancyProjPhi_ = dbe_->book1D(histo, histo, 72, 0., 360.);

    sprintf(histo, "EBOT EB rec hit occupancy");
    meEBRecHitOccupancy_ = dbe_->book2D(histo, histo, 72, 0., 360., 34, -85., 85.);
    sprintf(histo, "EBOT EB rec hit occupancy projection eta");
    meEBRecHitOccupancyProjEta_ = dbe_->book1D(histo, histo, 34, -85., 85.);
    sprintf(histo, "EBOT EB rec hit occupancy projection phi");
    meEBRecHitOccupancyProjPhi_ = dbe_->book1D(histo, histo, 72, 0., 360.);

    sprintf(histo, "EBOT EB trigger primitives digi occupancy");
    meEBTrigPrimDigiOccupancy_ = dbe_->book2D(histo, histo, 72, 0., 360., 34, -85., 85.);
    sprintf(histo, "EBOT EB trigger primitives digi occupancy projection eta");
    meEBTrigPrimDigiOccupancyProjEta_ = dbe_->book1D(histo, histo, 34, -85., 85.);
    sprintf(histo, "EBOT EB trigger primitives digi occupancy projection phi");
    meEBTrigPrimDigiOccupancyProjPhi_ = dbe_->book1D(histo, histo, 72, 0., 360.);

  }

}

void EBOccupancyTask::cleanup(void){

  if ( ! enableCleanup_ ) return;

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalBarrel/EBOccupancyTask");

    for (int i = 0; i < 36; i++) {
      if ( meOccupancy_[i] ) dbe_->removeElement( meOccupancy_[i]->getName() );
      meOccupancy_[i] = 0;
      if ( meOccupancyMem_[i] ) dbe_->removeElement( meOccupancyMem_[i]->getName() );
      meOccupancyMem_[i] = 0;
    }

    if ( meEBDigiOccupancy_ ) dbe_->removeElement( meEBDigiOccupancy_->getName() );
    meEBDigiOccupancy_ = 0;
    if ( meEBDigiOccupancyProjEta_ ) dbe_->removeElement( meEBDigiOccupancyProjEta_->getName() );
    meEBDigiOccupancyProjEta_ = 0;
    if ( meEBDigiOccupancyProjPhi_ ) dbe_->removeElement( meEBDigiOccupancyProjPhi_->getName() );
    meEBDigiOccupancyProjPhi_ = 0;

    if ( meEBRecHitOccupancy_ ) dbe_->removeElement( meEBRecHitOccupancy_->getName() );
    meEBRecHitOccupancy_ = 0;
    if ( meEBRecHitOccupancyProjEta_ ) dbe_->removeElement( meEBRecHitOccupancyProjEta_->getName() );
    meEBRecHitOccupancyProjEta_ = 0;
    if ( meEBRecHitOccupancyProjPhi_ ) dbe_->removeElement( meEBRecHitOccupancyProjPhi_->getName() );
    meEBRecHitOccupancyProjPhi_ = 0;

    if ( meEBTrigPrimDigiOccupancy_ ) dbe_->removeElement( meEBTrigPrimDigiOccupancy_->getName() );
    meEBTrigPrimDigiOccupancy_ = 0;
    if ( meEBTrigPrimDigiOccupancyProjEta_ ) dbe_->removeElement( meEBTrigPrimDigiOccupancyProjEta_->getName() );
    meEBTrigPrimDigiOccupancyProjEta_ = 0;
    if ( meEBTrigPrimDigiOccupancyProjPhi_ ) dbe_->removeElement( meEBTrigPrimDigiOccupancyProjPhi_->getName() );
    meEBTrigPrimDigiOccupancyProjPhi_ = 0;

  }

  init_ = false;

}

void EBOccupancyTask::endJob(void) {

  LogInfo("EBOccupancyTask") << "analyzed " << ievt_ << " events";

  if ( init_ ) this->cleanup();

}

void EBOccupancyTask::analyze(const Event& e, const EventSetup& c){

  if ( ! init_ ) this->setup();

  ievt_++;

  Handle<EBDigiCollection> digis;

  if ( e.getByLabel(EBDigiCollection_, digis) ) {

    int nebd = digis->size();
    LogDebug("EBOccupancyTask") << "event " << ievt_ << " digi collection size " << nebd;

    for ( EBDigiCollection::const_iterator digiItr = digis->begin(); digiItr != digis->end(); ++digiItr ) {

      EBDataFrame dataframe = (*digiItr);
      EBDetId id = dataframe.id();

      int ic = id.ic();
      int ie = (ic-1)/20 + 1;
      int ip = (ic-1)%20 + 1;

      int ism = Numbers::iSM( id );

      float xie = ie - 0.5;
      float xip = ip - 0.5;

      LogDebug("EBOccupancyTask") << " det id = " << id;
      LogDebug("EBOccupancyTask") << " sm, ieta, iphi " << ism << " " << ie << " " << ip;

      if ( xie <= 0. || xie >= 85. || xip <= 0. || xip >= 20. ) {
        LogWarning("EBOccupancyTask") << " det id = " << id;
        LogWarning("EBOccupancyTask") << " sm, ieta, iphi " << ism << " " << ie << " " << ip;
        LogWarning("EBOccupancyTask") << " xie, xip " << xie << " " << xip;
      }

      if ( meOccupancy_[ism-1] ) meOccupancy_[ism-1]->Fill(xie, xip);

      int ebeta = id.ieta();
      int ebphi = id.iphi();

      if ( meEBDigiOccupancy_ ) meEBDigiOccupancy_->Fill( ebphi, ebeta );
      if ( meEBDigiOccupancyProjEta_ ) meEBDigiOccupancyProjEta_->Fill( ebeta );
      if ( meEBDigiOccupancyProjPhi_ ) meEBDigiOccupancyProjPhi_->Fill( ebphi );

    }

  } else {

    LogWarning("EBOccupancyTask") << EBDigiCollection_ << " not available";

  }

  Handle<EcalPnDiodeDigiCollection> PNs;

  if ( e.getByLabel(EcalPnDiodeDigiCollection_, PNs) ) {

    // filling mem occupancy only for the 5 channels belonging
    // to a fully reconstructed PN's

    for ( EcalPnDiodeDigiCollection::const_iterator pnItr = PNs->begin(); pnItr != PNs->end(); ++pnItr ) {

      EcalPnDiodeDigi pn = (*pnItr);
      EcalPnDiodeDetId id = pn.id();

      if ( Numbers::subDet( id ) != EcalBarrel ) continue;

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

    LogWarning("EBOccupancyTask") << EcalPnDiodeDigiCollection_ << " not available";

  }

  Handle<EcalRecHitCollection> rechits;

  if ( e.getByLabel(EcalRecHitCollection_, rechits) ) {

    int nebrh = rechits->size();
    LogDebug("EBOccupancyTask") << "event " << ievt_ << " rec hits collection size " << nebrh;

    for ( EcalRecHitCollection::const_iterator rechitItr = rechits->begin(); rechitItr != rechits->end(); ++rechitItr ) {

      EBDetId id = rechitItr->id();
      int ebeta = id.ieta();
      int ebphi = id.iphi();

      if ( meEBRecHitOccupancy_ ) meEBRecHitOccupancy_->Fill( ebphi, ebeta );
      if ( meEBRecHitOccupancyProjEta_ ) meEBRecHitOccupancyProjEta_->Fill( ebeta );
      if ( meEBRecHitOccupancyProjPhi_ ) meEBRecHitOccupancyProjPhi_->Fill( ebphi );

    }

  } else {

    LogWarning("EBOccupancyTask") << EcalRecHitCollection_ << " not available";

  }

  Handle<EcalTrigPrimDigiCollection> trigPrimDigis;

  if ( e.getByLabel(EcalTrigPrimDigiCollection_, trigPrimDigis) ) {

    int nebtpg = trigPrimDigis->size();
    LogDebug("EBOccupancyTask") << "event " << ievt_ << " trigger primitives digis collection size " << nebtpg;

    for ( EcalTrigPrimDigiCollection::const_iterator tpdigiItr = trigPrimDigis->begin(); 
	  tpdigiItr != trigPrimDigis->end(); ++tpdigiItr ) {

      EcalTriggerPrimitiveDigi data = (*tpdigiItr);
      EcalTrigTowerDetId idt = data.id();

      int ebeta = idt.ieta();
      int ebphi = idt.iphi();

      if ( meEBTrigPrimDigiOccupancy_ ) meEBTrigPrimDigiOccupancy_->Fill( ebphi, ebeta );
      if ( meEBTrigPrimDigiOccupancyProjEta_ ) meEBTrigPrimDigiOccupancyProjEta_->Fill( ebeta );
      if ( meEBTrigPrimDigiOccupancyProjPhi_ ) meEBTrigPrimDigiOccupancyProjPhi_->Fill( ebphi );

    }

  } else {

    LogWarning("EBOccupancyTask") << EcalTrigPrimDigiCollection_ << " not available";

  }


}

