/*
 * \file EBCosmicTask.cc
 *
 * $Date: 2007/08/14 17:43:06 $
 * $Revision: 1.75 $
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

#include <DQM/EcalBarrelMonitorTasks/interface/EBCosmicTask.h>

using namespace cms;
using namespace edm;
using namespace std;

EBCosmicTask::EBCosmicTask(const ParameterSet& ps){

  init_ = false;

  // get hold of back-end interface
  dbe_ = Service<DaqMonitorBEInterface>().operator->();

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", true);

  EcalRawDataCollection_ = ps.getParameter<edm::InputTag>("EcalRawDataCollection");
  EcalRecHitCollection_ = ps.getParameter<edm::InputTag>("EcalRecHitCollection");

  for (int i = 0; i < 36 ; i++) {
    meCutMap_[i] = 0;
    meSelMap_[i] = 0;
    meSpectrumMap_[i] = 0;
  }

}

EBCosmicTask::~EBCosmicTask(){

}

void EBCosmicTask::beginJob(const EventSetup& c){

  ievt_ = 0;

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalBarrel/EBCosmicTask");
    dbe_->rmdir("EcalBarrel/EBCosmicTask");
  }

}

void EBCosmicTask::setup(void){

  init_ = true;

  Char_t histo[200];

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalBarrel/EBCosmicTask");

    dbe_->setCurrentFolder("EcalBarrel/EBCosmicTask/Cut");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBCT energy cut %s", Numbers::sEB(i+1).c_str());
      meCutMap_[i] = dbe_->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096., "s");
    }

    dbe_->setCurrentFolder("EcalBarrel/EBCosmicTask/Sel");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBCT energy sel %s", Numbers::sEB(i+1).c_str());
      meSelMap_[i] = dbe_->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096., "s");
    }

    dbe_->setCurrentFolder("EcalBarrel/EBCosmicTask/Spectrum");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBCT energy spectrum %s", Numbers::sEB(i+1).c_str());
      meSpectrumMap_[i] = dbe_->book1D(histo, histo, 100, 0., 1.5);
    }

  }

}

void EBCosmicTask::cleanup(void){

  if ( ! enableCleanup_ ) return;

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalBarrel/EBCosmicTask");

    dbe_->setCurrentFolder("EcalBarrel/EBCosmicTask/Cut");
    for (int i = 0; i < 36 ; i++) {
      if ( meCutMap_[i] ) dbe_->removeElement( meCutMap_[i]->getName() );
      meCutMap_[i] = 0;
    }

    dbe_->setCurrentFolder("EcalBarrel/EBCosmicTask/Sel");
    for (int i = 0; i < 36 ; i++) {
      if ( meSelMap_[i] ) dbe_->removeElement( meSelMap_[i]->getName() );
      meSelMap_[i] = 0;
    }

    dbe_->setCurrentFolder("EcalBarrel/EBCosmicTask/Spectrum");
    for (int i = 0; i < 36 ; i++) {
      if ( meSpectrumMap_[i] ) dbe_->removeElement( meSpectrumMap_[i]->getName() );
      meSpectrumMap_[i] = 0;
    }

  }

  init_ = false;

}

void EBCosmicTask::endJob(void){

  LogInfo("EBCosmicTask") << "analyzed " << ievt_ << " events";

  if ( init_ ) this->cleanup();

}

void EBCosmicTask::analyze(const Event& e, const EventSetup& c){

  Numbers::initGeometry(c);

  bool enable = false;
  map<int, EcalDCCHeaderBlock> dccMap;

  try {

    Handle<EcalRawDataCollection> dcchs;
    e.getByLabel(EcalRawDataCollection_, dcchs);

    for ( EcalRawDataCollection::const_iterator dcchItr = dcchs->begin(); dcchItr != dcchs->end(); ++dcchItr ) {

      EcalDCCHeaderBlock dcch = (*dcchItr);

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

  } catch ( exception& ex) {

    LogWarning("EBCosmicTask") << EcalRawDataCollection_ << " not available";

  }

  if ( ! enable ) return;

  if ( ! init_ ) this->setup();

  ievt_++;

  try {

    Handle<EcalRecHitCollection> hits;
    e.getByLabel(EcalRecHitCollection_, hits);

    int nebh = hits->size();
    LogDebug("EBCosmicTask") << "event " << ievt_ << " hits collection size " << nebh;

    for ( EcalRecHitCollection::const_iterator hitItr = hits->begin(); hitItr != hits->end(); ++hitItr ) {

      EcalRecHit hit = (*hitItr);
      EBDetId id = hit.id();

      int ic = id.ic();
      int ie = (ic-1)/20 + 1;
      int ip = (ic-1)%20 + 1;

      int ism = Numbers::iSM( id );

      float xie = ie - 0.5;
      float xip = ip - 0.5;

      map<int, EcalDCCHeaderBlock>::iterator i = dccMap.find(ism);
      if ( i == dccMap.end() ) continue;

      if ( ! ( dccMap[ism].getRunType() == EcalDCCHeaderBlock::COSMIC ||
               dccMap[ism].getRunType() == EcalDCCHeaderBlock::MTCC ||
               dccMap[ism].getRunType() == EcalDCCHeaderBlock::COSMICS_GLOBAL ||
               dccMap[ism].getRunType() == EcalDCCHeaderBlock::PHYSICS_GLOBAL ||
               dccMap[ism].getRunType() == EcalDCCHeaderBlock::COSMICS_LOCAL ||
               dccMap[ism].getRunType() == EcalDCCHeaderBlock::PHYSICS_LOCAL ) ) continue;

      LogDebug("EBCosmicTask") << " det id = " << id;
      LogDebug("EBCosmicTask") << " sm, eta, phi " << ism << " " << ie << " " << ip;

      float xval = hit.energy();
      if ( xval <= 0. ) xval = 0.0;

      LogDebug("EBCosmicTask") << " hit energy " << xval;

      const float lowThreshold  = 0.06125;
      const float highThreshold = 0.12500;

      if ( xval >= lowThreshold ) {
        if ( meCutMap_[ism-1] ) meCutMap_[ism-1]->Fill(xie, xip, xval);
      }

      if ( xval >= highThreshold ) {
        if ( meSelMap_[ism-1] ) meSelMap_[ism-1]->Fill(xie, xip, xval);
      }

      if ( meSpectrumMap_[ism-1] ) meSpectrumMap_[ism-1]->Fill(xval);

    }

  } catch ( exception& ex) {

    LogWarning("EBCosmicTask") << EcalRecHitCollection_ << " not available";

  }

}

