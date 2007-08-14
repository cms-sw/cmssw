/*
 * \file EECosmicTask.cc
 *
 * $Date: 2007/06/13 18:01:29 $
 * $Revision: 1.10 $
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

#include <DQM/EcalEndcapMonitorTasks/interface/EECosmicTask.h>

using namespace cms;
using namespace edm;
using namespace std;

EECosmicTask::EECosmicTask(const ParameterSet& ps){

  Numbers::maxSM = 18;

  init_ = false;

  // get hold of back-end interface
  dbe_ = Service<DaqMonitorBEInterface>().operator->();

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", true);

  EcalRawDataCollection_ = ps.getParameter<edm::InputTag>("EcalRawDataCollection");
  EcalRecHitCollection_ = ps.getParameter<edm::InputTag>("EcalRecHitCollection");

  for (int i = 0; i < 18 ; i++) {
    meCutMap_[i] = 0;
    meSelMap_[i] = 0;
    meSpectrumMap_[i] = 0;
  }

}

EECosmicTask::~EECosmicTask(){

}

void EECosmicTask::beginJob(const EventSetup& c){

  ievt_ = 0;

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalEndcap/EECosmicTask");
    dbe_->rmdir("EcalEndcap/EECosmicTask");
  }

}

void EECosmicTask::setup(void){

  init_ = true;

  Char_t histo[200];

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalEndcap/EECosmicTask");

    dbe_->setCurrentFolder("EcalEndcap/EECosmicTask/Cut");
    for (int i = 0; i < 18 ; i++) {
      sprintf(histo, "EECT energy cut %s", Numbers::sEE(i+1).c_str());
      meCutMap_[i] = dbe_->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096., "s");
    }

    dbe_->setCurrentFolder("EcalEndcap/EECosmicTask/Sel");
    for (int i = 0; i < 18 ; i++) {
      sprintf(histo, "EECT energy sel %s", Numbers::sEE(i+1).c_str());
      meSelMap_[i] = dbe_->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096., "s");
    }

    dbe_->setCurrentFolder("EcalEndcap/EECosmicTask/Spectrum");
    for (int i = 0; i < 18 ; i++) {
      sprintf(histo, "EECT energy spectrum %s", Numbers::sEE(i+1).c_str());
      meSpectrumMap_[i] = dbe_->book1D(histo, histo, 100, 0., 1.5);
    }

  }

}

void EECosmicTask::cleanup(void){

  if ( ! enableCleanup_ ) return;

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalEndcap/EECosmicTask");

    dbe_->setCurrentFolder("EcalEndcap/EECosmicTask/Cut");
    for (int i = 0; i < 18 ; i++) {
      if ( meCutMap_[i] ) dbe_->removeElement( meCutMap_[i]->getName() );
      meCutMap_[i] = 0;
    }

    dbe_->setCurrentFolder("EcalEndcap/EECosmicTask/Sel");
    for (int i = 0; i < 18 ; i++) {
      if ( meSelMap_[i] ) dbe_->removeElement( meSelMap_[i]->getName() );
      meSelMap_[i] = 0;
    }

    dbe_->setCurrentFolder("EcalEndcap/EECosmicTask/Spectrum");
    for (int i = 0; i < 18 ; i++) {
      if ( meSpectrumMap_[i] ) dbe_->removeElement( meSpectrumMap_[i]->getName() );
      meSpectrumMap_[i] = 0;
    }

  }

  init_ = false;

}

void EECosmicTask::endJob(void){

  LogInfo("EECosmicTask") << "analyzed " << ievt_ << " events";

  if ( init_ ) this->cleanup();

}

void EECosmicTask::analyze(const Event& e, const EventSetup& c){

  bool enable = false;
  map<int, EcalDCCHeaderBlock> dccMap;

  try {

    Handle<EcalRawDataCollection> dcchs;
    e.getByLabel(EcalRawDataCollection_, dcchs);

    for ( EcalRawDataCollection::const_iterator dcchItr = dcchs->begin(); dcchItr != dcchs->end(); ++dcchItr ) {

      EcalDCCHeaderBlock dcch = (*dcchItr);

      int ism = Numbers::iSM( dcch ); if ( ism > 18 ) continue;

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

    LogWarning("EECosmicTask") << EcalRawDataCollection_ << " not available";

  }

  if ( ! enable ) return;

  if ( ! init_ ) this->setup();

  ievt_++;

  try {

    Handle<EcalRecHitCollection> hits;
    e.getByLabel(EcalRecHitCollection_, hits);

    int nebh = hits->size();
    LogDebug("EECosmicTask") << "event " << ievt_ << " hits collection size " << nebh;

    for ( EcalRecHitCollection::const_iterator hitItr = hits->begin(); hitItr != hits->end(); ++hitItr ) {

      EcalRecHit hit = (*hitItr);
      EBDetId id = hit.id();

      int ic = id.ic();
      int ie = (ic-1)/20 + 1;
      int ip = (ic-1)%20 + 1;

      int ism = Numbers::iSM( id ); if ( ism > 18 ) continue;

      float xie = ie - 0.5;
      float xip = ip - 0.5;

      map<int, EcalDCCHeaderBlock>::iterator i = dccMap.find(ism);
      if ( i == dccMap.end() ) continue;

      if ( ! ( dccMap[ism].getRunType() == EcalDCCHeaderBlock::COSMIC ||
               dccMap[ism].getRunType() == EcalDCCHeaderBlock::MTCC ||
               dccMap[ism].getRunType() == EcalDCCHeaderBlock::COSMICS_GLOBAL ||
               dccMap[ism].getRunType() == EcalDCCHeaderBlock::PHYSICS_GLOBAL ||
               dccMap[ism].getRunType() == EcalDCCHeaderBlock::COSMICS_LOCAL ||
               dccMap[ism].getRunType() == EcalDCCHeaderBlock::PHYSICS_LOCAL ) ) enable = true;

      LogDebug("EECosmicTask") << " det id = " << id;
      LogDebug("EECosmicTask") << " sm, eta, phi " << ism << " " << ie << " " << ip;

      float xval = hit.energy();
      if ( xval <= 0. ) xval = 0.0;

      LogDebug("EECosmicTask") << " hit energy " << xval;

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

    LogWarning("EECosmicTask") << EcalRecHitCollection_ << " not available";

  }

}

