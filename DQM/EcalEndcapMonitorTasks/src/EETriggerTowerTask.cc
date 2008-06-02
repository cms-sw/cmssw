/*
 * \file EETriggerTowerTask.cc
 *
 * $Date: 2007/05/12 12:12:25 $
 * $Revision: 1.6 $
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

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include <DQM/EcalCommon/interface/Numbers.h>

#include <DQM/EcalEndcapMonitorTasks/interface/EETriggerTowerTask.h>

using namespace cms;
using namespace edm;
using namespace std;

EETriggerTowerTask::EETriggerTowerTask(const ParameterSet& ps){

  init_ = false;

  // get hold of back-end interface
  dbe_ = Service<DaqMonitorBEInterface>().operator->();

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", true);

  EcalTrigPrimDigiCollection_ = ps.getParameter<edm::InputTag>("EcalTrigPrimDigiCollection");
  EcalUncalibratedRecHitCollection_ = ps.getParameter<edm::InputTag>("EcalUncalibratedRecHitCollection");

  for (int i = 0; i < 18 ; i++) {
    meEtMap_[i] = 0;
    meVeto_[i] = 0;
    meFlags_[i] = 0;
    for (int j = 0; j < 68 ; j++) {
      meEtMapT_[i][j] = 0;
      meEtMapR_[i][j] = 0;
    }
  }

}

EETriggerTowerTask::~EETriggerTowerTask(){

}

void EETriggerTowerTask::beginJob(const EventSetup& c){

  ievt_ = 0;

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalEndcap/EETriggerTowerTask");
    dbe_->rmdir("EcalEndcap/EETriggerTowerTask");
  }

}

void EETriggerTowerTask::setup(void){

  init_ = true;

  Char_t histo[200];

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalEndcap/EETriggerTowerTask");

    for (int i = 0; i < 18 ; i++) {
      sprintf(histo, "EETTT Et map SM%02d", i+1);
      meEtMap_[i] = dbe_->bookProfile2D(histo, histo, 17, 0., 17., 4, 0., 4., 128, 0, 512., "s");
      dbe_->tag(meEtMap_[i], i+1);
      sprintf(histo, "EETTT FineGrainVeto SM%02d", i+1);
      meVeto_[i] = dbe_->book3D(histo, histo, 17, 0., 17., 4, 0., 4., 2, 0., 2.);
      dbe_->tag(meVeto_[i], i+1);
      sprintf(histo, "EETTT Flags SM%02d", i+1);
      meFlags_[i] = dbe_->book3D(histo, histo, 17, 0., 17., 4, 0., 4., 8, 0., 8.);
      dbe_->tag(meFlags_[i], i+1);
    }

    dbe_->setCurrentFolder("EcalEndcap/EETriggerTowerTask/EnergyMaps");

    for (int i = 0; i < 18 ; i++) {
      for (int j = 0; j < 68 ; j++) {
        sprintf(histo, "EETTT Et T SM%02d TT%02d", i+1, j+1);
        meEtMapT_[i][j] = dbe_->book1D(histo, histo, 128, 0.0001, 512.);
        dbe_->tag(meEtMapT_[i][j], i+1);
        sprintf(histo, "EETTT Et R SM%02d TT%02d", i+1, j+1);
        meEtMapR_[i][j] = dbe_->book1D(histo, histo, 128, 0.0001, 512.);
        dbe_->tag(meEtMapR_[i][j], i+1);
      }
    }

  }

}

void EETriggerTowerTask::cleanup(void){

  if ( ! enableCleanup_ ) return;

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalEndcap/EETriggerTowerTask");

    for ( int i = 0; i < 18; i++ ) {
      if ( meEtMap_[i] ) dbe_->removeElement( meEtMap_[i]->getName() );
      meEtMap_[i] = 0;
      if ( meVeto_[i] ) dbe_->removeElement( meVeto_[i]->getName() );
      meVeto_[i] = 0;
      if ( meFlags_[i] ) dbe_->removeElement( meFlags_[i]->getName() );
      meFlags_[i] = 0;
    }

    dbe_->setCurrentFolder("EcalEndcap/EETriggerTowerTask/EnergyMaps");

    for ( int i = 0; i < 18; i++ ) {
      for ( int j = 0; j < 18; j++ ) {
        if ( meEtMapT_[i][j] ) dbe_->removeElement( meEtMapT_[i][j]->getName() );
        meEtMapT_[i][j] = 0;
        if ( meEtMapR_[i][j] ) dbe_->removeElement( meEtMapR_[i][j]->getName() );
        meEtMapR_[i][j] = 0;
      }
    }

  }

  init_ = false;

}

void EETriggerTowerTask::endJob(void){

  LogInfo("EETriggerTowerTask") << "analyzed " << ievt_ << " events";

  if ( init_ ) this->cleanup();

}

void EETriggerTowerTask::analyze(const Event& e, const EventSetup& c){

  Numbers::initGeometry(c);

  if ( ! init_ ) this->setup();

  ievt_++;

  try {

    Handle<EcalTrigPrimDigiCollection> tpdigis;
    e.getByLabel(EcalTrigPrimDigiCollection_, tpdigis);

    int nebtpd = tpdigis->size();
    LogDebug("EETriggerTowerTask") << "event " << ievt_ << " trigger primitive digi collection size " << nebtpd;

    for ( EcalTrigPrimDigiCollection::const_iterator tpdigiItr = tpdigis->begin(); tpdigiItr != tpdigis->end(); ++tpdigiItr ) {

      EcalTriggerPrimitiveDigi data = (*tpdigiItr);
      EcalTrigTowerDetId id = data.id();

      int iet = id.ieta();
      int ipt = id.iphi();

      // phi_tower: change the range from global to SM-local
      ipt     = ( (ipt-1) % 4) +1;

      // phi_tower: range matters too
      //    if ( id.zside() >0)
      //      { ipt = 5 - ipt;      }

      int ismt = id.iDCC(); if ( ismt > 18 ) continue;

      int itt = 4*(iet-1)+(ipt-1)+1;

      float xiet = iet - 0.5;
      float xipt = ipt - 0.5;

      LogDebug("EETriggerTowerTask") << " det id = " << id;
      LogDebug("EETriggerTowerTask") << " sm, eta, phi " << ismt << " " << iet << " " << ipt;

      float xval;

      xval = data.compressedEt();
      if ( meEtMap_[ismt-1] ) meEtMap_[ismt-1]->Fill(xiet, xipt, xval);

      xval = 0.5 + data.fineGrain();
      if ( meVeto_[ismt-1] ) meVeto_[ismt-1]->Fill(xiet, xipt, xval);

      xval = 0.5 + data.ttFlag();
      if ( meFlags_[ismt-1] ) meFlags_[ismt-1]->Fill(xiet, xipt, xval);

      xval = data.compressedEt();
      if ( meEtMapT_[ismt-1][itt-1] ) meEtMapT_[ismt-1][itt-1]->Fill(xval);

    }

  } catch ( std::exception& ex) {
    LogWarning("EETriggerTowerTask") << EcalTrigPrimDigiCollection_ << " not available";
  }

  float xmap[18][68];

  for (int i = 0; i < 18 ; i++) {
    for (int j = 0; j < 68 ; j++) {
      xmap[i][j] = 0.;
    }
  }

  try {

    Handle<EcalUncalibratedRecHitCollection> hits;
    e.getByLabel(EcalUncalibratedRecHitCollection_, hits);

    int nebh = hits->size();
    LogDebug("EETriggerTowerTask") << "event " << ievt_ << " hits collection size " << nebh;

    for ( EcalUncalibratedRecHitCollection::const_iterator hitItr = hits->begin(); hitItr != hits->end(); ++hitItr ) {

      EcalUncalibratedRecHit hit = (*hitItr);
      EBDetId id = hit.id();

      int ic = id.ic();
      int ie = (ic-1)/20 + 1;
      int ip = (ic-1)%20 + 1;

      int ism = id.ism(); if ( ism > 18 ) continue;

      int iet = 1 + ((ie-1)/5);
      int ipt = 1 + ((ip-1)/5);

      int itt = 4*(iet-1) + (ipt-1) + 1;

      LogDebug("EETriggerTowerTask") << " det id = " << id;
      LogDebug("EETriggerTowerTask") << " sm, eta, phi " << ism << " " << ie << " " << ip;

      float xval = 0.;

      xval = hit.amplitude();
      if ( xval <= 0. ) xval = 0.0;

//      xval = xval * (1./16.) * TMath::Sin(2*TMath::ATan(TMath::Exp(-0.0174*(ie-0.5))));

      xval = xval * (1./16.);

      xmap[ism-1][itt-1] = xmap[ism-1][itt-1] + xval;

    }

    for (int i = 0; i < 18 ; i++) {
      for (int j = 0; j < 68 ; j++) {
         float xval = xmap[i][j];
         if ( meEtMapR_[i][j] && xval != 0 ) meEtMapR_[i][j]->Fill(xval);
      }
    }

  } catch ( std::exception& ex) {
    LogWarning("EETriggerTowerTask") << EcalUncalibratedRecHitCollection_ << " not available";
  }

}

