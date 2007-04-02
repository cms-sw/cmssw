/*
 * \file EETriggerTowerTask.cc
 *
 * $Date: 2007/03/26 17:34:07 $
 * $Revision: 1.29 $
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

#include <DQM/EcalEndcapMonitorTasks/interface/EETriggerTowerTask.h>

using namespace cms;
using namespace edm;
using namespace std;

EETriggerTowerTask::EETriggerTowerTask(const ParameterSet& ps){

  init_ = false;

  EcalTrigPrimDigiCollection_ = ps.getParameter<edm::InputTag>("EcalTrigPrimDigiCollection");
  EcalUncalibratedRecHitCollection_ = ps.getParameter<edm::InputTag>("EcalUncalibratedRecHitCollection");

  for (int i = 0; i < 36 ; i++) {
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

  DaqMonitorBEInterface* dbe = 0;

  // get hold of back-end interface
  dbe = Service<DaqMonitorBEInterface>().operator->();

  if ( dbe ) {
    dbe->setCurrentFolder("EcalEndcap/EETriggerTowerTask");
    dbe->rmdir("EcalEndcap/EETriggerTowerTask");
  }

}

void EETriggerTowerTask::setup(void){

  init_ = true;

  Char_t histo[200];

  DaqMonitorBEInterface* dbe = 0;

  // get hold of back-end interface
  dbe = Service<DaqMonitorBEInterface>().operator->();

  if ( dbe ) {
    dbe->setCurrentFolder("EcalEndcap/EETriggerTowerTask");

    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EETTT Et map SM%02d", i+1);
      meEtMap_[i] = dbe->bookProfile2D(histo, histo, 17, 0., 17., 4, 0., 4., 128, 0, 512., "s");
      dbe->tag(meEtMap_[i], i+1);
      sprintf(histo, "EETTT FineGrainVeto SM%02d", i+1);
      meVeto_[i] = dbe->book3D(histo, histo, 17, 0., 17., 4, 0., 4., 2, 0., 2.);
      dbe->tag(meVeto_[i], i+1);
      sprintf(histo, "EETTT Flags SM%02d", i+1);
      meFlags_[i] = dbe->book3D(histo, histo, 17, 0., 17., 4, 0., 4., 8, 0., 8.);
      dbe->tag(meFlags_[i], i+1);
    }

    dbe->setCurrentFolder("EcalEndcap/EETriggerTowerTask/EnergyMaps");

    for (int i = 0; i < 36 ; i++) {
      for (int j = 0; j < 68 ; j++) {
        sprintf(histo, "EETTT Et T SM%02d TT%02d", i+1, j+1);
        meEtMapT_[i][j] = dbe->book1D(histo, histo, 128, 0.0001, 512.);
        dbe->tag(meEtMapT_[i][j], i+1);
        sprintf(histo, "EETTT Et R SM%02d TT%02d", i+1, j+1);
        meEtMapR_[i][j] = dbe->book1D(histo, histo, 128, 0.0001, 512.);
        dbe->tag(meEtMapR_[i][j], i+1);
      }
    }

  }

}

void EETriggerTowerTask::cleanup(void){

  DaqMonitorBEInterface* dbe = 0;

  // get hold of back-end interface
  dbe = Service<DaqMonitorBEInterface>().operator->();

  if ( dbe ) {
    dbe->setCurrentFolder("EcalEndcap/EETriggerTowerTask");

    for ( int i = 0; i < 36; i++ ) {
      if ( meEtMap_[i] ) dbe->removeElement( meEtMap_[i]->getName() );
      meEtMap_[i] = 0;
      if ( meVeto_[i] ) dbe->removeElement( meVeto_[i]->getName() );
      meVeto_[i] = 0;
      if ( meFlags_[i] ) dbe->removeElement( meFlags_[i]->getName() );
      meFlags_[i] = 0;
    }

    dbe->setCurrentFolder("EcalEndcap/EETriggerTowerTask/EnergyMaps");

    for ( int i = 0; i < 36; i++ ) {
      for ( int j = 0; j < 36; j++ ) {
        if ( meEtMapT_[i][j] ) dbe->removeElement( meEtMapT_[i][j]->getName() );
        meEtMapT_[i][j] = 0;
        if ( meEtMapR_[i][j] ) dbe->removeElement( meEtMapR_[i][j]->getName() );
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

      int ismt = id.iDCC();

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

  float xmap[36][68];

  for (int i = 0; i < 36 ; i++) {
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

      int ism = id.ism();

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

    for (int i = 0; i < 36 ; i++) {
      for (int j = 0; j < 68 ; j++) {
         float xval = xmap[i][j];
         if ( meEtMapR_[i][j] && xval != 0 ) meEtMapR_[i][j]->Fill(xval);
      }
    }

  } catch ( std::exception& ex) {
    LogWarning("EETriggerTowerTask") << EcalUncalibratedRecHitCollection_ << " not available";
  }

}

