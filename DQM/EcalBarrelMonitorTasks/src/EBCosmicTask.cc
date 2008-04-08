/*
 * \file EBCosmicTask.cc
 *
 * $Date: 2008/04/08 15:32:09 $
 * $Revision: 1.103 $
 * \author G. Della Ricca
 *
*/

#include <iostream>
#include <fstream>
#include <vector>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/MonitorElement.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include <DQM/EcalCommon/interface/Numbers.h>

#include <DQM/EcalBarrelMonitorTasks/interface/EBCosmicTask.h>

using namespace cms;
using namespace edm;
using namespace std;

EBCosmicTask::EBCosmicTask(const ParameterSet& ps){

  init_ = false;

  dqmStore_ = Service<DQMStore>().operator->();

  prefixME_ = ps.getUntrackedParameter<string>("prefixME", "");

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  EcalRawDataCollection_ = ps.getParameter<edm::InputTag>("EcalRawDataCollection");
  EcalUncalibratedRecHitCollection_ = ps.getParameter<edm::InputTag>("EcalUncalibratedRecHitCollection");
  EcalRecHitCollection_ = ps.getParameter<edm::InputTag>("EcalRecHitCollection");

  lowThreshold_  = 0.06125; // 7 ADC counts at G200
  highThreshold_ = 0.12500; // typical muon energy deposit is 250 MeV

  minJitter_ = -2.0;
  maxJitter_ =  1.5;

  for (int i = 0; i < 36; i++) {
    meCutMap_[i] = 0;
    meSelMap_[i] = 0;
    meSpectrum_[0][i] = 0;
    meSpectrum_[1][i] = 0;
  }

}

EBCosmicTask::~EBCosmicTask(){

}

void EBCosmicTask::beginJob(const EventSetup& c){

  ievt_ = 0;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EBCosmicTask");
    dqmStore_->rmdir(prefixME_ + "/EBCosmicTask");
  }

  Numbers::initGeometry(c, false);

}

void EBCosmicTask::setup(void){

  init_ = true;

  char histo[200];

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EBCosmicTask");

    dqmStore_->setCurrentFolder(prefixME_ + "/EBCosmicTask/Cut");
    for (int i = 0; i < 36; i++) {
      sprintf(histo, "EBCT energy cut %s", Numbers::sEB(i+1).c_str());
      meCutMap_[i] = dqmStore_->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096., "s");
      meCutMap_[i]->setAxisTitle("ieta", 1);
      meCutMap_[i]->setAxisTitle("iphi", 2);
    }

    dqmStore_->setCurrentFolder(prefixME_ + "/EBCosmicTask/Sel");
    for (int i = 0; i < 36; i++) {
      sprintf(histo, "EBCT energy sel %s", Numbers::sEB(i+1).c_str());
      meSelMap_[i] = dqmStore_->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096., "s");
      meSelMap_[i]->setAxisTitle("ieta", 1);
      meSelMap_[i]->setAxisTitle("iphi", 2);
    }

    dqmStore_->setCurrentFolder(prefixME_ + "/EBCosmicTask/Spectrum");
    for (int i = 0; i < 36; i++) {
      sprintf(histo, "EBCT 1x1 energy spectrum %s", Numbers::sEB(i+1).c_str());
      meSpectrum_[0][i] = dqmStore_->book1D(histo, histo, 100, 0., 1.5);
      meSpectrum_[0][i]->setAxisTitle("energy (GeV)", 1);
      sprintf(histo, "EBCT 3x3 energy spectrum %s", Numbers::sEB(i+1).c_str());
      meSpectrum_[1][i] = dqmStore_->book1D(histo, histo, 100, 0., 1.5);
      meSpectrum_[1][i]->setAxisTitle("energy (GeV)", 1);
    }

  }

}

void EBCosmicTask::cleanup(void){

  if ( ! enableCleanup_ ) return;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EBCosmicTask");

    dqmStore_->setCurrentFolder(prefixME_ + "/EBCosmicTask/Cut");
    for (int i = 0; i < 36; i++) {
      if ( meCutMap_[i] ) dqmStore_->removeElement( meCutMap_[i]->getName() );
      meCutMap_[i] = 0;
    }

    dqmStore_->setCurrentFolder(prefixME_ + "/EBCosmicTask/Sel");
    for (int i = 0; i < 36; i++) {
      if ( meSelMap_[i] ) dqmStore_->removeElement( meSelMap_[i]->getName() );
      meSelMap_[i] = 0;
    }

    dqmStore_->setCurrentFolder(prefixME_ + "/EBCosmicTask/Spectrum");
    for (int i = 0; i < 36; i++) {
      if ( meSpectrum_[0][i] ) dqmStore_->removeElement( meSpectrum_[0][i]->getName() );
      meSpectrum_[0][i] = 0;
      if ( meSpectrum_[1][i] ) dqmStore_->removeElement( meSpectrum_[1][i]->getName() );
      meSpectrum_[1][i] = 0;
    }

  }

  init_ = false;

}

void EBCosmicTask::endJob(void){

  LogInfo("EBCosmicTask") << "analyzed " << ievt_ << " events";

  if ( init_ ) this->cleanup();

}

void EBCosmicTask::analyze(const Event& e, const EventSetup& c){

  bool isData = true;
  bool enable = false;
  map<int, EcalDCCHeaderBlock> dccMap;

  Handle<EcalRawDataCollection> dcchs;

  if ( e.getByLabel(EcalRawDataCollection_, dcchs) ) {

    for ( EcalRawDataCollection::const_iterator dcchItr = dcchs->begin(); dcchItr != dcchs->end(); ++dcchItr ) {

      EcalDCCHeaderBlock dcch = (*dcchItr);

      if ( Numbers::subDet( dcch ) != EcalBarrel ) continue;

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

  } else {

    isData = false; enable = true;
    LogWarning("EBCosmicTask") << EcalRawDataCollection_ << " not available";

  }

  if ( ! enable ) return;

  if ( ! init_ ) this->setup();

  ievt_++;

  Handle<EcalRecHitCollection> hits;

  if ( e.getByLabel(EcalRecHitCollection_, hits) ) {

    int nebh = hits->size();
    LogDebug("EBCosmicTask") << "event " << ievt_ << " hits collection size " << nebh;

    Handle<EcalUncalibratedRecHitCollection> uhits;

    if ( ! e.getByLabel(EcalUncalibratedRecHitCollection_, uhits) ) {
      LogWarning("EBCosmicTask") << EcalUncalibratedRecHitCollection_ << " not available";
    }

    for ( EcalRecHitCollection::const_iterator hitItr = hits->begin(); hitItr != hits->end(); ++hitItr ) {

      EcalRecHit hit = (*hitItr);
      EBDetId id = hit.id();

      int ic = id.ic();
      int ie = (ic-1)/20 + 1;
      int ip = (ic-1)%20 + 1;

      int ism = Numbers::iSM( id );

      float xie = ie - 0.5;
      float xip = ip - 0.5;

      if ( isData ) {
      map<int, EcalDCCHeaderBlock>::iterator i = dccMap.find(ism);
      if ( i == dccMap.end() ) continue;

      if ( ! ( dccMap[ism].getRunType() == EcalDCCHeaderBlock::COSMIC ||
               dccMap[ism].getRunType() == EcalDCCHeaderBlock::MTCC ||
               dccMap[ism].getRunType() == EcalDCCHeaderBlock::COSMICS_GLOBAL ||
               dccMap[ism].getRunType() == EcalDCCHeaderBlock::PHYSICS_GLOBAL ||
               dccMap[ism].getRunType() == EcalDCCHeaderBlock::COSMICS_LOCAL ||
               dccMap[ism].getRunType() == EcalDCCHeaderBlock::PHYSICS_LOCAL ) ) continue;
      }

      LogDebug("EBCosmicTask") << " det id = " << id;
      LogDebug("EBCosmicTask") << " sm, ieta, iphi " << ism << " " << ie << " " << ip;

      float xval = hit.energy();
      if ( xval <= 0. ) xval = 0.0;
      
      LogDebug("EBCosmicTask") << " hit energy " << xval;

      // look for the seeds 
      float e3x3 = 0.;
      bool isSeed = true;

      // evaluate 3x3 matrix around a seed
      for(int icry=0; icry<9; ++icry) {
        unsigned int row    = icry/3;
        unsigned int column = icry%3;
        int icryEta = id.ieta()+column-1;
        int icryPhi = id.iphi()+row-1;
        if ( EBDetId::validDetId(icryEta, icryPhi) ) {
          EBDetId id3x3 = EBDetId(icryEta, icryPhi, EBDetId::ETAPHIMODE);
          if ( hits->find(id3x3) != hits->end() ) {
            float neighbourEnergy = hits->find(id3x3)->energy();
            e3x3 += neighbourEnergy;
            if ( neighbourEnergy > xval ) isSeed = false;
          }
        }
      }

      // find the jitter of the seed
      float jitter = -999.;
      if ( isSeed ) {
        if ( uhits.isValid() ) {
          if ( uhits->find(id) != uhits->end() ) {
            jitter = uhits->find(id)->jitter();
          }
        }
      }

      if ( xval >= lowThreshold_ ) {
        if ( meCutMap_[ism-1] ) meCutMap_[ism-1]->Fill(xie, xip, xval);
      }

      if ( isSeed && e3x3 >= highThreshold_ && jitter > minJitter_ && jitter < maxJitter_ ) {
        if ( meSelMap_[ism-1] ) meSelMap_[ism-1]->Fill(xie, xip, e3x3);
      }

      if ( meSpectrum_[0][ism-1] ) meSpectrum_[0][ism-1]->Fill(xval);

      if ( isSeed && xval >= lowThreshold_ && jitter > minJitter_ && jitter < maxJitter_ ) {
        if ( meSpectrum_[1][ism-1] ) meSpectrum_[1][ism-1]->Fill(e3x3);
      }

    }

  } else {

    LogWarning("EBCosmicTask") << EcalRecHitCollection_ << " not available";

  }

}

