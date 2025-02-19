/*
 * \file EECosmicTask.cc
 *
 * $Date: 2012/04/27 13:46:14 $
 * $Revision: 1.60 $
 * \author G. Della Ricca
 *
*/

#include <iostream>
#include <fstream>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/MonitorElement.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "DQM/EcalCommon/interface/Numbers.h"

#include "DQM/EcalEndcapMonitorTasks/interface/EECosmicTask.h"

EECosmicTask::EECosmicTask(const edm::ParameterSet& ps){

  init_ = false;

  dqmStore_ = edm::Service<DQMStore>().operator->();

  prefixME_ = ps.getUntrackedParameter<std::string>("prefixME", "");

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  mergeRuns_ = ps.getUntrackedParameter<bool>("mergeRuns", false);

  EcalRawDataCollection_ = ps.getParameter<edm::InputTag>("EcalRawDataCollection");
  EcalUncalibratedRecHitCollection_ = ps.getParameter<edm::InputTag>("EcalUncalibratedRecHitCollection");
  EcalRecHitCollection_ = ps.getParameter<edm::InputTag>("EcalRecHitCollection");

  threshold_ = 0.12500; // typical muon energy deposit is 250 MeV

  minJitter_ = -2.0;
  maxJitter_ =  1.5;

  for (int i = 0; i < 18; i++) {
    meSelMap_[i] = 0;
    meSpectrum_[0][i] = 0;
    meSpectrum_[1][i] = 0;
  }

}

EECosmicTask::~EECosmicTask(){

}

void EECosmicTask::beginJob(void){

  ievt_ = 0;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EECosmicTask");
    dqmStore_->rmdir(prefixME_ + "/EECosmicTask");
  }

}

void EECosmicTask::beginRun(const edm::Run& r, const edm::EventSetup& c) {

  Numbers::initGeometry(c, false);

  if ( ! mergeRuns_ ) this->reset();

}

void EECosmicTask::endRun(const edm::Run& r, const edm::EventSetup& c) {

}

void EECosmicTask::reset(void) {

  for (int i = 0; i < 18; i++) {
    if ( meSelMap_[i] ) meSelMap_[i]->Reset();
    if ( meSpectrum_[0][i] ) meSpectrum_[0][i]->Reset();
    if ( meSpectrum_[1][i] ) meSpectrum_[1][i]->Reset();
  }

}

void EECosmicTask::setup(void){

  init_ = true;

  std::string name;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EECosmicTask");

    dqmStore_->setCurrentFolder(prefixME_ + "/EECosmicTask/Sel");
    for (int i = 0; i < 18; i++) {
      name = "EECT energy sel " + Numbers::sEE(i+1);
      meSelMap_[i] = dqmStore_->bookProfile2D(name, name, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 4096, 0., 4096., "s");
      meSelMap_[i]->setAxisTitle("ix", 1);
      if ( i+1 >= 1 && i+1 <= 9 ) meSelMap_[i]->setAxisTitle("101-ix", 1);
      meSelMap_[i]->setAxisTitle("iy", 2);
      meSelMap_[i]->setAxisTitle("energy (GeV)", 3);
    }

    dqmStore_->setCurrentFolder(prefixME_ + "/EECosmicTask/Spectrum");
    for (int i = 0; i < 18; i++) {
      name = "EECT 1x1 energy spectrum " + Numbers::sEE(i+1);
      meSpectrum_[0][i] = dqmStore_->book1D(name, name, 100, 0., 1.5);
      meSpectrum_[0][i]->setAxisTitle("energy (GeV)", 1);

      name = "EECT 3x3 energy spectrum " + Numbers::sEE(i+1);
      meSpectrum_[1][i] = dqmStore_->book1D(name, name, 100, 0., 1.5);
      meSpectrum_[1][i]->setAxisTitle("energy (GeV)", 1);
    }

  }

}

void EECosmicTask::cleanup(void){

  if ( ! init_ ) return;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EECosmicTask");

    dqmStore_->setCurrentFolder(prefixME_ + "/EECosmicTask/Sel");
    for (int i = 0; i < 18; i++) {
      if ( meSelMap_[i] ) dqmStore_->removeElement( meSelMap_[i]->getName() );
      meSelMap_[i] = 0;
    }

    dqmStore_->setCurrentFolder(prefixME_ + "/EECosmicTask/Spectrum");
    for (int i = 0; i < 18; i++) {
      if ( meSpectrum_[0][i] ) dqmStore_->removeElement( meSpectrum_[0][i]->getName() );
      meSpectrum_[0][i] = 0;
      if ( meSpectrum_[1][i] ) dqmStore_->removeElement( meSpectrum_[1][i]->getName() );
      meSpectrum_[1][i] = 0;
    }

  }

  init_ = false;

}

void EECosmicTask::endJob(void){

  edm::LogInfo("EECosmicTask") << "analyzed " << ievt_ << " events";

  if ( enableCleanup_ ) this->cleanup();

}

void EECosmicTask::analyze(const edm::Event& e, const edm::EventSetup& c){

  bool isData = true;
  bool enable = false;
  int runType[18];
  for (int i=0; i<18; i++) runType[i] = -1;

  edm::Handle<EcalRawDataCollection> dcchs;

  if ( e.getByLabel(EcalRawDataCollection_, dcchs) ) {

    for ( EcalRawDataCollection::const_iterator dcchItr = dcchs->begin(); dcchItr != dcchs->end(); ++dcchItr ) {

      if ( Numbers::subDet( *dcchItr ) != EcalEndcap ) continue;

      int ism = Numbers::iSM( *dcchItr, EcalEndcap );

      runType[ism-1] = dcchItr->getRunType();

      if ( dcchItr->getRunType() == EcalDCCHeaderBlock::COSMIC ||
           dcchItr->getRunType() == EcalDCCHeaderBlock::MTCC ||
           dcchItr->getRunType() == EcalDCCHeaderBlock::COSMICS_GLOBAL ||
           dcchItr->getRunType() == EcalDCCHeaderBlock::PHYSICS_GLOBAL ||
           dcchItr->getRunType() == EcalDCCHeaderBlock::COSMICS_LOCAL ||
           dcchItr->getRunType() == EcalDCCHeaderBlock::PHYSICS_LOCAL ) enable = true;

    }

  } else {

    isData = false; enable = true;
    edm::LogWarning("EECosmicTask") << EcalRawDataCollection_ << " not available";

  }

  if ( ! enable ) return;

  if ( ! init_ ) this->setup();

  ievt_++;

  edm::Handle<EcalRecHitCollection> hits;

  if ( e.getByLabel(EcalRecHitCollection_, hits) ) {

    int neeh = hits->size();
    LogDebug("EECosmicTask") << "event " << ievt_ << " hits collection size " << neeh;

    edm::Handle<EcalUncalibratedRecHitCollection> uhits;

    if ( ! e.getByLabel(EcalUncalibratedRecHitCollection_, uhits) ) {
      edm::LogWarning("EECosmicTask") << EcalUncalibratedRecHitCollection_ << " not available";
    }

    for ( EcalRecHitCollection::const_iterator hitItr = hits->begin(); hitItr != hits->end(); ++hitItr ) {

      EEDetId id = hitItr->id();

      int ix = id.ix();
      int iy = id.iy();

      int ism = Numbers::iSM( id );

      if ( ism >= 1 && ism <= 9 ) ix = 101 - ix;

      float xix = ix - 0.5;
      float xiy = iy - 0.5;

      int iz = 0;

      if ( ism >=  1 && ism <=  9 ) iz = -1;
      if ( ism >= 10 && ism <= 18 ) iz = +1;

      if ( isData ) {

        if ( ! ( runType[ism-1] == EcalDCCHeaderBlock::COSMIC ||
                 runType[ism-1] == EcalDCCHeaderBlock::MTCC ||
                 runType[ism-1] == EcalDCCHeaderBlock::COSMICS_GLOBAL ||
                 runType[ism-1] == EcalDCCHeaderBlock::PHYSICS_GLOBAL ||
                 runType[ism-1] == EcalDCCHeaderBlock::COSMICS_LOCAL ||
                 runType[ism-1] == EcalDCCHeaderBlock::PHYSICS_LOCAL ) ) continue;

      }

      float xval = hitItr->energy();
      if ( xval <= 0. ) xval = 0.0;

      // look for the seeds
      float e3x3 = 0.;
      bool isSeed = true;

      // evaluate 3x3 matrix around a seed
      for(int icry=0; icry<9; ++icry) {
        unsigned int row    = icry/3;
        unsigned int column = icry%3;
        int icryX = id.ix()+column-1;
        int icryY = id.iy()+row-1;
        if ( EEDetId::validDetId(icryX, icryY, iz) ) {
          EEDetId id3x3 = EEDetId(icryX, icryY, iz, EEDetId::XYMODE);
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

      if ( isSeed && e3x3 >= threshold_ && jitter > minJitter_ && jitter < maxJitter_ ) {
        if ( meSelMap_[ism-1] ) meSelMap_[ism-1]->Fill(xix, xiy, e3x3);
      }

      if ( meSpectrum_[0][ism-1] ) meSpectrum_[0][ism-1]->Fill(xval);

      if ( isSeed && xval >= threshold_ && jitter > minJitter_ && jitter < maxJitter_ ) {
        if ( meSpectrum_[1][ism-1] ) meSpectrum_[1][ism-1]->Fill(e3x3);
      }

    }

  } else {

    edm::LogWarning("EECosmicTask") << EcalRecHitCollection_ << " not available";

  }

}

