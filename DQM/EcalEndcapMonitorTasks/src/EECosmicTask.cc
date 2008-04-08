/*
 * \file EECosmicTask.cc
 *
 * $Date: 2008/04/08 15:32:10 $
 * $Revision: 1.41 $
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
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include <DQM/EcalCommon/interface/Numbers.h>

#include <DQM/EcalEndcapMonitorTasks/interface/EECosmicTask.h>

using namespace cms;
using namespace edm;
using namespace std;

EECosmicTask::EECosmicTask(const ParameterSet& ps){

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

  for (int i = 0; i < 18; i++) {
    meCutMap_[i] = 0;
    meSelMap_[i] = 0;
    meSpectrum_[0][i] = 0;
    meSpectrum_[1][i] = 0;
  }

}

EECosmicTask::~EECosmicTask(){

}

void EECosmicTask::beginJob(const EventSetup& c){

  ievt_ = 0;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EECosmicTask");
    dqmStore_->rmdir(prefixME_ + "/EECosmicTask");
  }

  Numbers::initGeometry(c, false);

}

void EECosmicTask::setup(void){

  init_ = true;

  char histo[200];

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EECosmicTask");

    dqmStore_->setCurrentFolder(prefixME_ + "/EECosmicTask/Cut");
    for (int i = 0; i < 18; i++) {
      sprintf(histo, "EECT energy cut %s", Numbers::sEE(i+1).c_str());
      meCutMap_[i] = dqmStore_->bookProfile2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 4096, 0., 4096., "s");
      meCutMap_[i]->setAxisTitle("jx", 1);
      meCutMap_[i]->setAxisTitle("jy", 2);
    }

    dqmStore_->setCurrentFolder(prefixME_ + "/EECosmicTask/Sel");
    for (int i = 0; i < 18; i++) {
      sprintf(histo, "EECT energy sel %s", Numbers::sEE(i+1).c_str());
      meSelMap_[i] = dqmStore_->bookProfile2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 4096, 0., 4096., "s");
      meSelMap_[i]->setAxisTitle("jx", 1);
      meSelMap_[i]->setAxisTitle("jy", 2);
    }

    dqmStore_->setCurrentFolder(prefixME_ + "/EECosmicTask/Spectrum");
    for (int i = 0; i < 18; i++) {
      sprintf(histo, "EECT 1x1 energy spectrum %s", Numbers::sEE(i+1).c_str());
      meSpectrum_[0][i] = dqmStore_->book1D(histo, histo, 100, 0., 1.5);
      meSpectrum_[0][i]->setAxisTitle("energy (GeV)", 1);
      sprintf(histo, "EECT 3x3 energy spectrum %s", Numbers::sEE(i+1).c_str());
      meSpectrum_[1][i] = dqmStore_->book1D(histo, histo, 100, 0., 1.5);
      meSpectrum_[1][i]->setAxisTitle("energy (GeV)", 1);
    }

  }

}

void EECosmicTask::cleanup(void){

  if ( ! enableCleanup_ ) return;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EECosmicTask");

    dqmStore_->setCurrentFolder(prefixME_ + "/EECosmicTask/Cut");
    for (int i = 0; i < 18; i++) {
      if ( meCutMap_[i] ) dqmStore_->removeElement( meCutMap_[i]->getName() );
      meCutMap_[i] = 0;
    }

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

  LogInfo("EECosmicTask") << "analyzed " << ievt_ << " events";

  if ( init_ ) this->cleanup();

}

void EECosmicTask::analyze(const Event& e, const EventSetup& c){

  bool isData = true;
  bool enable = false;
  map<int, EcalDCCHeaderBlock> dccMap;

  Handle<EcalRawDataCollection> dcchs;

  if ( e.getByLabel(EcalRawDataCollection_, dcchs) ) {

    for ( EcalRawDataCollection::const_iterator dcchItr = dcchs->begin(); dcchItr != dcchs->end(); ++dcchItr ) {

      EcalDCCHeaderBlock dcch = (*dcchItr);

      if ( Numbers::subDet( dcch ) != EcalEndcap ) continue;

      int ism = Numbers::iSM( dcch, EcalEndcap );

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
    LogWarning("EECosmicTask") << EcalRawDataCollection_ << " not available";

  }

  if ( ! enable ) return;

  if ( ! init_ ) this->setup();

  ievt_++;

  Handle<EcalRecHitCollection> hits;

  if ( e.getByLabel(EcalRecHitCollection_, hits) ) {

    int neeh = hits->size();
    LogDebug("EECosmicTask") << "event " << ievt_ << " hits collection size " << neeh;

    Handle<EcalUncalibratedRecHitCollection> uhits;

    if ( ! e.getByLabel(EcalUncalibratedRecHitCollection_, uhits) ) {
      LogWarning("EECosmicTask") << EcalUncalibratedRecHitCollection_ << " not available";
    }
    
    for ( EcalRecHitCollection::const_iterator hitItr = hits->begin(); hitItr != hits->end(); ++hitItr ) {

      EcalRecHit hit = (*hitItr);
      EEDetId id = hit.id();

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
      map<int, EcalDCCHeaderBlock>::iterator i = dccMap.find(ism);
      if ( i == dccMap.end() ) continue;

      if ( ! ( dccMap[ism].getRunType() == EcalDCCHeaderBlock::COSMIC ||
               dccMap[ism].getRunType() == EcalDCCHeaderBlock::MTCC ||
               dccMap[ism].getRunType() == EcalDCCHeaderBlock::COSMICS_GLOBAL ||
               dccMap[ism].getRunType() == EcalDCCHeaderBlock::PHYSICS_GLOBAL ||
               dccMap[ism].getRunType() == EcalDCCHeaderBlock::COSMICS_LOCAL ||
               dccMap[ism].getRunType() == EcalDCCHeaderBlock::PHYSICS_LOCAL ) ) continue;
      }

      LogDebug("EECosmicTask") << " det id = " << id;
      LogDebug("EECosmicTask") << " sm, ix, iy " << ism << " " << ix << " " << iy;

      float xval = hit.energy();
      if ( xval <= 0. ) xval = 0.0;

      LogDebug("EECosmicTask") << " hit energy " << xval;

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

      if ( xval >= lowThreshold_ ) {
        if ( meCutMap_[ism-1] ) meCutMap_[ism-1]->Fill(xix, xiy, xval);
      }

      if ( isSeed && e3x3 >= highThreshold_ && jitter > minJitter_ && jitter < maxJitter_ ) {
        if ( meSelMap_[ism-1] ) meSelMap_[ism-1]->Fill(xix, xiy, e3x3);
      }

      if ( meSpectrum_[0][ism-1] ) meSpectrum_[0][ism-1]->Fill(xval);

      if ( isSeed && xval >= lowThreshold_ && jitter > minJitter_ && jitter < maxJitter_ ) {
        if ( meSpectrum_[1][ism-1] ) meSpectrum_[1][ism-1]->Fill(e3x3);
      }

    }

  } else {

    LogWarning("EECosmicTask") << EcalRecHitCollection_ << " not available";

  }

}

