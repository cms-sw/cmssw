/*
 * \file EECosmicTask.cc
 *
 * $Date: 2008/01/09 12:17:31 $
 * $Revision: 1.24 $
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
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include <DQM/EcalCommon/interface/Numbers.h>

#include <DQM/EcalEndcapMonitorTasks/interface/EECosmicTask.h>

using namespace cms;
using namespace edm;
using namespace std;

EECosmicTask::EECosmicTask(const ParameterSet& ps){

  init_ = false;

  // get hold of back-end interface
  dbe_ = Service<DaqMonitorBEInterface>().operator->();

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  EcalRawDataCollection_ = ps.getParameter<edm::InputTag>("EcalRawDataCollection");
  EcalUncalibRecHitCollection_ = ps.getParameter<edm::InputTag>("EcalUncalibRecHitCollection");
  EcalRecHitCollection_ = ps.getParameter<edm::InputTag>("EcalRecHitCollection");

  lowThreshold_  = 0.06125; // 7 ADC counts at G200
  highThreshold_ = 0.12500; // typical muon energy deposit is 250 MeV

  minJitter_ = -2.0;
  maxJitter_ =  1.5;

  for (int i = 0; i < 18; i++) {
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

  Numbers::initGeometry(c);

}

void EECosmicTask::setup(void){

  init_ = true;

  Char_t histo[200];

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalEndcap/EECosmicTask");

    dbe_->setCurrentFolder("EcalEndcap/EECosmicTask/Cut");
    for (int i = 0; i < 18; i++) {
      sprintf(histo, "EECT energy cut %s", Numbers::sEE(i+1).c_str());
      meCutMap_[i] = dbe_->bookProfile2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 4096, 0., 4096., "s");
      meCutMap_[i]->setAxisTitle("jx", 1);
      meCutMap_[i]->setAxisTitle("jy", 2);
    }

    dbe_->setCurrentFolder("EcalEndcap/EECosmicTask/Sel");
    for (int i = 0; i < 18; i++) {
      sprintf(histo, "EECT energy sel %s", Numbers::sEE(i+1).c_str());
      meSelMap_[i] = dbe_->bookProfile2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 4096, 0., 4096., "s");
      meSelMap_[i]->setAxisTitle("jx", 1);
      meSelMap_[i]->setAxisTitle("jy", 2);
    }

    dbe_->setCurrentFolder("EcalEndcap/EECosmicTask/Spectrum");
    for (int i = 0; i < 18; i++) {
      sprintf(histo, "EECT energy spectrum %s", Numbers::sEE(i+1).c_str());
      meSpectrumMap_[i] = dbe_->book1D(histo, histo, 100, 0., 1.5);
      meSpectrumMap_[i]->setAxisTitle("energy (GeV)", 1);
    }

  }

}

void EECosmicTask::cleanup(void){

  if ( ! enableCleanup_ ) return;

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalEndcap/EECosmicTask");

    dbe_->setCurrentFolder("EcalEndcap/EECosmicTask/Cut");
    for (int i = 0; i < 18; i++) {
      if ( meCutMap_[i] ) dbe_->removeElement( meCutMap_[i]->getName() );
      meCutMap_[i] = 0;
    }

    dbe_->setCurrentFolder("EcalEndcap/EECosmicTask/Sel");
    for (int i = 0; i < 18; i++) {
      if ( meSelMap_[i] ) dbe_->removeElement( meSelMap_[i]->getName() );
      meSelMap_[i] = 0;
    }

    dbe_->setCurrentFolder("EcalEndcap/EECosmicTask/Spectrum");
    for (int i = 0; i < 18; i++) {
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

    if ( e.getByLabel(EcalUncalibRecHitCollection_, uhits) ) {
    
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

        if( ism >=  1 && ism <=  9 ) iz = -1;
        if( ism >= 10 && ism <= 18 ) iz = +1;

        map<int, EcalDCCHeaderBlock>::iterator i = dccMap.find(ism);
        if ( i == dccMap.end() ) continue;

        if ( ! ( dccMap[ism].getRunType() == EcalDCCHeaderBlock::COSMIC ||
                 dccMap[ism].getRunType() == EcalDCCHeaderBlock::MTCC ||
                 dccMap[ism].getRunType() == EcalDCCHeaderBlock::COSMICS_GLOBAL ||
                 dccMap[ism].getRunType() == EcalDCCHeaderBlock::PHYSICS_GLOBAL ||
                 dccMap[ism].getRunType() == EcalDCCHeaderBlock::COSMICS_LOCAL ||
                 dccMap[ism].getRunType() == EcalDCCHeaderBlock::PHYSICS_LOCAL ) ) continue;

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
            EEDetId Xtals3x3 = EEDetId(icryX, icryY, iz, EEDetId::XYMODE);
            float neighbourEnergy = hits->find(Xtals3x3)->energy();
            e3x3 += neighbourEnergy;
            if( neighbourEnergy > xval ) isSeed = false;
          }
        }

        // find the jitter of the seed
        float jitter = -999.;
        if( isSeed ) {
          EcalUncalibratedRecHitCollection::const_iterator uhitItr = uhits->find( hitItr->detid() );
          jitter = uhitItr->jitter();
        }

        if ( xval >= lowThreshold_ ) {
          if ( meCutMap_[ism-1] ) meCutMap_[ism-1]->Fill(xix, xiy, xval);
        }

        if ( isSeed && e3x3 >= highThreshold_ && jitter > minJitter_ && jitter < maxJitter_ ) {
          if ( meSelMap_[ism-1] ) meSelMap_[ism-1]->Fill(xix, xiy, e3x3);
        }

        if ( isSeed && jitter > minJitter_ && jitter < maxJitter_ ) {
          if ( meSpectrumMap_[ism-1] ) meSpectrumMap_[ism-1]->Fill(xval);
        }

      }

    }  else {

      LogWarning("EECosmicTask") << EcalUncalibRecHitCollection_ << " not available";

    }

  } else {

    LogWarning("EECosmicTask") << EcalRecHitCollection_ << " not available";

  }

}

