/*
 * \file EBCosmicTask.cc
 *
 * $Date: 2011/08/30 09:30:32 $
 * $Revision: 1.118 $
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
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "DQM/EcalCommon/interface/Numbers.h"

#include "DQM/EcalBarrelMonitorTasks/interface/EBCosmicTask.h"

EBCosmicTask::EBCosmicTask(const edm::ParameterSet& ps){

  init_ = false;

  dqmStore_ = edm::Service<DQMStore>().operator->();

  prefixME_ = ps.getUntrackedParameter<std::string>("prefixME", "");

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  mergeRuns_ = ps.getUntrackedParameter<bool>("mergeRuns", false);

  EcalRawDataCollection_ = ps.getParameter<edm::InputTag>("EcalRawDataCollection");
  EcalRecHitCollection_ = ps.getParameter<edm::InputTag>("EcalRecHitCollection");

  threshold_ = 0.12500; // typical muon energy deposit is 250 MeV

  for (int i = 0; i < 36; i++) {
    meSelMap_[i] = 0;
    meSpectrum_[0][i] = 0;
    meSpectrum_[1][i] = 0;
  }

  meSpectrumAll_ = 0;

  ievt_ = 0;

}

EBCosmicTask::~EBCosmicTask(){

}

void EBCosmicTask::beginJob(void){

  ievt_ = 0;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/Energy");
    dqmStore_->rmdir(prefixME_ + "/Energy");
  }

}

void EBCosmicTask::beginRun(const edm::Run& r, const edm::EventSetup& c) {

  Numbers::initGeometry(c, false);

  if ( ! mergeRuns_ ) this->reset();

}

void EBCosmicTask::endRun(const edm::Run& r, const edm::EventSetup& c) {

}

void EBCosmicTask::reset(void) {

  for (int i = 0; i < 36; i++) {
    if ( meSelMap_[i] ) meSelMap_[i]->Reset();
    if ( meSpectrum_[0][i] ) meSpectrum_[0][i]->Reset();
    if ( meSpectrum_[1][i] ) meSpectrum_[1][i]->Reset();
  }

  if(meSpectrumAll_) meSpectrumAll_->Reset();

}

void EBCosmicTask::setup(void){

  init_ = true;

  std::string name;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/Energy");

    dqmStore_->setCurrentFolder(prefixME_ + "/Energy/Profile");
    for (int i = 0; i < 36; i++) {
      name = "RecHitTask energy " + Numbers::sEB(i+1);
      meSelMap_[i] = dqmStore_->bookProfile2D(name, name, 85, 0., 85., 20, 0., 20., 0., 14000., "s");
      meSelMap_[i]->setAxisTitle("ieta", 1);
      meSelMap_[i]->setAxisTitle("iphi", 2);
      meSelMap_[i]->setAxisTitle("energy (GeV)", 3);
    }

    dqmStore_->setCurrentFolder(prefixME_ + "/Energy/Spectrum");
    for (int i = 0; i < 36; i++) {
      name = "RecHitTask energy 1D " + Numbers::sEB(i+1);
      meSpectrum_[0][i] = dqmStore_->book1D(name, name, 100, 0., 10.);
      meSpectrum_[0][i]->setAxisTitle("energy (GeV)", 1);
      name = "RecHitTask 3x3 " + Numbers::sEB(i+1);
      meSpectrum_[1][i] = dqmStore_->book1D(name, name, 100, 0., 10.);
      meSpectrum_[1][i]->setAxisTitle("energy (GeV)", 1);
    }

    name = "RecHitTask energy 1D all EB";
    meSpectrumAll_ = dqmStore_->book1D(name, name, 100, 0., 10.);
    meSpectrumAll_->setAxisTitle("energy (GeV)", 1);

  }

}

void EBCosmicTask::cleanup(void){

  if ( ! init_ ) return;

  if ( dqmStore_ ) {

    for (int i = 0; i < 36; i++) {
      if ( meSelMap_[i] ) dqmStore_->removeElement( meSelMap_[i]->getFullname() );
      meSelMap_[i] = 0;
    }

    for (int i = 0; i < 36; i++) {
      if ( meSpectrum_[0][i] ) dqmStore_->removeElement( meSpectrum_[0][i]->getFullname() );
      meSpectrum_[0][i] = 0;
      if ( meSpectrum_[1][i] ) dqmStore_->removeElement( meSpectrum_[1][i]->getFullname() );
      meSpectrum_[1][i] = 0;
    }

  }

  init_ = false;

}

void EBCosmicTask::endJob(void){

  edm::LogInfo("EBCosmicTask") << "analyzed " << ievt_ << " events";

  if ( enableCleanup_ ) this->cleanup();

}

void EBCosmicTask::analyze(const edm::Event& e, const edm::EventSetup& c){

  bool isData = true;
  bool enable = false;
  int runType[36];
  for (int i=0; i<36; i++) runType[i] = -1;

  edm::Handle<EcalRawDataCollection> dcchs;

  if ( e.getByLabel(EcalRawDataCollection_, dcchs) ) {

    for ( EcalRawDataCollection::const_iterator dcchItr = dcchs->begin(); dcchItr != dcchs->end(); ++dcchItr ) {

      if ( Numbers::subDet( *dcchItr ) != EcalBarrel ) continue;

      int ism = Numbers::iSM( *dcchItr, EcalBarrel );

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
    edm::LogWarning("EBCosmicTask") << EcalRawDataCollection_ << " not available";

  }

  if ( ! enable ) return;

  if ( ! init_ ) this->setup();

  ievt_++;

  uint32_t mask = 0xffffffff ^ ((0x1 << EcalRecHit::kGood));

  edm::Handle<EcalRecHitCollection> hits;

  if ( e.getByLabel(EcalRecHitCollection_, hits) ) {

    int nebh = hits->size();
    LogDebug("EBCosmicTask") << "event " << ievt_ << " hits collection size " << nebh;

    for ( EcalRecHitCollection::const_iterator hitItr = hits->begin(); hitItr != hits->end(); ++hitItr ) {

      if( hitItr->checkFlagMask(mask) ) continue;

      EBDetId id = hitItr->id();

      int ic = id.ic();
      int ie = (ic-1)/20 + 1;
      int ip = (ic-1)%20 + 1;

      int ism = Numbers::iSM( id );

      float xie = ie - 0.5;
      float xip = ip - 0.5;

      if ( isData ) {

        if ( ! ( runType[ism-1] == EcalDCCHeaderBlock::COSMIC ||
                 runType[ism-1] == EcalDCCHeaderBlock::MTCC ||
                 runType[ism-1] == EcalDCCHeaderBlock::COSMICS_GLOBAL ||
                 runType[ism-1] == EcalDCCHeaderBlock::PHYSICS_GLOBAL ||
                 runType[ism-1] == EcalDCCHeaderBlock::COSMICS_LOCAL ||
                 runType[ism-1] == EcalDCCHeaderBlock::PHYSICS_LOCAL ) ) continue;
      }

      float xval = hitItr->energy();

      if ( meSelMap_[ism-1] ) meSelMap_[ism-1]->Fill(xie, xip, xval);
      if ( meSpectrum_[0][ism-1] ) meSpectrum_[0][ism-1]->Fill(xval);
      if(meSpectrumAll_) meSpectrumAll_->Fill(xval);

      // look for the seeds
      float e3x3 = 0.;
      bool isSeed = true;

      EcalRecHitCollection::const_iterator cItr;
      // evaluate 3x3 matrix around a seed
      for(int icry=0; icry<9; ++icry) {
        unsigned int row    = icry/3;
        unsigned int column = icry%3;
        int icryEta = id.ieta()+column-1;
        int icryPhi = id.iphi()+row-1;
        if ( EBDetId::validDetId(icryEta, icryPhi) ) {
          EBDetId id3x3 = EBDetId(icryEta, icryPhi, EBDetId::ETAPHIMODE);
          if ( (cItr = hits->find(id3x3)) != hits->end() && !cItr->checkFlagMask(mask)) {
            float neighbourEnergy = cItr->energy();
            e3x3 += neighbourEnergy;
            if ( neighbourEnergy > xval ) isSeed = false;
          }
        }
      }

      if ( isSeed && e3x3 >= threshold_)
        if ( meSpectrum_[1][ism-1] ) meSpectrum_[1][ism-1]->Fill(e3x3);

    }

  } else {

    edm::LogWarning("EBCosmicTask") << EcalRecHitCollection_ << " not available";

  }

}

