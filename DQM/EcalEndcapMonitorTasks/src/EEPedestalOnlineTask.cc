/*
 * \file EEPedestalOnlineTask.cc
 *
 * $Date: 2012/06/28 12:14:30 $
 * $Revision: 1.38 $
 * \author G. Della Ricca
 *
*/

#include <iostream>
#include <fstream>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/MonitorElement.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

#include "DQM/EcalCommon/interface/Numbers.h"

#include "DQM/EcalEndcapMonitorTasks/interface/EEPedestalOnlineTask.h"

EEPedestalOnlineTask::EEPedestalOnlineTask(const edm::ParameterSet& ps){

  init_ = false;

  dqmStore_ = edm::Service<DQMStore>().operator->();

  prefixME_ = ps.getUntrackedParameter<std::string>("prefixME", "");

  subfolder_ = ps.getUntrackedParameter<std::string>("subfolder", "");

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  mergeRuns_ = ps.getUntrackedParameter<bool>("mergeRuns", false);

  EEDigiCollection_ = ps.getParameter<edm::InputTag>("EEDigiCollection");

  for (int i = 0; i < 18; i++) {
    mePedMapG12_[i] = 0;
  }

}

EEPedestalOnlineTask::~EEPedestalOnlineTask(){

}

void EEPedestalOnlineTask::beginJob(void){

  ievt_ = 0;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EEPedestalOnlineTask");
    dqmStore_->rmdir(prefixME_ + "/EEPedestalOnlineTask");
  }

}

void EEPedestalOnlineTask::beginRun(const edm::Run& r, const edm::EventSetup& c) {

  Numbers::initGeometry(c, false);

  if ( ! mergeRuns_ ) this->reset();

}

void EEPedestalOnlineTask::endRun(const edm::Run& r, const edm::EventSetup& c) {

}

void EEPedestalOnlineTask::reset(void) {

  for (int i = 0; i < 18; i++) {
    if ( mePedMapG12_[i] ) mePedMapG12_[i]->Reset();
  }

}

void EEPedestalOnlineTask::setup(void){

  init_ = true;

  std::string name;
  std::string dir;

  if ( dqmStore_ ) {
    dir = prefixME_ + "/EEPedestalOnlineTask";
    if(subfolder_.size())
      dir += "/" + subfolder_;

    dqmStore_->setCurrentFolder(dir);

    dqmStore_->setCurrentFolder(dir + "/Gain12");
    for (int i = 0; i < 18; i++) {
      name = "EEPOT pedestal " + Numbers::sEE(i+1) + " G12";
      mePedMapG12_[i] = dqmStore_->bookProfile2D(name, name, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 4096, 0., 4096., "s");
      mePedMapG12_[i]->setAxisTitle("ix", 1);
      if ( i+1 >= 1 && i+1 <= 9 ) mePedMapG12_[i]->setAxisTitle("101-ix", 1);
      mePedMapG12_[i]->setAxisTitle("iy", 2);
      dqmStore_->tag(mePedMapG12_[i], i+1);
    }

  }

}

void EEPedestalOnlineTask::cleanup(void){

  if ( ! init_ ) return;

  if ( dqmStore_ ) {
    std::string dir = prefixME_ + "/EEPedestalOnlineTask";
    if(subfolder_.size())
      dir += "/" + subfolder_;

    dqmStore_->setCurrentFolder(dir);

    dqmStore_->setCurrentFolder(dir + "/Gain12");
    for ( int i = 0; i < 18; i++ ) {
      if ( mePedMapG12_[i] ) dqmStore_->removeElement( mePedMapG12_[i]->getName() );
      mePedMapG12_[i] = 0;
    }

  }

  init_ = false;

}

void EEPedestalOnlineTask::endJob(void){

  edm::LogInfo("EEPedestalOnlineTask") << "analyzed " << ievt_ << " events";

  if ( enableCleanup_ ) this->cleanup();

}

void EEPedestalOnlineTask::analyze(const edm::Event& e, const edm::EventSetup& c){

  if ( ! init_ ) this->setup();

  ievt_++;

  edm::Handle<EEDigiCollection> digis;

  if ( e.getByLabel(EEDigiCollection_, digis) ) {

    int need = digis->size();
    LogDebug("EEPedestalOnlineTask") << "event " << ievt_ << " digi collection size " << need;

    for ( EEDigiCollection::const_iterator digiItr = digis->begin(); digiItr != digis->end(); ++digiItr ) {

      EEDetId id = digiItr->id();

      int ix = id.ix();
      int iy = id.iy();

      int ism = Numbers::iSM( id );

      if ( ism >= 1 && ism <= 9 ) ix = 101 - ix;

      float xix = ix - 0.5;
      float xiy = iy - 0.5;

      EEDataFrame dataframe = (*digiItr);

      int iMax(-1);
      int maxADC(0);
      for(int i(0); i < 10; i++){
        if(dataframe.sample(i).gainId() != 1) break;
        int adc(dataframe.sample(i).adc());
        if(adc > maxADC){
          maxADC = adc;
          iMax = i;
        }
      }

      if(iMax != 5) continue;

      for (int i = 0; i < 3; i++) {

        int adc = dataframe.sample(i).adc();

        MonitorElement* mePedMap = 0;

        if ( dataframe.sample(i).gainId() == 1 ) mePedMap = mePedMapG12_[ism-1];
        if ( dataframe.sample(i).gainId() == 2 ) mePedMap = 0;
        if ( dataframe.sample(i).gainId() == 3 ) mePedMap = 0;

        float xval = float(adc);

        if ( mePedMap ) mePedMap->Fill(xix, xiy, xval);

      }

    }

  } else {

    edm::LogWarning("EEPedestalOnlineTask") << EEDigiCollection_ << " not available";

  }

}

