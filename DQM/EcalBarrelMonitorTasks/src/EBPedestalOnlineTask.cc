/*
 * \file EBPedestalOnlineTask.cc
 *
 * $Date: 2012/06/28 12:14:29 $
 * $Revision: 1.51 $
 * \author G. Della Ricca
 *
*/

#include <iostream>
#include <fstream>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/MonitorElement.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

#include "DQM/EcalCommon/interface/Numbers.h"

#include "DQM/EcalBarrelMonitorTasks/interface/EBPedestalOnlineTask.h"

EBPedestalOnlineTask::EBPedestalOnlineTask(const edm::ParameterSet& ps){

  init_ = false;

  dqmStore_ = edm::Service<DQMStore>().operator->();

  prefixME_ = ps.getUntrackedParameter<std::string>("prefixME", "");

  subfolder_ = ps.getUntrackedParameter<std::string>("subfolder", "");

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  mergeRuns_ = ps.getUntrackedParameter<bool>("mergeRuns", false);

  EBDigiCollection_ = ps.getParameter<edm::InputTag>("EBDigiCollection");

  for (int i = 0; i < 36; i++) {
    mePedMapG12_[i] = 0;
  }

}

EBPedestalOnlineTask::~EBPedestalOnlineTask(){

}

void EBPedestalOnlineTask::beginJob(void){

  ievt_ = 0;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EBPedestalOnlineTask");
    dqmStore_->rmdir(prefixME_ + "/EBPedestalOnlineTask");
  }

}

void EBPedestalOnlineTask::beginRun(const edm::Run& r, const edm::EventSetup& c) {

  Numbers::initGeometry(c, false);

  if ( ! mergeRuns_ ) this->reset();

}

void EBPedestalOnlineTask::endRun(const edm::Run& r, const edm::EventSetup& c) {

}

void EBPedestalOnlineTask::reset(void) {

  for (int i = 0; i < 36; i++) {
    if ( mePedMapG12_[i] ) mePedMapG12_[i]->Reset();
  }

}

void EBPedestalOnlineTask::setup(void){

  init_ = true;

  std::string name;
  std::string dir;

  if ( dqmStore_ ) {
    dir = prefixME_ + "/EBPedestalOnlineTask";
    if(subfolder_.size())
      dir += "/" + subfolder_;

    dqmStore_->setCurrentFolder(dir);

    dqmStore_->setCurrentFolder(dir + "/Gain12");
    for (int i = 0; i < 36; i++) {
      name = "EBPOT pedestal " + Numbers::sEB(i+1) + " G12";
      mePedMapG12_[i] = dqmStore_->bookProfile2D(name, name, 85, 0., 85., 20, 0., 20., 4096, 0., 4096., "s");
      mePedMapG12_[i]->setAxisTitle("ieta", 1);
      mePedMapG12_[i]->setAxisTitle("iphi", 2);
      dqmStore_->tag(mePedMapG12_[i], i+1);
    }

  }

}

void EBPedestalOnlineTask::cleanup(void){

  if ( ! init_ ) return;

  if ( dqmStore_ ) {
    std::string dir = prefixME_ + "/EBPedestalOnlineTask";
    if(subfolder_.size())
      dir += "/" + subfolder_;

    dqmStore_->setCurrentFolder(dir);

    dqmStore_->setCurrentFolder(dir + "/Gain12");
    for ( int i = 0; i < 36; i++ ) {
      if ( mePedMapG12_[i] ) dqmStore_->removeElement( mePedMapG12_[i]->getName() );
      mePedMapG12_[i] = 0;
    }

  }

  init_ = false;

}

void EBPedestalOnlineTask::endJob(void){

  edm::LogInfo("EBPedestalOnlineTask") << "analyzed " << ievt_ << " events";

  if ( enableCleanup_ ) this->cleanup();

}

void EBPedestalOnlineTask::analyze(const edm::Event& e, const edm::EventSetup& c){

  if ( ! init_ ) this->setup();

  ievt_++;

  edm::Handle<EBDigiCollection> digis;

  if ( e.getByLabel(EBDigiCollection_, digis) ) {

    int nebd = digis->size();
    LogDebug("EBPedestalOnlineTask") << "event " << ievt_ << " digi collection size " << nebd;

    for ( EBDigiCollection::const_iterator digiItr = digis->begin(); digiItr != digis->end(); ++digiItr ) {

      EBDetId id = digiItr->id();

      int ic = id.ic();
      int ie = (ic-1)/20 + 1;
      int ip = (ic-1)%20 + 1;

      int ism = Numbers::iSM( id );

      float xie = ie - 0.5;
      float xip = ip - 0.5;

      EBDataFrame dataframe = (*digiItr);

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

        if ( mePedMap ) mePedMap->Fill(xie, xip, xval);

      }

    }

  } else {

    edm::LogWarning("EBPedestalOnlineTask") << EBDigiCollection_ << " not available";

  }

}

