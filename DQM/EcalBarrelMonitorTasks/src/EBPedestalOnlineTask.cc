/*
 * \file EBPedestalOnlineTask.cc
 *
 * $Date: 2008/04/08 15:35:12 $
 * $Revision: 1.39 $
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

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

#include <DQM/EcalCommon/interface/Numbers.h>

#include <DQM/EcalBarrelMonitorTasks/interface/EBPedestalOnlineTask.h>

using namespace cms;
using namespace edm;
using namespace std;

EBPedestalOnlineTask::EBPedestalOnlineTask(const ParameterSet& ps){

  init_ = false;

  dqmStore_ = Service<DQMStore>().operator->();

  prefixME_ = ps.getUntrackedParameter<string>("prefixME", "");

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  mergeRuns_ = ps.getUntrackedParameter<bool>("mergeRuns", false);

  EBDigiCollection_ = ps.getParameter<edm::InputTag>("EBDigiCollection");

  for (int i = 0; i < 36; i++) {
    mePedMapG12_[i] = 0;
  }

}

EBPedestalOnlineTask::~EBPedestalOnlineTask(){

}

void EBPedestalOnlineTask::beginJob(const EventSetup& c){

  ievt_ = 0;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EBPedestalOnlineTask");
    dqmStore_->rmdir(prefixME_ + "/EBPedestalOnlineTask");
  }

  Numbers::initGeometry(c, false);

}

void EBPedestalOnlineTask::beginRun(const Run& r, const EventSetup& c) {

  if ( ! mergeRuns_ ) this->reset();

}

void EBPedestalOnlineTask::endRun(const Run& r, const EventSetup& c) {

}

void EBPedestalOnlineTask::reset(void) {

  for (int i = 0; i < 36; i++) {
    if ( mePedMapG12_[i] ) mePedMapG12_[i]->Reset();
  }

}

void EBPedestalOnlineTask::setup(void){

  init_ = true;

  char histo[200];

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EBPedestalOnlineTask");

    dqmStore_->setCurrentFolder(prefixME_ + "/EBPedestalOnlineTask/Gain12");
    for (int i = 0; i < 36; i++) {
      sprintf(histo, "EBPOT pedestal %s G12", Numbers::sEB(i+1).c_str());
      mePedMapG12_[i] = dqmStore_->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096., "s");
      mePedMapG12_[i]->setAxisTitle("ieta", 1);
      mePedMapG12_[i]->setAxisTitle("iphi", 2);
      dqmStore_->tag(mePedMapG12_[i], i+1);
    }

  }

}

void EBPedestalOnlineTask::cleanup(void){

  if ( ! init_ ) return;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EBPedestalOnlineTask");

    dqmStore_->setCurrentFolder(prefixME_ + "/EBPedestalOnlineTask/Gain12");
    for ( int i = 0; i < 36; i++ ) {
      if ( mePedMapG12_[i] ) dqmStore_->removeElement( mePedMapG12_[i]->getName() );
      mePedMapG12_[i] = 0;
    }

  }

  init_ = false;

}

void EBPedestalOnlineTask::endJob(void){

  LogInfo("EBPedestalOnlineTask") << "analyzed " << ievt_ << " events";

  if ( enableCleanup_ ) this->cleanup();

}

void EBPedestalOnlineTask::analyze(const Event& e, const EventSetup& c){

  if ( ! init_ ) this->setup();

  ievt_++;

  Handle<EBDigiCollection> digis;

  if ( e.getByLabel(EBDigiCollection_, digis) ) {

    int nebd = digis->size();
    LogDebug("EBPedestalOnlineTask") << "event " << ievt_ << " digi collection size " << nebd;

    for ( EBDigiCollection::const_iterator digiItr = digis->begin(); digiItr != digis->end(); ++digiItr ) {

      EBDataFrame dataframe = (*digiItr);
      EBDetId id = dataframe.id();

      int ic = id.ic();
      int ie = (ic-1)/20 + 1;
      int ip = (ic-1)%20 + 1;

      int ism = Numbers::iSM( id );

      float xie = ie - 0.5;
      float xip = ip - 0.5;

      LogDebug("EBPedestalOnlineTask") << " det id = " << id;
      LogDebug("EBPedestalOnlineTask") << " sm, ieta, iphi " << ism << " " << ie << " " << ip;

      for (int i = 0; i < 3; i++) {

        EcalMGPASample sample = dataframe.sample(i);
        int adc = sample.adc();

        MonitorElement* mePedMap = 0;

        if ( sample.gainId() == 1 ) mePedMap = mePedMapG12_[ism-1];
        if ( sample.gainId() == 2 ) mePedMap = 0;
        if ( sample.gainId() == 3 ) mePedMap = 0;

        float xval = float(adc);

        if ( mePedMap ) mePedMap->Fill(xie, xip, xval);

      }

    }

  } else {

    LogWarning("EBPedestalOnlineTask") << EBDigiCollection_ << " not available";

  }

}

