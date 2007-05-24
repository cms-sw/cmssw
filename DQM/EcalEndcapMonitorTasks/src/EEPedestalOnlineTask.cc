/*
 * \file EEPedestalOnlineTask.cc
 *
 * $Date: 2007/05/21 09:57:46 $
 * $Revision: 1.8 $
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
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include <DQM/EcalCommon/interface/Numbers.h>

#include <DQM/EcalEndcapMonitorTasks/interface/EEPedestalOnlineTask.h>

using namespace cms;
using namespace edm;
using namespace std;

EEPedestalOnlineTask::EEPedestalOnlineTask(const ParameterSet& ps){

  Numbers::maxSM = 18;

  init_ = false;

  // get hold of back-end interface
  dbe_ = Service<DaqMonitorBEInterface>().operator->();

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", true);

  EBDigiCollection_ = ps.getParameter<edm::InputTag>("EBDigiCollection");

  for (int i = 0; i < 18 ; i++) {
    mePedMapG12_[i] = 0;
  }

}

EEPedestalOnlineTask::~EEPedestalOnlineTask(){

}

void EEPedestalOnlineTask::beginJob(const EventSetup& c){

  ievt_ = 0;

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalEndcap/EEPedestalOnlineTask");
    dbe_->rmdir("EcalEndcap/EEPedestalOnlineTask");
  }

}

void EEPedestalOnlineTask::setup(void){

  init_ = true;

  Char_t histo[200];

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalEndcap/EEPedestalOnlineTask");

    dbe_->setCurrentFolder("EcalEndcap/EEPedestalOnlineTask/Gain12");
    for (int i = 0; i < 18 ; i++) {
      sprintf(histo, "EEPOT pedestal %s G12", Numbers::sEE(i+1).c_str());
      mePedMapG12_[i] = dbe_->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096., "s");
      dbe_->tag(mePedMapG12_[i], i+1);
    }

  }

}

void EEPedestalOnlineTask::cleanup(void){

  if ( ! enableCleanup_ ) return;

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalEndcap/EEPedestalOnlineTask");

    dbe_->setCurrentFolder("EcalEndcap/EEPedestalOnlineTask/Gain12");
    for ( int i = 0; i < 18; i++ ) {
      if ( mePedMapG12_[i] ) dbe_->removeElement( mePedMapG12_[i]->getName() );
      mePedMapG12_[i] = 0;
    }

  }

  init_ = false;

}

void EEPedestalOnlineTask::endJob(void){

  LogInfo("EEPedestalOnlineTask") << "analyzed " << ievt_ << " events";

  if ( init_ ) this->cleanup();

}

void EEPedestalOnlineTask::analyze(const Event& e, const EventSetup& c){

  if ( ! init_ ) this->setup();

  ievt_++;

  try {

    Handle<EBDigiCollection> digis;
    e.getByLabel(EBDigiCollection_, digis);

    int nebd = digis->size();
    LogDebug("EEPedestalOnlineTask") << "event " << ievt_ << " digi collection size " << nebd;

    for ( EBDigiCollection::const_iterator digiItr = digis->begin(); digiItr != digis->end(); ++digiItr ) {

      EBDataFrame dataframe = (*digiItr);
      EBDetId id = dataframe.id();

      int ic = id.ic();
      int ie = (ic-1)/20 + 1;
      int ip = (ic-1)%20 + 1;

      int ism = Numbers::iSM( id ); if ( ism > 18 ) continue;

      float xie = ie - 0.5;
      float xip = ip - 0.5;

      LogDebug("EEPedestalOnlineTask") << " det id = " << id;
      LogDebug("EEPedestalOnlineTask") << " sm, eta, phi " << ism << " " << ie << " " << ip;

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

  } catch ( exception& ex) {

    LogWarning("EEPedestalOnlineTask") << EBDigiCollection_ << " not available";

  }

}

