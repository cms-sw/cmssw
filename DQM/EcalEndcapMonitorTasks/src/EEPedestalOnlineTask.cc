/*
 * \file EEPedestalOnlineTask.cc
 *
 * $Date: 2008/04/08 15:32:10 $
 * $Revision: 1.24 $
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

#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

#include <DQM/EcalCommon/interface/Numbers.h>

#include <DQM/EcalEndcapMonitorTasks/interface/EEPedestalOnlineTask.h>

using namespace cms;
using namespace edm;
using namespace std;

EEPedestalOnlineTask::EEPedestalOnlineTask(const ParameterSet& ps){

  init_ = false;

  dqmStore_ = Service<DQMStore>().operator->();

  prefixME_ = ps.getUntrackedParameter<string>("prefixME", "");

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  EEDigiCollection_ = ps.getParameter<edm::InputTag>("EEDigiCollection");

  for (int i = 0; i < 18; i++) {
    mePedMapG12_[i] = 0;
  }

}

EEPedestalOnlineTask::~EEPedestalOnlineTask(){

}

void EEPedestalOnlineTask::beginJob(const EventSetup& c){

  ievt_ = 0;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EEPedestalOnlineTask");
    dqmStore_->rmdir(prefixME_ + "/EEPedestalOnlineTask");
  }

  Numbers::initGeometry(c, false);

}

void EEPedestalOnlineTask::setup(void){

  init_ = true;

  char histo[200];

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EEPedestalOnlineTask");

    dqmStore_->setCurrentFolder(prefixME_ + "/EEPedestalOnlineTask/Gain12");
    for (int i = 0; i < 18; i++) {
      sprintf(histo, "EEPOT pedestal %s G12", Numbers::sEE(i+1).c_str());
      mePedMapG12_[i] = dqmStore_->bookProfile2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 4096, 0., 4096., "s");
      mePedMapG12_[i]->setAxisTitle("jx", 1);
      mePedMapG12_[i]->setAxisTitle("jy", 2);
      dqmStore_->tag(mePedMapG12_[i], i+1);
    }

  }

}

void EEPedestalOnlineTask::cleanup(void){

  if ( ! enableCleanup_ ) return;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EEPedestalOnlineTask");

    dqmStore_->setCurrentFolder(prefixME_ + "/EEPedestalOnlineTask/Gain12");
    for ( int i = 0; i < 18; i++ ) {
      if ( mePedMapG12_[i] ) dqmStore_->removeElement( mePedMapG12_[i]->getName() );
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

  Handle<EEDigiCollection> digis;

  if ( e.getByLabel(EEDigiCollection_, digis) ) {

    int need = digis->size();
    LogDebug("EEPedestalOnlineTask") << "event " << ievt_ << " digi collection size " << need;

    for ( EEDigiCollection::const_iterator digiItr = digis->begin(); digiItr != digis->end(); ++digiItr ) {

      EEDataFrame dataframe = (*digiItr);
      EEDetId id = dataframe.id();

      int ix = id.ix();
      int iy = id.iy();

      int ism = Numbers::iSM( id );

      if ( ism >= 1 && ism <= 9 ) ix = 101 - ix;

      float xix = ix - 0.5;
      float xiy = iy - 0.5;

      LogDebug("EEPedestalOnlineTask") << " det id = " << id;
      LogDebug("EEPedestalOnlineTask") << " sm, ix, iy " << ism << " " << ix << " " << iy;

      for (int i = 0; i < 3; i++) {

        EcalMGPASample sample = dataframe.sample(i);
        int adc = sample.adc();

        MonitorElement* mePedMap = 0;

        if ( sample.gainId() == 1 ) mePedMap = mePedMapG12_[ism-1];
        if ( sample.gainId() == 2 ) mePedMap = 0;
        if ( sample.gainId() == 3 ) mePedMap = 0;

        float xval = float(adc);

        if ( mePedMap ) mePedMap->Fill(xix, xiy, xval);

      }

    }

  } else {

    LogWarning("EEPedestalOnlineTask") << EEDigiCollection_ << " not available";

  }

}

