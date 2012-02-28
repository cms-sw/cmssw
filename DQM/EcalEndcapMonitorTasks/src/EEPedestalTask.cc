/*
 * \file EEPedestalTask.cc
 *
 * $Date: 2011/08/30 09:28:42 $
 * $Revision: 1.56 $
 * \author G. Della Ricca
 *
*/

#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/MonitorElement.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

#include "DQM/EcalCommon/interface/Numbers.h"

#include "DQM/EcalEndcapMonitorTasks/interface/EEPedestalTask.h"

// #define COMMON_NOISE_ANALYSIS

EEPedestalTask::EEPedestalTask(const edm::ParameterSet& ps){

  init_ = false;

  dqmStore_ = edm::Service<DQMStore>().operator->();

  prefixME_ = ps.getUntrackedParameter<std::string>("prefixME", "");

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  mergeRuns_ = ps.getUntrackedParameter<bool>("mergeRuns", false);

  EcalRawDataCollection_ = ps.getParameter<edm::InputTag>("EcalRawDataCollection");
  EEDigiCollection_ = ps.getParameter<edm::InputTag>("EEDigiCollection");
  EcalPnDiodeDigiCollection_ = ps.getParameter<edm::InputTag>("EcalPnDiodeDigiCollection");

  for ( unsigned int i = 1; i <= 3; i++ ) MGPAGains_.push_back(i);
  MGPAGains_ = ps.getUntrackedParameter<std::vector<int> >("MGPAGains", MGPAGains_);

  for ( unsigned int i = 1; i <= 3; i++ ) MGPAGainsPN_.push_back(i);
  MGPAGainsPN_ = ps.getUntrackedParameter<std::vector<int> >("MGPAGainsPN", MGPAGainsPN_);

  meOccupancy_[0] = 0;
  meOccupancy_[1] = 0;

  for (int i = 0; i < 18; i++) {
    mePedMapG01_[i] = 0;
    mePedMapG06_[i] = 0;
    mePedMapG12_[i] = 0;
    mePnPedMapG01_[i] = 0;
    mePnPedMapG16_[i] = 0;
  }

  ievt_ = 0;

}

EEPedestalTask::~EEPedestalTask(){

}

void EEPedestalTask::beginJob(void){

  ievt_ = 0;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/Pedestal");
    dqmStore_->rmdir(prefixME_ + "/Pedestal");
  }

}

void EEPedestalTask::beginRun(const edm::Run& r, const edm::EventSetup& c) {

  Numbers::initGeometry(c, false);

  if ( ! mergeRuns_ ) this->reset();

}

void EEPedestalTask::endRun(const edm::Run& r, const edm::EventSetup& c) {

}

void EEPedestalTask::reset(void) {

  for(int i(0); i < 2; i++)
    if(meOccupancy_[i]) meOccupancy_[i]->Reset();

  for (int i = 0; i < 18; i++) {
    if (find(MGPAGains_.begin(), MGPAGains_.end(), 1) != MGPAGains_.end() ) {
      if ( mePedMapG01_[i] ) mePedMapG01_[i]->Reset();
    }
    if (find(MGPAGains_.begin(), MGPAGains_.end(), 6) != MGPAGains_.end() ) {
      if ( mePedMapG06_[i] ) mePedMapG06_[i]->Reset();
    }
    if (find(MGPAGains_.begin(), MGPAGains_.end(), 12) != MGPAGains_.end() ) {
      if ( mePedMapG12_[i] ) mePedMapG12_[i]->Reset();
    }
    if (find(MGPAGainsPN_.begin(), MGPAGainsPN_.end(), 1) != MGPAGainsPN_.end() ) {
      if ( mePnPedMapG01_[i] ) mePnPedMapG01_[i]->Reset();
    }
    if (find(MGPAGainsPN_.begin(), MGPAGainsPN_.end(), 16) != MGPAGainsPN_.end() ) {
      if ( mePnPedMapG16_[i] ) mePnPedMapG16_[i]->Reset();
    }
  }

}

void EEPedestalTask::setup(void){

  init_ = true;

  std::string name;
  std::stringstream GainN, GN;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/Pedestal");

    std::string subdet[] = {"EE-", "EE+"};

    for(int i(0); i < 2; i++){
      name = "PedestalTask occupancy " + subdet[i];
      meOccupancy_[i] = dqmStore_->book2D(name, name, 20, 0., 100., 20, 0., 100.);
      meOccupancy_[i]->setAxisTitle("ix", 1);
      meOccupancy_[i]->setAxisTitle("iy", 2);
    }

    if (find(MGPAGains_.begin(), MGPAGains_.end(), 1) != MGPAGains_.end() ) {

      GainN.str("");
      GainN << "Gain" << std::setw(2) << std::setfill('0') << 1;
      GN.str("");
      GN << "G" << std::setw(2) << std::setfill('0') << 1;

      dqmStore_->setCurrentFolder(prefixME_ + "/Pedestal/" + GainN.str());
      for (int i = 0; i < 18; i++) {
	name = "PedestalTask pedestal " + GN.str() + " " + Numbers::sEE(i+1);
        mePedMapG01_[i] = dqmStore_->bookProfile2D(name, name, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 4096, 0., 4096., "s");
        mePedMapG01_[i]->setAxisTitle("ix", 1);
        if ( i+1 >= 1 && i+1 <= 9 ) mePedMapG01_[i]->setAxisTitle("101-ix", 1);
        mePedMapG01_[i]->setAxisTitle("iy", 2);
        dqmStore_->tag(mePedMapG01_[i], i+1);
      }

    }

    if (find(MGPAGains_.begin(), MGPAGains_.end(), 6) != MGPAGains_.end() ) {

      GainN.str("");
      GainN << "Gain" << std::setw(2) << std::setfill('0') << 6;
      GN.str("");
      GN << "G" << std::setw(2) << std::setfill('0') << 6;

      dqmStore_->setCurrentFolder(prefixME_ + "/Pedestal/" + GainN.str());
      for (int i = 0; i < 18; i++) {
	name = "PedestalTask pedestal " + GN.str() + " " + Numbers::sEE(i+1);
        mePedMapG06_[i] = dqmStore_->bookProfile2D(name, name, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 4096, 0., 4096., "s");
        mePedMapG06_[i]->setAxisTitle("ix", 1);
        if ( i+1 >= 1 && i+1 <= 9 ) mePedMapG06_[i]->setAxisTitle("101-ix", 1);
        mePedMapG06_[i]->setAxisTitle("iy", 2);
        dqmStore_->tag(mePedMapG06_[i], i+1);

      }

    }

    if (find(MGPAGains_.begin(), MGPAGains_.end(), 12) != MGPAGains_.end() ) {

      GainN.str("");
      GainN << "Gain" << std::setw(2) << std::setfill('0') << 12;
      GN.str("");
      GN << "G" << std::setw(2) << std::setfill('0') << 12;

      dqmStore_->setCurrentFolder(prefixME_ + "/Pedestal/" + GainN.str());
      for (int i = 0; i < 18; i++) {
	name = "PedestalTask pedestal " + GN.str() + " " + Numbers::sEE(i+1);
        mePedMapG12_[i] = dqmStore_->bookProfile2D(name, name, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 4096, 0., 4096., "s");
        mePedMapG12_[i]->setAxisTitle("ix", 1);
        if ( i+1 >= 1 && i+1 <= 9 ) mePedMapG12_[i]->setAxisTitle("101-ix", 1);
        mePedMapG12_[i]->setAxisTitle("iy", 2);
        dqmStore_->tag(mePedMapG12_[i], i+1);
      }

    }

    dqmStore_->setCurrentFolder(prefixME_ + "/Pedestal/PN");

    if (find(MGPAGainsPN_.begin(), MGPAGainsPN_.end(), 1) != MGPAGainsPN_.end() ) {

      GainN.str("");
      GainN << "Gain" << std::setw(2) << std::setfill('0') << 1;
      GN.str("");
      GN << "G" << std::setw(2) << std::setfill('0') << 1;

      dqmStore_->setCurrentFolder(prefixME_ + "/Pedestal/PN/" + GainN.str());
      for (int i = 0; i < 18; i++) {
	name = "PedestalTask PN pedestal " + GN.str() + " " + Numbers::sEE(i+1);
	mePnPedMapG01_[i] =  dqmStore_->bookProfile(name, name, 10, 0., 10., 4096, 0., 4096., "s");
        mePnPedMapG01_[i]->setAxisTitle("channel", 1);
        mePnPedMapG01_[i]->setAxisTitle("pedestal", 2);
        dqmStore_->tag(mePnPedMapG01_[i], i+1);
      }

    }

    if (find(MGPAGainsPN_.begin(), MGPAGainsPN_.end(), 16) != MGPAGainsPN_.end() ) {

      GainN.str("");
      GainN << "Gain" << std::setw(2) << std::setfill('0') << 16;
      GN.str("");
      GN << "G" << std::setw(2) << std::setfill('0') << 16;

      dqmStore_->setCurrentFolder(prefixME_ + "/Pedestal/PN/" + GainN.str());
      for (int i = 0; i < 18; i++) {
	name = "PedestalTask PN pedestal " + GN.str() + " " + Numbers::sEE(i+1);
	mePnPedMapG16_[i] =  dqmStore_->bookProfile(name, name, 10, 0., 10., 4096, 0., 4096., "s");
        mePnPedMapG16_[i]->setAxisTitle("channel", 1);
        mePnPedMapG16_[i]->setAxisTitle("pedestal", 2);
        dqmStore_->tag(mePnPedMapG16_[i], i+1);
      }

    }

  }

}

void EEPedestalTask::cleanup(void){

  if ( ! init_ ) return;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/Pedestal");

    if (find(MGPAGains_.begin(), MGPAGains_.end(), 1) != MGPAGains_.end() ) {

      dqmStore_->setCurrentFolder(prefixME_ + "/Pedestal/Gain01");
      for ( int i = 0; i < 18; i++ ) {
        if ( mePedMapG01_[i] ) dqmStore_->removeElement( mePedMapG01_[i]->getFullname() );
        mePedMapG01_[i] = 0;

      }

    }

    if (find(MGPAGains_.begin(), MGPAGains_.end(), 6) != MGPAGains_.end() ) {

      dqmStore_->setCurrentFolder(prefixME_ + "/Pedestal/Gain06");
      for ( int i = 0; i < 18; i++ ) {
        if ( mePedMapG06_[i] ) dqmStore_->removeElement( mePedMapG06_[i]->getFullname() );
        mePedMapG06_[i] = 0;

      }

    }

    if (find(MGPAGains_.begin(), MGPAGains_.end(), 12) != MGPAGains_.end() ) {

      dqmStore_->setCurrentFolder(prefixME_ + "/Pedestal/Gain12");
      for ( int i = 0; i < 18; i++ ) {
        if ( mePedMapG12_[i] ) dqmStore_->removeElement( mePedMapG12_[i]->getFullname() );
        mePedMapG12_[i] = 0;

      }

    }

    dqmStore_->setCurrentFolder(prefixME_ + "/Pedestal/PN");

    if (find(MGPAGainsPN_.begin(), MGPAGainsPN_.end(), 1) != MGPAGainsPN_.end() ) {

      dqmStore_->setCurrentFolder(prefixME_ + "/Pedestal/PN/Gain01");
      for ( int i = 0; i < 18; i++ ) {
        if ( mePnPedMapG01_[i]) dqmStore_->removeElement( mePnPedMapG01_[i]->getFullname() );
        mePnPedMapG01_[i] = 0;
      }

    }

    if (find(MGPAGainsPN_.begin(), MGPAGainsPN_.end(), 16) != MGPAGainsPN_.end() ) {

      dqmStore_->setCurrentFolder(prefixME_ + "/Pedestal/PN/Gain16");
      for ( int i = 0; i < 18; i++ ) {
        if ( mePnPedMapG16_[i]) dqmStore_->removeElement( mePnPedMapG16_[i]->getFullname() );
        mePnPedMapG16_[i] = 0;
      }

    }

  }

  init_ = false;

}

void EEPedestalTask::endJob(void){

  edm::LogInfo("EEPedestalTask") << "analyzed " << ievt_ << " events";

  if ( enableCleanup_ ) this->cleanup();

}

void EEPedestalTask::analyze(const edm::Event& e, const edm::EventSetup& c){

  bool enable = false;
  int runType[18];
  for (int i=0; i<18; i++) runType[i] = -1;

  edm::Handle<EcalRawDataCollection> dcchs;

  if ( e.getByLabel(EcalRawDataCollection_, dcchs) ) {

    for ( EcalRawDataCollection::const_iterator dcchItr = dcchs->begin(); dcchItr != dcchs->end(); ++dcchItr ) {

      if ( Numbers::subDet( *dcchItr ) != EcalEndcap ) continue;

      int ism = Numbers::iSM( *dcchItr, EcalEndcap );

      runType[ism-1] = dcchItr->getRunType();

      if ( dcchItr->getRunType() == EcalDCCHeaderBlock::PEDESTAL_STD ||
           dcchItr->getRunType() == EcalDCCHeaderBlock::PEDESTAL_GAP ) enable = true;

    }

  } else {

    edm::LogWarning("EEPedestalTask") << EcalRawDataCollection_ << " not available";

  }

  if ( ! enable ) return;

  if ( ! init_ ) this->setup();

  ievt_++;

  edm::Handle<EEDigiCollection> digis;

  if ( e.getByLabel(EEDigiCollection_, digis) ) {

    int need = digis->size();
    LogDebug("EEPedestalTask") << "event " << ievt_ << " digi collection size " << need;

    float xmap01[18][50][50];
    float xmap06[18][50][50];
    float xmap12[18][50][50];

    for ( int ism = 1; ism <= 18; ism++ ) {
      for ( int ix = 1; ix <= 50; ix++ ) {
        for ( int iy = 1; iy <= 50; iy++ ) {

          xmap01[ism-1][ix-1][iy-1] = 0.;
          xmap06[ism-1][ix-1][iy-1] = 0.;
          xmap12[ism-1][ix-1][iy-1] = 0.;

        }
      }
    }

    for ( EEDigiCollection::const_iterator digiItr = digis->begin(); digiItr != digis->end(); ++digiItr ) {

      EEDetId id = digiItr->id();

      int ix = id.ix();
      int iy = id.iy();

      int iz = id.zside() < 0 ? 0 : 1;

      if(meOccupancy_[iz]) meOccupancy_[iz]->Fill(ix - 0.5, iy - 0.5);

      int ism = Numbers::iSM( id );

      if ( ism >= 1 && ism <= 9 ) ix = 101 - ix;

      float xix = ix - 0.5;
      float xiy = iy - 0.5;

      if ( ! ( runType[ism-1] == EcalDCCHeaderBlock::PEDESTAL_STD ||
               runType[ism-1] == EcalDCCHeaderBlock::PEDESTAL_GAP ) ) continue;

      EEDataFrame dataframe = (*digiItr);

      for (int i = 0; i < 10; i++) {

        int adc = dataframe.sample(i).adc();

        MonitorElement* mePedMap = 0;

        if ( dataframe.sample(i).gainId() == 1 ) mePedMap = mePedMapG12_[ism-1];
        if ( dataframe.sample(i).gainId() == 2 ) mePedMap = mePedMapG06_[ism-1];
        if ( dataframe.sample(i).gainId() == 3 ) mePedMap = mePedMapG01_[ism-1];

        float xval = float(adc);

        if ( mePedMap ) mePedMap->Fill(xix, xiy, xval);

        if ( dataframe.sample(i).gainId() == 1 ) xmap12[ism-1][ix-1-Numbers::ix0EE(ism)][iy-1-Numbers::iy0EE(ism)] = xmap12[ism-1][ix-1-Numbers::ix0EE(ism)][iy-1-Numbers::iy0EE(ism)] + xval;
        if ( dataframe.sample(i).gainId() == 2 ) xmap06[ism-1][ix-1-Numbers::ix0EE(ism)][iy-1-Numbers::iy0EE(ism)] = xmap06[ism-1][ix-1-Numbers::ix0EE(ism)][iy-1-Numbers::iy0EE(ism)] + xval;
        if ( dataframe.sample(i).gainId() == 3 ) xmap01[ism-1][ix-1-Numbers::ix0EE(ism)][iy-1-Numbers::iy0EE(ism)] = xmap01[ism-1][ix-1-Numbers::ix0EE(ism)][iy-1-Numbers::iy0EE(ism)] + xval;

      }

      xmap12[ism-1][ix-1-Numbers::ix0EE(ism)][iy-1-Numbers::iy0EE(ism)]=xmap12[ism-1][ix-1-Numbers::ix0EE(ism)][iy-1-Numbers::iy0EE(ism)]/10.;
      xmap06[ism-1][ix-1-Numbers::ix0EE(ism)][iy-1-Numbers::iy0EE(ism)]=xmap06[ism-1][ix-1-Numbers::ix0EE(ism)][iy-1-Numbers::iy0EE(ism)]/10.;
      xmap01[ism-1][ix-1-Numbers::ix0EE(ism)][iy-1-Numbers::iy0EE(ism)]=xmap01[ism-1][ix-1-Numbers::ix0EE(ism)][iy-1-Numbers::iy0EE(ism)]/10.;

    }


  } else {

    edm::LogWarning("EEPedestalTask") << EEDigiCollection_ << " not available";

  }

  edm::Handle<EcalPnDiodeDigiCollection> pns;

  if ( e.getByLabel(EcalPnDiodeDigiCollection_, pns) ) {

    int nep = pns->size();
    LogDebug("EEPedestalTask") << "event " << ievt_ << " pns collection size " << nep;

    for ( EcalPnDiodeDigiCollection::const_iterator pnItr = pns->begin(); pnItr != pns->end(); ++pnItr ) {

      if ( Numbers::subDet( pnItr->id() ) != EcalEndcap ) continue;

      int ism = Numbers::iSM( pnItr->id() );

      int num = pnItr->id().iPnId();

      if ( ! ( runType[ism-1] == EcalDCCHeaderBlock::PEDESTAL_STD ||
               runType[ism-1] == EcalDCCHeaderBlock::PEDESTAL_GAP ) ) continue;

      for (int i = 0; i < 50; i++) {

        int adc = pnItr->sample(i).adc();

        MonitorElement* mePNPed = 0;

        if ( pnItr->sample(i).gainId() == 0 ) mePNPed = mePnPedMapG01_[ism-1];
        if ( pnItr->sample(i).gainId() == 1 ) mePNPed = mePnPedMapG16_[ism-1];

        float xval = float(adc);

        if ( mePNPed ) mePNPed->Fill(num - 0.5, xval);

      }

    }

  } else {

    edm::LogWarning("EEPedestalTask") << EcalPnDiodeDigiCollection_ << " not available";

  }

}

