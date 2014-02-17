/*
 * \file EBPedestalTask.cc
 *
 * $Date: 2012/04/27 13:46:02 $
 * $Revision: 1.105 $
 * \author G. Della Ricca
 *
*/

#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <algorithm>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/MonitorElement.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

#include "DQM/EcalCommon/interface/Numbers.h"

#include "DQM/EcalBarrelMonitorTasks/interface/EBPedestalTask.h"

// #define COMMON_NOISE_ANALYSIS

EBPedestalTask::EBPedestalTask(const edm::ParameterSet& ps){

  init_ = false;

  dqmStore_ = edm::Service<DQMStore>().operator->();

  prefixME_ = ps.getUntrackedParameter<std::string>("prefixME", "");

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  mergeRuns_ = ps.getUntrackedParameter<bool>("mergeRuns", false);

  EcalRawDataCollection_ = ps.getParameter<edm::InputTag>("EcalRawDataCollection");
  EBDigiCollection_ = ps.getParameter<edm::InputTag>("EBDigiCollection");
  EcalPnDiodeDigiCollection_ = ps.getParameter<edm::InputTag>("EcalPnDiodeDigiCollection");

  MGPAGains_.reserve(3);
  for ( unsigned int i = 1; i <= 3; i++ ) MGPAGains_.push_back(i);
  MGPAGains_ = ps.getUntrackedParameter<std::vector<int> >("MGPAGains", MGPAGains_);

  MGPAGainsPN_.reserve(2);
  for ( unsigned int i = 1; i <= 3; i++ ) MGPAGainsPN_.push_back(i);
  MGPAGainsPN_ = ps.getUntrackedParameter<std::vector<int> >("MGPAGainsPN", MGPAGainsPN_);

  for (int i = 0; i < 36; i++) {
    mePedMapG01_[i] = 0;
    mePedMapG06_[i] = 0;
    mePedMapG12_[i] = 0;
#ifdef COMMON_NOISE_ANALYSIS
    mePed3SumMapG01_[i] = 0;
    mePed3SumMapG06_[i] = 0;
    mePed3SumMapG12_[i] = 0;
    mePed5SumMapG01_[i] = 0;
    mePed5SumMapG06_[i] = 0;
    mePed5SumMapG12_[i] = 0;
#endif
    mePnPedMapG01_[i] = 0;
    mePnPedMapG16_[i] = 0;
  }

}

EBPedestalTask::~EBPedestalTask(){

}

void EBPedestalTask::beginJob(void){

  ievt_ = 0;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EBPedestalTask");
    dqmStore_->rmdir(prefixME_ + "/EBPedestalTask");
  }

}

void EBPedestalTask::beginRun(const edm::Run& r, const edm::EventSetup& c) {

  Numbers::initGeometry(c, false);

  if ( ! mergeRuns_ ) this->reset();

}

void EBPedestalTask::endRun(const edm::Run& r, const edm::EventSetup& c) {

}

void EBPedestalTask::reset(void) {

  for (int i = 0; i < 36; i++) {
    if (find(MGPAGains_.begin(), MGPAGains_.end(), 1) != MGPAGains_.end() ) {
      if ( mePedMapG01_[i] ) mePedMapG01_[i]->Reset();
    }
    if (find(MGPAGains_.begin(), MGPAGains_.end(), 6) != MGPAGains_.end() ) {
      if ( mePedMapG06_[i] ) mePedMapG06_[i]->Reset();
    }
    if (find(MGPAGains_.begin(), MGPAGains_.end(), 12) != MGPAGains_.end() ) {
      if ( mePedMapG12_[i] ) mePedMapG12_[i]->Reset();
    }
#ifdef COMMON_NOISE_ANALYSIS
    if ( mePed3SumMapG01_[i] ) mePed3SumMapG01_[i]->Reset();
    if ( mePed3SumMapG06_[i] ) mePed3SumMapG06_[i]->Reset();
    if ( mePed3SumMapG12_[i] ) mePed3SumMapG12_[i]->Reset();
    if ( mePed5SumMapG01_[i] ) mePed5SumMapG01_[i]->Reset();
    if ( mePed5SumMapG06_[i] ) mePed5SumMapG06_[i]->Reset();
    if ( mePed5SumMapG12_[i] ) mePed5SumMapG12_[i]->Reset();
#endif
    if (find(MGPAGainsPN_.begin(), MGPAGainsPN_.end(), 1) != MGPAGainsPN_.end() ) {
      if ( mePnPedMapG01_[i] ) mePnPedMapG01_[i]->Reset();
    }
    if (find(MGPAGainsPN_.begin(), MGPAGainsPN_.end(), 12) != MGPAGainsPN_.end() ) {
      if ( mePnPedMapG16_[i] ) mePnPedMapG16_[i]->Reset();
    }
  }

}

void EBPedestalTask::setup(void){

  init_ = true;

  std::string name;
  std::stringstream GainN, GN;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EBPedestalTask");

    if (find(MGPAGains_.begin(), MGPAGains_.end(), 1) != MGPAGains_.end() ) {

      GainN.str("");
      GainN << "Gain" << std::setw(2) << std::setfill('0') << 1;
      GN.str("");
      GN << "G" << std::setw(2) << std::setfill('0') << 1;

      dqmStore_->setCurrentFolder(prefixME_ + "/EBPedestalTask/" + GainN.str());
      for (int i = 0; i < 36; i++) {
        name = "EBPT pedestal " + Numbers::sEB(i+1) + " " + GN.str(); 
        mePedMapG01_[i] = dqmStore_->bookProfile2D(name, name, 85, 0., 85., 20, 0., 20., 4096, 0., 4096., "s");
        mePedMapG01_[i]->setAxisTitle("ieta", 1);
        mePedMapG01_[i]->setAxisTitle("iphi", 2);
        dqmStore_->tag(mePedMapG01_[i], i+1);
#ifdef COMMON_NOISE_ANALYSIS
	name = "EBPT pedestal 3sum " + Numbers::sEB(i+1) + " " + GN.str(); 
        mePed3SumMapG01_[i] = dqmStore_->bookProfile2D(name, name, 85, 0., 85., 20, 0., 20., 4096, 0., 4096., "s");
        mePed3SumMapG01_[i]->setAxisTitle("ieta", 1);
        mePed3SumMapG01_[i]->setAxisTitle("iphi", 2);
        dqmStore_->tag(mePed3SumMapG01_[i], i+1);
	name = "EBPT pedestal 5sum " + Numbers::sEB(i+1) + " " + GN.str(); 
        mePed5SumMapG01_[i] = dqmStore_->bookProfile2D(name, name, 85, 0., 85., 20, 0., 20., 4096, 0., 4096., "s");
        mePed5SumMapG01_[i]->setAxisTitle("ieta", 1);
        mePed5SumMapG01_[i]->setAxisTitle("iphi", 2);
        dqmStore_->tag(mePed5SumMapG01_[i], i+1);
#endif
      }

    }

    if (find(MGPAGains_.begin(), MGPAGains_.end(), 6) != MGPAGains_.end() ) {

      GainN.str("");
      GainN << "Gain" << std::setw(2) << std::setfill('0') << 6;
      GN.str("");
      GN << "G" << std::setw(2) << std::setfill('0') << 6;

      dqmStore_->setCurrentFolder(prefixME_ + "/EBPedestalTask/" + GainN.str());
      for (int i = 0; i < 36; i++) {
        name = "EBPT pedestal " + Numbers::sEB(i+1) + " " + GN.str(); 
        mePedMapG06_[i] = dqmStore_->bookProfile2D(name, name, 85, 0., 85., 20, 0., 20., 4096, 0., 4096., "s");
        mePedMapG06_[i]->setAxisTitle("ieta", 1);
        mePedMapG06_[i]->setAxisTitle("iphi", 2);
        dqmStore_->tag(mePedMapG06_[i], i+1);
#ifdef COMMON_NOISE_ANALYSIS
	name = "EBPT pedestal 3sum " + Numbers::sEB(i+1) + " " + GN.str(); 
        mePed3SumMapG06_[i] = dqmStore_->bookProfile2D(name, name, 85, 0., 85., 20, 0., 20., 4096, 0., 4096., "s");
        mePed3SumMapG06_[i]->setAxisTitle("ieta", 1);
        mePed3SumMapG06_[i]->setAxisTitle("iphi", 2);
        dqmStore_->tag(mePed3SumMapG06_[i], i+1);
	name = "EBPT pedestal 5sum " + Numbers::sEB(i+1) + " " + GN.str(); 
        mePed5SumMapG06_[i] = dqmStore_->bookProfile2D(name, name, 85, 0., 85., 20, 0., 20., 4096, 0., 4096., "s");
        mePed5SumMapG06_[i]->setAxisTitle("ieta", 1);
        mePed5SumMapG06_[i]->setAxisTitle("iphi", 2);
        dqmStore_->tag(mePed5SumMapG06_[i], i+1);
#endif
      }

    }

    if (find(MGPAGains_.begin(), MGPAGains_.end(), 12) != MGPAGains_.end() ) {

      GainN.str("");
      GainN << "Gain" << std::setw(2) << std::setfill('0') << 12;
      GN.str("");
      GN << "G" << std::setw(2) << std::setfill('0') << 12;

      dqmStore_->setCurrentFolder(prefixME_ + "/EBPedestalTask/" + GainN.str());
      for (int i = 0; i < 36; i++) {
        name = "EBPT pedestal " + Numbers::sEB(i+1) + " " + GN.str(); 
        mePedMapG12_[i] = dqmStore_->bookProfile2D(name, name, 85, 0., 85., 20, 0., 20., 4096, 0., 4096., "s");
        mePedMapG12_[i]->setAxisTitle("ieta", 1);
        mePedMapG12_[i]->setAxisTitle("iphi", 2);
        dqmStore_->tag(mePedMapG12_[i], i+1);
#ifdef COMMON_NOISE_ANALYSIS
	name = "EBPT pedestal 3sum " + Numbers::sEB(i+1) + " " + GN.str(); 
        mePed3SumMapG12_[i] = dqmStore_->bookProfile2D(name, name, 85, 0., 85., 20, 0., 20., 4096, 0., 4096., "s");
        mePed3SumMapG12_[i]->setAxisTitle("ieta", 1);
        mePed3SumMapG12_[i]->setAxisTitle("iphi", 2);
        dqmStore_->tag(mePed3SumMapG12_[i], i+1);
	name = "EBPT pedestal 5sum " + Numbers::sEB(i+1) + " " + GN.str(); 
        mePed5SumMapG12_[i] = dqmStore_->bookProfile2D(name, name, 85, 0., 85., 20, 0., 20., 4096, 0., 4096., "s");
        mePed5SumMapG12_[i]->setAxisTitle("ieta", 1);
        mePed5SumMapG12_[i]->setAxisTitle("iphi", 2);
        dqmStore_->tag(mePed5SumMapG12_[i], i+1);
#endif
      }
    }


    dqmStore_->setCurrentFolder(prefixME_ + "/EBPedestalTask/PN");

    if (find(MGPAGainsPN_.begin(), MGPAGainsPN_.end(), 1) != MGPAGainsPN_.end() ) {

      GainN.str("");
      GainN << "Gain" << std::setw(2) << std::setfill('0') << 1;
      GN.str("");
      GN << "G" << std::setw(2) << std::setfill('0') << 1;

      dqmStore_->setCurrentFolder(prefixME_ + "/EBPedestalTask/PN/" + GainN.str());
      for (int i = 0; i < 36; i++) {
	name = "EBPDT PNs pedestal " + Numbers::sEB(i+1) + " " + GN.str(); 
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

      dqmStore_->setCurrentFolder(prefixME_ + "/EBPedestalTask/PN/" + GainN.str());
      for (int i = 0; i < 36; i++) {
	name = "EBPDT PNs pedestal " + Numbers::sEB(i+1) + " " + GN.str(); 
        mePnPedMapG16_[i] =  dqmStore_->bookProfile(name, name, 10, 0., 10., 4096, 0., 4096., "s");
        mePnPedMapG16_[i]->setAxisTitle("channel", 1);
        mePnPedMapG16_[i]->setAxisTitle("pedestal", 2);
        dqmStore_->tag(mePnPedMapG16_[i], i+1);
      }

    }

  }

}

void EBPedestalTask::cleanup(void){

  if ( ! init_ ) return;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EBPedestalTask");

    if (find(MGPAGains_.begin(), MGPAGains_.end(), 1) != MGPAGains_.end() ) {

      dqmStore_->setCurrentFolder(prefixME_ + "/EBPedestalTask/Gain01");
      for ( int i = 0; i < 36; i++ ) {
        if ( mePedMapG01_[i] ) dqmStore_->removeElement( mePedMapG01_[i]->getName() );
        mePedMapG01_[i] = 0;
#ifdef COMMON_NOISE_ANALYSIS
        if ( mePed3SumMapG01_[i] ) dqmStore_->removeElement( mePed3SumMapG01_[i]->getName() );
        mePed3SumMapG01_[i] = 0;
        if ( mePed5SumMapG01_[i] ) dqmStore_->removeElement( mePed5SumMapG01_[i]->getName() );
        mePed5SumMapG01_[i] = 0;
#endif
      }

    }

    if (find(MGPAGains_.begin(), MGPAGains_.end(), 6) != MGPAGains_.end() ) {

      dqmStore_->setCurrentFolder(prefixME_ + "/EBPedestalTask/Gain06");
      for ( int i = 0; i < 36; i++ ) {
        if ( mePedMapG06_[i] ) dqmStore_->removeElement( mePedMapG06_[i]->getName() );
        mePedMapG06_[i] = 0;
#ifdef COMMON_NOISE_ANALYSIS
        if ( mePed3SumMapG06_[i] ) dqmStore_->removeElement( mePed3SumMapG06_[i]->getName() );
        mePed3SumMapG06_[i] = 0;
        if ( mePed5SumMapG06_[i] ) dqmStore_->removeElement( mePed5SumMapG06_[i]->getName() );
        mePed5SumMapG06_[i] = 0;
#endif
      }

    }

    if (find(MGPAGains_.begin(), MGPAGains_.end(), 12) != MGPAGains_.end() ) {

      dqmStore_->setCurrentFolder(prefixME_ + "/EBPedestalTask/Gain12");
      for ( int i = 0; i < 36; i++ ) {
        if ( mePedMapG12_[i] ) dqmStore_->removeElement( mePedMapG12_[i]->getName() );
        mePedMapG12_[i] = 0;
#ifdef COMMON_NOISE_ANALYSIS
        if ( mePed3SumMapG12_[i] ) dqmStore_->removeElement( mePed3SumMapG12_[i]->getName() );
        mePed3SumMapG12_[i] = 0;
        if ( mePed5SumMapG12_[i] ) dqmStore_->removeElement( mePed5SumMapG12_[i]->getName() );
        mePed5SumMapG12_[i] = 0;
#endif
      }

    }

    dqmStore_->setCurrentFolder(prefixME_ + "/EBPedestalTask/PN");

    if (find(MGPAGainsPN_.begin(), MGPAGainsPN_.end(), 1) != MGPAGainsPN_.end() ) {

      dqmStore_->setCurrentFolder(prefixME_ + "/EBPedestalTask/PN/Gain01");
      for ( int i = 0; i < 36; i++ ) {
        if ( mePnPedMapG01_[i]) dqmStore_->removeElement( mePnPedMapG01_[i]->getName() );
        mePnPedMapG01_[i] = 0;
      }

    }

    if (find(MGPAGains_.begin(), MGPAGains_.end(), 16) != MGPAGains_.end() ) {

      dqmStore_->setCurrentFolder(prefixME_ + "/EBPedestalTask/PN/Gain16");
      for ( int i = 0; i < 36; i++ ) {
        if ( mePnPedMapG16_[i]) dqmStore_->removeElement( mePnPedMapG16_[i]->getName() );
        mePnPedMapG16_[i] = 0;
      }

    }

  }

  init_ = false;

}

void EBPedestalTask::endJob(void){

  edm::LogInfo("EBPedestalTask") << "analyzed " << ievt_ << " events";

  if ( enableCleanup_ ) this->cleanup();

}

void EBPedestalTask::analyze(const edm::Event& e, const edm::EventSetup& c){

  bool enable = false;
  int runType[36];
  for (int i=0; i<36; i++) runType[i] = -1;

  edm::Handle<EcalRawDataCollection> dcchs;

  if ( e.getByLabel(EcalRawDataCollection_, dcchs) ) {

    for ( EcalRawDataCollection::const_iterator dcchItr = dcchs->begin(); dcchItr != dcchs->end(); ++dcchItr ) {

      if ( Numbers::subDet( *dcchItr ) != EcalBarrel ) continue;

      int ism = Numbers::iSM( *dcchItr, EcalBarrel );

      runType[ism-1] = dcchItr->getRunType();

      if ( dcchItr->getRunType() == EcalDCCHeaderBlock::PEDESTAL_STD ||
           dcchItr->getRunType() == EcalDCCHeaderBlock::PEDESTAL_GAP ) enable = true;

    }

  } else {

    edm::LogWarning("EBPedestalTask") << EcalRawDataCollection_ << " not available";

  }

  if ( ! enable ) return;

  if ( ! init_ ) this->setup();

  ievt_++;

  edm::Handle<EBDigiCollection> digis;

  if ( e.getByLabel(EBDigiCollection_, digis) ) {

    int nebd = digis->size();
    LogDebug("EBPedestalTask") << "event " << ievt_ << " digi collection size " << nebd;

    float xmap01[36][85][20];
    float xmap06[36][85][20];
    float xmap12[36][85][20];

    for ( int ism = 1; ism <= 36; ism++ ) {
      for ( int ie = 1; ie <= 85; ie++ ) {
        for ( int ip = 1; ip <= 20; ip++ ) {

          xmap01[ism-1][ie-1][ip-1] = 0.;
          xmap06[ism-1][ie-1][ip-1] = 0.;
          xmap12[ism-1][ie-1][ip-1] = 0.;

        }
      }
    }

    for ( EBDigiCollection::const_iterator digiItr = digis->begin(); digiItr != digis->end(); ++digiItr ) {

      EBDetId id = digiItr->id();

      int ic = id.ic();
      int ie = (ic-1)/20 + 1;
      int ip = (ic-1)%20 + 1;

      int ism = Numbers::iSM( id );

      float xie = ie - 0.5;
      float xip = ip - 0.5;

      if ( ! ( runType[ism-1] == EcalDCCHeaderBlock::PEDESTAL_STD ||
               runType[ism-1] == EcalDCCHeaderBlock::PEDESTAL_GAP ) ) continue;

      EBDataFrame dataframe = (*digiItr);

      for (int i = 0; i < 10; i++) {

        int adc = dataframe.sample(i).adc();

        MonitorElement* mePedMap = 0;

        if ( dataframe.sample(i).gainId() == 1 ) mePedMap = mePedMapG12_[ism-1];
        if ( dataframe.sample(i).gainId() == 2 ) mePedMap = mePedMapG06_[ism-1];
        if ( dataframe.sample(i).gainId() == 3 ) mePedMap = mePedMapG01_[ism-1];

        float xval = float(adc);

        if ( mePedMap ) mePedMap->Fill(xie, xip, xval);

        if ( dataframe.sample(i).gainId() == 1 ) xmap12[ism-1][ie-1][ip-1] = xmap12[ism-1][ie-1][ip-1] + xval;
        if ( dataframe.sample(i).gainId() == 2 ) xmap06[ism-1][ie-1][ip-1] = xmap06[ism-1][ie-1][ip-1] + xval;
        if ( dataframe.sample(i).gainId() == 3 ) xmap01[ism-1][ie-1][ip-1] = xmap01[ism-1][ie-1][ip-1] + xval;

      }

      xmap12[ism-1][ie-1][ip-1]=xmap12[ism-1][ie-1][ip-1]/10.;
      xmap06[ism-1][ie-1][ip-1]=xmap06[ism-1][ie-1][ip-1]/10.;
      xmap01[ism-1][ie-1][ip-1]=xmap01[ism-1][ie-1][ip-1]/10.;

    }

    // to be re-done using the 3x3 & 5x5 Selectors (if faster)

#ifdef COMMON_NOISE_ANALYSIS
    for ( int ism = 1; ism <= 36; ism++ ) {
      for ( int ie = 1; ie <= 85; ie++ ) {
        for ( int ip = 1; ip <= 20; ip++ ) {

          float xie = ie - 0.5;
          float xip = ip - 0.5;

          float x3val01;
          float x3val06;
          float x3val12;

          if ( ie >= 2 && ie <= 84 && ip >= 2 && ip <= 19 ) {

            x3val01 = 0.;
            x3val06 = 0.;
            x3val12 = 0.;
            for ( int i = -1; i <= +1; i++ ) {
              for ( int j = -1; j <= +1; j++ ) {

                x3val01 = x3val01 + xmap01[ism-1][ie-1+i][ip-1+j];
                x3val06 = x3val06 + xmap06[ism-1][ie-1+i][ip-1+j];
                x3val12 = x3val12 + xmap12[ism-1][ie-1+i][ip-1+j];

              }
            }
            x3val01 = x3val01 / 9.;
            x3val06 = x3val06 / 9.;
            x3val12 = x3val12 / 9.;
            if ( mePed3SumMapG01_[ism-1] && x3val01 != 0. ) mePed3SumMapG01_[ism-1]->Fill(xie, xip, x3val01);
            if ( mePed3SumMapG06_[ism-1] && x3val06 != 0. ) mePed3SumMapG06_[ism-1]->Fill(xie, xip, x3val06);
            if ( mePed3SumMapG12_[ism-1] && x3val12 != 0. ) mePed3SumMapG12_[ism-1]->Fill(xie, xip, x3val12);

          }

          float x5val01;
          float x5val06;
          float x5val12;

          if ( ie >= 3 && ie <= 83 && ip >= 3 && ip <= 18 ) {

            x5val01 = 0.;
            x5val06 = 0.;
            x5val12 = 0.;
            for ( int i = -2; i <= +2; i++ ) {
              for ( int j = -2; j <= +2; j++ ) {

                x5val01 = x5val01 + xmap01[ism-1][ie-1+i][ip-1+j];
                x5val06 = x5val06 + xmap06[ism-1][ie-1+i][ip-1+j];
                x5val12 = x5val12 + xmap12[ism-1][ie-1+i][ip-1+j];

              }
            }
            x5val01 = x5val01 / 25.;
            x5val06 = x5val06 / 25.;
            x5val12 = x5val12 / 25.;
            if ( mePed5SumMapG01_[ism-1] && x5val01 != 0. ) mePed5SumMapG01_[ism-1]->Fill(xie, xip, x5val01);
            if ( mePed5SumMapG06_[ism-1] && x5val06 != 0. ) mePed5SumMapG06_[ism-1]->Fill(xie, xip, x5val06);
            if ( mePed5SumMapG12_[ism-1] && x5val12 != 0. ) mePed5SumMapG12_[ism-1]->Fill(xie, xip, x5val12);

          }

        }
      }
    }
#endif

  } else {

    edm::LogWarning("EBPedestalTask") << EBDigiCollection_ << " not available";

  }

  edm::Handle<EcalPnDiodeDigiCollection> pns;

  if ( e.getByLabel(EcalPnDiodeDigiCollection_, pns) ) {

    int nep = pns->size();
    LogDebug("EBPedestalTask") << "event " << ievt_ << " pns collection size " << nep;

    for ( EcalPnDiodeDigiCollection::const_iterator pnItr = pns->begin(); pnItr != pns->end(); ++pnItr ) {

      if ( Numbers::subDet( pnItr->id() ) != EcalBarrel ) continue;

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

    edm::LogWarning("EBPedestalTask") << EcalPnDiodeDigiCollection_ << " not available";

  }

}

