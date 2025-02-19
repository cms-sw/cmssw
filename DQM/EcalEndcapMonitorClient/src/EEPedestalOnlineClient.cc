/*
 * \file EEPedestalOnlineClient.cc
 *
 * $Date: 2012/04/27 13:46:08 $
 * $Revision: 1.117 $
 * \author G. Della Ricca
 * \author F. Cossutti
 *
*/

#include <memory>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <math.h>

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#ifdef WITH_ECAL_COND_DB
#include "OnlineDB/EcalCondDB/interface/MonPedestalsOnlineDat.h"
#include "OnlineDB/EcalCondDB/interface/RunCrystalErrorsDat.h"
#include "OnlineDB/EcalCondDB/interface/RunTTErrorsDat.h"
#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"
#include "DQM/EcalCommon/interface/LogicID.h"
#endif

#include "DQM/EcalCommon/interface/Masks.h"

#include "DQM/EcalCommon/interface/UtilsClient.h"
#include "DQM/EcalCommon/interface/Numbers.h"

#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "DQM/EcalEndcapMonitorClient/interface/EEPedestalOnlineClient.h"

EEPedestalOnlineClient::EEPedestalOnlineClient(const edm::ParameterSet& ps) {

  // cloneME switch
  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);

  // verbose switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", true);

  // debug switch
  debug_ = ps.getUntrackedParameter<bool>("debug", false);

  // prefixME path
  prefixME_ = ps.getUntrackedParameter<std::string>("prefixME", "");

  subfolder_ = ps.getUntrackedParameter<std::string>("subfolder", "");

  // enableCleanup_ switch
  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  // vector of selected Super Modules (Defaults to all 18).
  superModules_.reserve(18);
  for ( unsigned int i = 1; i <= 18; i++ ) superModules_.push_back(i);
  superModules_ = ps.getUntrackedParameter<std::vector<int> >("superModules", superModules_);

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    h03_[ism-1] = 0;

  }

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    meg03_[ism-1] = 0;

    mep03_[ism-1] = 0;

    mer03_[ism-1] = 0;

  }

  expectedMean_ = 200.0;
  discrepancyMean_ = 25.0;
  RMSThreshold_ = 4.0;
  RMSThresholdInternal_ = 8.0;

}

EEPedestalOnlineClient::~EEPedestalOnlineClient() {

}

void EEPedestalOnlineClient::beginJob(void) {

  dqmStore_ = edm::Service<DQMStore>().operator->();

  if ( debug_ ) std::cout << "EEPedestalOnlineClient: beginJob" << std::endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EEPedestalOnlineClient::beginRun(void) {

  if ( debug_ ) std::cout << "EEPedestalOnlineClient: beginRun" << std::endl;

  jevt_ = 0;

  this->setup();

}

void EEPedestalOnlineClient::endJob(void) {

  if ( debug_ ) std::cout << "EEPedestalOnlineClient: endJob, ievt = " << ievt_ << std::endl;

  this->cleanup();

}

void EEPedestalOnlineClient::endRun(void) {

  if ( debug_ ) std::cout << "EEPedestalOnlineClient: endRun, jevt = " << jevt_ << std::endl;

  this->cleanup();

}

void EEPedestalOnlineClient::setup(void) {

  std::string name;

  dqmStore_->setCurrentFolder( prefixME_ + "/EEPedestalOnlineClient" );

  if(subfolder_.size())
    dqmStore_->setCurrentFolder(prefixME_ + "/EEPedestalOnlineClient/" + subfolder_);

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( meg03_[ism-1] ) dqmStore_->removeElement( meg03_[ism-1]->getName() );
    name = "EEPOT pedestal quality G12 " + Numbers::sEE(ism);
    meg03_[ism-1] = dqmStore_->book2D(name, name, 50, Numbers::ix0EE(ism)+0., Numbers::ix0EE(ism)+50., 50, Numbers::iy0EE(ism)+0., Numbers::iy0EE(ism)+50.);
    meg03_[ism-1]->setAxisTitle("ix", 1);
    if ( ism >= 1 && ism <= 9 ) meg03_[ism-1]->setAxisTitle("101-ix", 1);
    meg03_[ism-1]->setAxisTitle("iy", 2);

    if ( mep03_[ism-1] ) dqmStore_->removeElement( mep03_[ism-1]->getName() );
    name = "EEPOT pedestal mean G12 " + Numbers::sEE(ism);
    mep03_[ism-1] = dqmStore_->book1D(name, name, 100, 150., 250.);
    mep03_[ism-1]->setAxisTitle("mean", 1);

    if ( mer03_[ism-1] ) dqmStore_->removeElement( mer03_[ism-1]->getName() );
    name = "EEPOT pedestal rms G12 " + Numbers::sEE(ism);
    mer03_[ism-1] = dqmStore_->book1D(name, name, 100, 0.,  10.);
    mer03_[ism-1]->setAxisTitle("rms", 1);

  }

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( meg03_[ism-1] ) meg03_[ism-1]->Reset();

    for ( int ix = 1; ix <= 50; ix++ ) {
      for ( int iy = 1; iy <= 50; iy++ ) {

        meg03_[ism-1]->setBinContent( ix, iy, 6. );

        int jx = ix + Numbers::ix0EE(ism);
        int jy = iy + Numbers::iy0EE(ism);

        if ( ism >= 1 && ism <= 9 ) jx = 101 - jx;

        if ( Numbers::validEE(ism, jx, jy) ) {
          meg03_[ism-1]->setBinContent( ix, iy, 2. );
        }

      }
    }

    if ( mep03_[ism-1] ) mep03_[ism-1]->Reset();
    if ( mer03_[ism-1] ) mer03_[ism-1]->Reset();

  }

}

void EEPedestalOnlineClient::cleanup(void) {

  if ( ! enableCleanup_ ) return;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( cloneME_ ) {
      if ( h03_[ism-1] ) delete h03_[ism-1];
    }

    h03_[ism-1] = 0;

  }

  dqmStore_->setCurrentFolder( prefixME_ + "/EEPedestalOnlineClient" );

  if(subfolder_.size())
    dqmStore_->setCurrentFolder(prefixME_ + "/EEPedestalOnlineClient/" + subfolder_);

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( meg03_[ism-1] ) dqmStore_->removeElement( meg03_[ism-1]->getName() );
    meg03_[ism-1] = 0;

    if ( mep03_[ism-1] ) dqmStore_->removeElement( mep03_[ism-1]->getName() );
    mep03_[ism-1] = 0;

    if ( mer03_[ism-1] ) dqmStore_->removeElement( mer03_[ism-1]->getName() );
    mer03_[ism-1] = 0;

  }

}

#ifdef WITH_ECAL_COND_DB
bool EEPedestalOnlineClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov, bool& status) {

  status = true;

  EcalLogicID ecid;

  MonPedestalsOnlineDat p;
  std::map<EcalLogicID, MonPedestalsOnlineDat> dataset;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( verbose_ ) {
      std::cout << " " << Numbers::sEE(ism) << " (ism=" << ism << ")" << std::endl;
      std::cout << std::endl;
      UtilsClient::printBadChannels(meg03_[ism-1], h03_[ism-1]);
    }

    float num03;
    float mean03;
    float rms03;

    for ( int ix = 1; ix <= 50; ix++ ) {
      for ( int iy = 1; iy <= 50; iy++ ) {

        int jx = ix + Numbers::ix0EE(ism);
        int jy = iy + Numbers::iy0EE(ism);

        if ( ism >= 1 && ism <= 9 ) jx = 101 - jx;

        if ( ! Numbers::validEE(ism, jx, jy) ) continue;

        bool update03;

        update03 = UtilsClient::getBinStatistics(h03_[ism-1], ix, iy, num03, mean03, rms03);

        if ( update03 ) {

          if ( Numbers::icEE(ism, jx, jy) == 1 ) {

            if ( verbose_ ) {
              std::cout << "Preparing dataset for " << Numbers::sEE(ism) << " (ism=" << ism << ")" << std::endl;
              std::cout << "G12 (" << Numbers::ix0EE(i+1)+ix << "," << Numbers::iy0EE(i+1)+iy << ") " << num03  << " " << mean03 << " " << rms03  << std::endl;
              std::cout << std::endl;
            }

          }

          p.setADCMeanG12(mean03);
          p.setADCRMSG12(rms03);

          if ( UtilsClient::getBinStatus(meg03_[ism-1], ix, iy) ) {
            p.setTaskStatus(true);
          } else {
            p.setTaskStatus(false);
          }

          status = status && UtilsClient::getBinQuality(meg03_[ism-1], ix, iy);

          int ic = Numbers::indexEE(ism, jx, jy);

          if ( ic == -1 ) continue;

          if ( econn ) {
            ecid = LogicID::getEcalLogicID("EE_crystal_number", Numbers::iSM(ism, EcalEndcap), ic);
            dataset[ecid] = p;
          }

        }

      }
    }

  }

  if ( econn ) {
    try {
      if ( verbose_ ) std::cout << "Inserting MonPedestalsOnlineDat ..." << std::endl;
      if ( dataset.size() != 0 ) econn->insertDataArraySet(&dataset, moniov);
      if ( verbose_ ) std::cout << "done." << std::endl;
    } catch (std::runtime_error &e) {
      std::cerr << e.what() << std::endl;
    }
  }

  return true;

}
#endif

void EEPedestalOnlineClient::analyze(void) {

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( debug_ ) std::cout << "EEPedestalOnlineClient: ievt/jevt = " << ievt_ << "/" << jevt_ << std::endl;
  }

  uint32_t bits03 = 0;
  bits03 |= 1 << EcalDQMStatusHelper::PEDESTAL_ONLINE_HIGH_GAIN_MEAN_ERROR;
  bits03 |= 1 << EcalDQMStatusHelper::PEDESTAL_ONLINE_HIGH_GAIN_RMS_ERROR;

  std::string subdir(subfolder_.size() ? subfolder_ + "/" : "");

  MonitorElement* me;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    me = dqmStore_->get( prefixME_ + "/EEPedestalOnlineTask/" + subdir + "Gain12/EEPOT pedestal " + Numbers::sEE(ism) + " G12" );
    h03_[ism-1] = UtilsClient::getHisto( me, cloneME_, h03_[ism-1] );

    if ( meg03_[ism-1] ) meg03_[ism-1]->Reset();
    if ( mep03_[ism-1] ) mep03_[ism-1]->Reset();
    if ( mer03_[ism-1] ) mer03_[ism-1]->Reset();

    for ( int ix = 1; ix <= 50; ix++ ) {
      for ( int iy = 1; iy <= 50; iy++ ) {

        if ( meg03_[ism-1] ) meg03_[ism-1]->setBinContent(ix, iy, 6.);

        int jx = ix + Numbers::ix0EE(ism);
        int jy = iy + Numbers::iy0EE(ism);

        if ( ism >= 1 && ism <= 9 ) jx = 101 - jx;

        if ( Numbers::validEE(ism, jx, jy) ) {
          if ( meg03_[ism-1] ) meg03_[ism-1]->setBinContent( ix, iy, 2. );
        }

        // OOT pileup affects the region near beam pipe mostly. Use higher threshold for these crystals
        float radius = std::sqrt((jx-50.)*(jx-50.)+(jy-50.)*(jy-50.));

        bool update03;

        float num03;
        float mean03;
        float rms03;

        update03 = UtilsClient::getBinStatistics(h03_[ism-1], ix, iy, num03, mean03, rms03);

        if ( update03 ) {

          float val;

          val = 1.;
          if ( std::abs(mean03 - expectedMean_) > discrepancyMean_ )
            val = 0.;
          if ( (radius >= 20. && rms03 > RMSThreshold_) || (radius < 20. && rms03 > RMSThresholdInternal_) )
            val = 0.;
          if ( meg03_[ism-1] ) meg03_[ism-1]->setBinContent(ix, iy, val);

          if ( mep03_[ism-1] ) mep03_[ism-1]->Fill(mean03);
          if ( mer03_[ism-1] ) mer03_[ism-1]->Fill(rms03);

        }

        if ( Masks::maskChannel(ism, ix, iy, bits03, EcalEndcap) ) UtilsClient::maskBinContent( meg03_[ism-1], ix, iy );

      }
    }

  }

}

