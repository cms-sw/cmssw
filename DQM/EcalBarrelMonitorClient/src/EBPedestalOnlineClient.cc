/*
 * \file EBPedestalOnlineClient.cc
 *
 * $Date: 2012/04/27 13:45:59 $
 * $Revision: 1.167 $
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

#include "DQM/EcalBarrelMonitorClient/interface/EBPedestalOnlineClient.h"

EBPedestalOnlineClient::EBPedestalOnlineClient(const edm::ParameterSet& ps) {

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

  // vector of selected Super Modules (Defaults to all 36).
  superModules_.reserve(36);
  for ( unsigned int i = 1; i <= 36; i++ ) superModules_.push_back(i);
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
  RMSThreshold_ = 3.0;
  RMSThresholdHighEta_ = 6.0;

}

EBPedestalOnlineClient::~EBPedestalOnlineClient() {

}

void EBPedestalOnlineClient::beginJob(void) {

  dqmStore_ = edm::Service<DQMStore>().operator->();

  if ( debug_ ) std::cout << "EBPedestalOnlineClient: beginJob" << std::endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EBPedestalOnlineClient::beginRun(void) {

  if ( debug_ ) std::cout << "EBPedestalOnlineClient: beginRun" << std::endl;

  jevt_ = 0;

  this->setup();

}

void EBPedestalOnlineClient::endJob(void) {

  if ( debug_ ) std::cout << "EBPedestalOnlineClient: endJob, ievt = " << ievt_ << std::endl;

  this->cleanup();

}

void EBPedestalOnlineClient::endRun(void) {

  if ( debug_ ) std::cout << "EBPedestalOnlineClient: endRun, jevt = " << jevt_ << std::endl;

  this->cleanup();

}

void EBPedestalOnlineClient::setup(void) {

  std::string name;

  dqmStore_->setCurrentFolder( prefixME_ + "/EBPedestalOnlineClient" );

  if(subfolder_.size())
    dqmStore_->setCurrentFolder( prefixME_ + "/EBPedestalOnlineClient/" + subfolder_);

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( meg03_[ism-1] ) dqmStore_->removeElement( meg03_[ism-1]->getName() );
    name = "EBPOT pedestal quality G12 " + Numbers::sEB(ism);
    meg03_[ism-1] = dqmStore_->book2D(name, name, 85, 0., 85., 20, 0., 20.);
    meg03_[ism-1]->setAxisTitle("ieta", 1);
    meg03_[ism-1]->setAxisTitle("iphi", 2);

    if ( mep03_[ism-1] ) dqmStore_->removeElement( mep03_[ism-1]->getName() );
    name = "EBPOT pedestal mean G12 " + Numbers::sEB(ism);
    mep03_[ism-1] = dqmStore_->book1D(name, name, 100, 150., 250.);
    mep03_[ism-1]->setAxisTitle("mean", 1);

    if ( mer03_[ism-1] ) dqmStore_->removeElement( mer03_[ism-1]->getName() );
    name = "EBPOT pedestal rms G12 " + Numbers::sEB(ism);
    mer03_[ism-1] = dqmStore_->book1D(name, name, 100, 0.,  10.);
    mer03_[ism-1]->setAxisTitle("rms", 1);

  }

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( meg03_[ism-1] ) meg03_[ism-1]->Reset();

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

        meg03_[ism-1]->setBinContent( ie, ip, 2. );

      }
    }

    if ( mep03_[ism-1] ) mep03_[ism-1]->Reset();
    if ( mer03_[ism-1] ) mer03_[ism-1]->Reset();

  }

}

void EBPedestalOnlineClient::cleanup(void) {

  if ( ! enableCleanup_ ) return;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( cloneME_ ) {
      if ( h03_[ism-1] ) delete h03_[ism-1];
    }

    h03_[ism-1] = 0;

  }

  dqmStore_->setCurrentFolder( prefixME_ + "/EBPedestalOnlineClient" );

  if(subfolder_.size())
    dqmStore_->setCurrentFolder( prefixME_ + "/EBPedestalOnlineClient/" + subfolder_);

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
bool EBPedestalOnlineClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov, bool& status) {

  status = true;

  EcalLogicID ecid;

  MonPedestalsOnlineDat p;
  std::map<EcalLogicID, MonPedestalsOnlineDat> dataset;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( verbose_ ) {
      std::cout << " " << Numbers::sEB(ism) << " (ism=" << ism << ")" << std::endl;
      std::cout << std::endl;
      UtilsClient::printBadChannels(meg03_[ism-1], h03_[ism-1]);
    }

    float num03;
    float mean03;
    float rms03;

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

        bool update03;

        update03 = UtilsClient::getBinStatistics(h03_[ism-1], ie, ip, num03, mean03, rms03);

        if ( update03 ) {

          if ( Numbers::icEB(ism, ie, ip) == 1 ) {

            if ( verbose_ ) {
              std::cout << "Preparing dataset for " << Numbers::sEB(ism) << " (ism=" << ism << ")" << std::endl;
              std::cout << "G12 (" << ie << "," << ip << ") " << num03  << " " << mean03 << " " << rms03  << std::endl;
              std::cout << std::endl;
            }

          }

          p.setADCMeanG12(mean03);
          p.setADCRMSG12(rms03);

          if ( UtilsClient::getBinStatus(meg03_[ism-1], ie, ip) ) {
            p.setTaskStatus(true);
          } else {
            p.setTaskStatus(false);
          }

          status = status && UtilsClient::getBinQuality(meg03_[ism-1], ie, ip);

          int ic = Numbers::indexEB(ism, ie, ip);

          if ( econn ) {
            ecid = LogicID::getEcalLogicID("EB_crystal_number", Numbers::iSM(ism, EcalBarrel), ic);
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

void EBPedestalOnlineClient::analyze(void) {

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( debug_ ) std::cout << "EBPedestalOnlineClient: ievt/jevt = " << ievt_ << "/" << jevt_ << std::endl;
  }

  uint32_t bits03 = 0;
  bits03 |= 1 << EcalDQMStatusHelper::PEDESTAL_ONLINE_HIGH_GAIN_MEAN_ERROR;
  bits03 |= 1 << EcalDQMStatusHelper::PEDESTAL_ONLINE_HIGH_GAIN_RMS_ERROR;

  std::string subdir(subfolder_.size() ? subfolder_ + "/" : "");

  MonitorElement* me;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    me = dqmStore_->get( prefixME_ + "/EBPedestalOnlineTask/" + subdir + "Gain12/EBPOT pedestal " + Numbers::sEB(ism) + " G12" );
    h03_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h03_[ism-1] );
    if ( meg03_[ism-1] ) meg03_[ism-1]->Reset();
    if ( mep03_[ism-1] ) mep03_[ism-1]->Reset();
    if ( mer03_[ism-1] ) mer03_[ism-1]->Reset();

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

        if ( meg03_[ism-1] ) meg03_[ism-1]->setBinContent(ie, ip, 2.);

        bool update03;

        float num03;
        float mean03;
        float rms03;

        update03 = UtilsClient::getBinStatistics(h03_[ism-1], ie, ip, num03, mean03, rms03);

        if ( update03 ) {

          float val;

          val = 1.;
          if ( std::abs(mean03 - expectedMean_) > discrepancyMean_ )
            val = 0.;
          if ( (ie<=40 && rms03 > RMSThreshold_) || (ie>40 && rms03 > RMSThresholdHighEta_) )
            val = 0.;
          if ( meg03_[ism-1] ) meg03_[ism-1]->setBinContent(ie, ip, val);

          if ( mep03_[ism-1] ) mep03_[ism-1]->Fill(mean03);
          if ( mer03_[ism-1] ) mer03_[ism-1]->Fill(rms03);

        }

        if ( Masks::maskChannel(ism, ie, ip, bits03, EcalBarrel) ) UtilsClient::maskBinContent( meg03_[ism-1], ie, ip );

      }
    }

  }

}

