/*
 * \file EEStatusFlagsClient.cc
 *
 * $Date: 2012/04/27 13:46:08 $
 * $Revision: 1.51 $
 * \author G. Della Ricca
 *
*/

#include <memory>
#include <iostream>
#include <fstream>
#include <iomanip>

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#ifdef WITH_ECAL_COND_DB
#include "OnlineDB/EcalCondDB/interface/RunTTErrorsDat.h"
#include "DQM/EcalCommon/interface/LogicID.h"
#endif

#include "DQM/EcalCommon/interface/Masks.h"

#include "DQM/EcalCommon/interface/UtilsClient.h"
#include "DQM/EcalCommon/interface/Numbers.h"

#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "DQM/EcalEndcapMonitorClient/interface/EEStatusFlagsClient.h"

EEStatusFlagsClient::EEStatusFlagsClient(const edm::ParameterSet& ps) {

  // cloneME switch
  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);

  // verbose switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", true);

  // debug switch
  debug_ = ps.getUntrackedParameter<bool>("debug", false);

  // prefixME path
  prefixME_ = ps.getUntrackedParameter<std::string>("prefixME", "");

  // enableCleanup_ switch
  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  // vector of selected Super Modules (Defaults to all 18).
  superModules_.reserve(18);
  for ( unsigned int i = 1; i <= 18; i++ ) superModules_.push_back(i);
  superModules_ = ps.getUntrackedParameter<std::vector<int> >("superModules", superModules_);

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    h01_[ism-1] = 0;

    meh01_[ism-1] = 0;

    h02_[ism-1] = 0;

    meh02_[ism-1] = 0;

    h03_[ism-1] = 0;

    meh03_[ism-1] = 0;

  }

}

EEStatusFlagsClient::~EEStatusFlagsClient() {

}

void EEStatusFlagsClient::beginJob(void) {

  dqmStore_ = edm::Service<DQMStore>().operator->();

  if ( debug_ ) std::cout << "EEStatusFlagsClient: beginJob" << std::endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EEStatusFlagsClient::beginRun(void) {

  if ( debug_ ) std::cout << "EEStatusFlagsClient: beginRun" << std::endl;

  jevt_ = 0;

  this->setup();

}

void EEStatusFlagsClient::endJob(void) {

  if ( debug_ ) std::cout << "EEStatusFlagsClient: endJob, ievt = " << ievt_ << std::endl;

  this->cleanup();

}

void EEStatusFlagsClient::endRun(void) {

  if ( debug_ ) std::cout << "EEStatusFlagsClient: endRun, jevt = " << jevt_ << std::endl;

  this->cleanup();

}

void EEStatusFlagsClient::setup(void) {

  dqmStore_->setCurrentFolder( prefixME_ + "/EEStatusFlagsClient" );

}

void EEStatusFlagsClient::cleanup(void) {

  if ( ! enableCleanup_ ) return;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( cloneME_ ) {
      if ( h01_[ism-1] ) delete h01_[ism-1];
      if ( h02_[ism-1] ) delete h02_[ism-1];
      if ( h03_[ism-1] ) delete h03_[ism-1];
    }

    h01_[ism-1] = 0;
    h02_[ism-1] = 0;
    h03_[ism-1] = 0;

    meh01_[ism-1] = 0;
    meh02_[ism-1] = 0;
    meh03_[ism-1] = 0;

  }

  dqmStore_->setCurrentFolder( prefixME_ + "/EEStatusFlagsClient" );

}

#ifdef WITH_ECAL_COND_DB
bool EEStatusFlagsClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov, bool& status) {

  status = true;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( verbose_ ) {
      std::cout << " " << Numbers::sEE(ism) << " (ism=" << ism << ")" << std::endl;
      std::cout << std::endl;
      UtilsClient::printBadChannels(meh01_[ism-1], UtilsClient::getHisto<TH2F*>(meh01_[ism-1]), true);
    }

    if ( meh01_[ism-1] ) {
      if ( meh01_[ism-1]->getEntries() != 0 ) status = false;
    }

  }

  return true;

}
#endif

void EEStatusFlagsClient::analyze(void) {

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( debug_ ) std::cout << "EEStatusFlagsClient: ievt/jevt = " << ievt_ << "/" << jevt_ << std::endl;
  }

  uint32_t bits01 = 0;
  bits01 |= 1 << EcalDQMStatusHelper::STATUS_FLAG_ERROR;

  MonitorElement* me;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    me = dqmStore_->get( prefixME_ + "/EEStatusFlagsTask/FEStatus/EESFT front-end status " + Numbers::sEE(ism) );
    h01_[ism-1] = UtilsClient::getHisto<TH2F*>( me, cloneME_, h01_[ism-1] );
    meh01_[ism-1] = me;

    me = dqmStore_->get( prefixME_ + "/EEStatusFlagsTask/FEStatus/EESFT front-end status bits " + Numbers::sEE(ism) );
    h02_[ism-1] = UtilsClient::getHisto<TH1F*>( me, cloneME_, h02_[ism-1] );
    meh02_[ism-1] = me;

    me = dqmStore_->get( prefixME_ + "/EEStatusFlagsTask/FEStatus/EESFT MEM front-end status " + Numbers::sEE(ism) );
    h03_[ism-1] = UtilsClient::getHisto<TH2F*>( me, cloneME_, h01_[ism-1] );
    meh03_[ism-1] = me;

    for ( int ix = 1; ix <= 50; ix++ ) {
      for ( int iy = 1; iy <= 50; iy++ ) {
        if ( Masks::maskChannel(ism, ix, iy, bits01, EcalEndcap) ) {
          if ( meh01_[ism-1] ) meh01_[ism-1]->setBinError( ix, iy, 0.01 );
        }
      }
    }

    for ( int i = 1; i <= 10; i++ ) {
      if ( Masks::maskPn(ism, i, bits01, EcalEndcap) ) {
        int it = (i-1)/5 + 1;
        if ( meh03_[ism-1] ) meh03_[ism-1]->setBinError( it, 1, 0.01 );
      }
    }

  }

}

