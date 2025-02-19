/*
 * \file EBStatusFlagsClient.cc
 *
 * $Date: 2012/04/27 13:45:59 $
 * $Revision: 1.49 $
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

#include "DQM/EcalBarrelMonitorClient/interface/EBStatusFlagsClient.h"

EBStatusFlagsClient::EBStatusFlagsClient(const edm::ParameterSet& ps) {

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

  // vector of selected Super Modules (Defaults to all 36).
  superModules_.reserve(36);
  for ( unsigned int i = 1; i <= 36; i++ ) superModules_.push_back(i);
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

EBStatusFlagsClient::~EBStatusFlagsClient() {

}

void EBStatusFlagsClient::beginJob(void) {

  dqmStore_ = edm::Service<DQMStore>().operator->();

  if ( debug_ ) std::cout << "EBStatusFlagsClient: beginJob" << std::endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EBStatusFlagsClient::beginRun(void) {

  if ( debug_ ) std::cout << "EBStatusFlagsClient: beginRun" << std::endl;

  jevt_ = 0;

  this->setup();

}

void EBStatusFlagsClient::endJob(void) {

  if ( debug_ ) std::cout << "EBStatusFlagsClient: endJob, ievt = " << ievt_ << std::endl;

  this->cleanup();

}

void EBStatusFlagsClient::endRun(void) {

  if ( debug_ ) std::cout << "EBStatusFlagsClient: endRun, jevt = " << jevt_ << std::endl;

  this->cleanup();

}

void EBStatusFlagsClient::setup(void) {

  dqmStore_->setCurrentFolder( prefixME_ + "/EBStatusFlagsClient" );

}

void EBStatusFlagsClient::cleanup(void) {

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

  dqmStore_->setCurrentFolder( prefixME_ + "/EBStatusFlagsClient" );

}

#ifdef WITH_ECAL_COND_DB
bool EBStatusFlagsClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov, bool& status) {

  status = true;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( verbose_ ) {
      std::cout << " " << Numbers::sEB(ism) << " (ism=" << ism << ")" << std::endl;
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

void EBStatusFlagsClient::analyze(void) {

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( debug_ ) std::cout << "EBStatusFlagsClient: ievt/jevt = " << ievt_ << "/" << jevt_ << std::endl;
  }

  uint32_t bits01 = 0;
  bits01 |= 1 << EcalDQMStatusHelper::STATUS_FLAG_ERROR;

  MonitorElement* me;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    me = dqmStore_->get( prefixME_ + "/EBStatusFlagsTask/FEStatus/EBSFT front-end status " + Numbers::sEB(ism) );
    h01_[ism-1] = UtilsClient::getHisto( me, cloneME_, h01_[ism-1] );
    meh01_[ism-1] = me;

    me = dqmStore_->get( prefixME_ + "/EBStatusFlagsTask/FEStatus/EBSFT front-end status bits " + Numbers::sEB(ism) );
    h02_[ism-1] = UtilsClient::getHisto( me, cloneME_, h02_[ism-1] );
    meh02_[ism-1] = me;

    me = dqmStore_->get( prefixME_ + "/EBStatusFlagsTask/FEStatus/EBSFT MEM front-end status " + Numbers::sEB(ism) );
    h03_[ism-1] = UtilsClient::getHisto( me, cloneME_, h01_[ism-1] );
    meh03_[ism-1] = me;

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {
        if ( Masks::maskChannel(ism, ie, ip, bits01, EcalBarrel) ) {
          int iet = (ie-1)/5 + 1;
          int ipt = (ip-1)/5 + 1;
          if ( meh01_[ism-1] ) meh01_[ism-1]->setBinError( iet, ipt, 0.01 );
        }
      }
    }

    for ( int i = 1; i <= 10; i++ ) {
      if ( Masks::maskPn(ism, i, bits01, EcalBarrel) ) {
        int it = (i-1)/5 + 1;
        if ( meh03_[ism-1] ) meh03_[ism-1]->setBinError( it, 1, 0.01 );
      }
    }

  }

}

