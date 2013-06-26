/*
 * \file EECosmicClient.cc
 *
 * $Date: 2012/04/27 13:46:07 $
 * $Revision: 1.77 $
 * \author G. Della Ricca
 * \author F. Cossutti
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
#include "OnlineDB/EcalCondDB/interface/MonOccupancyDat.h"
#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"
#include "DQM/EcalCommon/interface/LogicID.h"
#endif

#include "DQM/EcalCommon/interface/UtilsClient.h"
#include "DQM/EcalCommon/interface/Numbers.h"

#include "DQM/EcalEndcapMonitorClient/interface/EECosmicClient.h"

EECosmicClient::EECosmicClient(const edm::ParameterSet& ps) {

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
    h02_[ism-1] = 0;
    h03_[ism-1] = 0;

    meh01_[ism-1] = 0;
    meh02_[ism-1] = 0;
    meh03_[ism-1] = 0;

  }

}

EECosmicClient::~EECosmicClient() {

}

void EECosmicClient::beginJob(void) {

  dqmStore_ = edm::Service<DQMStore>().operator->();

  if ( debug_ ) std::cout << "EECosmicClient: beginJob" << std::endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EECosmicClient::beginRun(void) {

  if ( debug_ ) std::cout << "EECosmicClient: beginRun" << std::endl;

  jevt_ = 0;

  this->setup();

}

void EECosmicClient::endJob(void) {

  if ( debug_ ) std::cout << "EECosmicClient: endJob, ievt = " << ievt_ << std::endl;

  this->cleanup();

}

void EECosmicClient::endRun(void) {

  if ( debug_ ) std::cout << "EECosmicClient: endRun, jevt = " << jevt_ << std::endl;

  this->cleanup();

}

void EECosmicClient::setup(void) {

}

void EECosmicClient::cleanup(void) {

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

}

#ifdef WITH_ECAL_COND_DB
bool EECosmicClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov, bool& status) {

  status = true;

  return true;

}
#endif

void EECosmicClient::analyze(void) {

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( debug_ ) std::cout << "EECosmicClient: ievt/jevt = " << ievt_ << "/" << jevt_ << std::endl;
  }

  MonitorElement* me;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    me = dqmStore_->get( prefixME_ + "/EECosmicTask/Sel/EECT energy sel " + Numbers::sEE(ism) );
    h01_[ism-1] = UtilsClient::getHisto( me, cloneME_, h01_[ism-1] );
    meh01_[ism-1] = me;

    me = dqmStore_->get( prefixME_ + "/EECosmicTask/Spectrum/EECT 1x1 energy spectrum " + Numbers::sEE(ism) );
    h02_[ism-1] = UtilsClient::getHisto( me, cloneME_, h02_[ism-1] );
    meh02_[ism-1] = me;

    me = dqmStore_->get( prefixME_ + "/EECosmicTask/Spectrum/EECT 3x3 energy spectrum " + Numbers::sEE(ism) );
    h03_[ism-1] = UtilsClient::getHisto( me, cloneME_, h03_[ism-1] );
    meh03_[ism-1] = me;

  }

}

