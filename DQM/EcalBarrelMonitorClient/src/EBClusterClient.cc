/*
 * \file EBClusterClient.cc
 *
 * $Date: 2012/04/27 13:45:58 $
 * $Revision: 1.80 $
 * \author G. Della Ricca
 * \author F. Cossutti
 * \author E. Di Marco
 *
*/

#include <memory>
#include <iostream>
#include <fstream>
#include <math.h>

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DQM/EcalCommon/interface/UtilsClient.h"

#include "DQM/EcalBarrelMonitorClient/interface/EBClusterClient.h"

EBClusterClient::EBClusterClient(const edm::ParameterSet& ps) {

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

  h01_[0] = 0;
  h01_[1] = 0;
  h01_[2] = 0;

  h02_[0] = 0;
  h02ProjEta_[0] = 0;
  h02ProjPhi_[0] = 0;
  h02_[1] = 0;
  h02ProjEta_[1] = 0;
  h02ProjPhi_[1] = 0;

  h03_ = 0;
  h03ProjEta_ = 0;
  h03ProjPhi_ = 0;

  h04_ = 0;
  h04ProjEta_ = 0;
  h04ProjPhi_ = 0;

  i01_[0] = 0;
  i01_[1] = 0;
  i01_[2] = 0;

  s01_[0] = 0;
  s01_[1] = 0;
  s01_[2] = 0;

}

EBClusterClient::~EBClusterClient() {

}

void EBClusterClient::beginJob(void) {

  dqmStore_ = edm::Service<DQMStore>().operator->();

  if ( debug_ ) std::cout << "EBClusterClient: beginJob" << std::endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EBClusterClient::beginRun(void) {

  if ( debug_ ) std::cout << "EBClusterClient: beginRun" << std::endl;

  jevt_ = 0;

  this->setup();

}

void EBClusterClient::endJob(void) {

  if ( debug_ ) std::cout << "EBClusterClient: endJob, ievt = " << ievt_ << std::endl;

  this->cleanup();

}

void EBClusterClient::endRun(void) {

  if ( debug_ ) std::cout << "EBClusterClient: endRun, jevt = " << jevt_ << std::endl;

  this->cleanup();

}

void EBClusterClient::setup(void) {

  dqmStore_->setCurrentFolder( prefixME_ + "/EBClusterClient" );

}

void EBClusterClient::cleanup(void) {

  if ( ! enableCleanup_ ) return;

  if ( cloneME_ ) {
    if ( h01_[0] ) delete h01_[0];
    if ( h01_[1] ) delete h01_[1];
    if ( h01_[2] ) delete h01_[2];

    if ( h02_[0] ) delete h02_[0];
    if ( h02ProjEta_[0] ) delete h02ProjEta_[0];
    if ( h02ProjPhi_[0] ) delete h02ProjPhi_[0];
    if ( h02_[1] ) delete h02_[1];
    if ( h02ProjEta_[1] ) delete h02ProjEta_[1];
    if ( h02ProjPhi_[1] ) delete h02ProjPhi_[1];

    if ( h03_ ) delete h03_;
    if ( h03ProjEta_ ) delete h03ProjEta_;
    if ( h03ProjPhi_ ) delete h03ProjPhi_;
    if ( h04_ ) delete h04_;
    if ( h04ProjEta_ ) delete h04ProjEta_;
    if ( h04ProjPhi_ ) delete h04ProjPhi_;

    if ( i01_[0] ) delete i01_[0];
    if ( i01_[1] ) delete i01_[1];
    if ( i01_[2] ) delete i01_[2];

    if ( s01_[0] ) delete s01_[0];
    if ( s01_[1] ) delete s01_[1];
    if ( s01_[2] ) delete s01_[2];

  }

  h01_[0] = 0;
  h01_[1] = 0;
  h01_[2] = 0;

  h02_[0] = 0;
  h02ProjEta_[0] = 0;
  h02ProjPhi_[0] = 0;
  h02_[1] = 0;
  h02ProjEta_[1] = 0;
  h02ProjPhi_[1] = 0;

  h03_ = 0;
  h03ProjEta_ = 0;
  h03ProjPhi_ = 0;
  h04_ = 0;
  h04ProjEta_ = 0;
  h04ProjPhi_ = 0;

  i01_[0] = 0;
  i01_[1] = 0;
  i01_[2] = 0;

  s01_[0] = 0;
  s01_[1] = 0;
  s01_[2] = 0;

}

#ifdef WITH_ECAL_COND_DB
bool EBClusterClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov, bool& status) {

  status = true;

  return true;

}
#endif

void EBClusterClient::analyze(void) {

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( debug_ ) std::cout << "EBClusterClient: ievt/jevt = " << ievt_ << "/" << jevt_ << std::endl;
  }

  MonitorElement* me;

  me = dqmStore_->get( prefixME_ + "/EBClusterTask/EBCLT BC energy" );
  h01_[0] = UtilsClient::getHisto( me, cloneME_, h01_[0] );

  me = dqmStore_->get( prefixME_ + "/EBClusterTask/EBCLT BC size" );
  h01_[1] = UtilsClient::getHisto( me, cloneME_, h01_[1] );

  me = dqmStore_->get( prefixME_ + "/EBClusterTask/EBCLT BC number" );
  h01_[2] = UtilsClient::getHisto( me, cloneME_, h01_[2] );

  me = dqmStore_->get( prefixME_ + "/EBClusterTask/EBCLT BC energy map" );
  h02_[0] = UtilsClient::getHisto( me, cloneME_, h02_[0] );

  me = dqmStore_->get( prefixME_ + "/EBClusterTask/EBCLT BC ET map" );
  h02_[1] = UtilsClient::getHisto( me, cloneME_, h02_[1] );

  me = dqmStore_->get( prefixME_ + "/EBClusterTask/EBCLT BC number map" );
  h03_ = UtilsClient::getHisto( me, cloneME_, h03_ );

  me = dqmStore_->get( prefixME_ + "/EBClusterTask/EBCLT BC size map" );
  h04_ = UtilsClient::getHisto( me, cloneME_, h04_ );

  me = dqmStore_->get( prefixME_ + "/EBClusterTask/EBCLT BC energy projection eta" );
  h02ProjEta_[0] = UtilsClient::getHisto( me, cloneME_, h02ProjEta_[0] );

  me = dqmStore_->get( prefixME_ + "/EBClusterTask/EBCLT BC energy projection phi" );
  h02ProjPhi_[0] = UtilsClient::getHisto( me, cloneME_, h02ProjPhi_[0] );

  me = dqmStore_->get( prefixME_ + "/EBClusterTask/EBCLT BC ET projection eta" );
  h02ProjEta_[1] = UtilsClient::getHisto( me, cloneME_, h02ProjEta_[1] );

  me = dqmStore_->get( prefixME_ + "/EBClusterTask/EBCLT BC ET projection phi" );
  h02ProjPhi_[1] = UtilsClient::getHisto( me, cloneME_, h02ProjPhi_[1] );

  me = dqmStore_->get( prefixME_ + "/EBClusterTask/EBCLT BC number projection eta" );
  h03ProjEta_ = UtilsClient::getHisto( me, cloneME_, h03ProjEta_ );

  me = dqmStore_->get( prefixME_ + "/EBClusterTask/EBCLT BC number projection phi" );
  h03ProjPhi_ = UtilsClient::getHisto( me, cloneME_, h03ProjPhi_ );

  me = dqmStore_->get( prefixME_ + "/EBClusterTask/EBCLT BC size projection eta" );
  h04ProjEta_ = UtilsClient::getHisto( me, cloneME_, h04ProjEta_ );

  me = dqmStore_->get( prefixME_ + "/EBClusterTask/EBCLT BC size projection phi" );
  h04ProjPhi_ = UtilsClient::getHisto( me, cloneME_, h04ProjPhi_ );

  me = dqmStore_->get( prefixME_ + "/EBClusterTask/EBCLT SC energy" );
  i01_[0] = UtilsClient::getHisto( me, cloneME_, i01_[0] );

  me = dqmStore_->get( prefixME_ + "/EBClusterTask/EBCLT SC size" );
  i01_[1] = UtilsClient::getHisto( me, cloneME_, i01_[1] );

  me = dqmStore_->get( prefixME_ + "/EBClusterTask/EBCLT SC number" );
  i01_[2] = UtilsClient::getHisto( me, cloneME_, i01_[2] );

  me = dqmStore_->get( prefixME_ + "/EBClusterTask/EBCLT s1s9" );
  s01_[0] = UtilsClient::getHisto( me, cloneME_, s01_[0] );

  me = dqmStore_->get( prefixME_ + "/EBClusterTask/EBCLT s9s25" );
  s01_[1] = UtilsClient::getHisto( me, cloneME_, s01_[1] );

  me = dqmStore_->get( prefixME_ + "/EBClusterTask/EBCLT dicluster invariant mass Pi0" );
  s01_[2] = UtilsClient::getHisto( me, cloneME_, s01_[2] );

}

