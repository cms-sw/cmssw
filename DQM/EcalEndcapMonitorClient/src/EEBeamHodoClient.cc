/*
 * \file EEBeamHodoClient.cc
 *
 * $Date: 2011/08/30 09:29:44 $
 * $Revision: 1.43 $
 * \author G. Della Ricca
 * \author G. Franzoni
 *
*/

#include <memory>
#include <iostream>
#include <sstream>
#include <iomanip>

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DQM/EcalCommon/interface/UtilsClient.h"
#include "DQM/EcalCommon/interface/Numbers.h"

#include "DQM/EcalEndcapMonitorClient/interface/EEBeamHodoClient.h"

EEBeamHodoClient::EEBeamHodoClient(const edm::ParameterSet& ps) {

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

  for (int i=0; i<4; i++) {

    ho01_[i] = 0;
    hr01_[i] = 0;

  }

  hp01_[0] = 0;
  hp01_[1] = 0;

  hp02_ = 0;

  hs01_[0] = 0;
  hs01_[1] = 0;

  hq01_[0] = 0;
  hq01_[1] = 0;

  ht01_ = 0;

  hc01_[0] = 0;
  hc01_[1] = 0;
  hc01_[2] = 0;

  hm01_    = 0;

  he01_[0] = 0;
  he01_[1] = 0;

  he02_[0] = 0;
  he02_[1] = 0;

  he03_[0] = 0;
  he03_[1] = 0;
  he03_[2] = 0;

}

EEBeamHodoClient::~EEBeamHodoClient() {

}

void EEBeamHodoClient::beginJob(void) {

  dqmStore_ = edm::Service<DQMStore>().operator->();

  if ( debug_ ) std::cout << "EEBeamHodoClient: beginJob" << std::endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EEBeamHodoClient::beginRun(void) {

  if ( debug_ ) std::cout << "EEBeamHodoClient: beginRun" << std::endl;

  jevt_ = 0;

  this->setup();

}

void EEBeamHodoClient::endJob(void) {

  if ( debug_ ) std::cout << "EEBeamHodoClient: endJob, ievt = " << ievt_ << std::endl;

  this->cleanup();

  if ( cloneME_ ) {

    for (int i=0; i<4; i++) {

      if ( ho01_[i] ) delete ho01_[i];
      if ( hr01_[i] ) delete hr01_[i];

    }

    if ( hp01_[0] ) delete hp01_[0];
    if ( hp01_[1] ) delete hp01_[1];

    if ( hp02_ ) delete hp02_;

    if ( hs01_[0] ) delete hs01_[0];
    if ( hs01_[1] ) delete hs01_[1];

    if ( hq01_[0] ) delete hq01_[0];
    if ( hq01_[1] ) delete hq01_[1];

    if ( ht01_ ) delete ht01_;

    if ( hc01_[0] ) delete hc01_[0];
    if ( hc01_[1] ) delete hc01_[1];
    if ( hc01_[2] ) delete hc01_[2];

    if ( hm01_ )    delete hm01_;

    if ( he01_[0] ) delete he01_[0];
    if ( he01_[1] ) delete he01_[1];

    if ( he02_[0] ) delete he02_[0];
    if ( he02_[1] ) delete he02_[1];

    if ( he03_[0] ) delete he03_[0];
    if ( he03_[1] ) delete he03_[1];
    if ( he03_[2] ) delete he03_[2];

  }

  for (int i=0; i<4; i++) {

    ho01_[i] = 0;
    hr01_[i] = 0;

  }

  hp01_[0] = 0;
  hp01_[1] = 0;

  hp02_ = 0;

  hs01_[0] = 0;
  hs01_[1] = 0;

  hq01_[0] = 0;
  hq01_[1] = 0;

  ht01_ = 0;

  hc01_[0] = 0;
  hc01_[1] = 0;
  hc01_[2] = 0;

  hm01_    = 0;

  he01_[0] = 0;
  he01_[1] = 0;

  he02_[0] = 0;
  he02_[1] = 0;

  he03_[0] = 0;
  he03_[1] = 0;
  he03_[2] = 0;

}

void EEBeamHodoClient::endRun(void) {

  if ( debug_ ) std::cout << "EEBeamHodoClient: endRun, jevt = " << jevt_ << std::endl;

  this->cleanup();

}

void EEBeamHodoClient::setup(void) {

  dqmStore_->setCurrentFolder( prefixME_ + "/EEBeamHodoClient" );

}

void EEBeamHodoClient::cleanup(void) {

  if ( ! enableCleanup_ ) return;

  dqmStore_->setCurrentFolder( prefixME_ + "/EEBeamHodoClient" );

}

#ifdef WITH_ECAL_COND_DB
bool EEBeamHodoClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov, bool& status) {

  status = true;

  return true;

}
#endif

void EEBeamHodoClient::analyze(void) {

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( debug_ ) std::cout << "EEBeamHodoClient: ievt/jevt = " << ievt_ << "/" << jevt_ << std::endl;
  }

  int smId = 1;

  MonitorElement* me;

  std::stringstream ss;

  for (int i=0; i<4; i++) {

    ss.str("");
    ss << prefixME_ << "/EEBeamHodoTask/EEBHT occup " << Numbers::sEE(smId).c_str() << " " << std::setfill('0') << std::setw(2) << i+1;
    me = dqmStore_->get( ss.str() );
    ho01_[i] = UtilsClient::getHisto( me, cloneME_, ho01_[i] );

    ss.str("");
    ss << prefixME_ << "/EEBeamHodoTask/EEBHT raw " << Numbers::sEE(smId) << " " << std::setfill('0') << std::setw(2) << i+1;
    me = dqmStore_->get(ss.str());
    hr01_[i] = UtilsClient::getHisto( me, cloneME_, hr01_[i] );

  }

  me = dqmStore_->get( prefixME_ + "/EEBeamHodoTask/EEBHT PosX rec " + Numbers::sEE(smId) );
  hp01_[0] = UtilsClient::getHisto( me, cloneME_, hp01_[0] );

  me = dqmStore_->get( prefixME_ + "/EEBeamHodoTask/EEBHT PosY rec " + Numbers::sEE(smId) );
  hp01_[1] = UtilsClient::getHisto( me, cloneME_, hp01_[1] );

  me = dqmStore_->get( prefixME_ + "/EEBeamHodoTask/EEBHT PosYX rec " + Numbers::sEE(smId) );
  hp02_ = UtilsClient::getHisto( me, cloneME_, hp02_ );

  me = dqmStore_->get( prefixME_ + "/EEBeamHodoTask/EEBHT SloX " + Numbers::sEE(smId) );
  hs01_[0] = UtilsClient::getHisto( me, cloneME_, hs01_[0] );

  me = dqmStore_->get( prefixME_ + "/EEBeamHodoTask/EEBHT SloY " + Numbers::sEE(smId) );
  hs01_[1] = UtilsClient::getHisto( me, cloneME_, hs01_[1] );

  me = dqmStore_->get( prefixME_ + "/EEBeamHodoTask/EEBHT QualX " + Numbers::sEE(smId) );
  hq01_[0] = UtilsClient::getHisto( me, cloneME_, hq01_[0] );

  me = dqmStore_->get( prefixME_ + "/EEBeamHodoTask/EEBHT QualY " + Numbers::sEE(smId) );
  hq01_[1] = UtilsClient::getHisto( me, cloneME_, hq01_[1] );

  me = dqmStore_->get( prefixME_ + "/EEBeamHodoTask/EEBHT TDC rec " + Numbers::sEE(smId) );
  ht01_ = UtilsClient::getHisto( me, cloneME_, ht01_ );

  me = dqmStore_->get( prefixME_ + "/EEBeamHodoTask/EEBHT Hodo-Calo X vs Cry " + Numbers::sEE(smId) );
  hc01_[0] = UtilsClient::getHisto( me, cloneME_, hc01_[0] );

  me = dqmStore_->get( prefixME_ + "/EEBeamHodoTask/EEBHT Hodo-Calo Y vs Cry " + Numbers::sEE(smId) );
  hc01_[1] = UtilsClient::getHisto( me, cloneME_, hc01_[1] );

  me = dqmStore_->get( prefixME_ + "/EEBeamHodoTask/EEBHT TDC-Calo vs Cry " + Numbers::sEE(smId) );
  hc01_[2] = UtilsClient::getHisto( me, cloneME_, hc01_[2] );

  me = dqmStore_->get( prefixME_ + "/EEBeamHodoTask/EEBHT Missing Collections " + Numbers::sEE(smId) );
  hm01_ = UtilsClient::getHisto( me, cloneME_, hm01_ );

  me = dqmStore_->get( prefixME_ + "/EEBeamHodoTask/EEBHT prof E1 vs X " + Numbers::sEE(smId) );
  he01_[0] = UtilsClient::getHisto( me, cloneME_, he01_[0] );

  me = dqmStore_->get( prefixME_ + "/EEBeamHodoTask/EEBHT prof E1 vs Y " + Numbers::sEE(smId) );
  he01_[1] = UtilsClient::getHisto( me, cloneME_, he01_[1] );

  me = dqmStore_->get( prefixME_ + "/EEBeamHodoTask/EEBHT his E1 vs X " + Numbers::sEE(smId) );
  he02_[0] = UtilsClient::getHisto( me, cloneME_, he02_[0] );

  me = dqmStore_->get( prefixME_ + "/EEBeamHodoTask/EEBHT his E1 vs Y " + Numbers::sEE(smId) );
  he02_[1] = UtilsClient::getHisto( me, cloneME_, he02_[1] );

  me = dqmStore_->get( prefixME_ + "/EEBeamHodoTask/EEBHT PosX Hodo-Calo " + Numbers::sEE(smId) );
  he03_[0] = UtilsClient::getHisto( me, cloneME_, he03_[0] );

  me = dqmStore_->get( prefixME_ + "/EEBeamHodoTask/EEBHT PosY Hodo-Calo " + Numbers::sEE(smId) );
  he03_[1] = UtilsClient::getHisto( me, cloneME_, he03_[1] );

  me = dqmStore_->get( prefixME_ + "/EEBeamHodoTask/EEBHT TimeMax TDC-Calo " + Numbers::sEE(smId) );
  he03_[2] = UtilsClient::getHisto( me, cloneME_, he03_[2] );

}

