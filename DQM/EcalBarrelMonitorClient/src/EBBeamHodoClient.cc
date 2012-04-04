/*
 * \file EBBeamHodoClient.cc
 *
 * $Date: 2011/08/30 09:33:51 $
 * $Revision: 1.74 $
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

#include "DQM/EcalBarrelMonitorClient/interface/EBBeamHodoClient.h"

EBBeamHodoClient::EBBeamHodoClient(const edm::ParameterSet& ps) {

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

EBBeamHodoClient::~EBBeamHodoClient() {

}

void EBBeamHodoClient::beginJob(void) {

  dqmStore_ = edm::Service<DQMStore>().operator->();

  if ( debug_ ) std::cout << "EBBeamHodoClient: beginJob" << std::endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EBBeamHodoClient::beginRun(void) {

  if ( debug_ ) std::cout << "EBBeamHodoClient: beginRun" << std::endl;

  jevt_ = 0;

  this->setup();

}

void EBBeamHodoClient::endJob(void) {

  if ( debug_ ) std::cout << "EBBeamHodoClient: endJob, ievt = " << ievt_ << std::endl;

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

void EBBeamHodoClient::endRun(void) {

  if ( debug_ ) std::cout << "EBBeamHodoClient: endRun, jevt = " << jevt_ << std::endl;

  this->cleanup();

}

void EBBeamHodoClient::setup(void) {

  dqmStore_->setCurrentFolder( prefixME_ + "/EBBeamHodoClient" );

}

void EBBeamHodoClient::cleanup(void) {

  if ( ! enableCleanup_ ) return;

  dqmStore_->setCurrentFolder( prefixME_ + "/EBBeamHodoClient" );

}

#ifdef WITH_ECAL_COND_DB
bool EBBeamHodoClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov, bool& status) {

  status = true;

  return true;

}
#endif

void EBBeamHodoClient::analyze(void) {

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( debug_ ) std::cout << "EBBeamHodoClient: ievt/jevt = " << ievt_ << "/" << jevt_ << std::endl;
  }

  int smId = 1;

  std::stringstream ss;

  MonitorElement* me;

  for (int i=0; i<4; i++) {

    ss.str("");
    ss << prefixME_ << "/EBBeamHodoTask/EBBHT occup " << Numbers::sEB(smId) << std::setfill('0') << std::setw(2) << i+1;
    me = dqmStore_->get(ss.str());
    ho01_[i] = UtilsClient::getHisto( me, cloneME_, ho01_[i] );

    ss.str("");
    ss << prefixME_ << "/EBBeamHodoTask/EBBHT raw " << Numbers::sEB(smId) << std::setfill('0') << std::setw(2) << i+1;
    me = dqmStore_->get(ss.str());
    hr01_[i] = UtilsClient::getHisto( me, cloneME_, hr01_[i] );

  }

  me = dqmStore_->get( prefixME_ + "/EBBeamHodoTask/EBBHT PosX rec " + Numbers::sEB(smId) );
  hp01_[0] = UtilsClient::getHisto( me, cloneME_, hp01_[0] );

  me = dqmStore_->get( prefixME_ + "/EBBeamHodoTask/EBBHT PosY rec " + Numbers::sEB(smId) );
  hp01_[1] = UtilsClient::getHisto( me, cloneME_, hp01_[1] );

  me = dqmStore_->get( prefixME_ + "/EBBeamHodoTask/EBBHT PosYX rec " + Numbers::sEB(smId) );
  hp02_ = UtilsClient::getHisto( me, cloneME_, hp02_ );

  me = dqmStore_->get( prefixME_ + "/EBBeamHodoTask/EBBHT SloX " + Numbers::sEB(smId) );
  hs01_[0] = UtilsClient::getHisto( me, cloneME_, hs01_[0] );

  me = dqmStore_->get( prefixME_ + "/EBBeamHodoTask/EBBHT SloY " + Numbers::sEB(smId) );
  hs01_[1] = UtilsClient::getHisto( me, cloneME_, hs01_[1] );

  me = dqmStore_->get( prefixME_ + "/EBBeamHodoTask/EBBHT QualX " + Numbers::sEB(smId) );
  hq01_[0] = UtilsClient::getHisto( me, cloneME_, hq01_[0] );

  me = dqmStore_->get( prefixME_ + "/EBBeamHodoTask/EBBHT QualY " + Numbers::sEB(smId) );
  hq01_[1] = UtilsClient::getHisto( me, cloneME_, hq01_[1] );

  me = dqmStore_->get( prefixME_ + "/EBBeamHodoTask/EBBHT TDC rec " + Numbers::sEB(smId) );
  ht01_ = UtilsClient::getHisto( me, cloneME_, ht01_ );

  me = dqmStore_->get( prefixME_ + "/EBBeamHodoTask/EBBHT Hodo-Calo X vs Cry " + Numbers::sEB(smId) );
  hc01_[0] = UtilsClient::getHisto( me, cloneME_, hc01_[0] );

  me = dqmStore_->get( prefixME_ + "/EBBeamHodoTask/EBBHT Hodo-Calo Y vs Cry " + Numbers::sEB(smId) );
  hc01_[1] = UtilsClient::getHisto( me, cloneME_, hc01_[1] );

  me = dqmStore_->get( prefixME_ + "/EBBeamHodoTask/EBBHT TDC-Calo vs Cry " + Numbers::sEB(smId) );
  hc01_[2] = UtilsClient::getHisto( me, cloneME_, hc01_[2] );

  me = dqmStore_->get( prefixME_ + "/EBBeamHodoTask/EBBHT Missing Collections " + Numbers::sEB(smId) );
  hm01_ = UtilsClient::getHisto( me, cloneME_, hm01_ );

  me = dqmStore_->get( prefixME_ + "/EBBeamHodoTask/EBBHT prof E1 vs X " + Numbers::sEB(smId) );
  he01_[0] = UtilsClient::getHisto( me, cloneME_, he01_[0] );

  me = dqmStore_->get( prefixME_ + "/EBBeamHodoTask/EBBHT prof E1 vs Y " + Numbers::sEB(smId) );
  he01_[1] = UtilsClient::getHisto( me, cloneME_, he01_[1] );

  me = dqmStore_->get( prefixME_ + "/EBBeamHodoTask/EBBHT his E1 vs X " + Numbers::sEB(smId) );
  he02_[0] = UtilsClient::getHisto( me, cloneME_, he02_[0] );

  me = dqmStore_->get( prefixME_ + "/EBBeamHodoTask/EBBHT his E1 vs Y " + Numbers::sEB(smId) );
  he02_[1] = UtilsClient::getHisto( me, cloneME_, he02_[1] );

  me = dqmStore_->get( prefixME_ + "/EBBeamHodoTask/EBBHT PosX Hodo-Calo " + Numbers::sEB(smId) );
  he03_[0] = UtilsClient::getHisto( me, cloneME_, he03_[0] );

  me = dqmStore_->get( prefixME_ + "/EBBeamHodoTask/EBBHT PosY Hodo-Calo " + Numbers::sEB(smId) );
  he03_[1] = UtilsClient::getHisto( me, cloneME_, he03_[1] );

  me = dqmStore_->get( prefixME_ + "/EBBeamHodoTask/EBBHT TimeMax TDC-Calo " + Numbers::sEB(smId) );
  he03_[2] = UtilsClient::getHisto( me, cloneME_, he03_[2] );

}

