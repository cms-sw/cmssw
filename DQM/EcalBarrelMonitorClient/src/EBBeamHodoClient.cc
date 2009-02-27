/*
 * \file EBBeamHodoClient.cc
 *
 * $Date: 2008/06/25 15:08:18 $
 * $Revision: 1.67 $
 * \author G. Della Ricca
 * \author G. Franzoni
 *
*/

#include <memory>
#include <iostream>
#include <fstream>

#include "DQMServices/Core/interface/DQMStore.h"

#include "DQM/EcalCommon/interface/UtilsClient.h"
#include "DQM/EcalCommon/interface/Numbers.h"

#include <DQM/EcalBarrelMonitorClient/interface/EBBeamHodoClient.h>

using namespace cms;
using namespace edm;
using namespace std;

EBBeamHodoClient::EBBeamHodoClient(const ParameterSet& ps) {

  // cloneME switch
  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);

  // verbose switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", true);

  // debug switch
  debug_ = ps.getUntrackedParameter<bool>("debug", false);

  // prefixME path
  prefixME_ = ps.getUntrackedParameter<string>("prefixME", "");

  // enableCleanup_ switch
  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  // vector of selected Super Modules (Defaults to all 36).
  superModules_.reserve(36);
  for ( unsigned int i = 1; i <= 36; i++ ) superModules_.push_back(i);
  superModules_ = ps.getUntrackedParameter<vector<int> >("superModules", superModules_);

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

void EBBeamHodoClient::beginJob(DQMStore* dqmStore) {

  dqmStore_ = dqmStore;

  if ( debug_ ) cout << "EBBeamHodoClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EBBeamHodoClient::beginRun(void) {

  if ( debug_ ) cout << "EBBeamHodoClient: beginRun" << endl;

  jevt_ = 0;

  this->setup();

}

void EBBeamHodoClient::endJob(void) {

  if ( debug_ ) cout << "EBBeamHodoClient: endJob, ievt = " << ievt_ << endl;

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

  if ( debug_ ) cout << "EBBeamHodoClient: endRun, jevt = " << jevt_ << endl;

  this->cleanup();

}

void EBBeamHodoClient::setup(void) {

  dqmStore_->setCurrentFolder( prefixME_ + "/EBBeamHodoClient" );

}

void EBBeamHodoClient::cleanup(void) {

  if ( ! enableCleanup_ ) return;

  dqmStore_->setCurrentFolder( prefixME_ + "/EBBeamHodoClient" );

}

bool EBBeamHodoClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov, bool& status, bool flag) {

  status = true;

  if ( ! flag ) return false;

  return true;

}

void EBBeamHodoClient::analyze(void) {

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( debug_ ) cout << "EBBeamHodoClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;
  }

  int smId = 1;

  char histo[200];

  MonitorElement* me;

  for (int i=0; i<4; i++) {

    sprintf(histo, (prefixME_ + "/EBBeamHodoTask/EBBHT occup %s %02d").c_str(), Numbers::sEB(smId).c_str(), i+1);
    me = dqmStore_->get(histo);
    ho01_[i] = UtilsClient::getHisto<TH1F*>( me, cloneME_, ho01_[i] );

    sprintf(histo, (prefixME_ + "/EBBeamHodoTask/EBBHT raw %s %02d").c_str(), Numbers::sEB(smId).c_str(), i+1);
    me = dqmStore_->get(histo);
    hr01_[i] = UtilsClient::getHisto<TH1F*>( me, cloneME_, hr01_[i] );

  }

  sprintf(histo, (prefixME_ + "/EBBeamHodoTask/EBBHT PosX rec %s").c_str(), Numbers::sEB(smId).c_str());
  me = dqmStore_->get(histo);
  hp01_[0] = UtilsClient::getHisto<TH1F*>( me, cloneME_, hp01_[0] );

  sprintf(histo, (prefixME_ + "/EBBeamHodoTask/EBBHT PosY rec %s").c_str(), Numbers::sEB(smId).c_str());
  me = dqmStore_->get(histo);
  hp01_[1] = UtilsClient::getHisto<TH1F*>( me, cloneME_, hp01_[1] );

  sprintf(histo, (prefixME_ + "/EBBeamHodoTask/EBBHT PosYX rec %s").c_str(), Numbers::sEB(smId).c_str());
  me = dqmStore_->get(histo);
  hp02_ = UtilsClient::getHisto<TH2F*>( me, cloneME_, hp02_ );

  sprintf(histo, (prefixME_ + "/EBBeamHodoTask/EBBHT SloX %s").c_str(), Numbers::sEB(smId).c_str());
  me = dqmStore_->get(histo);
  hs01_[0] = UtilsClient::getHisto<TH1F*>( me, cloneME_, hs01_[0] );

  sprintf(histo, (prefixME_ + "/EBBeamHodoTask/EBBHT SloY %s").c_str(), Numbers::sEB(smId).c_str());
  me = dqmStore_->get(histo);
  hs01_[1] = UtilsClient::getHisto<TH1F*>( me, cloneME_, hs01_[1] );

  sprintf(histo, (prefixME_ + "/EBBeamHodoTask/EBBHT QualX %s").c_str(), Numbers::sEB(smId).c_str());
  me = dqmStore_->get(histo);
  hq01_[0] = UtilsClient::getHisto<TH1F*>( me, cloneME_, hq01_[0] );

  sprintf(histo, (prefixME_ + "/EBBeamHodoTask/EBBHT QualY %s").c_str(), Numbers::sEB(smId).c_str());
  me = dqmStore_->get(histo);
  hq01_[1] = UtilsClient::getHisto<TH1F*>( me, cloneME_, hq01_[1] );

  sprintf(histo, (prefixME_ + "/EBBeamHodoTask/EBBHT TDC rec %s").c_str(), Numbers::sEB(smId).c_str());
  me = dqmStore_->get(histo);
  ht01_ = UtilsClient::getHisto<TH1F*>( me, cloneME_, ht01_ );

  sprintf(histo, (prefixME_ + "/EBBeamHodoTask/EBBHT Hodo-Calo X vs Cry %s").c_str(), Numbers::sEB(smId).c_str());
  me = dqmStore_->get(histo);
  hc01_[0] = UtilsClient::getHisto<TH1F*>( me, cloneME_, hc01_[0] );

  sprintf(histo, (prefixME_ + "/EBBeamHodoTask/EBBHT Hodo-Calo Y vs Cry %s").c_str(), Numbers::sEB(smId).c_str());
  me = dqmStore_->get(histo);
  hc01_[1] = UtilsClient::getHisto<TH1F*>( me, cloneME_, hc01_[1] );

  sprintf(histo, (prefixME_ + "/EBBeamHodoTask/EBBHT TDC-Calo vs Cry %s").c_str(), Numbers::sEB(smId).c_str());
  me = dqmStore_->get(histo);
  hc01_[2] = UtilsClient::getHisto<TH1F*>( me, cloneME_, hc01_[2] );

  sprintf(histo, (prefixME_ + "/EBBeamHodoTask/EBBHT Missing Collections %s").c_str(), Numbers::sEB(smId).c_str());
  me = dqmStore_->get(histo);
  hm01_ = UtilsClient::getHisto<TH1F*>( me, cloneME_, hm01_ );

  sprintf(histo, (prefixME_ + "/EBBeamHodoTask/EBBHT prof E1 vs X %s").c_str(), Numbers::sEB(smId).c_str());
  me = dqmStore_->get(histo);
  he01_[0] = UtilsClient::getHisto<TProfile*>( me, cloneME_, he01_[0] );

  sprintf(histo, (prefixME_ + "/EBBeamHodoTask/EBBHT prof E1 vs Y %s").c_str(), Numbers::sEB(smId).c_str());
  me = dqmStore_->get(histo);
  he01_[1] = UtilsClient::getHisto<TProfile*>( me, cloneME_, he01_[1] );

  sprintf(histo, (prefixME_ + "/EBBeamHodoTask/EBBHT his E1 vs X %s").c_str(), Numbers::sEB(smId).c_str());
  me = dqmStore_->get(histo);
  he02_[0] = UtilsClient::getHisto<TH2F*>( me, cloneME_, he02_[0] );

  sprintf(histo, (prefixME_ + "/EBBeamHodoTask/EBBHT his E1 vs Y %s").c_str(), Numbers::sEB(smId).c_str());
  me = dqmStore_->get(histo);
  he02_[1] = UtilsClient::getHisto<TH2F*>( me, cloneME_, he02_[1] );

  sprintf(histo, (prefixME_ + "/EBBeamHodoTask/EBBHT PosX Hodo-Calo %s").c_str(), Numbers::sEB(smId).c_str());
  me = dqmStore_->get(histo);
  he03_[0] = UtilsClient::getHisto<TH1F*>( me, cloneME_, he03_[0] );

  sprintf(histo, (prefixME_ + "/EBBeamHodoTask/EBBHT PosY Hodo-Calo %s").c_str(), Numbers::sEB(smId).c_str());
  me = dqmStore_->get(histo);
  he03_[1] = UtilsClient::getHisto<TH1F*>( me, cloneME_, he03_[1] );

  sprintf(histo, (prefixME_ + "/EBBeamHodoTask/EBBHT TimeMax TDC-Calo %s").c_str(), Numbers::sEB(smId).c_str());
  me = dqmStore_->get(histo);
  he03_[2] = UtilsClient::getHisto<TH1F*>( me, cloneME_, he03_[2] );

}

void EBBeamHodoClient::softReset(bool flag) {

}

