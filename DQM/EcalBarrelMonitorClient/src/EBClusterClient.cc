/*
 * \file EBClusterClient.cc
 *
 * $Date: 2008/08/05 15:37:22 $
 * $Revision: 1.69 $
 * \author G. Della Ricca
 * \author F. Cossutti
 * \author E. Di Marco
 *
*/

#include <memory>
#include <iostream>
#include <fstream>
#include <math.h>

#include "DQMServices/Core/interface/DQMStore.h"

#include "DQM/EcalCommon/interface/UtilsClient.h"

#include <DQM/EcalBarrelMonitorClient/interface/EBClusterClient.h>

using namespace cms;
using namespace edm;
using namespace std;

EBClusterClient::EBClusterClient(const ParameterSet& ps) {

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

void EBClusterClient::beginJob(DQMStore* dqmStore) {

  dqmStore_ = dqmStore;

  if ( debug_ ) cout << "EBClusterClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EBClusterClient::beginRun(void) {

  if ( debug_ ) cout << "EBClusterClient: beginRun" << endl;

  jevt_ = 0;

  this->setup();

}

void EBClusterClient::endJob(void) {

  if ( debug_ ) cout << "EBClusterClient: endJob, ievt = " << ievt_ << endl;

  this->cleanup();

}

void EBClusterClient::endRun(void) {

  if ( debug_ ) cout << "EBClusterClient: endRun, jevt = " << jevt_ << endl;

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

bool EBClusterClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov, bool& status, bool flag) {

  status = true;

  if ( ! flag ) return false;

  return true;

}

void EBClusterClient::analyze(void) {

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( debug_ ) cout << "EBClusterClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;
  }

  char histo[200];

  MonitorElement* me;

  sprintf(histo, (prefixME_ + "/EBClusterTask/EBCLT BC energy").c_str());
  me = dqmStore_->get(histo);
  h01_[0] = UtilsClient::getHisto<TH1F*>( me, cloneME_, h01_[0] );

  sprintf(histo, (prefixME_ + "/EBClusterTask/EBCLT BC size").c_str());
  me = dqmStore_->get(histo);
  h01_[1] = UtilsClient::getHisto<TH1F*>( me, cloneME_, h01_[1] );

  sprintf(histo, (prefixME_ + "/EBClusterTask/EBCLT BC number").c_str());
  me = dqmStore_->get(histo);
  h01_[2] = UtilsClient::getHisto<TH1F*>( me, cloneME_, h01_[2] );

  sprintf(histo, (prefixME_ + "/EBClusterTask/EBCLT BC energy map").c_str());
  me = dqmStore_->get(histo);
  h02_[0] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h02_[0] );

  sprintf(histo, (prefixME_ + "/EBClusterTask/EBCLT BC ET map").c_str());
  me = dqmStore_->get(histo);
  h02_[1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h02_[1] );

  sprintf(histo, (prefixME_ + "/EBClusterTask/EBCLT BC number map").c_str());
  me = dqmStore_->get(histo);
  h03_ = UtilsClient::getHisto<TH2F*>( me, cloneME_, h03_ );

  sprintf(histo, (prefixME_ + "/EBClusterTask/EBCLT BC size map").c_str());
  me = dqmStore_->get(histo);
  h04_ = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h04_ );

  sprintf(histo, (prefixME_ + "/EBClusterTask/EBCLT BC energy projection eta").c_str());
  me = dqmStore_->get(histo);
  h02ProjEta_[0] = UtilsClient::getHisto<TProfile*>( me, cloneME_, h02ProjEta_[0] );

  sprintf(histo, (prefixME_ + "/EBClusterTask/EBCLT BC energy projection phi").c_str());
  me = dqmStore_->get(histo);
  h02ProjPhi_[0] = UtilsClient::getHisto<TProfile*>( me, cloneME_, h02ProjPhi_[0] );

  sprintf(histo, (prefixME_ + "/EBClusterTask/EBCLT BC ET projection eta").c_str());
  me = dqmStore_->get(histo);
  h02ProjEta_[1] = UtilsClient::getHisto<TProfile*>( me, cloneME_, h02ProjEta_[1] );

  sprintf(histo, (prefixME_ + "/EBClusterTask/EBCLT BC ET projection phi").c_str());
  me = dqmStore_->get(histo);
  h02ProjPhi_[1] = UtilsClient::getHisto<TProfile*>( me, cloneME_, h02ProjPhi_[1] );

  sprintf(histo, (prefixME_ + "/EBClusterTask/EBCLT BC number projection eta").c_str());
  me = dqmStore_->get(histo);
  h03ProjEta_ = UtilsClient::getHisto<TH1F*>( me, cloneME_, h03ProjEta_ );

  sprintf(histo, (prefixME_ + "/EBClusterTask/EBCLT BC number projection phi").c_str());
  me = dqmStore_->get(histo);
  h03ProjPhi_ = UtilsClient::getHisto<TH1F*>( me, cloneME_, h03ProjPhi_ );

  sprintf(histo, (prefixME_ + "/EBClusterTask/EBCLT BC size projection eta").c_str());
  me = dqmStore_->get(histo);
  h04ProjEta_ = UtilsClient::getHisto<TProfile*>( me, cloneME_, h04ProjEta_ );

  sprintf(histo, (prefixME_ + "/EBClusterTask/EBCLT BC size projection phi").c_str());
  me = dqmStore_->get(histo);
  h04ProjPhi_ = UtilsClient::getHisto<TProfile*>( me, cloneME_, h04ProjPhi_ );

  sprintf(histo, (prefixME_ + "/EBClusterTask/EBCLT SC energy").c_str());
  me = dqmStore_->get(histo);
  i01_[0] = UtilsClient::getHisto<TH1F*>( me, cloneME_, i01_[0] );

  sprintf(histo, (prefixME_ + "/EBClusterTask/EBCLT SC size").c_str());
  me = dqmStore_->get(histo);
  i01_[1] = UtilsClient::getHisto<TH1F*>( me, cloneME_, i01_[1] );

  sprintf(histo, (prefixME_ + "/EBClusterTask/EBCLT SC number").c_str());
  me = dqmStore_->get(histo);
  i01_[2] = UtilsClient::getHisto<TH1F*>( me, cloneME_, i01_[2] );

  sprintf(histo, (prefixME_ + "/EBClusterTask/EBCLT s1s9").c_str());
  me = dqmStore_->get(histo);
  s01_[0] = UtilsClient::getHisto<TH1F*>( me, cloneME_, s01_[0] );

  sprintf(histo, (prefixME_ + "/EBClusterTask/EBCLT s9s25").c_str());
  me = dqmStore_->get(histo);
  s01_[1] = UtilsClient::getHisto<TH1F*>( me, cloneME_, s01_[1] );

  sprintf(histo, (prefixME_ + "/EBClusterTask/EBCLT dicluster invariant mass Pi0").c_str());
  me = dqmStore_->get(histo);
  s01_[2] = UtilsClient::getHisto<TH1F*>( me, cloneME_, s01_[2] );

}

void EBClusterClient::softReset(bool flag) {

}

