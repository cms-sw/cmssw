/*
 * \file EBOccupancyClient.cc
 *
 * $Date: 2009/01/29 11:17:45 $
 * $Revision: 1.31 $
 * \author G. Della Ricca
 * \author F. Cossutti
 *
*/

#include <memory>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <math.h>

#include "DQMServices/Core/interface/DQMStore.h"

#include "DQM/EcalCommon/interface/UtilsClient.h"
#include "DQM/EcalCommon/interface/LogicID.h"
#include "DQM/EcalCommon/interface/Numbers.h"

#include <DQM/EcalBarrelMonitorClient/interface/EBOccupancyClient.h>

using namespace cms;
using namespace edm;
using namespace std;

EBOccupancyClient::EBOccupancyClient(const ParameterSet& ps) {

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

  for ( int i=0; i<3; i++) {
    h01_[i] = 0;
    h01ProjEta_[i] = 0;
    h01ProjPhi_[i] = 0;
  }

  for ( int i=0; i<2; i++) {
    h02_[i] = 0;
    h02ProjEta_[i] = 0;
    h02ProjPhi_[i] = 0;
  }

}

EBOccupancyClient::~EBOccupancyClient() {

}

void EBOccupancyClient::beginJob(DQMStore* dqmStore) {

  dqmStore_ = dqmStore;

  if ( debug_ ) cout << "EBOccupancyClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EBOccupancyClient::beginRun(void) {

  if ( debug_ ) cout << "EBOccupancyClient: beginRun" << endl;

  jevt_ = 0;

  this->setup();

}

void EBOccupancyClient::endJob(void) {

  if ( debug_ ) cout << "EBOccupancyClient: endJob, ievt = " << ievt_ << endl;

  this->cleanup();

}

void EBOccupancyClient::endRun(void) {

  if ( debug_ ) cout << "EBOccupancyClient: endRun, jevt = " << jevt_ << endl;

  this->cleanup();

}

void EBOccupancyClient::setup(void) {

  dqmStore_->setCurrentFolder( prefixME_ + "/EBOccupancyClient" );

}

void EBOccupancyClient::cleanup(void) {

  if ( ! enableCleanup_ ) return;

  if ( cloneME_ ) {

    for ( int i=0; i<3; ++i ) {
      if ( h01_[i] ) delete h01_[i];
      if ( h01ProjEta_[i] ) delete h01ProjEta_[i];
      if ( h01ProjPhi_[i] ) delete h01ProjPhi_[i];
    }

    for ( int i=0; i<2; ++i ) {
      if ( h02_[i] ) delete h02_[i];
      if ( h02ProjEta_[i] ) delete h02ProjEta_[i];
      if ( h02ProjPhi_[i] ) delete h02ProjPhi_[i];
    }

  }

  for ( int i=0; i<3; ++i ) {
    h01_[i] = 0;
    h01ProjEta_[i] = 0;
    h01ProjPhi_[i] = 0;
  }

  for ( int i=0; i<2; ++i ) {
    h02_[i] = 0;
    h02ProjEta_[i] = 0;
    h02ProjPhi_[i] = 0;
  }

}

bool EBOccupancyClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov, bool& status, bool flag) {

  status = true;

  if ( ! flag ) return false;

  return true;

}

void EBOccupancyClient::analyze(void) {

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( debug_ ) cout << "EBOccupancyClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;
  }

  char histo[200];

  MonitorElement* me;

  sprintf(histo, (prefixME_ + "/EBOccupancyTask/EBOT digi occupancy").c_str());
  me = dqmStore_->get(histo);
  h01_[0] = UtilsClient::getHisto<TH2F*> ( me, cloneME_, h01_[0] );

  sprintf(histo, (prefixME_ + "/EBOccupancyTask/EBOT digi occupancy projection eta").c_str());
  me = dqmStore_->get(histo);
  h01ProjEta_[0] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h01ProjEta_[0] );

  sprintf(histo, (prefixME_ + "/EBOccupancyTask/EBOT digi occupancy projection phi").c_str());
  me = dqmStore_->get(histo);
  h01ProjPhi_[0] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h01ProjPhi_[0] );

  sprintf(histo, (prefixME_ + "/EBOccupancyTask/EBOT rec hit occupancy").c_str());
  me = dqmStore_->get(histo);
  h01_[1] = UtilsClient::getHisto<TH2F*> ( me, cloneME_, h01_[1] );

  sprintf(histo, (prefixME_ + "/EBOccupancyTask/EBOT rec hit occupancy projection eta").c_str());
  me = dqmStore_->get(histo);
  h01ProjEta_[1] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h01ProjEta_[1] );

  sprintf(histo, (prefixME_ + "/EBOccupancyTask/EBOT rec hit occupancy projection phi").c_str());
  me = dqmStore_->get(histo);
  h01ProjPhi_[1] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h01ProjPhi_[1] );

  sprintf(histo, (prefixME_ + "/EBOccupancyTask/EBOT TP digi occupancy").c_str());
  me = dqmStore_->get(histo);
  h01_[2] = UtilsClient::getHisto<TH2F*> ( me, cloneME_, h01_[2] );

  sprintf(histo, (prefixME_ + "/EBOccupancyTask/EBOT TP digi occupancy projection eta").c_str());
  me = dqmStore_->get(histo);
  h01ProjEta_[2] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h01ProjEta_[2] );

  sprintf(histo, (prefixME_ + "/EBOccupancyTask/EBOT TP digi occupancy projection phi").c_str());
  me = dqmStore_->get(histo);
  h01ProjPhi_[2] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h01ProjPhi_[2] );

  sprintf(histo, (prefixME_ + "/EBOccupancyTask/EBOT rec hit thr occupancy").c_str());
  me = dqmStore_->get(histo);
  h02_[0] = UtilsClient::getHisto<TH2F*> ( me, cloneME_, h02_[0] );

  sprintf(histo, (prefixME_ + "/EBOccupancyTask/EBOT rec hit thr occupancy projection eta").c_str());
  me = dqmStore_->get(histo);
  h02ProjEta_[0] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h02ProjEta_[0] );

  sprintf(histo, (prefixME_ + "/EBOccupancyTask/EBOT rec hit thr occupancy projection phi").c_str());
  me = dqmStore_->get(histo);
  h02ProjPhi_[0] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h02ProjPhi_[0] );

  sprintf(histo, (prefixME_ + "/EBOccupancyTask/EBOT TP digi thr occupancy").c_str());
  me = dqmStore_->get(histo);
  h02_[1] = UtilsClient::getHisto<TH2F*> ( me, cloneME_, h02_[1] );

  sprintf(histo, (prefixME_ + "/EBOccupancyTask/EBOT TP digi thr occupancy projection eta").c_str());
  me = dqmStore_->get(histo);
  h02ProjEta_[1] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h02ProjEta_[1] );

  sprintf(histo, (prefixME_ + "/EBOccupancyTask/EBOT TP digi thr occupancy projection phi").c_str());
  me = dqmStore_->get(histo);
  h02ProjPhi_[1] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h02ProjPhi_[1] );

}

void EBOccupancyClient::softReset(bool flag) {

}

