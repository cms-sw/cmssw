/*
 * \file EEOccupancyClient.cc
 *
 * $Date: 2009/02/12 11:28:12 $
 * $Revision: 1.30 $
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

#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"

#include "DQM/EcalCommon/interface/UtilsClient.h"
#include "DQM/EcalCommon/interface/LogicID.h"
#include "DQM/EcalCommon/interface/Numbers.h"

#include <DQM/EcalEndcapMonitorClient/interface/EEOccupancyClient.h>

using namespace cms;
using namespace edm;
using namespace std;

EEOccupancyClient::EEOccupancyClient(const ParameterSet& ps) {

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

  // vector of selected Super Modules (Defaults to all 18).
  superModules_.reserve(18);
  for ( unsigned int i = 1; i <= 18; i++ ) superModules_.push_back(i);
  superModules_ = ps.getUntrackedParameter<vector<int> >("superModules", superModules_);

  for ( int i=0; i<3; i++) {
    h01_[0][i] = 0;
    h01ProjR_[0][i] = 0;
    h01ProjPhi_[0][i] = 0;
    h01_[1][i] = 0;
    h01ProjR_[1][i] = 0;
    h01ProjPhi_[1][i] = 0;
  }

  for ( int i=0; i<2; i++) {
    h02_[0][i] = 0;
    h02ProjR_[0][i] = 0;
    h02ProjPhi_[0][i] = 0;
    h02_[1][i] = 0;
    h02ProjR_[1][i] = 0;
    h02ProjPhi_[1][i] = 0;
  }

}

EEOccupancyClient::~EEOccupancyClient() {

}

void EEOccupancyClient::beginJob(DQMStore* dqmStore) {

  dqmStore_ = dqmStore;

  if ( debug_ ) cout << "EEOccupancyClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EEOccupancyClient::beginRun(void) {

  if ( debug_ ) cout << "EEOccupancyClient: beginRun" << endl;

  jevt_ = 0;

  this->setup();

}

void EEOccupancyClient::endJob(void) {

  if ( debug_ ) cout << "EEOccupancyClient: endJob, ievt = " << ievt_ << endl;

  this->cleanup();

}

void EEOccupancyClient::endRun(void) {

  if ( debug_ ) cout << "EEOccupancyClient: endRun, jevt = " << jevt_ << endl;

  this->cleanup();

}

void EEOccupancyClient::setup(void) {

  dqmStore_->setCurrentFolder( prefixME_ + "/EEOccupancyClient" );

}

void EEOccupancyClient::cleanup(void) {

  if ( ! enableCleanup_ ) return;

  if ( cloneME_ ) {

    for ( int i=0; i<3; ++i ) {
      if ( h01_[0][i] ) delete h01_[0][i];
      if ( h01ProjR_[0][i] ) delete h01ProjR_[0][i];
      if ( h01ProjPhi_[0][i] ) delete h01ProjPhi_[0][i];
      if ( h01_[1][i] ) delete h01_[1][i];
      if ( h01ProjR_[1][i] ) delete h01ProjR_[1][i];
      if ( h01ProjPhi_[1][i] ) delete h01ProjPhi_[1][i];
    }

    for ( int i=0; i<2; ++i ) {
      if ( h02_[0][i] ) delete h02_[0][i];
      if ( h02ProjR_[0][i] ) delete h02ProjR_[0][i];
      if ( h02ProjPhi_[0][i] ) delete h02ProjPhi_[0][i];
      if ( h01_[1][i] ) delete h01_[1][i];
      if ( h01ProjR_[1][i] ) delete h01ProjR_[1][i];
      if ( h01ProjPhi_[1][i] ) delete h01ProjPhi_[1][i];
    }

  }

  for ( int i=0; i<3; i++) {
    h01_[0][i] = 0;
    h01ProjR_[0][i] = 0;
    h01ProjPhi_[0][i] = 0;
    h01_[1][i] = 0;
    h01ProjR_[1][i] = 0;
    h01ProjPhi_[1][i] = 0;
  }

  for ( int i=0; i<2; i++) {
    h02_[0][i] = 0;
    h02ProjR_[0][i] = 0;
    h02ProjPhi_[0][i] = 0;
    h02_[1][i] = 0;
    h02ProjR_[1][i] = 0;
    h02ProjPhi_[1][i] = 0;
  }

}

bool EEOccupancyClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov, bool& status, bool flag) {

  status = true;

  if ( ! flag ) return false;

  return true;

}

void EEOccupancyClient::analyze(void) {

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( debug_ ) cout << "EEOccupancyClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;
  }

  char histo[200];

  MonitorElement* me;

  sprintf(histo, (prefixME_ + "/EEOccupancyTask/EEOT digi occupancy EE -").c_str());
  me = dqmStore_->get(histo);
  h01_[0][0] = UtilsClient::getHisto<TH2F*> ( me, cloneME_, h01_[0][0] );

  sprintf(histo, (prefixME_ + "/EEOccupancyTask/EEOT digi occupancy EE - projection R").c_str());
  me = dqmStore_->get(histo);
  h01ProjR_[0][0] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h01ProjR_[0][0] );

  sprintf(histo, (prefixME_ + "/EEOccupancyTask/EEOT digi occupancy EE - projection phi").c_str());
  me = dqmStore_->get(histo);
  h01ProjPhi_[0][0] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h01ProjPhi_[0][0] );

  sprintf(histo, (prefixME_ + "/EEOccupancyTask/EEOT digi occupancy EE +").c_str());
  me = dqmStore_->get(histo);
  h01_[1][0] = UtilsClient::getHisto<TH2F*> ( me, cloneME_, h01_[1][0] );

  sprintf(histo, (prefixME_ + "/EEOccupancyTask/EEOT digi occupancy EE + projection R").c_str());
  me = dqmStore_->get(histo);
  h01ProjR_[1][0] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h01ProjR_[1][0] );

  sprintf(histo, (prefixME_ + "/EEOccupancyTask/EEOT digi occupancy EE + projection phi").c_str());
  me = dqmStore_->get(histo);
  h01ProjPhi_[1][0] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h01ProjPhi_[1][0] );

  sprintf(histo, (prefixME_ + "/EEOccupancyTask/EEOT rec hit occupancy EE -").c_str());
  me = dqmStore_->get(histo);
  h01_[0][1] = UtilsClient::getHisto<TH2F*> ( me, cloneME_, h01_[0][1] );

  sprintf(histo, (prefixME_ + "/EEOccupancyTask/EEOT rec hit occupancy EE - projection R").c_str());
  me = dqmStore_->get(histo);
  h01ProjR_[0][1] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h01ProjR_[0][1] );

  sprintf(histo, (prefixME_ + "/EEOccupancyTask/EEOT rec hit occupancy EE - projection phi").c_str());
  me = dqmStore_->get(histo);
  h01ProjPhi_[0][1] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h01ProjPhi_[0][1] );

  sprintf(histo, (prefixME_ + "/EEOccupancyTask/EEOT rec hit occupancy EE +").c_str());
  me = dqmStore_->get(histo);
  h01_[1][1] = UtilsClient::getHisto<TH2F*> ( me, cloneME_, h01_[1][1] );

  sprintf(histo, (prefixME_ + "/EEOccupancyTask/EEOT rec hit occupancy EE + projection R").c_str());
  me = dqmStore_->get(histo);
  h01ProjR_[1][1] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h01ProjR_[1][1] );

  sprintf(histo, (prefixME_ + "/EEOccupancyTask/EEOT rec hit occupancy EE + projection phi").c_str());
  me = dqmStore_->get(histo);
  h01ProjPhi_[1][1] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h01ProjPhi_[1][1] );

  sprintf(histo, (prefixME_ + "/EEOccupancyTask/EEOT TP digi occupancy EE -").c_str());
  me = dqmStore_->get(histo);
  h01_[0][2] = UtilsClient::getHisto<TH2F*> ( me, cloneME_, h01_[0][2] );

  sprintf(histo, (prefixME_ + "/EEOccupancyTask/EEOT TP digi occupancy EE - projection R").c_str());
  me = dqmStore_->get(histo);
  h01ProjR_[0][2] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h01ProjR_[0][2] );

  sprintf(histo, (prefixME_ + "/EEOccupancyTask/EEOT TP digi occupancy EE - projection phi").c_str());
  me = dqmStore_->get(histo);
  h01ProjPhi_[0][2] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h01ProjPhi_[0][2] );

  sprintf(histo, (prefixME_ + "/EEOccupancyTask/EEOT TP digi occupancy EE +").c_str());
  me = dqmStore_->get(histo);
  h01_[1][2] = UtilsClient::getHisto<TH2F*> ( me, cloneME_, h01_[1][2] );

  sprintf(histo, (prefixME_ + "/EEOccupancyTask/EEOT TP digi occupancy EE + projection R").c_str());
  me = dqmStore_->get(histo);
  h01ProjR_[1][2] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h01ProjR_[1][2] );

  sprintf(histo, (prefixME_ + "/EEOccupancyTask/EEOT TP digi occupancy EE + projection phi").c_str());
  me = dqmStore_->get(histo);
  h01ProjPhi_[1][2] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h01ProjPhi_[1][2] );

  sprintf(histo, (prefixME_ + "/EEOccupancyTask/EEOT rec hit thr occupancy EE -").c_str());
  me = dqmStore_->get(histo);
  h02_[0][0] = UtilsClient::getHisto<TH2F*> ( me, cloneME_, h02_[0][0] );

  sprintf(histo, (prefixME_ + "/EEOccupancyTask/EEOT rec hit thr occupancy EE - projection R").c_str());
  me = dqmStore_->get(histo);
  h02ProjR_[0][0] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h02ProjR_[0][0] );

  sprintf(histo, (prefixME_ + "/EEOccupancyTask/EEOT rec hit thr occupancy EE - projection phi").c_str());
  me = dqmStore_->get(histo);
  h02ProjPhi_[0][0] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h02ProjPhi_[0][0] );

  sprintf(histo, (prefixME_ + "/EEOccupancyTask/EEOT rec hit thr occupancy EE +").c_str());
  me = dqmStore_->get(histo);
  h02_[1][0] = UtilsClient::getHisto<TH2F*> ( me, cloneME_, h02_[1][0] );

  sprintf(histo, (prefixME_ + "/EEOccupancyTask/EEOT rec hit thr occupancy EE + projection R").c_str());
  me = dqmStore_->get(histo);
  h02ProjR_[1][0] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h02ProjR_[1][0] );

  sprintf(histo, (prefixME_ + "/EEOccupancyTask/EEOT rec hit thr occupancy EE + projection phi").c_str());
  me = dqmStore_->get(histo);
  h02ProjPhi_[1][0] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h02ProjPhi_[1][0] );

  sprintf(histo, (prefixME_ + "/EEOccupancyTask/EEOT TP digi thr occupancy EE -").c_str());
  me = dqmStore_->get(histo);
  h02_[0][1] = UtilsClient::getHisto<TH2F*> ( me, cloneME_, h02_[0][1] );

  sprintf(histo, (prefixME_ + "/EEOccupancyTask/EEOT TP digi thr occupancy EE - projection R").c_str());
  me = dqmStore_->get(histo);
  h02ProjR_[0][1] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h02ProjR_[0][1] );

  sprintf(histo, (prefixME_ + "/EEOccupancyTask/EEOT TP digi thr occupancy EE - projection phi").c_str());
  me = dqmStore_->get(histo);
  h02ProjPhi_[0][1] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h02ProjPhi_[0][1] );

  sprintf(histo, (prefixME_ + "/EEOccupancyTask/EEOT TP digi thr occupancy EE +").c_str());
  me = dqmStore_->get(histo);
  h02_[1][1] = UtilsClient::getHisto<TH2F*> ( me, cloneME_, h02_[1][1] );

  sprintf(histo, (prefixME_ + "/EEOccupancyTask/EEOT TP digi thr occupancy EE + projection R").c_str());
  me = dqmStore_->get(histo);
  h02ProjR_[1][1] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h02ProjR_[1][1] );

  sprintf(histo, (prefixME_ + "/EEOccupancyTask/EEOT TP digi thr occupancy EE + projection phi").c_str());
  me = dqmStore_->get(histo);
  h02ProjPhi_[1][1] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h02ProjPhi_[1][1] );

}

void EEOccupancyClient::softReset(bool flag) {

}

