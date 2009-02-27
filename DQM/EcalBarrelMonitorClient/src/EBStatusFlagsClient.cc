/*
 * \file EBStatusFlagsClient.cc
 *
 * $Date: 2008/06/25 15:08:18 $
 * $Revision: 1.23 $
 * \author G. Della Ricca
 *
*/

#include <memory>
#include <iostream>
#include <fstream>
#include <iomanip>

#include "DQMServices/Core/interface/DQMStore.h"

#include "DQM/EcalCommon/interface/UtilsClient.h"
#include "DQM/EcalCommon/interface/Numbers.h"

#include <DQM/EcalBarrelMonitorClient/interface/EBStatusFlagsClient.h>

using namespace cms;
using namespace edm;
using namespace std;

EBStatusFlagsClient::EBStatusFlagsClient(const ParameterSet& ps) {

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

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    h01_[ism-1] = 0;

    meh01_[ism-1] = 0;

    h02_[ism-1] = 0;

    meh02_[ism-1] = 0;

  }

}

EBStatusFlagsClient::~EBStatusFlagsClient() {

}

void EBStatusFlagsClient::beginJob(DQMStore* dqmStore) {

  dqmStore_ = dqmStore;

  if ( debug_ ) cout << "EBStatusFlagsClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EBStatusFlagsClient::beginRun(void) {

  if ( debug_ ) cout << "EBStatusFlagsClient: beginRun" << endl;

  jevt_ = 0;

  this->setup();

}

void EBStatusFlagsClient::endJob(void) {

  if ( debug_ ) cout << "EBStatusFlagsClient: endJob, ievt = " << ievt_ << endl;

  this->cleanup();

}

void EBStatusFlagsClient::endRun(void) {

  if ( debug_ ) cout << "EBStatusFlagsClient: endRun, jevt = " << jevt_ << endl;

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
    }

    h01_[ism-1] = 0;
    h02_[ism-1] = 0;

    meh01_[ism-1] = 0;
    meh02_[ism-1] = 0;

  }

  dqmStore_->setCurrentFolder( prefixME_ + "/EBStatusFlagsClient" );

}

bool EBStatusFlagsClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov, bool& status, bool flag) {

  status = true;

  if ( ! flag ) return false;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( verbose_ ) {
      cout << " " << Numbers::sEB(ism) << " (ism=" << ism << ")" << endl;
      cout << endl;
      UtilsClient::printBadChannels(meh01_[ism-1], UtilsClient::getHisto<TH2F*>(meh01_[ism-1]), true);
    }

    if ( meh01_[ism-1] ) {
      if ( meh01_[ism-1]->getEntries() != 0 ) status = false;
    }

  }

  return true;

}

void EBStatusFlagsClient::analyze(void) {

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( debug_ ) cout << "EBStatusFlagsClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;
  }

  char histo[200];

  MonitorElement* me;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    sprintf(histo, (prefixME_ + "/EBStatusFlagsTask/FEStatus/EBSFT front-end status %s").c_str(), Numbers::sEB(ism).c_str());
    me = dqmStore_->get(histo);
    h01_[ism-1] = UtilsClient::getHisto<TH2F*>( me, cloneME_, h01_[ism-1] );
    meh01_[ism-1] = me;

    sprintf(histo, (prefixME_ + "/EBStatusFlagsTask/FEStatus/EBSFT front-end status bits %s").c_str(), Numbers::sEB(ism).c_str());
    me = dqmStore_->get(histo);
    h02_[ism-1] = UtilsClient::getHisto<TH1F*>( me, cloneME_, h02_[ism-1] );
    meh02_[ism-1] = me;

  }

}

void EBStatusFlagsClient::softReset(bool flag) {

}

