/*
 * \file EBCosmicClient.cc
 *
 * $Date: 2008/06/25 15:08:18 $
 * $Revision: 1.117 $
 * \author G. Della Ricca
 * \author F. Cossutti
 *
*/

#include <memory>
#include <iostream>
#include <fstream>
#include <iomanip>

#include "DQMServices/Core/interface/DQMStore.h"

#include "OnlineDB/EcalCondDB/interface/MonOccupancyDat.h"

#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"

#include "DQM/EcalCommon/interface/UtilsClient.h"
#include "DQM/EcalCommon/interface/LogicID.h"
#include "DQM/EcalCommon/interface/Numbers.h"

#include <DQM/EcalBarrelMonitorClient/interface/EBCosmicClient.h>

using namespace cms;
using namespace edm;
using namespace std;

EBCosmicClient::EBCosmicClient(const ParameterSet& ps) {

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
    h02_[ism-1] = 0;
    h03_[ism-1] = 0;
    h04_[ism-1] = 0;

    meh01_[ism-1] = 0;
    meh02_[ism-1] = 0;
    meh03_[ism-1] = 0;
    meh04_[ism-1] = 0;

  }

}

EBCosmicClient::~EBCosmicClient() {

}

void EBCosmicClient::beginJob(DQMStore* dqmStore) {

  dqmStore_ = dqmStore;

  if ( debug_ ) cout << "EBCosmicClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EBCosmicClient::beginRun(void) {

  if ( debug_ ) cout << "EBCosmicClient: beginRun" << endl;

  jevt_ = 0;

  this->setup();

}

void EBCosmicClient::endJob(void) {

  if ( debug_ ) cout << "EBCosmicClient: endJob, ievt = " << ievt_ << endl;

  this->cleanup();

}

void EBCosmicClient::endRun(void) {

  if ( debug_ ) cout << "EBCosmicClient: endRun, jevt = " << jevt_ << endl;

  this->cleanup();

}

void EBCosmicClient::setup(void) {

}

void EBCosmicClient::cleanup(void) {

  if ( ! enableCleanup_ ) return;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( cloneME_ ) {
      if ( h01_[ism-1] ) delete h01_[ism-1];
      if ( h02_[ism-1] ) delete h02_[ism-1];
      if ( h03_[ism-1] ) delete h03_[ism-1];
      if ( h04_[ism-1] ) delete h04_[ism-1];
    }

    h01_[ism-1] = 0;
    h02_[ism-1] = 0;
    h03_[ism-1] = 0;
    h04_[ism-1] = 0;

    meh01_[ism-1] = 0;
    meh02_[ism-1] = 0;
    meh03_[ism-1] = 0;
    meh04_[ism-1] = 0;

  }

}

bool EBCosmicClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov, bool& status, bool flag) {

  status = true;

  if ( ! flag ) return false;

  EcalLogicID ecid;

  MonOccupancyDat o;
  map<EcalLogicID, MonOccupancyDat> dataset;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( verbose_ ) {
      cout << " " << Numbers::sEB(ism) << " (ism=" << ism << ")" << endl;
      cout << endl;
    }

    const float n_min_tot = 1000.;
    const float n_min_bin = 10.;

    float num01, num02;
    float mean01, mean02;
    float rms01, rms02;

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

        num01  = num02  = -1.;
        mean01 = mean02 = -1.;
        rms01  = rms02  = -1.;

        bool update_channel = false;

        if ( h01_[ism-1] && h01_[ism-1]->GetEntries() >= n_min_tot ) {
          num01 = h01_[ism-1]->GetBinEntries(h01_[ism-1]->GetBin(ie, ip));
          if ( num01 >= n_min_bin ) {
            mean01 = h01_[ism-1]->GetBinContent(ie, ip);
            rms01  = h01_[ism-1]->GetBinError(ie, ip);
            update_channel = true;
          }
        }

        if ( h02_[ism-1] && h02_[ism-1]->GetEntries() >= n_min_tot ) {
          num02 = h02_[ism-1]->GetBinEntries(h02_[ism-1]->GetBin(ie, ip));
          if ( num02 >= n_min_bin ) {
            mean02 = h02_[ism-1]->GetBinContent(ie, ip);
            rms02  = h02_[ism-1]->GetBinError(ie, ip);
            update_channel = true;
          }
        }

        if ( update_channel ) {

          if ( Numbers::icEB(ism, ie, ip) == 1 ) {

            if ( verbose_ ) {
              cout << "Preparing dataset for " << Numbers::sEB(ism) << " (ism=" << ism << ")" << endl;
              cout << "Cut (" << ie << "," << ip << ") " << num01  << " " << mean01 << " " << rms01  << endl;
              cout << "Sel (" << ie << "," << ip << ") " << num02  << " " << mean02 << " " << rms02  << endl;
              cout << endl;
            }

          }

          o.setEventsOverHighThreshold(int(num01));
          o.setEventsOverLowThreshold(int(num02));

          o.setAvgEnergy(mean01);

          int ic = Numbers::indexEB(ism, ie, ip);

          if ( econn ) {
            ecid = LogicID::getEcalLogicID("EB_crystal_number", Numbers::iSM(ism, EcalBarrel), ic);
            dataset[ecid] = o;
          }

        }

      }
    }

  }

  if ( econn ) {
    try {
      if ( verbose_ ) cout << "Inserting MonOccupancyDat ..." << endl;
      if ( dataset.size() != 0 ) econn->insertDataArraySet(&dataset, moniov);
      if ( verbose_ ) cout << "done." << endl;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

  return true;

}

void EBCosmicClient::analyze(void) {

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( debug_ ) cout << "EBCosmicClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;
  }

  char histo[200];

  MonitorElement* me;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    sprintf(histo, (prefixME_ + "/EBCosmicTask/Cut/EBCT energy cut %s").c_str(), Numbers::sEB(ism).c_str());
    me = dqmStore_->get(histo);
    h01_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h01_[ism-1] );
    meh01_[ism-1] = me;

    sprintf(histo, (prefixME_ + "/EBCosmicTask/Sel/EBCT energy sel %s").c_str(), Numbers::sEB(ism).c_str());
    me = dqmStore_->get(histo);
    h02_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h02_[ism-1] );
    meh02_[ism-1] = me;

    sprintf(histo, (prefixME_ + "/EBCosmicTask/Spectrum/EBCT 1x1 energy spectrum %s").c_str(), Numbers::sEB(ism).c_str());
    me = dqmStore_->get(histo);
    h03_[ism-1] = UtilsClient::getHisto<TH1F*>( me, cloneME_, h03_[ism-1] );
    meh03_[ism-1] = me;

    sprintf(histo, (prefixME_ + "/EBCosmicTask/Spectrum/EBCT 3x3 energy spectrum %s").c_str(), Numbers::sEB(ism).c_str());
    me = dqmStore_->get(histo);
    h04_[ism-1] = UtilsClient::getHisto<TH1F*>( me, cloneME_, h04_[ism-1] );
    meh04_[ism-1] = me;

  }

}

void EBCosmicClient::softReset(bool flag) {

}

