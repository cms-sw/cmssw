/*
 * \file EETriggerTowerClient.cc
 *
 * $Date: 2009/02/27 19:14:42 $
 * $Revision: 1.84 $
 * \author G. Della Ricca
 * \author F. Cossutti
 *
*/

#include <memory>
#include <iostream>
#include <fstream>
#include <iomanip>

#include "DQMServices/Core/interface/DQMStore.h"

#include "DQM/EcalCommon/interface/UtilsClient.h"
#include "DQM/EcalCommon/interface/Numbers.h"

#include <DataFormats/EcalDetId/interface/EEDetId.h>

#include <DQM/EcalEndcapMonitorClient/interface/EETriggerTowerClient.h>

using namespace cms;
using namespace edm;
using namespace std;

EETriggerTowerClient::EETriggerTowerClient(const ParameterSet& ps) {

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

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    l01_[ism-1] = 0;
    o01_[ism-1] = 0;

    mel01_[ism-1] = 0;
    meo01_[ism-1] = 0;

  }

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    me_o01_[ism-1] = 0;
    me_o02_[ism-1] = 0;

  }

}

EETriggerTowerClient::~EETriggerTowerClient() {

}

void EETriggerTowerClient::beginJob(DQMStore* dqmStore) {

  dqmStore_ = dqmStore;

  if ( debug_ ) cout << "EETriggerTowerClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EETriggerTowerClient::beginRun(void) {

  if ( debug_ ) cout << "EETriggerTowerClient: beginRun" << endl;

  jevt_ = 0;

  this->setup();

}

void EETriggerTowerClient::endJob(void) {

  if ( debug_ ) cout << "EETriggerTowerClient: endJob, ievt = " << ievt_ << endl;

  this->cleanup();

}

void EETriggerTowerClient::endRun(void) {

  if ( debug_ ) cout << "EETriggerTowerClient: endRun, jevt = " << jevt_ << endl;

  this->cleanup();

}

void EETriggerTowerClient::setup(void) {

  char histo[200];

  dqmStore_->setCurrentFolder( prefixME_ + "/EETriggerTowerClient" );

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( me_o01_[ism-1] ) dqmStore_->removeElement( me_o01_[ism-1]->getName() );
    sprintf(histo, "EETTT Trigger Primitives Timing %s", Numbers::sEE(ism).c_str());
    me_o01_[ism-1] = dqmStore_->book2D(histo, histo, 50, Numbers::ix0EE(ism)+0., Numbers::ix0EE(ism)+50., 50, Numbers::iy0EE(ism)+0., Numbers::iy0EE(ism)+50.);
    me_o01_[ism-1]->setAxisTitle("jx", 1);
    me_o01_[ism-1]->setAxisTitle("jy", 2);

    if ( me_o02_[ism-1] ) dqmStore_->removeElement( me_o02_[ism-1]->getName() );
    sprintf(histo, "EETTT Non Single Timing %s", Numbers::sEE(ism).c_str());
    me_o02_[ism-1] = dqmStore_->book2D(histo, histo, 50, Numbers::ix0EE(ism)+0., Numbers::ix0EE(ism)+50., 50, Numbers::iy0EE(ism)+0., Numbers::iy0EE(ism)+50.);
    me_o02_[ism-1]->setAxisTitle("ieta'", 1);
    me_o02_[ism-1]->setAxisTitle("iphi'", 2);
    me_o02_[ism-1]->setAxisTitle("fraction", 3);

  }

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( me_o01_[ism-1] ) me_o01_[ism-1]->Reset();
    if ( me_o02_[ism-1] ) me_o02_[ism-1]->Reset();

  }

}

void EETriggerTowerClient::cleanup(void) {

  if ( ! enableCleanup_ ) return;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( cloneME_ ) {
      if ( l01_[ism-1] ) delete l01_[ism-1];
      if ( o01_[ism-1] ) delete o01_[ism-1];
    }

    l01_[ism-1] = 0;
    o01_[ism-1] = 0;

    mel01_[ism-1] = 0;
    meo01_[ism-1] = 0;

  }

  dqmStore_->setCurrentFolder( prefixME_ + "/EETriggerTowerClient" );

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( me_o01_[ism-1] ) dqmStore_->removeElement( me_o01_[ism-1]->getName() );
    me_o01_[ism-1] = 0;
    if ( me_o02_[ism-1] ) dqmStore_->removeElement( me_o02_[ism-1]->getName() );
    me_o02_[ism-1] = 0;

  }

}

bool EETriggerTowerClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov, bool& status, bool flag) {

  status = true;

  if ( ! flag ) return false;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( verbose_ ) {
      cout << " " << Numbers::sEE(ism) << " (ism=" << ism << ")" << endl;
      cout << endl;
      UtilsClient::printBadChannels(mel01_[ism-1], UtilsClient::getHisto<TH2F*>(mel01_[ism-1]), true);
    }

  }

  return true;

}

void EETriggerTowerClient::analyze(void) {

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( debug_ ) cout << "EETriggerTowerClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;
  }

  char histo[200];

  MonitorElement* me;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    sprintf(histo, (prefixME_ + "/EETriggerTowerClient/EETTT EmulError %s").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    l01_[ism-1] = UtilsClient::getHisto<TH2F*>( me, cloneME_, l01_[ism-1] );
    mel01_[ism-1] = me;

    sprintf(histo, (prefixME_ + "/%s/EETTT EmulMatch %s").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    o01_[ism-1] = UtilsClient::getHisto<TH3F*>( me, cloneME_, o01_[ism-1] );
    meo01_[ism-1] = me;

    if ( me_o01_[ism-1] ) me_o01_[ism-1]->Reset();
    if ( me_o02_[ism-1] ) me_o02_[ism-1]->Reset();

    for (int ix = 1; ix <= 50; ix++) {
      for (int iy = 1; iy <= 50; iy++) {

        int jx = ix + Numbers::ix0EE(ism);
        int jy = iy + Numbers::iy0EE(ism);

        if ( o01_[ism-1] ) {
          // find the most frequent TP timing that matches the emulator
          float index=-1;
          double max=0;
          double total=0;
          for (int j=0; j<6; j++) {
            double sampleEntries = o01_[ism-1]->GetBinContent(jx, jy, j+1);
            if(sampleEntries > max) {
              index=j;
              max = sampleEntries;
            }
            total += sampleEntries;
          }
          if ( max > 0 ) {
            if ( index == 0 ) me_o01_[ism-1]->setBinContent(jx, jy, -1);
            else me_o01_[ism-1]->setBinContent(jx, jy, index );
          }
          double fraction = (total > 0) ? 1.0 - max/total : 0.;
          if ( me_o02_[ism-1] ) me_o02_[ism-1]->setBinContent(jx, jy, fraction);
        }

      }
    }

  }

}

void EETriggerTowerClient::softReset(bool flag) {

}

