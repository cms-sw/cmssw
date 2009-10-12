/*
 * \file EBTriggerTowerClient.cc
 *
 * $Date: 2009/08/23 20:44:31 $
 * $Revision: 1.119 $
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

#include <DataFormats/EcalDetId/interface/EBDetId.h>

#include <DQM/EcalBarrelMonitorClient/interface/EBTriggerTowerClient.h>

using namespace cms;
using namespace edm;
using namespace std;

EBTriggerTowerClient::EBTriggerTowerClient(const ParameterSet& ps) {

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

EBTriggerTowerClient::~EBTriggerTowerClient() {

}

void EBTriggerTowerClient::beginJob(DQMStore* dqmStore) {

  dqmStore_ = dqmStore;

  if ( debug_ ) cout << "EBTriggerTowerClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EBTriggerTowerClient::beginRun(void) {

  if ( debug_ ) cout << "EBTriggerTowerClient: beginRun" << endl;

  jevt_ = 0;

  this->setup();

}

void EBTriggerTowerClient::endJob(void) {

  if ( debug_ ) cout << "EBTriggerTowerClient: endJob, ievt = " << ievt_ << endl;

  this->cleanup();

}

void EBTriggerTowerClient::endRun(void) {

  if ( debug_ ) cout << "EBTriggerTowerClient: endRun, jevt = " << jevt_ << endl;

  this->cleanup();

}

void EBTriggerTowerClient::setup(void) {

  char histo[200];

  dqmStore_->setCurrentFolder( prefixME_ + "/EBTriggerTowerClient" );

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( me_o01_[ism-1] ) dqmStore_->removeElement( me_o01_[ism-1]->getName() );
    sprintf(histo, "EBTTT Trigger Primitives Timing %s", Numbers::sEB(ism).c_str());
    me_o01_[ism-1] = dqmStore_->book2D(histo, histo, 17, 0., 17., 4, 0., 4.);
    me_o01_[ism-1]->setAxisTitle("ieta'", 1);
    me_o01_[ism-1]->setAxisTitle("iphi'", 2);

    if ( me_o02_[ism-1] ) dqmStore_->removeElement( me_o02_[ism-1]->getName() );
    sprintf(histo, "EBTTT Non Single Timing %s", Numbers::sEB(ism).c_str());
    me_o02_[ism-1] = dqmStore_->book2D(histo, histo, 17, 0., 17., 4, 0., 4.);
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

void EBTriggerTowerClient::cleanup(void) {

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

  dqmStore_->setCurrentFolder( prefixME_ + "/EBTriggerTowerClient" );

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( me_o01_[ism-1] ) dqmStore_->removeElement( me_o01_[ism-1]->getName() );
    me_o01_[ism-1] = 0;
    if ( me_o02_[ism-1] ) dqmStore_->removeElement( me_o02_[ism-1]->getName() );
    me_o02_[ism-1] = 0;

  }

}

bool EBTriggerTowerClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov, bool& status) {

  status = true;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( verbose_ ) {
      cout << " " << Numbers::sEB(ism) << " (ism=" << ism << ")" << endl;
      cout << endl;
      UtilsClient::printBadChannels(mel01_[ism-1], UtilsClient::getHisto<TH2F*>(mel01_[ism-1]), true);
    }

  }

  return true;

}

void EBTriggerTowerClient::analyze(void) {

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( debug_ ) cout << "EBTriggerTowerClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;
  }

  char histo[200];

  MonitorElement* me;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    sprintf(histo, (prefixME_ + "/EBTriggerTowerTask/EBTTT EmulError %s").c_str(), Numbers::sEB(ism).c_str());
    me = dqmStore_->get(histo);
    l01_[ism-1] = UtilsClient::getHisto<TH2F*>( me, cloneME_, l01_[ism-1] );
    mel01_[ism-1] = me;

    sprintf(histo, (prefixME_ + "/EBTriggerTowerTask/EBTTT EmulMatch %s").c_str(), Numbers::sEB(ism).c_str());
    me = dqmStore_->get(histo);
    o01_[ism-1] = UtilsClient::getHisto<TH3F*>( me, cloneME_, o01_[ism-1] );
    meo01_[ism-1] = me;

    if ( me_o01_[ism-1] ) me_o01_[ism-1]->Reset();
    if ( me_o02_[ism-1] ) me_o02_[ism-1]->Reset();

    for (int ie = 1; ie <= 17; ie++) {
      for (int ip = 1; ip <= 4; ip++) {

        if ( o01_[ism-1] ) {
          // find the most frequent TP timing that matches the emulator
          float index=-1;
          double max=0;
          double total=0;
          for (int j=0; j<6; j++) {
            double sampleEntries = o01_[ism-1]->GetBinContent(ie, ip, j+1);
            if(sampleEntries > max) {
              index=j;
              max = sampleEntries;
            }
            total += sampleEntries;
          }
          if ( max > 0 ) {
            if ( index == 0 ) {
              me_o01_[ism-1]->setBinContent(ie, ip, -1);
            } else {
              me_o01_[ism-1]->setBinContent(ie, ip, index );
            }
          }
          double fraction = (total > 0) ? 1.0 - max/total : 0.;
          if ( me_o02_[ism-1] ) me_o02_[ism-1]->setBinContent(ie, ip, fraction);
        }

      }
    }

  }

}

