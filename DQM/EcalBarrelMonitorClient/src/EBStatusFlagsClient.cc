/*
 * \file EBStatusFlagsClient.cc
 *
 * $Date: 2010/02/15 14:14:14 $
 * $Revision: 1.32 $
 * \author G. Della Ricca
 *
*/

#include <memory>
#include <iostream>
#include <fstream>
#include <iomanip>

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"

#ifdef WITH_ECAL_COND_DB
#include "OnlineDB/EcalCondDB/interface/RunTTErrorsDat.h"
#endif

#include "CondTools/Ecal/interface/EcalErrorDictionary.h"

#include "DQM/EcalCommon/interface/EcalErrorMask.h"
#include "DQM/EcalCommon/interface/LogicID.h"

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

    h03_[ism-1] = 0;

    meh03_[ism-1] = 0;

  }

}

EBStatusFlagsClient::~EBStatusFlagsClient() {

}

void EBStatusFlagsClient::beginJob(void) {

  dqmStore_ = Service<DQMStore>().operator->();

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
      if ( h03_[ism-1] ) delete h03_[ism-1];
    }

    h01_[ism-1] = 0;
    h02_[ism-1] = 0;
    h03_[ism-1] = 0;

    meh01_[ism-1] = 0;
    meh02_[ism-1] = 0;
    meh03_[ism-1] = 0;

  }

  dqmStore_->setCurrentFolder( prefixME_ + "/EBStatusFlagsClient" );

}

#ifdef WITH_ECAL_COND_DB
bool EBStatusFlagsClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov, bool& status) {

  status = true;

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
#endif

void EBStatusFlagsClient::analyze(void) {

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( debug_ ) cout << "EBStatusFlagsClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;
  }

  uint64_t bits01 = 0;
  bits01 |= EcalErrorDictionary::getMask("STATUS_FLAG_ERROR");

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

    sprintf(histo, (prefixME_ + "/EBStatusFlagsTask/FEStatus/EBSFT MEM front-end status %s").c_str(), Numbers::sEB(ism).c_str());
    me = dqmStore_->get(histo);
    h03_[ism-1] = UtilsClient::getHisto<TH2F*>( me, cloneME_, h01_[ism-1] );
    meh03_[ism-1] = me;    

  }

#ifdef WITH_ECAL_COND_DB
  if ( EcalErrorMask::mapTTErrors_.size() != 0 ) {
    map<EcalLogicID, RunTTErrorsDat>::const_iterator m;
    for (m = EcalErrorMask::mapTTErrors_.begin(); m != EcalErrorMask::mapTTErrors_.end(); m++) {

      if ( (m->second).getErrorBits() & bits01 ) {
        EcalLogicID ecid = m->first;

        if ( strcmp(ecid.getMapsTo().c_str(), "EB_readout_tower") != 0 ) continue;

        int ism = Numbers::iSM(ecid.getID1(), EcalBarrel);
        int itt = ecid.getID2();

        int iet = (itt-1)/4 + 1;
        int ipt = (itt-1)%4 + 1;

        if ( itt > 70 ) continue;

        if ( itt >= 1 && itt <= 68 ) {
          if ( meh01_[ism-1] ) meh01_[ism-1]->setBinError( iet, ipt, 0.01 );
        } else if ( itt == 69 || itt == 70 ) {
          if ( meh03_[ism-1] ) meh03_[ism-1]->setBinError( itt-68, 1, 0.01 );
        }

      }

    }
  }
#endif

}

