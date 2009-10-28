/*
 * \file EEStatusFlagsClient.cc
 *
 * $Date: 2009/09/23 07:26:02 $
 * $Revision: 1.28 $
 * \author G. Della Ricca
 *
*/

#include <memory>
#include <iostream>
#include <fstream>
#include <iomanip>

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "OnlineDB/EcalCondDB/interface/RunTTErrorsDat.h"
#include "CondTools/Ecal/interface/EcalErrorDictionary.h"

#include "DQM/EcalCommon/interface/EcalErrorMask.h"
#include "DQM/EcalCommon/interface/LogicID.h"

#include "DQM/EcalCommon/interface/UtilsClient.h"
#include "DQM/EcalCommon/interface/Numbers.h"

#include <DQM/EcalEndcapMonitorClient/interface/EEStatusFlagsClient.h>

using namespace cms;
using namespace edm;
using namespace std;

EEStatusFlagsClient::EEStatusFlagsClient(const ParameterSet& ps) {

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

    h01_[ism-1] = 0;

    meh01_[ism-1] = 0;

    h02_[ism-1] = 0;

    meh02_[ism-1] = 0;

    h03_[ism-1] = 0;

    meh03_[ism-1] = 0;

  }

}

EEStatusFlagsClient::~EEStatusFlagsClient() {

}

void EEStatusFlagsClient::beginJob(void) {

  dqmStore_ = Service<DQMStore>().operator->();

  if ( debug_ ) cout << "EEStatusFlagsClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EEStatusFlagsClient::beginRun(void) {

  if ( debug_ ) cout << "EEStatusFlagsClient: beginRun" << endl;

  jevt_ = 0;

  this->setup();

}

void EEStatusFlagsClient::endJob(void) {

  if ( debug_ ) cout << "EEStatusFlagsClient: endJob, ievt = " << ievt_ << endl;

  this->cleanup();

}

void EEStatusFlagsClient::endRun(void) {

  if ( debug_ ) cout << "EEStatusFlagsClient: endRun, jevt = " << jevt_ << endl;

  this->cleanup();

}

void EEStatusFlagsClient::setup(void) {

  dqmStore_->setCurrentFolder( prefixME_ + "/EEStatusFlagsClient" );

}

void EEStatusFlagsClient::cleanup(void) {

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

  dqmStore_->setCurrentFolder( prefixME_ + "/EEStatusFlagsClient" );

}

bool EEStatusFlagsClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov, bool& status) {

  status = true;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( verbose_ ) {
      cout << " " << Numbers::sEE(ism) << " (ism=" << ism << ")" << endl;
      cout << endl;
      UtilsClient::printBadChannels(meh01_[ism-1], UtilsClient::getHisto<TH2F*>(meh01_[ism-1]), true);
    }

    if ( meh01_[ism-1] ) {
      if ( meh01_[ism-1]->getEntries() != 0 ) status = false;
    }

  }

  return true;

}

void EEStatusFlagsClient::analyze(void) {

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( debug_ ) cout << "EEStatusFlagsClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;
  }

  uint64_t bits01 = 0;
  bits01 |= EcalErrorDictionary::getMask("STATUS_FLAG_ERROR");

  map<EcalLogicID, RunTTErrorsDat> mask1;
  EcalErrorMask::fetchDataSet(&mask1);

  char histo[200];

  MonitorElement* me;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    sprintf(histo, (prefixME_ + "/EEStatusFlagsTask/FEStatus/EESFT front-end status %s").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    h01_[ism-1] = UtilsClient::getHisto<TH2F*>( me, cloneME_, h01_[ism-1] );
    meh01_[ism-1] = me;

    sprintf(histo, (prefixME_ + "/EEStatusFlagsTask/FEStatus/EESFT front-end status bits %s").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    h02_[ism-1] = UtilsClient::getHisto<TH1F*>( me, cloneME_, h02_[ism-1] );
    meh02_[ism-1] = me;

    sprintf(histo, (prefixME_ + "/EEStatusFlagsTask/FEStatus/EESFT MEM front-end status %s").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    h03_[ism-1] = UtilsClient::getHisto<TH2F*>( me, cloneME_, h01_[ism-1] );
    meh03_[ism-1] = me;    

    // TT masking
    
    for ( int ix = 1; ix <= 50; ix++ ) {
      for ( int iy = 1; iy <= 50; iy++ ) {
        
        int jx = ix + Numbers::ix0EE(ism);
        int jy = iy + Numbers::iy0EE(ism);

        if ( ism >= 1 && ism <= 9 ) jx = 101 - jx;

        if ( ! Numbers::validEE(ism, jx, jy) ) continue;

         if ( mask1.size() != 0 ) {
          map<EcalLogicID, RunTTErrorsDat>::const_iterator m;
          for (m = mask1.begin(); m != mask1.end(); m++) {

            EcalLogicID ecid = m->first;
      
            int itt = Numbers::iSC(ism, EcalEndcap, jx, jy);

            if ( itt > 70 ) continue;
            
            if ( itt >= 42 && itt <= 68 ) continue;

            if ( ( ism == 8 || ism == 17 ) && ( itt >= 18 && itt <= 24 ) ) continue;

            if ( ecid.getLogicID() == LogicID::getEcalLogicID("EE_readout_tower", Numbers::iSM(ism, EcalEndcap), itt).getLogicID() ) {
              if ( (m->second).getErrorBits() & bits01 ) {

                if ( itt >= 1 && itt <= 41 ) {
                  if ( meh01_[ism-1] ) meh01_[ism-1]->setBinError( ix, iy, 0.01 );
                } else if ( itt == 69 || itt == 70 ) {
                  if ( meh03_[ism-1] ) meh03_[ism-1]->setBinError( itt-68, 1, 0.01 );
                }

              }
            }
      
          }
        }
        
      }
    }

  }

}

