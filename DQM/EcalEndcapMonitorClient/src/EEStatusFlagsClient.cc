/*
 * \file EEStatusFlagsClient.cc
 *
 * $Date: 2010/04/14 16:13:40 $
 * $Revision: 1.40 $
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
#include "CondTools/Ecal/interface/EcalErrorDictionary.h"
#include "DQM/EcalCommon/interface/EcalErrorMask.h"
#include "DQM/EcalCommon/interface/LogicID.h"
#endif

#include "DQM/EcalCommon/interface/UtilsClient.h"
#include "DQM/EcalCommon/interface/Numbers.h"

#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include <DQM/EcalEndcapMonitorClient/interface/EEStatusFlagsClient.h>

EEStatusFlagsClient::EEStatusFlagsClient(const edm::ParameterSet& ps) {

  // cloneME switch
  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);

  // verbose switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", true);

  // debug switch
  debug_ = ps.getUntrackedParameter<bool>("debug", false);

  // prefixME path
  prefixME_ = ps.getUntrackedParameter<std::string>("prefixME", "");

  // enableCleanup_ switch
  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  // vector of selected Super Modules (Defaults to all 18).
  superModules_.reserve(18);
  for ( unsigned int i = 1; i <= 18; i++ ) superModules_.push_back(i);
  superModules_ = ps.getUntrackedParameter<std::vector<int> >("superModules", superModules_);

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

  dqmStore_ = edm::Service<DQMStore>().operator->();

  if ( debug_ ) std::cout << "EEStatusFlagsClient: beginJob" << std::endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EEStatusFlagsClient::beginRun(void) {

  if ( debug_ ) std::cout << "EEStatusFlagsClient: beginRun" << std::endl;

  jevt_ = 0;

  this->setup();

}

void EEStatusFlagsClient::endJob(void) {

  if ( debug_ ) std::cout << "EEStatusFlagsClient: endJob, ievt = " << ievt_ << std::endl;

  this->cleanup();

}

void EEStatusFlagsClient::endRun(void) {

  if ( debug_ ) std::cout << "EEStatusFlagsClient: endRun, jevt = " << jevt_ << std::endl;

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

#ifdef WITH_ECAL_COND_DB
bool EEStatusFlagsClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov, bool& status) {

  status = true;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( verbose_ ) {
      std::cout << " " << Numbers::sEE(ism) << " (ism=" << ism << ")" << std::endl;
      std::cout << std::endl;
      UtilsClient::printBadChannels(meh01_[ism-1], UtilsClient::getHisto<TH2F*>(meh01_[ism-1]), true);
    }

    if ( meh01_[ism-1] ) {
      if ( meh01_[ism-1]->getEntries() != 0 ) status = false;
    }

  }

  return true;

}
#endif

void EEStatusFlagsClient::analyze(void) {

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( debug_ ) std::cout << "EEStatusFlagsClient: ievt/jevt = " << ievt_ << "/" << jevt_ << std::endl;
  }

#ifdef WITH_ECAL_COND_DB
  uint64_t bits01 = 0;
  bits01 |= EcalErrorDictionary::getMask("STATUS_FLAG_ERROR");
#endif

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

  }

#ifdef WITH_ECAL_COND_DB
  if ( EcalErrorMask::mapTTErrors_.size() != 0 ) {
    map<EcalLogicID, RunTTErrorsDat>::const_iterator m;
    for (m = EcalErrorMask::mapTTErrors_.begin(); m != EcalErrorMask::mapTTErrors_.end(); m++) {

      if ( (m->second).getErrorBits() & bits01 ) {
        EcalLogicID ecid = m->first;

        if ( strcmp(ecid.getMapsTo().c_str(), "EE_readout_tower") != 0 ) continue;

        int idcc = ecid.getID1() - 600;

        int ism = -1;
        if ( idcc >=   1 && idcc <=   9 ) ism = idcc;
        if ( idcc >=  46 && idcc <=  54 ) ism = idcc - 45 + 9;
        std::vector<int>::iterator iter = find(superModules_.begin(), superModules_.end(), ism);
        if (iter == superModules_.end()) continue;

        int itt = ecid.getID2();

        if ( itt > 70 ) continue;

        if ( itt >= 42 && itt <= 68 ) continue;

        if ( ( ism == 8 || ism == 17 ) && ( itt >= 18 && itt <= 24 ) ) continue;

        if ( itt >= 1 && itt <= 68 ) {
          vector<DetId>* crystals = Numbers::crystals( idcc, itt );
          for ( unsigned int i=0; i<crystals->size(); i++ ) {
            EEDetId id = (*crystals)[i];
            int ix = id.ix();
            int iy = id.iy();
            if ( ism >= 1 && ism <= 9 ) ix = 101 - ix;
            int jx = ix - Numbers::ix0EE(ism);
            int jy = iy - Numbers::iy0EE(ism);
            if ( meh01_[ism-1] ) meh01_[ism-1]->setBinError( jx, jy, 0.01 );
          }
        } else if ( itt == 69 || itt == 70 ) {
          if ( meh03_[ism-1] ) meh03_[ism-1]->setBinError( itt-68, 1, 0.01 );
        }

      }

    }
  }
#endif

}

