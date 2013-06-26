/*
 * \file EEPedestalClient.cc
 *
 * $Date: 2012/04/27 13:46:07 $
 * $Revision: 1.125 $
 * \author G. Della Ricca
 * \author F. Cossutti
 *
*/

#include <memory>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <math.h>

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#ifdef WITH_ECAL_COND_DB
#include "OnlineDB/EcalCondDB/interface/MonPedestalsDat.h"
#include "OnlineDB/EcalCondDB/interface/MonPNPedDat.h"
#include "OnlineDB/EcalCondDB/interface/RunCrystalErrorsDat.h"
#include "OnlineDB/EcalCondDB/interface/RunTTErrorsDat.h"
#include "OnlineDB/EcalCondDB/interface/RunPNErrorsDat.h"
#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"
#include "DQM/EcalCommon/interface/LogicID.h"
#endif

#include "DQM/EcalCommon/interface/Masks.h"

#include "DQM/EcalCommon/interface/UtilsClient.h"
#include "DQM/EcalCommon/interface/Numbers.h"

#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "DQM/EcalEndcapMonitorClient/interface/EEPedestalClient.h"

// #define COMMON_NOISE_ANALYSIS

EEPedestalClient::EEPedestalClient(const edm::ParameterSet& ps) {

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

  MGPAGains_.reserve(3);
  for ( unsigned int i = 1; i <= 3; i++ ) MGPAGains_.push_back(i);
  MGPAGains_ = ps.getUntrackedParameter<std::vector<int> >("MGPAGains", MGPAGains_);

  MGPAGainsPN_.reserve(2);
  for ( unsigned int i = 1; i <= 3; i++ ) MGPAGainsPN_.push_back(i);
  MGPAGainsPN_ = ps.getUntrackedParameter<std::vector<int> >("MGPAGainsPN", MGPAGainsPN_);

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    h01_[ism-1] = 0;
    h02_[ism-1] = 0;
    h03_[ism-1] = 0;

    j01_[ism-1] = 0;
    j02_[ism-1] = 0;
    j03_[ism-1] = 0;

    k01_[ism-1] = 0;
    k02_[ism-1] = 0;
    k03_[ism-1] = 0;

    i01_[ism-1] = 0;
    i02_[ism-1] = 0;

  }

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    meg01_[ism-1] = 0;
    meg02_[ism-1] = 0;
    meg03_[ism-1] = 0;

    meg04_[ism-1] = 0;
    meg05_[ism-1] = 0;

    mep01_[ism-1] = 0;
    mep02_[ism-1] = 0;
    mep03_[ism-1] = 0;

    mer01_[ism-1] = 0;
    mer02_[ism-1] = 0;
    mer03_[ism-1] = 0;

    mer04_[ism-1] = 0;
    mer05_[ism-1] = 0;

#ifdef COMMON_NOISE_ANALYSIS
    mes01_[ism-1] = 0;
    mes02_[ism-1] = 0;
    mes03_[ism-1] = 0;

    met01_[ism-1] = 0;
    met02_[ism-1] = 0;
    met03_[ism-1] = 0;
#endif

  }

  expectedMean_[0] = 200.0;
  expectedMean_[1] = 200.0;
  expectedMean_[2] = 200.0;

  discrepancyMean_[0] = 25.0;
  discrepancyMean_[1] = 25.0;
  discrepancyMean_[2] = 25.0;

  RMSThreshold_[0] = 1.0;
  RMSThreshold_[1] = 1.5;
  RMSThreshold_[2] = 2.5;

  RMSThresholdInner_[0] = 2.0;
  RMSThresholdInner_[1] = 2.5;
  RMSThresholdInner_[2] = 3.5;

  expectedMeanPn_[0] = 750.0;
  expectedMeanPn_[1] = 750.0;

  discrepancyMeanPn_[0] = 100.0;
  discrepancyMeanPn_[1] = 100.0;

  RMSThresholdPn_[0] = 999.;
  RMSThresholdPn_[1] = 999.;

}

EEPedestalClient::~EEPedestalClient() {

}

void EEPedestalClient::beginJob(void) {

  dqmStore_ = edm::Service<DQMStore>().operator->();

  if ( debug_ ) std::cout << "EEPedestalClient: beginJob" << std::endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EEPedestalClient::beginRun(void) {

  if ( debug_ ) std::cout << "EEPedestalClient: beginRun" << std::endl;

  jevt_ = 0;

  this->setup();

}

void EEPedestalClient::endJob(void) {

  if ( debug_ ) std::cout << "EEPedestalClient: endJob, ievt = " << ievt_ << std::endl;

  this->cleanup();

}

void EEPedestalClient::endRun(void) {

  if ( debug_ ) std::cout << "EEPedestalClient: endRun, jevt = " << jevt_ << std::endl;

  this->cleanup();

}

void EEPedestalClient::setup(void) {

  std::string name;

  dqmStore_->setCurrentFolder( prefixME_ + "/EEPedestalClient" );

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if (find(MGPAGains_.begin(), MGPAGains_.end(), 1) != MGPAGains_.end() ) {
      if ( meg01_[ism-1] ) dqmStore_->removeElement( meg01_[ism-1]->getName() );
      name = "EEPT pedestal quality G01 " + Numbers::sEE(ism);
      meg01_[ism-1] = dqmStore_->book2D(name, name, 50, Numbers::ix0EE(ism)+0., Numbers::ix0EE(ism)+50., 50, Numbers::iy0EE(ism)+0., Numbers::iy0EE(ism)+50.);
      meg01_[ism-1]->setAxisTitle("ix", 1);
      if ( ism >= 1 && ism <= 9 ) meg01_[ism-1]->setAxisTitle("101-ix", 1);
      meg01_[ism-1]->setAxisTitle("iy", 2);
    }
    if (find(MGPAGains_.begin(), MGPAGains_.end(), 6) != MGPAGains_.end() ) {
      if ( meg02_[ism-1] ) dqmStore_->removeElement( meg02_[ism-1]->getName() );
      name = "EEPT pedestal quality G06 " + Numbers::sEE(ism);
      meg02_[ism-1] = dqmStore_->book2D(name, name, 50, Numbers::ix0EE(ism)+0., Numbers::ix0EE(ism)+50., 50, Numbers::iy0EE(ism)+0., Numbers::iy0EE(ism)+50.);
      meg02_[ism-1]->setAxisTitle("ix", 1);
      if ( ism >= 1 && ism <= 9 ) meg02_[ism-1]->setAxisTitle("101-ix", 1);
      meg02_[ism-1]->setAxisTitle("iy", 2);
    }
    if (find(MGPAGains_.begin(), MGPAGains_.end(), 12) != MGPAGains_.end() ) {
      if ( meg03_[ism-1] ) dqmStore_->removeElement( meg03_[ism-1]->getName() );
      name = "EEPT pedestal quality G12 " + Numbers::sEE(ism);
      meg03_[ism-1] = dqmStore_->book2D(name, name, 50, Numbers::ix0EE(ism)+0., Numbers::ix0EE(ism)+50., 50, Numbers::iy0EE(ism)+0., Numbers::iy0EE(ism)+50.);
      meg03_[ism-1]->setAxisTitle("ix", 1);
      if ( ism >= 1 && ism <= 9 ) meg03_[ism-1]->setAxisTitle("101-ix", 1);
      meg03_[ism-1]->setAxisTitle("iy", 2);
    }

    if (find(MGPAGainsPN_.begin(), MGPAGainsPN_.end(), 1) != MGPAGainsPN_.end() ) {
      if ( meg04_[ism-1] ) dqmStore_->removeElement( meg04_[ism-1]->getName() );
      name = "EEPT pedestal quality PNs G01 " + Numbers::sEE(ism);
      meg04_[ism-1] = dqmStore_->book2D(name, name, 10, 0., 10., 1, 0., 5.);
      meg04_[ism-1]->setAxisTitle("pseudo-strip", 1);
      meg04_[ism-1]->setAxisTitle("channel", 2);
    }
    if (find(MGPAGainsPN_.begin(), MGPAGainsPN_.end(), 16) != MGPAGainsPN_.end() ) {
      if ( meg05_[ism-1] ) dqmStore_->removeElement( meg05_[ism-1]->getName() );
      name = "EEPT pedestal quality PNs G16 " + Numbers::sEE(ism);
      meg05_[ism-1] = dqmStore_->book2D(name, name, 10, 0., 10., 1, 0., 5.);
      meg05_[ism-1]->setAxisTitle("pseudo-strip", 1);
      meg05_[ism-1]->setAxisTitle("channel", 2);
    }

    if (find(MGPAGains_.begin(), MGPAGains_.end(), 1) != MGPAGains_.end() ) {
      if ( mep01_[ism-1] ) dqmStore_->removeElement( mep01_[ism-1]->getName() );
      name = "EEPT pedestal mean G01 " + Numbers::sEE(ism);
      mep01_[ism-1] = dqmStore_->book1D(name, name, 100, 150., 250.);
      mep01_[ism-1]->setAxisTitle("mean", 1);
    }
    if (find(MGPAGains_.begin(), MGPAGains_.end(), 6) != MGPAGains_.end() ) {
      if ( mep02_[ism-1] ) dqmStore_->removeElement( mep02_[ism-1]->getName() );
      name = "EEPT pedestal mean G06 " + Numbers::sEE(ism);
      mep02_[ism-1] = dqmStore_->book1D(name, name, 100, 150., 250.);
      mep02_[ism-1]->setAxisTitle("mean", 1);
    }
    if (find(MGPAGains_.begin(), MGPAGains_.end(), 12) != MGPAGains_.end() ) {
      if ( mep03_[ism-1] ) dqmStore_->removeElement( mep03_[ism-1]->getName() );
      name = "EEPT pedestal mean G12 " + Numbers::sEE(ism);
      mep03_[ism-1] = dqmStore_->book1D(name, name, 100, 150., 250.);
      mep03_[ism-1]->setAxisTitle("mean", 1);
    }

    if (find(MGPAGains_.begin(), MGPAGains_.end(), 1) != MGPAGains_.end() ) {
      if ( mer01_[ism-1] ) dqmStore_->removeElement( mer01_[ism-1]->getName() );
      name = "EEPT pedestal rms G01 " + Numbers::sEE(ism);
      mer01_[ism-1] = dqmStore_->book1D(name, name, 100, 0., 10.);
      mer01_[ism-1]->setAxisTitle("rms", 1);
    }
    if (find(MGPAGains_.begin(), MGPAGains_.end(), 6) != MGPAGains_.end() ) {
      if ( mer02_[ism-1] ) dqmStore_->removeElement( mer02_[ism-1]->getName() );
      name = "EEPT pedestal rms G06 " + Numbers::sEE(ism);
      mer02_[ism-1] = dqmStore_->book1D(name, name, 100, 0., 10.);
      mer02_[ism-1]->setAxisTitle("rms", 1);
    }
    if (find(MGPAGains_.begin(), MGPAGains_.end(), 12) != MGPAGains_.end() ) {
      if ( mer03_[ism-1] ) dqmStore_->removeElement( mer03_[ism-1]->getName() );
      name = "EEPT pedestal rms G12 " + Numbers::sEE(ism);
      mer03_[ism-1] = dqmStore_->book1D(name, name, 100, 0., 10.);
      mer03_[ism-1]->setAxisTitle("rms", 1);
    }

    if (find(MGPAGainsPN_.begin(), MGPAGainsPN_.end(), 1) != MGPAGainsPN_.end() ) {
      if ( mer04_[ism-1] ) dqmStore_->removeElement( mer04_[ism-1]->getName() );
      name = "EEPDT PNs pedestal rms " + Numbers::sEE(ism) + " G01";
      mer04_[ism-1] = dqmStore_->book1D(name, name, 100, 0., 10.);
      mer04_[ism-1]->setAxisTitle("rms", 1);
    }
    if (find(MGPAGainsPN_.begin(), MGPAGainsPN_.end(), 16) != MGPAGainsPN_.end() ) {
      if ( mer05_[ism-1] ) dqmStore_->removeElement( mer05_[ism-1]->getName() );
      name = "EEPDT PNs pedestal rms " + Numbers::sEE(ism) + " G16";
      mer05_[ism-1] = dqmStore_->book1D(name, name, 100, 0., 10.);
      mer05_[ism-1]->setAxisTitle("rms", 1);
    }

#ifdef COMMON_NOISE_ANALYSIS
    if (find(MGPAGains_.begin(), MGPAGains_.end(), 1) != MGPAGains_.end() ) {
      if ( mes01_[ism-1] ) dqmStore_->removeElement( mes01_[ism-1]->getName() );
      name = "EEPT pedestal 3sum G01 " + Numbers::sEE(ism);
      mes01_[ism-1] = dqmStore_->book2D(name, name, 50, Numbers::ix0EE(ism)+0., Numbers::ix0EE(ism)+50., 50, Numbers::iy0EE(ism)+0., Numbers::iy0EE(ism)+50.);
      mes01_[ism-1]->setAxisTitle("ix", 1);
      if ( ism >= 1 && ism <= 9 ) mes01_[ism-1]->setAxisTitle("101-ix", 1);
      mes01_[ism-1]->setAxisTitle("iy", 2);
    }
    if (find(MGPAGains_.begin(), MGPAGains_.end(), 6) != MGPAGains_.end() ) {
      if ( mes02_[ism-1] ) dqmStore_->removeElement( mes02_[ism-1]->getName() );
      name = "EEPT pedestal 3sum G06 " + Numbers::sEE(ism);
      mes02_[ism-1] = dqmStore_->book2D(name, name, 50, Numbers::ix0EE(ism)+0., Numbers::ix0EE(ism)+50., 50, Numbers::iy0EE(ism)+0., Numbers::iy0EE(ism)+50.);
      mes02_[ism-1]->setAxisTitle("ix", 1);
      if ( ism >= 1 && ism <= 9 ) mes02_[ism-1]->setAxisTitle("101-ix", 1);
      mes02_[ism-1]->setAxisTitle("iy", 2);
    }
    if (find(MGPAGains_.begin(), MGPAGains_.end(), 12) != MGPAGains_.end() ) {
      if ( mes03_[ism-1] ) dqmStore_->removeElement( mes03_[ism-1]->getName() );
      name = "EEPT pedestal 3sum G12 " + Numbers::sEE(ism);
      mes03_[ism-1] = dqmStore_->book2D(name, name, 50, Numbers::ix0EE(ism)+0., Numbers::ix0EE(ism)+50., 50, Numbers::iy0EE(ism)+0., Numbers::iy0EE(ism)+50.);
      mes03_[ism-1]->setAxisTitle("ix", 1);
      if ( ism >= 1 && ism <= 9 ) mes03_[ism-1]->setAxisTitle("101-ix", 1);
      mes03_[ism-1]->setAxisTitle("iy", 2);
    }

    if (find(MGPAGains_.begin(), MGPAGains_.end(), 1) != MGPAGains_.end() ) {
      if ( met01_[ism-1] ) dqmStore_->removeElement( met01_[ism-1]->getName() );
      name = "EEPT pedestal 5sum G01 " + Numbers::sEE(ism);
      met01_[ism-1] = dqmStore_->book2D(name, name, 50, Numbers::ix0EE(ism)+0., Numbers::ix0EE(ism)+50., 50, Numbers::iy0EE(ism)+0., Numbers::iy0EE(ism)+50.);
      met01_[ism-1]->setAxisTitle("ix", 1);
      if ( ism >= 1 && ism <= 9 ) met01_[ism-1]->setAxisTitle("101-ix", 1);
      met01_[ism-1]->setAxisTitle("iy", 2);
    }
    if (find(MGPAGains_.begin(), MGPAGains_.end(), 6) != MGPAGains_.end() ) {
      if ( met02_[ism-1] ) dqmStore_->removeElement( met02_[ism-1]->getName() );
      name = "EEPT pedestal 5sum G06 " + Numbers::sEE(ism);
      met02_[ism-1] = dqmStore_->book2D(name, name, 50, Numbers::ix0EE(ism)+0., Numbers::ix0EE(ism)+50., 50, Numbers::iy0EE(ism)+0., Numbers::iy0EE(ism)+50.);
      met02_[ism-1]->setAxisTitle("ix", 1);
      if ( ism >= 1 && ism <= 9 ) met02_[ism-1]->setAxisTitle("101-ix", 1);
      met02_[ism-1]->setAxisTitle("iy", 2);
    }
    if (find(MGPAGains_.begin(), MGPAGains_.end(), 12) != MGPAGains_.end() ) {
      if ( met03_[ism-1] ) dqmStore_->removeElement( met03_[ism-1]->getName() );
      name = "EEPT pedestal 5sum G12 " + Numbers::sEE(ism);
      met03_[ism-1] = dqmStore_->book2D(name, name, 50, Numbers::ix0EE(ism)+0., Numbers::ix0EE(ism)+50., 50, Numbers::iy0EE(ism)+0., Numbers::iy0EE(ism)+50.);
      met03_[ism-1]->setAxisTitle("ix", 1);
      if ( ism >= 1 && ism <= 9 ) met03_[ism-1]->setAxisTitle("101-ix", 1);
      met03_[ism-1]->setAxisTitle("iy", 2);
    }
#endif

  }

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( meg01_[ism-1] ) meg01_[ism-1]->Reset();
    if ( meg02_[ism-1] ) meg02_[ism-1]->Reset();
    if ( meg03_[ism-1] ) meg03_[ism-1]->Reset();

    if ( meg04_[ism-1] ) meg04_[ism-1]->Reset();
    if ( meg05_[ism-1] ) meg05_[ism-1]->Reset();

    for ( int ix = 1; ix <= 50; ix++ ) {
      for ( int iy = 1; iy <= 50; iy++ ) {

        if ( meg01_[ism-1] ) meg01_[ism-1]->setBinContent( ix, iy, 6. );
        if ( meg02_[ism-1] ) meg02_[ism-1]->setBinContent( ix, iy, 6. );
        if ( meg03_[ism-1] ) meg03_[ism-1]->setBinContent( ix, iy, 6. );

        int jx = ix + Numbers::ix0EE(ism);
        int jy = iy + Numbers::iy0EE(ism);

        if ( ism >= 1 && ism <= 9 ) jx = 101 - jx;

        if ( Numbers::validEE(ism, jx, jy) ) {
          if ( meg01_[ism-1] ) meg01_[ism-1]->setBinContent( ix, iy, 2. );
          if ( meg02_[ism-1] ) meg02_[ism-1]->setBinContent( ix, iy, 2. );
          if ( meg03_[ism-1] ) meg03_[ism-1]->setBinContent( ix, iy, 2. );
        }

      }
    }

    for ( int i = 1; i <= 10; i++ ) {

      if ( meg04_[ism-1] ) meg04_[ism-1]->setBinContent( i, 1, 6. );
      if ( meg05_[ism-1] ) meg05_[ism-1]->setBinContent( i, 1, 6. );

      // non-existing mem
      if ( (ism >=  3 && ism <=  4) || (ism >=  7 && ism <=  9) ) continue;
      if ( (ism >= 12 && ism <= 13) || (ism >= 16 && ism <= 18) ) continue;

      if ( meg04_[ism-1] ) meg04_[ism-1]->setBinContent( i, 1, 2. );
      if ( meg05_[ism-1] ) meg05_[ism-1]->setBinContent( i, 1, 2. );

    }

    if ( mep01_[ism-1] ) mep01_[ism-1]->Reset();
    if ( mep02_[ism-1] ) mep02_[ism-1]->Reset();
    if ( mep03_[ism-1] ) mep03_[ism-1]->Reset();

    if ( mer01_[ism-1] ) mer01_[ism-1]->Reset();
    if ( mer02_[ism-1] ) mer02_[ism-1]->Reset();
    if ( mer03_[ism-1] ) mer03_[ism-1]->Reset();

    if ( mer04_[ism-1] ) mer04_[ism-1]->Reset();
    if ( mer05_[ism-1] ) mer05_[ism-1]->Reset();

#ifdef COMMON_NOISE_ANALYSIS
    if ( mes01_[ism-1] ) mes01_[ism-1]->Reset();
    if ( mes02_[ism-1] ) mes02_[ism-1]->Reset();
    if ( mes03_[ism-1] ) mes03_[ism-1]->Reset();

    if ( met01_[ism-1] ) met01_[ism-1]->Reset();
    if ( met02_[ism-1] ) met02_[ism-1]->Reset();
    if ( met03_[ism-1] ) met03_[ism-1]->Reset();

    for ( int ix = 1; ix <= 50; ix++ ) {
      for ( int iy = 1; iy <= 50; iy++ ) {

        if ( mes01_[ism-1] ) mes01_[ism-1]->setBinContent( ix, iy, -999. );
        if ( mes02_[ism-1] ) mes02_[ism-1]->setBinContent( ix, iy, -999. );
        if ( mes03_[ism-1] ) mes03_[ism-1]->setBinContent( ix, iy, -999. );

        if ( met01_[ism-1] ) met01_[ism-1]->setBinContent( ix, iy, -999. );
        if ( met02_[ism-1] ) met02_[ism-1]->setBinContent( ix, iy, -999. );
        if ( met03_[ism-1] ) met03_[ism-1]->setBinContent( ix, iy, -999. );

      }
    }
#endif

  }

}

void EEPedestalClient::cleanup(void) {

  if ( ! enableCleanup_ ) return;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( cloneME_ ) {
      if ( h01_[ism-1] ) delete h01_[ism-1];
      if ( h02_[ism-1] ) delete h02_[ism-1];
      if ( h03_[ism-1] ) delete h03_[ism-1];

      if ( j01_[ism-1] ) delete j01_[ism-1];
      if ( j02_[ism-1] ) delete j02_[ism-1];
      if ( j03_[ism-1] ) delete j03_[ism-1];

      if ( k01_[ism-1] ) delete k01_[ism-1];
      if ( k02_[ism-1] ) delete k02_[ism-1];
      if ( k03_[ism-1] ) delete k03_[ism-1];

      if ( i01_[ism-1] ) delete i01_[ism-1];
      if ( i02_[ism-1] ) delete i02_[ism-1];
    }

    h01_[ism-1] = 0;
    h02_[ism-1] = 0;
    h03_[ism-1] = 0;

    j01_[ism-1] = 0;
    j02_[ism-1] = 0;
    j03_[ism-1] = 0;

    k01_[ism-1] = 0;
    k02_[ism-1] = 0;
    k03_[ism-1] = 0;

    i01_[ism-1] = 0;
    i02_[ism-1] = 0;

  }

  dqmStore_->setCurrentFolder( prefixME_ + "/EEPedestalClient" );

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( meg01_[ism-1] ) dqmStore_->removeElement( meg01_[ism-1]->getName() );
    meg01_[ism-1] = 0;
    if ( meg02_[ism-1] ) dqmStore_->removeElement( meg02_[ism-1]->getName() );
    meg02_[ism-1] = 0;
    if ( meg03_[ism-1] ) dqmStore_->removeElement( meg03_[ism-1]->getName() );
    meg03_[ism-1] = 0;

    if ( meg04_[ism-1] ) dqmStore_->removeElement( meg04_[ism-1]->getName() );
    meg04_[ism-1] = 0;
    if ( meg05_[ism-1] ) dqmStore_->removeElement( meg05_[ism-1]->getName() );
    meg05_[ism-1] = 0;

    if ( mep01_[ism-1] ) dqmStore_->removeElement( mep01_[ism-1]->getName() );
    mep01_[ism-1] = 0;
    if ( mep02_[ism-1] ) dqmStore_->removeElement( mep02_[ism-1]->getName() );
    mep02_[ism-1] = 0;
    if ( mep03_[ism-1] ) dqmStore_->removeElement( mep03_[ism-1]->getName() );
    mep03_[ism-1] = 0;

    if ( mer01_[ism-1] ) dqmStore_->removeElement( mer01_[ism-1]->getName() );
    mer01_[ism-1] = 0;
    if ( mer02_[ism-1] ) dqmStore_->removeElement( mer02_[ism-1]->getName() );
    mer02_[ism-1] = 0;
    if ( mer03_[ism-1] ) dqmStore_->removeElement( mer03_[ism-1]->getName() );
    mer03_[ism-1] = 0;

    if ( mer04_[ism-1] ) dqmStore_->removeElement( mer04_[ism-1]->getName() );
    mer04_[ism-1] = 0;
    if ( mer05_[ism-1] ) dqmStore_->removeElement( mer05_[ism-1]->getName() );
    mer05_[ism-1] = 0;

#ifdef COMMON_NOISE_ANALYSIS
    if ( mes01_[ism-1] ) dqmStore_->removeElement( mes01_[ism-1]->getName() );
    mes01_[ism-1] = 0;
    if ( mes02_[ism-1] ) dqmStore_->removeElement( mes02_[ism-1]->getName() );
    mes02_[ism-1] = 0;
    if ( mes03_[ism-1] ) dqmStore_->removeElement( mes03_[ism-1]->getName() );
    mes03_[ism-1] = 0;

    if ( met01_[ism-1] ) dqmStore_->removeElement( met01_[ism-1]->getName() );
    met01_[ism-1] = 0;
    if ( met02_[ism-1] ) dqmStore_->removeElement( met02_[ism-1]->getName() );
    met02_[ism-1] = 0;
    if ( met03_[ism-1] ) dqmStore_->removeElement( met03_[ism-1]->getName() );
    met03_[ism-1] = 0;
#endif

  }

}

#ifdef WITH_ECAL_COND_DB
bool EEPedestalClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov, bool& status) {

  status = true;

  EcalLogicID ecid;

  MonPedestalsDat p;
  std::map<EcalLogicID, MonPedestalsDat> dataset1;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( verbose_ ) {
      std::cout << " " << Numbers::sEE(ism) << " (ism=" << ism << ")" << std::endl;
      std::cout << std::endl;
      if (find(MGPAGains_.begin(), MGPAGains_.end(), 1) != MGPAGains_.end() ) {
        UtilsClient::printBadChannels(meg01_[ism-1], h01_[ism-1]);
      }
      if (find(MGPAGains_.begin(), MGPAGains_.end(), 6) != MGPAGains_.end() ) {
        UtilsClient::printBadChannels(meg02_[ism-1], h02_[ism-1]);
      }
      if (find(MGPAGains_.begin(), MGPAGains_.end(), 12) != MGPAGains_.end() ) {
        UtilsClient::printBadChannels(meg03_[ism-1], h03_[ism-1]);
      }
    }

    for ( int ix = 1; ix <= 50; ix++ ) {
      for ( int iy = 1; iy <= 50; iy++ ) {

        int jx = ix + Numbers::ix0EE(ism);
        int jy = iy + Numbers::iy0EE(ism);

        if ( ism >= 1 && ism <= 9 ) jx = 101 - jx;

        if ( ! Numbers::validEE(ism, jx, jy) ) continue;

        bool update01;
        bool update02;
        bool update03;

        float num01, num02, num03;
        float mean01, mean02, mean03;
        float rms01, rms02, rms03;

        update01 = UtilsClient::getBinStatistics(h01_[ism-1], ix, iy, num01, mean01, rms01);
        update02 = UtilsClient::getBinStatistics(h02_[ism-1], ix, iy, num02, mean02, rms02);
        update03 = UtilsClient::getBinStatistics(h03_[ism-1], ix, iy, num03, mean03, rms03);

        if ( update01 || update02 || update03 ) {

          if ( Numbers::icEE(ism, jx, jy) == 1 ) {

            if ( verbose_ ) {
              std::cout << "Preparing dataset for " << Numbers::sEE(ism) << " (ism=" << ism << ")" << std::endl;
              std::cout << "G01 (" << Numbers::ix0EE(i+1)+ix << "," << Numbers::iy0EE(i+1)+iy << ") " << num01  << " " << mean01 << " " << rms01  << std::endl;
              std::cout << "G06 (" << Numbers::ix0EE(i+1)+ix << "," << Numbers::iy0EE(i+1)+iy << ") " << num02  << " " << mean02 << " " << rms02  << std::endl;
              std::cout << "G12 (" << Numbers::ix0EE(i+1)+ix << "," << Numbers::iy0EE(i+1)+iy << ") " << num03  << " " << mean03 << " " << rms03  << std::endl;
              std::cout << std::endl;
            }

          }

          p.setPedMeanG1(mean01);
          p.setPedRMSG1(rms01);

          p.setPedMeanG6(mean02);
          p.setPedRMSG6(rms02);

          p.setPedMeanG12(mean03);
          p.setPedRMSG12(rms03);

          if ( UtilsClient::getBinStatus(meg01_[ism-1], ix, iy) &&
               UtilsClient::getBinStatus(meg02_[ism-1], ix, iy) &&
               UtilsClient::getBinStatus(meg03_[ism-1], ix, iy) ) {
            p.setTaskStatus(true);
          } else {
            p.setTaskStatus(false);
          }

          status = status && UtilsClient::getBinQuality(meg01_[ism-1], ix, iy) &&
            UtilsClient::getBinQuality(meg02_[ism-1], ix, iy) &&
            UtilsClient::getBinQuality(meg03_[ism-1], ix, iy);

          int ic = Numbers::indexEE(ism, jx, jy);

          if ( ic == -1 ) continue;

          if ( econn ) {
            ecid = LogicID::getEcalLogicID("EE_crystal_number", Numbers::iSM(ism, EcalEndcap), ic);
            dataset1[ecid] = p;
          }

        }

      }
    }

  }

  if ( econn ) {
    try {
      if ( verbose_ ) std::cout << "Inserting MonPedestalsDat ..." << std::endl;
      if ( dataset1.size() != 0 ) econn->insertDataArraySet(&dataset1, moniov);
      if ( verbose_ ) std::cout << "done." << std::endl;
    } catch (std::runtime_error &e) {
      std::cerr << e.what() << std::endl;
    }
  }

  if ( verbose_ ) std::cout << std::endl;

  MonPNPedDat pn;
  std::map<EcalLogicID, MonPNPedDat> dataset2;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( verbose_ ) {
      std::cout << " " << Numbers::sEE(ism) << " (ism=" << ism << ")" << std::endl;
      std::cout << std::endl;
      if (find(MGPAGainsPN_.begin(), MGPAGainsPN_.end(), 1) != MGPAGainsPN_.end() ) {
        UtilsClient::printBadChannels(meg04_[ism-1], i01_[ism-1]);
      }
      if (find(MGPAGainsPN_.begin(), MGPAGains_.end(), 16) != MGPAGains_.end() ) {
        UtilsClient::printBadChannels(meg05_[ism-1], i02_[ism-1]);
      }
    }

    for ( int i = 1; i <= 10; i++ ) {

      bool update01;
      bool update02;

      float num01, num02;
      float mean01, mean02;
      float rms01, rms02;

      update01 = UtilsClient::getBinStatistics(i01_[ism-1], i, 0, num01, mean01, rms01);
      update02 = UtilsClient::getBinStatistics(i02_[ism-1], i, 0, num02, mean02, rms02);

      if ( update01 || update02 ) {

        if ( i == 1 ) {

          if ( verbose_ ) {
            std::cout << "Preparing dataset for " << Numbers::sEE(ism) << " (ism=" << ism << ")" << std::endl;
            std::cout << "PNs (" << i << ") G01 " << num01  << " " << mean01 << " " << rms01  << std::endl;
            std::cout << "PNs (" << i << ") G16 " << num01  << " " << mean01 << " " << rms01  << std::endl;
            std::cout << std::endl;
          }

        }

        pn.setPedMeanG1(mean01);
        pn.setPedRMSG1(rms01);

        pn.setPedMeanG16(mean02);
        pn.setPedRMSG16(rms02);

        if ( UtilsClient::getBinStatus(meg04_[ism-1], i, 1) &&
             UtilsClient::getBinStatus(meg05_[ism-1], i, 1) ) {
          pn.setTaskStatus(true);
        } else {
          pn.setTaskStatus(false);
        }

        status = status && UtilsClient::getBinQuality(meg04_[ism-1], i, 1) &&
          UtilsClient::getBinQuality(meg05_[ism-1], i, 1);

        if ( econn ) {
          ecid = LogicID::getEcalLogicID("EE_LM_PN", Numbers::iSM(ism, EcalEndcap), i-1);
          dataset2[ecid] = pn;
        }

      }

    }

  }

  if ( econn ) {
    try {
      if ( verbose_ ) std::cout << "Inserting MonPNPedDat ..." << std::endl;
      if ( dataset2.size() != 0 ) econn->insertDataArraySet(&dataset2, moniov);
      if ( verbose_ ) std::cout << "done." << std::endl;
    } catch (std::runtime_error &e) {
      std::cerr << e.what() << std::endl;
    }
  }

  return true;

}
#endif

void EEPedestalClient::analyze(void) {

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( debug_ ) std::cout << "EEPedestalClient: ievt/jevt = " << ievt_ << "/" << jevt_ << std::endl;
  }

  uint32_t bits01 = 0;
  bits01 |= 1 << EcalDQMStatusHelper::PEDESTAL_LOW_GAIN_MEAN_ERROR;
  bits01 |= 1 << EcalDQMStatusHelper::PEDESTAL_LOW_GAIN_RMS_ERROR;

  uint32_t bits02 = 0;
  bits02 |= 1 << EcalDQMStatusHelper::PEDESTAL_MIDDLE_GAIN_MEAN_ERROR;
  bits02 |= 1 << EcalDQMStatusHelper::PEDESTAL_MIDDLE_GAIN_RMS_ERROR;

  uint32_t bits03 = 0;
  bits03 |= 1 << EcalDQMStatusHelper::PEDESTAL_HIGH_GAIN_MEAN_ERROR;
  bits03 |= 1 << EcalDQMStatusHelper::PEDESTAL_HIGH_GAIN_RMS_ERROR;

  MonitorElement* me;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if (find(MGPAGains_.begin(), MGPAGains_.end(), 1) != MGPAGains_.end() ) {
      me = dqmStore_->get( prefixME_ + "/EEPedestalTask/Gain01/EEPT pedestal " + Numbers::sEE(ism) + " G01" );
      h01_[ism-1] = UtilsClient::getHisto( me, cloneME_, h01_[ism-1] );
    }

    if (find(MGPAGains_.begin(), MGPAGains_.end(), 6) != MGPAGains_.end() ) {
      me = dqmStore_->get( prefixME_ + "/EEPedestalTask/Gain06/EEPT pedestal " + Numbers::sEE(ism) + " G06" );
      h02_[ism-1] = UtilsClient::getHisto( me, cloneME_, h02_[ism-1] );
    }

    if (find(MGPAGains_.begin(), MGPAGains_.end(), 12) != MGPAGains_.end() ) {
      me = dqmStore_->get( prefixME_ + "/EEPedestalTask/Gain12/EEPT pedestal " + Numbers::sEE(ism) + " G12" );
      h03_[ism-1] = UtilsClient::getHisto( me, cloneME_, h03_[ism-1] );
    }

#ifdef COMMON_NOISE_ANALYSIS
    if (find(MGPAGains_.begin(), MGPAGains_.end(), 1) != MGPAGains_.end() ) {
      me = dqmStore_->get( prefixME_ + "/EEPedestalTask/Gain01/EEPT pedestal 3sum " + Numbers::sEE(ism) + " G01" );
      j01_[ism-1] = UtilsClient::getHisto( me, cloneME_, j01_[ism-1] );
    }

    if (find(MGPAGains_.begin(), MGPAGains_.end(), 6) != MGPAGains_.end() ) {
      me = dqmStore_->get( prefixME_ + "/EEPedestalTask/Gain06/EEPT pedestal 3sum " + Numbers::sEE(ism) + " G06" );
      j02_[ism-1] = UtilsClient::getHisto( me, cloneME_, j02_[ism-1] );
    }

    if (find(MGPAGains_.begin(), MGPAGains_.end(), 12) != MGPAGains_.end() ) {
      me = dqmStore_->get( prefixME_ + "/EEPedestalTask/Gain12/EEPT pedestal 3sum " + Numbers::sEE(ism) + " G12" );
      j03_[ism-1] = UtilsClient::getHisto( me, cloneME_, j03_[ism-1] );
    }

    if (find(MGPAGains_.begin(), MGPAGains_.end(), 1) != MGPAGains_.end() ) {
      me = dqmStore_->get( prefixME_ + "/EEPedestalTask/Gain01/EEPT pedestal 5sum " + Numbers::sEE(ism) + " G01" );
      k01_[ism-1] = UtilsClient::getHisto( me, cloneME_, k01_[ism-1] );
    }

    if (find(MGPAGains_.begin(), MGPAGains_.end(), 6) != MGPAGains_.end() ) {
      me = dqmStore_->get( prefixME_ + "/EEPedestalTask/Gain06/EEPT pedestal 5sum " + Numbers::sEE(ism) + " G06" );
      k02_[ism-1] = UtilsClient::getHisto( me, cloneME_, k02_[ism-1] );
    }

    if (find(MGPAGains_.begin(), MGPAGains_.end(), 12) != MGPAGains_.end() ) {
      me = dqmStore_->get( prefixME_ + "/EEPedestalTask/Gain12/EEPT pedestal 5sum " + Numbers::sEE(ism) + " G12" );
      k03_[ism-1] = UtilsClient::getHisto( me, cloneME_, k03_[ism-1] );
    }
#endif

    if (find(MGPAGainsPN_.begin(), MGPAGainsPN_.end(), 1) != MGPAGainsPN_.end() ) {
      me = dqmStore_->get( prefixME_ + "/EEPedestalTask/PN/Gain01/EEPDT PNs pedestal " + Numbers::sEE(ism) + " G01" );
      i01_[ism-1] = UtilsClient::getHisto( me, cloneME_, i01_[ism-1] );
    }

    if (find(MGPAGainsPN_.begin(), MGPAGainsPN_.end(), 16) != MGPAGainsPN_.end() ) {
      me = dqmStore_->get( prefixME_ + "/EEPedestalTask/PN/Gain16/EEPDT PNs pedestal " + Numbers::sEE(ism) + " G16" );
      i02_[ism-1] = UtilsClient::getHisto( me, cloneME_, i02_[ism-1] );
    }

    if ( meg01_[ism-1] ) meg01_[ism-1]->Reset();
    if ( meg02_[ism-1] ) meg02_[ism-1]->Reset();
    if ( meg03_[ism-1] ) meg03_[ism-1]->Reset();

    if ( meg04_[ism-1] ) meg04_[ism-1]->Reset();
    if ( meg05_[ism-1] ) meg05_[ism-1]->Reset();

    if ( mep01_[ism-1] ) mep01_[ism-1]->Reset();
    if ( mep02_[ism-1] ) mep02_[ism-1]->Reset();
    if ( mep03_[ism-1] ) mep03_[ism-1]->Reset();

    if ( mer01_[ism-1] ) mer01_[ism-1]->Reset();
    if ( mer02_[ism-1] ) mer02_[ism-1]->Reset();
    if ( mer03_[ism-1] ) mer03_[ism-1]->Reset();

    if ( mer04_[ism-1] ) mer04_[ism-1]->Reset();
    if ( mer05_[ism-1] ) mer05_[ism-1]->Reset();

#ifdef COMMON_NOISE_ANALYSIS
    if ( mes01_[ism-1] ) mes01_[ism-1]->Reset();
    if ( mes02_[ism-1] ) mes02_[ism-1]->Reset();
    if ( mes03_[ism-1] ) mes03_[ism-1]->Reset();

    if ( met01_[ism-1] ) met01_[ism-1]->Reset();
    if ( met02_[ism-1] ) met02_[ism-1]->Reset();
    if ( met03_[ism-1] ) met03_[ism-1]->Reset();
#endif

    for ( int ix = 1; ix <= 50; ix++ ) {
      for ( int iy = 1; iy <= 50; iy++ ) {

        if ( meg01_[ism-1] ) meg01_[ism-1]->setBinContent(ix, iy, 6.);
        if ( meg02_[ism-1] ) meg02_[ism-1]->setBinContent(ix, iy, 6.);
        if ( meg03_[ism-1] ) meg03_[ism-1]->setBinContent(ix, iy, 6.);

        int jx = ix + Numbers::ix0EE(ism);
        int jy = iy + Numbers::iy0EE(ism);

        if ( ism >= 1 && ism <= 9 ) jx = 101 - jx;

	bool innerCrystals = std::abs(jx-50) >= 5 && std::abs(jx-50) <= 10 && std::abs(jy-50) >= 5 && std::abs(jy-50) <= 10;

        if ( Numbers::validEE(ism, jx, jy) ) {
          if ( meg01_[ism-1] ) meg01_[ism-1]->setBinContent(ix, iy, 2.);
          if ( meg02_[ism-1] ) meg02_[ism-1]->setBinContent(ix, iy, 2.);
          if ( meg03_[ism-1] ) meg03_[ism-1]->setBinContent(ix, iy, 2.);
        }

        bool update01;
        bool update02;
        bool update03;

        float num01, num02, num03;
        float mean01, mean02, mean03;
        float rms01, rms02, rms03;

        update01 = UtilsClient::getBinStatistics(h01_[ism-1], ix, iy, num01, mean01, rms01);
        update02 = UtilsClient::getBinStatistics(h02_[ism-1], ix, iy, num02, mean02, rms02);
        update03 = UtilsClient::getBinStatistics(h03_[ism-1], ix, iy, num03, mean03, rms03);

        if ( update01 ) {

          float val;

          val = 1.;
          if ( std::abs(mean01 - expectedMean_[0]) > discrepancyMean_[0] )
            val = 0.;
          if ( (!innerCrystals && rms01 > RMSThreshold_[0]) ||
	       (innerCrystals && rms01 > RMSThresholdInner_[0]) )
            val = 0.;
          if ( meg01_[ism-1] ) meg01_[ism-1]->setBinContent(ix, iy, val);

          if ( mep01_[ism-1] ) mep01_[ism-1]->Fill(mean01);
          if ( mer01_[ism-1] ) mer01_[ism-1]->Fill(rms01);

        }

        if ( update02 ) {

          float val;

          val = 1.;
          if ( std::abs(mean02 - expectedMean_[1]) > discrepancyMean_[1] )
            val = 0.;
          if ( (!innerCrystals && rms02 > RMSThreshold_[1]) ||
	       (innerCrystals && rms02 > RMSThresholdInner_[1]) )
            val = 0.;
          if ( meg02_[ism-1] ) meg02_[ism-1]->setBinContent(ix, iy, val);

          if ( mep02_[ism-1] ) mep02_[ism-1]->Fill(mean02);
          if ( mer02_[ism-1] ) mer02_[ism-1]->Fill(rms02);

        }

        if ( update03 ) {

          float val;

          val = 1.;
          if ( std::abs(mean03 - expectedMean_[2]) > discrepancyMean_[2] )
            val = 0.;
          if ( (!innerCrystals && rms03 > RMSThreshold_[2]) ||
	       (innerCrystals && rms03 > RMSThresholdInner_[2]) )
            val = 0.;

          if ( meg03_[ism-1] ) meg03_[ism-1]->setBinContent(ix, iy, val);

          if ( mep03_[ism-1] ) mep03_[ism-1]->Fill(mean03);
          if ( mer03_[ism-1] ) mer03_[ism-1]->Fill(rms03);

        }

        if ( Masks::maskChannel(ism, ix, iy, bits01, EcalEndcap) ) UtilsClient::maskBinContent( meg01_[ism-1], ix, iy );
        if ( Masks::maskChannel(ism, ix, iy, bits02, EcalEndcap) ) UtilsClient::maskBinContent( meg02_[ism-1], ix, iy );
        if ( Masks::maskChannel(ism, ix, iy, bits03, EcalEndcap) ) UtilsClient::maskBinContent( meg03_[ism-1], ix, iy );

      }
    }

    // PN diodes

    for ( int i = 1; i <= 10; i++ ) {

      if ( meg04_[ism-1] ) meg04_[ism-1]->setBinContent( i, 1, 6. );
      if ( meg05_[ism-1] ) meg05_[ism-1]->setBinContent( i, 1, 6. );

      // non-existing mem
      if ( (ism >=  3 && ism <=  4) || (ism >=  7 && ism <=  9) ) continue;
      if ( (ism >= 12 && ism <= 13) || (ism >= 16 && ism <= 18) ) continue;

      if ( meg04_[ism-1] ) meg04_[ism-1]->setBinContent( i, 1, 2. );
      if ( meg05_[ism-1] ) meg05_[ism-1]->setBinContent( i, 1, 2. );

      bool update01;
      bool update02;

      float num01, num02;
      float mean01, mean02;
      float rms01, rms02;

      update01 = UtilsClient::getBinStatistics(i01_[ism-1], i, 0, num01, mean01, rms01);
      update02 = UtilsClient::getBinStatistics(i02_[ism-1], i, 0, num02, mean02, rms02);

      // filling projections
      if ( mer04_[ism-1] )  mer04_[ism-1]->Fill(rms01);
      if ( mer05_[ism-1] )  mer05_[ism-1]->Fill(rms02);

      if ( update01 ) {

        float val;

        val = 1.;
        if ( mean01 < (expectedMeanPn_[0] - discrepancyMeanPn_[0])
             || (expectedMeanPn_[0] + discrepancyMeanPn_[0]) <  mean01)
          val = 0.;
        if ( rms01 >  RMSThresholdPn_[0])
          val = 0.;

        if ( meg04_[ism-1] ) meg04_[ism-1]->setBinContent(i, 1, val);

      }

      if ( update02 ) {

        float val;

        val = 1.;
        if ( mean02 < (expectedMeanPn_[1] - discrepancyMeanPn_[1])
             || (expectedMeanPn_[1] + discrepancyMeanPn_[1]) <  mean02)
          val = 0.;
        if ( rms02 >  RMSThresholdPn_[1])
          val = 0.;

        if ( meg05_[ism-1] ) meg05_[ism-1]->setBinContent(i, 1, val);
      }

      if ( Masks::maskPn(ism, i, bits01, EcalEndcap) ) UtilsClient::maskBinContent( meg04_[ism-1], i, 1 );
      if ( Masks::maskPn(ism, i, bits03, EcalEndcap) ) UtilsClient::maskBinContent( meg05_[ism-1], i, 1 );

    }

#ifdef COMMON_NOISE_ANALYSIS
    for ( int ix = 1; ix <= 50; ix++ ) {
      for ( int iy = 1; iy <= 50; iy++ ) {

        float x3val01;
        float x3val02;
        float x3val03;

        float y3val01;
        float y3val02;
        float y3val03;

        float z3val01;
        float z3val02;
        float z3val03;

        float x5val01;
        float x5val02;
        float x5val03;

        float y5val01;
        float y5val02;
        float y5val03;

        float z5val01;
        float z5val02;
        float z5val03;

        if ( mes01_[ism-1] ) mes01_[ism-1]->setBinContent(ix, iy, -999.);
        if ( mes02_[ism-1] ) mes02_[ism-1]->setBinContent(ix, iy, -999.);
        if ( mes03_[ism-1] ) mes03_[ism-1]->setBinContent(ix, iy, -999.);

        if ( met01_[ism-1] ) met01_[ism-1]->setBinContent(ix, iy, -999.);
        if ( met02_[ism-1] ) met02_[ism-1]->setBinContent(ix, iy, -999.);
        if ( met03_[ism-1] ) met03_[ism-1]->setBinContent(ix, iy, -999.);

        int jx = ix + Numbers::ix0EE(ism);
        int jy = iy + Numbers::iy0EE(ism);

        if ( ism >= 1 && ism <= 9 ) jx = 101 - jx;

        if ( ! Numbers::validEE(ism, jx, jy) ) continue;

        if ( ix >= 2 && ix <= 49 && iy >= 2 && iy <= 49 ) {

          x3val01 = 0.;
          x3val02 = 0.;
          x3val03 = 0.;
          for ( int i = -1; i <= +1; i++ ) {
            for ( int j = -1; j <= +1; j++ ) {

              if ( h01_[ism-1] ) x3val01 = x3val01 + h01_[ism-1]->GetBinError(ix+i, iy+j) *
                                   h01_[ism-1]->GetBinError(ix+i, iy+j);

              if ( h02_[ism-1] ) x3val02 = x3val02 + h02_[ism-1]->GetBinError(ix+i, iy+j) *
                                   h02_[ism-1]->GetBinError(ix+i, iy+j);

              if ( h03_[ism-1] ) x3val03 = x3val03 + h03_[ism-1]->GetBinError(ix+i, iy+j) *
                                   h03_[ism-1]->GetBinError(ix+i, iy+j);

            }
          }
          x3val01 = x3val01 / (9.*9.);
          x3val02 = x3val02 / (9.*9.);
          x3val03 = x3val03 / (9.*9.);

          y3val01 = 0.;
          if ( j01_[ism-1] ) y3val01 = j01_[ism-1]->GetBinError(ix, iy) *
                               j01_[ism-1]->GetBinError(ix, iy);

          y3val02 = 0.;
          if ( j02_[ism-1] ) y3val02 = j02_[ism-1]->GetBinError(ix, iy) *
                               j02_[ism-1]->GetBinError(ix, iy);

          y3val03 = 0.;
          if ( j03_[ism-1] ) y3val03 = j03_[ism-1]->GetBinError(ix, iy) *
                               j03_[ism-1]->GetBinError(ix, iy);

          z3val01 = -999.;
          if ( x3val01 != 0 && y3val01 != 0 ) z3val01 = sqrt(std::abs(x3val01 - y3val01));
          if ( (x3val01 - y3val01) < 0 ) z3val01 = -z3val01;

          if ( mes01_[ism-1] ) mes01_[ism-1]->setBinContent(ix, iy, z3val01);

          z3val02 = -999.;
          if ( x3val02 != 0 && y3val02 != 0 ) z3val02 = sqrt(std::abs(x3val02 - y3val02));
          if ( (x3val02 - y3val02) < 0 ) z3val02 = -z3val02;

          if ( mes02_[ism-1] ) mes02_[ism-1]->setBinContent(ix, iy, z3val02);

          z3val03 = -999.;
          if ( x3val03 != 0 && y3val03 != 0 ) z3val03 = sqrt(std::abs(x3val03 - y3val03));
          if ( (x3val03 - y3val03) < 0 ) z3val03 = -z3val03;

          if ( mes03_[ism-1] ) mes03_[ism-1]->setBinContent(ix, iy, z3val03);

        }

        if ( ix >= 3 && ix <= 48 && iy >= 3 && iy <= 48 ) {

          x5val01 = 0.;
          x5val02 = 0.;
          x5val03 = 0.;
          for ( int i = -2; i <= +2; i++ ) {
            for ( int j = -2; j <= +2; j++ ) {

              if ( h01_[ism-1] ) x5val01 = x5val01 + h01_[ism-1]->GetBinError(ix+i, iy+j) *
                                   h01_[ism-1]->GetBinError(ix+i, iy+j);

              if ( h02_[ism-1] ) x5val02 = x5val02 + h02_[ism-1]->GetBinError(ix+i, iy+j) *
                                   h02_[ism-1]->GetBinError(ix+i, iy+j);

              if ( h03_[ism-1] ) x5val03 = x5val03 + h03_[ism-1]->GetBinError(ix+i, iy+j) *
                                   h03_[ism-1]->GetBinError(ix+i, iy+j);

            }
          }
          x5val01 = x5val01 / (25.*25.);
          x5val02 = x5val02 / (25.*25.);
          x5val03 = x5val03 / (25.*25.);

          y5val01 = 0.;
          if ( k01_[ism-1] ) y5val01 = k01_[ism-1]->GetBinError(ix, iy) *
                               k01_[ism-1]->GetBinError(ix, iy);

          y5val02 = 0.;
          if ( k02_[ism-1] ) y5val02 = k02_[ism-1]->GetBinError(ix, iy) *
                               k02_[ism-1]->GetBinError(ix, iy);

          y5val03 = 0.;
          if ( k03_[ism-1] ) y5val03 = k03_[ism-1]->GetBinError(ix, iy) *
                               k03_[ism-1]->GetBinError(ix, iy);

          z5val01 = -999.;
          if ( x5val01 != 0 && y5val01 != 0 ) z5val01 = sqrt(std::abs(x5val01 - y5val01));
          if ( (x5val01 - y5val01) < 0 ) z5val01 = -z5val01;

          if ( met01_[ism-1] ) met01_[ism-1]->setBinContent(ix, iy, z5val01);

          z5val02 = -999.;
          if ( x5val02 != 0 && y5val02 != 0 ) z5val02 = sqrt(std::abs(x5val02 - y5val02));
          if ( (x5val02 - y5val02) < 0 ) z5val02 = -z5val02;

          if ( met02_[ism-1] ) met02_[ism-1]->setBinContent(ix, iy, z5val02);

          z5val03 = -999.;
          if ( x5val03 != 0 && y5val03 != 0 ) z5val03 = sqrt(std::abs(x5val03 - y5val03));
          if ( (x5val03 - y5val03) < 0 ) z5val03 = -z5val03;

          if ( met03_[ism-1] ) met03_[ism-1]->setBinContent(ix, iy, z5val03);

        }

      }
    }
#endif

  }

}

