/*
 * \file EBPedestalClient.cc
 *
 * $Date: 2012/04/27 13:45:59 $
 * $Revision: 1.233 $
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

#include "DQM/EcalBarrelMonitorClient/interface/EBPedestalClient.h"

// #define COMMON_NOISE_ANALYSIS

EBPedestalClient::EBPedestalClient(const edm::ParameterSet& ps) {

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

  // vector of selected Super Modules (Defaults to all 36).
  superModules_.reserve(36);
  for ( unsigned int i = 1; i <= 36; i++ ) superModules_.push_back(i);
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
  RMSThreshold_[1] = 1.2;
  RMSThreshold_[2] = 2.0;

  expectedMeanPn_[0] = 750.0;
  expectedMeanPn_[1] = 750.0;

  discrepancyMeanPn_[0] = 100.0;
  discrepancyMeanPn_[1] = 100.0;

  RMSThresholdPn_[0] = 999.;
  RMSThresholdPn_[1] = 999.;

}

EBPedestalClient::~EBPedestalClient() {

}

void EBPedestalClient::beginJob(void) {

  dqmStore_ = edm::Service<DQMStore>().operator->();

  if ( debug_ ) std::cout << "EBPedestalClient: beginJob" << std::endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EBPedestalClient::beginRun(void) {

  if ( debug_ ) std::cout << "EBPedestalClient: beginRun" << std::endl;

  jevt_ = 0;

  this->setup();

}

void EBPedestalClient::endJob(void) {

  if ( debug_ ) std::cout << "EBPedestalClient: endJob, ievt = " << ievt_ << std::endl;

  this->cleanup();

}

void EBPedestalClient::endRun(void) {

  if ( debug_ ) std::cout << "EBPedestalClient: endRun, jevt = " << jevt_ << std::endl;

  this->cleanup();

}

void EBPedestalClient::setup(void) {

  std::string name;

  dqmStore_->setCurrentFolder( prefixME_ + "/EBPedestalClient" );

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if (find(MGPAGains_.begin(), MGPAGains_.end(), 1) != MGPAGains_.end() ) {
      if ( meg01_[ism-1] ) dqmStore_->removeElement( meg01_[ism-1]->getName() );
      name = "EBPT pedestal quality G01 " + Numbers::sEB(ism);
      meg01_[ism-1] = dqmStore_->book2D(name, name, 85, 0., 85., 20, 0., 20.);
      meg01_[ism-1]->setAxisTitle("ieta", 1);
      meg01_[ism-1]->setAxisTitle("iphi", 2);
    }
    if (find(MGPAGains_.begin(), MGPAGains_.end(), 6) != MGPAGains_.end() ) {
      if ( meg02_[ism-1] ) dqmStore_->removeElement( meg02_[ism-1]->getName() );
      name = "EBPT pedestal quality G06 " + Numbers::sEB(ism);
      meg02_[ism-1] = dqmStore_->book2D(name, name, 85, 0., 85., 20, 0., 20.);
      meg02_[ism-1]->setAxisTitle("ieta", 1);
      meg02_[ism-1]->setAxisTitle("iphi", 2);
    }
    if (find(MGPAGains_.begin(), MGPAGains_.end(), 12) != MGPAGains_.end() ) {
      if ( meg03_[ism-1] ) dqmStore_->removeElement( meg03_[ism-1]->getName() );
      name = "EBPT pedestal quality G12 " + Numbers::sEB(ism);
      meg03_[ism-1] = dqmStore_->book2D(name, name, 85, 0., 85., 20, 0., 20.);
      meg03_[ism-1]->setAxisTitle("ieta", 1);
      meg03_[ism-1]->setAxisTitle("iphi", 2);
    }

    if (find(MGPAGainsPN_.begin(), MGPAGainsPN_.end(), 1) != MGPAGainsPN_.end() ) {
      if ( meg04_[ism-1] ) dqmStore_->removeElement( meg04_[ism-1]->getName() );
      name = "EBPT pedestal quality PNs G01 " + Numbers::sEB(ism);
      meg04_[ism-1] = dqmStore_->book2D(name, name, 10, 0., 10., 1, 0., 5.);
      meg04_[ism-1]->setAxisTitle("pseudo-strip", 1);
      meg04_[ism-1]->setAxisTitle("channel", 2);
    }
    if (find(MGPAGainsPN_.begin(), MGPAGainsPN_.end(), 16) != MGPAGainsPN_.end() ) {
      if ( meg05_[ism-1] ) dqmStore_->removeElement( meg05_[ism-1]->getName() );
      name = "EBPT pedestal quality PNs G16 " + Numbers::sEB(ism);
      meg05_[ism-1] = dqmStore_->book2D(name, name, 10, 0., 10., 1, 0., 5.);
      meg05_[ism-1]->setAxisTitle("pseudo-strip", 1);
      meg05_[ism-1]->setAxisTitle("channel", 2);
    }

    if (find(MGPAGains_.begin(), MGPAGains_.end(), 1) != MGPAGains_.end() ) {
      if ( mep01_[ism-1] ) dqmStore_->removeElement( mep01_[ism-1]->getName() );
      name = "EBPT pedestal mean G01 " + Numbers::sEB(ism);
      mep01_[ism-1] = dqmStore_->book1D(name, name, 100, 150., 250.);
      mep01_[ism-1]->setAxisTitle("mean", 1);
    }
    if (find(MGPAGains_.begin(), MGPAGains_.end(), 6) != MGPAGains_.end() ) {
      if ( mep02_[ism-1] ) dqmStore_->removeElement( mep02_[ism-1]->getName() );
      name = "EBPT pedestal mean G06 " + Numbers::sEB(ism);
      mep02_[ism-1] = dqmStore_->book1D(name, name, 100, 150., 250.);
      mep02_[ism-1]->setAxisTitle("mean", 1);
    }
    if (find(MGPAGains_.begin(), MGPAGains_.end(), 12) != MGPAGains_.end() ) {
      if ( mep03_[ism-1] ) dqmStore_->removeElement( mep03_[ism-1]->getName() );
      name = "EBPT pedestal mean G12 " + Numbers::sEB(ism);
      mep03_[ism-1] = dqmStore_->book1D(name, name, 100, 150., 250.);
      mep03_[ism-1]->setAxisTitle("mean", 1);
    }

    if (find(MGPAGains_.begin(), MGPAGains_.end(), 1) != MGPAGains_.end() ) {
      if ( mer01_[ism-1] ) dqmStore_->removeElement( mer01_[ism-1]->getName() );
      name = "EBPT pedestal rms G01 " + Numbers::sEB(ism);
      mer01_[ism-1] = dqmStore_->book1D(name, name, 100, 0., 10.);
      mer01_[ism-1]->setAxisTitle("rms", 1);
    }
    if (find(MGPAGains_.begin(), MGPAGains_.end(), 6) != MGPAGains_.end() ) {
      if ( mer02_[ism-1] ) dqmStore_->removeElement( mer02_[ism-1]->getName() );
      name = "EBPT pedestal rms G06 " + Numbers::sEB(ism);
      mer02_[ism-1] = dqmStore_->book1D(name, name, 100, 0., 10.);
      mer02_[ism-1]->setAxisTitle("rms", 1);
    }
    if (find(MGPAGains_.begin(), MGPAGains_.end(), 12) != MGPAGains_.end() ) {
      if ( mer03_[ism-1] ) dqmStore_->removeElement( mer03_[ism-1]->getName() );
      name = "EBPT pedestal rms G12 " + Numbers::sEB(ism);
      mer03_[ism-1] = dqmStore_->book1D(name, name, 100, 0., 10.);
      mer03_[ism-1]->setAxisTitle("rms", 1);
    }

    if (find(MGPAGainsPN_.begin(), MGPAGainsPN_.end(), 1) != MGPAGainsPN_.end() ) {
      if ( mer04_[ism-1] ) dqmStore_->removeElement( mer04_[ism-1]->getName() );
      name = "EBPDT PNs pedestal rms " + Numbers::sEB(ism) + " G01"; 
      mer04_[ism-1] = dqmStore_->book1D(name, name, 100, 0., 10.);
      mer04_[ism-1]->setAxisTitle("rms", 1);
    }
    if (find(MGPAGainsPN_.begin(), MGPAGainsPN_.end(), 16) != MGPAGainsPN_.end() ) {
      if ( mer05_[ism-1] ) dqmStore_->removeElement( mer05_[ism-1]->getName() );
      name = "EBPDT PNs pedestal rms " + Numbers::sEB(ism) + " G16";
      mer05_[ism-1] = dqmStore_->book1D(name, name, 100, 0., 10.);
      mer05_[ism-1]->setAxisTitle("rms", 1);
    }

#ifdef COMMON_NOISE_ANALYSIS
    if (find(MGPAGains_.begin(), MGPAGains_.end(), 1) != MGPAGains_.end() ) {
      if ( mes01_[ism-1] ) dqmStore_->removeElement( mes01_[ism-1]->getName() );
      name = "EBPT pedestal 3sum G01 " + Numbers::sEB(ism);
      mes01_[ism-1] = dqmStore_->book2D(name, name, 85, 0., 85., 20, 0., 20.);
      mes01_[ism-1]->setAxisTitle("ieta", 1);
      mes01_[ism-1]->setAxisTitle("iphi", 2);
    }
    if (find(MGPAGains_.begin(), MGPAGains_.end(), 6) != MGPAGains_.end() ) {
      if ( mes02_[ism-1] ) dqmStore_->removeElement( mes02_[ism-1]->getName() );
      name = "EBPT pedestal 3sum G06 " + Numbers::sEB(ism);
      mes02_[ism-1] = dqmStore_->book2D(name, name, 85, 0., 85., 20, 0., 20.);
      mes02_[ism-1]->setAxisTitle("ieta", 1);
      mes02_[ism-1]->setAxisTitle("iphi", 2);
    }
    if (find(MGPAGains_.begin(), MGPAGains_.end(), 12) != MGPAGains_.end() ) {
      if ( mes03_[ism-1] ) dqmStore_->removeElement( mes03_[ism-1]->getName() );
      name = "EBPT pedestal 3sum G12 " + Numbers::sEB(ism);
      mes03_[ism-1] = dqmStore_->book2D(name, name, 85, 0., 85., 20, 0., 20.);
      mes03_[ism-1]->setAxisTitle("ieta", 1);
      mes03_[ism-1]->setAxisTitle("iphi", 2);
    }

    if (find(MGPAGains_.begin(), MGPAGains_.end(), 1) != MGPAGains_.end() ) {
      if ( met01_[ism-1] ) dqmStore_->removeElement( met01_[ism-1]->getName() );
      name = "EBPT pedestal 5sum G01 " + Numbers::sEB(ism);
      met01_[ism-1] = dqmStore_->book2D(name, name, 85, 0., 85., 20, 0., 20.);
      met01_[ism-1]->setAxisTitle("ieta", 1);
      met01_[ism-1]->setAxisTitle("iphi", 2);
    }
    if (find(MGPAGains_.begin(), MGPAGains_.end(), 6) != MGPAGains_.end() ) {
      if ( met02_[ism-1] ) dqmStore_->removeElement( met02_[ism-1]->getName() );
      name = "EBPT pedestal 5sum G06 " + Numbers::sEB(ism);
      met02_[ism-1] = dqmStore_->book2D(name, name, 85, 0., 85., 20, 0., 20.);
      met02_[ism-1]->setAxisTitle("ieta", 1);
      met02_[ism-1]->setAxisTitle("iphi", 2);
    }
    if (find(MGPAGains_.begin(), MGPAGains_.end(), 12) != MGPAGains_.end() ) {
      if ( met03_[ism-1] ) dqmStore_->removeElement( met03_[ism-1]->getName() );
      name = "EBPT pedestal 5sum G12 " + Numbers::sEB(ism);
      met03_[ism-1] = dqmStore_->book2D(name, name, 85, 0., 85., 20, 0., 20.);
      met03_[ism-1]->setAxisTitle("ieta", 1);
      met03_[ism-1]->setAxisTitle("iphi", 2);
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

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

        if ( meg01_[ism-1] ) meg01_[ism-1]->setBinContent( ie, ip, 2. );
        if ( meg02_[ism-1] ) meg02_[ism-1]->setBinContent( ie, ip, 2. );
        if ( meg03_[ism-1] ) meg03_[ism-1]->setBinContent( ie, ip, 2. );

      }
    }

    for ( int i = 1; i <= 10; i++ ) {

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

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

        if ( mes01_[ism-1] ) mes01_[ism-1]->setBinContent( ie, ip, -999. );
        if ( mes02_[ism-1] ) mes02_[ism-1]->setBinContent( ie, ip, -999. );
        if ( mes03_[ism-1] ) mes03_[ism-1]->setBinContent( ie, ip, -999. );

        if ( met01_[ism-1] ) met01_[ism-1]->setBinContent( ie, ip, -999. );
        if ( met02_[ism-1] ) met02_[ism-1]->setBinContent( ie, ip, -999. );
        if ( met03_[ism-1] ) met03_[ism-1]->setBinContent( ie, ip, -999. );

      }
    }
#endif

  }

}

void EBPedestalClient::cleanup(void) {

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

  dqmStore_->setCurrentFolder( prefixME_ + "/EBPedestalClient" );

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
bool EBPedestalClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov, bool& status) {

  status = true;

  EcalLogicID ecid;

  MonPedestalsDat p;
  std::map<EcalLogicID, MonPedestalsDat> dataset1;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( verbose_ ) {
      std::cout << " " << Numbers::sEB(ism) << " (ism=" << ism << ")" << std::endl;
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

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

        bool update01;
        bool update02;
        bool update03;

        float num01, num02, num03;
        float mean01, mean02, mean03;
        float rms01, rms02, rms03;

        update01 = UtilsClient::getBinStatistics(h01_[ism-1], ie, ip, num01, mean01, rms01);
        update02 = UtilsClient::getBinStatistics(h02_[ism-1], ie, ip, num02, mean02, rms02);
        update03 = UtilsClient::getBinStatistics(h03_[ism-1], ie, ip, num03, mean03, rms03);

        if ( update01 || update02 || update03 ) {

          if ( Numbers::icEB(ism, ie, ip) == 1 ) {

            if ( verbose_ ) {
              std::cout << "Preparing dataset for " << Numbers::sEB(ism) << " (ism=" << ism << ")" << std::endl;
              std::cout << "G01 (" << ie << "," << ip << ") " << num01  << " " << mean01 << " " << rms01  << std::endl;
              std::cout << "G06 (" << ie << "," << ip << ") " << num02  << " " << mean02 << " " << rms02  << std::endl;
              std::cout << "G12 (" << ie << "," << ip << ") " << num03  << " " << mean03 << " " << rms03  << std::endl;
              std::cout << std::endl;
            }

          }

          p.setPedMeanG1(mean01);
          p.setPedRMSG1(rms01);

          p.setPedMeanG6(mean02);
          p.setPedRMSG6(rms02);

          p.setPedMeanG12(mean03);
          p.setPedRMSG12(rms03);

          if ( UtilsClient::getBinStatus(meg01_[ism-1], ie, ip) &&
               UtilsClient::getBinStatus(meg02_[ism-1], ie, ip) &&
               UtilsClient::getBinStatus(meg03_[ism-1], ie, ip) ) {
            p.setTaskStatus(true);
          } else {
            p.setTaskStatus(false);
          }

          status = status && UtilsClient::getBinQuality(meg01_[ism-1], ie, ip) &&
            UtilsClient::getBinQuality(meg02_[ism-1], ie, ip) &&
            UtilsClient::getBinQuality(meg03_[ism-1], ie, ip);

          int ic = Numbers::indexEB(ism, ie, ip);

          if ( econn ) {
            ecid = LogicID::getEcalLogicID("EB_crystal_number", Numbers::iSM(ism, EcalBarrel), ic);
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
      std::cout << " " << Numbers::sEB(ism) << " (ism=" << ism << ")" << std::endl;
      std::cout << std::endl;
      if (find(MGPAGainsPN_.begin(), MGPAGainsPN_.end(), 1) != MGPAGainsPN_.end() ) {
        UtilsClient::printBadChannels(meg04_[ism-1], i01_[ism-1]);
      }
      if (find(MGPAGainsPN_.begin(), MGPAGainsPN_.end(), 16) != MGPAGainsPN_.end() ) {
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
            std::cout << "Preparing dataset for " << Numbers::sEB(ism) << " (ism=" << ism << ")" << std::endl;
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
          ecid = LogicID::getEcalLogicID("EB_LM_PN", Numbers::iSM(ism, EcalBarrel), i-1);
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

void EBPedestalClient::analyze(void) {

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( debug_ ) std::cout << "EBPedestalClient: ievt/jevt = " << ievt_ << "/" << jevt_ << std::endl;
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

      me = dqmStore_->get(prefixME_ + "/EBPedestalTask/Gain01/EBPT pedestal " + Numbers::sEB(ism) + " G01");
      h01_[ism-1] = UtilsClient::getHisto( me, cloneME_, h01_[ism-1] );

    }

    if (find(MGPAGains_.begin(), MGPAGains_.end(), 6) != MGPAGains_.end() ) {

      me = dqmStore_->get(prefixME_ + "/EBPedestalTask/Gain06/EBPT pedestal " + Numbers::sEB(ism) + " G06");
      h02_[ism-1] = UtilsClient::getHisto( me, cloneME_, h02_[ism-1] );

    }

    if (find(MGPAGains_.begin(), MGPAGains_.end(), 12) != MGPAGains_.end() ) {

      me = dqmStore_->get(prefixME_ + "/EBPedestalTask/Gain12/EBPT pedestal " + Numbers::sEB(ism) + " G12");
      h03_[ism-1] = UtilsClient::getHisto( me, cloneME_, h03_[ism-1] );

    }

#ifdef COMMON_NOISE_ANALYSIS
    if (find(MGPAGains_.begin(), MGPAGains_.end(), 1) != MGPAGains_.end() ) {

      me = dqmStore_->get(prefixME_ + "/EBPedestalTask/Gain01/EBPT pedestal 3sum " + Numbers::sEB(ism) + " G01");
      j01_[ism-1] = UtilsClient::getHisto( me, cloneME_, j01_[ism-1] );

    }

    if (find(MGPAGains_.begin(), MGPAGains_.end(), 6) != MGPAGains_.end() ) {

      me = dqmStore_->get(prefixME_ + "/EBPedestalTask/Gain06/EBPT pedestal 3sum " + Numbers::sEB(ism) + " G06");
      j02_[ism-1] = UtilsClient::getHisto( me, cloneME_, j02_[ism-1] );

    }

    if (find(MGPAGains_.begin(), MGPAGains_.end(), 12) != MGPAGains_.end() ) {

      me = dqmStore_->get(prefixME_ + "/EBPedestalTask/Gain12/EBPT pedestal 3sum " + Numbers::sEB(ism) + " G12");
      j03_[ism-1] = UtilsClient::getHisto( me, cloneME_, j03_[ism-1] );

    }

    if (find(MGPAGains_.begin(), MGPAGains_.end(), 1) != MGPAGains_.end() ) {

      me = dqmStore_->get(prefixME_ + "/EBPedestalTask/Gain01/EBPT pedestal 5sum " + Numbers::sEB(ism) + " G01");
      k01_[ism-1] = UtilsClient::getHisto( me, cloneME_, k01_[ism-1] );

    }

    if (find(MGPAGains_.begin(), MGPAGains_.end(), 6) != MGPAGains_.end() ) {

      me = dqmStore_->get(prefixME_ + "/EBPedestalTask/Gain06/EBPT pedestal 5sum " + Numbers::sEB(ism) + " G06");
      k02_[ism-1] = UtilsClient::getHisto( me, cloneME_, k02_[ism-1] );

    }

    if (find(MGPAGains_.begin(), MGPAGains_.end(), 12) != MGPAGains_.end() ) {

      me = dqmStore_->get(prefixME_ + "/EBPedestalTask/Gain12/EBPT pedestal 5sum " + Numbers::sEB(ism) + " G12");
      k03_[ism-1] = UtilsClient::getHisto( me, cloneME_, k03_[ism-1] );

    }
#endif

    if (find(MGPAGainsPN_.begin(), MGPAGainsPN_.end(), 1) != MGPAGainsPN_.end() ) {

      me = dqmStore_->get(prefixME_ + "/EBPedestalTask/PN/Gain01/EBPDT PNs pedestal " + Numbers::sEB(ism) + " G01");
      i01_[ism-1] = UtilsClient::getHisto( me, cloneME_, i01_[ism-1] );

    }

    if (find(MGPAGainsPN_.begin(), MGPAGainsPN_.end(), 16) != MGPAGainsPN_.end() ) {

      me = dqmStore_->get(prefixME_ + "/EBPedestalTask/PN/Gain16/EBPDT PNs pedestal " + Numbers::sEB(ism) + " G16");
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

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

        if ( meg01_[ism-1] ) meg01_[ism-1]->setBinContent(ie, ip, 2.);
        if ( meg02_[ism-1] ) meg02_[ism-1]->setBinContent(ie, ip, 2.);
        if ( meg03_[ism-1] ) meg03_[ism-1]->setBinContent(ie, ip, 2.);

        bool update01;
        bool update02;
        bool update03;

        float num01, num02, num03;
        float mean01, mean02, mean03;
        float rms01, rms02, rms03;

        update01 = UtilsClient::getBinStatistics(h01_[ism-1], ie, ip, num01, mean01, rms01);
        update02 = UtilsClient::getBinStatistics(h02_[ism-1], ie, ip, num02, mean02, rms02);
        update03 = UtilsClient::getBinStatistics(h03_[ism-1], ie, ip, num03, mean03, rms03);

        if ( update01 ) {

          float val;

          val = 1.;
          if ( std::abs(mean01 - expectedMean_[0]) > discrepancyMean_[0] )
            val = 0.;
          if ( rms01 > RMSThreshold_[0] )
            val = 0.;
          if ( meg01_[ism-1] ) meg01_[ism-1]->setBinContent(ie, ip, val);

          if ( mep01_[ism-1] ) mep01_[ism-1]->Fill(mean01);
          if ( mer01_[ism-1] ) mer01_[ism-1]->Fill(rms01);

        }

        if ( update02 ) {

          float val;

          val = 1.;
          if ( std::abs(mean02 - expectedMean_[1]) > discrepancyMean_[1] )
            val = 0.;
          if ( rms02 > RMSThreshold_[1] )
            val = 0.;
          if ( meg02_[ism-1] ) meg02_[ism-1]->setBinContent(ie, ip, val);

          if ( mep02_[ism-1] ) mep02_[ism-1]->Fill(mean02);
          if ( mer02_[ism-1] ) mer02_[ism-1]->Fill(rms02);

        }

        if ( update03 ) {

          float val;

          val = 1.;
          if ( std::abs(mean03 - expectedMean_[2]) > discrepancyMean_[2] )
            val = 0.;
          if ( rms03 > RMSThreshold_[2] )
            val = 0.;
          if ( meg03_[ism-1] ) meg03_[ism-1]->setBinContent(ie, ip, val);

          if ( mep03_[ism-1] ) mep03_[ism-1]->Fill(mean03);
          if ( mer03_[ism-1] ) mer03_[ism-1]->Fill(rms03);

        }

        if ( Masks::maskChannel(ism, ie, ip, bits01, EcalBarrel) ) UtilsClient::maskBinContent( meg01_[ism-1], ie, ip );
        if ( Masks::maskChannel(ism, ie, ip, bits02, EcalBarrel) ) UtilsClient::maskBinContent( meg02_[ism-1], ie, ip );
        if ( Masks::maskChannel(ism, ie, ip, bits03, EcalBarrel) ) UtilsClient::maskBinContent( meg03_[ism-1], ie, ip );

      }
    }

    // PN diodes

    for ( int i = 1; i <= 10; i++ ) {

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

      if ( Masks::maskPn(ism, i, bits01, EcalBarrel) ) UtilsClient::maskBinContent( meg04_[ism-1], i, 1 );
      if ( Masks::maskPn(ism, i, bits03, EcalBarrel) ) UtilsClient::maskBinContent( meg05_[ism-1], i, 1 );

    }

#ifdef COMMON_NOISE_ANALYSIS
    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

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

        if ( mes01_[ism-1] ) mes01_[ism-1]->setBinContent(ie, ip, -999.);
        if ( mes02_[ism-1] ) mes02_[ism-1]->setBinContent(ie, ip, -999.);
        if ( mes03_[ism-1] ) mes03_[ism-1]->setBinContent(ie, ip, -999.);

        if ( met01_[ism-1] ) met01_[ism-1]->setBinContent(ie, ip, -999.);
        if ( met02_[ism-1] ) met02_[ism-1]->setBinContent(ie, ip, -999.);
        if ( met03_[ism-1] ) met03_[ism-1]->setBinContent(ie, ip, -999.);

        if ( ie >= 2 && ie <= 84 && ip >= 2 && ip <= 19 ) {

          x3val01 = 0.;
          x3val02 = 0.;
          x3val03 = 0.;
          for ( int i = -1; i <= +1; i++ ) {
            for ( int j = -1; j <= +1; j++ ) {

              if ( h01_[ism-1] ) x3val01 = x3val01 + h01_[ism-1]->GetBinError(ie+i, ip+j) *
                                   h01_[ism-1]->GetBinError(ie+i, ip+j);

              if ( h02_[ism-1] ) x3val02 = x3val02 + h02_[ism-1]->GetBinError(ie+i, ip+j) *
                                   h02_[ism-1]->GetBinError(ie+i, ip+j);

              if ( h03_[ism-1] ) x3val03 = x3val03 + h03_[ism-1]->GetBinError(ie+i, ip+j) *
                                   h03_[ism-1]->GetBinError(ie+i, ip+j);

            }
          }
          x3val01 = x3val01 / (9.*9.);
          x3val02 = x3val02 / (9.*9.);
          x3val03 = x3val03 / (9.*9.);

          y3val01 = 0.;
          if ( j01_[ism-1] ) y3val01 = j01_[ism-1]->GetBinError(ie, ip) *
                               j01_[ism-1]->GetBinError(ie, ip);

          y3val02 = 0.;
          if ( j02_[ism-1] ) y3val02 = j02_[ism-1]->GetBinError(ie, ip) *
                               j02_[ism-1]->GetBinError(ie, ip);

          y3val03 = 0.;
          if ( j03_[ism-1] ) y3val03 = j03_[ism-1]->GetBinError(ie, ip) *
                               j03_[ism-1]->GetBinError(ie, ip);

          z3val01 = -999.;
          if ( x3val01 != 0 && y3val01 != 0 ) z3val01 = sqrt(std::abs(x3val01 - y3val01));
          if ( (x3val01 - y3val01) < 0 ) z3val01 = -z3val01;

          if ( mes01_[ism-1] ) mes01_[ism-1]->setBinContent(ie, ip, z3val01);

          z3val02 = -999.;
          if ( x3val02 != 0 && y3val02 != 0 ) z3val02 = sqrt(std::abs(x3val02 - y3val02));
          if ( (x3val02 - y3val02) < 0 ) z3val02 = -z3val02;

          if ( mes02_[ism-1] ) mes02_[ism-1]->setBinContent(ie, ip, z3val02);

          z3val03 = -999.;
          if ( x3val03 != 0 && y3val03 != 0 ) z3val03 = sqrt(std::abs(x3val03 - y3val03));
          if ( (x3val03 - y3val03) < 0 ) z3val03 = -z3val03;

          if ( mes03_[ism-1] ) mes03_[ism-1]->setBinContent(ie, ip, z3val03);

        }

        if ( ie >= 3 && ie <= 83 && ip >= 3 && ip <= 18 ) {

          x5val01 = 0.;
          x5val02 = 0.;
          x5val03 = 0.;
          for ( int i = -2; i <= +2; i++ ) {
            for ( int j = -2; j <= +2; j++ ) {

              if ( h01_[ism-1] ) x5val01 = x5val01 + h01_[ism-1]->GetBinError(ie+i, ip+j) *
                                   h01_[ism-1]->GetBinError(ie+i, ip+j);

              if ( h02_[ism-1] ) x5val02 = x5val02 + h02_[ism-1]->GetBinError(ie+i, ip+j) *
                                   h02_[ism-1]->GetBinError(ie+i, ip+j);

              if ( h03_[ism-1] ) x5val03 = x5val03 + h03_[ism-1]->GetBinError(ie+i, ip+j) *
                                   h03_[ism-1]->GetBinError(ie+i, ip+j);

            }
          }
          x5val01 = x5val01 / (25.*25.);
          x5val02 = x5val02 / (25.*25.);
          x5val03 = x5val03 / (25.*25.);

          y5val01 = 0.;
          if ( k01_[ism-1] ) y5val01 = k01_[ism-1]->GetBinError(ie, ip) *
                               k01_[ism-1]->GetBinError(ie, ip);

          y5val02 = 0.;
          if ( k02_[ism-1] ) y5val02 = k02_[ism-1]->GetBinError(ie, ip) *
                               k02_[ism-1]->GetBinError(ie, ip);

          y5val03 = 0.;
          if ( k03_[ism-1] ) y5val03 = k03_[ism-1]->GetBinError(ie, ip) *
                               k03_[ism-1]->GetBinError(ie, ip);

          z5val01 = -999.;
          if ( x5val01 != 0 && y5val01 != 0 ) z5val01 = sqrt(std::abs(x5val01 - y5val01));
          if ( (x5val01 - y5val01) < 0 ) z5val01 = -z5val01;

          if ( met01_[ism-1] ) met01_[ism-1]->setBinContent(ie, ip, z5val01);

          z5val02 = -999.;
          if ( x5val02 != 0 && y5val02 != 0 ) z5val02 = sqrt(std::abs(x5val02 - y5val02));
          if ( (x5val02 - y5val02) < 0 ) z5val02 = -z5val02;

          if ( met02_[ism-1] ) met02_[ism-1]->setBinContent(ie, ip, z5val02);

          z5val03 = -999.;
          if ( x5val03 != 0 && y5val03 != 0 ) z5val03 = sqrt(std::abs(x5val03 - y5val03));
          if ( (x5val03 - y5val03) < 0 ) z5val03 = -z5val03;

          if ( met03_[ism-1] ) met03_[ism-1]->setBinContent(ie, ip, z5val03);

        }

      }
    }
#endif

  }

}

