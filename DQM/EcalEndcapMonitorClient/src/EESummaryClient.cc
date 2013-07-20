/*
 * \file EESummaryClient.cc
 *
 * $Date: 2012/08/10 00:30:16 $
 * $Revision: 1.227 $
 * \author G. Della Ricca
 *
*/

#include <memory>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <math.h>
#include <utility>

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#ifdef WITH_ECAL_COND_DB
#include "OnlineDB/EcalCondDB/interface/RunTag.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"
#endif

#include "DataFormats/EcalDetId/interface/EcalScDetId.h"
#include "DataFormats/EcalDetId/interface/EcalTriggerElectronicsId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"

#include "DQM/EcalCommon/interface/UtilsClient.h"
#include "DQM/EcalCommon/interface/Numbers.h"
#include "DQM/EcalCommon/interface/Masks.h"

#include "DQM/EcalEndcapMonitorClient/interface/EEStatusFlagsClient.h"
#include "DQM/EcalEndcapMonitorClient/interface/EEIntegrityClient.h"
#include "DQM/EcalEndcapMonitorClient/interface/EEOccupancyClient.h"
#include "DQM/EcalEndcapMonitorClient/interface/EELaserClient.h"
#include "DQM/EcalEndcapMonitorClient/interface/EELedClient.h"
#include "DQM/EcalEndcapMonitorClient/interface/EEPedestalClient.h"
#include "DQM/EcalEndcapMonitorClient/interface/EEPedestalOnlineClient.h"
#include "DQM/EcalEndcapMonitorClient/interface/EETestPulseClient.h"
#include "DQM/EcalEndcapMonitorClient/interface/EETriggerTowerClient.h"
#include "DQM/EcalEndcapMonitorClient/interface/EEClusterClient.h"
#include "DQM/EcalEndcapMonitorClient/interface/EETimingClient.h"

#include "DQM/EcalEndcapMonitorClient/interface/EESummaryClient.h"

EESummaryClient::EESummaryClient(const edm::ParameterSet& ps) {

  // cloneME switch
  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);

  // verbose switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", true);

  // debug switch
  debug_ = ps.getUntrackedParameter<bool>("debug", false);

  // prefixME path
  prefixME_ = ps.getUntrackedParameter<std::string>("prefixME", "");

  subfolder_ = ps.getUntrackedParameter<std::string>("subfolder", "");

  // enableCleanup_ switch
  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  produceReports_ = ps.getUntrackedParameter<bool>("produceReports", true);

  reducedReports_ = ps.getUntrackedParameter<bool>("reducedReports", false);

  // vector of selected Super Modules (Defaults to all 18).
  superModules_.reserve(18);
  for ( unsigned int i = 1; i <= 18; i++ ) superModules_.push_back(i);
  superModules_ = ps.getUntrackedParameter<std::vector<int> >("superModules", superModules_);

  laserWavelengths_.reserve(4);
  for ( unsigned int i = 1; i <= 4; i++ ) laserWavelengths_.push_back(i);
  laserWavelengths_ = ps.getUntrackedParameter<std::vector<int> >("laserWavelengths", laserWavelengths_);

  ledWavelengths_.reserve(2);
  for ( unsigned int i = 1; i <= 2; i++ ) ledWavelengths_.push_back(i);
  ledWavelengths_ = ps.getUntrackedParameter<std::vector<int> >("ledWavelengths", ledWavelengths_);

  MGPAGains_.reserve(3);
  for ( unsigned int i = 1; i <= 3; i++ ) MGPAGains_.push_back(i);
  MGPAGains_ = ps.getUntrackedParameter<std::vector<int> >("MGPAGains", MGPAGains_);

  MGPAGainsPN_.reserve(2);
  for ( unsigned int i = 1; i <= 3; i++ ) MGPAGainsPN_.push_back(i);
  MGPAGainsPN_ = ps.getUntrackedParameter<std::vector<int> >("MGPAGainsPN", MGPAGainsPN_);

  timingNHitThreshold_ = ps.getUntrackedParameter<int>("timingNHitThreshold", 5);
  synchErrorThreshold_ = ps.getUntrackedParameter<double>("synchErrorThreshold", 0.01);

  // summary maps
  meIntegrity_[0]      = 0;
  meIntegrity_[1]      = 0;
  meIntegrityPN_       = 0;
  meOccupancy_[0]      = 0;
  meOccupancy_[1]      = 0;
  meOccupancyPN_       = 0;
  meStatusFlags_[0]    = 0;
  meStatusFlags_[1]    = 0;
  mePedestalOnline_[0] = 0;
  mePedestalOnline_[1] = 0;
  mePedestalOnlineRMSMap_[0] = 0;
  mePedestalOnlineRMSMap_[1] = 0;
  mePedestalOnlineMean_   = 0;
  mePedestalOnlineRMS_    = 0;

  meLaserL1_[0]        = 0;
  meLaserL1_[1]        = 0;
  meLaserL1PN_         = 0;
  meLaserL1Ampl_       = 0;
  meLaserL1Timing_     = 0;
  meLaserL1AmplOverPN_ = 0;

  meLaserL2_[0]        = 0;
  meLaserL2_[1]        = 0;
  meLaserL2PN_         = 0;
  meLaserL2Ampl_       = 0;
  meLaserL2Timing_     = 0;
  meLaserL2AmplOverPN_ = 0;

  meLaserL3_[0]        = 0;
  meLaserL3_[1]        = 0;
  meLaserL3PN_         = 0;
  meLaserL3Ampl_       = 0;
  meLaserL3Timing_     = 0;
  meLaserL3AmplOverPN_ = 0;

  meLaserL4_[0]        = 0;
  meLaserL4_[1]        = 0;
  meLaserL4PN_         = 0;
  meLaserL4Ampl_       = 0;
  meLaserL4Timing_     = 0;
  meLaserL4AmplOverPN_ = 0;

  meLedL1_[0]          = 0;
  meLedL1_[1]          = 0;
  meLedL1PN_           = 0;
  meLedL1Ampl_         = 0;
  meLedL1Timing_       = 0;
  meLedL1AmplOverPN_   = 0;

  meLedL2_[0]          = 0;
  meLedL2_[1]          = 0;
  meLedL2PN_           = 0;
  meLedL2Ampl_         = 0;
  meLedL2Timing_       = 0;
  meLedL2AmplOverPN_   = 0;

  mePedestalG01_[0]       = 0;
  mePedestalG01_[1]       = 0;
  mePedestalG06_[0]       = 0;
  mePedestalG06_[1]       = 0;
  mePedestalG12_[0]       = 0;
  mePedestalG12_[1]       = 0;
  mePedestalPNG01_        = 0;
  mePedestalPNG16_        = 0;
  meTestPulseG01_[0]      = 0;
  meTestPulseG01_[1]      = 0;
  meTestPulseG06_[0]      = 0;
  meTestPulseG06_[1]      = 0;
  meTestPulseG12_[0]      = 0;
  meTestPulseG12_[1]      = 0;
  meTestPulsePNG01_       = 0;
  meTestPulsePNG16_       = 0;
  meTestPulseAmplG01_ = 0;
  meTestPulseAmplG06_ = 0;
  meTestPulseAmplG12_ = 0;
  meGlobalSummary_[0]  = 0;
  meGlobalSummary_[1]  = 0;

  meRecHitEnergy_[0]   = 0;
  meRecHitEnergy_[1]   = 0;
  meTiming_[0]         = 0;
  meTiming_[1]         = 0;
  meTimingMean1D_[0]   = 0;
  meTimingMean1D_[1]   = 0;
  meTimingRMS1D_[0]   = 0;
  meTimingRMS1D_[1]   = 0;
  meTimingMean_ = 0;
  meTimingRMS_  = 0;

  meTriggerTowerEt_[0]        = 0;
  meTriggerTowerEt_[1]        = 0;
  meTriggerTowerEmulError_[0] = 0;
  meTriggerTowerEmulError_[1] = 0;
  meTriggerTowerTiming_[0] = 0;
  meTriggerTowerTiming_[1] = 0;
  meTriggerTowerNonSingleTiming_[0] = 0;
  meTriggerTowerNonSingleTiming_[1] = 0;

  // summary errors
  meIntegrityErr_       = 0;
  meOccupancy1D_        = 0;
  meStatusFlagsErr_     = 0;
  mePedestalOnlineErr_  = 0;
  meLaserL1Err_         = 0;
  meLaserL1PNErr_       = 0;
  meLaserL2Err_         = 0;
  meLaserL2PNErr_       = 0;
  meLaserL3Err_         = 0;
  meLaserL3PNErr_       = 0;
  meLaserL4Err_         = 0;
  meLaserL4PNErr_       = 0;
  meLedL1Err_           = 0;
  meLedL1PNErr_         = 0;
  meLedL2Err_           = 0;
  meLedL2PNErr_         = 0;

  meSummaryErr_ = 0;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    hpot01_[ism-1] = 0;
    httt01_[ism-1] = 0;

  }

}

EESummaryClient::~EESummaryClient() {

}

void EESummaryClient::beginJob(void) {

  dqmStore_ = edm::Service<DQMStore>().operator->();

  if ( debug_ ) std::cout << "EESummaryClient: beginJob" << std::endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EESummaryClient::beginRun(void) {

  if ( debug_ ) std::cout << "EESummaryClient: beginRun" << std::endl;

  jevt_ = 0;

  this->setup();

}

void EESummaryClient::endJob(void) {

  if ( debug_ ) std::cout << "EESummaryClient: endJob, ievt = " << ievt_ << std::endl;

  this->cleanup();

}

void EESummaryClient::endRun(void) {

  if ( debug_ ) std::cout << "EESummaryClient: endRun, jevt = " << jevt_ << std::endl;

  this->cleanup();

}

void EESummaryClient::setup(void) {

  bool integrityClient(false);
  bool occupancyClient(false);
  bool statusFlagsClient(false);
  bool pedestalOnlineClient(false);
  bool laserClient(false);
  bool ledClient(false);
  bool pedestalClient(false);
  bool testPulseClient(false);
  bool timingClient(false);
  bool triggerTowerClient(false);

  for(unsigned i = 0; i < clients_.size(); i++){

    if(dynamic_cast<EEIntegrityClient*>(clients_[i])) integrityClient = true;
    if(dynamic_cast<EEOccupancyClient*>(clients_[i])) occupancyClient = true;
    if(dynamic_cast<EEStatusFlagsClient*>(clients_[i])) statusFlagsClient = true;
    if(dynamic_cast<EEPedestalOnlineClient*>(clients_[i])) pedestalOnlineClient = true;
    if(dynamic_cast<EELaserClient*>(clients_[i])) laserClient = true;
    if(dynamic_cast<EELedClient*>(clients_[i])) ledClient = true;
    if(dynamic_cast<EEPedestalClient*>(clients_[i])) pedestalClient = true;
    if(dynamic_cast<EETestPulseClient*>(clients_[i])) testPulseClient = true;
    if(dynamic_cast<EETimingClient*>(clients_[i])) timingClient = true;
    if(dynamic_cast<EETriggerTowerClient*>(clients_[i])) triggerTowerClient = true;

  }

  std::string name;

  dqmStore_->setCurrentFolder( prefixME_ + "/EESummaryClient" );

  if(integrityClient){
    if(produceReports_){
      if ( meIntegrity_[0] ) dqmStore_->removeElement( meIntegrity_[0]->getName() );
      name = "EEIT EE - integrity quality summary";
      meIntegrity_[0] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
      meIntegrity_[0]->setAxisTitle("ix", 1);
      meIntegrity_[0]->setAxisTitle("iy", 2);

      if ( meIntegrity_[1] ) dqmStore_->removeElement( meIntegrity_[1]->getName() );
      name = "EEIT EE + integrity quality summary";
      meIntegrity_[1] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
      meIntegrity_[1]->setAxisTitle("ix", 1);
      meIntegrity_[1]->setAxisTitle("iy", 2);

      if ( meIntegrityErr_ ) dqmStore_->removeElement( meIntegrityErr_->getName() );
      name = "EEIT integrity quality errors summary";
      meIntegrityErr_ = dqmStore_->book1D(name, name, 18, 1, 19);
      for (int i = 0; i < 18; i++) {
	meIntegrityErr_->setBinLabel(i+1, Numbers::sEE(i+1), 1);
      }
    }
    if(laserClient){
      if ( meIntegrityPN_ ) dqmStore_->removeElement( meIntegrityPN_->getName() );
      name = "EEIT PN integrity quality summary";
      meIntegrityPN_ = dqmStore_->book2D(name, name, 45, 0., 45., 20, -10., 10.);
      meIntegrityPN_->setAxisTitle("jchannel", 1);
      meIntegrityPN_->setAxisTitle("jpseudo-strip", 2);
    }
  }

  if(occupancyClient){
    if(produceReports_){
      if ( meOccupancy_[0] ) dqmStore_->removeElement( meOccupancy_[0]->getName() );
      name = "EEOT EE - digi occupancy summary";
      meOccupancy_[0] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
      meOccupancy_[0]->setAxisTitle("ix", 1);
      meOccupancy_[0]->setAxisTitle("iy", 2);

      if ( meOccupancy_[1] ) dqmStore_->removeElement( meOccupancy_[1]->getName() );
      name = "EEOT EE + digi occupancy summary";
      meOccupancy_[1] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
      meOccupancy_[1]->setAxisTitle("ix", 1);
      meOccupancy_[1]->setAxisTitle("iy", 2);

      if ( meOccupancy1D_ ) dqmStore_->removeElement( meOccupancy1D_->getName() );
      name = "EEIT digi occupancy summary 1D";
      meOccupancy1D_ = dqmStore_->book1D(name, name, 18, 1, 19);
      for (int i = 0; i < 18; i++) {
	meOccupancy1D_->setBinLabel(i+1, Numbers::sEE(i+1), 1);
      }

      if( meRecHitEnergy_[0] ) dqmStore_->removeElement( meRecHitEnergy_[0]->getName() );
      name = "EEOT EE - energy summary";
      meRecHitEnergy_[0] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
      meRecHitEnergy_[0]->setAxisTitle("ix", 1);
      meRecHitEnergy_[0]->setAxisTitle("iy", 2);
      meRecHitEnergy_[0]->setAxisTitle("energy (GeV)", 3);

      if( meRecHitEnergy_[1] ) dqmStore_->removeElement( meRecHitEnergy_[1]->getName() );
      name = "EEOT EE + energy summary";
      meRecHitEnergy_[1] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
      meRecHitEnergy_[1]->setAxisTitle("ix", 1);
      meRecHitEnergy_[1]->setAxisTitle("iy", 2);
      meRecHitEnergy_[1]->setAxisTitle("energy (GeV)", 3);
    }
    if(laserClient){
      if ( meOccupancyPN_ ) dqmStore_->removeElement( meOccupancyPN_->getName() );
      name = "EEOT PN digi occupancy summary";
      meOccupancyPN_ = dqmStore_->book2D(name, name, 45, 0., 45., 20, -10., 10.);
      meOccupancyPN_->setAxisTitle("channel", 1);
      meOccupancyPN_->setAxisTitle("pseudo-strip", 2);
    }
  }

  if(statusFlagsClient && produceReports_){
    if ( meStatusFlags_[0] ) dqmStore_->removeElement( meStatusFlags_[0]->getName() );
    name = "EESFT EE - front-end status summary";
    meStatusFlags_[0] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
    meStatusFlags_[0]->setAxisTitle("ix", 1);
    meStatusFlags_[0]->setAxisTitle("iy", 2);

    if ( meStatusFlags_[1] ) dqmStore_->removeElement( meStatusFlags_[1]->getName() );
    name = "EESFT EE + front-end status summary";
    meStatusFlags_[1] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
    meStatusFlags_[1]->setAxisTitle("ix", 1);
    meStatusFlags_[1]->setAxisTitle("iy", 2);

    if ( meStatusFlagsErr_ ) dqmStore_->removeElement( meStatusFlagsErr_->getName() );
    name = "EESFT front-end status errors summary";
    meStatusFlagsErr_ = dqmStore_->book1D(name, name, 18, 1, 19);
    for (int i = 0; i < 18; i++) {
      meStatusFlagsErr_->setBinLabel(i+1, Numbers::sEE(i+1), 1);
    }
  }

  if(pedestalOnlineClient && produceReports_){
    if ( mePedestalOnline_[0] ) dqmStore_->removeElement( mePedestalOnline_[0]->getName() );
    name = "EEPOT EE - pedestal quality summary G12";
    mePedestalOnline_[0] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
    mePedestalOnline_[0]->setAxisTitle("ix", 1);
    mePedestalOnline_[0]->setAxisTitle("iy", 2);

    if ( mePedestalOnline_[1] ) dqmStore_->removeElement( mePedestalOnline_[1]->getName() );
    name = "EEPOT EE + pedestal quality summary G12";
    mePedestalOnline_[1] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
    mePedestalOnline_[1]->setAxisTitle("ix", 1);
    mePedestalOnline_[1]->setAxisTitle("iy", 2);

    if ( mePedestalOnlineRMSMap_[0] ) dqmStore_->removeElement( mePedestalOnlineRMSMap_[0]->getName() );
    name = "EEPOT EE - pedestal G12 RMS map";
    mePedestalOnlineRMSMap_[0] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
    mePedestalOnlineRMSMap_[0]->setAxisTitle("ix", 1);
    mePedestalOnlineRMSMap_[0]->setAxisTitle("iy", 2);
    mePedestalOnlineRMSMap_[0]->setAxisTitle("rms", 3);

    if ( mePedestalOnlineRMSMap_[1] ) dqmStore_->removeElement( mePedestalOnlineRMSMap_[1]->getName() );
    name = "EEPOT EE + pedestal G12 RMS map";
    mePedestalOnlineRMSMap_[1] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
    mePedestalOnlineRMSMap_[1]->setAxisTitle("ix", 1);
    mePedestalOnlineRMSMap_[1]->setAxisTitle("iy", 2);
    mePedestalOnlineRMSMap_[1]->setAxisTitle("rms", 3);

    if ( mePedestalOnlineMean_ ) dqmStore_->removeElement( mePedestalOnlineMean_->getName() );
    name = "EEPOT pedestal G12 mean";
    mePedestalOnlineMean_ = dqmStore_->bookProfile(name, name, 18, 1, 19, 100, 150., 250.);
    for (int i = 0; i < 18; i++) {
      mePedestalOnlineMean_->setBinLabel(i+1, Numbers::sEE(i+1), 1);
    }

    if ( mePedestalOnlineRMS_ ) dqmStore_->removeElement( mePedestalOnlineRMS_->getName() );
    name = "EEPOT pedestal G12 rms";
    mePedestalOnlineRMS_ = dqmStore_->bookProfile(name, name, 18, 1, 19, 100, 0., 10.);
    for (int i = 0; i < 18; i++) {
      mePedestalOnlineRMS_->setBinLabel(i+1, Numbers::sEE(i+1), 1);
    }

    if ( mePedestalOnlineErr_ ) dqmStore_->removeElement( mePedestalOnlineErr_->getName() );
    name = "EEPOT pedestal quality errors summary G12";
    mePedestalOnlineErr_ = dqmStore_->book1D(name, name, 18, 1, 19);
    for (int i = 0; i < 18; i++) {
      mePedestalOnlineErr_->setBinLabel(i+1, Numbers::sEE(i+1), 1);
    }
  }

  if(laserClient){
    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 1) != laserWavelengths_.end() ) {

      if ( meLaserL1_[0] ) dqmStore_->removeElement( meLaserL1_[0]->getName() );
      name = "EELT EE - laser quality summary L1";
      meLaserL1_[0] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
      meLaserL1_[0]->setAxisTitle("ix", 1);
      meLaserL1_[0]->setAxisTitle("iy", 2);

      if ( meLaserL1_[1] ) dqmStore_->removeElement( meLaserL1_[1]->getName() );
      name = "EELT EE + laser quality summary L1";
      meLaserL1_[1] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
      meLaserL1_[1]->setAxisTitle("ix", 1);
      meLaserL1_[1]->setAxisTitle("iy", 2);

      if ( meLaserL1Err_ ) dqmStore_->removeElement( meLaserL1Err_->getName() );
      name = "EELT laser quality errors summary L1";
      meLaserL1Err_ = dqmStore_->book1D(name, name, 18, 1, 19);
      for (int i = 0; i < 18; i++) {
	meLaserL1Err_->setBinLabel(i+1, Numbers::sEE(i+1), 1);
      }

      if ( meLaserL1PN_ ) dqmStore_->removeElement( meLaserL1PN_->getName() );
      name = "EELT PN laser quality summary L1";
      meLaserL1PN_ = dqmStore_->book2D(name, name, 45, 0., 45., 20, -10., 10.);
      meLaserL1PN_->setAxisTitle("jchannel", 1);
      meLaserL1PN_->setAxisTitle("jpseudo-strip", 2);

      if ( meLaserL1PNErr_ ) dqmStore_->removeElement( meLaserL1PNErr_->getName() );
      name = "EELT PN laser quality errors summary L1";
      meLaserL1PNErr_ = dqmStore_->book1D(name, name, 18, 1, 19);
      for (int i = 0; i < 18; i++) {
	meLaserL1PNErr_->setBinLabel(i+1, Numbers::sEE(i+1), 1);
      }

      if ( meLaserL1Ampl_ ) dqmStore_->removeElement( meLaserL1Ampl_->getName() );
      name = "EELT laser L1 amplitude summary";
      meLaserL1Ampl_ = dqmStore_->bookProfile(name, name, 18, 1, 19, 4096, 0., 4096., "s");
      for (int i = 0; i < 18; i++) {
	meLaserL1Ampl_->setBinLabel(i+1, Numbers::sEE(i+1), 1);
      }

      if ( meLaserL1Timing_ ) dqmStore_->removeElement( meLaserL1Timing_->getName() );
      name = "EELT laser L1 timing summary";
      meLaserL1Timing_ = dqmStore_->bookProfile(name, name, 18, 1, 19, 100, 0., 10., "s");
      for (int i = 0; i < 18; i++) {
	meLaserL1Timing_->setBinLabel(i+1, Numbers::sEE(i+1), 1);
      }

      if ( meLaserL1AmplOverPN_ ) dqmStore_->removeElement( meLaserL1AmplOverPN_->getName() );
      name = "EELT laser L1 amplitude over PN summary";
      meLaserL1AmplOverPN_ = dqmStore_->bookProfile(name, name, 18, 1, 19, 4096, 0., 4096.*12., "s");
      for (int i = 0; i < 18; i++) {
	meLaserL1AmplOverPN_->setBinLabel(i+1, Numbers::sEE(i+1), 1);
      }

    }

    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 2) != laserWavelengths_.end() ) {

      if ( meLaserL2_[0] ) dqmStore_->removeElement( meLaserL2_[0]->getName() );
      name = "EELT EE - laser quality summary L2";
      meLaserL2_[0] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
      meLaserL2_[0]->setAxisTitle("ix", 1);
      meLaserL2_[0]->setAxisTitle("iy", 2);

      if ( meLaserL2_[1] ) dqmStore_->removeElement( meLaserL2_[1]->getName() );
      name = "EELT EE + laser quality summary L2";
      meLaserL2_[1] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
      meLaserL2_[1]->setAxisTitle("ix", 1);
      meLaserL2_[1]->setAxisTitle("iy", 2);

      if ( meLaserL2Err_ ) dqmStore_->removeElement( meLaserL2Err_->getName() );
      name = "EELT laser quality errors summary L2";
      meLaserL2Err_ = dqmStore_->book1D(name, name, 18, 1, 19);
      for (int i = 0; i < 18; i++) {
	meLaserL2Err_->setBinLabel(i+1, Numbers::sEE(i+1), 1);
      }

      if ( meLaserL2PN_ ) dqmStore_->removeElement( meLaserL2PN_->getName() );
      name = "EELT PN laser quality summary L2";
      meLaserL2PN_ = dqmStore_->book2D(name, name, 45, 0., 45., 20, -10., 10.);
      meLaserL2PN_->setAxisTitle("jchannel", 1);
      meLaserL2PN_->setAxisTitle("jpseudo-strip", 2);

      if ( meLaserL2PNErr_ ) dqmStore_->removeElement( meLaserL2PNErr_->getName() );
      name = "EELT PN laser quality errors summary L2";
      meLaserL2PNErr_ = dqmStore_->book1D(name, name, 18, 1, 19);
      for (int i = 0; i < 18; i++) {
	meLaserL2PNErr_->setBinLabel(i+1, Numbers::sEE(i+1), 1);
      }

      if ( meLaserL2Ampl_ ) dqmStore_->removeElement( meLaserL2Ampl_->getName() );
      name = "EELT laser L2 amplitude summary";
      meLaserL2Ampl_ = dqmStore_->bookProfile(name, name, 18, 1, 19, 4096, 0., 4096., "s");
      for (int i = 0; i < 18; i++) {
	meLaserL2Ampl_->setBinLabel(i+1, Numbers::sEE(i+1), 1);
      }

      if ( meLaserL2Timing_ ) dqmStore_->removeElement( meLaserL2Timing_->getName() );
      name = "EELT laser L2 timing summary";
      meLaserL2Timing_ = dqmStore_->bookProfile(name, name, 18, 1, 19, 100, 0., 10., "s");
      for (int i = 0; i < 18; i++) {
	meLaserL2Timing_->setBinLabel(i+1, Numbers::sEE(i+1), 1);
      }

      if ( meLaserL2AmplOverPN_ ) dqmStore_->removeElement( meLaserL2AmplOverPN_->getName() );
      name = "EELT laser L2 amplitude over PN summary";
      meLaserL2AmplOverPN_ = dqmStore_->bookProfile(name, name, 18, 1, 19, 4096, 0., 4096.*12., "s");
      for (int i = 0; i < 18; i++) {
	meLaserL2AmplOverPN_->setBinLabel(i+1, Numbers::sEE(i+1), 1);
      }

    }

    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 3) != laserWavelengths_.end() ) {

      if ( meLaserL3_[0] ) dqmStore_->removeElement( meLaserL3_[0]->getName() );
      name = "EELT EE - laser quality summary L3";
      meLaserL3_[0] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
      meLaserL3_[0]->setAxisTitle("ix", 1);
      meLaserL3_[0]->setAxisTitle("iy", 2);

      if ( meLaserL3_[1] ) dqmStore_->removeElement( meLaserL3_[1]->getName() );
      name = "EELT EE + laser quality summary L3";
      meLaserL3_[1] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
      meLaserL3_[1]->setAxisTitle("ix", 1);
      meLaserL3_[1]->setAxisTitle("iy", 2);

      if ( meLaserL3Err_ ) dqmStore_->removeElement( meLaserL3Err_->getName() );
      name = "EELT laser quality errors summary L3";
      meLaserL3Err_ = dqmStore_->book1D(name, name, 18, 1, 19);
      for (int i = 0; i < 18; i++) {
	meLaserL3Err_->setBinLabel(i+1, Numbers::sEE(i+1), 1);
      }

      if ( meLaserL3PN_ ) dqmStore_->removeElement( meLaserL3PN_->getName() );
      name = "EELT PN laser quality summary L3";
      meLaserL3PN_ = dqmStore_->book2D(name, name, 45, 0., 45., 20, -10., 10.);
      meLaserL3PN_->setAxisTitle("jchannel", 1);
      meLaserL3PN_->setAxisTitle("jpseudo-strip", 2);

      if ( meLaserL3PNErr_ ) dqmStore_->removeElement( meLaserL3PNErr_->getName() );
      name = "EELT PN laser quality errors summary L3";
      meLaserL3PNErr_ = dqmStore_->book1D(name, name, 18, 1, 19);
      for (int i = 0; i < 18; i++) {
	meLaserL3PNErr_->setBinLabel(i+1, Numbers::sEE(i+1), 1);
      }

      if ( meLaserL3Ampl_ ) dqmStore_->removeElement( meLaserL3Ampl_->getName() );
      name = "EELT laser L3 amplitude summary";
      meLaserL3Ampl_ = dqmStore_->bookProfile(name, name, 18, 1, 19, 4096, 0., 4096., "s");
      for (int i = 0; i < 18; i++) {
	meLaserL3Ampl_->setBinLabel(i+1, Numbers::sEE(i+1), 1);
      }

      if ( meLaserL3Timing_ ) dqmStore_->removeElement( meLaserL3Timing_->getName() );
      name = "EELT laser L3 timing summary";
      meLaserL3Timing_ = dqmStore_->bookProfile(name, name, 18, 1, 19, 100, 0., 10., "s");
      for (int i = 0; i < 18; i++) {
	meLaserL3Timing_->setBinLabel(i+1, Numbers::sEE(i+1), 1);
      }

      if ( meLaserL3AmplOverPN_ ) dqmStore_->removeElement( meLaserL3AmplOverPN_->getName() );
      name = "EELT laser L3 amplitude over PN summary";
      meLaserL3AmplOverPN_ = dqmStore_->bookProfile(name, name, 18, 1, 19, 4096, 0., 4096.*12., "s");
      for (int i = 0; i < 18; i++) {
	meLaserL3AmplOverPN_->setBinLabel(i+1, Numbers::sEE(i+1), 1);
      }

    }

    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 4) != laserWavelengths_.end() ) {

      if ( meLaserL4_[0] ) dqmStore_->removeElement( meLaserL4_[0]->getName() );
      name = "EELT EE - laser quality summary L4";
      meLaserL4_[0] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
      meLaserL4_[0]->setAxisTitle("ix", 1);
      meLaserL4_[0]->setAxisTitle("iy", 2);

      if ( meLaserL4_[1] ) dqmStore_->removeElement( meLaserL4_[1]->getName() );
      name = "EELT EE + laser quality summary L4";
      meLaserL4_[1] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
      meLaserL4_[1]->setAxisTitle("ix", 1);
      meLaserL4_[1]->setAxisTitle("iy", 2);

      if ( meLaserL4Err_ ) dqmStore_->removeElement( meLaserL4Err_->getName() );
      name = "EELT laser quality errors summary L4";
      meLaserL4Err_ = dqmStore_->book1D(name, name, 18, 1, 19);
      for (int i = 0; i < 18; i++) {
	meLaserL4Err_->setBinLabel(i+1, Numbers::sEE(i+1), 1);
      }

      if ( meLaserL4PN_ ) dqmStore_->removeElement( meLaserL4PN_->getName() );
      name = "EELT PN laser quality summary L4";
      meLaserL4PN_ = dqmStore_->book2D(name, name, 45, 0., 45., 20, -10., 10.);
      meLaserL4PN_->setAxisTitle("jchannel", 1);
      meLaserL4PN_->setAxisTitle("jpseudo-strip", 2);

      if ( meLaserL4PNErr_ ) dqmStore_->removeElement( meLaserL4PNErr_->getName() );
      name = "EELT PN laser quality errors summary L4";
      meLaserL4PNErr_ = dqmStore_->book1D(name, name, 18, 1, 19);
      for (int i = 0; i < 18; i++) {
	meLaserL4PNErr_->setBinLabel(i+1, Numbers::sEE(i+1), 1);
      }

      if ( meLaserL4Ampl_ ) dqmStore_->removeElement( meLaserL4Ampl_->getName() );
      name = "EELT laser L4 amplitude summary";
      meLaserL4Ampl_ = dqmStore_->bookProfile(name, name, 18, 1, 19, 4096, 0., 4096., "s");
      for (int i = 0; i < 18; i++) {
	meLaserL4Ampl_->setBinLabel(i+1, Numbers::sEE(i+1), 1);
      }

      if ( meLaserL4Timing_ ) dqmStore_->removeElement( meLaserL4Timing_->getName() );
      name = "EELT laser L4 timing summary";
      meLaserL4Timing_ = dqmStore_->bookProfile(name, name, 18, 1, 19, 100, 0., 10., "s");
      for (int i = 0; i < 18; i++) {
	meLaserL4Timing_->setBinLabel(i+1, Numbers::sEE(i+1), 1);
      }

      if ( meLaserL4AmplOverPN_ ) dqmStore_->removeElement( meLaserL4AmplOverPN_->getName() );
      name = "EELT laser L4 amplitude over PN summary";
      meLaserL4AmplOverPN_ = dqmStore_->bookProfile(name, name, 18, 1, 19, 4096, 0., 4096.*12., "s");
      for (int i = 0; i < 18; i++) {
	meLaserL4AmplOverPN_->setBinLabel(i+1, Numbers::sEE(i+1), 1);
      }

    }
  }

  if(ledClient){
    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 1) != ledWavelengths_.end() ) {

      if ( meLedL1_[0] ) dqmStore_->removeElement( meLedL1_[0]->getName() );
      name = "EELDT EE - led quality summary L1";
      meLedL1_[0] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
      meLedL1_[0]->setAxisTitle("ix", 1);
      meLedL1_[0]->setAxisTitle("iy", 2);

      if ( meLedL1_[1] ) dqmStore_->removeElement( meLedL1_[1]->getName() );
      name = "EELDT EE + led quality summary L1";
      meLedL1_[1] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
      meLedL1_[1]->setAxisTitle("ix", 1);
      meLedL1_[1]->setAxisTitle("iy", 2);

      if ( meLedL1Err_ ) dqmStore_->removeElement( meLedL1Err_->getName() );
      name = "EELDT led quality errors summary L1";
      meLedL1Err_ = dqmStore_->book1D(name, name, 18, 1, 19);
      for (int i = 0; i < 18; i++) {
	meLedL1Err_->setBinLabel(i+1, Numbers::sEE(i+1), 1);
      }

      if ( meLedL1PN_ ) dqmStore_->removeElement( meLedL1PN_->getName() );
      name = "EELDT PN led quality summary L1";
      meLedL1PN_ = dqmStore_->book2D(name, name, 45, 0., 45., 20, -10., 10.);
      meLedL1PN_->setAxisTitle("jchannel", 1);
      meLedL1PN_->setAxisTitle("jpseudo-strip", 2);

      if ( meLedL1PNErr_ ) dqmStore_->removeElement( meLedL1PNErr_->getName() );
      name = "EELDT PN led quality errors summary L1";
      meLedL1PNErr_ = dqmStore_->book1D(name, name, 18, 1, 19);
      for (int i = 0; i < 18; i++) {
	meLedL1PNErr_->setBinLabel(i+1, Numbers::sEE(i+1), 1);
      }

      if ( meLedL1Ampl_ ) dqmStore_->removeElement( meLedL1Ampl_->getName() );
      name = "EELDT led L1 amplitude summary";
      meLedL1Ampl_ = dqmStore_->bookProfile(name, name, 18, 1, 19, 4096, 0., 4096., "s");
      for (int i = 0; i < 18; i++) {
	meLedL1Ampl_->setBinLabel(i+1, Numbers::sEE(i+1), 1);
      }

      if ( meLedL1Timing_ ) dqmStore_->removeElement( meLedL1Timing_->getName() );
      name = "EELDT led L1 timing summary";
      meLedL1Timing_ = dqmStore_->bookProfile(name, name, 18, 1, 19, 100, 0., 10., "s");
      for (int i = 0; i < 18; i++) {
	meLedL1Timing_->setBinLabel(i+1, Numbers::sEE(i+1), 1);
      }

      if ( meLedL1AmplOverPN_ ) dqmStore_->removeElement( meLedL1AmplOverPN_->getName() );
      name = "EELDT led L1 amplitude over PN summary";
      meLedL1AmplOverPN_ = dqmStore_->bookProfile(name, name, 18, 1, 19, 4096, 0., 4096.*12., "s");
      for (int i = 0; i < 18; i++) {
	meLedL1AmplOverPN_->setBinLabel(i+1, Numbers::sEE(i+1), 1);
      }

    }

    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 2) != ledWavelengths_.end() ) {

      if ( meLedL2_[0] ) dqmStore_->removeElement( meLedL2_[0]->getName() );
      name = "EELDT EE - led quality summary L2";
      meLedL2_[0] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
      meLedL2_[0]->setAxisTitle("ix", 1);
      meLedL2_[0]->setAxisTitle("iy", 2);

      if ( meLedL2_[1] ) dqmStore_->removeElement( meLedL2_[1]->getName() );
      name = "EELDT EE + led quality summary L2";
      meLedL2_[1] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
      meLedL2_[1]->setAxisTitle("ix", 1);
      meLedL2_[1]->setAxisTitle("iy", 2);

      if ( meLedL2Err_ ) dqmStore_->removeElement( meLedL2Err_->getName() );
      name = "EELDT led quality errors summary L2";
      meLedL2Err_ = dqmStore_->book1D(name, name, 18, 1, 19);
      for (int i = 0; i < 18; i++) {
	meLedL2Err_->setBinLabel(i+1, Numbers::sEE(i+1), 1);
      }

      if ( meLedL2PN_ ) dqmStore_->removeElement( meLedL2PN_->getName() );
      name = "EELDT PN led quality summary L2";
      meLedL2PN_ = dqmStore_->book2D(name, name, 45, 0., 45., 20, -10., 10.);
      meLedL2PN_->setAxisTitle("jchannel", 1);
      meLedL2PN_->setAxisTitle("jpseudo-strip", 2);

      if ( meLedL2PNErr_ ) dqmStore_->removeElement( meLedL2PNErr_->getName() );
      name = "EELDT PN led quality errors summary L2";
      meLedL2PNErr_ = dqmStore_->book1D(name, name, 18, 1, 19);
      for (int i = 0; i < 18; i++) {
	meLedL2PNErr_->setBinLabel(i+1, Numbers::sEE(i+1), 1);
      }

      if ( meLedL2Ampl_ ) dqmStore_->removeElement( meLedL2Ampl_->getName() );
      name = "EELDT led L2 amplitude summary";
      meLedL2Ampl_ = dqmStore_->bookProfile(name, name, 18, 1, 19, 4096, 0., 4096., "s");
      for (int i = 0; i < 18; i++) {
	meLedL2Ampl_->setBinLabel(i+1, Numbers::sEE(i+1), 1);
      }

      if ( meLedL2Timing_ ) dqmStore_->removeElement( meLedL2Timing_->getName() );
      name = "EELDT led L2 timing summary";
      meLedL2Timing_ = dqmStore_->bookProfile(name, name, 18, 1, 19, 100, 0., 10., "s");
      for (int i = 0; i < 18; i++) {
	meLedL2Timing_->setBinLabel(i+1, Numbers::sEE(i+1), 1);
      }

      if ( meLedL2AmplOverPN_ ) dqmStore_->removeElement( meLedL2AmplOverPN_->getName() );
      name = "EELDT led L2 amplitude over PN summary";
      meLedL2AmplOverPN_ = dqmStore_->bookProfile(name, name, 18, 1, 19, 4096, 0., 4096.*12., "s");
      for (int i = 0; i < 18; i++) {
	meLedL2AmplOverPN_->setBinLabel(i+1, Numbers::sEE(i+1), 1);
      }

    }
  }

  if(pedestalClient){
    if (find(MGPAGains_.begin(), MGPAGains_.end(), 1) != MGPAGains_.end() ) {

      if( mePedestalG01_[0] ) dqmStore_->removeElement( mePedestalG01_[0]->getName() );
      name = "EEPT EE - pedestal quality G01 summary";
      mePedestalG01_[0] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
      mePedestalG01_[0]->setAxisTitle("ix", 1);
      mePedestalG01_[0]->setAxisTitle("iy", 2);

    }

    if (find(MGPAGains_.begin(), MGPAGains_.end(), 6) != MGPAGains_.end() ) {

      if( mePedestalG06_[0] ) dqmStore_->removeElement( mePedestalG06_[0]->getName() );
      name = "EEPT EE - pedestal quality G06 summary";
      mePedestalG06_[0] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
      mePedestalG06_[0]->setAxisTitle("ix", 1);
      mePedestalG06_[0]->setAxisTitle("iy", 2);

    }

    if (find(MGPAGains_.begin(), MGPAGains_.end(), 12) != MGPAGains_.end() ) {

      if( mePedestalG12_[0] ) dqmStore_->removeElement( mePedestalG12_[0]->getName() );
      name = "EEPT EE - pedestal quality G12 summary";
      mePedestalG12_[0] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
      mePedestalG12_[0]->setAxisTitle("ix", 1);
      mePedestalG12_[0]->setAxisTitle("iy", 2);

    }


    if (find(MGPAGainsPN_.begin(), MGPAGainsPN_.end(), 1) != MGPAGainsPN_.end() ) {

      if( mePedestalPNG01_ ) dqmStore_->removeElement( mePedestalPNG01_->getName() );
      name = "EEPT PN pedestal quality G01 summary";
      mePedestalPNG01_ = dqmStore_->book2D(name, name, 45, 0., 45., 20, -10, 10.);
      mePedestalPNG01_->setAxisTitle("jchannel", 1);
      mePedestalPNG01_->setAxisTitle("jpseudo-strip", 2);

    }

    if (find(MGPAGainsPN_.begin(), MGPAGainsPN_.end(), 16) != MGPAGainsPN_.end() ) {

      if( mePedestalPNG16_ ) dqmStore_->removeElement( mePedestalPNG16_->getName() );
      name = "EEPT PN pedestal quality G16 summary";
      mePedestalPNG16_ = dqmStore_->book2D(name, name, 45, 0., 45., 20, -10, 10.);
      mePedestalPNG16_->setAxisTitle("jchannel", 1);
      mePedestalPNG16_->setAxisTitle("jpseudo-strip", 2);

    }

    if (find(MGPAGains_.begin(), MGPAGains_.end(), 1) != MGPAGains_.end() ) {

      if( mePedestalG01_[1] ) dqmStore_->removeElement( mePedestalG01_[1]->getName() );
      name = "EEPT EE + pedestal quality G01 summary";
      mePedestalG01_[1] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
      mePedestalG01_[1]->setAxisTitle("ix", 1);
      mePedestalG01_[1]->setAxisTitle("iy", 2);

    }


    if (find(MGPAGains_.begin(), MGPAGains_.end(), 6) != MGPAGains_.end() ) {

      if( mePedestalG06_[1] ) dqmStore_->removeElement( mePedestalG06_[1]->getName() );
      name = "EEPT EE + pedestal quality G06 summary";
      mePedestalG06_[1] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
      mePedestalG06_[1]->setAxisTitle("ix", 1);
      mePedestalG06_[1]->setAxisTitle("iy", 2);

    }

    if (find(MGPAGains_.begin(), MGPAGains_.end(), 12) != MGPAGains_.end() ) {

      if( mePedestalG12_[1] ) dqmStore_->removeElement( mePedestalG12_[1]->getName() );
      name = "EEPT EE + pedestal quality G12 summary";
      mePedestalG12_[1] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
      mePedestalG12_[1]->setAxisTitle("ix", 1);
      mePedestalG12_[1]->setAxisTitle("iy", 2);

    }
  }

  if(testPulseClient){
    if (find(MGPAGains_.begin(), MGPAGains_.end(), 1) != MGPAGains_.end() ) {

      if( meTestPulseG01_[0] ) dqmStore_->removeElement( meTestPulseG01_[0]->getName() );
      name = "EETPT EE - test pulse quality G01 summary";
      meTestPulseG01_[0] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
      meTestPulseG01_[0]->setAxisTitle("ix", 1);
      meTestPulseG01_[0]->setAxisTitle("iy", 2);

    }

    if (find(MGPAGains_.begin(), MGPAGains_.end(), 6) != MGPAGains_.end() ) {

      if( meTestPulseG06_[0] ) dqmStore_->removeElement( meTestPulseG06_[0]->getName() );
      name = "EETPT EE - test pulse quality G06 summary";
      meTestPulseG06_[0] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
      meTestPulseG06_[0]->setAxisTitle("ix", 1);
      meTestPulseG06_[0]->setAxisTitle("iy", 2);

    }

    if (find(MGPAGains_.begin(), MGPAGains_.end(), 12) != MGPAGains_.end() ) {

      if( meTestPulseG12_[0] ) dqmStore_->removeElement( meTestPulseG12_[0]->getName() );
      name = "EETPT EE - test pulse quality G12 summary";
      meTestPulseG12_[0] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
      meTestPulseG12_[0]->setAxisTitle("ix", 1);
      meTestPulseG12_[0]->setAxisTitle("iy", 2);

    }


    if (find(MGPAGainsPN_.begin(), MGPAGainsPN_.end(), 1) != MGPAGainsPN_.end() ) {

      if( meTestPulsePNG01_ ) dqmStore_->removeElement( meTestPulsePNG01_->getName() );
      name = "EETPT PN test pulse quality G01 summary";
      meTestPulsePNG01_ = dqmStore_->book2D(name, name, 45, 0., 45., 20, -10., 10.);
      meTestPulsePNG01_->setAxisTitle("jchannel", 1);
      meTestPulsePNG01_->setAxisTitle("jpseudo-strip", 2);

    }

    if (find(MGPAGainsPN_.begin(), MGPAGainsPN_.end(), 16) != MGPAGainsPN_.end() ) {

      if( meTestPulsePNG16_ ) dqmStore_->removeElement( meTestPulsePNG16_->getName() );
      name = "EETPT PN test pulse quality G16 summary";
      meTestPulsePNG16_ = dqmStore_->book2D(name, name, 45, 0., 45., 20, -10., 10.);
      meTestPulsePNG16_->setAxisTitle("jchannel", 1);
      meTestPulsePNG16_->setAxisTitle("jpseudo-strip", 2);

    }

    if (find(MGPAGains_.begin(), MGPAGains_.end(), 1) != MGPAGains_.end() ) {

      if( meTestPulseG01_[1] ) dqmStore_->removeElement( meTestPulseG01_[1]->getName() );
      name = "EETPT EE + test pulse quality G01 summary";
      meTestPulseG01_[1] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
      meTestPulseG01_[1]->setAxisTitle("ix", 1);
      meTestPulseG01_[1]->setAxisTitle("iy", 2);

    }

    if (find(MGPAGains_.begin(), MGPAGains_.end(), 6) != MGPAGains_.end() ) {

      if( meTestPulseG06_[1] ) dqmStore_->removeElement( meTestPulseG06_[1]->getName() );
      name = "EETPT EE + test pulse quality G06 summary";
      meTestPulseG06_[1] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
      meTestPulseG06_[1]->setAxisTitle("ix", 1);
      meTestPulseG06_[1]->setAxisTitle("iy", 2);

    }

    if (find(MGPAGains_.begin(), MGPAGains_.end(), 12) != MGPAGains_.end() ) {

      if( meTestPulseG12_[1] ) dqmStore_->removeElement( meTestPulseG12_[1]->getName() );
      name = "EETPT EE + test pulse quality G12 summary";
      meTestPulseG12_[1] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
      meTestPulseG12_[1]->setAxisTitle("ix", 1);
      meTestPulseG12_[1]->setAxisTitle("iy", 2);

    }

    if (find(MGPAGains_.begin(), MGPAGains_.end(), 1) != MGPAGains_.end() ) {

      if( meTestPulseAmplG01_ ) dqmStore_->removeElement( meTestPulseAmplG01_->getName() );
      name = "EETPT test pulse amplitude G01 summary";
      meTestPulseAmplG01_ = dqmStore_->bookProfile(name, name, 18, 1, 19, 4096, 0., 4096.*12., "s");
      for (int i = 0; i < 18; i++) {
	meTestPulseAmplG01_->setBinLabel(i+1, Numbers::sEE(i+1), 1);
      }

    }

    if (find(MGPAGains_.begin(), MGPAGains_.end(), 6) != MGPAGains_.end() ) {

      if( meTestPulseAmplG06_ ) dqmStore_->removeElement( meTestPulseAmplG06_->getName() );
      name = "EETPT test pulse amplitude G06 summary";
      meTestPulseAmplG06_ = dqmStore_->bookProfile(name, name, 18, 1, 19, 4096, 0., 4096.*12., "s");
      for (int i = 0; i < 18; i++) {
	meTestPulseAmplG06_->setBinLabel(i+1, Numbers::sEE(i+1), 1);
      }

    }

    if (find(MGPAGains_.begin(), MGPAGains_.end(), 12) != MGPAGains_.end() ) {

      if( meTestPulseAmplG12_ ) dqmStore_->removeElement( meTestPulseAmplG12_->getName() );
      name = "EETPT test pulse amplitude G12 summary";
      meTestPulseAmplG12_ = dqmStore_->bookProfile(name, name, 18, 1, 19, 4096, 0., 4096.*12., "s");
      for (int i = 0; i < 18; i++) {
	meTestPulseAmplG12_->setBinLabel(i+1, Numbers::sEE(i+1), 1);
      }

    }
  }

  if(timingClient){

    if( meTiming_[0] ) dqmStore_->removeElement( meTiming_[0]->getName() );
    name = "EETMT EE - timing quality summary";
    meTiming_[0] = dqmStore_->book2D(name, name, 20, 0., 100., 20, 0., 100.);
    meTiming_[0]->setAxisTitle("ix", 1);
    meTiming_[0]->setAxisTitle("iy", 2);

    if( meTiming_[1] ) dqmStore_->removeElement( meTiming_[1]->getName() );
    name = "EETMT EE + timing quality summary";
    meTiming_[1] = dqmStore_->book2D(name, name, 20, 0., 100., 20, 0., 100.);
    meTiming_[1]->setAxisTitle("ix", 1);
    meTiming_[1]->setAxisTitle("iy", 2);

    if( meTimingMean1D_[0] ) dqmStore_->removeElement( meTimingMean1D_[0]->getName() );
    name = "EETMT EE - timing mean 1D summary";
    meTimingMean1D_[0] = dqmStore_->book1D(name, name, 100, -25., 25.);
    meTimingMean1D_[0]->setAxisTitle("mean (ns)", 1);

    if( meTimingMean1D_[1] ) dqmStore_->removeElement( meTimingMean1D_[1]->getName() );
    name = "EETMT EE + timing mean 1D summary";
    meTimingMean1D_[1] = dqmStore_->book1D(name, name, 100, -25., 25.);
    meTimingMean1D_[1]->setAxisTitle("mean (ns)", 1);

    if( meTimingRMS1D_[0] ) dqmStore_->removeElement( meTimingRMS1D_[0]->getName() );
    name = "EETMT EE - timing rms 1D summary";
    meTimingRMS1D_[0] = dqmStore_->book1D(name, name, 100, 0.0, 10.0);
    meTimingRMS1D_[0]->setAxisTitle("rms (ns)", 1);

    if( meTimingRMS1D_[1] ) dqmStore_->removeElement( meTimingRMS1D_[1]->getName() );
    name = "EETMT EE + timing rms 1D summary";
    meTimingRMS1D_[1] = dqmStore_->book1D(name, name, 100, 0.0, 10.0);
    meTimingRMS1D_[1]->setAxisTitle("rms (ns)", 1);

    if ( meTimingMean_ ) dqmStore_->removeElement( meTimingMean_->getName() );
    name = "EETMT timing mean";
    meTimingMean_ = dqmStore_->bookProfile(name, name, 18, 1, 19, 100, -20., 20.,"");
    for (int i = 0; i < 18; i++) {
      meTimingMean_->setBinLabel(i+1, Numbers::sEE(i+1), 1);
    }
    meTimingMean_->setAxisTitle("mean (ns)", 2);

    if ( meTimingRMS_ ) dqmStore_->removeElement( meTimingRMS_->getName() );
    name = "EETMT timing rms";
    meTimingRMS_ = dqmStore_->bookProfile(name, name, 18, 1, 19, 100, 0., 10.,"");
    for (int i = 0; i < 18; i++) {
      meTimingRMS_->setBinLabel(i+1, Numbers::sEE(i+1), 1);
    }
    meTimingRMS_->setAxisTitle("rms (ns)", 2);
  }

  if(triggerTowerClient){
    if( meTriggerTowerEt_[0] ) dqmStore_->removeElement( meTriggerTowerEt_[0]->getName() );
    name = "EETTT EE - Et trigger tower summary";
    meTriggerTowerEt_[0] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
    meTriggerTowerEt_[0]->setAxisTitle("ix", 1);
    meTriggerTowerEt_[0]->setAxisTitle("iy", 2);
    meTriggerTowerEt_[0]->setAxisTitle("Et (GeV)", 3);

    if( meTriggerTowerEt_[1] ) dqmStore_->removeElement( meTriggerTowerEt_[1]->getName() );
    name = "EETTT EE + Et trigger tower summary";
    meTriggerTowerEt_[1] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
    meTriggerTowerEt_[1]->setAxisTitle("ix", 1);
    meTriggerTowerEt_[1]->setAxisTitle("iy", 2);
    meTriggerTowerEt_[1]->setAxisTitle("Et (GeV)", 3);

    if( meTriggerTowerEmulError_[0] ) dqmStore_->removeElement( meTriggerTowerEmulError_[0]->getName() );
    name = "EETTT EE - emulator error quality summary";
    meTriggerTowerEmulError_[0] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
    meTriggerTowerEmulError_[0]->setAxisTitle("ix", 1);
    meTriggerTowerEmulError_[0]->setAxisTitle("iy", 2);

    if( meTriggerTowerEmulError_[1] ) dqmStore_->removeElement( meTriggerTowerEmulError_[1]->getName() );
    name = "EETTT EE + emulator error quality summary";
    meTriggerTowerEmulError_[1] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
    meTriggerTowerEmulError_[1]->setAxisTitle("ix", 1);
    meTriggerTowerEmulError_[1]->setAxisTitle("iy", 2);

    if( meTriggerTowerTiming_[0] ) dqmStore_->removeElement( meTriggerTowerTiming_[0]->getName() );
    name = "EETTT EE - Trigger Primitives Timing summary";
    meTriggerTowerTiming_[0] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
    meTriggerTowerTiming_[0]->setAxisTitle("ix", 1);
    meTriggerTowerTiming_[0]->setAxisTitle("iy", 2);
    meTriggerTowerTiming_[0]->setAxisTitle("TP data matching emulator", 3);

    if( meTriggerTowerTiming_[1] ) dqmStore_->removeElement( meTriggerTowerTiming_[1]->getName() );
    name = "EETTT EE + Trigger Primitives Timing summary";
    meTriggerTowerTiming_[1] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
    meTriggerTowerTiming_[1]->setAxisTitle("ix", 1);
    meTriggerTowerTiming_[1]->setAxisTitle("iy", 2);
    meTriggerTowerTiming_[1]->setAxisTitle("TP data matching emulator", 3);

    if( meTriggerTowerNonSingleTiming_[0] ) dqmStore_->removeElement( meTriggerTowerNonSingleTiming_[0]->getName() );
    name = "EETTT EE - Trigger Primitives Non Single Timing summary";
    meTriggerTowerNonSingleTiming_[0] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
    meTriggerTowerNonSingleTiming_[0]->setAxisTitle("ix", 1);
    meTriggerTowerNonSingleTiming_[0]->setAxisTitle("iy", 2);
    meTriggerTowerNonSingleTiming_[0]->setAxisTitle("fraction", 3);

    if( meTriggerTowerNonSingleTiming_[1] ) dqmStore_->removeElement( meTriggerTowerNonSingleTiming_[1]->getName() );
    name = "EETTT EE + Trigger Primitives Non Single Timing summary";
    meTriggerTowerNonSingleTiming_[1] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
    meTriggerTowerNonSingleTiming_[1]->setAxisTitle("ix", 1);
    meTriggerTowerNonSingleTiming_[1]->setAxisTitle("iy", 2);
    meTriggerTowerNonSingleTiming_[1]->setAxisTitle("fraction", 3);
  }

  if(meIntegrity_[0] && mePedestalOnline_[0] && meStatusFlags_[0] && (reducedReports_ || (meTiming_[0] && meTriggerTowerEmulError_[0]))) {
    if( meGlobalSummary_[0] ) dqmStore_->removeElement( meGlobalSummary_[0]->getName() );
    name = "EE global summary EE -";
    meGlobalSummary_[0] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
    meGlobalSummary_[0]->setAxisTitle("ix", 1);
    meGlobalSummary_[0]->setAxisTitle("iy", 2);
  }

  if(meIntegrity_[1] && mePedestalOnline_[1] && meStatusFlags_[1] && (reducedReports_ || (meTiming_[1] && meTriggerTowerEmulError_[1]))) {
    if( meGlobalSummary_[1] ) dqmStore_->removeElement( meGlobalSummary_[1]->getName() );
    name = "EE global summary EE +";
    meGlobalSummary_[1] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
    meGlobalSummary_[1]->setAxisTitle("ix", 1);
    meGlobalSummary_[1]->setAxisTitle("iy", 2);
  }

  if(meGlobalSummary_[0] && meGlobalSummary_[1]){
    if(meSummaryErr_) dqmStore_->removeElement(meSummaryErr_->getName());
    name = "EE global summary errors";
    meSummaryErr_ = dqmStore_->book1D(name, name, 1, 0., 1.);
  }
}

void EESummaryClient::cleanup(void) {

  if ( ! enableCleanup_ ) return;

  dqmStore_->setCurrentFolder( prefixME_ + "/EESummaryClient" );

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( cloneME_ ) {
      if ( hpot01_[ism-1] ) delete hpot01_[ism-1];
      if ( httt01_[ism-1] ) delete httt01_[ism-1];
    }

    hpot01_[ism-1] = 0;
    httt01_[ism-1] = 0;

  }

  if ( meIntegrity_[0] ) dqmStore_->removeElement( meIntegrity_[0]->getName() );
  meIntegrity_[0] = 0;

  if ( meIntegrity_[1] ) dqmStore_->removeElement( meIntegrity_[1]->getName() );
  meIntegrity_[1] = 0;

  if ( meIntegrityErr_ ) dqmStore_->removeElement( meIntegrityErr_->getName() );
  meIntegrityErr_ = 0;

  if ( meIntegrityPN_ ) dqmStore_->removeElement( meIntegrityPN_->getName() );
  meIntegrityPN_ = 0;

  if ( meOccupancy_[0] ) dqmStore_->removeElement( meOccupancy_[0]->getName() );
  meOccupancy_[0] = 0;

  if ( meOccupancy_[1] ) dqmStore_->removeElement( meOccupancy_[1]->getName() );
  meOccupancy_[1] = 0;

  if ( meOccupancy1D_ ) dqmStore_->removeElement( meOccupancy1D_->getName() );
  meOccupancy1D_ = 0;

  if ( meOccupancyPN_ ) dqmStore_->removeElement( meOccupancyPN_->getName() );
  meOccupancyPN_ = 0;

  if ( meStatusFlags_[0] ) dqmStore_->removeElement( meStatusFlags_[0]->getName() );
  meStatusFlags_[0] = 0;

  if ( meStatusFlags_[1] ) dqmStore_->removeElement( meStatusFlags_[1]->getName() );
  meStatusFlags_[1] = 0;

  if ( meStatusFlagsErr_ ) dqmStore_->removeElement( meStatusFlagsErr_->getName() );
  meStatusFlagsErr_ = 0;

  if ( mePedestalOnline_[0] ) dqmStore_->removeElement( mePedestalOnline_[0]->getName() );
  mePedestalOnline_[0] = 0;

  if ( mePedestalOnline_[1] ) dqmStore_->removeElement( mePedestalOnline_[1]->getName() );
  mePedestalOnline_[1] = 0;

  if ( mePedestalOnlineErr_ ) dqmStore_->removeElement( mePedestalOnlineErr_->getName() );
  mePedestalOnlineErr_ = 0;

  if ( mePedestalOnlineMean_ ) dqmStore_->removeElement( mePedestalOnlineMean_->getName() );
  mePedestalOnlineMean_ = 0;

  if ( mePedestalOnlineRMS_ ) dqmStore_->removeElement( mePedestalOnlineRMS_->getName() );
  mePedestalOnlineRMS_ = 0;

  if ( mePedestalOnlineRMSMap_[0] ) dqmStore_->removeElement( mePedestalOnlineRMSMap_[0]->getName() );
  mePedestalOnlineRMSMap_[0] = 0;

  if ( mePedestalOnlineRMSMap_[1] ) dqmStore_->removeElement( mePedestalOnlineRMSMap_[1]->getName() );
  mePedestalOnlineRMSMap_[1] = 0;

  if ( meLaserL1_[0] ) dqmStore_->removeElement( meLaserL1_[0]->getName() );
  meLaserL1_[0] = 0;

  if ( meLaserL1_[1] ) dqmStore_->removeElement( meLaserL1_[1]->getName() );
  meLaserL1_[1] = 0;

  if ( meLaserL1Err_ ) dqmStore_->removeElement( meLaserL1Err_->getName() );
  meLaserL1Err_ = 0;

  if ( meLaserL1PN_ ) dqmStore_->removeElement( meLaserL1PN_->getName() );
  meLaserL1PN_ = 0;

  if ( meLaserL1PNErr_ ) dqmStore_->removeElement( meLaserL1PNErr_->getName() );
  meLaserL1PNErr_ = 0;

  if ( meLaserL1Ampl_ ) dqmStore_->removeElement( meLaserL1Ampl_->getName() );
  meLaserL1Ampl_ = 0;

  if ( meLaserL1Timing_ ) dqmStore_->removeElement( meLaserL1Timing_->getName() );
  meLaserL1Timing_ = 0;

  if ( meLaserL1AmplOverPN_ ) dqmStore_->removeElement( meLaserL1AmplOverPN_->getName() );
  meLaserL1AmplOverPN_ = 0;

  if ( meLaserL2_[0] ) dqmStore_->removeElement( meLaserL2_[0]->getName() );
  meLaserL2_[0] = 0;

  if ( meLaserL2_[1] ) dqmStore_->removeElement( meLaserL2_[1]->getName() );
  meLaserL2_[1] = 0;

  if ( meLaserL2Err_ ) dqmStore_->removeElement( meLaserL2Err_->getName() );
  meLaserL2Err_ = 0;

  if ( meLaserL2PN_ ) dqmStore_->removeElement( meLaserL2PN_->getName() );
  meLaserL2PN_ = 0;

  if ( meLaserL2PNErr_ ) dqmStore_->removeElement( meLaserL2PNErr_->getName() );
  meLaserL2PNErr_ = 0;

  if ( meLaserL2Ampl_ ) dqmStore_->removeElement( meLaserL2Ampl_->getName() );
  meLaserL2Ampl_ = 0;

  if ( meLaserL2Timing_ ) dqmStore_->removeElement( meLaserL2Timing_->getName() );
  meLaserL2Timing_ = 0;

  if ( meLaserL2AmplOverPN_ ) dqmStore_->removeElement( meLaserL2AmplOverPN_->getName() );
  meLaserL2AmplOverPN_ = 0;

  if ( meLaserL3_[0] ) dqmStore_->removeElement( meLaserL3_[0]->getName() );
  meLaserL3_[0] = 0;

  if ( meLaserL3_[1] ) dqmStore_->removeElement( meLaserL3_[1]->getName() );
  meLaserL3_[1] = 0;

  if ( meLaserL3Err_ ) dqmStore_->removeElement( meLaserL3Err_->getName() );
  meLaserL3Err_ = 0;

  if ( meLaserL3PN_ ) dqmStore_->removeElement( meLaserL3PN_->getName() );
  meLaserL3PN_ = 0;

  if ( meLaserL3PNErr_ ) dqmStore_->removeElement( meLaserL3PNErr_->getName() );
  meLaserL3PNErr_ = 0;

  if ( meLaserL3Ampl_ ) dqmStore_->removeElement( meLaserL3Ampl_->getName() );
  meLaserL3Ampl_ = 0;

  if ( meLaserL3Timing_ ) dqmStore_->removeElement( meLaserL3Timing_->getName() );
  meLaserL3Timing_ = 0;

  if ( meLaserL3AmplOverPN_ ) dqmStore_->removeElement( meLaserL3AmplOverPN_->getName() );
  meLaserL3AmplOverPN_ = 0;

  if ( meLaserL4_[0] ) dqmStore_->removeElement( meLaserL4_[0]->getName() );
  meLaserL4_[0] = 0;

  if ( meLaserL4_[1] ) dqmStore_->removeElement( meLaserL4_[1]->getName() );
  meLaserL4_[1] = 0;

  if ( meLaserL4Err_ ) dqmStore_->removeElement( meLaserL4Err_->getName() );
  meLaserL4Err_ = 0;

  if ( meLaserL4PN_ ) dqmStore_->removeElement( meLaserL4PN_->getName() );
  meLaserL4PN_ = 0;

  if ( meLaserL4PNErr_ ) dqmStore_->removeElement( meLaserL4PNErr_->getName() );
  meLaserL4PNErr_ = 0;

  if ( meLaserL4Ampl_ ) dqmStore_->removeElement( meLaserL4Ampl_->getName() );
  meLaserL4Ampl_ = 0;

  if ( meLaserL4Timing_ ) dqmStore_->removeElement( meLaserL4Timing_->getName() );
  meLaserL4Timing_ = 0;

  if ( meLaserL4AmplOverPN_ ) dqmStore_->removeElement( meLaserL4AmplOverPN_->getName() );
  meLaserL4AmplOverPN_ = 0;

  if ( meLedL1_[0] ) dqmStore_->removeElement( meLedL1_[0]->getName() );
  meLedL1_[0] = 0;

  if ( meLedL1_[1] ) dqmStore_->removeElement( meLedL1_[1]->getName() );
  meLedL1_[1] = 0;

  if ( meLedL1Err_ ) dqmStore_->removeElement( meLedL1Err_->getName() );
  meLedL1Err_ = 0;

  if ( meLedL1PN_ ) dqmStore_->removeElement( meLedL1PN_->getName() );
  meLedL1PN_ = 0;

  if ( meLedL1PNErr_ ) dqmStore_->removeElement( meLedL1PNErr_->getName() );
  meLedL1PNErr_ = 0;

  if ( meLedL1Ampl_ ) dqmStore_->removeElement( meLedL1Ampl_->getName() );
  meLedL1Ampl_ = 0;

  if ( meLedL1Timing_ ) dqmStore_->removeElement( meLedL1Timing_->getName() );
  meLedL1Timing_ = 0;

  if ( meLedL1AmplOverPN_ ) dqmStore_->removeElement( meLedL1AmplOverPN_->getName() );
  meLedL1AmplOverPN_ = 0;

  if ( meLedL2_[0] ) dqmStore_->removeElement( meLedL2_[0]->getName() );
  meLedL2_[0] = 0;

  if ( meLedL2_[1] ) dqmStore_->removeElement( meLedL2_[1]->getName() );
  meLedL2_[1] = 0;

  if ( meLedL2Err_ ) dqmStore_->removeElement( meLedL2Err_->getName() );
  meLedL2Err_ = 0;

  if ( meLedL2PN_ ) dqmStore_->removeElement( meLedL2PN_->getName() );
  meLedL2PN_ = 0;

  if ( meLedL2PNErr_ ) dqmStore_->removeElement( meLedL2PNErr_->getName() );
  meLedL2PNErr_ = 0;

  if ( meLedL2Ampl_ ) dqmStore_->removeElement( meLedL2Ampl_->getName() );
  meLedL2Ampl_ = 0;

  if ( meLedL2Timing_ ) dqmStore_->removeElement( meLedL2Timing_->getName() );
  meLedL2Timing_ = 0;

  if ( meLedL2AmplOverPN_ ) dqmStore_->removeElement( meLedL2AmplOverPN_->getName() );
  meLedL2AmplOverPN_ = 0;

  if ( mePedestalG01_[0] ) dqmStore_->removeElement( mePedestalG01_[0]->getName() );
  mePedestalG01_[0] = 0;

  if ( mePedestalG01_[1] ) dqmStore_->removeElement( mePedestalG01_[1]->getName() );
  mePedestalG01_[1] = 0;

  if ( mePedestalG06_[0] ) dqmStore_->removeElement( mePedestalG06_[0]->getName() );
  mePedestalG06_[0] = 0;

  if ( mePedestalG06_[1] ) dqmStore_->removeElement( mePedestalG06_[1]->getName() );
  mePedestalG06_[1] = 0;

  if ( mePedestalG12_[0] ) dqmStore_->removeElement( mePedestalG12_[0]->getName() );
  mePedestalG12_[0] = 0;

  if ( mePedestalG12_[1] ) dqmStore_->removeElement( mePedestalG12_[1]->getName() );
  mePedestalG12_[1] = 0;

  if ( mePedestalPNG01_ ) dqmStore_->removeElement( mePedestalPNG01_->getName() );
  mePedestalPNG01_ = 0;

  if ( mePedestalPNG16_ ) dqmStore_->removeElement( mePedestalPNG16_->getName() );
  mePedestalPNG16_ = 0;

  if ( meTestPulseG01_[0] ) dqmStore_->removeElement( meTestPulseG01_[0]->getName() );
  meTestPulseG01_[0] = 0;

  if ( meTestPulseG01_[1] ) dqmStore_->removeElement( meTestPulseG01_[1]->getName() );
  meTestPulseG01_[1] = 0;

  if ( meTestPulseG06_[0] ) dqmStore_->removeElement( meTestPulseG06_[0]->getName() );
  meTestPulseG06_[0] = 0;

  if ( meTestPulseG06_[1] ) dqmStore_->removeElement( meTestPulseG06_[1]->getName() );
  meTestPulseG06_[1] = 0;

  if ( meTestPulseG12_[0] ) dqmStore_->removeElement( meTestPulseG12_[0]->getName() );
  meTestPulseG12_[0] = 0;

  if ( meTestPulseG12_[1] ) dqmStore_->removeElement( meTestPulseG12_[1]->getName() );
  meTestPulseG12_[1] = 0;

  if ( meTestPulsePNG01_ ) dqmStore_->removeElement( meTestPulsePNG01_->getName() );
  meTestPulsePNG01_ = 0;

  if ( meTestPulsePNG16_ ) dqmStore_->removeElement( meTestPulsePNG16_->getName() );
  meTestPulsePNG16_ = 0;

  if ( meTestPulseAmplG01_ ) dqmStore_->removeElement( meTestPulseAmplG01_->getName() );
  meTestPulseAmplG01_ = 0;

  if ( meTestPulseAmplG06_ ) dqmStore_->removeElement( meTestPulseAmplG06_->getName() );
  meTestPulseAmplG06_ = 0;

  if ( meTestPulseAmplG12_ ) dqmStore_->removeElement( meTestPulseAmplG12_->getName() );
  meTestPulseAmplG12_ = 0;

  if ( meRecHitEnergy_[0] ) dqmStore_->removeElement( meRecHitEnergy_[0]->getName() );
  meRecHitEnergy_[0] = 0;

  if ( meRecHitEnergy_[1] ) dqmStore_->removeElement( meRecHitEnergy_[1]->getName() );
  meRecHitEnergy_[1] = 0;

  if ( meTiming_[0] ) dqmStore_->removeElement( meTiming_[0]->getName() );
  meTiming_[0] = 0;

  if ( meTiming_[1] ) dqmStore_->removeElement( meTiming_[1]->getName() );
  meTiming_[1] = 0;

  if ( meTimingMean1D_[0] ) dqmStore_->removeElement( meTimingMean1D_[0]->getName() );
  meTimingMean1D_[0] = 0;

  if ( meTimingMean1D_[1] ) dqmStore_->removeElement( meTimingMean1D_[1]->getName() );
  meTimingMean1D_[1] = 0;

  if ( meTimingRMS1D_[0] ) dqmStore_->removeElement( meTimingRMS1D_[0]->getName() );
  meTimingRMS1D_[0] = 0;

  if ( meTimingRMS1D_[1] ) dqmStore_->removeElement( meTimingRMS1D_[1]->getName() );
  meTimingRMS1D_[1] = 0;

  if ( meTriggerTowerEt_[0] ) dqmStore_->removeElement( meTriggerTowerEt_[0]->getName() );
  meTriggerTowerEt_[0] = 0;

  if ( meTriggerTowerEt_[1] ) dqmStore_->removeElement( meTriggerTowerEt_[1]->getName() );
  meTriggerTowerEt_[1] = 0;

  if ( meTriggerTowerEmulError_[0] ) dqmStore_->removeElement( meTriggerTowerEmulError_[0]->getName() );
  meTriggerTowerEmulError_[0] = 0;

  if ( meTriggerTowerEmulError_[1] ) dqmStore_->removeElement( meTriggerTowerEmulError_[1]->getName() );
  meTriggerTowerEmulError_[1] = 0;

  if ( meTriggerTowerTiming_[0] ) dqmStore_->removeElement( meTriggerTowerTiming_[0]->getName() );
  meTriggerTowerTiming_[0] = 0;

  if ( meTriggerTowerTiming_[1] ) dqmStore_->removeElement( meTriggerTowerTiming_[1]->getName() );
  meTriggerTowerTiming_[1] = 0;

  if ( meTriggerTowerNonSingleTiming_[0] ) dqmStore_->removeElement( meTriggerTowerNonSingleTiming_[0]->getName() );
  meTriggerTowerNonSingleTiming_[0] = 0;

  if ( meTriggerTowerNonSingleTiming_[1] ) dqmStore_->removeElement( meTriggerTowerNonSingleTiming_[1]->getName() );
  meTriggerTowerNonSingleTiming_[1] = 0;

  if ( meGlobalSummary_[0] ) dqmStore_->removeElement( meGlobalSummary_[0]->getName() );
  meGlobalSummary_[0] = 0;

  if ( meGlobalSummary_[1] ) dqmStore_->removeElement( meGlobalSummary_[1]->getName() );
  meGlobalSummary_[1] = 0;

  if(meSummaryErr_) dqmStore_->removeElement(meSummaryErr_->getName());
  meSummaryErr_ = 0;

}

#ifdef WITH_ECAL_COND_DB
bool EESummaryClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov, bool& status) {

  status = true;

  return true;

}
#endif

void EESummaryClient::analyze(void) {

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( debug_ ) std::cout << "EESummaryClient: ievt/jevt = " << ievt_ << "/" << jevt_ << std::endl;
  }

  uint32_t chWarnBit = 1 << EcalDQMStatusHelper::PHYSICS_BAD_CHANNEL_WARNING;

  for ( int ix = 1; ix <= 100; ix++ ) {
    for ( int iy = 1; iy <= 100; iy++ ) {

      if ( meIntegrity_[0] ) meIntegrity_[0]->setBinContent( ix, iy, 6. );
      if ( meIntegrity_[1] ) meIntegrity_[1]->setBinContent( ix, iy, 6. );
      if ( meOccupancy_[0] ) meOccupancy_[0]->setBinContent( ix, iy, 0. );
      if ( meOccupancy_[1] ) meOccupancy_[1]->setBinContent( ix, iy, 0. );
      if ( meStatusFlags_[0] ) meStatusFlags_[0]->setBinContent( ix, iy, 6. );
      if ( meStatusFlags_[1] ) meStatusFlags_[1]->setBinContent( ix, iy, 6. );
      if ( mePedestalOnline_[0] ) mePedestalOnline_[0]->setBinContent( ix, iy, 6. );
      if ( mePedestalOnline_[1] ) mePedestalOnline_[1]->setBinContent( ix, iy, 6. );
      if ( mePedestalOnlineRMSMap_[0] ) mePedestalOnlineRMSMap_[0]->setBinContent( ix, iy, -1. );
      if ( mePedestalOnlineRMSMap_[1] ) mePedestalOnlineRMSMap_[1]->setBinContent( ix, iy, -1. );

      if ( meLaserL1_[0] ) meLaserL1_[0]->setBinContent( ix, iy, 6. );
      if ( meLaserL1_[1] ) meLaserL1_[1]->setBinContent( ix, iy, 6. );
      if ( meLaserL2_[0] ) meLaserL2_[0]->setBinContent( ix, iy, 6. );
      if ( meLaserL2_[1] ) meLaserL2_[1]->setBinContent( ix, iy, 6. );
      if ( meLaserL3_[0] ) meLaserL3_[0]->setBinContent( ix, iy, 6. );
      if ( meLaserL3_[1] ) meLaserL3_[1]->setBinContent( ix, iy, 6. );
      if ( meLaserL4_[0] ) meLaserL4_[0]->setBinContent( ix, iy, 6. );
      if ( meLaserL4_[1] ) meLaserL4_[1]->setBinContent( ix, iy, 6. );
      if ( meLedL1_[0] ) meLedL1_[0]->setBinContent( ix, iy, 6. );
      if ( meLedL1_[1] ) meLedL1_[1]->setBinContent( ix, iy, 6. );
      if ( meLedL2_[0] ) meLedL2_[0]->setBinContent( ix, iy, 6. );
      if ( meLedL2_[1] ) meLedL2_[1]->setBinContent( ix, iy, 6. );
      if ( mePedestalG01_[0] ) mePedestalG01_[0]->setBinContent( ix, iy, 6. );
      if ( mePedestalG01_[1] ) mePedestalG01_[1]->setBinContent( ix, iy, 6. );
      if ( mePedestalG06_[0] ) mePedestalG06_[0]->setBinContent( ix, iy, 6. );
      if ( mePedestalG06_[1] ) mePedestalG06_[1]->setBinContent( ix, iy, 6. );
      if ( mePedestalG12_[0] ) mePedestalG12_[0]->setBinContent( ix, iy, 6. );
      if ( mePedestalG12_[1] ) mePedestalG12_[1]->setBinContent( ix, iy, 6. );
      if ( meTestPulseG01_[0] ) meTestPulseG01_[0]->setBinContent( ix, iy, 6. );
      if ( meTestPulseG01_[1] ) meTestPulseG01_[1]->setBinContent( ix, iy, 6. );
      if ( meTestPulseG06_[0] ) meTestPulseG06_[0]->setBinContent( ix, iy, 6. );
      if ( meTestPulseG06_[1] ) meTestPulseG06_[1]->setBinContent( ix, iy, 6. );
      if ( meTestPulseG12_[0] ) meTestPulseG12_[0]->setBinContent( ix, iy, 6. );
      if ( meTestPulseG12_[1] ) meTestPulseG12_[1]->setBinContent( ix, iy, 6. );
      if ( meRecHitEnergy_[0] ) meRecHitEnergy_[0]->setBinContent( ix, iy, 0. );
      if ( meRecHitEnergy_[1] ) meRecHitEnergy_[1]->setBinContent( ix, iy, 0. );

      if( meGlobalSummary_[0] ) meGlobalSummary_[0]->setBinContent( ix, iy, 6. );
      if( meGlobalSummary_[1] ) meGlobalSummary_[1]->setBinContent( ix, iy, 6. );

    }
  }

  // default is 6 because we want white for the non existing MEM
  for ( int ix = 1; ix <= 45; ix++ ) {
    for ( int iy = 1; iy <= 20; iy++ ) {

      if ( meIntegrityPN_ ) meIntegrityPN_->setBinContent( ix, iy, 6. );
      if ( meOccupancyPN_ ) meOccupancyPN_->setBinContent( ix, iy, 0. );
      if ( meLaserL1PN_ ) meLaserL1PN_->setBinContent( ix, iy, 6. );
      if ( meLaserL2PN_ ) meLaserL2PN_->setBinContent( ix, iy, 6. );
      if ( meLaserL3PN_ ) meLaserL3PN_->setBinContent( ix, iy, 6. );
      if ( meLaserL4PN_ ) meLaserL4PN_->setBinContent( ix, iy, 6. );
      if ( meLedL1PN_ ) meLedL1PN_->setBinContent( ix, iy, 6. );
      if ( meLedL2PN_ ) meLedL2PN_->setBinContent( ix, iy, 6. );
      if ( mePedestalPNG01_ ) mePedestalPNG01_->setBinContent( ix, iy, 6. );
      if ( mePedestalPNG16_ ) mePedestalPNG16_->setBinContent( ix, iy, 6. );
      if ( meTestPulsePNG01_ ) meTestPulsePNG01_->setBinContent( ix, iy, 6. );
      if ( meTestPulsePNG16_ ) meTestPulsePNG16_->setBinContent( ix, iy, 6. );

    }
  }

  for ( int ix = 1; ix <= 100; ix++ ) {
    for ( int iy = 1; iy <= 100; iy++ ) {
      if ( meTriggerTowerEt_[0] ) meTriggerTowerEt_[0]->setBinContent( ix, iy, 0. );
      if ( meTriggerTowerEt_[1] ) meTriggerTowerEt_[1]->setBinContent( ix, iy, 0. );
      if ( meTriggerTowerEmulError_[0] ) meTriggerTowerEmulError_[0]->setBinContent( ix, iy, 6. );
      if ( meTriggerTowerEmulError_[1] ) meTriggerTowerEmulError_[1]->setBinContent( ix, iy, 6. );
      if ( meTriggerTowerTiming_[0] ) meTriggerTowerTiming_[0]->setBinContent( ix, iy, 0. );
      if ( meTriggerTowerTiming_[1] ) meTriggerTowerTiming_[1]->setBinContent( ix, iy, 0. );
      if ( meTriggerTowerNonSingleTiming_[0] ) meTriggerTowerNonSingleTiming_[0]->setBinContent( ix, iy, -1 );
      if ( meTriggerTowerNonSingleTiming_[1] ) meTriggerTowerNonSingleTiming_[1]->setBinContent( ix, iy, -1 );
    }
  }

  for ( int ix = 1; ix <= 20; ix++ ) {
    for ( int iy = 1; iy <= 20; iy++ ) {
      if ( meTiming_[0] ) meTiming_[0]->setBinContent( ix, iy, 6. );
      if ( meTiming_[1] ) meTiming_[1]->setBinContent( ix, iy, 6. );
    }
  }

  if ( meIntegrity_[0] ) meIntegrity_[0]->setEntries( 0 );
  if ( meIntegrity_[1] ) meIntegrity_[1]->setEntries( 0 );
  if ( meIntegrityErr_ ) meIntegrityErr_->Reset();
  if ( meIntegrityPN_ ) meIntegrityPN_->setEntries( 0 );
  if ( meOccupancy_[0] ) meOccupancy_[0]->setEntries( 0 );
  if ( meOccupancy_[1] ) meOccupancy_[1]->setEntries( 0 );
  if ( meOccupancy1D_ ) meOccupancy1D_->Reset();
  if ( meOccupancyPN_ ) meOccupancyPN_->setEntries( 0 );
  if ( meStatusFlags_[0] ) meStatusFlags_[0]->setEntries( 0 );
  if ( meStatusFlags_[1] ) meStatusFlags_[1]->setEntries( 0 );
  if ( meStatusFlagsErr_ ) meStatusFlagsErr_->Reset();
  if ( mePedestalOnline_[0] ) mePedestalOnline_[0]->setEntries( 0 );
  if ( mePedestalOnline_[1] ) mePedestalOnline_[1]->setEntries( 0 );
  if ( mePedestalOnlineErr_ ) mePedestalOnlineErr_->Reset();
  if ( mePedestalOnlineMean_ ) mePedestalOnlineMean_->Reset();
  if ( mePedestalOnlineRMS_ ) mePedestalOnlineRMS_->Reset();
  if ( meLaserL1_[0] ) meLaserL1_[0]->setEntries( 0 );
  if ( meLaserL1_[1] ) meLaserL1_[1]->setEntries( 0 );
  if ( meLaserL1Err_ ) meLaserL1Err_->Reset();
  if ( meLaserL1PN_ ) meLaserL1PN_->setEntries( 0 );
  if ( meLaserL1PNErr_ ) meLaserL1PNErr_->Reset();
  if ( meLaserL1Ampl_ ) meLaserL1Ampl_->Reset();
  if ( meLaserL1Timing_ ) meLaserL1Timing_->Reset();
  if ( meLaserL1AmplOverPN_ ) meLaserL1AmplOverPN_->Reset();
  if ( meLaserL2_[0] ) meLaserL2_[0]->setEntries( 0 );
  if ( meLaserL2_[1] ) meLaserL2_[1]->setEntries( 0 );
  if ( meLaserL2Err_ ) meLaserL2Err_->Reset();
  if ( meLaserL2PN_ ) meLaserL2PN_->setEntries( 0 );
  if ( meLaserL2PNErr_ ) meLaserL2PNErr_->Reset();
  if ( meLaserL2Ampl_ ) meLaserL2Ampl_->Reset();
  if ( meLaserL2Timing_ ) meLaserL2Timing_->Reset();
  if ( meLaserL2AmplOverPN_ ) meLaserL2AmplOverPN_->Reset();
  if ( meLaserL3_[0] ) meLaserL3_[0]->setEntries( 0 );
  if ( meLaserL3_[1] ) meLaserL3_[1]->setEntries( 0 );
  if ( meLaserL3Err_ ) meLaserL3Err_->Reset();
  if ( meLaserL3PN_ ) meLaserL3PN_->setEntries( 0 );
  if ( meLaserL3PNErr_ ) meLaserL3PNErr_->Reset();
  if ( meLaserL3Ampl_ ) meLaserL3Ampl_->Reset();
  if ( meLaserL3Timing_ ) meLaserL3Timing_->Reset();
  if ( meLaserL3AmplOverPN_ ) meLaserL3AmplOverPN_->Reset();
  if ( meLaserL4_[0] ) meLaserL4_[0]->setEntries( 0 );
  if ( meLaserL4_[1] ) meLaserL4_[1]->setEntries( 0 );
  if ( meLaserL4Err_ ) meLaserL4Err_->Reset();
  if ( meLaserL4PN_ ) meLaserL4PN_->setEntries( 0 );
  if ( meLaserL4PNErr_ ) meLaserL4PNErr_->Reset();
  if ( meLaserL4Ampl_ ) meLaserL4Ampl_->Reset();
  if ( meLaserL4Timing_ ) meLaserL4Timing_->Reset();
  if ( meLaserL4AmplOverPN_ ) meLaserL4AmplOverPN_->Reset();
  if ( meLedL1_[0] ) meLedL1_[0]->setEntries( 0 );
  if ( meLedL1_[1] ) meLedL1_[1]->setEntries( 0 );
  if ( meLedL1Err_ ) meLedL1Err_->Reset();
  if ( meLedL1PN_ ) meLedL1PN_->setEntries( 0 );
  if ( meLedL1PNErr_ ) meLedL1PNErr_->Reset();
  if ( meLedL1Ampl_ ) meLedL1Ampl_->Reset();
  if ( meLedL1Timing_ ) meLedL1Timing_->Reset();
  if ( meLedL1AmplOverPN_ ) meLedL1AmplOverPN_->Reset();
  if ( meLedL2_[0] ) meLedL2_[0]->setEntries( 0 );
  if ( meLedL2_[1] ) meLedL2_[1]->setEntries( 0 );
  if ( meLedL2Err_ ) meLedL2Err_->Reset();
  if ( meLedL2PN_ ) meLedL2PN_->setEntries( 0 );
  if ( meLedL2PNErr_ ) meLedL2PNErr_->Reset();
  if ( meLedL2Ampl_ ) meLedL2Ampl_->Reset();
  if ( meLedL2Timing_ ) meLedL2Timing_->Reset();
  if ( meLedL2AmplOverPN_ ) meLedL2AmplOverPN_->Reset();
  if ( mePedestalG01_[0] ) mePedestalG01_[0]->setEntries( 0 );
  if ( mePedestalG01_[1] ) mePedestalG01_[1]->setEntries( 0 );
  if ( mePedestalG06_[0] ) mePedestalG06_[0]->setEntries( 0 );
  if ( mePedestalG06_[1] ) mePedestalG06_[1]->setEntries( 0 );
  if ( mePedestalG12_[0] ) mePedestalG12_[0]->setEntries( 0 );
  if ( mePedestalG12_[1] ) mePedestalG12_[1]->setEntries( 0 );
  if ( mePedestalPNG01_ ) mePedestalPNG01_->setEntries( 0 );
  if ( mePedestalPNG16_ ) mePedestalPNG16_->setEntries( 0 );
  if ( meTestPulseG01_[0] ) meTestPulseG01_[0]->setEntries( 0 );
  if ( meTestPulseG01_[1] ) meTestPulseG01_[1]->setEntries( 0 );
  if ( meTestPulseG06_[0] ) meTestPulseG06_[0]->setEntries( 0 );
  if ( meTestPulseG06_[1] ) meTestPulseG06_[1]->setEntries( 0 );
  if ( meTestPulseG12_[0] ) meTestPulseG12_[0]->setEntries( 0 );
  if ( meTestPulseG12_[1] ) meTestPulseG12_[1]->setEntries( 0 );
  if ( meTestPulsePNG01_ ) meTestPulsePNG01_->setEntries( 0 );
  if ( meTestPulsePNG16_ ) meTestPulsePNG16_->setEntries( 0 );
  if ( meTestPulseAmplG01_ ) meTestPulseAmplG01_->Reset();
  if ( meTestPulseAmplG06_ ) meTestPulseAmplG06_->Reset();
  if ( meTestPulseAmplG12_ ) meTestPulseAmplG12_->Reset();

  if ( meRecHitEnergy_[0] ) meRecHitEnergy_[0]->setEntries( 0 );
  if ( meRecHitEnergy_[1] ) meRecHitEnergy_[1]->setEntries( 0 );
  if ( meTiming_[0] ) meTiming_[0]->setEntries( 0 );
  if ( meTiming_[1] ) meTiming_[1]->setEntries( 0 );
  if ( meTimingMean1D_[0] ) meTimingMean1D_[0]->Reset();
  if ( meTimingMean1D_[1] ) meTimingMean1D_[1]->Reset();
  if ( meTimingRMS1D_[0] ) meTimingRMS1D_[0]->Reset();
  if ( meTimingRMS1D_[1] ) meTimingRMS1D_[1]->Reset();
  if ( meTimingMean_ ) meTimingMean_->Reset();
  if ( meTimingRMS_ ) meTimingRMS_->Reset();
  if ( meTriggerTowerEt_[0] ) meTriggerTowerEt_[0]->setEntries( 0 );
  if ( meTriggerTowerEt_[1] ) meTriggerTowerEt_[1]->setEntries( 0 );
  if ( meTriggerTowerEmulError_[0] ) meTriggerTowerEmulError_[0]->setEntries( 0 );
  if ( meTriggerTowerEmulError_[1] ) meTriggerTowerEmulError_[1]->setEntries( 0 );
  if ( meTriggerTowerTiming_[0] ) meTriggerTowerTiming_[0]->setEntries( 0 );
  if ( meTriggerTowerTiming_[1] ) meTriggerTowerTiming_[1]->setEntries( 0 );
  if ( meTriggerTowerNonSingleTiming_[0] ) meTriggerTowerNonSingleTiming_[0]->setEntries( 0 );
  if ( meTriggerTowerNonSingleTiming_[1] ) meTriggerTowerNonSingleTiming_[1]->setEntries( 0 );

  if( meGlobalSummary_[0] ) meGlobalSummary_[0]->setEntries( 0 );
  if( meGlobalSummary_[1] ) meGlobalSummary_[1]->setEntries(0);

  if(meSummaryErr_) meSummaryErr_->Reset();

  MonitorElement *me(0);
  me = dqmStore_->get(prefixME_ + "/EETimingTask/EETMT timing map EE +");
  TProfile2D *htmtp(0);
  htmtp = UtilsClient::getHisto(me, false, htmtp);
 
  me = dqmStore_->get(prefixME_ + "/EETimingTask/EETMT timing map EE -");
  TProfile2D *htmtm(0);
  htmtm = UtilsClient::getHisto(me, false, htmtm);

  std::string subdir(subfolder_ == "" ? "" : subfolder_ + "/");

  TH1F* oosTrend(0);

  for ( unsigned int i=0; i<clients_.size(); i++ ) {

    EEIntegrityClient* eeic = dynamic_cast<EEIntegrityClient*>(clients_[i]);
    EEStatusFlagsClient* eesfc = dynamic_cast<EEStatusFlagsClient*>(clients_[i]);
    if(!produceReports_) eesfc = 0;
    EEPedestalOnlineClient* eepoc = dynamic_cast<EEPedestalOnlineClient*>(clients_[i]);
    if(!produceReports_) eepoc = 0;

    EELaserClient* eelc = dynamic_cast<EELaserClient*>(clients_[i]);
    EELedClient* eeldc = dynamic_cast<EELedClient*>(clients_[i]);
    EEPedestalClient* eepc = dynamic_cast<EEPedestalClient*>(clients_[i]);
    EETestPulseClient* eetpc = dynamic_cast<EETestPulseClient*>(clients_[i]);

    EETimingClient* eetmc = dynamic_cast<EETimingClient*>(clients_[i]);
    EETriggerTowerClient* eetttc = dynamic_cast<EETriggerTowerClient*>(clients_[i]);

    MonitorElement *me_01, *me_02, *me_03;
    MonitorElement *me_04, *me_05;
    //    MonitorElement *me_f[6], *me_fg[2];
    TH2F* h2;
    TH2F* h3;

    me = dqmStore_->get( prefixME_ + "/EcalInfo/EEMM DCC" );
    norm01_ = UtilsClient::getHisto( me, cloneME_, norm01_ );

    me = dqmStore_->get( prefixME_ + "/EERawDataTask/" + subdir + "EERDT L1A FE errors" );
    synch01_ = UtilsClient::getHisto( me, cloneME_, synch01_ );

    me = dqmStore_->get(prefixME_ + "/EERawDataTask/" + subdir + "EERDT accumulated FE synchronization errors");
    oosTrend = UtilsClient::getHisto(me, cloneME_, oosTrend);

    for ( unsigned int i=0; i<superModules_.size(); i++ ) {

      int ism = superModules_[i];

      me = dqmStore_->get( prefixME_ + "/EEOccupancyTask/" + subdir + "EEOT rec hit energy " + Numbers::sEE(ism) );
      hot01_[ism-1] = UtilsClient::getHisto( me, cloneME_, hot01_[ism-1] );

      me = dqmStore_->get( prefixME_ + "/EEPedestalOnlineTask/" + subdir + "Gain12/EEPOT pedestal " + Numbers::sEE(ism) + " G12" );
      hpot01_[ism-1] = UtilsClient::getHisto( me, cloneME_, hpot01_[ism-1] );

      me = dqmStore_->get( prefixME_ + "/EETriggerTowerTask/EETTT Et map Real Digis " + Numbers::sEE(ism) );
      httt01_[ism-1] = UtilsClient::getHisto( me, cloneME_, httt01_[ism-1] );

      me = dqmStore_->get( prefixME_ + "/EETimingTask/EETMT timing " + Numbers::sEE(ism) );
      htmt01_[ism-1] = UtilsClient::getHisto( me, cloneME_, htmt01_[ism-1] );

      for ( int ix = 1; ix <= 50; ix++ ) {
        for ( int iy = 1; iy <= 50; iy++ ) {

          int jx = ix + Numbers::ix0EE(ism);
          int jy = iy + Numbers::iy0EE(ism);

          if ( ism >= 1 && ism <= 9 ) {
            if ( ! Numbers::validEE(ism, 101 - jx, jy) ) continue;
          } else {
            if ( ! Numbers::validEE(ism, jx, jy) ) continue;
          }

          if ( eeic ) {

            me = eeic->meg01_[ism-1];

            if ( me ) {

              float xval = me->getBinContent( ix, iy );

              if ( ism >= 1 && ism <= 9 ) {
                if(meIntegrity_[0]) meIntegrity_[0]->setBinContent( 101 - jx, jy, xval );
              } else {
                if(meIntegrity_[1]) meIntegrity_[1]->setBinContent( jx, jy, xval );
              }

              if ( xval == 0 && meIntegrityErr_) meIntegrityErr_->Fill( ism );

            }

            h2 = eeic->h_[ism-1];

            if ( h2 ) {

              float xval = h2->GetBinContent( ix, iy );

              if ( ism >= 1 && ism <= 9 ) {
                if ( xval != 0 && meOccupancy_[0]) meOccupancy_[0]->setBinContent( 101 - jx, jy, xval );
              } else {
                if ( xval != 0 && meOccupancy_[1]) meOccupancy_[1]->setBinContent( jx, jy, xval );
              }

              if(meOccupancy1D_) meOccupancy1D_->Fill( ism, xval );

            }

          }

          if ( eepoc ) {

            me = eepoc->meg03_[ism-1];

            if ( me ) {

              float xval = me->getBinContent( ix, iy );

              if ( ism >= 1 && ism <= 9 ) {
                if(mePedestalOnline_[0]) mePedestalOnline_[0]->setBinContent( 101 - jx, jy, xval );
              } else {
                if(mePedestalOnline_[1]) mePedestalOnline_[1]->setBinContent( jx, jy, xval );
              }

              if ( xval == 0 && mePedestalOnlineErr_) mePedestalOnlineErr_->Fill( ism );

            }

            float num01, mean01, rms01;
            bool update01 = UtilsClient::getBinStatistics(hpot01_[ism-1], ix, iy, num01, mean01, rms01);

            if ( update01 ) {

              if(mePedestalOnlineRMS_) mePedestalOnlineRMS_->Fill( ism, rms01 );
              if(mePedestalOnlineMean_) mePedestalOnlineMean_->Fill( ism, mean01 );

              if ( ism >= 1 && ism <= 9 ) {
                if(mePedestalOnlineRMSMap_[0]) mePedestalOnlineRMSMap_[0]->setBinContent( 101 - jx, jy, rms01 );
              } else {
                if(mePedestalOnlineRMSMap_[1]) mePedestalOnlineRMSMap_[1]->setBinContent( jx, jy, rms01 );
              }

            }

          }

          if ( eelc ) {

            if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 1) != laserWavelengths_.end() ) {

              me = eelc->meg01_[ism-1];

              if ( me ) {

                float xval = me->getBinContent( ix, iy );

                if ( me->getEntries() != 0 ) {
                  if ( ism >= 1 && ism <= 9 ) {
                    meLaserL1_[0]->setBinContent( 101 - jx, jy, xval );
                  } else {
                    meLaserL1_[1]->setBinContent( jx, jy, xval );
                  }

                  if ( xval == 0 ) meLaserL1Err_->Fill( ism );
                }

              }

            }

            if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 2) != laserWavelengths_.end() ) {

              me = eelc->meg02_[ism-1];

              if ( me ) {

                float xval = me->getBinContent( ix, iy );

                if ( me->getEntries() != 0 ) {
                  if ( ism >= 1 && ism <= 9 ) {
                    meLaserL2_[0]->setBinContent( 101 - jx, jy, xval );
                  } else {
                    meLaserL2_[1]->setBinContent( jx, jy, xval );
                  }

                  if ( xval == 0 ) meLaserL2Err_->Fill( ism );
                }

              }

            }

            if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 3) != laserWavelengths_.end() ) {

              me = eelc->meg03_[ism-1];

              if ( me ) {

                float xval = me->getBinContent( ix, iy );

                if ( me->getEntries() != 0 ) {
                  if ( ism >= 1 && ism <= 9 ) {
                    meLaserL3_[0]->setBinContent( 101 - jx, jy, xval );
                  } else {
                    meLaserL3_[1]->setBinContent( jx, jy, xval );
                  }

                  if ( xval == 0 ) meLaserL3Err_->Fill( ism );
                }

              }

            }

            if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 4) != laserWavelengths_.end() ) {

              me = eelc->meg04_[ism-1];

              if ( me ) {

                float xval = me->getBinContent( ix, iy );

                if ( me->getEntries() != 0 ) {
                  if ( ism >= 1 && ism <= 9 ) {
                    meLaserL4_[0]->setBinContent( 101 - jx, jy, xval );
                  } else {
                    meLaserL4_[1]->setBinContent( jx, jy, xval );
                  }

                  if ( xval == 0 ) meLaserL4Err_->Fill( ism );
                }

              }

            }

          }

          if ( eeldc ) {

            if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 1) != ledWavelengths_.end() ) {

              me = eeldc->meg01_[ism-1];

              if ( me ) {

                float xval = me->getBinContent( ix, iy );

                if ( me->getEntries() != 0 ) {
                  if ( ism >= 1 && ism <= 9 ) {
                    meLedL1_[0]->setBinContent( 101 - jx, jy, xval );
                  } else {
                    meLedL1_[1]->setBinContent( jx, jy, xval );
                  }

                  if ( xval == 0 ) meLedL1Err_->Fill( ism );
                }

              }

            }

            if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 2) != ledWavelengths_.end() ) {

              me = eeldc->meg02_[ism-1];

              if ( me ) {

                float xval = me->getBinContent( ix, iy );

                if ( me->getEntries() != 0 ) {
                  if ( ism >= 1 && ism <= 9 ) {
                    meLedL2_[0]->setBinContent( 101 - jx, jy, xval );
                  } else {
                    meLedL2_[1]->setBinContent( jx, jy, xval );
                  }

                  if ( xval == 0 ) meLedL2Err_->Fill( ism );
                }

              }

            }

          }

          if ( eepc ) {

            me_01 = eepc->meg01_[ism-1];
            me_02 = eepc->meg02_[ism-1];
            me_03 = eepc->meg03_[ism-1];

            if ( me_01 ) {
              float val_01=me_01->getBinContent(ix,iy);
              if ( me_01->getEntries() != 0 ) {
                if ( ism >= 1 && ism <= 9 ) {
                  mePedestalG01_[0]->setBinContent( 101 - jx, jy, val_01 );
                } else {
                  mePedestalG01_[1]->setBinContent( jx, jy, val_01 );
                }
              }
            }
            if ( me_02 ) {
              float val_02=me_02->getBinContent(ix,iy);
              if ( me_02->getEntries() != 0 ) {
                if ( ism >= 1 && ism <= 9 ) {
                  mePedestalG06_[0]->setBinContent( 101 - jx, jy, val_02 );
                } else {
                  mePedestalG06_[1]->setBinContent( jx, jy, val_02 );
                }
              }
            }
            if ( me_03 ) {
              float val_03=me_03->getBinContent(ix,iy);
              if ( me_03->getEntries() != 0 ) {
                if ( ism >= 1 && ism <= 9 ) {
                  mePedestalG12_[0]->setBinContent( 101 - jx, jy, val_03 );
                } else {
                  mePedestalG12_[1]->setBinContent( jx, jy, val_03 );
                }
              }
            }

          }

          if ( eetpc ) {

            me_01 = eetpc->meg01_[ism-1];
            me_02 = eetpc->meg02_[ism-1];
            me_03 = eetpc->meg03_[ism-1];

            if ( me_01 ) {
              float val_01=me_01->getBinContent(ix,iy);
              if ( me_01->getEntries() != 0 ) {
                if ( ism >= 1 && ism <= 9 ) {
                  meTestPulseG01_[0]->setBinContent( 101 - jx, jy, val_01 );
                } else {
                  meTestPulseG01_[1]->setBinContent( jx, jy, val_01 );
                }
              }
            }
            if ( me_02 ) {
              float val_02=me_02->getBinContent(ix,iy);
              if ( me_02->getEntries() != 0 ) {
                if ( ism >= 1 && ism <= 9 ) {
                  meTestPulseG06_[0]->setBinContent( 101 - jx, jy, val_02 );
                } else {
                  meTestPulseG06_[1]->setBinContent( jx, jy, val_02 );
                }
              }
            }
            if ( me_03 ) {
              float val_03=me_03->getBinContent(ix,iy);
              if ( me_03->getEntries() != 0 ) {
                if ( ism >= 1 && ism <= 9 ) {
                  meTestPulseG12_[0]->setBinContent( 101 - jx, jy, val_03 );
                } else {
                  meTestPulseG12_[1]->setBinContent( jx, jy, val_03 );
                }
              }
            }

          }

          if ( hot01_[ism-1] ) {

            float xval = hot01_[ism-1]->GetBinContent( ix, iy );

            if ( ism >= 1 && ism <= 9 ) {
              if(meRecHitEnergy_[0]) meRecHitEnergy_[0]->setBinContent( 101 - jx, jy, xval );
            } else {
              if(meRecHitEnergy_[1]) meRecHitEnergy_[1]->setBinContent( jx, jy, xval );
            }

          }

        }
      }

      for ( int ix = 1; ix <= 50; ix++ ) {
        for ( int iy = 1; iy <= 50; iy++ ) {

          int jx = ix + Numbers::ix0EE(ism);
          int jy = iy + Numbers::iy0EE(ism);

          if ( ism >= 1 && ism <= 9 ) {
            if ( ! Numbers::validEE(ism, 101 - jx, jy) ) continue;
          } else {
            if ( ! Numbers::validEE(ism, jx, jy) ) continue;
          }

          if ( eesfc ) {

            me = dqmStore_->get(prefixME_ + "/EcalInfo/EEMM DCC");

            float xval = 6;

            if ( me ) {

              xval = 2;
              if ( me->getBinContent( ism ) > 0 ) xval = 1;

            }

            me = eesfc->meh01_[ism-1];

            if ( me ) {

              if ( me->getBinContent( ix, iy ) > 0 ) xval = 0;

              if ( ism >= 1 && ism <= 9 ) {

                meStatusFlags_[0]->setBinContent( 101 - jx, jy, xval );

                if ( me->getBinError( ix, iy ) > 0 && me->getBinError( ix, iy ) < 0.1 ) {
                  UtilsClient::maskBinContent( meStatusFlags_[0], 101 - jx, jy );
                }
              } else {

                meStatusFlags_[1]->setBinContent( jx, jy, xval );

                if ( me->getBinError( ix, iy ) > 0 && me->getBinError( ix, iy ) < 0.1 ) {
                  UtilsClient::maskBinContent( meStatusFlags_[1], jx, jy );
                }
              }

              if ( xval == 0 ) meStatusFlagsErr_->Fill( ism );

            }

          }

        }
      }

      for ( int ix = 1; ix <= 50; ix++ ) {
        for ( int iy = 1; iy <= 50; iy++ ) {

          int jx = ix + Numbers::ix0EE(ism);
          int jy = iy + Numbers::iy0EE(ism);

          if ( eetttc ) {

            float mean01 = 0;
            bool hadNonZeroInterest = false;

            if ( httt01_[ism-1] ) {

              mean01 = httt01_[ism-1]->GetBinContent( ix, iy );

              if ( mean01 != 0. ) {
                if ( ism >= 1 && ism <= 9 ) {
                  if ( meTriggerTowerEt_[0] ) meTriggerTowerEt_[0]->setBinContent( 101 - jx, jy, mean01 );
                } else {
                  if ( meTriggerTowerEt_[1] ) meTriggerTowerEt_[1]->setBinContent( jx, jy, mean01 );
                }
              }

            }

            me = eetttc->me_o01_[ism-1];

            if ( me ) {

              float xval = me->getBinContent( ix, iy );

              if ( xval != 0. ) {
                if ( ism >= 1 && ism <= 9 ) {
                  meTriggerTowerTiming_[0]->setBinContent( 101 - jx, jy, xval );
                } else {
                  meTriggerTowerTiming_[1]->setBinContent( jx, jy, xval );
                }
                hadNonZeroInterest = true;
              }

            }

            me = eetttc->me_o02_[ism-1];

            if ( me ) {

              float xval = me->getBinContent( ix, iy );

              if ( xval != 0. ) {
                if ( ism >= 1 && ism <= 9 ) {
                  meTriggerTowerNonSingleTiming_[0]->setBinContent( 101 - jx, jy, xval );
                } else {
                  meTriggerTowerNonSingleTiming_[1]->setBinContent( jx, jy, xval );
                }
              }

            }

            float xval = 2;
            if( mean01 > 0. ) {

              h2 = eetttc->l01_[ism-1];
              h3 = eetttc->l02_[ism-1];

              if ( h2 && h3 ) {

                // float emulErrorVal = h2->GetBinContent( ix, iy ) + h3->GetBinContent( ix, iy );
                float emulErrorVal = h2->GetBinContent( ix, iy );

                float errorThresh(0.01);
                if((ix - 50) * (ix - 50) + (iy - 50) * (iy - 50) < 400.) errorThresh = 0.05;

                if( emulErrorVal > errorThresh * ievt_ && hadNonZeroInterest ) xval = 0;

              }

              if ( xval!=0 && hadNonZeroInterest ) xval = 1;

            }

            // see fix below
            if ( xval == 2 ) continue;

            if ( ism >= 1 && ism <= 9 ) {
              meTriggerTowerEmulError_[0]->setBinContent( 101 - jx, jy, xval );
            } else {
              meTriggerTowerEmulError_[1]->setBinContent( jx, jy, xval );
            }

          }

          if ( eetmc ) {

	    float num01, mean01, rms01;
	    bool update01 = UtilsClient::getBinStatistics(htmt01_[ism-1], ix, iy, num01, mean01, rms01, timingNHitThreshold_);
	    mean01 -= 50.;

	    if( update01 ){

	      if ( ism >= 1 && ism <= 9 ) {
		meTimingMean1D_[0]->Fill(mean01);
		meTimingRMS1D_[0]->Fill(rms01);
	      } else {
		meTimingMean1D_[1]->Fill(mean01);
		meTimingRMS1D_[1]->Fill(rms01);
	      }

	      meTimingMean_->Fill( ism, mean01 );

	      meTimingRMS_->Fill( ism, rms01 );

	    }

          }

        }
      }

      for ( int ix = 1; ix <= 10; ix++ ) {
	for( int iy = 1; iy <= 10; iy++ ) {

	  int jx = ix + Numbers::ix0EE(ism) / 5;
	  int jy = iy + Numbers::iy0EE(ism) / 5;

	  if( jx <= 0 || jx >= 21 || jy <= 0 || jy >= 21 ) continue;

	  if ( ism >= 1 && ism <= 9 ) {
	    if ( ! Numbers::validEESc(ism, 21 - jx, jy) ) continue;
	  } else {
	    if ( ! Numbers::validEESc(ism, jx, jy) ) continue;
	  }

          if ( eetmc ) {

            if ( htmt01_[ism-1] ) {

	      int ixedge = (ix-1) * 5;
	      int iyedge = (iy-1) * 5;
	      int jxedge = (jx-1) * 5;
	      int jyedge = (jy-1) * 5;

 	      float num(0);
 	      int nValid(0);
 	      bool mask(false);

	      for(int cx=1; cx<=5; cx++){
		for(int cy=1; cy<=5; cy++){
		  int scjx = (ism >= 1 && ism <= 9) ? 101 - (jxedge + cx) : jxedge + cx;
		  int scjy = jyedge + cy;
		  int scix = ixedge + cx;
		  int sciy = iyedge + cy;

		  if ( ! Numbers::validEE(ism, scjx, scjy) ) continue;

		  nValid += 1;

		  num += htmt01_[ism-1]->GetBinEntries(htmt01_[ism-1]->GetBin(scix, sciy));

		  if(Masks::maskChannel(ism, scix, sciy, chWarnBit, EcalEndcap) ) mask = true;
		}
	      }

 	      float nHitThreshold(timingNHitThreshold_ * 15. * nValid / 25.);

 	      bool update01(false);
 	      float num01, mean01, rms01;
 	      if(ism >= 1 && ism <= 9)
 		update01 = UtilsClient::getBinStatistics(htmtm, 21 - jx, jy, num01, mean01, rms01, nHitThreshold);
 	      else
 		update01 = UtilsClient::getBinStatistics(htmtp, jx, jy, num01, mean01, rms01, nHitThreshold);
 
 	      mean01 -= 50.;
 
 	      if(!update01){
 		mean01 = 0.;
 		rms01 = 0.;
 	      }

 	      update01 |= num > 1.4 * nHitThreshold; // allow 40% outliers

	      float xval = 2.;

 	      if( update01 ){

                float rmsThresh(6.);
                float meanThresh(3.);
                if((ix - 50) * (ix - 50) + (iy - 50) * (iy - 50) < 400.){
                  rmsThresh = 10.;
                  meanThresh = 6.;
                }

 		// quality BAD if mean large, rms large, or significantly more outliers (num: # events in +-20 ns time window)
 		if( std::abs(mean01) > meanThresh || rms01 > rmsThresh || num > 1.4 * num01 ) xval = 0.;
		else xval = 1.;

	      }

	      int ind;
	      if ( ism >= 1 && ism <= 9 ){
		jx = 21 - jx;
		ind = 0;
	      }else{
		ind = 1;
	      }

	      meTiming_[ind]->setBinContent( jx, jy, xval );
	      if ( mask ) UtilsClient::maskBinContent( meTiming_[ind], jx, jy );

	    }

          }

	}
      }
      // PN's summaries
      for( int i = 1; i <= 10; i++ ) {
	for( int j = 1; j <= 5; j++ ) {

	  int ichanx;
	  int ipseudostripx;

	  if(ism<=9) {
	    ichanx = i;
	    ipseudostripx = (ism<=3) ? j+5*(ism-1+6) : j+5*(ism-1-3);
	  } else {
	    ichanx = i+10;
	    ipseudostripx = (ism<=12) ? j+5*(ism-10+6) : j+5*(ism-10-3);
	  }

	  if ( eeic ) {

	    me_04 = eeic->meg02_[ism-1];
	    h2 = eeic->hmem_[ism-1];


	    if( me_04 ) {

	      float xval = me_04->getBinContent(i,j);
	      if(meIntegrityPN_) meIntegrityPN_->setBinContent( ipseudostripx, ichanx, xval );

	    }

	    if ( h2 ) {

	      float xval = h2->GetBinContent(i,1);
	      if(meOccupancyPN_) meOccupancyPN_->setBinContent( ipseudostripx, ichanx, xval );

	    }

	  }
	
          if ( eepc ) {

            me_04 = eepc->meg04_[ism-1];
            me_05 = eepc->meg05_[ism-1];

            if( me_04 ) {
              float val_04=me_04->getBinContent(i,1);
              mePedestalPNG01_->setBinContent( ipseudostripx, ichanx, val_04 );
            }
            if( me_05 ) {
              float val_05=me_05->getBinContent(i,1);
              mePedestalPNG16_->setBinContent( ipseudostripx, ichanx, val_05 );
            }

          }

          if ( eetpc ) {

            me_04 = eetpc->meg04_[ism-1];
            me_05 = eetpc->meg05_[ism-1];

            if( me_04 ) {
              float val_04=me_04->getBinContent(i,1);
              meTestPulsePNG01_->setBinContent( ipseudostripx, ichanx, val_04 );
            }
            if( me_05 ) {
              float val_05=me_05->getBinContent(i,1);
              meTestPulsePNG16_->setBinContent( ipseudostripx, ichanx, val_05 );
            }

          }

          if ( eelc ) {

            if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 1) != laserWavelengths_.end() ) {

              me = eelc->meg09_[ism-1];

              if( me ) {

                float xval = me->getBinContent(i,1);

                if ( me->getEntries() != 0 && me->getEntries() != 0 ) {
                  meLaserL1PN_->setBinContent( ipseudostripx, ichanx, xval );
                  if ( xval == 0 ) meLaserL1PNErr_->Fill( ism );
                }

              }

            }

            if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 2) != laserWavelengths_.end() ) {

              me = eelc->meg10_[ism-1];

              if( me ) {

                float xval = me->getBinContent(i,1);

                if ( me->getEntries() != 0 && me->getEntries() != 0 ) {
                  meLaserL2PN_->setBinContent( ipseudostripx, ichanx, xval );
                  if ( xval == 0 ) meLaserL2PNErr_->Fill( ism );
                }

              }

            }

            if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 3) != laserWavelengths_.end() ) {

              me = eelc->meg11_[ism-1];

              if( me ) {

                float xval = me->getBinContent(i,1);

                if ( me->getEntries() != 0 && me->getEntries() != 0 ) {
                  meLaserL3PN_->setBinContent( ipseudostripx, ichanx, xval );
                  if ( xval == 0 ) meLaserL3PNErr_->Fill( ism );
                }

              }

            }

            if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 4) != laserWavelengths_.end() ) {

              me = eelc->meg12_[ism-1];

              if( me ) {

                float xval = me->getBinContent(i,1);

                if ( me->getEntries() != 0 && me->getEntries() != 0 ) {
                  meLaserL4PN_->setBinContent( ipseudostripx, ichanx, xval );
                  if ( xval == 0 ) meLaserL4PNErr_->Fill( ism );
                }

              }

            }

          }

          if ( eeldc ) {

            if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 1) != ledWavelengths_.end() ) {

              me = eeldc->meg09_[ism-1];

              if( me ) {

                float xval = me->getBinContent(i,1);

                if ( me->getEntries() != 0 && me->getEntries() != 0 ) {
                  meLedL1PN_->setBinContent( ipseudostripx, ichanx, xval );
                  if ( xval == 0 ) meLedL1PNErr_->Fill( ism );
                }

              }

            }

            if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 2) != ledWavelengths_.end() ) {

              me = eeldc->meg10_[ism-1];

              if( me ) {

                float xval = me->getBinContent(i,1);

                if ( me->getEntries() != 0 && me->getEntries() != 0 ) {
                  meLedL2PN_->setBinContent( ipseudostripx, ichanx, xval );
                  if ( xval == 0 ) meLedL2PNErr_->Fill( ism );
                }

              }

            }

          }

        }
      }

      for ( int ix=1; ix<=50; ix++ ) {
        for (int iy=1; iy<=50; iy++ ) {

          int jx = ix + Numbers::ix0EE(ism);
          int jy = iy + Numbers::iy0EE(ism);
	  if( ism >= 1 && ism <= 9 ) jx = 101 - jx;

	  if( !Numbers::validEE(ism, jx, jy) ) continue;

          int ic = Numbers::icEE(ism, jx, jy);

	  if ( eelc ) {

	    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 1) != laserWavelengths_.end() ) {

	      MonitorElement *meg = eelc->meg01_[ism-1];

	      float xval = 2;
	      if ( meg ) xval = meg->getBinContent( ix, iy );

	      // exclude channels without laser data (yellow in the quality map)
	      if( xval != 2 && xval != 5 ) {

		MonitorElement* mea01 = eelc->mea01_[ism-1];
		MonitorElement* met01 = eelc->met01_[ism-1];
		MonitorElement* meaopn01 = eelc->meaopn01_[ism-1];

		if( mea01 && met01 && meaopn01 ) {
		  meLaserL1Ampl_->Fill( ism, mea01->getBinContent( ic ) );
		  if( met01->getBinContent( ic ) > 0. ) meLaserL1Timing_->Fill( ism, met01->getBinContent( ic ) );
		  meLaserL1AmplOverPN_->Fill( ism, meaopn01->getBinContent( ic ) );
		}

	      }

	    }

	    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 2) != laserWavelengths_.end() ) {

	      MonitorElement *meg = eelc->meg02_[ism-1];

	      float xval = 2;
	      if ( meg ) xval = meg->getBinContent( ix, iy );

	      // exclude channels without laser data (yellow in the quality map)
	      if( xval != 2 && xval != 5 ) {

		MonitorElement* mea02 = eelc->mea02_[ism-1];
		MonitorElement* met02 = eelc->met02_[ism-1];
		MonitorElement* meaopn02 = eelc->meaopn02_[ism-1];

		if( mea02 && met02 && meaopn02 ) {
		  meLaserL2Ampl_->Fill( ism, mea02->getBinContent( ic ) );
		  if( met02->getBinContent( ic ) > 0. ) meLaserL2Timing_->Fill( ism, met02->getBinContent( ic ) );
		  meLaserL2AmplOverPN_->Fill( ism, meaopn02->getBinContent( ic ) );
		}

	      }

	    }

	    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 3) != laserWavelengths_.end() ) {

	      MonitorElement *meg = eelc->meg03_[ism-1];

	      float xval = 2;
	      if ( meg ) xval = meg->getBinContent( ix, iy );

	      // exclude channels without laser data (yellow in the quality map)
	      if( xval != 2 && xval != 5 ) {

		MonitorElement* mea03 = eelc->mea03_[ism-1];
		MonitorElement* met03 = eelc->met03_[ism-1];
		MonitorElement* meaopn03 = eelc->meaopn03_[ism-1];

		if( mea03 && met03 && meaopn03 ) {
		  meLaserL3Ampl_->Fill( ism, mea03->getBinContent( ic ) );
		  if( met03->getBinContent( ic ) > 0. ) meLaserL3Timing_->Fill( ism, met03->getBinContent( ic ) );
		  meLaserL3AmplOverPN_->Fill( ism, meaopn03->getBinContent( ic ) );
		}

	      }

	    }

	    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 4) != laserWavelengths_.end() ) {

	      MonitorElement *meg = eelc->meg04_[ism-1];

	      float xval = 2;
	      if ( meg ) xval = meg->getBinContent( ix, iy );

	      // exclude channels without laser data (yellow in the quality map)
	      if( xval != 2 && xval != 5 ) {

		MonitorElement* mea04 = eelc->mea04_[ism-1];
		MonitorElement* met04 = eelc->met04_[ism-1];
		MonitorElement* meaopn04 = eelc->meaopn04_[ism-1];

		if( mea04 && met04 && meaopn04 ) {
		  meLaserL4Ampl_->Fill( ism, mea04->getBinContent( ic ) );
		  if( met04->getBinContent( ic ) > 0. ) meLaserL4Timing_->Fill( ism, met04->getBinContent( ic ) );
		  meLaserL4AmplOverPN_->Fill( ism, meaopn04->getBinContent( ic ) );
		}

	      }

	    }

	  }

	  if ( eeldc ) {

	    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 1) != ledWavelengths_.end() ) {

	      MonitorElement *meg = eeldc->meg01_[ism-1];

	      float xval = 2;
	      if ( meg )  xval = meg->getBinContent( ix, iy );

	      // exclude channels without led data (yellow in the quality map)
	      if( xval != 2 && xval != 5 ) {

		MonitorElement* mea01 = eeldc->mea01_[ism-1];
		MonitorElement* met01 = eeldc->met01_[ism-1];
		MonitorElement* meaopn01 = eeldc->meaopn01_[ism-1];

		if( mea01 && met01 && meaopn01 ) {
		  meLedL1Ampl_->Fill( ism, mea01->getBinContent( ic ) );
		  if( met01->getBinContent( ic ) > 0. ) meLedL1Timing_->Fill( ism, met01->getBinContent( ic ) );
		  meLedL1AmplOverPN_->Fill( ism, meaopn01->getBinContent( ic ) );
		}

	      }

	    }

	    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 2) != ledWavelengths_.end() ) {

	      MonitorElement *meg = eeldc->meg02_[ism-1];

	      float xval = 2;
	      if ( meg )  xval = meg->getBinContent( ix, iy );

	      // exclude channels without led data (yellow in the quality map)
	      if( xval != 2 && xval != 5 ) {

		MonitorElement* mea02 = eeldc->mea02_[ism-1];
		MonitorElement* met02 = eeldc->met02_[ism-1];
		MonitorElement* meaopn02 = eeldc->meaopn02_[ism-1];

		if( mea02 && met02 && meaopn02 ) {
		  meLedL2Ampl_->Fill( ism, mea02->getBinContent( ic ) );
		  if( met02->getBinContent( ic ) > 0. ) meLedL2Timing_->Fill( ism, met02->getBinContent( ic ) );
		  meLedL2AmplOverPN_->Fill( ism, meaopn02->getBinContent( ic ) );
		}

	      }

	    }

	  }

	  if ( eetpc ) {

	    MonitorElement *meg01 = eetpc->meg01_[ism-1];
	    MonitorElement *meg02 = eetpc->meg02_[ism-1];
	    MonitorElement *meg03 = eetpc->meg03_[ism-1];

	    if ( meg01 ) {

	      float xval01 = meg01->getBinContent( ix, iy );

	      if ( xval01 != 2 && xval01 != 5 ) {

		me = eetpc->mea01_[ism-1];

		if ( me ) {

		  meTestPulseAmplG01_->Fill( ism, me->getBinContent( ic ) );

		}

	      }

	    }

	    if ( meg02 ) {

	      float xval02 = meg02->getBinContent( ix, iy );

	      if ( xval02 != 2 && xval02 != 5 ) {

		me = eetpc->mea02_[ism-1];

		if ( me ) {

		  meTestPulseAmplG06_->Fill( ism, me->getBinContent( ic ) );

		}

	      }

	    }

	    if ( meg03 ) {

	      float xval03 = meg03->getBinContent( ix, iy );

	      if ( xval03 != 2 && xval03 != 5 ) {

		me = eetpc->mea03_[ism-1];

		if ( me ) {

		  meTestPulseAmplG12_->Fill( ism, me->getBinContent( ic ) );

		}

	      }

	    }

	  } //etpc


        } // loop on iy
      } // loop on ix

    } // loop on SM

    // fix TPG quality plots

    for ( unsigned int i=0; i<superModules_.size(); i++ ) {

      int ism = superModules_[i];

      for ( int ix = 1; ix <= 50; ix++ ) {
        for ( int iy = 1; iy <= 50; iy++ ) {

          int jx = ix + Numbers::ix0EE(ism);
          int jy = iy + Numbers::iy0EE(ism);

          if ( eetttc ) {

            if ( ism >= 1 && ism <= 9 ) {
              if ( meTriggerTowerEmulError_[0]->getBinContent( 101 - jx, jy ) == 6 ) {
                if ( Numbers::validEE(ism, 101 - jx, jy) ) meTriggerTowerEmulError_[0]->setBinContent( 101 - jx, jy, 2 );
              }
            } else {
              if ( meTriggerTowerEmulError_[1]->getBinContent( jx, jy ) == 6 ) {
                if ( Numbers::validEE(ism, jx, jy) ) meTriggerTowerEmulError_[1]->setBinContent( jx, jy, 2 );
              }
            }

          }

        }
      }

    }

  } // loop on clients

  // The global-summary
  int nGlobalErrors = 0;
  int nGlobalErrorsEE[18];
  int nValidChannels = 0;
  int nValidChannelsEE[18];

  for (int i = 0; i < 18; i++) {
    nGlobalErrorsEE[i] = 0;
    nValidChannelsEE[i] = 0;
  }

  for ( int jx = 1; jx <= 100; jx++ ) {
    for ( int jy = 1; jy <= 100; jy++ ) {

      if(meGlobalSummary_[0]) {

        float xval = 6;
        float val_in = meIntegrity_[0]->getBinContent(jx,jy);
        float val_po = mePedestalOnline_[0]->getBinContent(jx,jy);
        float val_tm = reducedReports_ ? 1. : meTiming_[0]->getBinContent((jx-1)/5+1,(jy-1)/5+1);
        float val_sf = meStatusFlags_[0]->getBinContent(jx,jy);
	float val_ee = reducedReports_ ? 1. : meTriggerTowerEmulError_[0]->getBinContent(jx,jy); // removed temporarily from the global summary
        // float val_ee = 1;

        // combine all the available wavelenghts in unique laser status
        // for each laser turn dark color and yellow into bright green
        float val_ls_1=2, val_ls_2=2, val_ls_3=2, val_ls_4=2;
        if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 1) != laserWavelengths_.end() ) {
          if ( meLaserL1_[0] ) val_ls_1 = meLaserL1_[0]->getBinContent(jx,jy);
          if(val_ls_1==2 || val_ls_1==3 || val_ls_1==4 || val_ls_1==5) val_ls_1=1;
        }
        if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 2) != laserWavelengths_.end() ) {
          if ( meLaserL2_[0] ) val_ls_2 = meLaserL2_[0]->getBinContent(jx,jy);
          if(val_ls_2==2 || val_ls_2==3 || val_ls_2==4 || val_ls_2==5) val_ls_2=1;
        }
        if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 3) != laserWavelengths_.end() ) {
          if ( meLaserL3_[0] ) val_ls_3 = meLaserL3_[0]->getBinContent(jx,jy);
          if(val_ls_3==2 || val_ls_3==3 || val_ls_3==4 || val_ls_3==5) val_ls_3=1;
        }
        if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 4) != laserWavelengths_.end() ) {
          if ( meLaserL4_[0] ) val_ls_4 = meLaserL4_[0]->getBinContent(jx,jy);
          if(val_ls_4==2 || val_ls_4==3 || val_ls_4==4 || val_ls_4==5) val_ls_4=1;
        }

        float val_ls = 1;
        if (val_ls_1 == 0 || val_ls_2==0 || val_ls_3==0 || val_ls_4==0) val_ls=0;

        // combine all the available wavelenghts in unique led status
        // for each laser turn dark color and yellow into bright green
        float val_ld_1=2, val_ld_2=2;
        if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 1) != ledWavelengths_.end() ) {
          if ( meLedL1_[0] ) val_ld_1 = meLedL1_[0]->getBinContent(jx,jy);
          if(val_ld_1==2 || val_ld_1==3 || val_ld_1==4 || val_ld_1==5) val_ld_1=1;
        }
        if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 2) != ledWavelengths_.end() ) {
          if ( meLedL2_[0] ) val_ld_2 = meLedL2_[0]->getBinContent(jx,jy);
          if(val_ld_2==2 || val_ld_2==3 || val_ld_2==4 || val_ld_2==5) val_ld_2=1;
        }

        float val_ld = 1;
        if (val_ld_1 == 0 || val_ld_2==0) val_ld=0;

        // DO NOT CONSIDER CALIBRATION EVENTS IN THE REPORT SUMMARY FOR NOW
        val_ls = 1;
        val_ld = 1;

        // turn each dark color (masked channel) to bright green
        // for laser & timing & trigger turn also yellow into bright green
        // for pedestal online too because is not computed in calibration events

        //  0/3 = red/dark red
        //  1/4 = green/dark green
        //  2/5 = yellow/dark yellow
        //  6   = unknown

        if(             val_in==3 || val_in==4 || val_in==5) val_in=1;
        if(val_po==2 || val_po==3 || val_po==4 || val_po==5) val_po=1;
        if(val_ls==2 || val_ls==3 || val_ls==4 || val_ls==5) val_ls=1;
        if(val_ld==2 || val_ld==3 || val_ld==4 || val_ld==5) val_ld=1;
        if(val_tm==2 || val_tm==3 || val_tm==4 || val_tm==5) val_tm=1;
        if(             val_sf==3 || val_sf==4 || val_sf==5) val_sf=1;
        if(val_ee==2 || val_ee==3 || val_ee==4 || val_ee==5) val_ee=1;

        if(val_in==6) xval=6;
        else if(val_in==0) xval=0;
        else if(val_po==0 || val_ls==0 || val_ld==0 || val_tm==0 || val_sf==0 || val_ee==0) xval=0;
        else if(val_po==2 || val_ls==2 || val_ld==2 || val_tm==2 || val_sf==2 || val_ee==2) xval=2;
        else xval=1;

        bool validCry = false;

        // if the SM is entirely not read, the masked channels
        // are reverted back to yellow
        float iEntries=0;

        for(int ism = 1; ism <= 9; ism++) {
          std::vector<int>::iterator iter = find(superModules_.begin(), superModules_.end(), ism);
          if (iter != superModules_.end()) {
            if ( Numbers::validEE(ism, jx, jy) ) {
              validCry = true;

              // recycle the validEE for the synch check of the DCC
              if(norm01_ && synch01_) {
                float frac_synch_errors = 0.;
                float norm = norm01_->GetBinContent(ism);
                if(norm > 0) frac_synch_errors = float(synch01_->GetBinContent(ism))/float(norm);
                if(frac_synch_errors > synchErrorThreshold_){
		  xval = 0;
		  if(oosTrend && oosTrend->GetBinContent(oosTrend->GetNbinsX()) - oosTrend->GetBinContent(1) < 1.) xval += 3.;
		}
              }

              for ( unsigned int i=0; i<clients_.size(); i++ ) {
                EEIntegrityClient* eeic = dynamic_cast<EEIntegrityClient*>(clients_[i]);
                if ( eeic ) {
                  TH2F* h2 = eeic->h_[ism-1];
                  if ( h2 ) {
                    iEntries = h2->GetEntries();
                  }
                }
              }
            }
          }
        }

        if ( validCry && iEntries==0 ) {
          xval=2;
        }

        meGlobalSummary_[0]->setBinContent( jx, jy, xval );

        if ( xval >= 0 && xval <= 5 ) {
          if ( xval != 2 && xval != 5 ) ++nValidChannels;
          for (int i = 1; i <= 9; i++) {
            if ( xval != 2 && xval != 5 ) {
              if ( Numbers::validEE(i, jx, jy) ) ++nValidChannelsEE[i-1];
            }
          }
          if ( xval == 0 ) ++nGlobalErrors;
          for (int i = 1; i <= 9; i++) {
            if ( xval == 0 ) {
              if ( Numbers::validEE(i, jx, jy) ) ++nGlobalErrorsEE[i-1];
            }
          }
        }

      }

      if(meGlobalSummary_[1]) {

        float xval = 6;
        float val_in = meIntegrity_[1]->getBinContent(jx,jy);
        float val_po = mePedestalOnline_[1]->getBinContent(jx,jy);
        float val_tm = reducedReports_ ? 1. : meTiming_[1]->getBinContent((jx-1)/5+1,(jy-1)/5+1);
        float val_sf = meStatusFlags_[1]->getBinContent(jx,jy);
	float val_ee = reducedReports_ ? 1. : meTriggerTowerEmulError_[1]->getBinContent(jx,jy); // removed temporarily from the global summary
        // float val_ee = 1;

        // combine all the available wavelenghts in unique laser status
        // for each laser turn dark color and yellow into bright green
        float val_ls_1=2, val_ls_2=2, val_ls_3=2, val_ls_4=2;
        if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 1) != laserWavelengths_.end() ) {
          if ( meLaserL1_[1] ) val_ls_1 = meLaserL1_[1]->getBinContent(jx,jy);
          if(val_ls_1==2 || val_ls_1==3 || val_ls_1==4 || val_ls_1==5) val_ls_1=1;
        }
        if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 2) != laserWavelengths_.end() ) {
          if ( meLaserL2_[1] ) val_ls_2 = meLaserL2_[1]->getBinContent(jx,jy);
          if(val_ls_2==2 || val_ls_2==3 || val_ls_2==4 || val_ls_2==5) val_ls_2=1;
        }
        if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 3) != laserWavelengths_.end() ) {
          if ( meLaserL3_[1] ) val_ls_3 = meLaserL3_[1]->getBinContent(jx,jy);
          if(val_ls_3==2 || val_ls_3==3 || val_ls_3==4 || val_ls_3==5) val_ls_3=1;
        }
        if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 4) != laserWavelengths_.end() ) {
          if ( meLaserL4_[1] ) val_ls_4 = meLaserL4_[1]->getBinContent(jx,jy);
          if(val_ls_4==2 || val_ls_4==3 || val_ls_4==4 || val_ls_4==5) val_ls_4=1;
        }

        float val_ls = 1;
        if (val_ls_1 == 0 || val_ls_2==0 || val_ls_3==0 || val_ls_4==0) val_ls=0;

        // combine all the available wavelenghts in unique laser status
        // for each laser turn dark color and yellow into bright green
        float val_ld_1=2, val_ld_2=2;
        if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 1) != ledWavelengths_.end() ) {
          if ( meLedL1_[1] ) val_ld_1 = meLedL1_[1]->getBinContent(jx,jy);
          if(val_ld_1==2 || val_ld_1==3 || val_ld_1==4 || val_ld_1==5) val_ld_1=1;
        }
        if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 2) != ledWavelengths_.end() ) {
          if ( meLedL2_[1] ) val_ld_2 = meLedL2_[1]->getBinContent(jx,jy);
          if(val_ld_2==2 || val_ld_2==3 || val_ld_2==4 || val_ld_2==5) val_ld_2=1;
        }

        float val_ld = 1;
        if (val_ld_1 == 0 || val_ld_2==0) val_ld=0;

        // DO NOT CONSIDER CALIBRATION EVENTS IN THE REPORT SUMMARY FOR NOW
        val_ls = 1;
        val_ld = 1;

        // turn each dark color to bright green
        // for laser & timing & trigger turn also yellow into bright green
        // for pedestal online too because is not computed in calibration events

        //  0/3 = red/dark red
        //  1/4 = green/dark green
        //  2/5 = yellow/dark yellow
        //  6   = unknown

        if(             val_in==3 || val_in==4 || val_in==5) val_in=1;
        if(val_po==2 || val_po==3 || val_po==4 || val_po==5) val_po=1;
        if(val_ls==2 || val_ls==3 || val_ls==4 || val_ls==5) val_ls=1;
        if(val_ld==2 || val_ld==3 || val_ld==4 || val_ld==5) val_ld=1;
        if(val_tm==2 || val_tm==3 || val_tm==4 || val_tm==5) val_tm=1;
        if(             val_sf==3 || val_sf==4 || val_sf==5) val_sf=1;
        if(val_ee==2 || val_ee==3 || val_ee==4 || val_ee==5) val_ee=1;

        if(val_in==6) xval=6;
        else if(val_in==0) xval=0;
        else if(val_po==0 || val_ls==0 || val_ld==0 || val_tm==0 || val_sf==0 || val_ee==0) xval=0;
        else if(val_po==2 || val_ls==2 || val_ld==2 || val_tm==2 || val_sf==2 || val_ee==2) xval=2;
        else xval=1;

        bool validCry = false;

        // if the SM is entirely not read, the masked channels
        // are reverted back in yellow
        float iEntries=0;

        for(int ism = 10; ism <= 18; ism++) {
          std::vector<int>::iterator iter = find(superModules_.begin(), superModules_.end(), ism);
          if (iter != superModules_.end()) {
            if ( Numbers::validEE(ism, jx, jy) ) {
              validCry = true;

              // recycle the validEE for the synch check of the DCC
              if(norm01_ && synch01_) {
                float frac_synch_errors = 0.;
                float norm = norm01_->GetBinContent(ism);
                if(norm > 0) frac_synch_errors = float(synch01_->GetBinContent(ism))/float(norm);
                if(frac_synch_errors > synchErrorThreshold_){
		  xval = 0.;
		  if(oosTrend && oosTrend->GetBinContent(oosTrend->GetNbinsX()) - oosTrend->GetBinContent(1) < 1.) xval += 3.;
		}
              }

              for ( unsigned int i=0; i<clients_.size(); i++ ) {
                EEIntegrityClient* eeic = dynamic_cast<EEIntegrityClient*>(clients_[i]);
                if ( eeic ) {
                  TH2F* h2 = eeic->h_[ism-1];
                  if ( h2 ) {
                    iEntries = h2->GetEntries();
                  }
                }
              }
            }
          }
        }

        if ( validCry && iEntries==0 ) {
          xval=2;
        }

        meGlobalSummary_[1]->setBinContent( jx, jy, xval );

        if ( xval >= 0 && xval <= 5 ) {
          if ( xval != 2 && xval != 5 ) ++nValidChannels;
          for (int i = 10; i <= 18; i++) {
            if ( xval != 2 && xval != 5 ) {
              if ( Numbers::validEE(i, jx, jy) ) ++nValidChannelsEE[i-1];
            }
          }
          if ( xval == 0 ) ++nGlobalErrors;
          for (int i = 10; i <= 18; i++) {
            if ( xval == 0 ) {
              if ( Numbers::validEE(i, jx, jy) ) ++nGlobalErrorsEE[i-1];
            }
          }
        }

      }

    }
  }

  if(meSummaryErr_)
    meSummaryErr_->setBinContent(1, double(nGlobalErrors) / double(nValidChannels));

  float reportSummary = -1.0;
  if ( nValidChannels != 0 )
    reportSummary = 1.0 - float(nGlobalErrors)/float(nValidChannels);
  me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummary");
  if ( me ) me->Fill(reportSummary);

  for (int i = 0; i < 18; i++) {
    float reportSummaryEE = -1.0;
    if ( nValidChannelsEE[i] != 0 )
      reportSummaryEE = 1.0 - float(nGlobalErrorsEE[i])/float(nValidChannelsEE[i]);
    me = dqmStore_->get( prefixME_ + "/EventInfo/reportSummaryContents/EcalEndcap_" + Numbers::sEE(i+1) );
    if ( me ) me->Fill(reportSummaryEE);
  }

  if(meGlobalSummary_[0] && meGlobalSummary_[1]){

    me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummaryMap");
    if ( me ) {

      int nValidChannelsSC[2][20][20];
      int nGlobalErrorsSC[2][20][20];
      for ( int iside = 0; iside < 2; iside++ ) {
	for ( int jxdcc = 0; jxdcc < 20; jxdcc++ ) {
	  for ( int jydcc = 0; jydcc < 20; jydcc++ ) {
	    nValidChannelsSC[iside][jxdcc][jydcc] = 0;
	    nGlobalErrorsSC[iside][jxdcc][jydcc] = 0;
	  }
	}
      }

      for (int iside = 0; iside < 2; iside++ ) {
	for ( int ix = 1; ix <= 100; ix++ ) {
	  for ( int iy = 1; iy <= 100; iy++ ) {

	    int jxsc = (ix-1)/5;
	    int jysc = (iy-1)/5;

	    float xval = meGlobalSummary_[iside]->getBinContent( ix, iy );

	    if ( xval >= 0 && xval <= 5 ) {
	      if ( xval != 2 && xval != 5 ) ++nValidChannelsSC[iside][jxsc][jysc];
	      if ( xval == 0 ) ++nGlobalErrorsSC[iside][jxsc][jysc];
	    }

	  }
	}
      }

      // Countermeasure to partial TR failure
      // make the whole Dee red if more than 2 towers within a 2x2 matrix fails

      for(int iside(0); iside < 2; iside++){
	for(int jy(1); jy < 20; jy++){
	  for(int jx(1); jx < 20; jx++){
            if(jy > 6 && jy < 14 && jx > 6 && jx < 14) continue;
	    int nErr(0);
	    if(nValidChannelsSC[iside][jx - 1][jy - 1] > 0 && nGlobalErrorsSC[iside][jx - 1][jy - 1] == nValidChannelsSC[iside][jx - 1][jy - 1]) nErr += 1;
	    if(nValidChannelsSC[iside][jx][jy - 1] > 0 && nGlobalErrorsSC[iside][jx][jy - 1] == nValidChannelsSC[iside][jx][jy - 1]) nErr += 1;
	    if(nValidChannelsSC[iside][jx - 1][jy] > 0 && nGlobalErrorsSC[iside][jx - 1][jy] == nValidChannelsSC[iside][jx - 1][jy]) nErr += 1;
	    if(nValidChannelsSC[iside][jx][jy] > 0 && nGlobalErrorsSC[iside][jx][jy] == nValidChannelsSC[iside][jx][jy]) nErr += 1;
	    if(nErr > 2){
	      int jx0(((jx - 1) / 10) * 10);
	      for(int jjx(jx0); jjx < jx0 + 10; jjx++){
		for(int jjy(0); jjy < 20; jjy++){
		  nGlobalErrorsSC[iside][jjx][jjy] = nValidChannelsSC[iside][jjx][jjy];
		}
	      }
	    }
	  }
	}
      }

      for (int iside = 0; iside < 2; iside++ ) {
	for ( int jxsc = 0; jxsc < 20; jxsc++ ) {
	  for ( int jysc = 0; jysc < 20; jysc++ ) {

	    float scval = -1;

	    if( nValidChannelsSC[iside][jxsc][jysc] != 0 )
	      scval = 1.0 - float(nGlobalErrorsSC[iside][jxsc][jysc])/float(nValidChannelsSC[iside][jxsc][jysc]);

	    me->setBinContent( jxsc+iside*20+1, jysc+1, scval );

	  }
	}
      }
    }

//     for ( int jxdcc = 0; jxdcc < 20; jxdcc++ ) {
//       for ( int jydcc = 0; jydcc < 20; jydcc++ ) {
//         for ( int iside = 0; iside < 2; iside++ ) {

//           float xval = -1.0;
//           if ( nOutOfGeometryTT[iside][jxdcc][jydcc] < 25 ) {
//             if ( nValidChannelsTT[iside][jxdcc][jydcc] != 0 )
//               xval = 1.0 - float(nGlobalErrorsTT[iside][jxdcc][jydcc])/float(nValidChannelsTT[iside][jxdcc][jydcc]);
//           }

//           me->setBinContent( 20*iside+jxdcc+1, jydcc+1, xval );

//         }
//       }
//     }

  }

}

