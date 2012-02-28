/*
 * \file EBSummaryClient.cc
 *
 * $Date: 2011/11/15 17:21:29 $
 * $Revision: 1.226 $
 * \author G. Della Ricca
 *
*/

#include <memory>
#include <iostream>
#include <fstream>
#include <iomanip>

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#ifdef WITH_ECAL_COND_DB
#include "OnlineDB/EcalCondDB/interface/RunTag.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"
#endif

#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"

#include "DQM/EcalCommon/interface/UtilsClient.h"
#include "DQM/EcalCommon/interface/Numbers.h"
#include "DQM/EcalCommon/interface/Masks.h"

#include "DQM/EcalBarrelMonitorClient/interface/EBStatusFlagsClient.h"
#include "DQM/EcalBarrelMonitorClient/interface/EBIntegrityClient.h"
#include "DQM/EcalBarrelMonitorClient/interface/EBLaserClient.h"
#include "DQM/EcalBarrelMonitorClient/interface/EBPedestalClient.h"
#include "DQM/EcalBarrelMonitorClient/interface/EBPedestalOnlineClient.h"
#include "DQM/EcalBarrelMonitorClient/interface/EBTestPulseClient.h"
#include "DQM/EcalBarrelMonitorClient/interface/EBTriggerTowerClient.h"
#include "DQM/EcalBarrelMonitorClient/interface/EBClusterClient.h"
#include "DQM/EcalBarrelMonitorClient/interface/EBTimingClient.h"

#include "DQM/EcalBarrelMonitorClient/interface/EBSummaryClient.h"

#include "TString.h"
#include "TPRegexp.h"
#include "TObjArray.h"
#include "TObjString.h"

EBSummaryClient::EBSummaryClient(const edm::ParameterSet& ps) {

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

  laserWavelengths_.reserve(4);
  for ( unsigned int i = 1; i <= 4; i++ ) laserWavelengths_.push_back(i);
  laserWavelengths_ = ps.getUntrackedParameter<std::vector<int> >("laserWavelengths", laserWavelengths_);

  MGPAGains_.reserve(3);
  for ( unsigned int i = 1; i <= 3; i++ ) MGPAGains_.push_back(i);
  MGPAGains_ = ps.getUntrackedParameter<std::vector<int> >("MGPAGains", MGPAGains_);

  MGPAGainsPN_.reserve(2);
  for ( unsigned int i = 1; i <= 3; i++ ) MGPAGainsPN_.push_back(i);
  MGPAGainsPN_ = ps.getUntrackedParameter<std::vector<int> >("MGPAGainsPN", MGPAGainsPN_);

  enabledClients_ = ps.getUntrackedParameter<std::vector<std::string> >("enabledClients");

  // summary maps
  meIntegrity_            = 0;
  meIntegrityPN_          = 0;
  meOccupancy_            = 0;
  meOccupancyPN_          = 0;
  meStatusFlags_          = 0;
  mePedestalOnline_       = 0;
  mePedestalOnlineRMSMap_ = 0;
  mePedestalOnlineMean_   = 0;
  mePedestalOnlineRMS_    = 0;

  meLaserL1_              = 0;
  meLaserL1PN_            = 0;
  meLaserL1Ampl_          = 0;
  meLaserL1Timing_        = 0;
  meLaserL1AmplOverPN_    = 0;

  meLaserL2_              = 0;
  meLaserL2PN_            = 0;
  meLaserL2Ampl_          = 0;
  meLaserL2Timing_        = 0;
  meLaserL2AmplOverPN_    = 0;

  meLaserL3_              = 0;
  meLaserL3PN_            = 0;
  meLaserL3Ampl_          = 0;
  meLaserL3Timing_        = 0;
  meLaserL3AmplOverPN_    = 0;

  meLaserL4_              = 0;
  meLaserL4PN_            = 0;
  meLaserL4Ampl_          = 0;
  meLaserL4Timing_        = 0;
  meLaserL4AmplOverPN_    = 0;

  mePedestalG01_          = 0;
  mePedestalG06_          = 0;
  mePedestalG12_          = 0;
  mePedestalPNG01_        = 0;
  mePedestalPNG16_        = 0;
  meTestPulseG01_         = 0;
  meTestPulseG06_         = 0;
  meTestPulseG12_         = 0;
  meTestPulsePNG01_       = 0;
  meTestPulsePNG16_       = 0;
  meTestPulseAmplG01_     = 0;
  meTestPulseAmplG06_     = 0;
  meTestPulseAmplG12_     = 0;
  meGlobalSummary_        = 0;

  meRecHitEnergy_         = 0;
  meTiming_         = 0;
  meTimingMean1D_   = 0;
  meTimingRMS1D_    = 0;
  meTimingMean_     = 0;
  meTimingRMS_      = 0;
  meTriggerTowerEt_        = 0;
  meTriggerTowerEmulError_ = 0;
  meTriggerTowerTiming_ = 0;
  meTriggerTowerNonSingleTiming_ = 0;

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

  // additional histograms from tasks
  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    hpot01_[ism-1] = 0;
    httt01_[ism-1] = 0;
    htmt01_[ism-1] = 0;

  }

  timingNHitThreshold_ = ps.getUntrackedParameter<int>("timingNHitThreshold", 5);
  synchErrorThreshold_ = ps.getUntrackedParameter<int>("synchErrorThreshold", 5);


  ievt_ = 0;
  jevt_ = 0;
  dqmStore_ = 0;


}

EBSummaryClient::~EBSummaryClient() {

}

void EBSummaryClient::beginJob(void) {

  dqmStore_ = edm::Service<DQMStore>().operator->();

  if ( debug_ ) std::cout << "EBSummaryClient: beginJob" << std::endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EBSummaryClient::beginRun(void) {

  if ( debug_ ) std::cout << "EBSummaryClient: beginRun" << std::endl;

  jevt_ = 0;

  this->setup();

}

void EBSummaryClient::endJob(void) {

  if ( debug_ ) std::cout << "EBSummaryClient: endJob, ievt = " << ievt_ << std::endl;

  this->cleanup();

}

void EBSummaryClient::endRun(void) {

  if ( debug_ ) std::cout << "EBSummaryClient: endRun, jevt = " << jevt_ << std::endl;

  this->cleanup();

}

void EBSummaryClient::setup(void) {

  std::string name;

  std::vector<std::string>::iterator clBegin(enabledClients_.begin()), clEnd(enabledClients_.end());

  dqmStore_->setCurrentFolder( prefixME_ + "/Summary" );

  if (std::find(clBegin, clEnd, "Integrity") != clEnd) {

    if ( meIntegrity_ ) dqmStore_->removeElement( meIntegrity_->getName() );
    name = "Summary integrity quality EB";
    meIntegrity_ = dqmStore_->book2D(name, name, 360, 0., 360., 170, -85., 85.);
    meIntegrity_->setAxisTitle("jphi", 1);
    meIntegrity_->setAxisTitle("jeta", 2);

    if ( meIntegrityErr_ ) dqmStore_->removeElement( meIntegrityErr_->getName() );
    name = "Summary integrity quality errors EB";
    meIntegrityErr_ = dqmStore_->book1D(name, name, 36, 1, 37);
    for (int i = 0; i < 36; i++) {
      meIntegrityErr_->setBinLabel(i+1, Numbers::sEB(i+1).c_str(), 1);
    }

    if ( meIntegrityPN_ ) dqmStore_->removeElement( meIntegrityPN_->getName() );
    name = "Summary PN integrity quality EB";
    meIntegrityPN_ = dqmStore_->book2D(name, name, 90, 0., 90., 20, -10., 10.);
    meIntegrityPN_->setAxisTitle("jchannel", 1);
    meIntegrityPN_->setAxisTitle("jpseudo-strip", 2);

  }

  if (std::find(clBegin, clEnd, "Occupancy") != clEnd) {

    if ( meOccupancy_ ) dqmStore_->removeElement( meOccupancy_->getName() );
    name = "Summary digi occupancy EB";
    meOccupancy_ = dqmStore_->book2D(name, name, 360, 0., 360., 170, -85., 85.);
    meOccupancy_->setAxisTitle("jphi", 1);
    meOccupancy_->setAxisTitle("jeta", 2);

    if ( meOccupancy1D_ ) dqmStore_->removeElement( meOccupancy1D_->getName() );
    name = "Summary digi occupancy 1D EB";
    meOccupancy1D_ = dqmStore_->book1D(name, name, 36, 1, 37);
    for (int i = 0; i < 36; i++) {
      meOccupancy1D_->setBinLabel(i+1, Numbers::sEB(i+1).c_str(), 1);
    }

    if ( meOccupancyPN_ ) dqmStore_->removeElement( meOccupancyPN_->getName() );
    name = "Summary PN digi occupancy EB";
    meOccupancyPN_ = dqmStore_->book2D(name, name, 90, 0., 90., 20, -10., 10.);
    meOccupancyPN_->setAxisTitle("jchannel", 1);
    meOccupancyPN_->setAxisTitle("jpseudo-strip", 2);

  }

  if (std::find(clBegin, clEnd, "StatusFlags") != clEnd) {

    if ( meStatusFlags_ ) dqmStore_->removeElement( meStatusFlags_->getName() );
    name = "Summary front-end status quality EB";
    meStatusFlags_ = dqmStore_->book2D(name, name, 72, 0., 360., 34, -85., 85.);
    meStatusFlags_->setAxisTitle("jphi'", 1);
    meStatusFlags_->setAxisTitle("jeta'", 2);

    if ( meStatusFlagsErr_ ) dqmStore_->removeElement( meStatusFlagsErr_->getName() );
    name = "Summary front-end status errors EB";
    meStatusFlagsErr_ = dqmStore_->book1D(name, name, 36, 1, 37);
    for (int i = 0; i < 36; i++) {
      meStatusFlagsErr_->setBinLabel(i+1, Numbers::sEB(i+1).c_str(), 1);
    }

  }

  if (std::find(clBegin, clEnd, "PedestalOnline") != clEnd) {

    if ( mePedestalOnline_ ) dqmStore_->removeElement( mePedestalOnline_->getName() );
    name = "Summary presample quality EB";
    mePedestalOnline_ = dqmStore_->book2D(name, name, 360, 0., 360., 170, -85., 85.);
    mePedestalOnline_->setAxisTitle("jphi", 1);
    mePedestalOnline_->setAxisTitle("jeta", 2);

    if ( mePedestalOnlineErr_ ) dqmStore_->removeElement( mePedestalOnlineErr_->getName() );
    name = "Summary presample quality errors EB";
    mePedestalOnlineErr_ = dqmStore_->book1D(name, name, 36, 1, 37);
    for (int i = 0; i < 36; i++) {
      mePedestalOnlineErr_->setBinLabel(i+1, Numbers::sEB(i+1).c_str(), 1);
    }

    if ( mePedestalOnlineRMSMap_ ) dqmStore_->removeElement( mePedestalOnlineRMSMap_->getName() );
    name = "Summary presample rms EB";
    mePedestalOnlineRMSMap_ = dqmStore_->book2D(name, name, 360, 0., 360., 170, -85., 85.);
    mePedestalOnlineRMSMap_->setAxisTitle("jphi", 1);
    mePedestalOnlineRMSMap_->setAxisTitle("jeta", 2);

    if ( mePedestalOnlineMean_ ) dqmStore_->removeElement( mePedestalOnlineMean_->getName() );
    name = "Summary presample mean EB";
    mePedestalOnlineMean_ = dqmStore_->bookProfile(name, name, 36, 1, 37, 100, 150., 250.);
    for (int i = 0; i < 36; i++) {
      mePedestalOnlineMean_->setBinLabel(i+1, Numbers::sEB(i+1).c_str(), 1);
    }

    if ( mePedestalOnlineRMS_ ) dqmStore_->removeElement( mePedestalOnlineRMS_->getName() );
    name = "Summary presample rms 1D EB";
    mePedestalOnlineRMS_ = dqmStore_->bookProfile(name, name, 36, 1, 37, 100, 0., 10.);
    for (int i = 0; i < 36; i++) {
      mePedestalOnlineRMS_->setBinLabel(i+1, Numbers::sEB(i+1).c_str(), 1);
    }

  }

  if (std::find(clBegin, clEnd, "Laser") != clEnd) {

    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 1) != laserWavelengths_.end() ) {

      if ( meLaserL1_ ) dqmStore_->removeElement( meLaserL1_->getName() );
      name = "Summary laser quality L1 EB";
      meLaserL1_ = dqmStore_->book2D(name, name, 360, 0., 360., 170, -85., 85.);
      meLaserL1_->setAxisTitle("jphi", 1);
      meLaserL1_->setAxisTitle("jeta", 2);

      if ( meLaserL1Err_ ) dqmStore_->removeElement( meLaserL1Err_->getName() );
      name = "Summary laser quality L1 errors EB";
      meLaserL1Err_ = dqmStore_->book1D(name, name, 36, 1, 37);
      for (int i = 0; i < 36; i++) {
	meLaserL1Err_->setBinLabel(i+1, Numbers::sEB(i+1).c_str(), 1);
      }

      if ( meLaserL1PN_ ) dqmStore_->removeElement( meLaserL1PN_->getName() );
      name = "Summary laser PN quality L1 EB";
      meLaserL1PN_ = dqmStore_->book2D(name, name, 90, 0., 90., 20, -10., 10.);
      meLaserL1PN_->setAxisTitle("jchannel", 1);
      meLaserL1PN_->setAxisTitle("jpseudo-strip", 2);

      if ( meLaserL1PNErr_ ) dqmStore_->removeElement( meLaserL1PNErr_->getName() );
      name = "Summary laser PN quality L1 errors EB";
      meLaserL1PNErr_ = dqmStore_->book1D(name, name, 36, 1, 37);
      for (int i = 0; i < 36; i++) {
	meLaserL1PNErr_->setBinLabel(i+1, Numbers::sEB(i+1).c_str(), 1);
      }

      if ( meLaserL1Ampl_ ) dqmStore_->removeElement( meLaserL1Ampl_->getName() );
      name = "Summary laser amplitude L1 EB";
      meLaserL1Ampl_ = dqmStore_->bookProfile(name, name, 36, 1, 37, 4096, 0., 4096., "s");
      for (int i = 0; i < 36; i++) {
	meLaserL1Ampl_->setBinLabel(i+1, Numbers::sEB(i+1).c_str(), 1);
      }

      if ( meLaserL1Timing_ ) dqmStore_->removeElement( meLaserL1Timing_->getName() );
      name = "Summary laser timing L1 EB";
      meLaserL1Timing_ = dqmStore_->bookProfile(name, name, 36, 1, 37, 100, 0., 10., "s");
      for (int i = 0; i < 36; i++) {
	meLaserL1Timing_->setBinLabel(i+1, Numbers::sEB(i+1).c_str(), 1);
      }

      if ( meLaserL1AmplOverPN_ ) dqmStore_->removeElement( meLaserL1AmplOverPN_->getName() );
      name = "Summary laser APD over PN L1 EB";
      meLaserL1AmplOverPN_ = dqmStore_->bookProfile(name, name, 36, 1, 37, 4096, 0., 4096.*12., "s");
      for (int i = 0; i < 36; i++) {
	meLaserL1AmplOverPN_->setBinLabel(i+1, Numbers::sEB(i+1).c_str(), 1);
      }

    }

    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 2) != laserWavelengths_.end() ) {

      if ( meLaserL2_ ) dqmStore_->removeElement( meLaserL2_->getName() );
      name = "Summary laser quality L2 EB";
      meLaserL2_ = dqmStore_->book2D(name, name, 360, 0., 360., 170, -85., 85.);
      meLaserL2_->setAxisTitle("jphi", 1);
      meLaserL2_->setAxisTitle("jeta", 2);

      if ( meLaserL2Err_ ) dqmStore_->removeElement( meLaserL2Err_->getName() );
      name = "Summary laser quality L2 errors EB";
      meLaserL2Err_ = dqmStore_->book1D(name, name, 36, 1, 37);
      for (int i = 0; i < 36; i++) {
	meLaserL2Err_->setBinLabel(i+1, Numbers::sEB(i+1).c_str(), 1);
      }

      if ( meLaserL2PN_ ) dqmStore_->removeElement( meLaserL2PN_->getName() );
      name = "Summary laser PN quality L2 EB";
      meLaserL2PN_ = dqmStore_->book2D(name, name, 90, 0., 90., 20, -10., 10.);
      meLaserL2PN_->setAxisTitle("jchannel", 1);
      meLaserL2PN_->setAxisTitle("jpseudo-strip", 2);

      if ( meLaserL2PNErr_ ) dqmStore_->removeElement( meLaserL2PNErr_->getName() );
      name = "Summary laser PN quality L2 errors EB";
      meLaserL2PNErr_ = dqmStore_->book1D(name, name, 36, 1, 37);
      for (int i = 0; i < 36; i++) {
	meLaserL2PNErr_->setBinLabel(i+1, Numbers::sEB(i+1).c_str(), 1);
      }

      if ( meLaserL2Ampl_ ) dqmStore_->removeElement( meLaserL2Ampl_->getName() );
      name = "Summary laser amplitude L2 EB";
      meLaserL2Ampl_ = dqmStore_->bookProfile(name, name, 36, 1, 37, 4096, 0., 4096., "s");
      for (int i = 0; i < 36; i++) {
	meLaserL2Ampl_->setBinLabel(i+1, Numbers::sEB(i+1).c_str(), 1);
      }

      if ( meLaserL2Timing_ ) dqmStore_->removeElement( meLaserL2Timing_->getName() );
      name = "Summary laser timing L2 EB";
      meLaserL2Timing_ = dqmStore_->bookProfile(name, name, 36, 1, 37, 100, 0., 10., "s");
      for (int i = 0; i < 36; i++) {
	meLaserL2Timing_->setBinLabel(i+1, Numbers::sEB(i+1).c_str(), 1);
      }

      if ( meLaserL2AmplOverPN_ ) dqmStore_->removeElement( meLaserL2AmplOverPN_->getName() );
      name = "Summary laser APD over PN L2 EB";
      meLaserL2AmplOverPN_ = dqmStore_->bookProfile(name, name, 36, 1, 37, 4096, 0., 4096.*12., "s");
      for (int i = 0; i < 36; i++) {
	meLaserL2AmplOverPN_->setBinLabel(i+1, Numbers::sEB(i+1).c_str(), 1);
      }

    }

    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 3) != laserWavelengths_.end() ) {

      if ( meLaserL3_ ) dqmStore_->removeElement( meLaserL3_->getName() );
      name = "Summary laser quality L3 EB";
      meLaserL3_ = dqmStore_->book2D(name, name, 360, 0., 360., 170, -85., 85.);
      meLaserL3_->setAxisTitle("jphi", 1);
      meLaserL3_->setAxisTitle("jeta", 2);

      if ( meLaserL3Err_ ) dqmStore_->removeElement( meLaserL3Err_->getName() );
      name = "Summary laser quality L3 errors EB";
      meLaserL3Err_ = dqmStore_->book1D(name, name, 36, 1, 37);
      for (int i = 0; i < 36; i++) {
	meLaserL3Err_->setBinLabel(i+1, Numbers::sEB(i+1).c_str(), 1);
      }

      if ( meLaserL3PN_ ) dqmStore_->removeElement( meLaserL3PN_->getName() );
      name = "Summary laser PN quality L3 EB";
      meLaserL3PN_ = dqmStore_->book2D(name, name, 90, 0., 90., 20, -10., 10.);
      meLaserL3PN_->setAxisTitle("jchannel", 1);
      meLaserL3PN_->setAxisTitle("jpseudo-strip", 2);

      if ( meLaserL3PNErr_ ) dqmStore_->removeElement( meLaserL3PNErr_->getName() );
      name = "Summary laser PN quality L3 errors EB";
      meLaserL3PNErr_ = dqmStore_->book1D(name, name, 36, 1, 37);
      for (int i = 0; i < 36; i++) {
	meLaserL3PNErr_->setBinLabel(i+1, Numbers::sEB(i+1).c_str(), 1);
      }

      if ( meLaserL3Ampl_ ) dqmStore_->removeElement( meLaserL3Ampl_->getName() );
      name = "Summary laser amplitude L3 EB";
      meLaserL3Ampl_ = dqmStore_->bookProfile(name, name, 36, 1, 37, 4096, 0., 4096., "s");
      for (int i = 0; i < 36; i++) {
	meLaserL3Ampl_->setBinLabel(i+1, Numbers::sEB(i+1).c_str(), 1);
      }

      if ( meLaserL3Timing_ ) dqmStore_->removeElement( meLaserL3Timing_->getName() );
      name = "Summary laser timing L3 EB";
      meLaserL3Timing_ = dqmStore_->bookProfile(name, name, 36, 1, 37, 100, 0., 10., "s");
      for (int i = 0; i < 36; i++) {
	meLaserL3Timing_->setBinLabel(i+1, Numbers::sEB(i+1).c_str(), 1);
      }

      if ( meLaserL3AmplOverPN_ ) dqmStore_->removeElement( meLaserL3AmplOverPN_->getName() );
      name = "Summary laser APD over PN L3 EB";
      meLaserL3AmplOverPN_ = dqmStore_->bookProfile(name, name, 36, 1, 37, 4096, 0., 4096.*12., "s");
      for (int i = 0; i < 36; i++) {
	meLaserL3AmplOverPN_->setBinLabel(i+1, Numbers::sEB(i+1).c_str(), 1);
      }

    }

    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 4) != laserWavelengths_.end() ) {

      if ( meLaserL4_ ) dqmStore_->removeElement( meLaserL4_->getName() );
      name = "Summary laser quality L4 EB";
      meLaserL4_ = dqmStore_->book2D(name, name, 360, 0., 360., 170, -85., 85.);
      meLaserL4_->setAxisTitle("jphi", 1);
      meLaserL4_->setAxisTitle("jeta", 2);

      if ( meLaserL4Err_ ) dqmStore_->removeElement( meLaserL4Err_->getName() );
      name = "Summary laser quality L4 errors EB";
      meLaserL4Err_ = dqmStore_->book1D(name, name, 36, 1, 37);
      for (int i = 0; i < 36; i++) {
	meLaserL4Err_->setBinLabel(i+1, Numbers::sEB(i+1).c_str(), 1);
      }

      if ( meLaserL4PN_ ) dqmStore_->removeElement( meLaserL4PN_->getName() );
      name = "Summary laser PN quality L4 EB";
      meLaserL4PN_ = dqmStore_->book2D(name, name, 90, 0., 90., 20, -10., 10.);
      meLaserL4PN_->setAxisTitle("jchannel", 1);
      meLaserL4PN_->setAxisTitle("jpseudo-strip", 2);

      if ( meLaserL4PNErr_ ) dqmStore_->removeElement( meLaserL4PNErr_->getName() );
      name = "Summary laser PN quality L4 errors EB";
      meLaserL4PNErr_ = dqmStore_->book1D(name, name, 36, 1, 37);
      for (int i = 0; i < 36; i++) {
	meLaserL4PNErr_->setBinLabel(i+1, Numbers::sEB(i+1).c_str(), 1);
      }

      if ( meLaserL4Ampl_ ) dqmStore_->removeElement( meLaserL4Ampl_->getName() );
      name = "Summary laser amplitude L4 EB";
      meLaserL4Ampl_ = dqmStore_->bookProfile(name, name, 36, 1, 37, 4096, 0., 4096., "s");
      for (int i = 0; i < 36; i++) {
	meLaserL4Ampl_->setBinLabel(i+1, Numbers::sEB(i+1).c_str(), 1);
      }

      if ( meLaserL4Timing_ ) dqmStore_->removeElement( meLaserL4Timing_->getName() );
      name = "Summary laser timing L4 EB";
      meLaserL4Timing_ = dqmStore_->bookProfile(name, name, 36, 1, 37, 100, 0., 10., "s");
      for (int i = 0; i < 36; i++) {
	meLaserL4Timing_->setBinLabel(i+1, Numbers::sEB(i+1).c_str(), 1);
      }

      if ( meLaserL4AmplOverPN_ ) dqmStore_->removeElement( meLaserL4AmplOverPN_->getName() );
      name = "Summary laser APD over PN L4 EB";
      meLaserL4AmplOverPN_ = dqmStore_->bookProfile(name, name, 36, 1, 37, 4096, 0., 4096.*12., "s");
      for (int i = 0; i < 36; i++) {
	meLaserL4AmplOverPN_->setBinLabel(i+1, Numbers::sEB(i+1).c_str(), 1);
      }

    }

  }

  if (std::find(clBegin, clEnd, "Pedestal") != clEnd) {

    if (find(MGPAGains_.begin(), MGPAGains_.end(), 1) != MGPAGains_.end() ) {

      if( mePedestalG01_ ) dqmStore_->removeElement( mePedestalG01_->getName() );
      name = "Summary pedestal quality G01 EB";
      mePedestalG01_ = dqmStore_->book2D(name, name, 360, 0., 360., 170, -85., 85.);
      mePedestalG01_->setAxisTitle("jphi", 1);
      mePedestalG01_->setAxisTitle("jeta", 2);

    }

    if (find(MGPAGains_.begin(), MGPAGains_.end(), 6) != MGPAGains_.end() ) {

      if( mePedestalG06_ ) dqmStore_->removeElement( mePedestalG06_->getName() );
      name = "Summary pedestal quality G06 EB";
      mePedestalG06_ = dqmStore_->book2D(name, name, 360, 0., 360., 170, -85., 85.);
      mePedestalG06_->setAxisTitle("jphi", 1);
      mePedestalG06_->setAxisTitle("jeta", 2);

    }

    if (find(MGPAGains_.begin(), MGPAGains_.end(), 12) != MGPAGains_.end() ) {

      if( mePedestalG12_ ) dqmStore_->removeElement( mePedestalG12_->getName() );
      name = "Summary pedestal quality G12 EB";
      mePedestalG12_ = dqmStore_->book2D(name, name, 360, 0., 360., 170, -85., 85.);
      mePedestalG12_->setAxisTitle("jphi", 1);
      mePedestalG12_->setAxisTitle("jeta", 2);

    }

    if (find(MGPAGainsPN_.begin(), MGPAGainsPN_.end(), 1) != MGPAGainsPN_.end() ) {

      if( mePedestalPNG01_ ) dqmStore_->removeElement( mePedestalPNG01_->getName() );
      name = "Summary PN pedestal quality G01 EB";
      mePedestalPNG01_ = dqmStore_->book2D(name, name, 90, 0., 90., 20, -10, 10.);
      mePedestalPNG01_->setAxisTitle("jchannel", 1);
      mePedestalPNG01_->setAxisTitle("jpseudo-strip", 2);

    }

    if (find(MGPAGainsPN_.begin(), MGPAGainsPN_.end(), 16) != MGPAGainsPN_.end() ) {

      if( mePedestalPNG16_ ) dqmStore_->removeElement( mePedestalPNG16_->getName() );
      name = "Summary PN pedestal quality G16 EB";
      mePedestalPNG16_ = dqmStore_->book2D(name, name, 90, 0., 90., 20, -10, 10.);
      mePedestalPNG16_->setAxisTitle("jchannel", 1);
      mePedestalPNG16_->setAxisTitle("jpseudo-strip", 2);

    }

  }

  if (std::find(clBegin, clEnd, "TestPulse") != clEnd) {

    if (find(MGPAGains_.begin(), MGPAGains_.end(), 1) != MGPAGains_.end() ) {

      if( meTestPulseG01_ ) dqmStore_->removeElement( meTestPulseG01_->getName() );
      name = "Summary test pulse quality G01 EB";
      meTestPulseG01_ = dqmStore_->book2D(name, name, 360, 0., 360., 170, -85., 85.);
      meTestPulseG01_->setAxisTitle("jphi", 1);
      meTestPulseG01_->setAxisTitle("jeta", 2);

    }

    if (find(MGPAGains_.begin(), MGPAGains_.end(), 6) != MGPAGains_.end() ) {

      if( meTestPulseG06_ ) dqmStore_->removeElement( meTestPulseG06_->getName() );
      name = "Summary test pulse quality G06 EB";
      meTestPulseG06_ = dqmStore_->book2D(name, name, 360, 0., 360., 170, -85., 85.);
      meTestPulseG06_->setAxisTitle("jphi", 1);
      meTestPulseG06_->setAxisTitle("jeta", 2);

    }

    if (find(MGPAGains_.begin(), MGPAGains_.end(), 12) != MGPAGains_.end() ) {

      if( meTestPulseG12_ ) dqmStore_->removeElement( meTestPulseG12_->getName() );
      name = "Summary test pulse quality G12 EB";
      meTestPulseG12_ = dqmStore_->book2D(name, name, 360, 0., 360., 170, -85., 85.);
      meTestPulseG12_->setAxisTitle("jphi", 1);
      meTestPulseG12_->setAxisTitle("jeta", 2);

    }

    if (find(MGPAGainsPN_.begin(), MGPAGainsPN_.end(), 1) != MGPAGainsPN_.end() ) {

      if( meTestPulsePNG01_ ) dqmStore_->removeElement( meTestPulsePNG01_->getName() );
      name = "Summary PN test pulse quality G01 EB";
      meTestPulsePNG01_ = dqmStore_->book2D(name, name, 90, 0., 90., 20, -10., 10.);
      meTestPulsePNG01_->setAxisTitle("jchannel", 1);
      meTestPulsePNG01_->setAxisTitle("jpseudo-strip", 2);

    }

    if (find(MGPAGainsPN_.begin(), MGPAGainsPN_.end(), 16) != MGPAGainsPN_.end() ) {

      if( meTestPulsePNG16_ ) dqmStore_->removeElement( meTestPulsePNG16_->getName() );
      name = "Summary PN test pulse quality G16 EB";
      meTestPulsePNG16_ = dqmStore_->book2D(name, name, 90, 0., 90., 20, -10., 10.);
      meTestPulsePNG16_->setAxisTitle("jchannel", 1);
      meTestPulsePNG16_->setAxisTitle("jpseudo-strip", 2);

    }

    if (find(MGPAGains_.begin(), MGPAGains_.end(), 1) != MGPAGains_.end() ) {

      if( meTestPulseAmplG01_ ) dqmStore_->removeElement( meTestPulseAmplG01_->getName() );
      name = "Summary test pulse amplitude G01 EB";
      meTestPulseAmplG01_ = dqmStore_->bookProfile(name, name, 36, 1, 37, 4096, 0., 4096.*12., "s");
      for (int i = 0; i < 36; i++) {
	meTestPulseAmplG01_->setBinLabel(i+1, Numbers::sEB(i+1).c_str(), 1);
      }

    }

    if (find(MGPAGains_.begin(), MGPAGains_.end(), 6) != MGPAGains_.end() ) {

      if( meTestPulseAmplG06_ ) dqmStore_->removeElement( meTestPulseAmplG06_->getName() );
      name = "Summary test pulse amplitude G06 EB";
      meTestPulseAmplG06_ = dqmStore_->bookProfile(name, name, 36, 1, 37, 4096, 0., 4096.*12., "s");
      for (int i = 0; i < 36; i++) {
	meTestPulseAmplG06_->setBinLabel(i+1, Numbers::sEB(i+1).c_str(), 1);
      }

    }

    if (find(MGPAGains_.begin(), MGPAGains_.end(), 12) != MGPAGains_.end() ) {

      if( meTestPulseAmplG12_ ) dqmStore_->removeElement( meTestPulseAmplG12_->getName() );
      name = "Summary test pulse amplitude G12 EB";
      meTestPulseAmplG12_ = dqmStore_->bookProfile(name, name, 36, 1, 37, 4096, 0., 4096.*12., "s");
      for (int i = 0; i < 36; i++) {
	meTestPulseAmplG12_->setBinLabel(i+1, Numbers::sEB(i+1).c_str(), 1);
      }

    }

  }

  if (std::find(clBegin, clEnd, "Cosmic") != clEnd) {

    if( meRecHitEnergy_ ) dqmStore_->removeElement( meRecHitEnergy_->getName() );
    name = "Summary rec hit energy EB";
    meRecHitEnergy_ = dqmStore_->book2D(name, name, 360, 0., 360., 170, -85., 85.);
    meRecHitEnergy_->setAxisTitle("jphi", 1);
    meRecHitEnergy_->setAxisTitle("jeta", 2);

  }

  if (std::find(clBegin, clEnd, "Timing") != clEnd) {

    if( meTiming_ ) dqmStore_->removeElement( meTiming_->getName() );
    name = "Summary timing quality EB";
    meTiming_ = dqmStore_->book2D(name, name, 72, 0., 360., 34, -85., 85.);
    meTiming_->setAxisTitle("jphi", 1);
    meTiming_->setAxisTitle("jeta", 2);

    if( meTimingMean1D_ ) dqmStore_->removeElement( meTimingMean1D_->getName() );
    name = "Summary timing mean 1D EB";
    meTimingMean1D_ = dqmStore_->book1D(name, name, 100, -25., 25.);
    meTimingMean1D_->setAxisTitle("mean (ns)", 1);

    if( meTimingRMS1D_ ) dqmStore_->removeElement( meTimingRMS1D_->getName() );
    name = "Summary timing rms 1D EB";
    meTimingRMS1D_ = dqmStore_->book1D(name, name, 100, 0.0, 10.0);
    meTimingRMS1D_->setAxisTitle("rms (ns)", 1);

    if ( meTimingMean_ ) dqmStore_->removeElement( meTimingMean_->getName() );
    name = "Summary timing mean EB";
    meTimingMean_ = dqmStore_->bookProfile(name, name, 36, 1, 37, -20., 20.,"");
    for (int i = 0; i < 36; i++) {
      meTimingMean_->setBinLabel(i+1, Numbers::sEB(i+1).c_str(), 1);
    }
    meTimingMean_->setAxisTitle("mean (ns)", 2);

    if ( meTimingRMS_ ) dqmStore_->removeElement( meTimingRMS_->getName() );
    name = "Summary timing rms EB";
    meTimingRMS_ = dqmStore_->bookProfile(name, name, 36, 1, 37, 100, 0., 10.,"");
    for (int i = 0; i < 36; i++) {
      meTimingRMS_->setBinLabel(i+1, Numbers::sEB(i+1).c_str(), 1);
    }
    meTimingRMS_->setAxisTitle("rms (ns)", 2);

  }

  if (std::find(clBegin, clEnd, "TriggerTower") != clEnd) {

    if( meTriggerTowerEt_ ) dqmStore_->removeElement( meTriggerTowerEt_->getName() );
    name = "Summary TP Et EB";
    meTriggerTowerEt_ = dqmStore_->book2D(name, name, 72, 0., 360., 34, -85., 85.);
    meTriggerTowerEt_->setAxisTitle("jphi'", 1);
    meTriggerTowerEt_->setAxisTitle("jeta'", 2);

    if( meTriggerTowerEmulError_ ) dqmStore_->removeElement( meTriggerTowerEmulError_->getName() );
    name = "Summary TP emul error quality EB";
    meTriggerTowerEmulError_ = dqmStore_->book2D(name, name, 72, 0., 360., 34, -85., 85.);
    meTriggerTowerEmulError_->setAxisTitle("jphi'", 1);
    meTriggerTowerEmulError_->setAxisTitle("jeta'", 2);

    if( meTriggerTowerTiming_ ) dqmStore_->removeElement( meTriggerTowerTiming_->getName() );
    name = "Summary TP timing EB";
    meTriggerTowerTiming_ = dqmStore_->book2D(name, name, 72, 0., 360., 34, -85., 85.);
    meTriggerTowerTiming_->setAxisTitle("jphi'", 1);
    meTriggerTowerTiming_->setAxisTitle("jeta'", 2);
    meTriggerTowerTiming_->setAxisTitle("TP data matching emulator", 3);

    if( meTriggerTowerNonSingleTiming_ ) dqmStore_->removeElement( meTriggerTowerNonSingleTiming_->getName() );
    name = "Summary TP non single timing errors EB";
    meTriggerTowerNonSingleTiming_ = dqmStore_->book2D(name, name, 72, 0., 360., 34, -85., 85.);
    meTriggerTowerNonSingleTiming_->setAxisTitle("jphi'", 1);
    meTriggerTowerNonSingleTiming_->setAxisTitle("jeta'", 2);
    meTriggerTowerNonSingleTiming_->setAxisTitle("fraction", 3);

  }

  if(meIntegrity_ && mePedestalOnline_ && meTiming_ && meStatusFlags_) {

    if( meGlobalSummary_ ) dqmStore_->removeElement( meGlobalSummary_->getName() );
    name = "Summary global quality EB";
    meGlobalSummary_ = dqmStore_->book2D(name, name, 360, 0., 360., 170, -85., 85.);
    meGlobalSummary_->setAxisTitle("jphi", 1);
    meGlobalSummary_->setAxisTitle("jeta", 2);

  }

}

void EBSummaryClient::cleanup(void) {

  if ( ! enableCleanup_ ) return;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( cloneME_ ) {
      if ( hpot01_[ism-1] ) delete hpot01_[ism-1];
      if ( httt01_[ism-1] ) delete httt01_[ism-1];
    }

    hpot01_[ism-1] = 0;
    httt01_[ism-1] = 0;

  }

  if ( meIntegrity_ ) dqmStore_->removeElement( meIntegrity_->getFullname() );
  meIntegrity_ = 0;

  if ( meIntegrityErr_ ) dqmStore_->removeElement( meIntegrityErr_->getFullname() );
  meIntegrityErr_ = 0;

  if ( meIntegrityPN_ ) dqmStore_->removeElement( meIntegrityPN_->getFullname() );
  meIntegrityPN_ = 0;

  if ( meOccupancy_ ) dqmStore_->removeElement( meOccupancy_->getFullname() );
  meOccupancy_ = 0;

  if ( meOccupancy1D_ ) dqmStore_->removeElement( meOccupancy1D_->getFullname() );
  meOccupancy1D_ = 0;

  if ( meOccupancyPN_ ) dqmStore_->removeElement( meOccupancyPN_->getFullname() );
  meOccupancyPN_ = 0;

  if ( meStatusFlags_ ) dqmStore_->removeElement( meStatusFlags_->getFullname() );
  meStatusFlags_ = 0;

  if ( meStatusFlagsErr_ ) dqmStore_->removeElement( meStatusFlagsErr_->getFullname() );
  meStatusFlagsErr_ = 0;

  if ( mePedestalOnline_ ) dqmStore_->removeElement( mePedestalOnline_->getFullname() );
  mePedestalOnline_ = 0;

  if ( mePedestalOnlineErr_ ) dqmStore_->removeElement( mePedestalOnlineErr_->getFullname() );
  mePedestalOnlineErr_ = 0;

  if ( mePedestalOnlineMean_ ) dqmStore_->removeElement( mePedestalOnlineMean_->getFullname() );
  mePedestalOnlineMean_ = 0;

  if ( mePedestalOnlineRMS_ ) dqmStore_->removeElement( mePedestalOnlineRMS_->getFullname() );
  mePedestalOnlineRMS_ = 0;

  if ( mePedestalOnlineRMSMap_ ) dqmStore_->removeElement( mePedestalOnlineRMSMap_->getFullname() );
  mePedestalOnlineRMSMap_ = 0;

  if ( meLaserL1_ ) dqmStore_->removeElement( meLaserL1_->getFullname() );
  meLaserL1_ = 0;

  if ( meLaserL1Err_ ) dqmStore_->removeElement( meLaserL1Err_->getFullname() );
  meLaserL1Err_ = 0;

  if ( meLaserL1Ampl_ ) dqmStore_->removeElement( meLaserL1Ampl_->getFullname() );
  meLaserL1Ampl_ = 0;

  if ( meLaserL1Timing_ ) dqmStore_->removeElement( meLaserL1Timing_->getFullname() );
  meLaserL1Timing_ = 0;

  if ( meLaserL1AmplOverPN_ ) dqmStore_->removeElement( meLaserL1AmplOverPN_->getFullname() );
  meLaserL1AmplOverPN_ = 0;

  if ( meLaserL1PN_ ) dqmStore_->removeElement( meLaserL1PN_->getFullname() );
  meLaserL1PN_ = 0;

  if ( meLaserL1PNErr_ ) dqmStore_->removeElement( meLaserL1PNErr_->getFullname() );
  meLaserL1PNErr_ = 0;

  if ( meLaserL2_ ) dqmStore_->removeElement( meLaserL2_->getFullname() );
  meLaserL2_ = 0;

  if ( meLaserL2Err_ ) dqmStore_->removeElement( meLaserL2Err_->getFullname() );
  meLaserL2Err_ = 0;

  if ( meLaserL2Ampl_ ) dqmStore_->removeElement( meLaserL2Ampl_->getFullname() );
  meLaserL2Ampl_ = 0;

  if ( meLaserL2Timing_ ) dqmStore_->removeElement( meLaserL2Timing_->getFullname() );
  meLaserL2Timing_ = 0;

  if ( meLaserL2AmplOverPN_ ) dqmStore_->removeElement( meLaserL2AmplOverPN_->getFullname() );
  meLaserL2AmplOverPN_ = 0;

  if ( meLaserL2PN_ ) dqmStore_->removeElement( meLaserL2PN_->getFullname() );
  meLaserL2PN_ = 0;

  if ( meLaserL2PNErr_ ) dqmStore_->removeElement( meLaserL2PNErr_->getFullname() );
  meLaserL2PNErr_ = 0;

  if ( meLaserL3_ ) dqmStore_->removeElement( meLaserL3_->getFullname() );
  meLaserL3_ = 0;

  if ( meLaserL3Err_ ) dqmStore_->removeElement( meLaserL3Err_->getFullname() );
  meLaserL3Err_ = 0;

  if ( meLaserL3Ampl_ ) dqmStore_->removeElement( meLaserL3Ampl_->getFullname() );
  meLaserL3Ampl_ = 0;

  if ( meLaserL3Timing_ ) dqmStore_->removeElement( meLaserL3Timing_->getFullname() );
  meLaserL3Timing_ = 0;

  if ( meLaserL3AmplOverPN_ ) dqmStore_->removeElement( meLaserL3AmplOverPN_->getFullname() );
  meLaserL3AmplOverPN_ = 0;

  if ( meLaserL3PN_ ) dqmStore_->removeElement( meLaserL3PN_->getFullname() );
  meLaserL3PN_ = 0;

  if ( meLaserL3PNErr_ ) dqmStore_->removeElement( meLaserL3PNErr_->getFullname() );
  meLaserL3PNErr_ = 0;

  if ( meLaserL4_ ) dqmStore_->removeElement( meLaserL4_->getFullname() );
  meLaserL4_ = 0;

  if ( meLaserL4Err_ ) dqmStore_->removeElement( meLaserL4Err_->getFullname() );
  meLaserL4Err_ = 0;

  if ( meLaserL4Ampl_ ) dqmStore_->removeElement( meLaserL4Ampl_->getFullname() );
  meLaserL4Ampl_ = 0;

  if ( meLaserL4Timing_ ) dqmStore_->removeElement( meLaserL4Timing_->getFullname() );
  meLaserL4Timing_ = 0;

  if ( meLaserL4AmplOverPN_ ) dqmStore_->removeElement( meLaserL4AmplOverPN_->getFullname() );
  meLaserL4AmplOverPN_ = 0;

  if ( meLaserL4PN_ ) dqmStore_->removeElement( meLaserL4PN_->getFullname() );
  meLaserL4PN_ = 0;

  if ( meLaserL4PNErr_ ) dqmStore_->removeElement( meLaserL4PNErr_->getFullname() );
  meLaserL4PNErr_ = 0;

  if ( mePedestalG01_ ) dqmStore_->removeElement( mePedestalG01_->getFullname() );
  mePedestalG01_ = 0;

  if ( mePedestalG06_ ) dqmStore_->removeElement( mePedestalG06_->getFullname() );
  mePedestalG06_ = 0;

  if ( mePedestalG12_ ) dqmStore_->removeElement( mePedestalG12_->getFullname() );
  mePedestalG12_ = 0;

  if ( meTestPulseG01_ ) dqmStore_->removeElement( meTestPulseG01_->getFullname() );
  meTestPulseG01_ = 0;

  if ( meTestPulseG06_ ) dqmStore_->removeElement( meTestPulseG06_->getFullname() );
  meTestPulseG06_ = 0;

  if ( meTestPulseG12_ ) dqmStore_->removeElement( meTestPulseG12_->getFullname() );
  meTestPulseG12_ = 0;

  if ( meTestPulseG01_ ) dqmStore_->removeElement( meTestPulseG01_->getFullname() );
  meTestPulseG01_ = 0;

  if ( meTestPulseAmplG01_ ) dqmStore_->removeElement( meTestPulseAmplG01_->getFullname() );
  meTestPulseAmplG01_ = 0;

  if ( meTestPulseAmplG06_ ) dqmStore_->removeElement( meTestPulseAmplG06_->getFullname() );
  meTestPulseAmplG06_ = 0;

  if ( meTestPulseAmplG12_ ) dqmStore_->removeElement( meTestPulseAmplG12_->getFullname() );
  meTestPulseAmplG12_ = 0;

  if ( meRecHitEnergy_ ) dqmStore_->removeElement( meRecHitEnergy_->getFullname() );
  meRecHitEnergy_ = 0;

  if ( meTiming_ ) dqmStore_->removeElement( meTiming_->getFullname() );
  meTiming_ = 0;

  if ( meTimingMean1D_ ) dqmStore_->removeElement( meTimingMean1D_->getFullname() );
  meTimingMean1D_ = 0;

  if ( meTimingRMS1D_ ) dqmStore_->removeElement( meTimingRMS1D_->getFullname() );
  meTimingRMS1D_ = 0;

  if ( meTimingMean_ ) dqmStore_->removeElement( meTimingMean_->getFullname() );
  meTimingMean_ = 0;

  if ( meTimingRMS_ ) dqmStore_->removeElement( meTimingRMS_->getFullname() );
  meTimingRMS_ = 0;

  if ( meTriggerTowerEt_ ) dqmStore_->removeElement( meTriggerTowerEt_->getFullname() );
  meTriggerTowerEt_ = 0;

  if ( meTriggerTowerEmulError_ ) dqmStore_->removeElement( meTriggerTowerEmulError_->getFullname() );
  meTriggerTowerEmulError_ = 0;

  if ( meTriggerTowerTiming_ ) dqmStore_->removeElement( meTriggerTowerTiming_->getFullname() );
  meTriggerTowerTiming_ = 0;

  if ( meTriggerTowerNonSingleTiming_ ) dqmStore_->removeElement( meTriggerTowerNonSingleTiming_->getFullname() );
  meTriggerTowerNonSingleTiming_ = 0;

  if ( meGlobalSummary_ ) dqmStore_->removeElement( meGlobalSummary_->getFullname() );
  meGlobalSummary_ = 0;

}

#ifdef WITH_ECAL_COND_DB
bool EBSummaryClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov, bool& status) {

  status = true;

  return true;

}
#endif

void EBSummaryClient::analyze(void) {

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( debug_ ) std::cout << "EBSummaryClient: ievt/jevt = " << ievt_ << "/" << jevt_ << std::endl;
  }

  uint32_t chWarnBit = 1 << EcalDQMStatusHelper::PHYSICS_BAD_CHANNEL_WARNING;

  for ( int iex = 1; iex <= 170; iex++ ) {
    for ( int ipx = 1; ipx <= 360; ipx++ ) {

      if ( meIntegrity_ ) meIntegrity_->setBinContent( ipx, iex, 6. );
      if ( meOccupancy_ ) meOccupancy_->setBinContent( ipx, iex, 0. );
      if ( meStatusFlags_ ) meStatusFlags_->setBinContent( ipx, iex, 6. );
      if ( mePedestalOnline_ ) mePedestalOnline_->setBinContent( ipx, iex, 6. );
      if ( mePedestalOnlineRMSMap_ ) mePedestalOnlineRMSMap_->setBinContent( ipx, iex, -1.);
      if ( meLaserL1_ ) meLaserL1_->setBinContent( ipx, iex, 6. );
      if ( meLaserL2_ ) meLaserL2_->setBinContent( ipx, iex, 6. );
      if ( meLaserL3_ ) meLaserL3_->setBinContent( ipx, iex, 6. );
      if ( meLaserL4_ ) meLaserL4_->setBinContent( ipx, iex, 6. );
      if ( mePedestalG01_ ) mePedestalG01_->setBinContent( ipx, iex, 6. );
      if ( mePedestalG06_ ) mePedestalG06_->setBinContent( ipx, iex, 6. );
      if ( mePedestalG12_ ) mePedestalG12_->setBinContent( ipx, iex, 6. );
      if ( meTestPulseG01_ ) meTestPulseG01_->setBinContent( ipx, iex, 6. );
      if ( meTestPulseG06_ ) meTestPulseG06_->setBinContent( ipx, iex, 6. );
      if ( meTestPulseG12_ ) meTestPulseG12_->setBinContent( ipx, iex, 6. );

      if ( meRecHitEnergy_ ) meRecHitEnergy_->setBinContent( ipx, iex, 0. );

      if ( meGlobalSummary_ ) meGlobalSummary_->setBinContent( ipx, iex, 6. );

    }
  }

  for ( int iex = 1; iex <= 20; iex++ ) {
    for ( int ipx = 1; ipx <= 90; ipx++ ) {

      if ( meIntegrityPN_ ) meIntegrityPN_->setBinContent( ipx, iex, 6. );
      if ( meOccupancyPN_ ) meOccupancyPN_->setBinContent( ipx, iex, 0. );
      if ( meLaserL1PN_ ) meLaserL1PN_->setBinContent( ipx, iex, 6. );
      if ( meLaserL2PN_ ) meLaserL2PN_->setBinContent( ipx, iex, 6. );
      if ( meLaserL3PN_ ) meLaserL3PN_->setBinContent( ipx, iex, 6. );
      if ( meLaserL4PN_ ) meLaserL4PN_->setBinContent( ipx, iex, 6. );
      if ( mePedestalPNG01_ ) mePedestalPNG01_->setBinContent( ipx, iex, 6. );
      if ( mePedestalPNG16_ ) mePedestalPNG16_->setBinContent( ipx, iex, 6. );
      if ( meTestPulsePNG01_ ) meTestPulsePNG01_->setBinContent( ipx, iex, 6. );
      if ( meTestPulsePNG16_ ) meTestPulsePNG16_->setBinContent( ipx, iex, 6. );

    }
  }

  for ( int iex = 1; iex <= 34; iex++ ) {
    for ( int ipx = 1; ipx <= 72; ipx++ ) {
      if ( meTriggerTowerEt_ ) meTriggerTowerEt_->setBinContent( ipx, iex, 0. );
      if ( meTriggerTowerEmulError_ ) meTriggerTowerEmulError_->setBinContent( ipx, iex, 6. );
      if ( meTriggerTowerTiming_ ) meTriggerTowerTiming_->setBinContent( ipx, iex, 0. );
      if ( meTriggerTowerNonSingleTiming_ ) meTriggerTowerNonSingleTiming_->setBinContent( ipx, iex, -1. );
      if ( meTiming_ ) meTiming_->setBinContent( ipx, iex, 6. );
    }
  }

  if ( meIntegrity_ ) meIntegrity_->setEntries( 0 );
  if ( meIntegrityErr_ ) meIntegrityErr_->Reset();
  if ( meIntegrityPN_ ) meIntegrityPN_->setEntries( 0 );
  if ( meOccupancy_ ) meOccupancy_->setEntries( 0 );
  if ( meOccupancy1D_ ) meOccupancy1D_->Reset();
  if ( meOccupancyPN_ ) meOccupancyPN_->setEntries( 0 );
  if ( meStatusFlags_ ) meStatusFlags_->setEntries( 0 );
  if ( meStatusFlagsErr_ ) meStatusFlagsErr_->Reset();
  if ( mePedestalOnline_ ) mePedestalOnline_->setEntries( 0 );
  if ( mePedestalOnlineErr_ ) mePedestalOnlineErr_->Reset();
  if ( mePedestalOnlineMean_ ) mePedestalOnlineMean_->Reset();
  if ( mePedestalOnlineRMS_ ) mePedestalOnlineRMS_->Reset();
  if ( mePedestalOnlineRMSMap_ ) mePedestalOnlineRMSMap_->Reset();

  if ( meLaserL1_ ) meLaserL1_->setEntries( 0 );
  if ( meLaserL1Err_ ) meLaserL1Err_->Reset();
  if ( meLaserL1Ampl_ ) meLaserL1Ampl_->Reset();
  if ( meLaserL1Timing_ ) meLaserL1Timing_->Reset();
  if ( meLaserL1AmplOverPN_ ) meLaserL1AmplOverPN_->Reset();
  if ( meLaserL1PN_ ) meLaserL1PN_->setEntries( 0 );
  if ( meLaserL1PNErr_ ) meLaserL1PNErr_->Reset();

  if ( meLaserL2_ ) meLaserL2_->setEntries( 0 );
  if ( meLaserL2Err_ ) meLaserL2Err_->Reset();
  if ( meLaserL2Ampl_ ) meLaserL2Ampl_->Reset();
  if ( meLaserL2Timing_ ) meLaserL2Timing_->Reset();
  if ( meLaserL2AmplOverPN_ ) meLaserL2AmplOverPN_->Reset();
  if ( meLaserL2PN_ ) meLaserL2PN_->setEntries( 0 );
  if ( meLaserL2PNErr_ ) meLaserL2PNErr_->Reset();

  if ( meLaserL3_ ) meLaserL3_->setEntries( 0 );
  if ( meLaserL3Err_ ) meLaserL3Err_->Reset();
  if ( meLaserL3Ampl_ ) meLaserL3Ampl_->Reset();
  if ( meLaserL3Timing_ ) meLaserL3Timing_->Reset();
  if ( meLaserL3AmplOverPN_ ) meLaserL3AmplOverPN_->Reset();
  if ( meLaserL3PN_ ) meLaserL3PN_->setEntries( 0 );
  if ( meLaserL3PNErr_ ) meLaserL3PNErr_->Reset();

  if ( meLaserL4_ ) meLaserL4_->setEntries( 0 );
  if ( meLaserL4Err_ ) meLaserL4Err_->Reset();
  if ( meLaserL4Ampl_ ) meLaserL4Ampl_->Reset();
  if ( meLaserL4Timing_ ) meLaserL4Timing_->Reset();
  if ( meLaserL4AmplOverPN_ ) meLaserL4AmplOverPN_->Reset();
  if ( meLaserL4PN_ ) meLaserL4PN_->setEntries( 0 );
  if ( meLaserL4PNErr_ ) meLaserL4PNErr_->Reset();

  if ( mePedestalG01_ ) mePedestalG01_->setEntries( 0 );
  if ( mePedestalG06_ ) mePedestalG06_->setEntries( 0 );
  if ( mePedestalG12_ ) mePedestalG12_->setEntries( 0 );
  if ( mePedestalPNG01_ ) mePedestalPNG01_->setEntries( 0 );
  if ( mePedestalPNG16_ ) mePedestalPNG16_->setEntries( 0 );
  if ( meTestPulseG01_ ) meTestPulseG01_->setEntries( 0 );
  if ( meTestPulseG06_ ) meTestPulseG06_->setEntries( 0 );
  if ( meTestPulseG12_ ) meTestPulseG12_->setEntries( 0 );
  if ( meTestPulsePNG01_ ) meTestPulsePNG01_->setEntries( 0 );
  if ( meTestPulsePNG16_ ) meTestPulsePNG16_->setEntries( 0 );
  if ( meTestPulseAmplG01_ ) meTestPulseAmplG01_->Reset();
  if ( meTestPulseAmplG06_ ) meTestPulseAmplG06_->Reset();
  if ( meTestPulseAmplG12_ ) meTestPulseAmplG12_->Reset();

  if ( meRecHitEnergy_ ) meRecHitEnergy_->setEntries( 0 );
  if ( meTiming_ ) meTiming_->setEntries( 0 );
  if ( meTimingMean1D_ ) meTimingMean1D_->Reset();
  if ( meTimingRMS1D_ ) meTimingRMS1D_->Reset();
  if ( meTimingMean_ ) meTimingMean_->Reset();
  if ( meTimingRMS_ ) meTimingRMS_->Reset();
  if ( meTriggerTowerEt_ ) meTriggerTowerEt_->setEntries( 0 );
  if ( meTriggerTowerEmulError_ ) meTriggerTowerEmulError_->setEntries( 0 );
  if ( meTriggerTowerTiming_ ) meTriggerTowerTiming_->setEntries( 0 );
  if ( meTriggerTowerNonSingleTiming_ ) meTriggerTowerNonSingleTiming_->setEntries( 0 );

  if (meGlobalSummary_) meGlobalSummary_->setEntries( 0 );

  MonitorElement *me(0);
  me = dqmStore_->get(prefixME_ + "/Timing/TimingTask timing EB");
  TProfile2D *htmt(0);
  htmt = UtilsClient::getHisto(me, false, htmt);

  for ( unsigned int i=0; i<clients_.size(); i++ ) {

    EBIntegrityClient* ebic = dynamic_cast<EBIntegrityClient*>(clients_[i]);
    EBStatusFlagsClient* ebsfc = dynamic_cast<EBStatusFlagsClient*>(clients_[i]);
    EBPedestalOnlineClient* ebpoc = dynamic_cast<EBPedestalOnlineClient*>(clients_[i]);

    EBLaserClient* eblc = dynamic_cast<EBLaserClient*>(clients_[i]);
    EBPedestalClient* ebpc = dynamic_cast<EBPedestalClient*>(clients_[i]);
    EBTestPulseClient* ebtpc = dynamic_cast<EBTestPulseClient*>(clients_[i]);

    EBTimingClient* ebtmc = dynamic_cast<EBTimingClient*>(clients_[i]);
    EBTriggerTowerClient* ebtttc = dynamic_cast<EBTriggerTowerClient*>(clients_[i]);

    MonitorElement *me_01, *me_02, *me_03;
    MonitorElement *me_04, *me_05;
    //    MonitorElement *me_f[6], *me_fg[2];
    TH2F* h2;

    for ( unsigned int i=0; i<superModules_.size(); i++ ) {

      int ism = superModules_[i];

      me = dqmStore_->get( prefixME_ + "/Energy/Profile/RecHitTask energy " + Numbers::sEB(ism) );
      hot01_[ism-1] = UtilsClient::getHisto( me, cloneME_, hot01_[ism-1] );

      me = dqmStore_->get( prefixME_ + "/Pedestal/Presample/PedestalTask presample G12 " + Numbers::sEB(ism) );
      hpot01_[ism-1] = UtilsClient::getHisto( me, cloneME_, hpot01_[ism-1] );

      me = dqmStore_->get( prefixME_ + "/TriggerPrimitives/Et/TrigPrimTask Et " + Numbers::sEB(ism) );
      httt01_[ism-1] = UtilsClient::getHisto( me, cloneME_, httt01_[ism-1] );

      me = dqmStore_->get( prefixME_ + "/Timing/Profile/TimingTask timing " + Numbers::sEB(ism) );
      htmt01_[ism-1] = UtilsClient::getHisto( me, cloneME_, htmt01_[ism-1] );

      me = dqmStore_->get( prefixME_ + "/RawData/RawDataTask FE-DCC L1A mismatch EB" );
      synch01_ = UtilsClient::getHisto( me, cloneME_, synch01_ );

      for ( int ie = 1; ie <= 85; ie++ ) {
        for ( int ip = 1; ip <= 20; ip++ ) {

          if ( ebic ) {

            me = ebic->meg01_[ism-1];

            if ( me ) {

              float xval = me->getBinContent( ie, ip );

              int iex;
              int ipx;

              if ( ism <= 18 ) {
                iex = 1+(85-ie);
                ipx = ip+20*(ism-1);
              } else {
                iex = 85+ie;
                ipx = 1+(20-ip)+20*(ism-19);
              }

              meIntegrity_->setBinContent( ipx, iex, xval );
              if( xval == 0 ) meIntegrityErr_->Fill( ism );

            }

            h2 = ebic->h_[ism-1];

            if ( h2 ) {

              float xval = h2->GetBinContent( ie, ip );

              int iex;
              int ipx;

              if ( ism <= 18 ) {
                iex = 1+(85-ie);
                ipx = ip+20*(ism-1);
              } else {
                iex = 85+ie;
                ipx = 1+(20-ip)+20*(ism-19);
              }

              meOccupancy_->setBinContent( ipx, iex, xval );
              if ( xval != 0 ) meOccupancy1D_->Fill( ism, xval );

            }

          }

          if ( ebpoc ) {

            me = ebpoc->meg03_[ism-1];

            if ( me ) {

              int iex;
              int ipx;

              if ( ism <= 18 ) {
                iex = 1+(85-ie);
                ipx = ip+20*(ism-1);
              } else {
                iex = 85+ie;
                ipx = 1+(20-ip)+20*(ism-19);
              }

              float xval = me->getBinContent( ie, ip );

              mePedestalOnline_->setBinContent( ipx, iex, xval );
              if ( xval == 0 ) mePedestalOnlineErr_->Fill( ism );

            }

            float num01, mean01, rms01;
            bool update01 = UtilsClient::getBinStatistics(hpot01_[ism-1], ie, ip, num01, mean01, rms01);

            if ( update01 ) {

              int iex;
              int ipx;

              if ( ism <= 18 ) {
                iex = 1+(85-ie);
                ipx = ip+20*(ism-1);
              } else {
                iex = 85+ie;
                ipx = 1+(20-ip)+20*(ism-19);
              }

              mePedestalOnlineRMSMap_->setBinContent( ipx, iex, rms01 );

              mePedestalOnlineRMS_->Fill( ism, rms01 );

              mePedestalOnlineMean_->Fill( ism, mean01 );

            }

          }

          if ( eblc ) {

            int iex;
            int ipx;

            if ( ism <= 18 ) {
              iex = 1+(85-ie);
              ipx = ip+20*(ism-1);
            } else {
              iex = 85+ie;
              ipx = 1+(20-ip)+20*(ism-19);
            }

            if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 1) != laserWavelengths_.end() ) {

              me = eblc->meg01_[ism-1];

              if ( me ) {

                float xval = me->getBinContent( ie, ip );

                if ( me->getEntries() != 0 ) {
                  meLaserL1_->setBinContent( ipx, iex, xval );
                  if ( xval == 0 ) meLaserL1Err_->Fill( ism );
                }

              }

            }

            if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 2) != laserWavelengths_.end() ) {

              me = eblc->meg02_[ism-1];

              if ( me ) {

                float xval = me->getBinContent( ie, ip );

                if ( me->getEntries() != 0 ) {
                  meLaserL2_->setBinContent( ipx, iex, xval );
                  if ( xval == 0 ) meLaserL2Err_->Fill( ism );
                }

              }

            }

            if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 3) != laserWavelengths_.end() ) {

              me = eblc->meg03_[ism-1];

              if ( me ) {

                float xval = me->getBinContent( ie, ip );

                if ( me->getEntries() != 0 ) {
                  meLaserL3_->setBinContent( ipx, iex, xval );
                  if ( xval == 0 ) meLaserL3Err_->Fill( ism );
                }

              }

            }

            if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 4) != laserWavelengths_.end() ) {

              me = eblc->meg04_[ism-1];

              if ( me ) {

                float xval = me->getBinContent( ie, ip );

                if ( me->getEntries() != 0 ) {
                  meLaserL4_->setBinContent( ipx, iex, xval );
                  if ( xval == 0 ) meLaserL4Err_->Fill( ism );
                }

              }

            }

          }

          if ( ebpc ) {

            me_01 = ebpc->meg01_[ism-1];
            me_02 = ebpc->meg02_[ism-1];
            me_03 = ebpc->meg03_[ism-1];

            int iex;
            int ipx;

            if ( ism <= 18 ) {
              iex = 1+(85-ie);
              ipx = ip+20*(ism-1);
            } else {
              iex = 85+ie;
              ipx = 1+(20-ip)+20*(ism-19);
            }

            if ( me_01 ) {
              float val_01=me_01->getBinContent(ie,ip);
              if ( me_01->getEntries() != 0 ) mePedestalG01_->setBinContent( ipx, iex, val_01 );
            }
            if ( me_02 ) {
              float val_02=me_02->getBinContent(ie,ip);
              if ( me_02->getEntries() != 0 ) mePedestalG06_->setBinContent( ipx, iex, val_02 );
            }
            if ( me_03 ) {
              float val_03=me_03->getBinContent(ie,ip);
              if ( me_03->getEntries() != 0 ) mePedestalG12_->setBinContent( ipx, iex, val_03 );
            }

          }

          if ( ebtpc ) {

            me_01 = ebtpc->meg01_[ism-1];
            me_02 = ebtpc->meg02_[ism-1];
            me_03 = ebtpc->meg03_[ism-1];

            int iex;
            int ipx;

            if ( ism <= 18 ) {
              iex = 1+(85-ie);
              ipx = ip+20*(ism-1);
            } else {
              iex = 85+ie;
              ipx = 1+(20-ip)+20*(ism-19);
            }

            if ( me_01 ) {

              float val_01=me_01->getBinContent(ie,ip);

              if ( me_01->getEntries() != 0 ) meTestPulseG01_->setBinContent( ipx, iex, val_01 );

            }
            if ( me_02 ) {

              float val_02=me_02->getBinContent(ie,ip);

              if ( me_02->getEntries() != 0 ) meTestPulseG06_->setBinContent( ipx, iex, val_02 );

            }
            if ( me_03 ) {

              float val_03=me_03->getBinContent(ie,ip);

              if ( me_03->getEntries() != 0 ) meTestPulseG12_->setBinContent( ipx, iex, val_03 );

            }


          }

          if ( hot01_[ism-1] ) {

            float xval = hot01_[ism-1]->GetBinContent( ie, ip );

            int iex;
            int ipx;

            if ( ism <= 18 ) {
              iex = 1+(85-ie);
              ipx = ip+20*(ism-1);
            } else {
              iex = 85+ie;
              ipx = 1+(20-ip)+20*(ism-19);
            }

            meRecHitEnergy_->setBinContent( ipx, iex, xval );

          }

          if ( ebtmc ) {

            float num02, mean02, rms02;

	    bool update02 = UtilsClient::getBinStatistics(htmt01_[ism-1], ie, ip, num02, mean02, rms02, timingNHitThreshold_);

            if ( update02 ) {

	      mean02 -= 50.;

              meTimingMean1D_->Fill(mean02);

              meTimingRMS1D_->Fill(rms02);

              meTimingMean_->Fill( ism, mean02 );

              meTimingRMS_->Fill( ism, rms02 );

            }

          }

        }
      }

      std::vector<float> festatus(17 * 4, 0.);
      if ( ebsfc ) {
	TString name;
	TPRegexp pattern("FEStatusTask Error FE ([0-9]+) ([0-9]+)");
	int idcc, iccu;

	std::vector<MonitorElement *> vme(dqmStore_->getContents(prefixME_ + "/FEStatus/Errors"));
	for(std::vector<MonitorElement *>::iterator meItr(vme.begin()); meItr != vme.end(); ++meItr){
	  if(!(*meItr)) continue;
	  name = (*meItr)->getName().c_str();
	  TObjArray* matches(pattern.MatchS(name));
	  if(matches->GetEntries() == 0) continue;
	  idcc = static_cast<TObjString*>(matches->At(1))->GetString().Atoi();
	  iccu = static_cast<TObjString*>(matches->At(2))->GetString().Atoi();
	  if(idcc != ism + 9) continue;
	  festatus[iccu - 1] = (*meItr)->getBinContent(1);
	}
      }

      std::vector<float> ttstatus(68, 0.);
      if (ebtttc) {
	TString name;
	TPRegexp pattern("TrigPrimClient non single timing TT ([0-9]+) ([0-9]+)");
	int itcc, itt;

	std::vector<MonitorElement *> vme(dqmStore_->getContents(prefixME_ + "/TriggerPrimitives/EmulationErrors/Timing"));
	for(std::vector<MonitorElement *>::iterator meItr(vme.begin()); meItr != vme.end(); ++meItr){
	  if(!(*meItr)) continue;
	  name = (*meItr)->getName().c_str();
	  TObjArray* matches(pattern.MatchS(name));
	  if(matches->GetEntries() == 0) continue;
	  itcc = static_cast<TObjString*>(matches->At(1))->GetString().Atoi();
	  itt = static_cast<TObjString*>(matches->At(2))->GetString().Atoi();
	  if(itcc != 36 + ism) continue;

	  int ieta(ism <= 18 ? 17 - (itt - 1) / 4 : 18 + (itt - 1) / 4);
	  int iphi(((ism - 1) % 18) * 4 + (itt - 1) % 4 + 1);

	  meTriggerTowerNonSingleTiming_->setBinContent(iphi, ieta, (*meItr)->getBinContent(1));
	}

	TPRegexp patternEt("TrigPrimTask emulation Et mismatch TT ([0-9]+) ([0-9]+)");
	vme = dqmStore_->getContents(prefixME_ + "/TriggerPrimitives/EmulationErrors/Et");
	for(std::vector<MonitorElement *>::iterator meItr(vme.begin()); meItr != vme.end(); ++meItr){
	  if(!(*meItr)) continue;
	  name = (*meItr)->getName().c_str();
	  TObjArray* matches(patternEt.MatchS(name));
	  if(matches->GetEntries() == 0) continue;
	  itcc = static_cast<TObjString*>(matches->At(1))->GetString().Atoi();
	  itt = static_cast<TObjString*>(matches->At(2))->GetString().Atoi();
	  if(itcc != ism + 36) continue;
	  ttstatus[itt - 1] = (*meItr)->getBinContent(1);
	}
      }


      for (int ie = 1; ie <= 17; ie++ ) {
        for (int ip = 1; ip <= 4; ip++ ) {

          int iex;
          int ipx;

          if ( ism <= 18 ) {
            iex = 1+(17-ie);
            ipx = ip+4*(ism-1);
          } else {
            iex = 17+ie;
            ipx = 1+(4-ip)+4*(ism-19);
          }

          if ( ebsfc ) {

            me = dqmStore_->get(prefixME_ + "/Occupancy/OccupancyTask DCC occupancy EB");

            float xval = 6;

            if ( me ) {

              xval = 2;
              if ( me->getBinContent( ism ) > 0 ) xval = 1;

            }

//             me = ebsfc->meh01_[ism-1];

//             if ( me ) {

//               if ( me->getBinContent( ie, ip ) > 0 ) xval = 0;

	    if(festatus[(ie - 1) * 4 + ip - 1] > 0) xval = 0;

	    meStatusFlags_->setBinContent( ipx, iex, xval );

//               if ( me->getBinError( ie, ip ) > 0 && me->getBinError( ie, ip ) < 0.1 ) UtilsClient::maskBinContent( meStatusFlags_, ipx, iex );

	    uint32_t mask(0x1 << EcalDQMStatusHelper::STATUS_FLAG_ERROR);

	    for(int localEta(1); localEta <= 5; localEta++){
	      for(int localPhi(1); localPhi <= 5; localPhi++){
		if(Masks::maskChannel(ism, (ie-1)*5+localEta, (ip-1)*5+localPhi, mask, EcalBarrel)){
		  UtilsClient::maskBinContent( meStatusFlags_, ipx, iex );
		  break;
		}
	      }
	    }

	    if ( xval == 0 ) meStatusFlagsErr_->Fill( ism );

//             }



          }

          if ( ebtttc ) {

            float mean01 = 0;
            bool hadNonZeroInterest = false;

            if ( httt01_[ism-1] ) {

              mean01 = httt01_[ism-1]->GetBinContent( ie, ip );

              if ( mean01 != 0. ) {
                if ( meTriggerTowerEt_ ) meTriggerTowerEt_->setBinContent( ipx, iex, mean01 );
              }

            }

            me = ebtttc->me_o01_[ism-1];

            if ( me ) {

              float xval = me->getBinContent( ie, ip );

              if ( xval != 0. ) {
                meTriggerTowerTiming_->setBinContent( ipx, iex, xval );
                hadNonZeroInterest = true;
              }

            }

//             me = ebtttc->me_o02_[ism-1];

//             if ( me ) {

//               float xval = me->getBinContent( ie, ip );

//               if ( xval != 0. ) {
//                 meTriggerTowerNonSingleTiming_->setBinContent( ipx, iex, xval );
//               }

//             }

	    float xval = 2;
	    if( mean01 > 0. ) {

//               h2 = ebtttc->l01_[ism-1];
//               h3 = ebtttc->l02_[ism-1];

//               if ( h2 && h3 ) {

//                 // float emulErrorVal = h2->GetBinContent( ie, ip ) + h3->GetBinContent( ie, ip );
//                 float emulErrorVal = h2->GetBinContent( ie, ip );

//                 if( emulErrorVal!=0 && hadNonZeroInterest ) xval = 0;
	      if(ttstatus[(ie - 1) * 4 + ip - 1] > 0 && hadNonZeroInterest) xval = 0.;
//               }

	      if ( xval!=0 && hadNonZeroInterest ) xval = 1;

//             }
	    }

	    meTriggerTowerEmulError_->setBinContent( ipx, iex, xval );

          }

          if ( ebtmc ) {

	    if( htmt01_[ism-1] ){

	      float num(0.);
	      bool mask(false);

	      for(int ce=1; ce<=5; ce++){
		for(int cp=1; cp<=5; cp++){

		  int scie = (ie - 1) * 5 + ce;
		  int scip = (ip - 1) * 5 + cp; 
		  num += htmt01_[ism-1]->GetBinEntries(htmt01_[ism-1]->GetBin( scie, scip ));

		  if( Masks::maskChannel(ism, scie, scip, chWarnBit, EcalBarrel) ) mask = true;
		}
	      }

	      float nHitThreshold(timingNHitThreshold_ * 18.);

	      bool update01(false);
	      float num01, mean01, rms01;
	      update01 = UtilsClient::getBinStatistics(htmt, ipx, iex, num01, mean01, rms01, nHitThreshold);

	      mean01 -= 50.;

	      if(!update01){
		mean01 = 0.;
		rms01 = 0.;
	      }


	      update01 |= num > 1.3 * nHitThreshold; // allow 10% outliers

	      float xval = 2.;
	      // there is not much point to make both single channel & TT hit threshold configurable
	      if( update01 ){

		// quality BAD if large mean, large rms, or large outliers (num: # events in +-20ns time window)
		if( std::abs(mean01) > 2. || rms01 > 6. || num > 1.3 * num01 ) xval = 0.;
		else xval = 1.;

	      }

	      meTiming_->setBinContent( ipx, iex, xval );
	      if ( mask ) UtilsClient::maskBinContent( meTiming_, ipx, iex );

	    }

          }

        }
      }

      // PN's summaries
      for( int i = 1; i <= 10; i++ ) {
        for( int j = 1; j <= 5; j++ ) {

          int ichanx;
          int ipseudostripx;

          if(ism<=18) {
            ichanx = i;
            ipseudostripx = j+5*(ism-1);
          } else {
            ichanx = i+10;
            ipseudostripx = j+5*(ism-19);
          }

          if ( ebic ) {

            me_04 = ebic->meg02_[ism-1];
            h2 = ebic->hmem_[ism-1];

            if( me_04 ) {

              float xval = me_04->getBinContent(i,j);
              meIntegrityPN_->setBinContent( ipseudostripx, ichanx, xval );

            }

            if ( h2 ) {

              float xval = h2->GetBinContent(i,1);
              meOccupancyPN_->setBinContent( ipseudostripx, ichanx, xval );

            }

          }

          if ( ebpc ) {

            me_04 = ebpc->meg04_[ism-1];
            me_05 = ebpc->meg05_[ism-1];

            if( me_04 ) {
              float val_04=me_04->getBinContent(i,1);
              mePedestalPNG01_->setBinContent( ipseudostripx, ichanx, val_04 );
            }
            if( me_05 ) {
              float val_05=me_05->getBinContent(i,1);
              mePedestalPNG16_->setBinContent( ipseudostripx, ichanx, val_05 );
            }

          }

          if ( ebtpc ) {

            me_04 = ebtpc->meg04_[ism-1];
            me_05 = ebtpc->meg05_[ism-1];

            if( me_04 ) {
              float val_04=me_04->getBinContent(i,1);
              meTestPulsePNG01_->setBinContent( ipseudostripx, ichanx, val_04 );
            }
            if( me_05 ) {
              float val_05=me_05->getBinContent(i,1);
              meTestPulsePNG16_->setBinContent( ipseudostripx, ichanx, val_05 );
            }

          }

          if ( eblc ) {

            if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 1) != laserWavelengths_.end() ) {

              me = eblc->meg09_[ism-1];

              if( me ) {

                float xval = me->getBinContent(i,1);

                if ( me->getEntries() != 0 && me->getEntries() != 0 ) {
                  meLaserL1PN_->setBinContent( ipseudostripx, ichanx, xval );
                  if ( xval == 0 ) meLaserL1PNErr_->Fill( ism );
                }

              }

            }

            if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 2) != laserWavelengths_.end() ) {

              me = eblc->meg10_[ism-1];

              if( me ) {

                float xval = me->getBinContent(i,1);

                if ( me->getEntries() != 0 && me->getEntries() != 0 ) {
                  meLaserL2PN_->setBinContent( ipseudostripx, ichanx, xval );
                  if ( xval == 0 ) meLaserL2PNErr_->Fill( ism );
                }

              }

            }

            if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 3) != laserWavelengths_.end() ) {

              me = eblc->meg11_[ism-1];

              if( me ) {

                float xval = me->getBinContent(i,1);

                if ( me->getEntries() != 0 && me->getEntries() != 0 ) {
                  meLaserL3PN_->setBinContent( ipseudostripx, ichanx, xval );
                  if ( xval == 0 ) meLaserL3PNErr_->Fill( ism );
                }

              }

            }

            if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 4) != laserWavelengths_.end() ) {

              me = eblc->meg12_[ism-1];

              if( me ) {

                float xval = me->getBinContent(i,1);

                if ( me->getEntries() != 0 && me->getEntries() != 0 ) {
                  meLaserL4PN_->setBinContent( ipseudostripx, ichanx, xval );
                  if ( xval == 0 ) meLaserL4PNErr_->Fill( ism );
                }

              }

            }

          }

        }
      }

      for(int chan=0; chan<1700; chan++) {

        int ie = (chan)/20 + 1;
        int ip = (chan)%20 + 1;

        // laser 1D summaries
        if ( eblc ) {

          if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 1) != laserWavelengths_.end() ) {

            MonitorElement *meg = eblc->meg01_[ism-1];

            float xval = 2;
            if ( meg ) xval = meg->getBinContent( ie, ip );

            // exclude channels without laser data (yellow in the quality map)
            if( xval != 2 && xval != 5 ) {

              MonitorElement* mea01 = eblc->mea01_[ism-1];
              MonitorElement* met01 = eblc->met01_[ism-1];
              MonitorElement* meaopn01 = eblc->meaopn01_[ism-1];

              if( mea01 && met01 && meaopn01 ) {
                meLaserL1Ampl_->Fill( ism, mea01->getBinContent( chan+1 ) );
                if( met01->getBinContent( chan+1 ) > 0. ) meLaserL1Timing_->Fill( ism, met01->getBinContent( chan+1 ) );
                meLaserL1AmplOverPN_->Fill( ism, meaopn01->getBinContent( chan+1 ) );
              }

            }

          }

          if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 2) != laserWavelengths_.end() ) {

            MonitorElement *meg = eblc->meg02_[ism-1];

            float xval = 2;
            if ( meg ) xval = meg->getBinContent( ie, ip );

            // exclude channels without laser data (yellow in the quality map)
            if( xval != 2 && xval != 5 ) {

              MonitorElement* mea02 = eblc->mea02_[ism-1];
              MonitorElement* met02 = eblc->met02_[ism-1];
              MonitorElement* meaopn02 = eblc->meaopn02_[ism-1];

              if( mea02 && met02 && meaopn02 ) {
                meLaserL2Ampl_->Fill( ism, mea02->getBinContent( chan+1 ) );
                if( met02->getBinContent( chan+1 ) > 0. ) meLaserL2Timing_->Fill( ism, met02->getBinContent( chan+1 ) );
                meLaserL2AmplOverPN_->Fill( ism, meaopn02->getBinContent( chan+1 ) );
              }

            }

          }

          if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 3) != laserWavelengths_.end() ) {

            MonitorElement *meg = eblc->meg03_[ism-1];

            float xval = 2;
            if ( meg ) xval = meg->getBinContent( ie, ip );

            // exclude channels without laser data (yellow in the quality map)
            if( xval != 2 && xval != 5 ) {

              MonitorElement* mea03 = eblc->mea03_[ism-1];
              MonitorElement* met03 = eblc->met03_[ism-1];
              MonitorElement* meaopn03 = eblc->meaopn03_[ism-1];

              if( mea03 && met03 && meaopn03 ) {
                meLaserL3Ampl_->Fill( ism, mea03->getBinContent( chan+1 ) );
                if( met03->getBinContent( chan+1 ) > 0. ) meLaserL3Timing_->Fill( ism, met03->getBinContent( chan+1 ) );
                meLaserL3AmplOverPN_->Fill( ism, meaopn03->getBinContent( chan+1 ) );
              }

            }

          }

          if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 4) != laserWavelengths_.end() ) {

            MonitorElement *meg = eblc->meg04_[ism-1];

            float xval = 2;
            if ( meg ) xval = meg->getBinContent( ie, ip );

            // exclude channels without laser data (yellow in the quality map)
            if( xval != 2 && xval != 5 ) {

              MonitorElement* mea04 = eblc->mea04_[ism-1];
              MonitorElement* met04 = eblc->met04_[ism-1];
              MonitorElement* meaopn04 = eblc->meaopn04_[ism-1];

              if( mea04 && met04 && meaopn04 ) {
                meLaserL4Ampl_->Fill( ism, mea04->getBinContent( chan+1 ) );
                if( met04->getBinContent( chan+1 ) > 0. ) meLaserL4Timing_->Fill( ism, met04->getBinContent( chan+1 ) );
                meLaserL4AmplOverPN_->Fill( ism, meaopn04->getBinContent( chan+1 ) );
              }

            }

          }

        }

        if ( ebtpc ) {

          MonitorElement *meg01 = ebtpc->meg01_[ism-1];
          MonitorElement *meg02 = ebtpc->meg02_[ism-1];
          MonitorElement *meg03 = ebtpc->meg03_[ism-1];

          if ( meg01 ) {

            float xval01 = meg01->getBinContent(ie,ip);

            if ( xval01 != 2 && xval01 != 5 ) {

              me = ebtpc->mea01_[ism-1];

              if ( me ) {

                meTestPulseAmplG01_->Fill( ism, me->getBinContent( chan+1 ) );

              }

            }

          }

          if ( meg02 ) {

            float xval02 = meg02->getBinContent(ie,ip);

            if ( xval02 != 2 && xval02 != 5 ) {

              me = ebtpc->mea02_[ism-1];

              if ( me ) {

                meTestPulseAmplG06_->Fill( ism, me->getBinContent( chan+1 ) );

              }

            }

          }

          if ( meg03 ) {

            float xval03 = meg03->getBinContent(ie,ip);

            if ( xval03 != 2 && xval03 != 5 ) {

              me = ebtpc->mea03_[ism-1];

              if ( me ) {

                meTestPulseAmplG12_->Fill( ism, me->getBinContent( chan+1 ) );

              }

            }

          }

        }

      }  // loop on channels

    } // loop on SM

  } // loop on clients

  // The global-summary
  int nGlobalErrors = 0;
  int nGlobalErrorsEB[36];
  int nValidChannels = 0;
  int nValidChannelsEB[36];

  for (int i = 0; i < 36; i++) {
    nGlobalErrorsEB[i] = 0;
    nValidChannelsEB[i] = 0;
  }

  for ( int iex = 1; iex <= 170; iex++ ) {
    for ( int ipx = 1; ipx <= 360; ipx++ ) {

      if(meIntegrity_ && mePedestalOnline_ && meTiming_ && meStatusFlags_ && meTriggerTowerEmulError_) {

        int ism = (ipx-1)/20 + 1 ;
        if ( iex>85 ) ism+=18;

	int iet = (iex-1)/5 + 1;
	int ipt = (ipx-1)/5 + 1;

        float xval = 6;
        float val_in = meIntegrity_->getBinContent(ipx,iex);
        float val_po = mePedestalOnline_->getBinContent(ipx,iex);
        float val_tm = meTiming_->getBinContent(ipt,iet);
        float val_sf = meStatusFlags_->getBinContent((ipx-1)/5+1,(iex-1)/5+1);
	// float val_ee = meTriggerTowerEmulError_->getBinContent((ipx-1)/5+1,(iex-1)/5+1); // removed from the global summary temporarily
        float val_ee = 1;

        // combine all the available wavelenghts in unique laser status
        // for each laser turn dark color and yellow into bright green
        float val_ls_1=2, val_ls_2=2, val_ls_3=2, val_ls_4=2;
        if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 1) != laserWavelengths_.end() ) {
          if ( meLaserL1_ ) val_ls_1 = meLaserL1_->getBinContent(ipx,iex);
          if(val_ls_1==2 || val_ls_1==3 || val_ls_1==4 || val_ls_1==5) val_ls_1=1;
        }
        if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 2) != laserWavelengths_.end() ) {
          if ( meLaserL2_ ) val_ls_2 = meLaserL2_->getBinContent(ipx,iex);
          if(val_ls_2==2 || val_ls_2==3 || val_ls_2==4 || val_ls_2==5) val_ls_2=1;
        }
        if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 3) != laserWavelengths_.end() ) {
          if ( meLaserL3_ ) val_ls_3 = meLaserL3_->getBinContent(ipx,iex);
          if(val_ls_3==2 || val_ls_3==3 || val_ls_3==4 || val_ls_3==5) val_ls_3=1;
        }
        if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 4) != laserWavelengths_.end() ) {
          if ( meLaserL4_ ) val_ls_4 = meLaserL4_->getBinContent(ipx,iex);
          if(val_ls_4==2 || val_ls_4==3 || val_ls_4==4 || val_ls_4==5) val_ls_4=1;
        }

        float val_ls = 1;
        if (val_ls_1 == 0 || val_ls_2==0 || val_ls_3==0 || val_ls_4==0) val_ls=0;

        // DO NOT CONSIDER CALIBRATION EVENTS IN THE REPORT SUMMARY UNTIL LHC COLLISIONS
        val_ls = 1;

        // turn each dark color (masked channel) to bright green
        // for laser & timing & trigger turn also yellow into bright green
        // for pedestal online too because is not computed in calibration events

        //  0/3 = red/dark red
        //  1/4 = green/dark green
        //  2/5 = yellow/dark yellow
        //  6   = unknown

        if(             val_in==3 || val_in==4 || val_in==5) val_in=1;
        if(val_po==2 || val_po==3 || val_po==4 || val_po==5) val_po=1;
        if(val_tm==2 || val_tm==3 || val_tm==4 || val_tm==5) val_tm=1;
        if(             val_sf==3 || val_sf==4 || val_sf==5) val_sf=1;
        if(val_ee==2 || val_ee==3 || val_ee==4 || val_ee==5) val_ee=1;

        if(val_in==6) xval=6;
        else if(val_in==0) xval=0;
        else if(val_po==0 || val_ls==0 || val_tm==0 || val_sf==0 || val_ee==0) xval=0;
        else if(val_po==2 || val_ls==2 || val_tm==2 || val_sf==2 || val_ee==2) xval=2;
        else xval=1;

        // if the SM is entirely not read, the masked channels
        // are reverted back to yellow
        float iEntries=0;

        if(synch01_) {
	  float synchErrors = synch01_->GetBinContent(ism);
          if(synchErrors > synchErrorThreshold_) xval=0;
        }

        std::vector<int>::iterator iter = find(superModules_.begin(), superModules_.end(), ism);
        if (iter != superModules_.end()) {
          for ( unsigned int i=0; i<clients_.size(); i++ ) {
            EBIntegrityClient* ebic = dynamic_cast<EBIntegrityClient*>(clients_[i]);
            if ( ebic ) {
              TH2F* h2 = ebic->h_[ism-1];
              if ( h2 ) {
                iEntries = h2->GetEntries();
              }
            }
          }
        }

        if ( iEntries==0 ) {
          xval=2;
        }

        if(meGlobalSummary_) meGlobalSummary_->setBinContent( ipx, iex, xval );

        if ( xval >= 0 && xval <= 5 ) {
          if ( xval != 2 && xval != 5 ) ++nValidChannels;
          if ( iex <= 85 ) {
            if ( xval != 2 && xval != 5 ) ++nValidChannelsEB[(ipx-1)/20];
          } else {
            if ( xval != 2 && xval != 5 ) ++nValidChannelsEB[18+(ipx-1)/20];
          }
          if ( xval == 0 ) ++nGlobalErrors;
          if ( iex <= 85 ) {
            if ( xval == 0 ) ++nGlobalErrorsEB[(ipx-1)/20];
          } else {
            if ( xval == 0 ) ++nGlobalErrorsEB[18+(ipx-1)/20];
          }
        }

      }

    }
  }

  float reportSummary = -1.0;
  float nEB(61200.);
  if ( nValidChannels != 0 )
    reportSummary = 1.0 - float(nGlobalErrors)/float(nValidChannels);
  me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummary EB");
  if ( me )
    me->Fill(reportSummary);
  else
    nEB = 0.;

  // workaround for new EB+EE configuration
  float eeSummary(0.);
  float nEE(14648.);
  me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummary EE");
  if (me)
    eeSummary = me->getFloatValue();
  else
    nEE = 0.;

  me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummary");
  if (me)
    me->Fill((reportSummary * nEB + eeSummary * nEE) / (nEB + nEE));

  for (int i = 0; i < 36; i++) {
    float reportSummaryEB = -1.0;
    if ( nValidChannelsEB[i] != 0 )
      reportSummaryEB = 1.0 - float(nGlobalErrorsEB[i])/float(nValidChannelsEB[i]);
    me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummaryContents/EcalBarrel_" + Numbers::sEB(i+1));
    if ( me ) me->Fill(reportSummaryEB);
  }

  me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummaryMap EB");
  MonitorElement* mecomb(dqmStore_->get(prefixME_ + "/EventInfo/reportSummaryMap"));
  if ( me && meGlobalSummary_) {

    int nValidChannelsTT[72][34];
    int nGlobalErrorsTT[72][34];
    for ( int iettx = 0; iettx < 34; iettx++ ) {
      for ( int ipttx = 0; ipttx < 72; ipttx++ ) {
        nValidChannelsTT[ipttx][iettx] = 0;
        nGlobalErrorsTT[ipttx][iettx] = 0;
      }
    }

    for ( int iex = 1; iex <= 170; iex++ ) {
      for ( int ipx = 1; ipx <= 360; ipx++ ) {

        int iettx = (iex-1)/5+1;
        int ipttx = (ipx-1)/5+1;

        float xval = meGlobalSummary_->getBinContent( ipx, iex );

        if ( xval >= 0 && xval <= 5 ) {
          if ( xval != 2 && xval != 5 ) ++nValidChannelsTT[ipttx-1][iettx-1];
          if ( xval == 0 ) ++nGlobalErrorsTT[ipttx-1][iettx-1];
        }

      }
    }

    for ( int iettx = 0; iettx < 34; iettx++ ) {
      for ( int ipttx = 0; ipttx < 72; ipttx++ ) {

        float xval = -1.0;
        if ( nValidChannelsTT[ipttx][iettx] != 0 )
          xval = 1.0 - float(nGlobalErrorsTT[ipttx][iettx])/float(nValidChannelsTT[ipttx][iettx]);

        me->setBinContent( ipttx+1, iettx+1, xval );
	if(mecomb)
	  mecomb->setBinContent(ipttx + 1, iettx + 21, xval);

      }
    }

  }

}

