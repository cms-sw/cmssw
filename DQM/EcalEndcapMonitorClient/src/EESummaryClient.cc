/*
 * \file EESummaryClient.cc
 *
 * $Date: 2009/08/02 15:46:40 $
 * $Revision: 1.174 $
 * \author G. Della Ricca
 *
*/

#include <memory>
#include <iostream>
#include <iomanip>
#include <map>
#include <math.h>

#include <DataFormats/EcalDetId/interface/EEDetId.h>

#include "DQMServices/Core/interface/DQMStore.h"

#include "OnlineDB/EcalCondDB/interface/RunTag.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"

#include <DQM/EcalCommon/interface/UtilsClient.h>
#include <DQM/EcalCommon/interface/Numbers.h>

#include <DQM/EcalEndcapMonitorClient/interface/EECosmicClient.h>
#include <DQM/EcalEndcapMonitorClient/interface/EEStatusFlagsClient.h>
#include <DQM/EcalEndcapMonitorClient/interface/EEIntegrityClient.h>
#include <DQM/EcalEndcapMonitorClient/interface/EELaserClient.h>
#include <DQM/EcalEndcapMonitorClient/interface/EELedClient.h>
#include <DQM/EcalEndcapMonitorClient/interface/EEPedestalClient.h>
#include <DQM/EcalEndcapMonitorClient/interface/EEPedestalOnlineClient.h>
#include <DQM/EcalEndcapMonitorClient/interface/EETestPulseClient.h>
#include <DQM/EcalEndcapMonitorClient/interface/EEBeamCaloClient.h>
#include <DQM/EcalEndcapMonitorClient/interface/EEBeamHodoClient.h>
#include <DQM/EcalEndcapMonitorClient/interface/EETriggerTowerClient.h>
#include <DQM/EcalEndcapMonitorClient/interface/EEClusterClient.h>
#include <DQM/EcalEndcapMonitorClient/interface/EETimingClient.h>

#include <DQM/EcalEndcapMonitorClient/interface/EESummaryClient.h>

using namespace cms;
using namespace edm;
using namespace std;

EESummaryClient::EESummaryClient(const ParameterSet& ps) {

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

  laserWavelengths_.reserve(4);
  for ( unsigned int i = 1; i <= 4; i++ ) laserWavelengths_.push_back(i);
  laserWavelengths_ = ps.getUntrackedParameter<vector<int> >("laserWavelengths", laserWavelengths_);

  ledWavelengths_.reserve(2);
  for ( unsigned int i = 1; i <= 2; i++ ) ledWavelengths_.push_back(i);
  ledWavelengths_ = ps.getUntrackedParameter<vector<int> >("ledWavelengths", ledWavelengths_);

  MGPAGains_.reserve(3);
  for ( unsigned int i = 1; i <= 3; i++ ) MGPAGains_.push_back(i);
  MGPAGains_ = ps.getUntrackedParameter<vector<int> >("MGPAGains", MGPAGains_);

  MGPAGainsPN_.reserve(2);
  for ( unsigned int i = 1; i <= 3; i++ ) MGPAGainsPN_.push_back(i);
  MGPAGainsPN_ = ps.getUntrackedParameter<vector<int> >("MGPAGainsPN", MGPAGainsPN_);

  // summary maps
  meIntegrity_[0]      = 0;
  meIntegrity_[1]      = 0;
  meOccupancy_[0]      = 0;
  meOccupancy_[1]      = 0;
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
  meLaserL1PN_[0]      = 0;
  meLaserL1PN_[1]      = 0;
  meLaserL1Ampl_ = 0;
  meLaserL1Timing_  = 0;
  meLaserL1AmplOverPN_ = 0;

  meLaserL2_[0]        = 0;
  meLaserL2_[1]        = 0;
  meLaserL2PN_[0]      = 0;
  meLaserL2PN_[1]      = 0;
  meLaserL2Ampl_ = 0;
  meLaserL2Timing_  = 0;
  meLaserL2AmplOverPN_ = 0;

  meLaserL3_[0]        = 0;
  meLaserL3_[1]        = 0;
  meLaserL3PN_[0]      = 0;
  meLaserL3PN_[1]      = 0;
  meLaserL3Ampl_ = 0;
  meLaserL3Timing_  = 0;
  meLaserL3AmplOverPN_ = 0;

  meLaserL4_[0]        = 0;
  meLaserL4_[1]        = 0;
  meLaserL4PN_[0]      = 0;
  meLaserL4PN_[1]      = 0;
  meLaserL4Ampl_ = 0;
  meLaserL4Timing_  = 0;
  meLaserL4AmplOverPN_ = 0;

  meLedL1_[0]          = 0;
  meLedL1_[1]          = 0;
  meLedL1PN_[0]        = 0;
  meLedL1PN_[1]        = 0;
  meLedL1Ampl_         = 0;
  meLedL1Timing_       = 0;
  meLedL1AmplOverPN_   = 0;

  meLedL2_[0]          = 0;
  meLedL2_[1]          = 0;
  meLedL2PN_[0]        = 0;
  meLedL2PN_[1]        = 0;
  meLedL2Ampl_         = 0;
  meLedL2Timing_       = 0;
  meLedL2AmplOverPN_   = 0;

  mePedestalG01_[0]       = 0;
  mePedestalG01_[1]       = 0;
  mePedestalG06_[0]       = 0;
  mePedestalG06_[1]       = 0;
  mePedestalG12_[0]       = 0;
  mePedestalG12_[1]       = 0;
  mePedestalPNG01_[0]     = 0;
  mePedestalPNG01_[1]     = 0;
  mePedestalPNG16_[0]     = 0;
  mePedestalPNG16_[1]     = 0;
  meTestPulseG01_[0]      = 0;
  meTestPulseG01_[1]      = 0;
  meTestPulseG06_[0]      = 0;
  meTestPulseG06_[1]      = 0;
  meTestPulseG12_[0]      = 0;
  meTestPulseG12_[1]      = 0;
  meTestPulsePNG01_[0]    = 0;
  meTestPulsePNG01_[1]    = 0;
  meTestPulsePNG16_[0]    = 0;
  meTestPulsePNG16_[1]    = 0;
  meTestPulseAmplG01_ = 0;
  meTestPulseAmplG06_ = 0;
  meTestPulseAmplG12_ = 0;
  meGlobalSummary_[0]  = 0;
  meGlobalSummary_[1]  = 0;

  meCosmic_[0]         = 0;
  meCosmic_[1]         = 0;
  meTiming_[0]         = 0;
  meTiming_[1]         = 0;
  meTriggerTowerEt_[0]        = 0;
  meTriggerTowerEt_[1]        = 0;
  meTriggerTowerEtSpectrum_[0] = 0;
  meTriggerTowerEtSpectrum_[1] = 0;
  meTriggerTowerEmulError_[0] = 0;
  meTriggerTowerEmulError_[1] = 0;
  meTriggerTowerTiming_[0] = 0;
  meTriggerTowerTiming_[1] = 0;

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

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    hpot01_[ism-1] = 0;
    httt01_[ism-1] = 0;

  }

}

EESummaryClient::~EESummaryClient() {

}

void EESummaryClient::beginJob(DQMStore* dqmStore) {

  dqmStore_ = dqmStore;

  if ( debug_ ) cout << "EESummaryClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

  // summary for DQM GUI

  char histo[200];

  MonitorElement* me;

  dqmStore_->setCurrentFolder( prefixME_ + "/EventInfo" );

  sprintf(histo, "reportSummary");
  me = dqmStore_->get(prefixME_ + "/EventInfo/" + histo);
  if ( me ) {
    dqmStore_->removeElement(me->getName());
  }
  me = dqmStore_->bookFloat(histo);
  me->Fill(-1.0);

  dqmStore_->setCurrentFolder( prefixME_ + "/EventInfo/reportSummaryContents" );

  for (int i = 0; i < 18; i++) {
    sprintf(histo, "EcalEndcap_%s", Numbers::sEE(i+1).c_str());
    me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummaryContents/" + histo);
    if ( me ) {
      dqmStore_->removeElement(me->getName());
    }
    me = dqmStore_->bookFloat(histo);
    me->Fill(-1.0);
  }

  dqmStore_->setCurrentFolder( prefixME_ + "/EventInfo" );

  sprintf(histo, "reportSummaryMap");
  me = dqmStore_->get(prefixME_ + "/EventInfo/" + histo);
  if ( me ) {
    dqmStore_->removeElement(me->getName());
  }
  me = dqmStore_->book2D(histo, histo, 200, 0., 200., 100, 0., 100);
  for ( int jx = 1; jx <= 200; jx++ ) {
    for ( int jy = 1; jy <= 100; jy++ ) {
      me->setBinContent( jx, jy, -1.0 );
    }
  }
  me->setAxisTitle("jx", 1);
  me->setAxisTitle("jy", 2);

}

void EESummaryClient::beginRun(void) {

  if ( debug_ ) cout << "EESummaryClient: beginRun" << endl;

  jevt_ = 0;

  this->setup();

}

void EESummaryClient::endJob(void) {

  if ( debug_ ) cout << "EESummaryClient: endJob, ievt = " << ievt_ << endl;

  this->cleanup();

}

void EESummaryClient::endRun(void) {

  if ( debug_ ) cout << "EESummaryClient: endRun, jevt = " << jevt_ << endl;

  this->cleanup();

}

void EESummaryClient::setup(void) {

  char histo[200];

  dqmStore_->setCurrentFolder( prefixME_ + "/EESummaryClient" );

  if ( meIntegrity_[0] ) dqmStore_->removeElement( meIntegrity_[0]->getName() );
  sprintf(histo, "EEIT EE - integrity quality summary");
  meIntegrity_[0] = dqmStore_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
  meIntegrity_[0]->setAxisTitle("jx", 1);
  meIntegrity_[0]->setAxisTitle("jy", 2);

  if ( meIntegrity_[1] ) dqmStore_->removeElement( meIntegrity_[0]->getName() );
  sprintf(histo, "EEIT EE + integrity quality summary");
  meIntegrity_[1] = dqmStore_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
  meIntegrity_[1]->setAxisTitle("jx", 1);
  meIntegrity_[1]->setAxisTitle("jy", 2);

  if ( meIntegrityErr_ ) dqmStore_->removeElement( meIntegrityErr_->getName() );
  sprintf(histo, "EEIT integrity quality errors summary");
  meIntegrityErr_ = dqmStore_->book1D(histo, histo, 18, 1, 19);
  for (int i = 0; i < 18; i++) {
    meIntegrityErr_->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
  }

  if ( meOccupancy_[0] ) dqmStore_->removeElement( meOccupancy_[0]->getName() );
  sprintf(histo, "EEOT EE - digi occupancy summary");
  meOccupancy_[0] = dqmStore_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
  meOccupancy_[0]->setAxisTitle("jx", 1);
  meOccupancy_[0]->setAxisTitle("jy", 2);

  if ( meOccupancy_[1] ) dqmStore_->removeElement( meOccupancy_[1]->getName() );
  sprintf(histo, "EEOT EE + digi occupancy summary");
  meOccupancy_[1] = dqmStore_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
  meOccupancy_[1]->setAxisTitle("jx", 1);
  meOccupancy_[1]->setAxisTitle("jy", 2);

  if ( meOccupancy1D_ ) dqmStore_->removeElement( meOccupancy1D_->getName() );
  sprintf(histo, "EEIT digi occupancy summary 1D");
  meOccupancy1D_ = dqmStore_->book1D(histo, histo, 18, 1, 19);
  for (int i = 0; i < 18; i++) {
    meOccupancy1D_->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
  }

  if ( meStatusFlags_[0] ) dqmStore_->removeElement( meStatusFlags_[0]->getName() );
  sprintf(histo, "EESFT EE - front-end status summary");
  meStatusFlags_[0] = dqmStore_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
  meStatusFlags_[0]->setAxisTitle("jx", 1);
  meStatusFlags_[0]->setAxisTitle("jy", 2);

  if ( meStatusFlags_[1] ) dqmStore_->removeElement( meStatusFlags_[1]->getName() );
  sprintf(histo, "EESFT EE + front-end status summary");
  meStatusFlags_[1] = dqmStore_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
  meStatusFlags_[1]->setAxisTitle("jx", 1);
  meStatusFlags_[1]->setAxisTitle("jy", 2);

  if ( meStatusFlagsErr_ ) dqmStore_->removeElement( meStatusFlagsErr_->getName() );
  sprintf(histo, "EESFT front-end status errors summary");
  meStatusFlagsErr_ = dqmStore_->book1D(histo, histo, 18, 1, 19);
  for (int i = 0; i < 18; i++) {
    meStatusFlagsErr_->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
  }

  if ( mePedestalOnline_[0] ) dqmStore_->removeElement( mePedestalOnline_[0]->getName() );
  sprintf(histo, "EEPOT EE - pedestal quality summary G12");
  mePedestalOnline_[0] = dqmStore_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
  mePedestalOnline_[0]->setAxisTitle("jx", 1);
  mePedestalOnline_[0]->setAxisTitle("jy", 2);

  if ( mePedestalOnline_[1] ) dqmStore_->removeElement( mePedestalOnline_[1]->getName() );
  sprintf(histo, "EEPOT EE + pedestal quality summary G12");
  mePedestalOnline_[1] = dqmStore_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
  mePedestalOnline_[1]->setAxisTitle("jx", 1);
  mePedestalOnline_[1]->setAxisTitle("jy", 2);

  if ( mePedestalOnlineRMSMap_[0] ) dqmStore_->removeElement( mePedestalOnlineRMSMap_[0]->getName() );
  sprintf(histo, "EEPOT EE - pedestal G12 RMS map");
  mePedestalOnlineRMSMap_[0] = dqmStore_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
  mePedestalOnlineRMSMap_[0]->setAxisTitle("jx", 1);
  mePedestalOnlineRMSMap_[0]->setAxisTitle("jy", 2);

  if ( mePedestalOnlineRMSMap_[1] ) dqmStore_->removeElement( mePedestalOnlineRMSMap_[1]->getName() );
  sprintf(histo, "EEPOT EE + pedestal G12 RMS map");
  mePedestalOnlineRMSMap_[1] = dqmStore_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
  mePedestalOnlineRMSMap_[1]->setAxisTitle("jx", 1);
  mePedestalOnlineRMSMap_[1]->setAxisTitle("jy", 2);

  if ( mePedestalOnlineMean_ ) dqmStore_->removeElement( mePedestalOnlineMean_->getName() );
  sprintf(histo, "EEPOT pedestal G12 mean");
  mePedestalOnlineMean_ = dqmStore_->bookProfile(histo, histo, 18, 1, 19, 100, 150., 250.);
  for (int i = 0; i < 18; i++) {
    mePedestalOnlineMean_->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
  }

  if ( mePedestalOnlineRMS_ ) dqmStore_->removeElement( mePedestalOnlineRMS_->getName() );
  sprintf(histo, "EEPOT pedestal G12 rms");
  mePedestalOnlineRMS_ = dqmStore_->bookProfile(histo, histo, 18, 1, 19, 100, 0., 10.);
  for (int i = 0; i < 18; i++) {
    mePedestalOnlineRMS_->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
  }

  if ( mePedestalOnlineErr_ ) dqmStore_->removeElement( mePedestalOnlineErr_->getName() );
  sprintf(histo, "EEPOT pedestal quality errors summary G12");
  mePedestalOnlineErr_ = dqmStore_->book1D(histo, histo, 18, 1, 19);
  for (int i = 0; i < 18; i++) {
    mePedestalOnlineErr_->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
  }

  if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 1) != laserWavelengths_.end() ) {

    if ( meLaserL1_[0] ) dqmStore_->removeElement( meLaserL1_[0]->getName() );
    sprintf(histo, "EELT EE - laser quality summary L1");
    meLaserL1_[0] = dqmStore_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
    meLaserL1_[0]->setAxisTitle("jx", 1);
    meLaserL1_[0]->setAxisTitle("jy", 2);

    if ( meLaserL1PN_[0] ) dqmStore_->removeElement( meLaserL1PN_[0]->getName() );
    sprintf(histo, "EELT EE - PN laser quality summary L1");
    meLaserL1PN_[0] = dqmStore_->book2D(histo, histo, 90, 0., 90., 20, -10., 10.);
    meLaserL1PN_[0]->setAxisTitle("jx", 1);
    meLaserL1PN_[0]->setAxisTitle("jy", 2);

    if ( meLaserL1_[1] ) dqmStore_->removeElement( meLaserL1_[1]->getName() );
    sprintf(histo, "EELT EE + laser quality summary L1");
    meLaserL1_[1] = dqmStore_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
    meLaserL1_[1]->setAxisTitle("jx", 1);
    meLaserL1_[1]->setAxisTitle("jy", 2);

    if ( meLaserL1PN_[1] ) dqmStore_->removeElement( meLaserL1PN_[1]->getName() );
    sprintf(histo, "EELT EE + PN laser quality summary L1");
    meLaserL1PN_[1] = dqmStore_->book2D(histo, histo, 90, 0., 90., 20, -10., 10.);
    meLaserL1PN_[1]->setAxisTitle("jx", 1);
    meLaserL1PN_[1]->setAxisTitle("jy", 2);

    if ( meLaserL1Err_ ) dqmStore_->removeElement( meLaserL1Err_->getName() );
    sprintf(histo, "EELT laser quality errors summary L1");
    meLaserL1Err_ = dqmStore_->book1D(histo, histo, 18, 1, 19);
    for (int i = 0; i < 18; i++) {
      meLaserL1Err_->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
    }

    if ( meLaserL1PNErr_ ) dqmStore_->removeElement( meLaserL1PNErr_->getName() );
    sprintf(histo, "EELT PN laser quality errors summary L1");
    meLaserL1PNErr_ = dqmStore_->book1D(histo, histo, 18, 1, 19);
    for (int i = 0; i < 18; i++) {
      meLaserL1PNErr_->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
    }

    if ( meLaserL1Ampl_ ) dqmStore_->removeElement( meLaserL1Ampl_->getName() );
    sprintf(histo, "EELT laser L1 amplitude summary");
    meLaserL1Ampl_ = dqmStore_->bookProfile(histo, histo, 18, 1, 19, 100, 0., 2000., "s");
    for (int i = 0; i < 18; i++) {
      meLaserL1Ampl_->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
    }

    if ( meLaserL1Timing_ ) dqmStore_->removeElement( meLaserL1Timing_->getName() );
    sprintf(histo, "EELT laser L1 timing summary");
    meLaserL1Timing_ = dqmStore_->bookProfile(histo, histo, 18, 1, 19, 100, 0., 10., "s");
    for (int i = 0; i < 18; i++) {
      meLaserL1Timing_->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
    }
  
    if ( meLaserL1AmplOverPN_ ) dqmStore_->removeElement( meLaserL1AmplOverPN_->getName() );
    sprintf(histo, "EELT laser L1 amplitude over PN summary");
    meLaserL1AmplOverPN_ = dqmStore_->bookProfile(histo, histo, 18, 1, 19, 100, 0., 20., "s");
    for (int i = 0; i < 18; i++) {
      meLaserL1AmplOverPN_->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
    }

  }

  if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 2) != laserWavelengths_.end() ) {

    if ( meLaserL2_[0] ) dqmStore_->removeElement( meLaserL2_[0]->getName() );
    sprintf(histo, "EELT EE - laser quality summary L2");
    meLaserL2_[0] = dqmStore_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
    meLaserL2_[0]->setAxisTitle("jx", 1);
    meLaserL2_[0]->setAxisTitle("jy", 2);

    if ( meLaserL2PN_[0] ) dqmStore_->removeElement( meLaserL2PN_[0]->getName() );
    sprintf(histo, "EELT EE - PN laser quality summary L2");
    meLaserL2PN_[0] = dqmStore_->book2D(histo, histo, 90, 0., 90., 20, -10., 10.);
    meLaserL2PN_[0]->setAxisTitle("jx", 1);
    meLaserL2PN_[0]->setAxisTitle("jy", 2);

    if ( meLaserL2_[1] ) dqmStore_->removeElement( meLaserL2_[1]->getName() );
    sprintf(histo, "EELT EE + laser quality summary L2");
    meLaserL2_[1] = dqmStore_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
    meLaserL2_[1]->setAxisTitle("jx", 1);
    meLaserL2_[1]->setAxisTitle("jy", 2);

    if ( meLaserL2PN_[1] ) dqmStore_->removeElement( meLaserL2PN_[1]->getName() );
    sprintf(histo, "EELT EE + PN laser quality summary L2");
    meLaserL2PN_[1] = dqmStore_->book2D(histo, histo, 90, 0., 90., 20, -10., 10.);
    meLaserL2PN_[1]->setAxisTitle("jx", 1);
    meLaserL2PN_[1]->setAxisTitle("jy", 2);

    if ( meLaserL2Err_ ) dqmStore_->removeElement( meLaserL2Err_->getName() );
    sprintf(histo, "EELT laser quality errors summary L2");
    meLaserL2Err_ = dqmStore_->book1D(histo, histo, 18, 1, 19);
    for (int i = 0; i < 18; i++) {
      meLaserL2Err_->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
    }

    if ( meLaserL2PNErr_ ) dqmStore_->removeElement( meLaserL2PNErr_->getName() );
    sprintf(histo, "EELT PN laser quality errors summary L2");
    meLaserL2PNErr_ = dqmStore_->book1D(histo, histo, 18, 1, 19);
    for (int i = 0; i < 18; i++) {
      meLaserL2PNErr_->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
    }

    if ( meLaserL2Ampl_ ) dqmStore_->removeElement( meLaserL2Ampl_->getName() );
    sprintf(histo, "EELT laser L2 amplitude summary");
    meLaserL2Ampl_ = dqmStore_->bookProfile(histo, histo, 18, 1, 19, 100, 0., 2000., "s");
    for (int i = 0; i < 18; i++) {
      meLaserL2Ampl_->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
    }

    if ( meLaserL2Timing_ ) dqmStore_->removeElement( meLaserL2Timing_->getName() );
    sprintf(histo, "EELT laser L2 timing summary");
    meLaserL2Timing_ = dqmStore_->bookProfile(histo, histo, 18, 1, 19, 100, 0., 10., "s");
    for (int i = 0; i < 18; i++) {
      meLaserL2Timing_->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
    }
  
    if ( meLaserL2AmplOverPN_ ) dqmStore_->removeElement( meLaserL2AmplOverPN_->getName() );
    sprintf(histo, "EELT laser L2 amplitude over PN summary");
    meLaserL2AmplOverPN_ = dqmStore_->bookProfile(histo, histo, 18, 1, 19, 100, 0., 20., "s");
    for (int i = 0; i < 18; i++) {
      meLaserL2AmplOverPN_->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
    }

  }

  if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 3) != laserWavelengths_.end() ) {

    if ( meLaserL3_[0] ) dqmStore_->removeElement( meLaserL3_[0]->getName() );
    sprintf(histo, "EELT EE - laser quality summary L3");
    meLaserL3_[0] = dqmStore_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
    meLaserL3_[0]->setAxisTitle("jx", 1);
    meLaserL3_[0]->setAxisTitle("jy", 2);

    if ( meLaserL3PN_[0] ) dqmStore_->removeElement( meLaserL3PN_[0]->getName() );
    sprintf(histo, "EELT EE - PN laser quality summary L3");
    meLaserL3PN_[0] = dqmStore_->book2D(histo, histo, 90, 0., 90., 20, -10., 10.);
    meLaserL3PN_[0]->setAxisTitle("jx", 1);
    meLaserL3PN_[0]->setAxisTitle("jy", 2);

    if ( meLaserL3_[1] ) dqmStore_->removeElement( meLaserL3_[1]->getName() );
    sprintf(histo, "EELT EE + laser quality summary L3");
    meLaserL3_[1] = dqmStore_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
    meLaserL3_[1]->setAxisTitle("jx", 1);
    meLaserL3_[1]->setAxisTitle("jy", 2);

    if ( meLaserL3PN_[1] ) dqmStore_->removeElement( meLaserL3PN_[1]->getName() );
    sprintf(histo, "EELT EE + PN laser quality summary L3");
    meLaserL3PN_[1] = dqmStore_->book2D(histo, histo, 90, 0., 90., 20, -10., 10.);
    meLaserL3PN_[1]->setAxisTitle("jx", 1);
    meLaserL3PN_[1]->setAxisTitle("jy", 2);

    if ( meLaserL3Err_ ) dqmStore_->removeElement( meLaserL3Err_->getName() );
    sprintf(histo, "EELT laser quality errors summary L3");
    meLaserL3Err_ = dqmStore_->book1D(histo, histo, 18, 1, 19);
    for (int i = 0; i < 18; i++) {
      meLaserL3Err_->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
    }

    if ( meLaserL3PNErr_ ) dqmStore_->removeElement( meLaserL3PNErr_->getName() );
    sprintf(histo, "EELT PN laser quality errors summary L3");
    meLaserL3PNErr_ = dqmStore_->book1D(histo, histo, 18, 1, 19);
    for (int i = 0; i < 18; i++) {
      meLaserL3PNErr_->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
    }

    if ( meLaserL3Ampl_ ) dqmStore_->removeElement( meLaserL3Ampl_->getName() );
    sprintf(histo, "EELT laser L3 amplitude summary");
    meLaserL3Ampl_ = dqmStore_->bookProfile(histo, histo, 18, 1, 19, 100, 0., 2000., "s");
    for (int i = 0; i < 18; i++) {
      meLaserL3Ampl_->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
    }

    if ( meLaserL3Timing_ ) dqmStore_->removeElement( meLaserL3Timing_->getName() );
    sprintf(histo, "EELT laser L3 timing summary");
    meLaserL3Timing_ = dqmStore_->bookProfile(histo, histo, 18, 1, 19, 100, 0., 10., "s");
    for (int i = 0; i < 18; i++) {
      meLaserL3Timing_->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
    }
  
    if ( meLaserL3AmplOverPN_ ) dqmStore_->removeElement( meLaserL3AmplOverPN_->getName() );
    sprintf(histo, "EELT laser L3 amplitude over PN summary");
    meLaserL3AmplOverPN_ = dqmStore_->bookProfile(histo, histo, 18, 1, 19, 100, 0., 20., "s");
    for (int i = 0; i < 18; i++) {
      meLaserL3AmplOverPN_->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
    }

  }

  if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 4) != laserWavelengths_.end() ) {

    if ( meLaserL4_[0] ) dqmStore_->removeElement( meLaserL4_[0]->getName() );
    sprintf(histo, "EELT EE - laser quality summary L4");
    meLaserL4_[0] = dqmStore_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
    meLaserL4_[0]->setAxisTitle("jx", 1);
    meLaserL4_[0]->setAxisTitle("jy", 2);

    if ( meLaserL4PN_[0] ) dqmStore_->removeElement( meLaserL4PN_[0]->getName() );
    sprintf(histo, "EELT EE - PN laser quality summary L4");
    meLaserL4PN_[0] = dqmStore_->book2D(histo, histo, 90, 0., 90., 20, -10., 10.);
    meLaserL4PN_[0]->setAxisTitle("jx", 1);
    meLaserL4PN_[0]->setAxisTitle("jy", 2);

    if ( meLaserL4_[1] ) dqmStore_->removeElement( meLaserL4_[1]->getName() );
    sprintf(histo, "EELT EE + laser quality summary L4");
    meLaserL4_[1] = dqmStore_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
    meLaserL4_[1]->setAxisTitle("jx", 1);
    meLaserL4_[1]->setAxisTitle("jy", 2);

    if ( meLaserL4PN_[1] ) dqmStore_->removeElement( meLaserL4PN_[1]->getName() );
    sprintf(histo, "EELT EE + PN laser quality summary L4");
    meLaserL4PN_[1] = dqmStore_->book2D(histo, histo, 90, 0., 90., 20, -10., 10.);
    meLaserL4PN_[1]->setAxisTitle("jx", 1);
    meLaserL4PN_[1]->setAxisTitle("jy", 2);

    if ( meLaserL4Err_ ) dqmStore_->removeElement( meLaserL4Err_->getName() );
    sprintf(histo, "EELT laser quality errors summary L4");
    meLaserL4Err_ = dqmStore_->book1D(histo, histo, 18, 1, 19);
    for (int i = 0; i < 18; i++) {
      meLaserL4Err_->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
    }

    if ( meLaserL4PNErr_ ) dqmStore_->removeElement( meLaserL4PNErr_->getName() );
    sprintf(histo, "EELT PN laser quality errors summary L4");
    meLaserL4PNErr_ = dqmStore_->book1D(histo, histo, 18, 1, 19);
    for (int i = 0; i < 18; i++) {
      meLaserL4PNErr_->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
    }

    if ( meLaserL4Ampl_ ) dqmStore_->removeElement( meLaserL4Ampl_->getName() );
    sprintf(histo, "EELT laser L4 amplitude summary");
    meLaserL4Ampl_ = dqmStore_->bookProfile(histo, histo, 18, 1, 19, 100, 0., 2000., "s");
    for (int i = 0; i < 18; i++) {
      meLaserL4Ampl_->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
    }

    if ( meLaserL4Timing_ ) dqmStore_->removeElement( meLaserL4Timing_->getName() );
    sprintf(histo, "EELT laser L4 timing summary");
    meLaserL4Timing_ = dqmStore_->bookProfile(histo, histo, 18, 1, 19, 100, 0., 10., "s");
    for (int i = 0; i < 18; i++) {
      meLaserL4Timing_->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
    }
  
    if ( meLaserL4AmplOverPN_ ) dqmStore_->removeElement( meLaserL4AmplOverPN_->getName() );
    sprintf(histo, "EELT laser L4 amplitude over PN summary");
    meLaserL4AmplOverPN_ = dqmStore_->bookProfile(histo, histo, 18, 1, 19, 100, 0., 20., "s");
    for (int i = 0; i < 18; i++) {
      meLaserL4AmplOverPN_->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
    }

  }

  if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 1) != ledWavelengths_.end() ) {

    if ( meLedL1_[0] ) dqmStore_->removeElement( meLedL1_[0]->getName() );
    sprintf(histo, "EELDT EE - led quality summary L1");
    meLedL1_[0] = dqmStore_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
    meLedL1_[0]->setAxisTitle("jx", 1);
    meLedL1_[0]->setAxisTitle("jy", 2);

    if ( meLedL1PN_[0] ) dqmStore_->removeElement( meLedL1PN_[0]->getName() );
    sprintf(histo, "EELDT EE - PN led quality summary L1");
    meLedL1PN_[0] = dqmStore_->book2D(histo, histo, 90, 0., 90., 20, -10., 10.);
    meLedL1PN_[0]->setAxisTitle("jx", 1);
    meLedL1PN_[0]->setAxisTitle("jy", 2);

    if ( meLedL1_[1] ) dqmStore_->removeElement( meLedL1_[1]->getName() );
    sprintf(histo, "EELDT EE + led quality summary L1");
    meLedL1_[1] = dqmStore_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
    meLedL1_[1]->setAxisTitle("jx", 1);
    meLedL1_[1]->setAxisTitle("jy", 2);

    if ( meLedL1PN_[1] ) dqmStore_->removeElement( meLedL1PN_[1]->getName() );
    sprintf(histo, "EELDT EE + PN led quality summary L1");
    meLedL1PN_[1] = dqmStore_->book2D(histo, histo, 90, 0., 90., 20, -10., 10.);
    meLedL1PN_[1]->setAxisTitle("jx", 1);
    meLedL1PN_[1]->setAxisTitle("jy", 2);

    if ( meLedL1Err_ ) dqmStore_->removeElement( meLedL1Err_->getName() );
    sprintf(histo, "EELDT led quality errors summary L1");
    meLedL1Err_ = dqmStore_->book1D(histo, histo, 18, 1, 19);
    for (int i = 0; i < 18; i++) {
      meLedL1Err_->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
    }

    if ( meLedL1PNErr_ ) dqmStore_->removeElement( meLedL1PNErr_->getName() );
    sprintf(histo, "EELDT PN led quality errors summary L1");
    meLedL1PNErr_ = dqmStore_->book1D(histo, histo, 18, 1, 19);
    for (int i = 0; i < 18; i++) {
      meLedL1PNErr_->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
    }

    if ( meLedL1Ampl_ ) dqmStore_->removeElement( meLedL1Ampl_->getName() );
    sprintf(histo, "EELDT led L1 amplitude summary");
    meLedL1Ampl_ = dqmStore_->bookProfile(histo, histo, 18, 1, 19, 100, 0., 2000., "s");
    for (int i = 0; i < 18; i++) {
      meLedL1Ampl_->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
    }

    if ( meLedL1Timing_ ) dqmStore_->removeElement( meLedL1Timing_->getName() );
    sprintf(histo, "EELDT led L1 timing summary");
    meLedL1Timing_ = dqmStore_->bookProfile(histo, histo, 18, 1, 19, 100, 0., 10., "s");
    for (int i = 0; i < 18; i++) {
      meLedL1Timing_->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
    }
  
    if ( meLedL1AmplOverPN_ ) dqmStore_->removeElement( meLedL1AmplOverPN_->getName() );
    sprintf(histo, "EELDT led L1 amplitude over PN summary");
    meLedL1AmplOverPN_ = dqmStore_->bookProfile(histo, histo, 18, 1, 19, 100, 0., 20., "s");
    for (int i = 0; i < 18; i++) {
      meLedL1AmplOverPN_->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
    }

  }

  if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 2) != ledWavelengths_.end() ) {

    if ( meLedL2_[0] ) dqmStore_->removeElement( meLedL2_[0]->getName() );
    sprintf(histo, "EELDT EE - led quality summary L2");
    meLedL2_[0] = dqmStore_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
    meLedL2_[0]->setAxisTitle("jx", 1);
    meLedL2_[0]->setAxisTitle("jy", 2);

    if ( meLedL2PN_[0] ) dqmStore_->removeElement( meLedL2PN_[0]->getName() );
    sprintf(histo, "EELDT EE - PN led quality summary L2");
    meLedL2PN_[0] = dqmStore_->book2D(histo, histo, 90, 0., 90., 20, -10., 10.);
    meLedL2PN_[0]->setAxisTitle("jx", 1);
    meLedL2PN_[0]->setAxisTitle("jy", 2);

    if ( meLedL2_[1] ) dqmStore_->removeElement( meLedL2_[1]->getName() );
    sprintf(histo, "EELDT EE + led quality summary L2");
    meLedL2_[1] = dqmStore_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
    meLedL2_[1]->setAxisTitle("jx", 1);
    meLedL2_[1]->setAxisTitle("jy", 2);

    if ( meLedL2PN_[1] ) dqmStore_->removeElement( meLedL2PN_[1]->getName() );
    sprintf(histo, "EELDT EE + PN led quality summary L2");
    meLedL2PN_[1] = dqmStore_->book2D(histo, histo, 90, 0., 90., 20, -10., 10.);
    meLedL2PN_[1]->setAxisTitle("jx", 1);
    meLedL2PN_[1]->setAxisTitle("jy", 2);

    if ( meLedL2Err_ ) dqmStore_->removeElement( meLedL2Err_->getName() );
    sprintf(histo, "EELDT led quality errors summary L2");
    meLedL2Err_ = dqmStore_->book1D(histo, histo, 18, 1, 19);
    for (int i = 0; i < 18; i++) {
      meLedL2Err_->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
    }

    if ( meLedL2PNErr_ ) dqmStore_->removeElement( meLedL2PNErr_->getName() );
    sprintf(histo, "EELDT PN led quality errors summary L2");
    meLedL2PNErr_ = dqmStore_->book1D(histo, histo, 18, 1, 19);
    for (int i = 0; i < 18; i++) {
      meLedL2PNErr_->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
    }

    if ( meLedL2Ampl_ ) dqmStore_->removeElement( meLedL2Ampl_->getName() );
    sprintf(histo, "EELDT led L2 amplitude summary");
    meLedL2Ampl_ = dqmStore_->bookProfile(histo, histo, 18, 1, 19, 100, 0., 2000., "s");
    for (int i = 0; i < 18; i++) {
      meLedL2Ampl_->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
    }

    if ( meLedL2Timing_ ) dqmStore_->removeElement( meLedL2Timing_->getName() );
    sprintf(histo, "EELDT led L2 timing summary");
    meLedL2Timing_ = dqmStore_->bookProfile(histo, histo, 18, 1, 19, 100, 0., 10., "s");
    for (int i = 0; i < 18; i++) {
      meLedL2Timing_->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
    }
  
    if ( meLedL2AmplOverPN_ ) dqmStore_->removeElement( meLedL2AmplOverPN_->getName() );
    sprintf(histo, "EELDT led L2 amplitude over PN summary");
    meLedL2AmplOverPN_ = dqmStore_->bookProfile(histo, histo, 18, 1, 19, 100, 0., 20., "s");
    for (int i = 0; i < 18; i++) {
      meLedL2AmplOverPN_->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
    }

  }

  if (find(MGPAGains_.begin(), MGPAGains_.end(), 1) != MGPAGains_.end() ) {

    if( mePedestalG01_[0] ) dqmStore_->removeElement( mePedestalG01_[0]->getName() );
    sprintf(histo, "EEPT EE - pedestal quality G01 summary");
    mePedestalG01_[0] = dqmStore_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
    mePedestalG01_[0]->setAxisTitle("jx", 1);
    mePedestalG01_[0]->setAxisTitle("jy", 2);

  }

  if (find(MGPAGains_.begin(), MGPAGains_.end(), 6) != MGPAGains_.end() ) {

    if( mePedestalG06_[0] ) dqmStore_->removeElement( mePedestalG06_[0]->getName() );
    sprintf(histo, "EEPT EE - pedestal quality G06 summary");
    mePedestalG06_[0] = dqmStore_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
    mePedestalG06_[0]->setAxisTitle("jx", 1);
    mePedestalG06_[0]->setAxisTitle("jy", 2);

  }

  if (find(MGPAGains_.begin(), MGPAGains_.end(), 12) != MGPAGains_.end() ) {

    if( mePedestalG12_[0] ) dqmStore_->removeElement( mePedestalG12_[0]->getName() );
    sprintf(histo, "EEPT EE - pedestal quality G12 summary");
    mePedestalG12_[0] = dqmStore_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
    mePedestalG12_[0]->setAxisTitle("jx", 1);
    mePedestalG12_[0]->setAxisTitle("jy", 2);

  }


  if (find(MGPAGainsPN_.begin(), MGPAGainsPN_.end(), 1) != MGPAGainsPN_.end() ) {

    if( mePedestalPNG01_[0] ) dqmStore_->removeElement( mePedestalPNG01_[0]->getName() );
    sprintf(histo, "EEPT EE - PN pedestal quality G01 summary");
    mePedestalPNG01_[0] = dqmStore_->book2D(histo, histo, 90, 0., 90., 20, -10, 10.);
    mePedestalPNG01_[0]->setAxisTitle("jx", 1);
    mePedestalPNG01_[0]->setAxisTitle("jy", 2);

  }

  if (find(MGPAGainsPN_.begin(), MGPAGainsPN_.end(), 16) != MGPAGainsPN_.end() ) {

    if( mePedestalPNG16_[0] ) dqmStore_->removeElement( mePedestalPNG16_[0]->getName() );
    sprintf(histo, "EEPT EE - PN pedestal quality G16 summary");
    mePedestalPNG16_[0] = dqmStore_->book2D(histo, histo, 90, 0., 90., 20, -10, 10.);
    mePedestalPNG16_[0]->setAxisTitle("jx", 1);
    mePedestalPNG16_[0]->setAxisTitle("jy", 2);

  }

  if (find(MGPAGains_.begin(), MGPAGains_.end(), 1) != MGPAGains_.end() ) {

    if( mePedestalG01_[1] ) dqmStore_->removeElement( mePedestalG01_[1]->getName() );
    sprintf(histo, "EEPT EE + pedestal quality G01 summary");
    mePedestalG01_[1] = dqmStore_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
    mePedestalG01_[1]->setAxisTitle("jx", 1);
    mePedestalG01_[1]->setAxisTitle("jy", 2);

  }


  if (find(MGPAGains_.begin(), MGPAGains_.end(), 6) != MGPAGains_.end() ) {

    if( mePedestalG06_[1] ) dqmStore_->removeElement( mePedestalG06_[1]->getName() );
    sprintf(histo, "EEPT EE + pedestal quality G06 summary");
    mePedestalG06_[1] = dqmStore_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
    mePedestalG06_[1]->setAxisTitle("jx", 1);
    mePedestalG06_[1]->setAxisTitle("jy", 2);

  }

  if (find(MGPAGains_.begin(), MGPAGains_.end(), 12) != MGPAGains_.end() ) {

    if( mePedestalG12_[1] ) dqmStore_->removeElement( mePedestalG12_[1]->getName() );
    sprintf(histo, "EEPT EE + pedestal quality G12 summary");
    mePedestalG12_[1] = dqmStore_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
    mePedestalG12_[1]->setAxisTitle("jx", 1);
    mePedestalG12_[1]->setAxisTitle("jy", 2);

  }

  if (find(MGPAGainsPN_.begin(), MGPAGainsPN_.end(), 1) != MGPAGainsPN_.end() ) {

    if( mePedestalPNG01_[1] ) dqmStore_->removeElement( mePedestalPNG01_[1]->getName() );
    sprintf(histo, "EEPT EE + PN pedestal quality G01 summary");
    mePedestalPNG01_[1] = dqmStore_->book2D(histo, histo, 90, 0., 90., 20, -10, 10.);
    mePedestalPNG01_[1]->setAxisTitle("jx", 1);
    mePedestalPNG01_[1]->setAxisTitle("jy", 2);

  }

  if (find(MGPAGainsPN_.begin(), MGPAGainsPN_.end(), 16) != MGPAGainsPN_.end() ) {

    if( mePedestalPNG16_[1] ) dqmStore_->removeElement( mePedestalPNG16_[1]->getName() );
    sprintf(histo, "EEPT EE + PN pedestal quality G16 summary");
    mePedestalPNG16_[1] = dqmStore_->book2D(histo, histo, 90, 0., 90., 20, -10, 10.);
    mePedestalPNG16_[1]->setAxisTitle("jx", 1);
    mePedestalPNG16_[1]->setAxisTitle("jy", 2);

  }

  if (find(MGPAGains_.begin(), MGPAGains_.end(), 1) != MGPAGains_.end() ) {

    if( meTestPulseG01_[0] ) dqmStore_->removeElement( meTestPulseG01_[0]->getName() );
    sprintf(histo, "EETPT EE - test pulse quality G01 summary");
    meTestPulseG01_[0] = dqmStore_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
    meTestPulseG01_[0]->setAxisTitle("jx", 1);
    meTestPulseG01_[0]->setAxisTitle("jy", 2);

  }

  if (find(MGPAGains_.begin(), MGPAGains_.end(), 6) != MGPAGains_.end() ) {

    if( meTestPulseG06_[0] ) dqmStore_->removeElement( meTestPulseG06_[0]->getName() );
    sprintf(histo, "EETPT EE - test pulse quality G06 summary");
    meTestPulseG06_[0] = dqmStore_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
    meTestPulseG06_[0]->setAxisTitle("jx", 1);
    meTestPulseG06_[0]->setAxisTitle("jy", 2);

  }

  if (find(MGPAGains_.begin(), MGPAGains_.end(), 12) != MGPAGains_.end() ) {

    if( meTestPulseG12_[0] ) dqmStore_->removeElement( meTestPulseG12_[0]->getName() );
    sprintf(histo, "EETPT EE - test pulse quality G12 summary");
    meTestPulseG12_[0] = dqmStore_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
    meTestPulseG12_[0]->setAxisTitle("jx", 1);
    meTestPulseG12_[0]->setAxisTitle("jy", 2);

  }


  if (find(MGPAGainsPN_.begin(), MGPAGainsPN_.end(), 1) != MGPAGainsPN_.end() ) {

    if( meTestPulsePNG01_[0] ) dqmStore_->removeElement( meTestPulsePNG01_[0]->getName() );
    sprintf(histo, "EETPT EE - PN test pulse quality G01 summary");
    meTestPulsePNG01_[0] = dqmStore_->book2D(histo, histo, 90, 0., 90., 20, -10., 10.);
    meTestPulsePNG01_[0]->setAxisTitle("jx", 1);
    meTestPulsePNG01_[0]->setAxisTitle("jy", 2);

  }

  if (find(MGPAGainsPN_.begin(), MGPAGainsPN_.end(), 16) != MGPAGainsPN_.end() ) {

    if( meTestPulsePNG16_[0] ) dqmStore_->removeElement( meTestPulsePNG16_[0]->getName() );
    sprintf(histo, "EETPT EE - PN test pulse quality G16 summary");
    meTestPulsePNG16_[0] = dqmStore_->book2D(histo, histo, 90, 0., 90., 20, -10., 10.);
    meTestPulsePNG16_[0]->setAxisTitle("jx", 1);
    meTestPulsePNG16_[0]->setAxisTitle("jy", 2);

  }

  if (find(MGPAGains_.begin(), MGPAGains_.end(), 1) != MGPAGains_.end() ) {

    if( meTestPulseG01_[1] ) dqmStore_->removeElement( meTestPulseG01_[1]->getName() );
    sprintf(histo, "EETPT EE + test pulse quality G01 summary");
    meTestPulseG01_[1] = dqmStore_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
    meTestPulseG01_[1]->setAxisTitle("jx", 1);
    meTestPulseG01_[1]->setAxisTitle("jy", 2);

  }

  if (find(MGPAGains_.begin(), MGPAGains_.end(), 6) != MGPAGains_.end() ) {

    if( meTestPulseG06_[1] ) dqmStore_->removeElement( meTestPulseG06_[1]->getName() );
    sprintf(histo, "EETPT EE + test pulse quality G06 summary");
    meTestPulseG06_[1] = dqmStore_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
    meTestPulseG06_[1]->setAxisTitle("jx", 1);
    meTestPulseG06_[1]->setAxisTitle("jy", 2);

  }

  if (find(MGPAGains_.begin(), MGPAGains_.end(), 12) != MGPAGains_.end() ) {

    if( meTestPulseG12_[1] ) dqmStore_->removeElement( meTestPulseG12_[1]->getName() );
    sprintf(histo, "EETPT EE + test pulse quality G12 summary");
    meTestPulseG12_[1] = dqmStore_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
    meTestPulseG12_[1]->setAxisTitle("jx", 1);
    meTestPulseG12_[1]->setAxisTitle("jy", 2);

  }

  if (find(MGPAGainsPN_.begin(), MGPAGainsPN_.end(), 1) != MGPAGainsPN_.end() ) {

    if( meTestPulsePNG01_[1] ) dqmStore_->removeElement( meTestPulsePNG01_[1]->getName() );
    sprintf(histo, "EETPT EE + PN test pulse quality G01 summary");
    meTestPulsePNG01_[1] = dqmStore_->book2D(histo, histo, 90, 0., 90., 20, -10., 10.);
    meTestPulsePNG01_[1]->setAxisTitle("jx", 1);
    meTestPulsePNG01_[1]->setAxisTitle("jy", 2);

  }

  if (find(MGPAGainsPN_.begin(), MGPAGainsPN_.end(), 16) != MGPAGainsPN_.end() ) {

    if( meTestPulsePNG16_[1] ) dqmStore_->removeElement( meTestPulsePNG16_[1]->getName() );
    sprintf(histo, "EETPT EE + PN test pulse quality G16 summary");
    meTestPulsePNG16_[1] = dqmStore_->book2D(histo, histo, 90, 0., 90., 20, -10., 10.);
    meTestPulsePNG16_[1]->setAxisTitle("jx", 1);
    meTestPulsePNG16_[1]->setAxisTitle("jy", 2);

  }

  if (find(MGPAGains_.begin(), MGPAGains_.end(), 1) != MGPAGains_.end() ) {

    if( meTestPulseAmplG01_ ) dqmStore_->removeElement( meTestPulseAmplG01_->getName() );
    sprintf(histo, "EETPT test pulse amplitude G01 summary");
    meTestPulseAmplG01_ = dqmStore_->bookProfile(histo, histo, 18, 1, 19, 4096, 0., 4096., "s");
    for (int i = 0; i < 18; i++) {
      meTestPulseAmplG01_->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
    }

  }

  if (find(MGPAGains_.begin(), MGPAGains_.end(), 6) != MGPAGains_.end() ) {

    if( meTestPulseAmplG06_ ) dqmStore_->removeElement( meTestPulseAmplG06_->getName() );
    sprintf(histo, "EETPT test pulse amplitude G06 summary");
    meTestPulseAmplG06_ = dqmStore_->bookProfile(histo, histo, 18, 1, 19, 4096, 0., 4096., "s");
    for (int i = 0; i < 18; i++) {
      meTestPulseAmplG06_->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
    }

  }

  if (find(MGPAGains_.begin(), MGPAGains_.end(), 12) != MGPAGains_.end() ) {

    if( meTestPulseAmplG12_ ) dqmStore_->removeElement( meTestPulseAmplG12_->getName() );
    sprintf(histo, "EETPT test pulse amplitude G12 summary");
    meTestPulseAmplG12_ = dqmStore_->bookProfile(histo, histo, 18, 1, 19, 4096, 0., 4096., "s");
    for (int i = 0; i < 18; i++) {
      meTestPulseAmplG12_->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
    }

  }

  if( meCosmic_[0] ) dqmStore_->removeElement( meCosmic_[0]->getName() );
  sprintf(histo, "EECT EE - cosmic summary");
  meCosmic_[0] = dqmStore_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
  meCosmic_[0]->setAxisTitle("jx", 1);
  meCosmic_[0]->setAxisTitle("jy", 2);

  if( meCosmic_[1] ) dqmStore_->removeElement( meCosmic_[1]->getName() );
  sprintf(histo, "EECT EE + cosmic summary");
  meCosmic_[1] = dqmStore_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
  meCosmic_[1]->setAxisTitle("jx", 1);
  meCosmic_[1]->setAxisTitle("jy", 2);

  if( meTiming_[0] ) dqmStore_->removeElement( meTiming_[0]->getName() );
  sprintf(histo, "EETMT EE - timing quality summary");
  meTiming_[0] = dqmStore_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
  meTiming_[0]->setAxisTitle("jx", 1);
  meTiming_[0]->setAxisTitle("jy", 2);

  if( meTiming_[1] ) dqmStore_->removeElement( meTiming_[1]->getName() );
  sprintf(histo, "EETMT EE + timing quality summary");
  meTiming_[1] = dqmStore_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
  meTiming_[1]->setAxisTitle("jx", 1);
  meTiming_[1]->setAxisTitle("jy", 2);

  if( meTriggerTowerEt_[0] ) dqmStore_->removeElement( meTriggerTowerEt_[0]->getName() );
  sprintf(histo, "EETTT EE - Et trigger tower summary");
  meTriggerTowerEt_[0] = dqmStore_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
  meTriggerTowerEt_[0]->setAxisTitle("jx", 1);
  meTriggerTowerEt_[0]->setAxisTitle("jy", 2);

  if( meTriggerTowerEt_[1] ) dqmStore_->removeElement( meTriggerTowerEt_[1]->getName() );
  sprintf(histo, "EETTT EE + Et trigger tower summary");
  meTriggerTowerEt_[1] = dqmStore_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
  meTriggerTowerEt_[1]->setAxisTitle("jx", 1);
  meTriggerTowerEt_[1]->setAxisTitle("jy", 2);

  if( meTriggerTowerEtSpectrum_[0] ) dqmStore_->removeElement( meTriggerTowerEtSpectrum_[0]->getName() );
  sprintf(histo, "EETTT EE - Et trigger tower spectrum");
  meTriggerTowerEtSpectrum_[0] = dqmStore_->book1D(histo, histo, 256, 1., 256.);
  meTriggerTowerEtSpectrum_[0]->setAxisTitle("transverse energy (GeV)", 1);

  if( meTriggerTowerEtSpectrum_[1] ) dqmStore_->removeElement( meTriggerTowerEtSpectrum_[1]->getName() );
  sprintf(histo, "EETTT EE + Et trigger tower spectrum");
  meTriggerTowerEtSpectrum_[1] = dqmStore_->book1D(histo, histo, 256, 1., 256.);
  meTriggerTowerEtSpectrum_[1]->setAxisTitle("transverse energy (GeV)", 1);

  if( meTriggerTowerEmulError_[0] ) dqmStore_->removeElement( meTriggerTowerEmulError_[0]->getName() );
  sprintf(histo, "EETTT EE - emulator error quality summary");
  meTriggerTowerEmulError_[0] = dqmStore_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
  meTriggerTowerEmulError_[0]->setAxisTitle("jx", 1);
  meTriggerTowerEmulError_[0]->setAxisTitle("jy", 2);

  if( meTriggerTowerEmulError_[1] ) dqmStore_->removeElement( meTriggerTowerEmulError_[1]->getName() );
  sprintf(histo, "EETTT EE + emulator error quality summary");
  meTriggerTowerEmulError_[1] = dqmStore_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
  meTriggerTowerEmulError_[1]->setAxisTitle("jx", 1);
  meTriggerTowerEmulError_[1]->setAxisTitle("jy", 2);

  if( meTriggerTowerTiming_[0] ) dqmStore_->removeElement( meTriggerTowerTiming_[0]->getName() );
  sprintf(histo, "EETTT EE - Trigger Primitives Timing summary");
  meTriggerTowerTiming_[0] = dqmStore_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
  meTriggerTowerTiming_[0]->setAxisTitle("jx", 1);
  meTriggerTowerTiming_[0]->setAxisTitle("jy", 2);

  if( meTriggerTowerTiming_[1] ) dqmStore_->removeElement( meTriggerTowerTiming_[1]->getName() );
  sprintf(histo, "EETTT EE + Trigger Primitives Timing summary");
  meTriggerTowerTiming_[1] = dqmStore_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
  meTriggerTowerTiming_[1]->setAxisTitle("jx", 1);
  meTriggerTowerTiming_[1]->setAxisTitle("jy", 2);

  if( meGlobalSummary_[0] ) dqmStore_->removeElement( meGlobalSummary_[0]->getName() );
  sprintf(histo, "EE global summary EE -");
  meGlobalSummary_[0] = dqmStore_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
  meGlobalSummary_[0]->setAxisTitle("jx", 1);
  meGlobalSummary_[0]->setAxisTitle("jy", 2);

  if( meGlobalSummary_[1] ) dqmStore_->removeElement( meGlobalSummary_[1]->getName() );
  sprintf(histo, "EE global summary EE +");
  meGlobalSummary_[1] = dqmStore_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
  meGlobalSummary_[1]->setAxisTitle("jx", 1);
  meGlobalSummary_[1]->setAxisTitle("jy", 2);

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

  if ( meOccupancy_[0] ) dqmStore_->removeElement( meOccupancy_[0]->getName() );
  meOccupancy_[0] = 0;

  if ( meOccupancy_[1] ) dqmStore_->removeElement( meOccupancy_[1]->getName() );
  meOccupancy_[1] = 0;

  if ( meOccupancy1D_ ) dqmStore_->removeElement( meOccupancy1D_->getName() );
  meOccupancy1D_ = 0;

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

  if ( meLaserL1PN_[0] ) dqmStore_->removeElement( meLaserL1PN_[0]->getName() );
  meLaserL1PN_[0] = 0;

  if ( meLaserL1PN_[1] ) dqmStore_->removeElement( meLaserL1PN_[1]->getName() );
  meLaserL1PN_[1] = 0;

  if ( meLaserL1PNErr_ ) dqmStore_->removeElement( meLaserL1PNErr_->getName() );
  meLaserL1PNErr_ = 0;

  if ( meLaserL1Ampl_ ) dqmStore_->removeElement( meLaserL1Ampl_->getName() );
  meLaserL1Ampl_ = 0;

  if ( meLaserL1Timing_ ) dqmStore_->removeElement( meLaserL1Timing_->getName() );
  meLaserL1Timing_ = 0;

  if ( meLaserL1AmplOverPN_ ) dqmStore_->removeElement( meLaserL1AmplOverPN_->getName() );
  meLaserL1AmplOverPN_ = 0;

  if ( meLedL1_[0] ) dqmStore_->removeElement( meLedL1_[0]->getName() );
  meLedL1_[0] = 0;

  if ( meLedL1_[1] ) dqmStore_->removeElement( meLedL1_[1]->getName() );
  meLedL1_[1] = 0;

  if ( meLedL1Err_ ) dqmStore_->removeElement( meLedL1Err_->getName() );
  meLedL1Err_ = 0;

  if ( meLedL1PN_[0] ) dqmStore_->removeElement( meLedL1PN_[0]->getName() );
  meLedL1PN_[0] = 0;

  if ( meLedL1PN_[1] ) dqmStore_->removeElement( meLedL1PN_[1]->getName() );
  meLedL1PN_[1] = 0;

  if ( meLedL1PNErr_ ) dqmStore_->removeElement( meLedL1PNErr_->getName() );
  meLedL1PNErr_ = 0;

  if ( meLedL1Ampl_ ) dqmStore_->removeElement( meLedL1Ampl_->getName() );
  meLedL1Ampl_ = 0;

  if ( meLedL1Timing_ ) dqmStore_->removeElement( meLedL1Timing_->getName() );
  meLedL1Timing_ = 0;

  if ( meLedL1AmplOverPN_ ) dqmStore_->removeElement( meLedL1AmplOverPN_->getName() );
  meLedL1AmplOverPN_ = 0;

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

  if ( mePedestalPNG01_[0] ) dqmStore_->removeElement( mePedestalPNG01_[0]->getName() );
  mePedestalPNG01_[0] = 0;

  if ( mePedestalPNG01_[1] ) dqmStore_->removeElement( mePedestalPNG01_[1]->getName() );
  mePedestalPNG01_[1] = 0;

  if ( mePedestalPNG16_[0] ) dqmStore_->removeElement( mePedestalPNG16_[0]->getName() );
  mePedestalPNG16_[0] = 0;

  if ( mePedestalPNG16_[1] ) dqmStore_->removeElement( mePedestalPNG16_[1]->getName() );
  mePedestalPNG16_[1] = 0;

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

  if ( meTestPulsePNG01_[0] ) dqmStore_->removeElement( meTestPulsePNG01_[0]->getName() );
  meTestPulsePNG01_[0] = 0;

  if ( meTestPulsePNG01_[1] ) dqmStore_->removeElement( meTestPulsePNG01_[1]->getName() );
  meTestPulsePNG01_[1] = 0;

  if ( meTestPulsePNG16_[0] ) dqmStore_->removeElement( meTestPulsePNG16_[0]->getName() );
  meTestPulsePNG16_[0] = 0;

  if ( meTestPulsePNG16_[1] ) dqmStore_->removeElement( meTestPulsePNG16_[1]->getName() );
  meTestPulsePNG16_[1] = 0;

  if ( meTestPulseAmplG01_ ) dqmStore_->removeElement( meTestPulseAmplG01_->getName() );
  meTestPulseAmplG01_ = 0;

  if ( meTestPulseAmplG06_ ) dqmStore_->removeElement( meTestPulseAmplG06_->getName() );
  meTestPulseAmplG06_ = 0;

  if ( meTestPulseAmplG12_ ) dqmStore_->removeElement( meTestPulseAmplG12_->getName() );
  meTestPulseAmplG12_ = 0;

  if ( meCosmic_[0] ) dqmStore_->removeElement( meCosmic_[0]->getName() );
  meCosmic_[0] = 0;

  if ( meCosmic_[1] ) dqmStore_->removeElement( meCosmic_[1]->getName() );
  meCosmic_[1] = 0;

  if ( meTiming_[0] ) dqmStore_->removeElement( meTiming_[0]->getName() );
  meTiming_[0] = 0;

  if ( meTiming_[1] ) dqmStore_->removeElement( meTiming_[1]->getName() );
  meTiming_[1] = 0;

  if ( meTriggerTowerEt_[0] ) dqmStore_->removeElement( meTriggerTowerEt_[0]->getName() );
  meTriggerTowerEt_[0] = 0;

  if ( meTriggerTowerEt_[1] ) dqmStore_->removeElement( meTriggerTowerEt_[1]->getName() );
  meTriggerTowerEt_[1] = 0;

  if ( meTriggerTowerEtSpectrum_[0] ) dqmStore_->removeElement( meTriggerTowerEtSpectrum_[0]->getName() );
  meTriggerTowerEtSpectrum_[0] = 0;

  if ( meTriggerTowerEtSpectrum_[1] ) dqmStore_->removeElement( meTriggerTowerEtSpectrum_[1]->getName() );
  meTriggerTowerEtSpectrum_[1] = 0;

  if ( meTriggerTowerEmulError_[0] ) dqmStore_->removeElement( meTriggerTowerEmulError_[0]->getName() );
  meTriggerTowerEmulError_[0] = 0;

  if ( meTriggerTowerEmulError_[1] ) dqmStore_->removeElement( meTriggerTowerEmulError_[1]->getName() );
  meTriggerTowerEmulError_[1] = 0;

  if ( meTriggerTowerTiming_[0] ) dqmStore_->removeElement( meTriggerTowerTiming_[0]->getName() );
  meTriggerTowerTiming_[0] = 0;

  if ( meTriggerTowerTiming_[1] ) dqmStore_->removeElement( meTriggerTowerTiming_[1]->getName() );
  meTriggerTowerTiming_[1] = 0;

  if ( meGlobalSummary_[0] ) dqmStore_->removeElement( meGlobalSummary_[0]->getName() );
  meGlobalSummary_[0] = 0;

  if ( meGlobalSummary_[1] ) dqmStore_->removeElement( meGlobalSummary_[1]->getName() );
  meGlobalSummary_[1] = 0;

}

bool EESummaryClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov, bool& status, bool flag) {

  status = true;

  if ( ! flag ) return false;

  return true;

}

void EESummaryClient::analyze(void) {

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( debug_ ) cout << "EESummaryClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;
  }

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
      if ( meCosmic_[0] ) meCosmic_[0]->setBinContent( ix, iy, 0. );
      if ( meCosmic_[1] ) meCosmic_[1]->setBinContent( ix, iy, 0. );
      if ( meTiming_[0] ) meTiming_[0]->setBinContent( ix, iy, 6. );
      if ( meTiming_[1] ) meTiming_[1]->setBinContent( ix, iy, 6. );

      if ( meGlobalSummary_[0] ) meGlobalSummary_[0]->setBinContent( ix, iy, 6. );
      if ( meGlobalSummary_[1] ) meGlobalSummary_[1]->setBinContent( ix, iy, 6. );

    }
  }

  for ( int ix = 1; ix <= 20; ix++ ) {
    for ( int iy = 1; iy <= 90; iy++ ) {

      if ( meLaserL1PN_[0] ) meLaserL1PN_[0]->setBinContent( ix, iy, 6. );
      if ( meLaserL1PN_[1] ) meLaserL1PN_[1]->setBinContent( ix, iy, 6. );
      if ( meLaserL2PN_[0] ) meLaserL2PN_[0]->setBinContent( ix, iy, 6. );
      if ( meLaserL2PN_[1] ) meLaserL2PN_[1]->setBinContent( ix, iy, 6. );
      if ( meLaserL3PN_[0] ) meLaserL3PN_[0]->setBinContent( ix, iy, 6. );
      if ( meLaserL3PN_[1] ) meLaserL3PN_[1]->setBinContent( ix, iy, 6. );
      if ( meLaserL4PN_[0] ) meLaserL4PN_[0]->setBinContent( ix, iy, 6. );
      if ( meLaserL4PN_[1] ) meLaserL4PN_[1]->setBinContent( ix, iy, 6. );
      if ( meLedL1PN_[0] ) meLedL1PN_[0]->setBinContent( ix, iy, 6. );
      if ( meLedL1PN_[1] ) meLedL1PN_[1]->setBinContent( ix, iy, 6. );
      if ( meLedL2PN_[0] ) meLedL2PN_[0]->setBinContent( ix, iy, 6. );
      if ( meLedL2PN_[1] ) meLedL2PN_[1]->setBinContent( ix, iy, 6. );
      if ( mePedestalPNG01_[0] ) mePedestalPNG01_[0]->setBinContent( ix, iy, 6. );
      if ( mePedestalPNG01_[1] ) mePedestalPNG01_[1]->setBinContent( ix, iy, 6. );
      if ( mePedestalPNG16_[0] ) mePedestalPNG16_[0]->setBinContent( ix, iy, 6. );
      if ( mePedestalPNG16_[1] ) mePedestalPNG16_[1]->setBinContent( ix, iy, 6. );
      if ( meTestPulsePNG01_[0] ) meTestPulsePNG01_[0]->setBinContent( ix, iy, 6. );
      if ( meTestPulsePNG01_[1] ) meTestPulsePNG01_[1]->setBinContent( ix, iy, 6. );
      if ( meTestPulsePNG16_[0] ) meTestPulsePNG16_[0]->setBinContent( ix, iy, 6. );
      if ( meTestPulsePNG16_[1] ) meTestPulsePNG16_[1]->setBinContent( ix, iy, 6. );

    }
  }

  for ( int ix = 1; ix <= 100; ix++ ) {
    for ( int iy = 1; iy <= 100; iy++ ) {
      if ( meTriggerTowerEt_[0] ) meTriggerTowerEt_[0]->setBinContent( ix, iy, 0. );
      if ( meTriggerTowerEt_[1] ) meTriggerTowerEt_[1]->setBinContent( ix, iy, 0. );
      if ( meTriggerTowerEmulError_[0] ) meTriggerTowerEmulError_[0]->setBinContent( ix, iy, 6. );
      if ( meTriggerTowerEmulError_[1] ) meTriggerTowerEmulError_[1]->setBinContent( ix, iy, 6. );
      if ( meTriggerTowerTiming_[0] ) meTriggerTowerTiming_[0]->setBinContent( ix, iy, -1 );
      if ( meTriggerTowerTiming_[1] ) meTriggerTowerTiming_[1]->setBinContent( ix, iy, -1 );
    }
  }

  if ( meIntegrity_[0] ) meIntegrity_[0]->setEntries( 0 );
  if ( meIntegrity_[1] ) meIntegrity_[1]->setEntries( 0 );
  if ( meIntegrityErr_ ) meIntegrityErr_->Reset();
  if ( meOccupancy_[0] ) meOccupancy_[0]->setEntries( 0 );
  if ( meOccupancy_[1] ) meOccupancy_[1]->setEntries( 0 );
  if ( meOccupancy1D_ ) meOccupancy1D_->Reset();
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
  if ( meLaserL1PN_[0] ) meLaserL1PN_[0]->setEntries( 0 );
  if ( meLaserL1PN_[1] ) meLaserL1PN_[1]->setEntries( 0 );
  if ( meLaserL1PNErr_ ) meLaserL1PNErr_->Reset();
  if ( meLaserL1Ampl_ ) meLaserL1Ampl_->Reset();
  if ( meLaserL1Timing_ ) meLaserL1Timing_->Reset();
  if ( meLaserL1AmplOverPN_ ) meLaserL1AmplOverPN_->Reset();
  if ( meLaserL2_[0] ) meLaserL2_[0]->setEntries( 0 );
  if ( meLaserL2_[1] ) meLaserL2_[1]->setEntries( 0 );
  if ( meLaserL2Err_ ) meLaserL2Err_->Reset();
  if ( meLaserL2PN_[0] ) meLaserL2PN_[0]->setEntries( 0 );
  if ( meLaserL2PN_[1] ) meLaserL2PN_[1]->setEntries( 0 );
  if ( meLaserL2PNErr_ ) meLaserL2PNErr_->Reset();
  if ( meLaserL2Ampl_ ) meLaserL2Ampl_->Reset();
  if ( meLaserL2Timing_ ) meLaserL2Timing_->Reset();
  if ( meLaserL2AmplOverPN_ ) meLaserL2AmplOverPN_->Reset();
  if ( meLaserL3_[0] ) meLaserL3_[0]->setEntries( 0 );
  if ( meLaserL3_[1] ) meLaserL3_[1]->setEntries( 0 );
  if ( meLaserL3Err_ ) meLaserL3Err_->Reset();
  if ( meLaserL3PN_[0] ) meLaserL3PN_[0]->setEntries( 0 );
  if ( meLaserL3PN_[1] ) meLaserL3PN_[1]->setEntries( 0 );
  if ( meLaserL3PNErr_ ) meLaserL3PNErr_->Reset();
  if ( meLaserL3Ampl_ ) meLaserL3Ampl_->Reset();
  if ( meLaserL3Timing_ ) meLaserL3Timing_->Reset();
  if ( meLaserL3AmplOverPN_ ) meLaserL3AmplOverPN_->Reset();
  if ( meLaserL4_[0] ) meLaserL4_[0]->setEntries( 0 );
  if ( meLaserL4_[1] ) meLaserL4_[1]->setEntries( 0 );
  if ( meLaserL4Err_ ) meLaserL4Err_->Reset();
  if ( meLaserL4PN_[0] ) meLaserL4PN_[0]->setEntries( 0 );
  if ( meLaserL4PN_[1] ) meLaserL4PN_[1]->setEntries( 0 );
  if ( meLaserL4PNErr_ ) meLaserL4PNErr_->Reset();
  if ( meLaserL4Ampl_ ) meLaserL4Ampl_->Reset();
  if ( meLaserL4Timing_ ) meLaserL4Timing_->Reset();
  if ( meLaserL4AmplOverPN_ ) meLaserL4AmplOverPN_->Reset();
  if ( meLedL1_[0] ) meLedL1_[0]->setEntries( 0 );
  if ( meLedL1_[1] ) meLedL1_[1]->setEntries( 0 );
  if ( meLedL1Err_ ) meLedL1Err_->Reset();
  if ( meLedL1PN_[0] ) meLedL1PN_[0]->setEntries( 0 );
  if ( meLedL1PN_[1] ) meLedL1PN_[1]->setEntries( 0 );
  if ( meLedL1PNErr_ ) meLedL1PNErr_->Reset();
  if ( meLedL1Ampl_ ) meLedL1Ampl_->Reset();
  if ( meLedL1Timing_ ) meLedL1Timing_->Reset();
  if ( meLedL1AmplOverPN_ ) meLedL1AmplOverPN_->Reset();
  if ( meLedL2_[0] ) meLedL2_[0]->setEntries( 0 );
  if ( meLedL2_[1] ) meLedL2_[1]->setEntries( 0 );
  if ( meLedL2Err_ ) meLedL2Err_->Reset();
  if ( meLedL2PN_[0] ) meLedL2PN_[0]->setEntries( 0 );
  if ( meLedL2PN_[1] ) meLedL2PN_[1]->setEntries( 0 );
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
  if ( mePedestalPNG01_[0] ) mePedestalPNG01_[0]->setEntries( 0 );
  if ( mePedestalPNG01_[1] ) mePedestalPNG01_[1]->setEntries( 0 );
  if ( mePedestalPNG16_[0] ) mePedestalPNG16_[0]->setEntries( 0 );
  if ( mePedestalPNG16_[1] ) mePedestalPNG16_[1]->setEntries( 0 );
  if ( meTestPulseG01_[0] ) meTestPulseG01_[0]->setEntries( 0 );
  if ( meTestPulseG01_[1] ) meTestPulseG01_[1]->setEntries( 0 );
  if ( meTestPulseG06_[0] ) meTestPulseG06_[0]->setEntries( 0 );
  if ( meTestPulseG06_[1] ) meTestPulseG06_[1]->setEntries( 0 );
  if ( meTestPulseG12_[0] ) meTestPulseG12_[0]->setEntries( 0 );
  if ( meTestPulseG12_[1] ) meTestPulseG12_[1]->setEntries( 0 );
  if ( meTestPulsePNG01_[0] ) meTestPulsePNG01_[0]->setEntries( 0 );
  if ( meTestPulsePNG01_[1] ) meTestPulsePNG01_[1]->setEntries( 0 );
  if ( meTestPulsePNG16_[0] ) meTestPulsePNG16_[0]->setEntries( 0 );
  if ( meTestPulsePNG16_[1] ) meTestPulsePNG16_[1]->setEntries( 0 );
  if ( meTestPulseAmplG01_ ) meTestPulseAmplG01_->Reset();
  if ( meTestPulseAmplG06_ ) meTestPulseAmplG06_->Reset();
  if ( meTestPulseAmplG12_ ) meTestPulseAmplG12_->Reset();

  if ( meCosmic_[0] ) meCosmic_[0]->setEntries( 0 );
  if ( meCosmic_[1] ) meCosmic_[1]->setEntries( 0 );
  if ( meTiming_[0] ) meTiming_[0]->setEntries( 0 );
  if ( meTiming_[1] ) meTiming_[1]->setEntries( 0 );
  if ( meTriggerTowerEt_[0] ) meTriggerTowerEt_[0]->setEntries( 0 );
  if ( meTriggerTowerEt_[1] ) meTriggerTowerEt_[1]->setEntries( 0 );
  if ( meTriggerTowerEtSpectrum_[0] ) meTriggerTowerEtSpectrum_[0]->Reset();
  if ( meTriggerTowerEtSpectrum_[1] ) meTriggerTowerEtSpectrum_[1]->Reset();
  if ( meTriggerTowerEmulError_[0] ) meTriggerTowerEmulError_[0]->setEntries( 0 );
  if ( meTriggerTowerEmulError_[1] ) meTriggerTowerEmulError_[1]->setEntries( 0 );
  if ( meTriggerTowerTiming_[0] ) meTriggerTowerTiming_[0]->setEntries( 0 );
  if ( meTriggerTowerTiming_[1] ) meTriggerTowerTiming_[1]->setEntries( 0 );

  if ( meGlobalSummary_[0] ) meGlobalSummary_[0]->setEntries( 0 );
  if ( meGlobalSummary_[1] ) meGlobalSummary_[1]->setEntries( 0 );

  for ( unsigned int i=0; i<clients_.size(); i++ ) {

    EEIntegrityClient* eeic = dynamic_cast<EEIntegrityClient*>(clients_[i]);
    EEStatusFlagsClient* eesfc = dynamic_cast<EEStatusFlagsClient*>(clients_[i]);
    EEPedestalOnlineClient* eepoc = dynamic_cast<EEPedestalOnlineClient*>(clients_[i]);

    EELaserClient* eelc = dynamic_cast<EELaserClient*>(clients_[i]);
    EELedClient* eeldc = dynamic_cast<EELedClient*>(clients_[i]);
    EEPedestalClient* eepc = dynamic_cast<EEPedestalClient*>(clients_[i]);
    EETestPulseClient* eetpc = dynamic_cast<EETestPulseClient*>(clients_[i]);

    EECosmicClient* eecc = dynamic_cast<EECosmicClient*>(clients_[i]);
    EETimingClient* eetmc = dynamic_cast<EETimingClient*>(clients_[i]);
    EETriggerTowerClient* eetttc = dynamic_cast<EETriggerTowerClient*>(clients_[i]);

    MonitorElement *me;
    MonitorElement *me_01, *me_02, *me_03;
    //    MonitorElement *me_f[6], *me_fg[2];
//    MonitorElement *me_04, *me_05;

    TH2F* h2;
    TProfile2D* h2d;

    for ( unsigned int i=0; i<superModules_.size(); i++ ) {

      int ism = superModules_[i];

      char histo[200];

      sprintf(histo, (prefixME_ + "/EEPedestalOnlineTask/Gain12/EEPOT pedestal %s G12").c_str(), Numbers::sEE(ism).c_str());
      me = dqmStore_->get(histo);
      hpot01_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, hpot01_[ism-1] );

      sprintf(histo, (prefixME_ + "/EETriggerTowerTask/EETTT Et map Real Digis %s").c_str(), Numbers::sEE(ism).c_str());
      me = dqmStore_->get(histo);
      httt01_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, httt01_[ism-1] );

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
              
              if ( ism >= 1 && ism <= 9 ) meIntegrity_[0]->setBinContent( 101 - jx, jy, xval );
              else meIntegrity_[1]->setBinContent( jx, jy, xval );
              
              if ( xval == 0 ) meIntegrityErr_->Fill( ism );
              
            }
            
            h2 = eeic->h_[ism-1];
            
            if ( h2 ) {
              
              float xval = h2->GetBinContent( ix, iy );
              
              if ( ism >= 1 && ism <= 9 ) {
                if ( xval != 0 ) meOccupancy_[0]->setBinContent( 101 - jx, jy, xval );
              } else {
                if ( xval != 0 ) meOccupancy_[1]->setBinContent( jx, jy, xval );
              }
              
              meOccupancy1D_->Fill( ism, xval );
              
            }
            
          }
          
          if ( eepoc ) {

            me = eepoc->meg03_[ism-1];
          
            if ( me ) {
            
              float xval = me->getBinContent( ix, iy );
              
              if ( ism >= 1 && ism <= 9 ) mePedestalOnline_[0]->setBinContent( 101 - jx, jy, xval );
              else mePedestalOnline_[1]->setBinContent( jx, jy, xval );
              
              if ( xval == 0 ) mePedestalOnlineErr_->Fill( ism );
              
            }

          }

          float num01, mean01, rms01;
          bool update01 = UtilsClient::getBinStatistics(hpot01_[ism-1], ix, iy, num01, mean01, rms01);

          if ( update01 ) {
            
            mePedestalOnlineRMS_->Fill( ism, rms01 );
            mePedestalOnlineMean_->Fill( ism, mean01 );

            if ( ism >= 1 && ism <= 9 ) mePedestalOnlineRMSMap_[0]->setBinContent( 101 - jx, jy, rms01 );
            else mePedestalOnlineRMSMap_[1]->setBinContent( jx, jy, rms01 );

          }

          if ( eelc ) {

            if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 1) != laserWavelengths_.end() ) {

              me = eelc->meg01_[ism-1];

              if ( me ) {

                float xval = me->getBinContent( ix, iy );

                if ( me->getEntries() != 0 ) {
                  if ( ism >= 1 && ism <= 9 ) meLaserL1_[0]->setBinContent( 101 - jx, jy, xval );
                  else meLaserL1_[1]->setBinContent( jx, jy, xval );

                  if ( xval == 0 ) meLaserL1Err_->Fill( ism );
                }

              }
              
            }

            if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 2) != laserWavelengths_.end() ) {

              me = eelc->meg02_[ism-1];

              if ( me ) {

                float xval = me->getBinContent( ix, iy );

                if ( me->getEntries() != 0 ) {
                  if ( ism >= 1 && ism <= 9 ) meLaserL2_[0]->setBinContent( 101 - jx, jy, xval );
                  else meLaserL2_[1]->setBinContent( jx, jy, xval );

                  if ( xval == 0 ) meLaserL2Err_->Fill( ism );
                }

              }

            }

            if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 3) != laserWavelengths_.end() ) {

              me = eelc->meg03_[ism-1];

              if ( me ) {

                float xval = me->getBinContent( ix, iy );

                if ( me->getEntries() != 0 ) {
                  if ( ism >= 1 && ism <= 9 ) meLaserL3_[0]->setBinContent( 101 - jx, jy, xval );
                  else meLaserL3_[1]->setBinContent( jx, jy, xval );

                  if ( xval == 0 ) meLaserL3Err_->Fill( ism );
                }

              }
              
            }

            if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 4) != laserWavelengths_.end() ) {

              me = eelc->meg04_[ism-1];

              if ( me ) {

                float xval = me->getBinContent( ix, iy );

                if ( me->getEntries() != 0 ) {
                  if ( ism >= 1 && ism <= 9 ) meLaserL4_[0]->setBinContent( 101 - jx, jy, xval );
                  else meLaserL4_[1]->setBinContent( jx, jy, xval );
                  
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
                  if ( ism >= 1 && ism <= 9 ) meLedL1_[0]->setBinContent( 101 - jx, jy, xval );
                  else meLedL1_[1]->setBinContent( jx, jy, xval );
                  
                  if ( xval == 0 ) meLedL1Err_->Fill( ism );
                }
                
              }

            }

            if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 2) != ledWavelengths_.end() ) {

              me = eeldc->meg02_[ism-1];

              if ( me ) {
                
                float xval = me->getBinContent( ix, iy );
                
                if ( me->getEntries() != 0 ) {
                  if ( ism >= 1 && ism <= 9 ) meLedL2_[0]->setBinContent( 101 - jx, jy, xval );
                  else meLedL2_[1]->setBinContent( jx, jy, xval );

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
                if ( ism >= 1 && ism <= 9 ) mePedestalG01_[0]->setBinContent( 101 - jx, jy, val_01 );
                else mePedestalG01_[1]->setBinContent( jx, jy, val_01 );
              }
            }
            if ( me_02 ) {
              float val_02=me_02->getBinContent(ix,iy);
              if ( me_02->getEntries() != 0 ) {
                if ( ism >= 1 && ism <= 9 ) mePedestalG06_[0]->setBinContent( 101 - jx, jy, val_02 );
                else mePedestalG06_[1]->setBinContent( jx, jy, val_02 );
              }
            }
            if ( me_03 ) {
              float val_03=me_03->getBinContent(ix,iy);
              if ( me_03->getEntries() != 0 ) {
                if ( ism >= 1 && ism <= 9 ) mePedestalG12_[0]->setBinContent( 101 - jx, jy, val_03 );
                else mePedestalG12_[1]->setBinContent( jx, jy, val_03 );
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
                if ( ism >= 1 && ism <= 9 ) meTestPulseG01_[0]->setBinContent( 101 - jx, jy, val_01 );
                else meTestPulseG01_[1]->setBinContent( jx, jy, val_01 );
              }
            }
            if ( me_02 ) {
              float val_02=me_02->getBinContent(ix,iy);
              if ( me_02->getEntries() != 0 ) {
                if ( ism >= 1 && ism <= 9 ) meTestPulseG06_[0]->setBinContent( 101 - jx, jy, val_02 );
                else meTestPulseG06_[1]->setBinContent( jx, jy, val_02 );
              }
            }
            if ( me_03 ) {
              float val_03=me_03->getBinContent(ix,iy);
              if ( me_03->getEntries() != 0 ) {
                if ( ism >= 1 && ism <= 9 ) meTestPulseG12_[0]->setBinContent( 101 - jx, jy, val_03 );
                else meTestPulseG12_[1]->setBinContent( jx, jy, val_03 );
              }
            }

          }

          if ( eecc ) {

            h2d = eecc->h02_[ism-1];

            if ( h2d ) {

              float xval = h2d->GetBinContent( ix, iy );

              if ( ism >= 1 && ism <= 9 ) {
                if ( xval != 0 ) meCosmic_[0]->setBinContent( 101 - jx, jy, xval );
              } else {
                if ( xval != 0 ) meCosmic_[1]->setBinContent( jx, jy, xval );
              }

            }

          }

          if ( eetmc ) {

            me = eetmc->meg01_[ism-1];

            if ( me ) {

              float xval = me->getBinContent( ix, iy );

              if ( ism >= 1 && ism <= 9 ) meTiming_[0]->setBinContent( 101 - jx, jy, xval );
              else meTiming_[1]->setBinContent( jx, jy, xval );

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

            me = eesfc->meh01_[ism-1];

            if ( me ) {

              float xval = 6;

              if ( me->getBinContent( ix, iy ) < 0 ) xval = 2;
              if ( me->getBinContent( ix, iy ) == 0 ) xval = 1;
              if ( me->getBinContent( ix, iy ) > 0 ) xval = 0;

              if ( ism >= 1 && ism <= 9 ) meStatusFlags_[0]->setBinContent( 101 - jx, jy, xval );
              else meStatusFlags_[1]->setBinContent( jx, jy, xval );
              
              if ( xval == 0 ) meStatusFlagsErr_->Fill( ism );

            }

          }

          if ( eetttc ) {

            float num01, mean01, rms01;
            bool update01 = UtilsClient::getBinStatistics(httt01_[ism-1], jx, jy, num01, mean01, rms01);
            
            if ( update01 ) {
              if ( ism >= 1 && ism <= 9 ) {
                if ( meTriggerTowerEt_[0] ) meTriggerTowerEt_[0]->setBinContent( 101 - jx, jy, mean01 );
                if ( meTriggerTowerEtSpectrum_[0] ) meTriggerTowerEtSpectrum_[0]->Fill( mean01 );
              }
              else {
                if ( meTriggerTowerEt_[1] ) meTriggerTowerEt_[1]->setBinContent( jx, jy, mean01 );
                if ( meTriggerTowerEtSpectrum_[1] ) meTriggerTowerEtSpectrum_[1]->Fill( mean01 );
              }
            }

            me = eetttc->me_o01_[ism-1];

            if ( me ) {

              float xval = me->getBinContent( ix, iy );

              if ( ism >= 1 && ism <= 9 ) {
                meTriggerTowerTiming_[0]->setBinContent( 101 - jx, jy, xval );
              } else {
                meTriggerTowerTiming_[1]->setBinContent( jx, jy, xval );
              }

            }

            float xval = 6;
            if( mean01 <= 0 ) xval = 2;
            else {

              h2 = eetttc->l01_[ism-1];

              if ( h2 ) {

                float emulErrorVal = h2->GetBinContent( ix, iy );
                if( emulErrorVal!=0 ) xval = 0;

              }

              // do not propagate the flag bits to the summary for now
//               for ( int iflag=0; iflag<6; iflag++ ) {

//                 me_f[iflag] = eetttc->me_m01_[ism-1][iflag];

//                 if ( me_f[iflag] ) {

//                   float emulFlagErrorVal = me_f[iflag]->getBinContent( ix, iy );
//                   if ( emulFlagErrorVal!=0 ) xval = 0;

//                 }

//               }

//               for ( int ifg=0; ifg<2; ifg++) {

//                 me_fg[ifg] = eetttc->me_n01_[ism-1][ifg];
//                 if ( me_fg[ifg] ) {

//                   float emulFineGrainVetoErrorVal = me_fg[ifg]->getBinContent( ix, iy );
//                   if ( emulFineGrainVetoErrorVal!=0 ) xval = 0;

//                 }

//               }

              if ( xval!=0 ) xval = 1;

            }

            // see fix below
            if ( xval == 2 ) continue;

            if ( ism >= 1 && ism <= 9 ) {
              meTriggerTowerEmulError_[0]->setBinContent( 101 - jx, jy, xval );
            } else {
              meTriggerTowerEmulError_[1]->setBinContent( jx, jy, xval );
            }

          }

        }
      }

      // PN's summaries
      for( int i = 1; i <= 10; i++ ) {
        for( int j = 1; j <= 5; j++ ) {

          if ( eepc ) {

          }

          if ( eetpc ) {

          }

          if ( eelc ) {

          }

          if ( eeldc ) {


          }

        }
      }

      for ( int ix=1; ix<=50; ix++ ) {
        for (int iy=1; iy<=50; iy++ ) {

          int jx = ix + Numbers::ix0EE(ism);
          int jy = iy + Numbers::iy0EE(ism);
          int ic = Numbers::icEE(ism, jx, jy);

          if( ic != -1 ) {

            if ( eelc ) {

              if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 1) != laserWavelengths_.end() ) {

                MonitorElement *meg = eelc->meg01_[ism-1];

                float xval = 2;
                if ( meg ) xval = meg->getBinContent( ix, iy );

                // exclude channels without laser data (yellow in the quality map)
                if( xval != 2 && xval != 5 ) { 
                
                  MonitorElement *mea01 = eelc->mea01_[ism-1];
                  MonitorElement *met01 = eelc->met01_[ism-1];
                  MonitorElement *meaopn01 = eelc->meaopn01_[ism-1];

                  if( mea01 && met01 && meaopn01 ) {
                    meLaserL1Ampl_->Fill( ism, mea01->getBinContent( ic ) );
                    meLaserL1Timing_->Fill( ism, met01->getBinContent( ic ) );
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
                
                  MonitorElement *mea02 = eelc->mea02_[ism-1];
                  MonitorElement *met02 = eelc->met02_[ism-1];
                  MonitorElement *meaopn02 = eelc->meaopn02_[ism-1];

                  if( mea02 && met02 && meaopn02 ) {
                    meLaserL2Ampl_->Fill( ism, mea02->getBinContent( ic ) );
                    meLaserL2Timing_->Fill( ism, met02->getBinContent( ic ) );
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
                
                  MonitorElement *mea03 = eelc->mea03_[ism-1];
                  MonitorElement *met03 = eelc->met03_[ism-1];
                  MonitorElement *meaopn03 = eelc->meaopn03_[ism-1];

                  if( mea03 && met03 && meaopn03 ) {
                    meLaserL3Ampl_->Fill( ism, mea03->getBinContent( ic ) );
                    meLaserL3Timing_->Fill( ism, met03->getBinContent( ic ) );
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
                
                  MonitorElement *mea04 = eelc->mea04_[ism-1];
                  MonitorElement *met04 = eelc->met04_[ism-1];
                  MonitorElement *meaopn04 = eelc->meaopn04_[ism-1];

                  if( mea04 && met04 && meaopn04 ) {
                    meLaserL4Ampl_->Fill( ism, mea04->getBinContent( ic ) );
                    meLaserL4Timing_->Fill( ism, met04->getBinContent( ic ) );
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
                
                  MonitorElement *mea01 = eeldc->mea01_[ism-1];
                  MonitorElement *met01 = eeldc->met01_[ism-1];
                  MonitorElement *meaopn01 = eeldc->meaopn01_[ism-1];

                  if( mea01 && met01 && meaopn01 ) {
                    meLedL1Ampl_->Fill( ism, mea01->getBinContent( ic ) );
                    meLedL1Timing_->Fill( ism, met01->getBinContent( ic ) );
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
                
                  MonitorElement *mea02 = eeldc->mea02_[ism-1];
                  MonitorElement *met02 = eeldc->met02_[ism-1];
                  MonitorElement *meaopn02 = eeldc->meaopn02_[ism-1];

                  if( mea02 && met02 && meaopn02 ) {
                    meLedL2Ampl_->Fill( ism, mea02->getBinContent( ic ) );
                    meLedL2Timing_->Fill( ism, met02->getBinContent( ic ) );
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

          }

        } // loop on jy
      } // loop on jx

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

      if(meIntegrity_[0] && mePedestalOnline_[0] && meTiming_[0] && meStatusFlags_[0] && meTriggerTowerEmulError_[0]) {

        float xval = 6;
        float val_in = meIntegrity_[0]->getBinContent(jx,jy);
        float val_po = mePedestalOnline_[0]->getBinContent(jx,jy);
        float val_tm = meTiming_[0]->getBinContent(jx,jy);
        float val_sf = meStatusFlags_[0]->getBinContent(jx,jy);
	// float val_ee = meTriggerTowerEmulError_[0]->getBinContent(jx,jy); // removed temporarily from the global summary
	float val_ee = 1;

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

        // DO NOT CONSIDER CALIBRATION EVENTS IN THE REPORT SUMMARY UNTIL LHC COLLISIONS
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
          vector<int>::iterator iter = find(superModules_.begin(), superModules_.end(), ism);
          if (iter != superModules_.end()) {
            if ( Numbers::validEE(ism, jx, jy) ) {
              validCry = true;
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

      if(meIntegrity_[1] && mePedestalOnline_[1] && meTiming_[1] && meStatusFlags_[1] && meTriggerTowerEmulError_[1]) {

        float xval = 6;
        float val_in = meIntegrity_[1]->getBinContent(jx,jy);
        float val_po = mePedestalOnline_[1]->getBinContent(jx,jy);
        float val_tm = meTiming_[1]->getBinContent(jx,jy);
        float val_sf = meStatusFlags_[1]->getBinContent(jx,jy);
        // float val_ee = meTriggerTowerEmulError_[1]->getBinContent(jx,jy); // removed temporarily from the global summary
	float val_ee = 1;

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

        // DO NOT CONSIDER CALIBRATION EVENTS IN THE REPORT SUMMARY UNTIL LHC COLLISIONS
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
          if ( Numbers::validEE(ism, jx, jy) ) {
            validCry = true;
            for ( unsigned int i=0; i<clients_.size(); i++ ) {
              EEIntegrityClient* eeic = dynamic_cast<EEIntegrityClient*>(clients_[i]);
              if ( eeic ) {
                TH2F *h2 = eeic->h_[ism-1];
                if ( h2 ) {
                  iEntries = h2->GetEntries();
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

  MonitorElement* me;

  float reportSummary = -1.0;
  if ( nValidChannels != 0 )
    reportSummary = 1.0 - float(nGlobalErrors)/float(nValidChannels);
  me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummary");
  if ( me ) me->Fill(reportSummary);

  char histo[200];

  for (int i = 0; i < 18; i++) {
    float reportSummaryEE = -1.0;
    if ( nValidChannelsEE[i] != 0 )
      reportSummaryEE = 1.0 - float(nGlobalErrorsEE[i])/float(nValidChannelsEE[i]);
    sprintf(histo, "EcalEndcap_%s", Numbers::sEE(i+1).c_str());
    me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummaryContents/" + histo);
    if ( me ) me->Fill(reportSummaryEE);
  }

  me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummaryMap");
  if ( me ) {

    int nValidChannelsTT[2][20][20];
    int nGlobalErrorsTT[2][20][20];
    int nOutOfGeometryTT[2][20][20];
    for ( int jxdcc = 0; jxdcc < 20; jxdcc++ ) {
      for ( int jydcc = 0; jydcc < 20; jydcc++ ) {
        for ( int iside = 0; iside < 2; iside++ ) {
          nValidChannelsTT[iside][jxdcc][jydcc] = 0;
          nGlobalErrorsTT[iside][jxdcc][jydcc] = 0;
          nOutOfGeometryTT[iside][jxdcc][jydcc] = 0;
        }
      }
    }

    int ttx[200][100];
    int tty[200][100];
    for ( int jx = 1; jx <= 100; jx++ ) {
      for ( int jy = 1; jy <= 100; jy++ ) {
        for ( int iside = 0; iside < 2; iside++ ) {

          int jxdcc = (jx-1)/5+1;
          int jydcc = (jy-1)/5+1;

          float xval = meGlobalSummary_[iside]->getBinContent( jx, jy );

          if ( xval >= 0 && xval <= 5 ) {
            if ( xval != 2 && xval != 5 ) ++nValidChannelsTT[iside][jxdcc-1][jydcc-1];
            if ( xval == 0 ) ++nGlobalErrorsTT[iside][jxdcc-1][jydcc-1];
          } else {
            nOutOfGeometryTT[iside][jxdcc-1][jydcc-1]++;
          }

          int ix = (iside==0) ? jx-1 : jx-1+100;
          int iy = jy-1;
          ttx[ix][iy] = jxdcc-1;
          tty[ix][iy] = jydcc-1;
        }
      }
    }

    for ( int iz = -1; iz < 2; iz+=2 ) {
      for ( int ix = 1; ix <= 100; ix++ ) {
        for ( int iy = 1; iy <= 100; iy++ ) {

          int jx = (iz==1) ? 100 + ix : ix;
          int jy = iy;

          float xval = -1;

          if( EEDetId::validDetId(ix, iy, iz) ) {
            
            int TTx = ttx[jx-1][jy-1];
            int TTy = tty[jx-1][jy-1];

            int iside = (iz==1) ? 1 : 0;

            if( nValidChannelsTT[iside][TTx][TTy] != 0 ) 
              xval = 1.0 - float(nGlobalErrorsTT[iside][TTx][TTy])/float(nValidChannelsTT[iside][TTx][TTy]);
            
            me->setBinContent( jx, jy, xval );

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

void EESummaryClient::softReset(bool flag) {

}

