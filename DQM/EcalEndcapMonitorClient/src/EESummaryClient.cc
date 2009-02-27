/*
 * \file EESummaryClient.cc
 *
 * $Date: 2009/02/27 12:31:33 $
 * $Revision: 1.157 $
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
  mePedestalOnlineMean_[0]   = 0;
  mePedestalOnlineMean_[1]   = 0;
  mePedestalOnlineRMS_[0]    = 0;
  mePedestalOnlineRMS_[1]    = 0;
  meLaserL1_[0]        = 0;
  meLaserL1_[1]        = 0;
  meLaserL1PN_[0]      = 0;
  meLaserL1PN_[1]      = 0;
  meLaserL1AmplOverPN_[0] = 0;
  meLaserL1AmplOverPN_[1] = 0;
  meLaserL1Timing_[0]  = 0;
  meLaserL1Timing_[1]  = 0;
  meLedL1_[0]          = 0;
  meLedL1_[1]          = 0;
  meLedL1PN_[0]        = 0;
  meLedL1PN_[1]        = 0;
  meLedL1AmplOverPN_[0] = 0;
  meLedL1AmplOverPN_[1] = 0;
  meLedL1Timing_[0]     = 0;
  meLedL1Timing_[1]     = 0;
  mePedestal_[0]       = 0;
  mePedestal_[1]       = 0;
  mePedestalPN_[0]     = 0;
  mePedestalPN_[1]     = 0;
  meTestPulse_[0]      = 0;
  meTestPulse_[1]      = 0;
  meTestPulsePN_[0]    = 0;
  meTestPulsePN_[1]    = 0;
  meTestPulseAmplG01_[0] = 0;
  meTestPulseAmplG01_[1] = 0;
  meTestPulseAmplG06_[0] = 0;
  meTestPulseAmplG06_[1] = 0;
  meTestPulseAmplG12_[0] = 0;
  meTestPulseAmplG12_[1] = 0;
  meGlobalSummary_[0]  = 0;
  meGlobalSummary_[1]  = 0;

  meCosmic_[0]         = 0;
  meCosmic_[1]         = 0;
  meTiming_[0]         = 0;
  meTiming_[1]         = 0;
  meTriggerTowerEt_[0]        = 0;
  meTriggerTowerEt_[1]        = 0;
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
  meLedL1Err_           = 0;
  meLedL1PNErr_         = 0;
  mePedestalErr_        = 0;
  mePedestalPNErr_      = 0;
  meTestPulseErr_       = 0;
  meTestPulsePNErr_     = 0;

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
  if ( me = dqmStore_->get(prefixME_ + "/EventInfo/" + histo) ) {
    dqmStore_->removeElement(me->getName());
  }
  me = dqmStore_->bookFloat(histo);
  me->Fill(-1.0);

  dqmStore_->setCurrentFolder( prefixME_ + "/EventInfo/reportSummaryContents" );

  for (int i = 0; i < 18; i++) {
    sprintf(histo, "EcalEndcap_%s", Numbers::sEE(i+1).c_str());
    if ( me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummaryContents/" + histo) ) {
      dqmStore_->removeElement(me->getName());
    }
    me = dqmStore_->bookFloat(histo);
    me->Fill(-1.0);
  }

  dqmStore_->setCurrentFolder( prefixME_ + "/EventInfo" );

  sprintf(histo, "reportSummaryMap");
  if ( me = dqmStore_->get(prefixME_ + "/EventInfo/" + histo) ) {
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

  if ( mePedestalOnlineMean_[0] ) dqmStore_->removeElement( mePedestalOnlineMean_[0]->getName() );
  sprintf(histo, "EEPOT EE - pedestal G12 mean");
  mePedestalOnlineMean_[0] = dqmStore_->book1D(histo, histo, 100, 150., 250.);

  if ( mePedestalOnlineMean_[1] ) dqmStore_->removeElement( mePedestalOnlineMean_[1]->getName() );
  sprintf(histo, "EEPOT EE + pedestal G12 mean");
  mePedestalOnlineMean_[1] = dqmStore_->book1D(histo, histo, 100, 150., 250.);

  if ( mePedestalOnlineRMS_[0] ) dqmStore_->removeElement( mePedestalOnlineRMS_[0]->getName() );
  sprintf(histo, "EEPOT EE - pedestal G12 rms");
  mePedestalOnlineRMS_[0] = dqmStore_->book1D(histo, histo, 100, 0., 10.);

  if ( mePedestalOnlineRMS_[1] ) dqmStore_->removeElement( mePedestalOnlineRMS_[1]->getName() );
  sprintf(histo, "EEPOT EE + pedestal G12 rms");
  mePedestalOnlineRMS_[1] = dqmStore_->book1D(histo, histo, 100, 0., 10.);

  if ( mePedestalOnlineErr_ ) dqmStore_->removeElement( mePedestalOnlineErr_->getName() );
  sprintf(histo, "EEPOT pedestal quality errors summary G12");
  mePedestalOnlineErr_ = dqmStore_->book1D(histo, histo, 18, 1, 19);
  for (int i = 0; i < 18; i++) {
    mePedestalOnlineErr_->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
  }

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

  if ( meLaserL1AmplOverPN_[0] ) dqmStore_->removeElement( meLaserL1AmplOverPN_[0]->getName() );
  sprintf(histo, "EELT EE - laser L1 amplitude over PN");
  meLaserL1AmplOverPN_[0] = dqmStore_->book1D(histo, histo, 100, 0., 20.);

  if ( meLaserL1AmplOverPN_[1] ) dqmStore_->removeElement( meLaserL1AmplOverPN_[1]->getName() );
  sprintf(histo, "EELT EE + laser L1 amplitude over PN");
  meLaserL1AmplOverPN_[1] = dqmStore_->book1D(histo, histo, 100, 0., 20.);

  if ( meLaserL1Timing_[0] ) dqmStore_->removeElement( meLaserL1Timing_[0]->getName() );
  sprintf(histo, "EELT EE - laser L1 timing");
  meLaserL1Timing_[0] = dqmStore_->book1D(histo, histo, 100, 0., 10.);

  if ( meLaserL1Timing_[1] ) dqmStore_->removeElement( meLaserL1Timing_[1]->getName() );
  sprintf(histo, "EELT EE + laser L1 timing");
  meLaserL1Timing_[1] = dqmStore_->book1D(histo, histo, 100, 0., 10.);

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

  if ( meLedL1AmplOverPN_[0] ) dqmStore_->removeElement( meLedL1AmplOverPN_[0]->getName() );
  sprintf(histo, "EELT EE - led L1 amplitude over PN");
  meLedL1AmplOverPN_[0] = dqmStore_->book1D(histo, histo, 100, 0., 20.);

  if ( meLedL1AmplOverPN_[1] ) dqmStore_->removeElement( meLedL1AmplOverPN_[1]->getName() );
  sprintf(histo, "EELT EE + led L1 amplitude over PN");
  meLedL1AmplOverPN_[1] = dqmStore_->book1D(histo, histo, 100, 0., 20.);

  if ( meLedL1Timing_[0] ) dqmStore_->removeElement( meLedL1Timing_[0]->getName() );
  sprintf(histo, "EELT EE - led L1 timing");
  meLedL1Timing_[0] = dqmStore_->book1D(histo, histo, 100, 0., 10.);

  if ( meLedL1Timing_[1] ) dqmStore_->removeElement( meLedL1Timing_[1]->getName() );
  sprintf(histo, "EELT EE + led L1 timing");
  meLedL1Timing_[1] = dqmStore_->book1D(histo, histo, 100, 0., 10.);

  if( mePedestal_[0] ) dqmStore_->removeElement( mePedestal_[0]->getName() );
  sprintf(histo, "EEPT EE - pedestal quality summary");
  mePedestal_[0] = dqmStore_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
  mePedestal_[0]->setAxisTitle("jx", 1);
  mePedestal_[0]->setAxisTitle("jy", 2);

  if( mePedestalPN_[0] ) dqmStore_->removeElement( mePedestalPN_[0]->getName() );
  sprintf(histo, "EEPT EE - PN pedestal quality summary");
  mePedestalPN_[0] = dqmStore_->book2D(histo, histo, 90, 0., 90., 20, -10, 10.);
  mePedestalPN_[0]->setAxisTitle("jx", 1);
  mePedestalPN_[0]->setAxisTitle("jy", 2);

  if( mePedestal_[1] ) dqmStore_->removeElement( mePedestal_[1]->getName() );
  sprintf(histo, "EEPT EE + pedestal quality summary");
  mePedestal_[1] = dqmStore_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
  mePedestal_[1]->setAxisTitle("jx", 1);
  mePedestal_[1]->setAxisTitle("jy", 2);

  if( mePedestalPN_[1] ) dqmStore_->removeElement( mePedestalPN_[1]->getName() );
  sprintf(histo, "EEPT EE + PN pedestal quality summary");
  mePedestalPN_[1] = dqmStore_->book2D(histo, histo, 90, 0., 90., 20, -10, 10.);
  mePedestalPN_[1]->setAxisTitle("jx", 1);
  mePedestalPN_[1]->setAxisTitle("jy", 2);

  if ( mePedestalErr_ ) dqmStore_->removeElement( mePedestalErr_->getName() );
  sprintf(histo, "EEPT pedestal quality errors summary");
  mePedestalErr_ = dqmStore_->book1D(histo, histo, 18, 1, 19);
  for (int i = 0; i < 18; i++) {
    mePedestalErr_->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
  }

  if ( mePedestalPNErr_ ) dqmStore_->removeElement( mePedestalPNErr_->getName() );
  sprintf(histo, "EEPT PN pedestal quality errors summary");
  mePedestalPNErr_ = dqmStore_->book1D(histo, histo, 18, 1, 19);
  for (int i = 0; i < 18; i++) {
    mePedestalPNErr_->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
  }

  if( meTestPulse_[0] ) dqmStore_->removeElement( meTestPulse_[0]->getName() );
  sprintf(histo, "EETPT EE - test pulse quality summary");
  meTestPulse_[0] = dqmStore_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
  meTestPulse_[0]->setAxisTitle("jx", 1);
  meTestPulse_[0]->setAxisTitle("jy", 2);

  if( meTestPulsePN_[0] ) dqmStore_->removeElement( meTestPulsePN_[0]->getName() );
  sprintf(histo, "EETPT EE - PN test pulse quality summary");
  meTestPulsePN_[0] = dqmStore_->book2D(histo, histo, 90, 0., 90., 20, -10., 10.);
  meTestPulsePN_[0]->setAxisTitle("jx", 1);
  meTestPulsePN_[0]->setAxisTitle("jy", 2);

  if( meTestPulse_[1] ) dqmStore_->removeElement( meTestPulse_[1]->getName() );
  sprintf(histo, "EETPT EE + test pulse quality summary");
  meTestPulse_[1] = dqmStore_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
  meTestPulse_[1]->setAxisTitle("jx", 1);
  meTestPulse_[1]->setAxisTitle("jy", 2);

  if( meTestPulsePN_[1] ) dqmStore_->removeElement( meTestPulsePN_[1]->getName() );
  sprintf(histo, "EETPT EE + PN test pulse quality summary");
  meTestPulsePN_[1] = dqmStore_->book2D(histo, histo, 90, 0., 90., 20, -10., 10.);
  meTestPulsePN_[1]->setAxisTitle("jx", 1);
  meTestPulsePN_[1]->setAxisTitle("jy", 2);

  if ( meTestPulseErr_ ) dqmStore_->removeElement( meTestPulseErr_->getName() );
  sprintf(histo, "EETPT test pulse quality errors summary");
  meTestPulseErr_ = dqmStore_->book1D(histo, histo, 18, 1, 19);
  for (int i = 0; i < 18; i++) {
    meTestPulseErr_->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
  }

  if ( meTestPulsePNErr_ ) dqmStore_->removeElement( meTestPulsePNErr_->getName() );
  sprintf(histo, "EETPT PN test pulse quality errors summary");
  meTestPulsePNErr_ = dqmStore_->book1D(histo, histo, 18, 1, 19);
  for (int i = 0; i < 18; i++) {
    meTestPulsePNErr_->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
  }

  if( meTestPulseAmplG01_[0] ) dqmStore_->removeElement( meTestPulseAmplG01_[0]->getName() );
  sprintf(histo, "EETPT EE - test pulse amplitude G01 summary");
  meTestPulseAmplG01_[0] = dqmStore_->book1D(histo, histo, 100, 2000, 4000);

  if( meTestPulseAmplG01_[1] ) dqmStore_->removeElement( meTestPulseAmplG01_[1]->getName() );
  sprintf(histo, "EETPT EE + test pulse amplitude G01 summary");
  meTestPulseAmplG01_[1] = dqmStore_->book1D(histo, histo, 100, 2000, 4000);

  if( meTestPulseAmplG06_[0] ) dqmStore_->removeElement( meTestPulseAmplG06_[0]->getName() );
  sprintf(histo, "EETPT EE - test pulse amplitude G06 summary");
  meTestPulseAmplG06_[0] = dqmStore_->book1D(histo, histo, 100, 2000, 4000);

  if( meTestPulseAmplG06_[1] ) dqmStore_->removeElement( meTestPulseAmplG06_[1]->getName() );
  sprintf(histo, "EETPT EE + test pulse amplitude G06 summary");
  meTestPulseAmplG06_[1] = dqmStore_->book1D(histo, histo, 100, 2000, 4000);

  if( meTestPulseAmplG12_[0] ) dqmStore_->removeElement( meTestPulseAmplG12_[0]->getName() );
  sprintf(histo, "EETPT EE - test pulse amplitude G12 summary");
  meTestPulseAmplG12_[0] = dqmStore_->book1D(histo, histo, 100, 2000, 4000);

  if( meTestPulseAmplG12_[1] ) dqmStore_->removeElement( meTestPulseAmplG12_[1]->getName() );
  sprintf(histo, "EETPT EE + test pulse amplitude G12 summary");
  meTestPulseAmplG12_[1] = dqmStore_->book1D(histo, histo, 100, 2000, 4000);

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

  if ( mePedestalOnlineMean_[0] ) dqmStore_->removeElement( mePedestalOnlineMean_[0]->getName() );
  mePedestalOnlineMean_[0] = 0;

  if ( mePedestalOnlineMean_[1] ) dqmStore_->removeElement( mePedestalOnlineMean_[1]->getName() );
  mePedestalOnlineMean_[1] = 0;

  if ( mePedestalOnlineRMS_[0] ) dqmStore_->removeElement( mePedestalOnlineRMS_[0]->getName() );
  mePedestalOnlineRMS_[0] = 0;

  if ( mePedestalOnlineRMS_[1] ) dqmStore_->removeElement( mePedestalOnlineRMS_[1]->getName() );
  mePedestalOnlineRMS_[1] = 0;

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

  if ( meLaserL1AmplOverPN_[0] ) dqmStore_->removeElement( meLaserL1AmplOverPN_[0]->getName() );
  meLaserL1AmplOverPN_[0] = 0;

  if ( meLaserL1AmplOverPN_[1] ) dqmStore_->removeElement( meLaserL1AmplOverPN_[1]->getName() );
  meLaserL1AmplOverPN_[1] = 0;

  if ( meLaserL1Timing_[0] ) dqmStore_->removeElement( meLaserL1Timing_[0]->getName() );
  meLaserL1Timing_[0] = 0;

  if ( meLaserL1Timing_[1] ) dqmStore_->removeElement( meLaserL1Timing_[1]->getName() );
  meLaserL1Timing_[1] = 0;

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

  if ( meLedL1AmplOverPN_[0] ) dqmStore_->removeElement( meLedL1AmplOverPN_[0]->getName() );
  meLedL1AmplOverPN_[0] = 0;

  if ( meLedL1AmplOverPN_[1] ) dqmStore_->removeElement( meLedL1AmplOverPN_[1]->getName() );
  meLedL1AmplOverPN_[1] = 0;

  if ( meLedL1Timing_[0] ) dqmStore_->removeElement( meLedL1Timing_[0]->getName() );
  meLedL1Timing_[0] = 0;

  if ( meLedL1Timing_[1] ) dqmStore_->removeElement( meLedL1Timing_[1]->getName() );
  meLedL1Timing_[1] = 0;

  if ( mePedestal_[0] ) dqmStore_->removeElement( mePedestal_[0]->getName() );
  mePedestal_[0] = 0;

  if ( mePedestal_[1] ) dqmStore_->removeElement( mePedestal_[1]->getName() );
  mePedestal_[1] = 0;

  if ( mePedestalErr_ ) dqmStore_->removeElement( mePedestalErr_->getName() );
  mePedestalErr_ = 0;

  if ( mePedestalPN_[0] ) dqmStore_->removeElement( mePedestalPN_[0]->getName() );
  mePedestalPN_[0] = 0;

  if ( mePedestalPN_[1] ) dqmStore_->removeElement( mePedestalPN_[1]->getName() );
  mePedestalPN_[1] = 0;

  if ( mePedestalPNErr_ ) dqmStore_->removeElement( mePedestalPNErr_->getName() );
  mePedestalPNErr_ = 0;

  if ( meTestPulse_[0] ) dqmStore_->removeElement( meTestPulse_[0]->getName() );
  meTestPulse_[0] = 0;

  if ( meTestPulse_[1] ) dqmStore_->removeElement( meTestPulse_[1]->getName() );
  meTestPulse_[1] = 0;

  if ( meTestPulseErr_ ) dqmStore_->removeElement( meTestPulseErr_->getName() );
  meTestPulseErr_ = 0;

  if ( meTestPulsePN_[0] ) dqmStore_->removeElement( meTestPulsePN_[0]->getName() );
  meTestPulsePN_[0] = 0;

  if ( meTestPulsePN_[1] ) dqmStore_->removeElement( meTestPulsePN_[1]->getName() );
  meTestPulsePN_[1] = 0;

  if ( meTestPulsePNErr_ ) dqmStore_->removeElement( meTestPulsePNErr_->getName() );
  meTestPulsePNErr_ = 0;

  if ( meTestPulseAmplG01_[0] ) dqmStore_->removeElement( meTestPulseAmplG01_[0]->getName() );
  meTestPulseAmplG01_[0] = 0;

  if ( meTestPulseAmplG01_[1] ) dqmStore_->removeElement( meTestPulseAmplG01_[1]->getName() );
  meTestPulseAmplG01_[1] = 0;

  if ( meTestPulseAmplG06_[0] ) dqmStore_->removeElement( meTestPulseAmplG06_[0]->getName() );
  meTestPulseAmplG06_[0] = 0;

  if ( meTestPulseAmplG06_[1] ) dqmStore_->removeElement( meTestPulseAmplG06_[1]->getName() );
  meTestPulseAmplG06_[1] = 0;

  if ( meTestPulseAmplG12_[0] ) dqmStore_->removeElement( meTestPulseAmplG12_[0]->getName() );
  meTestPulseAmplG12_[0] = 0;

  if ( meTestPulseAmplG12_[1] ) dqmStore_->removeElement( meTestPulseAmplG12_[1]->getName() );
  meTestPulseAmplG12_[1] = 0;

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

      meIntegrity_[0]->setBinContent( ix, iy, 6. );
      meIntegrity_[1]->setBinContent( ix, iy, 6. );
      meOccupancy_[0]->setBinContent( ix, iy, 0. );
      meOccupancy_[1]->setBinContent( ix, iy, 0. );
      meStatusFlags_[0]->setBinContent( ix, iy, 6. );
      meStatusFlags_[1]->setBinContent( ix, iy, 6. );
      mePedestalOnline_[0]->setBinContent( ix, iy, 6. );
      mePedestalOnline_[1]->setBinContent( ix, iy, 6. );
      mePedestalOnlineRMSMap_[0]->setBinContent( ix, iy, -1. );
      mePedestalOnlineRMSMap_[1]->setBinContent( ix, iy, -1. );

      meLaserL1_[0]->setBinContent( ix, iy, 6. );
      meLaserL1_[1]->setBinContent( ix, iy, 6. );
      meLedL1_[0]->setBinContent( ix, iy, 6. );
      meLedL1_[1]->setBinContent( ix, iy, 6. );
      mePedestal_[0]->setBinContent( ix, iy, 6. );
      mePedestal_[1]->setBinContent( ix, iy, 6. );
      meTestPulse_[0]->setBinContent( ix, iy, 6. );
      meTestPulse_[1]->setBinContent( ix, iy, 6. );

      meCosmic_[0]->setBinContent( ix, iy, 0. );
      meCosmic_[1]->setBinContent( ix, iy, 0. );
      meTiming_[0]->setBinContent( ix, iy, 6. );
      meTiming_[1]->setBinContent( ix, iy, 6. );

      meGlobalSummary_[0]->setBinContent( ix, iy, 6. );
      meGlobalSummary_[1]->setBinContent( ix, iy, 6. );

    }
  }

  for ( int ix = 1; ix <= 20; ix++ ) {
    for ( int iy = 1; iy <= 90; iy++ ) {

      meLaserL1PN_[0]->setBinContent( ix, iy, 6. );
      meLaserL1PN_[1]->setBinContent( ix, iy, 6. );
      mePedestalPN_[0]->setBinContent( ix, iy, 6. );
      mePedestalPN_[1]->setBinContent( ix, iy, 6. );
      meTestPulsePN_[0]->setBinContent( ix, iy, 6. );
      meTestPulsePN_[1]->setBinContent( ix, iy, 6. );

    }
  }

  for ( int ix = 1; ix <= 100; ix++ ) {
    for ( int iy = 1; iy <= 100; iy++ ) {
      meTriggerTowerEt_[0]->setBinContent( ix, iy, 0. );
      meTriggerTowerEt_[1]->setBinContent( ix, iy, 0. );
      meTriggerTowerEmulError_[0]->setBinContent( ix, iy, 6. );
      meTriggerTowerEmulError_[1]->setBinContent( ix, iy, 6. );
      meTriggerTowerTiming_[0]->setBinContent( ix, iy, -1 );
      meTriggerTowerTiming_[1]->setBinContent( ix, iy, -1 );
    }
  }

  meIntegrity_[0]->setEntries( 0 );
  meIntegrity_[1]->setEntries( 0 );
  meIntegrityErr_->Reset();
  meOccupancy_[0]->setEntries( 0 );
  meOccupancy_[1]->setEntries( 0 );
  meOccupancy1D_->Reset();
  meStatusFlags_[0]->setEntries( 0 );
  meStatusFlags_[1]->setEntries( 0 );
  meStatusFlagsErr_->Reset();
  mePedestalOnline_[0]->setEntries( 0 );
  mePedestalOnline_[1]->setEntries( 0 );
  mePedestalOnlineErr_->Reset();
  mePedestalOnlineMean_[0]->Reset();
  mePedestalOnlineMean_[1]->Reset();
  mePedestalOnlineRMS_[0]->Reset();
  mePedestalOnlineRMS_[1]->Reset();
  meLaserL1_[0]->setEntries( 0 );
  meLaserL1_[1]->setEntries( 0 );
  meLaserL1Err_->Reset();
  meLaserL1PN_[0]->setEntries( 0 );
  meLaserL1PN_[1]->setEntries( 0 );
  meLaserL1PNErr_->Reset();
  meLaserL1AmplOverPN_[0]->Reset();
  meLaserL1AmplOverPN_[1]->Reset();
  meLaserL1Timing_[0]->Reset();
  meLaserL1Timing_[1]->Reset();
  meLedL1_[0]->setEntries( 0 );
  meLedL1_[1]->setEntries( 0 );
  meLedL1Err_->Reset();
  meLedL1PN_[0]->setEntries( 0 );
  meLedL1PN_[1]->setEntries( 0 );
  meLedL1PNErr_->Reset();
  meLedL1AmplOverPN_[0]->Reset();
  meLedL1AmplOverPN_[1]->Reset();
  meLedL1Timing_[0]->Reset();
  meLedL1Timing_[1]->Reset();
  mePedestal_[0]->setEntries( 0 );
  mePedestal_[1]->setEntries( 0 );
  mePedestalErr_->Reset();
  mePedestalPN_[0]->setEntries( 0 );
  mePedestalPN_[1]->setEntries( 0 );
  mePedestalPNErr_->Reset();
  meTestPulse_[0]->setEntries( 0 );
  meTestPulse_[1]->setEntries( 0 );
  meTestPulseErr_->Reset();
  meTestPulsePN_[0]->setEntries( 0 );
  meTestPulsePN_[1]->setEntries( 0 );
  meTestPulsePNErr_->Reset();
  meTestPulseAmplG01_[0]->Reset();
  meTestPulseAmplG01_[1]->Reset();
  meTestPulseAmplG06_[0]->Reset();
  meTestPulseAmplG06_[1]->Reset();
  meTestPulseAmplG12_[0]->Reset();
  meTestPulseAmplG12_[1]->Reset();

  meCosmic_[0]->setEntries( 0 );
  meCosmic_[1]->setEntries( 0 );
  meTiming_[0]->setEntries( 0 );
  meTiming_[1]->setEntries( 0 );
  meTriggerTowerEt_[0]->setEntries( 0 );
  meTriggerTowerEt_[1]->setEntries( 0 );
  meTriggerTowerEmulError_[0]->setEntries( 0 );
  meTriggerTowerEmulError_[1]->setEntries( 0 );
  meTriggerTowerTiming_[0]->setEntries( 0 );
  meTriggerTowerTiming_[1]->setEntries( 0 );

  meGlobalSummary_[0]->setEntries( 0 );
  meGlobalSummary_[1]->setEntries( 0 );

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

    // fill the gain value priority map<id,priority>
    map<float,float> priority;
    priority.insert( pair<float,float>(0,3) );
    priority.insert( pair<float,float>(1,1) );
    priority.insert( pair<float,float>(2,2) );
    priority.insert( pair<float,float>(3,2) );
    priority.insert( pair<float,float>(4,3) );
    priority.insert( pair<float,float>(5,1) );

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

          if ( eeic ) {

            me = eeic->meg01_[ism-1];

            if ( me ) {

              float xval = me->getBinContent( ix, iy );

              if ( ism >= 1 && ism <= 9 ) {
                if ( Numbers::validEE(ism, 101 - jx, jy) ) meIntegrity_[0]->setBinContent( 101 - jx, jy, xval );
              } else {
                if ( Numbers::validEE(ism, jx, jy) ) meIntegrity_[1]->setBinContent( jx, jy, xval );
              }
              if ( xval == 0 ) meIntegrityErr_->Fill( ism );

            }

            h2 = eeic->h_[ism-1];

            if ( h2 ) {

              float xval = h2->GetBinContent( ix, iy );

              if ( ism >= 1 && ism <= 9 ) {
                if ( xval != 0 ) {
                  if ( Numbers::validEE(ism, 101 - jx, jy) ) meOccupancy_[0]->setBinContent( 101 - jx, jy, xval );
                }
              } else {
                if ( xval != 0 ) {
                  if ( Numbers::validEE(ism, jx, jy) ) meOccupancy_[1]->setBinContent( jx, jy, xval );
                }
              }
              meOccupancy1D_->Fill( ism, xval );

            }

          }

          if ( eepoc ) {

            me = eepoc->meg03_[ism-1];

            if ( me ) {

              float xval = me->getBinContent( ix, iy );

              if ( ism >= 1 && ism <= 9 ) {
                if ( Numbers::validEE(ism, 101 - jx, jy) ) mePedestalOnline_[0]->setBinContent( 101 - jx, jy, xval );
              } else {
                if ( Numbers::validEE(ism, jx, jy) ) mePedestalOnline_[1]->setBinContent( jx, jy, xval );
              }
              if ( xval == 0 ) mePedestalOnlineErr_->Fill( ism );

            }

          }

          float num01, mean01, rms01;
          bool update01 = UtilsClient::getBinStatistics(hpot01_[ism-1], ix, iy, num01, mean01, rms01);

          if ( update01 ) {

            if ( ism >= 1 && ism <= 9 ) {
              if ( Numbers::validEE(ism, 101 - jx, jy) ) {
                mePedestalOnlineRMSMap_[0]->setBinContent( 101 - jx, jy, rms01 );
                mePedestalOnlineRMS_[0]->Fill( rms01 );
                mePedestalOnlineMean_[0]->Fill( mean01 );
              }
            } else {
              if ( Numbers::validEE(ism, jx, jy) ) {
                mePedestalOnlineRMSMap_[1]->setBinContent( jx, jy, rms01 );
                mePedestalOnlineRMS_[1]->Fill( rms01 );
                mePedestalOnlineMean_[1]->Fill( mean01 );
              }
            }
          }

          if ( eelc ) {

            me = eelc->meg01_[ism-1];

            if ( me ) {

              float xval = me->getBinContent( ix, iy );

              if ( me->getEntries() != 0 ) {
                if ( ism >= 1 && ism <= 9 ) {
                  if ( Numbers::validEE(ism, 101 - jx, jy) ) meLaserL1_[0]->setBinContent( 101 - jx, jy, xval );
                } else {
                  if ( Numbers::validEE(ism, jx, jy) ) meLaserL1_[1]->setBinContent( jx, jy, xval );
                }
                if ( xval == 0 ) meLaserL1Err_->Fill( ism );
              }

            }

          }

          if ( eeldc ) {

            me = eeldc->meg01_[ism-1];

            if ( me ) {

              float xval = me->getBinContent( ix, iy );

              if ( me->getEntries() != 0 ) {
                if ( ism >= 1 && ism <= 9 ) {
                  if ( Numbers::validEE(ism, 101 - jx, jy) ) meLedL1_[0]->setBinContent( 101 - jx, jy, xval );
                } else {
                  if ( Numbers::validEE(ism, jx, jy) ) meLedL1_[1]->setBinContent( jx, jy, xval );
                }
                if ( xval == 0 ) meLedL1Err_->Fill( ism );
              }

            }

          }

          if ( eepc ) {

            me_01 = eepc->meg01_[ism-1];
            me_02 = eepc->meg02_[ism-1];
            me_03 = eepc->meg03_[ism-1];

            if ( me_01 && me_02 && me_03 ) {
              float xval=2;
              float val_01=me_01->getBinContent(ix,iy);
              float val_02=me_02->getBinContent(ix,iy);
              float val_03=me_03->getBinContent(ix,iy);

              vector<float> maskedVal, unmaskedVal;
              (val_01>=3&&val_01<=5) ? maskedVal.push_back(val_01) : unmaskedVal.push_back(val_01);
              (val_02>=3&&val_02<=5) ? maskedVal.push_back(val_02) : unmaskedVal.push_back(val_02);
              (val_03>=3&&val_03<=5) ? maskedVal.push_back(val_03) : unmaskedVal.push_back(val_03);

              float brightColor=6, darkColor=6;
              float maxPriority=-1;

              vector<float>::const_iterator Val;
              for(Val=unmaskedVal.begin(); Val<unmaskedVal.end(); Val++) {
                if(priority[*Val]>maxPriority) brightColor=*Val;
              }
              maxPriority=-1;
              for(Val=maskedVal.begin(); Val<maskedVal.end(); Val++) {
                if(priority[*Val]>maxPriority) darkColor=*Val;
              }
              if(unmaskedVal.size()==3) xval = brightColor;
              else if(maskedVal.size()==3) xval = darkColor;
              else {
                if(brightColor==1 && darkColor==5) xval = 5;
                else xval = brightColor;
              }

              if ( me_01->getEntries() != 0 && me_02->getEntries() != 0 && me_03->getEntries() != 0 ) {
                if ( ism >= 1 && ism <= 9 ) {
                  if ( Numbers::validEE(ism, 101 - jx, jy) ) mePedestal_[0]->setBinContent( 101 - jx, jy, xval );
                } else {
                  if ( Numbers::validEE(ism, jx, jy) ) mePedestal_[1]->setBinContent( jx, jy, xval );
                }
                if ( xval == 0 ) mePedestalErr_->Fill( ism );
              }

            }

          }

          if ( eetpc ) {

            me_01 = eetpc->meg01_[ism-1];
            me_02 = eetpc->meg02_[ism-1];
            me_03 = eetpc->meg03_[ism-1];

            if ( me_01 && me_02 && me_03 ) {
              float xval=2;
              float val_01=me_01->getBinContent(ix,iy);
              float val_02=me_02->getBinContent(ix,iy);
              float val_03=me_03->getBinContent(ix,iy);

              vector<float> maskedVal, unmaskedVal;
              (val_01>=3&&val_01<=5) ? maskedVal.push_back(val_01) : unmaskedVal.push_back(val_01);
              (val_02>=3&&val_02<=5) ? maskedVal.push_back(val_02) : unmaskedVal.push_back(val_02);
              (val_03>=3&&val_03<=5) ? maskedVal.push_back(val_03) : unmaskedVal.push_back(val_03);

              float brightColor=6, darkColor=6;
              float maxPriority=-1;

              vector<float>::const_iterator Val;
              for(Val=unmaskedVal.begin(); Val<unmaskedVal.end(); Val++) {
                if(priority[*Val]>maxPriority) brightColor=*Val;
              }
              maxPriority=-1;
              for(Val=maskedVal.begin(); Val<maskedVal.end(); Val++) {
                if(priority[*Val]>maxPriority) darkColor=*Val;
              }
              if(unmaskedVal.size()==3) xval = brightColor;
              else if(maskedVal.size()==3) xval = darkColor;
              else {
                if(brightColor==1 && darkColor==5) xval = 5;
                else xval = brightColor;
              }

              if ( me_01->getEntries() != 0 && me_02->getEntries() != 0 && me_03->getEntries() != 0 ) {
                if ( ism >= 1 && ism <= 9 ) {
                  if ( Numbers::validEE(ism, 101 - jx, jy) ) meTestPulse_[0]->setBinContent( 101 - jx, jy, xval );
                } else {
                  if ( Numbers::validEE(ism, jx, jy) ) meTestPulse_[1]->setBinContent( jx, jy, xval );
                }
                if ( xval == 0 ) meTestPulseErr_->Fill( ism );
              }

            }

          }

          if ( eecc ) {

            h2d = eecc->h02_[ism-1];

            if ( h2d ) {

              float xval = h2d->GetBinContent( ix, iy );

              if ( ism >= 1 && ism <= 9 ) {
                if ( xval != 0 ) {
                  if ( Numbers::validEE(ism, 101 - jx, jy) ) meCosmic_[0]->setBinContent( 101 - jx, jy, xval );
                }
              } else {
                if ( xval != 0 ) {
                  if ( Numbers::validEE(ism, jx, jy) ) meCosmic_[1]->setBinContent( jx, jy, xval );
                }
              }

            }

          }

          if ( eetmc ) {

            me = eetmc->meg01_[ism-1];

            if ( me ) {

              float xval = me->getBinContent( ix, iy );

              if ( ism >= 1 && ism <= 9 ) {
                if ( Numbers::validEE(ism, 101 - jx, jy) ) meTiming_[0]->setBinContent( 101 - jx, jy, xval );
              } else {
                if ( Numbers::validEE(ism, jx, jy) ) meTiming_[1]->setBinContent( jx, jy, xval );
              }

            }

          }

        }
      }

      for ( int ix = 1; ix <= 50; ix++ ) {
        for ( int iy = 1; iy <= 50; iy++ ) {

          int jx = ix + Numbers::ix0EE(ism);
          int jy = iy + Numbers::iy0EE(ism);

          if ( eesfc ) {

            me = eesfc->meh01_[ism-1];

            if ( me ) {

              float xval = 6;

              if ( me->getBinContent( ix, iy ) == 6 ) xval = 2;
              if ( me->getBinContent( ix, iy ) == 0 ) xval = 1;
              if ( me->getBinContent( ix, iy ) > 0 ) xval = 0;

              if ( me->getEntries() != 0 ) {
              if ( ism >= 1 && ism <= 9 ) {
                if ( Numbers::validEE(ism, 101 - jx, jy) ) meStatusFlags_[0]->setBinContent( 101 - jx, jy, xval );
              } else {
                if ( Numbers::validEE(ism, jx, jy) ) meStatusFlags_[1]->setBinContent( jx, jy, xval );
              }
              if ( xval == 0 ) meStatusFlagsErr_->Fill( ism );
              }

            }

          }

          float num01, mean01, rms01;
          bool update01 = UtilsClient::getBinStatistics(httt01_[ism-1], jx, jy, num01, mean01, rms01);

          if ( update01 ) {
            if ( ism >= 1 && ism <= 9 ) meTriggerTowerEt_[0]->setBinContent( 101 - jx, jy, mean01 );
            else meTriggerTowerEt_[1]->setBinContent( jx, jy, mean01 );
          }

          if ( eetttc ) {

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

            int side = ( ism >= 1 && ism <= 9 ) ? 0 : 1;

            if ( eelc ) {

              MonitorElement *meg = eelc->meg01_[ism-1];

              float xval = 2;
              if ( meg ) xval = meg->getBinContent( ix, iy );

              // exclude channels without laser data (yellow in the quality map)
              if( xval != 2 && xval != 5 ) { 
                
                int RtHalf = 0;
                // EE-05
                if ( ism ==  8 && ix > 50 ) RtHalf = 1;
                
                // EE+05
                if ( ism == 17 && ix > 50 ) RtHalf = 1;
                
                //! Ampl / PN
                // L1A
                me = eelc->meaopn01_[ism-1];

                if( me && RtHalf == 0 ) {
                  meLaserL1AmplOverPN_[side]->Fill( me->getBinContent( ic ) );
                }

                // L1B
                me = eelc->meaopn05_[ism-1];
                if( me && RtHalf == 0 ) {
                  meLaserL1AmplOverPN_[side]->Fill( me->getBinContent( ic ) );
                }
                
                //! timing
                // L1A
                me = eelc->met01_[ism-1];
            
                if( me && RtHalf == 0 ) {
                  meLaserL1Timing_[side]->Fill( me->getBinContent( ic ) );
                }
            
                // L1B
                me = eelc->met05_[ism-1];
                
                if ( me && RtHalf == 1 ) {
                  meLaserL1Timing_[side]->Fill( me->getBinContent( ic ) );
                }

              }
              
            }

            if ( eeldc ) {

              MonitorElement *meg = eeldc->meg01_[ism-1];

              float xval = 2;
              if ( meg )  xval = meg->getBinContent( ix, iy );

              // exclude channels without led data (yellow in the quality map)
              if( xval != 2 && xval != 5 ) { 
                
                int RtHalf = 0;
                // EE-05
                if ( ism ==  8 && ix > 50 ) RtHalf = 1;
                
                // EE+05
                if ( ism == 17 && ix > 50 ) RtHalf = 1;
                
                //! Ampl / PN
                // L1A
                me = eeldc->meaopn01_[ism-1];

                if( me && RtHalf == 0 ) {
                  meLaserL1AmplOverPN_[side]->Fill( me->getBinContent( ic ) );
                }
          
                // L1B
                me = eeldc->meaopn05_[ism-1];
                if( me && RtHalf == 0 ) {
                  meLaserL1AmplOverPN_[side]->Fill( me->getBinContent( ic ) );
                }
                
                //! timing
                // L1A
                me = eeldc->met01_[ism-1];
            
                if( me && RtHalf == 0 ) {
                  meLaserL1Timing_[side]->Fill( me->getBinContent( ic ) );
                }
            
                // L1B (rectangular)
                me = eeldc->met05_[ism-1];
                
                if ( me && RtHalf == 1 ) {
                  meLaserL1Timing_[side]->Fill( me->getBinContent( ic ) );
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
                
                    meTestPulseAmplG01_[side]->Fill( me->getBinContent( ic ) );

                  }

                }
                
              }

              if ( meg02 ) {
            
                float xval02 = meg02->getBinContent( ix, iy );

                if ( xval02 != 2 && xval02 != 5 ) {
                  
                  me = eetpc->mea02_[ism-1];
                  
                  if ( me ) {
                    
                    meTestPulseAmplG06_[side]->Fill( me->getBinContent( ic ) );
                    
                  }
                  
                }
                
              }
              
              if ( meg03 ) {
                
                float xval03 = meg03->getBinContent( ix, iy );
                
                if ( xval03 != 2 && xval03 != 5 ) {
                  
                  me = eetpc->mea03_[ism-1];
                  
                  if ( me ) {
                    
                    meTestPulseAmplG12_[side]->Fill( me->getBinContent( ic ) );
                    
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

      if(meIntegrity_[0] && mePedestalOnline_[0] && meLaserL1_[0] && meLedL1_[0] && meTiming_[0] && meStatusFlags_[0] && meTriggerTowerEmulError_[0]) {

        float xval = 6;
        float val_in = meIntegrity_[0]->getBinContent(jx,jy);
        float val_po = mePedestalOnline_[0]->getBinContent(jx,jy);
        float val_ls = meLaserL1_[0]->getBinContent(jx,jy);
        float val_ld = meLedL1_[0]->getBinContent(jx,jy);
        float val_tm = meTiming_[0]->getBinContent(jx,jy);
        float val_sf = meStatusFlags_[0]->getBinContent(jx,jy);
	// float val_ee = meTriggerTowerEmulError_[0]->getBinContent(jx,jy); // removed temporarily from the global summary
	float val_ee = 1;

        // turn each dark color (masked channel) to bright green
        // for laser & timing & trigger turn also yellow into bright green

        //  0/3 = red/dark red
        //  1/4 = green/dark green
        //  2/5 = yellow/dark yellow
        //  6   = unknown

        if(             val_in==3 || val_in==4 || val_in==5) val_in=1;
        if(             val_po==3 || val_po==4 || val_po==5) val_po=1;
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

      if(meIntegrity_[1] && mePedestalOnline_[1] && meLaserL1_[1] && meLedL1_[1] && meTiming_[1] && meStatusFlags_[1] && meTriggerTowerEmulError_[1]) {

        float xval = 6;
        float val_in = meIntegrity_[1]->getBinContent(jx,jy);
        float val_po = mePedestalOnline_[1]->getBinContent(jx,jy);
        float val_ls = meLaserL1_[1]->getBinContent(jx,jy);
        float val_ld = meLedL1_[1]->getBinContent(jx,jy);
        float val_tm = meTiming_[1]->getBinContent(jx,jy);
        float val_sf = meStatusFlags_[1]->getBinContent(jx,jy);
        // float val_ee = meTriggerTowerEmulError_[1]->getBinContent(jx,jy); // removed temporarily from the global summary
	float val_ee = 1;

        // turn each dark color to bright green
        // for laser & timing & trigger turn also yellow into bright green

        //  0/3 = red/dark red
        //  1/4 = green/dark green
        //  2/5 = yellow/dark yellow
        //  6   = unknown

        if(             val_in==3 || val_in==4 || val_in==5) val_in=1;
        if(             val_po==3 || val_po==4 || val_po==5) val_po=1;
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

