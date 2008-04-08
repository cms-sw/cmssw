/*
 * \file EESummaryClient.cc
 *
 * $Date: 2008/04/08 15:06:25 $
 * $Revision: 1.106 $
 * \author G. Della Ricca
 *
*/

#include <memory>
#include <iostream>
#include <iomanip>
#include <map>
#include <math.h>

#include "TCanvas.h"
#include "TStyle.h"
#include "TGraph.h"
#include "TLine.h"

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

EESummaryClient::EESummaryClient(const ParameterSet& ps){

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
  meLaserL1_[0]        = 0;
  meLaserL1_[1]        = 0;
  meLaserL1PN_[0]      = 0;
  meLaserL1PN_[1]      = 0;
  meLedL1_[0]          = 0;
  meLedL1_[1]          = 0;
  meLedL1PN_[0]        = 0;
  meLedL1PN_[1]        = 0;
  mePedestal_[0]       = 0;
  mePedestal_[1]       = 0;
  mePedestalPN_[0]     = 0;
  mePedestalPN_[1]     = 0;
  meTestPulse_[0]      = 0;
  meTestPulse_[1]      = 0;
  meTestPulsePN_[0]    = 0;
  meTestPulsePN_[1]    = 0;
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

}

EESummaryClient::~EESummaryClient(){

}

void EESummaryClient::beginJob(DQMStore* dqmStore){

  dqmStore_ = dqmStore;

  if ( debug_ ) cout << "EESummaryClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EESummaryClient::beginRun(void){

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

  if( meCosmic_[0] ) dqmStore_->removeElement( meCosmic_[0]->getName() );
  sprintf(histo, "EECT EE - quality summary");
  meCosmic_[0] = dqmStore_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
  meCosmic_[0]->setAxisTitle("jx", 1);
  meCosmic_[0]->setAxisTitle("jy", 2);

  if( meCosmic_[1] ) dqmStore_->removeElement( meCosmic_[1]->getName() );
  sprintf(histo, "EECT EE + quality summary");
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

  // summary for DQM GUI

  dqmStore_->setCurrentFolder( prefixME_ + "/EventInfo" );

  MonitorElement* me;

  sprintf(histo, "errorSummaryXY_EEM");
  me = dqmStore_->book2D(histo, histo, 20, 0., 20., 20, 0., 20);
  me->setAxisTitle("jx", 1);
  me->setAxisTitle("jy", 2);

  sprintf(histo, "errorSummaryXY_EEP");
  me = dqmStore_->book2D(histo, histo, 20, 0., 20., 20, 0., 20);
  me->setAxisTitle("jx", 1);
  me->setAxisTitle("jy", 2);

}

void EESummaryClient::cleanup(void) {

  if ( ! enableCleanup_ ) return;

  dqmStore_->setCurrentFolder( prefixME_ + "/EESummaryClient" );

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

  if ( meGlobalSummary_[0] ) dqmStore_->removeElement( meGlobalSummary_[0]->getName() );
  meGlobalSummary_[0] = 0;

  if ( meGlobalSummary_[1] ) dqmStore_->removeElement( meGlobalSummary_[1]->getName() );
  meGlobalSummary_[1] = 0;

  // summary for DQM GUI

  dqmStore_->setCurrentFolder( prefixME_ + "/EventInfo" );

  // to be done

}

bool EESummaryClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov) {

  bool status = true;

  return status;

}

void EESummaryClient::analyze(void){

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( debug_ ) cout << "EESummaryClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;
  }

  for ( int ix = 1; ix <= 100; ix++ ) {
    for ( int iy = 1; iy <= 100; iy++ ) {

      meIntegrity_[0]->setBinContent( ix, iy, -1. );
      meIntegrity_[1]->setBinContent( ix, iy, -1. );
      meOccupancy_[0]->setBinContent( ix, iy, 0. );
      meOccupancy_[1]->setBinContent( ix, iy, 0. );
      meStatusFlags_[0]->setBinContent( ix, iy, -1. );
      meStatusFlags_[1]->setBinContent( ix, iy, -1. );
      mePedestalOnline_[0]->setBinContent( ix, iy, -1. );
      mePedestalOnline_[1]->setBinContent( ix, iy, -1. );

      meLaserL1_[0]->setBinContent( ix, iy, -1. );
      meLaserL1_[1]->setBinContent( ix, iy, -1. );
      meLedL1_[0]->setBinContent( ix, iy, -1. );
      meLedL1_[1]->setBinContent( ix, iy, -1. );
      mePedestal_[0]->setBinContent( ix, iy, -1. );
      mePedestal_[1]->setBinContent( ix, iy, -1. );
      meTestPulse_[0]->setBinContent( ix, iy, -1. );
      meTestPulse_[1]->setBinContent( ix, iy, -1. );

      meCosmic_[0]->setBinContent( ix, iy, 0. );
      meCosmic_[1]->setBinContent( ix, iy, 0. );
      meTiming_[0]->setBinContent( ix, iy, -1. );
      meTiming_[1]->setBinContent( ix, iy, -1. );

      meGlobalSummary_[0]->setBinContent( ix, iy, -1. );
      meGlobalSummary_[1]->setBinContent( ix, iy, -1. );

    }
  }

  for ( int ix = 1; ix <= 20; ix++ ) {
    for ( int iy = 1; iy <= 90; iy++ ) {

      meLaserL1PN_[0]->setBinContent( ix, iy, -1. );
      meLaserL1PN_[1]->setBinContent( ix, iy, -1. );
      mePedestalPN_[0]->setBinContent( ix, iy, -1. );
      mePedestalPN_[1]->setBinContent( ix, iy, -1. );
      meTestPulsePN_[0]->setBinContent( ix, iy, -1. );
      meTestPulsePN_[1]->setBinContent( ix, iy, -1. );

    }
  }

  for ( int ix = 1; ix <= 100; ix++ ) {
    for ( int iy = 1; iy <= 100; iy++ ) {
      meTriggerTowerEt_[0]->setBinContent( ix, iy, 0. );
      meTriggerTowerEt_[1]->setBinContent( ix, iy, 0. );
      meTriggerTowerEmulError_[0]->setBinContent( ix, iy, -1. );
      meTriggerTowerEmulError_[1]->setBinContent( ix, iy, -1. );
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

  meLaserL1_[0]->setEntries( 0 );
  meLaserL1_[1]->setEntries( 0 );
  meLaserL1Err_->Reset();
  meLaserL1PN_[0]->setEntries( 0 );
  meLaserL1PN_[1]->setEntries( 0 );
  meLaserL1PNErr_->Reset();
  meLedL1_[0]->setEntries( 0 );
  meLedL1_[1]->setEntries( 0 );
  meLedL1Err_->Reset();
  meLedL1PN_[0]->setEntries( 0 );
  meLedL1PN_[1]->setEntries( 0 );
  meLedL1PNErr_->Reset();
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

  meCosmic_[0]->setEntries( 0 );
  meCosmic_[1]->setEntries( 0 );
  meTiming_[0]->setEntries( 0 );
  meTiming_[1]->setEntries( 0 );
  meTriggerTowerEt_[0]->setEntries( 0 );
  meTriggerTowerEt_[1]->setEntries( 0 );
  meTriggerTowerEmulError_[0]->setEntries( 0 );
  meTriggerTowerEmulError_[1]->setEntries( 0 );

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

    MonitorElement* me;
    MonitorElement *me_01, *me_02, *me_03;
//    MonitorElement *me_04, *me_05;

    TH2F* h2;
    TProfile2D* h2d;

    // fill the gain value priority map<id,priority>
    std::map<float,float> priority;
    priority.insert( make_pair(0,3) );
    priority.insert( make_pair(1,1) );
    priority.insert( make_pair(2,2) );
    priority.insert( make_pair(3,2) );
    priority.insert( make_pair(4,3) );
    priority.insert( make_pair(5,1) );

    for ( unsigned int i=0; i<superModules_.size(); i++ ) {

      int ism = superModules_[i];

      for ( int ix = 1; ix <= 50; ix++ ) {
        for ( int iy = 1; iy <= 50; iy++ ) {

          int jx = ix + Numbers::ix0EE(ism);
          int jy = iy + Numbers::iy0EE(ism);

          if ( eeic ) {

            me = eeic->meg01_[ism-1];

            if ( me ) {

              float xval = me->getBinContent( ix, iy );

              if ( ism >= 1 && ism <= 9 ) {
                if ( Numbers::validEE(ism, 101 - jx, jy) ) meIntegrity_[0]->setBinContent( jx, jy, xval );
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
                  if ( Numbers::validEE(ism, 101 - jx, jy) ) meOccupancy_[0]->setBinContent( jx, jy, xval );
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
                if ( Numbers::validEE(ism, 101 - jx, jy) ) mePedestalOnline_[0]->setBinContent( jx, jy, xval );
              } else {
                if ( Numbers::validEE(ism, jx, jy) ) mePedestalOnline_[1]->setBinContent( jx, jy, xval );
              }
              if ( xval == 0 ) mePedestalOnlineErr_->Fill( ism );

            }

          }

          if ( eelc ) {

            me = eelc->meg01_[ism-1];

            if ( me ) {

              float xval = me->getBinContent( ix, iy );

              if ( me->getEntries() != 0 ) {
                if ( ism >= 1 && ism <= 9 ) {
                  if ( Numbers::validEE(ism, 101 - jx, jy) ) meLaserL1_[0]->setBinContent( jx, jy, xval );
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
                  if ( Numbers::validEE(ism, 101 - jx, jy) ) meLedL1_[0]->setBinContent( jx, jy, xval );
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

            if (me_01 && me_02 && me_03 ) {
              float xval=2;
              float val_01=me_01->getBinContent(ix,iy);
              float val_02=me_02->getBinContent(ix,iy);
              float val_03=me_03->getBinContent(ix,iy);

              std::vector<float> maskedVal, unmaskedVal;
              (val_01>2) ? maskedVal.push_back(val_01) : unmaskedVal.push_back(val_01);
              (val_02>2) ? maskedVal.push_back(val_02) : unmaskedVal.push_back(val_02);
              (val_03>2) ? maskedVal.push_back(val_03) : unmaskedVal.push_back(val_03);

              float brightColor=-1, darkColor=-1;
              float maxPriority=-1;
              std::vector<float>::const_iterator Val;
              for(Val=unmaskedVal.begin(); Val<unmaskedVal.end(); Val++) {
                if(priority[*Val]>maxPriority) brightColor=*Val;
              }
              maxPriority=-1;
              for(Val=maskedVal.begin(); Val<maskedVal.end(); Val++) {
                if(priority[*Val]>maxPriority) darkColor=*Val;
              }
              if(unmaskedVal.size()==3)  xval = brightColor;
              else if(maskedVal.size()==3)  xval = darkColor;
              else {
                if(brightColor==1 && darkColor==5) xval = 5;
                else xval = brightColor;
              }

              if ( me_01->getEntries() != 0 && me_02->getEntries() != 0 && me_03->getEntries() != 0 ) {
                if ( ism >= 1 && ism <= 9 ) {
                  if ( Numbers::validEE(ism, 101 - jx, jy) ) mePedestal_[0]->setBinContent( jx, jy, xval );
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

            if (me_01 && me_02 && me_03 ) {
              float xval=2;
              float val_01=me_01->getBinContent(ix,iy);
              float val_02=me_02->getBinContent(ix,iy);
              float val_03=me_03->getBinContent(ix,iy);

              std::vector<float> maskedVal, unmaskedVal;
              (val_01>2) ? maskedVal.push_back(val_01) : unmaskedVal.push_back(val_01);
              (val_02>2) ? maskedVal.push_back(val_02) : unmaskedVal.push_back(val_02);
              (val_03>2) ? maskedVal.push_back(val_03) : unmaskedVal.push_back(val_03);

              float brightColor=-1, darkColor=-1;
              float maxPriority=-1;
              std::vector<float>::const_iterator Val;
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
                  if ( Numbers::validEE(ism, 101 - jx, jy) ) meTestPulse_[0]->setBinContent( jx, jy, xval );
                } else {
                  if ( Numbers::validEE(ism, jx, jy) ) meTestPulse_[1]->setBinContent( jx, jy, xval );
                }
                if ( xval == 0 ) meTestPulseErr_->Fill( ism );
              }

            }

          }

          if ( eecc ) {

            h2d = eecc->h01_[ism-1];

            if ( h2d ) {

              float xval = h2d->GetBinContent( ix, iy );

              if ( ism >= 1 && ism <= 9 ) {
                if ( xval != 0 ) {
                  if ( Numbers::validEE(ism, 101 - jx, jy) ) meCosmic_[0]->setBinContent( jx, jy, xval );
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
                if ( Numbers::validEE(ism, 101 - jx, jy) ) meTiming_[0]->setBinContent( jx, jy, xval );
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

              float xval = -1;

              if ( me->getBinContent( ix, iy ) == -1 ) xval = 2;
              if ( me->getBinContent( ix, iy ) == 0 ) xval = 1;
              if ( me->getBinContent( ix, iy ) > 0 ) xval = 0;

              if ( me->getEntries() != 0 ) {
              if ( ism >= 1 && ism <= 9 ) {
                if ( Numbers::validEE(ism, 101 - jx, jy) ) meStatusFlags_[0]->setBinContent( jx, jy, xval );
              } else {
                if ( Numbers::validEE(ism, jx, jy) ) meStatusFlags_[1]->setBinContent( jx, jy, xval );
              }
              if ( xval == 0 ) meStatusFlagsErr_->Fill( ism );
              }

            }

          }

          if ( eetttc ) {

            me = eetttc->me_h01_[ism-1];

            bool hasRealDigi = false;

            if ( me ) {

              float xval = me->getBinContent( ix, iy );

              TProfile2D* obj = UtilsClient::getHisto<TProfile2D*>(me);
              if(obj && obj->GetBinEntries(obj->GetBin( ix, iy ))!=0) hasRealDigi = true;

              if ( ism >= 1 && ism <= 9 ) {
                if ( xval != 0 ) {
                  meTriggerTowerEt_[0]->setBinContent( jx, jy, xval );
                }
              } else {
                if ( xval != 0 ) {
                  meTriggerTowerEt_[1]->setBinContent( jx, jy, xval );
                }
              }

            }

            h2 = eetttc->l01_[ism-1];

            if ( h2 ) {

              float xval = -1;
              float emulErrorVal = h2->GetBinContent( ix, iy );

              if(!hasRealDigi) xval = 2;
              else if(hasRealDigi && emulErrorVal!=0) xval = 0;
              else xval = 1;

              // see fix below
              if ( xval == 2 ) continue;

              if ( ism >= 1 && ism <= 9 ) {
                meTriggerTowerEmulError_[0]->setBinContent( jx, jy, xval );
              } else {
                meTriggerTowerEmulError_[1]->setBinContent( jx, jy, xval );
              }

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
              if ( meTriggerTowerEmulError_[0]->getBinContent( jx, jy ) == -1 ) {
                if ( Numbers::validEE(ism, 101 - jx, jy) ) meTriggerTowerEmulError_[0]->setBinContent( jx, jy, 2 );
              }
            } else {
              if ( meTriggerTowerEmulError_[1]->getBinContent( jx, jy ) == -1 ) {
                if ( Numbers::validEE(ism, jx, jy) ) meTriggerTowerEmulError_[1]->setBinContent( jx, jy, 2 );
              }
            }

          }

        }
      }

    }

  } // loop on clients

  // The global-summary
  // right now a summary of Integrity and PO
  int nGlobalErrors = 0, nGlobalErrorsEEM = 0, nGlobalErrorsEEP = 0;
  int nValidChannels = 0, nValidChannelsEEM = 0, nValidChannelsEEP = 0;
  for ( int jx = 1; jx <= 100; jx++ ) {
    for ( int jy = 1; jy <= 100; jy++ ) {

      if(meIntegrity_[0] && mePedestalOnline_[0] && meLaserL1_[0] && meTiming_[0] && meStatusFlags_[0] && meTriggerTowerEmulError_[0]) {

        float xval = -1;
        float val_in = meIntegrity_[0]->getBinContent(jx,jy);
        float val_po = mePedestalOnline_[0]->getBinContent(jx,jy);
        float val_ls = meLaserL1_[0]->getBinContent(jx,jy);
        float val_tm = meTiming_[0]->getBinContent(jx,jy);
        float val_sf = meStatusFlags_[0]->getBinContent(jx,jy);
        float val_ee = meTriggerTowerEmulError_[0]->getBinContent(jx,jy);

        // turn each dark color (masked channel) to bright green
        // for laser & timing turn also yellow into bright green
        if(val_in> 2) val_in=1;
        if(val_po> 2) val_po=1;
        if(val_ls>=2) val_ls=1;
        if(val_tm>=2) val_tm=1;
        if(val_sf> 2) val_sf=1;
        if(val_ee> 2) val_ee=1;

        // -1 = unknown
        //  0 = red
        //  1 = green
        //  2 = yellow

        if(val_in==-1) xval=-1;
        else if(val_in == 0) xval=0;
        else if(val_po == 0 || val_ls == 0 || val_tm == 0 || val_sf == 0 || val_ee == 0) xval = 0;
        else if(val_po == 2 || val_ls == 2 || val_tm == 2 || val_sf == 2 || val_ee == 2) xval = 2;
        else xval=1;

        meGlobalSummary_[0]->setBinContent( jx, jy, xval );

        if ( xval > -1 ) {
          ++nValidChannels;
          ++nValidChannelsEEM;
        }
        if ( xval == 0 ) {
          ++nGlobalErrors;
          ++nGlobalErrorsEEM;
        }

      }

      if(meIntegrity_[1] && mePedestalOnline_[1] && meLaserL1_[1] && meTiming_[1] && meStatusFlags_[1] && meTriggerTowerEmulError_[1]) {

        float xval = -1;
        float val_in = meIntegrity_[1]->getBinContent(jx,jy);
        float val_po = mePedestalOnline_[1]->getBinContent(jx,jy);
        float val_ls = meLaserL1_[1]->getBinContent(jx,jy);
        float val_tm = meTiming_[1]->getBinContent(jx,jy);
        float val_sf = meStatusFlags_[1]->getBinContent(jx,jy);
        float val_ee = meTriggerTowerEmulError_[1]->getBinContent(jx,jy);

        // turn each dark color to bright green
        // for laser turn also yellow into bright green
        if(val_in>2) val_in=1;
        if(val_po>2) val_po=1;
        if(val_ls>=2) val_ls=1;
        if(val_tm>2)  val_tm=1;
        if(val_sf>2)  val_sf=1;
        if(val_ee>2)  val_ee=1;

        // -1 = unknown
        //  0 = red
        //  1 = green
        //  2 = yellow

        if(val_in==-1) xval=-1;
        else if(val_in == 0) xval=0;
        else if(val_po == 0 || val_ls == 0 || val_tm == 0 || val_sf == 0 || val_ee == 0) xval = 0;
        else if(val_po == 2 || val_tm == 2 || val_sf == 2 || val_ee == 2) xval = 2;
        else xval=1;

        meGlobalSummary_[1]->setBinContent( jx, jy, xval );

        if ( xval > -1 ) {
          ++nValidChannels;
          ++nValidChannelsEEP;
        }
        if ( xval == 0 ) {
          ++nGlobalErrors;
          ++nGlobalErrorsEEP;
        }

      }

    }
  }

  float errorSummary = -1.0;
  float errorSummaryEEM = -1.0;
  float errorSummaryEEP = -1.0;

  if ( nValidChannels != 0 )
    errorSummary = 1.0 - float(nGlobalErrors)/float(nValidChannels);
  if ( nValidChannelsEEM != 0 )
    errorSummaryEEM = 1.0 - float(nGlobalErrorsEEM)/float(nValidChannelsEEM);
  if ( nValidChannelsEEP != 0 )
    errorSummaryEEP = 1.0 - float(nGlobalErrorsEEP)/float(nValidChannelsEEP);

  MonitorElement* me;

  me = dqmStore_->get(prefixME_ + "/EventInfo/errorSummary");
  if (me) me->Fill(errorSummary);

  me = dqmStore_->get(prefixME_ + "/EventInfo/errorSummarySegments/Segment00");
  if (me) me->Fill(errorSummaryEEM);

  me = dqmStore_->get(prefixME_ + "/EventInfo/errorSummarySegments/Segment01");
  if (me) me->Fill(errorSummaryEEP);

  MonitorElement* meside[2];

  meside[0] = dqmStore_->get(prefixME_ + "/EventInfo/errorSummaryXY_EEM");
  meside[1] = dqmStore_->get(prefixME_ + "/EventInfo/errorSummaryXY_EEP");
  if (meside[0] && meside[1]) {

    int nValidChannelsTT[2][20][20];
    int nGlobalErrorsTT[2][20][20];
    int nOutOfGeometryTT[2][20][20];
    for ( int jxdcc = 0; jxdcc < 20; jxdcc++ ) {
      for ( int jydcc = 0; jydcc < 20; jydcc++ ) {
        for ( int iside = 0; iside < 2; iside++ ) {
          nValidChannelsTT[iside][jxdcc][jydcc]=0;
          nGlobalErrorsTT[iside][jxdcc][jydcc]=0;
          nOutOfGeometryTT[iside][jxdcc][jydcc]=0;
        }
      }
    }

    for ( int jx = 1; jx <= 100; jx++ ) {
      for ( int jy = 1; jy <= 100; jy++ ) {
        for ( int iside = 0; iside < 2; iside++ ) {

          int jxdcc = (jx-1)/5+1;
          int jydcc = (jy-1)/5+1;

          float xval = meGlobalSummary_[iside]->getBinContent( jx, jy );

          if ( xval > -1 ) {
            if ( xval != 2 && xval != 5 ) nValidChannelsTT[iside][jxdcc-1][jydcc-1]++;
            if ( xval == 0 ) nGlobalErrorsTT[iside][jxdcc-1][jydcc-1]++;
          } else {
            nOutOfGeometryTT[iside][jxdcc-1][jydcc-1]++;
          }

        }
      }
    }

    for ( int jxdcc = 0; jxdcc < 20; jxdcc++ ) {
      for ( int jydcc = 0; jydcc < 20; jydcc++ ) {
        for ( int iside = 0; iside < 2; iside++ ) {

          float xval = -1.0;
          if ( nOutOfGeometryTT[iside][jxdcc][jydcc] < 25 ) {
            if ( nValidChannelsTT[iside][jxdcc][jydcc] != 0 )
              xval = 1.0 - float(nGlobalErrorsTT[iside][jxdcc][jydcc])/float(nValidChannelsTT[iside][jxdcc][jydcc]);
          } else {
            xval = 0.0;
          }

          meside[iside]->setBinContent( jxdcc+1, jydcc+1, xval );

        }
      }
    }

  }

}

void EESummaryClient::htmlOutput(int run, string& htmlDir, string& htmlName){

  if ( verbose_ ) cout << "Preparing EESummaryClient html output ..." << endl;

  ofstream htmlFile;

  htmlFile.open((htmlDir + htmlName).c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlFile << "<html>  " << endl;
  htmlFile << "<head>  " << endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlFile << " http-equiv=\"content-type\">  " << endl;
  htmlFile << "  <title>Monitor:Summary output</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlFile << "<body>  " << endl;
  //htmlFile << "<br>  " << endl;
  htmlFile << "<a name=""top""></a>" << endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << run << "</span></h2>" << endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">SUMMARY</span></h2> " << endl;
  htmlFile << "<hr>" << endl;
  htmlFile << "<table border=1><tr><td bgcolor=red>channel has problems in this task</td>" << endl;
  htmlFile << "<td bgcolor=lime>channel has NO problems</td>" << endl;
  htmlFile << "<td bgcolor=yellow>channel is missing</td></table>" << endl;
  htmlFile << "<br>" << endl;

  // Produce the plots to be shown as .png files from existing histograms

  const int csize = 250;

//  const double histMax = 1.e15;

  int pCol3[6] = { 301, 302, 303, 304, 305, 306 };
  int pCol4[10];
  for ( int i = 0; i < 10; i++ ) pCol4[i] = 401+i;

  // dummy histogram labelling the SM's
  TH2C labelGrid1("labelGrid1","label grid for EE -", 10, 0., 100., 10, 0., 100.);

  for ( int i=1; i<=10; i++) {
    for ( int j=1; j<=10; j++) {
      labelGrid1.SetBinContent(i, j, -10);
    }
  }

  labelGrid1.SetBinContent(2, 5, -7);
  labelGrid1.SetBinContent(2, 7, -8);
  labelGrid1.SetBinContent(4, 9, -9);
  labelGrid1.SetBinContent(7, 9, -1);
  labelGrid1.SetBinContent(9, 7, -2);
  labelGrid1.SetBinContent(9, 5, -3);
  labelGrid1.SetBinContent(8, 3, -4);
  labelGrid1.SetBinContent(6, 2, -5);
  labelGrid1.SetBinContent(3, 3, -6);

  labelGrid1.SetMarkerSize(2);
  labelGrid1.SetMinimum(-9.01);
  labelGrid1.SetMaximum(-0.01);

  TH2C labelGrid2("labelGrid2","label grid for EE +", 10, 0., 100., 10, 0., 100.);

  for ( int i=1; i<=10; i++) {
    for ( int j=1; j<=10; j++) {
      labelGrid2.SetBinContent(i, j, -10);
    }
  }

  labelGrid2.SetBinContent(2, 5, +3);
  labelGrid2.SetBinContent(2, 7, +2);
  labelGrid2.SetBinContent(4, 9, +1);
  labelGrid2.SetBinContent(7, 9, +9);
  labelGrid2.SetBinContent(9, 7, +8);
  labelGrid2.SetBinContent(9, 5, +7);
  labelGrid2.SetBinContent(8, 3, +6);
  labelGrid2.SetBinContent(5, 2, +5);
  labelGrid2.SetBinContent(3, 3, +4);

  labelGrid2.SetMarkerSize(2);
  labelGrid2.SetMinimum(+0.01);
  labelGrid2.SetMaximum(+9.01);

  string imgNameMapI[2], imgNameMapO[2];
  string imgNameMapDF[2];
  string imgNameMapPO[2];
  string imgNameMapLL1[2], imgNameMapLL1_PN[2];
  string imgNameMapLD[2], imgNameMapLD_PN[2];
  string imgNameMapP[2], imgNameMapP_PN[2];
  string imgNameMapTP[2], imgNameMapTP_PN[2];
  string imgNameMapC[2];
  string imgNameMapTM[2];
  string imgNameMapTTEt[2];
  string imgNameMapTTEmulError[2];
  string imgName, meName;
  string imgNameMapGS[2];

  TCanvas* cMap = new TCanvas("cMap", "Temp", int(1.5*csize), int(1.5*csize));

  float saveHeigth = gStyle->GetTitleH();
  gStyle->SetTitleH(0.07);
  float saveFontSize = gStyle->GetTitleFontSize();
  gStyle->SetTitleFontSize(14);

  TH2F* obj2f;

  gStyle->SetPaintTextFormat("+g");

  imgNameMapI[0] = "";

  obj2f = 0;
  obj2f = UtilsClient::getHisto<TH2F*>( meIntegrity_[0] );

  if ( obj2f ) {

    meName = obj2f->GetName();

    replace(meName.begin(), meName.end(), ' ', '_');
    imgNameMapI[0] = meName + ".png";
    imgName = htmlDir + imgNameMapI[0];

    cMap->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(6, pCol3);
    cMap->SetGridx();
    cMap->SetGridy();
    obj2f->SetMinimum(-0.00000001);
    obj2f->SetMaximum(6.0);
    obj2f->GetXaxis()->SetLabelSize(0.03);
    obj2f->GetYaxis()->SetLabelSize(0.03);
    obj2f->Draw("col");
    labelGrid1.Draw("text,same");
    cMap->SetBit(TGraph::kClipFrame);
    TLine l;
    l.SetLineWidth(1);
    for ( int i=0; i<201; i=i+1){
      if ( (Numbers::ixSectorsEE[i]!=0 || Numbers::iySectorsEE[i]!=0) && (Numbers::ixSectorsEE[i+1]!=0 || Numbers::iySectorsEE[i+1]!=0) ) {
        l.DrawLine(Numbers::ixSectorsEE[i], Numbers::iySectorsEE[i], Numbers::ixSectorsEE[i+1], Numbers::iySectorsEE[i+1]);
      }
    }
    cMap->Update();
    cMap->SaveAs(imgName.c_str());

  }

  imgNameMapI[1] = "";

  obj2f = 0;
  obj2f = UtilsClient::getHisto<TH2F*>( meIntegrity_[1] );

  if ( obj2f ) {

    meName = obj2f->GetName();

    replace(meName.begin(), meName.end(), ' ', '_');
    imgNameMapI[1] = meName + ".png";
    imgName = htmlDir + imgNameMapI[1];

    cMap->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(6, pCol3);
    cMap->SetGridx();
    cMap->SetGridy();
    obj2f->SetMinimum(-0.00000001);
    obj2f->SetMaximum(6.0);
    obj2f->GetXaxis()->SetLabelSize(0.03);
    obj2f->GetYaxis()->SetLabelSize(0.03);
    obj2f->Draw("col");
    labelGrid2.Draw("text,same");
    cMap->SetBit(TGraph::kClipFrame);
    TLine l;
    l.SetLineWidth(1);
    for ( int i=0; i<201; i=i+1){
      if ( (Numbers::ixSectorsEE[i]!=0 || Numbers::iySectorsEE[i]!=0) && (Numbers::ixSectorsEE[i+1]!=0 || Numbers::iySectorsEE[i+1]!=0) ) {
        l.DrawLine(Numbers::ixSectorsEE[i], Numbers::iySectorsEE[i], Numbers::ixSectorsEE[i+1], Numbers::iySectorsEE[i+1]);
      }
    }
    cMap->Update();
    cMap->SaveAs(imgName.c_str());

  }

  imgNameMapO[0] = "";

  obj2f = 0;
  obj2f = UtilsClient::getHisto<TH2F*>( meOccupancy_[0] );

  if ( obj2f ) {

    meName = obj2f->GetName();

    replace(meName.begin(), meName.end(), ' ', '_');
    imgNameMapO[0] = meName + ".png";
    imgName = htmlDir + imgNameMapO[0];

    cMap->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(10, pCol4);
    cMap->SetGridx();
    cMap->SetGridy();
    obj2f->SetMinimum(0.0);
    obj2f->GetXaxis()->SetLabelSize(0.03);
    obj2f->GetYaxis()->SetLabelSize(0.03);
    obj2f->GetZaxis()->SetLabelSize(0.03);
    obj2f->Draw("colz");
    labelGrid1.Draw("text,same");
    cMap->SetBit(TGraph::kClipFrame);
    TLine l;
    l.SetLineWidth(1);
    for ( int i=0; i<201; i=i+1){
      if ( (Numbers::ixSectorsEE[i]!=0 || Numbers::iySectorsEE[i]!=0) && (Numbers::ixSectorsEE[i+1]!=0 || Numbers::iySectorsEE[i+1]!=0) ) {
        l.DrawLine(Numbers::ixSectorsEE[i], Numbers::iySectorsEE[i], Numbers::ixSectorsEE[i+1], Numbers::iySectorsEE[i+1]);
      }
    }
    cMap->Update();
    cMap->SaveAs(imgName.c_str());

  }

  imgNameMapO[1] = "";

  obj2f = 0;
  obj2f = UtilsClient::getHisto<TH2F*>( meOccupancy_[1] );

  if ( obj2f ) {

    meName = obj2f->GetName();

    replace(meName.begin(), meName.end(), ' ', '_');
    imgNameMapO[1] = meName + ".png";
    imgName = htmlDir + imgNameMapO[1];

    cMap->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(10, pCol4);
    cMap->SetGridx();
    cMap->SetGridy();
    obj2f->SetMinimum(0.0);
    obj2f->GetXaxis()->SetLabelSize(0.03);
    obj2f->GetYaxis()->SetLabelSize(0.03);
    obj2f->GetZaxis()->SetLabelSize(0.03);
    obj2f->Draw("colz");
    labelGrid2.Draw("text,same");
    cMap->SetBit(TGraph::kClipFrame);
    TLine l;
    l.SetLineWidth(1);
    for ( int i=0; i<201; i=i+1){
      if ( (Numbers::ixSectorsEE[i]!=0 || Numbers::iySectorsEE[i]!=0) && (Numbers::ixSectorsEE[i+1]!=0 || Numbers::iySectorsEE[i+1]!=0) ) {
        l.DrawLine(Numbers::ixSectorsEE[i], Numbers::iySectorsEE[i], Numbers::ixSectorsEE[i+1], Numbers::iySectorsEE[i+1]);
      }
    }
    cMap->Update();
    cMap->SaveAs(imgName.c_str());

  }

  imgNameMapDF[0] = "";

  obj2f = 0;
  obj2f = UtilsClient::getHisto<TH2F*>( meStatusFlags_[0] );

  if ( obj2f && obj2f->GetEntries() != 0 ) {

    meName = obj2f->GetName();

    replace(meName.begin(), meName.end(), ' ', '_');
    imgNameMapDF[0] = meName + ".png";
    imgName = htmlDir + imgNameMapDF[0];

    cMap->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(6, pCol3);
    cMap->SetGridx();
    cMap->SetGridy();
    obj2f->SetMinimum(-0.00000001);
    obj2f->SetMaximum(6.0);
    obj2f->GetXaxis()->SetLabelSize(0.03);
    obj2f->GetYaxis()->SetLabelSize(0.03);
    obj2f->Draw("col");
    labelGrid1.Draw("text,same");
    cMap->SetBit(TGraph::kClipFrame);
    TLine l;
    l.SetLineWidth(1);
    for ( int i=0; i<201; i=i+1){
      if ( (Numbers::ixSectorsEE[i]!=0 || Numbers::iySectorsEE[i]!=0) && (Numbers::ixSectorsEE[i+1]!=0 || Numbers::iySectorsEE[i+1]!=0) ) {
        l.DrawLine(Numbers::ixSectorsEE[i], Numbers::iySectorsEE[i], Numbers::ixSectorsEE[i+1], Numbers::iySectorsEE[i+1]);
      }
    }
    cMap->Update();
    cMap->SaveAs(imgName.c_str());

  }

  imgNameMapDF[1] = "";

  obj2f = 0;
  obj2f = UtilsClient::getHisto<TH2F*>( meStatusFlags_[1] );

  if ( obj2f && obj2f->GetEntries() != 0 ) {

    meName = obj2f->GetName();

    replace(meName.begin(), meName.end(), ' ', '_');
    imgNameMapDF[1] = meName + ".png";
    imgName = htmlDir + imgNameMapDF[1];

    cMap->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(6, pCol3);
    cMap->SetGridx();
    cMap->SetGridy();
    obj2f->SetMinimum(-0.00000001);
    obj2f->SetMaximum(6.0);
    obj2f->GetXaxis()->SetLabelSize(0.03);
    obj2f->GetYaxis()->SetLabelSize(0.03);
    obj2f->Draw("col");
    labelGrid1.Draw("text,same");
    cMap->SetBit(TGraph::kClipFrame);
    TLine l;
    l.SetLineWidth(1);
    for ( int i=0; i<201; i=i+1){
      if ( (Numbers::ixSectorsEE[i]!=0 || Numbers::iySectorsEE[i]!=0) && (Numbers::ixSectorsEE[i+1]!=0 || Numbers::iySectorsEE[i+1]!=0) ) {
        l.DrawLine(Numbers::ixSectorsEE[i], Numbers::iySectorsEE[i], Numbers::ixSectorsEE[i+1], Numbers::iySectorsEE[i+1]);
      }
    }
    cMap->Update();
    cMap->SaveAs(imgName.c_str());

  }

  imgNameMapPO[0] = "";

  obj2f = 0;
  obj2f = UtilsClient::getHisto<TH2F*>( mePedestalOnline_[0] );

  if ( obj2f && obj2f->GetEntries() != 0 ) {

    meName = obj2f->GetName();

    replace(meName.begin(), meName.end(), ' ', '_');
    imgNameMapPO[0] = meName + ".png";
    imgName = htmlDir + imgNameMapPO[0];

    cMap->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(6, pCol3);
    cMap->SetGridx();
    cMap->SetGridy();
    obj2f->SetMinimum(-0.00000001);
    obj2f->SetMaximum(6.0);
    obj2f->GetXaxis()->SetLabelSize(0.03);
    obj2f->GetYaxis()->SetLabelSize(0.03);
    obj2f->Draw("col");
    labelGrid1.Draw("text,same");
    cMap->SetBit(TGraph::kClipFrame);
    TLine l;
    l.SetLineWidth(1);
    for ( int i=0; i<201; i=i+1){
      if ( (Numbers::ixSectorsEE[i]!=0 || Numbers::iySectorsEE[i]!=0) && (Numbers::ixSectorsEE[i+1]!=0 || Numbers::iySectorsEE[i+1]!=0) ) {
        l.DrawLine(Numbers::ixSectorsEE[i], Numbers::iySectorsEE[i], Numbers::ixSectorsEE[i+1], Numbers::iySectorsEE[i+1]);
      }
    }
    cMap->Update();
    cMap->SaveAs(imgName.c_str());

  }

  imgNameMapPO[1] = "";

  obj2f = 0;
  obj2f = UtilsClient::getHisto<TH2F*>( mePedestalOnline_[1] );

  if ( obj2f && obj2f->GetEntries() != 0 ) {

    meName = obj2f->GetName();

    replace(meName.begin(), meName.end(), ' ', '_');
    imgNameMapPO[1] = meName + ".png";
    imgName = htmlDir + imgNameMapPO[1];

    cMap->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(6, pCol3);
    cMap->SetGridx();
    cMap->SetGridy();
    obj2f->SetMinimum(-0.00000001);
    obj2f->SetMaximum(6.0);
    obj2f->GetXaxis()->SetLabelSize(0.03);
    obj2f->GetYaxis()->SetLabelSize(0.03);
    obj2f->Draw("col");
    labelGrid2.Draw("text,same");
    cMap->SetBit(TGraph::kClipFrame);
    TLine l;
    l.SetLineWidth(1);
    for ( int i=0; i<201; i=i+1){
      if ( (Numbers::ixSectorsEE[i]!=0 || Numbers::iySectorsEE[i]!=0) && (Numbers::ixSectorsEE[i+1]!=0 || Numbers::iySectorsEE[i+1]!=0) ) {
        l.DrawLine(Numbers::ixSectorsEE[i], Numbers::iySectorsEE[i], Numbers::ixSectorsEE[i+1], Numbers::iySectorsEE[i+1]);
      }
    }
    cMap->Update();
    cMap->SaveAs(imgName.c_str());

  }

  imgNameMapLL1[0] = "";

  obj2f = 0;
  obj2f = UtilsClient::getHisto<TH2F*>( meLaserL1_[0] );

  if ( obj2f && obj2f->GetEntries() != 0 ) {

    meName = obj2f->GetName();

    replace(meName.begin(), meName.end(), ' ', '_');
    imgNameMapLL1[0] = meName + ".png";
    imgName = htmlDir + imgNameMapLL1[0];

    cMap->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(6, pCol3);
    cMap->SetGridx();
    cMap->SetGridy();
    obj2f->SetMinimum(-0.00000001);
    obj2f->SetMaximum(6.0);
    obj2f->GetXaxis()->SetLabelSize(0.03);
    obj2f->GetYaxis()->SetLabelSize(0.03);
    obj2f->Draw("col");
    labelGrid1.Draw("text,same");
    cMap->SetBit(TGraph::kClipFrame);
    TLine l;
    l.SetLineWidth(1);
    for ( int i=0; i<201; i=i+1){
      if ( (Numbers::ixSectorsEE[i]!=0 || Numbers::iySectorsEE[i]!=0) && (Numbers::ixSectorsEE[i+1]!=0 || Numbers::iySectorsEE[i+1]!=0) ) {
        l.DrawLine(Numbers::ixSectorsEE[i], Numbers::iySectorsEE[i], Numbers::ixSectorsEE[i+1], Numbers::iySectorsEE[i+1]);
      }
    }
    cMap->Update();
    cMap->SaveAs(imgName.c_str());

  }

  imgNameMapLL1[1] = "";

  obj2f = 0;
  obj2f = UtilsClient::getHisto<TH2F*>( meLaserL1_[1] );

  if ( obj2f && obj2f->GetEntries() != 0 ) {

    meName = obj2f->GetName();

    replace(meName.begin(), meName.end(), ' ', '_');
    imgNameMapLL1[1] = meName + ".png";
    imgName = htmlDir + imgNameMapLL1[1];

    cMap->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(6, pCol3);
    cMap->SetGridx();
    cMap->SetGridy();
    obj2f->SetMinimum(-0.00000001);
    obj2f->SetMaximum(6.0);
    obj2f->GetXaxis()->SetLabelSize(0.03);
    obj2f->GetYaxis()->SetLabelSize(0.03);
    obj2f->Draw("col");
    labelGrid2.Draw("text,same");
    cMap->SetBit(TGraph::kClipFrame);
    TLine l;
    l.SetLineWidth(1);
    for ( int i=0; i<201; i=i+1){
      if ( (Numbers::ixSectorsEE[i]!=0 || Numbers::iySectorsEE[i]!=0) && (Numbers::ixSectorsEE[i+1]!=0 || Numbers::iySectorsEE[i+1]!=0) ) {
        l.DrawLine(Numbers::ixSectorsEE[i], Numbers::iySectorsEE[i], Numbers::ixSectorsEE[i+1], Numbers::iySectorsEE[i+1]);
      }
    }
    cMap->Update();
    cMap->SaveAs(imgName.c_str());

  }

//
  imgNameMapLL1_PN[0] = "";
  imgNameMapLL1_PN[1] = "";
//

  imgNameMapLD[0] = "";

  obj2f = 0;
  obj2f = UtilsClient::getHisto<TH2F*>( meLedL1_[0] );

  if ( obj2f && obj2f->GetEntries() != 0 ) {

    meName = obj2f->GetName();

    replace(meName.begin(), meName.end(), ' ', '_');
    imgNameMapLD[0] = meName + ".png";
    imgName = htmlDir + imgNameMapLD[0];

    cMap->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(6, pCol3);
    cMap->SetGridx();
    cMap->SetGridy();
    obj2f->SetMinimum(-0.00000001);
    obj2f->SetMaximum(6.0);
    obj2f->GetXaxis()->SetLabelSize(0.03);
    obj2f->GetYaxis()->SetLabelSize(0.03);
    obj2f->Draw("col");
    labelGrid1.Draw("text,same");
    cMap->SetBit(TGraph::kClipFrame);
    TLine l;
    l.SetLineWidth(1);
    for ( int i=0; i<201; i=i+1){
      if ( (Numbers::ixSectorsEE[i]!=0 || Numbers::iySectorsEE[i]!=0) && (Numbers::ixSectorsEE[i+1]!=0 || Numbers::iySectorsEE[i+1]!=0) ) {
        l.DrawLine(Numbers::ixSectorsEE[i], Numbers::iySectorsEE[i], Numbers::ixSectorsEE[i+1], Numbers::iySectorsEE[i+1]);
      }
    }
    cMap->Update();
    cMap->SaveAs(imgName.c_str());

  }

  imgNameMapLD[1] = "";

  obj2f = 0;
  obj2f = UtilsClient::getHisto<TH2F*>( meLedL1_[1] );

  if ( obj2f && obj2f->GetEntries() != 0 ) {

    meName = obj2f->GetName();

    replace(meName.begin(), meName.end(), ' ', '_');
    imgNameMapLD[1] = meName + ".png";
    imgName = htmlDir + imgNameMapLD[1];

    cMap->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(6, pCol3);
    cMap->SetGridx();
    cMap->SetGridy();
    obj2f->SetMinimum(-0.00000001);
    obj2f->SetMaximum(6.0);
    obj2f->GetXaxis()->SetLabelSize(0.03);
    obj2f->GetYaxis()->SetLabelSize(0.03);
    obj2f->Draw("col");
    labelGrid2.Draw("text,same");
    cMap->SetBit(TGraph::kClipFrame);
    TLine l;
    l.SetLineWidth(1);
    for ( int i=0; i<201; i=i+1){
      if ( (Numbers::ixSectorsEE[i]!=0 || Numbers::iySectorsEE[i]!=0) && (Numbers::ixSectorsEE[i+1]!=0 || Numbers::iySectorsEE[i+1]!=0) ) {
        l.DrawLine(Numbers::ixSectorsEE[i], Numbers::iySectorsEE[i], Numbers::ixSectorsEE[i+1], Numbers::iySectorsEE[i+1]);
      }
    }
    cMap->Update();
    cMap->SaveAs(imgName.c_str());

  }

//
  imgNameMapLD_PN[0] = "";
  imgNameMapLD_PN[1] = "";
//

  imgNameMapP[0] = "";

  obj2f = 0;
  obj2f = UtilsClient::getHisto<TH2F*>( mePedestal_[0] );

  if ( obj2f && obj2f->GetEntries() != 0 ) {

    meName = obj2f->GetName();

    replace(meName.begin(), meName.end(), ' ', '_');
    imgNameMapP[0] = meName + ".png";
    imgName = htmlDir + imgNameMapP[0];

    cMap->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(6, pCol3);
    cMap->SetGridx();
    cMap->SetGridy();
    obj2f->SetMinimum(-0.00000001);
    obj2f->SetMaximum(6.0);
    obj2f->GetXaxis()->SetLabelSize(0.03);
    obj2f->GetYaxis()->SetLabelSize(0.03);
    obj2f->Draw("col");
    labelGrid1.Draw("text,same");
    cMap->SetBit(TGraph::kClipFrame);
    TLine l;
    l.SetLineWidth(1);
    for ( int i=0; i<201; i=i+1){
      if ( (Numbers::ixSectorsEE[i]!=0 || Numbers::iySectorsEE[i]!=0) && (Numbers::ixSectorsEE[i+1]!=0 || Numbers::iySectorsEE[i+1]!=0) ) {
        l.DrawLine(Numbers::ixSectorsEE[i], Numbers::iySectorsEE[i], Numbers::ixSectorsEE[i+1], Numbers::iySectorsEE[i+1]);
      }
    }
    cMap->Update();
    cMap->SaveAs(imgName.c_str());

  }

  imgNameMapP[1] = "";

  obj2f = 0;
  obj2f = UtilsClient::getHisto<TH2F*>( mePedestal_[1] );

  if ( obj2f && obj2f->GetEntries() != 0 ) {

    meName = obj2f->GetName();

    replace(meName.begin(), meName.end(), ' ', '_');
    imgNameMapP[1] = meName + ".png";
    imgName = htmlDir + imgNameMapP[1];

    cMap->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(6, pCol3);
    cMap->SetGridx();
    cMap->SetGridy();
    obj2f->SetMinimum(-0.00000001);
    obj2f->SetMaximum(6.0);
    obj2f->GetXaxis()->SetLabelSize(0.03);
    obj2f->GetYaxis()->SetLabelSize(0.03);
    obj2f->Draw("col");
    labelGrid2.Draw("text,same");
    cMap->SetBit(TGraph::kClipFrame);
    TLine l;
    l.SetLineWidth(1);
    for ( int i=0; i<201; i=i+1){
      if ( (Numbers::ixSectorsEE[i]!=0 || Numbers::iySectorsEE[i]!=0) && (Numbers::ixSectorsEE[i+1]!=0 || Numbers::iySectorsEE[i+1]!=0) ) {
        l.DrawLine(Numbers::ixSectorsEE[i], Numbers::iySectorsEE[i], Numbers::ixSectorsEE[i+1], Numbers::iySectorsEE[i+1]);
      }
    }
    cMap->Update();
    cMap->SaveAs(imgName.c_str());

  }

//
  imgNameMapP_PN[0] = "";
  imgNameMapP_PN[1] = "";
//

  imgNameMapTP[0] = "";

  obj2f = 0;
  obj2f = UtilsClient::getHisto<TH2F*>( meTestPulse_[0] );

  if ( obj2f && obj2f->GetEntries() != 0 ) {

    meName = obj2f->GetName();

    replace(meName.begin(), meName.end(), ' ', '_');
    imgNameMapTP[0] = meName + ".png";
    imgName = htmlDir + imgNameMapTP[0];

    cMap->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(6, pCol3);
    cMap->SetGridx();
    cMap->SetGridy();
    obj2f->SetMinimum(-0.00000001);
    obj2f->SetMaximum(6.0);
    obj2f->GetXaxis()->SetLabelSize(0.03);
    obj2f->GetYaxis()->SetLabelSize(0.03);
    obj2f->Draw("col");
    labelGrid1.Draw("text,same");
    cMap->SetBit(TGraph::kClipFrame);
    TLine l;
    l.SetLineWidth(1);
    for ( int i=0; i<201; i=i+1){
      if ( (Numbers::ixSectorsEE[i]!=0 || Numbers::iySectorsEE[i]!=0) && (Numbers::ixSectorsEE[i+1]!=0 || Numbers::iySectorsEE[i+1]!=0) ) {
        l.DrawLine(Numbers::ixSectorsEE[i], Numbers::iySectorsEE[i], Numbers::ixSectorsEE[i+1], Numbers::iySectorsEE[i+1]);
      }
    }
    cMap->Update();
    cMap->SaveAs(imgName.c_str());

  }

  imgNameMapTP[1] = "";

  obj2f = 0;
  obj2f = UtilsClient::getHisto<TH2F*>( meTestPulse_[1] );

  if ( obj2f && obj2f->GetEntries() != 0 ) {

    meName = obj2f->GetName();

    replace(meName.begin(), meName.end(), ' ', '_');
    imgNameMapTP[1] = meName + ".png";
    imgName = htmlDir + imgNameMapTP[1];

    cMap->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(6, pCol3);
    cMap->SetGridx();
    cMap->SetGridy();
    obj2f->SetMinimum(-0.00000001);
    obj2f->SetMaximum(6.0);
    obj2f->GetXaxis()->SetLabelSize(0.03);
    obj2f->GetYaxis()->SetLabelSize(0.03);
    obj2f->Draw("col");
    labelGrid2.Draw("text,same");
    cMap->SetBit(TGraph::kClipFrame);
    TLine l;
    l.SetLineWidth(1);
    for ( int i=0; i<201; i=i+1){
      if ( (Numbers::ixSectorsEE[i]!=0 || Numbers::iySectorsEE[i]!=0) && (Numbers::ixSectorsEE[i+1]!=0 || Numbers::iySectorsEE[i+1]!=0) ) {
        l.DrawLine(Numbers::ixSectorsEE[i], Numbers::iySectorsEE[i], Numbers::ixSectorsEE[i+1], Numbers::iySectorsEE[i+1]);
      }
    }
    cMap->Update();
    cMap->SaveAs(imgName.c_str());

  }

//
  imgNameMapTP_PN[0] = "";
  imgNameMapTP_PN[1] = "";
//

  imgNameMapC[0] = "";

  obj2f = 0;
  obj2f = UtilsClient::getHisto<TH2F*>( meCosmic_[0] );

  if ( obj2f && obj2f->GetEntries() != 0 ) {

    meName = obj2f->GetName();

    replace(meName.begin(), meName.end(), ' ', '_');
    imgNameMapC[0] = meName + ".png";
    imgName = htmlDir + imgNameMapC[0];

    cMap->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(10, pCol4);
    cMap->SetGridx();
    cMap->SetGridy();
    obj2f->SetMinimum(0.0);
    obj2f->GetXaxis()->SetLabelSize(0.03);
    obj2f->GetYaxis()->SetLabelSize(0.03);
    obj2f->GetZaxis()->SetLabelSize(0.03);
    obj2f->Draw("colz");
    labelGrid1.Draw("text,same");
    cMap->SetBit(TGraph::kClipFrame);
    TLine l;
    l.SetLineWidth(1);
    for ( int i=0; i<201; i=i+1){
      if ( (Numbers::ixSectorsEE[i]!=0 || Numbers::iySectorsEE[i]!=0) && (Numbers::ixSectorsEE[i+1]!=0 || Numbers::iySectorsEE[i+1]!=0) ) {
        l.DrawLine(Numbers::ixSectorsEE[i], Numbers::iySectorsEE[i], Numbers::ixSectorsEE[i+1], Numbers::iySectorsEE[i+1]);
      }
    }
    cMap->Update();
    cMap->SaveAs(imgName.c_str());

  }

  imgNameMapC[1] = "";

  obj2f = 0;
  obj2f = UtilsClient::getHisto<TH2F*>( meCosmic_[1] );

  if ( obj2f && obj2f->GetEntries() != 0 ) {

    meName = obj2f->GetName();

    replace(meName.begin(), meName.end(), ' ', '_');
    imgNameMapC[1] = meName + ".png";
    imgName = htmlDir + imgNameMapC[1];

    cMap->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(10, pCol4);
    cMap->SetGridx();
    cMap->SetGridy();
    obj2f->SetMinimum(0.0);
    obj2f->GetXaxis()->SetLabelSize(0.03);
    obj2f->GetYaxis()->SetLabelSize(0.03);
    obj2f->GetZaxis()->SetLabelSize(0.03);
    obj2f->Draw("colz");
    labelGrid2.Draw("text,same");
    cMap->SetBit(TGraph::kClipFrame);
    TLine l;
    l.SetLineWidth(1);
    for ( int i=0; i<201; i=i+1){
      if ( (Numbers::ixSectorsEE[i]!=0 || Numbers::iySectorsEE[i]!=0) && (Numbers::ixSectorsEE[i+1]!=0 || Numbers::iySectorsEE[i+1]!=0) ) {
        l.DrawLine(Numbers::ixSectorsEE[i], Numbers::iySectorsEE[i], Numbers::ixSectorsEE[i+1], Numbers::iySectorsEE[i+1]);
      }
    }
    cMap->Update();
    cMap->SaveAs(imgName.c_str());

  }

  imgNameMapTM[0] = "";

  obj2f = 0;
  obj2f = UtilsClient::getHisto<TH2F*>( meTiming_[0] );

  if ( obj2f && obj2f->GetEntries() != 0 ) {

    meName = obj2f->GetName();

    replace(meName.begin(), meName.end(), ' ', '_');
    imgNameMapTM[0] = meName + ".png";
    imgName = htmlDir + imgNameMapTM[0];

    cMap->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(6, pCol3);
    cMap->SetGridx();
    cMap->SetGridy();
    obj2f->SetMinimum(-0.00000001);
    obj2f->SetMaximum(6.0);
    obj2f->GetXaxis()->SetLabelSize(0.03);
    obj2f->GetYaxis()->SetLabelSize(0.03);
    obj2f->Draw("col");
    labelGrid1.Draw("text,same");
    cMap->SetBit(TGraph::kClipFrame);
    TLine l;
    l.SetLineWidth(1);
    for ( int i=0; i<201; i=i+1){
      if ( (Numbers::ixSectorsEE[i]!=0 || Numbers::iySectorsEE[i]!=0) && (Numbers::ixSectorsEE[i+1]!=0 || Numbers::iySectorsEE[i+1]!=0) ) {
        l.DrawLine(Numbers::ixSectorsEE[i], Numbers::iySectorsEE[i], Numbers::ixSectorsEE[i+1], Numbers::iySectorsEE[i+1]);
      }
    }
    cMap->Update();
    cMap->SaveAs(imgName.c_str());

  }

  imgNameMapTM[1] = "";

  obj2f = 0;
  obj2f = UtilsClient::getHisto<TH2F*>( meTiming_[1] );

  if ( obj2f && obj2f->GetEntries() != 0 ) {

    meName = obj2f->GetName();

    replace(meName.begin(), meName.end(), ' ', '_');
    imgNameMapTM[1] = meName + ".png";
    imgName = htmlDir + imgNameMapTM[1];

    cMap->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(6, pCol3);
    cMap->SetGridx();
    cMap->SetGridy();
    obj2f->SetMinimum(-0.00000001);
    obj2f->SetMaximum(6.0);
    obj2f->GetXaxis()->SetLabelSize(0.03);
    obj2f->GetYaxis()->SetLabelSize(0.03);
    obj2f->Draw("col");
    labelGrid1.Draw("text,same");
    cMap->SetBit(TGraph::kClipFrame);
    TLine l;
    l.SetLineWidth(1);
    for ( int i=0; i<201; i=i+1){
      if ( (Numbers::ixSectorsEE[i]!=0 || Numbers::iySectorsEE[i]!=0) && (Numbers::ixSectorsEE[i+1]!=0 || Numbers::iySectorsEE[i+1]!=0) ) {
        l.DrawLine(Numbers::ixSectorsEE[i], Numbers::iySectorsEE[i], Numbers::ixSectorsEE[i+1], Numbers::iySectorsEE[i+1]);
      }
    }
    cMap->Update();
    cMap->SaveAs(imgName.c_str());

  }

  imgNameMapTTEmulError[0] = "";

  obj2f = 0;
  obj2f = UtilsClient::getHisto<TH2F*>( meTriggerTowerEmulError_[0] );

  if ( obj2f && obj2f->GetEntries() != 0 ) {

    meName = obj2f->GetName();

    replace(meName.begin(), meName.end(), ' ', '_');
    imgNameMapTTEmulError[0] = meName + ".png";
    imgName = htmlDir + imgNameMapTTEmulError[0];

    cMap->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(6, pCol3);
    cMap->SetGridx();
    cMap->SetGridy();
    obj2f->SetMinimum(-0.00000001);
    obj2f->SetMaximum(6.0);
    obj2f->GetXaxis()->SetLabelSize(0.03);
    obj2f->GetYaxis()->SetLabelSize(0.03);
    obj2f->Draw("col");
    labelGrid1.Draw("text,same");
    cMap->SetBit(TGraph::kClipFrame);
    TLine l;
    l.SetLineWidth(1);
    for ( int i=0; i<201; i=i+1){
      if ( (Numbers::ixSectorsEE[i]!=0 || Numbers::iySectorsEE[i]!=0) && (Numbers::ixSectorsEE[i+1]!=0 || Numbers::iySectorsEE[i+1]!=0) ) {
        l.DrawLine(Numbers::ixSectorsEE[i], Numbers::iySectorsEE[i], Numbers::ixSectorsEE[i+1], Numbers::iySectorsEE[i+1]);
      }
    }
    cMap->Update();
    cMap->SaveAs(imgName.c_str());

  }

  imgNameMapTTEmulError[1] = "";

  obj2f = 0;
  obj2f = UtilsClient::getHisto<TH2F*>( meTriggerTowerEmulError_[1] );

  if ( obj2f && obj2f->GetEntries() != 0 ) {

    meName = obj2f->GetName();

    replace(meName.begin(), meName.end(), ' ', '_');
    imgNameMapTTEmulError[1] = meName + ".png";
    imgName = htmlDir + imgNameMapTTEmulError[1];

    cMap->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(6, pCol3);
    cMap->SetGridx();
    cMap->SetGridy();
    obj2f->SetMinimum(-0.00000001);
    obj2f->SetMaximum(6.0);
    obj2f->GetXaxis()->SetLabelSize(0.03);
    obj2f->GetYaxis()->SetLabelSize(0.03);
    obj2f->Draw("col");
    labelGrid1.Draw("text,same");
    cMap->SetBit(TGraph::kClipFrame);
    TLine l;
    l.SetLineWidth(1);
    for ( int i=0; i<201; i=i+1){
      if ( (Numbers::ixSectorsEE[i]!=0 || Numbers::iySectorsEE[i]!=0) && (Numbers::ixSectorsEE[i+1]!=0 || Numbers::iySectorsEE[i+1]!=0) ) {
        l.DrawLine(Numbers::ixSectorsEE[i], Numbers::iySectorsEE[i], Numbers::ixSectorsEE[i+1], Numbers::iySectorsEE[i+1]);
      }
    }
    cMap->Update();
    cMap->SaveAs(imgName.c_str());

  }

  imgNameMapTTEt[0] = "";

  obj2f = 0;
  obj2f = UtilsClient::getHisto<TH2F*>( meTriggerTowerEt_[0] );

  if ( obj2f && obj2f->GetEntries() != 0 ) {

    meName = obj2f->GetName();

    replace(meName.begin(), meName.end(), ' ', '_');
    imgNameMapTTEt[0] = meName + ".png";
    imgName = htmlDir + imgNameMapTTEt[0];

    cMap->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(10, pCol4);
    cMap->SetGridx();
    cMap->SetGridy();
    obj2f->SetMinimum(0.0);
    obj2f->GetXaxis()->SetLabelSize(0.03);
    obj2f->GetYaxis()->SetLabelSize(0.03);
    obj2f->GetZaxis()->SetLabelSize(0.03);
    obj2f->Draw("colz");
    labelGrid1.Draw("text,same");
    cMap->SetBit(TGraph::kClipFrame);
    TLine l;
    l.SetLineWidth(1);
    for ( int i=0; i<201; i=i+1){
      if ( (Numbers::ixSectorsEE[i]!=0 || Numbers::iySectorsEE[i]!=0) && (Numbers::ixSectorsEE[i+1]!=0 || Numbers::iySectorsEE[i+1]!=0) ) {
        l.DrawLine(Numbers::ixSectorsEE[i], Numbers::iySectorsEE[i], Numbers::ixSectorsEE[i+1], Numbers::iySectorsEE[i+1]);
      }
    }
    cMap->Update();
    cMap->SaveAs(imgName.c_str());

  }

  imgNameMapTTEt[1] = "";

  obj2f = 0;
  obj2f = UtilsClient::getHisto<TH2F*>( meTriggerTowerEt_[1] );

  if ( obj2f && obj2f->GetEntries() != 0 ) {

    meName = obj2f->GetName();

    replace(meName.begin(), meName.end(), ' ', '_');
    imgNameMapTTEt[1] = meName + ".png";
    imgName = htmlDir + imgNameMapTTEt[1];

    cMap->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(10, pCol4);
    cMap->SetGridx();
    cMap->SetGridy();
    obj2f->SetMinimum(0.0);
    obj2f->GetXaxis()->SetLabelSize(0.03);
    obj2f->GetYaxis()->SetLabelSize(0.03);
    obj2f->GetZaxis()->SetLabelSize(0.03);
    obj2f->Draw("colz");
    labelGrid1.Draw("text,same");
    cMap->SetBit(TGraph::kClipFrame);
    TLine l;
    l.SetLineWidth(1);
    for ( int i=0; i<201; i=i+1){
      if ( (Numbers::ixSectorsEE[i]!=0 || Numbers::iySectorsEE[i]!=0) && (Numbers::ixSectorsEE[i+1]!=0 || Numbers::iySectorsEE[i+1]!=0) ) {
        l.DrawLine(Numbers::ixSectorsEE[i], Numbers::iySectorsEE[i], Numbers::ixSectorsEE[i+1], Numbers::iySectorsEE[i+1]);
      }
    }
    cMap->Update();
    cMap->SaveAs(imgName.c_str());

  }

  imgNameMapGS[0] = "";

  obj2f = 0;
  obj2f = UtilsClient::getHisto<TH2F*>( meGlobalSummary_[0] );

  if ( obj2f ) {

    meName = obj2f->GetName();

    replace(meName.begin(), meName.end(), ' ', '_');
    imgNameMapGS[0] = meName + ".png";
    imgName = htmlDir + imgNameMapGS[0];

    cMap->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(6, pCol3);
    cMap->SetGridx();
    cMap->SetGridy();
    obj2f->SetMinimum(-0.00000001);
    obj2f->SetMaximum(6.0);
    obj2f->GetXaxis()->SetLabelSize(0.03);
    obj2f->GetYaxis()->SetLabelSize(0.03);
    obj2f->Draw("col");
    labelGrid1.Draw("text,same");
    cMap->SetBit(TGraph::kClipFrame);
    TLine l;
    l.SetLineWidth(1);
    for ( int i=0; i<201; i=i+1){
      if ( (Numbers::ixSectorsEE[i]!=0 || Numbers::iySectorsEE[i]!=0) && (Numbers::ixSectorsEE[i+1]!=0 || Numbers::iySectorsEE[i+1]!=0) ) {
        l.DrawLine(Numbers::ixSectorsEE[i], Numbers::iySectorsEE[i], Numbers::ixSectorsEE[i+1], Numbers::iySectorsEE[i+1]);
      }
    }
    cMap->Update();
    cMap->SaveAs(imgName.c_str());

  }

  imgNameMapGS[1] = "";

  obj2f = 0;
  obj2f = UtilsClient::getHisto<TH2F*>( meGlobalSummary_[1] );

  if ( obj2f ) {

    meName = obj2f->GetName();

    replace(meName.begin(), meName.end(), ' ', '_');
    imgNameMapGS[1] = meName + ".png";
    imgName = htmlDir + imgNameMapGS[1];

    cMap->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(6, pCol3);
    cMap->SetGridx();
    cMap->SetGridy();
    obj2f->SetMinimum(-0.00000001);
    obj2f->SetMaximum(6.0);
    obj2f->GetXaxis()->SetLabelSize(0.03);
    obj2f->GetYaxis()->SetLabelSize(0.03);
    obj2f->Draw("col");
    labelGrid2.Draw("text,same");
    cMap->SetBit(TGraph::kClipFrame);
    TLine l;
    l.SetLineWidth(1);
    for ( int i=0; i<201; i=i+1){
      if ( (Numbers::ixSectorsEE[i]!=0 || Numbers::iySectorsEE[i]!=0) && (Numbers::ixSectorsEE[i+1]!=0 || Numbers::iySectorsEE[i+1]!=0) ) {
        l.DrawLine(Numbers::ixSectorsEE[i], Numbers::iySectorsEE[i], Numbers::ixSectorsEE[i+1], Numbers::iySectorsEE[i+1]);
      }
    }
    cMap->Update();
    cMap->SaveAs(imgName.c_str());

  }

  gStyle->SetPaintTextFormat();

  if ( imgNameMapI[0].size() != 0 || imgNameMapI[1].size() != 0 ) {
    htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
    htmlFile << "<tr align=\"center\">" << endl;
    if ( imgNameMapI[0].size() != 0 ) htmlFile << "<td><img src=\"" << imgNameMapI[0] << "\" usemap=\"#Integrity_0\" border=0></td>" << endl;
    if ( imgNameMapI[1].size() != 0 ) htmlFile << "<td><img src=\"" << imgNameMapI[1] << "\" usemap=\"#Integrity_1\" border=0></td>" << endl;
    htmlFile << "</tr>" << endl;
    htmlFile << "</table>" << endl;
    htmlFile << "<br>" << endl;
  }

  if ( imgNameMapO[0].size() != 0 || imgNameMapO[1].size() != 0 ) {
    htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
    htmlFile << "<tr align=\"center\">" << endl;
    if ( imgNameMapO[0].size() != 0 ) htmlFile << "<td><img src=\"" << imgNameMapO[0] << "\" usemap=\"#Occupancy_0\" border=0></td>" << endl;
    if ( imgNameMapO[1].size() != 0 ) htmlFile << "<td><img src=\"" << imgNameMapO[1] << "\" usemap=\"#Occupancy_1\" border=0></td>" << endl;
    htmlFile << "</tr>" << endl;
    htmlFile << "</table>" << endl;
    htmlFile << "<br>" << endl;
  }

  if ( imgNameMapDF[0].size() != 0 || imgNameMapDF[1].size() != 0 ) {
    htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
    htmlFile << "<tr align=\"center\">" << endl;
    if ( imgNameMapDF[0].size() != 0 ) htmlFile << "<td><img src=\"" << imgNameMapDF[0] << "\" usemap=\"#StatusFlags_0\" border=0></td>" << endl;
    if ( imgNameMapDF[1].size() != 0 ) htmlFile << "<td><img src=\"" << imgNameMapDF[1] << "\" usemap=\"#StatusFlags_1\" border=0></td>" << endl;
    htmlFile << "</tr>" << endl;
    htmlFile << "</table>" << endl;
    htmlFile << "<br>" << endl;
  }

  if ( imgNameMapPO[0].size() != 0 || imgNameMapPO[1].size() != 0 ) {
    htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
    htmlFile << "<tr align=\"center\">" << endl;
    if ( imgNameMapPO[0].size() != 0 ) htmlFile << "<td><img src=\"" << imgNameMapPO[0] << "\" usemap=\"#PedestalOnline_0\" border=0></td>" << endl;
    if ( imgNameMapPO[1].size() != 0 ) htmlFile << "<td><img src=\"" << imgNameMapPO[1] << "\" usemap=\"#PedestalOnline_1\" border=0></td>" << endl;
    htmlFile << "</tr>" << endl;
    htmlFile << "</table>" << endl;
    htmlFile << "<br>" << endl;
  }

  if ( imgNameMapLL1[0].size() != 0 || imgNameMapLL1[1].size() != 0 ) {
    htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
    htmlFile << "<tr align=\"center\">" << endl;
    if ( imgNameMapLL1[0].size() != 0 ) htmlFile << "<td><img src=\"" << imgNameMapLL1[0] << "\" usemap=\"#LaserL1_0\" border=0></td>" << endl;
    if ( imgNameMapLL1[1].size() != 0 ) htmlFile << "<td><img src=\"" << imgNameMapLL1[1] << "\" usemap=\"#LaserL1_1\" border=0></td>" << endl;
    htmlFile << "</tr>" << endl;
    htmlFile << "</table>" << endl;
    htmlFile << "<br>" << endl;
  }

  if ( imgNameMapLD[0].size() != 0 || imgNameMapLD[1].size() != 0 ) {
    htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
    htmlFile << "<tr align=\"center\">" << endl;
    if ( imgNameMapLD[0].size() != 0 ) htmlFile << "<td><img src=\"" << imgNameMapLD[0] << "\" usemap=\"#Led_0\" border=0></td>" << endl;
    if ( imgNameMapLD[1].size() != 0 ) htmlFile << "<td><img src=\"" << imgNameMapLD[1] << "\" usemap=\"#Led_1\" border=0></td>" << endl;
    htmlFile << "</tr>" << endl;
    htmlFile << "</table>" << endl;
    htmlFile << "<br>" << endl;
  }

  if ( imgNameMapP[0].size() != 0 || imgNameMapP[1].size() != 0 ) {
    htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
    htmlFile << "<tr align=\"center\">" << endl;
    if ( imgNameMapP[0].size() != 0 ) htmlFile << "<td><img src=\"" << imgNameMapP[0] << "\" usemap=\"#Pedestal_0\" border=0></td>" << endl;
    if ( imgNameMapP[1].size() != 0 ) htmlFile << "<td><img src=\"" << imgNameMapP[1] << "\" usemap=\"#Pedestal_1\" border=0></td>" << endl;
    htmlFile << "</tr>" << endl;
    htmlFile << "</table>" << endl;
    htmlFile << "<br>" << endl;
  }

  if ( imgNameMapTP[0].size() != 0 || imgNameMapTP[1].size() != 0 ) {
    htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
    htmlFile << "<tr align=\"center\">" << endl;
    if ( imgNameMapTP[0].size() != 0 ) htmlFile << "<td><img src=\"" << imgNameMapTP[0] << "\" usemap=\"#TestPulse_0\" border=0></td>" << endl;
    if ( imgNameMapTP[1].size() != 0 ) htmlFile << "<td><img src=\"" << imgNameMapTP[1] << "\" usemap=\"#TestPulse_1\" border=0></td>" << endl;
    htmlFile << "</tr>" << endl;
    htmlFile << "</table>" << endl;
    htmlFile << "<br>" << endl;
  }

  if ( imgNameMapC[0].size() != 0 || imgNameMapC[1].size() != 0 ) {
    htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
    htmlFile << "<tr align=\"center\">" << endl;
    if ( imgNameMapC[0].size() != 0 ) htmlFile << "<td><img src=\"" << imgNameMapC[0] << "\" usemap=\"#Cosmic_0\" border=0></td>" << endl;
    if ( imgNameMapC[1].size() != 0 ) htmlFile << "<td><img src=\"" << imgNameMapC[1] << "\" usemap=\"#Cosmic_1\" border=0></td>" << endl;
    htmlFile << "</tr>" << endl;
    htmlFile << "</table>" << endl;
    htmlFile << "<br>" << endl;
  }

  if ( imgNameMapTM[0].size() != 0 || imgNameMapTM[1].size() != 0 ) {
    htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
    htmlFile << "<tr align=\"center\">" << endl;
    if ( imgNameMapTM[0].size() != 0 ) htmlFile << "<td><img src=\"" << imgNameMapTM[0] << "\" usemap=\"#Timing_0\" border=0></td>" << endl;
    if ( imgNameMapTM[1].size() != 0 ) htmlFile << "<td><img src=\"" << imgNameMapTM[1] << "\" usemap=\"#Timing_1\" border=0></td>" << endl;
    htmlFile << "</tr>" << endl;
    htmlFile << "</table>" << endl;
    htmlFile << "<br>" << endl;
  }

  if ( imgNameMapTTEmulError[0].size() != 0 || imgNameMapTTEmulError[1].size() != 0 ) {
    htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
    htmlFile << "<tr align=\"center\">" << endl;
    if ( imgNameMapTTEmulError[0].size() != 0 ) htmlFile << "<td><img src=\"" << imgNameMapTTEmulError[0] << "\" usemap=\"#TriggerTower_0\" border=0></td>" << endl;
    if ( imgNameMapTTEmulError[1].size() != 0 ) htmlFile << "<td><img src=\"" << imgNameMapTTEmulError[1] << "\" usemap=\"#TriggerTower_1\" border=0></td>" << endl;
    htmlFile << "</tr>" << endl;
    htmlFile << "</table>" << endl;
    htmlFile << "<br>" << endl;
  }

  if ( imgNameMapTTEt[0].size() != 0 || imgNameMapTTEt[1].size() != 0 ) {
    htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
    htmlFile << "<tr align=\"center\">" << endl;
    if ( imgNameMapTTEt[0].size() != 0 ) htmlFile << "<td><img src=\"" << imgNameMapTTEt[0] << "\" usemap=\"#TriggerTower_0\" border=0></td>" << endl;
    if ( imgNameMapTTEt[1].size() != 0 ) htmlFile << "<td><img src=\"" << imgNameMapTTEt[1] << "\" usemap=\"#TriggerTower_1\" border=0></td>" << endl;
    htmlFile << "</tr>" << endl;
    htmlFile << "</table>" << endl;
    htmlFile << "<br>" << endl;
  }

  delete cMap;

  gStyle->SetPaintTextFormat();

  if ( imgNameMapI[0].size() != 0 || imgNameMapI[1].size() != 0 ) this->writeMap( htmlFile, "Integrity" );
  if ( imgNameMapO[0].size() != 0 || imgNameMapO[1].size() != 0 ) this->writeMap( htmlFile, "Occupancy" );
  if ( imgNameMapDF[0].size() != 0 || imgNameMapDF[1].size() != 0 ) this->writeMap( htmlFile, "StatusFlags" );
  if ( imgNameMapPO[0].size() != 0 || imgNameMapPO[1].size() != 0 ) this->writeMap( htmlFile, "PedestalOnline" );
  if ( imgNameMapLL1[0].size() != 0 || imgNameMapLL1[1].size() != 0 ) this->writeMap( htmlFile, "LaserL1" );
  if ( imgNameMapLD[0].size() != 0 || imgNameMapLD[1].size() != 0 ) this->writeMap( htmlFile, "Led" );
  if ( imgNameMapP[0].size() != 0  || imgNameMapP[1].size() != 0 ) this->writeMap( htmlFile, "Pedestal" );
  if ( imgNameMapTP[0].size() != 0 || imgNameMapTP[1].size() != 0 ) this->writeMap( htmlFile, "TestPulse" );

  if ( imgNameMapC[0].size() != 0 || imgNameMapC[1].size() != 0 ) this->writeMap( htmlFile, "Cosmic" );
  if ( imgNameMapTM[0].size() != 0 || imgNameMapTM[1].size() != 0 ) this->writeMap( htmlFile, "Timing" );
  if ( imgNameMapTTEt[0].size() != 0 || imgNameMapTTEt[1].size() != 0 ) this->writeMap( htmlFile, "TriggerTower" );
  if ( imgNameMapTTEmulError[0].size() != 0 || imgNameMapTTEmulError[1].size() != 0 ) this->writeMap( htmlFile, "TriggerTower" );

  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  htmlFile.close();

  gStyle->SetTitleH( saveHeigth );
  gStyle->SetTitleFontSize( saveFontSize );

}

void EESummaryClient::writeMap( std::ofstream& hf, const char* mapname ) {

  std::map<std::string, std::string> refhtml;
  refhtml["Integrity"] = "EEIntegrityClient.html";
  refhtml["Occupancy"] = "EEIntegrityClient.html";
  refhtml["StatusFlags"] = "EEStatusFlagsClient.html";
  refhtml["PedestalOnline"] = "EEPedestalOnlineClient.html";
  refhtml["LaserL1"] = "EELaserClient.html";
  refhtml["Led"] = "EELedClient.html";
  refhtml["Pedestal"] = "EEPedestalClient.html";
  refhtml["TestPulse"] = "EETestPulseClient.html";

  refhtml["Cosmic"] = "EECosmicClient.html";
  refhtml["Timing"] = "EETimingClient.html";
  refhtml["TriggerTower"] = "EETriggerTowerClient.html";

  const int A0 =  38;
  const int A1 = 334;
  const int B0 =  33;
  const int B1 = 312;

  const int C0 = 34;
  const int C1 = 148;

  hf << "<map name=\"" << mapname << "_0\">" << std::endl;
  for( unsigned int sm=0; sm<superModules_.size(); sm++ ) {
    if( superModules_[sm] >= 1 && superModules_[sm] <= 9 ) {
      int i=superModules_[sm]-1;
      int j=superModules_[sm];
      int x0 = (A0+A1)/2 + int(C0*cos(M_PI/2+3*2*M_PI/9-i*2*M_PI/9));
      int x1 = (A0+A1)/2 + int(C0*cos(M_PI/2+3*2*M_PI/9-j*2*M_PI/9));
      int x2 = (A0+A1)/2 + int(C1*cos(M_PI/2+3*2*M_PI/9-j*2*M_PI/9));
      int x3 = (A0+A1)/2 + int(C1*cos(M_PI/2+3*2*M_PI/9-i*2*M_PI/9));
      int y0 = (B0+B1)/2 - int(C0*sin(M_PI/2+3*2*M_PI/9-i*2*M_PI/9));
      int y1 = (B0+B1)/2 - int(C0*sin(M_PI/2+3*2*M_PI/9-j*2*M_PI/9));
      int y2 = (B0+B1)/2 - int(C1*sin(M_PI/2+3*2*M_PI/9-j*2*M_PI/9));
      int y3 = (B0+B1)/2 - int(C1*sin(M_PI/2+3*2*M_PI/9-i*2*M_PI/9));
      hf << "<area title=\"" << Numbers::sEE(superModules_[sm])
         << "\" shape=\"poly\" href=\"" << refhtml[mapname]
         << "#" << Numbers::sEE(superModules_[sm])
         << "\" coords=\"" << x0 << ", " << y0 << ", "
                           << x1 << ", " << y1 << ", "
                           << x2 << ", " << y2 << ", "
                           << x3 << ", " << y3 << "\">"
         << std::endl;
    }
  }
  hf << "</map>" << std::endl;

  hf << "<map name=\"" << mapname << "_1\">" << std::endl;
  for( unsigned int sm=0; sm<superModules_.size(); sm++ ) {
    if( superModules_[sm] >= 10 && superModules_[sm] <= 18 ) {
      int i=superModules_[sm]-9-1;
      int j=superModules_[sm]-9;
      int x0 = (A0+A1)/2 + int(C0*cos(M_PI/2-3*2*M_PI/9+i*2*M_PI/9));
      int x1 = (A0+A1)/2 + int(C0*cos(M_PI/2-3*2*M_PI/9+j*2*M_PI/9));
      int x2 = (A0+A1)/2 + int(C1*cos(M_PI/2-3*2*M_PI/9+j*2*M_PI/9));
      int x3 = (A0+A1)/2 + int(C1*cos(M_PI/2-3*2*M_PI/9+i*2*M_PI/9));
      int y0 = (B0+B1)/2 - int(C0*sin(M_PI/2-3*2*M_PI/9+i*2*M_PI/9));
      int y1 = (B0+B1)/2 - int(C0*sin(M_PI/2-3*2*M_PI/9+j*2*M_PI/9));
      int y2 = (B0+B1)/2 - int(C1*sin(M_PI/2-3*2*M_PI/9+j*2*M_PI/9));
      int y3 = (B0+B1)/2 - int(C1*sin(M_PI/2-3*2*M_PI/9+i*2*M_PI/9));
      hf << "<area title=\"" << Numbers::sEE(superModules_[sm])
         << "\" shape=\"poly\" href=\"" << refhtml[mapname]
         << "#" << Numbers::sEE(superModules_[sm])
         << "\" coords=\"" << x0 << ", " << y0 << ", "
                           << x1 << ", " << y1 << ", "
                           << x2 << ", " << y2 << ", "
                           << x3 << ", " << y3 << "\">"
         << std::endl;
    }
  }
  hf << "</map>" << std::endl;

}

