/*
 * \file EBSummaryClient.cc
 *
 * $Date: 2008/05/06 14:32:54 $
 * $Revision: 1.141 $
 * \author G. Della Ricca
 *
*/

#include <memory>
#include <iostream>
#include <iomanip>
#include <map>

#include "TCanvas.h"
#include "TStyle.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "OnlineDB/EcalCondDB/interface/RunTag.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"

#include <DQM/EcalCommon/interface/UtilsClient.h>
#include <DQM/EcalCommon/interface/Numbers.h>

#include <DQM/EcalBarrelMonitorClient/interface/EBCosmicClient.h>
#include <DQM/EcalBarrelMonitorClient/interface/EBStatusFlagsClient.h>
#include <DQM/EcalBarrelMonitorClient/interface/EBIntegrityClient.h>
#include <DQM/EcalBarrelMonitorClient/interface/EBLaserClient.h>
#include <DQM/EcalBarrelMonitorClient/interface/EBPedestalClient.h>
#include <DQM/EcalBarrelMonitorClient/interface/EBPedestalOnlineClient.h>
#include <DQM/EcalBarrelMonitorClient/interface/EBTestPulseClient.h>
#include <DQM/EcalBarrelMonitorClient/interface/EBBeamCaloClient.h>
#include <DQM/EcalBarrelMonitorClient/interface/EBBeamHodoClient.h>
#include <DQM/EcalBarrelMonitorClient/interface/EBTriggerTowerClient.h>
#include <DQM/EcalBarrelMonitorClient/interface/EBClusterClient.h>
#include <DQM/EcalBarrelMonitorClient/interface/EBTimingClient.h>

#include <DQM/EcalBarrelMonitorClient/interface/EBSummaryClient.h>

using namespace cms;
using namespace edm;
using namespace std;

EBSummaryClient::EBSummaryClient(const ParameterSet& ps){

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

  // summary maps
  meIntegrity_      = 0;
  meOccupancy_      = 0;
  meStatusFlags_    = 0;
  mePedestalOnline_ = 0;
  meLaserL1_        = 0;
  meLaserL1PN_      = 0;
  mePedestal_       = 0;
  mePedestalPN_     = 0;
  meTestPulse_      = 0;
  meTestPulsePN_    = 0;
  meGlobalSummary_  = 0;

  meCosmic_         = 0;
  meTiming_         = 0;
  meTriggerTowerEt_        = 0;
  meTriggerTowerEmulError_ = 0;

  // summary errors
  meIntegrityErr_       = 0;
  meOccupancy1D_        = 0;
  meStatusFlagsErr_     = 0;
  mePedestalOnlineErr_  = 0;
  meLaserL1Err_         = 0;
  meLaserL1PNErr_       = 0;
  mePedestalErr_        = 0;
  mePedestalPNErr_      = 0;
  meTestPulseErr_       = 0;
  meTestPulsePNErr_     = 0;

}

EBSummaryClient::~EBSummaryClient(){

}

void EBSummaryClient::beginJob(DQMStore* dqmStore){

  dqmStore_ = dqmStore;

  if ( debug_ ) cout << "EBSummaryClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EBSummaryClient::beginRun(void){

  if ( debug_ ) cout << "EBSummaryClient: beginRun" << endl;

  jevt_ = 0;

  this->setup();

}

void EBSummaryClient::endJob(void) {

  if ( debug_ ) cout << "EBSummaryClient: endJob, ievt = " << ievt_ << endl;

  this->cleanup();

}

void EBSummaryClient::endRun(void) {

  if ( debug_ ) cout << "EBSummaryClient: endRun, jevt = " << jevt_ << endl;

  this->cleanup();

}

void EBSummaryClient::setup(void) {

  char histo[200];

  dqmStore_->setCurrentFolder( prefixME_ + "/EBSummaryClient" );

  if ( meIntegrity_ ) dqmStore_->removeElement( meIntegrity_->getName() );
  sprintf(histo, "EBIT integrity quality summary");
  meIntegrity_ = dqmStore_->book2D(histo, histo, 360, 0., 360., 170, -85., 85.);
  meIntegrity_->setAxisTitle("jphi", 1);
  meIntegrity_->setAxisTitle("jeta", 2);

  if ( meIntegrityErr_ ) dqmStore_->removeElement( meIntegrityErr_->getName() );
  sprintf(histo, "EBIT integrity quality errors summary");
  meIntegrityErr_ = dqmStore_->book1D(histo, histo, 36, 1, 37);
  for (int i = 0; i < 36; i++) {
    meIntegrityErr_->setBinLabel(i+1, Numbers::sEB(i+1).c_str(), 1);
  }

  if ( meOccupancy_ ) dqmStore_->removeElement( meOccupancy_->getName() );
  sprintf(histo, "EBOT digi occupancy summary");
  meOccupancy_ = dqmStore_->book2D(histo, histo, 360, 0., 360., 170, -85., 85.);
  meOccupancy_->setAxisTitle("jphi", 1);
  meOccupancy_->setAxisTitle("jeta", 2);

  if ( meOccupancy1D_ ) dqmStore_->removeElement( meOccupancy1D_->getName() );
  sprintf(histo, "EBOT digi occupancy summary 1D");
  meOccupancy1D_ = dqmStore_->book1D(histo, histo, 36, 1, 37);
  for (int i = 0; i < 36; i++) {
    meOccupancy1D_->setBinLabel(i+1, Numbers::sEB(i+1).c_str(), 1);
  }

  if ( meStatusFlags_ ) dqmStore_->removeElement( meStatusFlags_->getName() );
  sprintf(histo, "EBSFT front-end status summary");
  meStatusFlags_ = dqmStore_->book2D(histo, histo, 72, 0., 72., 34, -17., 17.);
  meStatusFlags_->setAxisTitle("jphi'", 1);
  meStatusFlags_->setAxisTitle("jeta'", 2);

  if ( meStatusFlagsErr_ ) dqmStore_->removeElement( meStatusFlagsErr_->getName() );
  sprintf(histo, "EBSFT front-end status errors summary");
  meStatusFlagsErr_ = dqmStore_->book1D(histo, histo, 36, 1, 37);
  for (int i = 0; i < 36; i++) {
    meStatusFlagsErr_->setBinLabel(i+1, Numbers::sEB(i+1).c_str(), 1);
  }

  if ( mePedestalOnline_ ) dqmStore_->removeElement( mePedestalOnline_->getName() );
  sprintf(histo, "EBPOT pedestal quality summary G12");
  mePedestalOnline_ = dqmStore_->book2D(histo, histo, 360, 0., 360., 170, -85., 85.);
  mePedestalOnline_->setAxisTitle("jphi", 1);
  mePedestalOnline_->setAxisTitle("jeta", 2);

  if ( mePedestalOnlineErr_ ) dqmStore_->removeElement( mePedestalOnlineErr_->getName() );
  sprintf(histo, "EBPOT pedestal quality errors summary G12");
  mePedestalOnlineErr_ = dqmStore_->book1D(histo, histo, 36, 1, 37);
  for (int i = 0; i < 36; i++) {
    mePedestalOnlineErr_->setBinLabel(i+1, Numbers::sEB(i+1).c_str(), 1);
  }

  if ( meLaserL1_ ) dqmStore_->removeElement( meLaserL1_->getName() );
  sprintf(histo, "EBLT laser quality summary L1");
  meLaserL1_ = dqmStore_->book2D(histo, histo, 360, 0., 360., 170, -85., 85.);
  meLaserL1_->setAxisTitle("jphi", 1);
  meLaserL1_->setAxisTitle("jeta", 2);

  if ( meLaserL1Err_ ) dqmStore_->removeElement( meLaserL1Err_->getName() );
  sprintf(histo, "EBLT laser quality errors summary L1");
  meLaserL1Err_ = dqmStore_->book1D(histo, histo, 36, 1, 37);
  for (int i = 0; i < 36; i++) {
    meLaserL1Err_->setBinLabel(i+1, Numbers::sEB(i+1).c_str(), 1);
  }

  if ( meLaserL1PN_ ) dqmStore_->removeElement( meLaserL1PN_->getName() );
  sprintf(histo, "EBLT PN laser quality summary L1");
  meLaserL1PN_ = dqmStore_->book2D(histo, histo, 90, 0., 90., 20, -10., 10.);
  meLaserL1PN_->setAxisTitle("jphi", 1);
  meLaserL1PN_->setAxisTitle("jeta", 2);

  if ( meLaserL1PNErr_ ) dqmStore_->removeElement( meLaserL1PNErr_->getName() );
  sprintf(histo, "EBLT PN laser quality errors summary L1");
  meLaserL1PNErr_ = dqmStore_->book1D(histo, histo, 36, 1, 37);
  for (int i = 0; i < 36; i++) {
    meLaserL1PNErr_->setBinLabel(i+1, Numbers::sEB(i+1).c_str(), 1);
  }

  if( mePedestal_ ) dqmStore_->removeElement( mePedestal_->getName() );
  sprintf(histo, "EBPT pedestal quality summary");
  mePedestal_ = dqmStore_->book2D(histo, histo, 360, 0., 360., 170, -85., 85.);
  mePedestal_->setAxisTitle("jphi", 1);
  mePedestal_->setAxisTitle("jeta", 2);

  if( mePedestalErr_ ) dqmStore_->removeElement( mePedestalErr_->getName() );
  sprintf(histo, "EBPT pedestal quality errors summary");
  mePedestalErr_ = dqmStore_->book1D(histo, histo, 36, 1, 37);
  for (int i = 0; i < 36; i++) {
    mePedestalErr_->setBinLabel(i+1, Numbers::sEB(i+1).c_str(), 1);
  }

  if( mePedestalPN_ ) dqmStore_->removeElement( mePedestalPN_->getName() );
  sprintf(histo, "EBPT PN pedestal quality summary");
  mePedestalPN_ = dqmStore_->book2D(histo, histo, 90, 0., 90., 20, -10, 10.);
  mePedestalPN_->setAxisTitle("jphi", 1);
  mePedestalPN_->setAxisTitle("jeta", 2);

  if( mePedestalPNErr_ ) dqmStore_->removeElement( mePedestalPNErr_->getName() );
  sprintf(histo, "EBPT PN pedestal quality errors summary");
  mePedestalPNErr_ = dqmStore_->book1D(histo, histo, 36, 1, 37);
  for (int i = 0; i < 36; i++) {
    mePedestalPNErr_->setBinLabel(i+1, Numbers::sEB(i+1).c_str(), 1);
  }

  if( meTestPulse_ ) dqmStore_->removeElement( meTestPulse_->getName() );
  sprintf(histo, "EBTPT test pulse quality summary");
  meTestPulse_ = dqmStore_->book2D(histo, histo, 360, 0., 360., 170, -85., 85.);
  meTestPulse_->setAxisTitle("jphi", 1);
  meTestPulse_->setAxisTitle("jeta", 2);

  if( meTestPulseErr_ ) dqmStore_->removeElement( meTestPulseErr_->getName() );
  sprintf(histo, "EBTPT test pulse quality errors summary");
  meTestPulseErr_ = dqmStore_->book1D(histo, histo, 36, 1, 37);
  for (int i = 0; i < 36; i++) {
    meTestPulseErr_->setBinLabel(i+1, Numbers::sEB(i+1).c_str(), 1);
  }

  if( meTestPulsePN_ ) dqmStore_->removeElement( meTestPulsePN_->getName() );
  sprintf(histo, "EBTPT PN test pulse quality summary");
  meTestPulsePN_ = dqmStore_->book2D(histo, histo, 90, 0., 90., 20, -10., 10.);
  meTestPulsePN_->setAxisTitle("jphi", 1);
  meTestPulsePN_->setAxisTitle("jeta", 2);

  if( meTestPulsePNErr_ ) dqmStore_->removeElement( meTestPulsePNErr_->getName() );
  sprintf(histo, "EBTPT PN test pulse quality errors summary");
  meTestPulsePNErr_ = dqmStore_->book1D(histo, histo, 36, 1, 37);
  for (int i = 0; i < 36; i++) {
    meTestPulsePNErr_->setBinLabel(i+1, Numbers::sEB(i+1).c_str(), 1);
  }

  if( meCosmic_ ) dqmStore_->removeElement( meCosmic_->getName() );
  sprintf(histo, "EBCT cosmic summary");
  meCosmic_ = dqmStore_->book2D(histo, histo, 360, 0., 360., 170, -85., 85.);
  meCosmic_->setAxisTitle("jphi", 1);
  meCosmic_->setAxisTitle("jeta", 2);

  if( meTiming_ ) dqmStore_->removeElement( meTiming_->getName() );
  sprintf(histo, "EBTMT timing quality summary");
  meTiming_ = dqmStore_->book2D(histo, histo, 360, 0., 360., 170, -85., 85.);
  meTiming_->setAxisTitle("jphi", 1);
  meTiming_->setAxisTitle("jeta", 2);

  if( meTriggerTowerEt_ ) dqmStore_->removeElement( meTriggerTowerEt_->getName() );
  sprintf(histo, "EBTTT Et trigger tower summary");
  meTriggerTowerEt_ = dqmStore_->book2D(histo, histo, 72, 0., 72., 34, -17., 17.);
  meTriggerTowerEt_->setAxisTitle("jphi'", 1);
  meTriggerTowerEt_->setAxisTitle("jeta'", 2);

  if( meTriggerTowerEmulError_ ) dqmStore_->removeElement( meTriggerTowerEmulError_->getName() );
  sprintf(histo, "EBTTT emulator error quality summary");
  meTriggerTowerEmulError_ = dqmStore_->book2D(histo, histo, 72, 0., 72., 34, -17., 17.);
  meTriggerTowerEmulError_->setAxisTitle("jphi'", 1);
  meTriggerTowerEmulError_->setAxisTitle("jeta'", 2);

  if( meGlobalSummary_ ) dqmStore_->removeElement( meGlobalSummary_->getName() );
  sprintf(histo, "EB global summary");
  meGlobalSummary_ = dqmStore_->book2D(histo, histo, 360, 0., 360., 170, -85., 85.);
  meGlobalSummary_->setAxisTitle("jphi", 1);
  meGlobalSummary_->setAxisTitle("jeta", 2);

  // summary for DQM GUI

  MonitorElement* me;

  dqmStore_->setCurrentFolder( prefixME_ + "/EventInfo" );

  sprintf(histo, "reportSummary");
  me = dqmStore_->bookFloat(histo);

  dqmStore_->setCurrentFolder( prefixME_ + "/EventInfo/reportSummaryContents" );

  for (int i = 0; i < 36; i++) {
    sprintf(histo, "status %s", Numbers::sEB(i+1).c_str());
    me = dqmStore_->bookFloat(histo);
  }

  dqmStore_->setCurrentFolder( prefixME_ + "/EventInfo" );

  sprintf(histo, "reportSummaryMap");
  me = dqmStore_->book2D(histo, histo, 72, 0., 72., 34, 0., 34);
  me->setAxisTitle("jphi", 1);
  me->setAxisTitle("jeta", 2);

}

void EBSummaryClient::cleanup(void) {

  if ( ! enableCleanup_ ) return;

  dqmStore_->setCurrentFolder( prefixME_ + "/EBSummaryClient" );

  if ( meIntegrity_ ) dqmStore_->removeElement( meIntegrity_->getName() );
  meIntegrity_ = 0;

  if ( meIntegrityErr_ ) dqmStore_->removeElement( meIntegrityErr_->getName() );
  meIntegrityErr_ = 0;

  if ( meOccupancy_ ) dqmStore_->removeElement( meOccupancy_->getName() );
  meOccupancy_ = 0;

  if ( meOccupancy1D_ ) dqmStore_->removeElement( meOccupancy1D_->getName() );
  meOccupancy1D_ = 0;

  if ( meStatusFlags_ ) dqmStore_->removeElement( meStatusFlags_->getName() );
  meStatusFlags_ = 0;

  if ( meStatusFlagsErr_ ) dqmStore_->removeElement( meStatusFlagsErr_->getName() );
  meStatusFlagsErr_ = 0;

  if ( mePedestalOnline_ ) dqmStore_->removeElement( mePedestalOnline_->getName() );
  mePedestalOnline_ = 0;

  if ( mePedestalOnlineErr_ ) dqmStore_->removeElement( mePedestalOnlineErr_->getName() );
  mePedestalOnlineErr_ = 0;

  if ( meLaserL1_ ) dqmStore_->removeElement( meLaserL1_->getName() );
  meLaserL1_ = 0;

  if ( meLaserL1Err_ ) dqmStore_->removeElement( meLaserL1Err_->getName() );
  meLaserL1Err_ = 0;

  if ( meLaserL1PN_ ) dqmStore_->removeElement( meLaserL1PN_->getName() );
  meLaserL1PN_ = 0;

  if ( meLaserL1PNErr_ ) dqmStore_->removeElement( meLaserL1PNErr_->getName() );
  meLaserL1PNErr_ = 0;

  if ( mePedestal_ ) dqmStore_->removeElement( mePedestal_->getName() );
  mePedestal_ = 0;

  if ( mePedestalErr_ ) dqmStore_->removeElement( mePedestalErr_->getName() );
  mePedestalErr_ = 0;

  if ( mePedestalPN_ ) dqmStore_->removeElement( mePedestalPN_->getName() );
  mePedestalPN_ = 0;

  if ( mePedestalPNErr_ ) dqmStore_->removeElement( mePedestalPNErr_->getName() );
  mePedestalPNErr_ = 0;

  if ( meTestPulse_ ) dqmStore_->removeElement( meTestPulse_->getName() );
  meTestPulse_ = 0;

  if ( meTestPulseErr_ ) dqmStore_->removeElement( meTestPulseErr_->getName() );
  meTestPulseErr_ = 0;

  if ( meTestPulsePN_ ) dqmStore_->removeElement( meTestPulsePN_->getName() );
  meTestPulsePN_ = 0;

  if ( meTestPulsePNErr_ ) dqmStore_->removeElement( meTestPulsePNErr_->getName() );
  meTestPulsePNErr_ = 0;

  if ( meCosmic_ ) dqmStore_->removeElement( meCosmic_->getName() );
  meCosmic_ = 0;

  if ( meTiming_ ) dqmStore_->removeElement( meTiming_->getName() );
  meTiming_ = 0;

  if ( meTriggerTowerEt_ ) dqmStore_->removeElement( meTriggerTowerEt_->getName() );
  meTriggerTowerEt_ = 0;

  if ( meTriggerTowerEmulError_ ) dqmStore_->removeElement( meTriggerTowerEmulError_->getName() );
  meTriggerTowerEmulError_ = 0;

  if ( meGlobalSummary_ ) dqmStore_->removeElement( meGlobalSummary_->getName() );
  meGlobalSummary_ = 0;

}

bool EBSummaryClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov) {

  bool status = true;

  return status;

}

void EBSummaryClient::analyze(void){

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( debug_ ) cout << "EBSummaryClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;
  }

  for ( int iex = 1; iex <= 170; iex++ ) {
    for ( int ipx = 1; ipx <= 360; ipx++ ) {

      meIntegrity_->setBinContent( ipx, iex, -1. );
      meOccupancy_->setBinContent( ipx, iex, 0. );
      meStatusFlags_->setBinContent( ipx, iex, -1. );
      mePedestalOnline_->setBinContent( ipx, iex, -1. );

      meLaserL1_->setBinContent( ipx, iex, -1. );
      mePedestal_->setBinContent( ipx, iex, -1. );
      meTestPulse_->setBinContent( ipx, iex, -1. );

      meCosmic_->setBinContent( ipx, iex, 0. );
      meTiming_->setBinContent( ipx, iex, -1. );

      meGlobalSummary_->setBinContent( ipx, iex, -1. );

    }
  }

  for ( int iex = 1; iex <= 20; iex++ ) {
    for ( int ipx = 1; ipx <= 90; ipx++ ) {

      meLaserL1PN_->setBinContent( ipx, iex, -1. );
      mePedestalPN_->setBinContent( ipx, iex, -1. );
      meTestPulsePN_->setBinContent( ipx, iex, -1. );

    }
  }

  for ( int iex = 1; iex <= 34; iex++ ) {
    for ( int ipx = 1; ipx <= 72; ipx++ ) {
      meTriggerTowerEt_->setBinContent( ipx, iex, 0. );
      meTriggerTowerEmulError_->setBinContent( ipx, iex, -1. );
    }
  }

  meIntegrity_->setEntries( 0 );
  meIntegrityErr_->Reset();
  meOccupancy_->setEntries( 0 );
  meOccupancy1D_->Reset();
  meStatusFlags_->setEntries( 0 );
  meStatusFlagsErr_->Reset();
  mePedestalOnline_->setEntries( 0 );
  mePedestalOnlineErr_->Reset();

  meLaserL1_->setEntries( 0 );
  meLaserL1Err_->Reset();
  meLaserL1PN_->setEntries( 0 );
  meLaserL1PNErr_->Reset();
  mePedestal_->setEntries( 0 );
  mePedestalErr_->Reset();
  mePedestalPN_->setEntries( 0 );
  mePedestalPNErr_->Reset();
  meTestPulse_->setEntries( 0 );
  meTestPulseErr_->Reset();
  meTestPulsePN_->setEntries( 0 );
  meTestPulsePNErr_->Reset();

  meCosmic_->setEntries( 0 );
  meTiming_->setEntries( 0 );
  meTriggerTowerEt_->setEntries( 0 );
  meTriggerTowerEmulError_->setEntries( 0 );

  meGlobalSummary_->setEntries( 0 );

  for ( unsigned int i=0; i<clients_.size(); i++ ) {

    EBIntegrityClient* ebic = dynamic_cast<EBIntegrityClient*>(clients_[i]);
    EBStatusFlagsClient* ebsfc = dynamic_cast<EBStatusFlagsClient*>(clients_[i]);
    EBPedestalOnlineClient* ebpoc = dynamic_cast<EBPedestalOnlineClient*>(clients_[i]);

    EBLaserClient* eblc = dynamic_cast<EBLaserClient*>(clients_[i]);
    EBPedestalClient* ebpc = dynamic_cast<EBPedestalClient*>(clients_[i]);
    EBTestPulseClient* ebtpc = dynamic_cast<EBTestPulseClient*>(clients_[i]);

    EBCosmicClient* ebcc = dynamic_cast<EBCosmicClient*>(clients_[i]);
    EBTimingClient* ebtmc = dynamic_cast<EBTimingClient*>(clients_[i]);
    EBTriggerTowerClient* ebtttc = dynamic_cast<EBTriggerTowerClient*>(clients_[i]);

    MonitorElement *me;
    MonitorElement *me_01, *me_02, *me_03;
    MonitorElement *me_04, *me_05;
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

              mePedestalOnline_->setBinContent( ipx, iex, xval );
              if ( xval == 0 ) mePedestalOnlineErr_->Fill( ism );

            }

          }

          if ( eblc ) {

            me = eblc->meg01_[ism-1];

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

              if ( me->getEntries() != 0 ) {
                meLaserL1_->setBinContent( ipx, iex, xval );
                if ( xval == 0 ) meLaserL1Err_->Fill( ism );
              }

            }

          }

          if ( ebpc ) {

            me_01 = ebpc->meg01_[ism-1];
            me_02 = ebpc->meg02_[ism-1];
            me_03 = ebpc->meg03_[ism-1];

            if (me_01 && me_02 && me_03 ) {
              float xval=2;
              float val_01=me_01->getBinContent(ie,ip);
              float val_02=me_02->getBinContent(ie,ip);
              float val_03=me_03->getBinContent(ie,ip);

              vector<float> maskedVal, unmaskedVal;
              (val_01>2) ? maskedVal.push_back(val_01) : unmaskedVal.push_back(val_01);
              (val_02>2) ? maskedVal.push_back(val_02) : unmaskedVal.push_back(val_02);
              (val_03>2) ? maskedVal.push_back(val_03) : unmaskedVal.push_back(val_03);

              float brightColor=-1, darkColor=-1;
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

              int iex;
              int ipx;

              if ( ism <= 18 ) {
                iex = 1+(85-ie);
                ipx = ip+20*(ism-1);
              } else {
                iex = 85+ie;
                ipx = 1+(20-ip)+20*(ism-19);
              }

              if ( me_01->getEntries() != 0 && me_02->getEntries() != 0 && me_03->getEntries() != 0 ) {
                mePedestal_->setBinContent( ipx, iex, xval );
                if ( xval == 0 ) mePedestalErr_->Fill( ism );
              }

            }

          }

          if ( ebtpc ) {

            me_01 = ebtpc->meg01_[ism-1];
            me_02 = ebtpc->meg02_[ism-1];
            me_03 = ebtpc->meg03_[ism-1];

            if (me_01 && me_02 && me_03 ) {
              float xval=2;
              float val_01=me_01->getBinContent(ie,ip);
              float val_02=me_02->getBinContent(ie,ip);
              float val_03=me_03->getBinContent(ie,ip);

              vector<float> maskedVal, unmaskedVal;
              (val_01>2) ? maskedVal.push_back(val_01) : unmaskedVal.push_back(val_01);
              (val_02>2) ? maskedVal.push_back(val_02) : unmaskedVal.push_back(val_02);
              (val_03>2) ? maskedVal.push_back(val_03) : unmaskedVal.push_back(val_03);

              float brightColor=-1, darkColor=-1;
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

              int iex;
              int ipx;

              if ( ism <= 18 ) {
                iex = 1+(85-ie);
                ipx = ip+20*(ism-1);
              } else {
                iex = 85+ie;
                ipx = 1+(20-ip)+20*(ism-19);
              }

              if ( me_01->getEntries() != 0 && me_02->getEntries() != 0 && me_03->getEntries() != 0 ) {
                meTestPulse_->setBinContent( ipx, iex, xval );
                if( xval == 0 ) meTestPulseErr_->Fill( ism );
              }

            }

          }

          if ( ebcc ) {

            h2d = ebcc->h02_[ism-1];

            if ( h2d ) {

              float xval = h2d->GetBinContent( ie, ip );

              int iex;
              int ipx;

              if ( ism <= 18 ) {
                iex = 1+(85-ie);
                ipx = ip+20*(ism-1);
              } else {
                iex = 85+ie;
                ipx = 1+(20-ip)+20*(ism-19);
              }

              meCosmic_->setBinContent( ipx, iex, xval );

            }

          }

          if ( ebtmc ) {

            me = ebtmc->meg01_[ism-1];

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

              meTiming_->setBinContent( ipx, iex, xval );

            }

          }

        }
      }

      for (int ie = 1; ie <= 17; ie++ ) {
        for (int ip = 1; ip <= 4; ip++ ) {

          if ( ebsfc ) {

            me = ebsfc->meh01_[ism-1];

            if ( me ) {

              float xval = -1;

              if ( me->getBinContent( ie, ip ) == -1 ) xval = 2;
              if ( me->getBinContent( ie, ip ) == 0 ) xval = 1;
              if ( me->getBinContent( ie, ip ) > 0 ) xval = 0;

              int iex;
              int ipx;

              if ( ism <= 18 ) {
                iex = 1+(17-ie);
                ipx = ip+4*(ism-1);
              } else {
                iex = 17+ie;
                ipx = 1+(4-ip)+4*(ism-19);
              }

              if ( me->getEntries() != 0 ) {
                meStatusFlags_->setBinContent( ipx, iex, xval );
                if ( xval == 0 ) meStatusFlagsErr_->Fill( ism );
              }

            }

          }

          if ( ebtttc ) {

            me = ebtttc->me_h01_[ism-1];

            bool hasRealDigi = false;

            if ( me ) {

              float xval = me->getBinContent( ie, ip );

              TProfile2D* obj = UtilsClient::getHisto<TProfile2D*>(me);
              if(obj && obj->GetBinEntries(obj->GetBin( ie, ip ))!=0) hasRealDigi = true;

              int iex;
              int ipx;

              if ( ism <= 18 ) {
                iex = 1+(17-ie);
                ipx = ip+4*(ism-1);
              } else {
                iex = 17+ie;
                ipx = 1+(4-ip)+4*(ism-19);
              }

              meTriggerTowerEt_->setBinContent( ipx, iex, xval );

            }

            h2 = ebtttc->l01_[ism-1];

            if ( h2 ) {

              float xval = -1;
              float emulErrorVal = h2->GetBinContent( ie, ip );

              if(!hasRealDigi) xval = 2;
              else if(hasRealDigi && emulErrorVal!=0) xval = 0;
              else xval = 1;

              int iex;
              int ipx;

              if ( ism <= 18 ) {
                iex = 1+(17-ie);
                ipx = ip+4*(ism-1);
              } else {
                iex = 17+ie;
                ipx = 1+(4-ip)+4*(ism-19);
              }

              meTriggerTowerEmulError_->setBinContent( ipx, iex, xval );

            }

          }
        }
      }

      // PN's summaries
      for( int i = 1; i <= 10; i++ ) {
        for( int j = 1; j <= 5; j++ ) {

          if ( ebpc ) {

            me_04 = ebpc->meg04_[ism-1];
            me_05 = ebpc->meg05_[ism-1];

            if( me_04 && me_05) {
              float xval=2;
              float val_04=me_04->getBinContent(i,1);
              float val_05=me_05->getBinContent(i,1);

              vector<float> maskedVal, unmaskedVal;
              (val_04>2) ? maskedVal.push_back(val_04) : unmaskedVal.push_back(val_04);
              (val_05>2) ? maskedVal.push_back(val_05) : unmaskedVal.push_back(val_05);

              float brightColor=-1, darkColor=-1;
              float maxPriority=-1;

              vector<float>::const_iterator Val;
              for(Val=unmaskedVal.begin(); Val<unmaskedVal.end(); Val++) {
                if(priority[*Val]>maxPriority) brightColor=*Val;
              }
              maxPriority=-1;
              for(Val=maskedVal.begin(); Val<maskedVal.end(); Val++) {
                if(priority[*Val]>maxPriority) darkColor=*Val;
              }
              if(unmaskedVal.size()==2) xval = brightColor;
              else if(maskedVal.size()==2) xval = darkColor;
              else {
                if(brightColor==1 && darkColor==5) xval = 5;
                else xval = brightColor;
              }

              int iex;
              int ipx;

              if(ism<=18) {
                iex = i;
                ipx = j+5*(ism-1);
              } else {
                iex = i+10;
                ipx = j+5*(ism-19);
              }

              if ( me_04->getEntries() != 0 && me_05->getEntries() != 0 ) {
                mePedestalPN_->setBinContent( ipx, iex, xval );
                if( xval == 0 ) mePedestalPNErr_->Fill( ism );
              }

            }

          }

          if ( ebtpc ) {

            me_04 = ebtpc->meg04_[ism-1];
            me_05 = ebtpc->meg05_[ism-1];

            if( me_04 && me_05) {
              float xval=2;
              float val_04=me_04->getBinContent(i,1);
              float val_05=me_05->getBinContent(i,1);

              vector<float> maskedVal, unmaskedVal;
              (val_04>2) ? maskedVal.push_back(val_04) : unmaskedVal.push_back(val_04);
              (val_05>2) ? maskedVal.push_back(val_05) : unmaskedVal.push_back(val_05);

              float brightColor=-1, darkColor=-1;
              float maxPriority=-1;

              vector<float>::const_iterator Val;
              for(Val=unmaskedVal.begin(); Val<unmaskedVal.end(); Val++) {
                if(priority[*Val]>maxPriority) brightColor=*Val;
              }
              maxPriority=-1;
              for(Val=maskedVal.begin(); Val<maskedVal.end(); Val++) {
                if(priority[*Val]>maxPriority) darkColor=*Val;
              }
              if(unmaskedVal.size()==2) xval = brightColor;
              else if(maskedVal.size()==2) xval = darkColor;
              else {
                if(brightColor==1 && darkColor==5) xval = 5;
                else xval = brightColor;
              }

              int iex;
              int ipx;

              if(ism<=18) {
                iex = i;
                ipx = j+5*(ism-1);
              } else {
                iex = i+10;
                ipx = j+5*(ism-19);
              }

              if ( me_04->getEntries() != 0 && me_05->getEntries() != 0 ) {
                meTestPulsePN_->setBinContent( ipx, iex, xval );
                if ( xval == 0 ) meTestPulsePNErr_->Fill ( ism );
              }

            }
          }

          if ( eblc ) {

            me = eblc->meg09_[ism-1];

            if( me ) {

              float xval = me->getBinContent(i,1);

              int iex;
              int ipx;

              if(ism<=18) {
                iex = i;
                ipx = j+5*(ism-1);
              } else {
                iex = i+10;
                ipx = j+5*(ism-19);
              }

              if ( me->getEntries() != 0 && me->getEntries() != 0 ) {
                meLaserL1PN_->setBinContent( ipx, iex, xval );
                if ( xval == 0 ) meLaserL1PNErr_->Fill( ism );
              }

            }

          }

        }
      }

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

      if(meIntegrity_ && mePedestalOnline_ && meLaserL1_ && meTiming_ && meStatusFlags_ && meTriggerTowerEmulError_) {

        float xval = -1;
        float val_in = meIntegrity_->getBinContent(ipx,iex);
        float val_po = mePedestalOnline_->getBinContent(ipx,iex);
        float val_ls = meLaserL1_->getBinContent(ipx,iex);
        float val_tm = meTiming_->getBinContent(ipx,iex);
        float val_sf = meStatusFlags_->getBinContent((ipx-1)/5+1,(iex-1)/5+1);
        float val_ee = meTriggerTowerEmulError_->getBinContent((ipx-1)/5+1,(iex-1)/5+1);

        // turn each dark color (masked channel) to bright green
        // for laser&timing turn also yellow into bright green
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

        meGlobalSummary_->setBinContent( ipx, iex, xval );

        if ( xval > -1 ) {
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

  MonitorElement* me;

  float reportSummary = -1.0;
  if ( nValidChannels != 0 )
    reportSummary = 1.0 - float(nGlobalErrors)/float(nValidChannels);
  me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummary");
  if (me) me->Fill(reportSummary);

  char histo[200];

  for (int i = 0; i < 36; i++) {
    float reportSummaryEB = -1.0;
    if ( nValidChannelsEB[i] != 0 )
      reportSummaryEB = 1.0 - float(nGlobalErrorsEB[i])/float(nValidChannelsEB[i]);
    sprintf(histo, "status %s", Numbers::sEB(i+1).c_str());
    me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummaryContents/" + histo);
    if (me) me->Fill(reportSummaryEB);
  }

  me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummaryMap");
  if (me) {

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

        if ( xval > -1 ) {
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

      }
    }

  }

}

void EBSummaryClient::htmlOutput(int run, string& htmlDir, string& htmlName){

  if ( verbose_ ) cout << "Preparing EBSummaryClient html output ..." << endl;

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

  const int csize = 400;

//  const double histMax = 1.e15;

  int pCol3[6] = { 301, 302, 303, 304, 305, 306 };
  int pCol4[10];
  for ( int i = 0; i < 10; i++ ) pCol4[i] = 401+i;

  // dummy histogram labelling the SM's
  TH2C labelGrid("labelGrid","label grid for SM", 18, 0., 360., 2, -85., 85.);
  for ( short sm=0; sm<36; sm++ ) {
    int x = 1 + sm%18;
    int y = 1 + sm/18;
    labelGrid.SetBinContent(x, y, Numbers::iEB(sm+1));
  }
  labelGrid.SetMarkerSize(2);
  labelGrid.SetMinimum(-18.01);

  TH2C labelGridPN("labelGridPN","label grid for SM", 18, 0., 90., 2, -10., 10.);
  for ( short sm=0; sm<36; sm++ ) {
    int x = 1 + sm%18;
    int y = 1 + sm/18;
    labelGridPN.SetBinContent(x, y, Numbers::iEB(sm+1));
  }
  labelGridPN.SetMarkerSize(4);
  labelGridPN.SetMinimum(-18.01);

  TH2C labelGridTT("labelGridTT","label grid for SM", 18, 0., 72., 2, -17., 17.);
  for ( short sm=0; sm<36; sm++ ) {
    int x = 1 + sm%18;
    int y = 1 + sm/18;
    labelGridTT.SetBinContent(x, y, Numbers::iEB(sm+1));
  }
  labelGridTT.SetMarkerSize(2);
  labelGridTT.SetMinimum(-18.01);

  string imgNameMapI, imgNameMapO;
  string imgNameMapSF;
  string imgNameMapPO;
  string imgNameMapLL1, imgNameMapLL1_PN;
  string imgNameMapP, imgNameMapP_PN;
  string imgNameMapTP, imgNameMapTP_PN;
  string imgNameMapC;
  string imgNameMapTM;
  string imgNameMapTTEt;
  string imgNameMapTTEmulError;
  string imgName, meName;
  string imgNameMapGS;

  TCanvas* cMap = new TCanvas("cMap", "Temp", int(360./170.*csize), csize);
  TCanvas* cMapPN = new TCanvas("cMapPN", "Temp", int(360./170.*csize), int(20./90.*360./170.*csize));

  float saveHeigth = gStyle->GetTitleH();
  gStyle->SetTitleH(0.07);
  float saveFontSize = gStyle->GetTitleFontSize();
  gStyle->SetTitleFontSize(14);
  float saveTitleOffset = gStyle->GetTitleX();

  TH2F* obj2f;

  imgNameMapI = "";

  gStyle->SetPaintTextFormat("+g");

  obj2f = 0;
  obj2f = UtilsClient::getHisto<TH2F*>( meIntegrity_ );

  if ( obj2f ) {

    meName = obj2f->GetName();

    replace(meName.begin(), meName.end(), ' ', '_');
    imgNameMapI = meName + ".png";
    imgName = htmlDir + imgNameMapI;

    cMap->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(6, pCol3);
    obj2f->GetXaxis()->SetNdivisions(18, kFALSE);
    obj2f->GetYaxis()->SetNdivisions(2);
    cMap->SetGridx();
    cMap->SetGridy();
    obj2f->SetMinimum(-0.00000001);
    obj2f->SetMaximum(6.0);
    obj2f->GetXaxis()->SetLabelSize(0.03);
    obj2f->GetYaxis()->SetLabelSize(0.03);
    obj2f->Draw("col");
    labelGrid.Draw("text,same");
    cMap->Update();
    cMap->SaveAs(imgName.c_str());

  }

  imgNameMapO = "";

  obj2f = 0;
  obj2f = UtilsClient::getHisto<TH2F*>( meOccupancy_ );

  if ( obj2f ) {

    meName = obj2f->GetName();

    replace(meName.begin(), meName.end(), ' ', '_');
    imgNameMapO = meName + ".png";
    imgName = htmlDir + imgNameMapO;

    cMap->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(10, pCol4);
    obj2f->GetXaxis()->SetNdivisions(18, kFALSE);
    obj2f->GetYaxis()->SetNdivisions(2);
    cMap->SetGridx();
    cMap->SetGridy();
    obj2f->SetMinimum(0.0);
    obj2f->GetXaxis()->SetLabelSize(0.03);
    obj2f->GetYaxis()->SetLabelSize(0.03);
    obj2f->GetZaxis()->SetLabelSize(0.03);
    obj2f->Draw("colz");
    labelGrid.Draw("text,same");
    cMap->Update();
    cMap->SaveAs(imgName.c_str());

  }

  imgNameMapSF = "";

  gStyle->SetPaintTextFormat("+g");

  obj2f = 0;
  obj2f = UtilsClient::getHisto<TH2F*>( meStatusFlags_ );

  if ( obj2f && obj2f->GetEntries() != 0 ) {

    meName = obj2f->GetName();

    replace(meName.begin(), meName.end(), ' ', '_');
    imgNameMapSF = meName + ".png";
    imgName = htmlDir + imgNameMapSF;

    cMap->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(6, pCol3);
    obj2f->GetXaxis()->SetNdivisions(18, kFALSE);
    obj2f->GetYaxis()->SetNdivisions(2);
    cMap->SetGridx();
    cMap->SetGridy();
    obj2f->SetMinimum(-0.00000001);
    obj2f->SetMaximum(6.0);
    obj2f->GetXaxis()->SetLabelSize(0.03);
    obj2f->GetYaxis()->SetLabelSize(0.03);
    obj2f->Draw("col");
    labelGridTT.Draw("text,same");
    cMap->Update();
    cMap->SaveAs(imgName.c_str());

  }

  imgNameMapPO = "";

  obj2f = 0;
  obj2f = UtilsClient::getHisto<TH2F*>( mePedestalOnline_ );

  if ( obj2f && obj2f->GetEntries() != 0 ) {

    meName = obj2f->GetName();

    replace(meName.begin(), meName.end(), ' ', '_');
    imgNameMapPO = meName + ".png";
    imgName = htmlDir + imgNameMapPO;

    cMap->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(6, pCol3);
    obj2f->GetXaxis()->SetNdivisions(18, kFALSE);
    obj2f->GetYaxis()->SetNdivisions(2);
    cMap->SetGridx();
    cMap->SetGridy();
    obj2f->SetMinimum(-0.00000001);
    obj2f->SetMaximum(6.0);
    obj2f->GetXaxis()->SetLabelSize(0.03);
    obj2f->GetYaxis()->SetLabelSize(0.03);
    obj2f->Draw("col");
    labelGrid.Draw("text,same");
    cMap->Update();
    cMap->SaveAs(imgName.c_str());

  }

  imgNameMapLL1 = "";

  obj2f = 0;
  obj2f = UtilsClient::getHisto<TH2F*>( meLaserL1_ );

  if ( obj2f && obj2f->GetEntries() != 0 ) {

    meName = obj2f->GetName();

    replace(meName.begin(), meName.end(), ' ', '_');
    imgNameMapLL1 = meName + ".png";
    imgName = htmlDir + imgNameMapLL1;

    cMap->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(6, pCol3);
    obj2f->GetXaxis()->SetNdivisions(18, kFALSE);
    obj2f->GetYaxis()->SetNdivisions(2);
    cMap->SetGridx();
    cMap->SetGridy();
    obj2f->SetMinimum(-0.00000001);
    obj2f->SetMaximum(6.0);
    obj2f->GetXaxis()->SetLabelSize(0.03);
    obj2f->GetYaxis()->SetLabelSize(0.03);
    obj2f->Draw("col");
    labelGrid.Draw("text,same");
    cMap->Update();
    cMap->SaveAs(imgName.c_str());

  }

  imgNameMapLL1_PN = "";

  obj2f = 0;
  obj2f = UtilsClient::getHisto<TH2F*>( meLaserL1PN_ );

  if ( obj2f && obj2f->GetEntries() != 0 ) {

    meName = obj2f->GetName();

    replace(meName.begin(), meName.end(), ' ', '_');
    imgNameMapLL1_PN = meName + ".png";
    imgName = htmlDir + imgNameMapLL1_PN;

    cMapPN->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(6, pCol3);
    obj2f->GetXaxis()->SetNdivisions(18, kFALSE);
    obj2f->GetYaxis()->SetNdivisions(2, kFALSE);
    cMapPN->SetGridx();
    cMapPN->SetGridy();
    obj2f->SetMinimum(-0.00000001);
    obj2f->SetMaximum(6.0);
    obj2f->GetXaxis()->SetLabelSize(0.07);
    obj2f->GetYaxis()->SetLabelSize(0.07);
    gStyle->SetTitleX(0.15);
    obj2f->Draw("col");
    labelGridPN.Draw("text,same");
    cMapPN->Update();
    cMapPN->SaveAs(imgName.c_str());
    gStyle->SetTitleX(saveTitleOffset);
  }

  imgNameMapP = "";

  obj2f = 0;
  obj2f = UtilsClient::getHisto<TH2F*>( mePedestal_ );

  if ( obj2f && obj2f->GetEntries() != 0 ) {

    meName = obj2f->GetName();

    replace(meName.begin(), meName.end(), ' ', '_');
    imgNameMapP = meName + ".png";
    imgName = htmlDir + imgNameMapP;

    cMap->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(6, pCol3);
    obj2f->GetXaxis()->SetNdivisions(18, kFALSE);
    obj2f->GetYaxis()->SetNdivisions(2);
    cMap->SetGridx();
    cMap->SetGridy();
    obj2f->SetMinimum(-0.00000001);
    obj2f->SetMaximum(6.0);
    obj2f->GetXaxis()->SetLabelSize(0.03);
    obj2f->GetYaxis()->SetLabelSize(0.03);
    obj2f->Draw("col");
    labelGrid.Draw("text,same");
    cMap->Update();
    cMap->SaveAs(imgName.c_str());

  }

  imgNameMapP_PN = "";

  obj2f = 0;
  obj2f = UtilsClient::getHisto<TH2F*>( mePedestalPN_ );

  if ( obj2f && obj2f->GetEntries() != 0 ) {

    meName = obj2f->GetName();

    replace(meName.begin(), meName.end(), ' ', '_');
    imgNameMapP_PN = meName + ".png";
    imgName = htmlDir + imgNameMapP_PN;

    cMapPN->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(6, pCol3);
    obj2f->GetXaxis()->SetNdivisions(18, kFALSE);
    obj2f->GetYaxis()->SetNdivisions(2, kFALSE);
    cMapPN->SetGridx();
    cMapPN->SetGridy();
    obj2f->SetMinimum(-0.00000001);
    obj2f->SetMaximum(6.0);
    obj2f->GetXaxis()->SetLabelSize(0.07);
    obj2f->GetYaxis()->SetLabelSize(0.07);
    gStyle->SetTitleX(0.15);
    obj2f->Draw("col");
    labelGridPN.Draw("text,same");
    cMapPN->Update();
    cMapPN->SaveAs(imgName.c_str());
    gStyle->SetTitleX(saveTitleOffset);
  }


  imgNameMapTP = "";

  obj2f = 0;
  obj2f = UtilsClient::getHisto<TH2F*>( meTestPulse_ );

  if ( obj2f && obj2f->GetEntries() != 0 ) {

    meName = obj2f->GetName();

    replace(meName.begin(), meName.end(), ' ', '_');
    imgNameMapTP = meName + ".png";
    imgName = htmlDir + imgNameMapTP;

    cMap->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(6, pCol3);
    obj2f->GetXaxis()->SetNdivisions(18, kFALSE);
    obj2f->GetYaxis()->SetNdivisions(2);
    cMap->SetGridx();
    cMap->SetGridy();
    obj2f->SetMinimum(-0.00000001);
    obj2f->SetMaximum(6.0);
    obj2f->GetXaxis()->SetLabelSize(0.03);
    obj2f->GetYaxis()->SetLabelSize(0.03);
    obj2f->Draw("col");
    labelGrid.Draw("text,same");
    cMap->Update();
    cMap->SaveAs(imgName.c_str());

  }

  imgNameMapTP_PN = "";

  obj2f = 0;
  obj2f = UtilsClient::getHisto<TH2F*>( meTestPulsePN_ );

  if ( obj2f && obj2f->GetEntries() != 0 ) {

    meName = obj2f->GetName();

    replace(meName.begin(), meName.end(), ' ', '_');
    imgNameMapTP_PN = meName + ".png";
    imgName = htmlDir + imgNameMapTP_PN;

    cMapPN->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(6, pCol3);
    obj2f->GetXaxis()->SetNdivisions(18, kFALSE);
    obj2f->GetYaxis()->SetNdivisions(2);
    cMapPN->SetGridx();
    cMapPN->SetGridy();
    obj2f->SetMinimum(-0.00000001);
    obj2f->SetMaximum(6.0);
    obj2f->GetXaxis()->SetLabelSize(0.07);
    obj2f->GetYaxis()->SetLabelSize(0.07);
    gStyle->SetTitleX(0.15);
    obj2f->Draw("col");
    labelGridPN.Draw("text,same");
    cMapPN->Update();
    cMapPN->SaveAs(imgName.c_str());
    gStyle->SetTitleX(saveTitleOffset);
  }

  imgNameMapC = "";

  obj2f = 0;
  obj2f = UtilsClient::getHisto<TH2F*>( meCosmic_ );

  if ( obj2f && obj2f->GetEntries() != 0 ) {

    meName = obj2f->GetName();

    replace(meName.begin(), meName.end(), ' ', '_');
    imgNameMapC = meName + ".png";
    imgName = htmlDir + imgNameMapC;

    cMap->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(10, pCol4);
    obj2f->GetXaxis()->SetNdivisions(18, kFALSE);
    obj2f->GetYaxis()->SetNdivisions(2);
    cMap->SetGridx();
    cMap->SetGridy();
    obj2f->SetMinimum(0.0);
    obj2f->GetXaxis()->SetLabelSize(0.03);
    obj2f->GetYaxis()->SetLabelSize(0.03);
    obj2f->GetZaxis()->SetLabelSize(0.03);
    obj2f->Draw("colz");
    labelGrid.Draw("text,same");
    cMap->Update();
    cMap->SaveAs(imgName.c_str());

  }

  imgNameMapTM = "";

  obj2f = 0;
  obj2f = UtilsClient::getHisto<TH2F*>( meTiming_ );

  if ( obj2f && obj2f->GetEntries() != 0 ) {

    meName = obj2f->GetName();

    replace(meName.begin(), meName.end(), ' ', '_');
    imgNameMapTM = meName + ".png";
    imgName = htmlDir + imgNameMapTM;

    cMap->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(6, pCol3);
    obj2f->GetXaxis()->SetNdivisions(18, kFALSE);
    obj2f->GetYaxis()->SetNdivisions(2);
    cMap->SetGridx();
    cMap->SetGridy();
    obj2f->SetMinimum(-0.00000001);
    obj2f->SetMaximum(6.0);
    obj2f->GetXaxis()->SetLabelSize(0.03);
    obj2f->GetYaxis()->SetLabelSize(0.03);
    obj2f->Draw("col");
    labelGrid.Draw("text,same");
    cMap->Update();
    cMap->SaveAs(imgName.c_str());

  }

  imgNameMapTTEmulError = "";

  obj2f = 0;
  obj2f = UtilsClient::getHisto<TH2F*>( meTriggerTowerEmulError_ );

  if ( obj2f && obj2f->GetEntries() != 0 ) {

    meName = obj2f->GetName();

    replace(meName.begin(), meName.end(), ' ', '_');
    imgNameMapTTEmulError = meName + ".png";
    imgName = htmlDir + imgNameMapTTEmulError;

    cMap->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(6, pCol3);
    obj2f->GetXaxis()->SetNdivisions(18, kFALSE);
    obj2f->GetYaxis()->SetNdivisions(2);
    cMap->SetGridx();
    cMap->SetGridy();
    obj2f->SetMinimum(-0.00000001);
    obj2f->SetMaximum(6.0);
    obj2f->GetXaxis()->SetLabelSize(0.03);
    obj2f->GetYaxis()->SetLabelSize(0.03);
    obj2f->Draw("col");
    labelGridTT.Draw("text,same");
    cMap->Update();
    cMap->SaveAs(imgName.c_str());

  }

  imgNameMapTTEt = "";

  obj2f = 0;
  obj2f = UtilsClient::getHisto<TH2F*>( meTriggerTowerEt_ );

  if ( obj2f && obj2f->GetEntries() != 0 ) {

    meName = obj2f->GetName();

    replace(meName.begin(), meName.end(), ' ', '_');
    imgNameMapTTEt = meName + ".png";
    imgName = htmlDir + imgNameMapTTEt;

    cMap->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(10, pCol4);
    obj2f->GetXaxis()->SetNdivisions(18, kFALSE);
    obj2f->GetYaxis()->SetNdivisions(2);
    cMap->SetGridx();
    cMap->SetGridy();
    obj2f->SetMinimum(0.0);
    obj2f->GetXaxis()->SetLabelSize(0.03);
    obj2f->GetYaxis()->SetLabelSize(0.03);
    obj2f->GetZaxis()->SetLabelSize(0.03);
    obj2f->Draw("colz");
    labelGridTT.Draw("text,same");
    cMap->Update();
    cMap->SaveAs(imgName.c_str());

  }

  imgNameMapGS = "";

  obj2f = 0;
  obj2f = UtilsClient::getHisto<TH2F*>( meGlobalSummary_ );

  if ( obj2f ) {

    meName = obj2f->GetName();

    replace(meName.begin(), meName.end(), ' ', '_');
    imgNameMapGS = meName + ".png";
    imgName = htmlDir + imgNameMapGS;

    cMap->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(6, pCol3);
    obj2f->GetXaxis()->SetNdivisions(18, kFALSE);
    obj2f->GetYaxis()->SetNdivisions(2);
    cMap->SetGridx();
    cMap->SetGridy();
    obj2f->SetMinimum(-0.00000001);
    obj2f->SetMaximum(6.0);
    obj2f->GetXaxis()->SetLabelSize(0.03);
    obj2f->GetYaxis()->SetLabelSize(0.03);
    obj2f->Draw("col");
    labelGrid.Draw("text,same");
    cMap->Update();
    cMap->SaveAs(imgName.c_str());

  }


  gStyle->SetPaintTextFormat();

  if ( imgNameMapI.size() != 0 ) {
    htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
    htmlFile << "<tr align=\"center\">" << endl;
    htmlFile << "<td><img src=\"" << imgNameMapI << "\" usemap=\"#Integrity\" border=0></td>" << endl;
    htmlFile << "</tr>" << endl;
    htmlFile << "</table>" << endl;
    htmlFile << "<br>" << endl;
  }

  if ( imgNameMapO.size() != 0 ) {
    htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
    htmlFile << "<tr align=\"center\">" << endl;
    htmlFile << "<td><img src=\"" << imgNameMapO << "\" usemap=\"#Occupancy\" border=0></td>" << endl;
    htmlFile << "</tr>" << endl;
    htmlFile << "</table>" << endl;
    htmlFile << "<br>" << endl;
  }

  if ( imgNameMapSF.size() != 0 ) {
    htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
    htmlFile << "<tr align=\"center\">" << endl;
    htmlFile << "<td><img src=\"" << imgNameMapSF << "\" usemap=\"#StatusFlags\" border=0></td>" << endl;
    htmlFile << "</tr>" << endl;
    htmlFile << "</table>" << endl;
    htmlFile << "<br>" << endl;
  }

  if ( imgNameMapPO.size() != 0 ) {
    htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
    htmlFile << "<tr align=\"center\">" << endl;
    htmlFile << "<td><img src=\"" << imgNameMapPO << "\" usemap=\"#PedestalOnline\" border=0></td>" << endl;
    htmlFile << "</tr>" << endl;
    htmlFile << "</table>" << endl;
    htmlFile << "<br>" << endl;
  }

  if ( imgNameMapLL1.size() != 0 ) {
    htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
    htmlFile << "<tr align=\"center\">" << endl;
    htmlFile << "<td><img src=\"" << imgNameMapLL1 << "\" usemap=\"#LaserL1\" border=0></td>" << endl;
    htmlFile << "</tr>" << endl;
    htmlFile << "</table>" << endl;
    htmlFile << "<br>" << endl;
  }

  if ( imgNameMapLL1_PN.size() != 0 ) {
    htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
    htmlFile << "<tr align=\"center\">" << endl;
    htmlFile << "<td><img src=\"" << imgNameMapLL1_PN << "\" border=0></td>" << endl;
    htmlFile << "</tr>" << endl;
    htmlFile << "</table>" << endl;
    htmlFile << "<br>" << endl;
  }

  if ( imgNameMapP.size() != 0 ) {
    htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
    htmlFile << "<tr align=\"center\">" << endl;
    htmlFile << "<td><img src=\"" << imgNameMapP << "\" usemap=\"#Pedestal\" border=0></td>" << endl;
    htmlFile << "</tr>" << endl;
    htmlFile << "</table>" << endl;
    htmlFile << "<br>" << endl;
  }

  if ( imgNameMapP_PN.size() != 0 ) {
    htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
    htmlFile << "<tr align=\"center\">" << endl;
    htmlFile << "<td><img src=\"" << imgNameMapP_PN << "\" border=0></td>" << endl;
    htmlFile << "</tr>" << endl;
    htmlFile << "</table>" << endl;
    htmlFile << "<br>" << endl;
  }

  if ( imgNameMapTP.size() != 0 ) {
    htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
    htmlFile << "<tr align=\"center\">" << endl;
    htmlFile << "<td><img src=\"" << imgNameMapTP << "\" usemap=\"#TestPulse\" border=0></td>" << endl;
    htmlFile << "</tr>" << endl;
    htmlFile << "</table>" << endl;
    htmlFile << "<br>" << endl;
  }

  if ( imgNameMapTP_PN.size() != 0 ) {
    htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
    htmlFile << "<tr align=\"center\">" << endl;
    htmlFile << "<td><img src=\"" << imgNameMapTP_PN << "\" border=0></td>" << endl;
    htmlFile << "</tr>" << endl;
    htmlFile << "</table>" << endl;
    htmlFile << "<br>" << endl;
  }

  if ( imgNameMapC.size() != 0 ) {
    htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
    htmlFile << "<tr align=\"center\">" << endl;
    htmlFile << "<td><img src=\"" << imgNameMapC << "\" usemap=\"#Cosmic\" border=0></td>" << endl;
    htmlFile << "</tr>" << endl;
    htmlFile << "</table>" << endl;
    htmlFile << "<br>" << endl;
  }

  if ( imgNameMapTM.size() != 0 ) {
    htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
    htmlFile << "<tr align=\"center\">" << endl;
    htmlFile << "<td><img src=\"" << imgNameMapTM << "\" usemap=\"#Timing\" border=0></td>" << endl;
    htmlFile << "</tr>" << endl;
    htmlFile << "</table>" << endl;
    htmlFile << "<br>" << endl;
  }

  if ( imgNameMapTTEmulError.size() != 0 ) {
    htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
    htmlFile << "<tr align=\"center\">" << endl;
    htmlFile << "<td><img src=\"" << imgNameMapTTEmulError << "\" usemap=\"#TriggerTower\" border=0></td>" << endl;
    htmlFile << "</tr>" << endl;
    htmlFile << "</table>" << endl;
    htmlFile << "<br>" << endl;
  }

  if ( imgNameMapTTEt.size() != 0 ) {
    htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
    htmlFile << "<tr align=\"center\">" << endl;
    htmlFile << "<td><img src=\"" << imgNameMapTTEt << "\" usemap=\"#TriggerTower\" border=0></td>" << endl;
    htmlFile << "</tr>" << endl;
    htmlFile << "</table>" << endl;
    htmlFile << "<br>" << endl;
  }

  delete cMap;
  delete cMapPN;

  gStyle->SetPaintTextFormat();

  if ( imgNameMapI.size() != 0 ) this->writeMap( htmlFile, "Integrity" );
  if ( imgNameMapO.size() != 0 ) this->writeMap( htmlFile, "Occupancy" );
  if ( imgNameMapSF.size() != 0 ) this->writeMap( htmlFile, "StatusFlags" );
  if ( imgNameMapPO.size() != 0 ) this->writeMap( htmlFile, "PedestalOnline" );
  if ( imgNameMapLL1.size() != 0 ) this->writeMap( htmlFile, "LaserL1" );
  if ( imgNameMapP.size() != 0 ) this->writeMap( htmlFile, "Pedestal" );
  if ( imgNameMapTP.size() != 0 ) this->writeMap( htmlFile, "TestPulse" );

  if ( imgNameMapC.size() != 0 ) this->writeMap( htmlFile, "Cosmic" );
  if ( imgNameMapTM.size() != 0 ) this->writeMap( htmlFile, "Timing" );
  if ( imgNameMapTTEt.size() != 0 ) this->writeMap( htmlFile, "TriggerTower" );
  if ( imgNameMapTTEmulError.size() != 0 ) this->writeMap( htmlFile, "TriggerTower" );

  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  htmlFile.close();

  gStyle->SetTitleH( saveHeigth );
  gStyle->SetTitleFontSize( saveFontSize );

}

void EBSummaryClient::writeMap( ofstream& hf, const char* mapname ) {

  map<string, string> refhtml;
  refhtml["Integrity"] = "EBIntegrityClient.html";
  refhtml["Occupancy"] = "EBIntegrityClient.html";
  refhtml["StatusFlags"] = "EBStatusFlagsClient.html";
  refhtml["PedestalOnline"] = "EBPedestalOnlineClient.html";
  refhtml["LaserL1"] = "EBLaserClient.html";
  refhtml["Pedestal"] = "EBPedestalClient.html";
  refhtml["TestPulse"] = "EBTestPulseClient.html";

  refhtml["Cosmic"] = "EBCosmicClient.html";
  refhtml["Timing"] = "EBTimingClient.html";
  refhtml["TriggerTower"] = "EBTriggerTowerClient.html";

  const int A0 =  85;
  const int A1 = 759;
  const int B0 =  35;
  const int B1 = 334;

  hf << "<map name=\"" << mapname << "\">" << endl;
  for( unsigned int sm=0; sm<superModules_.size(); sm++ ) {
    int i=(superModules_[sm]-1)/18;
    int j=(superModules_[sm]-1)%18;
    int x0 = A0 + (A1-A0)*j/18;
    int x1 = A0 + (A1-A0)*(j+1)/18;
    int y0 = B0 + (B1-B0)*(1-i)/2;
    int y1 = B0 + (B1-B0)*((1-i)+1)/2;
    hf << "<area title=\"" << Numbers::sEB(superModules_[sm])
       << "\" shape=\"rect\" href=\"" << refhtml[mapname]
       << "#" << Numbers::sEB(superModules_[sm])
       << "\" coords=\"" << x0 << ", " << y0 << ", "
                         << x1 << ", " << y1 << "\">"
       << endl;
  }
  hf << "</map>" << endl;

}

