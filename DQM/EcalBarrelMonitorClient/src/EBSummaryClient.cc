/*
 * \file EBSummaryClient.cc
 *
 * $Date: 2009/02/27 19:13:42 $
 * $Revision: 1.175 $
 * \author G. Della Ricca
 *
*/

#include <memory>
#include <iostream>
#include <iomanip>
#include <map>

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

EBSummaryClient::EBSummaryClient(const ParameterSet& ps) {

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
  meIntegrity_            = 0;
  meOccupancy_            = 0;
  meStatusFlags_          = 0;
  mePedestalOnline_       = 0;
  mePedestalOnlineRMSMap_ = 0;
  mePedestalOnlineMean_   = 0;
  mePedestalOnlineRMS_    = 0;
  meLaserL1_              = 0;
  meLaserL1PN_            = 0;
  meLaserL1AmplOverPN_    = 0;
  meLaserL1Timing_        = 0;
  mePedestal_             = 0;
  mePedestalG01_          = 0;
  mePedestalG06_          = 0;
  mePedestalG12_          = 0;
  mePedestalPN_           = 0;
  mePedestalPNG01_        = 0;
  mePedestalPNG16_        = 0;
  meTestPulse_            = 0;
  meTestPulseG01_         = 0;
  meTestPulseG06_         = 0;
  meTestPulseG12_         = 0;
  meTestPulsePN_          = 0;
  meTestPulsePNG01_       = 0;
  meTestPulsePNG16_       = 0;
  meTestPulseAmplG01_     = 0;
  meTestPulseAmplG06_     = 0;
  meTestPulseAmplG12_     = 0;
  meGlobalSummary_        = 0;

  meCosmic_         = 0;
  meTiming_         = 0;
  meTriggerTowerEt_        = 0;
  meTriggerTowerEtSpectrum_ = 0;
  meTriggerTowerEmulError_ = 0;
  meTriggerTowerTiming_ = 0;

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

  // additional histograms from tasks
  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    hpot01_[ism-1] = 0;
    httt01_[ism-1] = 0;
    
  }

}

EBSummaryClient::~EBSummaryClient() {

}

void EBSummaryClient::beginJob(DQMStore* dqmStore) {

  dqmStore_ = dqmStore;

  if ( debug_ ) cout << "EBSummaryClient: beginJob" << endl;

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

  for (int i = 0; i < 36; i++) {
    sprintf(histo, "EcalBarrel_%s", Numbers::sEB(i+1).c_str());
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
  me = dqmStore_->book2D(histo, histo, 72, 0., 72., 34, 0., 34);
  for ( int iettx = 0; iettx < 34; iettx++ ) {
    for ( int ipttx = 0; ipttx < 72; ipttx++ ) {
      me->setBinContent( ipttx+1, iettx+1, -1.0 );
    }
  }
  me->setAxisTitle("jphi", 1);
  me->setAxisTitle("jeta", 2);

}

void EBSummaryClient::beginRun(void) {

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

  if ( mePedestalOnlineRMSMap_ ) dqmStore_->removeElement( mePedestalOnlineRMSMap_->getName() );
  sprintf(histo, "EBPOT pedestal G12 RMS map");
  mePedestalOnlineRMSMap_ = dqmStore_->book2D(histo, histo, 360, 0., 360., 170, -85., 85.);
  mePedestalOnlineRMSMap_->setAxisTitle("jphi", 1);
  mePedestalOnlineRMSMap_->setAxisTitle("jeta", 2);

  if ( mePedestalOnlineMean_ ) dqmStore_->removeElement( mePedestalOnlineMean_->getName() );
  sprintf(histo, "EBPOT pedestal G12 mean");
  mePedestalOnlineMean_ = dqmStore_->book1D(histo, histo, 100, 150., 250.);

  if ( mePedestalOnlineRMS_ ) dqmStore_->removeElement( mePedestalOnlineRMS_->getName() );
  sprintf(histo, "EBPOT pedestal G12 rms");
  mePedestalOnlineRMS_ = dqmStore_->book1D(histo, histo, 100, 0., 10.);

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

  if ( meLaserL1AmplOverPN_ ) dqmStore_->removeElement( meLaserL1AmplOverPN_->getName() );
  sprintf(histo, "EBLT laser L1 amplitude over PN");
  meLaserL1AmplOverPN_ = dqmStore_->book1D(histo, histo, 100, 0., 20.);

  if ( meLaserL1Timing_ ) dqmStore_->removeElement( meLaserL1Timing_->getName() );
  sprintf(histo, "EBLT laser L1 timing");
  meLaserL1Timing_ = dqmStore_->book1D(histo, histo, 100, 0., 10.);

  if( mePedestal_ ) dqmStore_->removeElement( mePedestal_->getName() );
  sprintf(histo, "EBPT pedestal quality summary");
  mePedestal_ = dqmStore_->book2D(histo, histo, 360, 0., 360., 170, -85., 85.);
  mePedestal_->setAxisTitle("jphi", 1);
  mePedestal_->setAxisTitle("jeta", 2);

  if( mePedestalG01_ ) dqmStore_->removeElement( mePedestalG01_->getName() );
  sprintf(histo, "EBPT pedestal quality G01 summary");
  mePedestalG01_ = dqmStore_->book2D(histo, histo, 360, 0., 360., 170, -85., 85.);
  mePedestalG01_->setAxisTitle("jphi", 1);
  mePedestalG01_->setAxisTitle("jeta", 2);

  if( mePedestalG06_ ) dqmStore_->removeElement( mePedestalG06_->getName() );
  sprintf(histo, "EBPT pedestal quality G06 summary");
  mePedestalG06_ = dqmStore_->book2D(histo, histo, 360, 0., 360., 170, -85., 85.);
  mePedestalG06_->setAxisTitle("jphi", 1);
  mePedestalG06_->setAxisTitle("jeta", 2);

  if( mePedestalG12_ ) dqmStore_->removeElement( mePedestalG12_->getName() );
  sprintf(histo, "EBPT pedestal quality G12 summary");
  mePedestalG12_ = dqmStore_->book2D(histo, histo, 360, 0., 360., 170, -85., 85.);
  mePedestalG12_->setAxisTitle("jphi", 1);
  mePedestalG12_->setAxisTitle("jeta", 2);

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

  if( mePedestalPNG01_ ) dqmStore_->removeElement( mePedestalPNG01_->getName() );
  sprintf(histo, "EBPT PN pedestal quality G01 summary");
  mePedestalPNG01_ = dqmStore_->book2D(histo, histo, 90, 0., 90., 20, -10, 10.);
  mePedestalPNG01_->setAxisTitle("jphi", 1);
  mePedestalPNG01_->setAxisTitle("jeta", 2);

  if( mePedestalPNG16_ ) dqmStore_->removeElement( mePedestalPNG16_->getName() );
  sprintf(histo, "EBPT PN pedestal quality G16 summary");
  mePedestalPNG16_ = dqmStore_->book2D(histo, histo, 90, 0., 90., 20, -10, 10.);
  mePedestalPNG16_->setAxisTitle("jphi", 1);
  mePedestalPNG16_->setAxisTitle("jeta", 2);

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

  if( meTestPulseG01_ ) dqmStore_->removeElement( meTestPulseG01_->getName() );
  sprintf(histo, "EBTPT test pulse quality G01 summary");
  meTestPulseG01_ = dqmStore_->book2D(histo, histo, 360, 0., 360., 170, -85., 85.);
  meTestPulseG01_->setAxisTitle("jphi", 1);
  meTestPulseG01_->setAxisTitle("jeta", 2);

  if( meTestPulseG06_ ) dqmStore_->removeElement( meTestPulseG06_->getName() );
  sprintf(histo, "EBTPT test pulse quality G06 summary");
  meTestPulseG06_ = dqmStore_->book2D(histo, histo, 360, 0., 360., 170, -85., 85.);
  meTestPulseG06_->setAxisTitle("jphi", 1);
  meTestPulseG06_->setAxisTitle("jeta", 2);

  if( meTestPulseG12_ ) dqmStore_->removeElement( meTestPulseG12_->getName() );
  sprintf(histo, "EBTPT test pulse quality G12 summary");
  meTestPulseG12_ = dqmStore_->book2D(histo, histo, 360, 0., 360., 170, -85., 85.);
  meTestPulseG12_->setAxisTitle("jphi", 1);
  meTestPulseG12_->setAxisTitle("jeta", 2);

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

  if( meTestPulsePNG01_ ) dqmStore_->removeElement( meTestPulsePNG01_->getName() );
  sprintf(histo, "EBTPT PN test pulse quality G01 summary");
  meTestPulsePNG01_ = dqmStore_->book2D(histo, histo, 90, 0., 90., 20, -10., 10.);
  meTestPulsePNG01_->setAxisTitle("jphi", 1);
  meTestPulsePNG01_->setAxisTitle("jeta", 2);

  if( meTestPulsePNG16_ ) dqmStore_->removeElement( meTestPulsePNG16_->getName() );
  sprintf(histo, "EBTPT PN test pulse quality G16 summary");
  meTestPulsePNG16_ = dqmStore_->book2D(histo, histo, 90, 0., 90., 20, -10., 10.);
  meTestPulsePNG16_->setAxisTitle("jphi", 1);
  meTestPulsePNG16_->setAxisTitle("jeta", 2);

  if( meTestPulsePNErr_ ) dqmStore_->removeElement( meTestPulsePNErr_->getName() );
  sprintf(histo, "EBTPT PN test pulse quality errors summary");
  meTestPulsePNErr_ = dqmStore_->book1D(histo, histo, 36, 1, 37);
  for (int i = 0; i < 36; i++) {
    meTestPulsePNErr_->setBinLabel(i+1, Numbers::sEB(i+1).c_str(), 1);
  }

  if( meTestPulseAmplG01_ ) dqmStore_->removeElement( meTestPulseAmplG01_->getName() );
  sprintf(histo, "EBTPT test pulse amplitude G01 summary");
  meTestPulseAmplG01_ = dqmStore_->book1D(histo, histo, 100, 2000, 4000);

  if( meTestPulseAmplG06_ ) dqmStore_->removeElement( meTestPulseAmplG06_->getName() );
  sprintf(histo, "EBTPT test pulse amplitude G06 summary");
  meTestPulseAmplG06_ = dqmStore_->book1D(histo, histo, 100, 2000, 4000);

  if( meTestPulseAmplG12_ ) dqmStore_->removeElement( meTestPulseAmplG12_->getName() );
  sprintf(histo, "EBTPT test pulse amplitude G12 summary");
  meTestPulseAmplG12_ = dqmStore_->book1D(histo, histo, 100, 2000, 4000);

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

  if( meTriggerTowerEtSpectrum_ ) dqmStore_->removeElement( meTriggerTowerEtSpectrum_->getName() );
  sprintf(histo, "EBTTT Et trigger tower spectrum");
  meTriggerTowerEtSpectrum_ = dqmStore_->book1D(histo, histo, 256, 0., 256.);
  meTriggerTowerEtSpectrum_->setAxisTitle("transverse energy (GeV)", 1);

  if( meTriggerTowerEmulError_ ) dqmStore_->removeElement( meTriggerTowerEmulError_->getName() );
  sprintf(histo, "EBTTT emulator error quality summary");
  meTriggerTowerEmulError_ = dqmStore_->book2D(histo, histo, 72, 0., 72., 34, -17., 17.);
  meTriggerTowerEmulError_->setAxisTitle("jphi'", 1);
  meTriggerTowerEmulError_->setAxisTitle("jeta'", 2);

  if( meTriggerTowerTiming_ ) dqmStore_->removeElement( meTriggerTowerTiming_->getName() );
  sprintf(histo, "EBTTT Trigger Primitives Timing summary");
  meTriggerTowerTiming_ = dqmStore_->book2D(histo, histo, 72, 0., 72., 34, -17., 17.);
  meTriggerTowerTiming_->setAxisTitle("jphi'", 1);
  meTriggerTowerTiming_->setAxisTitle("jeta'", 2);

  if( meGlobalSummary_ ) dqmStore_->removeElement( meGlobalSummary_->getName() );
  sprintf(histo, "EB global summary");
  meGlobalSummary_ = dqmStore_->book2D(histo, histo, 360, 0., 360., 170, -85., 85.);
  meGlobalSummary_->setAxisTitle("jphi", 1);
  meGlobalSummary_->setAxisTitle("jeta", 2);

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

  if ( mePedestalOnlineMean_ ) dqmStore_->removeElement( mePedestalOnlineMean_->getName() );
  mePedestalOnlineMean_ = 0;

  if ( mePedestalOnlineRMS_ ) dqmStore_->removeElement( mePedestalOnlineRMS_->getName() );
  mePedestalOnlineRMS_ = 0;

  if ( mePedestalOnlineRMSMap_ ) dqmStore_->removeElement( mePedestalOnlineRMSMap_->getName() );
  mePedestalOnlineRMSMap_ = 0;

  if ( meLaserL1_ ) dqmStore_->removeElement( meLaserL1_->getName() );
  meLaserL1_ = 0;

  if ( meLaserL1Err_ ) dqmStore_->removeElement( meLaserL1Err_->getName() );
  meLaserL1Err_ = 0;

  if ( meLaserL1AmplOverPN_ ) dqmStore_->removeElement( meLaserL1AmplOverPN_->getName() );
  meLaserL1AmplOverPN_ = 0;

  if ( meLaserL1Timing_ ) dqmStore_->removeElement( meLaserL1Timing_->getName() );
  meLaserL1Timing_ = 0;

  if ( meLaserL1PN_ ) dqmStore_->removeElement( meLaserL1PN_->getName() );
  meLaserL1PN_ = 0;

  if ( meLaserL1PNErr_ ) dqmStore_->removeElement( meLaserL1PNErr_->getName() );
  meLaserL1PNErr_ = 0;

  if ( mePedestal_ ) dqmStore_->removeElement( mePedestal_->getName() );
  mePedestal_ = 0;

  if ( mePedestalG01_ ) dqmStore_->removeElement( mePedestalG01_->getName() );
  mePedestalG01_ = 0;

  if ( mePedestalG06_ ) dqmStore_->removeElement( mePedestalG06_->getName() );
  mePedestalG06_ = 0;

  if ( mePedestalG12_ ) dqmStore_->removeElement( mePedestalG12_->getName() );
  mePedestalG12_ = 0;

  if ( mePedestalErr_ ) dqmStore_->removeElement( mePedestalErr_->getName() );
  mePedestalErr_ = 0;

  if ( mePedestalPN_ ) dqmStore_->removeElement( mePedestalPN_->getName() );
  mePedestalPN_ = 0;

  if ( mePedestalPNErr_ ) dqmStore_->removeElement( mePedestalPNErr_->getName() );
  mePedestalPNErr_ = 0;

  if ( meTestPulse_ ) dqmStore_->removeElement( meTestPulse_->getName() );
  meTestPulse_ = 0;

  if ( meTestPulseG01_ ) dqmStore_->removeElement( meTestPulseG01_->getName() );
  meTestPulseG01_ = 0;

  if ( meTestPulseG06_ ) dqmStore_->removeElement( meTestPulseG06_->getName() );
  meTestPulseG06_ = 0;

  if ( meTestPulseG12_ ) dqmStore_->removeElement( meTestPulseG12_->getName() );
  meTestPulseG12_ = 0;

  if ( meTestPulseG01_ ) dqmStore_->removeElement( meTestPulseG01_->getName() );
  meTestPulseG01_ = 0;

  if ( meTestPulseErr_ ) dqmStore_->removeElement( meTestPulseErr_->getName() );
  meTestPulseErr_ = 0;

  if ( meTestPulsePN_ ) dqmStore_->removeElement( meTestPulsePN_->getName() );
  meTestPulsePN_ = 0;

  if ( meTestPulsePNErr_ ) dqmStore_->removeElement( meTestPulsePNErr_->getName() );
  meTestPulsePNErr_ = 0;

  if ( meTestPulseAmplG01_ ) dqmStore_->removeElement( meTestPulseAmplG01_->getName() );
  meTestPulseAmplG01_ = 0;

  if ( meTestPulseAmplG06_ ) dqmStore_->removeElement( meTestPulseAmplG06_->getName() );
  meTestPulseAmplG06_ = 0;

  if ( meTestPulseAmplG12_ ) dqmStore_->removeElement( meTestPulseAmplG12_->getName() );
  meTestPulseAmplG12_ = 0;

  if ( meCosmic_ ) dqmStore_->removeElement( meCosmic_->getName() );
  meCosmic_ = 0;

  if ( meTiming_ ) dqmStore_->removeElement( meTiming_->getName() );
  meTiming_ = 0;

  if ( meTriggerTowerEt_ ) dqmStore_->removeElement( meTriggerTowerEt_->getName() );
  meTriggerTowerEt_ = 0;

  if ( meTriggerTowerEtSpectrum_ ) dqmStore_->removeElement( meTriggerTowerEtSpectrum_->getName() );
  meTriggerTowerEtSpectrum_ = 0;

  if ( meTriggerTowerEmulError_ ) dqmStore_->removeElement( meTriggerTowerEmulError_->getName() );
  meTriggerTowerEmulError_ = 0;

  if ( meTriggerTowerTiming_ ) dqmStore_->removeElement( meTriggerTowerTiming_->getName() );
  meTriggerTowerTiming_ = 0;

  if ( meGlobalSummary_ ) dqmStore_->removeElement( meGlobalSummary_->getName() );
  meGlobalSummary_ = 0;

}

bool EBSummaryClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov, bool& status, bool flag) {

  status = true;

  if ( ! flag ) return false;

  return true;

}

void EBSummaryClient::analyze(void) {

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( debug_ ) cout << "EBSummaryClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;
  }

  for ( int iex = 1; iex <= 170; iex++ ) {
    for ( int ipx = 1; ipx <= 360; ipx++ ) {

      meIntegrity_->setBinContent( ipx, iex, 6. );
      meOccupancy_->setBinContent( ipx, iex, 0. );
      meStatusFlags_->setBinContent( ipx, iex, 6. );
      mePedestalOnline_->setBinContent( ipx, iex, 6. );
      mePedestalOnlineRMSMap_->setBinContent( ipx, iex, -1.);

      meLaserL1_->setBinContent( ipx, iex, 6. );
      mePedestal_->setBinContent( ipx, iex, 6. );
      mePedestalG01_->setBinContent( ipx, iex, 6. );
      mePedestalG06_->setBinContent( ipx, iex, 6. );
      mePedestalG12_->setBinContent( ipx, iex, 6. );
      meTestPulse_->setBinContent( ipx, iex, 6. );
      meTestPulseG01_->setBinContent( ipx, iex, 6. );
      meTestPulseG06_->setBinContent( ipx, iex, 6. );
      meTestPulseG12_->setBinContent( ipx, iex, 6. );

      meCosmic_->setBinContent( ipx, iex, 0. );
      meTiming_->setBinContent( ipx, iex, 6. );

      meGlobalSummary_->setBinContent( ipx, iex, 6. );

    }
  }

  for ( int iex = 1; iex <= 20; iex++ ) {
    for ( int ipx = 1; ipx <= 90; ipx++ ) {

      meLaserL1PN_->setBinContent( ipx, iex, 6. );
      mePedestalPN_->setBinContent( ipx, iex, 6. );
      mePedestalPNG01_->setBinContent( ipx, iex, 6. );
      mePedestalPNG16_->setBinContent( ipx, iex, 6. );
      meTestPulsePN_->setBinContent( ipx, iex, 6. );
      meTestPulsePNG01_->setBinContent( ipx, iex, 6. );
      meTestPulsePNG16_->setBinContent( ipx, iex, 6. );

    }
  }

  for ( int iex = 1; iex <= 34; iex++ ) {
    for ( int ipx = 1; ipx <= 72; ipx++ ) {
      meTriggerTowerEt_->setBinContent( ipx, iex, 0. );
      meTriggerTowerEmulError_->setBinContent( ipx, iex, 6. );
      meTriggerTowerTiming_->setBinContent( ipx, iex, -1. );
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
  mePedestalOnlineMean_->Reset();
  mePedestalOnlineRMS_->Reset();
  mePedestalOnlineRMSMap_->Reset();

  meLaserL1_->setEntries( 0 );
  meLaserL1Err_->Reset();
  meLaserL1AmplOverPN_->Reset();
  meLaserL1Timing_->Reset();
  meLaserL1PN_->setEntries( 0 );
  meLaserL1PNErr_->Reset();
  mePedestal_->setEntries( 0 );
  mePedestalG01_->setEntries( 0 );
  mePedestalG06_->setEntries( 0 );
  mePedestalG12_->setEntries( 0 );
  mePedestalErr_->Reset();
  mePedestalPN_->setEntries( 0 );
  mePedestalPNG01_->setEntries( 0 );
  mePedestalPNG16_->setEntries( 0 );
  mePedestalPNErr_->Reset();
  meTestPulse_->setEntries( 0 );
  meTestPulseG01_->setEntries( 0 );
  meTestPulseG06_->setEntries( 0 );
  meTestPulseG12_->setEntries( 0 );
  meTestPulseErr_->Reset();
  meTestPulsePN_->setEntries( 0 );
  meTestPulsePNG01_->setEntries( 0 );
  meTestPulsePNG16_->setEntries( 0 );
  meTestPulsePNErr_->Reset();
  meTestPulseAmplG01_->Reset();
  meTestPulseAmplG06_->Reset();
  meTestPulseAmplG12_->Reset();

  meCosmic_->setEntries( 0 );
  meTiming_->setEntries( 0 );
  meTriggerTowerEt_->setEntries( 0 );
  meTriggerTowerEtSpectrum_->Reset();
  meTriggerTowerEmulError_->setEntries( 0 );
  meTriggerTowerTiming_->setEntries( 0 );

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
    //    MonitorElement *me_f[6], *me_fg[2];
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

      sprintf(histo, (prefixME_ + "/EBPedestalOnlineTask/Gain12/EBPOT pedestal %s G12").c_str(), Numbers::sEB(ism).c_str());
      me = dqmStore_->get(histo);
      hpot01_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, hpot01_[ism-1] );

      sprintf(histo, (prefixME_ + "/EBTriggerTowerTask/EBTTT Et map Real Digis %s").c_str(), Numbers::sEB(ism).c_str());
      me = dqmStore_->get(histo);
      httt01_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, httt01_[ism-1] );

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
            
            mePedestalOnlineRMS_->Fill( rms01 );
            
            mePedestalOnlineMean_->Fill( mean01 );
            
          }
          
          if ( eblc ) {
            
            me = eblc->meg01_[ism-1];
            
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

            if ( me_01 && me_02 && me_03 ) {
              float xval=2;
              float val_01=me_01->getBinContent(ie,ip);
              float val_02=me_02->getBinContent(ie,ip);
              float val_03=me_03->getBinContent(ie,ip);

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

              int iex;
              int ipx;

              if ( ism <= 18 ) {
                iex = 1+(85-ie);
                ipx = ip+20*(ism-1);
              } else {
                iex = 85+ie;
                ipx = 1+(20-ip)+20*(ism-19);
              }

              if ( me_01->getEntries() != 0 ) mePedestalG01_->setBinContent( ipx, iex, val_01 );
              if ( me_02->getEntries() != 0 ) mePedestalG06_->setBinContent( ipx, iex, val_02 );
              if ( me_03->getEntries() != 0 ) mePedestalG12_->setBinContent( ipx, iex, val_03 );

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

            if ( me_01 && me_02 && me_03 ) {
              float xval=2;
              float val_01=me_01->getBinContent(ie,ip);
              float val_02=me_02->getBinContent(ie,ip);
              float val_03=me_03->getBinContent(ie,ip);

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

              int iex;
              int ipx;

              if ( ism <= 18 ) {
                iex = 1+(85-ie);
                ipx = ip+20*(ism-1);
              } else {
                iex = 85+ie;
                ipx = 1+(20-ip)+20*(ism-19);
              }

              if ( me_01->getEntries() != 0 ) meTestPulseG01_->setBinContent( ipx, iex, val_01 );
              if ( me_02->getEntries() != 0 ) meTestPulseG06_->setBinContent( ipx, iex, val_02 );
              if ( me_03->getEntries() != 0 ) meTestPulseG12_->setBinContent( ipx, iex, val_03 );

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

            me = ebsfc->meh01_[ism-1];

            if ( me ) {

              float xval = 6;

              if ( me->getBinContent( ie, ip ) == 6 ) xval = 2;
              if ( me->getBinContent( ie, ip ) == 0 ) xval = 1;
              if ( me->getBinContent( ie, ip ) > 0 ) xval = 0;

              if ( me->getEntries() != 0 ) {
                meStatusFlags_->setBinContent( ipx, iex, xval );
                if ( xval == 0 ) meStatusFlagsErr_->Fill( ism );
              }

            }

          }

          if ( ebtttc ) {

            float num01, mean01, rms01;
            bool update01 = UtilsClient::getBinStatistics(httt01_[ism-1], ie, ip, num01, mean01, rms01);
            
            if ( update01 ) { 
              if ( meTriggerTowerEt_ ) meTriggerTowerEt_->setBinContent( ipx, iex, mean01 );
              if ( meTriggerTowerEtSpectrum_ ) meTriggerTowerEtSpectrum_->Fill( mean01 );
            }
              
            me = ebtttc->me_o01_[ism-1];

            if ( me ) {

              float xval = me->getBinContent( ie, ip );

              meTriggerTowerTiming_->setBinContent( ipx, iex, xval );

            }
            
            float xval = 6;
            if( mean01 <= 0 ) xval = 2;
            else {

              h2 = ebtttc->l01_[ism-1];

              if ( h2 ) {

                float emulErrorVal = h2->GetBinContent( ie, ip );
                if( emulErrorVal!=0 ) xval = 0;

              }

              // do not propagate the flag bits to the summary for now
//               for ( int iflag=0; iflag<6; iflag++ ) {

//                 me_f[iflag] = ebtttc->me_m01_[ism-1][iflag];

//                 if ( me_f[iflag] ) {

//                   float emulFlagErrorVal = me_f[iflag]->getBinContent( ie, ip );
//                   if ( emulFlagErrorVal!=0 ) xval = 0;

//                 }

//               }

//               for ( int ifg=0; ifg<2; ifg++) {

//                 me_fg[ifg] = ebtttc->me_n01_[ism-1][ifg];
//                 if ( me_fg[ifg] ) {

//                   float emulFineGrainVetoErrorVal = me_fg[ifg]->getBinContent( ie, ip );
//                   if ( emulFineGrainVetoErrorVal!=0 ) xval = 0;

//                 }

//               }

              if ( xval!=0 ) xval = 1;

            }

            meTriggerTowerEmulError_->setBinContent( ipx, iex, xval );

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
              (val_04>=3&&val_04<=5) ? maskedVal.push_back(val_04) : unmaskedVal.push_back(val_04);
              (val_05>=3&&val_05<=5) ? maskedVal.push_back(val_05) : unmaskedVal.push_back(val_05);

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

              if ( me_04->getEntries() != 0 ) mePedestalPNG01_->setBinContent( ipx, iex, val_04 );
              if ( me_05->getEntries() != 0 ) mePedestalPNG16_->setBinContent( ipx, iex, val_05 );

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
              (val_04>=3&&val_04<=5) ? maskedVal.push_back(val_04) : unmaskedVal.push_back(val_04);
              (val_05>=3&&val_05<=5) ? maskedVal.push_back(val_05) : unmaskedVal.push_back(val_05);

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

              if ( me_04->getEntries() != 0 ) meTestPulsePNG01_->setBinContent( ipx, iex, val_04 );
              if ( me_05->getEntries() != 0 ) meTestPulsePNG16_->setBinContent( ipx, iex, val_05 );

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

      for(int chan=0; chan<1700; chan++) {

        int ie = (chan)/20 + 1;
        int ip = (chan)%20 + 1;

        // laser 1D summaries
        if ( eblc ) {
          
          MonitorElement *meg = eblc->meg01_[ism-1];

          float xval = 2;
          if ( meg ) xval = meg->getBinContent( ie, ip );

          // exclude channels without laser data (yellow in the quality map)
          if( xval != 2 && xval != 5 ) { 
          
            int RtHalf = 0;
            if( ie > 5 && ip < 11 ) RtHalf = 1;
          
            //! Ampl / PN
            // L1A (L-shaped)
            me = eblc->meaopn01_[ism-1];
            
            if( me && RtHalf == 0 ) {
              meLaserL1AmplOverPN_->Fill( me->getBinContent( chan+1 ) );
            }
            
            // L1B (rectangular)
            me = eblc->meaopn05_[ism-1];
            
            if ( me && RtHalf == 1 ) {
              meLaserL1AmplOverPN_->Fill( me->getBinContent( chan+1 ) );
            }
            
            //! timing
            // L1A (L-shaped)
            me = eblc->met01_[ism-1];
            
            if( me && RtHalf == 0 ) {
              meLaserL1Timing_->Fill( me->getBinContent( chan+1 ) );
            }
            
            // L1B (rectangular)
            me = eblc->met05_[ism-1];
            
            if ( me && RtHalf == 1 ) {
              meLaserL1Timing_->Fill( me->getBinContent( chan+1 ) );
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
                
                meTestPulseAmplG01_->Fill( me->getBinContent( chan+1 ) );

              }

            }
            
          }

          if ( meg02 ) {
            
            float xval02 = meg02->getBinContent(ie,ip);

            if ( xval02 != 2 && xval02 != 5 ) {

              me = ebtpc->mea02_[ism-1];
              
              if ( me ) {
                
                meTestPulseAmplG06_->Fill( me->getBinContent( chan+1 ) );

              }

            }
            
          }

          if ( meg03 ) {
            
            float xval03 = meg03->getBinContent(ie,ip);

            if ( xval03 != 2 && xval03 != 5 ) {

              me = ebtpc->mea03_[ism-1];
              
              if ( me ) {
                
                meTestPulseAmplG12_->Fill( me->getBinContent( chan+1 ) );

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

      if(meIntegrity_ && mePedestalOnline_ && meLaserL1_ && meTiming_ && meStatusFlags_ && meTriggerTowerEmulError_) {

        float xval = 6;
        float val_in = meIntegrity_->getBinContent(ipx,iex);
        float val_po = mePedestalOnline_->getBinContent(ipx,iex);
        float val_ls = meLaserL1_->getBinContent(ipx,iex);
        float val_tm = meTiming_->getBinContent(ipx,iex);
        float val_sf = meStatusFlags_->getBinContent((ipx-1)/5+1,(iex-1)/5+1);
	// float val_ee = meTriggerTowerEmulError_->getBinContent((ipx-1)/5+1,(iex-1)/5+1); // removed from the global summary temporarily
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

        int ism = (ipx-1)/20 + 1 ;
        if ( iex>85 ) ism+=18;

        vector<int>::iterator iter = find(superModules_.begin(), superModules_.end(), ism);
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

        meGlobalSummary_->setBinContent( ipx, iex, xval );

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

  MonitorElement* me;

  float reportSummary = -1.0;
  if ( nValidChannels != 0 )
    reportSummary = 1.0 - float(nGlobalErrors)/float(nValidChannels);
  me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummary");
  if ( me ) me->Fill(reportSummary);

  char histo[200];

  for (int i = 0; i < 36; i++) {
    float reportSummaryEB = -1.0;
    if ( nValidChannelsEB[i] != 0 )
      reportSummaryEB = 1.0 - float(nGlobalErrorsEB[i])/float(nValidChannelsEB[i]);
    sprintf(histo, "EcalBarrel_%s", Numbers::sEB(i+1).c_str());
    me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummaryContents/" + histo);
    if ( me ) me->Fill(reportSummaryEB);
  }

  me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummaryMap");
  if ( me ) {

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

      }
    }

  }

}

void EBSummaryClient::softReset(bool flag) {

}

