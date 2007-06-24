/*
 * \file EEPedestalClient.cc
 *
 * $Date: 2007/06/12 18:18:06 $
 * $Revision: 1.14 $
 * \author G. Della Ricca
 * \author F. Cossutti
 *
*/

#include <memory>
#include <iostream>
#include <fstream>
#include <iomanip>

#include "TStyle.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/UI/interface/MonitorUIRoot.h"
#include "DQMServices/Core/interface/QTestStatus.h"
#include "DQMServices/QualityTests/interface/QCriterionRoot.h"

#include "OnlineDB/EcalCondDB/interface/RunTag.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"
#include "OnlineDB/EcalCondDB/interface/MonPedestalsDat.h"
#include "OnlineDB/EcalCondDB/interface/MonPNPedDat.h"
#include "OnlineDB/EcalCondDB/interface/RunCrystalErrorsDat.h"
#include "OnlineDB/EcalCondDB/interface/RunPNErrorsDat.h"

#include "CondTools/Ecal/interface/EcalErrorDictionary.h"

#include "DQM/EcalCommon/interface/EcalErrorMask.h"
#include <DQM/EcalCommon/interface/UtilsClient.h>
#include <DQM/EcalCommon/interface/LogicID.h>
#include <DQM/EcalCommon/interface/Numbers.h>

#include <DQM/EcalEndcapMonitorClient/interface/EEPedestalClient.h>

using namespace cms;
using namespace edm;
using namespace std;

EEPedestalClient::EEPedestalClient(const ParameterSet& ps){

  // collateSources switch
  collateSources_ = ps.getUntrackedParameter<bool>("collateSources", false);

  // cloneME switch
  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);

  // enableQT switch
  enableQT_ = ps.getUntrackedParameter<bool>("enableQT", true);

  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

  // MonitorDaemon switch
  enableMonitorDaemon_ = ps.getUntrackedParameter<bool>("enableMonitorDaemon", true);

  // prefix to ME paths
  prefixME_ = ps.getUntrackedParameter<string>("prefixME", "");

  // vector of selected Super Modules (Defaults to all 18).
  superModules_.reserve(18);
  for ( unsigned int i = 1; i < 19; i++ ) superModules_.push_back(i);
  superModules_ = ps.getUntrackedParameter<vector<int> >("superModules", superModules_);

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

    mes01_[ism-1] = 0;
    mes02_[ism-1] = 0;
    mes03_[ism-1] = 0;

    met01_[ism-1] = 0;
    met02_[ism-1] = 0;
    met03_[ism-1] = 0;

    qth01_[ism-1] = 0;
    qth02_[ism-1] = 0;
    qth03_[ism-1] = 0;

    qth04_[ism-1] = 0;
    qth05_[ism-1] = 0;

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
  
  RMSThresholdPn_[0] = 1.0;
  RMSThresholdPn_[1] = 3.0;
  
}

EEPedestalClient::~EEPedestalClient(){

}

void EEPedestalClient::beginJob(MonitorUserInterface* mui){

  mui_ = mui;

  if ( verbose_ ) cout << "EEPedestalClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

  if ( enableQT_ ) {

    Char_t qtname[200];

    for ( unsigned int i=0; i<superModules_.size(); i++ ) {

      int ism = superModules_[i];

      sprintf(qtname, "EEPT quality %s G01", Numbers::sEE(ism).c_str());
      qth01_[ism-1] = dynamic_cast<MEContentsProf2DWithinRangeROOT*> (mui_->createQTest(ContentsProf2DWithinRangeROOT::getAlgoName(), qtname));

      sprintf(qtname, "EEPT quality %s G06", Numbers::sEE(ism).c_str());
      qth02_[ism-1] = dynamic_cast<MEContentsProf2DWithinRangeROOT*> (mui_->createQTest(ContentsProf2DWithinRangeROOT::getAlgoName(), qtname));

      sprintf(qtname, "EEPT quality %s G12", Numbers::sEE(ism).c_str());
      qth03_[ism-1] = dynamic_cast<MEContentsProf2DWithinRangeROOT*> (mui_->createQTest(ContentsProf2DWithinRangeROOT::getAlgoName(), qtname));

      sprintf(qtname, "EEPT pedestal quality PNs %s G01", Numbers::sEE(ism).c_str());
      qth04_[ism-1] = dynamic_cast<MEContentsProf2DWithinRangeROOT*> (mui_->createQTest(ContentsProf2DWithinRangeROOT::getAlgoName(), qtname));

      sprintf(qtname, "EEPT pedestal quality PNs %s G16", Numbers::sEE(ism).c_str());
      qth05_[ism-1] = dynamic_cast<MEContentsProf2DWithinRangeROOT*> (mui_->createQTest(ContentsProf2DWithinRangeROOT::getAlgoName(), qtname));

      qth01_[ism-1]->setMeanRange(expectedMean_[0] - discrepancyMean_[0], expectedMean_[0] + discrepancyMean_[0]);
      qth02_[ism-1]->setMeanRange(expectedMean_[1] - discrepancyMean_[1], expectedMean_[1] + discrepancyMean_[1]);
      qth03_[ism-1]->setMeanRange(expectedMean_[2] - discrepancyMean_[2], expectedMean_[2] + discrepancyMean_[2]);
      
      qth04_[ism-1]->setMeanRange(expectedMeanPn_[0] - discrepancyMeanPn_[0], expectedMeanPn_[0] + discrepancyMeanPn_[0]);
      qth05_[ism-1]->setMeanRange(expectedMeanPn_[1] - discrepancyMeanPn_[1], expectedMeanPn_[1] + discrepancyMeanPn_[1]);
      
      qth01_[ism-1]->setRMSRange(0.0, RMSThreshold_[0]);
      qth02_[ism-1]->setRMSRange(0.0, RMSThreshold_[1]);
      qth03_[ism-1]->setRMSRange(0.0, RMSThreshold_[2]);
      
      qth04_[ism-1]->setRMSRange(0.0, RMSThresholdPn_[0]);
      qth05_[ism-1]->setRMSRange(0.0, RMSThresholdPn_[1]);
      
      qth01_[ism-1]->setMinimumEntries(10*1700);
      qth02_[ism-1]->setMinimumEntries(10*1700);
      qth03_[ism-1]->setMinimumEntries(10*1700);

      qth04_[ism-1]->setMinimumEntries(10*10);
      qth05_[ism-1]->setMinimumEntries(10*10);

      qth01_[ism-1]->setErrorProb(1.00);
      qth02_[ism-1]->setErrorProb(1.00);
      qth03_[ism-1]->setErrorProb(1.00);

      qth04_[ism-1]->setErrorProb(1.00);
      qth05_[ism-1]->setErrorProb(1.00);

    }

  }

}

void EEPedestalClient::beginRun(void){

  if ( verbose_ ) cout << "EEPedestalClient: beginRun" << endl;

  jevt_ = 0;

  this->setup();

  this->subscribe();

}

void EEPedestalClient::endJob(void) {

  if ( verbose_ ) cout << "EEPedestalClient: endJob, ievt = " << ievt_ << endl;

  this->unsubscribe();

  this->cleanup();

}

void EEPedestalClient::endRun(void) {

  if ( verbose_ ) cout << "EEPedestalClient: endRun, jevt = " << jevt_ << endl;

  this->unsubscribe();

  this->cleanup();

}

void EEPedestalClient::setup(void) {

  Char_t histo[200];

  mui_->setCurrentFolder( "EcalEndcap/EEPedestalClient" );
  DaqMonitorBEInterface* dbe = mui_->getBEInterface();

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( meg01_[ism-1] ) dbe->removeElement( meg01_[ism-1]->getName() );
    sprintf(histo, "EEPT pedestal quality G01 %s", Numbers::sEE(ism).c_str());
    meg01_[ism-1] = dbe->book2D(histo, histo, 85, 0., 85., 20, 0., 20.);
    if ( meg02_[ism-1] ) dbe->removeElement( meg02_[ism-1]->getName() );
    sprintf(histo, "EEPT pedestal quality G06 %s", Numbers::sEE(ism).c_str());
    meg02_[ism-1] = dbe->book2D(histo, histo, 85, 0., 85., 20, 0., 20.);
    if ( meg03_[ism-1] ) dbe->removeElement( meg03_[ism-1]->getName() );
    sprintf(histo, "EEPT pedestal quality G12 %s", Numbers::sEE(ism).c_str());
    meg03_[ism-1] = dbe->book2D(histo, histo, 85, 0., 85., 20, 0., 20.);

    if ( meg04_[ism-1] ) dbe->removeElement( meg04_[ism-1]->getName() );
    sprintf(histo, "EEPT pedestal quality PNs G01 %s", Numbers::sEE(ism).c_str());
    meg04_[ism-1] = dbe->book2D(histo, histo, 10, 0., 10., 1, 0., 5.);
    if ( meg05_[ism-1] ) dbe->removeElement( meg05_[ism-1]->getName() );
    sprintf(histo, "EEPT pedestal quality PNs G16 %s", Numbers::sEE(ism).c_str());
    meg05_[ism-1] = dbe->book2D(histo, histo, 10, 0., 10., 1, 0., 5.);

    if ( mep01_[ism-1] ) dbe->removeElement( mep01_[ism-1]->getName() );
    sprintf(histo, "EEPT pedestal mean G01 %s", Numbers::sEE(ism).c_str());
    mep01_[ism-1] = dbe->book1D(histo, histo, 100, 150., 250.);
    if ( mep02_[ism-1] ) dbe->removeElement( mep02_[ism-1]->getName() );
    sprintf(histo, "EEPT pedestal mean G06 %s", Numbers::sEE(ism).c_str());
    mep02_[ism-1] = dbe->book1D(histo, histo, 100, 150., 250.);
    if ( mep03_[ism-1] ) dbe->removeElement( mep03_[ism-1]->getName() );
    sprintf(histo, "EEPT pedestal mean G12 %s", Numbers::sEE(ism).c_str());
    mep03_[ism-1] = dbe->book1D(histo, histo, 100, 150., 250.);

    if ( mer01_[ism-1] ) dbe->removeElement( mer01_[ism-1]->getName() );
    sprintf(histo, "EEPT pedestal rms G01 %s", Numbers::sEE(ism).c_str());
    mer01_[ism-1] = dbe->book1D(histo, histo, 100, 0., 10.);
    if ( mer02_[ism-1] ) dbe->removeElement( mer02_[ism-1]->getName() );
    sprintf(histo, "EEPT pedestal rms G06 %s", Numbers::sEE(ism).c_str());
    mer02_[ism-1] = dbe->book1D(histo, histo, 100, 0., 10.);
    if ( mer03_[ism-1] ) dbe->removeElement( mer03_[ism-1]->getName() );
    sprintf(histo, "EEPT pedestal rms G12 %s", Numbers::sEE(ism).c_str());
    mer03_[ism-1] = dbe->book1D(histo, histo, 100, 0., 10.);
    
    if ( mer04_[ism-1] ) dbe->removeElement( mer04_[ism-1]->getName() );
    sprintf(histo, "EEPDT PNs pedestal rms %s G01", Numbers::sEE(ism).c_str());
    mer04_[ism-1] = dbe->book1D(histo, histo, 100, 0., 10.);
    if ( mer05_[ism-1] ) dbe->removeElement( mer05_[ism-1]->getName() );
    sprintf(histo, "EEPDT PNs pedestal rms %s G16", Numbers::sEE(ism).c_str());
    mer05_[ism-1] = dbe->book1D(histo, histo, 100, 0., 10.);
    
    if ( mes01_[ism-1] ) dbe->removeElement( mes01_[ism-1]->getName() );
    sprintf(histo, "EEPT pedestal 3sum G01 %s", Numbers::sEE(ism).c_str());
    mes01_[ism-1] = dbe->book2D(histo, histo, 85, 0., 85., 20, 0., 20.);
    if ( mes02_[ism-1] ) dbe->removeElement( mes02_[ism-1]->getName() );
    sprintf(histo, "EEPT pedestal 3sum G06 %s", Numbers::sEE(ism).c_str());
    mes02_[ism-1] = dbe->book2D(histo, histo, 85, 0., 85., 20, 0., 20.);
    if ( mes03_[ism-1] ) dbe->removeElement( mes03_[ism-1]->getName() );
    sprintf(histo, "EEPT pedestal 3sum G12 %s", Numbers::sEE(ism).c_str());
    mes03_[ism-1] = dbe->book2D(histo, histo, 85, 0., 85., 20, 0., 20.);

    if ( met01_[ism-1] ) dbe->removeElement( met01_[ism-1]->getName() );
    sprintf(histo, "EEPT pedestal 5sum G01 %s", Numbers::sEE(ism).c_str());
    met01_[ism-1] = dbe->book2D(histo, histo, 85, 0., 85., 20, 0., 20.);
    if ( met02_[ism-1] ) dbe->removeElement( met02_[ism-1]->getName() );
    sprintf(histo, "EEPT pedestal 5sum G06 %s", Numbers::sEE(ism).c_str());
    met02_[ism-1] = dbe->book2D(histo, histo, 85, 0., 85., 20, 0., 20.);
    if ( met03_[ism-1] ) dbe->removeElement( met03_[ism-1]->getName() );
    sprintf(histo, "EEPT pedestal 5sum G12 %s", Numbers::sEE(ism).c_str());
    met03_[ism-1] = dbe->book2D(histo, histo, 85, 0., 85., 20, 0., 20.);

  }

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    UtilsClient::resetHisto( meg01_[ism-1] );
    UtilsClient::resetHisto( meg02_[ism-1] );
    UtilsClient::resetHisto( meg03_[ism-1] );

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

        meg01_[ism-1]->setBinContent( ie, ip, 2. );
        meg02_[ism-1]->setBinContent( ie, ip, 2. );
        meg03_[ism-1]->setBinContent( ie, ip, 2. );

      }
    }

    for ( int i = 1; i <= 10; i++ ) {

        meg04_[ism-1]->setBinContent( i, 1, 2. );
        meg05_[ism-1]->setBinContent( i, 1, 2. );

    }

    UtilsClient::resetHisto( mep01_[ism-1] );
    UtilsClient::resetHisto( mep02_[ism-1] );
    UtilsClient::resetHisto( mep03_[ism-1] );

    UtilsClient::resetHisto( mer01_[ism-1] );
    UtilsClient::resetHisto( mer02_[ism-1] );
    UtilsClient::resetHisto( mer03_[ism-1] );
    
    UtilsClient::resetHisto( mer04_[ism-1] );
    UtilsClient::resetHisto( mer05_[ism-1] );
    
    UtilsClient::resetHisto( mes01_[ism-1] );
    UtilsClient::resetHisto( mes02_[ism-1] );
    UtilsClient::resetHisto( mes03_[ism-1] );

    UtilsClient::resetHisto( met01_[ism-1] );
    UtilsClient::resetHisto( met02_[ism-1] );
    UtilsClient::resetHisto( met03_[ism-1] );

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

        mes01_[ism-1]->setBinContent( ie, ip, -999. );
        mes02_[ism-1]->setBinContent( ie, ip, -999. );
        mes03_[ism-1]->setBinContent( ie, ip, -999. );

        met01_[ism-1]->setBinContent( ie, ip, -999. );
        met02_[ism-1]->setBinContent( ie, ip, -999. );
        met03_[ism-1]->setBinContent( ie, ip, -999. );

      }
    }

  }

}

void EEPedestalClient::cleanup(void) {

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

  mui_->setCurrentFolder( "EcalEndcap/EEPedestalClient" );
  DaqMonitorBEInterface* dbe = mui_->getBEInterface();

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( meg01_[ism-1] ) dbe->removeElement( meg01_[ism-1]->getName() );
    meg01_[ism-1] = 0;
    if ( meg02_[ism-1] ) dbe->removeElement( meg02_[ism-1]->getName() );
    meg02_[ism-1] = 0;
    if ( meg03_[ism-1] ) dbe->removeElement( meg03_[ism-1]->getName() );
    meg03_[ism-1] = 0;

    if ( meg04_[ism-1] ) dbe->removeElement( meg04_[ism-1]->getName() );
    meg04_[ism-1] = 0;
    if ( meg05_[ism-1] ) dbe->removeElement( meg05_[ism-1]->getName() );
    meg05_[ism-1] = 0;

    if ( mep01_[ism-1] ) dbe->removeElement( mep01_[ism-1]->getName() );
    mep01_[ism-1] = 0;
    if ( mep02_[ism-1] ) dbe->removeElement( mep02_[ism-1]->getName() );
    mep02_[ism-1] = 0;
    if ( mep03_[ism-1] ) dbe->removeElement( mep03_[ism-1]->getName() );
    mep03_[ism-1] = 0;

    if ( mer01_[ism-1] ) dbe->removeElement( mer01_[ism-1]->getName() );
    mer01_[ism-1] = 0;
    if ( mer02_[ism-1] ) dbe->removeElement( mer02_[ism-1]->getName() );
    mer02_[ism-1] = 0;
    if ( mer03_[ism-1] ) dbe->removeElement( mer03_[ism-1]->getName() );
    mer03_[ism-1] = 0;
    
    if ( mer04_[ism-1] ) dbe->removeElement( mer04_[ism-1]->getName() );
    mer04_[ism-1] = 0;
    if ( mer05_[ism-1] ) dbe->removeElement( mer05_[ism-1]->getName() );
    mer05_[ism-1] = 0;
    
    if ( mes01_[ism-1] ) dbe->removeElement( mes01_[ism-1]->getName() );
    mes01_[ism-1] = 0;
    if ( mes02_[ism-1] ) dbe->removeElement( mes02_[ism-1]->getName() );
    mes02_[ism-1] = 0;
    if ( mes03_[ism-1] ) dbe->removeElement( mes03_[ism-1]->getName() );
    mes03_[ism-1] = 0;

    if ( met01_[ism-1] ) dbe->removeElement( met01_[ism-1]->getName() );
    met01_[ism-1] = 0;
    if ( met02_[ism-1] ) dbe->removeElement( met02_[ism-1]->getName() );
    met02_[ism-1] = 0;
    if ( met03_[ism-1] ) dbe->removeElement( met03_[ism-1]->getName() );
    met03_[ism-1] = 0;

  }

}

bool EEPedestalClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov, int ism) {

  bool status = true;

  UtilsClient::printBadChannels(qth01_[ism-1]);
  UtilsClient::printBadChannels(qth02_[ism-1]);
  UtilsClient::printBadChannels(qth03_[ism-1]);

  UtilsClient::printBadChannels(qth04_[ism-1]);
  UtilsClient::printBadChannels(qth05_[ism-1]);

  EcalLogicID ecid;
  MonPedestalsDat p;
  map<EcalLogicID, MonPedestalsDat> dataset1;

  for ( int ie = 1; ie <= 85; ie++ ) {
    for ( int ip = 1; ip <= 20; ip++ ) {

      bool update01;
      bool update02;
      bool update03;

      float num01, num02, num03;
      float mean01, mean02, mean03;
      float rms01, rms02, rms03;

      update01 = UtilsClient::getBinStats(h01_[ism-1], ie, ip, num01, mean01, rms01);
      update02 = UtilsClient::getBinStats(h02_[ism-1], ie, ip, num02, mean02, rms02);
      update03 = UtilsClient::getBinStats(h03_[ism-1], ie, ip, num03, mean03, rms03);

      if ( update01 || update02 || update03 ) {

        if ( ie == 1 && ip == 1 ) {

          cout << "Preparing dataset for SM=" << ism << endl;

          cout << "G01 (" << ie << "," << ip << ") " << num01  << " " << mean01 << " " << rms01  << endl;
          cout << "G06 (" << ie << "," << ip << ") " << num02  << " " << mean02 << " " << rms02  << endl;
          cout << "G12 (" << ie << "," << ip << ") " << num03  << " " << mean03 << " " << rms03  << endl;

          cout << endl;

        }

        p.setPedMeanG1(mean01);
        p.setPedRMSG1(rms01);

        p.setPedMeanG6(mean02);
        p.setPedRMSG6(rms02);

        p.setPedMeanG12(mean03);
        p.setPedRMSG12(rms03);

        if ( meg01_[ism-1] && int(meg01_[ism-1]->getBinContent( ie, ip )) % 3 == 1. &&
             meg02_[ism-1] && int(meg02_[ism-1]->getBinContent( ie, ip )) % 3 == 1. &&
             meg03_[ism-1] && int(meg03_[ism-1]->getBinContent( ie, ip )) % 3 == 1. ) {
          p.setTaskStatus(true);
        } else {
          p.setTaskStatus(false);
        }

        status = status && UtilsClient::getBinQual(meg01_[ism-1], ie, ip) &&
                           UtilsClient::getBinQual(meg02_[ism-1], ie, ip) &&
                           UtilsClient::getBinQual(meg03_[ism-1], ie, ip);

        int ic = (ip-1) + 20*(ie-1) + 1;

        if ( econn ) {
          try {
            ecid = LogicID::getEcalLogicID("EB_crystal_number", Numbers::iSM(ism), ic);
            dataset1[ecid] = p;
          } catch (runtime_error &e) {
            cerr << e.what() << endl;
          }
        }

      }

    }
  }

  if ( econn ) {
    try {
      cout << "Inserting MonPedestalsDat ... " << flush;
      if ( dataset1.size() != 0 ) econn->insertDataSet(&dataset1, moniov);
      cout << "done." << endl;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

  MonPNPedDat pn;
  map<EcalLogicID, MonPNPedDat> dataset2;

  for ( int i = 1; i <= 10; i++ ) {

    bool update01;
    bool update02;

    float num01, num02;
    float mean01, mean02;
    float rms01, rms02;

    update01 = UtilsClient::getBinStats(i01_[ism-1], 1, i, num01, mean01, rms01);
    update02 = UtilsClient::getBinStats(i02_[ism-1], 1, i, num02, mean02, rms02);

    if ( update01 || update02 ) {

      if ( i == 1 ) {

        cout << "Preparing dataset for SM=" << ism << endl;

        cout << "PNs (" << i << ") G01 " << num01  << " " << mean01 << " " << rms01  << endl;
        cout << "PNs (" << i << ") G16 " << num01  << " " << mean01 << " " << rms01  << endl;

        cout << endl;

      }

      pn.setPedMeanG1(mean01);
      pn.setPedRMSG1(rms01);

      pn.setPedMeanG16(mean02);
      pn.setPedRMSG16(rms02);

      if ( meg04_[ism-1] && int(meg04_[ism-1]->getBinContent( i, 1 )) % 3 == 1. &&
           meg05_[ism-1] && int(meg05_[ism-1]->getBinContent( i, 1 )) % 3 == 1. ) {
        pn.setTaskStatus(true);
      } else {
        pn.setTaskStatus(false);
      }

      status = status && UtilsClient::getBinQual(meg04_[ism-1], i, 1) &&
                         UtilsClient::getBinQual(meg05_[ism-1], i, 1);

      if ( econn ) {
        try {
          ecid = LogicID::getEcalLogicID("EB_LM_PN", Numbers::iSM(ism), i-1);
          dataset2[ecid] = pn;
        } catch (runtime_error &e) {
          cerr << e.what() << endl;
        }
      }

    }

  }

  if ( econn ) {
    try {
      cout << "Inserting MonPNPedDat ... " << flush;
      if ( dataset2.size() != 0 ) econn->insertDataSet(&dataset2, moniov);
      cout << "done." << endl;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

  return status;

}

void EEPedestalClient::subscribe(void){

  if ( verbose_ ) cout << "EEPedestalClient: subscribe" << endl;

  Char_t histo[200];

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    unsigned int ism = superModules_[i];

    sprintf(histo, "*/EcalEndcap/EEPedestalTask/Gain01/EEPT pedestal %s G01", Numbers::sEE(ism).c_str());
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EEPedestalTask/Gain06/EEPT pedestal %s G06", Numbers::sEE(ism).c_str());
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EEPedestalTask/Gain12/EEPT pedestal %s G12", Numbers::sEE(ism).c_str());
    mui_->subscribe(histo, ism);

    sprintf(histo, "*/EcalEndcap/EEPedestalTask/Gain01/EEPT pedestal 3sum %s G01", Numbers::sEE(ism).c_str());
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EEPedestalTask/Gain06/EEPT pedestal 3sum %s G06", Numbers::sEE(ism).c_str());
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EEPedestalTask/Gain12/EEPT pedestal 3sum %s G12", Numbers::sEE(ism).c_str());
    mui_->subscribe(histo, ism);

    sprintf(histo, "*/EcalEndcap/EEPedestalTask/Gain01/EEPT pedestal 5sum %s G01", Numbers::sEE(ism).c_str());
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EEPedestalTask/Gain06/EEPT pedestal 5sum %s G06", Numbers::sEE(ism).c_str());
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EEPedestalTask/Gain12/EEPT pedestal 5sum %s G12", Numbers::sEE(ism).c_str());
    mui_->subscribe(histo, ism);

    sprintf(histo, "*/EcalEndcap/EEPedestalTask/PN/Gain01/EEPDT PNs pedestal %s G01", Numbers::sEE(ism).c_str());
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EEPedestalTask/PN/Gain16/EEPDT PNs pedestal %s G16", Numbers::sEE(ism).c_str());
    mui_->subscribe(histo, ism);

  }

  if ( collateSources_ ) {

    if ( verbose_ ) cout << "EEPedestalClient: collate" << endl;

    for ( unsigned int i=0; i<superModules_.size(); i++ ) {

      int ism = superModules_[i];

      sprintf(histo, "EEPT pedestal %s G01", Numbers::sEE(ism).c_str());
      me_h01_[ism-1] = mui_->collateProf2D(histo, histo, "EcalEndcap/Sums/EEPedestalTask/Gain01");
      sprintf(histo, "*/EcalEndcap/EEPedestalTask/Gain01/EEPT pedestal %s G01", Numbers::sEE(ism).c_str());
      mui_->add(me_h01_[ism-1], histo);

      sprintf(histo, "EEPT pedestal %s G06", Numbers::sEE(ism).c_str());
      me_h02_[ism-1] = mui_->collateProf2D(histo, histo, "EcalEndcap/Sums/EEPedestalTask/Gain06");
      sprintf(histo, "*/EcalEndcap/EEPedestalTask/Gain06/EEPT pedestal %s G06", Numbers::sEE(ism).c_str());
      mui_->add(me_h02_[ism-1], histo);

      sprintf(histo, "EEPT pedestal %s G12", Numbers::sEE(ism).c_str());
      me_h03_[ism-1] = mui_->collateProf2D(histo, histo, "EcalEndcap/Sums/EEPedestalTask/Gain12");
      sprintf(histo, "*/EcalEndcap/EEPedestalTask/Gain12/EEPT pedestal %s G12", Numbers::sEE(ism).c_str());
      mui_->add(me_h03_[ism-1], histo);

      sprintf(histo, "EEPT pedestal 3sum %s G01", Numbers::sEE(ism).c_str());
      me_j01_[ism-1] = mui_->collateProf2D(histo, histo, "EcalEndcap/Sums/EEPedestalTask/Gain01");
      sprintf(histo, "*/EcalEndcap/EEPedestalTask/Gain01/EEPT pedestal 3sum %s G01", Numbers::sEE(ism).c_str());
      mui_->add(me_j01_[ism-1], histo);

      sprintf(histo, "EEPT pedestal 3sum %s G06", Numbers::sEE(ism).c_str());
      me_j02_[ism-1] = mui_->collateProf2D(histo, histo, "EcalEndcap/Sums/EEPedestalTask/Gain06");
      sprintf(histo, "*/EcalEndcap/EEPedestalTask/Gain06/EEPT pedestal 3sum %s G06", Numbers::sEE(ism).c_str());
      mui_->add(me_j02_[ism-1], histo);

      sprintf(histo, "EEPT pedestal 3sum %s G12", Numbers::sEE(ism).c_str());
      me_j03_[ism-1] = mui_->collateProf2D(histo, histo, "EcalEndcap/Sums/EEPedestalTask/Gain12");
      sprintf(histo, "*/EcalEndcap/EEPedestalTask/Gain12/EEPT pedestal 3sum %s G12", Numbers::sEE(ism).c_str());
      mui_->add(me_j03_[ism-1], histo);

      sprintf(histo, "EEPT pedestal 5sum %s G01", Numbers::sEE(ism).c_str());
      me_k01_[ism-1] = mui_->collateProf2D(histo, histo, "EcalEndcap/Sums/EEPedestalTask/Gain01");
      sprintf(histo, "*/EcalEndcap/EEPedestalTask/Gain01/EEPT pedestal 5sum %s G01", Numbers::sEE(ism).c_str());
      mui_->add(me_k01_[ism-1], histo);

      sprintf(histo, "EEPT pedestal 5sum %s G06", Numbers::sEE(ism).c_str());
      me_k02_[ism-1] = mui_->collateProf2D(histo, histo, "EcalEndcap/Sums/EEPedestalTask/Gain06");
      sprintf(histo, "*/EcalEndcap/EEPedestalTask/Gain06/EEPT pedestal 5sum %s G06", Numbers::sEE(ism).c_str());
      mui_->add(me_k02_[ism-1], histo);

      sprintf(histo, "EEPT pedestal 5sum %s G12", Numbers::sEE(ism).c_str());
      me_k03_[ism-1] = mui_->collateProf2D(histo, histo, "EcalEndcap/Sums/EEPedestalTask/Gain12");
      sprintf(histo, "*/EcalEndcap/EEPedestalTask/Gain12/EEPT pedestal 5sum %s G12", Numbers::sEE(ism).c_str());
      mui_->add(me_k03_[ism-1], histo);

      sprintf(histo, "EEPDT PNs pedestal %s G01", Numbers::sEE(ism).c_str());
      me_i01_[ism-1] = mui_->collateProf2D(histo, histo, "EcalEndcap/Sums/EEPedestalTask/PN/Gain01");
      sprintf(histo, "*/EcalEndcap/EEPedestalTask/PN/Gain01/EEPDT PNs pedestal %s G01", Numbers::sEE(ism).c_str());
      mui_->add(me_i01_[ism-1], histo);

      sprintf(histo, "EEPDT PNs pedestal %s G16", Numbers::sEE(ism).c_str());
      me_i02_[ism-1] = mui_->collateProf2D(histo, histo, "EcalEndcap/Sums/EEPedestalTask/PN/Gain16");
      sprintf(histo, "*/EcalEndcap/EEPedestalTask/PN/Gain16/EEPDT PNs pedestal %s G16", Numbers::sEE(ism).c_str());
      mui_->add(me_i02_[ism-1], histo);

    }

  }

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( collateSources_ ) {
      sprintf(histo, "EcalEndcap/Sums/EEPedestalTask/Gain01/EEPT pedestal %s G01", Numbers::sEE(ism).c_str());
      if ( qth01_[ism-1] ) mui_->useQTest(histo, qth01_[ism-1]->getName());
      sprintf(histo, "EcalEndcap/Sums/EEPedestalTask/Gain06/EEPT pedestal %s G06", Numbers::sEE(ism).c_str());
      if ( qth02_[ism-1] ) mui_->useQTest(histo, qth02_[ism-1]->getName());
      sprintf(histo, "EcalEndcap/Sums/EEPedestalTask/Gain12/EEPT pedestal %s G12", Numbers::sEE(ism).c_str());
      if ( qth03_[ism-1] ) mui_->useQTest(histo, qth03_[ism-1]->getName());
      sprintf(histo, "EcalEndcap/Sums/EEPedestalTask/PN/Gain01/EEPDT PNs pedestal %s G01", Numbers::sEE(ism).c_str());
      if ( qth04_[ism-1] ) mui_->useQTest(histo, qth04_[ism-1]->getName());
      sprintf(histo, "EcalEndcap/Sums/EEPedestalTask/PN/Gain16/EEPDT PNs pedestal %s G16", Numbers::sEE(ism).c_str());
      if ( qth05_[ism-1] ) mui_->useQTest(histo, qth05_[ism-1]->getName());
    } else {
      if ( enableMonitorDaemon_ ) {
        sprintf(histo, "*/EcalEndcap/EEPedestalTask/Gain01/EEPT pedestal %s G01", Numbers::sEE(ism).c_str());
        if ( qth01_[ism-1] ) mui_->useQTest(histo, qth01_[ism-1]->getName());
        sprintf(histo, "*/EcalEndcap/EEPedestalTask/Gain06/EEPT pedestal %s G06", Numbers::sEE(ism).c_str());
        if ( qth02_[ism-1] ) mui_->useQTest(histo, qth02_[ism-1]->getName());
        sprintf(histo, "*/EcalEndcap/EEPedestalTask/Gain12/EEPT pedestal %s G12", Numbers::sEE(ism).c_str());
        if ( qth03_[ism-1] ) mui_->useQTest(histo, qth03_[ism-1]->getName());
        sprintf(histo, "*/EcalEndcap/EEPedestalTask/PN/Gain01/EEPDT PNs pedestal %s G01", Numbers::sEE(ism).c_str());
        if ( qth04_[ism-1] ) mui_->useQTest(histo, qth04_[ism-1]->getName());
        sprintf(histo, "*/EcalEndcap/EEPedestalTask/PN/Gain16/EEPDT PNs pedestal %s G16", Numbers::sEE(ism).c_str());
        if ( qth05_[ism-1] ) mui_->useQTest(histo, qth05_[ism-1]->getName());
      } else {
        sprintf(histo, "EcalEndcap/EEPedestalTask/Gain01/EEPT pedestal %s G01", Numbers::sEE(ism).c_str());
        if ( qth01_[ism-1] ) mui_->useQTest(histo, qth01_[ism-1]->getName());
        sprintf(histo, "EcalEndcap/EEPedestalTask/Gain06/EEPT pedestal %s G06", Numbers::sEE(ism).c_str());
        if ( qth02_[ism-1] ) mui_->useQTest(histo, qth02_[ism-1]->getName());
        sprintf(histo, "EcalEndcap/EEPedestalTask/Gain12/EEPT pedestal %s G12", Numbers::sEE(ism).c_str());
        if ( qth03_[ism-1] ) mui_->useQTest(histo, qth03_[ism-1]->getName());
        sprintf(histo, "EcalEndcap/EEPedestalTask/PN/Gain01/EEPDT PNs pedestal %s G01", Numbers::sEE(ism).c_str());
        if ( qth04_[ism-1] ) mui_->useQTest(histo, qth04_[ism-1]->getName());
        sprintf(histo, "EcalEndcap/EEPedestalTask/PN/Gain16/EEPDT PNs pedestal %s G16", Numbers::sEE(ism).c_str());
        if ( qth05_[ism-1] ) mui_->useQTest(histo, qth05_[ism-1]->getName());
      }
    }

  }

}

void EEPedestalClient::subscribeNew(void){

  Char_t histo[200];

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    unsigned int ism = superModules_[i];

    sprintf(histo, "*/EcalEndcap/EEPedestalTask/Gain01/EEPT pedestal %s G01", Numbers::sEE(ism).c_str());
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalEndcap/EEPedestalTask/Gain06/EEPT pedestal %s G06", Numbers::sEE(ism).c_str());
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalEndcap/EEPedestalTask/Gain12/EEPT pedestal %s G12", Numbers::sEE(ism).c_str());
    mui_->subscribeNew(histo, ism);

    sprintf(histo, "*/EcalEndcap/EEPedestalTask/Gain01/EEPT pedestal 3sum %s G01", Numbers::sEE(ism).c_str());
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalEndcap/EEPedestalTask/Gain06/EEPT pedestal 3sum %s G06", Numbers::sEE(ism).c_str());
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalEndcap/EEPedestalTask/Gain12/EEPT pedestal 3sum %s G12", Numbers::sEE(ism).c_str());
    mui_->subscribeNew(histo, ism);

    sprintf(histo, "*/EcalEndcap/EEPedestalTask/Gain01/EEPT pedestal 5sum %s G01", Numbers::sEE(ism).c_str());
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalEndcap/EEPedestalTask/Gain06/EEPT pedestal 5sum %s G06", Numbers::sEE(ism).c_str());
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalEndcap/EEPedestalTask/Gain12/EEPT pedestal 5sum %s G12", Numbers::sEE(ism).c_str());
    mui_->subscribeNew(histo, ism);

    sprintf(histo, "*/EcalEndcap/EEPedestalTask/PN/Gain01/EEPDT PNs pedestal %s G01", Numbers::sEE(ism).c_str());
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalEndcap/EEPedestalTask/PN/Gain16/EEPDT PNs pedestal %s G16", Numbers::sEE(ism).c_str());
    mui_->subscribeNew(histo, ism);

  }

}

void EEPedestalClient::unsubscribe(void){

  if ( verbose_ ) cout << "EEPedestalClient: unsubscribe" << endl;

  if ( collateSources_ ) {

    if ( verbose_ ) cout << "EEPedestalClient: uncollate" << endl;

    if ( mui_ ) {

    for ( unsigned int i=0; i<superModules_.size(); i++ ) {

        int ism = superModules_[i];

        mui_->removeCollate(me_h01_[ism-1]);
        mui_->removeCollate(me_h02_[ism-1]);
        mui_->removeCollate(me_h03_[ism-1]);

        mui_->removeCollate(me_j01_[ism-1]);
        mui_->removeCollate(me_j02_[ism-1]);
        mui_->removeCollate(me_j03_[ism-1]);

        mui_->removeCollate(me_k01_[ism-1]);
        mui_->removeCollate(me_k02_[ism-1]);
        mui_->removeCollate(me_k03_[ism-1]);

        mui_->removeCollate(me_i01_[ism-1]);
        mui_->removeCollate(me_i02_[ism-1]);

      }

    }

  }

  Char_t histo[200];

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    unsigned int ism = superModules_[i];

    sprintf(histo, "*/EcalEndcap/EEPedestalTask/Gain01/EEPT pedestal %s G01", Numbers::sEE(ism).c_str());
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EEPedestalTask/Gain06/EEPT pedestal %s G06", Numbers::sEE(ism).c_str());
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EEPedestalTask/Gain12/EEPT pedestal %s G12", Numbers::sEE(ism).c_str());
    mui_->unsubscribe(histo, ism);

    sprintf(histo, "*/EcalEndcap/EEPedestalTask/Gain01/EEPT pedestal 3sum %s G01", Numbers::sEE(ism).c_str());
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EEPedestalTask/Gain06/EEPT pedestal 3sum %s G06", Numbers::sEE(ism).c_str());
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EEPedestalTask/Gain12/EEPT pedestal 3sum %s G12", Numbers::sEE(ism).c_str());
    mui_->unsubscribe(histo, ism);

    sprintf(histo, "*/EcalEndcap/EEPedestalTask/Gain01/EEPT pedestal 5sum %s G01", Numbers::sEE(ism).c_str());
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EEPedestalTask/Gain06/EEPT pedestal 5sum %s G06", Numbers::sEE(ism).c_str());
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EEPedestalTask/Gain12/EEPT pedestal 5sum %s G12", Numbers::sEE(ism).c_str());
    mui_->unsubscribe(histo, ism);

    sprintf(histo, "*/EcalEndcap/EEPedestalTask/PN/Gain01/EEPDT PNs pedestal %s G01", Numbers::sEE(ism).c_str());
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EEPedestalTask/PN/Gain16/EEPDT PNs pedestal %s G16", Numbers::sEE(ism).c_str());
    mui_->unsubscribe(histo, ism);

  }

}

void EEPedestalClient::softReset(void){

}

void EEPedestalClient::analyze(void){

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( verbose_ ) cout << "EEPedestalClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;
  }

  uint64_t bits01 = 0;
  bits01 |= EcalErrorDictionary::getMask("PEDESTAL_LOW_GAIN_MEAN_WARNING");
  bits01 |= EcalErrorDictionary::getMask("PEDESTAL_LOW_GAIN_RMS_WARNING");
  bits01 |= EcalErrorDictionary::getMask("PEDESTAL_LOW_GAIN_MEAN_ERROR");
  bits01 |= EcalErrorDictionary::getMask("PEDESTAL_LOW_GAIN_RMS_ERROR");

  uint64_t bits02 = 0;
  bits02 |= EcalErrorDictionary::getMask("PEDESTAL_MIDDLE_GAIN_MEAN_WARNING");
  bits02 |= EcalErrorDictionary::getMask("PEDESTAL_MIDDLE_GAIN_RMS_WARNING");
  bits02 |= EcalErrorDictionary::getMask("PEDESTAL_MIDDLE_GAIN_MEAN_ERROR");
  bits02 |= EcalErrorDictionary::getMask("PEDESTAL_MIDDLE_GAIN_RMS_ERROR");

  uint64_t bits03 = 0;
  bits03 |= EcalErrorDictionary::getMask("PEDESTAL_HIGH_GAIN_MEAN_WARNING");
  bits03 |= EcalErrorDictionary::getMask("PEDESTAL_HIGH_GAIN_RMS_WARNING");
  bits03 |= EcalErrorDictionary::getMask("PEDESTAL_HIGH_GAIN_MEAN_ERROR");
  bits03 |= EcalErrorDictionary::getMask("PEDESTAL_HIGH_GAIN_RMS_ERROR");

  map<EcalLogicID, RunCrystalErrorsDat> mask1;
  map<EcalLogicID, RunPNErrorsDat> mask2;

  EcalErrorMask::fetchDataSet(&mask1);
  EcalErrorMask::fetchDataSet(&mask2);

  Char_t histo[200];

  MonitorElement* me;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( collateSources_ ) {
      sprintf(histo, "EcalEndcap/Sums/EEPedestalTask/Gain01/EEPT pedestal %s G01", Numbers::sEE(ism).c_str());
    } else {
      sprintf(histo, (prefixME_+"EcalEndcap/EEPedestalTask/Gain01/EEPT pedestal %s G01").c_str(), Numbers::sEE(ism).c_str());
    }
    me = mui_->get(histo);
    h01_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h01_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalEndcap/Sums/EEPedestalTask/Gain06/EEPT pedestal %s G06", Numbers::sEE(ism).c_str());
    } else {
      sprintf(histo, (prefixME_+"EcalEndcap/EEPedestalTask/Gain06/EEPT pedestal %s G06").c_str(), Numbers::sEE(ism).c_str());
    }
    me = mui_->get(histo);
    h02_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h02_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalEndcap/Sums/EEPedestalTask/Gain12/EEPT pedestal %s G12", Numbers::sEE(ism).c_str());
    } else {
      sprintf(histo, (prefixME_+"EcalEndcap/EEPedestalTask/Gain12/EEPT pedestal %s G12").c_str(), Numbers::sEE(ism).c_str());
    }
    me = mui_->get(histo);
    h03_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h03_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalEndcap/Sums/EEPedestalTask/Gain01/EEPT pedestal 3sum %s G01", Numbers::sEE(ism).c_str());
    } else {
      sprintf(histo, (prefixME_+"EcalEndcap/EEPedestalTask/Gain01/EEPT pedestal 3sum %s G01").c_str(), Numbers::sEE(ism).c_str());
    }
    me = mui_->get(histo);
    j01_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, j01_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalEndcap/Sums/EEPedestalTask/Gain06/EEPT pedestal 3sum %s G06", Numbers::sEE(ism).c_str());
    } else {
      sprintf(histo, (prefixME_+"EcalEndcap/EEPedestalTask/Gain06/EEPT pedestal 3sum %s G06").c_str(), Numbers::sEE(ism).c_str());
    }
    me = mui_->get(histo);
    j02_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, j02_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalEndcap/Sums/EEPedestalTask/Gain12/EEPT pedestal 3sum %s G12", Numbers::sEE(ism).c_str());
    } else {
      sprintf(histo, (prefixME_+"EcalEndcap/EEPedestalTask/Gain12/EEPT pedestal 3sum %s G12").c_str(), Numbers::sEE(ism).c_str());
    }
    me = mui_->get(histo);
    j03_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, j03_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalEndcap/Sums/EEPedestalTask/Gain01/EEPT pedestal 5sum %s G01", Numbers::sEE(ism).c_str());
    } else {
      sprintf(histo, (prefixME_+"EcalEndcap/EEPedestalTask/Gain01/EEPT pedestal 5sum %s G01").c_str(), Numbers::sEE(ism).c_str());
    }
    me = mui_->get(histo);
    k01_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, k01_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalEndcap/Sums/EEPedestalTask/Gain06/EEPT pedestal 5sum %s G06", Numbers::sEE(ism).c_str());
    } else {
      sprintf(histo, (prefixME_+"EcalEndcap/EEPedestalTask/Gain06/EEPT pedestal 5sum %s G06").c_str(), Numbers::sEE(ism).c_str());
    }
    me = mui_->get(histo);
    k02_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, k02_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalEndcap/Sums/EEPedestalTask/Gain12/EEPT pedestal 5sum %s G12", Numbers::sEE(ism).c_str());
    } else {
      sprintf(histo, (prefixME_+"EcalEndcap/EEPedestalTask/Gain12/EEPT pedestal 5sum %s G12").c_str(), Numbers::sEE(ism).c_str());
    }
    me = mui_->get(histo);
    k03_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, k03_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalEndcap/Sums/EEPedestalTask/PN/Gain01/EEPDT PNs pedestal %s G01", Numbers::sEE(ism).c_str());
    } else {
      sprintf(histo, (prefixME_+"EcalEndcap/EEPedestalTask/PN/Gain01/EEPDT PNs pedestal %s G01").c_str(), Numbers::sEE(ism).c_str());
    }
    me = mui_->get(histo);
    i01_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, i01_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalEndcap/Sums/EEPedestalTask/PN/Gain16/EEPDT PNs pedestal %s G16", Numbers::sEE(ism).c_str());
    } else {
      sprintf(histo, (prefixME_+"EcalEndcap/EEPedestalTask/PN/Gain16/EEPDT PNs pedestal %s G16").c_str(), Numbers::sEE(ism).c_str());
    }
    me = mui_->get(histo);
    i02_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, i02_[ism-1] );

    UtilsClient::resetHisto( meg01_[ism-1] );
    UtilsClient::resetHisto( meg02_[ism-1] );
    UtilsClient::resetHisto( meg03_[ism-1] );

    UtilsClient::resetHisto( meg04_[ism-1] );
    UtilsClient::resetHisto( meg05_[ism-1] );

    UtilsClient::resetHisto( mep01_[ism-1] );
    UtilsClient::resetHisto( mep02_[ism-1] );
    UtilsClient::resetHisto( mep03_[ism-1] );

    UtilsClient::resetHisto( mer01_[ism-1] );
    UtilsClient::resetHisto( mer02_[ism-1] );
    UtilsClient::resetHisto( mer03_[ism-1] );

    UtilsClient::resetHisto( mer04_[ism-1] );
    UtilsClient::resetHisto( mer05_[ism-1] );
    
    UtilsClient::resetHisto( mes01_[ism-1] );
    UtilsClient::resetHisto( mes02_[ism-1] );
    UtilsClient::resetHisto( mes03_[ism-1] );

    UtilsClient::resetHisto( met01_[ism-1] );
    UtilsClient::resetHisto( met02_[ism-1] );
    UtilsClient::resetHisto( met03_[ism-1] );

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

        update01 = UtilsClient::getBinStats(h01_[ism-1], ie, ip, num01, mean01, rms01);
        update02 = UtilsClient::getBinStats(h02_[ism-1], ie, ip, num02, mean02, rms02);
        update03 = UtilsClient::getBinStats(h03_[ism-1], ie, ip, num03, mean03, rms03);

        if ( update01 ) {

          float val;

          val = 1.;
          if ( fabs(mean01 - expectedMean_[0]) > discrepancyMean_[0] )
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
          if ( fabs(mean02 - expectedMean_[1]) > discrepancyMean_[1] )
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
          if ( fabs(mean03 - expectedMean_[2]) > discrepancyMean_[2] )
            val = 0.;
          if ( rms03 > RMSThreshold_[2] )
            val = 0.;
          if ( meg03_[ism-1] ) meg03_[ism-1]->setBinContent(ie, ip, val);

          if ( mep03_[ism-1] ) mep03_[ism-1]->Fill(mean03);
          if ( mer03_[ism-1] ) mer03_[ism-1]->Fill(rms03);

        }

        // masking

        if ( mask1.size() != 0 ) {
          map<EcalLogicID, RunCrystalErrorsDat>::const_iterator m;
          for (m = mask1.begin(); m != mask1.end(); m++) {

            EcalLogicID ecid = m->first;

            int ic = (ip-1) + 20*(ie-1) + 1;

            if ( ecid.getID1() == Numbers::iSM(ism) && ecid.getID2() == ic ) {
              if ( (m->second).getErrorBits() & bits01 ) {
                if ( meg01_[ism-1] ) {
                  float val = int(meg01_[ism-1]->getBinContent(ie, ip)) % 3;
                  meg01_[ism-1]->setBinContent( ie, ip, val+3 );
                }
              }
              if ( (m->second).getErrorBits() & bits02 ) {
                if ( meg02_[ism-1] ) {
                  float val = int(meg02_[ism-1]->getBinContent(ie, ip)) % 3;
                  meg02_[ism-1]->setBinContent( ie, ip, val+3 );
                }
              }
              if ( (m->second).getErrorBits() & bits03 ) {
                if ( meg03_[ism-1] ) {
                  float val = int(meg03_[ism-1]->getBinContent(ie, ip)) % 3;
                  meg03_[ism-1]->setBinContent( ie, ip, val+3 );
                }
              }
            }

          }
        }

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

      update01 = UtilsClient::getBinStats(i01_[ism-1], 1, i, num01, mean01, rms01);
      update02 = UtilsClient::getBinStats(i02_[ism-1], 1, i, num02, mean02, rms02);

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
	//        if ( mean02 < pedestalThresholdPn_ )
         if ( mean02 < (expectedMeanPn_[1] - discrepancyMeanPn_[1])
              || (expectedMeanPn_[1] + discrepancyMeanPn_[1]) <  mean02)
           val = 0.;
         if ( rms02 >  RMSThresholdPn_[1])
           val = 0.;

        if ( meg05_[ism-1] ) meg05_[ism-1]->setBinContent(i, 1, val);
      }

      // masking

      if ( mask2.size() != 0 ) {
        map<EcalLogicID, RunPNErrorsDat>::const_iterator m;
        for (m = mask2.begin(); m != mask2.end(); m++) {

          EcalLogicID ecid = m->first;

          if ( ecid.getID1() == Numbers::iSM(ism) && ecid.getID2() == i-1 ) {
            if ( (m->second).getErrorBits() & bits01 ) {
              if ( meg04_[ism-1] ) {
                float val = int(meg04_[ism-1]->getBinContent(i, 1)) % 3;
                meg04_[ism-1]->setBinContent( i, 1, val+3 );
              }
            }
            if ( (m->second).getErrorBits() & bits03 ) {
              if ( meg05_[ism-1] ) {
                float val = int(meg05_[ism-1]->getBinContent(i, 1)) % 3;
                meg05_[ism-1]->setBinContent( i, 1, val+3 );
              }
            }
          }

        }
      }

    }

    vector<dqm::me_util::Channel> badChannels;

    if ( qth01_[ism-1] ) badChannels = qth01_[ism-1]->getBadChannels();

//    if ( ! badChannels.empty() ) {
//      for ( vector<dqm::me_util::Channel>::iterator it = badChannels.begin(); it != badChannels.end(); ++it ) {
//        if ( meg01_[ism-1] ) meg01_[ism-1]->setBinContent(it->getBinX(), it->getBinY(), 0.);
//      }
//    }

    if ( qth02_[ism-1] ) badChannels = qth02_[ism-1]->getBadChannels();

//    if ( ! badChannels.empty() ) {
//      for ( vector<dqm::me_util::Channel>::iterator it = badChannels.begin(); it != badChannels.end(); ++it ) {
//        if ( meg02_[ism-1] ) meg02_[ism-1]->setBinContent(it->getBinX(), it->getBinY(), 0.);
//      }
//    }

    if ( qth03_[ism-1] ) badChannels = qth03_[ism-1]->getBadChannels();

//    if ( ! badChannels.empty() ) {
//      for ( vector<dqm::me_util::Channel>::iterator it = badChannels.begin(); it != badChannels.end(); ++it ) {
//        if ( meg03_[ism-1] ) meg03_[ism-1]->setBinContent(it->getBinX(), it->getBinY(), 0.);
//      }
//    }

    if ( qth04_[ism-1] ) badChannels = qth04_[ism-1]->getBadChannels();

//    if ( ! badChannels.empty() ) {
//      for ( vector<dqm::me_util::Channel>::iterator it = badChannels.begin(); it != badChannels.end(); ++it ) {
//        if ( meg04_[ism-1] ) meg04_[ism-1]->setBinContent(it->getBinX(), it->getBinY(), 0.);
//      }
//    }

    if ( qth05_[ism-1] ) badChannels = qth05_[ism-1]->getBadChannels();

//    if ( ! badChannels.empty() ) {
//      for ( vector<dqm::me_util::Channel>::iterator it = badChannels.begin(); it != badChannels.end(); ++it ) {
//        if ( meg05_[ism-1] ) meg05_[ism-1]->setBinContent(it->getBinX(), it->getBinY(), 0.);
//      }
//    }

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

        if ( mes01_[ism-1] ) mes01_[ism-1]->setBinContent(ie, ip, -999.);
        if ( mes02_[ism-1] ) mes02_[ism-1]->setBinContent(ie, ip, -999.);
        if ( mes03_[ism-1] ) mes03_[ism-1]->setBinContent(ie, ip, -999.);

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
          if ( x3val01 != 0 && y3val01 != 0 ) z3val01 = sqrt(fabs(x3val01 - y3val01));
          if ( (x3val01 - y3val01) < 0 ) z3val01 = -z3val01;

          if ( mes01_[ism-1] ) mes01_[ism-1]->setBinContent(ie, ip, z3val01);

          z3val02 = -999.;
          if ( x3val02 != 0 && y3val02 != 0 ) z3val02 = sqrt(fabs(x3val02 - y3val02));
          if ( (x3val02 - y3val02) < 0 ) z3val02 = -z3val02;

          if ( mes02_[ism-1] ) mes02_[ism-1]->setBinContent(ie, ip, z3val02);

          z3val03 = -999.;
          if ( x3val03 != 0 && y3val03 != 0 ) z3val03 = sqrt(fabs(x3val03 - y3val03));
          if ( (x3val03 - y3val03) < 0 ) z3val03 = -z3val03;

          if ( mes03_[ism-1] ) mes03_[ism-1]->setBinContent(ie, ip, z3val03);

        }

        float x5val01;
        float x5val02;
        float x5val03;

        float y5val01;
        float y5val02;
        float y5val03;

        float z5val01;
        float z5val02;
        float z5val03;

        if ( met01_[ism-1] ) met01_[ism-1]->setBinContent(ie, ip, -999.);
        if ( met02_[ism-1] ) met02_[ism-1]->setBinContent(ie, ip, -999.);
        if ( met03_[ism-1] ) met03_[ism-1]->setBinContent(ie, ip, -999.);

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
          if ( x5val01 != 0 && y5val01 != 0 ) z5val01 = sqrt(fabs(x5val01 - y5val01));
          if ( (x5val01 - y5val01) < 0 ) z5val01 = -z5val01;

          if ( met01_[ism-1] ) met01_[ism-1]->setBinContent(ie, ip, z5val01);

          z5val02 = -999.;
          if ( x5val02 != 0 && y5val02 != 0 ) z5val02 = sqrt(fabs(x5val02 - y5val02));
          if ( (x5val02 - y5val02) < 0 ) z5val02 = -z5val02;

          if ( met02_[ism-1] ) met02_[ism-1]->setBinContent(ie, ip, z5val02);

          z5val03 = -999.;
          if ( x5val03 != 0 && y5val03 != 0 ) z5val03 = sqrt(fabs(x5val03 - y5val03));
          if ( (x5val03 - y5val03) < 0 ) z5val03 = -z5val03;

          if ( met03_[ism-1] ) met03_[ism-1]->setBinContent(ie, ip, z5val03);

        }

      }
    }

  }

}

void EEPedestalClient::htmlOutput(int run, string htmlDir, string htmlName){

  cout << "Preparing EEPedestalClient html output ..." << endl;

  ofstream htmlFile;

  htmlFile.open((htmlDir + htmlName).c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlFile << "<html>  " << endl;
  htmlFile << "<head>  " << endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlFile << " http-equiv=\"content-type\">  " << endl;
  htmlFile << "  <title>Monitor:PedestalTask output</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlFile << "<body>  " << endl;
  //htmlFile << "<br>  " << endl;
  htmlFile << "<a name=""top""></a>" << endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << run << "</span></h2>" << endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">PEDESTAL</span></h2> " << endl;
  htmlFile << "<hr>" << endl;
  htmlFile << "<table border=1><tr><td bgcolor=red>channel has problems in this task</td>" << endl;
  htmlFile << "<td bgcolor=lime>channel has NO problems</td>" << endl;
  htmlFile << "<td bgcolor=yellow>channel is missing</td></table>" << endl;
  htmlFile << "<br>" << endl;
  htmlFile << "<table border=1>" << std::endl;
  for ( unsigned int i=0; i<superModules_.size(); i ++ ) {
    htmlFile << "<td bgcolor=white><a href=""#"
	     << Numbers::sEE(superModules_[i]).c_str() << ">"
	     << setfill( '0' ) << setw(2) << superModules_[i] << "</a></td>";
  }
  htmlFile << std::endl << "</table>" << std::endl;

  // Produce the plots to be shown as .png files from existing histograms

  const int csize = 250;

  const double histMax = 1.e15;

  int pCol3[6] = { 301, 302, 303, 304, 305, 306 };

  int pCol4[10];
  for ( int i = 0; i < 10; i++ ) pCol4[i] = 401+i;

  TH2C dummy( "dummy", "dummy for sm", 85, 0., 85., 20, 0., 20. );
  for ( int i = 0; i < 68; i++ ) {
    int a = 2 + ( i/4 ) * 5;
    int b = 2 + ( i%4 ) * 5;
    dummy.Fill( a, b, i+1 );
  }
  dummy.SetMarkerSize(2);
  dummy.SetMinimum(0.1);

  TH2C dummy1( "dummy1", "dummy1 for sm mem", 10, 0, 10, 5, 0, 5 );
  for ( short i=0; i<2; i++ ) {
    int a = 2 + i*5;
    int b = 2;
    dummy1.Fill( a, b, i+1+68 );
  }
  dummy1.SetMarkerSize(2);
  dummy1.SetMinimum(0.1);

  string imgNameQual[3], imgNameMean[3], imgNameRMS[3], imgName3Sum[3], imgName5Sum[3], imgNameMEPnQual[2], imgNameMEPnPed[2],imgNameMEPnPedRms[2], imgName, meName;
  
  TCanvas* cQual = new TCanvas("cQual", "Temp", 2*csize, csize);
  TCanvas* cMean = new TCanvas("cMean", "Temp", csize, csize);
  TCanvas* cRMS = new TCanvas("cRMS", "Temp", csize, csize);
  TCanvas* c3Sum = new TCanvas("c3Sum", "Temp", 2*csize, csize);
  TCanvas* c5Sum = new TCanvas("c5Sum", "Temp", 2*csize, csize);
  TCanvas* cPed = new TCanvas("cPed", "Temp", csize, csize);

  TH2F* obj2f;
  TH1F* obj1f;
  TH1D* obj1d;

  // Loop on barrel supermodules

  for ( unsigned int i=0; i<superModules_.size(); i ++ ) {

    int ism = superModules_[i];

    // Loop on gains

    for ( int iCanvas = 1 ; iCanvas <= 3 ; iCanvas++ ) {

      // Quality plots

      imgNameQual[iCanvas-1] = "";

      obj2f = 0;
      switch ( iCanvas ) {
        case 1:
          obj2f = UtilsClient::getHisto<TH2F*>( meg01_[ism-1] );
          break;
        case 2:
          obj2f = UtilsClient::getHisto<TH2F*>( meg02_[ism-1] );
          break;
        case 3:
          obj2f = UtilsClient::getHisto<TH2F*>( meg03_[ism-1] );
          break;
        default:
          break;
      }

      if ( obj2f ) {

        meName = obj2f->GetName();

        for ( unsigned int i = 0; i < meName.size(); i++ ) {
          if ( meName.substr(i, 1) == " " )  {
            meName.replace(i, 1, "_");
          }
        }
        imgNameQual[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgNameQual[iCanvas-1];

        cQual->cd();
        gStyle->SetOptStat(" ");
        gStyle->SetPalette(6, pCol3);
        obj2f->GetXaxis()->SetNdivisions(17);
        obj2f->GetYaxis()->SetNdivisions(4);
        cQual->SetGridx();
        cQual->SetGridy();
        obj2f->SetMinimum(-0.00000001);
        obj2f->SetMaximum(6.0);
        obj2f->Draw("col");
        dummy.Draw("text,same");
        cQual->Update();
        cQual->SaveAs(imgName.c_str());

      }

      // Mean distributions

      imgNameMean[iCanvas-1] = "";

      obj1f = 0;
      switch ( iCanvas ) {
        case 1:
          obj1f = UtilsClient::getHisto<TH1F*>( mep01_[ism-1] );
          break;
        case 2:
          obj1f = UtilsClient::getHisto<TH1F*>( mep02_[ism-1] );
          break;
        case 3:
          obj1f = UtilsClient::getHisto<TH1F*>( mep03_[ism-1] );
          break;
        default:
            break;
      }

      if ( obj1f ) {

        meName = obj1f->GetName();

        for ( unsigned int i = 0; i < meName.size(); i++ ) {
          if ( meName.substr(i, 1) == " " )  {
            meName.replace(i, 1 ,"_" );
          }
        }
        imgNameMean[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgNameMean[iCanvas-1];

        cMean->cd();
        gStyle->SetOptStat("euomr");
        obj1f->SetStats(kTRUE);
        if ( obj1f->GetMaximum(histMax) > 0. ) {
          gPad->SetLogy(1);
        } else {
          gPad->SetLogy(0);
        }
        obj1f->Draw();
        cMean->Update();
        cMean->SaveAs(imgName.c_str());
        gPad->SetLogy(0);

      }

      // RMS distributions

      imgNameRMS[iCanvas-1] = "";

      obj1f = 0;
      switch ( iCanvas ) {
        case 1:
          obj1f = UtilsClient::getHisto<TH1F*>( mer01_[ism-1] );
          break;
        case 2:
          obj1f = UtilsClient::getHisto<TH1F*>( mer02_[ism-1] );
          break;
        case 3:
          obj1f = UtilsClient::getHisto<TH1F*>( mer03_[ism-1] );
          break;
        default:
          break;
      }

      if ( obj1f ) {

        meName = obj1f->GetName();

        for ( unsigned int i = 0; i < meName.size(); i++ ) {
          if ( meName.substr(i, 1) == " " )  {
            meName.replace(i, 1, "_");
          }
        }
        imgNameRMS[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgNameRMS[iCanvas-1];

        cRMS->cd();
        gStyle->SetOptStat("euomr");
        obj1f->SetStats(kTRUE);
        if ( obj1f->GetMaximum(histMax) > 0. ) {
          gPad->SetLogy(1);
        } else {
          gPad->SetLogy(0);
        }
        obj1f->Draw();
        cRMS->Update();
        cRMS->SaveAs(imgName.c_str());
        gPad->SetLogy(0);

      }

      // 3Sum distributions

      imgName3Sum[iCanvas-1] = "";

      obj2f = 0;
      switch ( iCanvas ) {
        case 1:
          obj2f = UtilsClient::getHisto<TH2F*>( mes01_[ism-1] );
          break;
        case 2:
          obj2f = UtilsClient::getHisto<TH2F*>( mes02_[ism-1] );
          break;
        case 3:
          obj2f = UtilsClient::getHisto<TH2F*>( mes03_[ism-1] );
          break;
        default:
          break;
      }

      if ( obj2f ) {

        meName = obj2f->GetName();

        for ( unsigned int i = 0; i < meName.size(); i++ ) {
          if ( meName.substr(i, 1) == " " )  {
            meName.replace(i, 1, "_");
          }
        }
        imgName3Sum[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgName3Sum[iCanvas-1];

        c3Sum->cd();
        gStyle->SetOptStat(" ");
        gStyle->SetPalette(10, pCol4);
        obj2f->GetXaxis()->SetNdivisions(17);
        obj2f->GetYaxis()->SetNdivisions(4);
        c3Sum->SetGridx();
        c3Sum->SetGridy();
        obj2f->SetMinimum(-0.5);
        obj2f->SetMaximum(+0.5);
        obj2f->Draw("colz");
        dummy.Draw("text,same");
        c3Sum->Update();
        c3Sum->SaveAs(imgName.c_str());

      }

      // 5Sum distributions

      imgName5Sum[iCanvas-1] = "";

      obj2f = 0;
      switch ( iCanvas ) {
        case 1:
          obj2f = UtilsClient::getHisto<TH2F*>( met01_[ism-1] );
          break;
        case 2:
          obj2f = UtilsClient::getHisto<TH2F*>( met02_[ism-1] );
          break;
        case 3:
          obj2f = UtilsClient::getHisto<TH2F*>( met03_[ism-1] );
          break;
        default:
          break;
      }

      if ( obj2f ) {

        meName = obj2f->GetName();

        for ( unsigned int i = 0; i < meName.size(); i++ ) {
          if ( meName.substr(i, 1) == " " )  {
            meName.replace(i, 1, "_");
          }
        }
        imgName5Sum[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgName5Sum[iCanvas-1];

        c5Sum->cd();
        gStyle->SetOptStat(" ");
        gStyle->SetPalette(10, pCol4);
        obj2f->GetXaxis()->SetNdivisions(17);
        obj2f->GetYaxis()->SetNdivisions(4);
        c5Sum->SetGridx();
        c5Sum->SetGridy();
        obj2f->SetMinimum(-0.5);
        obj2f->SetMaximum(+0.5);
        obj2f->Draw("colz");
        dummy.Draw("text,same");
        c5Sum->Update();
        c5Sum->SaveAs(imgName.c_str());

      }

    }

    // Loop on gains

    for ( int iCanvas = 1 ; iCanvas <= 2 ; iCanvas++ ) {

      // Monitoring elements plots

      imgNameMEPnQual[iCanvas-1] = "";

      obj2f = 0;
      switch ( iCanvas ) {
      case 1:
        obj2f = UtilsClient::getHisto<TH2F*>( meg04_[ism-1] );
        break;
      case 2:
        obj2f = UtilsClient::getHisto<TH2F*>( meg05_[ism-1] );
        break;
      default:
        break;
      }

      if ( obj2f ) {

        meName = obj2f->GetName();

        for ( unsigned int i = 0; i < meName.size(); i++ ) {
          if ( meName.substr(i, 1) == " " )  {
            meName.replace(i, 1, "_");
          }
        }
        imgNameMEPnQual[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgNameMEPnQual[iCanvas-1];

        cQual->cd();
        gStyle->SetOptStat(" ");
        gStyle->SetPalette(6, pCol3);
        obj2f->GetXaxis()->SetNdivisions(10);
        obj2f->GetYaxis()->SetNdivisions(5);
        cQual->SetGridx();
        cQual->SetGridy(0);
        obj2f->SetMinimum(-0.00000001);
        obj2f->SetMaximum(6.0);
        obj2f->Draw("col");
        dummy1.Draw("text,same");
        cQual->Update();
        cQual->SaveAs(imgName.c_str());

      }

      imgNameMEPnPed[iCanvas-1] = "";

      obj1d = 0;
      switch ( iCanvas ) {
        case 1:
          if ( i01_[ism-1] ) obj1d = i01_[ism-1]->ProjectionY("_py", 1, 1, "e");
          break;
        case 2:
          if ( i02_[ism-1] ) obj1d = i02_[ism-1]->ProjectionY("_py", 1, 1, "e");
          break;
        default:
          break;
      }

      if ( obj1d ) {

        meName = obj1d->GetName();

        for ( unsigned int i = 0; i < meName.size(); i++ ) {
          if ( meName.substr(i, 1) == " " )  {
            meName.replace(i, 1 ,"_" );
          }
        }
        imgNameMEPnPed[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgNameMEPnPed[iCanvas-1];

        cPed->cd();
        gStyle->SetOptStat("euo");
        obj1d->SetStats(kTRUE);
//        if ( obj1d->GetMaximum(histMax) > 0. ) {
//          gPad->SetLogy(1);
//        } else {
//          gPad->SetLogy(0);
//        }
        obj1d->SetMinimum(0.0);
        obj1d->Draw();
        cPed->Update();
        cPed->SaveAs(imgName.c_str());
        gPad->SetLogy(0);

        delete obj1d;

      }
      
      imgNameMEPnPedRms[iCanvas-1] = "";
      
      obj1f = 0;
      switch ( iCanvas ) {
      case 1:
	if ( mer04_[ism-1] ) obj1f =  UtilsClient::getHisto<TH1F*>(mer04_[ism-1]);
	break;
      case 2:
	if ( mer05_[ism-1] ) obj1f =  UtilsClient::getHisto<TH1F*>(mer05_[ism-1]);
	break;
      default:
	break;
      }
      
      if ( obj1f ) {
  	
	meName = obj1f->GetName();
  	
	for ( unsigned int i = 0; i < meName.size(); i++ ) {
	  if ( meName.substr(i, 1) == " " )  {
	    meName.replace(i, 1 ,"_" );
	  }
	}
	imgNameMEPnPedRms[iCanvas-1] = meName + ".png";
	imgName = htmlDir + imgNameMEPnPedRms[iCanvas-1];
  	
	cPed->cd();
	gStyle->SetOptStat("euomr");
	obj1f->SetStats(kTRUE);
	//        if ( obj1f->GetMaximum(histMax) > 0. ) {
	//          gPad->SetLogy(1);
	//        } else {
	//          gPad->SetLogy(0);
	//        }
	obj1f->Draw();
	cPed->Update();
	cPed->SaveAs(imgName.c_str());
	gPad->SetLogy(0);
      }
      
    }
    
    if( i>0 ) htmlFile << "<a href=""#top"">Top</a>" << std::endl;
    htmlFile << "<hr>" << std::endl;
    htmlFile << "<h3><a name="""
	     << Numbers::sEE(ism).c_str() << """></a><strong>"
	     << Numbers::sEE(ism).c_str() << "</strong></h3>" << endl;
    htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
    htmlFile << "<tr align=\"center\">" << endl;

    for ( int iCanvas = 1 ; iCanvas <= 3 ; iCanvas++ ) {

      if ( imgNameQual[iCanvas-1].size() != 0 )
        htmlFile << "<td colspan=\"2\"><img src=\"" << imgNameQual[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td colspan=\"2\"><img src=\"" << " " << "\"></td>" << endl;

    }

    htmlFile << "</tr>" << endl;
    htmlFile << "<tr>" << endl;

    for ( int iCanvas = 1 ; iCanvas <= 3 ; iCanvas++ ) {

      if ( imgNameMean[iCanvas-1].size() != 0 )
        htmlFile << "<td><img src=\"" << imgNameMean[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

      if ( imgNameRMS[iCanvas-1].size() != 0 )
        htmlFile << "<td><img src=\"" << imgNameRMS[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

    }

    htmlFile << "</tr>" << endl;

    htmlFile << "<tr align=\"center\"><td colspan=\"2\">Gain 1</td><td colspan=\"2\">Gain 6</td><td colspan=\"2\">Gain 12</td></tr>" << endl;
    htmlFile << "</table>" << endl;
    htmlFile << "<br>" << endl;

    htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
    htmlFile << "<tr align=\"center\">" << endl;

    for ( int iCanvas = 1 ; iCanvas <= 3 ; iCanvas++ ) {

      if ( imgName3Sum[iCanvas-1].size() != 0 )
        htmlFile << "<td><img src=\"" << imgName3Sum[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

    }

    htmlFile << "</tr>" << endl;

    htmlFile << "</table>" << endl;
    htmlFile << "<br>" << endl;

    htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
    htmlFile << "<tr align=\"center\">" << endl;

    for ( int iCanvas = 1 ; iCanvas <= 3 ; iCanvas++ ) {

      if ( imgName5Sum[iCanvas-1].size() != 0 )
        htmlFile << "<td><img src=\"" << imgName5Sum[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

    }

    htmlFile << "</tr>" << endl;

    htmlFile << "<tr align=\"center\"><td>Gain 1</td><td>Gain 6</td><td>Gain 12</td></tr>" << endl;
    htmlFile << "</table>" << endl;
    htmlFile << "<br>" << endl;

    htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\"> " << endl;
    htmlFile << "<tr align=\"center\">" << endl;

    for ( int iCanvas = 1 ; iCanvas <= 2 ; iCanvas++ ) {

      if ( imgNameMEPnQual[iCanvas-1].size() != 0 )
        htmlFile << "<td colspan=\"2\"><img src=\"" << imgNameMEPnQual[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td colspan=\"2\"><img src=\"" << " " << "\"></td>" << endl;

    }

    htmlFile << "</tr>" << endl;
    htmlFile << "</table>" << endl;
    htmlFile << "<br>" << endl;

    htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
    htmlFile << "<tr align=\"center\">" << endl;

    for ( int iCanvas = 1 ; iCanvas <= 2 ; iCanvas++ ) {
      
      if ( imgNameMEPnPed[iCanvas-1].size() != 0 ){
        htmlFile << "<td colspan=\"2\"><img src=\"" << imgNameMEPnPed[iCanvas-1] << "\"></td>" << endl;
	
	if ( imgNameMEPnPedRms[iCanvas-1].size() != 0 )
	  htmlFile << "<td colspan=\"2\"><img src=\"" << imgNameMEPnPedRms[iCanvas-1] << "\"></td>" << endl;
	else
	  htmlFile << "<td colspan=\"2\"><img src=\"" << " " << "\"></td>" << endl;
	
      }
      
      else{
	htmlFile << "<td colspan=\"2\"><img src=\"" << " " << "\"></td>" << endl;
	
	if ( imgNameMEPnPedRms[iCanvas-1].size() != 0 )
	  htmlFile << "<td colspan=\"2\"><img src=\"" << imgNameMEPnPedRms[iCanvas-1] << "\"></td>" << endl;
	else
	  htmlFile << "<td colspan=\"2\"><img src=\"" << " " << "\"></td>" << endl;
      }
      
    }  
    
    htmlFile << "</tr>" << endl;
    
    htmlFile << "<tr align=\"right\"><td colspan=\"2\">Gain 1</td>  <td colspan=\"2\"> </td> <td colspan=\"2\">Gain 16</td></tr>" << endl;
    htmlFile << "</table>" << endl;
    htmlFile << "<br>" << endl;
    
   }
  
  delete cQual;
  delete cMean;
  delete cRMS;
  delete c3Sum;
  delete c5Sum;
  delete cPed;

  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  htmlFile.close();

  }
