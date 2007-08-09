/*
 * \file EELedClient.cc
 *
 * $Date: 2007/07/27 16:41:56 $
 * $Revision: 1.4 $
 * \author G. Della Ricca
 * \author G. Franzoni
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
#include "OnlineDB/EcalCondDB/interface/MonLaserBlueDat.h"
#include "OnlineDB/EcalCondDB/interface/MonPNBlueDat.h"
#include "OnlineDB/EcalCondDB/interface/RunCrystalErrorsDat.h"
#include "OnlineDB/EcalCondDB/interface/RunPNErrorsDat.h"

#include "CondTools/Ecal/interface/EcalErrorDictionary.h"

#include "DQM/EcalCommon/interface/EcalErrorMask.h"
#include <DQM/EcalCommon/interface/UtilsClient.h>
#include <DQM/EcalCommon/interface/LogicID.h>
#include <DQM/EcalCommon/interface/Numbers.h>

#include <DQM/EcalEndcapMonitorClient/interface/EELedClient.h>

using namespace cms;
using namespace edm;
using namespace std;

EELedClient::EELedClient(const ParameterSet& ps){

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

    h09_[ism-1] = 0;

    h13_[ism-1] = 0;
    h14_[ism-1] = 0;

    h21_[ism-1] = 0;

    hs01_[ism-1] = 0;

    hs05_[ism-1] = 0;

    i01_[ism-1] = 0;

    i05_[ism-1] = 0;

    i09_[ism-1] = 0;

    i13_[ism-1] = 0;

  }

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    meg01_[ism-1] = 0;

    meg05_[ism-1] = 0;

    meg09_[ism-1] = 0;

    mea01_[ism-1] = 0;

    mea05_[ism-1] = 0;

    met01_[ism-1] = 0;

    met05_[ism-1] = 0;

    metav01_[ism-1] = 0;

    metav05_[ism-1] = 0;

    metrms01_[ism-1] = 0;

    metrms05_[ism-1] = 0;

    meaopn01_[ism-1] = 0;

    meaopn05_[ism-1] = 0;
 
    mepnprms01_[ism-1] = 0;

    mepnprms05_[ism-1] = 0;
    
    qth01_[ism-1] = 0;

    qth05_[ism-1] = 0;

    qth09_[ism-1] = 0;

    qth13_[ism-1] = 0;

    qth17_[ism-1] = 0;

    qth21_[ism-1] = 0;

    qtg01_[ism-1] = 0;

    qtg05_[ism-1] = 0;

    qtg09_[ism-1] = 0;

  }

  percentVariation_ = 0.4;

  amplitudeThresholdPnG01_ = 50.;
  amplitudeThresholdPnG16_ = 50.;
  
  pedPnExpectedMean_[0] = 750.0;
  pedPnExpectedMean_[1] = 750.0;
  
  pedPnDiscrepancyMean_[0] = 100.0;
  pedPnDiscrepancyMean_[1] = 100.0;
  
  pedPnRMSThreshold_[0] = 1.0; // value at h4; expected nominal: 0.5
  pedPnRMSThreshold_[1] = 3.0; // value at h4; expected nominal: 1.6
  
}

EELedClient::~EELedClient(){

}

void EELedClient::beginJob(MonitorUserInterface* mui){

  mui_ = mui;

  if ( verbose_ ) cout << "EELedClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

  if ( enableQT_ ) {

    Char_t qtname[200];

    for ( unsigned int i=0; i<superModules_.size(); i++ ) {

      int ism = superModules_[i];

      sprintf(qtname, "EELDT led quality %s A", Numbers::sEE(ism).c_str());
      qth01_[ism-1] = dynamic_cast<MEContentsProf2DWithinRangeROOT*> (mui_->createQTest(ContentsProf2DWithinRangeROOT::getAlgoName(), qtname));

      sprintf(qtname, "EELDT led quality %s B", Numbers::sEE(ism).c_str());
      qth05_[ism-1] = dynamic_cast<MEContentsProf2DWithinRangeROOT*> (mui_->createQTest(ContentsProf2DWithinRangeROOT::getAlgoName(), qtname));

      sprintf(qtname, "EELDT led amplitude quality PNs %s G01", Numbers::sEE(ism).c_str());
      qth09_[ism-1] = dynamic_cast<MEContentsProf2DWithinRangeROOT*> (mui_->createQTest(ContentsProf2DWithinRangeROOT::getAlgoName(), qtname));

      sprintf(qtname, "EELDT led pedestal quality PNs %s G01", Numbers::sEE(ism).c_str());
      qth13_[ism-1] = dynamic_cast<MEContentsProf2DWithinRangeROOT*> (mui_->createQTest(ContentsProf2DWithinRangeROOT::getAlgoName(), qtname));

      sprintf(qtname, "EELDT led amplitude quality PNs %s G16", Numbers::sEE(ism).c_str());
      qth17_[ism-1] = dynamic_cast<MEContentsProf2DWithinRangeROOT*> (mui_->createQTest(ContentsProf2DWithinRangeROOT::getAlgoName(), qtname));

      sprintf(qtname, "EELDT led pedestal quality PNs %s G16", Numbers::sEE(ism).c_str());
      qth21_[ism-1] = dynamic_cast<MEContentsProf2DWithinRangeROOT*> (mui_->createQTest(ContentsProf2DWithinRangeROOT::getAlgoName(), qtname));

      qth01_[ism-1]->setMeanRange(100.0, 4096.0*12.);

      qth05_[ism-1]->setMeanRange(100.0, 4096.0*12.);

      qth09_[ism-1]->setMeanRange(amplitudeThresholdPnG01_, 4096.0);

      qth13_[ism-1]->setMeanRange(pedPnExpectedMean_[0] - pedPnDiscrepancyMean_[0],
				  pedPnExpectedMean_[0] + pedPnDiscrepancyMean_[0]);

      qth17_[ism-1]->setMeanRange(amplitudeThresholdPnG16_, 4096.0);

      qth21_[ism-1]->setMeanRange(pedPnExpectedMean_[1] - pedPnDiscrepancyMean_[1],
				  pedPnExpectedMean_[1] + pedPnDiscrepancyMean_[1]);

      qth01_[ism-1]->setMeanTolerance(percentVariation_);

      qth05_[ism-1]->setMeanTolerance(percentVariation_);

      qth09_[ism-1]->setRMSRange(0.0, 4096.0);

      qth13_[ism-1]->setRMSRange(0.0, pedPnRMSThreshold_[0]);

      qth17_[ism-1]->setRMSRange(0.0, 4096.0);

      qth21_[ism-1]->setRMSRange(0.0, pedPnRMSThreshold_[1]);

      qth01_[ism-1]->setMinimumEntries(10*1700);

      qth05_[ism-1]->setMinimumEntries(10*1700);

      qth09_[ism-1]->setMinimumEntries(10*10);

      qth13_[ism-1]->setMinimumEntries(10*10);

      qth17_[ism-1]->setMinimumEntries(10*10);

      qth21_[ism-1]->setMinimumEntries(10*10);

      qth01_[ism-1]->setErrorProb(1.00);

      qth05_[ism-1]->setErrorProb(1.00);

      qth09_[ism-1]->setErrorProb(1.00);

      qth13_[ism-1]->setErrorProb(1.00);

      qth17_[ism-1]->setErrorProb(1.00);

      qth21_[ism-1]->setErrorProb(1.00);

      sprintf(qtname, "EELDT quality test %s", Numbers::sEE(ism).c_str());
      qtg01_[ism-1] = dynamic_cast<MEContentsTH2FWithinRangeROOT*> (mui_->createQTest(ContentsTH2FWithinRangeROOT::getAlgoName(), qtname));

      qtg01_[ism-1]->setMeanRange(1., 6.);

      qtg01_[ism-1]->setErrorProb(1.00);

      sprintf(qtname, "EELDT quality test PNs %s G01", Numbers::sEE(ism).c_str());
      qtg05_[ism-1] = dynamic_cast<MEContentsTH2FWithinRangeROOT*> (mui_->createQTest(ContentsTH2FWithinRangeROOT::getAlgoName(), qtname));

      sprintf(qtname, "EELDT quality test PNs %s G16", Numbers::sEE(ism).c_str());
      qtg09_[ism-1] = dynamic_cast<MEContentsTH2FWithinRangeROOT*> (mui_->createQTest(ContentsTH2FWithinRangeROOT::getAlgoName(), qtname));

      qtg05_[ism-1]->setMeanRange(1., 6.);

      qtg09_[ism-1]->setMeanRange(1., 6.);

      qtg05_[ism-1]->setErrorProb(1.00);

      qtg09_[ism-1]->setErrorProb(1.00);

    }

  }

}

void EELedClient::beginRun(void){

  if ( verbose_ ) cout << "EELedClient: beginRun" << endl;

  jevt_ = 0;

  this->setup();

  this->subscribe();

}

void EELedClient::endJob(void) {

  if ( verbose_ ) cout << "EELedClient: endJob, ievt = " << ievt_ << endl;

  this->unsubscribe();

  this->cleanup();

}

void EELedClient::endRun(void) {

  if ( verbose_ ) cout << "EELedClient: endRun, jevt = " << jevt_ << endl;

  this->unsubscribe();

  this->cleanup();

}

void EELedClient::setup(void) {

  Char_t histo[200];

  mui_->setCurrentFolder( "EcalEndcap/EELedClient" );

  DaqMonitorBEInterface* dbe = mui_->getBEInterface();

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( meg01_[ism-1] ) dbe->removeElement( meg01_[ism-1]->getName() );
    sprintf(histo, "EELDT led quality %s", Numbers::sEE(ism).c_str());
    meg01_[ism-1] = dbe->book2D(histo, histo, 85, 0., 85., 20, 0., 20.);

    if ( meg05_[ism-1] ) dbe->removeElement( meg05_[ism-1]->getName() );
    sprintf(histo, "EELDT led quality PNs %s G01", Numbers::sEE(ism).c_str());
    meg05_[ism-1] = dbe->book2D(histo, histo, 10, 0., 10., 1, 0., 5.);

    if ( meg09_[ism-1] ) dbe->removeElement( meg09_[ism-1]->getName() );
    sprintf(histo, "EELDT led quality PNs %s G16", Numbers::sEE(ism).c_str());
    meg09_[ism-1] = dbe->book2D(histo, histo, 10, 0., 10., 1, 0., 5.);

    if ( mea01_[ism-1] ) dbe->removeElement( mea01_[ism-1]->getName() );;
    sprintf(histo, "EELDT amplitude A %s", Numbers::sEE(ism).c_str());
    mea01_[ism-1] = dbe->book1D(histo, histo, 1700, 0., 1700.);

    if ( mea05_[ism-1] ) dbe->removeElement( mea05_[ism-1]->getName() );;
    sprintf(histo, "EELDT amplitude B %s", Numbers::sEE(ism).c_str());
    mea05_[ism-1] = dbe->book1D(histo, histo, 1700, 0., 1700.);

    if ( met01_[ism-1] ) dbe->removeElement( met01_[ism-1]->getName() );
    sprintf(histo, "EELDT timing A %s", Numbers::sEE(ism).c_str());
    met01_[ism-1] = dbe->book1D(histo, histo, 1700, 0., 1700.);

    if ( met05_[ism-1] ) dbe->removeElement( met05_[ism-1]->getName() );
    sprintf(histo, "EELDT timing B %s", Numbers::sEE(ism).c_str());
    met05_[ism-1] = dbe->book1D(histo, histo, 1700, 0., 1700.);

    if ( metav01_[ism-1] ) dbe->removeElement( metav01_[ism-1]->getName() );
    sprintf(histo, "EELDT timing mean A %s", Numbers::sEE(ism).c_str());
    metav01_[ism-1] = dbe->book1D(histo, histo, 100, 0., 10.);

    if ( metav05_[ism-1] ) dbe->removeElement( metav05_[ism-1]->getName() );
    sprintf(histo, "EELDT timing mean B %s", Numbers::sEE(ism).c_str());
    metav05_[ism-1] = dbe->book1D(histo, histo, 100, 0., 10.);

    if ( metrms01_[ism-1] ) dbe->removeElement( metrms01_[ism-1]->getName() );
    sprintf(histo, "EELDT timing rms A %s", Numbers::sEE(ism).c_str());
    metrms01_[ism-1] = dbe->book1D(histo, histo, 100, 0., 0.5);

    if ( metrms05_[ism-1] ) dbe->removeElement( metrms05_[ism-1]->getName() );
    sprintf(histo, "EELDT timing rms B %s", Numbers::sEE(ism).c_str());
    metrms05_[ism-1] = dbe->book1D(histo, histo, 100, 0., 0.5);

    if ( meaopn01_[ism-1] ) dbe->removeElement( meaopn01_[ism-1]->getName() );
    sprintf(histo, "EELDT amplitude over PN A %s", Numbers::sEE(ism).c_str());
    meaopn01_[ism-1] = dbe->book1D(histo, histo, 1700, 0., 1700.);

    if ( meaopn05_[ism-1] ) dbe->removeElement( meaopn05_[ism-1]->getName() );
    sprintf(histo, "EELDT amplitude over PN B %s", Numbers::sEE(ism).c_str());
    meaopn05_[ism-1] = dbe->book1D(histo, histo, 1700, 0., 1700.);

    if ( mepnprms01_[ism-1] ) dbe->removeElement( mepnprms01_[ism-1]->getName() );
    sprintf(histo, "EEPDT PNs pedestal rms %s G01", Numbers::sEE(ism).c_str());
    mepnprms01_[ism-1] = dbe->book1D(histo, histo, 100, 0., 10.);

    if ( mepnprms05_[ism-1] ) dbe->removeElement( mepnprms05_[ism-1]->getName() );
    sprintf(histo, "EEPDT PNs pedestal rms %s G16", Numbers::sEE(ism).c_str());
    mepnprms05_[ism-1] = dbe->book1D(histo, histo, 100, 0., 10.);

  }

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    UtilsClient::resetHisto( meg01_[ism-1] );

    UtilsClient::resetHisto( meg05_[ism-1] );

    UtilsClient::resetHisto( meg09_[ism-1] );

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

        meg01_[ism-1]->setBinContent( ie, ip, 2. );

      }
    }

    for ( int i = 1; i <= 10; i++ ) {

        meg05_[ism-1]->setBinContent( i, 1, 2. );

        meg09_[ism-1]->setBinContent( i, 1, 2. );

    }

    UtilsClient::resetHisto( mea01_[ism-1] );

    UtilsClient::resetHisto( mea05_[ism-1] );

    UtilsClient::resetHisto( met01_[ism-1] );

    UtilsClient::resetHisto( met05_[ism-1] );

    UtilsClient::resetHisto( metav01_[ism-1] );

    UtilsClient::resetHisto( metav05_[ism-1] );

    UtilsClient::resetHisto( metrms01_[ism-1] );

    UtilsClient::resetHisto( metrms05_[ism-1] );

    UtilsClient::resetHisto( meaopn01_[ism-1] );

    UtilsClient::resetHisto( meaopn05_[ism-1] );
    
    UtilsClient::resetHisto( mepnprms01_[ism-1] );

    UtilsClient::resetHisto( mepnprms05_[ism-1] );
    
  }

}

void EELedClient::cleanup(void) {

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( cloneME_ ) {
      if ( h01_[ism-1] ) delete h01_[ism-1];
      if ( h02_[ism-1] ) delete h02_[ism-1];

      if ( h09_[ism-1] ) delete h09_[ism-1];

      if ( h13_[ism-1] ) delete h13_[ism-1];
      if ( h14_[ism-1] ) delete h14_[ism-1];

      if ( h21_[ism-1] ) delete h21_[ism-1];

      if ( hs01_[ism-1] ) delete hs01_[ism-1];

      if ( hs05_[ism-1] ) delete hs05_[ism-1];

      if ( i01_[ism-1] ) delete i01_[ism-1];

      if ( i05_[ism-1] ) delete i05_[ism-1];

      if ( i09_[ism-1] ) delete i09_[ism-1];

      if ( i13_[ism-1] ) delete i13_[ism-1];
    }

    h01_[ism-1] = 0;
    h02_[ism-1] = 0;

    h09_[ism-1] = 0;

    h13_[ism-1] = 0;
    h14_[ism-1] = 0;

    h21_[ism-1] = 0;

    hs01_[ism-1] = 0;

    hs05_[ism-1] = 0;

    i01_[ism-1] = 0;

    i05_[ism-1] = 0;

    i09_[ism-1] = 0;

    i13_[ism-1] = 0;

  }

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    mui_->setCurrentFolder( "EcalEndcap/EELedClient" );
    DaqMonitorBEInterface* dbe = mui_->getBEInterface();

    if ( meg01_[ism-1] ) dbe->removeElement( meg01_[ism-1]->getName() );
    meg01_[ism-1] = 0;

    if ( meg05_[ism-1] ) dbe->removeElement( meg05_[ism-1]->getName() );
    meg05_[ism-1] = 0;

    if ( meg09_[ism-1] ) dbe->removeElement( meg09_[ism-1]->getName() );
    meg09_[ism-1] = 0;

    if ( mea01_[ism-1] ) dbe->removeElement( mea01_[ism-1]->getName() );
    mea01_[ism-1] = 0;

    if ( mea05_[ism-1] ) dbe->removeElement( mea05_[ism-1]->getName() );
    mea05_[ism-1] = 0;

    if ( met01_[ism-1] ) dbe->removeElement( met01_[ism-1]->getName() );
    met01_[ism-1] = 0;

    if ( met05_[ism-1] ) dbe->removeElement( met05_[ism-1]->getName() );
    met05_[ism-1] = 0;

    if ( metav01_[ism-1] ) dbe->removeElement( metav01_[ism-1]->getName() );
    metav01_[ism-1] = 0;

    if ( metav05_[ism-1] ) dbe->removeElement( metav05_[ism-1]->getName() );
    metav05_[ism-1] = 0;

    if ( metrms01_[ism-1] ) dbe->removeElement( metrms01_[ism-1]->getName() );
    metrms01_[ism-1] = 0;

    if ( metrms05_[ism-1] ) dbe->removeElement( metrms05_[ism-1]->getName() );
    metrms05_[ism-1] = 0;

    if ( meaopn01_[ism-1] ) dbe->removeElement( meaopn01_[ism-1]->getName() );
    meaopn01_[ism-1] = 0;

    if ( meaopn05_[ism-1] ) dbe->removeElement( meaopn05_[ism-1]->getName() );
    meaopn05_[ism-1] = 0;

    if ( mepnprms01_[ism-1] ) dbe->removeElement( mepnprms01_[ism-1]->getName() );
    mepnprms01_[ism-1] = 0;

    if ( mepnprms05_[ism-1] ) dbe->removeElement( mepnprms05_[ism-1]->getName() );
    mepnprms05_[ism-1] = 0;

  }

}

bool EELedClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov) {

  bool status = true;

  EcalLogicID ecid;

  MonLaserBlueDat apd_bl;
  map<EcalLogicID, MonLaserBlueDat> dataset1_bl;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    cout << " SM=" << ism << endl;

    UtilsClient::printBadChannels(qth01_[ism-1]);
    UtilsClient::printBadChannels(qth05_[ism-1]);

    UtilsClient::printBadChannels(qth09_[ism-1]);

    UtilsClient::printBadChannels(qth13_[ism-1]);

    UtilsClient::printBadChannels(qth17_[ism-1]);

    UtilsClient::printBadChannels(qth21_[ism-1]);

//    UtilsClient::printBadChannels(qtg01_[ism-1]);

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

        bool update01;
        bool update02;

        float num01, num02;
        float mean01, mean02;
        float rms01, rms02;

        update01 = UtilsClient::getBinStats(h01_[ism-1], ie, ip, num01, mean01, rms01);
        update02 = UtilsClient::getBinStats(h02_[ism-1], ie, ip, num02, mean02, rms02);

        if ( ! update01 )
          update01 = UtilsClient::getBinStats(h13_[ism-1], ie, ip, num01, mean01, rms01);
        if ( ! update02 )
          update02 = UtilsClient::getBinStats(h14_[ism-1], ie, ip, num02, mean02, rms02);

        if ( update01 || update02 ) {

          if ( ie == 1 && ip == 1 ) {

            cout << "Preparing dataset for SM=" << ism << endl;

            cout << "(" << ie << "," << ip << ") " << num01 << " " << mean01 << " " << rms01 << endl;

            cout << endl;

          }

          apd_bl.setAPDMean(mean01);
          apd_bl.setAPDRMS(rms01);

          apd_bl.setAPDOverPNMean(mean02);
          apd_bl.setAPDOverPNRMS(rms02);

          if ( meg01_[ism-1] && int(meg01_[ism-1]->getBinContent( ie, ip )) % 3 == 1. ) {
            apd_bl.setTaskStatus(true);
          } else {
            apd_bl.setTaskStatus(false);
          }

          status = status && UtilsClient::getBinQual(meg01_[ism-1], ie, ip);

          int ic = (ip-1) + 20*(ie-1) + 1;

          if ( econn ) {
            try {
              ecid = LogicID::getEcalLogicID("EB_crystal_number", Numbers::iSM(ism), ic);
              dataset1_bl[ecid] = apd_bl;
            } catch (runtime_error &e) {
              cerr << e.what() << endl;
            }
          }

        }

      }
    }

  }

  if ( econn ) {
    try {
      cout << "Inserting MonLedDat ... " << flush;
/// FIXME
///      if ( dataset1_bl.size() != 0 ) econn->insertDataSet(&dataset1_bl, moniov);
      cout << "done." << endl;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

  cout << endl;

  MonPNBlueDat pn_bl;
  map<EcalLogicID, MonPNBlueDat> dataset2_bl;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    cout << " SM=" << ism << endl;

//    UtilsClient::printBadChannels(qtg05_[ism-1]);

//    UtilsClient::printBadChannels(qtg09_[ism-1]);

    for ( int i = 1; i <= 10; i++ ) {

      bool update01;

      bool update05;

      bool update09;

      bool update13;

      float num01, num05;
      float num09, num13;
      float mean01, mean05;
      float mean09, mean13;
      float rms01, rms05;
      float rms09, rms13;

      update01 = UtilsClient::getBinStats(i01_[ism-1], 1, i, num01, mean01, rms01);

      update05 = UtilsClient::getBinStats(i05_[ism-1], 1, i, num05, mean05, rms05);

      update09 = UtilsClient::getBinStats(i09_[ism-1], 1, i, num09, mean09, rms09);

      update13 = UtilsClient::getBinStats(i13_[ism-1], 1, i, num13, mean13, rms13);

      if ( update01 || update05 || update09 || update13 ) {

        if ( i == 1 ) {

          cout << "Preparing dataset for SM=" << ism << endl;

          cout << "PNs (" << i << ") G01 " << num01  << " " << mean01 << " " << rms01  << endl;
          cout << "PNs (" << i << ") G16 " << num09  << " " << mean09 << " " << rms09  << endl;

          cout << endl;

        }

        pn_bl.setADCMeanG1(mean01);
        pn_bl.setADCRMSG1(rms01);

        pn_bl.setPedMeanG1(mean05);
        pn_bl.setPedRMSG1(rms05);

        pn_bl.setADCMeanG16(mean09);
        pn_bl.setADCRMSG16(rms09);

        pn_bl.setPedMeanG16(mean13);
        pn_bl.setPedRMSG16(rms13);

        if ( meg05_[ism-1] && int(meg05_[ism-1]->getBinContent( i, 1 )) % 3 == 1. ||
             meg09_[ism-1] && int(meg09_[ism-1]->getBinContent( i, 1 )) % 3 == 1. ) {
          pn_bl.setTaskStatus(true);
        } else {
          pn_bl.setTaskStatus(false);
        }

        status = status && ( UtilsClient::getBinQual(meg05_[ism-1], i, 1) ||
                             UtilsClient::getBinQual(meg09_[ism-1], i, 1) );

        if ( econn ) {
          try {
            ecid = LogicID::getEcalLogicID("EB_LM_PN", Numbers::iSM(ism), i-1);
            dataset2_bl[ecid] = pn_bl;
          } catch (runtime_error &e) {
            cerr << e.what() << endl;
          }
        }

      }

    }

  }

  if ( econn ) {
    try {
      cout << "Inserting MonPnDat ... " << flush;
/// FIXME
///      if ( dataset2_bl.size() != 0 ) econn->insertDataSet(&dataset2_bl, moniov);
      cout << "done." << endl;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

  return status;

}

void EELedClient::subscribe(void){

  if ( verbose_ ) cout << "EELedClient: subscribe" << endl;

  Char_t histo[200];

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    unsigned int ism = superModules_[i];

    sprintf(histo, "*/EcalEndcap/EELedTask/EELDT amplitude %s A", Numbers::sEE(ism).c_str());
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EELedTask/EELDT timing %s A", Numbers::sEE(ism).c_str());
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EELedTask/EELDT amplitude over PN %s A", Numbers::sEE(ism).c_str());
    mui_->subscribe(histo, ism);

    sprintf(histo, "*/EcalEndcap/EELedTask/EELDT amplitude %s B", Numbers::sEE(ism).c_str());
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EELedTask/EELDT timing %s B", Numbers::sEE(ism).c_str());
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EELedTask/EELDT amplitude over PN %s B", Numbers::sEE(ism).c_str());
    mui_->subscribe(histo, ism);

    sprintf(histo, "*/EcalEndcap/EELedTask/EELDT shape %s A", Numbers::sEE(ism).c_str());
    mui_->subscribe(histo, ism);

    sprintf(histo, "*/EcalEndcap/EELedTask/EELDT shape %s B", Numbers::sEE(ism).c_str());
    mui_->subscribe(histo, ism);

    sprintf(histo, "*/EcalEndcap/EELedTask/PN/Gain01/EEPDT PNs amplitude %s G01", Numbers::sEE(ism).c_str());
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EELedTask/PN/Gain01/EEPDT PNs pedestal %s G01", Numbers::sEE(ism).c_str());
    mui_->subscribe(histo, ism);

    sprintf(histo, "*/EcalEndcap/EELedTask/PN/Gain16/EEPDT PNs amplitude %s G16", Numbers::sEE(ism).c_str());
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EELedTask/PN/Gain16/EEPDT PNs pedestal %s G16", Numbers::sEE(ism).c_str());
    mui_->subscribe(histo, ism);

  }

  if ( collateSources_ ) {

    if ( verbose_ ) cout << "EELedClient: collate" << endl;

    for ( unsigned int i=0; i<superModules_.size(); i++ ) {

      int ism = superModules_[i];

      sprintf(histo, "EELDT amplitude %s A", Numbers::sEE(ism).c_str());
      me_h01_[ism-1] = mui_->collateProf2D(histo, histo, "EcalEndcap/Sums/EELedTask");
      sprintf(histo, "*/EcalEndcap/EELedTask/EELDT amplitude %s A", Numbers::sEE(ism).c_str());
      mui_->add(me_h01_[ism-1], histo);

      sprintf(histo, "EELDT amplitude over PN %s A", Numbers::sEE(ism).c_str());
      me_h02_[ism-1] = mui_->collateProf2D(histo, histo, "EcalEndcap/Sums/EELedTask");
      sprintf(histo, "*/EcalEndcap/EELedTask/EELDT amplitude over PN %s A", Numbers::sEE(ism).c_str());
      mui_->add(me_h02_[ism-1], histo);

      sprintf(histo, "EELDT timing %s A", Numbers::sEE(ism).c_str());
      me_h09_[ism-1] = mui_->collateProf2D(histo, histo, "EcalEndcap/Sums/EELedTask");
      sprintf(histo, "*/EcalEndcap/EELedTask/EELDT timing %s A", Numbers::sEE(ism).c_str());
      mui_->add(me_h09_[ism-1], histo);

      sprintf(histo, "EELDT amplitude %s B", Numbers::sEE(ism).c_str());
      me_h13_[ism-1] = mui_->collateProf2D(histo, histo, "EcalEndcap/Sums/EELedTask");
      sprintf(histo, "*/EcalEndcap/EELedTask/EELDT amplitude %s B", Numbers::sEE(ism).c_str());
      mui_->add(me_h13_[ism-1], histo);

      sprintf(histo, "EELDT amplitude over PN %s B", Numbers::sEE(ism).c_str());
      me_h14_[ism-1] = mui_->collateProf2D(histo, histo, "EcalEndcap/Sums/EELedTask");
      sprintf(histo, "*/EcalEndcap/EELedTask/EELDT amplitude over PN %s B", Numbers::sEE(ism).c_str());
      mui_->add(me_h14_[ism-1], histo);

      sprintf(histo, "EELDT timing %s B", Numbers::sEE(ism).c_str());
      me_h21_[ism-1] = mui_->collateProf2D(histo, histo, "EcalEndcap/Sums/EELedTask");
      sprintf(histo, "*/EcalEndcap/EELedTask/EELDT timing %s B", Numbers::sEE(ism).c_str());
      mui_->add(me_h21_[ism-1], histo);

      sprintf(histo, "EELDT shape %s A", Numbers::sEE(ism).c_str());
      me_hs01_[ism-1] = mui_->collateProf2D(histo, histo, "EcalEndcap/Sums/EELedTask");
      sprintf(histo, "*/EcalEndcap/EELedTask/EELDT shape %s A", Numbers::sEE(ism).c_str());
      mui_->add(me_hs01_[ism-1], histo);

      sprintf(histo, "EELDT shape %s B", Numbers::sEE(ism).c_str());
      me_hs05_[ism-1] = mui_->collateProf2D(histo, histo, "EcalEndcap/Sums/EELedTask");
      sprintf(histo, "*/EcalEndcap/EELedTask/EELDT shape %s B", Numbers::sEE(ism).c_str());
      mui_->add(me_hs05_[ism-1], histo);

      sprintf(histo, "EEPDT PNs amplitude %s G01", Numbers::sEE(ism).c_str());
      me_i01_[ism-1] = mui_->collateProf2D(histo, histo, "EcalEndcap/Sums/EELedTask/PN/Gain01");
      sprintf(histo, "*/EcalEndcap/EELedTask/PN/Gain01/EEPDT PNs amplitude %s G01", Numbers::sEE(ism).c_str());
      mui_->add(me_i01_[ism-1], histo);

      sprintf(histo, "EEPDT PNs pedestal %s G01", Numbers::sEE(ism).c_str());
      me_i05_[ism-1] = mui_->collateProf2D(histo, histo, "EcalEndcap/Sums/EELedTask/PN/Gain01");
      sprintf(histo, "*/EcalEndcap/EELedTask/PN/Gain01/EEPDT PNs pedestal %s G01", Numbers::sEE(ism).c_str());
      mui_->add(me_i05_[ism-1], histo);

      sprintf(histo, "EEPDT PNs amplitude %s G16", Numbers::sEE(ism).c_str());
      me_i09_[ism-1] = mui_->collateProf2D(histo, histo, "EcalEndcap/Sums/EELedTask/PN/Gain16");
      sprintf(histo, "*/EcalEndcap/EELedTask/PN/Gain16/EEPDT PNs amplitude %s G16", Numbers::sEE(ism).c_str());
      mui_->add(me_i09_[ism-1], histo);

      sprintf(histo, "EEPDT PNs pedestal %s G16", Numbers::sEE(ism).c_str());
      me_i13_[ism-1] = mui_->collateProf2D(histo, histo, "EcalEndcap/Sums/EELedTask/PN/Gain16");
      sprintf(histo, "*/EcalEndcap/EELedTask/PN/Gain16/EEPDT PNs pedestal %s G16", Numbers::sEE(ism).c_str());
      mui_->add(me_i13_[ism-1], histo);

    }

  }

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( collateSources_ ) {
      sprintf(histo, "EcalEndcap/Sums/EELedTask/EELDT amplitude %s A", Numbers::sEE(ism).c_str());
      if ( qth01_[ism-1] ) mui_->useQTest(histo, qth01_[ism-1]->getName());
      sprintf(histo, "EcalEndcap/Sums/EELedTask/EELDT amplitude %s B", Numbers::sEE(ism).c_str());
      if ( qth05_[ism-1] ) mui_->useQTest(histo, qth05_[ism-1]->getName());
      sprintf(histo, "EcalEndcap/Sums/EELedTask/PN/Gain01/EEPDT PNs amplitude %s G01", Numbers::sEE(ism).c_str());
      if ( qth09_[ism-1] ) mui_->useQTest(histo, qth09_[ism-1]->getName());
      sprintf(histo, "EcalEndcap/Sums/EELedTask/PN/Gain01/EEPDT PNs pedestal %s G01", Numbers::sEE(ism).c_str());
      if ( qth13_[ism-1] ) mui_->useQTest(histo, qth13_[ism-1]->getName());
      sprintf(histo, "EcalEndcap/Sums/EELedTask/PN/Gain16/EEPDT PNs amplitude %s G16", Numbers::sEE(ism).c_str());
      if ( qth17_[ism-1] ) mui_->useQTest(histo, qth17_[ism-1]->getName());
      sprintf(histo, "EcalEndcap/Sums/EELedTask/PN/Gain16/EEPDT PNs pedestal %s G16", Numbers::sEE(ism).c_str());
      if ( qth21_[ism-1] ) mui_->useQTest(histo, qth21_[ism-1]->getName());
    } else {
      if ( enableMonitorDaemon_ ) {
        sprintf(histo, "*/EcalEndcap/EELedTask/EELDT amplitude %s A", Numbers::sEE(ism).c_str());
        if ( qth01_[ism-1] ) mui_->useQTest(histo, qth01_[ism-1]->getName());
        sprintf(histo, "*/EcalEndcap/EELedTask/EELDT amplitude %s B", Numbers::sEE(ism).c_str());
        if ( qth05_[ism-1] ) mui_->useQTest(histo, qth05_[ism-1]->getName());
        sprintf(histo, "*/EcalEndcap/EELedTask/PN/Gain01/EEPDT PNs amplitude %s G01", Numbers::sEE(ism).c_str());
        if ( qth09_[ism-1] ) mui_->useQTest(histo, qth09_[ism-1]->getName());
        sprintf(histo, "*/EcalEndcap/EELedTask/PN/Gain01/EEPDT PNs pedestal %s G01", Numbers::sEE(ism).c_str());
        if ( qth13_[ism-1] ) mui_->useQTest(histo, qth13_[ism-1]->getName());
        sprintf(histo, "*/EcalEndcap/EELedTask/PN/Gain16/EEPDT PNs amplitude %s G16", Numbers::sEE(ism).c_str());
        if ( qth17_[ism-1] ) mui_->useQTest(histo, qth17_[ism-1]->getName());
        sprintf(histo, "*/EcalEndcap/EELedTask/PN/Gain16/EEPDT PNs pedestal %s G16", Numbers::sEE(ism).c_str());
        if ( qth21_[ism-1] ) mui_->useQTest(histo, qth21_[ism-1]->getName());
      } else {
        sprintf(histo, "EcalEndcap/EELedTask/EELDT amplitude %s A", Numbers::sEE(ism).c_str());
        if ( qth01_[ism-1] ) mui_->useQTest(histo, qth01_[ism-1]->getName());
        sprintf(histo, "EcalEndcap/EELedTask/EELDT amplitude %s B", Numbers::sEE(ism).c_str());
        if ( qth05_[ism-1] ) mui_->useQTest(histo, qth05_[ism-1]->getName());
        sprintf(histo, "EcalEndcap/EELedTask/PN/Gain01/EEPDT PNs amplitude %s G01", Numbers::sEE(ism).c_str());
        if ( qth09_[ism-1] ) mui_->useQTest(histo, qth09_[ism-1]->getName());
        sprintf(histo, "EcalEndcap/EELedTask/PN/Gain01/EEPDT PNs pedestal %s G01", Numbers::sEE(ism).c_str());
        if ( qth13_[ism-1] ) mui_->useQTest(histo, qth13_[ism-1]->getName());
        sprintf(histo, "EcalEndcap/EELedTask/PN/Gain16/EEPDT PNs amplitude %s G16", Numbers::sEE(ism).c_str());
        if ( qth17_[ism-1] ) mui_->useQTest(histo, qth17_[ism-1]->getName());
        sprintf(histo, "EcalEndcap/EELedTask/PN/Gain16/EEPDT PNs pedestal %s G16", Numbers::sEE(ism).c_str());
        if ( qth21_[ism-1] ) mui_->useQTest(histo, qth21_[ism-1]->getName());
      }
    }

  }

}

void EELedClient::subscribeNew(void){

  Char_t histo[200];

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    unsigned int ism = superModules_[i];

    sprintf(histo, "*/EcalEndcap/EELedTask/EELDT amplitude %s A", Numbers::sEE(ism).c_str());
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalEndcap/EELedTask/EELDT timing %s A", Numbers::sEE(ism).c_str());
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalEndcap/EELedTask/EELDT amplitude over PN %s A", Numbers::sEE(ism).c_str());
    mui_->subscribeNew(histo, ism);

    sprintf(histo, "*/EcalEndcap/EELedTask/EELDT amplitude %s B", Numbers::sEE(ism).c_str());
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalEndcap/EELedTask/EELDT timing %s B", Numbers::sEE(ism).c_str());
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalEndcap/EELedTask/EELDT amplitude over PN %s B", Numbers::sEE(ism).c_str());
    mui_->subscribeNew(histo, ism);

    sprintf(histo, "*/EcalEndcap/EELedTask/EELDT shape %s A", Numbers::sEE(ism).c_str());
    mui_->subscribeNew(histo, ism);

    sprintf(histo, "*/EcalEndcap/EELedTask/EELDT shape %s B", Numbers::sEE(ism).c_str());
    mui_->subscribeNew(histo, ism);

    sprintf(histo, "*/EcalEndcap/EELedTask/PN/Gain01/EEPDT PNs amplitude %s G01", Numbers::sEE(ism).c_str());
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalEndcap/EELedTask/PN/Gain01/EEPDT PNs pedestal %s G01", Numbers::sEE(ism).c_str());
    mui_->subscribeNew(histo, ism);

    sprintf(histo, "*/EcalEndcap/EELedTask/PN/Gain16/EEPDT PNs amplitude %s G16", Numbers::sEE(ism).c_str());
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalEndcap/EELedTask/PN/Gain16/EEPDT PNs pedestal %s G16", Numbers::sEE(ism).c_str());
    mui_->subscribeNew(histo, ism);

  }

}

void EELedClient::unsubscribe(void){

  if ( verbose_ ) cout << "EELedClient: unsubscribe" << endl;

  if ( collateSources_ ) {

    if ( verbose_ ) cout << "EELedClient: uncollate" << endl;

    if ( mui_ ) {

      for ( unsigned int i=0; i<superModules_.size(); i++ ) {

        int ism = superModules_[i];

        mui_->removeCollate(me_h01_[ism-1]);
        mui_->removeCollate(me_h02_[ism-1]);

        mui_->removeCollate(me_h09_[ism-1]);

        mui_->removeCollate(me_h13_[ism-1]);
        mui_->removeCollate(me_h14_[ism-1]);

        mui_->removeCollate(me_h21_[ism-1]);

        mui_->removeCollate(me_hs01_[ism-1]);

        mui_->removeCollate(me_hs05_[ism-1]);

        mui_->removeCollate(me_i01_[ism-1]);

        mui_->removeCollate(me_i05_[ism-1]);

        mui_->removeCollate(me_i09_[ism-1]);

        mui_->removeCollate(me_i13_[ism-1]);

      }

    }

  }

  Char_t histo[200];

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    unsigned int ism = superModules_[i];

    sprintf(histo, "*/EcalEndcap/EELedTask/EELDT amplitude %s A", Numbers::sEE(ism).c_str());
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EELedTask/EELDT timing %s A", Numbers::sEE(ism).c_str());
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EELedTask/EELDT amplitude over PN %s A", Numbers::sEE(ism).c_str());
    mui_->unsubscribe(histo, ism);

    sprintf(histo, "*/EcalEndcap/EELedTask/EELDT amplitude %s B", Numbers::sEE(ism).c_str());
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EELedTask/EELDT timing %s B", Numbers::sEE(ism).c_str());
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EELedTask/EELDT amplitude over PN %s B", Numbers::sEE(ism).c_str());
    mui_->unsubscribe(histo, ism);

    sprintf(histo, "*/EcalEndcap/EELedTask/EELDT shape %s A", Numbers::sEE(ism).c_str());
    mui_->unsubscribe(histo, ism);

    sprintf(histo, "*/EcalEndcap/EELedTask/EELDT shape %s B", Numbers::sEE(ism).c_str());
    mui_->unsubscribe(histo, ism);

    sprintf(histo, "*/EcalEndcap/EELedTask/PN/Gain01/EEPDT PNs amplitude %s G01", Numbers::sEE(ism).c_str());
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EELedTask/PN/Gain01/EEPDT PNs pedestal %s G01", Numbers::sEE(ism).c_str());
    mui_->unsubscribe(histo, ism);

    sprintf(histo, "*/EcalEndcap/EELedTask/PN/Gain16/EEPDT PNs amplitude %s G16", Numbers::sEE(ism).c_str());
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EELedTask/PN/Gain16/EEPDT PNs pedestal %s G16", Numbers::sEE(ism).c_str());
    mui_->unsubscribe(histo, ism);

  }

}

void EELedClient::softReset(void){

}

void EELedClient::analyze(void){

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( verbose_ ) cout << "EELedClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;
  }

  uint64_t bits01 = 0;
  bits01 |= EcalErrorDictionary::getMask("LASER_MEAN_WARNING");
  bits01 |= EcalErrorDictionary::getMask("LASER_RMS_WARNING");
  bits01 |= EcalErrorDictionary::getMask("LASER_MEAN_OVER_PN_WARNING");
  bits01 |= EcalErrorDictionary::getMask("LASER_RMS_OVER_PN_WARNING");

  uint64_t bits02 = 0;
  bits02 |= EcalErrorDictionary::getMask("PEDESTAL_LOW_GAIN_MEAN_WARNING");
  bits02 |= EcalErrorDictionary::getMask("PEDESTAL_LOW_GAIN_RMS_WARNING");
  bits02 |= EcalErrorDictionary::getMask("PEDESTAL_LOW_GAIN_MEAN_ERROR");
  bits02 |= EcalErrorDictionary::getMask("PEDESTAL_LOW_GAIN_RMS_ERROR");

  uint64_t bits03 = 0;
  bits03 |= EcalErrorDictionary::getMask("PEDESTAL_MIDDLE_GAIN_MEAN_WARNING");
  bits03 |= EcalErrorDictionary::getMask("PEDESTAL_MIDDLE_GAIN_RMS_WARNING");
  bits03 |= EcalErrorDictionary::getMask("PEDESTAL_MIDDLE_GAIN_MEAN_ERROR");
  bits03 |= EcalErrorDictionary::getMask("PEDESTAL_MIDDLE_GAIN_RMS_ERROR");

  uint64_t bits04 = 0;
  bits04 |= EcalErrorDictionary::getMask("PEDESTAL_HIGH_GAIN_MEAN_WARNING");
  bits04 |= EcalErrorDictionary::getMask("PEDESTAL_HIGH_GAIN_RMS_WARNING");
  bits04 |= EcalErrorDictionary::getMask("PEDESTAL_HIGH_GAIN_MEAN_ERROR");
  bits04 |= EcalErrorDictionary::getMask("PEDESTAL_HIGH_GAIN_RMS_ERROR");

  map<EcalLogicID, RunCrystalErrorsDat> mask1;
  map<EcalLogicID, RunPNErrorsDat> mask2;

  EcalErrorMask::fetchDataSet(&mask1);
  EcalErrorMask::fetchDataSet(&mask2);

  Char_t histo[200];

  MonitorElement* me;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( collateSources_ ) {
      sprintf(histo, "EcalEndcap/Sums/EELedTask/EELDT amplitude %s A", Numbers::sEE(ism).c_str());
    } else {
      sprintf(histo, (prefixME_+"EcalEndcap/EELedTask/EELDT amplitude %s A").c_str(), Numbers::sEE(ism).c_str());
    }
    me = mui_->get(histo);
    h01_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h01_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalEndcap/Sums/EELedTask/EELDT amplitude over PN %s A", Numbers::sEE(ism).c_str());
    } else {
      sprintf(histo, (prefixME_+"EcalEndcap/EELedTask/EELDT amplitude over PN %s A").c_str(), Numbers::sEE(ism).c_str());
    }
    me = mui_->get(histo);
    h02_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h02_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalEndcap/Sums/EELedTask/EELDT timing %s A", Numbers::sEE(ism).c_str());
    } else {
      sprintf(histo, (prefixME_+"EcalEndcap/EELedTask/EELDT timing %s A").c_str(), Numbers::sEE(ism).c_str());
    }
    me = mui_->get(histo);
    h09_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h09_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalEndcap/Sums/EELedTask/EELDT amplitude %s B", Numbers::sEE(ism).c_str());
    } else {
      sprintf(histo, (prefixME_+"EcalEndcap/EELedTask/EELDT amplitude %s B").c_str(), Numbers::sEE(ism).c_str());
    }
    me = mui_->get(histo);
    h13_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h13_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalEndcap/Sums/EELedTask/EELDT amplitude over PN %s B", Numbers::sEE(ism).c_str());
    } else {
      sprintf(histo, (prefixME_+"EcalEndcap/EELedTask/EELDT amplitude over PN %s B").c_str(), Numbers::sEE(ism).c_str());
    }
    me = mui_->get(histo);
    h14_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h14_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalEndcap/Sums/EELedTask/EELDT timing %s B", Numbers::sEE(ism).c_str());
    } else {
      sprintf(histo, (prefixME_+"EcalEndcap/EELedTask/EELDT timing %s B").c_str(), Numbers::sEE(ism).c_str());
    }
    me = mui_->get(histo);
    h21_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h21_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalEndcap/Sums/EELedTask/EELDT shape %s A", Numbers::sEE(ism).c_str());
    } else {
      sprintf(histo, (prefixME_+"EcalEndcap/EELedTask/EELDT shape %s A").c_str(), Numbers::sEE(ism).c_str());
    }
    me = mui_->get(histo);
    hs01_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, hs01_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalEndcap/Sums/EELedTask/EELDT shape %s B", Numbers::sEE(ism).c_str());
    } else {
      sprintf(histo, (prefixME_+"EcalEndcap/EELedTask/EELDT shape %s B").c_str(), Numbers::sEE(ism).c_str());
    }
    me = mui_->get(histo);
    hs05_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, hs05_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalEndcap/Sums/EELedTask/PN/Gain01/EEPDT PNs amplitude %s G01", Numbers::sEE(ism).c_str());
    } else {
      sprintf(histo, (prefixME_+"EcalEndcap/EELedTask/PN/Gain01/EEPDT PNs amplitude %s G01").c_str(), Numbers::sEE(ism).c_str());
    }
    me = mui_->get(histo);
    i01_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, i01_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalEndcap/Sums/EELedTask/PN/Gain01/EEPDT PNs pedestal %s G01", Numbers::sEE(ism).c_str());
    } else {
      sprintf(histo, (prefixME_+"EcalEndcap/EELedTask/PN/Gain01/EEPDT PNs pedestal %s G01").c_str(), Numbers::sEE(ism).c_str());
    }
    me = mui_->get(histo);
    i05_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, i05_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalEndcap/Sums/EELedTask/PN/Gain16/EEPDT PNs amplitude %s G16", Numbers::sEE(ism).c_str());
    } else {
      sprintf(histo, (prefixME_+"EcalEndcap/EELedTask/PN/Gain16/EEPDT PNs amplitude %s G16").c_str(), Numbers::sEE(ism).c_str());
    }
    me = mui_->get(histo);
    i09_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, i09_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalEndcap/Sums/EELedTask/PN/Gain16/EEPDT PNs pedestal %s G16", Numbers::sEE(ism).c_str());
    } else {
      sprintf(histo, (prefixME_+"EcalEndcap/EELedTask/PN/Gain16/EEPDT PNs pedestal %s G16").c_str(), Numbers::sEE(ism).c_str());
    }
    me = mui_->get(histo);
    i13_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, i13_[ism-1] );

    UtilsClient::resetHisto( meg01_[ism-1] );

    UtilsClient::resetHisto( meg05_[ism-1] );

    UtilsClient::resetHisto( meg09_[ism-1] );

    UtilsClient::resetHisto( mea01_[ism-1] );

    UtilsClient::resetHisto( mea05_[ism-1] );

    UtilsClient::resetHisto( met01_[ism-1] );

    UtilsClient::resetHisto( met05_[ism-1] );

    UtilsClient::resetHisto( metav01_[ism-1] );

    UtilsClient::resetHisto( metav05_[ism-1] );

    UtilsClient::resetHisto( metrms01_[ism-1] );

    UtilsClient::resetHisto( metrms05_[ism-1] );

    UtilsClient::resetHisto( meaopn01_[ism-1] );

    UtilsClient::resetHisto( meaopn05_[ism-1] );

    UtilsClient::resetHisto( mepnprms01_[ism-1] );

    UtilsClient::resetHisto( mepnprms05_[ism-1] );
    
    float meanAmplA;
    float meanAmplB;

    int nCryA;
    int nCryB;

    meanAmplA = 0.;
    meanAmplB = 0.;

    nCryA = 0;
    nCryB = 0;

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

        bool update01;

        bool update05;

        float num01, num05;
        float mean01, mean05;
        float rms01, rms05;

        update01 = UtilsClient::getBinStats(h01_[ism-1], ie, ip, num01, mean01, rms01);
        update05 = UtilsClient::getBinStats(h13_[ism-1], ie, ip, num05, mean05, rms05);

        if ( update01 ) {
          meanAmplA += mean01;
          nCryA++;
        }

        if ( update05 ) {
          meanAmplB += mean05;
          nCryB++;
        }

      }
    }

    if ( nCryA > 0 ) meanAmplA /= float (nCryA);
    if ( nCryB > 0 ) meanAmplB /= float (nCryB);

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

        if ( meg01_[ism-1] ) meg01_[ism-1]->setBinContent( ie, ip, 2.);

        bool update01;
        bool update02;

        bool update09;

        float num01, num02;
        float num09;
        float mean01, mean02;
        float mean09;
        float rms01, rms02;
        float rms09;

        update01 = UtilsClient::getBinStats(h01_[ism-1], ie, ip, num01, mean01, rms01);
        update02 = UtilsClient::getBinStats(h02_[ism-1], ie, ip, num02, mean02, rms02);

        update09 = UtilsClient::getBinStats(h09_[ism-1], ie, ip, num09, mean09, rms09);
	
         // other SM half
	
        if ( ! update01 )
          update01 = UtilsClient::getBinStats(h13_[ism-1], ie, ip, num01, mean01, rms01);
        if ( ! update02 )
          update02 = UtilsClient::getBinStats(h14_[ism-1], ie, ip, num02, mean02, rms02);
        if ( ! update09 )
          update09 = UtilsClient::getBinStats(h21_[ism-1], ie, ip, num09, mean09, rms09);

        if ( update01 ) {

          float val;

          if ( ie < 6 || ip > 10 ) {

            val = 1.;
            if ( fabs(mean01 - meanAmplA) > fabs(percentVariation_ * meanAmplA) )
              val = 0.;
            if ( meg01_[ism-1] ) meg01_[ism-1]->setBinContent( ie, ip, val );

            if ( mea01_[ism-1] ) {
              if ( mean01 > 0. ) {
                mea01_[ism-1]->setBinContent( ip+20*(ie-1), mean01 );
                mea01_[ism-1]->setBinError( ip+20*(ie-1), rms01 );
              } else {
                mea01_[ism-1]->setEntries( 1.+mea01_[ism-1]->getEntries() );
              }
            }

          } else {

            val = 1.;
            if ( fabs(mean01 - meanAmplB) > fabs(percentVariation_ * meanAmplB) )
              val = 0.;
            if ( meg01_[ism-1] ) meg01_[ism-1]->setBinContent( ie, ip, val );

            if ( mea05_[ism-1] ) {
              if ( mean01 > 0. ) {
                mea05_[ism-1]->setBinContent( ip+20*(ie-1), mean01 );
                mea05_[ism-1]->setBinError( ip+20*(ie-1), rms01 );
              } else {
                mea05_[ism-1]->setEntries( 1.+mea05_[ism-1]->getEntries() );
              }
            }

          }

        }

        if ( update02 ) {

          if ( ie < 6 || ip > 10 ) {

            if ( meaopn01_[ism-1] ) {
              if ( mean02 > 0. ) {
                meaopn01_[ism-1]->setBinContent( ip+20*(ie-1), mean02 );
                meaopn01_[ism-1]->setBinError( ip+20*(ie-1), rms02 );
              } else {
                meaopn01_[ism-1]->setEntries( 1.+meaopn01_[ism-1]->getEntries() );
              }
            }

          } else {


            if ( meaopn05_[ism-1] ) {
              if ( mean02 > 0. ) {
                meaopn05_[ism-1]->setBinContent( ip+20*(ie-1), mean02 );
                meaopn05_[ism-1]->setBinError( ip+20*(ie-1), rms02 );
              } else {
                meaopn05_[ism-1]->setEntries( 1.+meaopn05_[ism-1]->getEntries() );
              }
            }

          }

        }

        if ( update09 ) {

          if ( ie < 6 || ip > 10 ) {

            if ( met01_[ism-1] ) {
              if ( mean09 > 0. ) {
                met01_[ism-1]->setBinContent( ip+20*(ie-1), mean09 );
                met01_[ism-1]->setBinError( ip+20*(ie-1), rms09 );
              } else {
                met01_[ism-1]->setEntries(1.+met01_[ism-1]->getEntries());
              }
            }

            if ( metav01_[ism-1] )
              metav01_[ism-1] ->Fill(mean09);
            if ( metrms01_[ism-1] )
              metrms01_[ism-1]->Fill(rms09);

          } else {

            if ( met05_[ism-1] ) {
              if ( mean09 > 0. ) {
                met05_[ism-1]->setBinContent( ip+20*(ie-1), mean09 );
                met05_[ism-1]->setBinError( ip+20*(ie-1), rms09 );
              } else {
                met05_[ism-1]->setEntries(1.+met05_[ism-1]->getEntries());
              }

            if ( metav05_[ism-1] )
              metav05_[ism-1] ->Fill(mean09);
            if ( metrms05_[ism-1] )
              metrms05_[ism-1]->Fill(rms09);

            }

          }

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
            }

          }
        }

      }
    }

    for ( int i = 1; i <= 10; i++ ) {

      if ( meg05_[ism-1] ) meg05_[ism-1]->setBinContent( i, 1, 2. );

      if ( meg09_[ism-1] ) meg09_[ism-1]->setBinContent( i, 1, 2. );

      bool update01;

      bool update05;

      bool update09;

      bool update13;

      float num01, num05;
      float num09, num13;
      float mean01, mean05;
      float mean09, mean13;
      float rms01, rms05;
      float rms09, rms13;

      update01 = UtilsClient::getBinStats(i01_[ism-1], 1, i, num01, mean01, rms01);

      update05 = UtilsClient::getBinStats(i05_[ism-1], 1, i, num05, mean05, rms05);

      update09 = UtilsClient::getBinStats(i09_[ism-1], 1, i, num09, mean09, rms09);

      update13 = UtilsClient::getBinStats(i13_[ism-1], 1, i, num13, mean13, rms13);

      if ( update01 && update05 ) {

        float val;

        val = 1.;
        if ( mean01 < amplitudeThresholdPnG01_ )
          val = 0.;
        if ( mean05 <  pedPnExpectedMean_[0] - pedPnDiscrepancyMean_[0] ||
	     pedPnExpectedMean_[0] + pedPnDiscrepancyMean_[0] < mean05)
          val = 0.;
        if ( rms05 > pedPnRMSThreshold_[0] )
          val = 0.;

        if ( meg05_[ism-1] ) meg05_[ism-1]->setBinContent(i, 1, val);
        if ( mepnprms01_[ism-1] ) mepnprms01_[ism-1]->Fill(rms05);

      }

      if ( update09 && update13 ) {

        float val;

        val = 1.;
        if ( mean09 < amplitudeThresholdPnG16_ )
          val = 0.;
        if ( mean13 <  pedPnExpectedMean_[1] - pedPnDiscrepancyMean_[1] ||
	     pedPnExpectedMean_[1] + pedPnDiscrepancyMean_[1] < mean13)
          val = 0.;
        if ( rms13 > pedPnRMSThreshold_[1] )
          val = 0.;
	
        if ( meg09_[ism-1] )           meg09_[ism-1]->setBinContent(i, 1, val);
        if ( mepnprms05_[ism-1] ) mepnprms05_[ism-1]->Fill(rms13);

      }

      // masking

      if ( mask2.size() != 0 ) {
        map<EcalLogicID, RunPNErrorsDat>::const_iterator m;
        for (m = mask2.begin(); m != mask2.end(); m++) {

          EcalLogicID ecid = m->first;

          if ( ecid.getID1() == Numbers::iSM(ism) && ecid.getID2() == i-1 ) {
            if ( (m->second).getErrorBits() & (bits01|bits02) ) {
              if ( meg05_[ism-1] ) {
                float val = int(meg05_[ism-1]->getBinContent(i, 1)) % 3;
                meg05_[ism-1]->setBinContent( i, 1, val+3 );
              }
            }
            if ( (m->second).getErrorBits() & (bits01|bits04) ) {
              if ( meg09_[ism-1] ) {
                float val = int(meg09_[ism-1]->getBinContent(i, 1)) % 3;
                meg09_[ism-1]->setBinContent( i, 1, val+3 );
              }
            }
          }

        }
      }

    }

    vector<dqm::me_util::Channel> badChannels01;

    vector<dqm::me_util::Channel> badChannels05;

    if ( qth01_[ism-1] ) badChannels01 = qth01_[ism-1]->getBadChannels();

    if ( qth05_[ism-1] ) badChannels05 = qth05_[ism-1]->getBadChannels();

    vector<dqm::me_util::Channel> badChannels09;

    vector<dqm::me_util::Channel> badChannels13;

    vector<dqm::me_util::Channel> badChannels17;

    vector<dqm::me_util::Channel> badChannels21;

    if ( qth09_[ism-1] ) badChannels09 = qth09_[ism-1]->getBadChannels();

    if ( qth13_[ism-1] ) badChannels13 = qth13_[ism-1]->getBadChannels();

    if ( qth17_[ism-1] ) badChannels09 = qth17_[ism-1]->getBadChannels();

    if ( qth21_[ism-1] ) badChannels13 = qth21_[ism-1]->getBadChannels();

  }

}

void EELedClient::htmlOutput(int run, string htmlDir, string htmlName){

  cout << "Preparing EELedClient html output ..." << endl;

  ofstream htmlFile;

  htmlFile.open((htmlDir + htmlName).c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlFile << "<html>  " << endl;
  htmlFile << "<head>  " << endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlFile << " http-equiv=\"content-type\">  " << endl;
  htmlFile << "  <title>Monitor:LedTask output</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlFile << "<body>  " << endl;
  //htmlFile << "<br>  " << endl;
  htmlFile << "<a name=""top""></a>" << endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << run << "</span></h2>" << endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">LED</span></h2> " << endl;
  htmlFile << "<hr>" << endl;
  htmlFile << "<table border=1><tr><td bgcolor=red>channel has problems in this task</td>" << endl;
  htmlFile << "<td bgcolor=lime>channel has NO problems</td>" << endl;
  htmlFile << "<td bgcolor=yellow>channel is missing</td></table>" << endl;
  htmlFile << "<hr>" << endl;
  htmlFile << "<table style=\"width: 600px;\" border=\"0\">" << endl;
  htmlFile << "<tbody>" << endl;
  htmlFile << "<tr>" << endl;
  htmlFile << "<td align=\"left\">" << endl;
  htmlFile << "<div style=\"text-align: center;\"> </div>" << endl;
  htmlFile << "<table style=\"width: 255px; height: 35px;\" border=\"1\">" << endl;
  htmlFile << "<tbody>" << endl;
  htmlFile << "<tr>" << endl;
  htmlFile << "<td style=\"text-align: center;\">A=L-shaped half</td>" << endl;
  htmlFile << "<td style=\"vertical-align: top; text-align: center;\">B=notL-shaped half<br>" << endl;
  htmlFile << "</td>" << endl;
  htmlFile << "</tr>" << endl;
  htmlFile << "</tbody>" << endl;
  htmlFile << "</table>" << endl;
  htmlFile << "</td>" << endl;
  htmlFile << "</tr>" << endl;
  htmlFile << "</tbody>" << endl;
  htmlFile << "</table>" << endl;
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
  
  string imgNameQual[8], imgNameAmp[8], imgNameTim[8], imgNameTimav[8], imgNameTimrms[8], imgNameShape[8], imgNameAmpoPN[8], imgNameMEPnQualG01[8], imgNameMEPnG01[8], imgNameMEPnPedG01[8], imgNameMEPnRmsPedG01[8], imgNameMEPnQualG16[8], imgNameMEPnG16[8], imgNameMEPnPedG16[8], imgNameMEPnRmsPedG16[8], imgName, meName;

  TCanvas* cQual   = new TCanvas("cQual", "Temp", 2*csize, csize);
  TCanvas* cAmp    = new TCanvas("cAmp", "Temp", csize, csize);
  TCanvas* cTim    = new TCanvas("cTim", "Temp", csize, csize);
  TCanvas* cTimav  = new TCanvas("cTimav", "Temp", csize, csize);
  TCanvas* cTimrms = new TCanvas("cTimrms", "Temp", csize, csize);
  TCanvas* cShape  = new TCanvas("cShape", "Temp", csize, csize);
  TCanvas* cAmpoPN = new TCanvas("cAmpoPN", "Temp", csize, csize);
  TCanvas* cPed    = new TCanvas("cPed", "Temp", csize, csize);

  TH2F* obj2f;
  TH1F* obj1f;
  TH1D* obj1d;

  // Loop on barrel supermodules

  for ( unsigned int i=0; i<superModules_.size(); i ++ ) {

    int ism = superModules_[i];

    // Loop on 2 'sides'

    for ( int iCanvas = 1 ; iCanvas <= 4*2 ; iCanvas++ ) {

      // Quality plots

      imgNameQual[iCanvas-1] = "";

      obj2f = 0;
      switch ( iCanvas ) {
        case 1:
          obj2f = UtilsClient::getHisto<TH2F*>( meg01_[ism-1] );
          break;
        case 2:
        case 3:
        case 4:
        case 5:
        case 6:
        case 7:
        case 8:
          obj2f = 0;
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

      // Amplitude distributions

      imgNameAmp[iCanvas-1] = "";

      obj1f = 0;
      switch ( iCanvas ) {
        case 1:
          obj1f = UtilsClient::getHisto<TH1F*>( mea01_[ism-1] );
          break;
        case 2:
        case 3:
        case 4:
          obj2f = 0;
          break;
        case 5:
          obj1f = UtilsClient::getHisto<TH1F*>( mea05_[ism-1] );
          break;
        case 6:
        case 7:
        case 8:
          obj2f = 0;
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
        imgNameAmp[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgNameAmp[iCanvas-1];

        cAmp->cd();
        gStyle->SetOptStat("euo");
        obj1f->SetStats(kTRUE);
//        if ( obj1f->GetMaximum(histMax) > 0. ) {
//          gPad->SetLogy(1);
//        } else {
//          gPad->SetLogy(0);
//        }
        obj1f->SetMinimum(0.0);
        obj1f->Draw();
        cAmp->Update();
        cAmp->SaveAs(imgName.c_str());
        gPad->SetLogy(0);

      }

      // Timing distributions

      imgNameTim[iCanvas-1] = "";

      obj1f = 0;
      switch ( iCanvas ) {
        case 1:
          obj1f = UtilsClient::getHisto<TH1F*>( met01_[ism-1] );
          break;
        case 2:
        case 3:
        case 4:
          obj1f = 0;
          break;
        case 5:
          obj1f = UtilsClient::getHisto<TH1F*>( met05_[ism-1] );
          break;
        case 6:
        case 7:
        case 8:
          obj1f = 0;
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
        imgNameTim[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgNameTim[iCanvas-1];

        cTim->cd();
        gStyle->SetOptStat("euo");
        obj1f->SetStats(kTRUE);
        obj1f->SetMinimum(0.0);
        obj1f->SetMaximum(10.0);
        obj1f->Draw();
        cTim->Update();
        cTim->SaveAs(imgName.c_str());
        gPad->SetLogy(0);

      }

      // Timing mean distributions

      imgNameTimav[iCanvas-1] = "";

      obj1f = 0;
      switch ( iCanvas ) {
        case 1:
          obj1f = UtilsClient::getHisto<TH1F*>( metav01_[ism-1] );
          break;
        case 2:
        case 3:
        case 4:
          obj1f = 0;
          break;
        case 5:
          obj1f = UtilsClient::getHisto<TH1F*>( metav05_[ism-1] );
          break;
        case 6:
        case 7:
        case 8:
          obj1f = 0;
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
        imgNameTimav[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgNameTimav[iCanvas-1];

        cTimav->cd();
        gStyle->SetOptStat("euomr");
        obj1f->SetStats(kTRUE);
        if ( obj1f->GetMaximum(histMax) > 0. ) {
          gPad->SetLogy(1);
        } else {
          gPad->SetLogy(0);
        }
        obj1f->Draw();
        cTimav->Update();
        cTimav->SaveAs(imgName.c_str());
        gPad->SetLogy(0);

      }

      // Timing rms distributions

      imgNameTimrms[iCanvas-1] = "";

      obj1f = 0;
      switch ( iCanvas ) {
        case 1:
          obj1f = UtilsClient::getHisto<TH1F*>( metrms01_[ism-1] );
          break;
        case 2:
        case 3:
        case 4:
          obj1f = 0;
          break;
        case 5:
          obj1f = UtilsClient::getHisto<TH1F*>( metrms05_[ism-1] );
          break;
        case 6:
        case 7:
        case 8:
          obj1f = 0;
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
        imgNameTimrms[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgNameTimrms[iCanvas-1];

        cTimrms->cd();
        gStyle->SetOptStat("euomr");
        obj1f->SetStats(kTRUE);
        if ( obj1f->GetMaximum(histMax) > 0. ) {
          gPad->SetLogy(1);
        } else {
          gPad->SetLogy(0);
        }
        obj1f->Draw();
        cTimrms->Update();
        cTimrms->SaveAs(imgName.c_str());
        gPad->SetLogy(0);

      }

      // Shape distributions

      imgNameShape[iCanvas-1] = "";

      obj1d = 0;
      switch ( iCanvas ) {
        case 1:
          if ( hs01_[ism-1] ) obj1d = hs01_[ism-1]->ProjectionY("_py", 1, 1, "e");
          break;
        case 2:
        case 3:
        case 4:
          obj1d = 0;
          break;
        case 5:
          if ( hs05_[ism-1] ) obj1d = hs05_[ism-1]->ProjectionY("_py", 1681, 1681, "e");
          break;
        case 6:
        case 7:
        case 8:
          obj1d = 0;
          break;
        default:
          break;
      }

      if ( obj1d ) {
        meName = obj1d->GetName();

        for ( unsigned int i = 0; i < meName.size(); i++ ) {
          if ( meName.substr(i, 1) == " " )  {
            meName.replace(i, 1, "_");
          }
        }
        imgNameShape[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgNameShape[iCanvas-1];

        cShape->cd();
        gStyle->SetOptStat("euo");
        obj1d->SetStats(kTRUE);
//        if ( obj1d->GetMaximum(histMax) > 0. ) {
//          gPad->SetLogy(1);
//        } else {
//          gPad->SetLogy(0);
//        }
        obj1d->Draw();
        cShape->Update();
        cShape->SaveAs(imgName.c_str());
        gPad->SetLogy(0);

        delete obj1d;

      }

      // Amplitude over PN distributions

      imgNameAmpoPN[iCanvas-1] = "";

      obj1f = 0;
      switch ( iCanvas ) {
        case 1:
          obj1f = UtilsClient::getHisto<TH1F*>( meaopn01_[ism-1] );
          break;
        case 2:
        case 3:
        case 4:
          obj1f = 0;
          break;
        case 5:
          obj1f = UtilsClient::getHisto<TH1F*>( meaopn05_[ism-1] );
          break;
        case 6:
        case 7:
        case 8:
          obj1f = 0;
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
        imgNameAmpoPN[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgNameAmpoPN[iCanvas-1];

        cAmpoPN->cd();
        gStyle->SetOptStat("euo");
        obj1f->SetStats(kTRUE);
//        if ( obj1f->GetMaximum(histMax) > 0. ) {
//          gPad->SetLogy(1);
//        } else {
//          gPad->SetLogy(0);
//        }
        obj1f->SetMinimum(0.0);
        obj1f->SetMaximum(20.0);
        obj1f->Draw();
        cAmpoPN->Update();
        cAmpoPN->SaveAs(imgName.c_str());
        gPad->SetLogy(0);

      }

      // Monitoring elements plots

      imgNameMEPnQualG01[iCanvas-1] = "";

      obj2f = 0;
      switch ( iCanvas ) {
      case 1:
        obj2f = UtilsClient::getHisto<TH2F*>( meg05_[ism-1] );
        break;
      case 2:
      case 3:
      case 4:
      case 5:
      case 6:
      case 7:
      case 8:
        obj2f = 0;
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
        imgNameMEPnQualG01[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgNameMEPnQualG01[iCanvas-1];

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

      imgNameMEPnQualG16[iCanvas-1] = "";

      obj2f = 0;
      switch ( iCanvas ) {
      case 1:
        obj2f = UtilsClient::getHisto<TH2F*>( meg09_[ism-1] );
        break;
      case 2:
      case 3:
      case 4:
      case 5:
      case 6:
      case 7:
      case 8:
        obj2f = 0;
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
        imgNameMEPnQualG16[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgNameMEPnQualG16[iCanvas-1];

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

      imgNameMEPnG01[iCanvas-1] = "";

      obj1d = 0;
      switch ( iCanvas ) {
        case 1:
          if ( i01_[ism-1] ) obj1d = i01_[ism-1]->ProjectionY("_py", 1, 1, "e");
          break;
        case 2:
        case 3:
        case 4:
        case 5:
        case 6:
        case 7:
        case 8:
          obj2f = 0;
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
        imgNameMEPnG01[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgNameMEPnG01[iCanvas-1];

        cAmp->cd();
        gStyle->SetOptStat("euo");
        obj1d->SetStats(kTRUE);
//        if ( obj1d->GetMaximum(histMax) > 0. ) {
//          gPad->SetLogy(1);
//        } else {
//          gPad->SetLogy(0);
//        }
        obj1d->SetMinimum(0.0);
        obj1d->Draw();
        cAmp->Update();
        cAmp->SaveAs(imgName.c_str());
        gPad->SetLogy(0);

        delete obj1d;

      }

      imgNameMEPnG16[iCanvas-1] = "";

      obj1d = 0;
      switch ( iCanvas ) {
        case 1:
          if ( i09_[ism-1] ) obj1d = i09_[ism-1]->ProjectionY("_py", 1, 1, "e");
          break;
        case 2:
        case 3:
        case 4:
        case 5:
        case 6:
        case 7:
        case 8:
          obj2f = 0;
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
        imgNameMEPnG16[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgNameMEPnG16[iCanvas-1];

        cAmp->cd();
        gStyle->SetOptStat("euo");
        obj1d->SetStats(kTRUE);
//        if ( obj1d->GetMaximum(histMax) > 0. ) {
//          gPad->SetLogy(1);
//        } else {
//          gPad->SetLogy(0);
//        }
        obj1d->SetMinimum(0.0);
        obj1d->Draw();
        cAmp->Update();
        cAmp->SaveAs(imgName.c_str());
        gPad->SetLogy(0);

        delete obj1d;

      }

      // Monitoring elements plots

      imgNameMEPnPedG01[iCanvas-1] = "";

      obj1d = 0;
      switch ( iCanvas ) {
        case 1:
          if ( i05_[ism-1] ) obj1d = i05_[ism-1]->ProjectionY("_py", 1, 1, "e");
          break;
        case 2:
        case 3:
        case 4:
        case 5:
        case 6:
        case 7:
        case 8:
          obj2f = 0;
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
        imgNameMEPnPedG01[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgNameMEPnPedG01[iCanvas-1];

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
      
      
      imgNameMEPnRmsPedG01[iCanvas-1] = "";
      
      obj1f = 0;
      switch ( iCanvas ) {
      case 1:
	if ( mepnprms01_[ism-1] ) obj1f =  UtilsClient::getHisto<TH1F*>(mepnprms01_[ism-1]);
	break;
      case 2:
      case 3:
      case 4:
      case 5:
      case 6:
      case 7:
      case 8:
	obj2f = 0;
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
	imgNameMEPnRmsPedG01[iCanvas-1] = meName + ".png";
	imgName = htmlDir + imgNameMEPnRmsPedG01[iCanvas-1];
  	
	cPed->cd();
	gStyle->SetOptStat("euomr");
	obj1f->SetStats(kTRUE);
//        if ( obj1f->GetMaximum(histMax) > 0. ) {
//          gPad->SetLogy(1);
//        } else {
//          gPad->SetLogy(0);
//        }
	obj1f->SetMinimum(0.0);
	obj1f->Draw();
	cPed->Update();
	cPed->SaveAs(imgName.c_str());
	gPad->SetLogy(0);

      }
      
      
      imgNameMEPnRmsPedG16[iCanvas-1] = "";
      
      obj1f = 0;
      switch ( iCanvas ) {
      case 1:
	if ( mepnprms05_[ism-1] ) obj1f =  UtilsClient::getHisto<TH1F*>(mepnprms05_[ism-1]);
	break;
      case 2:
      case 3:
      case 4:
      case 5:
      case 6:
      case 7:
      case 8:
	obj2f = 0;
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
	imgNameMEPnRmsPedG16[iCanvas-1] = meName + ".png";
	imgName = htmlDir + imgNameMEPnRmsPedG16[iCanvas-1];
  	
	cPed->cd();
	gStyle->SetOptStat("euomr");
	obj1f->SetStats(kTRUE);
//        if ( obj1f->GetMaximum(histMax) > 0. ) {
//          gPad->SetLogy(1);
//        } else {
//          gPad->SetLogy(0);
//        }
	obj1f->SetMinimum(0.0);
	obj1f->Draw();
	cPed->Update();
	cPed->SaveAs(imgName.c_str());
	gPad->SetLogy(0);
  	
      }
      
      imgNameMEPnPedG16[iCanvas-1] = "";
      
      obj1d = 0;
      switch ( iCanvas ) {
      case 1:
	if ( i13_[ism-1] ) obj1d = i13_[ism-1]->ProjectionY("_py", 1, 1, "e");
	break;
      case 2:
      case 3:
      case 4:
      case 5:
      case 6:
      case 7:
      case 8:
	obj2f = 0;
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
        imgNameMEPnPedG16[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgNameMEPnPedG16[iCanvas-1];
	
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
      
    }
    
    if( i>0 ) htmlFile << "<a href=""#top"">Top</a>" << std::endl;
    htmlFile << "<hr>" << std::endl;
    htmlFile << "<h3><a name="""
	     << Numbers::sEE(ism).c_str() << """></a><strong>"
             << Numbers::sEE(ism).c_str() << "</strong></h3>" << endl;
    htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
    htmlFile << "<tr align=\"center\">" << endl;

    for ( int iCanvas = 1 ; iCanvas <= 1 ; iCanvas++ ) {

      if ( imgNameQual[iCanvas-1].size() != 0 )
        htmlFile << "<td colspan=\"2\"><img src=\"" << imgNameQual[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td colspan=\"2\"><img src=\"" << " " << "\"></td>" << endl;

    }

    htmlFile << "</tr>" << endl;

    htmlFile << "<tr>" << endl;

    for ( int iCanvas = 1 ; iCanvas <= 1 ; iCanvas++ ) {

      if ( imgNameAmp[iCanvas-1].size() != 0 )
        htmlFile << "<td><img src=\"" << imgNameAmp[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

      if ( imgNameAmpoPN[iCanvas-1].size() != 0 )
        htmlFile << "<td><img src=\"" << imgNameAmpoPN[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

    }

    htmlFile << "</tr>" << endl;

    htmlFile << "<tr>" << endl;

    for ( int iCanvas = 1 ; iCanvas <= 1 ; iCanvas++ ) {

      if ( imgNameAmp[4+iCanvas-1].size() != 0 )
        htmlFile << "<td><img src=\"" << imgNameAmp[4+iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

      if ( imgNameAmpoPN[4+iCanvas-1].size() != 0 )
        htmlFile << "<td><img src=\"" << imgNameAmpoPN[4+iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

    }

    htmlFile << "</tr>" << endl;

    htmlFile << "<tr align=\"center\">" << endl;

    for ( int iCanvas = 1 ; iCanvas <= 1 ; iCanvas++ ) {

      if ( imgNameTim[iCanvas-1].size() != 0 )
        htmlFile << "<td><img src=\"" << imgNameTim[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

      if ( imgNameShape[iCanvas-1].size() != 0 )
        htmlFile << "<td><img src=\"" << imgNameShape[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

    }

    htmlFile << "</tr>" << endl;

    htmlFile << "<tr align=\"center\">" << endl;

    for ( int iCanvas = 1 ; iCanvas <= 1 ; iCanvas++ ) {

      if ( imgNameTimav[iCanvas-1].size() != 0 )
        htmlFile << "<td><img src=\"" << imgNameTimav[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

      if ( imgNameTimrms[iCanvas-1].size() != 0 )
        htmlFile << "<td><img src=\"" << imgNameTimrms[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

    }

    htmlFile << "</tr>" << endl;

    htmlFile << "<tr align=\"center\">" << endl;

    for ( int iCanvas = 1 ; iCanvas <= 1 ; iCanvas++ ) {

      if ( imgNameTim[4+iCanvas-1].size() != 0 )
        htmlFile << "<td><img src=\"" << imgNameTim[4+iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

      if ( imgNameShape[4+iCanvas-1].size() != 0 )
        htmlFile << "<td><img src=\"" << imgNameShape[4+iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

    }

    htmlFile << "</tr>" << endl;

    htmlFile << "<tr align=\"center\">" << endl;

    for ( int iCanvas = 1 ; iCanvas <= 1 ; iCanvas++ ) {

      if ( imgNameTimav[4+iCanvas-1].size() != 0 )
        htmlFile << "<td><img src=\"" << imgNameTimav[4+iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

      if ( imgNameTimrms[4+iCanvas-1].size() != 0 )
        htmlFile << "<td><img src=\"" << imgNameTimrms[4+iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

    }

    htmlFile << "</tr>" << endl;

    htmlFile << "<tr align=\"center\">" << endl;

    htmlFile << "</tr>" << endl;
    htmlFile << "</table>" << endl;
    htmlFile << "<br>" << endl;

    htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
    htmlFile << "<tr align=\"center\">" << endl;

    for ( int iCanvas = 1 ; iCanvas <= 1 ; iCanvas++ ) {

      if ( imgNameMEPnQualG01[iCanvas-1].size() != 0 )
        htmlFile << "<td colspan=\"2\"><img src=\"" << imgNameMEPnQualG01[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td colspan=\"2\"><img src=\"" << " " << "\"></td>" << endl;

    }

    htmlFile << "</tr>" << endl;

    htmlFile << "<tr align=\"center\">" << endl;

    for ( int iCanvas = 1 ; iCanvas <= 1 ; iCanvas++ ) {

      if ( imgNameMEPnPedG01[iCanvas-1].size() != 0 )
        htmlFile << "<td><img src=\"" << imgNameMEPnPedG01[iCanvas-1] << "\"></td>" << endl;
      else
       htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

      if ( imgNameMEPnRmsPedG01[iCanvas-1].size() != 0 )
        htmlFile << "<td><img src=\"" << imgNameMEPnRmsPedG01[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

      if ( imgNameMEPnG01[iCanvas-1].size() != 0 )
        htmlFile << "<td><img src=\"" << imgNameMEPnG01[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

    }

    htmlFile << "</tr>" << endl;

    htmlFile << "<tr align=\"center\">" << endl;

    htmlFile << "<td colspan=\"1\"> </td> <td colspan=\"1\">PN Gain 1</td> <td colspan=\"1\">" << endl;

    htmlFile << "</tr>" << endl;

    htmlFile << "<tr align=\"center\">" << endl;

    for ( int iCanvas = 1 ; iCanvas <= 1 ; iCanvas++ ) {

      if ( imgNameMEPnQualG16[iCanvas-1].size() != 0 )
        htmlFile << "<td colspan=\"2\"><img src=\"" << imgNameMEPnQualG16[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td colspan=\"2\"><img src=\"" << " " << "\"></td>" << endl;

    }

    htmlFile << "</tr>" << endl;

    htmlFile << "<tr align=\"center\">" << endl;

    for ( int iCanvas = 1 ; iCanvas <= 1 ; iCanvas++ ) {

      if ( imgNameMEPnPedG16[iCanvas-1].size() != 0 )
        htmlFile << "<td><img src=\"" << imgNameMEPnPedG16[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

      if ( imgNameMEPnRmsPedG16[iCanvas-1].size() != 0 )
        htmlFile << "<td><img src=\"" << imgNameMEPnRmsPedG16[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

      if ( imgNameMEPnG16[iCanvas-1].size() != 0 )
        htmlFile << "<td><img src=\"" << imgNameMEPnG16[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

    }

    htmlFile << "<tr align=\"center\">" << endl;

    htmlFile << "<td colspan=\"1\"> </td> <td colspan=\"1\">PN Gain 16</td> <td colspan=\"1\"> </td>" << endl;

    htmlFile << "</tr>" << endl;

    htmlFile << "</tr>" << endl;
    htmlFile << "</table>" << endl;

    htmlFile << "<br>" << endl;

  }

  delete cQual;
  delete cAmp;
  delete cTim;
  delete cTimav;
  delete cTimrms;
  delete cShape;
  delete cAmpoPN;
  delete cPed;

  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  htmlFile.close();

}

