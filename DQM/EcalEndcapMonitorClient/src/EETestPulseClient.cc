/*
 * \file EETestPulseClient.cc
 *
 * $Date: 2007/05/22 09:53:49 $
 * $Revision: 1.10 $
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
#include "OnlineDB/EcalCondDB/interface/MonTestPulseDat.h"
#include "OnlineDB/EcalCondDB/interface/MonPulseShapeDat.h"
#include "OnlineDB/EcalCondDB/interface/MonPNMGPADat.h"
#include "OnlineDB/EcalCondDB/interface/RunCrystalErrorsDat.h"
#include "OnlineDB/EcalCondDB/interface/RunPNErrorsDat.h"

#include "CondTools/Ecal/interface/EcalErrorDictionary.h"

#include "DQM/EcalCommon/interface/EcalErrorMask.h"
#include <DQM/EcalCommon/interface/UtilsClient.h>
#include <DQM/EcalCommon/interface/LogicID.h>
#include <DQM/EcalCommon/interface/Numbers.h>

#include <DQM/EcalEndcapMonitorClient/interface/EETestPulseClient.h>

using namespace cms;
using namespace edm;
using namespace std;

EETestPulseClient::EETestPulseClient(const ParameterSet& ps){

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

    ha01_[ism-1] = 0;
    ha02_[ism-1] = 0;
    ha03_[ism-1] = 0;

    hs01_[ism-1] = 0;
    hs02_[ism-1] = 0;
    hs03_[ism-1] = 0;

    he01_[ism-1] = 0;
    he02_[ism-1] = 0;
    he03_[ism-1] = 0;

    i01_[ism-1] = 0;
    i02_[ism-1] = 0;
    i03_[ism-1] = 0;
    i04_[ism-1] = 0;

  }

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    meg01_[ism-1] = 0;
    meg02_[ism-1] = 0;
    meg03_[ism-1] = 0;

    meg04_[ism-1] = 0;
    meg05_[ism-1] = 0;

    mea01_[ism-1] = 0;
    mea02_[ism-1] = 0;
    mea03_[ism-1] = 0;

    qtha01_[ism-1] = 0;
    qtha02_[ism-1] = 0;
    qtha03_[ism-1] = 0;

    qtha04_[ism-1] = 0;
    qtha05_[ism-1] = 0;
    qtha06_[ism-1] = 0;
    qtha07_[ism-1] = 0;

  }

  amplitudeThreshold_ = 400.0;
  RMSThreshold_ = 300.0;
  threshold_on_AmplitudeErrorsNumber_ = 0.02;

  amplitudeThresholdPnG01_ = 200./16.;
  amplitudeThresholdPnG16_ = 200.;
  pedestalThresholdPn_ = 200.;

}

EETestPulseClient::~EETestPulseClient(){

}

void EETestPulseClient::beginJob(MonitorUserInterface* mui){

  mui_ = mui;

  if ( verbose_ ) cout << "EETestPulseClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

  if ( enableQT_ ) {

    Char_t qtname[200];

    for ( unsigned int i=0; i<superModules_.size(); i++ ) {

      int ism = superModules_[i];

      sprintf(qtname, "EETPT quality %s G01", Numbers::sEE(ism).c_str());
      qtha01_[ism-1] = dynamic_cast<MEContentsProf2DWithinRangeROOT*> (mui_->createQTest(ContentsProf2DWithinRangeROOT::getAlgoName(), qtname));

      sprintf(qtname, "EETPT quality %s G06", Numbers::sEE(ism).c_str());
      qtha02_[ism-1] = dynamic_cast<MEContentsProf2DWithinRangeROOT*> (mui_->createQTest(ContentsProf2DWithinRangeROOT::getAlgoName(), qtname));

      sprintf(qtname, "EETPT quality %s G12", Numbers::sEE(ism).c_str());
      qtha03_[ism-1] = dynamic_cast<MEContentsProf2DWithinRangeROOT*> (mui_->createQTest(ContentsProf2DWithinRangeROOT::getAlgoName(), qtname));

      sprintf(qtname, "EETPT amplitude quality PNs %s G01", Numbers::sEE(ism).c_str());
      qtha04_[ism-1] = dynamic_cast<MEContentsProf2DWithinRangeROOT*> (mui_->createQTest(ContentsProf2DWithinRangeROOT::getAlgoName(), qtname));

      sprintf(qtname, "EETPT amplitude quality PNs %s G16", Numbers::sEE(ism).c_str());
      qtha05_[ism-1] = dynamic_cast<MEContentsProf2DWithinRangeROOT*> (mui_->createQTest(ContentsProf2DWithinRangeROOT::getAlgoName(), qtname));

      sprintf(qtname, "EETPT pedestal quality PNs %s G01", Numbers::sEE(ism).c_str());
      qtha06_[ism-1] = dynamic_cast<MEContentsProf2DWithinRangeROOT*> (mui_->createQTest(ContentsProf2DWithinRangeROOT::getAlgoName(), qtname));

      sprintf(qtname, "EETPT pedestal quality PNs %s G16", Numbers::sEE(ism).c_str());
      qtha07_[ism-1] = dynamic_cast<MEContentsProf2DWithinRangeROOT*> (mui_->createQTest(ContentsProf2DWithinRangeROOT::getAlgoName(), qtname));

      qtha01_[ism-1]->setMeanRange(amplitudeThreshold_, 4096.0*12.);
      qtha02_[ism-1]->setMeanRange(amplitudeThreshold_, 4096.0*12.);
      qtha03_[ism-1]->setMeanRange(amplitudeThreshold_, 4096.0*12.);

      qtha04_[ism-1]->setMeanRange(amplitudeThresholdPnG01_, 4096.0);
      qtha05_[ism-1]->setMeanRange(amplitudeThresholdPnG16_, 4096.0);
      qtha06_[ism-1]->setMeanRange(pedestalThresholdPn_, 4096.0);
      qtha07_[ism-1]->setMeanRange(pedestalThresholdPn_, 4096.0);

      qtha01_[ism-1]->setRMSRange(0.0, RMSThreshold_);
      qtha02_[ism-1]->setRMSRange(0.0, RMSThreshold_);
      qtha03_[ism-1]->setRMSRange(0.0, RMSThreshold_);

      qtha04_[ism-1]->setRMSRange(0.0, 4096.0);
      qtha05_[ism-1]->setRMSRange(0.0, 4096.0);
      qtha06_[ism-1]->setRMSRange(0.0, 4096.0);
      qtha07_[ism-1]->setRMSRange(0.0, 4096.0);

      qtha01_[ism-1]->setMinimumEntries(10*1700);
      qtha02_[ism-1]->setMinimumEntries(10*1700);
      qtha03_[ism-1]->setMinimumEntries(10*1700);

      qtha04_[ism-1]->setMinimumEntries(10*10);
      qtha05_[ism-1]->setMinimumEntries(10*10);
      qtha06_[ism-1]->setMinimumEntries(10*10);
      qtha07_[ism-1]->setMinimumEntries(10*10);

      qtha01_[ism-1]->setErrorProb(1.00);
      qtha02_[ism-1]->setErrorProb(1.00);
      qtha03_[ism-1]->setErrorProb(1.00);

      qtha04_[ism-1]->setErrorProb(1.00);
      qtha05_[ism-1]->setErrorProb(1.00);
      qtha06_[ism-1]->setErrorProb(1.00);
      qtha07_[ism-1]->setErrorProb(1.00);

    }

  }

}

void EETestPulseClient::beginRun(void){

  if ( verbose_ ) cout << "EETestPulseClient: beginRun" << endl;

  jevt_ = 0;

  this->setup();

  this->subscribe();

}

void EETestPulseClient::endJob(void) {

  if ( verbose_ ) cout << "EETestPulseClient: endJob, ievt = " << ievt_ << endl;

  this->unsubscribe();

  this->cleanup();

}

void EETestPulseClient::endRun(void) {

  if ( verbose_ ) cout << "EETestPulseClient: endRun, jevt = " << jevt_ << endl;

  this->unsubscribe();

  this->cleanup();

}

void EETestPulseClient::setup(void) {

  Char_t histo[200];

  mui_->setCurrentFolder( "EcalEndcap/EETestPulseClient" );
  DaqMonitorBEInterface* bei = mui_->getBEInterface();

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( meg01_[ism-1] ) bei->removeElement( meg01_[ism-1]->getName() );
    sprintf(histo, "EETPT test pulse quality G01 %s", Numbers::sEE(ism).c_str());
    meg01_[ism-1] = bei->book2D(histo, histo, 85, 0., 85., 20, 0., 20.);
    if ( meg02_[ism-1] ) bei->removeElement( meg02_[ism-1]->getName() );
    sprintf(histo, "EETPT test pulse quality G06 %s", Numbers::sEE(ism).c_str());
    meg02_[ism-1] = bei->book2D(histo, histo, 85, 0., 85., 20, 0., 20.);
    if ( meg03_[ism-1] ) bei->removeElement( meg03_[ism-1]->getName() );
    sprintf(histo, "EETPT test pulse quality G12 %s", Numbers::sEE(ism).c_str());
    meg03_[ism-1] = bei->book2D(histo, histo, 85, 0., 85., 20, 0., 20.);

    if ( meg04_[ism-1] ) bei->removeElement( meg04_[ism-1]->getName() );
    sprintf(histo, "EETPT test pulse quality PNs G01 %s", Numbers::sEE(ism).c_str());
    meg04_[ism-1] = bei->book2D(histo, histo, 10, 0., 10., 1, 0., 5.);
    if ( meg05_[ism-1] ) bei->removeElement( meg05_[ism-1]->getName() );
    sprintf(histo, "EETPT test pulse quality PNs G16 %s", Numbers::sEE(ism).c_str());
    meg05_[ism-1] = bei->book2D(histo, histo, 10, 0., 10., 1, 0., 5.);

    if ( mea01_[ism-1] ) bei->removeElement( mea01_[ism-1]->getName() );
    sprintf(histo, "EETPT test pulse amplitude G01 %s", Numbers::sEE(ism).c_str());
    mea01_[ism-1] = bei->book1D(histo, histo, 1700, 0., 1700.);
    if ( mea02_[ism-1] ) bei->removeElement( mea02_[ism-1]->getName() );
    sprintf(histo, "EETPT test pulse amplitude G06 %s", Numbers::sEE(ism).c_str());
    mea02_[ism-1] = bei->book1D(histo, histo, 1700, 0., 1700.);
    if ( mea03_[ism-1] ) bei->removeElement( mea03_[ism-1]->getName() );
    sprintf(histo, "EETPT test pulse amplitude G12 %s", Numbers::sEE(ism).c_str());
    mea03_[ism-1] = bei->book1D(histo, histo, 1700, 0., 1700.);

  }

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    UtilsClient::resetHisto( meg01_[ism-1] );
    UtilsClient::resetHisto( meg02_[ism-1] );
    UtilsClient::resetHisto( meg03_[ism-1] );

    UtilsClient::resetHisto( meg04_[ism-1] );
    UtilsClient::resetHisto( meg05_[ism-1] );

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

    UtilsClient::resetHisto( mea01_[ism-1] );
    UtilsClient::resetHisto( mea02_[ism-1] );
    UtilsClient::resetHisto( mea03_[ism-1] );

  }

}

void EETestPulseClient::cleanup(void) {

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( cloneME_ ) {
      if ( ha01_[ism-1] ) delete ha01_[ism-1];
      if ( ha02_[ism-1] ) delete ha02_[ism-1];
      if ( ha03_[ism-1] ) delete ha03_[ism-1];

      if ( hs01_[ism-1] ) delete hs01_[ism-1];
      if ( hs02_[ism-1] ) delete hs02_[ism-1];
      if ( hs03_[ism-1] ) delete hs03_[ism-1];

      if ( he01_[ism-1] ) delete he01_[ism-1];
      if ( he02_[ism-1] ) delete he02_[ism-1];
      if ( he03_[ism-1] ) delete he03_[ism-1];

      if ( i01_[ism-1] ) delete i01_[ism-1];
      if ( i02_[ism-1] ) delete i02_[ism-1];
      if ( i03_[ism-1] ) delete i03_[ism-1];
      if ( i04_[ism-1] ) delete i04_[ism-1];
    }

    ha01_[ism-1] = 0;
    ha02_[ism-1] = 0;
    ha03_[ism-1] = 0;

    hs01_[ism-1] = 0;
    hs02_[ism-1] = 0;
    hs03_[ism-1] = 0;

    he01_[ism-1] = 0;
    he02_[ism-1] = 0;
    he03_[ism-1] = 0;

    i01_[ism-1] = 0;
    i02_[ism-1] = 0;
    i03_[ism-1] = 0;
    i04_[ism-1] = 0;

  }

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    mui_->setCurrentFolder( "EcalEndcap/EETestPulseClient" );
    DaqMonitorBEInterface* bei = mui_->getBEInterface();

    if ( meg01_[ism-1] ) bei->removeElement( meg01_[ism-1]->getName() );
    meg01_[ism-1] = 0;
    if ( meg02_[ism-1] ) bei->removeElement( meg02_[ism-1]->getName() );
    meg02_[ism-1] = 0;
    if ( meg03_[ism-1] ) bei->removeElement( meg03_[ism-1]->getName() );
    meg03_[ism-1] = 0;

    if ( meg04_[ism-1] ) bei->removeElement( meg04_[ism-1]->getName() );
    meg04_[ism-1] = 0;
    if ( meg05_[ism-1] ) bei->removeElement( meg05_[ism-1]->getName() );
    meg05_[ism-1] = 0;

    if ( mea01_[ism-1] ) bei->removeElement( mea01_[ism-1]->getName() );
    mea01_[ism-1] = 0;
    if ( mea02_[ism-1] ) bei->removeElement( mea02_[ism-1]->getName() );
    mea02_[ism-1] = 0;
    if ( mea03_[ism-1] ) bei->removeElement( mea03_[ism-1]->getName() );
    mea03_[ism-1] = 0;

  }

}

bool EETestPulseClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov, int ism) {

  bool status = true;

  UtilsClient::printBadChannels(qtha01_[ism-1]);
  UtilsClient::printBadChannels(qtha02_[ism-1]);
  UtilsClient::printBadChannels(qtha03_[ism-1]);

  UtilsClient::printBadChannels(qtha04_[ism-1]);
  UtilsClient::printBadChannels(qtha05_[ism-1]);
  UtilsClient::printBadChannels(qtha06_[ism-1]);
  UtilsClient::printBadChannels(qtha07_[ism-1]);

  EcalLogicID ecid;
  MonTestPulseDat adc;
  map<EcalLogicID, MonTestPulseDat> dataset1;
  MonPulseShapeDat shape;
  map<EcalLogicID, MonPulseShapeDat> dataset2;

  for ( int ie = 1; ie <= 85; ie++ ) {
    for ( int ip = 1; ip <= 20; ip++ ) {

      bool update01;
      bool update02;
      bool update03;

      float num01, num02, num03;
      float mean01, mean02, mean03;
      float rms01, rms02, rms03;

      update01 = UtilsClient::getBinStats(ha01_[ism-1], ie, ip, num01, mean01, rms01);
      update02 = UtilsClient::getBinStats(ha02_[ism-1], ie, ip, num02, mean02, rms02);
      update03 = UtilsClient::getBinStats(ha03_[ism-1], ie, ip, num03, mean03, rms03);

      if ( update01 || update02 || update03 ) {

        if ( ie == 1 && ip == 1 ) {

          cout << "Preparing dataset for SM=" << ism << endl;
          cout << "G01 (" << ie << "," << ip << ") " << num01 << " " << mean01 << " " << rms01 << endl;
          cout << "G06 (" << ie << "," << ip << ") " << num02 << " " << mean02 << " " << rms02 << endl;
          cout << "G12 (" << ie << "," << ip << ") " << num03 << " " << mean03 << " " << rms03 << endl;

          cout << endl;

        }

        adc.setADCMeanG1(mean01);
        adc.setADCRMSG1(rms01);

        adc.setADCMeanG6(mean02);
        adc.setADCRMSG6(rms02);

        adc.setADCMeanG12(mean03);
        adc.setADCRMSG12(rms03);

        if ( meg01_[ism-1] && int(meg01_[ism-1]->getBinContent( ie, ip )) % 3 == 1. &&
             meg02_[ism-1] && int(meg02_[ism-1]->getBinContent( ie, ip )) % 3 == 1. &&
             meg03_[ism-1] && int(meg03_[ism-1]->getBinContent( ie, ip )) % 3 == 1. ) {
          adc.setTaskStatus(true);
        } else {
          adc.setTaskStatus(false);
        }

        status = status && UtilsClient::getBinQual(meg01_[ism-1], ie, ip) &&
                           UtilsClient::getBinQual(meg02_[ism-1], ie, ip) &&
                           UtilsClient::getBinQual(meg03_[ism-1], ie, ip);

        if ( ie == 1 && ip == 1 ) {

          vector<float> sample01, sample02, sample03;

          sample01.clear();
          sample02.clear();
          sample03.clear();

          const float n_min_tot = 1000.;

          if ( hs01_[ism-1] && hs01_[ism-1]->GetEntries() >= n_min_tot ) {
            for ( int i = 1; i <= 10; i++ ) {
              sample01.push_back(int(hs01_[ism-1]->GetBinContent(1, i)));
            }
          } else {
            for ( int i = 1; i <= 10; i++ ) { sample01.push_back(-1.); }
          }

          if ( hs02_[ism-1] && hs02_[ism-1]->GetEntries() >= n_min_tot ) {
            for ( int i = 1; i <= 10; i++ ) {
              sample02.push_back(int(hs02_[ism-1]->GetBinContent(1, i)));
            }
          } else {
            for ( int i = 1; i <= 10; i++ ) { sample02.push_back(-1.); }
          }

          if ( hs03_[ism-1] && hs03_[ism-1]->GetEntries() >= n_min_tot ) {
            for ( int i = 1; i <= 10; i++ ) {
              sample03.push_back(int(hs03_[ism-1]->GetBinContent(1, i)));
            }
          } else {
            for ( int i = 1; i <= 10; i++ ) { sample03.push_back(-1.); }
          }

          cout << "sample01 = " << flush;
          for ( unsigned int i = 0; i < sample01.size(); i++ ) {
            cout << sample01[i] << " " << flush;
          }
          cout << endl;

          cout << "sample02 = " << flush;
          for ( unsigned int i = 0; i < sample02.size(); i++ ) {
            cout << sample02[i] << " " << flush;
          }
          cout << endl;

          cout << "sample03 = " << flush;
          for ( unsigned int i = 0; i < sample03.size(); i++ ) {
            cout << sample03[i] << " " << flush;
          }
          cout << endl;

          shape.setSamples(sample01,  1);
          shape.setSamples(sample02,  6);
          shape.setSamples(sample03, 12);

        }

        int ic = (ip-1) + 20*(ie-1) + 1;

        if ( econn ) {
          try {
            ecid = LogicID::getEcalLogicID("EB_crystal_number", Numbers::iSM(ism), ic);
            dataset1[ecid] = adc;
            if ( ie == 1 && ip == 1 ) dataset2[ecid] = shape;
          } catch (runtime_error &e) {
            cerr << e.what() << endl;
          }
        }

      }

    }
  }

  if ( econn ) {
    try {
      cout << "Inserting MonTestPulseDat ... " << flush;
      if ( dataset1.size() != 0 ) econn->insertDataSet(&dataset1, moniov);
      if ( dataset2.size() != 0 ) econn->insertDataSet(&dataset2, moniov);
      cout << "done." << endl;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

  MonPNMGPADat pn;
  map<EcalLogicID, MonPNMGPADat> dataset3;

  for ( int i = 1; i <= 10; i++ ) {

    bool update01;
    bool update02;
    bool update03;
    bool update04;

    float num01, num02, num03, num04;
    float mean01, mean02, mean03, mean04;
    float rms01, rms02, rms03, rms04;

    update01 = UtilsClient::getBinStats(i01_[ism-1], 1, i, num01, mean01, rms01);
    update02 = UtilsClient::getBinStats(i02_[ism-1], 1, i, num02, mean02, rms02);
    update03 = UtilsClient::getBinStats(i03_[ism-1], 1, i, num03, mean03, rms03);
    update04 = UtilsClient::getBinStats(i04_[ism-1], 1, i, num04, mean04, rms04);

    if ( update01 || update02 || update03 || update04 ) {

      if ( i == 1 ) {

        cout << "Preparing dataset for SM=" << ism << endl;

        cout << "PNs (" << i << ") G01 " << num01  << " " << mean01 << " " << rms01 << " " << num03 << " " << mean03 << " " << rms03 << endl;
        cout << "PNs (" << i << ") G16 " << num02  << " " << mean02 << " " << rms02 << " " << num04 << " " << mean04 << " " << rms04 << endl;

        cout << endl;

      }

      pn.setADCMeanG1(mean01);
      pn.setADCRMSG1(rms01);

      pn.setPedMeanG1(mean03);
      pn.setPedRMSG1(rms03);

      pn.setADCMeanG16(mean02);
      pn.setADCRMSG16(rms02);

      pn.setPedMeanG16(mean04);
      pn.setPedRMSG16(rms04);

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
          dataset3[ecid] = pn;
        } catch (runtime_error &e) {
          cerr << e.what() << endl;
        }
      }

    }

  }

  if ( econn ) {
    try {
      cout << "Inserting MonPNMGPADat ... " << flush;
      if ( dataset3.size() != 0 ) econn->insertDataSet(&dataset3, moniov);
      cout << "done." << endl;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

  return status;

}

void EETestPulseClient::subscribe(void){

  if ( verbose_ ) cout << "EETestPulseClient: subscribe" << endl;

  Char_t histo[200];

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    unsigned int ism = superModules_[i];

    sprintf(histo, "*/EcalEndcap/EETestPulseTask/Gain01/EETPT amplitude %s G01", Numbers::sEE(ism).c_str());
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EETestPulseTask/Gain01/EETPT shape %s G01", Numbers::sEE(ism).c_str());
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EETestPulseTask/Gain01/EETPT amplitude error %s G01", Numbers::sEE(ism).c_str());
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EETestPulseTask/Gain06/EETPT amplitude %s G06", Numbers::sEE(ism).c_str());
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EETestPulseTask/Gain06/EETPT shape %s G06", Numbers::sEE(ism).c_str());
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EETestPulseTask/Gain06/EETPT amplitude error %s G06", Numbers::sEE(ism).c_str());
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EETestPulseTask/Gain12/EETPT amplitude %s G12", Numbers::sEE(ism).c_str());
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EETestPulseTask/Gain12/EETPT shape %s G12", Numbers::sEE(ism).c_str());
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EETestPulseTask/Gain12/EETPT amplitude error %s G12", Numbers::sEE(ism).c_str());
    mui_->subscribe(histo, ism);

    sprintf(histo, "*/EcalEndcap/EEPnDiodeTask/Gain01/EEPDT PNs amplitude %s G01", Numbers::sEE(ism).c_str());
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EEPnDiodeTask/Gain16/EEPDT PNs amplitude %s G16", Numbers::sEE(ism).c_str());
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EEPnDiodeTask/Gain01/EEPDT PNs pedestal %s G01", Numbers::sEE(ism).c_str());
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EEPnDiodeTask/Gain16/EEPDT PNs pedestal %s G16", Numbers::sEE(ism).c_str());
    mui_->subscribe(histo, ism);

  }

  if ( collateSources_ ) {

    if ( verbose_ ) cout << "EETestPulseClient: collate" << endl;

    for ( unsigned int i=0; i<superModules_.size(); i++ ) {

      int ism = superModules_[i];

      sprintf(histo, "EETPT amplitude %s G01", Numbers::sEE(ism).c_str());
      me_ha01_[ism-1] = mui_->collateProf2D(histo, histo, "EcalEndcap/Sums/EETestPulseTask/Gain01");
      sprintf(histo, "*/EcalEndcap/EETestPulseTask/Gain01/EETPT amplitude %s G01", Numbers::sEE(ism).c_str());
      mui_->add(me_ha01_[ism-1], histo);

      sprintf(histo, "EETPT amplitude %s G06", Numbers::sEE(ism).c_str());
      me_ha02_[ism-1] = mui_->collateProf2D(histo, histo, "EcalEndcap/Sums/EETestPulseTask/Gain06");
      sprintf(histo, "*/EcalEndcap/EETestPulseTask/Gain06/EETPT amplitude %s G06", Numbers::sEE(ism).c_str());
      mui_->add(me_ha02_[ism-1], histo);

      sprintf(histo, "EETPT amplitude %s G12", Numbers::sEE(ism).c_str());
      me_ha03_[ism-1] = mui_->collateProf2D(histo, histo, "EcalEndcap/Sums/EETestPulseTask/Gain12");
      sprintf(histo, "*/EcalEndcap/EETestPulseTask/Gain12/EETPT amplitude %s G12", Numbers::sEE(ism).c_str());
      mui_->add(me_ha03_[ism-1], histo);

      sprintf(histo, "EETPT shape %s G01", Numbers::sEE(ism).c_str());
      me_hs01_[ism-1] = mui_->collateProf2D(histo, histo, "EcalEndcap/Sums/EETestPulseTask/Gain01");
      sprintf(histo, "*/EcalEndcap/EETestPulseTask/Gain01/EETPT shape %s G01", Numbers::sEE(ism).c_str());
      mui_->add(me_hs01_[ism-1], histo);

      sprintf(histo, "EETPT shape %s G06", Numbers::sEE(ism).c_str());
      me_hs02_[ism-1] = mui_->collateProf2D(histo, histo, "EcalEndcap/Sums/EETestPulseTask/Gain06");
      sprintf(histo, "*/EcalEndcap/EETestPulseTask/Gain06/EETPT shape %s G06", Numbers::sEE(ism).c_str());
      mui_->add(me_hs02_[ism-1], histo);

      sprintf(histo, "EETPT shape %s G12", Numbers::sEE(ism).c_str());
      me_hs03_[ism-1] = mui_->collateProf2D(histo, histo, "EcalEndcap/Sums/EETestPulseTask/Gain12");
      sprintf(histo, "*/EcalEndcap/EETestPulseTask/Gain12/EETPT shape %s G12", Numbers::sEE(ism).c_str());
      mui_->add(me_hs03_[ism-1], histo);

      sprintf(histo, "EETPT amplitude error %s G01", Numbers::sEE(ism).c_str());
      me_he01_[ism-1] = mui_->collate2D(histo, histo, "EcalEndcap/Sums/EETestPulseTask/Gain01");
      sprintf(histo, "*/EcalEndcap/EETestPulseTask/Gain01/EETPT amplitude error %s G01", Numbers::sEE(ism).c_str());
      mui_->add(me_he01_[ism-1], histo);

      sprintf(histo, "EETPT amplitude error %s G06", Numbers::sEE(ism).c_str());
      me_he02_[ism-1] = mui_->collate2D(histo, histo, "EcalEndcap/Sums/EETestPulseTask/Gain06");
      sprintf(histo, "*/EcalEndcap/EETestPulseTask/Gain06/EETPT amplitude error %s G06", Numbers::sEE(ism).c_str());
      mui_->add(me_he02_[ism-1], histo);

      sprintf(histo, "EETPT amplitude error %s G12", Numbers::sEE(ism).c_str());
      me_he03_[ism-1] = mui_->collate2D(histo, histo, "EcalEndcap/Sums/EETestPulseTask/Gain12");
      sprintf(histo, "*/EcalEndcap/EETestPulseTask/Gain12/EETPT amplitude error %s G12", Numbers::sEE(ism).c_str());
      mui_->add(me_he03_[ism-1], histo);

      sprintf(histo, "EEPDT PNs amplitude %s G01", Numbers::sEE(ism).c_str());
      me_i01_[ism-1] = mui_->collateProf2D(histo, histo, "EcalEndcap/Sums/EEPnDiodeTask/Gain01");
      sprintf(histo, "*/EcalEndcap/EEPnDiodeTask/Gain01/EEPDT PNs amplitude %s G01", Numbers::sEE(ism).c_str());
      mui_->add(me_i01_[ism-1], histo);

      sprintf(histo, "EEPDT PNs amplitude %s G16", Numbers::sEE(ism).c_str());
      me_i02_[ism-1] = mui_->collateProf2D(histo, histo, "EcalEndcap/Sums/EEPnDiodeTask/Gain16");
      sprintf(histo, "*/EcalEndcap/EEPnDiodeTask/Gain16/EEPDT PNs amplitude %s G16", Numbers::sEE(ism).c_str());
      mui_->add(me_i02_[ism-1], histo);

      sprintf(histo, "EEPDT PNs pedestal %s G01", Numbers::sEE(ism).c_str());
      me_i03_[ism-1] = mui_->collateProf2D(histo, histo, "EcalEndcap/Sums/EEPnDiodeTask/Gain01");
      sprintf(histo, "*/EcalEndcap/EEPnDiodeTask/Gain01/EEPDT PNs pedestal %s G01", Numbers::sEE(ism).c_str());
      mui_->add(me_i03_[ism-1], histo);

      sprintf(histo, "EEPDT PNs pedestal %s G16", Numbers::sEE(ism).c_str());
      me_i04_[ism-1] = mui_->collateProf2D(histo, histo, "EcalEndcap/Sums/EEPnDiodeTask/Gain16");
      sprintf(histo, "*/EcalEndcap/EEPnDiodeTask/Gain16/EEPDT PNs pedestal %s G16", Numbers::sEE(ism).c_str());
      mui_->add(me_i04_[ism-1], histo);

    }

  }

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( collateSources_ ) {
      sprintf(histo, "EcalEndcap/Sums/EETestPulseTask/Gain01/EETPT amplitude %s G01", Numbers::sEE(ism).c_str());
      if ( qtha01_[ism-1] ) mui_->useQTest(histo, qtha01_[ism-1]->getName());
      sprintf(histo, "EcalEndcap/Sums/EETestPulseTask/Gain06/EETPT amplitude %s G06", Numbers::sEE(ism).c_str());
      if ( qtha02_[ism-1] ) mui_->useQTest(histo, qtha02_[ism-1]->getName());
      sprintf(histo, "EcalEndcap/Sums/EETestPulseTask/Gain12/EETPT amplitude %s G12", Numbers::sEE(ism).c_str());
      if ( qtha03_[ism-1] ) mui_->useQTest(histo, qtha03_[ism-1]->getName());
      sprintf(histo, "EcalEndcap/Sums/EEPnDiodeTask/Gain01/EEPDT PNs amplitude %s G01", Numbers::sEE(ism).c_str());
      if ( qtha04_[ism-1] ) mui_->useQTest(histo, qtha04_[ism-1]->getName());
      sprintf(histo, "EcalEndcap/Sums/EEPnDiodeTask/Gain16/EEPDT PNs amplitude %s G16", Numbers::sEE(ism).c_str());
      if ( qtha05_[ism-1] ) mui_->useQTest(histo, qtha05_[ism-1]->getName());
      sprintf(histo, "EcalEndcap/Sums/EEPnDiodeTask/Gain01/EEPDT PNs pedestal %s G01", Numbers::sEE(ism).c_str());
      if ( qtha06_[ism-1] ) mui_->useQTest(histo, qtha06_[ism-1]->getName());
      sprintf(histo, "EcalEndcap/Sums/EEPnDiodeTask/Gain16/EEPDT PNs pedestal %s G16", Numbers::sEE(ism).c_str());
      if ( qtha07_[ism-1] ) mui_->useQTest(histo, qtha07_[ism-1]->getName());
    } else {
      if ( enableMonitorDaemon_ ) {
        sprintf(histo, "*/EcalEndcap/EETestPulseTask/Gain01/EETPT amplitude %s G01", Numbers::sEE(ism).c_str());
        if ( qtha01_[ism-1] ) mui_->useQTest(histo, qtha01_[ism-1]->getName());
        sprintf(histo, "*/EcalEndcap/EETestPulseTask/Gain06/EETPT amplitude %s G06", Numbers::sEE(ism).c_str());
        if ( qtha02_[ism-1] ) mui_->useQTest(histo, qtha02_[ism-1]->getName());
        sprintf(histo, "*/EcalEndcap/EETestPulseTask/Gain12/EETPT amplitude %s G12", Numbers::sEE(ism).c_str());
        if ( qtha03_[ism-1] ) mui_->useQTest(histo, qtha03_[ism-1]->getName());
        sprintf(histo, "*/EcalEndcap/EEPnDiodeTask/Gain01/EEPDT PNs amplitude %s G01", Numbers::sEE(ism).c_str());
        if ( qtha04_[ism-1] ) mui_->useQTest(histo, qtha04_[ism-1]->getName());
        sprintf(histo, "*/EcalEndcap/EEPnDiodeTask/Gain16/EEPDT PNs amplitude %s G16", Numbers::sEE(ism).c_str());
        if ( qtha05_[ism-1] ) mui_->useQTest(histo, qtha05_[ism-1]->getName());
        sprintf(histo, "*/EcalEndcap/EEPnDiodeTask/Gain01/EEPDT PNs pedestal %s G01", Numbers::sEE(ism).c_str());
        if ( qtha06_[ism-1] ) mui_->useQTest(histo, qtha06_[ism-1]->getName());
        sprintf(histo, "*/EcalEndcap/EEPnDiodeTask/Gain16/EEPDT PNs pedestal %s G16", Numbers::sEE(ism).c_str());
        if ( qtha07_[ism-1] ) mui_->useQTest(histo, qtha07_[ism-1]->getName());
      } else {
        sprintf(histo, "EcalEndcap/EETestPulseTask/Gain01/EETPT amplitude %s G01", Numbers::sEE(ism).c_str());
        if ( qtha01_[ism-1] ) mui_->useQTest(histo, qtha01_[ism-1]->getName());
        sprintf(histo, "EcalEndcap/EETestPulseTask/Gain06/EETPT amplitude %s G06", Numbers::sEE(ism).c_str());
        if ( qtha02_[ism-1] ) mui_->useQTest(histo, qtha02_[ism-1]->getName());
        sprintf(histo, "EcalEndcap/EETestPulseTask/Gain12/EETPT amplitude %s G12", Numbers::sEE(ism).c_str());
        if ( qtha03_[ism-1] ) mui_->useQTest(histo, qtha03_[ism-1]->getName());
        sprintf(histo, "EcalEndcap/EEPnDiodeTask/Gain01/EEPDT PNs amplitude %s G01", Numbers::sEE(ism).c_str());
        if ( qtha04_[ism-1] ) mui_->useQTest(histo, qtha04_[ism-1]->getName());
        sprintf(histo, "EcalEndcap/EEPnDiodeTask/Gain16/EEPDT PNs amplitude %s G16", Numbers::sEE(ism).c_str());
        if ( qtha05_[ism-1] ) mui_->useQTest(histo, qtha05_[ism-1]->getName());
        sprintf(histo, "EcalEndcap/EEPnDiodeTask/Gain01/EEPDT PNs pedestal %s G01", Numbers::sEE(ism).c_str());
        if ( qtha06_[ism-1] ) mui_->useQTest(histo, qtha06_[ism-1]->getName());
        sprintf(histo, "EcalEndcap/EEPnDiodeTask/Gain16/EEPDT PNs pedestal %s G16", Numbers::sEE(ism).c_str());
        if ( qtha07_[ism-1] ) mui_->useQTest(histo, qtha07_[ism-1]->getName());
      }
    }

  }

}

void EETestPulseClient::subscribeNew(void){

  Char_t histo[200];

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    unsigned int ism = superModules_[i];

    sprintf(histo, "*/EcalEndcap/EETestPulseTask/Gain01/EETPT amplitude %s G01", Numbers::sEE(ism).c_str());
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalEndcap/EETestPulseTask/Gain01/EETPT shape %s G01", Numbers::sEE(ism).c_str());
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalEndcap/EETestPulseTask/Gain01/EETPT amplitude error %s G01", Numbers::sEE(ism).c_str());
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalEndcap/EETestPulseTask/Gain06/EETPT amplitude %s G06", Numbers::sEE(ism).c_str());
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalEndcap/EETestPulseTask/Gain06/EETPT shape %s G06", Numbers::sEE(ism).c_str());
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalEndcap/EETestPulseTask/Gain06/EETPT amplitude error %s G06", Numbers::sEE(ism).c_str());
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalEndcap/EETestPulseTask/Gain12/EETPT amplitude %s G12", Numbers::sEE(ism).c_str());
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalEndcap/EETestPulseTask/Gain12/EETPT shape %s G12", Numbers::sEE(ism).c_str());
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalEndcap/EETestPulseTask/Gain12/EETPT amplitude error %s G12", Numbers::sEE(ism).c_str());
    mui_->subscribeNew(histo, ism);

    sprintf(histo, "*/EcalEndcap/EEPnDiodeTask/Gain01/EEPDT PNs amplitude %s G01", Numbers::sEE(ism).c_str());
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalEndcap/EEPnDiodeTask/Gain16/EEPDT PNs amplitude %s G16", Numbers::sEE(ism).c_str());
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalEndcap/EEPnDiodeTask/Gain01/EEPDT PNs pedestal %s G01", Numbers::sEE(ism).c_str());
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalEndcap/EEPnDiodeTask/Gain16/EEPDT PNs pedestal %s G16", Numbers::sEE(ism).c_str());
    mui_->subscribeNew(histo, ism);

  }

}

void EETestPulseClient::unsubscribe(void){

  if ( verbose_ ) cout << "EETestPulseClient: unsubscribe" << endl;

  if ( collateSources_ ) {

    if ( verbose_ ) cout << "EETestPulseClient: uncollate" << endl;

    if ( mui_ ) {

      for ( unsigned int i=0; i<superModules_.size(); i++ ) {

        int ism = superModules_[i];

        mui_->removeCollate(me_ha01_[ism-1]);
        mui_->removeCollate(me_ha02_[ism-1]);
        mui_->removeCollate(me_ha03_[ism-1]);

        mui_->removeCollate(me_hs01_[ism-1]);
        mui_->removeCollate(me_hs02_[ism-1]);
        mui_->removeCollate(me_hs03_[ism-1]);

        mui_->removeCollate(me_he01_[ism-1]);
        mui_->removeCollate(me_he02_[ism-1]);
        mui_->removeCollate(me_he03_[ism-1]);

        mui_->removeCollate(me_i01_[ism-1]);
        mui_->removeCollate(me_i02_[ism-1]);
        mui_->removeCollate(me_i03_[ism-1]);
        mui_->removeCollate(me_i04_[ism-1]);

      }

    }

  }

  Char_t histo[200];

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    unsigned int ism = superModules_[i];

    sprintf(histo, "*/EcalEndcap/EETestPulseTask/Gain01/EETPT amplitude %s G01", Numbers::sEE(ism).c_str());
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EETestPulseTask/Gain01/EETPT shape %s G01", Numbers::sEE(ism).c_str());
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EETestPulseTask/Gain01/EETPT amplitude error %s G01", Numbers::sEE(ism).c_str());
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EETestPulseTask/Gain06/EETPT amplitude %s G06", Numbers::sEE(ism).c_str());
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EETestPulseTask/Gain06/EETPT shape %s G06", Numbers::sEE(ism).c_str());
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EETestPulseTask/Gain06/EETPT amplitude error %s G06", Numbers::sEE(ism).c_str());
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EETestPulseTask/Gain12/EETPT amplitude %s G12", Numbers::sEE(ism).c_str());
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EETestPulseTask/Gain12/EETPT shape %s G12", Numbers::sEE(ism).c_str());
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EETestPulseTask/Gain12/EETPT amplitude error %s G12", Numbers::sEE(ism).c_str());
    mui_->unsubscribe(histo, ism);

    sprintf(histo, "*/EcalEndcap/EEPnDiodeTask/Gain01/EEPDT PNs amplitude %s G01", Numbers::sEE(ism).c_str());
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EEPnDiodeTask/Gain16/EEPDT PNs amplitude %s G16", Numbers::sEE(ism).c_str());
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EEPnDiodeTask/Gain01/EEPDT PNs pedestal %s G01", Numbers::sEE(ism).c_str());
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EEPnDiodeTask/Gain16/EEPDT PNs pedestal %s G16", Numbers::sEE(ism).c_str());
    mui_->unsubscribe(histo, ism);

  }

}

void EETestPulseClient::softReset(void){

}

void EETestPulseClient::analyze(void){

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( verbose_ ) cout << "EETestPulseClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;
  }

  uint64_t bits01 = 0;
  bits01 |= EcalErrorDictionary::getMask("TESTPULSE_LOW_GAIN_MEAN_WARNING");
  bits01 |= EcalErrorDictionary::getMask("TESTPULSE_LOW_GAIN_RMS_WARNING");

  uint64_t bits02 = 0;
  bits02 |= EcalErrorDictionary::getMask("TESTPULSE_MIDDLE_GAIN_MEAN_WARNING");
  bits02 |= EcalErrorDictionary::getMask("TESTPULSE_MIDDLE_GAIN_RMS_WARNING");

  uint64_t bits03 = 0;
  bits03 |= EcalErrorDictionary::getMask("TESTPULSE_HIGH_GAIN_MEAN_WARNING");
  bits03 |= EcalErrorDictionary::getMask("TESTPULSE_HIGH_GAIN_RMS_WARNING");

  uint64_t bits04 = 0;
  bits04 |= EcalErrorDictionary::getMask("PEDESTAL_LOW_GAIN_MEAN_WARNING");
  bits04 |= EcalErrorDictionary::getMask("PEDESTAL_LOW_GAIN_RMS_WARNING");
  bits04 |= EcalErrorDictionary::getMask("PEDESTAL_LOW_GAIN_MEAN_ERROR");
  bits04 |= EcalErrorDictionary::getMask("PEDESTAL_LOW_GAIN_RMS_ERROR");

  uint64_t bits05 = 0;
  bits05 |= EcalErrorDictionary::getMask("PEDESTAL_MIDDLE_GAIN_MEAN_WARNING");
  bits05 |= EcalErrorDictionary::getMask("PEDESTAL_MIDDLE_GAIN_RMS_WARNING");
  bits05 |= EcalErrorDictionary::getMask("PEDESTAL_MIDDLE_GAIN_MEAN_ERROR");
  bits05 |= EcalErrorDictionary::getMask("PEDESTAL_MIDDLE_GAIN_RMS_ERROR");

  uint64_t bits06 = 0;
  bits06 |= EcalErrorDictionary::getMask("PEDESTAL_HIGH_GAIN_MEAN_WARNING");
  bits06 |= EcalErrorDictionary::getMask("PEDESTAL_HIGH_GAIN_RMS_WARNING");
  bits06 |= EcalErrorDictionary::getMask("PEDESTAL_HIGH_GAIN_MEAN_ERROR");
  bits06 |= EcalErrorDictionary::getMask("PEDESTAL_HIGH_GAIN_RMS_ERROR");

  map<EcalLogicID, RunCrystalErrorsDat> mask1;
  map<EcalLogicID, RunPNErrorsDat> mask2;

  EcalErrorMask::fetchDataSet(&mask1);
  EcalErrorMask::fetchDataSet(&mask2);

  Char_t histo[200];

  MonitorElement* me;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( collateSources_ ) {
      sprintf(histo, "EcalEndcap/Sums/EETestPulseTask/Gain01/EETPT amplitude %s G01", Numbers::sEE(ism).c_str());
    } else {
      sprintf(histo, (prefixME_+"EcalEndcap/EETestPulseTask/Gain01/EETPT amplitude %s G01").c_str(), Numbers::sEE(ism).c_str());
    }
    me = mui_->get(histo);
    ha01_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, ha01_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalEndcap/Sums/EETestPulseTask/Gain06/EETPT amplitude %s G06", Numbers::sEE(ism).c_str());
    } else {
      sprintf(histo, (prefixME_+"EcalEndcap/EETestPulseTask/Gain06/EETPT amplitude %s G06").c_str(), Numbers::sEE(ism).c_str());
    }
    me = mui_->get(histo);
    ha02_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, ha02_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalEndcap/Sums/EETestPulseTask/Gain12/EETPT amplitude %s G12", Numbers::sEE(ism).c_str());
    } else {
      sprintf(histo, (prefixME_+"EcalEndcap/EETestPulseTask/Gain12/EETPT amplitude %s G12").c_str(), Numbers::sEE(ism).c_str());
    }
    me = mui_->get(histo);
    ha03_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, ha03_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalEndcap/Sums/EETestPulseTask/Gain01/EETPT shape %s G01", Numbers::sEE(ism).c_str());
    } else {
      sprintf(histo, (prefixME_+"EcalEndcap/EETestPulseTask/Gain01/EETPT shape %s G01").c_str(), Numbers::sEE(ism).c_str());
    }
    me = mui_->get(histo);
    hs01_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, hs01_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalEndcap/Sums/EETestPulseTask/Gain06/EETPT shape %s G06", Numbers::sEE(ism).c_str());
    } else {
      sprintf(histo, (prefixME_+"EcalEndcap/EETestPulseTask/Gain06/EETPT shape %s G06").c_str(), Numbers::sEE(ism).c_str());
    }
    me = mui_->get(histo);
    hs02_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, hs02_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalEndcap/Sums/EETestPulseTask/Gain12/EETPT shape %s G12", Numbers::sEE(ism).c_str());
    } else {
      sprintf(histo, (prefixME_+"EcalEndcap/EETestPulseTask/Gain12/EETPT shape %s G12").c_str(), Numbers::sEE(ism).c_str());
    }
    me = mui_->get(histo);
    hs03_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, hs03_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalEndcap/Sums/EETestPulseTask/Gain01/EETPT amplitude error %s G01", Numbers::sEE(ism).c_str());
    } else {
      sprintf(histo, (prefixME_+"EcalEndcap/EETestPulseTask/Gain01/EETPT amplitude error %s G01").c_str(), Numbers::sEE(ism).c_str());
    }
    me = mui_->get(histo);
    he01_[ism-1] = UtilsClient::getHisto<TH2F*>( me, cloneME_, he01_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalEndcap/Sums/EETestPulseTask/Gain06/EETPT amplitude error %s G06", Numbers::sEE(ism).c_str());
    } else {
      sprintf(histo, (prefixME_+"EcalEndcap/EETestPulseTask/Gain06/EETPT amplitude error %s G06").c_str(), Numbers::sEE(ism).c_str());
    }
    me = mui_->get(histo);
    he02_[ism-1] = UtilsClient::getHisto<TH2F*>( me, cloneME_, he02_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalEndcap/Sums/EETestPulseTask/Gain12/EETPT amplitude error %s G12", Numbers::sEE(ism).c_str());
    } else {
      sprintf(histo, (prefixME_+"EcalEndcap/EETestPulseTask/Gain12/EETPT amplitude error %s G12").c_str(), Numbers::sEE(ism).c_str());
    }
    me = mui_->get(histo);
    he03_[ism-1] = UtilsClient::getHisto<TH2F*>( me, cloneME_, he03_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalEndcap/Sums/EEPnDiodeTask/Gain01/EEPDT PNs amplitude %s G01", Numbers::sEE(ism).c_str());
    } else {
      sprintf(histo, (prefixME_+"EcalEndcap/EEPnDiodeTask/Gain01/EEPDT PNs amplitude %s G01").c_str(), Numbers::sEE(ism).c_str());
    }
    me = mui_->get(histo);
    i01_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, i01_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalEndcap/Sums/EEPnDiodeTask/Gain16/EEPDT PNs amplitude %s G16", Numbers::sEE(ism).c_str());
    } else {
      sprintf(histo, (prefixME_+"EcalEndcap/EEPnDiodeTask/Gain16/EEPDT PNs amplitude %s G16").c_str(), Numbers::sEE(ism).c_str());
    }
    me = mui_->get(histo);
    i02_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, i02_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalEndcap/Sums/EEPnDiodeTask/Gain01/EEPDT PNs pedestal %s G01", Numbers::sEE(ism).c_str());
    } else {
      sprintf(histo, (prefixME_+"EcalEndcap/EEPnDiodeTask/Gain01/EEPDT PNs pedestal %s G01").c_str(), Numbers::sEE(ism).c_str());
    }
    me = mui_->get(histo);
    i03_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, i03_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalEndcap/Sums/EEPnDiodeTask/Gain16/EEPDT PNs pedestal %s G16", Numbers::sEE(ism).c_str());
    } else {
      sprintf(histo, (prefixME_+"EcalEndcap/EEPnDiodeTask/Gain16/EEPDT PNs pedestal %s G16").c_str(), Numbers::sEE(ism).c_str());
    }
    me = mui_->get(histo);
    i04_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, i04_[ism-1] );

    UtilsClient::resetHisto( meg01_[ism-1] );
    UtilsClient::resetHisto( meg02_[ism-1] );
    UtilsClient::resetHisto( meg03_[ism-1] );

    UtilsClient::resetHisto( meg04_[ism-1] );
    UtilsClient::resetHisto( meg05_[ism-1] );

    UtilsClient::resetHisto( mea01_[ism-1] );
    UtilsClient::resetHisto( mea02_[ism-1] );
    UtilsClient::resetHisto( mea03_[ism-1] );

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

        if ( meg01_[ism-1] ) meg01_[ism-1]->setBinContent( ie, ip, 2. );
        if ( meg02_[ism-1] ) meg02_[ism-1]->setBinContent( ie, ip, 2. );
        if ( meg03_[ism-1] ) meg03_[ism-1]->setBinContent( ie, ip, 2. );

        float numEventsinCry[3] = {0., 0., 0.};

        if ( ha01_[ism-1] ) numEventsinCry[0] = ha01_[ism-1]->GetBinEntries(ha01_[ism-1]->GetBin(ie, ip));
        if ( ha02_[ism-1] ) numEventsinCry[1] = ha02_[ism-1]->GetBinEntries(ha02_[ism-1]->GetBin(ie, ip));
        if ( ha03_[ism-1] ) numEventsinCry[2] = ha03_[ism-1]->GetBinEntries(ha03_[ism-1]->GetBin(ie, ip));

        bool update01;
        bool update02;
        bool update03;

        float num01, num02, num03;
        float mean01, mean02, mean03;
        float rms01, rms02, rms03;

        update01 = UtilsClient::getBinStats(ha01_[ism-1], ie, ip, num01, mean01, rms01);
        update02 = UtilsClient::getBinStats(ha02_[ism-1], ie, ip, num02, mean02, rms02);
        update03 = UtilsClient::getBinStats(ha03_[ism-1], ie, ip, num03, mean03, rms03);

        if ( update01 ) {

          float val;

          val = 1.;
          if ( mean01 < amplitudeThreshold_ )
            val = 0.;
          if ( rms01 > RMSThreshold_ )
            val = 0.;
          if ( he01_[ism-1] && numEventsinCry[0] > 0 ) {
            float errorRate = he01_[ism-1]->GetBinContent(ie, ip) / numEventsinCry[0];
            if ( errorRate > threshold_on_AmplitudeErrorsNumber_ ) val = 0.;
          }
          if ( meg01_[ism-1] ) meg01_[ism-1]->setBinContent( ie, ip, val );

          if ( mea01_[ism-1] ) {
            if ( mean01 > 0. ) {
              mea01_[ism-1]->setBinContent( ip+20*(ie-1), mean01 );
              mea01_[ism-1]->setBinError( ip+20*(ie-1), rms01 );
            } else {
              mea01_[ism-1]->setEntries( 1.+mea01_[ism-1]->getEntries() );
            }
          }

        }

        if ( update02 ) {

          float val;

          val = 1.;
          if ( mean02 < amplitudeThreshold_ )
            val = 0.;
          if ( rms02 > RMSThreshold_ )
            val = 0.;
          if ( he02_[ism-1] && numEventsinCry[1] > 0 ) {
            float errorRate = he02_[ism-1]->GetBinContent(ie, ip) / numEventsinCry[1];
            if ( errorRate > threshold_on_AmplitudeErrorsNumber_ ) val = 0.;
          }
          if ( meg02_[ism-1] ) meg02_[ism-1]->setBinContent( ie, ip, val );

          if ( mea02_[ism-1] ) {
            if ( mean02 > 0. ) {
              mea02_[ism-1]->setBinContent( ip+20*(ie-1), mean02 );
              mea02_[ism-1]->setBinError( ip+20*(ie-1), rms02 );
            } else {
              mea02_[ism-1]->setEntries( 1.+mea02_[ism-1]->getEntries() );
            }
          }

        }

        if ( update03 ) {

          float val;

          val = 1.;
          if ( mean03 < amplitudeThreshold_ )
            val = 0.;
          if ( rms03 > RMSThreshold_ )
            val = 0.;
          if ( he03_[ism-1] && numEventsinCry[2] > 0 ) {
            float errorRate = he03_[ism-1]->GetBinContent(ie, ip) / numEventsinCry[2];
            if ( errorRate > threshold_on_AmplitudeErrorsNumber_ ) val = 0.;
          }
          if ( meg03_[ism-1] ) meg03_[ism-1]->setBinContent( ie, ip, val );

          if ( mea03_[ism-1] ) {
            if ( mean03 > 0. ) {
              mea03_[ism-1]->setBinContent( ip+20*(ie-1), mean03 );
              mea03_[ism-1]->setBinError( ip+20*(ie-1), rms03 );
            } else {
              mea03_[ism-1]->setEntries( 1.+mea03_[ism-1]->getEntries() );
            }
          }

        }

        // masking

        if ( mask1.size() != 0 ) {
          map<EcalLogicID, RunCrystalErrorsDat>::const_iterator m;
          for (m = mask1.begin(); m != mask1.end(); m++) {

            EcalLogicID ecid = m->first;

            int ic = (ip-1) + 20*(ie-1) + 1;

            if ( ecid.getID1() == ism && ecid.getID2() == ic ) {
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

    for ( int i = 1; i <= 10; i++ ) {

      if ( meg04_[ism-1] ) meg04_[ism-1]->setBinContent( i, 1, 2. );
      if ( meg05_[ism-1] ) meg05_[ism-1]->setBinContent( i, 1, 2. );

      bool update01;
      bool update02;
      bool update03;
      bool update04;

      float num01, num02, num03, num04;
      float mean01, mean02, mean03, mean04;
      float rms01, rms02, rms03, rms04;

      update01 = UtilsClient::getBinStats(i01_[ism-1], 1, i, num01, mean01, rms01);
      update02 = UtilsClient::getBinStats(i02_[ism-1], 1, i, num02, mean02, rms02);
      update03 = UtilsClient::getBinStats(i03_[ism-1], 1, i, num03, mean03, rms03);
      update04 = UtilsClient::getBinStats(i04_[ism-1], 1, i, num04, mean04, rms04);

      if ( update01 && update03 ) {

        float val;

        val = 1.;
        if ( mean01 < amplitudeThresholdPnG01_ )
          val = 0.;
        if ( mean03 < pedestalThresholdPn_ )
          val = 0.;
        if ( meg04_[ism-1] ) meg04_[ism-1]->setBinContent(i, 1, val);

      }

      if ( update02 && update04 ) {

        float val;

        val = 1.;
        if ( mean02 < amplitudeThresholdPnG16_ )
          val = 0.;
        if ( mean04 < pedestalThresholdPn_ )
          val = 0.;
        if ( meg05_[ism-1] ) meg05_[ism-1]->setBinContent(i, 1, val);

      }

      // masking

      if ( mask2.size() != 0 ) {
        map<EcalLogicID, RunPNErrorsDat>::const_iterator m;
        for (m = mask2.begin(); m != mask2.end(); m++) {

          EcalLogicID ecid = m->first;

          if ( ecid.getID1() == ism && ecid.getID2() == i-1 ) {
            if ( (m->second).getErrorBits() & (bits01|bits04) ) {
              if ( meg04_[ism-1] ) {
                float val = int(meg04_[ism-1]->getBinContent(i, 1)) % 3;
                meg04_[ism-1]->setBinContent( i, 1, val+3 );
              }
            }
            if ( (m->second).getErrorBits() & (bits03|bits06) ) {
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

    if ( qtha01_[ism-1] ) badChannels = qtha01_[ism-1]->getBadChannels();

//    if ( ! badChannels.empty() ) {
//      for ( vector<dqm::me_util::Channel>::iterator it = badChannels.begin(); it != badChannels.end(); ++it ) {
//        if ( meg01_[ism-1] ) meg01_[ism-1]->setBinContent(it->getBinX(), it->getBinY(), 0.);
//      }
//    }

    if ( qtha02_[ism-1] ) badChannels = qtha02_[ism-1]->getBadChannels();

//    if ( ! badChannels.empty() ) {
//      for ( vector<dqm::me_util::Channel>::iterator it = badChannels.begin(); it != badChannels.end(); ++it ) {
//        if ( meg02_[ism-1] ) meg02_[ism-1]->setBinContent(it->getBinX(), it->getBinY(), 0.);
//      }
//    }

    if ( qtha03_[ism-1] ) badChannels = qtha03_[ism-1]->getBadChannels();

//    if ( ! badChannels.empty() ) {
//      for ( vector<dqm::me_util::Channel>::iterator it = badChannels.begin(); it != badChannels.end(); ++it ) {
//        if ( meg03_[ism-1] ) meg03_[ism-1]->setBinContent(it->getBinX(), it->getBinY(), 0.);
//      }
//    }

    if ( qtha04_[ism-1] ) badChannels = qtha04_[ism-1]->getBadChannels();

//    if ( ! badChannels.empty() ) {
//      for ( vector<dqm::me_util::Channel>::iterator it = badChannels.begin(); it != badChannels.end(); ++it ) {
//        if ( meg04_[ism-1] ) meg04_[ism-1]->setBinContent(it->getBinX(), it->getBinY(), 0.);
//      }
//    }

    if ( qtha05_[ism-1] ) badChannels = qtha05_[ism-1]->getBadChannels();

//    if ( ! badChannels.empty() ) {
//      for ( vector<dqm::me_util::Channel>::iterator it = badChannels.begin(); it != badChannels.end(); ++it ) {
//        if ( meg05_[ism-1] ) meg05_[ism-1]->setBinContent(it->getBinX(), it->getBinY(), 0.);
//      }
//    }

    if ( qtha06_[ism-1] ) badChannels = qtha06_[ism-1]->getBadChannels();

//    if ( ! badChannels.empty() ) {
//      for ( vector<dqm::me_util::Channel>::iterator it = badChannels.begin(); it != badChannels.end(); ++it ) {
//        if ( meg06_[ism-1] ) meg06_[ism-1]->setBinContent(it->getBinX(), it->getBinY(), 0.);
//      }
//    }

    if ( qtha07_[ism-1] ) badChannels = qtha07_[ism-1]->getBadChannels();

//    if ( ! badChannels.empty() ) {
//      for ( vector<dqm::me_util::Channel>::iterator it = badChannels.begin(); it != badChannels.end(); ++it ) {
//        if ( meg07_[ism-1] ) meg07_[ism-1]->setBinContent(it->getBinX(), it->getBinY(), 0.);
//      }
//    }

  }

}

void EETestPulseClient::htmlOutput(int run, string htmlDir, string htmlName){

  cout << "Preparing EETestPulseClient html output ..." << endl;

  ofstream htmlFile;

  htmlFile.open((htmlDir + htmlName).c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlFile << "<html>  " << endl;
  htmlFile << "<head>  " << endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlFile << " http-equiv=\"content-type\">  " << endl;
  htmlFile << "  <title>Monitor:TestPulseTask output</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlFile << "<body>  " << endl;
  //htmlFile << "<br>  " << endl;
  htmlFile << "<a name=""top""></a>" << endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << run << "</span></h2>" << endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">TEST PULSE</span></h2> " << endl;
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

//  const double histMax = 1.e15;

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

  string imgNameQual[3], imgNameAmp[3], imgNameShape[3], imgNameMEPnQual[2], imgNameMEPn[2], imgNameMEPnPed[2], imgName, meName;

  TCanvas* cQual = new TCanvas("cQual", "Temp", 2*csize, csize);
  TCanvas* cAmp = new TCanvas("cAmp", "Temp", csize, csize);
  TCanvas* cShape = new TCanvas("cShape", "Temp", csize, csize);
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

      // Amplitude distributions

      imgNameAmp[iCanvas-1] = "";

      obj1f = 0;
      switch ( iCanvas ) {
        case 1:
          obj1f = UtilsClient::getHisto<TH1F*>( mea01_[ism-1] );
          break;
        case 2:
          obj1f = UtilsClient::getHisto<TH1F*>( mea02_[ism-1] );
          break;
        case 3:
          obj1f = UtilsClient::getHisto<TH1F*>( mea03_[ism-1] );
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
        gPad->SetLogy(0);
        cAmp->SaveAs(imgName.c_str());
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
          if ( hs02_[ism-1] ) obj1d = hs02_[ism-1]->ProjectionY("_py", 1, 1, "e");
          break;
        case 3:
          if ( hs03_[ism-1] ) obj1d = hs03_[ism-1]->ProjectionY("_py", 1, 1, "e");
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

    }

    // Loop on gain

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

      imgNameMEPn[iCanvas-1] = "";

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
        imgNameMEPn[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgNameMEPn[iCanvas-1];

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

      imgNameMEPnPed[iCanvas-1] = "";

      obj1d = 0;
      switch ( iCanvas ) {
        case 1:
          if ( i03_[ism-1] ) obj1d = i03_[ism-1]->ProjectionY("_py", 1, 1, "e");
          break;
        case 2:
          if ( i04_[ism-1] ) obj1d = i04_[ism-1]->ProjectionY("_py", 1, 1, "e");
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

      if ( imgNameAmp[iCanvas-1].size() != 0 )
        htmlFile << "<td><img src=\"" << imgNameAmp[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

      if ( imgNameShape[iCanvas-1].size() != 0 )
        htmlFile << "<td><img src=\"" << imgNameShape[iCanvas-1] << "\"></td>" << endl;
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

      if ( imgNameMEPnPed[iCanvas-1].size() != 0 )
        htmlFile << "<td colspan=\"2\"><img src=\"" << imgNameMEPnPed[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td colspan=\"2\"><img src=\"" << " " << "\"></td>" << endl;

      if ( imgNameMEPn[iCanvas-1].size() != 0 )
        htmlFile << "<td colspan=\"2\"><img src=\"" << imgNameMEPn[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td colspan=\"2\"><img src=\"" << " " << "\"></td>" << endl;

    }

    htmlFile << "</tr>" << endl;

    htmlFile << "<tr align=\"center\"><td colspan=\"4\">Gain 1</td><td colspan=\"4\">Gain 16</td></tr>" << endl;
    htmlFile << "</table>" << endl;
    htmlFile << "<br>" << endl;

  }

  delete cQual;
  delete cAmp;
  delete cShape;
  delete cPed;

  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  htmlFile.close();

}

