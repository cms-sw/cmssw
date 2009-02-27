/*
 * \file EBTestPulseClient.cc
 *
 * $Date: 2008/10/07 18:34:22 $
 * $Revision: 1.210 $
 * \author G. Della Ricca
 * \author F. Cossutti
 *
*/

#include <memory>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <math.h>

#include "DQMServices/Core/interface/DQMStore.h"

#include "OnlineDB/EcalCondDB/interface/MonTestPulseDat.h"
#include "OnlineDB/EcalCondDB/interface/MonPulseShapeDat.h"
#include "OnlineDB/EcalCondDB/interface/MonPNMGPADat.h"
#include "OnlineDB/EcalCondDB/interface/RunCrystalErrorsDat.h"
#include "OnlineDB/EcalCondDB/interface/RunTTErrorsDat.h"
#include "OnlineDB/EcalCondDB/interface/RunPNErrorsDat.h"

#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"

#include "CondTools/Ecal/interface/EcalErrorDictionary.h"

#include "DQM/EcalCommon/interface/EcalErrorMask.h"
#include "DQM/EcalCommon/interface/UtilsClient.h"
#include "DQM/EcalCommon/interface/LogicID.h"
#include "DQM/EcalCommon/interface/Numbers.h"

#include <DQM/EcalBarrelMonitorClient/interface/EBTestPulseClient.h>

using namespace cms;
using namespace edm;
using namespace std;

EBTestPulseClient::EBTestPulseClient(const ParameterSet& ps) {

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

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    ha01_[ism-1] = 0;
    ha02_[ism-1] = 0;
    ha03_[ism-1] = 0;

    hs01_[ism-1] = 0;
    hs02_[ism-1] = 0;
    hs03_[ism-1] = 0;

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

    mer04_[ism-1] = 0;
    mer05_[ism-1] = 0;

    me_hs01_[ism-1] = 0;
    me_hs02_[ism-1] = 0;
    me_hs03_[ism-1] = 0;

  }

  percentVariation_ = 0.2;
  RMSThreshold_ = 300.0;

  amplitudeThreshold_ = 100.;
  
  amplitudeThresholdPnG01_ = 200./16.;
  amplitudeThresholdPnG16_ = 200.;

  pedPnExpectedMean_[0] = 750.0;
  pedPnExpectedMean_[1] = 750.0;

  pedPnDiscrepancyMean_[0] = 100.0;
  pedPnDiscrepancyMean_[1] = 100.0;

  pedPnRMSThreshold_[0] = 1.0;
  pedPnRMSThreshold_[1] = 3.0;

}

EBTestPulseClient::~EBTestPulseClient() {

}

void EBTestPulseClient::beginJob(DQMStore* dqmStore) {

  dqmStore_ = dqmStore;

  if ( debug_ ) cout << "EBTestPulseClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EBTestPulseClient::beginRun(void) {

  if ( debug_ ) cout << "EBTestPulseClient: beginRun" << endl;

  jevt_ = 0;

  this->setup();

}

void EBTestPulseClient::endJob(void) {

  if ( debug_ ) cout << "EBTestPulseClient: endJob, ievt = " << ievt_ << endl;

  this->cleanup();

}

void EBTestPulseClient::endRun(void) {

  if ( debug_ ) cout << "EBTestPulseClient: endRun, jevt = " << jevt_ << endl;

  this->cleanup();

}

void EBTestPulseClient::setup(void) {

  char histo[200];

  dqmStore_->setCurrentFolder( prefixME_ + "/EBTestPulseClient" );

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( meg01_[ism-1] ) dqmStore_->removeElement( meg01_[ism-1]->getName() );
    sprintf(histo, "EBTPT test pulse quality G01 %s", Numbers::sEB(ism).c_str());
    meg01_[ism-1] = dqmStore_->book2D(histo, histo, 85, 0., 85., 20, 0., 20.);
    meg01_[ism-1]->setAxisTitle("ieta", 1);
    meg01_[ism-1]->setAxisTitle("iphi", 2);
    if ( meg02_[ism-1] ) dqmStore_->removeElement( meg02_[ism-1]->getName() );
    sprintf(histo, "EBTPT test pulse quality G06 %s", Numbers::sEB(ism).c_str());
    meg02_[ism-1] = dqmStore_->book2D(histo, histo, 85, 0., 85., 20, 0., 20.);
    meg02_[ism-1]->setAxisTitle("ieta", 1);
    meg02_[ism-1]->setAxisTitle("iphi", 2);
    if ( meg03_[ism-1] ) dqmStore_->removeElement( meg03_[ism-1]->getName() );
    sprintf(histo, "EBTPT test pulse quality G12 %s", Numbers::sEB(ism).c_str());
    meg03_[ism-1] = dqmStore_->book2D(histo, histo, 85, 0., 85., 20, 0., 20.);
    meg03_[ism-1]->setAxisTitle("ieta", 1);
    meg03_[ism-1]->setAxisTitle("iphi", 2);

    if ( meg04_[ism-1] ) dqmStore_->removeElement( meg04_[ism-1]->getName() );
    sprintf(histo, "EBTPT test pulse quality PNs G01 %s", Numbers::sEB(ism).c_str());
    meg04_[ism-1] = dqmStore_->book2D(histo, histo, 10, 0., 10., 1, 0., 5.);
    meg04_[ism-1]->setAxisTitle("pseudo-strip", 1);
    meg04_[ism-1]->setAxisTitle("channel", 2);
    if ( meg05_[ism-1] ) dqmStore_->removeElement( meg05_[ism-1]->getName() );
    sprintf(histo, "EBTPT test pulse quality PNs G16 %s", Numbers::sEB(ism).c_str());
    meg05_[ism-1] = dqmStore_->book2D(histo, histo, 10, 0., 10., 1, 0., 5.);
    meg05_[ism-1]->setAxisTitle("pseudo-strip", 1);
    meg05_[ism-1]->setAxisTitle("channel", 2);

    if ( mea01_[ism-1] ) dqmStore_->removeElement( mea01_[ism-1]->getName() );
    sprintf(histo, "EBTPT test pulse amplitude G01 %s", Numbers::sEB(ism).c_str());
    mea01_[ism-1] = dqmStore_->book1D(histo, histo, 1700, 0., 1700.);
    mea01_[ism-1]->setAxisTitle("channel", 1);
    mea01_[ism-1]->setAxisTitle("amplitude", 2);
    if ( mea02_[ism-1] ) dqmStore_->removeElement( mea02_[ism-1]->getName() );
    sprintf(histo, "EBTPT test pulse amplitude G06 %s", Numbers::sEB(ism).c_str());
    mea02_[ism-1] = dqmStore_->book1D(histo, histo, 1700, 0., 1700.);
    mea02_[ism-1]->setAxisTitle("channel", 1);
    mea02_[ism-1]->setAxisTitle("amplitude", 2);
    if ( mea03_[ism-1] ) dqmStore_->removeElement( mea03_[ism-1]->getName() );
    sprintf(histo, "EBTPT test pulse amplitude G12 %s", Numbers::sEB(ism).c_str());
    mea03_[ism-1] = dqmStore_->book1D(histo, histo, 1700, 0., 1700.);
    mea03_[ism-1]->setAxisTitle("channel", 1);
    mea03_[ism-1]->setAxisTitle("amplitude", 2);

    if ( mer04_[ism-1] ) dqmStore_->removeElement( mer04_[ism-1]->getName() );
    sprintf(histo, "EBPDT PNs pedestal rms %s G01", Numbers::sEB(ism).c_str());
    mer04_[ism-1] = dqmStore_->book1D(histo, histo, 100, 0., 10.);
    mer04_[ism-1]->setAxisTitle("rms", 1);
    if ( mer05_[ism-1] ) dqmStore_->removeElement( mer05_[ism-1]->getName() );
    sprintf(histo, "EBPDT PNs pedestal rms %s G16", Numbers::sEB(ism).c_str());
    mer05_[ism-1] = dqmStore_->book1D(histo, histo, 100, 0., 10.);
    mer05_[ism-1]->setAxisTitle("rms", 1);

    if ( me_hs01_[ism-1] ) dqmStore_->removeElement( me_hs01_[ism-1]->getName() );
    sprintf(histo, "EBTPT test pulse shape G01 %s", Numbers::sEB(ism).c_str());
    me_hs01_[ism-1] = dqmStore_->book1D(histo, histo, 10, 0., 10.);
    me_hs01_[ism-1]->setAxisTitle("sample", 1);
    me_hs01_[ism-1]->setAxisTitle("amplitude", 2);
    if ( me_hs02_[ism-1] ) dqmStore_->removeElement( me_hs02_[ism-1]->getName() );
    sprintf(histo, "EBTPT test pulse shape G06 %s", Numbers::sEB(ism).c_str());
    me_hs02_[ism-1] = dqmStore_->book1D(histo, histo, 10, 0., 10.);
    me_hs02_[ism-1]->setAxisTitle("sample", 1);
    me_hs02_[ism-1]->setAxisTitle("amplitude", 2);
    if ( me_hs03_[ism-1] ) dqmStore_->removeElement( me_hs03_[ism-1]->getName() );
    sprintf(histo, "EBTPT test pulse shape G12 %s", Numbers::sEB(ism).c_str());
    me_hs03_[ism-1] = dqmStore_->book1D(histo, histo, 10, 0., 10.);
    me_hs03_[ism-1]->setAxisTitle("sample", 1);
    me_hs03_[ism-1]->setAxisTitle("amplitude", 2);

  }

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    meg01_[ism-1]->Reset();
    meg02_[ism-1]->Reset();
    meg03_[ism-1]->Reset();

    meg04_[ism-1]->Reset();
    meg05_[ism-1]->Reset();

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

    mea01_[ism-1]->Reset();
    mea02_[ism-1]->Reset();
    mea03_[ism-1]->Reset();

    mer04_[ism-1]->Reset();
    mer05_[ism-1]->Reset();

    me_hs01_[ism-1]->Reset();
    me_hs02_[ism-1]->Reset();
    me_hs03_[ism-1]->Reset();

  }

}

void EBTestPulseClient::cleanup(void) {

  if ( ! enableCleanup_ ) return;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( cloneME_ ) {
      if ( ha01_[ism-1] ) delete ha01_[ism-1];
      if ( ha02_[ism-1] ) delete ha02_[ism-1];
      if ( ha03_[ism-1] ) delete ha03_[ism-1];

      if ( hs01_[ism-1] ) delete hs01_[ism-1];
      if ( hs02_[ism-1] ) delete hs02_[ism-1];
      if ( hs03_[ism-1] ) delete hs03_[ism-1];

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

    i01_[ism-1] = 0;
    i02_[ism-1] = 0;
    i03_[ism-1] = 0;
    i04_[ism-1] = 0;

  }

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    dqmStore_->setCurrentFolder( prefixME_ + "/EBTestPulseClient" );

    if ( meg01_[ism-1] ) dqmStore_->removeElement( meg01_[ism-1]->getName() );
    meg01_[ism-1] = 0;
    if ( meg02_[ism-1] ) dqmStore_->removeElement( meg02_[ism-1]->getName() );
    meg02_[ism-1] = 0;
    if ( meg03_[ism-1] ) dqmStore_->removeElement( meg03_[ism-1]->getName() );
    meg03_[ism-1] = 0;

    if ( meg04_[ism-1] ) dqmStore_->removeElement( meg04_[ism-1]->getName() );
    meg04_[ism-1] = 0;
    if ( meg05_[ism-1] ) dqmStore_->removeElement( meg05_[ism-1]->getName() );
    meg05_[ism-1] = 0;

    if ( mea01_[ism-1] ) dqmStore_->removeElement( mea01_[ism-1]->getName() );
    mea01_[ism-1] = 0;
    if ( mea02_[ism-1] ) dqmStore_->removeElement( mea02_[ism-1]->getName() );
    mea02_[ism-1] = 0;
    if ( mea03_[ism-1] ) dqmStore_->removeElement( mea03_[ism-1]->getName() );
    mea03_[ism-1] = 0;

    if ( mer04_[ism-1] ) dqmStore_->removeElement( mer04_[ism-1]->getName() );
    mer04_[ism-1] = 0;
    if ( mer05_[ism-1] ) dqmStore_->removeElement( mer05_[ism-1]->getName() );
    mer05_[ism-1] = 0;

    if ( me_hs01_[ism-1] ) dqmStore_->removeElement( me_hs01_[ism-1]->getName() );
    me_hs01_[ism-1] = 0;
    if ( me_hs02_[ism-1] ) dqmStore_->removeElement( me_hs02_[ism-1]->getName() );
    me_hs02_[ism-1] = 0;
    if ( me_hs03_[ism-1] ) dqmStore_->removeElement( me_hs03_[ism-1]->getName() );
    me_hs03_[ism-1] = 0;

  }

}

bool EBTestPulseClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov, bool& status, bool flag) {

  status = true;

  if ( ! flag ) return false;

  EcalLogicID ecid;

  MonTestPulseDat adc;
  map<EcalLogicID, MonTestPulseDat> dataset1;
  MonPulseShapeDat shape;
  map<EcalLogicID, MonPulseShapeDat> dataset2;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( verbose_ ) {
      cout << " " << Numbers::sEB(ism) << " (ism=" << ism << ")" << endl;
      cout << endl;
      UtilsClient::printBadChannels(meg01_[ism-1], ha01_[ism-1]);
      UtilsClient::printBadChannels(meg02_[ism-1], ha02_[ism-1]);
      UtilsClient::printBadChannels(meg03_[ism-1], ha03_[ism-1]);
    }

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

        bool update01;
        bool update02;
        bool update03;

        float num01, num02, num03;
        float mean01, mean02, mean03;
        float rms01, rms02, rms03;

        update01 = UtilsClient::getBinStatistics(ha01_[ism-1], ie, ip, num01, mean01, rms01);
        update02 = UtilsClient::getBinStatistics(ha02_[ism-1], ie, ip, num02, mean02, rms02);
        update03 = UtilsClient::getBinStatistics(ha03_[ism-1], ie, ip, num03, mean03, rms03);

        if ( update01 || update02 || update03 ) {

          if ( Numbers::icEB(ism, ie, ip) == 1 ) {

            if ( verbose_ ) {
              cout << "Preparing dataset for " << Numbers::sEB(ism) << " (ism=" << ism << ")" << endl;
              cout << "G01 (" << ie << "," << ip << ") " << num01 << " " << mean01 << " " << rms01 << endl;
              cout << "G06 (" << ie << "," << ip << ") " << num02 << " " << mean02 << " " << rms02 << endl;
              cout << "G12 (" << ie << "," << ip << ") " << num03 << " " << mean03 << " " << rms03 << endl;
              cout << endl;
            }

          }

          adc.setADCMeanG1(mean01);
          adc.setADCRMSG1(rms01);

          adc.setADCMeanG6(mean02);
          adc.setADCRMSG6(rms02);

          adc.setADCMeanG12(mean03);
          adc.setADCRMSG12(rms03);

          if ( UtilsClient::getBinStatus(meg01_[ism-1], ie, ip) &&
               UtilsClient::getBinStatus(meg02_[ism-1], ie, ip) &&
               UtilsClient::getBinStatus(meg03_[ism-1], ie, ip) ) {
            adc.setTaskStatus(true);
          } else {
            adc.setTaskStatus(false);
          }

          status = status && UtilsClient::getBinQuality(meg01_[ism-1], ie, ip) &&
                             UtilsClient::getBinQuality(meg02_[ism-1], ie, ip) &&
                             UtilsClient::getBinQuality(meg03_[ism-1], ie, ip);

          if ( Numbers::icEB(ism, ie, ip) == 1 ) {

            vector<float> sample01, sample02, sample03;

            sample01.clear();
            sample02.clear();
            sample03.clear();

            if ( me_hs01_[ism-1] ) {
              for ( int i = 1; i <= 10; i++ ) {
                sample01.push_back(int(me_hs01_[ism-1]->getBinContent(i)));
              }
            } else {
              for ( int i = 1; i <= 10; i++ ) { sample01.push_back(-1.); }
            }

            if ( me_hs02_[ism-1] ) {
              for ( int i = 1; i <= 10; i++ ) {
                sample02.push_back(int(me_hs02_[ism-1]->getBinContent(i)));
              }
            } else {
              for ( int i = 1; i <= 10; i++ ) { sample02.push_back(-1.); }
            }

            if ( me_hs03_[ism-1] ) {
              for ( int i = 1; i <= 10; i++ ) {
                sample03.push_back(int(me_hs03_[ism-1]->getBinContent(i)));
              }
            } else {
              for ( int i = 1; i <= 10; i++ ) { sample03.push_back(-1.); }
            }

            if ( verbose_ ) {
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
            }

            if ( verbose_ ) cout << endl;

            shape.setSamples(sample01,  1);
            shape.setSamples(sample02,  6);
            shape.setSamples(sample03, 12);

          }

          int ic = Numbers::indexEB(ism, ie, ip);

          if ( econn ) {
            ecid = LogicID::getEcalLogicID("EB_crystal_number", Numbers::iSM(ism, EcalBarrel), ic);
            dataset1[ecid] = adc;
            if ( Numbers::icEB(ism, ie, ip) == 1 ) dataset2[ecid] = shape;
          }

        }

      }
    }

  }

  if ( econn ) {
    try {
      if ( verbose_ ) cout << "Inserting MonTestPulseDat ..." << endl;
      if ( dataset1.size() != 0 ) econn->insertDataArraySet(&dataset1, moniov);
      if ( dataset2.size() != 0 ) econn->insertDataSet(&dataset2, moniov);
      if ( verbose_ ) cout << "done." << endl;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

  if ( verbose_ ) cout << endl;

  MonPNMGPADat pn;
  map<EcalLogicID, MonPNMGPADat> dataset3;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( verbose_ ) {
      cout << " " << Numbers::sEB(ism) << " (ism=" << ism << ")" << endl;
      cout << endl;
      UtilsClient::printBadChannels(meg04_[ism-1], i01_[ism-1]);
      UtilsClient::printBadChannels(meg04_[ism-1], i03_[ism-1]);
      UtilsClient::printBadChannels(meg05_[ism-1], i02_[ism-1]);
      UtilsClient::printBadChannels(meg05_[ism-1], i04_[ism-1]);
    }

    for ( int i = 1; i <= 10; i++ ) {

      bool update01;
      bool update02;
      bool update03;
      bool update04;

      float num01, num02, num03, num04;
      float mean01, mean02, mean03, mean04;
      float rms01, rms02, rms03, rms04;

      update01 = UtilsClient::getBinStatistics(i01_[ism-1], i, 0, num01, mean01, rms01);
      update02 = UtilsClient::getBinStatistics(i02_[ism-1], i, 0, num02, mean02, rms02);
      update03 = UtilsClient::getBinStatistics(i03_[ism-1], i, 1, num03, mean03, rms03);
      update04 = UtilsClient::getBinStatistics(i04_[ism-1], i, 1, num04, mean04, rms04);

      if ( update01 || update02 || update03 || update04 ) {

        if ( i == 1 ) {

          if ( verbose_ ) {
            cout << "Preparing dataset for " << Numbers::sEB(ism) << " (ism=" << ism << ")" << endl;
            cout << "PNs (" << i << ") G01 " << num01  << " " << mean01 << " " << rms01 << " " << num03 << " " << mean03 << " " << rms03 << endl;
            cout << "PNs (" << i << ") G16 " << num02  << " " << mean02 << " " << rms02 << " " << num04 << " " << mean04 << " " << rms04 << endl;
            cout << endl;
          }

        }

        pn.setADCMeanG1(mean01);
        pn.setADCRMSG1(rms01);

        pn.setPedMeanG1(mean03);
        pn.setPedRMSG1(rms03);

        pn.setADCMeanG16(mean02);
        pn.setADCRMSG16(rms02);

        pn.setPedMeanG16(mean04);
        pn.setPedRMSG16(rms04);

        if ( UtilsClient::getBinStatus(meg04_[ism-1], i, 1) &&
             UtilsClient::getBinStatus(meg05_[ism-1], i, 1) ) {
          pn.setTaskStatus(true);
        } else {
          pn.setTaskStatus(false);
        }

        status = status && UtilsClient::getBinQuality(meg04_[ism-1], i, 1) &&
                           UtilsClient::getBinQuality(meg05_[ism-1], i, 1);

        if ( econn ) {
          ecid = LogicID::getEcalLogicID("EB_LM_PN", Numbers::iSM(ism, EcalBarrel), i-1);
          dataset3[ecid] = pn;
        }

      }

    }

  }

  if ( econn ) {
    try {
      if ( verbose_ ) cout << "Inserting MonPNMGPADat ..." << endl;
      if ( dataset3.size() != 0 ) econn->insertDataArraySet(&dataset3, moniov);
      if ( verbose_ ) cout << "done." << endl;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

  return true;

}

void EBTestPulseClient::analyze(void) {

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( debug_ ) cout << "EBTestPulseClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;
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
  map<EcalLogicID, RunTTErrorsDat> mask3;

  EcalErrorMask::fetchDataSet(&mask1);
  EcalErrorMask::fetchDataSet(&mask2);
  EcalErrorMask::fetchDataSet(&mask3);

  char histo[200];

  MonitorElement* me;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    sprintf(histo, (prefixME_ + "/EBTestPulseTask/Gain01/EBTPT amplitude %s G01").c_str(), Numbers::sEB(ism).c_str());
    me = dqmStore_->get(histo);
    ha01_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, ha01_[ism-1] );

    sprintf(histo, (prefixME_ + "/EBTestPulseTask/Gain06/EBTPT amplitude %s G06").c_str(), Numbers::sEB(ism).c_str());
    me = dqmStore_->get(histo);
    ha02_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, ha02_[ism-1] );

    sprintf(histo, (prefixME_ + "/EBTestPulseTask/Gain12/EBTPT amplitude %s G12").c_str(), Numbers::sEB(ism).c_str());
    me = dqmStore_->get(histo);
    ha03_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, ha03_[ism-1] );

    sprintf(histo, (prefixME_ + "/EBTestPulseTask/Gain01/EBTPT shape %s G01").c_str(), Numbers::sEB(ism).c_str());
    me = dqmStore_->get(histo);
    hs01_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, hs01_[ism-1] );

    sprintf(histo, (prefixME_ + "/EBTestPulseTask/Gain06/EBTPT shape %s G06").c_str(), Numbers::sEB(ism).c_str());
    me = dqmStore_->get(histo);
    hs02_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, hs02_[ism-1] );

    sprintf(histo, (prefixME_ + "/EBTestPulseTask/Gain12/EBTPT shape %s G12").c_str(), Numbers::sEB(ism).c_str());
    me = dqmStore_->get(histo);
    hs03_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, hs03_[ism-1] );

    sprintf(histo, (prefixME_ + "/EBTestPulseTask/PN/Gain01/EBPDT PNs amplitude %s G01").c_str(), Numbers::sEB(ism).c_str());
    me = dqmStore_->get(histo);
    i01_[ism-1] = UtilsClient::getHisto<TProfile*>( me, cloneME_, i01_[ism-1] );

    sprintf(histo, (prefixME_ + "/EBTestPulseTask/PN/Gain16/EBPDT PNs amplitude %s G16").c_str(), Numbers::sEB(ism).c_str());
    me = dqmStore_->get(histo);
    i02_[ism-1] = UtilsClient::getHisto<TProfile*>( me, cloneME_, i02_[ism-1] );

    sprintf(histo, (prefixME_ + "/EBTestPulseTask/PN/Gain01/EBPDT PNs pedestal %s G01").c_str(), Numbers::sEB(ism).c_str());
    me = dqmStore_->get(histo);
    i03_[ism-1] = UtilsClient::getHisto<TProfile*>( me, cloneME_, i03_[ism-1] );

    sprintf(histo, (prefixME_ + "/EBTestPulseTask/PN/Gain16/EBPDT PNs pedestal %s G16").c_str(), Numbers::sEB(ism).c_str());
    me = dqmStore_->get(histo);
    i04_[ism-1] = UtilsClient::getHisto<TProfile*>( me, cloneME_, i04_[ism-1] );

    if ( meg01_[ism-1] ) meg01_[ism-1]->Reset();
    if ( meg02_[ism-1] ) meg02_[ism-1]->Reset();
    if ( meg03_[ism-1] ) meg03_[ism-1]->Reset();

    if ( meg04_[ism-1] ) meg04_[ism-1]->Reset();
    if ( meg05_[ism-1] ) meg05_[ism-1]->Reset();

    if ( mea01_[ism-1] ) mea01_[ism-1]->Reset();
    if ( mea02_[ism-1] ) mea02_[ism-1]->Reset();
    if ( mea03_[ism-1] ) mea03_[ism-1]->Reset();

    if ( mer04_[ism-1] ) mer04_[ism-1]->Reset();
    if ( mer05_[ism-1] ) mer05_[ism-1]->Reset();

    if ( me_hs01_[ism-1] ) me_hs01_[ism-1]->Reset();
    if ( me_hs02_[ism-1] ) me_hs02_[ism-1]->Reset();
    if ( me_hs03_[ism-1] ) me_hs03_[ism-1]->Reset();

    float meanAmpl01, meanAmpl02, meanAmpl03;

    int nCry01, nCry02, nCry03;

    meanAmpl01 = meanAmpl02 = meanAmpl03 = 0.;

    nCry01 = nCry02 = nCry03 = 0;

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

        bool update01;
        bool update02;
        bool update03;

        float num01, num02, num03;
        float mean01, mean02, mean03;
        float rms01, rms02, rms03;

        update01 = UtilsClient::getBinStatistics(ha01_[ism-1], ie, ip, num01, mean01, rms01);
        update02 = UtilsClient::getBinStatistics(ha02_[ism-1], ie, ip, num02, mean02, rms02);
        update03 = UtilsClient::getBinStatistics(ha03_[ism-1], ie, ip, num03, mean03, rms03);

        if ( update01 ) {
          meanAmpl01 += mean01;
          nCry01++;
        }

        if ( update02 ) {
          meanAmpl02 += mean02;
          nCry02++;
        }

        if ( update03 ) {
          meanAmpl03 += mean03;
          nCry03++;
        }

      }
    }

    if ( nCry01 > 0 ) meanAmpl01 /= float (nCry01);
    if ( nCry02 > 0 ) meanAmpl02 /= float (nCry02);
    if ( nCry03 > 0 ) meanAmpl03 /= float (nCry03);

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

        update01 = UtilsClient::getBinStatistics(ha01_[ism-1], ie, ip, num01, mean01, rms01);
        update02 = UtilsClient::getBinStatistics(ha02_[ism-1], ie, ip, num02, mean02, rms02);
        update03 = UtilsClient::getBinStatistics(ha03_[ism-1], ie, ip, num03, mean03, rms03);

        if ( update01 ) {

          float val;

          val = 1.;
          if ( fabs(mean01 - meanAmpl01) > fabs(percentVariation_ * meanAmpl01) || mean01 < amplitudeThreshold_ )
            val = 0.;
          if ( rms01 > RMSThreshold_ )
            val = 0.;
          if ( meg01_[ism-1] ) meg01_[ism-1]->setBinContent( ie, ip, val );

          int ic = Numbers::icEB(ism, ie, ip);

          if ( mea01_[ism-1] ) {
            if ( mean01 > 0. ) {
              mea01_[ism-1]->setBinContent( ic, mean01 );
              mea01_[ism-1]->setBinError( ic, rms01 );
            } else {
              mea01_[ism-1]->setEntries( 1.+mea01_[ism-1]->getEntries() );
            }
          }

        }

        if ( update02 ) {

          float val;

          val = 1.;
          if ( fabs(mean02 - meanAmpl02) > fabs(percentVariation_ * meanAmpl02) || mean02 < amplitudeThreshold_ )
            val = 0.;
          if ( rms02 > RMSThreshold_ )
            val = 0.;
          if ( meg02_[ism-1] ) meg02_[ism-1]->setBinContent( ie, ip, val );

          int ic = Numbers::icEB(ism, ie, ip);

          if ( mea02_[ism-1] ) {
            if ( mean02 > 0. ) {
              mea02_[ism-1]->setBinContent( ic, mean02 );
              mea02_[ism-1]->setBinError( ic, rms02 );
            } else {
              mea02_[ism-1]->setEntries( 1.+mea02_[ism-1]->getEntries() );
            }
          }

        }

        if ( update03 ) {

          float val;

          val = 1.;
          if ( fabs(mean03 - meanAmpl03) > fabs(percentVariation_ * meanAmpl03) || mean03 < amplitudeThreshold_ )
            val = 0.;
          if ( rms03 > RMSThreshold_ )
            val = 0.;
          if ( meg03_[ism-1] ) meg03_[ism-1]->setBinContent( ie, ip, val );

          int ic = Numbers::icEB(ism, ie, ip);

          if ( mea03_[ism-1] ) {
            if ( mean03 > 0. ) {
              mea03_[ism-1]->setBinContent( ic, mean03 );
              mea03_[ism-1]->setBinError( ic, rms03 );
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

            int ic = Numbers::indexEB(ism, ie, ip);

            if ( ecid.getLogicID() == LogicID::getEcalLogicID("EB_crystal_number", Numbers::iSM(ism, EcalBarrel), ic).getLogicID() ) {
              if ( (m->second).getErrorBits() & bits01 ) {
                UtilsClient::maskBinContent( meg01_[ism-1], ie, ip );
              }
              if ( (m->second).getErrorBits() & bits02 ) {
                UtilsClient::maskBinContent( meg02_[ism-1], ie, ip );
              }
              if ( (m->second).getErrorBits() & bits03 ) {
                UtilsClient::maskBinContent( meg03_[ism-1], ie, ip );
              }
            }

          }
        }

        // TT masking

        if ( mask3.size() != 0 ) {
          map<EcalLogicID, RunTTErrorsDat>::const_iterator m;
          for (m = mask3.begin(); m != mask3.end(); m++) {

            EcalLogicID ecid = m->first;

            int itt = Numbers::iTT(ism, EcalBarrel, ie, ip);

            if ( ecid.getLogicID() == LogicID::getEcalLogicID("EB_trigger_tower", Numbers::iSM(ism, EcalBarrel), itt).getLogicID() ) {
              if ( (m->second).getErrorBits() & bits01 ) {
                UtilsClient::maskBinContent( meg01_[ism-1], ie, ip );
              }
              if ( (m->second).getErrorBits() & bits02 ) {
                UtilsClient::maskBinContent( meg02_[ism-1], ie, ip );
              }
              if ( (m->second).getErrorBits() & bits03 ) {
                UtilsClient::maskBinContent( meg03_[ism-1], ie, ip );
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

      update01 = UtilsClient::getBinStatistics(i01_[ism-1], i, 0, num01, mean01, rms01);
      update02 = UtilsClient::getBinStatistics(i02_[ism-1], i, 0, num02, mean02, rms02);
      update03 = UtilsClient::getBinStatistics(i03_[ism-1], i, 0, num03, mean03, rms03);
      update04 = UtilsClient::getBinStatistics(i04_[ism-1], i, 0, num04, mean04, rms04);

      if ( mer04_[ism-1] ) mer04_[ism-1]->Fill(rms03);
      if ( mer05_[ism-1] ) mer05_[ism-1]->Fill(rms04);

      if ( update01 && update03 ) {

        float val;

        val = 1.;
        if ( mean01 < amplitudeThresholdPnG01_ )
          val = 0.;
        if ( mean03 <  pedPnExpectedMean_[0] - pedPnDiscrepancyMean_[0] ||
             pedPnExpectedMean_[0] + pedPnDiscrepancyMean_[0] < mean03)
          val = 0.;
        if ( rms03 > pedPnRMSThreshold_[0] )
          val = 0.;
        if ( meg04_[ism-1] ) meg04_[ism-1]->setBinContent(i, 1, val);

      }

      if ( update02 && update04 ) {

        float val;

        val = 1.;
        if ( mean02 < amplitudeThresholdPnG16_ )
          val = 0.;
        if ( mean04 <  pedPnExpectedMean_[1] - pedPnDiscrepancyMean_[1] ||
             pedPnExpectedMean_[1] + pedPnDiscrepancyMean_[1] < mean04)
          val = 0.;
        if ( rms04 > pedPnRMSThreshold_[1] )
          val = 0.;
        if ( meg05_[ism-1] ) meg05_[ism-1]->setBinContent(i, 1, val);

      }

      // masking

      if ( mask2.size() != 0 ) {
        map<EcalLogicID, RunPNErrorsDat>::const_iterator m;
        for (m = mask2.begin(); m != mask2.end(); m++) {

          EcalLogicID ecid = m->first;

          if ( ecid.getLogicID() == LogicID::getEcalLogicID("EB_LM_PN", Numbers::iSM(ism, EcalBarrel), i-1).getLogicID() ) {
            if ( (m->second).getErrorBits() & (bits01|bits04) ) {
              UtilsClient::maskBinContent( meg04_[ism-1], i, 1 );
            }
            if ( (m->second).getErrorBits() & (bits03|bits06) ) {
              UtilsClient::maskBinContent( meg05_[ism-1], i, 1 );
            }
          }

        }
      }

    }

    for ( int i = 1; i <= 10; i++ ) {

      if ( hs01_[ism-1] ) {
        int ic = UtilsClient::getFirstNonEmptyChannel( hs01_[ism-1] );
        if ( me_hs01_[ism-1] ) {
          me_hs01_[ism-1]->setBinContent( i, hs01_[ism-1]->GetBinContent(ic, i) );
          me_hs01_[ism-1]->setBinError( i, hs01_[ism-1]->GetBinError(ic, i) );
        }
      }

      if ( hs02_[ism-1] ) {
        int ic = UtilsClient::getFirstNonEmptyChannel( hs02_[ism-1] );
        if ( me_hs02_[ism-1] ) {
          me_hs02_[ism-1]->setBinContent( i, hs02_[ism-1]->GetBinContent(ic, i) );
          me_hs02_[ism-1]->setBinError( i, hs02_[ism-1]->GetBinError(ic, i) );
        }
      }

      if ( hs03_[ism-1] ) {
        int ic = UtilsClient::getFirstNonEmptyChannel( hs03_[ism-1] );
        if ( me_hs03_[ism-1] ) {
          me_hs03_[ism-1]->setBinContent( i, hs03_[ism-1]->GetBinContent(ic, i) );
          me_hs03_[ism-1]->setBinError( i, hs03_[ism-1]->GetBinError(ic, i) );
        }
      }

    }

  }

}

void EBTestPulseClient::softReset(bool flag) {

}

