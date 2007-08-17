
/*
 * \file EEIntegrityClient.cc
 *
 * $Date: 2007/08/17 09:05:12 $
 * $Revision: 1.20 $
 * \author G. Della Ricca
 * \author G. Franzoni
 *
 */

#include <memory>
#include <iostream>
#include <fstream>
#include <iomanip>

#include "TStyle.h"
#include "TGraph.h"
#include "TLine.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/UI/interface/MonitorUIRoot.h"
#include "DQMServices/Core/interface/QTestStatus.h"
#include "DQMServices/QualityTests/interface/QCriterionRoot.h"

#include "OnlineDB/EcalCondDB/interface/RunTag.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"
#include "OnlineDB/EcalCondDB/interface/MonCrystalConsistencyDat.h"
#include "OnlineDB/EcalCondDB/interface/MonTTConsistencyDat.h"
#include "OnlineDB/EcalCondDB/interface/MonMemChConsistencyDat.h"
#include "OnlineDB/EcalCondDB/interface/MonMemTTConsistencyDat.h"
#include "OnlineDB/EcalCondDB/interface/RunCrystalErrorsDat.h"
#include "OnlineDB/EcalCondDB/interface/RunTTErrorsDat.h"
#include "OnlineDB/EcalCondDB/interface/RunPNErrorsDat.h"
#include "OnlineDB/EcalCondDB/interface/RunMemChErrorsDat.h"
#include "OnlineDB/EcalCondDB/interface/RunMemTTErrorsDat.h"

#include "CondTools/Ecal/interface/EcalErrorDictionary.h"

#include "DQM/EcalCommon/interface/EcalErrorMask.h"
#include <DQM/EcalCommon/interface/UtilsClient.h>
#include <DQM/EcalCommon/interface/LogicID.h>
#include <DQM/EcalCommon/interface/Numbers.h>

#include <DQM/EcalEndcapMonitorClient/interface/EEIntegrityClient.h>

using namespace cms;
using namespace edm;
using namespace std;

EEIntegrityClient::EEIntegrityClient(const ParameterSet& ps){

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

  h00_ = 0;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    h_[ism-1] = 0;
    hmem_[ism-1] = 0;

    h01_[ism-1] = 0;
    h02_[ism-1] = 0;
    h03_[ism-1] = 0;
    h04_[ism-1] = 0;
    h05_[ism-1] = 0;
    h06_[ism-1] = 0;
    h07_[ism-1] = 0;
    h08_[ism-1] = 0;
    h09_[ism-1] = 0;
    h10_[ism-1] = 0;

  }

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    // integrity summary histograms
    meg01_[ism-1] = 0;
    meg02_[ism-1] = 0;

    qth01_[ism-1] = 0;
    qth02_[ism-1] = 0;
    qth03_[ism-1] = 0;
    qth04_[ism-1] = 0;
    qth05_[ism-1] = 0;
    qth06_[ism-1] = 0;
    qth07_[ism-1] = 0;
    qth08_[ism-1] = 0;
    qth09_[ism-1] = 0;
    qth10_[ism-1] = 0;

    qtg01_[ism-1] = 0;
    qtg02_[ism-1] = 0;

  }

  threshCry_ = 0.;

}

EEIntegrityClient::~EEIntegrityClient(){

}

void EEIntegrityClient::beginJob(MonitorUserInterface* mui){

  mui_ = mui;
  dbe_ = mui->getBEInterface();

  if ( verbose_ ) cout << "EEIntegrityClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

  if ( enableQT_ ) {

    Char_t qtname[200];

    for ( unsigned int i=0; i<superModules_.size(); i++ ) {

      int ism = superModules_[i];

      sprintf(qtname, "EEIT data integrity quality gain %s", Numbers::sEE(ism).c_str());
      qth01_[ism-1] = dynamic_cast<MEContentsTH2FWithinRangeROOT*> (dbe_->createQTest(ContentsTH2FWithinRangeROOT::getAlgoName(), qtname));

      sprintf(qtname, "EEIT data integrity quality ChId %s", Numbers::sEE(ism).c_str());
      qth02_[ism-1] = dynamic_cast<MEContentsTH2FWithinRangeROOT*> (dbe_->createQTest(ContentsTH2FWithinRangeROOT::getAlgoName(), qtname));

      sprintf(qtname, "EEIT data integrity quality gain switch %s", Numbers::sEE(ism).c_str());
      qth03_[ism-1] = dynamic_cast<MEContentsTH2FWithinRangeROOT*> (dbe_->createQTest(ContentsTH2FWithinRangeROOT::getAlgoName(), qtname));

      sprintf(qtname, "EEIT data integrity quality gain switch stay %s", Numbers::sEE(ism).c_str());
      qth04_[ism-1] = dynamic_cast<MEContentsTH2FWithinRangeROOT*> (dbe_->createQTest(ContentsTH2FWithinRangeROOT::getAlgoName(), qtname));

      sprintf(qtname, "EEIT data integrity quality TTId %s", Numbers::sEE(ism).c_str());
      qth05_[ism-1] = dynamic_cast<MEContentsTH2FWithinRangeROOT*> (dbe_->createQTest(ContentsTH2FWithinRangeROOT::getAlgoName(), qtname));

      sprintf(qtname, "EEIT data integrity quality TTBlockSize %s", Numbers::sEE(ism).c_str());
      qth06_[ism-1] = dynamic_cast<MEContentsTH2FWithinRangeROOT*> (dbe_->createQTest(ContentsTH2FWithinRangeROOT::getAlgoName(), qtname));

      sprintf(qtname, "EEIT data integrity quality MemChId %s", Numbers::sEE(ism).c_str());
      qth07_[ism-1] = dynamic_cast<MEContentsTH2FWithinRangeROOT*> (dbe_->createQTest(ContentsTH2FWithinRangeROOT::getAlgoName(), qtname));

      sprintf(qtname, "EEIT data integrity quality MemGain %s", Numbers::sEE(ism).c_str());
      qth08_[ism-1] = dynamic_cast<MEContentsTH2FWithinRangeROOT*> (dbe_->createQTest(ContentsTH2FWithinRangeROOT::getAlgoName(), qtname));

      sprintf(qtname, "EEIT data integrity quality MemTTId %s", Numbers::sEE(ism).c_str());
      qth09_[ism-1] = dynamic_cast<MEContentsTH2FWithinRangeROOT*> (dbe_->createQTest(ContentsTH2FWithinRangeROOT::getAlgoName(), qtname));

      sprintf(qtname, "EEIT data integrity quality MemSize %s", Numbers::sEE(ism).c_str());
      qth10_[ism-1] = dynamic_cast<MEContentsTH2FWithinRangeROOT*> (dbe_->createQTest(ContentsTH2FWithinRangeROOT::getAlgoName(), qtname));

      qth01_[ism-1]->setMeanRange(-1.0, threshCry_);
      qth02_[ism-1]->setMeanRange(-1.0, threshCry_);
      qth03_[ism-1]->setMeanRange(-1.0, threshCry_);
      qth04_[ism-1]->setMeanRange(-1.0, threshCry_);
      qth05_[ism-1]->setMeanRange(-1.0, threshCry_);
      qth06_[ism-1]->setMeanRange(-1.0, threshCry_);
      qth07_[ism-1]->setMeanRange(-1.0, threshCry_);
      qth08_[ism-1]->setMeanRange(-1.0, threshCry_);
      qth09_[ism-1]->setMeanRange(-1.0, threshCry_);
      qth10_[ism-1]->setMeanRange(-1.0, threshCry_);

      qth01_[ism-1]->setMinimumEntries(0);
      qth02_[ism-1]->setMinimumEntries(0);
      qth03_[ism-1]->setMinimumEntries(0);
      qth04_[ism-1]->setMinimumEntries(0);
      qth05_[ism-1]->setMinimumEntries(0);
      qth06_[ism-1]->setMinimumEntries(0);
      qth07_[ism-1]->setMinimumEntries(0);
      qth08_[ism-1]->setMinimumEntries(0);
      qth09_[ism-1]->setMinimumEntries(0);
      qth10_[ism-1]->setMinimumEntries(0);

      qth01_[ism-1]->setErrorProb(1.00);
      qth02_[ism-1]->setErrorProb(1.00);
      qth03_[ism-1]->setErrorProb(1.00);
      qth04_[ism-1]->setErrorProb(1.00);
      qth05_[ism-1]->setErrorProb(1.00);
      qth06_[ism-1]->setErrorProb(1.00);
      qth07_[ism-1]->setErrorProb(1.00);
      qth08_[ism-1]->setErrorProb(1.00);
      qth09_[ism-1]->setErrorProb(1.00);
      qth10_[ism-1]->setErrorProb(1.00);

      sprintf(qtname, "EEIT quality test %s", Numbers::sEE(ism).c_str());
      qtg01_[ism-1] = dynamic_cast<MEContentsTH2FWithinRangeROOT*> (dbe_->createQTest(ContentsTH2FWithinRangeROOT::getAlgoName(), qtname));

      sprintf(qtname, "EEIT quality test MEM %s", Numbers::sEE(ism).c_str());
      qtg02_[ism-1] = dynamic_cast<MEContentsTH2FWithinRangeROOT*> (dbe_->createQTest(ContentsTH2FWithinRangeROOT::getAlgoName(), qtname));

      qtg01_[ism-1]->setMeanRange(1., 6.);
      qtg02_[ism-1]->setMeanRange(1., 6.);

      qtg01_[ism-1]->setErrorProb(1.00);
      qtg02_[ism-1]->setErrorProb(1.00);

    }

  }

}

void EEIntegrityClient::beginRun(void){

  if ( verbose_ ) cout << "EEIntegrityClient: beginRun" << endl;

  jevt_ = 0;

  this->setup();

  this->subscribe();

}

void EEIntegrityClient::endJob(void) {

  if ( verbose_ ) cout << "EEIntegrityClient: endJob, ievt = " << ievt_ << endl;

  this->unsubscribe();

  this->cleanup();

}

void EEIntegrityClient::endRun(void) {

  if ( verbose_ ) cout << "EEIntegrityClient: endRun, jevt = " << jevt_ << endl;

  this->unsubscribe();

  this->cleanup();

}

void EEIntegrityClient::setup(void) {

  Char_t histo[200];

  dbe_->setCurrentFolder( "EcalEndcap/EEIntegrityClient" );

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( meg01_[ism-1] ) dbe_->removeElement( meg01_[ism-1]->getName() );
    sprintf(histo, "EEIT data integrity quality %s", Numbers::sEE(ism).c_str());
    meg01_[ism-1] = dbe_->book2D(histo, histo, 50, Numbers::ix0EE(ism)+0., Numbers::ix0EE(ism)+50., 50, Numbers::iy0EE(ism)+0., Numbers::iy0EE(ism)+50.);

    if ( meg02_[ism-1] ) dbe_->removeElement( meg02_[ism-1]->getName() );
    sprintf(histo, "EEIT data integrity quality MEM %s", Numbers::sEE(ism).c_str());
    meg02_[ism-1] = dbe_->book2D(histo, histo, 10, 0., 10., 5, 0.,5.);

  }

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    UtilsClient::resetHisto( meg01_[ism-1] );
    UtilsClient::resetHisto( meg02_[ism-1] );

    for ( int ix = 1; ix <= 50; ix++ ) {
      for ( int iy = 1; iy <= 50; iy++ ) {

        meg01_[ism-1]->setBinContent( ix, iy, -1. );

        int jx = ix + Numbers::ix0EE(ism);
        int jy = iy + Numbers::iy0EE(ism);

        if ( Numbers::validEE(ism, 101 - jx, jy) ) {
          meg01_[ism-1]->setBinContent( ix, iy, 2. );
        }

      }
    }

    for ( int ie = 1; ie <= 10; ie++ ) {
      for ( int ip = 1; ip <= 5; ip++ ) {

        meg02_[ism-1]->setBinContent( ie, ip, 2. );

      }
    }

  }

}

void EEIntegrityClient::cleanup(void) {

  if ( cloneME_ ) {
    if ( h00_ ) delete h00_;
  }

  h00_ = 0;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( cloneME_ ) {
      if ( h_[ism-1] )    delete h_[ism-1];
      if ( hmem_[ism-1] ) delete hmem_[ism-1];

      if ( h01_[ism-1] ) delete h01_[ism-1];
      if ( h02_[ism-1] ) delete h02_[ism-1];
      if ( h03_[ism-1] ) delete h03_[ism-1];
      if ( h04_[ism-1] ) delete h04_[ism-1];
      if ( h05_[ism-1] ) delete h05_[ism-1];
      if ( h06_[ism-1] ) delete h06_[ism-1];
      if ( h07_[ism-1] ) delete h07_[ism-1];
      if ( h08_[ism-1] ) delete h08_[ism-1];
      if ( h09_[ism-1] ) delete h09_[ism-1];
      if ( h10_[ism-1] ) delete h10_[ism-1];
    }

    h_[ism-1] = 0;
    hmem_[ism-1] = 0;

    h01_[ism-1] = 0;
    h02_[ism-1] = 0;
    h03_[ism-1] = 0;
    h04_[ism-1] = 0;
    h05_[ism-1] = 0;
    h06_[ism-1] = 0;
    h07_[ism-1] = 0;
    h08_[ism-1] = 0;
    h09_[ism-1] = 0;
    h10_[ism-1] = 0;

  }

  dbe_->setCurrentFolder( "EcalEndcap/EEIntegrityClient" );

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( meg01_[ism-1] ) dbe_->removeElement( meg01_[ism-1]->getName() );
    meg01_[ism-1] = 0;

    if ( meg02_[ism-1] ) dbe_->removeElement( meg02_[ism-1]->getName() );
    meg02_[ism-1] = 0;

  }

}

bool EEIntegrityClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov) {

  bool status = true;

  EcalLogicID ecid;

  MonCrystalConsistencyDat c1;
  map<EcalLogicID, MonCrystalConsistencyDat> dataset1;
  MonTTConsistencyDat c2;
  map<EcalLogicID, MonTTConsistencyDat> dataset2;
  MonMemChConsistencyDat c3;
  map<EcalLogicID, MonMemChConsistencyDat> dataset3;
  MonMemTTConsistencyDat c4;
  map<EcalLogicID, MonMemTTConsistencyDat> dataset4;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    cout << " SM=" << ism << endl;

    UtilsClient::printBadChannels(qth01_[ism-1]);
    UtilsClient::printBadChannels(qth02_[ism-1]);
    UtilsClient::printBadChannels(qth03_[ism-1]);
    UtilsClient::printBadChannels(qth04_[ism-1]);
    UtilsClient::printBadChannels(qth05_[ism-1]);
    UtilsClient::printBadChannels(qth06_[ism-1]);

    UtilsClient::printBadChannels(qth07_[ism-1]);
    UtilsClient::printBadChannels(qth08_[ism-1]);
    UtilsClient::printBadChannels(qth09_[ism-1]);
    UtilsClient::printBadChannels(qth10_[ism-1]);

//    UtilsClient::printBadChannels(qtg01_[ism-1]);
//    UtilsClient::printBadChannels(qtg02_[ism-1]);

    float num00;

    num00 = 0.;

    bool update0 = false;

    if ( h00_ ) {
      num00  = h00_->GetBinContent(ism);
      if ( num00 > 0 ) update0 = true;
    }

    float num01, num02, num03, num04;

    for ( int ix = 1; ix <= 50; ix++ ) {
      for ( int iy = 1; iy <= 50; iy++ ) {

        int jx = ix + Numbers::ix0EE(ism);
        int jy = iy + Numbers::iy0EE(ism);

        if ( ! Numbers::validEE(ism, 101 - jx, jy) ) continue;

        num01 = num02 = num03 = num04 = 0.;

        bool update1 = false;

        float numTot = -1.;

        if ( h_[ism-1] ) numTot = h_[ism-1]->GetBinContent(ix, iy);

        if ( h01_[ism-1] ) {
          num01  = h01_[ism-1]->GetBinContent(ix, iy);
          if ( num01 > 0 ) update1 = true;
        }

        if ( h02_[ism-1] ) {
          num02  = h02_[ism-1]->GetBinContent(ix, iy);
          if ( num02 > 0 ) update1 = true;
        }

        if ( h03_[ism-1] ) {
          num03  = h03_[ism-1]->GetBinContent(ix, iy);
          if ( num03 > 0 ) update1 = true;
        }

        if ( h04_[ism-1] ) {
          num04  = h04_[ism-1]->GetBinContent(ix, iy);
          if ( num04 > 0 ) update1 = true;
        }

        if ( update0 || update1 ) {

          if ( ix == 1 && iy == 1 ) {

            cout << "Preparing dataset for SM=" << ism << endl;

            cout << "(" << ix << "," << iy << ") " << num00 << " " << num01 << " " << num02 << " " << num03 << " " << num04 << endl;

            cout << endl;

          }

          c1.setProcessedEvents(int(numTot));
          c1.setProblematicEvents(int(num01+num02+num03+num04));
          c1.setProblemsGainZero(int(num01));
          c1.setProblemsID(int(num02));
          c1.setProblemsGainSwitch(int(num03+num04));

          bool val;

          val = true;
          if ( numTot > 0 ) {
            float errorRate1 = num00 / numTot;
            if ( errorRate1 > threshCry_ )
              val = false;
            errorRate1 = ( num01 + num02 + num03 + num04 ) / numTot / 4.;
            if ( errorRate1 > threshCry_ )
              val = false;
          } else {
            if ( num00 > 0 )
              val = false;
            if ( ( num01 + num02 + num03 + num04 ) > 0 )
              val = false;
          }
          c1.setTaskStatus(val);

          int ic = Numbers::icEE(ism, ix, iy);

          if ( ic == -1 ) continue;

          if ( econn ) {
            try {
              ecid = LogicID::getEcalLogicID("EB_crystal_number", Numbers::iSM(ism, EcalEndcap), ic);
              dataset1[ecid] = c1;
            } catch (runtime_error &e) {
              cerr << e.what() << endl;
            }
          }

          status = status && val;

        }

      }
    }

    float num05, num06;

    for ( int ixt = 1; ixt <= 10; ixt++ ) {
      for ( int iyt = 1; iyt <= 10; iyt++ ) {

        num05 = num06 = 0.;

        bool update1 = false;

        float numTot = -1.;

        if ( h_[ism-1] ) {
          numTot = 0.;
          for ( int ix = 1 + 5*(ixt-1); ix <= 5*ixt; ix++ ) {
            for ( int iy = 1 + 5*(iyt-1); iy <= 5*iyt; iy++ ) {
              numTot += h_[ism-1]->GetBinContent(ix, iy);
            }
          }
        }

        if ( h05_[ism-1] ) {
          num05  = h05_[ism-1]->GetBinContent(ixt, iyt);
          if ( num05 > 0 ) update1 = true;
        }

        if ( h06_[ism-1] ) {
          num06  = h06_[ism-1]->GetBinContent(ixt, iyt);
          if ( num06 > 0 ) update1 = true;
        }

        if ( update0 || update1 ) {

          if ( ixt == 1 && iyt == 1 ) {

            cout << "Preparing dataset for SM=" << ism << endl;

            cout << "(" << ixt << "," << iyt << ") " << num00 << " " << num05 << " " << num06 << endl;

            cout << endl;

          }

          c2.setProcessedEvents(int(numTot));
          c2.setProblematicEvents(int(num05+num06));
          c2.setProblemsID(int(num05));
          c2.setProblemsSize(int(num06));
          c2.setProblemsLV1(int(-1.));
          c2.setProblemsBunchX(int(-1.));

          bool val;

          val = true;
          if ( numTot > 0 ) {
            float errorRate2 = num00 / numTot;
            if ( errorRate2 > threshCry_ )
              val = false;
            errorRate2 = ( num05 + num06 ) / numTot / 2.;
            if ( errorRate2 > threshCry_ )
              val = false;
          } else {
            if ( num00 > 0 )
              val = false;
            if ( ( num05 + num06 ) > 0 )
              val = false;
          }
          c2.setTaskStatus(val);

          int itt = (iyt-1) + 4*(ixt-1) + 1;

          if ( econn ) {
            try {
              ecid = LogicID::getEcalLogicID("EB_trigger_tower", Numbers::iSM(ism, EcalEndcap), itt);
              dataset2[ecid] = c2;
            } catch (runtime_error &e) {
              cerr << e.what() << endl;
            }
          }

          status = status && val;

        }

      }
    }

    float num07, num08;

    for ( int ix = 1; ix <= 10; ix++ ) {
      for ( int iy = 1; iy <= 5; iy++ ) {

        num07 = num08 = 0.;

        bool update1 = false;

        float numTot = -1.;

        if ( hmem_[ism-1] ) numTot = hmem_[ism-1]->GetBinContent(ix, iy);

        if ( h07_[ism-1] ) {
          num07  = h07_[ism-1]->GetBinContent(ix, iy);
          if ( num07 > 0 ) update1 = true;
        }

        if ( h08_[ism-1] ) {
          num08  = h08_[ism-1]->GetBinContent(ix, iy);
          if ( num08 > 0 ) update1 = true;
        }

        if ( update0 || update1 ) {

          if ( ix == 1 && iy == 1 ) {

            cout << "Preparing dataset for mem of SM=" << ism << endl;

            cout << "(" << ix << "," << iy << ") " << num07 << " " << num08 << endl;

            cout << endl;

          }

          c3.setProcessedEvents( int (numTot));
          c3.setProblematicEvents(int (num07+num08));
          c3.setProblemsID(int (num07) );
          c3.setProblemsGainZero(int (num08));
          // c3.setProblemsGainSwitch(int prob);

          bool val;

          val = true;
          if ( numTot > 0 ) {
            float errorRate1 = num00 / numTot;
            if ( errorRate1 > threshCry_ )
              val = false;
            errorRate1 = ( num07 + num08 ) / numTot / 2.;
            if ( errorRate1 > threshCry_ )
              val = false;
          } else {
            if ( num00 > 0 )
             val = false;
            if ( ( num07 + num08 ) > 0 )
              val = false;
          }
          c3. setTaskStatus(val);

          int ic = EEIntegrityClient::chNum[ (ix-1)%5 ][ (iy-1) ] + (ix-1)/5 * 25;

          if ( econn ) {
            try {
              ecid = LogicID::getEcalLogicID("EB_mem_channel", Numbers::iSM(ism, EcalEndcap), ic);
              dataset3[ecid] = c3;
            } catch (runtime_error &e) {
              cerr << e.what() << endl;
            }
          }

          status = status && val;

        }

      }
    }

    float num09, num10;

    for ( int ixt = 1; ixt <= 2; ixt++ ) {

      num09 = num10 = 0.;

      bool update1 = false;

      float numTot = -1.;

      if ( hmem_[ism-1] ) {
        numTot = 0.;
        for ( int ix = 1 + 5*(ixt-1); ix <= 5*ixt; ix++ ) {
          for ( int iy = 1 ; iy <= 5; iy++ ) {
            numTot += hmem_[ism-1]->GetBinContent(ix, iy);
          }
        }
      }

      if ( h09_[ism-1] ) {
        num09  = h09_[ism-1]->GetBinContent(ixt, 1);
        if ( num09 > 0 ) update1 = true;
      }

      if ( h10_[ism-1] ) {
        num10  = h10_[ism-1]->GetBinContent(ixt, 1);
        if ( num10 > 0 ) update1 = true;
      }

      if ( update0 || update1 ) {

        if ( ixt == 1 ) {

          cout << "Preparing dataset for SM=" << ism << endl;

          cout << "(" << ixt <<  ") " << num09 << " " << num10 << endl;

          cout << endl;

        }

        c4.setProcessedEvents( int(numTot) );
        c4.setProblematicEvents( int(num09 + num10) );
        c4.setProblemsID( int(num09) );
        c4.setProblemsSize(int (num10) );
        // setProblemsLV1(int LV1);
        // setProblemsBunchX(int bunchX);

        bool val;

        val = true;
        if ( numTot > 0 ) {
          float errorRate2 = num00 / numTot;
          if ( errorRate2 > threshCry_ )
            val = false;
          errorRate2 = ( num09 + num10 ) / numTot / 2.;
          if ( errorRate2 > threshCry_ )
            val = false;
        } else {
          if ( num00 > 0 )
            val = false;
          if ( ( num09 + num10 ) > 0 )
            val = false;
        }
        c4.setTaskStatus(val);

        int itt = 68 + ixt;

        if ( econn ) {
          try {
            ecid = LogicID::getEcalLogicID("EB_mem_TT", Numbers::iSM(ism, EcalEndcap), itt);
            dataset4[ecid] = c4;
          } catch (runtime_error &e) {
            cerr << e.what() << endl;
          }
        }

        status = status && val;

      }

    }

  }

  if ( econn ) {
    try {
      cout << "Inserting MonCrystalConsistencyDat ... " << flush;
      if ( dataset1.size() != 0 ) econn->insertDataArraySet(&dataset1, moniov);
      if ( dataset2.size() != 0 ) econn->insertDataArraySet(&dataset2, moniov);
      if ( dataset3.size() != 0 ) econn->insertDataArraySet(&dataset3, moniov);
      if ( dataset4.size() != 0 ) econn->insertDataArraySet(&dataset4, moniov);
      cout << "done." << endl;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

  return status;

}

void EEIntegrityClient::subscribe(void){

  if ( verbose_ ) cout << "EEIntegrityClient: subscribe" << endl;

  Char_t histo[200];

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    unsigned int ism = superModules_[i];

    sprintf(histo, "*/EcalEndcap/EEOccupancyTask/EEOT occupancy %s", Numbers::sEE(ism).c_str());
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EEOccupancyTask/EEOT MEM occupancy %s", Numbers::sEE(ism).c_str());
    mui_->subscribe(histo, ism);

    sprintf(histo, "*/EcalEndcap/EEIntegrityTask/EEIT DCC size error");
    mui_->subscribe(histo);
    sprintf(histo, "*/EcalEndcap/EEIntegrityTask/Gain/EEIT gain %s", Numbers::sEE(ism).c_str());
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EEIntegrityTask/ChId/EEIT ChId %s", Numbers::sEE(ism).c_str());
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EEIntegrityTask/GainSwitch/EEIT gain switch %s", Numbers::sEE(ism).c_str());
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EEIntegrityTask/GainSwitchStay/EEIT gain switch stay %s", Numbers::sEE(ism).c_str());
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EEIntegrityTask/TTId/EEIT TTId %s", Numbers::sEE(ism).c_str());
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EEIntegrityTask/TTBlockSize/EEIT TTBlockSize %s", Numbers::sEE(ism).c_str());
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EEIntegrityTask/MemChId/EEIT MemChId %s", Numbers::sEE(ism).c_str());
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EEIntegrityTask/MemGain/EEIT MemGain %s", Numbers::sEE(ism).c_str());
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EEIntegrityTask/MemTTId/EEIT MemTTId %s", Numbers::sEE(ism).c_str());
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EEIntegrityTask/MemSize/EEIT MemSize %s", Numbers::sEE(ism).c_str());
    mui_->subscribe(histo, ism);

  }

  if ( collateSources_ ) {

    if ( verbose_ ) cout << "EEIntegrityClient: collate" << endl;

    sprintf(histo, "EEIT DCC size error");
    me_h00_ = mui_->collate1D(histo, histo, "EcalEndcap/Sums/EEIntegrityTask");
    sprintf(histo, "*/EcalEndcap/EEIntegrityTask/EEIT DCC size error");
    mui_->add(me_h00_, histo);

    for ( unsigned int i=0; i<superModules_.size(); i++ ) {

      int ism = superModules_[i];

      sprintf(histo, "EEOT occupancy %s", Numbers::sEE(ism).c_str());
      me_h_[ism-1] = mui_->collateProf2D(histo, histo, "EcalEndcap/Sums/EEOccupancyTask");
      sprintf(histo, "*/EcalEndcap/EEOccupancyTask/EEOT occupancy %s", Numbers::sEE(ism).c_str());
      mui_->add(me_h_[ism-1], histo);

      sprintf(histo, "EEOT MEM occupancy %s", Numbers::sEE(ism).c_str());
      me_hmem_[ism-1] = mui_->collateProf2D(histo, histo, "EcalEndcap/Sums/EEOccupancyTask");
      sprintf(histo, "*/EcalEndcap/EEOccupancyTask/EEOT MEM occupancy %s", Numbers::sEE(ism).c_str());
      mui_->add(me_hmem_[ism-1], histo);

      sprintf(histo, "EEIT gain %s", Numbers::sEE(ism).c_str());
      me_h01_[ism-1] = mui_->collate2D(histo, histo, "EcalEndcap/Sums/EEIntegrityTask/Gain");
      sprintf(histo, "*/EcalEndcap/EEIntegrityTask/Gain/EEIT gain %s", Numbers::sEE(ism).c_str());
      mui_->add(me_h01_[ism-1], histo);

      sprintf(histo, "EEIT ChId %s", Numbers::sEE(ism).c_str());
      me_h02_[ism-1] = mui_->collate2D(histo, histo, "EcalEndcap/Sums/EEIntegrityTask/ChId");
      sprintf(histo, "*/EcalEndcap/EEIntegrityTask/ChId/EEIT ChId %s", Numbers::sEE(ism).c_str());
      mui_->add(me_h02_[ism-1], histo);

      sprintf(histo, "EEIT gain switch %s", Numbers::sEE(ism).c_str());
      me_h03_[ism-1] = mui_->collate2D(histo, histo, "EcalEndcap/Sums/EEIntegrityTask/GainSwitch");
      sprintf(histo, "*/EcalEndcap/EEIntegrityTask/GainSwitch/EEIT gain switch %s", Numbers::sEE(ism).c_str());
      mui_->add(me_h03_[ism-1], histo);

      sprintf(histo, "EEIT gain switch stay %s", Numbers::sEE(ism).c_str());
      me_h04_[ism-1] = mui_->collate2D(histo, histo, "EcalEndcap/Sums/EEIntegrityTask/GainSwitchStay");
      sprintf(histo, "*/EcalEndcap/EEIntegrityTask/GainSwitchStay/EEIT gain switch stay %s", Numbers::sEE(ism).c_str());
      mui_->add(me_h04_[ism-1], histo);

      sprintf(histo, "EEIT TTId %s", Numbers::sEE(ism).c_str());
      me_h05_[ism-1] = mui_->collate2D(histo, histo, "EcalEndcap/Sums/EEIntegrityTask/TTId");
      sprintf(histo, "*/EcalEndcap/EEIntegrityTask/TTId/EEIT TTId %s", Numbers::sEE(ism).c_str());
      mui_->add(me_h05_[ism-1], histo);

      sprintf(histo, "EEIT TTBlockSize %s", Numbers::sEE(ism).c_str());
      me_h06_[ism-1] = mui_->collate2D(histo, histo, "EcalEndcap/Sums/EEIntegrityTask/TTBlockSize");
      sprintf(histo, "*/EcalEndcap/EEIntegrityTask/TTBlockSize/EEIT TTBlockSize %s", Numbers::sEE(ism).c_str());
      mui_->add(me_h06_[ism-1], histo);

      sprintf(histo, "EEIT MemChId %s", Numbers::sEE(ism).c_str());
      me_h07_[ism-1] = mui_->collate2D(histo, histo, "EcalEndcap/Sums/EEIntegrityTask/MemChId");
      sprintf(histo, "*/EcalEndcap/EEIntegrityTask/MemChId/EEIT MemChId %s", Numbers::sEE(ism).c_str());
      mui_->add(me_h07_[ism-1], histo);

      sprintf(histo, "EEIT MemGain %s", Numbers::sEE(ism).c_str());
      me_h08_[ism-1] = mui_->collate2D(histo, histo, "EcalEndcap/Sums/EEIntegrityTask/MemGain");
      sprintf(histo, "*/EcalEndcap/EEIntegrityTask/MemGain/EEIT MemGain %s", Numbers::sEE(ism).c_str());
      mui_->add(me_h08_[ism-1], histo);

      sprintf(histo, "EEIT MemTTId %s", Numbers::sEE(ism).c_str());
      me_h09_[ism-1] = mui_->collate2D(histo, histo, "EcalEndcap/Sums/EEIntegrityTask/MemTTId");
      sprintf(histo, "*/EcalEndcap/EEIntegrityTask/MemTTId/EEIT MemTTId %s", Numbers::sEE(ism).c_str());
      mui_->add(me_h09_[ism-1], histo);

      sprintf(histo, "EEIT MemSize %s", Numbers::sEE(ism).c_str());
      me_h10_[ism-1] = mui_->collate2D(histo, histo, "EcalEndcap/Sums/EEIntegrityTask/MemSize");
      sprintf(histo, "*/EcalEndcap/EEIntegrityTask/MemSize/EEIT MemSize %s", Numbers::sEE(ism).c_str());
      mui_->add(me_h10_[ism-1], histo);

    }

  }

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( collateSources_ ) {
      sprintf(histo, "EcalEndcap/Sums/EEIntegrityTask/Gain/EEIT gain %s", Numbers::sEE(ism).c_str());
      if ( qth01_[ism-1] ) mui_->useQTest(histo, qth01_[ism-1]->getName());
      sprintf(histo, "EcalEndcap/Sums/EEIntegrityTask/ChId/EEIT ChId %s", Numbers::sEE(ism).c_str());
      if ( qth02_[ism-1] ) mui_->useQTest(histo, qth02_[ism-1]->getName());
      sprintf(histo, "EcalEndcap/Sums/EEIntegrityTask/GainSwitch/EEIT gain switch %s", Numbers::sEE(ism).c_str());
      if ( qth03_[ism-1] ) mui_->useQTest(histo, qth03_[ism-1]->getName());
      sprintf(histo, "EcalEndcap/Sums/EEIntegrityTask/GainSwitchStay/EEIT gain switch stay %s", Numbers::sEE(ism).c_str());
      if ( qth04_[ism-1] ) mui_->useQTest(histo, qth04_[ism-1]->getName());
      sprintf(histo, "EcalEndcap/Sums/EEIntegrityTask/TTId/EEIT TTId %s", Numbers::sEE(ism).c_str());
      if ( qth05_[ism-1] ) mui_->useQTest(histo, qth05_[ism-1]->getName());
      sprintf(histo, "EcalEndcap/Sums/EEIntegrityTask/TTBlockSize/EEIT TTBlockSize %s", Numbers::sEE(ism).c_str());
      if ( qth06_[ism-1] ) mui_->useQTest(histo, qth06_[ism-1]->getName());
      sprintf(histo, "EcalEndcap/Sums/EEIntegrityTask/MemChId/EEIT MemChId %s", Numbers::sEE(ism).c_str());
      if ( qth07_[ism-1] ) mui_->useQTest(histo, qth07_[ism-1]->getName());
      sprintf(histo, "EcalEndcap/Sums/EEIntegrityTask/MemGain %s", Numbers::sEE(ism).c_str());
      if ( qth08_[ism-1] ) mui_->useQTest(histo, qth08_[ism-1]->getName());
      sprintf(histo, "EcalEndcap/Sums/EEIntegrityTask/MemTTId/EEIT MemTTId %s", Numbers::sEE(ism).c_str());
      if ( qth09_[ism-1] ) mui_->useQTest(histo, qth09_[ism-1]->getName());
      sprintf(histo, "EcalEndcap/Sums/EEIntegrityTask/MemSize/EEIT MemSize %s", Numbers::sEE(ism).c_str());
      if ( qth10_[ism-1] ) mui_->useQTest(histo, qth10_[ism-1]->getName());
    } else {
      if ( enableMonitorDaemon_ ) {
        sprintf(histo, "*/EcalEndcap/EEIntegrityTask/Gain/EEIT gain %s", Numbers::sEE(ism).c_str());
        if ( qth01_[ism-1] ) mui_->useQTest(histo, qth01_[ism-1]->getName());
        sprintf(histo, "*/EcalEndcap/EEIntegrityTask/ChId/EEIT ChId %s", Numbers::sEE(ism).c_str());
        if ( qth02_[ism-1] ) mui_->useQTest(histo, qth02_[ism-1]->getName());
        sprintf(histo, "*/EcalEndcap/EEIntegrityTask/GainSwitch/EEIT gain switch %s", Numbers::sEE(ism).c_str());
        if ( qth03_[ism-1] ) mui_->useQTest(histo, qth03_[ism-1]->getName());
        sprintf(histo, "*/EcalEndcap/EEIntegrityTask/GainSwitchStay/EEIT gain switch stay %s", Numbers::sEE(ism).c_str());
        if ( qth04_[ism-1] ) mui_->useQTest(histo, qth04_[ism-1]->getName());
        sprintf(histo, "*/EcalEndcap/EEIntegrityTask/TTId/EEIT TTId %s", Numbers::sEE(ism).c_str());
        if ( qth05_[ism-1] ) mui_->useQTest(histo, qth05_[ism-1]->getName());
        sprintf(histo, "*/EcalEndcap/EEIntegrityTask/TTBlockSize/EEIT TTBlockSize %s", Numbers::sEE(ism).c_str());
        if ( qth06_[ism-1] ) mui_->useQTest(histo, qth06_[ism-1]->getName());
        sprintf(histo, "*/EcalEndcap/EEIntegrityTask/MemChId/EEIT MemChId %s", Numbers::sEE(ism).c_str());
        if ( qth07_[ism-1] ) mui_->useQTest(histo, qth07_[ism-1]->getName());
        sprintf(histo, "*/EcalEndcap/EEIntegrityTask/MemGain %s", Numbers::sEE(ism).c_str());
        if ( qth08_[ism-1] ) mui_->useQTest(histo, qth08_[ism-1]->getName());
        sprintf(histo, "*/EcalEndcap/EEIntegrityTask/MemTTId/EEIT MemTTId %s", Numbers::sEE(ism).c_str());
        if ( qth09_[ism-1] ) mui_->useQTest(histo, qth09_[ism-1]->getName());
        sprintf(histo, "*/EcalEndcap/EEIntegrityTask/MemSize/EEIT MemSize %s", Numbers::sEE(ism).c_str());
        if ( qth10_[ism-1] ) mui_->useQTest(histo, qth10_[ism-1]->getName());
      } else {
        sprintf(histo, "EcalEndcap/EEIntegrityTask/Gain/EEIT gain %s", Numbers::sEE(ism).c_str());
        if ( qth01_[ism-1] ) mui_->useQTest(histo, qth01_[ism-1]->getName());
        sprintf(histo, "EcalEndcap/EEIntegrityTask/ChId/EEIT ChId %s", Numbers::sEE(ism).c_str());
        if ( qth02_[ism-1] ) mui_->useQTest(histo, qth02_[ism-1]->getName());
        sprintf(histo, "EcalEndcap/EEIntegrityTask/GainSwitch/EEIT gain switch %s", Numbers::sEE(ism).c_str());
        if ( qth03_[ism-1] ) mui_->useQTest(histo, qth03_[ism-1]->getName());
        sprintf(histo, "EcalEndcap/EEIntegrityTask/GainSwitchStay/EEIT gain switch stay %s", Numbers::sEE(ism).c_str());
        if ( qth04_[ism-1] ) mui_->useQTest(histo, qth04_[ism-1]->getName());
        sprintf(histo, "EcalEndcap/EEIntegrityTask/TTId/EEIT TTId %s", Numbers::sEE(ism).c_str());
        if ( qth05_[ism-1] ) mui_->useQTest(histo, qth05_[ism-1]->getName());
        sprintf(histo, "EcalEndcap/EEIntegrityTask/TTBlockSize/EEIT TTBlockSize %s", Numbers::sEE(ism).c_str());
        if ( qth06_[ism-1] ) mui_->useQTest(histo, qth06_[ism-1]->getName());
        sprintf(histo, "EcalEndcap/EEIntegrityTask/MemChId/EEIT MemChId %s", Numbers::sEE(ism).c_str());
        if ( qth07_[ism-1] ) mui_->useQTest(histo, qth07_[ism-1]->getName());
        sprintf(histo, "EcalEndcap/EEIntegrityTask/MemGain %s", Numbers::sEE(ism).c_str());
        if ( qth08_[ism-1] ) mui_->useQTest(histo, qth08_[ism-1]->getName());
        sprintf(histo, "EcalEndcap/EEIntegrityTask/MemTTId/EEIT MemTTId %s", Numbers::sEE(ism).c_str());
        if ( qth09_[ism-1] ) mui_->useQTest(histo, qth09_[ism-1]->getName());
        sprintf(histo, "EcalEndcap/EEIntegrityTask/MemSize/EEIT MemSize %s", Numbers::sEE(ism).c_str());
        if ( qth10_[ism-1] ) mui_->useQTest(histo, qth10_[ism-1]->getName());
      }
    }

    sprintf(histo, "EcalEndcap/EEIntegrityTask/EEIT data integrity quality %s", Numbers::sEE(ism).c_str());
    if ( qtg01_[ism-1] ) mui_->useQTest(histo, qtg01_[ism-1]->getName());

    sprintf(histo, "EcalEndcap/EEIntegrityTask/EEIT data integrity quality MEM %s", Numbers::sEE(ism).c_str());
    if ( qtg02_[ism-1] ) mui_->useQTest(histo, qtg02_[ism-1]->getName());

  }

}

void EEIntegrityClient::subscribeNew(void){

  Char_t histo[200];

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    unsigned int ism = superModules_[i];

    sprintf(histo, "*/EcalEndcap/EEOccupancyTask/EEOT occupancy %s", Numbers::sEE(ism).c_str());
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalEndcap/EEOccupancyTask/EEOT MEM occupancy %s", Numbers::sEE(ism).c_str());
    mui_->subscribeNew(histo, ism);

    sprintf(histo, "*/EcalEndcap/EEIntegrityTask/EEIT DCC size error");
    mui_->subscribeNew(histo);
    sprintf(histo, "*/EcalEndcap/EEIntegrityTask/Gain/EEIT gain %s", Numbers::sEE(ism).c_str());
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalEndcap/EEIntegrityTask/ChId/EEIT ChId %s", Numbers::sEE(ism).c_str());
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalEndcap/EEIntegrityTask/GainSwitch/EEIT gain switch %s", Numbers::sEE(ism).c_str());
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalEndcap/EEIntegrityTask/GainSwitchStay/EEIT gain switch stay %s", Numbers::sEE(ism).c_str());
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalEndcap/EEIntegrityTask/TTId/EEIT TTId %s", Numbers::sEE(ism).c_str());
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalEndcap/EEIntegrityTask/TTBlockSize/EEIT TTBlockSize %s", Numbers::sEE(ism).c_str());
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalEndcap/EEIntegrityTask/MemChId/EEIT MemChId %s", Numbers::sEE(ism).c_str());
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalEndcap/EEIntegrityTask/MemGain/EEIT MemGain %s", Numbers::sEE(ism).c_str());
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalEndcap/EEIntegrityTask/MemTTId/EEIT MemTTId %s", Numbers::sEE(ism).c_str());
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalEndcap/EEIntegrityTask/MemSize/EEIT MemSize %s", Numbers::sEE(ism).c_str());
    mui_->subscribeNew(histo, ism);

  }

}

void EEIntegrityClient::unsubscribe(void){

  if ( verbose_ ) cout << "EEIntegrityClient: unsubscribe" << endl;

  if ( collateSources_ ) {

    if ( verbose_ ) cout << "EEIntegrityClient: uncollate" << endl;

    if ( mui_ ) {

      mui_->removeCollate(me_h00_);

      for ( unsigned int i=0; i<superModules_.size(); i++ ) {

        int ism = superModules_[i];

        mui_->removeCollate(me_h_[ism-1]);
        mui_->removeCollate(me_hmem_[ism-1]);

        mui_->removeCollate(me_h01_[ism-1]);
        mui_->removeCollate(me_h02_[ism-1]);
        mui_->removeCollate(me_h03_[ism-1]);
        mui_->removeCollate(me_h04_[ism-1]);
        mui_->removeCollate(me_h05_[ism-1]);
        mui_->removeCollate(me_h06_[ism-1]);
        mui_->removeCollate(me_h07_[ism-1]);
        mui_->removeCollate(me_h08_[ism-1]);
        mui_->removeCollate(me_h09_[ism-1]);
        mui_->removeCollate(me_h10_[ism-1]);

      }

    }

  }

  Char_t histo[200];

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    unsigned int ism = superModules_[i];

    sprintf(histo, "*/EcalEndcap/EEOccupancyTask/EEOT occupancy %s", Numbers::sEE(ism).c_str());
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EEOccupancyTask/EEOT MEM occupancy %s", Numbers::sEE(ism).c_str());
    mui_->unsubscribe(histo, ism);

    sprintf(histo, "*/EcalEndcap/EEIntegrityTask/EEIT DCC size error");
    mui_->unsubscribe(histo);
    sprintf(histo, "*/EcalEndcap/EEIntegrityTask/Gain/EEIT gain %s", Numbers::sEE(ism).c_str());
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EEIntegrityTask/ChId/EEIT ChId %s", Numbers::sEE(ism).c_str());
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EEIntegrityTask/GainSwitch/EEIT gain switch %s", Numbers::sEE(ism).c_str());
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EEIntegrityTask/GainSwitchStay/EEIT gain switch stay %s", Numbers::sEE(ism).c_str());
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EEIntegrityTask/TTId/EEIT TTId %s", Numbers::sEE(ism).c_str());
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EEIntegrityTask/TTBlockSize/EEIT TTBlockSize %s", Numbers::sEE(ism).c_str());
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EEIntegrityTask/MemChId/EEIT MemChId %s", Numbers::sEE(ism).c_str());
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EEIntegrityTask/MemGain/EEIT MemGain %s", Numbers::sEE(ism).c_str());
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EEIntegrityTask/MemTTId/EEIT MemTTId %s", Numbers::sEE(ism).c_str());
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EEIntegrityTask/MemSize/EEIT MemSize %s", Numbers::sEE(ism).c_str());
    mui_->unsubscribe(histo, ism);

  }

}

void EEIntegrityClient::softReset(void){

}

void EEIntegrityClient::analyze(void){

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( verbose_ ) cout << "EEIntegrityClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;
  }

  uint64_t bits01 = 0;
  bits01 |= EcalErrorDictionary::getMask("CH_ID_WARNING");
  bits01 |= EcalErrorDictionary::getMask("CH_GAIN_ZERO_WARNING");
  bits01 |= EcalErrorDictionary::getMask("CH_GAIN_SWITCH_WARNING");
  bits01 |= EcalErrorDictionary::getMask("CH_ID_ERROR");
  bits01 |= EcalErrorDictionary::getMask("CH_GAIN_ZERO_ERROR");
  bits01 |= EcalErrorDictionary::getMask("CH_GAIN_SWITCH_ERROR");

  uint64_t bits02 = 0;
  bits02 |= EcalErrorDictionary::getMask("TT_ID_WARNING");
  bits02 |= EcalErrorDictionary::getMask("TT_SIZE_WARNING");
  bits02 |= EcalErrorDictionary::getMask("TT_LV1_WARNING");
  bits02 |= EcalErrorDictionary::getMask("TT_BUNCH_X_WARNING");
  bits02 |= EcalErrorDictionary::getMask("TT_ID_ERROR");
  bits02 |= EcalErrorDictionary::getMask("TT_SIZE_ERROR");
  bits02 |= EcalErrorDictionary::getMask("TT_LV1_ERROR");
  bits02 |= EcalErrorDictionary::getMask("TT_BUNCH_X_ERROR");

  map<EcalLogicID, RunCrystalErrorsDat> mask1;
  map<EcalLogicID, RunTTErrorsDat> mask2;
  map<EcalLogicID, RunMemChErrorsDat> mask3;
  map<EcalLogicID, RunMemTTErrorsDat> mask4;

  EcalErrorMask::fetchDataSet(&mask1);
  EcalErrorMask::fetchDataSet(&mask2);
  EcalErrorMask::fetchDataSet(&mask3);
  EcalErrorMask::fetchDataSet(&mask4);

  Char_t histo[200];

  MonitorElement* me;

  if ( collateSources_ ) {
    sprintf(histo, "EcalEndcap/Sums/EEIntegrityTask/EEIT DCC size error");
  } else {
    sprintf(histo, (prefixME_+"EcalEndcap/EEIntegrityTask/EEIT DCC size error").c_str());
  }
  me = dbe_->get(histo);
  h00_ = UtilsClient::getHisto<TH1F*>( me, cloneME_, h00_ );

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( collateSources_ ) {
      sprintf(histo, "EcalEndcap/Sums/EEOccupancyTask/EEOT occupancy %s", Numbers::sEE(ism).c_str());
    } else {
      sprintf(histo, (prefixME_+"EcalEndcap/EEOccupancyTask/EEOT occupancy %s").c_str(), Numbers::sEE(ism).c_str());
    }
    me = dbe_->get(histo);
    h_[ism-1] = UtilsClient::getHisto<TH2F*>( me, cloneME_, h_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalEndcap/Sums/EEOccupancyTask/EEOT MEM occupancy %s", Numbers::sEE(ism).c_str());
    } else {
      sprintf(histo, (prefixME_+"EcalEndcap/EEOccupancyTask/EEOT MEM occupancy %s").c_str(), Numbers::sEE(ism).c_str());
    }
    me = dbe_->get(histo);
    hmem_[ism-1] = UtilsClient::getHisto<TH2F*>( me, cloneME_, hmem_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalEndcap/Sums/EEIntegrityTask/Gain/EEIT gain %s", Numbers::sEE(ism).c_str());
    } else {
      sprintf(histo, (prefixME_+"EcalEndcap/EEIntegrityTask/Gain/EEIT gain %s").c_str(), Numbers::sEE(ism).c_str());
    }
    me = dbe_->get(histo);
    h01_[ism-1] = UtilsClient::getHisto<TH2F*>( me, cloneME_, h01_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalEndcap/Sums/EEIntegrityTask/ChId/EEIT ChId %s", Numbers::sEE(ism).c_str());
    } else {
      sprintf(histo, (prefixME_+"EcalEndcap/EEIntegrityTask/ChId/EEIT ChId %s").c_str(), Numbers::sEE(ism).c_str());
    }
    me = dbe_->get(histo);
    h02_[ism-1] = UtilsClient::getHisto<TH2F*>( me, cloneME_, h02_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalEndcap/Sums/EEIntegrityTask/GainSwitch/EEIT gain switch %s", Numbers::sEE(ism).c_str());
    } else {
      sprintf(histo, (prefixME_+"EcalEndcap/EEIntegrityTask/GainSwitch/EEIT gain switch %s").c_str(), Numbers::sEE(ism).c_str());
    }
    me = dbe_->get(histo);
    h03_[ism-1] = UtilsClient::getHisto<TH2F*>( me, cloneME_, h03_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalEndcap/Sums/EEIntegrityTask/GainSwitchStay/EEIT gain switch stay %s", Numbers::sEE(ism).c_str());
    } else {
      sprintf(histo, (prefixME_+"EcalEndcap/EEIntegrityTask/GainSwitchStay/EEIT gain switch stay %s").c_str(), Numbers::sEE(ism).c_str());
    }
    me = dbe_->get(histo);
    h04_[ism-1] = UtilsClient::getHisto<TH2F*>( me, cloneME_, h04_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalEndcap/Sums/EEIntegrityTask/TTId/EEIT TTId %s", Numbers::sEE(ism).c_str());
    } else {
      sprintf(histo, (prefixME_+"EcalEndcap/EEIntegrityTask/TTId/EEIT TTId %s").c_str(), Numbers::sEE(ism).c_str());
    }
    me = dbe_->get(histo);
    h05_[ism-1] = UtilsClient::getHisto<TH2F*>( me, cloneME_, h05_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalEndcap/Sums/EEIntegrityTask/TTBlockSize/EEIT TTBlockSize %s", Numbers::sEE(ism).c_str());
    } else {
      sprintf(histo, (prefixME_+"EcalEndcap/EEIntegrityTask/TTBlockSize/EEIT TTBlockSize %s").c_str(), Numbers::sEE(ism).c_str());
    }
    me = dbe_->get(histo);
    h06_[ism-1] = UtilsClient::getHisto<TH2F*>( me, cloneME_, h06_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalEndcap/Sums/EEIntegrityTask/MemChId/EEIT MemChId %s", Numbers::sEE(ism).c_str());
    } else {
      sprintf(histo, (prefixME_+"EcalEndcap/EEIntegrityTask/MemChId/EEIT MemChId %s").c_str(), Numbers::sEE(ism).c_str());
    }
    me = dbe_->get(histo);
    h07_[ism-1] = UtilsClient::getHisto<TH2F*>( me, cloneME_, h07_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalEndcap/Sums/EEIntegrityTask/MemGain/EEIT MemGain %s", Numbers::sEE(ism).c_str());
    } else {
      sprintf(histo, (prefixME_+"EcalEndcap/EEIntegrityTask/MemGain/EEIT MemGain %s").c_str(), Numbers::sEE(ism).c_str());
    }
    me = dbe_->get(histo);
    h08_[ism-1] = UtilsClient::getHisto<TH2F*>( me, cloneME_, h08_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalEndcap/Sums/EEIntegrityTask/MemTTId/EEIT MemTTId %s", Numbers::sEE(ism).c_str());
    } else {
      sprintf(histo, (prefixME_+"EcalEndcap/EEIntegrityTask/MemTTId/EEIT MemTTId %s").c_str(), Numbers::sEE(ism).c_str());
    }
    me = dbe_->get(histo);
    h09_[ism-1] = UtilsClient::getHisto<TH2F*>( me, cloneME_, h09_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalEndcap/Sums/EEIntegrityTask/MemSize/EEIT MemSize %s", Numbers::sEE(ism).c_str());
    } else {
      sprintf(histo, (prefixME_+"EcalEndcap/EEIntegrityTask/MemSize/EEIT MemSize %s").c_str(), Numbers::sEE(ism).c_str());
    }
    me = dbe_->get(histo);
    h10_[ism-1] = UtilsClient::getHisto<TH2F*>( me, cloneME_, h10_[ism-1] );

    float num00;

    // integrity summary histograms
    UtilsClient::resetHisto( meg01_[ism-1] );
    UtilsClient::resetHisto( meg02_[ism-1] );

    num00 = 0.;

    bool update0 = false;

    // dcc size errors
    if ( h00_ ) {
      num00  = h00_->GetBinContent(ism);
      update0 = true;
    }

    float num01, num02, num03, num04, num05, num06;

    for ( int ix = 1; ix <= 50; ix++ ) {
      for ( int iy = 1; iy <= 50; iy++ ) {

        num01 = num02 = num03 = num04 = num05 = num06 = 0.;

        if ( meg01_[ism-1] ) meg01_[ism-1]->setBinContent( ix, iy, -1. );

        bool update1 = false;
        bool update2 = false;

        float numTot = -1.;

        if ( h_[ism-1] ) numTot = h_[ism-1]->GetBinContent(ix, iy);

        if ( h01_[ism-1] ) {
          num01  = h01_[ism-1]->GetBinContent(ix, iy);
          update1 = true;
        }

        if ( h02_[ism-1] ) {
          num02  = h02_[ism-1]->GetBinContent(ix, iy);
          update1 = true;
        }

        if ( h03_[ism-1] ) {
          num03  = h03_[ism-1]->GetBinContent(ix, iy);
          update1 = true;
        }

        if ( h04_[ism-1] ) {
          num04  = h04_[ism-1]->GetBinContent(ix, iy);
          update1 = true;
        }

        int iet = 1 + ((ix-1)/5);
        int ipt = 1 + ((iy-1)/5);

        if ( h05_[ism-1] ) {
          num05  = h05_[ism-1]->GetBinContent(iet, ipt);
          update2 = true;
        }

        if ( h06_[ism-1] ) {
          num06  = h06_[ism-1]->GetBinContent(iet, ipt);
          update2 = true;
        }

        if ( update0 || update1 || update2 ) {

          float val;

          val = 1.;
          // numer of events on a channel
          if ( numTot > 0 ) {
            float errorRate1 =  num00 / numTot;
            if ( errorRate1 > threshCry_ )
              val = 0.;
            errorRate1 = ( num01 + num02 + num03 + num04 ) / numTot / 4.;
            if ( errorRate1 > threshCry_ )
              val = 0.;
            float errorRate2 = ( num05 + num06 ) / numTot / 2.;
            if ( errorRate2 > threshCry_ )
              val = 0.;
          } else {
            val = 2.;
            if ( num00 > 0 )
              val = 0.;
            if ( ( num01 + num02 + num03 + num04 ) > 0 )
              val = 0.;
            if ( ( num05 + num06 ) > 0 )
              val = 0.;
          }

          int jx = ix + Numbers::ix0EE(ism);
          int jy = iy + Numbers::iy0EE(ism);

          // filling the summary for SM channels
          if ( Numbers::validEE(ism, 101 - jx, jy) ) {
            if ( meg01_[ism-1] ) meg01_[ism-1]->setBinContent( ix, iy, val );
          }

        }

        // masking

        if ( mask1.size() != 0 ) {
          map<EcalLogicID, RunCrystalErrorsDat>::const_iterator m;
          for (m = mask1.begin(); m != mask1.end(); m++) {

            int jx = ix + Numbers::ix0EE(ism);
            int jy = iy + Numbers::iy0EE(ism);
 
            if ( ! Numbers::validEE(ism, 101 - jx, jy) ) continue;

            int ic = Numbers::icEE(ism, ix, iy);

            if ( ic == -1 ) continue;

            EcalLogicID ecid = m->first;

            if ( ecid.getID1() == Numbers::iSM(ism, EcalEndcap) && ecid.getID2() == ic ) {
              if ( (m->second).getErrorBits() & bits01 ) {
                if ( meg01_[ism-1] ) {
                  float val = int(meg01_[ism-1]->getBinContent(ix, iy)) % 3;
                  meg01_[ism-1]->setBinContent( ix, iy, val+3 );
                }
              }
            }

          }
        }

        if ( mask2.size() != 0 ) {
          map<EcalLogicID, RunTTErrorsDat>::const_iterator m;
          for (m = mask2.begin(); m != mask2.end(); m++) {

            EcalLogicID ecid = m->first;

            int iet = 1 + ((ix-1)/5);
            int ipt = 1 + ((iy-1)/5);
            int itt = (ipt-1) + 4*(iet-1) + 1;

            if ( ecid.getID1() == Numbers::iSM(ism, EcalEndcap) && ecid.getID2() == itt ) {
              if ( (m->second).getErrorBits() & bits02 ) {
                if ( meg01_[ism-1] ) {
                  float val = int(meg01_[ism-1]->getBinContent(ix, iy)) % 3;
                  meg01_[ism-1]->setBinContent( ix, iy, val+3 );
                }
              }
            }

          }
        }

      }
    }// end of loop on crystals to fill summary plot

    vector<dqm::me_util::Channel> badChannels01;
    vector<dqm::me_util::Channel> badChannels02;
    vector<dqm::me_util::Channel> badChannels03;
    vector<dqm::me_util::Channel> badChannels04;
    vector<dqm::me_util::Channel> badChannels05;
    vector<dqm::me_util::Channel> badChannels06;

    if ( qth01_[ism-1] ) badChannels01 = qth01_[ism-1]->getBadChannels();
    if ( qth02_[ism-1] ) badChannels02 = qth02_[ism-1]->getBadChannels();
    if ( qth03_[ism-1] ) badChannels03 = qth03_[ism-1]->getBadChannels();
    if ( qth04_[ism-1] ) badChannels04 = qth04_[ism-1]->getBadChannels();
    if ( qth05_[ism-1] ) badChannels05 = qth05_[ism-1]->getBadChannels();
    if ( qth06_[ism-1] ) badChannels06 = qth06_[ism-1]->getBadChannels();

    // summaries for mem channels
    float num07, num08, num09, num10;

    for ( int ie = 1; ie <= 10; ie++ ) {
      for ( int ip = 1; ip <= 5; ip++ ) {

        num07 = num08 = num09 = num10 = 0.;

        // initialize summary histo for mem
        if ( meg02_[ism-1] ) meg02_[ism-1]->setBinContent( ie, ip, 2. );

        bool update1 = false;
        bool update2 = false;

        float numTotmem = -1.;

        if ( hmem_[ism-1] ) numTotmem = hmem_[ism-1]->GetBinContent(ie, ip);

        if ( h07_[ism-1] ) {
          num07  = h07_[ism-1]->GetBinContent(ie, ip);
          update1 = true;
        }

        if ( h08_[ism-1] ) {
          num08  = h08_[ism-1]->GetBinContent(ie, ip);
          update1 = true;
        }

        int iet = 1 + ((ie-1)/5);
        int ipt = 1;

        if ( h09_[ism-1] ) {
          num09  = h09_[ism-1]->GetBinContent(iet, ipt);
          update2 = true;
        }

        if ( h10_[ism-1] ) {
          num10  = h10_[ism-1]->GetBinContent(iet, ipt);
          update2 = true;
        }


        if ( update0 || update1 || update2 ) {

          float val;

          val = 1.;
          // numer of events on a channel
          if ( numTotmem > 0 ) {
            float errorRate1 = ( num07 + num08 ) / numTotmem / 2.;
            if ( errorRate1 > threshCry_ )
              val = 0.;
            float errorRate2 = ( num09 + num10 ) / numTotmem / 2.;
            if ( errorRate2 > threshCry_ )
              val = 0.;
          } else {
            val = 2.;
            if ( ( num07 + num08 ) > 0 )
              val = 0.;
            if ( ( num09 + num10 ) > 0 )
              val = 0.;
          }

          // filling summary for mem channels
          if ( meg02_[ism-1] ) meg02_[ism-1]->setBinContent( ie, ip, val );

        }

        // masking

        if ( mask3.size() != 0 ) {
          map<EcalLogicID, RunMemChErrorsDat>::const_iterator m;
          for (m = mask3.begin(); m != mask3.end(); m++) {

            EcalLogicID ecid = m->first;

            int ic = EEIntegrityClient::chNum[ (ie-1)%5 ][ (ip-1) ] + (ie-1)/5 * 25;

            if ( ecid.getID1() == Numbers::iSM(ism, EcalEndcap) && ecid.getID2() == ic ) {
              if ( (m->second).getErrorBits() & bits01 ) {
                if ( meg02_[ism-1] ) {
                  float val = int(meg02_[ism-1]->getBinContent(ie, ip)) % 3;
                  meg02_[ism-1]->setBinContent( ie, ip, val+3 );
                }
              }
            }
          }
        }

        if ( mask4.size() != 0 ) {
          map<EcalLogicID, RunMemTTErrorsDat>::const_iterator m;
          for (m = mask4.begin(); m != mask4.end(); m++) {

            EcalLogicID ecid = m->first;

            int iet = 1 + ((ie-1)/5);
            int itt = 68 + iet;

            if ( ecid.getID1() == Numbers::iSM(ism, EcalEndcap) && ecid.getID2() == itt ) {
              if ( (m->second).getErrorBits() & bits02 ) {
                if ( meg02_[ism-1] ) {
                  float val = int(meg02_[ism-1]->getBinContent(ie, ip)) % 3;
                  meg02_[ism-1]->setBinContent( ie, ip, val+3 );
                }
              }
            }
          }
        }

      }
    }  // end loop on mem channels

    vector<dqm::me_util::Channel> badChannels07;
    vector<dqm::me_util::Channel> badChannels08;
    vector<dqm::me_util::Channel> badChannels09;
    vector<dqm::me_util::Channel> badChannels10;

    if ( qth07_[ism-1] ) badChannels01 = qth07_[ism-1]->getBadChannels();
    if ( qth08_[ism-1] ) badChannels02 = qth08_[ism-1]->getBadChannels();
    if ( qth09_[ism-1] ) badChannels03 = qth09_[ism-1]->getBadChannels();
    if ( qth10_[ism-1] ) badChannels04 = qth10_[ism-1]->getBadChannels();

  }// end loop on supermodules

}

void EEIntegrityClient::htmlOutput(int run, string htmlDir, string htmlName){

  cout << "Preparing EEIntegrityClient html output ..." << endl;

  ofstream htmlFile;

  htmlFile.open((htmlDir + htmlName).c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlFile << "<html>  " << endl;
  htmlFile << "<head>  " << endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlFile << " http-equiv=\"content-type\">  " << endl;
  htmlFile << "  <title>Monitor:IntegrityTask output</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlFile << "<body>  " << endl;
  //htmlFile << "<br>  " << endl;
  htmlFile << "<a name=""top""></a>" << endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << run << "</span></h2>" << endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">INTEGRITY</span></h2> " << endl;
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
  htmlFile << "<hr>" << std::endl;

  // Produce the plots to be shown as .png files from existing histograms

  const int csize = 250;

  int pCol3[6] = { 301, 302, 303, 304, 305, 306 };
  int pCol4[10];
  for ( int i = 0; i < 10; i++ ) pCol4[i] = 401+i;

  TH2C dummy3( "dummy3", "dummy3 for sm mem", 10, 0, 10, 5, 0, 5 );
  for ( short i=0; i<2; i++ ) {
    int a = 2 + i*5;
    int b = 2;
    dummy3.Fill( a, b, i+1+68 );
  }
  dummy3.SetMarkerSize(2);
  dummy3.SetMinimum(0.1);

  TH2C dummy4 ("dummy4", "dummy4 for sm mem", 2, 0, 2, 1, 0, 1 );
  for ( short i=0; i<2; i++ ) {
    int a =  i;
    int b = 0;
    dummy4.Fill( a, b, i+1+68 );
  }
  dummy4.SetMarkerSize(2);
  dummy4.SetMinimum(0.1);

  string imgNameDCC, imgNameOcc, imgNameQual,imgNameOccMem, imgNameQualMem, imgNameME[10], imgName, meName;

  TCanvas* cDCC = new TCanvas("cDCC", "Temp", 2*csize, csize);
  TCanvas* cOcc = new TCanvas("cOcc", "Temp", 2*csize, 2*csize);
  TCanvas* cQual = new TCanvas("cQual", "Temp", 2*csize, 2*csize);
  TCanvas* cMe = new TCanvas("cMe", "Temp", 2*csize, 2*csize);
  TCanvas* cMeMem = new TCanvas("cMeMem", "Temp", 2*csize, csize);

  TH1F* obj1f;
  TH2F* obj2f;

  // DCC size error

  imgNameDCC = "";

  obj1f = h00_;

  if ( obj1f ) {

    meName = obj1f->GetName();

    for ( unsigned int i = 0; i < meName.size(); i++ ) {
      if ( meName.substr(i, 1) == " " )  {
        meName.replace(i, 1, "_");
      }
    }
    imgNameDCC = meName + ".png";
    imgName = htmlDir + imgNameDCC;

    cDCC->cd();
    gStyle->SetOptStat(" ");
    obj1f->Draw();
    cDCC->Update();
    cDCC->SaveAs(imgName.c_str());

  }

  htmlFile << "<h3><strong>DCC size error</strong></h3>" << endl;

  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\"> " << endl;
  htmlFile << "<tr align=\"left\">" << endl;

  if ( imgNameDCC.size() != 0 )
    htmlFile << "<td><img src=\"" << imgNameDCC << "\"></td>" << endl;
  else
    htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

  htmlFile << "</tr>" << endl;
  htmlFile << "</table>" << endl;
  htmlFile << "<br>" << endl;

  // Loop on barrel supermodules

  for ( unsigned int i=0; i<superModules_.size(); i ++ ) {

    int ism = superModules_[i];

    // Quality plots

    imgNameQual = "";

    obj2f = UtilsClient::getHisto<TH2F*>( meg01_[ism-1] );

    if ( obj2f ) {

      meName = obj2f->GetName();

      for ( unsigned int i = 0; i < meName.size(); i++ ) {
        if ( meName.substr(i, 1) == " " )  {
          meName.replace(i, 1, "_");
        }
      }
      imgNameQual = meName + ".png";
      imgName = htmlDir + imgNameQual;

      cQual->cd();
      gStyle->SetOptStat(" ");
      gStyle->SetPalette(6, pCol3);
      cQual->SetGridx();
      cQual->SetGridy();
      obj2f->SetMinimum(-0.00000001);
      obj2f->SetMaximum(6.0);
      obj2f->GetXaxis()->SetLabelSize(0.02);
      obj2f->GetYaxis()->SetLabelSize(0.02);
      obj2f->Draw("col");
      cQual->SetBit(TGraph::kClipFrame);
      TLine l;
      l.SetLineWidth(1);
      for ( int i=0; i<201; i=i+1){
        if ( (Numbers::ixSectorsEE[i]!=0 || Numbers::iySectorsEE[i]!=0) && (Numbers::ixSectorsEE[i+1]!=0 || Numbers::iySectorsEE[i+1]!=0) ) {
          l.DrawLine(Numbers::ixSectorsEE[i], Numbers::iySectorsEE[i], Numbers::ixSectorsEE[i+1], Numbers::iySectorsEE[i+1]);
        }
      }
      cQual->Update();
      cQual->SaveAs(imgName.c_str());

    }

    // Occupancy plots

    imgNameOcc = "";

    obj2f = h_[ism-1];

    if ( obj2f ) {

      meName = obj2f->GetName();

      for ( unsigned int i = 0; i < meName.size(); i++ ) {
        if ( meName.substr(i, 1) == " " )  {
          meName.replace(i, 1, "_");
        }
      }

      imgNameOcc = meName + ".png";
      imgName = htmlDir + imgNameOcc;

      cOcc->cd();
      gStyle->SetOptStat(" ");
      gStyle->SetPalette(10, pCol4);
      cOcc->SetGridx();
      cOcc->SetGridy();
      obj2f->GetXaxis()->SetLabelSize(0.02);
      obj2f->GetYaxis()->SetLabelSize(0.02);
      obj2f->GetZaxis()->SetLabelSize(0.02);
      obj2f->SetMinimum(0.0);
      obj2f->Draw("colz");
      cOcc->SetBit(TGraph::kClipFrame);
      TLine l;
      l.SetLineWidth(1);
      for ( int i=0; i<201; i=i+1){
        if ( (Numbers::ixSectorsEE[i]!=0 || Numbers::iySectorsEE[i]!=0) && (Numbers::ixSectorsEE[i+1]!=0 || Numbers::iySectorsEE[i+1]!=0) ) {
          l.DrawLine(Numbers::ixSectorsEE[i], Numbers::iySectorsEE[i], Numbers::ixSectorsEE[i+1], Numbers::iySectorsEE[i+1]);
        }
      }
      cOcc->Update();
      cOcc->SaveAs(imgName.c_str());

    }

    // Monitoring elements plots

    for ( int iCanvas = 1; iCanvas <= 6; iCanvas++ ) {

      imgNameME[iCanvas-1] = "";

      obj2f = 0;
      switch ( iCanvas ) {
      case 1:
        obj2f = h01_[ism-1];
        break;
      case 2:
        obj2f = h02_[ism-1];
        break;
      case 3:
        obj2f = h03_[ism-1];
        break;
      case 4:
        obj2f = h04_[ism-1];
        break;
      case 5:
        obj2f = h05_[ism-1];
        break;
      case 6:
        obj2f = h06_[ism-1];
        break;
      default:
        break;
      }

      if ( obj2f ) {

        meName = obj2f->GetName();

        for ( unsigned int iMe = 0; iMe < meName.size(); iMe++ ) {
          if ( meName.substr(iMe, 1) == " " )  {
            meName.replace(iMe, 1, "_");
          }
        }
        imgNameME[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgNameME[iCanvas-1];

        cMe->cd();
        gStyle->SetOptStat(" ");
        gStyle->SetPalette(10, pCol4);
        cMe->SetGridx();
        cMe->SetGridy();
        obj2f->GetXaxis()->SetLabelSize(0.02);
        obj2f->GetYaxis()->SetLabelSize(0.02);
        obj2f->GetZaxis()->SetLabelSize(0.02);
        obj2f->SetMinimum(0.0);
        obj2f->Draw("colz");
        cMe->SetBit(TGraph::kClipFrame);
        if ( iCanvas > 4 ) {
          TLine l;
          l.SetLineWidth(1);
          for ( int i=0; i<201; i=i+1){
            if ( (Numbers::ixSectorsEE[i]!=0 || Numbers::iySectorsEE[i]!=0) && (Numbers::ixSectorsEE[i+1]!=0 || Numbers::iySectorsEE[i+1]!=0) ) {
              l.DrawLine(Numbers::ixSectorsEE[i]/5, Numbers::iySectorsEE[i]/5, Numbers::ixSectorsEE[i+1]/5, Numbers::iySectorsEE[i+1]/5);
            }
          }
        } else {
          TLine l;
          l.SetLineWidth(1);
          for ( int i=0; i<201; i=i+1){
            if ( (Numbers::ixSectorsEE[i]!=0 || Numbers::iySectorsEE[i]!=0) && (Numbers::ixSectorsEE[i+1]!=0 || Numbers::iySectorsEE[i+1]!=0) ) {
              l.DrawLine(Numbers::ixSectorsEE[i], Numbers::iySectorsEE[i], Numbers::ixSectorsEE[i+1], Numbers::iySectorsEE[i+1]);
            }
          }
        }
        cMe->Update();
        cMe->SaveAs(imgName.c_str());

      }

    }

    // MEM Quality plots

    imgNameQualMem = "";

    obj2f = UtilsClient::getHisto<TH2F*>( meg02_[ism-1] );

    if ( obj2f ) {

      meName = obj2f->GetName();

      for ( unsigned int i = 0; i < meName.size(); i++ ) {
        if ( meName.substr(i, 1) == " " )  {
          meName.replace(i, 1, "_");
        }
      }
      imgNameQualMem = meName + ".png";
      imgName = htmlDir + imgNameQualMem;

      cMeMem->cd();
      gStyle->SetOptStat(" ");
      gStyle->SetPalette(6, pCol3);
      obj2f->GetXaxis()->SetNdivisions(10);
      obj2f->GetYaxis()->SetNdivisions(5);
      cMeMem->SetGridx();
      cMeMem->SetGridy(0);
      obj2f->SetMinimum(-0.00000001);
      obj2f->SetMaximum(6.0);
      obj2f->Draw("col");
      dummy3.Draw("text,same");
      cMeMem->Update();
      cMeMem->SaveAs(imgName.c_str());

    }

    // MEM Occupancy plots

    imgNameOccMem = "";

    obj2f = hmem_[ism-1];

    if ( obj2f ) {

      meName = obj2f->GetName();

      for ( unsigned int i = 0; i < meName.size(); i++ ) {
        if ( meName.substr(i, 1) == " " )  {
          meName.replace(i, 1, "_");
        }
      }

      imgNameOccMem = meName + ".png";
      imgName = htmlDir + imgNameOccMem;

      cMeMem->cd();
      gStyle->SetOptStat(" ");
      gStyle->SetPalette(10, pCol4);
      obj2f->GetXaxis()->SetNdivisions(10);
      obj2f->GetYaxis()->SetNdivisions(5);
      cMeMem->SetGridx();
      cMeMem->SetGridy(0);
      obj2f->GetXaxis()->SetLabelSize(0.02);
      obj2f->GetYaxis()->SetLabelSize(0.02);
      obj2f->GetZaxis()->SetLabelSize(0.02);
      obj2f->SetMinimum(0.0);
      obj2f->Draw("colz");
      dummy3.Draw("text,same");
      cMeMem->Update();
      cMeMem->SaveAs(imgName.c_str());

    }

    // MeM Monitoring elements plots

    for ( int iCanvas = 7; iCanvas <= 10; iCanvas++ ) {

      imgNameME[iCanvas-1] = "";

      obj2f = 0;
      switch ( iCanvas ) {
      case 7:
        obj2f = h07_[ism-1];
        break;
      case 8:
        obj2f = h08_[ism-1];
        break;
      case 9:
        obj2f = h09_[ism-1];
        break;
      case 10:
        obj2f = h10_[ism-1];
        break;
      default:
        break;
      }

      if ( obj2f ) {

        meName = obj2f->GetName();

        for ( unsigned int iMe = 0; iMe < meName.size(); iMe++ ) {
          if ( meName.substr(iMe, 1) == " " )  {
            meName.replace(iMe, 1, "_");
          }
        }
        imgNameME[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgNameME[iCanvas-1];

        cMeMem->cd();
        gStyle->SetOptStat(" ");
        gStyle->SetPalette(10, pCol4);
        obj2f->SetMinimum(0.0);
        obj2f->Draw("colz");
        if ( iCanvas < 9 ){
          obj2f->GetXaxis()->SetNdivisions(10);
          obj2f->GetYaxis()->SetNdivisions(5);
          cMeMem->SetGridx();
          cMeMem->SetGridy(0);
          dummy3.Draw("text,same");
        }
        else{
          obj2f->GetXaxis()->SetNdivisions(2);
          obj2f->GetYaxis()->SetNdivisions(1);
          cMeMem->SetGridx();
          cMeMem->SetGridy();
          dummy4.Draw("text,same");
        }
        cMeMem->Update();
        cMeMem->SaveAs(imgName.c_str());

      }

    }

    if( i>0 ) htmlFile << "<a href=""#top"">Top</a>" << std::endl;
    htmlFile << "<hr>" << std::endl;
    htmlFile << "<h3><a name="""
	     << Numbers::sEE(ism).c_str() << """></a><strong>"
	     << Numbers::sEE(ism).c_str() << "</strong></h3>" << endl;
    htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\"> " << endl;
    htmlFile << "<tr align=\"left\">" << endl;

    if ( imgNameQual.size() != 0 )
      htmlFile << "<td><img src=\"" << imgNameQual << "\"></td>" << endl;
    else
      htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

    if ( imgNameOcc.size() != 0 )
      htmlFile << "<td><img src=\"" << imgNameOcc << "\"></td>" << endl;
    else
      htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

    htmlFile << "</tr>" << endl;
    htmlFile << "</table>" << endl;
    htmlFile << "<br>" << endl;

    htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
    htmlFile << "<tr align=\"center\">" << endl;

    for ( int iCanvas = 1 ; iCanvas <= 2 ; iCanvas++ ) {

      if ( imgNameME[iCanvas-1].size() != 0 )
        htmlFile << "<td><img src=\"" << imgNameME[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

    }

    htmlFile << "</tr>" << endl;
    htmlFile << "</table>" << endl;
    htmlFile << "<br>" << endl;

    htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
    htmlFile << "<tr align=\"center\">" << endl;

    for ( int iCanvas = 3 ; iCanvas <= 4 ; iCanvas++ ) {

      if ( imgNameME[iCanvas-1].size() != 0 )
        htmlFile << "<td><img src=\"" << imgNameME[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

    }

    htmlFile << "</tr>" << endl;
    htmlFile << "</table>" << endl;
    htmlFile << "<br>" << endl;

    htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
    htmlFile << "<tr align=\"center\">" << endl;

    for ( int iCanvas = 5 ; iCanvas <= 6 ; iCanvas++ ) {

      if ( imgNameME[iCanvas-1].size() != 0 )
        htmlFile << "<td><img src=\"" << imgNameME[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

    }

    htmlFile << "</tr>" << endl;
    htmlFile << "</table>" << endl;

    htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\"> " << endl;
    htmlFile << "<tr align=\"left\">" << endl;

    if ( imgNameQualMem.size() != 0 )
      htmlFile << "<td><img src=\"" << imgNameQualMem << "\"></td>" << endl;
    else
      htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

    if ( imgNameOccMem.size() != 0 )
      htmlFile << "<td><img src=\"" << imgNameOccMem << "\"></td>" << endl;
    else
      htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

    htmlFile << "</tr>" << endl;
    htmlFile << "</table>" << endl;
    htmlFile << "<br>" << endl;


    htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
    htmlFile << "<tr>" << endl;

    for ( int iCanvas = 7 ; iCanvas <= 8 ; iCanvas++ ) {

      if ( imgNameME[iCanvas-1].size() != 0 )
        htmlFile << "<td><img src=\"" << imgNameME[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

    }

    htmlFile << "</tr>" << endl;
    htmlFile << "</table>" << endl;
    htmlFile << "<br>" << endl;

    htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
    htmlFile << "<tr>" << endl;

    for ( int iCanvas = 9 ; iCanvas <= 10 ; iCanvas++ ) {

      if ( imgNameME[iCanvas-1].size() != 0 )
        htmlFile << "<td><img src=\"" << imgNameME[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

    }

    htmlFile << "</tr>" << endl;
    htmlFile << "</table>" << endl;
    htmlFile << "<br>" << endl;

  }

  delete cDCC;
  delete cOcc;
  delete cQual;
  delete cMe;
  delete cMeMem;

  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  htmlFile.close();

}

const int  EEIntegrityClient::chNum [5][5] = {
  { 1,  2,  3,  4,  5},
  {10,  9,  8,  7,  6},
  {11, 12, 13, 14, 15},
  {20, 19, 18, 17, 16},
  {21, 22, 23, 24, 25}
};

