
/*
 * \file EBIntegrityClient.cc
 *
 * $Date: 2007/08/09 12:24:18 $
 * $Revision: 1.153 $
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

#include <DQM/EcalBarrelMonitorClient/interface/EBIntegrityClient.h>

using namespace cms;
using namespace edm;
using namespace std;

EBIntegrityClient::EBIntegrityClient(const ParameterSet& ps){

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

  // vector of selected Super Modules (Defaults to all 36).
  superModules_.reserve(36);
  for ( unsigned int i = 1; i < 37; i++ ) superModules_.push_back(i);
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

EBIntegrityClient::~EBIntegrityClient(){

}

void EBIntegrityClient::beginJob(MonitorUserInterface* mui){

  mui_ = mui;

  if ( verbose_ ) cout << "EBIntegrityClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

  if ( enableQT_ ) {

    Char_t qtname[200];

    for ( unsigned int i=0; i<superModules_.size(); i++ ) {

      int ism = superModules_[i];

      sprintf(qtname, "EBIT data integrity quality gain %s", Numbers::sEB(ism).c_str());
      qth01_[ism-1] = dynamic_cast<MEContentsTH2FWithinRangeROOT*> (mui_->createQTest(ContentsTH2FWithinRangeROOT::getAlgoName(), qtname));

      sprintf(qtname, "EBIT data integrity quality ChId %s", Numbers::sEB(ism).c_str());
      qth02_[ism-1] = dynamic_cast<MEContentsTH2FWithinRangeROOT*> (mui_->createQTest(ContentsTH2FWithinRangeROOT::getAlgoName(), qtname));

      sprintf(qtname, "EBIT data integrity quality gain switch %s", Numbers::sEB(ism).c_str());
      qth03_[ism-1] = dynamic_cast<MEContentsTH2FWithinRangeROOT*> (mui_->createQTest(ContentsTH2FWithinRangeROOT::getAlgoName(), qtname));

      sprintf(qtname, "EBIT data integrity quality gain switch stay %s", Numbers::sEB(ism).c_str());
      qth04_[ism-1] = dynamic_cast<MEContentsTH2FWithinRangeROOT*> (mui_->createQTest(ContentsTH2FWithinRangeROOT::getAlgoName(), qtname));

      sprintf(qtname, "EBIT data integrity quality TTId %s", Numbers::sEB(ism).c_str());
      qth05_[ism-1] = dynamic_cast<MEContentsTH2FWithinRangeROOT*> (mui_->createQTest(ContentsTH2FWithinRangeROOT::getAlgoName(), qtname));

      sprintf(qtname, "EBIT data integrity quality TTBlockSize %s", Numbers::sEB(ism).c_str());
      qth06_[ism-1] = dynamic_cast<MEContentsTH2FWithinRangeROOT*> (mui_->createQTest(ContentsTH2FWithinRangeROOT::getAlgoName(), qtname));

      sprintf(qtname, "EBIT data integrity quality MemChId %s", Numbers::sEB(ism).c_str());
      qth07_[ism-1] = dynamic_cast<MEContentsTH2FWithinRangeROOT*> (mui_->createQTest(ContentsTH2FWithinRangeROOT::getAlgoName(), qtname));

      sprintf(qtname, "EBIT data integrity quality MemGain %s", Numbers::sEB(ism).c_str());
      qth08_[ism-1] = dynamic_cast<MEContentsTH2FWithinRangeROOT*> (mui_->createQTest(ContentsTH2FWithinRangeROOT::getAlgoName(), qtname));

      sprintf(qtname, "EBIT data integrity quality MemTTId %s", Numbers::sEB(ism).c_str());
      qth09_[ism-1] = dynamic_cast<MEContentsTH2FWithinRangeROOT*> (mui_->createQTest(ContentsTH2FWithinRangeROOT::getAlgoName(), qtname));

      sprintf(qtname, "EBIT data integrity quality MemSize %s", Numbers::sEB(ism).c_str());
      qth10_[ism-1] = dynamic_cast<MEContentsTH2FWithinRangeROOT*> (mui_->createQTest(ContentsTH2FWithinRangeROOT::getAlgoName(), qtname));

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

      sprintf(qtname, "EBIT quality test %s", Numbers::sEB(ism).c_str());
      qtg01_[ism-1] = dynamic_cast<MEContentsTH2FWithinRangeROOT*> (mui_->createQTest(ContentsTH2FWithinRangeROOT::getAlgoName(), qtname));

      sprintf(qtname, "EBIT quality test MEM %s", Numbers::sEB(ism).c_str());
      qtg02_[ism-1] = dynamic_cast<MEContentsTH2FWithinRangeROOT*> (mui_->createQTest(ContentsTH2FWithinRangeROOT::getAlgoName(), qtname));

      qtg01_[ism-1]->setMeanRange(1., 6.);
      qtg02_[ism-1]->setMeanRange(1., 6.);

      qtg01_[ism-1]->setErrorProb(1.00);
      qtg02_[ism-1]->setErrorProb(1.00);

    }

  }

}

void EBIntegrityClient::beginRun(void){

  if ( verbose_ ) cout << "EBIntegrityClient: beginRun" << endl;

  jevt_ = 0;

  this->setup();

  this->subscribe();

}

void EBIntegrityClient::endJob(void) {

  if ( verbose_ ) cout << "EBIntegrityClient: endJob, ievt = " << ievt_ << endl;

  this->unsubscribe();

  this->cleanup();

}

void EBIntegrityClient::endRun(void) {

  if ( verbose_ ) cout << "EBIntegrityClient: endRun, jevt = " << jevt_ << endl;

  this->unsubscribe();

  this->cleanup();

}

void EBIntegrityClient::setup(void) {

  Char_t histo[200];

  mui_->setCurrentFolder( "EcalBarrel/EBIntegrityClient" );
  DaqMonitorBEInterface* dbe = mui_->getBEInterface();

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( meg01_[ism-1] ) dbe->removeElement( meg01_[ism-1]->getName() );
    sprintf(histo, "EBIT data integrity quality %s", Numbers::sEB(ism).c_str());
    meg01_[ism-1] = dbe->book2D(histo, histo, 85, 0., 85., 20, 0., 20.);

    if ( meg02_[ism-1] ) dbe->removeElement( meg02_[ism-1]->getName() );
    sprintf(histo, "EBIT data integrity quality MEM %s", Numbers::sEB(ism).c_str());
    meg02_[ism-1] = dbe->book2D(histo, histo, 10, 0., 10., 5, 0.,5.);

  }

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    UtilsClient::resetHisto( meg01_[ism-1] );
    UtilsClient::resetHisto( meg02_[ism-1] );

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

        meg01_[ism-1]->setBinContent( ie, ip, 2. );

      }
    }


    for ( int ie = 1; ie <= 10; ie++ ) {
      for ( int ip = 1; ip <= 5; ip++ ) {

        meg02_[ism-1]->setBinContent( ie, ip, 2. );

      }
    }

  }

}

void EBIntegrityClient::cleanup(void) {

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

  mui_->setCurrentFolder( "EcalBarrel/EBIntegrityClient" );
  DaqMonitorBEInterface* dbe = mui_->getBEInterface();

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( meg01_[ism-1] ) dbe->removeElement( meg01_[ism-1]->getName() );
    meg01_[ism-1] = 0;

    if ( meg02_[ism-1] ) dbe->removeElement( meg02_[ism-1]->getName() );
    meg02_[ism-1] = 0;

  }

}

bool EBIntegrityClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov) {

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

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

        num01 = num02 = num03 = num04 = 0.;

        bool update1 = false;

        float numTot = -1.;

        if ( h_[ism-1] ) numTot = h_[ism-1]->GetBinContent(ie, ip);

        if ( h01_[ism-1] ) {
          num01  = h01_[ism-1]->GetBinContent(ie, ip);
          if ( num01 > 0 ) update1 = true;
        }

        if ( h02_[ism-1] ) {
          num02  = h02_[ism-1]->GetBinContent(ie, ip);
          if ( num02 > 0 ) update1 = true;
        }

        if ( h03_[ism-1] ) {
          num03  = h03_[ism-1]->GetBinContent(ie, ip);
          if ( num03 > 0 ) update1 = true;
        }

        if ( h04_[ism-1] ) {
          num04  = h04_[ism-1]->GetBinContent(ie, ip);
          if ( num04 > 0 ) update1 = true;
        }

        if ( update0 || update1 ) {

          if ( ie == 1 && ip == 1 ) {

            cout << "Preparing dataset for SM=" << ism << endl;

            cout << "(" << ie << "," << ip << ") " << num00 << " " << num01 << " " << num02 << " " << num03 << " " << num04 << endl;

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

          int ic = (ip-1) + 20*(ie-1) + 1;

          if ( econn ) {
            try {
              ecid = LogicID::getEcalLogicID("EB_crystal_number", Numbers::iSM(ism, EcalBarrel), ic);
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

    for ( int iet = 1; iet <= 17; iet++ ) {
      for ( int ipt = 1; ipt <= 4; ipt++ ) {

        num05 = num06 = 0.;

        bool update1 = false;

        float numTot = -1.;

        if ( h_[ism-1] ) {
          numTot = 0.;
          for ( int ie = 1 + 5*(iet-1); ie <= 5*iet; ie++ ) {
            for ( int ip = 1 + 5*(ipt-1); ip <= 5*ipt; ip++ ) {
              numTot += h_[ism-1]->GetBinContent(ie, ip);
            }
          }
        }

        if ( h05_[ism-1] ) {
          num05  = h05_[ism-1]->GetBinContent(iet, ipt);
          if ( num05 > 0 ) update1 = true;
        }

        if ( h06_[ism-1] ) {
          num06  = h06_[ism-1]->GetBinContent(iet, ipt);
          if ( num06 > 0 ) update1 = true;
        }

        if ( update0 || update1 ) {

          if ( iet == 1 && ipt == 1 ) {

            cout << "Preparing dataset for SM=" << ism << endl;

            cout << "(" << iet << "," << ipt << ") " << num00 << " " << num05 << " " << num06 << endl;

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

          int itt = (ipt-1) + 4*(iet-1) + 1;

          if ( econn ) {
            try {
              ecid = LogicID::getEcalLogicID("EB_trigger_tower", Numbers::iSM(ism, EcalBarrel), itt);
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

    for ( int ie = 1; ie <= 10; ie++ ) {
      for ( int ip = 1; ip <= 5; ip++ ) {

        num07 = num08 = 0.;

        bool update1 = false;

        float numTot = -1.;

        if ( hmem_[ism-1] ) numTot = hmem_[ism-1]->GetBinContent(ie, ip);

        if ( h07_[ism-1] ) {
          num07  = h07_[ism-1]->GetBinContent(ie, ip);
          if ( num07 > 0 ) update1 = true;
        }

        if ( h08_[ism-1] ) {
          num08  = h08_[ism-1]->GetBinContent(ie, ip);
          if ( num08 > 0 ) update1 = true;
        }

        if ( update0 || update1 ) {

          if ( ie == 1 && ip == 1 ) {

            cout << "Preparing dataset for mem of SM=" << ism << endl;

            cout << "(" << ie << "," << ip << ") " << num07 << " " << num08 << endl;

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

          int ic = EBIntegrityClient::chNum[ (ie-1)%5 ][ (ip-1) ] + (ie-1)/5 * 25;

          if ( econn ) {
            try {
              ecid = LogicID::getEcalLogicID("EB_mem_channel", Numbers::iSM(ism, EcalBarrel), ic);
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

    for ( int iet = 1; iet <= 2; iet++ ) {

      num09 = num10 = 0.;

      bool update1 = false;

      float numTot = -1.;

      if ( hmem_[ism-1] ) {
        numTot = 0.;
        for ( int ie = 1 + 5*(iet-1); ie <= 5*iet; ie++ ) {
          for ( int ip = 1 ; ip <= 5; ip++ ) {
            numTot += hmem_[ism-1]->GetBinContent(ie, ip);
          }
        }
      }

      if ( h09_[ism-1] ) {
        num09  = h09_[ism-1]->GetBinContent(iet, 1);
        if ( num09 > 0 ) update1 = true;
      }

      if ( h10_[ism-1] ) {
        num10  = h10_[ism-1]->GetBinContent(iet, 1);
        if ( num10 > 0 ) update1 = true;
      }

      if ( update0 || update1 ) {

        if ( iet == 1 ) {

          cout << "Preparing dataset for SM=" << ism << endl;

          cout << "(" << iet <<  ") " << num09 << " " << num10 << endl;

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

        int itt = 68 + iet;

        if ( econn ) {
          try {
            ecid = LogicID::getEcalLogicID("EB_mem_TT", Numbers::iSM(ism, EcalBarrel), itt);
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

void EBIntegrityClient::subscribe(void){

  if ( verbose_ ) cout << "EBIntegrityClient: subscribe" << endl;

  Char_t histo[200];

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    unsigned int ism = superModules_[i];

    sprintf(histo, "*/EcalBarrel/EBOccupancyTask/EBOT occupancy %s", Numbers::sEB(ism).c_str());
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBOccupancyTask/EBOT MEM occupancy %s", Numbers::sEB(ism).c_str());
    mui_->subscribe(histo, ism);

    sprintf(histo, "*/EcalBarrel/EBIntegrityTask/EBIT DCC size error");
    mui_->subscribe(histo);
    sprintf(histo, "*/EcalBarrel/EBIntegrityTask/Gain/EBIT gain %s", Numbers::sEB(ism).c_str());
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBIntegrityTask/ChId/EBIT ChId %s", Numbers::sEB(ism).c_str());
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBIntegrityTask/GainSwitch/EBIT gain switch %s", Numbers::sEB(ism).c_str());
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBIntegrityTask/GainSwitchStay/EBIT gain switch stay %s", Numbers::sEB(ism).c_str());
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBIntegrityTask/TTId/EBIT TTId %s", Numbers::sEB(ism).c_str());
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBIntegrityTask/TTBlockSize/EBIT TTBlockSize %s", Numbers::sEB(ism).c_str());
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBIntegrityTask/MemChId/EBIT MemChId %s", Numbers::sEB(ism).c_str());
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBIntegrityTask/MemGain/EBIT MemGain %s", Numbers::sEB(ism).c_str());
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBIntegrityTask/MemTTId/EBIT MemTTId %s", Numbers::sEB(ism).c_str());
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBIntegrityTask/MemSize/EBIT MemSize %s", Numbers::sEB(ism).c_str());
    mui_->subscribe(histo, ism);

  }

  if ( collateSources_ ) {

    if ( verbose_ ) cout << "EBIntegrityClient: collate" << endl;

    sprintf(histo, "EBIT DCC size error");
    me_h00_ = mui_->collate1D(histo, histo, "EcalBarrel/Sums/EBIntegrityTask");
    sprintf(histo, "*/EcalBarrel/EBIntegrityTask/EBIT DCC size error");
    mui_->add(me_h00_, histo);

    for ( unsigned int i=0; i<superModules_.size(); i++ ) {

      int ism = superModules_[i];

      sprintf(histo, "EBOT occupancy %s", Numbers::sEB(ism).c_str());
      me_h_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBOccupancyTask");
      sprintf(histo, "*/EcalBarrel/EBOccupancyTask/EBOT occupancy %s", Numbers::sEB(ism).c_str());
      mui_->add(me_h_[ism-1], histo);

      sprintf(histo, "EBOT MEM occupancy %s", Numbers::sEB(ism).c_str());
      me_hmem_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBOccupancyTask");
      sprintf(histo, "*/EcalBarrel/EBOccupancyTask/EBOT MEM occupancy %s", Numbers::sEB(ism).c_str());
      mui_->add(me_hmem_[ism-1], histo);

      sprintf(histo, "EBIT gain %s", Numbers::sEB(ism).c_str());
      me_h01_[ism-1] = mui_->collate2D(histo, histo, "EcalBarrel/Sums/EBIntegrityTask/Gain");
      sprintf(histo, "*/EcalBarrel/EBIntegrityTask/Gain/EBIT gain %s", Numbers::sEB(ism).c_str());
      mui_->add(me_h01_[ism-1], histo);

      sprintf(histo, "EBIT ChId %s", Numbers::sEB(ism).c_str());
      me_h02_[ism-1] = mui_->collate2D(histo, histo, "EcalBarrel/Sums/EBIntegrityTask/ChId");
      sprintf(histo, "*/EcalBarrel/EBIntegrityTask/ChId/EBIT ChId %s", Numbers::sEB(ism).c_str());
      mui_->add(me_h02_[ism-1], histo);

      sprintf(histo, "EBIT gain switch %s", Numbers::sEB(ism).c_str());
      me_h03_[ism-1] = mui_->collate2D(histo, histo, "EcalBarrel/Sums/EBIntegrityTask/GainSwitch");
      sprintf(histo, "*/EcalBarrel/EBIntegrityTask/GainSwitch/EBIT gain switch %s", Numbers::sEB(ism).c_str());
      mui_->add(me_h03_[ism-1], histo);

      sprintf(histo, "EBIT gain switch stay %s", Numbers::sEB(ism).c_str());
      me_h04_[ism-1] = mui_->collate2D(histo, histo, "EcalBarrel/Sums/EBIntegrityTask/GainSwitchStay");
      sprintf(histo, "*/EcalBarrel/EBIntegrityTask/GainSwitchStay/EBIT gain switch stay %s", Numbers::sEB(ism).c_str());
      mui_->add(me_h04_[ism-1], histo);

      sprintf(histo, "EBIT TTId %s", Numbers::sEB(ism).c_str());
      me_h05_[ism-1] = mui_->collate2D(histo, histo, "EcalBarrel/Sums/EBIntegrityTask/TTId");
      sprintf(histo, "*/EcalBarrel/EBIntegrityTask/TTId/EBIT TTId %s", Numbers::sEB(ism).c_str());
      mui_->add(me_h05_[ism-1], histo);

      sprintf(histo, "EBIT TTBlockSize %s", Numbers::sEB(ism).c_str());
      me_h06_[ism-1] = mui_->collate2D(histo, histo, "EcalBarrel/Sums/EBIntegrityTask/TTBlockSize");
      sprintf(histo, "*/EcalBarrel/EBIntegrityTask/TTBlockSize/EBIT TTBlockSize %s", Numbers::sEB(ism).c_str());
      mui_->add(me_h06_[ism-1], histo);

      sprintf(histo, "EBIT MemChId %s", Numbers::sEB(ism).c_str());
      me_h07_[ism-1] = mui_->collate2D(histo, histo, "EcalBarrel/Sums/EBIntegrityTask/MemChId");
      sprintf(histo, "*/EcalBarrel/EBIntegrityTask/MemChId/EBIT MemChId %s", Numbers::sEB(ism).c_str());
      mui_->add(me_h07_[ism-1], histo);

      sprintf(histo, "EBIT MemGain %s", Numbers::sEB(ism).c_str());
      me_h08_[ism-1] = mui_->collate2D(histo, histo, "EcalBarrel/Sums/EBIntegrityTask/MemGain");
      sprintf(histo, "*/EcalBarrel/EBIntegrityTask/MemGain/EBIT MemGain %s", Numbers::sEB(ism).c_str());
      mui_->add(me_h08_[ism-1], histo);

      sprintf(histo, "EBIT MemTTId %s", Numbers::sEB(ism).c_str());
      me_h09_[ism-1] = mui_->collate2D(histo, histo, "EcalBarrel/Sums/EBIntegrityTask/MemTTId");
      sprintf(histo, "*/EcalBarrel/EBIntegrityTask/MemTTId/EBIT MemTTId %s", Numbers::sEB(ism).c_str());
      mui_->add(me_h09_[ism-1], histo);

      sprintf(histo, "EBIT MemSize %s", Numbers::sEB(ism).c_str());
      me_h10_[ism-1] = mui_->collate2D(histo, histo, "EcalBarrel/Sums/EBIntegrityTask/MemSize");
      sprintf(histo, "*/EcalBarrel/EBIntegrityTask/MemSize/EBIT MemSize %s", Numbers::sEB(ism).c_str());
      mui_->add(me_h10_[ism-1], histo);

    }

  }

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBIntegrityTask/Gain/EBIT gain %s", Numbers::sEB(ism).c_str());
      if ( qth01_[ism-1] ) mui_->useQTest(histo, qth01_[ism-1]->getName());
      sprintf(histo, "EcalBarrel/Sums/EBIntegrityTask/ChId/EBIT ChId %s", Numbers::sEB(ism).c_str());
      if ( qth02_[ism-1] ) mui_->useQTest(histo, qth02_[ism-1]->getName());
      sprintf(histo, "EcalBarrel/Sums/EBIntegrityTask/GainSwitch/EBIT gain switch %s", Numbers::sEB(ism).c_str());
      if ( qth03_[ism-1] ) mui_->useQTest(histo, qth03_[ism-1]->getName());
      sprintf(histo, "EcalBarrel/Sums/EBIntegrityTask/GainSwitchStay/EBIT gain switch stay %s", Numbers::sEB(ism).c_str());
      if ( qth04_[ism-1] ) mui_->useQTest(histo, qth04_[ism-1]->getName());
      sprintf(histo, "EcalBarrel/Sums/EBIntegrityTask/TTId/EBIT TTId %s", Numbers::sEB(ism).c_str());
      if ( qth05_[ism-1] ) mui_->useQTest(histo, qth05_[ism-1]->getName());
      sprintf(histo, "EcalBarrel/Sums/EBIntegrityTask/TTBlockSize/EBIT TTBlockSize %s", Numbers::sEB(ism).c_str());
      if ( qth06_[ism-1] ) mui_->useQTest(histo, qth06_[ism-1]->getName());
      sprintf(histo, "EcalBarrel/Sums/EBIntegrityTask/MemChId/EBIT MemChId %s", Numbers::sEB(ism).c_str());
      if ( qth07_[ism-1] ) mui_->useQTest(histo, qth07_[ism-1]->getName());
      sprintf(histo, "EcalBarrel/Sums/EBIntegrityTask/MemGain %s", Numbers::sEB(ism).c_str());
      if ( qth08_[ism-1] ) mui_->useQTest(histo, qth08_[ism-1]->getName());
      sprintf(histo, "EcalBarrel/Sums/EBIntegrityTask/MemTTId/EBIT MemTTId %s", Numbers::sEB(ism).c_str());
      if ( qth09_[ism-1] ) mui_->useQTest(histo, qth09_[ism-1]->getName());
      sprintf(histo, "EcalBarrel/Sums/EBIntegrityTask/MemSize/EBIT MemSize %s", Numbers::sEB(ism).c_str());
      if ( qth10_[ism-1] ) mui_->useQTest(histo, qth10_[ism-1]->getName());
    } else {
      if ( enableMonitorDaemon_ ) {
        sprintf(histo, "*/EcalBarrel/EBIntegrityTask/Gain/EBIT gain %s", Numbers::sEB(ism).c_str());
        if ( qth01_[ism-1] ) mui_->useQTest(histo, qth01_[ism-1]->getName());
        sprintf(histo, "*/EcalBarrel/EBIntegrityTask/ChId/EBIT ChId %s", Numbers::sEB(ism).c_str());
        if ( qth02_[ism-1] ) mui_->useQTest(histo, qth02_[ism-1]->getName());
        sprintf(histo, "*/EcalBarrel/EBIntegrityTask/GainSwitch/EBIT gain switch %s", Numbers::sEB(ism).c_str());
        if ( qth03_[ism-1] ) mui_->useQTest(histo, qth03_[ism-1]->getName());
        sprintf(histo, "*/EcalBarrel/EBIntegrityTask/GainSwitchStay/EBIT gain switch stay %s", Numbers::sEB(ism).c_str());
        if ( qth04_[ism-1] ) mui_->useQTest(histo, qth04_[ism-1]->getName());
        sprintf(histo, "*/EcalBarrel/EBIntegrityTask/TTId/EBIT TTId %s", Numbers::sEB(ism).c_str());
        if ( qth05_[ism-1] ) mui_->useQTest(histo, qth05_[ism-1]->getName());
        sprintf(histo, "*/EcalBarrel/EBIntegrityTask/TTBlockSize/EBIT TTBlockSize %s", Numbers::sEB(ism).c_str());
        if ( qth06_[ism-1] ) mui_->useQTest(histo, qth06_[ism-1]->getName());
        sprintf(histo, "*/EcalBarrel/EBIntegrityTask/MemChId/EBIT MemChId %s", Numbers::sEB(ism).c_str());
        if ( qth07_[ism-1] ) mui_->useQTest(histo, qth07_[ism-1]->getName());
        sprintf(histo, "*/EcalBarrel/EBIntegrityTask/MemGain %s", Numbers::sEB(ism).c_str());
        if ( qth08_[ism-1] ) mui_->useQTest(histo, qth08_[ism-1]->getName());
        sprintf(histo, "*/EcalBarrel/EBIntegrityTask/MemTTId/EBIT MemTTId %s", Numbers::sEB(ism).c_str());
        if ( qth09_[ism-1] ) mui_->useQTest(histo, qth09_[ism-1]->getName());
        sprintf(histo, "*/EcalBarrel/EBIntegrityTask/MemSize/EBIT MemSize %s", Numbers::sEB(ism).c_str());
        if ( qth10_[ism-1] ) mui_->useQTest(histo, qth10_[ism-1]->getName());
      } else {
        sprintf(histo, "EcalBarrel/EBIntegrityTask/Gain/EBIT gain %s", Numbers::sEB(ism).c_str());
        if ( qth01_[ism-1] ) mui_->useQTest(histo, qth01_[ism-1]->getName());
        sprintf(histo, "EcalBarrel/EBIntegrityTask/ChId/EBIT ChId %s", Numbers::sEB(ism).c_str());
        if ( qth02_[ism-1] ) mui_->useQTest(histo, qth02_[ism-1]->getName());
        sprintf(histo, "EcalBarrel/EBIntegrityTask/GainSwitch/EBIT gain switch %s", Numbers::sEB(ism).c_str());
        if ( qth03_[ism-1] ) mui_->useQTest(histo, qth03_[ism-1]->getName());
        sprintf(histo, "EcalBarrel/EBIntegrityTask/GainSwitchStay/EBIT gain switch stay %s", Numbers::sEB(ism).c_str());
        if ( qth04_[ism-1] ) mui_->useQTest(histo, qth04_[ism-1]->getName());
        sprintf(histo, "EcalBarrel/EBIntegrityTask/TTId/EBIT TTId %s", Numbers::sEB(ism).c_str());
        if ( qth05_[ism-1] ) mui_->useQTest(histo, qth05_[ism-1]->getName());
        sprintf(histo, "EcalBarrel/EBIntegrityTask/TTBlockSize/EBIT TTBlockSize %s", Numbers::sEB(ism).c_str());
        if ( qth06_[ism-1] ) mui_->useQTest(histo, qth06_[ism-1]->getName());
        sprintf(histo, "EcalBarrel/EBIntegrityTask/MemChId/EBIT MemChId %s", Numbers::sEB(ism).c_str());
        if ( qth07_[ism-1] ) mui_->useQTest(histo, qth07_[ism-1]->getName());
        sprintf(histo, "EcalBarrel/EBIntegrityTask/MemGain %s", Numbers::sEB(ism).c_str());
        if ( qth08_[ism-1] ) mui_->useQTest(histo, qth08_[ism-1]->getName());
        sprintf(histo, "EcalBarrel/EBIntegrityTask/MemTTId/EBIT MemTTId %s", Numbers::sEB(ism).c_str());
        if ( qth09_[ism-1] ) mui_->useQTest(histo, qth09_[ism-1]->getName());
        sprintf(histo, "EcalBarrel/EBIntegrityTask/MemSize/EBIT MemSize %s", Numbers::sEB(ism).c_str());
        if ( qth10_[ism-1] ) mui_->useQTest(histo, qth10_[ism-1]->getName());
      }
    }

    sprintf(histo, "EcalBarrel/EBIntegrityClient/EBIT data integrity quality %s", Numbers::sEB(ism).c_str());
    if ( qtg01_[ism-1] ) mui_->useQTest(histo, qtg01_[ism-1]->getName());

    sprintf(histo, "EcalBarrel/EBIntegrityClient/EBIT data integrity quality MEM %s", Numbers::sEB(ism).c_str());
    if ( qtg02_[ism-1] ) mui_->useQTest(histo, qtg02_[ism-1]->getName());

  }

}

void EBIntegrityClient::subscribeNew(void){

  Char_t histo[200];

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    unsigned int ism = superModules_[i];

    sprintf(histo, "*/EcalBarrel/EBOccupancyTask/EBOT occupancy %s", Numbers::sEB(ism).c_str());
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBOccupancyTask/EBOT MEM occupancy %s", Numbers::sEB(ism).c_str());
    mui_->subscribeNew(histo, ism);

    sprintf(histo, "*/EcalBarrel/EBIntegrityTask/EBIT DCC size error");
    mui_->subscribeNew(histo);
    sprintf(histo, "*/EcalBarrel/EBIntegrityTask/Gain/EBIT gain %s", Numbers::sEB(ism).c_str());
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBIntegrityTask/ChId/EBIT ChId %s", Numbers::sEB(ism).c_str());
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBIntegrityTask/GainSwitch/EBIT gain switch %s", Numbers::sEB(ism).c_str());
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBIntegrityTask/GainSwitchStay/EBIT gain switch stay %s", Numbers::sEB(ism).c_str());
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBIntegrityTask/TTId/EBIT TTId %s", Numbers::sEB(ism).c_str());
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBIntegrityTask/TTBlockSize/EBIT TTBlockSize %s", Numbers::sEB(ism).c_str());
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBIntegrityTask/MemChId/EBIT MemChId %s", Numbers::sEB(ism).c_str());
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBIntegrityTask/MemGain/EBIT MemGain %s", Numbers::sEB(ism).c_str());
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBIntegrityTask/MemTTId/EBIT MemTTId %s", Numbers::sEB(ism).c_str());
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBIntegrityTask/MemSize/EBIT MemSize %s", Numbers::sEB(ism).c_str());
    mui_->subscribeNew(histo, ism);

  }

}

void EBIntegrityClient::unsubscribe(void){

  if ( verbose_ ) cout << "EBIntegrityClient: unsubscribe" << endl;

  if ( collateSources_ ) {

    if ( verbose_ ) cout << "EBIntegrityClient: uncollate" << endl;

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

    sprintf(histo, "*/EcalBarrel/EBOccupancyTask/EBOT occupancy %s", Numbers::sEB(ism).c_str());
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBOccupancyTask/EBOT MEM occupancy %s", Numbers::sEB(ism).c_str());
    mui_->unsubscribe(histo, ism);

    sprintf(histo, "*/EcalBarrel/EBIntegrityTask/EBIT DCC size error");
    mui_->unsubscribe(histo);
    sprintf(histo, "*/EcalBarrel/EBIntegrityTask/Gain/EBIT gain %s", Numbers::sEB(ism).c_str());
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBIntegrityTask/ChId/EBIT ChId %s", Numbers::sEB(ism).c_str());
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBIntegrityTask/GainSwitch/EBIT gain switch %s", Numbers::sEB(ism).c_str());
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBIntegrityTask/GainSwitchStay/EBIT gain switch stay %s", Numbers::sEB(ism).c_str());
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBIntegrityTask/TTId/EBIT TTId %s", Numbers::sEB(ism).c_str());
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBIntegrityTask/TTBlockSize/EBIT TTBlockSize %s", Numbers::sEB(ism).c_str());
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBIntegrityTask/MemChId/EBIT MemChId %s", Numbers::sEB(ism).c_str());
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBIntegrityTask/MemGain/EBIT MemGain %s", Numbers::sEB(ism).c_str());
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBIntegrityTask/MemTTId/EBIT MemTTId %s", Numbers::sEB(ism).c_str());
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBIntegrityTask/MemSize/EBIT MemSize %s", Numbers::sEB(ism).c_str());
    mui_->unsubscribe(histo, ism);

  }

}

void EBIntegrityClient::softReset(void){

}

void EBIntegrityClient::analyze(void){

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( verbose_ ) cout << "EBIntegrityClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;
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
    sprintf(histo, "EcalBarrel/Sums/EBIntegrityTask/EBIT DCC size error");
  } else {
    sprintf(histo, (prefixME_+"EcalBarrel/EBIntegrityTask/EBIT DCC size error").c_str());
  }
  me = mui_->get(histo);
  h00_ = UtilsClient::getHisto<TH1F*>( me, cloneME_, h00_ );

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBOccupancyTask/EBOT occupancy %s", Numbers::sEB(ism).c_str());
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBOccupancyTask/EBOT occupancy %s").c_str(), Numbers::sEB(ism).c_str());
    }
    me = mui_->get(histo);
    h_[ism-1] = UtilsClient::getHisto<TH2F*>( me, cloneME_, h_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBOccupancyTask/EBOT MEM occupancy %s", Numbers::sEB(ism).c_str());
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBOccupancyTask/EBOT MEM occupancy %s").c_str(), Numbers::sEB(ism).c_str());
    }
    me = mui_->get(histo);
    hmem_[ism-1] = UtilsClient::getHisto<TH2F*>( me, cloneME_, hmem_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBIntegrityTask/Gain/EBIT gain %s", Numbers::sEB(ism).c_str());
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBIntegrityTask/Gain/EBIT gain %s").c_str(), Numbers::sEB(ism).c_str());
    }
    me = mui_->get(histo);
    h01_[ism-1] = UtilsClient::getHisto<TH2F*>( me, cloneME_, h01_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBIntegrityTask/ChId/EBIT ChId %s", Numbers::sEB(ism).c_str());
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBIntegrityTask/ChId/EBIT ChId %s").c_str(), Numbers::sEB(ism).c_str());
    }
    me = mui_->get(histo);
    h02_[ism-1] = UtilsClient::getHisto<TH2F*>( me, cloneME_, h02_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBIntegrityTask/GainSwitch/EBIT gain switch %s", Numbers::sEB(ism).c_str());
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBIntegrityTask/GainSwitch/EBIT gain switch %s").c_str(), Numbers::sEB(ism).c_str());
    }
    me = mui_->get(histo);
    h03_[ism-1] = UtilsClient::getHisto<TH2F*>( me, cloneME_, h03_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBIntegrityTask/GainSwitchStay/EBIT gain switch stay %s", Numbers::sEB(ism).c_str());
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBIntegrityTask/GainSwitchStay/EBIT gain switch stay %s").c_str(), Numbers::sEB(ism).c_str());
    }
    me = mui_->get(histo);
    h04_[ism-1] = UtilsClient::getHisto<TH2F*>( me, cloneME_, h04_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBIntegrityTask/TTId/EBIT TTId %s", Numbers::sEB(ism).c_str());
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBIntegrityTask/TTId/EBIT TTId %s").c_str(), Numbers::sEB(ism).c_str());
    }
    me = mui_->get(histo);
    h05_[ism-1] = UtilsClient::getHisto<TH2F*>( me, cloneME_, h05_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBIntegrityTask/TTBlockSize/EBIT TTBlockSize %s", Numbers::sEB(ism).c_str());
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBIntegrityTask/TTBlockSize/EBIT TTBlockSize %s").c_str(), Numbers::sEB(ism).c_str());
    }
    me = mui_->get(histo);
    h06_[ism-1] = UtilsClient::getHisto<TH2F*>( me, cloneME_, h06_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBIntegrityTask/MemChId/EBIT MemChId %s", Numbers::sEB(ism).c_str());
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBIntegrityTask/MemChId/EBIT MemChId %s").c_str(), Numbers::sEB(ism).c_str());
    }
    me = mui_->get(histo);
    h07_[ism-1] = UtilsClient::getHisto<TH2F*>( me, cloneME_, h07_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBIntegrityTask/MemGain/EBIT MemGain %s", Numbers::sEB(ism).c_str());
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBIntegrityTask/MemGain/EBIT MemGain %s").c_str(), Numbers::sEB(ism).c_str());
    }
    me = mui_->get(histo);
    h08_[ism-1] = UtilsClient::getHisto<TH2F*>( me, cloneME_, h08_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBIntegrityTask/MemTTId/EBIT MemTTId %s", Numbers::sEB(ism).c_str());
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBIntegrityTask/MemTTId/EBIT MemTTId %s").c_str(), Numbers::sEB(ism).c_str());
    }
    me = mui_->get(histo);
    h09_[ism-1] = UtilsClient::getHisto<TH2F*>( me, cloneME_, h09_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBIntegrityTask/MemSize/EBIT MemSize %s", Numbers::sEB(ism).c_str());
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBIntegrityTask/MemSize/EBIT MemSize %s").c_str(), Numbers::sEB(ism).c_str());
    }
    me = mui_->get(histo);
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

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

        num01 = num02 = num03 = num04 = num05 = num06 = 0.;

        if ( meg01_[ism-1] ) meg01_[ism-1]->setBinContent( ie, ip, 2. );

        bool update1 = false;
        bool update2 = false;

        float numTot = -1.;

        if ( h_[ism-1] ) numTot = h_[ism-1]->GetBinContent(ie, ip);

        if ( h01_[ism-1] ) {
          num01  = h01_[ism-1]->GetBinContent(ie, ip);
          update1 = true;
        }

        if ( h02_[ism-1] ) {
          num02  = h02_[ism-1]->GetBinContent(ie, ip);
          update1 = true;
        }

        if ( h03_[ism-1] ) {
          num03  = h03_[ism-1]->GetBinContent(ie, ip);
          update1 = true;
        }

        if ( h04_[ism-1] ) {
          num04  = h04_[ism-1]->GetBinContent(ie, ip);
          update1 = true;
        }

        int iet = 1 + ((ie-1)/5);
        int ipt = 1 + ((ip-1)/5);

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

          // filling the summary for SM channels
          if ( meg01_[ism-1] ) meg01_[ism-1]->setBinContent( ie, ip, val );

        }

        // masking

        if ( mask1.size() != 0 ) {
          map<EcalLogicID, RunCrystalErrorsDat>::const_iterator m;
          for (m = mask1.begin(); m != mask1.end(); m++) {

            EcalLogicID ecid = m->first;

            int ic = (ip-1) + 20*(ie-1) + 1;

            if ( ecid.getID1() == Numbers::iSM(ism, EcalBarrel) && ecid.getID2() == ic ) {
              if ( (m->second).getErrorBits() & bits01 ) {
                if ( meg01_[ism-1] ) {
                  float val = int(meg01_[ism-1]->getBinContent(ie, ip)) % 3;
                  meg01_[ism-1]->setBinContent( ie, ip, val+3 );
                }
              }
            }

          }
        }

        if ( mask2.size() != 0 ) {
          map<EcalLogicID, RunTTErrorsDat>::const_iterator m;
          for (m = mask2.begin(); m != mask2.end(); m++) {

            EcalLogicID ecid = m->first;

            int iet = 1 + ((ie-1)/5);
            int ipt = 1 + ((ip-1)/5);
            int itt = (ipt-1) + 4*(iet-1) + 1;

            if ( ecid.getID1() == Numbers::iSM(ism, EcalBarrel) && ecid.getID2() == itt ) {
              if ( (m->second).getErrorBits() & bits02 ) {
                if ( meg01_[ism-1] ) {
                  float val = int(meg01_[ism-1]->getBinContent(ie, ip)) % 3;
                  meg01_[ism-1]->setBinContent( ie, ip, val+3 );
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

            int ic = EBIntegrityClient::chNum[ (ie-1)%5 ][ (ip-1) ] + (ie-1)/5 * 25;

            if ( ecid.getID1() == Numbers::iSM(ism, EcalBarrel) && ecid.getID2() == ic ) {
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

            if ( ecid.getID1() == Numbers::iSM(ism, EcalBarrel) && ecid.getID2() == itt ) {
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

void EBIntegrityClient::htmlOutput(int run, string htmlDir, string htmlName){

  cout << "Preparing EBIntegrityClient html output ..." << endl;

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
	     << Numbers::sEB(superModules_[i]).c_str() << ">"
	     << setfill( '0' ) << setw(2) << superModules_[i] << "</a></td>";
  }
  htmlFile << std::endl << "</table>" << std::endl;
  htmlFile << "<hr>" << std::endl;

  // Produce the plots to be shown as .png files from existing histograms

  const int csize = 250;

  int pCol3[6] = { 301, 302, 303, 304, 305, 306 };
  int pCol4[10];
  for ( int i = 0; i < 10; i++ ) pCol4[i] = 401+i;

  TH2C dummy1( "dummy1", "dummy1 for sm", 85, 0, 85, 20, 0, 20 );
  for ( short i=0; i<68; i++ ) {
    int a = 2 + ( i/4 ) * 5;
    int b = 2 + ( i%4 ) * 5;
    dummy1.Fill( a, b, i+1 );
  }
  dummy1.SetMarkerSize(2);
  dummy1.SetMinimum(0.1);

  TH2C dummy2( "dummy2", "dummy2 for sm", 17, 0, 17, 4, 0, 4 );
  for ( short i=0; i<68; i++ ) {
    int a = ( i/4 );
    int b = ( i%4 );
    dummy2.Fill( a, b, i+1 );
  }
  dummy2.SetMarkerSize(2);
  dummy2.SetMinimum(0.1);

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

  TCanvas* cDCC = new TCanvas("cDCC", "Temp", 3*csize, csize);
  TCanvas* cOcc = new TCanvas("cOcc", "Temp", 3*csize, csize);
  TCanvas* cQual = new TCanvas("cQual", "Temp", 3*csize, csize);
  TCanvas* cMe = new TCanvas("cMe", "Temp", 3*csize, csize);
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
      obj2f->GetXaxis()->SetNdivisions(17);
      obj2f->GetYaxis()->SetNdivisions(4);
      cQual->SetGridx();
      cQual->SetGridy();
      obj2f->SetMinimum(-0.00000001);
      obj2f->SetMaximum(6.0);
      obj2f->Draw("col");
      dummy1.Draw("text,same");
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
      obj2f->GetXaxis()->SetNdivisions(17);
      obj2f->GetYaxis()->SetNdivisions(4);
      cOcc->SetGridx();
      cOcc->SetGridy();
      obj2f->SetMinimum(0.0);
      obj2f->Draw("colz");
      dummy1.Draw("text,same");
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
        obj2f->GetXaxis()->SetNdivisions(17);
        obj2f->GetYaxis()->SetNdivisions(4);
        cMe->SetGridx();
        cMe->SetGridy();
        obj2f->SetMinimum(0.0);
        obj2f->Draw("colz");
        if ( iCanvas < 5 )
          dummy1.Draw("text,same");
        else
          dummy2.Draw("text,same");
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
	     << Numbers::sEB(ism).c_str() << """></a><strong>"
	     << Numbers::sEB(ism).c_str() << "</strong></h3>" << endl;
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

const int  EBIntegrityClient::chNum [5][5] = {
  { 1,  2,  3,  4,  5},
  {10,  9,  8,  7,  6},
  {11, 12, 13, 14, 15},
  {20, 19, 18, 17, 16},
  {21, 22, 23, 24, 25}
};

