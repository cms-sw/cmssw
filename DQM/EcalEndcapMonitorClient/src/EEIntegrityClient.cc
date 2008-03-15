
/*
 * \file EEIntegrityClient.cc
 *
 * $Date: 2008/03/14 14:38:58 $
 * $Revision: 1.67 $
 * \author G. Della Ricca
 * \author G. Franzoni
 *
 */

#include <memory>
#include <iostream>
#include <fstream>
#include <iomanip>

#include "TCanvas.h"
#include "TStyle.h"
#include "TGraph.h"
#include "TLine.h"

#include "DQMServices/Core/interface/DQMStore.h"

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

#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"

#include "CondTools/Ecal/interface/EcalErrorDictionary.h"

#include "DQM/EcalCommon/interface/EcalErrorMask.h"
#include "DQM/EcalCommon/interface/UtilsClient.h"
#include "DQM/EcalCommon/interface/LogicID.h"
#include "DQM/EcalCommon/interface/Numbers.h"

#include <DQM/EcalEndcapMonitorClient/interface/EEIntegrityClient.h>

#include <DataFormats/EcalDetId/interface/EEDetId.h>

using namespace cms;
using namespace edm;
using namespace std;

EEIntegrityClient::EEIntegrityClient(const ParameterSet& ps){

  // cloneME switch
  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);

  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

  // enableCleanup_ switch
  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  // prefix to ME paths
  prefixME_ = ps.getUntrackedParameter<string>("prefixME", "");

  // vector of selected Super Modules (Defaults to all 18).
  superModules_.reserve(18);
  for ( unsigned int i = 1; i <= 18; i++ ) superModules_.push_back(i);
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

  }

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    // integrity summary histograms
    meg01_[ism-1] = 0;
    meg02_[ism-1] = 0;

  }

  threshCry_ = 0.;

}

EEIntegrityClient::~EEIntegrityClient(){

}

void EEIntegrityClient::beginJob(DQMStore* dbe){

  dbe_ = dbe;

  if ( verbose_ ) cout << "EEIntegrityClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EEIntegrityClient::beginRun(void){

  if ( verbose_ ) cout << "EEIntegrityClient: beginRun" << endl;

  jevt_ = 0;

  this->setup();

}

void EEIntegrityClient::endJob(void) {

  if ( verbose_ ) cout << "EEIntegrityClient: endJob, ievt = " << ievt_ << endl;

  this->cleanup();

}

void EEIntegrityClient::endRun(void) {

  if ( verbose_ ) cout << "EEIntegrityClient: endRun, jevt = " << jevt_ << endl;

  this->cleanup();

}

void EEIntegrityClient::setup(void) {

  char histo[200];

  dbe_->setCurrentFolder( "EcalEndcap/EEIntegrityClient" );

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( meg01_[ism-1] ) dbe_->removeElement( meg01_[ism-1]->getName() );
    sprintf(histo, "EEIT data integrity quality %s", Numbers::sEE(ism).c_str());
    meg01_[ism-1] = dbe_->book2D(histo, histo, 50, Numbers::ix0EE(ism)+0., Numbers::ix0EE(ism)+50., 50, Numbers::iy0EE(ism)+0., Numbers::iy0EE(ism)+50.);
    meg01_[ism-1]->setAxisTitle("jx", 1);
    meg01_[ism-1]->setAxisTitle("jy", 2);

    if ( meg02_[ism-1] ) dbe_->removeElement( meg02_[ism-1]->getName() );
    sprintf(histo, "EEIT data integrity quality MEM %s", Numbers::sEE(ism).c_str());
    meg02_[ism-1] = dbe_->book2D(histo, histo, 10, 0., 10., 5, 0.,5.);
    meg02_[ism-1]->setAxisTitle("pseudo-strip", 1);
    meg02_[ism-1]->setAxisTitle("channel", 2);

  }

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( meg01_[ism-1] ) meg01_[ism-1]->Reset();
    if ( meg02_[ism-1] ) meg02_[ism-1]->Reset();

    for ( int ix = 1; ix <= 50; ix++ ) {
      for ( int iy = 1; iy <= 50; iy++ ) {

        if ( meg01_[ism-1] ) meg01_[ism-1]->setBinContent( ix, iy, -1. );

        int jx = ix + Numbers::ix0EE(ism);
        int jy = iy + Numbers::iy0EE(ism);

        if ( ism >= 1 && ism <= 9 ) jx = 101 - jx;

        if ( Numbers::validEE(ism, jx, jy) ) {
          if ( meg01_[ism-1] ) meg01_[ism-1]->setBinContent( ix, iy, 2. );
        }

      }
    }

    for ( int ie = 1; ie <= 10; ie++ ) {
      for ( int ip = 1; ip <= 5; ip++ ) {

        if ( meg02_[ism-1] ) meg02_[ism-1]->setBinContent( ie, ip, 2. );

      }
    }

  }

}

void EEIntegrityClient::cleanup(void) {

  if ( ! enableCleanup_ ) return;

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

    cout << " " << Numbers::sEE(ism) << " (ism=" << ism << ")" << endl;
    cout << endl;

    if ( h00_ && h00_->GetBinContent(ism) != 0 ) {
      cout << " DCC failed " << h00_->GetBinContent(ism) << " times" << endl;
      cout << endl;
    }

    UtilsClient::printBadChannels(meg01_[ism-1], h01_[ism-1], true);
    UtilsClient::printBadChannels(meg01_[ism-1], h02_[ism-1], true);
    UtilsClient::printBadChannels(meg01_[ism-1], h03_[ism-1], true);
    UtilsClient::printBadChannels(meg01_[ism-1], h04_[ism-1], true);
    UtilsClient::printBadChannels(meg01_[ism-1], h05_[ism-1], true);

    UtilsClient::printBadChannels(meg02_[ism-1], h06_[ism-1], true);
    UtilsClient::printBadChannels(meg02_[ism-1], h07_[ism-1], true);
    UtilsClient::printBadChannels(meg02_[ism-1], h08_[ism-1], true);
    UtilsClient::printBadChannels(meg02_[ism-1], h09_[ism-1], true);

    float num00;

    num00 = 0.;

    bool update0 = false;

    if ( h00_ ) {
      num00 = h00_->GetBinContent(ism);
      if ( num00 > 0 ) update0 = true;
    }

    float num01, num02, num03;

    for ( int ix = 1; ix <= 50; ix++ ) {
      for ( int iy = 1; iy <= 50; iy++ ) {

        int jx = ix + Numbers::ix0EE(ism);
        int jy = iy + Numbers::iy0EE(ism);

        if ( ism >= 1 && ism <= 9 ) jx = 101 - jx;

        if ( ! Numbers::validEE(ism, jx, jy) ) continue;

        num01 = num02 = num03 = 0.;

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

        if ( update0 || update1 ) {

          if ( Numbers::icEE(ism, jx, jy) == 1 ) {

            cout << "Preparing dataset for " << Numbers::sEE(ism) << " (ism=" << ism << ")" << endl;

            cout << "(" << Numbers::ix0EE(i+1)+ix << "," << Numbers::iy0EE(i+1)+iy << ") " << num00 << " " << num01 << " " << num02 << " " << num03 << endl;

            cout << endl;

          }

          c1.setProcessedEvents(int(numTot));
          c1.setProblematicEvents(int(num01+num02+num03));
          c1.setProblemsGainZero(int(num01));
          c1.setProblemsID(int(num02));
          c1.setProblemsGainSwitch(int(num03));

          bool val;

          val = true;
          if ( numTot > 0 ) {
            float errorRate1 = num00 / numTot;
            if ( errorRate1 > threshCry_ )
              val = false;
            errorRate1 = ( num01 + num02 + num03 ) / numTot / 4.;
            if ( errorRate1 > threshCry_ )
              val = false;
          } else {
            if ( num00 > 0 )
              val = false;
            if ( ( num01 + num02 + num03 ) > 0 )
              val = false;
          }
          c1.setTaskStatus(val);

          int ic = Numbers::indexEE(ism, jx, jy);

          if ( ic == -1 ) continue;

          if ( econn ) {
            ecid = LogicID::getEcalLogicID("EE_crystal_number", Numbers::iSM(ism, EcalEndcap), ic);
            dataset1[ecid] = c1;
          }

          status = status && val;

        }

      }
    }

    float num04, num05;

    for ( int ixt = 1; ixt <= 10; ixt++ ) {
      for ( int iyt = 1; iyt <= 10; iyt++ ) {

        int jxt = Numbers::ix0EE(ism) + 1 + 5*(ixt-1);
        int jyt = Numbers::iy0EE(ism) + 1 + 5*(iyt-1);

        if ( ism >= 1 && ism <= 9 ) jxt = 101 - jxt;

        num04 = num05 = 0.;

        bool update1 = false;

        float numTot = -1.;

        if ( h_[ism-1] ) {
          numTot = 0.;
          for ( int ix = 1 + 5*(ixt-1); ix <= 5*ixt; ix++ ) {
            for ( int iy = 1 + 5*(iyt-1); iy <= 5*iyt; iy++ ) {
              int jx = ix + Numbers::ix0EE(ism);
              int jy = iy + Numbers::iy0EE(ism);
              if ( ism >= 1 && ism <= 9 ) jx = 101 - jx;
              if ( ! Numbers::validEE(ism, jx, jy) ) continue;
              numTot += h_[ism-1]->GetBinContent(ix, iy);
            }
          }
        }

        if ( h04_[ism-1] ) {
          for ( int ix = 1 + 5*(ixt-1); ix <= 5*ixt; ix++ ) {
            for ( int iy = 1 + 5*(iyt-1); iy <= 5*iyt; iy++ ) {
              int jx = ix + Numbers::ix0EE(ism);
              int jy = iy + Numbers::iy0EE(ism);
              if ( ism >= 1 && ism <= 9 ) jx = 101 - jx;
              if ( ! Numbers::validEE(ism, jx, jy) ) continue;
              num04  = h04_[ism-1]->GetBinContent(ix, iy);
              if ( num04 > 0 ) update1 = true;
            }
          }
        }

        if ( h05_[ism-1] ) {
          for ( int ix = 1 + 5*(ixt-1); ix <= 5*ixt; ix++ ) {
            for ( int iy = 1 + 5*(iyt-1); iy <= 5*iyt; iy++ ) {
              int jx = ix + Numbers::ix0EE(ism);
              int jy = iy + Numbers::iy0EE(ism);
              if ( ism >= 1 && ism <= 9 ) jx = 101 - jx;
              if ( ! Numbers::validEE(ism, jx, jy) ) continue;
              num05  = h05_[ism-1]->GetBinContent(ix, iy);
              if ( num05 > 0 ) update1 = true;
            }
          }
        }

        if ( update0 || update1 ) {

          if ( Numbers::iTT(ism, EcalEndcap, jxt, jyt) == 1 ) {

            cout << "Preparing dataset for " << Numbers::sEE(ism) << " (ism=" << ism << ")" << endl;

            cout << "(" << 1+(Numbers::ix0EE(ism)+1+5*(ixt-1))/5 << "," << 1+(Numbers::iy0EE(ism)+1+5*(iyt-1))/5 << ") " << num00 << " " << num04 << " " << num05 << endl;

            cout << endl;

          }

          c2.setProcessedEvents(int(numTot));
          c2.setProblematicEvents(int(num04+num05));
          c2.setProblemsID(int(num04));
          c2.setProblemsSize(int(num05));
          c2.setProblemsLV1(int(-1.));
          c2.setProblemsBunchX(int(-1.));

          bool val;

          val = true;
          if ( numTot > 0 ) {
            float errorRate2 = num00 / numTot;
            if ( errorRate2 > threshCry_ )
              val = false;
            errorRate2 = ( num04 + num05 ) / numTot / 2.;
            if ( errorRate2 > threshCry_ )
              val = false;
          } else {
            if ( num00 > 0 )
              val = false;
            if ( ( num04 + num05 ) > 0 )
              val = false;
          }
          c2.setTaskStatus(val);

          int itt = Numbers::iTT(ism, EcalEndcap, jxt, jyt);

          if ( itt == -1 ) continue;

          if ( econn ) {
            ecid = LogicID::getEcalLogicID("EE_readout_tower", Numbers::iSM(ism, EcalEndcap), itt);
            dataset2[ecid] = c2;
          }

          status = status && val;

        }

      }
    }

    float num06, num07;

    for ( int ix = 1; ix <= 10; ix++ ) {
      for ( int iy = 1; iy <= 5; iy++ ) {

        num06 = num07 = 0.;

        bool update1 = false;

        float numTot = -1.;

        if ( hmem_[ism-1] ) numTot = hmem_[ism-1]->GetBinContent(ix, iy);

        if ( h06_[ism-1] ) {
          num06  = h06_[ism-1]->GetBinContent(ix, iy);
          if ( num06 > 0 ) update1 = true;
        }

        if ( h07_[ism-1] ) {
          num07  = h07_[ism-1]->GetBinContent(ix, iy);
          if ( num07 > 0 ) update1 = true;
        }

        if ( update0 || update1 ) {

          if ( ix ==1 && iy == 1 ) {

            cout << "Preparing dataset for mem of SM=" << ism << endl;

            cout << "(" << ix << "," << iy << ") " << num06 << " " << num07 << endl;

            cout << endl;

          }

          c3.setProcessedEvents( int (numTot));
          c3.setProblematicEvents(int (num06+num07));
          c3.setProblemsID(int (num06) );
          c3.setProblemsGainZero(int (num07));
          // c3.setProblemsGainSwitch(int prob);

          bool val;

          val = true;
          if ( numTot > 0 ) {
            float errorRate1 = num00 / numTot;
            if ( errorRate1 > threshCry_ )
              val = false;
            errorRate1 = ( num06 + num07 ) / numTot / 2.;
            if ( errorRate1 > threshCry_ )
              val = false;
          } else {
            if ( num00 > 0 )
             val = false;
            if ( ( num06 + num07 ) > 0 )
              val = false;
          }
          c3. setTaskStatus(val);

          int ic = EEIntegrityClient::chNum[ (ix-1)%5 ][ (iy-1) ] + (ix-1)/5 * 25;

          if ( econn ) {
            ecid = LogicID::getEcalLogicID("EE_mem_channel", Numbers::iSM(ism, EcalEndcap), ic);
            dataset3[ecid] = c3;
          }

          status = status && val;

        }

      }
    }

    float num08, num09;

    for ( int ixt = 1; ixt <= 2; ixt++ ) {

      num08 = num09 = 0.;

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

      if ( h08_[ism-1] ) {
        num08  = h08_[ism-1]->GetBinContent(ixt, 1);
        if ( num08 > 0 ) update1 = true;
      }

      if ( h09_[ism-1] ) {
        num09  = h09_[ism-1]->GetBinContent(ixt, 1);
        if ( num09 > 0 ) update1 = true;
      }

      if ( update0 || update1 ) {

        if ( ixt == 1 ) {

          cout << "Preparing dataset for " << Numbers::sEE(ism) << " (ism=" << ism << ")" << endl;

          cout << "(" << ixt <<  ") " << num08 << " " << num09 << endl;

          cout << endl;

        }

        c4.setProcessedEvents( int(numTot) );
        c4.setProblematicEvents( int(num08 + num09) );
        c4.setProblemsID( int(num08) );
        c4.setProblemsSize(int (num09) );
        // setProblemsLV1(int LV1);
        // setProblemsBunchX(int bunchX);

        bool val;

        val = true;
        if ( numTot > 0 ) {
          float errorRate2 = num00 / numTot;
          if ( errorRate2 > threshCry_ )
            val = false;
          errorRate2 = ( num08 + num09 ) / numTot / 2.;
          if ( errorRate2 > threshCry_ )
            val = false;
        } else {
          if ( num00 > 0 )
            val = false;
          if ( ( num08 + num09 ) > 0 )
            val = false;
        }
        c4.setTaskStatus(val);

        int itt = 68 + ixt;

        if ( econn ) {
          ecid = LogicID::getEcalLogicID("EE_mem_TT", Numbers::iSM(ism, EcalEndcap), itt);
          dataset4[ecid] = c4;
        }

        status = status && val;

      }

    }

  }

  if ( econn ) {
    try {
      cout << "Inserting MonConsistencyDat ..." << endl;
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

  char histo[200];

  MonitorElement* me;

  sprintf(histo, (prefixME_+"EcalEndcap/EEIntegrityTask/EEIT DCC size error").c_str());
  me = dbe_->get(histo);
  h00_ = UtilsClient::getHisto<TH1F*>( me, cloneME_, h00_ );

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    sprintf(histo, (prefixME_+"EcalEndcap/EEOccupancyTask/EEOT digi occupancy %s").c_str(), Numbers::sEE(ism).c_str());
    me = dbe_->get(histo);
    h_[ism-1] = UtilsClient::getHisto<TH2F*>( me, cloneME_, h_[ism-1] );

    sprintf(histo, (prefixME_+"EcalEndcap/EEOccupancyTask/EEOT MEM digi occupancy %s").c_str(), Numbers::sEE(ism).c_str());
    me = dbe_->get(histo);
    hmem_[ism-1] = UtilsClient::getHisto<TH2F*>( me, cloneME_, hmem_[ism-1] );

    sprintf(histo, (prefixME_+"EcalEndcap/EEIntegrityTask/Gain/EEIT gain %s").c_str(), Numbers::sEE(ism).c_str());
    me = dbe_->get(histo);
    h01_[ism-1] = UtilsClient::getHisto<TH2F*>( me, cloneME_, h01_[ism-1] );

    sprintf(histo, (prefixME_+"EcalEndcap/EEIntegrityTask/ChId/EEIT ChId %s").c_str(), Numbers::sEE(ism).c_str());
    me = dbe_->get(histo);
    h02_[ism-1] = UtilsClient::getHisto<TH2F*>( me, cloneME_, h02_[ism-1] );

    sprintf(histo, (prefixME_+"EcalEndcap/EEIntegrityTask/GainSwitch/EEIT gain switch %s").c_str(), Numbers::sEE(ism).c_str());
    me = dbe_->get(histo);
    h03_[ism-1] = UtilsClient::getHisto<TH2F*>( me, cloneME_, h03_[ism-1] );

    sprintf(histo, (prefixME_+"EcalEndcap/EEIntegrityTask/TTId/EEIT TTId %s").c_str(), Numbers::sEE(ism).c_str());
    me = dbe_->get(histo);
    h04_[ism-1] = UtilsClient::getHisto<TH2F*>( me, cloneME_, h04_[ism-1] );

    sprintf(histo, (prefixME_+"EcalEndcap/EEIntegrityTask/TTBlockSize/EEIT TTBlockSize %s").c_str(), Numbers::sEE(ism).c_str());
    me = dbe_->get(histo);
    h05_[ism-1] = UtilsClient::getHisto<TH2F*>( me, cloneME_, h05_[ism-1] );

    sprintf(histo, (prefixME_+"EcalEndcap/EEIntegrityTask/MemChId/EEIT MemChId %s").c_str(), Numbers::sEE(ism).c_str());
    me = dbe_->get(histo);
    h06_[ism-1] = UtilsClient::getHisto<TH2F*>( me, cloneME_, h06_[ism-1] );

    sprintf(histo, (prefixME_+"EcalEndcap/EEIntegrityTask/MemGain/EEIT MemGain %s").c_str(), Numbers::sEE(ism).c_str());
    me = dbe_->get(histo);
    h07_[ism-1] = UtilsClient::getHisto<TH2F*>( me, cloneME_, h07_[ism-1] );

    sprintf(histo, (prefixME_+"EcalEndcap/EEIntegrityTask/MemTTId/EEIT MemTTId %s").c_str(), Numbers::sEE(ism).c_str());
    me = dbe_->get(histo);
    h08_[ism-1] = UtilsClient::getHisto<TH2F*>( me, cloneME_, h08_[ism-1] );

    sprintf(histo, (prefixME_+"EcalEndcap/EEIntegrityTask/MemSize/EEIT MemSize %s").c_str(), Numbers::sEE(ism).c_str());
    me = dbe_->get(histo);
    h09_[ism-1] = UtilsClient::getHisto<TH2F*>( me, cloneME_, h09_[ism-1] );

    float num00;

    // integrity summary histograms
    if ( meg01_[ism-1] ) meg01_[ism-1]->Reset();
    if ( meg01_[ism-1] ) meg02_[ism-1]->Reset();

    num00 = 0.;

    bool update0 = false;

    // dcc size errors
    if ( h00_ ) {
      num00  = h00_->GetBinContent(ism);
      update0 = true;
    }

    float num01, num02, num03, num04, num05;

    for ( int ix = 1; ix <= 50; ix++ ) {
      for ( int iy = 1; iy <= 50; iy++ ) {

        num01 = num02 = num03 = num04 = num05 = 0.;

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
          update2 = true;
        }

        if ( h05_[ism-1] ) {
          num05  = h05_[ism-1]->GetBinContent(ix, iy);
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
            errorRate1 = ( num01 + num02 + num03 ) / numTot / 4.;
            if ( errorRate1 > threshCry_ )
              val = 0.;
            float errorRate2 = ( num04 + num05 ) / numTot / 2.;
            if ( errorRate2 > threshCry_ )
              val = 0.;
          } else {
            val = 2.;
            if ( num00 > 0 )
              val = 0.;
            if ( ( num01 + num02 + num03 ) > 0 )
              val = 0.;
            if ( ( num04 + num05 ) > 0 )
              val = 0.;
          }

          int jx = ix + Numbers::ix0EE(ism);
          int jy = iy + Numbers::iy0EE(ism);

          if ( ism >= 1 && ism <= 9 ) jx = 101 - jx;

          // filling the summary for SM channels
          if ( Numbers::validEE(ism, jx, jy) ) {
            if ( meg01_[ism-1] ) meg01_[ism-1]->setBinContent( ix, iy, val );
          }

        }

        // masking

        if ( mask1.size() != 0 ) {
          map<EcalLogicID, RunCrystalErrorsDat>::const_iterator m;
          for (m = mask1.begin(); m != mask1.end(); m++) {

            int jx = ix + Numbers::ix0EE(ism);
            int jy = iy + Numbers::iy0EE(ism);

            if ( ism >= 1 && ism <= 9 ) jx = 101 - jx;

            if ( ! Numbers::validEE(ism, jx, jy) ) continue;

            int ic = Numbers::indexEE(ism, jx, jy);

            if ( ic == -1 ) continue;

            EcalLogicID ecid = m->first;

            if ( ecid.getLogicID() == LogicID::getEcalLogicID("EE_crystal_number", Numbers::iSM(ism, EcalEndcap), ic).getLogicID() ) {
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

            int itt = Numbers::iTT(ism, EcalEndcap, ix, iy);

            if ( ecid.getLogicID() == LogicID::getEcalLogicID("EE_readout_tower", Numbers::iSM(ism, EcalEndcap), itt).getLogicID() ) {
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

    // summaries for mem channels
    float num06, num07, num08, num09;

    for ( int ie = 1; ie <= 10; ie++ ) {
      for ( int ip = 1; ip <= 5; ip++ ) {

        num06 = num07 = num08 = num09 = 0.;

        // initialize summary histo for mem
        if ( meg02_[ism-1] ) meg02_[ism-1]->setBinContent( ie, ip, 2. );

        bool update1 = false;
        bool update2 = false;

        float numTotmem = -1.;

        if ( hmem_[ism-1] ) numTotmem = hmem_[ism-1]->GetBinContent(ie, ip);

        if ( h06_[ism-1] ) {
          num06  = h06_[ism-1]->GetBinContent(ie, ip);
          update1 = true;
        }

        if ( h07_[ism-1] ) {
          num07  = h07_[ism-1]->GetBinContent(ie, ip);
          update1 = true;
        }

        int iet = 1 + ((ie-1)/5);
        int ipt = 1;

        if ( h08_[ism-1] ) {
          num08  = h08_[ism-1]->GetBinContent(iet, ipt);
          update2 = true;
        }

        if ( h09_[ism-1] ) {
          num09  = h09_[ism-1]->GetBinContent(iet, ipt);
          update2 = true;
        }


        if ( update0 || update1 || update2 ) {

          float val;

          val = 1.;
          // numer of events on a channel
          if ( numTotmem > 0 ) {
            float errorRate1 = ( num06 + num06 ) / numTotmem / 2.;
            if ( errorRate1 > threshCry_ )
              val = 0.;
            float errorRate2 = ( num08 + num09 ) / numTotmem / 2.;
            if ( errorRate2 > threshCry_ )
              val = 0.;
          } else {
            val = 2.;
            if ( ( num06 + num07 ) > 0 )
              val = 0.;
            if ( ( num08 + num09 ) > 0 )
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

            if ( ecid.getLogicID() == LogicID::getEcalLogicID("EE_mem_channel", Numbers::iSM(ism, EcalEndcap), ic).getLogicID() ) {
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

            if ( ecid.getLogicID() == LogicID::getEcalLogicID("EE_mem_TT", Numbers::iSM(ism, EcalEndcap), itt).getLogicID() ) {
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

  }// end loop on supermodules

}

void EEIntegrityClient::htmlOutput(int run, string& htmlDir, string& htmlName){

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
             << Numbers::sEE(superModules_[i]) << ">"
             << setfill( '0' ) << setw(2) << superModules_[i] << "</a></td>";
  }
  htmlFile << std::endl << "</table>" << std::endl;
  htmlFile << "<hr>" << std::endl;

  // Produce the plots to be shown as .png files from existing histograms

  const int csize = 250;

  int pCol3[6] = { 301, 302, 303, 304, 305, 306 };
  int pCol4[10];
  for ( int i = 0; i < 10; i++ ) pCol4[i] = 401+i;
  int pCol5[10];
  for ( int i = 0; i < 10; i++ ) pCol5[i] = 501+i;

  TH2S labelGrid("labelGrid","label grid", 100, -2., 98., 100, -2., 98.);
  for ( short j=0; j<400; j++ ) {
    int x = 5*(1 + j%20);
    int y = 5*(1 + j/20);
    labelGrid.SetBinContent(x, y, Numbers::inTowersEE[j]);
  }
  labelGrid.SetMarkerSize(1);
  labelGrid.SetMinimum(0.1);

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

  string imgNameDCC, imgNameOcc, imgNameQual,imgNameOccMem, imgNameQualMem, imgNameME[9], imgName, meName;

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

    replace(meName.begin(), meName.end(), ' ', '_');
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

  // Loop on endcap sectors

  for ( unsigned int i=0; i<superModules_.size(); i ++ ) {

    int ism = superModules_[i];

    // Quality plots

    imgNameQual = "";

    obj2f = UtilsClient::getHisto<TH2F*>( meg01_[ism-1] );

    if ( obj2f ) {

      meName = obj2f->GetName();

      replace(meName.begin(), meName.end(), ' ', '_');
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
      obj2f->GetXaxis()->SetTitleSize(0.02);
      obj2f->GetYaxis()->SetLabelSize(0.02);
      obj2f->GetYaxis()->SetTitleSize(0.02);
      obj2f->Draw("col");
      int x1 = labelGrid.GetXaxis()->FindFixBin(Numbers::ix0EE(ism)+0.);
      int x2 = labelGrid.GetXaxis()->FindFixBin(Numbers::ix0EE(ism)+50.);
      int y1 = labelGrid.GetYaxis()->FindFixBin(Numbers::iy0EE(ism)+0.);
      int y2 = labelGrid.GetYaxis()->FindFixBin(Numbers::iy0EE(ism)+50.);
      labelGrid.GetXaxis()->SetRange(x1, x2);
      labelGrid.GetYaxis()->SetRange(y1, y2);
      labelGrid.Draw("text,same");
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

      replace(meName.begin(), meName.end(), ' ', '_');
      imgNameOcc = meName + ".png";
      imgName = htmlDir + imgNameOcc;

      cOcc->cd();
      gStyle->SetOptStat(" ");
      gStyle->SetPalette(10, pCol4);
      cOcc->SetGridx();
      cOcc->SetGridy();
      obj2f->GetXaxis()->SetLabelSize(0.02);
      obj2f->GetXaxis()->SetTitleSize(0.02);
      obj2f->GetYaxis()->SetLabelSize(0.02);
      obj2f->GetYaxis()->SetTitleSize(0.02);
      obj2f->GetZaxis()->SetLabelSize(0.02);
      obj2f->SetMinimum(0.0);
      obj2f->Draw("colz");
      int x1 = labelGrid.GetXaxis()->FindFixBin(Numbers::ix0EE(ism)+0.);
      int x2 = labelGrid.GetXaxis()->FindFixBin(Numbers::ix0EE(ism)+50.);
      int y1 = labelGrid.GetYaxis()->FindFixBin(Numbers::iy0EE(ism)+0.);
      int y2 = labelGrid.GetYaxis()->FindFixBin(Numbers::iy0EE(ism)+50.);
      labelGrid.GetXaxis()->SetRange(x1, x2);
      labelGrid.GetYaxis()->SetRange(y1, y2);
      labelGrid.Draw("text,same");
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

    for ( int iCanvas = 1; iCanvas <= 5; iCanvas++ ) {

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
      default:
        break;
      }

      if ( obj2f ) {

        meName = obj2f->GetName();

        replace(meName.begin(), meName.end(), ' ', '_');
        imgNameME[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgNameME[iCanvas-1];

        cMe->cd();
        gStyle->SetOptStat(" ");
        gStyle->SetPalette(10, pCol5);
        cMe->SetGridx();
        cMe->SetGridy();
        obj2f->GetXaxis()->SetLabelSize(0.02);
        obj2f->GetXaxis()->SetTitleSize(0.02);
        obj2f->GetYaxis()->SetLabelSize(0.02);
        obj2f->GetYaxis()->SetTitleSize(0.02);
        obj2f->GetZaxis()->SetLabelSize(0.02);
        obj2f->SetMinimum(0.0);
        obj2f->Draw("colz");
        int x1 = labelGrid.GetXaxis()->FindFixBin(Numbers::ix0EE(ism)+0.);
        int x2 = labelGrid.GetXaxis()->FindFixBin(Numbers::ix0EE(ism)+50.);
        int y1 = labelGrid.GetYaxis()->FindFixBin(Numbers::iy0EE(ism)+0.);
        int y2 = labelGrid.GetYaxis()->FindFixBin(Numbers::iy0EE(ism)+50.);
        labelGrid.GetXaxis()->SetRange(x1, x2);
        labelGrid.GetYaxis()->SetRange(y1, y2);
        labelGrid.Draw("text,same");
        cMe->SetBit(TGraph::kClipFrame);
        TLine l;
        l.SetLineWidth(1);
        for ( int i=0; i<201; i=i+1){
          if ( (Numbers::ixSectorsEE[i]!=0 || Numbers::iySectorsEE[i]!=0) && (Numbers::ixSectorsEE[i+1]!=0 || Numbers::iySectorsEE[i+1]!=0) ) {
            l.DrawLine(Numbers::ixSectorsEE[i], Numbers::iySectorsEE[i], Numbers::ixSectorsEE[i+1], Numbers::iySectorsEE[i+1]);
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

      replace(meName.begin(), meName.end(), ' ', '_');
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

      replace(meName.begin(), meName.end(), ' ', '_');
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
      obj2f->GetXaxis()->SetTitleSize(0.02);
      obj2f->GetYaxis()->SetLabelSize(0.02);
      obj2f->GetYaxis()->SetTitleSize(0.02);
      obj2f->GetZaxis()->SetLabelSize(0.02);
      obj2f->SetMinimum(0.0);
      obj2f->Draw("colz");
      dummy3.Draw("text,same");
      cMeMem->Update();
      cMeMem->SaveAs(imgName.c_str());

    }

    // MeM Monitoring elements plots

    for ( int iCanvas = 6; iCanvas <= 9; iCanvas++ ) {

      imgNameME[iCanvas-1] = "";

      obj2f = 0;
      switch ( iCanvas ) {
      case 6:
        obj2f = h06_[ism-1];
        break;
      case 7:
        obj2f = h07_[ism-1];
        break;
      case 8:
        obj2f = h08_[ism-1];
        break;
      case 9:
        obj2f = h09_[ism-1];
        break;
      default:
        break;
      }

      if ( obj2f ) {

        meName = obj2f->GetName();

        replace(meName.begin(), meName.end(), ' ', '_');
        imgNameME[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgNameME[iCanvas-1];

        cMeMem->cd();
        gStyle->SetOptStat(" ");
        gStyle->SetPalette(10, pCol5);
        obj2f->SetMinimum(0.0);
        obj2f->Draw("colz");
        if ( iCanvas <= 7 ){
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
             << Numbers::sEE(ism) << """></a><strong>"
             << Numbers::sEE(ism) << "</strong></h3>" << endl;
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
    htmlFile << "cellpadding=\"10\"> " << endl;
    htmlFile << "<tr align=\"left\">" << endl;

    int iCanvas = 3;

    if ( imgNameME[iCanvas-1].size() != 0 )
      htmlFile << "<td><img src=\"" << imgNameME[iCanvas-1] << "\"></td>" << endl;
    else
      htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

    htmlFile << "</tr>" << endl;
    htmlFile << "</table>" << endl;
    htmlFile << "<br>" << endl;

    htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
    htmlFile << "<tr align=\"center\">" << endl;

    for ( int iCanvas = 4 ; iCanvas <= 5 ; iCanvas++ ) {

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

    for ( int iCanvas = 6 ; iCanvas <= 7 ; iCanvas++ ) {

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

    for ( int iCanvas = 8 ; iCanvas <= 9 ; iCanvas++ ) {

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

