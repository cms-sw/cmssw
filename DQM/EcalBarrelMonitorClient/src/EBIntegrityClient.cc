
/*
 * \file EBIntegrityClient.cc
 *
 * $Date: 2012/04/27 13:45:59 $
 * $Revision: 1.235 $
 * \author G. Della Ricca
 * \author G. Franzoni
 *
 */

#include <memory>
#include <iostream>
#include <fstream>
#include <iomanip>

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#ifdef WITH_ECAL_COND_DB
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
#include "DQM/EcalCommon/interface/LogicID.h"
#endif

#include "DQM/EcalCommon/interface/Masks.h"

#include "DQM/EcalCommon/interface/UtilsClient.h"
#include "DQM/EcalCommon/interface/Numbers.h"

#include "DQM/EcalBarrelMonitorClient/interface/EBIntegrityClient.h"

EBIntegrityClient::EBIntegrityClient(const edm::ParameterSet& ps) {

  // cloneME switch
  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);

  // verbose switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", true);

  // debug switch
  debug_ = ps.getUntrackedParameter<bool>("debug", false);

  // prefixME path
  prefixME_ = ps.getUntrackedParameter<std::string>("prefixME", "");

  subfolder_ = ps.getUntrackedParameter<std::string>("subfolder", "");

  // enableCleanup_ switch
  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  // vector of selected Super Modules (Defaults to all 36).
  superModules_.reserve(36);
  for ( unsigned int i = 1; i <= 36; i++ ) superModules_.push_back(i);
  superModules_ = ps.getUntrackedParameter<std::vector<int> >("superModules", superModules_);

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

  threshCry_ = 0.01;

}

EBIntegrityClient::~EBIntegrityClient() {

}

void EBIntegrityClient::beginJob(void) {

  dqmStore_ = edm::Service<DQMStore>().operator->();

  if ( debug_ ) std::cout << "EBIntegrityClient: beginJob" << std::endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EBIntegrityClient::beginRun(void) {

  if ( debug_ ) std::cout << "EBIntegrityClient: beginRun" << std::endl;

  jevt_ = 0;

  this->setup();

}

void EBIntegrityClient::endJob(void) {

  if ( debug_ ) std::cout << "EBIntegrityClient: endJob, ievt = " << ievt_ << std::endl;

  this->cleanup();

}

void EBIntegrityClient::endRun(void) {

  if ( debug_ ) std::cout << "EBIntegrityClient: endRun, jevt = " << jevt_ << std::endl;

  this->cleanup();

}

void EBIntegrityClient::setup(void) {

  std::string name;

  dqmStore_->setCurrentFolder( prefixME_ + "/EBIntegrityClient" );

  if(subfolder_.size())
    dqmStore_->setCurrentFolder( prefixME_ + "/EBIntegrityClient/" + subfolder_);

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( meg01_[ism-1] ) dqmStore_->removeElement( meg01_[ism-1]->getName() );
    name = "EBIT data integrity quality " + Numbers::sEB(ism);
    meg01_[ism-1] = dqmStore_->book2D(name, name, 85, 0., 85., 20, 0., 20.);
    meg01_[ism-1]->setAxisTitle("ieta", 1);
    meg01_[ism-1]->setAxisTitle("iphi", 2);

    if ( meg02_[ism-1] ) dqmStore_->removeElement( meg02_[ism-1]->getName() );
    name = "EBIT data integrity quality MEM " + Numbers::sEB(ism);
    meg02_[ism-1] = dqmStore_->book2D(name, name, 10, 0., 10., 5, 0.,5.);
    meg02_[ism-1]->setAxisTitle("pseudo-strip", 1);
    meg02_[ism-1]->setAxisTitle("channel", 2);

  }

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( meg01_[ism-1] ) meg01_[ism-1]->Reset();
    if ( meg02_[ism-1] ) meg02_[ism-1]->Reset();

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

        if ( meg01_[ism-1] ) meg01_[ism-1]->setBinContent( ie, ip, 2. );

      }
    }

    for ( int ie = 1; ie <= 10; ie++ ) {
      for ( int ip = 1; ip <= 5; ip++ ) {

        if ( meg02_[ism-1] ) meg02_[ism-1]->setBinContent( ie, ip, 2. );

      }
    }

  }

}

void EBIntegrityClient::cleanup(void) {

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

  dqmStore_->setCurrentFolder( prefixME_ + "/EBIntegrityClient" );

  if(subfolder_.size())
    dqmStore_->setCurrentFolder( prefixME_ + "/EBIntegrityClient/" + subfolder_);

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( meg01_[ism-1] ) dqmStore_->removeElement( meg01_[ism-1]->getName() );
    meg01_[ism-1] = 0;

    if ( meg02_[ism-1] ) dqmStore_->removeElement( meg02_[ism-1]->getName() );
    meg02_[ism-1] = 0;

  }

}

#ifdef WITH_ECAL_COND_DB
bool EBIntegrityClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov, bool& status) {

  status = true;

  EcalLogicID ecid;

  MonCrystalConsistencyDat c1;
  std::map<EcalLogicID, MonCrystalConsistencyDat> dataset1;
  MonTTConsistencyDat c2;
  std::map<EcalLogicID, MonTTConsistencyDat> dataset2;
  MonMemChConsistencyDat c3;
  std::map<EcalLogicID, MonMemChConsistencyDat> dataset3;
  MonMemTTConsistencyDat c4;
  std::map<EcalLogicID, MonMemTTConsistencyDat> dataset4;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( h00_ && h00_->GetBinContent(ism) != 0 ) {
      std::cerr << " DCC failed " << h00_->GetBinContent(ism) << " times" << std::endl;
      std::cerr << std::endl;
    }

    if ( verbose_ ) {
      std::cout << " " << Numbers::sEB(ism) << " (ism=" << ism << ")" << std::endl;
      std::cout << std::endl;
      UtilsClient::printBadChannels(meg01_[ism-1], h01_[ism-1], true);
      UtilsClient::printBadChannels(meg01_[ism-1], h02_[ism-1], true);
      UtilsClient::printBadChannels(meg01_[ism-1], h03_[ism-1], true);
      UtilsClient::printBadChannels(meg01_[ism-1], h04_[ism-1], true);
      UtilsClient::printBadChannels(meg01_[ism-1], h05_[ism-1], true);

      UtilsClient::printBadChannels(meg02_[ism-1], h06_[ism-1], true);
      UtilsClient::printBadChannels(meg02_[ism-1], h07_[ism-1], true);
      UtilsClient::printBadChannels(meg02_[ism-1], h08_[ism-1], true);
      UtilsClient::printBadChannels(meg02_[ism-1], h09_[ism-1], true);
    }

    float num00;

    num00 = 0.;

    bool update0 = false;

    if ( h00_ ) {
      num00 = h00_->GetBinContent(ism);
      if ( num00 > 0 ) update0 = true;
    }

    float num01, num02, num03;

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

        num01 = num02 = num03 = 0.;

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

        if ( update0 || update1 ) {

          if ( Numbers::icEB(ism, ie, ip) == 1 ) {

            if ( verbose_ ) {
              std::cout << "Preparing dataset for " << Numbers::sEB(ism) << " (ism=" << ism << ")" << std::endl;
              std::cout << "(" << ie << "," << ip << ") " << num00 << " " << num01 << " " << num02 << " " << num03 << std::endl;
              std::cout << std::endl;
            }

          }

          c1.setProcessedEvents(int(numTot));
          c1.setProblematicEvents(int(num01+num02+num03));
          c1.setProblemsGainZero(int(num01));
          c1.setProblemsID(int(num02));
          c1.setProblemsGainSwitch(int(num03));

          bool val;

          val = false;
          if ( numTot > 0 ) {
            float errorRate1 = num00 / ( numTot + num01 + num02 + num03 );
            if ( errorRate1 > threshCry_ )
              val = true;
            errorRate1 = ( num01 + num02 + num03 ) / ( numTot + num01 + num02 + num03 ) / 3.;
            if ( errorRate1 > threshCry_ )
              val = true;
          } else {
            if ( num00 > 0 )
              val = true;
            if ( ( num01 + num02 + num03 ) > 0 )
              val = true;
          }
          c1.setTaskStatus(val);

          int ic = Numbers::indexEB(ism, ie, ip);

          if ( econn ) {
            ecid = LogicID::getEcalLogicID("EB_crystal_number", Numbers::iSM(ism, EcalBarrel), ic);
            dataset1[ecid] = c1;
          }

          status = status && !val;

        }

      }
    }

    float num04, num05;

    for ( int iet = 1; iet <= 17; iet++ ) {
      for ( int ipt = 1; ipt <= 4; ipt++ ) {

        num04 = num05 = 0.;

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

        if ( h04_[ism-1] ) {
          num04  = h04_[ism-1]->GetBinContent(iet, ipt);
          if ( num04 > 0 ) update1 = true;
        }

        if ( h05_[ism-1] ) {
          num05  = h05_[ism-1]->GetBinContent(iet, ipt);
          if ( num05 > 0 ) update1 = true;
        }

        if ( update0 || update1 ) {

          if ( Numbers::iSC(ism, EcalBarrel, 1+5*(iet-1), 1+5*(ipt-1)) == 1 ) {

            if ( verbose_ ) {
              std::cout << "Preparing dataset for " << Numbers::sEB(ism) << " (ism=" << ism << ")" << std::endl;
              std::cout << "(" << iet << "," << ipt << ") " << num00 << " " << num04 << " " << num05 << std::endl;
              std::cout << std::endl;
            }

          }

          c2.setProcessedEvents(int(numTot));
          c2.setProblematicEvents(int(num04+num05));
          c2.setProblemsID(int(num04));
          c2.setProblemsSize(int(num05));
          c2.setProblemsLV1(int(-1.));
          c2.setProblemsBunchX(int(-1.));

          bool val;

          val = false;
          if ( numTot > 0 ) {
            float errorRate2 = num00 / ( numTot/25. + num04 + num05 );
            if ( errorRate2 > threshCry_ )
              val = true;
            errorRate2 = ( num04 + num05 ) / ( numTot/25. + num04 + num05 ) / 2.;
            if ( errorRate2 > threshCry_ )
              val = true;
          } else {
            if ( num00 > 0 )
              val = true;
            if ( ( num04 + num05 ) > 0 )
              val = true;
          }
          c2.setTaskStatus(val);

          int itt = Numbers::iSC(ism, EcalBarrel, 1+5*(iet-1), 1+5*(ipt-1));

          if ( econn ) {
            ecid = LogicID::getEcalLogicID("EB_trigger_tower", Numbers::iSM(ism, EcalBarrel), itt);
            dataset2[ecid] = c2;
          }

          status = status && !val;

        }

      }
    }

    float num06, num07;

    for ( int ie = 1; ie <= 10; ie++ ) {
      for ( int ip = 1; ip <= 5; ip++ ) {

        num06 = num07 = 0.;

        bool update1 = false;

        float numTot = -1.;

        if ( hmem_[ism-1] ) numTot = hmem_[ism-1]->GetBinContent(ie, ip);

        if ( h06_[ism-1] ) {
          num06  = h06_[ism-1]->GetBinContent(ie, ip);
          if ( num06 > 0 ) update1 = true;
        }

        if ( h07_[ism-1] ) {
          num07  = h07_[ism-1]->GetBinContent(ie, ip);
          if ( num07 > 0 ) update1 = true;
        }

        if ( update0 || update1 ) {

          if ( ie ==1 && ip == 1 ) {

            if ( verbose_ ) {
              std::cout << "Preparing dataset for mem of SM=" << ism << std::endl;
              std::cout << "(" << ie << "," << ip << ") " << num06 << " " << num07 << std::endl;
              std::cout << std::endl;
            }

          }

          c3.setProcessedEvents( int (numTot));
          c3.setProblematicEvents(int (num06+num07));
          c3.setProblemsID(int (num06) );
          c3.setProblemsGainZero(int (num07));
          // c3.setProblemsGainSwitch(int prob);

          bool val;

          val = false;
          if ( numTot > 0 ) {
            float errorRate1 = num00 / ( numTot + num06 + num07 );
            if ( errorRate1 > threshCry_ )
              val = true;
            errorRate1 = ( num06 + num07 ) / ( numTot + num06 + num07 ) / 2.;
            if ( errorRate1 > threshCry_ )
              val = true;
          } else {
            if ( num00 > 0 )
             val = true;
            if ( ( num06 + num07 ) > 0 )
              val = true;
          }
          c3. setTaskStatus(val);

          int ic = EBIntegrityClient::chNum[ (ie-1)%5 ][ (ip-1) ] + (ie-1)/5 * 25;

          if ( econn ) {
            ecid = LogicID::getEcalLogicID("EB_mem_channel", Numbers::iSM(ism, EcalBarrel), ic);
            dataset3[ecid] = c3;
          }

          status = status && !val;

        }

      }
    }

    float num08, num09;

    for ( int iet = 1; iet <= 2; iet++ ) {

      num08 = num09 = 0.;

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

      if ( h08_[ism-1] ) {
        num08  = h08_[ism-1]->GetBinContent(iet, 1);
        if ( num08 > 0 ) update1 = true;
      }

      if ( h09_[ism-1] ) {
        num09  = h09_[ism-1]->GetBinContent(iet, 1);
        if ( num09 > 0 ) update1 = true;
      }

      if ( update0 || update1 ) {

        if ( iet == 1 ) {

          if ( verbose_ ) {
            std::cout << "Preparing dataset for " << Numbers::sEB(ism) << " (ism=" << ism << ")" << std::endl;
            std::cout << "(" << iet <<  ") " << num08 << " " << num09 << std::endl;
            std::cout << std::endl;
          }

        }

        c4.setProcessedEvents( int(numTot) );
        c4.setProblematicEvents( int(num08 + num09) );
        c4.setProblemsID( int(num08) );
        c4.setProblemsSize(int (num09) );
        // setProblemsLV1(int LV1);
        // setProblemsBunchX(int bunchX);

        bool val;

        val = false;
        if ( numTot > 0 ) {
          float errorRate2 = num00 / ( numTot/25. + num08 + num09 );
          if ( errorRate2 > threshCry_ )
            val = true;
          errorRate2 = ( num08 + num09 ) / ( numTot/25. + num08 + num09 ) / 2.;
          if ( errorRate2 > threshCry_ )
            val = true;
        } else {
          if ( num00 > 0 )
            val = true;
          if ( ( num08 + num09 ) > 0 )
            val = true;
        }
        c4.setTaskStatus(val);

        int itt = 68 + iet;

        if ( econn ) {
          ecid = LogicID::getEcalLogicID("EB_mem_TT", Numbers::iSM(ism, EcalBarrel), itt);
          dataset4[ecid] = c4;
        }

        status = status && !val;

      }

    }

  }

  if ( econn ) {
    try {
      if ( verbose_ ) std::cout << "Inserting MonConsistencyDat ..." << std::endl;
      if ( dataset1.size() != 0 ) econn->insertDataArraySet(&dataset1, moniov);
      if ( dataset2.size() != 0 ) econn->insertDataArraySet(&dataset2, moniov);
      if ( dataset3.size() != 0 ) econn->insertDataArraySet(&dataset3, moniov);
      if ( dataset4.size() != 0 ) econn->insertDataArraySet(&dataset4, moniov);
      if ( verbose_ ) std::cout << "done." << std::endl;
    } catch (std::runtime_error &e) {
      std::cerr << e.what() << std::endl;
    }
  }

  return true;

}
#endif

void EBIntegrityClient::analyze(void) {

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( debug_ ) std::cout << "EBIntegrityClient: ievt/jevt = " << ievt_ << "/" << jevt_ << std::endl;
  }

  uint32_t bits01 = 0;
  bits01 |= 1 << EcalDQMStatusHelper::CH_ID_ERROR;
  bits01 |= 1 << EcalDQMStatusHelper::CH_GAIN_ZERO_ERROR;
  bits01 |= 1 << EcalDQMStatusHelper::CH_GAIN_SWITCH_ERROR;
  bits01 |= 1 << EcalDQMStatusHelper::TT_ID_ERROR;
  bits01 |= 1 << EcalDQMStatusHelper::TT_SIZE_ERROR;

  std::string subdir(subfolder_.size() ? subfolder_ + "/" : "");

  MonitorElement* me;

  me = dqmStore_->get( prefixME_ + "/EBIntegrityTask/" + subdir + "EBIT DCC size error" );
  h00_ = UtilsClient::getHisto( me, cloneME_, h00_ );

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    me = dqmStore_->get( prefixME_ + "/EBOccupancyTask/" + subdir + "EBOT digi occupancy " + Numbers::sEB(ism) );
    h_[ism-1] = UtilsClient::getHisto( me, cloneME_, h_[ism-1] );

    me = dqmStore_->get( prefixME_ + "/EBOccupancyTask/" + subdir + "EBOT MEM digi occupancy " + Numbers::sEB(ism) );
    hmem_[ism-1] = UtilsClient::getHisto( me, cloneME_, hmem_[ism-1] );

    me = dqmStore_->get( prefixME_ + "/EBIntegrityTask/" + subdir + "Gain/EBIT gain " + Numbers::sEB(ism) );
    h01_[ism-1] = UtilsClient::getHisto( me, cloneME_, h01_[ism-1] );

    me = dqmStore_->get( prefixME_ + "/EBIntegrityTask/" + subdir + "ChId/EBIT ChId " + Numbers::sEB(ism) );
    h02_[ism-1] = UtilsClient::getHisto( me, cloneME_, h02_[ism-1] );

    me = dqmStore_->get( prefixME_ + "/EBIntegrityTask/" + subdir + "GainSwitch/EBIT gain switch " + Numbers::sEB(ism) );
    h03_[ism-1] = UtilsClient::getHisto( me, cloneME_, h03_[ism-1] );

    me = dqmStore_->get( prefixME_ + "/EBIntegrityTask/" + subdir + "TTId/EBIT TTId " + Numbers::sEB(ism) );
    h04_[ism-1] = UtilsClient::getHisto( me, cloneME_, h04_[ism-1] );

    me = dqmStore_->get( prefixME_ + "/EBIntegrityTask/" + subdir + "TTBlockSize/EBIT TTBlockSize " + Numbers::sEB(ism) );
    h05_[ism-1] = UtilsClient::getHisto( me, cloneME_, h05_[ism-1] );

    me = dqmStore_->get( prefixME_ + "/EBIntegrityTask/" + subdir + "MemChId/EBIT MemChId " + Numbers::sEB(ism) );
    h06_[ism-1] = UtilsClient::getHisto( me, cloneME_, h06_[ism-1] );

    me = dqmStore_->get( prefixME_ + "/EBIntegrityTask/" + subdir + "MemGain/EBIT MemGain " + Numbers::sEB(ism) );
    h07_[ism-1] = UtilsClient::getHisto( me, cloneME_, h07_[ism-1] );

    me = dqmStore_->get( prefixME_ + "/EBIntegrityTask/" + subdir + "MemTTId/EBIT MemTTId " + Numbers::sEB(ism) );
    h08_[ism-1] = UtilsClient::getHisto( me, cloneME_, h08_[ism-1] );

    me = dqmStore_->get( prefixME_ + "/EBIntegrityTask/" + subdir + "MemSize/EBIT MemSize " + Numbers::sEB(ism) );
    h09_[ism-1] = UtilsClient::getHisto( me, cloneME_, h09_[ism-1] );

    float num00;

    // integrity summary histograms
    if ( meg01_[ism-1] ) meg01_[ism-1]->Reset();
    if ( meg02_[ism-1] ) meg02_[ism-1]->Reset();

    num00 = 0.;

    bool update0 = false;

    // dcc size errors
    if ( h00_ ) {
      num00  = h00_->GetBinContent(ism);
      update0 = true;
    }

    float num01, num02, num03, num04, num05;

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

        num01 = num02 = num03 = num04 = num05 = 0.;

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

        int iet = 1 + ((ie-1)/5);
        int ipt = 1 + ((ip-1)/5);

        if ( h04_[ism-1] ) {
          num04  = h04_[ism-1]->GetBinContent(iet, ipt);
          update2 = true;
        }

        if ( h05_[ism-1] ) {
          num05  = h05_[ism-1]->GetBinContent(iet, ipt);
          update2 = true;
        }

        if ( update0 || update1 || update2 ) {

          float val;

          val = 1.;
          // number of events on a channel
          if ( numTot > 0 ) {
            float errorRate1 =  num00 / ( numTot + num01 + num02 + num03 );
            if ( errorRate1 > threshCry_ )
              val = 0.;
            errorRate1 = ( num01 + num02 + num03 ) / ( numTot + num01 + num02 + num03 ) / 3.;
            if ( errorRate1 > threshCry_ )
              val = 0.;
            float errorRate2 = ( num04 + num05 ) / ( numTot/25. + num04 + num05 ) / 2.;
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

          // filling the summary for SM channels
          if ( meg01_[ism-1] ) meg01_[ism-1]->setBinContent( ie, ip, val );

        }

        if ( Masks::maskChannel(ism, ie, ip, bits01, EcalBarrel) ) UtilsClient::maskBinContent( meg01_[ism-1], ie, ip );

      }
    } // end of loop on crystals

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
          // number of events on a channel
          if ( numTotmem > 0 ) {
            float errorRate1 = ( num06 + num07 ) / ( numTotmem + num06 + num07 ) / 2.;
            if ( errorRate1 > threshCry_ )
              val = 0.;
            float errorRate2 = ( num08 + num09 ) / ( numTotmem/25. + num08 + num09 ) / 2.;
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

        if ( Masks::maskPn(ism, ie, bits01, EcalBarrel) ) UtilsClient::maskBinContent( meg02_[ism-1], ie, ip );

      }
    }  // end loop on mem channels

  } // end loop on supermodules

}

const int EBIntegrityClient::chNum[5][5] = {
  { 1,  2,  3,  4,  5},
  {10,  9,  8,  7,  6},
  {11, 12, 13, 14, 15},
  {20, 19, 18, 17, 16},
  {21, 22, 23, 24, 25}
};

