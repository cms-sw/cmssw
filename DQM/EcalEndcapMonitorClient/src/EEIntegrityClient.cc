
/*
 * \file EEIntegrityClient.cc
 *
 * $Date: 2011/09/02 13:55:02 $
 * $Revision: 1.114 $
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

#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EcalScDetId.h"
#include "DataFormats/EcalDetId/interface/EcalElectronicsId.h"

#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"

#include "DQM/EcalEndcapMonitorClient/interface/EEIntegrityClient.h"

#include "TString.h"
#include "TObjString.h"
#include "TPRegexp.h"
#include "TObjArray.h"

EEIntegrityClient::EEIntegrityClient(const edm::ParameterSet& ps) {

  // cloneME switch
  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);

  // verbose switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", true);

  // debug switch
  debug_ = ps.getUntrackedParameter<bool>("debug", false);

  // prefixME path
  prefixME_ = ps.getUntrackedParameter<std::string>("prefixME", "");

  // enableCleanup_ switch
  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  // vector of selected Super Modules (Defaults to all 18).
  superModules_.reserve(18);
  for ( unsigned int i = 1; i <= 18; i++ ) superModules_.push_back(i);
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

  ievt_ = 0;
  jevt_ = 0;
  dqmStore_ = 0;

}

EEIntegrityClient::~EEIntegrityClient() {

}

void EEIntegrityClient::beginJob(void) {

  dqmStore_ = edm::Service<DQMStore>().operator->();

  if ( debug_ ) std::cout << "EEIntegrityClient: beginJob" << std::endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EEIntegrityClient::beginRun(void) {

  if ( debug_ ) std::cout << "EEIntegrityClient: beginRun" << std::endl;

  jevt_ = 0;

  this->setup();

}

void EEIntegrityClient::endJob(void) {

  if ( debug_ ) std::cout << "EEIntegrityClient: endJob, ievt = " << ievt_ << std::endl;

  this->cleanup();

}

void EEIntegrityClient::endRun(void) {

  if ( debug_ ) std::cout << "EEIntegrityClient: endRun, jevt = " << jevt_ << std::endl;

  this->cleanup();

}

void EEIntegrityClient::setup(void) {

  std::string name;

  dqmStore_->setCurrentFolder( prefixME_ + "/IntegrityErrors" );
  dqmStore_->setCurrentFolder( prefixME_ + "/IntegrityErrors/Quality" );

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( meg01_[ism-1] ) dqmStore_->removeElement( meg01_[ism-1]->getFullname() );
    name = "IntegrityClient data integrity quality " + Numbers::sEE(ism);
    meg01_[ism-1] = dqmStore_->book2D(name, name, 50, Numbers::ix0EE(ism)+0., Numbers::ix0EE(ism)+50., 50, Numbers::iy0EE(ism)+0., Numbers::iy0EE(ism)+50.);
    meg01_[ism-1]->setAxisTitle("ix", 1);
    if ( ism >= 1 && ism <= 9 ) meg01_[ism-1]->setAxisTitle("101-ix", 1);
    meg01_[ism-1]->setAxisTitle("iy", 2);

    if ( meg02_[ism-1] ) dqmStore_->removeElement( meg02_[ism-1]->getFullname() );
    name = "IntegrityClient data integrity quality MEM " + Numbers::sEE(ism);
    meg02_[ism-1] = dqmStore_->book2D(name, name, 10, 0., 10., 5, 0.,5.);
    meg02_[ism-1]->setAxisTitle("pseudo-strip", 1);
    meg02_[ism-1]->setAxisTitle("channel", 2);

  }

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( meg01_[ism-1] ) meg01_[ism-1]->Reset();
    if ( meg02_[ism-1] ) meg02_[ism-1]->Reset();

    for ( int ix = 1; ix <= 50; ix++ ) {
      for ( int iy = 1; iy <= 50; iy++ ) {

        if ( meg01_[ism-1] ) meg01_[ism-1]->setBinContent( ix, iy, 6. );

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

        if ( meg02_[ism-1] ) meg02_[ism-1]->setBinContent( ie, ip, 6. );

        // non-existing mem
        if ( (ism >=  3 && ism <=  4) || (ism >=  7 && ism <=  9) ) continue;
        if ( (ism >= 12 && ism <=  3) || (ism >= 16 && ism <= 18) ) continue;

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

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( meg01_[ism-1] ) dqmStore_->removeElement( meg01_[ism-1]->getFullname() );
    meg01_[ism-1] = 0;

    if ( meg02_[ism-1] ) dqmStore_->removeElement( meg02_[ism-1]->getFullname() );
    meg02_[ism-1] = 0;

  }

}

#ifdef WITH_ECAL_COND_DB
bool EEIntegrityClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov, bool& status) {

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
    int zside = ism <= 9 ? -1 : 1;
    int idcc = ism <= 9 ? ism : ism + 36;

    if ( h00_ && h00_->GetBinContent(ism) != 0 ) {
      std::cerr << std::endl;
      std::cerr << " DCC failed " << h00_->GetBinContent(ism) << " times" << std::endl;
      std::cerr << std::endl;
    }

//     if ( verbose_ ) {
//       std::cout << " " << Numbers::sEE(ism) << " (ism=" << ism << ")" << std::endl;
//       std::cout << std::endl;
//       UtilsClient::printBadChannels(meg01_[ism-1], h01_[ism-1], true);
//       UtilsClient::printBadChannels(meg01_[ism-1], h02_[ism-1], true);
//       UtilsClient::printBadChannels(meg01_[ism-1], h03_[ism-1], true);
//       UtilsClient::printBadChannels(meg01_[ism-1], h04_[ism-1], true);
//       UtilsClient::printBadChannels(meg01_[ism-1], h05_[ism-1], true);

//       UtilsClient::printBadChannels(meg02_[ism-1], h06_[ism-1], true);
//       UtilsClient::printBadChannels(meg02_[ism-1], h07_[ism-1], true);
//       UtilsClient::printBadChannels(meg02_[ism-1], h08_[ism-1], true);
//       UtilsClient::printBadChannels(meg02_[ism-1], h09_[ism-1], true);
//     }

    float num00;

    num00 = 0.;

    bool update0 = false;

    if ( h00_ ) {
      num00 = h00_->GetBinContent(ism);
      if ( num00 > 0 ) update0 = true;
    }

    std::map<uint32_t, float>::iterator mapItr;

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

	uint32_t id(EEDetId(jx, jy, zside).rawId());

	if((mapItr = gain_.find(id)) != gain_.end()){
	  num01 = mapItr->second;
	  if(num01 > 0) update1 = true;
	}

	if((mapItr = chid_.find(id)) != chid_.end()){
	  num02 = mapItr->second;
	  if(num02 > 0) update1 = true;
	}

	if((mapItr = gainswitch_.find(id)) != gainswitch_.end()){
	  num03 = mapItr->second;
	  if(num03 > 0) update1 = true;
	}

        if ( update0 || update1 ) {

          if ( Numbers::icEE(ism, jx, jy) == 1 ) {

            if ( verbose_ ) {
              std::cout << "Preparing dataset for " << Numbers::sEE(ism) << " (ism=" << ism << ")" << std::endl;
              std::cout << "(" << Numbers::ix0EE(i+1)+ix << "," << Numbers::iy0EE(i+1)+iy << ") " << num00 << " " << num01 << " " << num02 << " " << num03 << std::endl;
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

          int ic = Numbers::indexEE(ism, jx, jy);

          if ( ic == -1 ) continue;

          if ( econn ) {
            ecid = LogicID::getEcalLogicID("EE_crystal_number", Numbers::iSM(ism, EcalEndcap), ic);
            dataset1[ecid] = c1;
          }

          status = status && !val;

        }

      }
    }

    float num04, num05;

    for ( int ixt = 1; ixt <= 10; ixt++ ) {
      for ( int iyt = 1; iyt <= 10; iyt++ ) {

        int jxt = ixt + Numbers::ix0EE(ism) / 5;
        int jyt = iyt + Numbers::iy0EE(ism) / 5;

        if ( ism >= 1 && ism <= 9 ) jxt = 21 - jxt;

	if(jxt <= 0 || jxt > 20 || jyt <= 0 || jyt > 20 || !EcalScDetId::validDetId(jxt, jyt, zside)) continue;

        num04 = num05 = 0.;

        bool update1 = false;

        float numTot = -1.;
	float nCrystals(0.);

        if ( h_[ism-1] ) {
          numTot = 0.;
          for ( int ix = 1 + 5*(ixt-1); ix <= 5*ixt; ix++ ) {
            for ( int iy = 1 + 5*(iyt-1); iy <= 5*iyt; iy++ ) {
              if ( ! Numbers::validEE(ism, ix, iy) ) continue;
	      nCrystals += 1.;
              numTot += h_[ism-1]->GetBinContent(ix, iy);
            }
          }
        }

	EcalScDetId scid(jxt, jyt, zside);
	std::pair<int, int> dccsc(Numbers::getElectronicsMapping()->getDCCandSC(scid));
	uint32_t id(EcalElectronicsId(dccsc.first, dccsc.second, 1, 1).rawId());

	if((mapItr = ttid_.find(id)) != ttid_.end()){
	  num04 = mapItr->second * nCrystals;
	  if(num04 > 0) update1 = true;
	}

	if((mapItr = ttblocksize_.find(id)) != ttblocksize_.end()){
	  num05 = mapItr->second * nCrystals;
	  if(num05 > 0) update1 = true;
	}

        if ( update0 || update1 ) {

//           if ( Numbers::iSC(ism, EcalEndcap, jxt, jyt) == 1 ) {

//             if ( verbose_ ) {
//               std::cout << "Preparing dataset for " << Numbers::sEE(ism) << " (ism=" << ism << ")" << std::endl;
//               std::cout << "(" << 1+(Numbers::ix0EE(ism)+1+5*(ixt-1))/5 << "," << 1+(Numbers::iy0EE(ism)+1+5*(iyt-1))/5 << ") " << num00 << " " << num04 << " " << num05 << std::endl;
//               std::cout << std::endl;
//             }

//           }

          c2.setProcessedEvents(int(numTot));
          c2.setProblematicEvents(int(num04+num05));
          c2.setProblemsID(int(num04));
          c2.setProblemsSize(int(num05));
          c2.setProblemsLV1(int(-1.));
          c2.setProblemsBunchX(int(-1.));

          bool val;

          val = false;
          if ( numTot > 0 ) {
            float errorRate2 = num00 / ( numTot + num04 + num05 );
            if ( errorRate2 > threshCry_ )
              val = true;
            errorRate2 = ( num04 + num05 ) / ( numTot + num04 + num05 ) / 2.;
            if ( errorRate2 > threshCry_ )
              val = true;
          } else {
            if ( num00 > 0 )
              val = true;
            if ( ( num04 + num05 ) > 0 )
              val = true;
          }
          c2.setTaskStatus(val);

          if ( econn ) {
            ecid = LogicID::getEcalLogicID("EE_readout_tower", Numbers::iSM(ism, EcalEndcap), dccsc.second);
            dataset2[ecid] = c2;
          }

          status = status && !val;

        }

      }
    }

    float num06, num07;

    for(int ife(0); ife < 2; ife++){
      float numTotMem[5] = {-1., -1., -1., -1., -1.};
      for ( int ixtal = 1; ixtal <= 5; ixtal++ ) {
	if ( hmem_[ism-1] ) numTotMem[ixtal - 1] = hmem_[ism-1]->GetBinContent(ife * 5 + ixtal, 1);
      }

      for ( int ixtal = 1; ixtal <= 5; ixtal++ ) {
	for ( int istrip = 1; istrip <= 5; istrip++ ) {

	  num06 = num07 = 0.;

	  bool update1 = false;

	  float numTot;
	  if(istrip % 2 == 1) numTot = numTotMem[ixtal - 1];
	  else numTot = numTotMem[6 - ixtal];

	  uint32_t id(EcalElectronicsId(idcc, ife + 68, istrip, ixtal).rawId());

	  if((mapItr = memchid_.find(id)) != memchid_.end()){
	    num06 = mapItr->second;
	    if ( num06 > 0 ) update1 = true;
	  }

	  if((mapItr = memgain_.find(id)) != memgain_.end()){
	    num07 = mapItr->second;
	    if ( num07 > 0 ) update1 = true;
	  }

	  if ( update0 || update1 ) {

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

	    int ic = EEIntegrityClient::chNum[istrip - 1][ixtal - 1] + 25 * ife;

	    if ( econn ) {
	      ecid = LogicID::getEcalLogicID("EE_mem_channel", Numbers::iSM(ism, EcalEndcap), ic);
	      dataset3[ecid] = c3;
	    }

	    status = status && !val;

	  }

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
	  numTot += hmem_[ism-1]->GetBinContent(ix, 1);
        }
      }

      uint32_t id(EcalElectronicsId(idcc, ixt + 68, 1, 1).rawId());

      if((mapItr = memttid_.find(id)) != memttid_.end()){
	num08 = mapItr->second;
	if ( num08 > 0 ) update1 = true;
      }

      if((mapItr = memblocksize_.find(id)) != memblocksize_.end()){
	num09 = mapItr->second;
	if ( num09 > 0 ) update1 = true;
      }

      if ( update0 || update1 ) {

        if ( ixt == 1 ) {

          if ( verbose_ ) {
            std::cout << "Preparing dataset for " << Numbers::sEE(ism) << " (ism=" << ism << ")" << std::endl;
            std::cout << "(" << ixt <<  ") " << num08 << " " << num09 << std::endl;
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
          float errorRate2 = num00 / ( numTot + num08 + num09 );
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

        int itt = 68 + ixt;

        if ( econn ) {
          ecid = LogicID::getEcalLogicID("EE_mem_TT", Numbers::iSM(ism, EcalEndcap), itt);
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

void EEIntegrityClient::analyze(void) {

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( debug_ ) std::cout << "EEIntegrityClient: ievt/jevt = " << ievt_ << "/" << jevt_ << std::endl;
  }

  uint32_t bits01 = 0;
  bits01 |= 1 << EcalDQMStatusHelper::CH_ID_ERROR;
  bits01 |= 1 << EcalDQMStatusHelper::CH_GAIN_ZERO_ERROR;
  bits01 |= 1 << EcalDQMStatusHelper::CH_GAIN_SWITCH_ERROR;
  bits01 |= 1 << EcalDQMStatusHelper::TT_ID_ERROR;
  bits01 |= 1 << EcalDQMStatusHelper::TT_SIZE_ERROR;

  MonitorElement* me;

  gain_.clear();
  chid_.clear();
  gainswitch_.clear();
  ttid_.clear();
  ttblocksize_.clear();
  memchid_.clear();
  memgain_.clear();
  memttid_.clear();
  memblocksize_.clear();

  std::vector<MonitorElement *> vme;
  TString name;
  TPRegexp reDetId("IntegrityTask [a-zA-Z ]+ EE ([0-9+-]+) ([0-9]+)");
  TPRegexp reEleId("IntegrityTask [a-zA-Z ]+ FE ([0-9]+) ([0-9]+)( ([0-9]+) ([0-9]+)|)");
  TObjArray* matches(0);
  int ix, iy, zside, idcc, iccu, istrip, ixtal;

  vme = dqmStore_->getContents(prefixME_ + "/IntegrityErrors/Gain");
  for(std::vector<MonitorElement *>::iterator meItr(vme.begin()); meItr != vme.end(); ++meItr){
    if(!(*meItr)) continue;
    name = (*meItr)->getName().c_str();
    matches = reDetId.MatchS(name);
    if(matches->GetEntries() != 3) continue;
    ix = static_cast<TObjString*>(matches->At(1))->GetString().Atoi();
    iy = static_cast<TObjString*>(matches->At(2))->GetString().Atoi();
    if(ix < 0){
      zside = -1;
      ix = -ix;
    }else{
      zside = 1;
    }
    EEDetId id(ix, iy, zside);
    gain_[id.rawId()] = (*meItr)->getBinContent(1);
  }

  vme = dqmStore_->getContents(prefixME_ + "/IntegrityErrors/ChId");
  for(std::vector<MonitorElement *>::iterator meItr(vme.begin()); meItr != vme.end(); ++meItr){
    if(!(*meItr)) continue;
    name = (*meItr)->getName().c_str();
    matches = reDetId.MatchS(name);
    if(matches->GetEntries() != 3) continue;
    ix = static_cast<TObjString*>(matches->At(1))->GetString().Atoi();
    iy = static_cast<TObjString*>(matches->At(2))->GetString().Atoi();
    if(ix < 0){
      zside = -1;
      ix = -ix;
    }else{
      zside = 1;
    }
    EEDetId id(ix, iy, zside);
    chid_[id.rawId()] = (*meItr)->getBinContent(1);
  }

  vme = dqmStore_->getContents(prefixME_ + "/IntegrityErrors/GainSwitch");
  for(std::vector<MonitorElement *>::iterator meItr(vme.begin()); meItr != vme.end(); ++meItr){
    if(!(*meItr)) continue;
    name = (*meItr)->getName().c_str();
    matches = reDetId.MatchS(name);
    if(matches->GetEntries() != 3) continue;
    ix = static_cast<TObjString*>(matches->At(1))->GetString().Atoi();
    iy = static_cast<TObjString*>(matches->At(2))->GetString().Atoi();
    if(ix < 0){
      zside = -1;
      ix = -ix;
    }else{
      zside = 1;
    }
    EEDetId id(ix, iy, zside);
    gainswitch_[id.rawId()] = (*meItr)->getBinContent(1);
  }

  vme = dqmStore_->getContents(prefixME_ + "/IntegrityErrors/TowerId");
  for(std::vector<MonitorElement *>::iterator meItr(vme.begin()); meItr != vme.end(); ++meItr){
    if(!(*meItr)) continue;
    name = (*meItr)->getName().c_str();
    matches = reEleId.MatchS(name);
    if(matches->GetEntries() != 4) continue;
    idcc = static_cast<TObjString*>(matches->At(1))->GetString().Atoi();
    iccu = static_cast<TObjString*>(matches->At(2))->GetString().Atoi();
    if(idcc >= 10 && idcc <= 45) continue;
    EcalElectronicsId id(idcc, iccu, 1, 1);
    ttid_[id.rawId()] = (*meItr)->getBinContent(1);
  }

  vme = dqmStore_->getContents(prefixME_ + "/IntegrityErrors/BlockSize");
  for(std::vector<MonitorElement *>::iterator meItr(vme.begin()); meItr != vme.end(); ++meItr){
    if(!(*meItr)) continue;
    name = (*meItr)->getName().c_str();
    matches = reEleId.MatchS(name);
    if(matches->GetEntries() != 4) continue;
    idcc = static_cast<TObjString*>(matches->At(1))->GetString().Atoi();
    iccu = static_cast<TObjString*>(matches->At(2))->GetString().Atoi();
    if(idcc >= 10 && idcc <= 45) continue;
    EcalElectronicsId id(idcc, iccu, 1, 1);
    ttblocksize_[id.rawId()] = (*meItr)->getBinContent(1);
  }

  vme = dqmStore_->getContents(prefixME_ + "/IntegrityErrors/MEMBlockSize");
  for(std::vector<MonitorElement *>::iterator meItr(vme.begin()); meItr != vme.end(); ++meItr){
    if(!(*meItr)) continue;
    name = (*meItr)->getName().c_str();
    matches = reEleId.MatchS(name);
    if(matches->GetEntries() != 4) continue;
    idcc = static_cast<TObjString*>(matches->At(1))->GetString().Atoi();
    iccu = static_cast<TObjString*>(matches->At(2))->GetString().Atoi();
    if(idcc >= 10 && idcc <= 45) continue;
    EcalElectronicsId id(idcc, iccu, 1, 1);
    memblocksize_[id.rawId()] = (*meItr)->getBinContent(1);
  }

  vme = dqmStore_->getContents(prefixME_ + "/IntegrityErrors/MEMTowerId");
  for(std::vector<MonitorElement *>::iterator meItr(vme.begin()); meItr != vme.end(); ++meItr){
    if(!(*meItr)) continue;
    name = (*meItr)->getName().c_str();
    matches = reEleId.MatchS(name);
    if(matches->GetEntries() != 4) continue;
    idcc = static_cast<TObjString*>(matches->At(1))->GetString().Atoi();
    iccu = static_cast<TObjString*>(matches->At(2))->GetString().Atoi();
    if(idcc >= 10 && idcc <= 45) continue;
    EcalElectronicsId id(idcc, iccu, 1, 1);
    memttid_[id.rawId()] = (*meItr)->getBinContent(1);
  }

  vme = dqmStore_->getContents(prefixME_ + "/IntegrityErrors/MEMChId");
  for(std::vector<MonitorElement *>::iterator meItr(vme.begin()); meItr != vme.end(); ++meItr){
    if(!(*meItr)) continue;
    name = (*meItr)->getName().c_str();
    matches = reEleId.MatchS(name);
    if(matches->GetEntries() != 6) continue;
    idcc = static_cast<TObjString*>(matches->At(1))->GetString().Atoi();
    iccu = static_cast<TObjString*>(matches->At(2))->GetString().Atoi();
    istrip = static_cast<TObjString*>(matches->At(4))->GetString().Atoi();
    ixtal = static_cast<TObjString*>(matches->At(5))->GetString().Atoi();
    if(idcc >= 10 && idcc <= 45) continue;
    EcalElectronicsId id(idcc, iccu, istrip, ixtal);
    memchid_[id.rawId()] = (*meItr)->getBinContent(1);
  }

  vme = dqmStore_->getContents(prefixME_ + "/IntegrityErrors/MEMGain");
  for(std::vector<MonitorElement *>::iterator meItr(vme.begin()); meItr != vme.end(); ++meItr){
    if(!(*meItr)) continue;
    name = (*meItr)->getName().c_str();
    matches = reEleId.MatchS(name);
    if(matches->GetEntries() != 6) continue;
    idcc = static_cast<TObjString*>(matches->At(1))->GetString().Atoi();
    iccu = static_cast<TObjString*>(matches->At(2))->GetString().Atoi();
    istrip = static_cast<TObjString*>(matches->At(4))->GetString().Atoi();
    ixtal = static_cast<TObjString*>(matches->At(5))->GetString().Atoi();
    if(idcc >= 10 && idcc <= 45) continue;
    EcalElectronicsId id(idcc, iccu, istrip, ixtal);
    memgain_[id.rawId()] = (*meItr)->getBinContent(1);
  }


  me = dqmStore_->get( prefixME_ + "/IntegrityErrors/IntegrityTask DCC size error EE" );
  h00_ = UtilsClient::getHisto( me, cloneME_, h00_ );

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    me = dqmStore_->get( prefixME_ + "/Occupancy/Digi/OccupancyTask digi occupancy " + Numbers::sEE(ism) );
    h_[ism-1] = UtilsClient::getHisto( me, cloneME_, h_[ism-1] );

    me = dqmStore_->get( prefixME_ + "/Occupancy/MEMDigi/OccupancyTask MEM digi occupancy " + Numbers::sEE(ism) );
    hmem_[ism-1] = UtilsClient::getHisto( me, cloneME_, hmem_[ism-1] );

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

    std::map<uint32_t, float>::iterator mapItr;

    for ( int ix = 1; ix <= 50; ix++ ) {
      for ( int iy = 1; iy <= 50; iy++ ) {

        num01 = num02 = num03 = num04 = num05 = 0.;

        if ( meg01_[ism-1] ) meg01_[ism-1]->setBinContent( ix, iy, 6. );

	int jx = ix + Numbers::ix0EE(ism);
	int jy = iy + Numbers::iy0EE(ism);
	int zside = ism <= 9 ? -1 : 1;
	if(zside < 0) jx = 101 - jx;

	if(!EEDetId::validDetId(jx, jy, zside)) continue;

        bool update1 = false;
        bool update2 = false;

        float numTot = -1.;

        if ( h_[ism-1] ) numTot = h_[ism-1]->GetBinContent(ix, iy);

	uint32_t eeid(EEDetId(jx, jy, zside).rawId());
	EcalElectronicsId tmpEid(Numbers::getElectronicsMapping()->getElectronicsId(DetId(eeid)));
	uint32_t eid(EcalElectronicsId(tmpEid.dccId(), tmpEid.towerId(), 1, 1).rawId());

	if((mapItr = gain_.find(eeid)) != gain_.end()){
	  num01 = mapItr->second;
	  update1 = true;
	}

	if((mapItr = chid_.find(eeid)) != chid_.end()){
	  num02 = mapItr->second;
	  update1 = true;
	}

	if((mapItr = gainswitch_.find(eeid)) != gainswitch_.end()){
	  num03 = mapItr->second;
	  update1 = true;
	}

	if((mapItr = ttid_.find(eid)) != ttid_.end()){
	  num04 = mapItr->second;
	  update2 = true;
	}

	if((mapItr = ttblocksize_.find(eid)) != ttblocksize_.end()){
	  num05 = mapItr->second;
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
            float errorRate2 = ( num04 + num05 ) / ( numTot + num04 + num05 ) / 2.;
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

        if ( Masks::maskChannel(ism, ix, iy, bits01, EcalEndcap) ) UtilsClient::maskBinContent( meg01_[ism-1], ix, iy );

      }
    } // end of loop on crystals to fill summary plot

    // summaries for mem channels
    float num06, num07, num08, num09;

    for ( int ie = 1; ie <= 10; ie++ ) {
      for ( int ip = 1; ip <= 5; ip++ ) {

        num06 = num07 = num08 = num09 = 0.;

        // initialize summary histo for mem
        if ( meg02_[ism-1] ) meg02_[ism-1]->setBinContent( ie, ip, 6. );

        // non-existing mem
        if ( (ism >=  3 && ism <=  4) || (ism >=  7 && ism <=  9) ) continue;
        if ( (ism >= 12 && ism <= 13) || (ism >= 16 && ism <= 18) ) continue;

        if ( meg02_[ism-1] ) meg02_[ism-1]->setBinContent( ie, ip, 2. );

        bool update1 = false;
        bool update2 = false;

        float numTotmem = -1.;

        if ( hmem_[ism-1] ) numTotmem = hmem_[ism-1]->GetBinContent(ie, 1);

	uint32_t towerid(EcalElectronicsId(ism + 9, 69 + (ie - 1) / 5, 1, 1).rawId());
	istrip = (ie - 1) % 5 + 1;
	ixtal = (ie % 2) ? ip : 6 - ip;
	uint32_t memid(EcalElectronicsId(ism + 9, 69 + (ie - 1) / 5, istrip, ixtal).rawId());

	if((mapItr = memchid_.find(memid)) != memchid_.end()){
	  num06 = mapItr->second;
	  update1 = true;
	}

	if((mapItr = memgain_.find(memid)) != memgain_.end()){
	  num07 = mapItr->second;
	  update1 = true;
	}

	if((mapItr = memttid_.find(towerid)) != memttid_.end()){
	  num08 = mapItr->second;
	  update2 = true;
	}

	if((mapItr = memblocksize_.find(towerid)) != memblocksize_.end()){
	  num08 = mapItr->second;
	  update2 = true;
	}

        if ( update0 || update1 || update2 ) {

          float val;

          val = 1.;
          // number of events on a channel
          if ( numTotmem > 0 ) {
            float errorRate1 = ( num06 + num07 ) / ( numTotmem + num06 + num07 )/ 2.;
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

        if ( Masks::maskPn(ism, ie, bits01, EcalEndcap) ) UtilsClient::maskBinContent( meg02_[ism-1], ie, ip );

      }
    }  // end loop on mem channels

  } // end loop on supermodules

}

const int EEIntegrityClient::chNum[5][5] = {
  { 1,  2,  3,  4,  5},
  {10,  9,  8,  7,  6},
  {11, 12, 13, 14, 15},
  {20, 19, 18, 17, 16},
  {21, 22, 23, 24, 25}
};

