/*
 * \file EETimingClient.cc
 *
 * $Date: 2012/04/27 13:46:08 $
 * $Revision: 1.115 $
 * \author G. Della Ricca
 *
*/

#include <memory>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <math.h>

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#ifdef WITH_ECAL_COND_DB
#include "OnlineDB/EcalCondDB/interface/MonTimingCrystalDat.h"
#include "OnlineDB/EcalCondDB/interface/RunCrystalErrorsDat.h"
#include "OnlineDB/EcalCondDB/interface/RunTTErrorsDat.h"
#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"
#include "DQM/EcalCommon/interface/LogicID.h"
#endif

#include "DQM/EcalCommon/interface/Masks.h"

#include "DQM/EcalCommon/interface/UtilsClient.h"
#include "DQM/EcalCommon/interface/Numbers.h"

#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "DQM/EcalEndcapMonitorClient/interface/EETimingClient.h"

EETimingClient::EETimingClient(const edm::ParameterSet& ps) {

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

  nHitThreshold_ = ps.getUntrackedParameter<int>("timingNHitThreshold", 5);

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    h01_[ism-1] = 0;
    h02_[ism-1] = 0;

    meh01_[ism-1] = 0;
    meh02_[ism-1] = 0;

  }

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    meg01_[ism-1] = 0;

    mea01_[ism-1] = 0;

    mep01_[ism-1] = 0;

    mer01_[ism-1] = 0;

  }

  meTimeSummaryMapProjEta_[0] = 0;
  meTimeSummaryMapProjEta_[1] = 0;
  meTimeSummaryMapProjPhi_[0] = 0;
  meTimeSummaryMapProjPhi_[1] = 0;

  expectedMean_ = 0.;
  meanThreshold_ = 3.;
  rmsThreshold_ = 6.;
}

EETimingClient::~EETimingClient() {

}

void EETimingClient::beginJob(void) {

  dqmStore_ = edm::Service<DQMStore>().operator->();

  if ( debug_ ) std::cout << "EETimingClient: beginJob" <<  std::endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EETimingClient::beginRun(void) {

  if ( debug_ ) std::cout << "EETimingClient: beginRun" <<  std::endl;

  jevt_ = 0;

  this->setup();

}

void EETimingClient::endJob(void) {

  if ( debug_ ) std::cout << "EETimingClient: endJob, ievt = " << ievt_ <<  std::endl;

  this->cleanup();

}

void EETimingClient::endRun(void) {

  if ( debug_ ) std::cout << "EETimingClient: endRun, jevt = " << jevt_ <<  std::endl;

  this->cleanup();

}

void EETimingClient::setup(void) {

  std::string name;

  dqmStore_->setCurrentFolder( prefixME_ + "/EETimingClient" );

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( meg01_[ism-1] ) dqmStore_->removeElement( meg01_[ism-1]->getName() );
    name = "EETMT timing quality " + Numbers::sEE(ism);
    meg01_[ism-1] = dqmStore_->book2D(name, name, 50, Numbers::ix0EE(ism)+0., Numbers::ix0EE(ism)+50., 50, Numbers::iy0EE(ism)+0., Numbers::iy0EE(ism)+50.);
    meg01_[ism-1]->setAxisTitle("ix", 1);
    if ( ism >= 1 && ism <= 9 ) meg01_[ism-1]->setAxisTitle("101-ix", 1);
    meg01_[ism-1]->setAxisTitle("iy", 2);

    if ( mea01_[ism-1] ) dqmStore_->removeElement( mea01_[ism-1]->getName() );
    name = "EETMT timing " + Numbers::sEE(ism);
    mea01_[ism-1] = dqmStore_->book1D(name, name, 850, 0., 850.);
    mea01_[ism-1]->setAxisTitle("channel", 1);
    mea01_[ism-1]->setAxisTitle("time (ns)", 2);

    if ( mep01_[ism-1] ) dqmStore_->removeElement( mep01_[ism-1]->getName() );
    name = "EETMT timing mean " + Numbers::sEE(ism);
    mep01_[ism-1] = dqmStore_->book1D(name, name, 100, -25., 25.);
    mep01_[ism-1]->setAxisTitle("mean (ns)", 1);

    if ( mer01_[ism-1] ) dqmStore_->removeElement( mer01_[ism-1]->getName() );
    name = "EETMT timing rms " + Numbers::sEE(ism);
    mer01_[ism-1] = dqmStore_->book1D(name, name, 100, 0.0, 10.);
    mer01_[ism-1]->setAxisTitle("rms (ns)", 1);

  }

  name = "EETMT timing projection eta EE -";
  meTimeSummaryMapProjEta_[0] = dqmStore_->bookProfile(name, name, 20, -3.0, -1.479, -20., 20.,"");
  meTimeSummaryMapProjEta_[0]->setAxisTitle("eta", 1);
  meTimeSummaryMapProjEta_[0]->setAxisTitle("time (ns)", 2);

  name = "EETMT timing projection eta EE +";
  meTimeSummaryMapProjEta_[1] = dqmStore_->bookProfile(name, name, 20, 1.479, 3.0, -20., 20.,"");
  meTimeSummaryMapProjEta_[1]->setAxisTitle("eta", 1);
  meTimeSummaryMapProjEta_[1]->setAxisTitle("time (ns)", 2);

  name = "EETMT timing projection phi EE -";
  meTimeSummaryMapProjPhi_[0] = dqmStore_->bookProfile(name, name, 50, -M_PI, M_PI, -20., 20.,"");
  meTimeSummaryMapProjPhi_[0]->setAxisTitle("phi", 1);
  meTimeSummaryMapProjPhi_[0]->setAxisTitle("time (ns)", 2);

  name = "EETMT timing projection phi EE +";
  meTimeSummaryMapProjPhi_[1] = dqmStore_->bookProfile(name, name, 50, -M_PI, M_PI, -20., 20.,"");
  meTimeSummaryMapProjPhi_[1]->setAxisTitle("phi", 1);
  meTimeSummaryMapProjPhi_[1]->setAxisTitle("time (ns)", 2);

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( meg01_[ism-1] ) meg01_[ism-1]->Reset();

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

    if ( mea01_[ism-1] ) mea01_[ism-1]->Reset();
    if ( mep01_[ism-1] ) mep01_[ism-1]->Reset();
    if ( mer01_[ism-1] ) mer01_[ism-1]->Reset();

  }

}

void EETimingClient::cleanup(void) {

  if ( ! enableCleanup_ ) return;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( cloneME_ ) {
      if ( h01_[ism-1] ) delete h01_[ism-1];
      if ( h02_[ism-1] ) delete h02_[ism-1];
    }

    h01_[ism-1] = 0;
    h02_[ism-1] = 0;

    meh01_[ism-1] = 0;
    meh02_[ism-1] = 0;

  }

  dqmStore_->setCurrentFolder( prefixME_ + "/EETimingClient" );

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( meg01_[ism-1] ) dqmStore_->removeElement( meg01_[ism-1]->getName() );
    meg01_[ism-1] = 0;

    if ( mea01_[ism-1] ) dqmStore_->removeElement( mea01_[ism-1]->getName() );
    mea01_[ism-1] = 0;

    if ( mep01_[ism-1] ) dqmStore_->removeElement( mep01_[ism-1]->getName() );
    mep01_[ism-1] = 0;

    if ( mer01_[ism-1] ) dqmStore_->removeElement( mer01_[ism-1]->getName() );
    mer01_[ism-1] = 0;

  }

  for(int i=0; i<2; i++){
    if ( meTimeSummaryMapProjEta_[i] ) dqmStore_->removeElement( meTimeSummaryMapProjEta_[i]->getName() );
    meTimeSummaryMapProjEta_[i] = 0;

    if ( meTimeSummaryMapProjPhi_[i] ) dqmStore_->removeElement( meTimeSummaryMapProjPhi_[i]->getName() );
    meTimeSummaryMapProjPhi_[i] = 0;
  }

}

#ifdef WITH_ECAL_COND_DB
bool EETimingClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov, bool& status) {

  status = true;

  EcalLogicID ecid;

  MonTimingCrystalDat t;
  std::map<EcalLogicID, MonTimingCrystalDat> dataset;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( verbose_ ) {
      std::cout << " " << Numbers::sEE(ism) << " (ism=" << ism << ")" <<  std::endl;
      std::cout <<  std::endl;
      UtilsClient::printBadChannels(meg01_[ism-1], h01_[ism-1]);
    }

    float num01;
    float mean01;
    float rms01;

    for ( int ix = 1; ix <= 50; ix++ ) {
      for ( int iy = 1; iy <= 50; iy++ ) {

        int jx = ix + Numbers::ix0EE(ism);
        int jy = iy + Numbers::iy0EE(ism);

        if ( ism >= 1 && ism <= 9 ) jx = 101 - jx;

        if ( ! Numbers::validEE(ism, jx, jy) ) continue;

        bool update01;

        update01 = UtilsClient::getBinStatistics(h01_[ism-1], ix, iy, num01, mean01, rms01, nHitThreshold_);
        // Task timing map is shifted of +50 ns for graphical reasons. Shift back it.
        mean01 -= 50.;

        if ( update01 ) {

          if ( Numbers::icEE(ism, jx, jy) == 1 ) {

            if ( verbose_ ) {
              std::cout << "Preparing dataset for " << Numbers::sEE(ism) << " (ism=" << ism << ")" <<  std::endl;
              std::cout << "crystal (" << Numbers::ix0EE(i+1)+ix << "," << Numbers::iy0EE(i+1)+iy << ") " << num01  << " " << mean01 << " " << rms01  <<  std::endl;
              std::cout <<  std::endl;
            }

          }

          t.setTimingMean(mean01);
          t.setTimingRMS(rms01);

          if ( UtilsClient::getBinStatus(meg01_[ism-1], ix, iy) ) {
            t.setTaskStatus(true);
          } else {
            t.setTaskStatus(false);
          }

          status = status && UtilsClient::getBinQuality(meg01_[ism-1], ix, iy);

          int ic = Numbers::indexEE(ism, jx, jy);

          if ( ic == -1 ) continue;

          if ( econn ) {
            ecid = LogicID::getEcalLogicID("EE_crystal_number", Numbers::iSM(ism, EcalEndcap), ic);
            dataset[ecid] = t;
          }

        }

      }
    }

  }

  if ( econn ) {
    try {
      if ( verbose_ ) std::cout << "Inserting MonTimingCrystalDat ..." <<  std::endl;
      if ( dataset.size() != 0 ) econn->insertDataArraySet(&dataset, moniov);
      if ( verbose_ ) std::cout << "done." <<  std::endl;
    } catch (std::runtime_error &e) {
      std::cerr << e.what() <<  std::endl;
    }
  }

  return true;

}
#endif

void EETimingClient::analyze(void) {

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( debug_ ) std::cout << "EETimingClient: ievt/jevt = " << ievt_ << "/" << jevt_ <<  std::endl;
  }

  uint32_t bits01 = 0;
  bits01 |= 1 << EcalDQMStatusHelper::PHYSICS_BAD_CHANNEL_WARNING;

  MonitorElement* me;

  if( meTimeSummaryMapProjEta_[0] && meTimeSummaryMapProjEta_[1] && meTimeSummaryMapProjPhi_[0] && meTimeSummaryMapProjPhi_[1] ){

    for(int iz=0; iz<2; iz++){
      int zside = -1 + iz * 2;
      me = dqmStore_->get(prefixME_ + "/EETimingTask/EETMT timing map EE " + (zside==0 ? "-" : "+"));
      TProfile2D *hmap = 0;
      if( me ) hmap = (TProfile2D *)me->getRootObject();
      if( hmap ){

	int nx = hmap->GetNbinsX();
	int ny = hmap->GetNbinsY();

	for(int jx=1; jx<=nx; jx++){
	  for(int jy=1; jy<=ny; jy++){

	    int ix = (jx-1)*5 + 1;
	    int iy = (jy-1)*5 + 1;
	    if( !EEDetId::validDetId(ix, iy, zside) ) continue;

	    EEDetId id(ix, iy, zside);

	    float yval = hmap->GetBinContent(jx,jy) - 50.;

	    meTimeSummaryMapProjEta_[iz]->Fill(Numbers::eta(id), yval);
	    meTimeSummaryMapProjPhi_[iz]->Fill(Numbers::phi(id), yval);
	  }
	}
      }
    }

  }

  for(int i=0; i<2; i++){
    if( meTimeSummaryMapProjEta_[i] ) meTimeSummaryMapProjEta_[i]->Reset();
    if( meTimeSummaryMapProjPhi_[i] ) meTimeSummaryMapProjPhi_[i]->Reset();
  }

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    me = dqmStore_->get( prefixME_ + "/EETimingTask/EETMT timing " + Numbers::sEE(ism) );
    h01_[ism-1] = UtilsClient::getHisto( me, cloneME_, h01_[ism-1] );
    meh01_[ism-1] = me;

    me = dqmStore_->get( prefixME_ + "/EETimingTask/EETMT timing vs amplitude " + Numbers::sEE(ism) );
    h02_[ism-1] = UtilsClient::getHisto( me, cloneME_, h02_[ism-1] );
    meh02_[ism-1] = me;

    if ( meg01_[ism-1] ) meg01_[ism-1]->Reset();
    if ( mea01_[ism-1] ) mea01_[ism-1]->Reset();
    if ( mep01_[ism-1] ) mep01_[ism-1]->Reset();
    if ( mer01_[ism-1] ) mer01_[ism-1]->Reset();

    int iz = ism / 10;

    for ( int ix = 1; ix <= 50; ix++ ) {
      for ( int iy = 1; iy <= 50; iy++ ) {

        if ( meg01_[ism-1] ) meg01_[ism-1]->setBinContent(ix, iy, 6.);

        int jx = ix + Numbers::ix0EE(ism);
        int jy = iy + Numbers::iy0EE(ism);

        if ( ism >= 1 && ism <= 9 ) jx = 101 - jx;

        if ( Numbers::validEE(ism, jx, jy) ) {
          if ( meg01_[ism-1] ) meg01_[ism-1]->setBinContent( ix, iy, 2. );
        }

        bool update01;

        float num01;
        float mean01;
        float rms01;

        update01 = UtilsClient::getBinStatistics(h01_[ism-1], ix, iy, num01, mean01, rms01, nHitThreshold_);
        // Task timing map is shifted of +50 ns for graphical reasons. Shift back it.
        mean01 -= 50.;

        if ( update01 ) {

	  EEDetId id(jx, jy, -1 + iz * 2);

          float val(1.);

          if ( std::abs(mean01 - expectedMean_) > meanThreshold_ || rms01 > rmsThreshold_ ) val = 0.;

          if ( meg01_[ism-1] ) meg01_[ism-1]->setBinContent(ix, iy, val);

          int ic = Numbers::icEE(ism, jx, jy);

          if ( ic != -1 ) {
            if ( mea01_[ism-1] ) {
              mea01_[ism-1]->setBinContent(ic, mean01);
              mea01_[ism-1]->setBinError(ic, rms01);
            }

            if ( mep01_[ism-1] ) mep01_[ism-1]->Fill(mean01);
            if ( mer01_[ism-1] ) mer01_[ism-1]->Fill(rms01);
          }

	  if( meTimeSummaryMapProjEta_[iz] ) meTimeSummaryMapProjEta_[iz]->Fill(Numbers::eta(id), mean01);
	  if( meTimeSummaryMapProjPhi_[iz] ) meTimeSummaryMapProjPhi_[iz]->Fill(Numbers::phi(id), mean01);

        }

        if ( Masks::maskChannel(ism, ix, iy, bits01, EcalEndcap) ) UtilsClient::maskBinContent( meg01_[ism-1], ix, iy );

      }
    }

  }

}

