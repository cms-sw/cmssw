/*
 * \file EBTimingClient.cc
 *
 * $Date: 2012/04/27 13:45:59 $
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

#include "DQM/EcalBarrelMonitorClient/interface/EBTimingClient.h"

EBTimingClient::EBTimingClient(const edm::ParameterSet& ps) {

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

  // vector of selected Super Modules (Defaults to all 36).
  superModules_.reserve(36);
  for ( unsigned int i = 1; i <= 36; i++ ) superModules_.push_back(i);
  superModules_ = ps.getUntrackedParameter<std::vector<int> >("superModules", superModules_);

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

  meTimeSummaryMapProjEta_ = 0;
  meTimeSummaryMapProjPhi_ = 0;

  expectedMean_ = 0.0;
  discrepancyMean_ = 2.0;
  RMSThreshold_ = 6.0;

  nHitThreshold_ = ps.getUntrackedParameter<int>("timingNHitThrehold", 5);

  ievt_ = 0;
  jevt_ = 0;
  dqmStore_ = 0;
}

EBTimingClient::~EBTimingClient() {

}

void EBTimingClient::beginJob(void) {

  dqmStore_ = edm::Service<DQMStore>().operator->();

  if ( debug_ ) std::cout << "EBTimingClient: beginJob" << std::endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EBTimingClient::beginRun(void) {

  if ( debug_ ) std::cout << "EBTimingClient: beginRun" << std::endl;

  jevt_ = 0;

  this->setup();

}

void EBTimingClient::endJob(void) {

  if ( debug_ ) std::cout << "EBTimingClient: endJob, ievt = " << ievt_ << std::endl;

  this->cleanup();

}

void EBTimingClient::endRun(void) {

  if ( debug_ ) std::cout << "EBTimingClient: endRun, jevt = " << jevt_ << std::endl;

  this->cleanup();

}

void EBTimingClient::setup(void) {

  std::string name;

  dqmStore_->setCurrentFolder( prefixME_ + "/EBTimingClient" );

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( meg01_[ism-1] ) dqmStore_->removeElement( meg01_[ism-1]->getName() );
    name = "EBTMT timing quality " + Numbers::sEB(ism);
    meg01_[ism-1] = dqmStore_->book2D(name, name, 85, 0., 85., 20, 0., 20.);
    meg01_[ism-1]->setAxisTitle("ieta", 1);
    meg01_[ism-1]->setAxisTitle("iphi", 2);

    if ( mea01_[ism-1] ) dqmStore_->removeElement( mea01_[ism-1]->getName() );
    name = "EBTMT timing " + Numbers::sEB(ism);
    mea01_[ism-1] = dqmStore_->book1D(name, name, 1700, 0., 1700.);
    mea01_[ism-1]->setAxisTitle("channel", 1);
    mea01_[ism-1]->setAxisTitle("time (ns)", 2);

    if ( mep01_[ism-1] ) dqmStore_->removeElement( mep01_[ism-1]->getName() );
    name = "EBTMT timing mean " + Numbers::sEB(ism);
    mep01_[ism-1] = dqmStore_->book1D(name, name, 100, -25.0, 25.0);
    mep01_[ism-1]->setAxisTitle("mean (ns)", 1);

    if ( mer01_[ism-1] ) dqmStore_->removeElement( mer01_[ism-1]->getName() );
    name = "EBTMT timing rms " + Numbers::sEB(ism);
    mer01_[ism-1] = dqmStore_->book1D(name, name, 100, 0.0, 10.0);
    mer01_[ism-1]->setAxisTitle("rms (ns)", 1);

  }

  name = "EBTMT timing projection eta";
  meTimeSummaryMapProjEta_ = dqmStore_->bookProfile(name, name, 34, -85., 85., -20., 20., "");
  meTimeSummaryMapProjEta_->setAxisTitle("jeta", 1);
  meTimeSummaryMapProjEta_->setAxisTitle("time (ns)", 2);

  name = "EBTMT timing projection phi";
  meTimeSummaryMapProjPhi_ = dqmStore_->bookProfile(name, name, 72, 0., 360., -20., 20., "");
  meTimeSummaryMapProjPhi_->setAxisTitle("jphi", 1);
  meTimeSummaryMapProjPhi_->setAxisTitle("time (ns)", 2);

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( meg01_[ism-1] ) meg01_[ism-1]->Reset();

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

        if ( meg01_[ism-1] ) meg01_[ism-1]->setBinContent( ie, ip, 2. );

      }
    }

    if ( mea01_[ism-1] ) mea01_[ism-1]->Reset();
    if ( mep01_[ism-1] ) mep01_[ism-1]->Reset();
    if ( mer01_[ism-1] ) mer01_[ism-1]->Reset();

  }



}

void EBTimingClient::cleanup(void) {

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

  dqmStore_->setCurrentFolder( prefixME_ + "/EBTimingClient" );

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

  if ( meTimeSummaryMapProjEta_ ) dqmStore_->removeElement( meTimeSummaryMapProjEta_->getName() );
  if ( meTimeSummaryMapProjPhi_ ) dqmStore_->removeElement( meTimeSummaryMapProjPhi_->getName() );

}

#ifdef WITH_ECAL_COND_DB
bool EBTimingClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov, bool& status) {

  status = true;

  EcalLogicID ecid;

  MonTimingCrystalDat t;
  std::map<EcalLogicID, MonTimingCrystalDat> dataset;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( verbose_ ) {
      std::cout << " " << Numbers::sEB(ism) << " (ism=" << ism << ")" << std::endl;
      std::cout << std::endl;
      UtilsClient::printBadChannels(meg01_[ism-1], h01_[ism-1]);
    }

    float num01;
    float mean01;
    float rms01;

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

        bool update01;

        update01 = UtilsClient::getBinStatistics(h01_[ism-1], ie, ip, num01, mean01, rms01, nHitThreshold_);
        // Task timing map is shifted of +50 ns for graphical reasons. Shift back it.
        mean01 -= 50.;

        if ( update01 ) {

          if ( Numbers::icEB(ism, ie, ip) == 1 ) {

            if ( verbose_ ) {
              std::cout << "Preparing dataset for " << Numbers::sEB(ism) << " (ism=" << ism << ")" << std::endl;
              std::cout << "crystal (" << ie << "," << ip << ") " << num01  << " " << mean01 << " " << rms01  << std::endl;
              std::cout << std::endl;
            }

          }

          t.setTimingMean(mean01);
          t.setTimingRMS(rms01);

          if ( UtilsClient::getBinStatus(meg01_[ism-1], ie, ip) ) {
            t.setTaskStatus(true);
          } else {
            t.setTaskStatus(false);
          }

          status = status && UtilsClient::getBinQuality(meg01_[ism-1], ie, ip);

          int ic = Numbers::indexEB(ism, ie, ip);

          if ( econn ) {
            ecid = LogicID::getEcalLogicID("EB_crystal_number", Numbers::iSM(ism, EcalBarrel), ic);
            dataset[ecid] = t;
          }

        }

      }
    }

  }

  if ( econn ) {
    try {
      if ( verbose_ ) std::cout << "Inserting MonTimingCrystalDat ..." << std::endl;
      if ( dataset.size() != 0 ) econn->insertDataArraySet(&dataset, moniov);
      if ( verbose_ ) std::cout << "done." << std::endl;
    } catch (std::runtime_error &e) {
      std::cerr << e.what() << std::endl;
    }
  }

  return true;

}
#endif

void EBTimingClient::analyze(void) {

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( debug_ ) std::cout << "EBTimingClient: ievt/jevt = " << ievt_ << "/" << jevt_ << std::endl;
  }

  uint32_t bits01 = 0;
  bits01 |= 1 << EcalDQMStatusHelper::PHYSICS_BAD_CHANNEL_WARNING;

  MonitorElement* me;

  if( meTimeSummaryMapProjEta_ ) meTimeSummaryMapProjEta_->Reset();
  if( meTimeSummaryMapProjPhi_ ) meTimeSummaryMapProjPhi_->Reset();

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    me = dqmStore_->get( prefixME_ + "/EBTimingTask/EBTMT timing " + Numbers::sEB(ism) );
    h01_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h01_[ism-1] );
    meh01_[ism-1] = me;

    me = dqmStore_->get( prefixME_ + "/EBTimingTask/EBTMT timing vs amplitude " + Numbers::sEB(ism) );
    h02_[ism-1] = UtilsClient::getHisto<TH2F*>( me, cloneME_, h02_[ism-1] );
    meh02_[ism-1] = me;

    if ( meg01_[ism-1] ) meg01_[ism-1]->Reset();
    if ( mea01_[ism-1] ) mea01_[ism-1]->Reset();
    if ( mep01_[ism-1] ) mep01_[ism-1]->Reset();
    if ( mer01_[ism-1] ) mer01_[ism-1]->Reset();

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

        if ( meg01_[ism-1] ) meg01_[ism-1]->setBinContent(ie, ip, 2.);

        bool update01;

        float num01;
        float mean01;
        float rms01;

        update01 = UtilsClient::getBinStatistics(h01_[ism-1], ie, ip, num01, mean01, rms01, nHitThreshold_);
        // Task timing map is shifted of +50 ns for graphical reasons. Shift back it.
        mean01 -= 50.;

        if ( update01 ) {

          float val;

          val = 1.;
          if ( std::abs(mean01 - expectedMean_) > discrepancyMean_ )
            val = 0.;
          if ( rms01 > RMSThreshold_ )
            val = 0.;
          if ( meg01_[ism-1] ) meg01_[ism-1]->setBinContent(ie, ip, val);

          int ic = Numbers::icEB(ism, ie, ip);

          if ( mea01_[ism-1] ) {
            mea01_[ism-1]->setBinContent(ic, mean01);
            mea01_[ism-1]->setBinError(ic, rms01);
          }
          if ( mep01_[ism-1] ) mep01_[ism-1]->Fill(mean01);
          if ( mer01_[ism-1] ) mer01_[ism-1]->Fill(rms01);

	  float eta, phi;

	  if ( ism <= 18 ) {
	    eta = -ie+0.5;
	    phi = ip+20*(ism-1)-0.5;
	  } else {
	    eta = ie-0.5;
	    phi = (20-ip)+20*(ism-19)+0.5;
	  }

	  if( meTimeSummaryMapProjEta_ ) meTimeSummaryMapProjEta_->Fill(eta, mean01);
	  if( meTimeSummaryMapProjPhi_ ) meTimeSummaryMapProjPhi_->Fill(phi, mean01);

        }

        if ( Masks::maskChannel(ism, ie, ip, bits01, EcalBarrel) ) UtilsClient::maskBinContent( meg01_[ism-1], ie, ip );



      }
    }

  }

}

