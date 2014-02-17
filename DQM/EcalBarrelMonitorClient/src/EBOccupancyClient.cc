/*
 * \file EBOccupancyClient.cc
 *
 * $Date: 2012/04/27 13:45:59 $
 * $Revision: 1.47 $
 * \author G. Della Ricca
 * \author F. Cossutti
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
#include "OnlineDB/EcalCondDB/interface/MonOccupancyDat.h"
#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"
#include "DQM/EcalCommon/interface/LogicID.h"
#endif

#include "DQM/EcalCommon/interface/UtilsClient.h"
#include "DQM/EcalCommon/interface/Numbers.h"

#include "DQM/EcalBarrelMonitorClient/interface/EBOccupancyClient.h"

EBOccupancyClient::EBOccupancyClient(const edm::ParameterSet& ps) {

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
    i01_[ism-1] = 0;
    i02_[ism-1] = 0;
  }

  for ( int i=0; i<3; i++) {
    h01_[i] = 0;
    h01ProjEta_[i] = 0;
    h01ProjPhi_[i] = 0;
  }

  for ( int i=0; i<2; i++) {
    h02_[i] = 0;
    h02ProjEta_[i] = 0;
    h02ProjPhi_[i] = 0;
  }

}

EBOccupancyClient::~EBOccupancyClient() {

}

void EBOccupancyClient::beginJob(void) {

  dqmStore_ = edm::Service<DQMStore>().operator->();

  if ( debug_ ) std::cout << "EBOccupancyClient: beginJob" << std::endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EBOccupancyClient::beginRun(void) {

  if ( debug_ ) std::cout << "EBOccupancyClient: beginRun" << std::endl;

  jevt_ = 0;

  this->setup();

}

void EBOccupancyClient::endJob(void) {

  if ( debug_ ) std::cout << "EBOccupancyClient: endJob, ievt = " << ievt_ << std::endl;

  this->cleanup();

}

void EBOccupancyClient::endRun(void) {

  if ( debug_ ) std::cout << "EBOccupancyClient: endRun, jevt = " << jevt_ << std::endl;

  this->cleanup();

}

void EBOccupancyClient::setup(void) {

  dqmStore_->setCurrentFolder( prefixME_ + "/EBOccupancyClient" );

}

void EBOccupancyClient::cleanup(void) {

  if ( ! enableCleanup_ ) return;

  if ( cloneME_ ) {

    for ( unsigned int i=0; i<superModules_.size(); i++ ) {
      int ism = superModules_[i];
      if ( i01_[ism-1] ) delete i01_[ism-1];
      if ( i02_[ism-1] ) delete i02_[ism-1];
    }

    for ( int i=0; i<3; ++i ) {
      if ( h01_[i] ) delete h01_[i];
      if ( h01ProjEta_[i] ) delete h01ProjEta_[i];
      if ( h01ProjPhi_[i] ) delete h01ProjPhi_[i];
    }

    for ( int i=0; i<2; ++i ) {
      if ( h02_[i] ) delete h02_[i];
      if ( h02ProjEta_[i] ) delete h02ProjEta_[i];
      if ( h02ProjPhi_[i] ) delete h02ProjPhi_[i];
    }

  }

  for ( int i=0; i<3; ++i ) {
    h01_[i] = 0;
    h01ProjEta_[i] = 0;
    h01ProjPhi_[i] = 0;
  }

  for ( int i=0; i<2; ++i ) {
    h02_[i] = 0;
    h02ProjEta_[i] = 0;
    h02ProjPhi_[i] = 0;
  }

}

#ifdef WITH_ECAL_COND_DB
bool EBOccupancyClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov, bool& status) {

  status = true;

  EcalLogicID ecid;

  MonOccupancyDat o;
  std::map<EcalLogicID, MonOccupancyDat> dataset;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( verbose_ ) {
      std::cout << " " << Numbers::sEB(ism) << " (ism=" << ism << ")" << std::endl;
      std::cout << std::endl;
    }

    const float n_min_tot = 1000.;
    const float n_min_bin = 10.;

    float num01, num02;
    float mean01, mean02;
    float rms01, rms02;

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

        num01  = num02  = -1.;
        mean01 = mean02 = -1.;
        rms01  = rms02  = -1.;

        bool update_channel = false;

        if ( i01_[ism-1] && i01_[ism-1]->GetEntries() >= n_min_tot ) {
          num01 = i01_[ism-1]->GetBinContent(ie, ip);
          if ( num01 >= n_min_bin ) update_channel = true;
        }

        if ( i02_[ism-1] && i02_[ism-1]->GetEntries() >= n_min_tot ) {
          num02 = i02_[ism-1]->GetBinEntries(i02_[ism-1]->GetBin(ie, ip));
          if ( num02 >= n_min_bin ) {
            mean02 = i02_[ism-1]->GetBinContent(ie, ip);
            rms02  = i02_[ism-1]->GetBinError(ie, ip);
          }
        }

        if ( update_channel ) {

          if ( Numbers::icEB(ism, ie, ip) == 1 ) {

            if ( verbose_ ) {
              std::cout << "Preparing dataset for " << Numbers::sEB(ism) << " (ism=" << ism << ")" << std::endl;
              std::cout << "Digi (" << ie << "," << ip << ") " << num01  << " " << mean01 << " " << rms01  << std::endl;
              std::cout << "RecHitThr (" << ie << "," << ip << ") " << num02  << " " << mean02 << " " << rms02  << std::endl;
              std::cout << std::endl;
            }

          }

          o.setEventsOverLowThreshold(int(num01));
          o.setEventsOverHighThreshold(int(num02));

          o.setAvgEnergy(mean02);

          int ic = Numbers::indexEB(ism, ie, ip);

          if ( econn ) {
            ecid = LogicID::getEcalLogicID("EB_crystal_number", Numbers::iSM(ism, EcalBarrel), ic);
            dataset[ecid] = o;
          }

        }

      }
    }

  }

  if ( econn ) {
    try {
      if ( verbose_ ) std::cout << "Inserting MonOccupancyDat ..." << std::endl;
      if ( dataset.size() != 0 ) econn->insertDataArraySet(&dataset, moniov);
      if ( verbose_ ) std::cout << "done." << std::endl;
    } catch (std::runtime_error &e) {
      std::cerr << e.what() << std::endl;
    }
  }

  return true;

}
#endif

void EBOccupancyClient::analyze(void) {

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( debug_ ) std::cout << "EBOccupancyClient: ievt/jevt = " << ievt_ << "/" << jevt_ << std::endl;
  }

  MonitorElement* me;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    me = dqmStore_->get( prefixME_ + "/EBOccupancyTask/EBOT digi occupancy " + Numbers::sEB(ism) );
    i01_[ism-1] = UtilsClient::getHisto<TH2F*>( me, cloneME_, i01_[ism-1] );

    me = dqmStore_->get( prefixME_ + "/EBOccupancyTask/EBOT rec hit energy " + Numbers::sEB(ism) );
    i02_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, i02_[ism-1] );

  }

  me = dqmStore_->get( prefixME_ + "/EBOccupancyTask/EBOT digi occupancy" );
  h01_[0] = UtilsClient::getHisto<TH2F*> ( me, cloneME_, h01_[0] );

  me = dqmStore_->get( prefixME_ + "/EBOccupancyTask/EBOT digi occupancy projection eta" );
  h01ProjEta_[0] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h01ProjEta_[0] );

  me = dqmStore_->get( prefixME_ + "/EBOccupancyTask/EBOT digi occupancy projection phi" );
  h01ProjPhi_[0] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h01ProjPhi_[0] );

  me = dqmStore_->get( prefixME_ + "/EBOccupancyTask/EBOT rec hit occupancy" );
  h01_[1] = UtilsClient::getHisto<TH2F*> ( me, cloneME_, h01_[1] );

  me = dqmStore_->get( prefixME_ + "/EBOccupancyTask/EBOT rec hit occupancy projection eta" );
  h01ProjEta_[1] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h01ProjEta_[1] );

  me = dqmStore_->get( prefixME_ + "/EBOccupancyTask/EBOT rec hit occupancy projection phi" );
  h01ProjPhi_[1] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h01ProjPhi_[1] );

  me = dqmStore_->get( prefixME_ + "/EBOccupancyTask/EBOT TP digi occupancy" );
  h01_[2] = UtilsClient::getHisto<TH2F*> ( me, cloneME_, h01_[2] );

  me = dqmStore_->get( prefixME_ + "/EBOccupancyTask/EBOT TP digi occupancy projection eta" );
  h01ProjEta_[2] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h01ProjEta_[2] );

  me = dqmStore_->get( prefixME_ + "/EBOccupancyTask/EBOT TP digi occupancy projection phi" );
  h01ProjPhi_[2] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h01ProjPhi_[2] );

  me = dqmStore_->get( prefixME_ + "/EBOccupancyTask/EBOT rec hit thr occupancy" );
  h02_[0] = UtilsClient::getHisto<TH2F*> ( me, cloneME_, h02_[0] );

  me = dqmStore_->get( prefixME_ + "/EBOccupancyTask/EBOT rec hit thr occupancy projection eta" );
  h02ProjEta_[0] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h02ProjEta_[0] );

  me = dqmStore_->get( prefixME_ + "/EBOccupancyTask/EBOT rec hit thr occupancy projection phi" );
  h02ProjPhi_[0] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h02ProjPhi_[0] );

  me = dqmStore_->get( prefixME_ + "/EBOccupancyTask/EBOT TP digi thr occupancy" );
  h02_[1] = UtilsClient::getHisto<TH2F*> ( me, cloneME_, h02_[1] );

  me = dqmStore_->get( prefixME_ + "/EBOccupancyTask/EBOT TP digi thr occupancy projection eta" );
  h02ProjEta_[1] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h02ProjEta_[1] );

  me = dqmStore_->get( prefixME_ + "/EBOccupancyTask/EBOT TP digi thr occupancy projection phi" );
  h02ProjPhi_[1] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h02ProjPhi_[1] );

}

