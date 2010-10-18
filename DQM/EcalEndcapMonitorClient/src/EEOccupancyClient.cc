/*
 * \file EEOccupancyClient.cc
 *
 * $Date: 2010/08/04 08:20:13 $
 * $Revision: 1.39 $
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

#ifdef WITH_ECAL_COND_DB
#include "OnlineDB/EcalCondDB/interface/MonOccupancyDat.h"
#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"
#include "DQM/EcalCommon/interface/LogicID.h"
#endif

#include "DQM/EcalCommon/interface/UtilsClient.h"
#include "DQM/EcalCommon/interface/Numbers.h"

#include "DQM/EcalEndcapMonitorClient/interface/EEOccupancyClient.h"

EEOccupancyClient::EEOccupancyClient(const edm::ParameterSet& ps) {

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

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {
    int ism = superModules_[i];
    i01_[ism-1] = 0;
    i02_[ism-1] = 0;
  }

  for ( int i=0; i<3; i++) {
    h01_[0][i] = 0;
    h01ProjEta_[0][i] = 0;
    h01ProjPhi_[0][i] = 0;
    h01_[1][i] = 0;
    h01ProjEta_[1][i] = 0;
    h01ProjPhi_[1][i] = 0;
  }

  for ( int i=0; i<2; i++) {
    h02_[0][i] = 0;
    h02ProjEta_[0][i] = 0;
    h02ProjPhi_[0][i] = 0;
    h02_[1][i] = 0;
    h02ProjEta_[1][i] = 0;
    h02ProjPhi_[1][i] = 0;
  }

}

EEOccupancyClient::~EEOccupancyClient() {

}

void EEOccupancyClient::beginJob(void) {

  dqmStore_ = edm::Service<DQMStore>().operator->();

  if ( debug_ ) std::cout << "EEOccupancyClient: beginJob" << std::endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EEOccupancyClient::beginRun(void) {

  if ( debug_ ) std::cout << "EEOccupancyClient: beginRun" << std::endl;

  jevt_ = 0;

  this->setup();

}

void EEOccupancyClient::endJob(void) {

  if ( debug_ ) std::cout << "EEOccupancyClient: endJob, ievt = " << ievt_ << std::endl;

  this->cleanup();

}

void EEOccupancyClient::endRun(void) {

  if ( debug_ ) std::cout << "EEOccupancyClient: endRun, jevt = " << jevt_ << std::endl;

  this->cleanup();

}

void EEOccupancyClient::setup(void) {

  dqmStore_->setCurrentFolder( prefixME_ + "/EEOccupancyClient" );

}

void EEOccupancyClient::cleanup(void) {

  if ( ! enableCleanup_ ) return;

  if ( cloneME_ ) {

    for ( unsigned int i=0; i<superModules_.size(); i++ ) {
      int ism = superModules_[i];
      if ( i01_[ism-1] ) delete i01_[ism-1];
      if ( i02_[ism-1] ) delete i02_[ism-1];
    }

    for ( int i=0; i<3; ++i ) {
      if ( h01_[0][i] ) delete h01_[0][i];
      if ( h01ProjEta_[0][i] ) delete h01ProjEta_[0][i];
      if ( h01ProjPhi_[0][i] ) delete h01ProjPhi_[0][i];
      if ( h01_[1][i] ) delete h01_[1][i];
      if ( h01ProjEta_[1][i] ) delete h01ProjEta_[1][i];
      if ( h01ProjPhi_[1][i] ) delete h01ProjPhi_[1][i];
    }

    for ( int i=0; i<2; ++i ) {
      if ( h02_[0][i] ) delete h02_[0][i];
      if ( h02ProjEta_[0][i] ) delete h02ProjEta_[0][i];
      if ( h02ProjPhi_[0][i] ) delete h02ProjPhi_[0][i];
      if ( h01_[1][i] ) delete h01_[1][i];
      if ( h01ProjEta_[1][i] ) delete h01ProjEta_[1][i];
      if ( h01ProjPhi_[1][i] ) delete h01ProjPhi_[1][i];
    }

  }

  for ( int i=0; i<3; i++) {
    h01_[0][i] = 0;
    h01ProjEta_[0][i] = 0;
    h01ProjPhi_[0][i] = 0;
    h01_[1][i] = 0;
    h01ProjEta_[1][i] = 0;
    h01ProjPhi_[1][i] = 0;
  }

  for ( int i=0; i<2; i++) {
    h02_[0][i] = 0;
    h02ProjEta_[0][i] = 0;
    h02ProjPhi_[0][i] = 0;
    h02_[1][i] = 0;
    h02ProjEta_[1][i] = 0;
    h02ProjPhi_[1][i] = 0;
  }

}

#ifdef WITH_ECAL_COND_DB
bool EEOccupancyClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov, bool& status) {

  status = true;

  EcalLogicID ecid;

  MonOccupancyDat o;
  map<EcalLogicID, MonOccupancyDat> dataset;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( verbose_ ) {
      std::cout << " " << Numbers::sEE(ism) << " (ism=" << ism << ")" << std::endl;
      std::cout << std::endl;
    }

    const float n_min_tot = 1000.;
    const float n_min_bin = 10.;

    float num01, num02;
    float mean01, mean02;
    float rms01, rms02;

    for ( int ix = 1; ix <= 50; ix++ ) {
      for ( int iy = 1; iy <= 50; iy++ ) {

        int jx = ix + Numbers::ix0EE(ism);
        int jy = iy + Numbers::iy0EE(ism);

        if ( ism >= 1 && ism <= 9 ) jx = 101 - jx;

        if ( ! Numbers::validEE(ism, jx, jy) ) continue;

        num01  = num02  = -1.;
        mean01 = mean02 = -1.;
        rms01  = rms02  = -1.;

        bool update_channel = false;

        if ( i01_[ism-1] && i01_[ism-1]->GetEntries() >= n_min_tot ) {
          num01 = i01_[ism-1]->GetBinContent(i01_[ism-1]->GetBin(ix, iy));
          if ( num01 >= n_min_bin ) update_channel = true;
        }

        if ( i02_[ism-1] && i02_[ism-1]->GetEntries() >= n_min_tot ) {
          num02 = i02_[ism-1]->GetBinEntries(i02_[ism-1]->GetBin(ix, iy));
          if ( num02 >= n_min_bin ) {
            mean02 = i02_[ism-1]->GetBinContent(ix, iy);
            rms02  = i02_[ism-1]->GetBinError(ix, iy);
          }
        }

        if ( update_channel ) {

          if ( Numbers::icEE(ism, jx, jy) == 1 ) {

            if ( verbose_ ) {
              std::cout << "Preparing dataset for " << Numbers::sEE(ism) << " (ism=" << ism << ")" << std::endl;
              std::cout << "Digi (" << Numbers::ix0EE(i+1)+ix << "," << Numbers::iy0EE(i+1)+iy << ") " << num01  << " " << mean01 << " " << rms01  << std::endl;
              std::cout << "RecHitThr (" << Numbers::ix0EE(i+1)+ix << "," << Numbers::iy0EE(i+1)+iy << ") " << num02  << " " << mean02 << " " << rms02  << std::endl;
              std::cout << std::endl;
            }

          }

          o.setEventsOverLowThreshold(int(num01));
          o.setEventsOverHighThreshold(int(num02));

          o.setAvgEnergy(mean02);

          int ic = Numbers::indexEE(ism, jx, jy);

          if ( ic == -1 ) continue;

          if ( econn ) {
            ecid = LogicID::getEcalLogicID("EE_crystal_number", Numbers::iSM(ism, EcalEndcap), ic);
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
    } catch (runtime_error &e) {
      cerr << e.what() << std::endl;
    }
  }

  return true;

}
#endif

void EEOccupancyClient::analyze(void) {

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( debug_ ) std::cout << "EEOccupancyClient: ievt/jevt = " << ievt_ << "/" << jevt_ << std::endl;
  }

  char histo[200];

  MonitorElement* me;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    sprintf(histo, (prefixME_ + "/EEOccupancyTask/EEOT digi occupancy %s").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    i01_[ism-1] = UtilsClient::getHisto<TH2F*>( me, cloneME_, i01_[ism-1] );

    sprintf(histo, (prefixME_ + "/EEOccupancyTask/EEOT rec hit energy %s").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    i02_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, i02_[ism-1] );

  }

  sprintf(histo, (prefixME_ + "/EEOccupancyTask/EEOT digi occupancy EE -").c_str());
  me = dqmStore_->get(histo);
  h01_[0][0] = UtilsClient::getHisto<TH2F*> ( me, cloneME_, h01_[0][0] );

  sprintf(histo, (prefixME_ + "/EEOccupancyTask/EEOT digi occupancy EE - projection eta").c_str());
  me = dqmStore_->get(histo);
  h01ProjEta_[0][0] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h01ProjEta_[0][0] );

  sprintf(histo, (prefixME_ + "/EEOccupancyTask/EEOT digi occupancy EE - projection phi").c_str());
  me = dqmStore_->get(histo);
  h01ProjPhi_[0][0] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h01ProjPhi_[0][0] );

  sprintf(histo, (prefixME_ + "/EEOccupancyTask/EEOT digi occupancy EE +").c_str());
  me = dqmStore_->get(histo);
  h01_[1][0] = UtilsClient::getHisto<TH2F*> ( me, cloneME_, h01_[1][0] );

  sprintf(histo, (prefixME_ + "/EEOccupancyTask/EEOT digi occupancy EE + projection eta").c_str());
  me = dqmStore_->get(histo);
  h01ProjEta_[1][0] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h01ProjEta_[1][0] );

  sprintf(histo, (prefixME_ + "/EEOccupancyTask/EEOT digi occupancy EE + projection phi").c_str());
  me = dqmStore_->get(histo);
  h01ProjPhi_[1][0] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h01ProjPhi_[1][0] );

  sprintf(histo, (prefixME_ + "/EEOccupancyTask/EEOT rec hit occupancy EE -").c_str());
  me = dqmStore_->get(histo);
  h01_[0][1] = UtilsClient::getHisto<TH2F*> ( me, cloneME_, h01_[0][1] );

  sprintf(histo, (prefixME_ + "/EEOccupancyTask/EEOT rec hit occupancy EE - projection eta").c_str());
  me = dqmStore_->get(histo);
  h01ProjEta_[0][1] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h01ProjEta_[0][1] );

  sprintf(histo, (prefixME_ + "/EEOccupancyTask/EEOT rec hit occupancy EE - projection phi").c_str());
  me = dqmStore_->get(histo);
  h01ProjPhi_[0][1] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h01ProjPhi_[0][1] );

  sprintf(histo, (prefixME_ + "/EEOccupancyTask/EEOT rec hit occupancy EE +").c_str());
  me = dqmStore_->get(histo);
  h01_[1][1] = UtilsClient::getHisto<TH2F*> ( me, cloneME_, h01_[1][1] );

  sprintf(histo, (prefixME_ + "/EEOccupancyTask/EEOT rec hit occupancy EE + projection eta").c_str());
  me = dqmStore_->get(histo);
  h01ProjEta_[1][1] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h01ProjEta_[1][1] );

  sprintf(histo, (prefixME_ + "/EEOccupancyTask/EEOT rec hit occupancy EE + projection phi").c_str());
  me = dqmStore_->get(histo);
  h01ProjPhi_[1][1] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h01ProjPhi_[1][1] );

  sprintf(histo, (prefixME_ + "/EEOccupancyTask/EEOT TP digi occupancy EE -").c_str());
  me = dqmStore_->get(histo);
  h01_[0][2] = UtilsClient::getHisto<TH2F*> ( me, cloneME_, h01_[0][2] );

  sprintf(histo, (prefixME_ + "/EEOccupancyTask/EEOT TP digi occupancy EE - projection eta").c_str());
  me = dqmStore_->get(histo);
  h01ProjEta_[0][2] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h01ProjEta_[0][2] );

  sprintf(histo, (prefixME_ + "/EEOccupancyTask/EEOT TP digi occupancy EE - projection phi").c_str());
  me = dqmStore_->get(histo);
  h01ProjPhi_[0][2] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h01ProjPhi_[0][2] );

  sprintf(histo, (prefixME_ + "/EEOccupancyTask/EEOT TP digi occupancy EE +").c_str());
  me = dqmStore_->get(histo);
  h01_[1][2] = UtilsClient::getHisto<TH2F*> ( me, cloneME_, h01_[1][2] );

  sprintf(histo, (prefixME_ + "/EEOccupancyTask/EEOT TP digi occupancy EE + projection eta").c_str());
  me = dqmStore_->get(histo);
  h01ProjEta_[1][2] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h01ProjEta_[1][2] );

  sprintf(histo, (prefixME_ + "/EEOccupancyTask/EEOT TP digi occupancy EE + projection phi").c_str());
  me = dqmStore_->get(histo);
  h01ProjPhi_[1][2] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h01ProjPhi_[1][2] );

  sprintf(histo, (prefixME_ + "/EEOccupancyTask/EEOT rec hit thr occupancy EE -").c_str());
  me = dqmStore_->get(histo);
  h02_[0][0] = UtilsClient::getHisto<TH2F*> ( me, cloneME_, h02_[0][0] );

  sprintf(histo, (prefixME_ + "/EEOccupancyTask/EEOT rec hit thr occupancy EE - projection eta").c_str());
  me = dqmStore_->get(histo);
  h02ProjEta_[0][0] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h02ProjEta_[0][0] );

  sprintf(histo, (prefixME_ + "/EEOccupancyTask/EEOT rec hit thr occupancy EE - projection phi").c_str());
  me = dqmStore_->get(histo);
  h02ProjPhi_[0][0] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h02ProjPhi_[0][0] );

  sprintf(histo, (prefixME_ + "/EEOccupancyTask/EEOT rec hit thr occupancy EE +").c_str());
  me = dqmStore_->get(histo);
  h02_[1][0] = UtilsClient::getHisto<TH2F*> ( me, cloneME_, h02_[1][0] );

  sprintf(histo, (prefixME_ + "/EEOccupancyTask/EEOT rec hit thr occupancy EE + projection eta").c_str());
  me = dqmStore_->get(histo);
  h02ProjEta_[1][0] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h02ProjEta_[1][0] );

  sprintf(histo, (prefixME_ + "/EEOccupancyTask/EEOT rec hit thr occupancy EE + projection phi").c_str());
  me = dqmStore_->get(histo);
  h02ProjPhi_[1][0] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h02ProjPhi_[1][0] );

  sprintf(histo, (prefixME_ + "/EEOccupancyTask/EEOT TP digi thr occupancy EE -").c_str());
  me = dqmStore_->get(histo);
  h02_[0][1] = UtilsClient::getHisto<TH2F*> ( me, cloneME_, h02_[0][1] );

  sprintf(histo, (prefixME_ + "/EEOccupancyTask/EEOT TP digi thr occupancy EE - projection eta").c_str());
  me = dqmStore_->get(histo);
  h02ProjEta_[0][1] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h02ProjEta_[0][1] );

  sprintf(histo, (prefixME_ + "/EEOccupancyTask/EEOT TP digi thr occupancy EE - projection phi").c_str());
  me = dqmStore_->get(histo);
  h02ProjPhi_[0][1] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h02ProjPhi_[0][1] );

  sprintf(histo, (prefixME_ + "/EEOccupancyTask/EEOT TP digi thr occupancy EE +").c_str());
  me = dqmStore_->get(histo);
  h02_[1][1] = UtilsClient::getHisto<TH2F*> ( me, cloneME_, h02_[1][1] );

  sprintf(histo, (prefixME_ + "/EEOccupancyTask/EEOT TP digi thr occupancy EE + projection eta").c_str());
  me = dqmStore_->get(histo);
  h02ProjEta_[1][1] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h02ProjEta_[1][1] );

  sprintf(histo, (prefixME_ + "/EEOccupancyTask/EEOT TP digi thr occupancy EE + projection phi").c_str());
  me = dqmStore_->get(histo);
  h02ProjPhi_[1][1] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h02ProjPhi_[1][1] );

}

