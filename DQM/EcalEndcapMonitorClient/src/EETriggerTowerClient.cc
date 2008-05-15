/*
 * \file EETriggerTowerClient.cc
 *
 * $Date: 2008/04/08 15:06:26 $
 * $Revision: 1.68 $
 * \author G. Della Ricca
 * \author F. Cossutti
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

#include "DQM/EcalCommon/interface/UtilsClient.h"
#include "DQM/EcalCommon/interface/Numbers.h"

#include <DQM/EcalEndcapMonitorClient/interface/EETriggerTowerClient.h>

using namespace cms;
using namespace edm;
using namespace std;

EETriggerTowerClient::EETriggerTowerClient(const ParameterSet& ps){

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

  // vector of selected Super Modules (Defaults to all 18).
  superModules_.reserve(18);
  for ( unsigned int i = 1; i <= 18; i++ ) superModules_.push_back(i);
  superModules_ = ps.getUntrackedParameter<vector<int> >("superModules", superModules_);

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    h01_[ism-1] = 0;
    h02_[ism-1] = 0;
    i01_[ism-1] = 0;
    i02_[ism-1] = 0;
    j01_[ism-1] = 0;
    j02_[ism-1] = 0;

    l01_[ism-1] = 0;
    m01_[ism-1] = 0;
    n01_[ism-1] = 0;

    meh01_[ism-1] = 0;
    meh02_[ism-1] = 0;
    mei01_[ism-1] = 0;
    mei02_[ism-1] = 0;
    mej01_[ism-1] = 0;
    mej02_[ism-1] = 0;

    mel01_[ism-1] = 0;
    mem01_[ism-1] = 0;
    men01_[ism-1] = 0;

//     for (int j=0; j<34; j++) {
//
//       k01_[ism-1][j] = 0;
//       k02_[ism-1][j] = 0;
//
//       mek01_[ism-1][j] = 0;
//       mek02_[ism-1][j] = 0;
//
//     }

    me_h01_[ism-1] = 0;
    me_h02_[ism-1] = 0;
    for (int j=0; j<2; j++) {
      me_i01_[ism-1][j] = 0;
      me_i02_[ism-1][j] = 0;
      me_n01_[ism-1][j] = 0;
    }
    for (int j=0; j<6; j++) {
      me_j01_[ism-1][j] = 0;
      me_j02_[ism-1][j] = 0;
      me_m01_[ism-1][j] = 0;
    }

  }

}

EETriggerTowerClient::~EETriggerTowerClient(){

}

void EETriggerTowerClient::beginJob(DQMStore* dqmStore){

  dqmStore_ = dqmStore;

  if ( debug_ ) cout << "EETriggerTowerClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EETriggerTowerClient::beginRun(void) {

  if ( debug_ ) cout << "EETriggerTowerClient: beginRun" << endl;

  jevt_ = 0;

  this->setup();

}

void EETriggerTowerClient::endJob(void) {

  if ( debug_ ) cout << "EETriggerTowerClient: endJob, ievt = " << ievt_ << endl;

  this->cleanup();

}

void EETriggerTowerClient::endRun(void) {

  if ( debug_ ) cout << "EETriggerTowerClient: endRun, jevt = " << jevt_ << endl;

  this->cleanup();

}

void EETriggerTowerClient::setup(void) {

  char histo[200];

  dqmStore_->setCurrentFolder( prefixME_ + "/EETriggerTowerClient" );

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( me_h01_[ism-1] ) dqmStore_->removeElement( me_h01_[ism-1]->getName() );
    sprintf(histo, "EETTT Et map Real Digis %s", Numbers::sEE(ism).c_str());
    me_h01_[ism-1] = dqmStore_->bookProfile2D(histo, histo, 50, Numbers::ix0EE(ism)+0., Numbers::ix0EE(ism)+50., 50, Numbers::iy0EE(ism)+0., Numbers::iy0EE(ism)+50., 256, 0., 256., "s");
    me_h01_[ism-1]->setAxisTitle("jx", 1);
    me_h01_[ism-1]->setAxisTitle("jy", 2);
    if ( me_h02_[ism-1] ) dqmStore_->removeElement( me_h02_[ism-1]->getName() );
    sprintf(histo, "EETTT Et map Emulated Digis %s", Numbers::sEE(ism).c_str());
    me_h02_[ism-1] = dqmStore_->bookProfile2D(histo, histo, 50, Numbers::ix0EE(ism)+0., Numbers::ix0EE(ism)+50., 50, Numbers::iy0EE(ism)+0., Numbers::iy0EE(ism)+50., 256, 0., 256., "s");
    me_h02_[ism-1]->setAxisTitle("jx", 1);
    me_h02_[ism-1]->setAxisTitle("jy", 2);
    for (int j=0; j<2; j++) {
      if ( me_i01_[ism-1][j] ) dqmStore_->removeElement( me_i01_[ism-1][j]->getName() );
      sprintf(histo, "EETTT FineGrainVeto Real Digis Flag %d %s", j, Numbers::sEE(ism).c_str());
      me_i01_[ism-1][j] = dqmStore_->book2D(histo, histo, 50, Numbers::ix0EE(ism)+0., Numbers::ix0EE(ism)+50., 50, Numbers::iy0EE(ism)+0., Numbers::iy0EE(ism)+50.);
      me_i01_[ism-1][j]->setAxisTitle("jx", 1);
      me_i01_[ism-1][j]->setAxisTitle("jy", 2);
      if ( me_i02_[ism-1][j] ) dqmStore_->removeElement( me_i02_[ism-1][j]->getName() );
      sprintf(histo, "EETTT FineGrainVeto Emulated Digis Flag %d %s", j, Numbers::sEE(ism).c_str());
      me_i02_[ism-1][j] = dqmStore_->book2D(histo, histo, 50, Numbers::ix0EE(ism)+0., Numbers::ix0EE(ism)+50., 50, Numbers::iy0EE(ism)+0., Numbers::iy0EE(ism)+50.);
      me_i02_[ism-1][j]->setAxisTitle("jx", 1);
      me_i02_[ism-1][j]->setAxisTitle("jy", 2);
      if ( me_n01_[ism-1][j] ) dqmStore_->removeElement( me_n01_[ism-1][j]->getName() );
      sprintf(histo, "EETTT EmulFineGrainVetoError Flag %d %s", j, Numbers::sEE(ism).c_str());
      me_n01_[ism-1][j] = dqmStore_->book2D(histo, histo, 50, Numbers::ix0EE(ism)+0., Numbers::ix0EE(ism)+50., 50, Numbers::iy0EE(ism)+0., Numbers::iy0EE(ism)+50.);
      me_n01_[ism-1][j]->setAxisTitle("jx", 1);
      me_n01_[ism-1][j]->setAxisTitle("jy", 2);
    }
    for (int j=0; j<6; j++) {
      string bits;
      if ( j == 0 ) bits = "Bit 000";
      if ( j == 1 ) bits = "Bit 001";
      if ( j == 2 ) bits = "Bit 011";
      if ( j == 3 ) bits = "Bit 100";
      if ( j == 4 ) bits = "Bit 101";
      if ( j == 5 ) bits = "Bits 110+111";
      if ( me_j01_[ism-1][j] ) dqmStore_->removeElement( me_j01_[ism-1][j]->getName() );
      sprintf(histo, "EETTT Flags Real Digis %s %s", bits.c_str(), Numbers::sEE(ism).c_str());
      me_j01_[ism-1][j] = dqmStore_->book2D(histo, histo, 50, Numbers::ix0EE(ism)+0., Numbers::ix0EE(ism)+50., 50, Numbers::iy0EE(ism)+0., Numbers::iy0EE(ism)+50.);
      me_j01_[ism-1][j]->setAxisTitle("jx", 1);
      me_j01_[ism-1][j]->setAxisTitle("jy", 2);
      if ( me_j02_[ism-1][j] ) dqmStore_->removeElement( me_j02_[ism-1][j]->getName() );
      sprintf(histo, "EETTT Flags Emulated Digis %s %s", bits.c_str(), Numbers::sEE(ism).c_str());
      me_j02_[ism-1][j] = dqmStore_->book2D(histo, histo, 50, Numbers::ix0EE(ism)+0., Numbers::ix0EE(ism)+50., 50, Numbers::iy0EE(ism)+0., Numbers::iy0EE(ism)+50.);
      me_j02_[ism-1][j]->setAxisTitle("jx", 1);
      me_j02_[ism-1][j]->setAxisTitle("jy", 2);
      if ( me_m01_[ism-1][j] ) dqmStore_->removeElement( me_m01_[ism-1][j]->getName() );
      sprintf(histo, "EETTT EmulFlagError %s %s", bits.c_str(), Numbers::sEE(ism).c_str());
      me_m01_[ism-1][j] = dqmStore_->book2D(histo, histo, 50, Numbers::ix0EE(ism)+0., Numbers::ix0EE(ism)+50., 50, Numbers::iy0EE(ism)+0., Numbers::iy0EE(ism)+50.);
      me_m01_[ism-1][j]->setAxisTitle("jx", 1);
      me_m01_[ism-1][j]->setAxisTitle("jy", 2);
    }

  }

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( me_h01_[ism-1] ) me_h01_[ism-1]->Reset();
    if ( me_h02_[ism-1] ) me_h02_[ism-1]->Reset();
    for (int j=0; j<2; j++) {
      if ( me_i01_[ism-1][j] ) me_i01_[ism-1][j]->Reset();
      if ( me_i02_[ism-1][j] ) me_i02_[ism-1][j]->Reset();
      if ( me_n01_[ism-1][j] ) me_n01_[ism-1][j]->Reset();
    }
    for (int j=0; j<6; j++) {
      if ( me_j01_[ism-1][j] ) me_j01_[ism-1][j]->Reset();
      if ( me_j02_[ism-1][j] ) me_j02_[ism-1][j]->Reset();
      if ( me_m01_[ism-1][j] ) me_m01_[ism-1][j]->Reset();
    }

  }

}

void EETriggerTowerClient::cleanup(void) {

  if ( ! enableCleanup_ ) return;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( cloneME_ ) {
      if ( h01_[ism-1] ) delete h01_[ism-1];
      if ( h02_[ism-1] ) delete h02_[ism-1];
      if ( i01_[ism-1] ) delete i01_[ism-1];
      if ( i02_[ism-1] ) delete i02_[ism-1];
      if ( j01_[ism-1] ) delete j01_[ism-1];
      if ( j02_[ism-1] ) delete j02_[ism-1];
      if ( l01_[ism-1] ) delete l01_[ism-1];
      if ( m01_[ism-1] ) delete m01_[ism-1];
      if ( n01_[ism-1] ) delete n01_[ism-1];
    }

    h01_[ism-1] = 0;
    h02_[ism-1] = 0;
    i01_[ism-1] = 0;
    i02_[ism-1] = 0;
    j01_[ism-1] = 0;
    j02_[ism-1] = 0;

    l01_[ism-1] = 0;
    m01_[ism-1] = 0;
    n01_[ism-1] = 0;

    meh01_[ism-1] = 0;
    meh02_[ism-1] = 0;
    mei01_[ism-1] = 0;
    mei02_[ism-1] = 0;
    mej01_[ism-1] = 0;
    mej02_[ism-1] = 0;

    mel01_[ism-1] = 0;
    mem01_[ism-1] = 0;
    men01_[ism-1] = 0;

//     for (int j=0; j<34; j++) {
//
//       if ( cloneME_ ) {
//         if ( k01_[ism-1][j] ) delete k01_[ism-1][j];
//         if ( k02_[ism-1][j] ) delete k02_[ism-1][j];
//       }
//
//       k01_[ism-1][j] = 0;
//       k02_[ism-1][j] = 0;
//
//       mek01_[ism-1][j] = 0;
//       mek02_[ism-1][j] = 0;
//
//     }

  }

  dqmStore_->setCurrentFolder( prefixME_ + "/EETriggerTowerClient" );

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( me_h01_[ism-1] ) dqmStore_->removeElement( me_h01_[ism-1]->getName() );
    me_h01_[ism-1] = 0;
    if ( me_h02_[ism-1] ) dqmStore_->removeElement( me_h02_[ism-1]->getName() );
    me_h02_[ism-1] = 0;
    for (int j=0; j<2; j++) {
      if ( me_i01_[ism-1][j] ) dqmStore_->removeElement( me_i01_[ism-1][j]->getName() );
      me_i01_[ism-1][j] = 0;
      if ( me_i02_[ism-1][j] ) dqmStore_->removeElement( me_i02_[ism-1][j]->getName() );
      me_i02_[ism-1][j] = 0;
      if ( me_n01_[ism-1][j] ) dqmStore_->removeElement( me_n01_[ism-1][j]->getName() );
      me_n01_[ism-1][j] = 0;
    }
    for (int j=0; j<6; j++) {
      if ( me_j01_[ism-1][j] ) dqmStore_->removeElement( me_j01_[ism-1][j]->getName() );
      me_j01_[ism-1][j] = 0;
      if ( me_j02_[ism-1][j] ) dqmStore_->removeElement( me_j02_[ism-1][j]->getName() );
      me_j02_[ism-1][j] = 0;
      if ( me_m01_[ism-1][j] ) dqmStore_->removeElement( me_m01_[ism-1][j]->getName() );
      me_m01_[ism-1][j] = 0;
    }

  }

}

bool EETriggerTowerClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov) {

  bool status = true;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( verbose_ ) {
      cout << " " << Numbers::sEE(ism) << " (ism=" << ism << ")" << endl;
      cout << endl;
      UtilsClient::printBadChannels(mel01_[ism-1], UtilsClient::getHisto<TH2F*>(mel01_[ism-1]), true);
      for (int j=0; j<2; j++) {
        UtilsClient::printBadChannels(me_n01_[ism-1][j], UtilsClient::getHisto<TH2F*>(me_n01_[ism-1][j]), true);
      }
      for (int j=0; j<6; j++) {
        UtilsClient::printBadChannels(me_m01_[ism-1][j], UtilsClient::getHisto<TH2F*>(me_m01_[ism-1][j]), true);
      }
    }

  }

  return status;

}

void EETriggerTowerClient::analyze(void){

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( debug_ ) cout << "EETriggerTowerClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;
  }

  analyze("Real Digis",
          "EETriggerTowerTask", false );

  analyze("Emulated Digis",
          "EETriggerTowerTask/Emulated", true );

}

void EETriggerTowerClient::analyze(const char* nameext,
                                   const char* folder,
                                   bool emulated) {
  char histo[200];

  MonitorElement* me;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    sprintf(histo, (prefixME_ + "/%s/EETTT Et map %s %s").c_str(), folder, nameext, Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    if(!emulated) {
      h01_[ism-1] = UtilsClient::getHisto<TH3F*>( me, cloneME_, h01_[ism-1] );
      meh01_[ism-1] = me;
    }
    else {
      h02_[ism-1] = UtilsClient::getHisto<TH3F*>( me, cloneME_, h02_[ism-1] );
      meh02_[ism-1] = me;
    }

    sprintf(histo, (prefixME_ + "/%s/EETTT FineGrainVeto %s %s").c_str(), folder, nameext, Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    if(!emulated) {
      i01_[ism-1] = UtilsClient::getHisto<TH3F*>( me, cloneME_, i01_[ism-1] );
      mei01_[ism-1] = me;
    }
    else {
      i02_[ism-1] = UtilsClient::getHisto<TH3F*>( me, cloneME_, i02_[ism-1] );
      mei02_[ism-1] = me;
    }

    sprintf(histo, (prefixME_ + "/%s/EETTT Flags %s %s").c_str(), folder, nameext, Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    if(!emulated) {
      j01_[ism-1] = UtilsClient::getHisto<TH3F*>( me, cloneME_, j01_[ism-1] );
      mej01_[ism-1] = me;
    }
    else {
      j02_[ism-1] = UtilsClient::getHisto<TH3F*>( me, cloneME_, j02_[ism-1] );
      mej02_[ism-1] = me;
    }

    if(!emulated) {
      sprintf(histo, (prefixME_ + "/%s/EETTT EmulError %s").c_str(), folder, Numbers::sEE(ism).c_str());
      me = dqmStore_->get(histo);
      l01_[ism-1] = UtilsClient::getHisto<TH2F*>( me, cloneME_, l01_[ism-1] );
      mel01_[ism-1] = me;

      sprintf(histo, (prefixME_ + "/%s/EETTT EmulFlagError %s").c_str(), folder, Numbers::sEE(ism).c_str());
      me = dqmStore_->get(histo);
      m01_[ism-1] = UtilsClient::getHisto<TH3F*>( me, cloneME_, m01_[ism-1] );
      mem01_[ism-1] = me;

      sprintf(histo, (prefixME_ + "/%s/EETTT EmulFineGrainVetoError %s").c_str(), folder, Numbers::sEE(ism).c_str());
      me = dqmStore_->get(histo);
      n01_[ism-1] = UtilsClient::getHisto<TH3F*>( me, cloneME_, n01_[ism-1] );
      men01_[ism-1] = me;

    }

//     for (int j=0; j<34; j++) {
//
//       sprintf(histo, (prefixME_ + "/EETriggerTowerTask/EnergyMaps/EETTT Et T %s TT%02d").c_str(), ism, j+1);
//       me = dqmStore_->get(histo);
//       k01_[ism-1][j] = UtilsClient::getHisto<TH1F*>( me, cloneME_, k01_[ism-1][j] );
//       mek01_[ism-1][j] = me;
//
//       sprintf(histo, (prefixME_ + "/EETriggerTowerTask/EnergyMaps/EETTT Et R %s TT%02d").c_str(), ism, j+1);
//       me = dqmStore_->get(histo);
//       k02_[ism-1][j] = UtilsClient::getHisto<TH1F*>( me, cloneME_, k02_[ism-1][j] );
//       mek02_[ism-1][j] = me;
//
//     }

    if ( me_h01_[ism-1] ) me_h01_[ism-1]->Reset();
    if ( me_h02_[ism-1] ) me_h02_[ism-1]->Reset();
    for (int j=0; j<2; j++) {
      if ( me_i01_[ism-1][j] ) me_i01_[ism-1][j]->Reset();
      if ( me_i02_[ism-1][j] ) me_i02_[ism-1][j]->Reset();
      if ( me_n01_[ism-1][j] ) me_n01_[ism-1][j]->Reset();
    }
    for (int j=0; j<6; j++) {
      if ( me_j01_[ism-1][j] ) me_j01_[ism-1][j]->Reset();
      if ( me_j02_[ism-1][j] ) me_j02_[ism-1][j]->Reset();
      if ( me_m01_[ism-1][j] ) me_m01_[ism-1][j]->Reset();
    }


    for (int ix = 1; ix <= 50; ix++) {
      for (int iy = 1; iy <= 50; iy++) {

        int jx = ix + Numbers::ix0EE(ism);
        int jy = iy + Numbers::iy0EE(ism);

        for (int j = 1; j <= 256; j++) {
          if ( h01_[ism-1] ) me_h01_[ism-1]->Fill(jx-0.5, jy-0.5, j-0.5, h01_[ism-1]->GetBinContent(ix, iy, j));
          if ( h02_[ism-1] ) me_h02_[ism-1]->Fill(jx-0.5, jy-0.5, j-0.5, h02_[ism-1]->GetBinContent(ix, iy, j));
        }
        for (int j=0; j<2; j++) {
          if ( i01_[ism-1] ) me_i01_[ism-1][j]->Fill(jx-0.5, jy-0.5, i01_[ism-1]->GetBinContent(ix, iy, j+1));
          if ( i02_[ism-1] ) me_i02_[ism-1][j]->Fill(jx-0.5, jy-0.5, i02_[ism-1]->GetBinContent(ix, iy, j+1));
          if ( n01_[ism-1] ) me_n01_[ism-1][j]->Fill(jx-0.5, jy-0.5, n01_[ism-1]->GetBinContent(ix, iy, j+1));
        }
        for (int j=0; j<6; j++) {
          if ( j == 0 ) {
            if ( j01_[ism-1] ) me_j01_[ism-1][j]->Fill(jx-0.5, jy-0.5, j01_[ism-1]->GetBinContent(ix, iy, j+1));
            if ( j02_[ism-1] ) me_j02_[ism-1][j]->Fill(jx-0.5, jy-0.5, j02_[ism-1]->GetBinContent(ix, iy, j+1));
            if ( m01_[ism-1] ) me_m01_[ism-1][j]->Fill(jx-0.5, jy-0.5, m01_[ism-1]->GetBinContent(ix, iy, j+1));
          }
          if ( j == 1 ) {
            if ( j01_[ism-1] ) me_j01_[ism-1][j]->Fill(jx-0.5, jy-0.5, j01_[ism-1]->GetBinContent(ix, iy, j+1));
            if ( j02_[ism-1] ) me_j02_[ism-1][j]->Fill(jx-0.5, jy-0.5, j02_[ism-1]->GetBinContent(ix, iy, j+1));
            if ( m01_[ism-1] ) me_m01_[ism-1][j]->Fill(jx-0.5, jy-0.5, m01_[ism-1]->GetBinContent(ix, iy, j+1));
          }
          if ( j == 2 ) {
            if ( j01_[ism-1] ) me_j01_[ism-1][j]->Fill(jx-0.5, jy-0.5, j01_[ism-1]->GetBinContent(ix, iy, j+2));
            if ( j02_[ism-1] ) me_j02_[ism-1][j]->Fill(jx-0.5, jy-0.5, j02_[ism-1]->GetBinContent(ix, iy, j+2));
            if ( m01_[ism-1] ) me_m01_[ism-1][j]->Fill(jx-0.5, jy-0.5, m01_[ism-1]->GetBinContent(ix, iy, j+2));
          }
          if ( j == 3 ) {
            if ( j01_[ism-1] ) me_j01_[ism-1][j]->Fill(jx-0.5, jy-0.5, j01_[ism-1]->GetBinContent(ix, iy, j+2));
            if ( j02_[ism-1] ) me_j02_[ism-1][j]->Fill(jx-0.5, jy-0.5, j02_[ism-1]->GetBinContent(ix, iy, j+2));
            if ( m01_[ism-1] ) me_m01_[ism-1][j]->Fill(jx-0.5, jy-0.5, m01_[ism-1]->GetBinContent(ix, iy, j+2));
          }
          if ( j == 4 ) {
            if ( j01_[ism-1] ) me_j01_[ism-1][j]->Fill(jx-0.5, jy-0.5, j01_[ism-1]->GetBinContent(ix, iy, j+2));
            if ( j02_[ism-1] ) me_j02_[ism-1][j]->Fill(jx-0.5, jy-0.5, j02_[ism-1]->GetBinContent(ix, iy, j+2));
            if ( m01_[ism-1] ) me_m01_[ism-1][j]->Fill(jx-0.5, jy-0.5, m01_[ism-1]->GetBinContent(ix, iy, j+2));
          }
          if ( j == 5 ) {
            if ( j01_[ism-1] ) {
              me_j01_[ism-1][j]->Fill(jx-0.5, jy-0.5, j01_[ism-1]->GetBinContent(ix, iy, j+2));
              me_j01_[ism-1][j]->Fill(jx-0.5, jy-0.5, j01_[ism-1]->GetBinContent(ix, iy, j+3));
            }
            if ( j02_[ism-1] ) {
              me_j02_[ism-1][j]->Fill(jx-0.5, jy-0.5, j02_[ism-1]->GetBinContent(ix, iy, j+2));
              me_j02_[ism-1][j]->Fill(jx-0.5, jy-0.5, j02_[ism-1]->GetBinContent(ix, iy, j+3));
            }
            if ( m01_[ism-1] ) {
              me_m01_[ism-1][j]->Fill(jx-0.5, jy-0.5, m01_[ism-1]->GetBinContent(ix, iy, j+2));
              me_m01_[ism-1][j]->Fill(jx-0.5, jy-0.5, m01_[ism-1]->GetBinContent(ix, iy, j+3));
            }
          }
        }

      }
    }

  }

}

void EETriggerTowerClient::htmlOutput(int run, string& htmlDir, string& htmlName){

  if ( verbose_ ) cout << "Preparing EETriggerTowerClient html output ..." << std::endl;

  std::ofstream htmlFile[19];

  htmlFile[0].open((htmlDir + htmlName).c_str());

  // html page header
  htmlFile[0] << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << std::endl;
  htmlFile[0] << "<html>  " << std::endl;
  htmlFile[0] << "<head>  " << std::endl;
  htmlFile[0] << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << std::endl;
  htmlFile[0] << " http-equiv=\"content-type\">  " << std::endl;
  htmlFile[0] << "  <title>Monitor:TriggerTowerTask output</title> " << std::endl;
  htmlFile[0] << "</head>  " << std::endl;
  htmlFile[0] << "<style type=\"text/css\"> td { font-weight: bold } </style>" << std::endl;
  htmlFile[0] << "<body>  " << std::endl;
  //htmlFile[0] << "<br>  " << std::endl;
  htmlFile[0] << "<a name=""top""></a>" << endl;
  htmlFile[0] << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << std::endl;
  htmlFile[0] << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << std::endl;
  htmlFile[0] << " style=\"color: rgb(0, 0, 153);\">" << run << "</span></h2>" << std::endl;
  htmlFile[0] << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << std::endl;
  htmlFile[0] << " style=\"color: rgb(0, 0, 153);\">TRIGGER TOWER</span></h2> " << std::endl;
  htmlFile[0] << "<br>" << std::endl;
  //htmlFile[0] << "<table border=1><tr><td bgcolor=red>channel has problems in this task</td>" << std::endl;
  //htmlFile[0] << "<td bgcolor=lime>channel has NO problems</td>" << std::endl;
  //htmlFile[0] << "<td bgcolor=yellow>channel is missing</td></table>" << std::endl;
  htmlFile[0] << "<hr>" << std::endl;
  htmlFile[0] << "<table border=1>" << std::endl;
  for ( unsigned int i=0; i<superModules_.size(); i ++ ) {
    htmlFile[0] << "<td bgcolor=white><a href=""#"
                << Numbers::sEE(superModules_[i]) << ">"
                << setfill( '0' ) << setw(2) << superModules_[i] << "</a></td>";
  }
  htmlFile[0] << std::endl << "</table>" << std::endl;

  // Produce the plots to be shown as .png files from existing histograms

  const int csize = 250;

  //const double histMax = 1.e15;

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

  string imgMeName[3], imgName, meName;

  TCanvas* cMe1 = new TCanvas("cMe1", "Temp", 2*csize, 2*csize);
  //  TCanvas* cMe2 = new TCanvas("cMe2", "Temp", int(1.2*csize), int(1.2*csize));
  //  TCanvas* cMe3 = new TCanvas("cMe3", "Temp", int(0.4*csize), int(0.4*csize));
  TCanvas* cMe2 = new TCanvas("cMe2", "Temp", int(1.8*csize), int(1.8*csize));
  TCanvas* cMe3 = new TCanvas("cMe3", "Temp", int(0.9*csize), int(0.9*csize));

  TH2F* obj2f;
  TProfile2D* obj2p;

  // Loop on endcap supermodules

  for ( unsigned int i=0; i<superModules_.size(); i ++ ) {

    int ism = superModules_[i];

    if ( i>0 ) htmlFile[0] << "<a href=""#top"">Top</a>" << std::endl;
    htmlFile[0] << "<hr>" << std::endl;
    htmlFile[0] << "<h3><a name="""
                << Numbers::sEE(ism) << """></a><strong>"
                << Numbers::sEE(ism) << "</strong></h3>" << endl;


    // ---------------------------  Emulator Error

    htmlFile[0] << "<h3><strong>Emulator Error</strong></h3>" << std::endl;
    htmlFile[0] << "<table border=\"0\" cellspacing=\"0\" " << std::endl;
    htmlFile[0] << "cellpadding=\"10\" align=\"center\"> " << std::endl;
    htmlFile[0] << "<tr align=\"center\">" << std::endl;


    imgMeName[0] = "";

    obj2f = l01_[ism-1];

    if ( obj2f ) {

      meName = obj2f->GetName();

      replace(meName.begin(), meName.end(), ' ', '_');
      imgMeName[0] = meName + ".png";
      imgName = htmlDir + imgMeName[0];

      cMe2->cd();
      gStyle->SetOptStat(" ");
      gStyle->SetPalette(10, pCol5);
      cMe2->SetGridx();
      cMe2->SetGridy();
      obj2f->GetXaxis()->SetLabelSize(0.02);
      obj2f->GetXaxis()->SetTitleSize(0.02);
      obj2f->GetYaxis()->SetLabelSize(0.02);
      obj2f->GetYaxis()->SetTitleSize(0.02);
      obj2f->GetZaxis()->SetLabelSize(0.02);
      obj2f->SetMinimum(0);
      obj2f->Draw("colz");
      int x1 = labelGrid.GetXaxis()->FindFixBin(Numbers::ix0EE(ism)+0.);
      int x2 = labelGrid.GetXaxis()->FindFixBin(Numbers::ix0EE(ism)+50.);
      int y1 = labelGrid.GetYaxis()->FindFixBin(Numbers::iy0EE(ism)+0.);
      int y2 = labelGrid.GetYaxis()->FindFixBin(Numbers::iy0EE(ism)+50.);
      labelGrid.GetXaxis()->SetRange(x1, x2);
      labelGrid.GetYaxis()->SetRange(y1, y2);
      labelGrid.Draw("text,same");
      cMe2->SetBit(TGraph::kClipFrame);
      TLine l;
      l.SetLineWidth(1);
      for ( int i=0; i<201; i=i+1){
        if ( (Numbers::ixSectorsEE[i]!=0 || Numbers::iySectorsEE[i]!=0) && (Numbers::ixSectorsEE[i+1]!=0 || Numbers::iySectorsEE[i+1]!=0) ) {
          l.DrawLine(Numbers::ixSectorsEE[i], Numbers::iySectorsEE[i], Numbers::ixSectorsEE[i+1], Numbers::iySectorsEE[i+1]);
        }
      }
      cMe2->Update();
      cMe2->SaveAs(imgName.c_str());
    }

    htmlFile[0] << "<img src=\"" << imgMeName[0] << "\"><br>" << std::endl;
    htmlFile[0] << "</tr>" << std::endl;
    htmlFile[0] << "<br><br>" << std::endl;

    // ---------------------------  Et plots

    for(int iemu=0; iemu<2; iemu++) {

      imgMeName[iemu] = "";

      obj2p = 0;
      switch ( iemu ) {
        case 0:
          obj2p = UtilsClient::getHisto<TProfile2D*>( me_h01_[ism-1] );
          break;
        case 1:
          obj2p = UtilsClient::getHisto<TProfile2D*>( me_h02_[ism-1] );
          break;
        default:
          break;
      }

      if ( obj2p ) {

        meName = obj2p->GetName();

        replace(meName.begin(), meName.end(), ' ', '_');
        imgMeName[iemu] = meName + ".png";
        imgName = htmlDir + imgMeName[iemu];

        cMe1->cd();
        gStyle->SetOptStat(" ");
        gStyle->SetPalette(10, pCol4);
        cMe1->SetGridx();
        cMe1->SetGridy();
        obj2p->GetXaxis()->SetLabelSize(0.02);
        obj2p->GetXaxis()->SetTitleSize(0.02);
        obj2p->GetYaxis()->SetLabelSize(0.02);
        obj2p->GetYaxis()->SetTitleSize(0.02);
        obj2p->GetZaxis()->SetLabelSize(0.02);
        obj2p->Draw("colz");
        int x1 = labelGrid.GetXaxis()->FindFixBin(Numbers::ix0EE(ism)+0.);
        int x2 = labelGrid.GetXaxis()->FindFixBin(Numbers::ix0EE(ism)+50.);
        int y1 = labelGrid.GetYaxis()->FindFixBin(Numbers::iy0EE(ism)+0.);
        int y2 = labelGrid.GetYaxis()->FindFixBin(Numbers::iy0EE(ism)+50.);
        labelGrid.GetXaxis()->SetRange(x1, x2);
        labelGrid.GetYaxis()->SetRange(y1, y2);
        labelGrid.Draw("text,same");
        cMe1->SetBit(TGraph::kClipFrame);
        TLine l;
        l.SetLineWidth(1);
          for ( int i=0; i<201; i=i+1){
          if ( (Numbers::ixSectorsEE[i]!=0 || Numbers::iySectorsEE[i]!=0) && (Numbers::ixSectorsEE[i+1]!=0 || Numbers::iySectorsEE[i+1]!=0) ) {
            l.DrawLine(Numbers::ixSectorsEE[i], Numbers::iySectorsEE[i], Numbers::ixSectorsEE[i+1], Numbers::iySectorsEE[i+1]);
          }
        }
        cMe1->Update();
        cMe1->SaveAs(imgName.c_str());
      }
    }

    htmlFile[0] << "<td><img src=\"" << imgMeName[0] << "\"></td>" << std::endl;
    htmlFile[0] << "<td><img src=\"" << imgMeName[1] << "\"></td>" << std::endl;
    htmlFile[0] << "</table>" << std::endl;
    htmlFile[0] << "<br>" << std::endl;

    std::stringstream subpage;
    subpage << htmlName.substr( 0, htmlName.find( ".html" ) ) << "_" << Numbers::sEE(ism) << ".html";
    htmlFile[0] << "<a href=\"" << subpage.str() << "\">" << Numbers::sEE(ism) << " details</a><br>" << std::endl;
    htmlFile[0] << "<hr>" << std::endl;

    htmlFile[ism].open((htmlDir + subpage.str()).c_str());

    // html page header
    htmlFile[ism] << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << std::endl;
    htmlFile[ism] << "<html>  " << std::endl;
    htmlFile[ism] << "<head>  " << std::endl;
    htmlFile[ism] << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << std::endl;
    htmlFile[ism] << " http-equiv=\"content-type\">  " << std::endl;
    htmlFile[ism] << "  <title>Monitor:TriggerTowerTask output " << Numbers::sEE(ism) << "</title> " << std::endl;
    htmlFile[ism] << "</head>  " << std::endl;
    htmlFile[ism] << "<style type=\"text/css\"> td { font-weight: bold } </style>" << std::endl;
    htmlFile[ism] << "<body>  " << std::endl;
    htmlFile[ism] << "<br>  " << std::endl;
    htmlFile[ism] << "<h3>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << std::endl;
    htmlFile[ism] << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << std::endl;
    htmlFile[ism] << " style=\"color: rgb(0, 0, 153);\">" << run << "</span></h3>" << std::endl;
    htmlFile[ism] << "<h3>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << std::endl;
    htmlFile[ism] << " style=\"color: rgb(0, 0, 153);\">TRIGGER TOWER</span></h3> " << std::endl;
    htmlFile[ism] << "<h3>SM:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << std::endl;
    htmlFile[ism] << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << std::endl;
    htmlFile[ism] << " style=\"color: rgb(0, 0, 153);\">" << Numbers::sEE(ism) << "</span></h3>" << std::endl;
    htmlFile[ism] << "<hr>" << std::endl;

    // ---------------------------  Flag bits plots

    htmlFile[ism] << "<h3><strong>Trigger Tower Flags</strong></h3>" << std::endl;
    htmlFile[ism] << "<table border=\"0\" cellspacing=\"0\" " << std::endl;
    htmlFile[ism] << "cellpadding=\"10\" align=\"center\"> " << std::endl;
    htmlFile[ism] << "<tr align=\"center\">" << std::endl;

    int counter = 0;

    for (int j=0; j<6; j++) {

      for(int iemu=0; iemu<3; iemu++) {

        imgMeName[iemu] = "";

        obj2f = 0;
        switch ( iemu ) {
          case 0:
            obj2f = UtilsClient::getHisto<TH2F*>( me_m01_[ism-1][j] );
            break;
          case 1:
            obj2f = UtilsClient::getHisto<TH2F*>( me_j01_[ism-1][j] );
            break;
          case 2:
            obj2f = UtilsClient::getHisto<TH2F*>( me_j02_[ism-1][j] );
            break;
          default:
            break;
        }

        if ( obj2f ) {

          meName = obj2f->GetName();

          replace(meName.begin(), meName.end(), ' ', '_');
          imgMeName[iemu] = meName + ".png";
          imgName = htmlDir + imgMeName[iemu];

          counter++;

          cMe2->cd();
          gStyle->SetOptStat(" ");
          if (iemu == 0 ) gStyle->SetPalette(10, pCol5);
          if (iemu == 1 || iemu == 2) gStyle->SetPalette(10, pCol4);
          cMe2->SetGridx();
          cMe2->SetGridy();
          obj2f->GetXaxis()->SetLabelSize(0.02);
          obj2f->GetXaxis()->SetTitleSize(0.02);
          obj2f->GetYaxis()->SetLabelSize(0.02);
          obj2f->GetYaxis()->SetTitleSize(0.02);
          obj2f->GetZaxis()->SetLabelSize(0.02);
          obj2f->SetMinimum(0);
          obj2f->Draw("colz");
          int x1 = labelGrid.GetXaxis()->FindFixBin(Numbers::ix0EE(ism)+0.);
          int x2 = labelGrid.GetXaxis()->FindFixBin(Numbers::ix0EE(ism)+50.);
          int y1 = labelGrid.GetYaxis()->FindFixBin(Numbers::iy0EE(ism)+0.);
          int y2 = labelGrid.GetYaxis()->FindFixBin(Numbers::iy0EE(ism)+50.);
          labelGrid.GetXaxis()->SetRange(x1, x2);
          labelGrid.GetYaxis()->SetRange(y1, y2);
          labelGrid.Draw("text,same");
          cMe2->SetBit(TGraph::kClipFrame);
          TLine l;
          l.SetLineWidth(1);
          for ( int i=0; i<201; i=i+1){
            if ( (Numbers::ixSectorsEE[i]!=0 || Numbers::iySectorsEE[i]!=0) && (Numbers::ixSectorsEE[i+1]!=0 || Numbers::iySectorsEE[i+1]!=0) ) {
              l.DrawLine(Numbers::ixSectorsEE[i], Numbers::iySectorsEE[i], Numbers::ixSectorsEE[i+1], Numbers::iySectorsEE[i+1]);
            }
          }
          cMe2->Update();
          cMe2->SaveAs(imgName.c_str());

          htmlFile[ism] << "<td><img src=\"" << imgMeName[iemu] << "\"></td>" << std::endl;
          if ( counter%3 == 0 ) htmlFile[ism] << "</tr><tr>" << std::endl;

        }
      }
    }

    htmlFile[ism] << "</tr>" << std::endl << "</table>" << std::endl;



    // ---------------------------  Fine Grain Veto

    htmlFile[ism] << "<h3><strong>Fine Grain Veto</strong></h3>" << std::endl;
    htmlFile[ism] << "<table border=\"0\" cellspacing=\"0\" " << std::endl;
    htmlFile[ism] << "cellpadding=\"10\" align=\"center\"> " << std::endl;
    htmlFile[ism] << "<tr align=\"center\">" << std::endl;


    for (int j=0; j<2; j++) {

      for(int iemu=0; iemu<3; iemu++) {

        imgMeName[iemu] = "";

        obj2f = 0;
        switch ( iemu ) {
          case 0:
            obj2f = UtilsClient::getHisto<TH2F*>( me_n01_[ism-1][j] );
            break;
          case 1:
            obj2f = UtilsClient::getHisto<TH2F*>( me_i01_[ism-1][j] );
            break;
          case 2:
            obj2f = UtilsClient::getHisto<TH2F*>( me_i02_[ism-1][j] );
            break;
          default:
            break;
        }

        if ( obj2f ) {

          meName = obj2f->GetName();

          replace(meName.begin(), meName.end(), ' ', '_');
          imgMeName[iemu] = meName + ".png";
          imgName = htmlDir + imgMeName[iemu];

          cMe2->cd();
          gStyle->SetOptStat(" ");
          if (iemu == 0 ) gStyle->SetPalette(10, pCol5);
          if (iemu == 1 || iemu == 2) gStyle->SetPalette(10, pCol4);
          cMe2->SetGridx();
          cMe2->SetGridy();
          obj2f->GetXaxis()->SetLabelSize(0.02);
          obj2f->GetXaxis()->SetTitleSize(0.02);
          obj2f->GetYaxis()->SetLabelSize(0.02);
          obj2f->GetYaxis()->SetTitleSize(0.02);
          obj2f->GetZaxis()->SetLabelSize(0.02);
          obj2f->SetMinimum(0);
          obj2f->Draw("colz");
          int x1 = labelGrid.GetXaxis()->FindFixBin(Numbers::ix0EE(ism)+0.);
          int x2 = labelGrid.GetXaxis()->FindFixBin(Numbers::ix0EE(ism)+50.);
          int y1 = labelGrid.GetYaxis()->FindFixBin(Numbers::iy0EE(ism)+0.);
          int y2 = labelGrid.GetYaxis()->FindFixBin(Numbers::iy0EE(ism)+50.);
          labelGrid.GetXaxis()->SetRange(x1, x2);
          labelGrid.GetYaxis()->SetRange(y1, y2);
          labelGrid.Draw("text,same");
          cMe2->SetBit(TGraph::kClipFrame);
          TLine l;
          l.SetLineWidth(1);
          for ( int i=0; i<201; i=i+1){
            if ( (Numbers::ixSectorsEE[i]!=0 || Numbers::iySectorsEE[i]!=0) && (Numbers::ixSectorsEE[i+1]!=0 || Numbers::iySectorsEE[i+1]!=0) ) {
              l.DrawLine(Numbers::ixSectorsEE[i], Numbers::iySectorsEE[i], Numbers::ixSectorsEE[i+1], Numbers::iySectorsEE[i+1]);
            }
          }
          cMe2->Update();
          cMe2->SaveAs(imgName.c_str());

          htmlFile[ism] << "<td><img src=\"" << imgMeName[iemu] << "\"></td>" << std::endl;
        }
      }
      htmlFile[ism] << "</tr><tr>" << std::endl;
    }

    htmlFile[ism] << "</tr>" << std::endl << "</table>" << std::endl;

    // ---------------------------  Et plots per Tower

    //     htmlFile[ism] << "<h3><strong>Et</strong></h3>" << std::endl;
    //     htmlFile[ism] << "<table border=\"0\" cellspacing=\"0\" " << std::endl;
    //     htmlFile[ism] << "cellpadding=\"10\" align=\"center\"> " << std::endl;
    //     htmlFile[ism] << "<tr align=\"center\">" << std::endl;
    //
    //     for (int j=0; j<34; j++) {
    //
    //       TH1F* obj1f1 = k01_[ism-1][j];
    //       TH1F* obj1f2 = k02_[ism-1][j];
    //
    //       if ( obj1f1 ) {
    //
    //         imgMeName[iemu] = "";
    //
    //         meName = obj1f1->GetName();
    //
    //         replace(meName.begin(), meName.end(), ' ', '_');
    //         imgMeName[iemu] = meName + ".png";
    //         imgName = htmlDir + imgMeName[iemu];
    //
    //         cMe3->cd();
    //         gStyle->SetOptStat("euomr");
    //         if ( obj1f2 ) {
    //           float m = TMath::Max( obj1f1->GetMaximum(), obj1f2->GetMaximum() );
    //           obj1f1->SetMaximum( m + 1. );
    //         }
    //         obj1f1->SetStats(kTRUE);
    //         gStyle->SetStatW( gStyle->GetStatW() * 1.5 );
    //         obj1f1->Draw();
    //         cMe3->Update();
    //
    //         if ( obj1f2 ) {
    //           gStyle->SetStatY( gStyle->GetStatY() - 1.25*gStyle->GetStatH() );
    //           gStyle->SetStatTextColor( kRed );
    //           obj1f2->SetStats(kTRUE);
    //           obj1f2->SetLineColor( kRed );
    //           obj1f2->Draw( "sames" );
    //           cMe3->Update();
    //           gStyle->SetStatY( gStyle->GetStatY() + 1.25*gStyle->GetStatH() );
    //           gStyle->SetStatTextColor( kBlack );
    //         }
    //
    //         gStyle->SetStatW( gStyle->GetStatW() / 1.5 );
    //         cMe3->SaveAs(imgName.c_str());
    //
    //         htmlFile[ism] << "<td><img src=\"" << imgMeName[iemu] << "\"></td>" << std::endl;
    //
    //       }
    //
    //       if ( (j+1)%4 == 0 ) htmlFile[ism] << "</tr><tr>" << std::endl;
    //
    //     }

    htmlFile[ism] << "</tr>" << std::endl << "</table>" << std::endl;

    // html page footer
    htmlFile[ism] << "</body> " << std::endl;
    htmlFile[ism] << "</html> " << std::endl;
    htmlFile[ism].close();
  }

  delete cMe1;
  delete cMe2;
  delete cMe3;

  // html page footer
  htmlFile[0] << "</body> " << std::endl;
  htmlFile[0] << "</html> " << std::endl;
  htmlFile[0].close();

}

