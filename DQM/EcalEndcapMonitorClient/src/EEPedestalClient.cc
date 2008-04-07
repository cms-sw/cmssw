/*
 * \file EEPedestalClient.cc
 *
 * $Date: 2008/04/05 10:03:04 $
 * $Revision: 1.70 $
 * \author G. Della Ricca
 * \author F. Cossutti
 *
*/

#include <memory>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <math.h>

#include "TCanvas.h"
#include "TStyle.h"
#include "TGraph.h"
#include "TLine.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "OnlineDB/EcalCondDB/interface/MonPedestalsDat.h"
#include "OnlineDB/EcalCondDB/interface/MonPNPedDat.h"
#include "OnlineDB/EcalCondDB/interface/RunCrystalErrorsDat.h"
#include "OnlineDB/EcalCondDB/interface/RunPNErrorsDat.h"

#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"

#include "CondTools/Ecal/interface/EcalErrorDictionary.h"

#include "DQM/EcalCommon/interface/EcalErrorMask.h"
#include "DQM/EcalCommon/interface/UtilsClient.h"
#include "DQM/EcalCommon/interface/LogicID.h"
#include "DQM/EcalCommon/interface/Numbers.h"

#include <DQM/EcalEndcapMonitorClient/interface/EEPedestalClient.h>

using namespace cms;
using namespace edm;
using namespace std;

EEPedestalClient::EEPedestalClient(const ParameterSet& ps){

  // cloneME switch
  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);

  // debug switch
  debug_ = ps.getUntrackedParameter<bool>("debug", false);

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
    h03_[ism-1] = 0;

    j01_[ism-1] = 0;
    j02_[ism-1] = 0;
    j03_[ism-1] = 0;

    k01_[ism-1] = 0;
    k02_[ism-1] = 0;
    k03_[ism-1] = 0;

    i01_[ism-1] = 0;
    i02_[ism-1] = 0;

  }

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    meg01_[ism-1] = 0;
    meg02_[ism-1] = 0;
    meg03_[ism-1] = 0;

    meg04_[ism-1] = 0;
    meg05_[ism-1] = 0;

    mep01_[ism-1] = 0;
    mep02_[ism-1] = 0;
    mep03_[ism-1] = 0;

    mer01_[ism-1] = 0;
    mer02_[ism-1] = 0;
    mer03_[ism-1] = 0;

    mer04_[ism-1] = 0;
    mer05_[ism-1] = 0;

    mes01_[ism-1] = 0;
    mes02_[ism-1] = 0;
    mes03_[ism-1] = 0;

    met01_[ism-1] = 0;
    met02_[ism-1] = 0;
    met03_[ism-1] = 0;

  }

  expectedMean_[0] = 200.0;
  expectedMean_[1] = 200.0;
  expectedMean_[2] = 200.0;

  discrepancyMean_[0] = 25.0;
  discrepancyMean_[1] = 25.0;
  discrepancyMean_[2] = 25.0;

  RMSThreshold_[0] = 1.0;
  RMSThreshold_[1] = 1.5;
  RMSThreshold_[2] = 2.5;

  expectedMeanPn_[0] = 750.0;
  expectedMeanPn_[1] = 750.0;

  discrepancyMeanPn_[0] = 100.0;
  discrepancyMeanPn_[1] = 100.0;

  RMSThresholdPn_[0] = 1.0;
  RMSThresholdPn_[1] = 3.0;

}

EEPedestalClient::~EEPedestalClient(){

}

void EEPedestalClient::beginJob(DQMStore* dbe){

  dbe_ = dbe;

  if ( debug_ ) cout << "EEPedestalClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EEPedestalClient::beginRun(void){

  if ( debug_ ) cout << "EEPedestalClient: beginRun" << endl;

  jevt_ = 0;

  this->setup();

}

void EEPedestalClient::endJob(void) {

  if ( debug_ ) cout << "EEPedestalClient: endJob, ievt = " << ievt_ << endl;

  this->cleanup();

}

void EEPedestalClient::endRun(void) {

  if ( debug_ ) cout << "EEPedestalClient: endRun, jevt = " << jevt_ << endl;

  this->cleanup();

}

void EEPedestalClient::setup(void) {

  char histo[200];

  dbe_->setCurrentFolder( "EcalEndcap/EEPedestalClient" );

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( meg01_[ism-1] ) dbe_->removeElement( meg01_[ism-1]->getName() );
    sprintf(histo, "EEPT pedestal quality G01 %s", Numbers::sEE(ism).c_str());
    meg01_[ism-1] = dbe_->book2D(histo, histo, 50, Numbers::ix0EE(ism)+0., Numbers::ix0EE(ism)+50., 50, Numbers::iy0EE(ism)+0., Numbers::iy0EE(ism)+50.);
    meg01_[ism-1]->setAxisTitle("jx", 1);
    meg01_[ism-1]->setAxisTitle("jy", 2);
    if ( meg02_[ism-1] ) dbe_->removeElement( meg02_[ism-1]->getName() );
    sprintf(histo, "EEPT pedestal quality G06 %s", Numbers::sEE(ism).c_str());
    meg02_[ism-1] = dbe_->book2D(histo, histo, 50, Numbers::ix0EE(ism)+0., Numbers::ix0EE(ism)+50., 50, Numbers::iy0EE(ism)+0., Numbers::iy0EE(ism)+50.);
    meg02_[ism-1]->setAxisTitle("jx", 1);
    meg02_[ism-1]->setAxisTitle("jy", 2);
    if ( meg03_[ism-1] ) dbe_->removeElement( meg03_[ism-1]->getName() );
    sprintf(histo, "EEPT pedestal quality G12 %s", Numbers::sEE(ism).c_str());
    meg03_[ism-1] = dbe_->book2D(histo, histo, 50, Numbers::ix0EE(ism)+0., Numbers::ix0EE(ism)+50., 50, Numbers::iy0EE(ism)+0., Numbers::iy0EE(ism)+50.);
    meg03_[ism-1]->setAxisTitle("jx", 1);
    meg03_[ism-1]->setAxisTitle("jy", 2);

    if ( meg04_[ism-1] ) dbe_->removeElement( meg04_[ism-1]->getName() );
    sprintf(histo, "EEPT pedestal quality PNs G01 %s", Numbers::sEE(ism).c_str());
    meg04_[ism-1] = dbe_->book2D(histo, histo, 10, 0., 10., 1, 0., 5.);
    meg04_[ism-1]->setAxisTitle("pseudo-strip", 1);
    meg04_[ism-1]->setAxisTitle("channel", 2);
    if ( meg05_[ism-1] ) dbe_->removeElement( meg05_[ism-1]->getName() );
    sprintf(histo, "EEPT pedestal quality PNs G16 %s", Numbers::sEE(ism).c_str());
    meg05_[ism-1] = dbe_->book2D(histo, histo, 10, 0., 10., 1, 0., 5.);
    meg05_[ism-1]->setAxisTitle("pseudo-strip", 1);
    meg05_[ism-1]->setAxisTitle("channel", 2);

    if ( mep01_[ism-1] ) dbe_->removeElement( mep01_[ism-1]->getName() );
    sprintf(histo, "EEPT pedestal mean G01 %s", Numbers::sEE(ism).c_str());
    mep01_[ism-1] = dbe_->book1D(histo, histo, 100, 150., 250.);
    mep01_[ism-1]->setAxisTitle("mean", 1);
    if ( mep02_[ism-1] ) dbe_->removeElement( mep02_[ism-1]->getName() );
    sprintf(histo, "EEPT pedestal mean G06 %s", Numbers::sEE(ism).c_str());
    mep02_[ism-1] = dbe_->book1D(histo, histo, 100, 150., 250.);
    mep02_[ism-1]->setAxisTitle("mean", 1);
    if ( mep03_[ism-1] ) dbe_->removeElement( mep03_[ism-1]->getName() );
    sprintf(histo, "EEPT pedestal mean G12 %s", Numbers::sEE(ism).c_str());
    mep03_[ism-1] = dbe_->book1D(histo, histo, 100, 150., 250.);
    mep03_[ism-1]->setAxisTitle("mean", 1);

    if ( mer01_[ism-1] ) dbe_->removeElement( mer01_[ism-1]->getName() );
    sprintf(histo, "EEPT pedestal rms G01 %s", Numbers::sEE(ism).c_str());
    mer01_[ism-1] = dbe_->book1D(histo, histo, 100, 0., 10.);
    mer01_[ism-1]->setAxisTitle("rms", 1);
    if ( mer02_[ism-1] ) dbe_->removeElement( mer02_[ism-1]->getName() );
    sprintf(histo, "EEPT pedestal rms G06 %s", Numbers::sEE(ism).c_str());
    mer02_[ism-1] = dbe_->book1D(histo, histo, 100, 0., 10.);
    mer02_[ism-1]->setAxisTitle("rms", 1);
    if ( mer03_[ism-1] ) dbe_->removeElement( mer03_[ism-1]->getName() );
    sprintf(histo, "EEPT pedestal rms G12 %s", Numbers::sEE(ism).c_str());
    mer03_[ism-1] = dbe_->book1D(histo, histo, 100, 0., 10.);
    mer03_[ism-1]->setAxisTitle("rms", 1);

    if ( mer04_[ism-1] ) dbe_->removeElement( mer04_[ism-1]->getName() );
    sprintf(histo, "EEPDT PNs pedestal rms %s G01", Numbers::sEE(ism).c_str());
    mer04_[ism-1] = dbe_->book1D(histo, histo, 100, 0., 10.);
    mer04_[ism-1]->setAxisTitle("rms", 1);
    if ( mer05_[ism-1] ) dbe_->removeElement( mer05_[ism-1]->getName() );
    sprintf(histo, "EEPDT PNs pedestal rms %s G16", Numbers::sEE(ism).c_str());
    mer05_[ism-1] = dbe_->book1D(histo, histo, 100, 0., 10.);
    mer05_[ism-1]->setAxisTitle("rms", 1);

    if ( mes01_[ism-1] ) dbe_->removeElement( mes01_[ism-1]->getName() );
    sprintf(histo, "EEPT pedestal 3sum G01 %s", Numbers::sEE(ism).c_str());
    mes01_[ism-1] = dbe_->book2D(histo, histo, 50, Numbers::ix0EE(ism)+0., Numbers::ix0EE(ism)+50., 50, Numbers::iy0EE(ism)+0., Numbers::iy0EE(ism)+50.);
    mes01_[ism-1]->setAxisTitle("jx", 1);
    mes01_[ism-1]->setAxisTitle("jy", 2);
    if ( mes02_[ism-1] ) dbe_->removeElement( mes02_[ism-1]->getName() );
    sprintf(histo, "EEPT pedestal 3sum G06 %s", Numbers::sEE(ism).c_str());
    mes02_[ism-1] = dbe_->book2D(histo, histo, 50, Numbers::ix0EE(ism)+0., Numbers::ix0EE(ism)+50., 50, Numbers::iy0EE(ism)+0., Numbers::iy0EE(ism)+50.);
    mes02_[ism-1]->setAxisTitle("jx", 1);
    mes02_[ism-1]->setAxisTitle("jy", 2);
    if ( mes03_[ism-1] ) dbe_->removeElement( mes03_[ism-1]->getName() );
    sprintf(histo, "EEPT pedestal 3sum G12 %s", Numbers::sEE(ism).c_str());
    mes03_[ism-1] = dbe_->book2D(histo, histo, 50, Numbers::ix0EE(ism)+0., Numbers::ix0EE(ism)+50., 50, Numbers::iy0EE(ism)+0., Numbers::iy0EE(ism)+50.);
    mes03_[ism-1]->setAxisTitle("jx", 1);
    mes03_[ism-1]->setAxisTitle("jy", 2);

    if ( met01_[ism-1] ) dbe_->removeElement( met01_[ism-1]->getName() );
    sprintf(histo, "EEPT pedestal 5sum G01 %s", Numbers::sEE(ism).c_str());
    met01_[ism-1] = dbe_->book2D(histo, histo, 50, Numbers::ix0EE(ism)+0., Numbers::ix0EE(ism)+50., 50, Numbers::iy0EE(ism)+0., Numbers::iy0EE(ism)+50.);
    met01_[ism-1]->setAxisTitle("jx", 1);
    met01_[ism-1]->setAxisTitle("jy", 2);
    if ( met02_[ism-1] ) dbe_->removeElement( met02_[ism-1]->getName() );
    sprintf(histo, "EEPT pedestal 5sum G06 %s", Numbers::sEE(ism).c_str());
    met02_[ism-1] = dbe_->book2D(histo, histo, 50, Numbers::ix0EE(ism)+0., Numbers::ix0EE(ism)+50., 50, Numbers::iy0EE(ism)+0., Numbers::iy0EE(ism)+50.);
    met02_[ism-1]->setAxisTitle("jx", 1);
    met02_[ism-1]->setAxisTitle("jy", 2);
    if ( met03_[ism-1] ) dbe_->removeElement( met03_[ism-1]->getName() );
    sprintf(histo, "EEPT pedestal 5sum G12 %s", Numbers::sEE(ism).c_str());
    met03_[ism-1] = dbe_->book2D(histo, histo, 50, Numbers::ix0EE(ism)+0., Numbers::ix0EE(ism)+50., 50, Numbers::iy0EE(ism)+0., Numbers::iy0EE(ism)+50.);
    met03_[ism-1]->setAxisTitle("jx", 1);
    met03_[ism-1]->setAxisTitle("jy", 2);

  }

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( meg01_[ism-1] ) meg01_[ism-1]->Reset();
    if ( meg02_[ism-1] ) meg02_[ism-1]->Reset();
    if ( meg03_[ism-1] ) meg03_[ism-1]->Reset();

    if ( meg04_[ism-1] ) meg04_[ism-1]->Reset();
    if ( meg05_[ism-1] ) meg05_[ism-1]->Reset();

    for ( int ix = 1; ix <= 50; ix++ ) {
      for ( int iy = 1; iy <= 50; iy++ ) {

        if ( meg01_[ism-1] ) meg01_[ism-1]->setBinContent( ix, iy, -1. );
        if ( meg02_[ism-1] ) meg02_[ism-1]->setBinContent( ix, iy, -1. );
        if ( meg03_[ism-1] ) meg03_[ism-1]->setBinContent( ix, iy, -1. );

        int jx = ix + Numbers::ix0EE(ism);
        int jy = iy + Numbers::iy0EE(ism);

        if ( ism >= 1 && ism <= 9 ) jx = 101 - jx;

        if ( Numbers::validEE(ism, jx, jy) ) {
          if ( meg01_[ism-1] ) meg01_[ism-1]->setBinContent( ix, iy, 2. );
          if ( meg02_[ism-1] ) meg02_[ism-1]->setBinContent( ix, iy, 2. );
          if ( meg03_[ism-1] ) meg03_[ism-1]->setBinContent( ix, iy, 2. );
        }

      }
    }

    for ( int i = 1; i <= 10; i++ ) {

        if ( meg04_[ism-1] ) meg04_[ism-1]->setBinContent( i, 1, 2. );
        if ( meg05_[ism-1] ) meg05_[ism-1]->setBinContent( i, 1, 2. );

    }

    if ( mep01_[ism-1] ) mep01_[ism-1]->Reset();
    if ( mep02_[ism-1] ) mep02_[ism-1]->Reset();
    if ( mep03_[ism-1] ) mep03_[ism-1]->Reset();

    if ( mer01_[ism-1] ) mer01_[ism-1]->Reset();
    if ( mer02_[ism-1] ) mer02_[ism-1]->Reset();
    if ( mer03_[ism-1] ) mer03_[ism-1]->Reset();

    if ( mer04_[ism-1] ) mer04_[ism-1]->Reset();
    if ( mer05_[ism-1] ) mer05_[ism-1]->Reset();

    if ( mes01_[ism-1] ) mes01_[ism-1]->Reset();
    if ( mes02_[ism-1] ) mes02_[ism-1]->Reset();
    if ( mes03_[ism-1] ) mes03_[ism-1]->Reset();

    if ( met01_[ism-1] ) met01_[ism-1]->Reset();
    if ( met02_[ism-1] ) met02_[ism-1]->Reset();
    if ( met03_[ism-1] ) met03_[ism-1]->Reset();

    for ( int ix = 1; ix <= 50; ix++ ) {
      for ( int iy = 1; iy <= 50; iy++ ) {

        if ( mes01_[ism-1] ) mes01_[ism-1]->setBinContent( ix, iy, -999. );
        if ( mes02_[ism-1] ) mes02_[ism-1]->setBinContent( ix, iy, -999. );
        if ( mes03_[ism-1] ) mes03_[ism-1]->setBinContent( ix, iy, -999. );

        if ( met01_[ism-1] ) met01_[ism-1]->setBinContent( ix, iy, -999. );
        if ( met02_[ism-1] ) met02_[ism-1]->setBinContent( ix, iy, -999. );
        if ( met03_[ism-1] ) met03_[ism-1]->setBinContent( ix, iy, -999. );

      }
    }

  }

}

void EEPedestalClient::cleanup(void) {

  if ( ! enableCleanup_ ) return;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( cloneME_ ) {
      if ( h01_[ism-1] ) delete h01_[ism-1];
      if ( h02_[ism-1] ) delete h02_[ism-1];
      if ( h03_[ism-1] ) delete h03_[ism-1];

      if ( j01_[ism-1] ) delete j01_[ism-1];
      if ( j02_[ism-1] ) delete j02_[ism-1];
      if ( j03_[ism-1] ) delete j03_[ism-1];

      if ( k01_[ism-1] ) delete k01_[ism-1];
      if ( k02_[ism-1] ) delete k02_[ism-1];
      if ( k03_[ism-1] ) delete k03_[ism-1];

      if ( i01_[ism-1] ) delete i01_[ism-1];
      if ( i02_[ism-1] ) delete i02_[ism-1];
    }

    h01_[ism-1] = 0;
    h02_[ism-1] = 0;
    h03_[ism-1] = 0;

    j01_[ism-1] = 0;
    j02_[ism-1] = 0;
    j03_[ism-1] = 0;

    k01_[ism-1] = 0;
    k02_[ism-1] = 0;
    k03_[ism-1] = 0;

    i01_[ism-1] = 0;
    i02_[ism-1] = 0;

  }

  dbe_->setCurrentFolder( "EcalEndcap/EEPedestalClient" );

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( meg01_[ism-1] ) dbe_->removeElement( meg01_[ism-1]->getName() );
    meg01_[ism-1] = 0;
    if ( meg02_[ism-1] ) dbe_->removeElement( meg02_[ism-1]->getName() );
    meg02_[ism-1] = 0;
    if ( meg03_[ism-1] ) dbe_->removeElement( meg03_[ism-1]->getName() );
    meg03_[ism-1] = 0;

    if ( meg04_[ism-1] ) dbe_->removeElement( meg04_[ism-1]->getName() );
    meg04_[ism-1] = 0;
    if ( meg05_[ism-1] ) dbe_->removeElement( meg05_[ism-1]->getName() );
    meg05_[ism-1] = 0;

    if ( mep01_[ism-1] ) dbe_->removeElement( mep01_[ism-1]->getName() );
    mep01_[ism-1] = 0;
    if ( mep02_[ism-1] ) dbe_->removeElement( mep02_[ism-1]->getName() );
    mep02_[ism-1] = 0;
    if ( mep03_[ism-1] ) dbe_->removeElement( mep03_[ism-1]->getName() );
    mep03_[ism-1] = 0;

    if ( mer01_[ism-1] ) dbe_->removeElement( mer01_[ism-1]->getName() );
    mer01_[ism-1] = 0;
    if ( mer02_[ism-1] ) dbe_->removeElement( mer02_[ism-1]->getName() );
    mer02_[ism-1] = 0;
    if ( mer03_[ism-1] ) dbe_->removeElement( mer03_[ism-1]->getName() );
    mer03_[ism-1] = 0;

    if ( mer04_[ism-1] ) dbe_->removeElement( mer04_[ism-1]->getName() );
    mer04_[ism-1] = 0;
    if ( mer05_[ism-1] ) dbe_->removeElement( mer05_[ism-1]->getName() );
    mer05_[ism-1] = 0;

    if ( mes01_[ism-1] ) dbe_->removeElement( mes01_[ism-1]->getName() );
    mes01_[ism-1] = 0;
    if ( mes02_[ism-1] ) dbe_->removeElement( mes02_[ism-1]->getName() );
    mes02_[ism-1] = 0;
    if ( mes03_[ism-1] ) dbe_->removeElement( mes03_[ism-1]->getName() );
    mes03_[ism-1] = 0;

    if ( met01_[ism-1] ) dbe_->removeElement( met01_[ism-1]->getName() );
    met01_[ism-1] = 0;
    if ( met02_[ism-1] ) dbe_->removeElement( met02_[ism-1]->getName() );
    met02_[ism-1] = 0;
    if ( met03_[ism-1] ) dbe_->removeElement( met03_[ism-1]->getName() );
    met03_[ism-1] = 0;

  }

}

bool EEPedestalClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov) {

  bool status = true;

  EcalLogicID ecid;

  MonPedestalsDat p;
  map<EcalLogicID, MonPedestalsDat> dataset1;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    cout << " " << Numbers::sEE(ism) << " (ism=" << ism << ")" << endl;
    cout << endl;

    UtilsClient::printBadChannels(meg01_[ism-1], h01_[ism-1]);
    UtilsClient::printBadChannels(meg02_[ism-1], h02_[ism-1]);
    UtilsClient::printBadChannels(meg03_[ism-1], h03_[ism-1]);

    for ( int ix = 1; ix <= 50; ix++ ) {
      for ( int iy = 1; iy <= 50; iy++ ) {

        int jx = ix + Numbers::ix0EE(ism);
        int jy = iy + Numbers::iy0EE(ism);

        if ( ism >= 1 && ism <= 9 ) jx = 101 - jx;

        if ( ! Numbers::validEE(ism, jx, jy) ) continue;

        bool update01;
        bool update02;
        bool update03;

        float num01, num02, num03;
        float mean01, mean02, mean03;
        float rms01, rms02, rms03;

        update01 = UtilsClient::getBinStats(h01_[ism-1], ix, iy, num01, mean01, rms01);
        update02 = UtilsClient::getBinStats(h02_[ism-1], ix, iy, num02, mean02, rms02);
        update03 = UtilsClient::getBinStats(h03_[ism-1], ix, iy, num03, mean03, rms03);

        if ( update01 || update02 || update03 ) {

          if ( Numbers::icEE(ism, jx, jy) == 1 ) {

            cout << "Preparing dataset for " << Numbers::sEE(ism) << " (ism=" << ism << ")" << endl;

            cout << "G01 (" << Numbers::ix0EE(i+1)+ix << "," << Numbers::iy0EE(i+1)+iy << ") " << num01  << " " << mean01 << " " << rms01  << endl;
            cout << "G06 (" << Numbers::ix0EE(i+1)+ix << "," << Numbers::iy0EE(i+1)+iy << ") " << num02  << " " << mean02 << " " << rms02  << endl;
            cout << "G12 (" << Numbers::ix0EE(i+1)+ix << "," << Numbers::iy0EE(i+1)+iy << ") " << num03  << " " << mean03 << " " << rms03  << endl;

            cout << endl;

          }

          p.setPedMeanG1(mean01);
          p.setPedRMSG1(rms01);

          p.setPedMeanG6(mean02);
          p.setPedRMSG6(rms02);

          p.setPedMeanG12(mean03);
          p.setPedRMSG12(rms03);

          if ( meg01_[ism-1] && int(meg01_[ism-1]->getBinContent( ix, iy )) % 3 == 1. &&
               meg02_[ism-1] && int(meg02_[ism-1]->getBinContent( ix, iy )) % 3 == 1. &&
               meg03_[ism-1] && int(meg03_[ism-1]->getBinContent( ix, iy )) % 3 == 1. ) {
            p.setTaskStatus(true);
          } else {
            p.setTaskStatus(false);
          }

          status = status && UtilsClient::getBinQual(meg01_[ism-1], ix, iy) &&
                             UtilsClient::getBinQual(meg02_[ism-1], ix, iy) &&
                             UtilsClient::getBinQual(meg03_[ism-1], ix, iy);

          int ic = Numbers::indexEE(ism, jx, jy);

          if ( ic == -1 ) continue;

          if ( econn ) {
            ecid = LogicID::getEcalLogicID("EE_crystal_number", Numbers::iSM(ism, EcalEndcap), ic);
            dataset1[ecid] = p;
          }

        }

      }
    }

  }

  if ( econn ) {
    try {
      cout << "Inserting MonPedestalsDat ..." << endl;
      if ( dataset1.size() != 0 ) econn->insertDataArraySet(&dataset1, moniov);
      cout << "done." << endl;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

  cout << endl;

  MonPNPedDat pn;
  map<EcalLogicID, MonPNPedDat> dataset2;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    cout << " " << Numbers::sEE(ism) << " (ism=" << ism << ")" << endl;
    cout << endl;

    UtilsClient::printBadChannels(meg04_[ism-1], i01_[ism-1]);
    UtilsClient::printBadChannels(meg05_[ism-1], i02_[ism-1]);

    for ( int i = 1; i <= 10; i++ ) {

      bool update01;
      bool update02;

      float num01, num02;
      float mean01, mean02;
      float rms01, rms02;

      update01 = UtilsClient::getBinStats(i01_[ism-1], i, 0, num01, mean01, rms01);
      update02 = UtilsClient::getBinStats(i02_[ism-1], i, 0, num02, mean02, rms02);

      if ( update01 || update02 ) {

        if ( i == 1 ) {

          cout << "Preparing dataset for " << Numbers::sEE(ism) << " (ism=" << ism << ")" << endl;

          cout << "PNs (" << i << ") G01 " << num01  << " " << mean01 << " " << rms01  << endl;
          cout << "PNs (" << i << ") G16 " << num01  << " " << mean01 << " " << rms01  << endl;

          cout << endl;

        }

        pn.setPedMeanG1(mean01);
        pn.setPedRMSG1(rms01);

        pn.setPedMeanG16(mean02);
        pn.setPedRMSG16(rms02);

        if ( meg04_[ism-1] && int(meg04_[ism-1]->getBinContent( i, 1 )) % 3 == 1. &&
             meg05_[ism-1] && int(meg05_[ism-1]->getBinContent( i, 1 )) % 3 == 1. ) {
          pn.setTaskStatus(true);
        } else {
          pn.setTaskStatus(false);
        }

        status = status && UtilsClient::getBinQual(meg04_[ism-1], i, 1) &&
                           UtilsClient::getBinQual(meg05_[ism-1], i, 1);

        if ( econn ) {
          ecid = LogicID::getEcalLogicID("EE_LM_PN", Numbers::iSM(ism, EcalEndcap), i-1);
          dataset2[ecid] = pn;
        }

      }

    }

  }

  if ( econn ) {
    try {
      cout << "Inserting MonPNPedDat ..." << endl;
      if ( dataset2.size() != 0 ) econn->insertDataArraySet(&dataset2, moniov);
      cout << "done." << endl;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

  return status;

}

void EEPedestalClient::analyze(void){

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( debug_ ) cout << "EEPedestalClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;
  }

  uint64_t bits01 = 0;
  bits01 |= EcalErrorDictionary::getMask("PEDESTAL_LOW_GAIN_MEAN_WARNING");
  bits01 |= EcalErrorDictionary::getMask("PEDESTAL_LOW_GAIN_RMS_WARNING");
  bits01 |= EcalErrorDictionary::getMask("PEDESTAL_LOW_GAIN_MEAN_ERROR");
  bits01 |= EcalErrorDictionary::getMask("PEDESTAL_LOW_GAIN_RMS_ERROR");

  uint64_t bits02 = 0;
  bits02 |= EcalErrorDictionary::getMask("PEDESTAL_MIDDLE_GAIN_MEAN_WARNING");
  bits02 |= EcalErrorDictionary::getMask("PEDESTAL_MIDDLE_GAIN_RMS_WARNING");
  bits02 |= EcalErrorDictionary::getMask("PEDESTAL_MIDDLE_GAIN_MEAN_ERROR");
  bits02 |= EcalErrorDictionary::getMask("PEDESTAL_MIDDLE_GAIN_RMS_ERROR");

  uint64_t bits03 = 0;
  bits03 |= EcalErrorDictionary::getMask("PEDESTAL_HIGH_GAIN_MEAN_WARNING");
  bits03 |= EcalErrorDictionary::getMask("PEDESTAL_HIGH_GAIN_RMS_WARNING");
  bits03 |= EcalErrorDictionary::getMask("PEDESTAL_HIGH_GAIN_MEAN_ERROR");
  bits03 |= EcalErrorDictionary::getMask("PEDESTAL_HIGH_GAIN_RMS_ERROR");

  map<EcalLogicID, RunCrystalErrorsDat> mask1;
  map<EcalLogicID, RunPNErrorsDat> mask2;

  EcalErrorMask::fetchDataSet(&mask1);
  EcalErrorMask::fetchDataSet(&mask2);

  char histo[200];

  MonitorElement* me;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    sprintf(histo, "EcalEndcap/EEPedestalTask/Gain01/EEPT pedestal %s G01", Numbers::sEE(ism).c_str());
    me = dbe_->get(histo);
    h01_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h01_[ism-1] );

    sprintf(histo, "EcalEndcap/EEPedestalTask/Gain06/EEPT pedestal %s G06", Numbers::sEE(ism).c_str());
    me = dbe_->get(histo);
    h02_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h02_[ism-1] );

    sprintf(histo, "EcalEndcap/EEPedestalTask/Gain12/EEPT pedestal %s G12", Numbers::sEE(ism).c_str());
    me = dbe_->get(histo);
    h03_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h03_[ism-1] );

    sprintf(histo, "EcalEndcap/EEPedestalTask/Gain01/EEPT pedestal 3sum %s G01", Numbers::sEE(ism).c_str());
    me = dbe_->get(histo);
    j01_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, j01_[ism-1] );

    sprintf(histo, "EcalEndcap/EEPedestalTask/Gain06/EEPT pedestal 3sum %s G06", Numbers::sEE(ism).c_str());
    me = dbe_->get(histo);
    j02_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, j02_[ism-1] );

    sprintf(histo, "EcalEndcap/EEPedestalTask/Gain12/EEPT pedestal 3sum %s G12", Numbers::sEE(ism).c_str());
    me = dbe_->get(histo);
    j03_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, j03_[ism-1] );

    sprintf(histo, "EcalEndcap/EEPedestalTask/Gain01/EEPT pedestal 5sum %s G01", Numbers::sEE(ism).c_str());
    me = dbe_->get(histo);
    k01_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, k01_[ism-1] );

    sprintf(histo, "EcalEndcap/EEPedestalTask/Gain06/EEPT pedestal 5sum %s G06", Numbers::sEE(ism).c_str());
    me = dbe_->get(histo);
    k02_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, k02_[ism-1] );

    sprintf(histo, "EcalEndcap/EEPedestalTask/Gain12/EEPT pedestal 5sum %s G12", Numbers::sEE(ism).c_str());
    me = dbe_->get(histo);
    k03_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, k03_[ism-1] );

    sprintf(histo, "EcalEndcap/EEPedestalTask/PN/Gain01/EEPDT PNs pedestal %s G01", Numbers::sEE(ism).c_str());
    me = dbe_->get(histo);
    i01_[ism-1] = UtilsClient::getHisto<TProfile*>( me, cloneME_, i01_[ism-1] );

    sprintf(histo, "EcalEndcap/EEPedestalTask/PN/Gain16/EEPDT PNs pedestal %s G16", Numbers::sEE(ism).c_str());
    me = dbe_->get(histo);
    i02_[ism-1] = UtilsClient::getHisto<TProfile*>( me, cloneME_, i02_[ism-1] );

    if ( meg01_[ism-1] ) meg01_[ism-1]->Reset();
    if ( meg02_[ism-1] ) meg02_[ism-1]->Reset();
    if ( meg03_[ism-1] ) meg03_[ism-1]->Reset();

    if ( meg04_[ism-1] ) meg04_[ism-1]->Reset();
    if ( meg05_[ism-1] ) meg05_[ism-1]->Reset();

    if ( mep01_[ism-1] ) mep01_[ism-1]->Reset();
    if ( mep02_[ism-1] ) mep02_[ism-1]->Reset();
    if ( mep03_[ism-1] ) mep03_[ism-1]->Reset();

    if ( mer01_[ism-1] ) mer01_[ism-1]->Reset();
    if ( mer02_[ism-1] ) mer02_[ism-1]->Reset();
    if ( mer03_[ism-1] ) mer03_[ism-1]->Reset();

    if ( mer04_[ism-1] ) mer04_[ism-1]->Reset();
    if ( mer05_[ism-1] ) mer05_[ism-1]->Reset();

    if ( mes01_[ism-1] ) mes01_[ism-1]->Reset();
    if ( mes02_[ism-1] ) mes02_[ism-1]->Reset();
    if ( mes03_[ism-1] ) mes03_[ism-1]->Reset();

    if ( met01_[ism-1] ) met01_[ism-1]->Reset();
    if ( met02_[ism-1] ) met02_[ism-1]->Reset();
    if ( met03_[ism-1] ) met03_[ism-1]->Reset();

    for ( int ix = 1; ix <= 50; ix++ ) {
      for ( int iy = 1; iy <= 50; iy++ ) {

        if ( meg01_[ism-1] ) meg01_[ism-1]->setBinContent(ix, iy, -1.);
        if ( meg02_[ism-1] ) meg02_[ism-1]->setBinContent(ix, iy, -1.);
        if ( meg03_[ism-1] ) meg03_[ism-1]->setBinContent(ix, iy, -1.);

        int jx = ix + Numbers::ix0EE(ism);
        int jy = iy + Numbers::iy0EE(ism);

        if ( ism >= 1 && ism <= 9 ) jx = 101 - jx;

        if ( Numbers::validEE(ism, jx, jy) ) {
          if ( meg01_[ism-1] ) meg01_[ism-1]->setBinContent(ix, iy, 2.);
          if ( meg02_[ism-1] ) meg02_[ism-1]->setBinContent(ix, iy, 2.);
          if ( meg03_[ism-1] ) meg03_[ism-1]->setBinContent(ix, iy, 2.);
        }

        bool update01;
        bool update02;
        bool update03;

        float num01, num02, num03;
        float mean01, mean02, mean03;
        float rms01, rms02, rms03;

        update01 = UtilsClient::getBinStats(h01_[ism-1], ix, iy, num01, mean01, rms01);
        update02 = UtilsClient::getBinStats(h02_[ism-1], ix, iy, num02, mean02, rms02);
        update03 = UtilsClient::getBinStats(h03_[ism-1], ix, iy, num03, mean03, rms03);

        if ( update01 ) {

          float val;

          val = 1.;
          if ( fabs(mean01 - expectedMean_[0]) > discrepancyMean_[0] )
            val = 0.;
          if ( rms01 > RMSThreshold_[0] )
            val = 0.;
          if ( meg01_[ism-1] ) meg01_[ism-1]->setBinContent(ix, iy, val);

          if ( mep01_[ism-1] ) mep01_[ism-1]->Fill(mean01);
          if ( mer01_[ism-1] ) mer01_[ism-1]->Fill(rms01);

        }

        if ( update02 ) {

          float val;

          val = 1.;
          if ( fabs(mean02 - expectedMean_[1]) > discrepancyMean_[1] )
            val = 0.;
          if ( rms02 > RMSThreshold_[1] )
            val = 0.;
          if ( meg02_[ism-1] ) meg02_[ism-1]->setBinContent(ix, iy, val);

          if ( mep02_[ism-1] ) mep02_[ism-1]->Fill(mean02);
          if ( mer02_[ism-1] ) mer02_[ism-1]->Fill(rms02);

        }

        if ( update03 ) {

          float val;

          val = 1.;
          if ( fabs(mean03 - expectedMean_[2]) > discrepancyMean_[2] )
            val = 0.;
          if ( rms03 > RMSThreshold_[2] )
            val = 0.;
          if ( meg03_[ism-1] ) meg03_[ism-1]->setBinContent(ix, iy, val);

          if ( mep03_[ism-1] ) mep03_[ism-1]->Fill(mean03);
          if ( mer03_[ism-1] ) mer03_[ism-1]->Fill(rms03);

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
              if ( (m->second).getErrorBits() & bits02 ) {
                if ( meg02_[ism-1] ) {
                  float val = int(meg02_[ism-1]->getBinContent(ix, iy)) % 3;
                  meg02_[ism-1]->setBinContent( ix, iy, val+3 );
                }
              }
              if ( (m->second).getErrorBits() & bits03 ) {
                if ( meg03_[ism-1] ) {
                  float val = int(meg03_[ism-1]->getBinContent(ix, iy)) % 3;
                  meg03_[ism-1]->setBinContent( ix, iy, val+3 );
                }
              }
            }

          }
        }

      }
    }

    // PN diodes

    for ( int i = 1; i <= 10; i++ ) {

      if ( meg04_[ism-1] ) meg04_[ism-1]->setBinContent( i, 1, 2. );
      if ( meg05_[ism-1] ) meg05_[ism-1]->setBinContent( i, 1, 2. );

      bool update01;
      bool update02;

      float num01, num02;
      float mean01, mean02;
      float rms01, rms02;

      update01 = UtilsClient::getBinStats(i01_[ism-1], i, 0, num01, mean01, rms01);
      update02 = UtilsClient::getBinStats(i02_[ism-1], i, 0, num02, mean02, rms02);

      // filling projections
      if ( mer04_[ism-1] )  mer04_[ism-1]->Fill(rms01);
      if ( mer05_[ism-1] )  mer05_[ism-1]->Fill(rms02);

      if ( update01 ) {

        float val;

        val = 1.;
        if ( mean01 < (expectedMeanPn_[0] - discrepancyMeanPn_[0])
             || (expectedMeanPn_[0] + discrepancyMeanPn_[0]) <  mean01)
           val = 0.;
        if ( rms01 >  RMSThresholdPn_[0])
          val = 0.;

        if ( meg04_[ism-1] ) meg04_[ism-1]->setBinContent(i, 1, val);

      }

      if ( update02 ) {

        float val;

        val = 1.;
        //        if ( mean02 < pedestalThresholdPn_ )
         if ( mean02 < (expectedMeanPn_[1] - discrepancyMeanPn_[1])
              || (expectedMeanPn_[1] + discrepancyMeanPn_[1]) <  mean02)
           val = 0.;
         if ( rms02 >  RMSThresholdPn_[1])
           val = 0.;

        if ( meg05_[ism-1] ) meg05_[ism-1]->setBinContent(i, 1, val);
      }

      // masking

      if ( mask2.size() != 0 ) {
        map<EcalLogicID, RunPNErrorsDat>::const_iterator m;
        for (m = mask2.begin(); m != mask2.end(); m++) {

          EcalLogicID ecid = m->first;

          if ( ecid.getLogicID() == LogicID::getEcalLogicID("EE_LM_PN", Numbers::iSM(ism, EcalEndcap), i-1).getLogicID() ) {
            if ( (m->second).getErrorBits() & bits01 ) {
              if ( meg04_[ism-1] ) {
                float val = int(meg04_[ism-1]->getBinContent(i, 1)) % 3;
                meg04_[ism-1]->setBinContent( i, 1, val+3 );
              }
            }
            if ( (m->second).getErrorBits() & bits03 ) {
              if ( meg05_[ism-1] ) {
                float val = int(meg05_[ism-1]->getBinContent(i, 1)) % 3;
                meg05_[ism-1]->setBinContent( i, 1, val+3 );
              }
            }
          }

        }
      }

    }

    for ( int ix = 1; ix <= 50; ix++ ) {
      for ( int iy = 1; iy <= 50; iy++ ) {

        float x3val01;
        float x3val02;
        float x3val03;

        float y3val01;
        float y3val02;
        float y3val03;

        float z3val01;
        float z3val02;
        float z3val03;

        float x5val01;
        float x5val02;
        float x5val03;

        float y5val01;
        float y5val02;
        float y5val03;

        float z5val01;
        float z5val02;
        float z5val03;

        if ( mes01_[ism-1] ) mes01_[ism-1]->setBinContent(ix, iy, -999.);
        if ( mes02_[ism-1] ) mes02_[ism-1]->setBinContent(ix, iy, -999.);
        if ( mes03_[ism-1] ) mes03_[ism-1]->setBinContent(ix, iy, -999.);

        if ( met01_[ism-1] ) met01_[ism-1]->setBinContent(ix, iy, -999.);
        if ( met02_[ism-1] ) met02_[ism-1]->setBinContent(ix, iy, -999.);
        if ( met03_[ism-1] ) met03_[ism-1]->setBinContent(ix, iy, -999.);

        int jx = ix + Numbers::ix0EE(ism);
        int jy = iy + Numbers::iy0EE(ism);

        if ( ism >= 1 && ism <= 9 ) jx = 101 - jx;

        if ( ! Numbers::validEE(ism, jx, jy) ) continue;

        if ( ix >= 2 && ix <= 49 && iy >= 2 && iy <= 49 ) {

          x3val01 = 0.;
          x3val02 = 0.;
          x3val03 = 0.;
          for ( int i = -1; i <= +1; i++ ) {
            for ( int j = -1; j <= +1; j++ ) {

              if ( h01_[ism-1] ) x3val01 = x3val01 + h01_[ism-1]->GetBinError(ix+i, iy+j) *
                                                     h01_[ism-1]->GetBinError(ix+i, iy+j);

              if ( h02_[ism-1] ) x3val02 = x3val02 + h02_[ism-1]->GetBinError(ix+i, iy+j) *
                                                     h02_[ism-1]->GetBinError(ix+i, iy+j);

              if ( h03_[ism-1] ) x3val03 = x3val03 + h03_[ism-1]->GetBinError(ix+i, iy+j) *
                                                     h03_[ism-1]->GetBinError(ix+i, iy+j);

            }
          }
          x3val01 = x3val01 / (9.*9.);
          x3val02 = x3val02 / (9.*9.);
          x3val03 = x3val03 / (9.*9.);

          y3val01 = 0.;
          if ( j01_[ism-1] ) y3val01 = j01_[ism-1]->GetBinError(ix, iy) *
                                       j01_[ism-1]->GetBinError(ix, iy);

          y3val02 = 0.;
          if ( j02_[ism-1] ) y3val02 = j02_[ism-1]->GetBinError(ix, iy) *
                                       j02_[ism-1]->GetBinError(ix, iy);

          y3val03 = 0.;
          if ( j03_[ism-1] ) y3val03 = j03_[ism-1]->GetBinError(ix, iy) *
                                       j03_[ism-1]->GetBinError(ix, iy);

          z3val01 = -999.;
          if ( x3val01 != 0 && y3val01 != 0 ) z3val01 = sqrt(fabs(x3val01 - y3val01));
          if ( (x3val01 - y3val01) < 0 ) z3val01 = -z3val01;

          if ( mes01_[ism-1] ) mes01_[ism-1]->setBinContent(ix, iy, z3val01);

          z3val02 = -999.;
          if ( x3val02 != 0 && y3val02 != 0 ) z3val02 = sqrt(fabs(x3val02 - y3val02));
          if ( (x3val02 - y3val02) < 0 ) z3val02 = -z3val02;

          if ( mes02_[ism-1] ) mes02_[ism-1]->setBinContent(ix, iy, z3val02);

          z3val03 = -999.;
          if ( x3val03 != 0 && y3val03 != 0 ) z3val03 = sqrt(fabs(x3val03 - y3val03));
          if ( (x3val03 - y3val03) < 0 ) z3val03 = -z3val03;

          if ( mes03_[ism-1] ) mes03_[ism-1]->setBinContent(ix, iy, z3val03);

        }

        if ( ix >= 3 && ix <= 48 && iy >= 3 && iy <= 48 ) {

          x5val01 = 0.;
          x5val02 = 0.;
          x5val03 = 0.;
          for ( int i = -2; i <= +2; i++ ) {
            for ( int j = -2; j <= +2; j++ ) {

              if ( h01_[ism-1] ) x5val01 = x5val01 + h01_[ism-1]->GetBinError(ix+i, iy+j) *
                                                     h01_[ism-1]->GetBinError(ix+i, iy+j);

              if ( h02_[ism-1] ) x5val02 = x5val02 + h02_[ism-1]->GetBinError(ix+i, iy+j) *
                                                     h02_[ism-1]->GetBinError(ix+i, iy+j);

              if ( h03_[ism-1] ) x5val03 = x5val03 + h03_[ism-1]->GetBinError(ix+i, iy+j) *
                                                     h03_[ism-1]->GetBinError(ix+i, iy+j);

            }
          }
          x5val01 = x5val01 / (25.*25.);
          x5val02 = x5val02 / (25.*25.);
          x5val03 = x5val03 / (25.*25.);

          y5val01 = 0.;
          if ( k01_[ism-1] ) y5val01 = k01_[ism-1]->GetBinError(ix, iy) *
                                       k01_[ism-1]->GetBinError(ix, iy);

          y5val02 = 0.;
          if ( k02_[ism-1] ) y5val02 = k02_[ism-1]->GetBinError(ix, iy) *
                                       k02_[ism-1]->GetBinError(ix, iy);

          y5val03 = 0.;
          if ( k03_[ism-1] ) y5val03 = k03_[ism-1]->GetBinError(ix, iy) *
                                       k03_[ism-1]->GetBinError(ix, iy);

          z5val01 = -999.;
          if ( x5val01 != 0 && y5val01 != 0 ) z5val01 = sqrt(fabs(x5val01 - y5val01));
          if ( (x5val01 - y5val01) < 0 ) z5val01 = -z5val01;

          if ( met01_[ism-1] ) met01_[ism-1]->setBinContent(ix, iy, z5val01);

          z5val02 = -999.;
          if ( x5val02 != 0 && y5val02 != 0 ) z5val02 = sqrt(fabs(x5val02 - y5val02));
          if ( (x5val02 - y5val02) < 0 ) z5val02 = -z5val02;

          if ( met02_[ism-1] ) met02_[ism-1]->setBinContent(ix, iy, z5val02);

          z5val03 = -999.;
          if ( x5val03 != 0 && y5val03 != 0 ) z5val03 = sqrt(fabs(x5val03 - y5val03));
          if ( (x5val03 - y5val03) < 0 ) z5val03 = -z5val03;

          if ( met03_[ism-1] ) met03_[ism-1]->setBinContent(ix, iy, z5val03);

        }

      }
    }

  }

}

void EEPedestalClient::htmlOutput(int run, string& htmlDir, string& htmlName){

  cout << "Preparing EEPedestalClient html output ..." << endl;

  ofstream htmlFile;

  htmlFile.open((htmlDir + htmlName).c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlFile << "<html>  " << endl;
  htmlFile << "<head>  " << endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlFile << " http-equiv=\"content-type\">  " << endl;
  htmlFile << "  <title>Monitor:PedestalTask output</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlFile << "<body>  " << endl;
  //htmlFile << "<br>  " << endl;
  htmlFile << "<a name=""top""></a>" << endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << run << "</span></h2>" << endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">PEDESTAL</span></h2> " << endl;
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

  // Produce the plots to be shown as .png files from existing histograms

  const int csize = 250;

  const double histMax = 1.e15;

  int pCol3[6] = { 301, 302, 303, 304, 305, 306 };

  int pCol4[10];
  for ( int i = 0; i < 10; i++ ) pCol4[i] = 401+i;

  TH2S labelGrid("labelGrid","label grid", 100, -2., 98., 100, -2., 98.);
  for ( short j=0; j<400; j++ ) {
    int x = 5*(1 + j%20);
    int y = 5*(1 + j/20);
    labelGrid.SetBinContent(x, y, Numbers::inTowersEE[j]);
  }
  labelGrid.SetMarkerSize(1);
  labelGrid.SetMinimum(0.1);

  TH2C dummy1( "dummy1", "dummy1 for sm mem", 10, 0, 10, 5, 0, 5 );
  for ( short i=0; i<2; i++ ) {
    int a = 2 + i*5;
    int b = 2;
    dummy1.Fill( a, b, i+1+68 );
  }
  dummy1.SetMarkerSize(2);
  dummy1.SetMinimum(0.1);

  string imgNameQual[3], imgNameMean[3], imgNameRMS[3], imgName3Sum[3], imgName5Sum[3], imgNameMEPnQual[2], imgNameMEPnPed[2],imgNameMEPnPedRms[2], imgName, meName;

  TCanvas* cQual = new TCanvas("cQual", "Temp", 2*csize, 2*csize);
  TCanvas* cQualPN = new TCanvas("cQualPN", "Temp", 2*csize, csize);
  TCanvas* cMean = new TCanvas("cMean", "Temp", csize, csize);
  TCanvas* cRMS = new TCanvas("cRMS", "Temp", csize, csize);
  TCanvas* c3Sum = new TCanvas("c3Sum", "Temp", 2*csize, 2*csize);
  TCanvas* c5Sum = new TCanvas("c5Sum", "Temp", 2*csize, 2*csize);
  TCanvas* cPed = new TCanvas("cPed", "Temp", csize, csize);

  TH2F* obj2f;
  TH1F* obj1f;
  TProfile* objp;

  // Loop on endcap sectors

  for ( unsigned int i=0; i<superModules_.size(); i ++ ) {

    int ism = superModules_[i];

    // Loop on gains

    for ( int iCanvas = 1 ; iCanvas <= 3 ; iCanvas++ ) {

      // Quality plots

      imgNameQual[iCanvas-1] = "";

      obj2f = 0;
      switch ( iCanvas ) {
        case 1:
          obj2f = UtilsClient::getHisto<TH2F*>( meg01_[ism-1] );
          break;
        case 2:
          obj2f = UtilsClient::getHisto<TH2F*>( meg02_[ism-1] );
          break;
        case 3:
          obj2f = UtilsClient::getHisto<TH2F*>( meg03_[ism-1] );
          break;
        default:
          break;
      }

      if ( obj2f ) {

        meName = obj2f->GetName();

        replace(meName.begin(), meName.end(), ' ', '_');
        imgNameQual[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgNameQual[iCanvas-1];

        cQual->cd();
        gStyle->SetOptStat(" ");
        gStyle->SetPalette(6, pCol3);
        cQual->SetGridx();
        cQual->SetGridy();
        obj2f->GetXaxis()->SetLabelSize(0.02);
        obj2f->GetXaxis()->SetTitleSize(0.02);
        obj2f->GetYaxis()->SetLabelSize(0.02);
        obj2f->GetYaxis()->SetTitleSize(0.02);
        obj2f->SetMinimum(-0.00000001);
        obj2f->SetMaximum(6.0);
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

      // Mean distributions

      imgNameMean[iCanvas-1] = "";

      obj1f = 0;
      switch ( iCanvas ) {
        case 1:
          obj1f = UtilsClient::getHisto<TH1F*>( mep01_[ism-1] );
          break;
        case 2:
          obj1f = UtilsClient::getHisto<TH1F*>( mep02_[ism-1] );
          break;
        case 3:
          obj1f = UtilsClient::getHisto<TH1F*>( mep03_[ism-1] );
          break;
        default:
            break;
      }

      if ( obj1f ) {

        meName = obj1f->GetName();

        replace(meName.begin(), meName.end(), ' ', '_');
        imgNameMean[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgNameMean[iCanvas-1];

        cMean->cd();
        gStyle->SetOptStat("euomr");
        obj1f->SetStats(kTRUE);
        if ( obj1f->GetMaximum(histMax) > 0. ) {
          gPad->SetLogy(kTRUE);
        } else {
          gPad->SetLogy(kFALSE);
        }
        obj1f->Draw();
        cMean->Update();
        cMean->SaveAs(imgName.c_str());
        gPad->SetLogy(kFALSE);

      }

      // RMS distributions

      imgNameRMS[iCanvas-1] = "";

      obj1f = 0;
      switch ( iCanvas ) {
        case 1:
          obj1f = UtilsClient::getHisto<TH1F*>( mer01_[ism-1] );
          break;
        case 2:
          obj1f = UtilsClient::getHisto<TH1F*>( mer02_[ism-1] );
          break;
        case 3:
          obj1f = UtilsClient::getHisto<TH1F*>( mer03_[ism-1] );
          break;
        default:
          break;
      }

      if ( obj1f ) {

        meName = obj1f->GetName();

        replace(meName.begin(), meName.end(), ' ', '_');
        imgNameRMS[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgNameRMS[iCanvas-1];

        cRMS->cd();
        gStyle->SetOptStat("euomr");
        obj1f->SetStats(kTRUE);
        if ( obj1f->GetMaximum(histMax) > 0. ) {
          gPad->SetLogy(kTRUE);
        } else {
          gPad->SetLogy(kFALSE);
        }
        obj1f->Draw();
        cRMS->Update();
        cRMS->SaveAs(imgName.c_str());
        gPad->SetLogy(kFALSE);

      }

      // 3Sum distributions

      imgName3Sum[iCanvas-1] = "";

      obj2f = 0;
      switch ( iCanvas ) {
        case 1:
          obj2f = UtilsClient::getHisto<TH2F*>( mes01_[ism-1] );
          break;
        case 2:
          obj2f = UtilsClient::getHisto<TH2F*>( mes02_[ism-1] );
          break;
        case 3:
          obj2f = UtilsClient::getHisto<TH2F*>( mes03_[ism-1] );
          break;
        default:
          break;
      }

      if ( obj2f ) {

        meName = obj2f->GetName();

        replace(meName.begin(), meName.end(), ' ', '_');
        imgName3Sum[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgName3Sum[iCanvas-1];

        c3Sum->cd();
        gStyle->SetOptStat(" ");
        gStyle->SetPalette(10, pCol4);
        c3Sum->SetGridx();
        c3Sum->SetGridy();
        obj2f->GetXaxis()->SetLabelSize(0.02);
        obj2f->GetXaxis()->SetTitleSize(0.02);
        obj2f->GetYaxis()->SetLabelSize(0.02);
        obj2f->GetYaxis()->SetTitleSize(0.02);
        obj2f->GetZaxis()->SetLabelSize(0.02);
        obj2f->SetMinimum(-0.5);
        obj2f->SetMaximum(+0.5);
        obj2f->Draw("colz");
        int x1 = labelGrid.GetXaxis()->FindFixBin(Numbers::ix0EE(ism)+0.);
        int x2 = labelGrid.GetXaxis()->FindFixBin(Numbers::ix0EE(ism)+50.);
        int y1 = labelGrid.GetYaxis()->FindFixBin(Numbers::iy0EE(ism)+0.);
        int y2 = labelGrid.GetYaxis()->FindFixBin(Numbers::iy0EE(ism)+50.);
        labelGrid.GetXaxis()->SetRange(x1, x2);
        labelGrid.GetYaxis()->SetRange(y1, y2);
        labelGrid.Draw("text,same");
        c3Sum->SetBit(TGraph::kClipFrame);
        TLine l;
        l.SetLineWidth(1);
        for ( int i=0; i<201; i=i+1){
          if ( (Numbers::ixSectorsEE[i]!=0 || Numbers::iySectorsEE[i]!=0) && (Numbers::ixSectorsEE[i+1]!=0 || Numbers::iySectorsEE[i+1]!=0) ) {
            l.DrawLine(Numbers::ixSectorsEE[i], Numbers::iySectorsEE[i], Numbers::ixSectorsEE[i+1], Numbers::iySectorsEE[i+1]);
          }
        }
        c3Sum->Update();
        c3Sum->SaveAs(imgName.c_str());

      }

      // 5Sum distributions

      imgName5Sum[iCanvas-1] = "";

      obj2f = 0;
      switch ( iCanvas ) {
        case 1:
          obj2f = UtilsClient::getHisto<TH2F*>( met01_[ism-1] );
          break;
        case 2:
          obj2f = UtilsClient::getHisto<TH2F*>( met02_[ism-1] );
          break;
        case 3:
          obj2f = UtilsClient::getHisto<TH2F*>( met03_[ism-1] );
          break;
        default:
          break;
      }

      if ( obj2f ) {

        meName = obj2f->GetName();

        replace(meName.begin(), meName.end(), ' ', '_');
        imgName5Sum[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgName5Sum[iCanvas-1];

        c5Sum->cd();
        gStyle->SetOptStat(" ");
        gStyle->SetPalette(10, pCol4);
        c5Sum->SetGridx();
        c5Sum->SetGridy();
        obj2f->GetXaxis()->SetLabelSize(0.02);
        obj2f->GetXaxis()->SetTitleSize(0.02);
        obj2f->GetYaxis()->SetLabelSize(0.02);
        obj2f->GetYaxis()->SetTitleSize(0.02);
        obj2f->GetZaxis()->SetLabelSize(0.02);
        obj2f->SetMinimum(-0.5);
        obj2f->SetMaximum(+0.5);
        obj2f->Draw("colz");
        int x1 = labelGrid.GetXaxis()->FindFixBin(Numbers::ix0EE(ism)+0.);
        int x2 = labelGrid.GetXaxis()->FindFixBin(Numbers::ix0EE(ism)+50.);
        int y1 = labelGrid.GetYaxis()->FindFixBin(Numbers::iy0EE(ism)+0.);
        int y2 = labelGrid.GetYaxis()->FindFixBin(Numbers::iy0EE(ism)+50.);
        labelGrid.GetXaxis()->SetRange(x1, x2);
        labelGrid.GetYaxis()->SetRange(y1, y2);
        labelGrid.Draw("text,same");
        c5Sum->SetBit(TGraph::kClipFrame);
        TLine l;
        l.SetLineWidth(1);
        for ( int i=0; i<201; i=i+1){
          if ( (Numbers::ixSectorsEE[i]!=0 || Numbers::iySectorsEE[i]!=0) && (Numbers::ixSectorsEE[i+1]!=0 || Numbers::iySectorsEE[i+1]!=0) ) {
            l.DrawLine(Numbers::ixSectorsEE[i], Numbers::iySectorsEE[i], Numbers::ixSectorsEE[i+1], Numbers::iySectorsEE[i+1]);
          }
        }
        c5Sum->Update();
        c5Sum->SaveAs(imgName.c_str());

      }

    }

    // Loop on gains

    for ( int iCanvas = 1 ; iCanvas <= 2 ; iCanvas++ ) {

      // Monitoring elements plots

      imgNameMEPnQual[iCanvas-1] = "";

      obj2f = 0;
      switch ( iCanvas ) {
      case 1:
        obj2f = UtilsClient::getHisto<TH2F*>( meg04_[ism-1] );
        break;
      case 2:
        obj2f = UtilsClient::getHisto<TH2F*>( meg05_[ism-1] );
        break;
      default:
        break;
      }

      if ( obj2f ) {

        meName = obj2f->GetName();

        replace(meName.begin(), meName.end(), ' ', '_');
        imgNameMEPnQual[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgNameMEPnQual[iCanvas-1];

        cQualPN->cd();
        gStyle->SetOptStat(" ");
        gStyle->SetPalette(6, pCol3);
        obj2f->GetXaxis()->SetNdivisions(10);
        obj2f->GetYaxis()->SetNdivisions(5);
        cQualPN->SetGridx();
        cQualPN->SetGridy(0);
        obj2f->SetMinimum(-0.00000001);
        obj2f->SetMaximum(6.0);
        obj2f->Draw("col");
        dummy1.Draw("text,same");
        cQualPN->Update();
        cQualPN->SaveAs(imgName.c_str());

      }

      imgNameMEPnPed[iCanvas-1] = "";

      objp = 0;
      switch ( iCanvas ) {
        case 1:
          objp = i01_[ism-1];
          break;
        case 2:
          objp = i02_[ism-1];
          break;
        default:
          break;
      }

      if ( objp ) {

        meName = objp->GetName();

        replace(meName.begin(), meName.end(), ' ', '_');
        imgNameMEPnPed[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgNameMEPnPed[iCanvas-1];

        cPed->cd();
        gStyle->SetOptStat("euo");
        objp->SetStats(kTRUE);
//        if ( objp->GetMaximum(histMax) > 0. ) {
//          gPad->SetLogy(kTRUE);
//        } else {
//          gPad->SetLogy(kFALSE);
//        }
        objp->SetMinimum(0.0);
        objp->Draw();
        cPed->Update();
        cPed->SaveAs(imgName.c_str());
        gPad->SetLogy(kFALSE);

      }

      imgNameMEPnPedRms[iCanvas-1] = "";

      obj1f = 0;
      switch ( iCanvas ) {
      case 1:
        if ( mer04_[ism-1] ) obj1f =  UtilsClient::getHisto<TH1F*>(mer04_[ism-1]);
        break;
      case 2:
        if ( mer05_[ism-1] ) obj1f =  UtilsClient::getHisto<TH1F*>(mer05_[ism-1]);
        break;
      default:
        break;
      }

      if ( obj1f ) {

        meName = obj1f->GetName();

        replace(meName.begin(), meName.end(), ' ', '_');
        imgNameMEPnPedRms[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgNameMEPnPedRms[iCanvas-1];

        cPed->cd();
        gStyle->SetOptStat("euomr");
        obj1f->SetStats(kTRUE);
        //        if ( obj1f->GetMaximum(histMax) > 0. ) {
        //          gPad->SetLogy(kTRUE);
        //        } else {
        //          gPad->SetLogy(kFALSE);
        //        }
        obj1f->Draw();
        cPed->Update();
        cPed->SaveAs(imgName.c_str());
        gPad->SetLogy(kFALSE);
      }

    }

    if( i>0 ) htmlFile << "<a href=""#top"">Top</a>" << std::endl;
    htmlFile << "<hr>" << std::endl;
    htmlFile << "<h3><a name="""
             << Numbers::sEE(ism) << """></a><strong>"
             << Numbers::sEE(ism) << "</strong></h3>" << endl;
    htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
    htmlFile << "<tr align=\"center\">" << endl;

    for ( int iCanvas = 1 ; iCanvas <= 3 ; iCanvas++ ) {

      if ( imgNameQual[iCanvas-1].size() != 0 )
        htmlFile << "<td colspan=\"2\"><img src=\"" << imgNameQual[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td colspan=\"2\"><img src=\"" << " " << "\"></td>" << endl;

    }

    htmlFile << "</tr>" << endl;
    htmlFile << "<tr>" << endl;

    for ( int iCanvas = 1 ; iCanvas <= 3 ; iCanvas++ ) {

      if ( imgNameMean[iCanvas-1].size() != 0 )
        htmlFile << "<td><img src=\"" << imgNameMean[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

      if ( imgNameRMS[iCanvas-1].size() != 0 )
        htmlFile << "<td><img src=\"" << imgNameRMS[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

    }

    htmlFile << "</tr>" << endl;

    htmlFile << "<tr align=\"center\"><td colspan=\"2\">Gain 1</td><td colspan=\"2\">Gain 6</td><td colspan=\"2\">Gain 12</td></tr>" << endl;
    htmlFile << "</table>" << endl;
    htmlFile << "<br>" << endl;

    htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
    htmlFile << "<tr align=\"center\">" << endl;

    for ( int iCanvas = 1 ; iCanvas <= 3 ; iCanvas++ ) {

      if ( imgName3Sum[iCanvas-1].size() != 0 )
        htmlFile << "<td><img src=\"" << imgName3Sum[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

    }

    htmlFile << "</tr>" << endl;

    htmlFile << "</table>" << endl;
    htmlFile << "<br>" << endl;

    htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
    htmlFile << "<tr align=\"center\">" << endl;

    for ( int iCanvas = 1 ; iCanvas <= 3 ; iCanvas++ ) {

      if ( imgName5Sum[iCanvas-1].size() != 0 )
        htmlFile << "<td><img src=\"" << imgName5Sum[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

    }

    htmlFile << "</tr>" << endl;

    htmlFile << "<tr align=\"center\"><td>Gain 1</td><td>Gain 6</td><td>Gain 12</td></tr>" << endl;
    htmlFile << "</table>" << endl;
    htmlFile << "<br>" << endl;

    htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\"> " << endl;
    htmlFile << "<tr align=\"center\">" << endl;

    for ( int iCanvas = 1 ; iCanvas <= 2 ; iCanvas++ ) {

      if ( imgNameMEPnQual[iCanvas-1].size() != 0 )
        htmlFile << "<td colspan=\"2\"><img src=\"" << imgNameMEPnQual[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td colspan=\"2\"><img src=\"" << " " << "\"></td>" << endl;

    }

    htmlFile << "</tr>" << endl;
    htmlFile << "</table>" << endl;
    htmlFile << "<br>" << endl;

    htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
    htmlFile << "<tr align=\"center\">" << endl;

    for ( int iCanvas = 1 ; iCanvas <= 2 ; iCanvas++ ) {

      if ( imgNameMEPnPed[iCanvas-1].size() != 0 ){
        htmlFile << "<td colspan=\"2\"><img src=\"" << imgNameMEPnPed[iCanvas-1] << "\"></td>" << endl;

        if ( imgNameMEPnPedRms[iCanvas-1].size() != 0 )
          htmlFile << "<td colspan=\"2\"><img src=\"" << imgNameMEPnPedRms[iCanvas-1] << "\"></td>" << endl;
        else
          htmlFile << "<td colspan=\"2\"><img src=\"" << " " << "\"></td>" << endl;

      }

      else{
        htmlFile << "<td colspan=\"2\"><img src=\"" << " " << "\"></td>" << endl;

        if ( imgNameMEPnPedRms[iCanvas-1].size() != 0 )
          htmlFile << "<td colspan=\"2\"><img src=\"" << imgNameMEPnPedRms[iCanvas-1] << "\"></td>" << endl;
        else
          htmlFile << "<td colspan=\"2\"><img src=\"" << " " << "\"></td>" << endl;
      }

    }

    htmlFile << "</tr>" << endl;

    htmlFile << "<tr align=\"right\"><td colspan=\"2\">Gain 1</td>  <td colspan=\"2\"> </td> <td colspan=\"2\">Gain 16</td></tr>" << endl;
    htmlFile << "</table>" << endl;
    htmlFile << "<br>" << endl;

   }

  delete cQual;
  delete cQualPN;
  delete cMean;
  delete cRMS;
  delete c3Sum;
  delete c5Sum;
  delete cPed;

  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  htmlFile.close();

  }
