/*
 * \file EELedClient.cc
 *
 * $Date: 2009/06/29 13:27:50 $
 * $Revision: 1.94 $
 * \author G. Della Ricca
 * \author G. Franzoni
 *
*/

#include <memory>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <math.h>

#include "DQMServices/Core/interface/DQMStore.h"

#include "OnlineDB/EcalCondDB/interface/MonLed1Dat.h"
#include "OnlineDB/EcalCondDB/interface/MonLed2Dat.h"
#include "OnlineDB/EcalCondDB/interface/MonPNLed1Dat.h"
#include "OnlineDB/EcalCondDB/interface/MonPNLed2Dat.h"
#include "OnlineDB/EcalCondDB/interface/MonTimingLed2CrystalDat.h"
#include "OnlineDB/EcalCondDB/interface/MonTimingLed1CrystalDat.h"
#include "OnlineDB/EcalCondDB/interface/RunCrystalErrorsDat.h"
#include "OnlineDB/EcalCondDB/interface/RunTTErrorsDat.h"
#include "OnlineDB/EcalCondDB/interface/RunPNErrorsDat.h"

#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"

#include "CondTools/Ecal/interface/EcalErrorDictionary.h"

#include "DQM/EcalCommon/interface/EcalErrorMask.h"
#include "DQM/EcalCommon/interface/UtilsClient.h"
#include "DQM/EcalCommon/interface/LogicID.h"
#include "DQM/EcalCommon/interface/Numbers.h"

#include <DQM/EcalEndcapMonitorClient/interface/EELedClient.h>

using namespace cms;
using namespace edm;
using namespace std;

EELedClient::EELedClient(const ParameterSet& ps) {

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

  ledWavelengths_.reserve(2);
  for ( unsigned int i = 1; i <= 2; i++ ) ledWavelengths_.push_back(i);
  ledWavelengths_ = ps.getUntrackedParameter<vector<int> >("ledWavelengths", ledWavelengths_);

  if ( verbose_ ) {
    cout << " Led wavelengths:" << endl;
    for ( unsigned int i = 0; i < ledWavelengths_.size(); i++ ) {
      cout << " " << ledWavelengths_[i];
    }
    cout << endl;
  }

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    h01_[ism-1] = 0;
    h02_[ism-1] = 0;
    h03_[ism-1] = 0;
    h04_[ism-1] = 0;

    h09_[ism-1] = 0;
    h10_[ism-1] = 0;

    hs01_[ism-1] = 0;
    hs02_[ism-1] = 0;

    i01_[ism-1] = 0;
    i02_[ism-1] = 0;

    i05_[ism-1] = 0;
    i06_[ism-1] = 0;

    i09_[ism-1] = 0;
    i10_[ism-1] = 0;

    i13_[ism-1] = 0;
    i14_[ism-1] = 0;

  }

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    meg01_[ism-1] = 0;
    meg02_[ism-1] = 0;

    meg05_[ism-1] = 0;
    meg06_[ism-1] = 0;

    meg09_[ism-1] = 0;
    meg10_[ism-1] = 0;

    mea01_[ism-1] = 0;
    mea02_[ism-1] = 0;

    met01_[ism-1] = 0;
    met02_[ism-1] = 0;

    metav01_[ism-1] = 0;
    metav02_[ism-1] = 0;

    metrms01_[ism-1] = 0;
    metrms02_[ism-1] = 0;

    meaopn01_[ism-1] = 0;
    meaopn02_[ism-1] = 0;

    mepnprms01_[ism-1] = 0;
    mepnprms02_[ism-1] = 0;

    mepnprms05_[ism-1] = 0;
    mepnprms06_[ism-1] = 0;

    me_hs01_[ism-1] = 0;
    me_hs02_[ism-1] = 0;

  }

  percentVariation_ = 0.4; // not used nor not normalized VPTs
  
  amplitudeThreshold_ = 100.;

  amplitudeThresholdPnG01_ = 50.;
  amplitudeThresholdPnG16_ = 50.;

  pedPnExpectedMean_[0] = 750.0;
  pedPnExpectedMean_[1] = 750.0;

  pedPnDiscrepancyMean_[0] = 100.0;
  pedPnDiscrepancyMean_[1] = 100.0;

  pedPnRMSThreshold_[0] = 1.0; // value at h4; expected nominal: 0.5
  pedPnRMSThreshold_[1] = 3.0; // value at h4; expected nominal: 1.6

}

EELedClient::~EELedClient() {

}

void EELedClient::beginJob(DQMStore* dqmStore) {

  dqmStore_ = dqmStore;

  if ( debug_ ) cout << "EELedClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EELedClient::beginRun(void) {

  if ( debug_ ) cout << "EELedClient: beginRun" << endl;

  jevt_ = 0;

  this->setup();

}

void EELedClient::endJob(void) {

  if ( debug_ ) cout << "EELedClient: endJob, ievt = " << ievt_ << endl;

  this->cleanup();

}

void EELedClient::endRun(void) {

  if ( debug_ ) cout << "EELedClient: endRun, jevt = " << jevt_ << endl;

  this->cleanup();

}

void EELedClient::setup(void) {

  char histo[200];

  dqmStore_->setCurrentFolder( prefixME_ + "/EELedClient" );

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 1) != ledWavelengths_.end() ) {
      if ( meg01_[ism-1] ) dqmStore_->removeElement( meg01_[ism-1]->getName() );
      sprintf(histo, "EELDT led quality L1 %s", Numbers::sEE(ism).c_str());
      meg01_[ism-1] = dqmStore_->book2D(histo, histo, 50, Numbers::ix0EE(ism)+0., Numbers::ix0EE(ism)+50., 50, Numbers::iy0EE(ism)+0., Numbers::iy0EE(ism)+50.);
      meg01_[ism-1]->setAxisTitle("jx", 1);
      meg01_[ism-1]->setAxisTitle("jy", 2);
    }
    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 2) != ledWavelengths_.end() ) {
      if ( meg02_[ism-1] ) dqmStore_->removeElement( meg02_[ism-1]->getName() );
      sprintf(histo, "EELDT led quality L2 %s", Numbers::sEE(ism).c_str());
      meg02_[ism-1] = dqmStore_->book2D(histo, histo, 50, Numbers::ix0EE(ism)+0., Numbers::ix0EE(ism)+50., 50, Numbers::iy0EE(ism)+0., Numbers::iy0EE(ism)+50.);
      meg02_[ism-1]->setAxisTitle("jx", 1);
      meg02_[ism-1]->setAxisTitle("jy", 2);
    }

    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 1) != ledWavelengths_.end() ) {
      if ( meg05_[ism-1] ) dqmStore_->removeElement( meg05_[ism-1]->getName() );
      sprintf(histo, "EELDT led quality L1 PNs G01 %s", Numbers::sEE(ism).c_str());
      meg05_[ism-1] = dqmStore_->book2D(histo, histo, 10, 0., 10., 1, 0., 5.);
      meg05_[ism-1]->setAxisTitle("pseudo-strip", 1);
      meg05_[ism-1]->setAxisTitle("channel", 2);
    }
    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 2) != ledWavelengths_.end() ) {
      if ( meg06_[ism-1] ) dqmStore_->removeElement( meg06_[ism-1]->getName() );
      sprintf(histo, "EELDT led quality L2 PNs G01 %s", Numbers::sEE(ism).c_str());
      meg06_[ism-1] = dqmStore_->book2D(histo, histo, 10, 0., 10., 1, 0., 5.);
      meg06_[ism-1]->setAxisTitle("pseudo-strip", 1);
      meg06_[ism-1]->setAxisTitle("channel", 2);
    }

    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 1) != ledWavelengths_.end() ) {
      if ( meg09_[ism-1] ) dqmStore_->removeElement( meg09_[ism-1]->getName() );
      sprintf(histo, "EELDT led quality L1 PNs G16 %s", Numbers::sEE(ism).c_str());
      meg09_[ism-1] = dqmStore_->book2D(histo, histo, 10, 0., 10., 1, 0., 5.);
      meg09_[ism-1]->setAxisTitle("pseudo-strip", 1);
      meg09_[ism-1]->setAxisTitle("channel", 2);
    }
    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 2) != ledWavelengths_.end() ) {
      if ( meg10_[ism-1] ) dqmStore_->removeElement( meg10_[ism-1]->getName() );
      sprintf(histo, "EELDT led quality L2 PNs G16 %s", Numbers::sEE(ism).c_str());
      meg10_[ism-1] = dqmStore_->book2D(histo, histo, 10, 0., 10., 1, 0., 5.);
      meg10_[ism-1]->setAxisTitle("pseudo-strip", 1);
      meg10_[ism-1]->setAxisTitle("channel", 2);
    }

    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 1) != ledWavelengths_.end() ) {
      if ( mea01_[ism-1] ) dqmStore_->removeElement( mea01_[ism-1]->getName() );;
      sprintf(histo, "EELDT amplitude L1 %s", Numbers::sEE(ism).c_str());
      mea01_[ism-1] = dqmStore_->book1D(histo, histo, 850, 0., 850.);
      mea01_[ism-1]->setAxisTitle("channel", 1);
      mea01_[ism-1]->setAxisTitle("amplitude", 2);
    }
    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 2) != ledWavelengths_.end() ) {
      if ( mea02_[ism-1] ) dqmStore_->removeElement( mea02_[ism-1]->getName() );
      sprintf(histo, "EELDT amplitude L2 %s", Numbers::sEE(ism).c_str());
      mea02_[ism-1] = dqmStore_->book1D(histo, histo, 850, 0., 850.);
      mea02_[ism-1]->setAxisTitle("channel", 1);
      mea02_[ism-1]->setAxisTitle("amplitude", 2);
    }

    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 1) != ledWavelengths_.end() ) {
      if ( met01_[ism-1] ) dqmStore_->removeElement( met01_[ism-1]->getName() );
      sprintf(histo, "EELDT led timing L1 %s", Numbers::sEE(ism).c_str());
      met01_[ism-1] = dqmStore_->book1D(histo, histo, 850, 0., 850.);
      met01_[ism-1]->setAxisTitle("channel", 1);
      met01_[ism-1]->setAxisTitle("jitter", 2);
    }
    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 2) != ledWavelengths_.end() ) {
      if ( met02_[ism-1] ) dqmStore_->removeElement( met02_[ism-1]->getName() );
      sprintf(histo, "EELDT led timing L2 %s", Numbers::sEE(ism).c_str());
      met02_[ism-1] = dqmStore_->book1D(histo, histo, 850, 0., 850.);
      met02_[ism-1]->setAxisTitle("channel", 1);
      met02_[ism-1]->setAxisTitle("jitter", 2);
    }

    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 1) != ledWavelengths_.end() ) {
      if ( metav01_[ism-1] ) dqmStore_->removeElement( metav01_[ism-1]->getName() );
      sprintf(histo, "EELDT led timing mean L1 %s", Numbers::sEE(ism).c_str());
      metav01_[ism-1] = dqmStore_->book1D(histo, histo, 100, 0., 10.);
      metav01_[ism-1]->setAxisTitle("mean", 1);
    }
    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 2) != ledWavelengths_.end() ) {
      if ( metav02_[ism-1] ) dqmStore_->removeElement( metav02_[ism-1]->getName() );
      sprintf(histo, "EELDT led timing mean L2 %s", Numbers::sEE(ism).c_str());
      metav02_[ism-1] = dqmStore_->book1D(histo, histo, 100, 0., 10.);
      metav02_[ism-1]->setAxisTitle("mean", 1);
    }

    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 1) != ledWavelengths_.end() ) {
      if ( metrms01_[ism-1] ) dqmStore_->removeElement( metrms01_[ism-1]->getName() );
      sprintf(histo, "EELDT led timing rms L1 %s", Numbers::sEE(ism).c_str());
      metrms01_[ism-1] = dqmStore_->book1D(histo, histo, 100, 0., 0.5);
      metrms01_[ism-1]->setAxisTitle("rms", 1);
    }
    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 2) != ledWavelengths_.end() ) {
      if ( metrms02_[ism-1] ) dqmStore_->removeElement( metrms02_[ism-1]->getName() );
      sprintf(histo, "EELDT led timing rms L2 %s", Numbers::sEE(ism).c_str());
      metrms02_[ism-1] = dqmStore_->book1D(histo, histo, 100, 0., 0.5);
      metrms02_[ism-1]->setAxisTitle("rms", 1);
    }

    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 1) != ledWavelengths_.end() ) {
      if ( meaopn01_[ism-1] ) dqmStore_->removeElement( meaopn01_[ism-1]->getName() );
      sprintf(histo, "EELDT amplitude over PN L1 %s", Numbers::sEE(ism).c_str());
      meaopn01_[ism-1] = dqmStore_->book1D(histo, histo, 850, 0., 850.);
      meaopn01_[ism-1]->setAxisTitle("channel", 1);
      meaopn01_[ism-1]->setAxisTitle("amplitude/PN", 2);
    }
    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 2) != ledWavelengths_.end() ) {
      if ( meaopn02_[ism-1] ) dqmStore_->removeElement( meaopn02_[ism-1]->getName() );
      sprintf(histo, "EELDT amplitude over PN L2 %s", Numbers::sEE(ism).c_str());
      meaopn02_[ism-1] = dqmStore_->book1D(histo, histo, 850, 0., 850.);
      meaopn02_[ism-1]->setAxisTitle("channel", 1);
      meaopn02_[ism-1]->setAxisTitle("amplitude/PN", 2);
    }

    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 1) != ledWavelengths_.end() ) {
      if ( mepnprms01_[ism-1] ) dqmStore_->removeElement( mepnprms01_[ism-1]->getName() );
      sprintf(histo, "EEPDT PNs pedestal rms %s G01 L1", Numbers::sEE(ism).c_str());
      mepnprms01_[ism-1] = dqmStore_->book1D(histo, histo, 100, 0., 10.);
      mepnprms01_[ism-1]->setAxisTitle("rms", 1);
    }
    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 2) != ledWavelengths_.end() ) {
      if ( mepnprms02_[ism-1] ) dqmStore_->removeElement( mepnprms02_[ism-1]->getName() );
      sprintf(histo, "EEPDT PNs pedestal rms %s G01 L2", Numbers::sEE(ism).c_str());
      mepnprms02_[ism-1] = dqmStore_->book1D(histo, histo, 100, 0., 10.);
      mepnprms02_[ism-1]->setAxisTitle("rms", 1);
    }

    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 1) != ledWavelengths_.end() ) {
      if ( mepnprms05_[ism-1] ) dqmStore_->removeElement( mepnprms05_[ism-1]->getName() );
      sprintf(histo, "EEPDT PNs pedestal rms %s G16 L1", Numbers::sEE(ism).c_str());
      mepnprms05_[ism-1] = dqmStore_->book1D(histo, histo, 100, 0., 10.);
      mepnprms05_[ism-1]->setAxisTitle("rms", 1);
    }
    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 2) != ledWavelengths_.end() ) {
      if ( mepnprms06_[ism-1] ) dqmStore_->removeElement( mepnprms06_[ism-1]->getName() );
      sprintf(histo, "EEPDT PNs pedestal rms %s G16 L2", Numbers::sEE(ism).c_str());
      mepnprms06_[ism-1] = dqmStore_->book1D(histo, histo, 100, 0., 10.);
      mepnprms06_[ism-1]->setAxisTitle("rms", 1);
    }

    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 1) != ledWavelengths_.end() ) {
      if ( me_hs01_[ism-1] ) dqmStore_->removeElement( me_hs01_[ism-1]->getName() );
      sprintf(histo, "EELDT led shape L1 %s", Numbers::sEE(ism).c_str());
      me_hs01_[ism-1] = dqmStore_->book1D(histo, histo, 10, 0., 10.);
      me_hs01_[ism-1]->setAxisTitle("sample", 1);
      me_hs01_[ism-1]->setAxisTitle("amplitude", 2);
    }
    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 2) != ledWavelengths_.end() ) {
      if ( me_hs02_[ism-1] ) dqmStore_->removeElement( me_hs02_[ism-1]->getName() );
      sprintf(histo, "EELDT led shape L2 %s", Numbers::sEE(ism).c_str());
      me_hs02_[ism-1] = dqmStore_->book1D(histo, histo, 10, 0., 10.);
      me_hs02_[ism-1]->setAxisTitle("sample", 1);
      me_hs02_[ism-1]->setAxisTitle("amplitude", 2);
    }

  }

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( meg01_[ism-1] ) meg01_[ism-1]->Reset();
    if ( meg02_[ism-1] ) meg02_[ism-1]->Reset();

    if ( meg05_[ism-1] ) meg05_[ism-1]->Reset();
    if ( meg06_[ism-1] ) meg06_[ism-1]->Reset();

    if ( meg09_[ism-1] ) meg09_[ism-1]->Reset();
    if ( meg10_[ism-1] ) meg10_[ism-1]->Reset();

    for ( int ix = 1; ix <= 50; ix++ ) {
      for ( int iy = 1; iy <= 50; iy++ ) {

        if ( meg01_[ism-1] ) meg01_[ism-1]->setBinContent( ix, iy, 6. );
        if ( meg02_[ism-1] ) meg02_[ism-1]->setBinContent( ix, iy, 6. );

        int jx = ix + Numbers::ix0EE(ism);
        int jy = iy + Numbers::iy0EE(ism);

        if ( ism >= 1 && ism <= 9 ) jx = 101 - jx;

        if ( Numbers::validEE(ism, jx, jy) ) {
          if ( meg01_[ism-1] ) meg01_[ism-1]->setBinContent( ix, iy, 2. );
          if ( meg02_[ism-1] ) meg02_[ism-1]->setBinContent( ix, iy, 2. );
        }

      }
    }

    for ( int i = 1; i <= 10; i++ ) {

        if ( meg05_[ism-1] ) meg05_[ism-1]->setBinContent( i, 1, 2. );
        if ( meg06_[ism-1] ) meg06_[ism-1]->setBinContent( i, 1, 2. );

        if ( meg09_[ism-1] ) meg09_[ism-1]->setBinContent( i, 1, 2. );
        if ( meg10_[ism-1] ) meg10_[ism-1]->setBinContent( i, 1, 2. );

    }

    if ( mea01_[ism-1] ) mea01_[ism-1]->Reset();
    if ( mea02_[ism-1] ) mea02_[ism-1]->Reset();

    if ( met01_[ism-1] ) met01_[ism-1]->Reset();
    if ( met02_[ism-1] ) met02_[ism-1]->Reset();

    if ( metav01_[ism-1] ) metav01_[ism-1]->Reset();
    if ( metav02_[ism-1] ) metav02_[ism-1]->Reset();

    if ( metrms01_[ism-1] ) metrms01_[ism-1]->Reset();
    if ( metrms02_[ism-1] ) metrms02_[ism-1]->Reset();

    if ( meaopn01_[ism-1] ) meaopn01_[ism-1]->Reset();
    if ( meaopn02_[ism-1] ) meaopn02_[ism-1]->Reset();

    if ( mepnprms01_[ism-1] ) mepnprms01_[ism-1]->Reset();
    if ( mepnprms02_[ism-1] ) mepnprms02_[ism-1]->Reset();

    if ( mepnprms05_[ism-1] ) mepnprms05_[ism-1]->Reset();
    if ( mepnprms06_[ism-1] ) mepnprms06_[ism-1]->Reset();

    if ( me_hs01_[ism-1] ) me_hs01_[ism-1]->Reset();
    if ( me_hs02_[ism-1] ) me_hs02_[ism-1]->Reset();

  }

}

void EELedClient::cleanup(void) {

  if ( ! enableCleanup_ ) return;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( cloneME_ ) {
      if ( h01_[ism-1] ) delete h01_[ism-1];
      if ( h02_[ism-1] ) delete h02_[ism-1];
      if ( h03_[ism-1] ) delete h03_[ism-1];
      if ( h04_[ism-1] ) delete h04_[ism-1];

      if ( h09_[ism-1] ) delete h09_[ism-1];
      if ( h10_[ism-1] ) delete h10_[ism-1];

      if ( hs01_[ism-1] ) delete hs01_[ism-1];
      if ( hs02_[ism-1] ) delete hs02_[ism-1];

      if ( i01_[ism-1] ) delete i01_[ism-1];
      if ( i02_[ism-1] ) delete i02_[ism-1];

      if ( i05_[ism-1] ) delete i05_[ism-1];
      if ( i06_[ism-1] ) delete i06_[ism-1];

      if ( i09_[ism-1] ) delete i09_[ism-1];
      if ( i10_[ism-1] ) delete i10_[ism-1];

      if ( i13_[ism-1] ) delete i13_[ism-1];
      if ( i14_[ism-1] ) delete i14_[ism-1];
    }

    h01_[ism-1] = 0;
    h02_[ism-1] = 0;
    h03_[ism-1] = 0;
    h04_[ism-1] = 0;

    h09_[ism-1] = 0;
    h10_[ism-1] = 0;

    hs01_[ism-1] = 0;
    hs02_[ism-1] = 0;

    i01_[ism-1] = 0;
    i02_[ism-1] = 0;

    i05_[ism-1] = 0;
    i06_[ism-1] = 0;

    i09_[ism-1] = 0;
    i10_[ism-1] = 0;

    i13_[ism-1] = 0;
    i14_[ism-1] = 0;

  }

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    dqmStore_->setCurrentFolder( prefixME_ + "/EELedClient" );

    if ( meg01_[ism-1] ) dqmStore_->removeElement( meg01_[ism-1]->getName() );
    meg01_[ism-1] = 0;
    if ( meg02_[ism-1] ) dqmStore_->removeElement( meg02_[ism-1]->getName() );
    meg02_[ism-1] = 0;

    if ( meg05_[ism-1] ) dqmStore_->removeElement( meg05_[ism-1]->getName() );
    meg05_[ism-1] = 0;
    if ( meg06_[ism-1] ) dqmStore_->removeElement( meg06_[ism-1]->getName() );
    meg06_[ism-1] = 0;

    if ( meg09_[ism-1] ) dqmStore_->removeElement( meg09_[ism-1]->getName() );
    meg09_[ism-1] = 0;
    if ( meg10_[ism-1] ) dqmStore_->removeElement( meg10_[ism-1]->getName() );
    meg10_[ism-1] = 0;

    if ( mea01_[ism-1] ) dqmStore_->removeElement( mea01_[ism-1]->getName() );
    mea01_[ism-1] = 0;
    if ( mea02_[ism-1] ) dqmStore_->removeElement( mea02_[ism-1]->getName() );
    mea02_[ism-1] = 0;

    if ( met01_[ism-1] ) dqmStore_->removeElement( met01_[ism-1]->getName() );
    met01_[ism-1] = 0;
    if ( met02_[ism-1] ) dqmStore_->removeElement( met02_[ism-1]->getName() );
    met02_[ism-1] = 0;

    if ( metav01_[ism-1] ) dqmStore_->removeElement( metav01_[ism-1]->getName() );
    metav01_[ism-1] = 0;
    if ( metav02_[ism-1] ) dqmStore_->removeElement( metav02_[ism-1]->getName() );
    metav02_[ism-1] = 0;

    if ( metrms01_[ism-1] ) dqmStore_->removeElement( metrms01_[ism-1]->getName() );
    metrms01_[ism-1] = 0;
    if ( metrms02_[ism-1] ) dqmStore_->removeElement( metrms02_[ism-1]->getName() );
    metrms02_[ism-1] = 0;

    if ( meaopn01_[ism-1] ) dqmStore_->removeElement( meaopn01_[ism-1]->getName() );
    meaopn01_[ism-1] = 0;
    if ( meaopn02_[ism-1] ) dqmStore_->removeElement( meaopn02_[ism-1]->getName() );
    meaopn02_[ism-1] = 0;

    if ( mepnprms01_[ism-1] ) dqmStore_->removeElement( mepnprms01_[ism-1]->getName() );
    mepnprms01_[ism-1] = 0;
    if ( mepnprms02_[ism-1] ) dqmStore_->removeElement( mepnprms02_[ism-1]->getName() );
    mepnprms02_[ism-1] = 0;

    if ( mepnprms05_[ism-1] ) dqmStore_->removeElement( mepnprms05_[ism-1]->getName() );
    mepnprms05_[ism-1] = 0;
    if ( mepnprms06_[ism-1] ) dqmStore_->removeElement( mepnprms06_[ism-1]->getName() );
    mepnprms06_[ism-1] = 0;

    if ( me_hs01_[ism-1] ) dqmStore_->removeElement( me_hs01_[ism-1]->getName() );
    me_hs01_[ism-1] = 0;
    if ( me_hs02_[ism-1] ) dqmStore_->removeElement( me_hs02_[ism-1]->getName() );
    me_hs02_[ism-1] = 0;

  }

}

bool EELedClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov, bool& status, bool flag) {

  status = true;

  if ( ! flag ) return false;

  EcalLogicID ecid;

  MonLed1Dat vpt_l1;
  map<EcalLogicID, MonLed1Dat> dataset1_l1;
  MonLed2Dat vpt_l2;
  map<EcalLogicID, MonLed2Dat> dataset1_l2;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( verbose_ ) {
      cout << " " << Numbers::sEE(ism) << " (ism=" << ism << ")" << endl;
      cout << endl;
    }

    if ( verbose_ ) {
      if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 1) != ledWavelengths_.end() ) {
        UtilsClient::printBadChannels(meg01_[ism-1], h01_[ism-1]);
      }
      if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 2) != ledWavelengths_.end() ) {
        UtilsClient::printBadChannels(meg02_[ism-1], h03_[ism-1]);
      }
    }

    for ( int ix = 1; ix <= 50; ix++ ) {
      for ( int iy = 1; iy <= 50; iy++ ) {

        int jx = ix + Numbers::ix0EE(ism);
        int jy = iy + Numbers::iy0EE(ism);

        if ( ism >= 1 && ism <= 9 ) jx = 101 - jx;

        if ( ! Numbers::validEE(ism, jx, jy) ) continue;

        bool update01;
        bool update02;
        bool update03;
        bool update04;

        float num01, num02, num03, num04;
        float mean01, mean02, mean03, mean04;
        float rms01, rms02, rms03, rms04;

        update01 = UtilsClient::getBinStatistics(h01_[ism-1], ix, iy, num01, mean01, rms01);
        update02 = UtilsClient::getBinStatistics(h02_[ism-1], ix, iy, num02, mean02, rms02);
        update03 = UtilsClient::getBinStatistics(h03_[ism-1], ix, iy, num03, mean03, rms03);
        update04 = UtilsClient::getBinStatistics(h04_[ism-1], ix, iy, num04, mean04, rms04);

        if ( update01 || update02 ) {

          if ( Numbers::icEE(ism, jx, jy) == 1 ) {

            if ( verbose_ ) {
              cout << "Preparing dataset for " << Numbers::sEE(ism) << " (ism=" << ism << ")" << endl;
              cout << "L1 (" << Numbers::ix0EE(i+1)+ix << "," << Numbers::iy0EE(i+1)+iy << ") " << num01 << " " << mean01 << " " << rms01 << endl;
              cout << endl;
            }

          }

          vpt_l1.setVPTMean(mean01);
          vpt_l1.setVPTRMS(rms01);

          vpt_l1.setVPTOverPNMean(mean02);
          vpt_l1.setVPTOverPNRMS(rms02);

          if ( UtilsClient::getBinStatus(meg01_[ism-1], ix, iy) ) {
            vpt_l1.setTaskStatus(true);
          } else {
            vpt_l1.setTaskStatus(false);
          }

          status = status && UtilsClient::getBinQuality(meg01_[ism-1], ix, iy);

          int ic = Numbers::indexEE(ism, jx, jy);

          if ( ic == -1 ) continue;

          if ( econn ) {
            ecid = LogicID::getEcalLogicID("EE_crystal_number", Numbers::iSM(ism, EcalEndcap), ic);
            dataset1_l1[ecid] = vpt_l1;
          }

        }

        if ( update03 || update04 ) {

          if ( Numbers::icEE(ism, jx, jy) == 1 ) {

            if ( verbose_ ) {
              cout << "Preparing dataset for " << Numbers::sEE(ism) << " (ism=" << ism << ")" << endl;
              cout << "L2 (" << Numbers::ix0EE(i+1)+ix << "," << Numbers::iy0EE(i+1)+iy << ") " << num03 << " " << mean03 << " " << rms03 << endl;
              cout << endl;
            }

          }

          vpt_l2.setVPTMean(mean03);
          vpt_l2.setVPTRMS(rms03);

          vpt_l2.setVPTOverPNMean(mean04);
          vpt_l2.setVPTOverPNRMS(rms04);

          if ( UtilsClient::getBinStatus(meg02_[ism-1], ix, iy) ) {
            vpt_l2.setTaskStatus(true);
          } else {
            vpt_l2.setTaskStatus(false);
          }

          status = status && UtilsClient::getBinQuality(meg02_[ism-1], ix, iy);

          int ic = Numbers::indexEE(ism, jx, jy);

          if ( ic == -1 ) continue;

          if ( econn ) {
            ecid = LogicID::getEcalLogicID("EE_crystal_number", Numbers::iSM(ism, EcalEndcap), ic);
            dataset1_l2[ecid] = vpt_l2;
          }

        }

      }
    }

  }

  if ( econn ) {
    try {
      if ( verbose_ ) cout << "Inserting MonLedDat ..." << endl;
      if ( dataset1_l1.size() != 0 ) econn->insertDataArraySet(&dataset1_l1, moniov);
      if ( dataset1_l2.size() != 0 ) econn->insertDataArraySet(&dataset1_l2, moniov);
      if ( verbose_ ) cout << "done." << endl;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

  if ( verbose_ ) cout << endl;

  MonPNLed1Dat pn_l1;
  map<EcalLogicID, MonPNLed1Dat> dataset2_l1;
  MonPNLed2Dat pn_l2;
  map<EcalLogicID, MonPNLed2Dat> dataset2_l2;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( verbose_ ) {
      cout << " " << Numbers::sEE(ism) << " (ism=" << ism << ")" << endl;
      cout << endl;
    }

    if ( verbose_ ) {
      if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 1) != ledWavelengths_.end() ) {
        UtilsClient::printBadChannels(meg05_[ism-1], i01_[ism-1]);
        UtilsClient::printBadChannels(meg05_[ism-1], i05_[ism-1]);
      }
      if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 2) != ledWavelengths_.end() ) {
        UtilsClient::printBadChannels(meg06_[ism-1], i02_[ism-1]);
        UtilsClient::printBadChannels(meg06_[ism-1], i06_[ism-1]);
      }

      if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 1) != ledWavelengths_.end() ) {
        UtilsClient::printBadChannels(meg09_[ism-1], i09_[ism-1]);
        UtilsClient::printBadChannels(meg09_[ism-1], i13_[ism-1]);
      }
      if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 2) != ledWavelengths_.end() ) {
        UtilsClient::printBadChannels(meg10_[ism-1], i10_[ism-1]);
        UtilsClient::printBadChannels(meg10_[ism-1], i14_[ism-1]);
      }
    }

    for ( int i = 1; i <= 10; i++ ) {

      bool update01;
      bool update02;

      bool update05;
      bool update06;

      bool update09;
      bool update10;

      bool update13;
      bool update14;

      float num01, num02, num05, num06;
      float num09, num10, num13, num14;
      float mean01, mean02, mean05, mean06;
      float mean09, mean10, mean13, mean14;
      float rms01, rms02, rms05, rms06;
      float rms09, rms10, rms13, rms14;

      update01 = UtilsClient::getBinStatistics(i01_[ism-1], i, 0, num01, mean01, rms01);
      update02 = UtilsClient::getBinStatistics(i02_[ism-1], i, 0, num02, mean02, rms02);

      update05 = UtilsClient::getBinStatistics(i05_[ism-1], i, 0, num05, mean05, rms05);
      update06 = UtilsClient::getBinStatistics(i06_[ism-1], i, 0, num06, mean06, rms06);

      update09 = UtilsClient::getBinStatistics(i09_[ism-1], i, 0, num09, mean09, rms09);
      update10 = UtilsClient::getBinStatistics(i10_[ism-1], i, 0, num10, mean10, rms10);

      update13 = UtilsClient::getBinStatistics(i13_[ism-1], i, 0, num13, mean13, rms13);
      update14 = UtilsClient::getBinStatistics(i14_[ism-1], i, 0, num14, mean14, rms14);

      if ( update01 || update05 || update09 || update13 ) {

        if ( i == 1 ) {

          if ( verbose_ ) {
            cout << "Preparing dataset for " << Numbers::sEE(ism) << " (ism=" << ism << ")" << endl;
            cout << "PNs (" << i << ") L1 G01 " << num01  << " " << mean01 << " " << rms01  << endl;
            cout << "PNs (" << i << ") L1 G16 " << num09  << " " << mean09 << " " << rms09  << endl;
            cout << endl;
          }

        }

        pn_l1.setADCMeanG1(mean01);
        pn_l1.setADCRMSG1(rms01);

        pn_l1.setPedMeanG1(mean05);
        pn_l1.setPedRMSG1(rms05);

        pn_l1.setADCMeanG16(mean09);
        pn_l1.setADCRMSG16(rms09);

        pn_l1.setPedMeanG16(mean13);
        pn_l1.setPedRMSG16(rms13);

        if ( UtilsClient::getBinStatus(meg05_[ism-1], i, 1) ||
             UtilsClient::getBinStatus(meg09_[ism-1], i, 1) ) {
          pn_l1.setTaskStatus(true);
        } else {
          pn_l1.setTaskStatus(false);
        }

        status = status && ( UtilsClient::getBinQuality(meg05_[ism-1], i, 1) ||
                             UtilsClient::getBinQuality(meg09_[ism-1], i, 1) );

        if ( econn ) {
          ecid = LogicID::getEcalLogicID("EE_LM_PN", Numbers::iSM(ism, EcalEndcap), i-1);
          dataset2_l1[ecid] = pn_l1;
        }

      }

      if ( update02 || update06 || update10 || update14 ) {

        if ( i == 1 ) {

          if ( verbose_ ) {
            cout << "Preparing dataset for " << Numbers::sEE(ism) << " (ism=" << ism << ")" << endl;
            cout << "PNs (" << i << ") L2 G01 " << num02  << " " << mean02 << " " << rms02  << endl;
            cout << "PNs (" << i << ") L2 G16 " << num10  << " " << mean10 << " " << rms10  << endl;
            cout << endl;
          }

        }

        pn_l2.setADCMeanG1(mean02);
        pn_l2.setADCRMSG1(rms02);

        pn_l2.setPedMeanG1(mean06);
        pn_l2.setPedRMSG1(rms06);

        pn_l2.setADCMeanG16(mean10);
        pn_l2.setADCRMSG16(rms10);

        pn_l2.setPedMeanG16(mean14);
        pn_l2.setPedRMSG16(rms14);

        if ( UtilsClient::getBinStatus(meg06_[ism-1], i, 1) ||
             UtilsClient::getBinStatus(meg10_[ism-1], i, 1) ) {
          pn_l2.setTaskStatus(true);
        } else {
          pn_l2.setTaskStatus(false);
        }

        status = status && ( UtilsClient::getBinQuality(meg06_[ism-1], i, 1) ||
                             UtilsClient::getBinQuality(meg10_[ism-1], i, 1) );

        if ( econn ) {
          ecid = LogicID::getEcalLogicID("EE_LM_PN", Numbers::iSM(ism, EcalEndcap), i-1);
          dataset2_l2[ecid] = pn_l2;
        }

      }

    }

  }

  if ( econn ) {
    try {
      if ( verbose_ ) cout << "Inserting MonPnDat ..." << endl;
      if ( dataset2_l1.size() != 0 ) econn->insertDataArraySet(&dataset2_l1, moniov);
      if ( dataset2_l2.size() != 0 ) econn->insertDataArraySet(&dataset2_l2, moniov);
      if ( verbose_ ) cout << "done." << endl;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

  if ( verbose_ ) cout << endl;

  MonTimingLed1CrystalDat t_l1;
  map<EcalLogicID, MonTimingLed1CrystalDat> dataset3_l1;
  MonTimingLed2CrystalDat t_l2;
  map<EcalLogicID, MonTimingLed2CrystalDat> dataset3_l2;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( verbose_ ) {
      cout << " " << Numbers::sEE(ism) << " (ism=" << ism << ")" << endl;
      cout << endl;
    }

    for ( int ix = 1; ix <= 50; ix++ ) {
      for ( int iy = 1; iy <= 50; iy++ ) {

        int jx = ix + Numbers::ix0EE(ism);
        int jy = iy + Numbers::iy0EE(ism);

        if ( ism >= 1 && ism <= 9 ) jx = 101 - jx;

        if ( ! Numbers::validEE(ism, jx, jy) ) continue;

        bool update01;
        bool update02;

        float num01, num02;
        float mean01, mean02;
        float rms01, rms02;

        update01 = UtilsClient::getBinStatistics(h09_[ism-1], ix, iy, num01, mean01, rms01);
        update02 = UtilsClient::getBinStatistics(h10_[ism-1], ix, iy, num02, mean02, rms02);

        if ( update01 ) {

          if ( Numbers::icEE(ism, ix, iy) == 1 ) {

            if ( verbose_ ) {
              cout << "Preparing dataset for " << Numbers::sEE(ism) << " (ism=" << ism << ")" << endl;
              cout << "L1 crystal (" << Numbers::ix0EE(i+1)+ix << "," << Numbers::iy0EE(i+1)+iy << ") " << num01  << " " << mean01 << " " << rms01  << endl;
              cout << endl;
            }

          }

          t_l1.setTimingMean(mean01);
          t_l1.setTimingRMS(rms01);

          if ( UtilsClient::getBinStatus(meg01_[ism-1], ix, iy) ) {
            t_l1.setTaskStatus(true);
          } else {
            t_l1.setTaskStatus(false);
          }

          status = status && UtilsClient::getBinQuality(meg01_[ism-1], ix, iy);

          int ic = Numbers::indexEE(ism, ix, iy);

          if ( ic == -1 ) continue;

          if ( econn ) {
            ecid = LogicID::getEcalLogicID("EE_crystal_number", Numbers::iSM(ism, EcalEndcap), ic);
            dataset3_l1[ecid] = t_l1;
          }

        }

        if ( update02 ) {

          if ( Numbers::icEE(ism, ix, iy) == 1 ) {

            if ( verbose_ ) {
              cout << "Preparing dataset for " << Numbers::sEE(ism) << " (ism=" << ism << ")" << endl;
              cout << "L2 crystal (" << Numbers::ix0EE(i+1)+ix << "," << Numbers::iy0EE(i+1)+iy << ") " << num02  << " " << mean02 << " " << rms02  << endl;
              cout << endl;
            }

          }

          t_l2.setTimingMean(mean02);
          t_l2.setTimingRMS(rms02);

          if ( UtilsClient::getBinStatus(meg02_[ism-1], ix, iy) ) {
            t_l2.setTaskStatus(true);
          } else {
            t_l2.setTaskStatus(false);
          }

          status = status && UtilsClient::getBinQuality(meg02_[ism-1], ix, iy);

          int ic = Numbers::indexEE(ism, ix, iy);

          if ( ic == -1 ) continue;

          if ( econn ) {
            ecid = LogicID::getEcalLogicID("EE_crystal_number", Numbers::iSM(ism, EcalEndcap), ic);
            dataset3_l2[ecid] = t_l2;
          }

        }

      }
    }

  }

  if ( econn ) {
    try {
      if ( verbose_ ) cout << "Inserting MonTimingLaserCrystalDat ..." << endl;
      if ( dataset3_l1.size() != 0 ) econn->insertDataArraySet(&dataset3_l1, moniov);
      if ( dataset3_l2.size() != 0 ) econn->insertDataArraySet(&dataset3_l2, moniov);
      if ( verbose_ ) cout << "done." << endl;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

  return true;

}

void EELedClient::analyze(void) {

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( debug_ ) cout << "EELedClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;
  }

  uint64_t bits01 = 0;
  bits01 |= EcalErrorDictionary::getMask("LED_MEAN_WARNING");
  bits01 |= EcalErrorDictionary::getMask("LED_RMS_WARNING");
  bits01 |= EcalErrorDictionary::getMask("LED_MEAN_OVER_PN_WARNING");
  bits01 |= EcalErrorDictionary::getMask("LED_RMS_OVER_PN_WARNING");

  uint64_t bits02 = 0;
  bits02 |= EcalErrorDictionary::getMask("PEDESTAL_LOW_GAIN_MEAN_WARNING");
  bits02 |= EcalErrorDictionary::getMask("PEDESTAL_LOW_GAIN_RMS_WARNING");
  bits02 |= EcalErrorDictionary::getMask("PEDESTAL_LOW_GAIN_MEAN_ERROR");
  bits02 |= EcalErrorDictionary::getMask("PEDESTAL_LOW_GAIN_RMS_ERROR");

  uint64_t bits03 = 0;
  bits03 |= EcalErrorDictionary::getMask("PEDESTAL_MIDDLE_GAIN_MEAN_WARNING");
  bits03 |= EcalErrorDictionary::getMask("PEDESTAL_MIDDLE_GAIN_RMS_WARNING");
  bits03 |= EcalErrorDictionary::getMask("PEDESTAL_MIDDLE_GAIN_MEAN_ERROR");
  bits03 |= EcalErrorDictionary::getMask("PEDESTAL_MIDDLE_GAIN_RMS_ERROR");

  uint64_t bits04 = 0;
  bits04 |= EcalErrorDictionary::getMask("PEDESTAL_HIGH_GAIN_MEAN_WARNING");
  bits04 |= EcalErrorDictionary::getMask("PEDESTAL_HIGH_GAIN_RMS_WARNING");
  bits04 |= EcalErrorDictionary::getMask("PEDESTAL_HIGH_GAIN_MEAN_ERROR");
  bits04 |= EcalErrorDictionary::getMask("PEDESTAL_HIGH_GAIN_RMS_ERROR");

  map<EcalLogicID, RunCrystalErrorsDat> mask1;
  map<EcalLogicID, RunPNErrorsDat> mask2;
  map<EcalLogicID, RunTTErrorsDat> mask3;

  EcalErrorMask::fetchDataSet(&mask1);
  EcalErrorMask::fetchDataSet(&mask2);
  EcalErrorMask::fetchDataSet(&mask3);

  char histo[200];

  MonitorElement* me;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 1) != ledWavelengths_.end() ) {

      sprintf(histo, (prefixME_ + "/EELedTask/Led1/EELDT amplitude %s L1").c_str(), Numbers::sEE(ism).c_str());
      me = dqmStore_->get(histo);
      h01_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h01_[ism-1] );

      sprintf(histo, (prefixME_ + "/EELedTask/Led1/EELDT amplitude over PN %s L1").c_str(), Numbers::sEE(ism).c_str());
      me = dqmStore_->get(histo);
      h02_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h02_[ism-1] );

    }

    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 2) != ledWavelengths_.end() ) {

      sprintf(histo, (prefixME_ + "/EELedTask/Led2/EELDT amplitude %s L2").c_str(), Numbers::sEE(ism).c_str());
      me = dqmStore_->get(histo);
      h03_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h03_[ism-1] );

      sprintf(histo, (prefixME_ + "/EELedTask/Led2/EELDT amplitude over PN %s L2").c_str(), Numbers::sEE(ism).c_str());
      me = dqmStore_->get(histo);
      h04_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h04_[ism-1] );

    }

    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 1) != ledWavelengths_.end() ) {

      sprintf(histo, (prefixME_ + "/EELedTask/Led1/EELDT timing %s L1").c_str(), Numbers::sEE(ism).c_str());
      me = dqmStore_->get(histo);
      h09_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h09_[ism-1] );

    }

    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 2) != ledWavelengths_.end() ) {

      sprintf(histo, (prefixME_ + "/EELedTask/Led2/EELDT timing %s L2").c_str(), Numbers::sEE(ism).c_str());
      me = dqmStore_->get(histo);
      h10_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h10_[ism-1] );

    }

    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 1) != ledWavelengths_.end() ) {

      sprintf(histo, (prefixME_ + "/EELedTask/Led1/EELDT shape %s L1").c_str(), Numbers::sEE(ism).c_str());
      me = dqmStore_->get(histo);
      hs01_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, hs01_[ism-1] );

    }

    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 2) != ledWavelengths_.end() ) {

      sprintf(histo, (prefixME_ + "/EELedTask/Led2/EELDT shape %s L2").c_str(), Numbers::sEE(ism).c_str());
      me = dqmStore_->get(histo);
      hs02_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, hs02_[ism-1] );

    }

    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 1) != ledWavelengths_.end() ) {

      sprintf(histo, (prefixME_ + "/EELedTask/Led1/PN/Gain01/EEPDT PNs amplitude %s G01 L1").c_str(), Numbers::sEE(ism).c_str());
      me = dqmStore_->get(histo);
      i01_[ism-1] = UtilsClient::getHisto<TProfile*>( me, cloneME_, i01_[ism-1] );

    }

    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 2) != ledWavelengths_.end() ) {

      sprintf(histo, (prefixME_ + "/EELedTask/Led2/PN/Gain01/EEPDT PNs amplitude %s G01 L2").c_str(), Numbers::sEE(ism).c_str());
      me = dqmStore_->get(histo);
      i02_[ism-1] = UtilsClient::getHisto<TProfile*>( me, cloneME_, i02_[ism-1] );

    }

    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 1) != ledWavelengths_.end() ) {

      sprintf(histo, (prefixME_ + "/EELedTask/Led1/PN/Gain01/EEPDT PNs pedestal %s G01 L1").c_str(), Numbers::sEE(ism).c_str());
      me = dqmStore_->get(histo);
      i05_[ism-1] = UtilsClient::getHisto<TProfile*>( me, cloneME_, i05_[ism-1] );

    }

    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 2) != ledWavelengths_.end() ) {

      sprintf(histo, (prefixME_ + "/EELedTask/Led2/PN/Gain01/EEPDT PNs pedestal %s G01 L2").c_str(), Numbers::sEE(ism).c_str());
      me = dqmStore_->get(histo);
      i06_[ism-1] = UtilsClient::getHisto<TProfile*>( me, cloneME_, i06_[ism-1] );

    }

    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 1) != ledWavelengths_.end() ) {

      sprintf(histo, (prefixME_ + "/EELedTask/Led1/PN/Gain16/EEPDT PNs amplitude %s G16 L1").c_str(), Numbers::sEE(ism).c_str());
      me = dqmStore_->get(histo);
      i09_[ism-1] = UtilsClient::getHisto<TProfile*>( me, cloneME_, i09_[ism-1] );

    }

    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 2) != ledWavelengths_.end() ) {

      sprintf(histo, (prefixME_ + "/EELedTask/Led2/PN/Gain16/EEPDT PNs amplitude %s G16 L2").c_str(), Numbers::sEE(ism).c_str());
      me = dqmStore_->get(histo);
      i10_[ism-1] = UtilsClient::getHisto<TProfile*>( me, cloneME_, i10_[ism-1] );

    }

    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 1) != ledWavelengths_.end() ) {

      sprintf(histo, (prefixME_ + "/EELedTask/Led1/PN/Gain16/EEPDT PNs pedestal %s G16 L1").c_str(), Numbers::sEE(ism).c_str());
      me = dqmStore_->get(histo);
      i13_[ism-1] = UtilsClient::getHisto<TProfile*>( me, cloneME_, i13_[ism-1] );

    }

    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 2) != ledWavelengths_.end() ) {

      sprintf(histo, (prefixME_ + "/EELedTask/Led2/PN/Gain16/EEPDT PNs pedestal %s G16 L2").c_str(), Numbers::sEE(ism).c_str());
      me = dqmStore_->get(histo);
      i14_[ism-1] = UtilsClient::getHisto<TProfile*>( me, cloneME_, i14_[ism-1] );

    }

    if ( meg01_[ism-1] ) meg01_[ism-1]->Reset();
    if ( meg02_[ism-1] ) meg02_[ism-1]->Reset();

    if ( meg05_[ism-1] ) meg05_[ism-1]->Reset();
    if ( meg06_[ism-1] ) meg06_[ism-1]->Reset();

    if ( meg09_[ism-1] ) meg09_[ism-1]->Reset();
    if ( meg10_[ism-1] ) meg10_[ism-1]->Reset();

    if ( mea01_[ism-1] ) mea01_[ism-1]->Reset();
    if ( mea02_[ism-1] ) mea02_[ism-1]->Reset();

    if ( met01_[ism-1] ) met01_[ism-1]->Reset();
    if ( met02_[ism-1] ) met02_[ism-1]->Reset();

    if ( metav01_[ism-1] ) metav01_[ism-1]->Reset();
    if ( metav02_[ism-1] ) metav02_[ism-1]->Reset();

    if ( metrms01_[ism-1] ) metrms01_[ism-1]->Reset();
    if ( metrms02_[ism-1] ) metrms02_[ism-1]->Reset();

    if ( meaopn01_[ism-1] ) meaopn01_[ism-1]->Reset();
    if ( meaopn02_[ism-1] ) meaopn02_[ism-1]->Reset();

    if ( mepnprms01_[ism-1] ) mepnprms01_[ism-1]->Reset();
    if ( mepnprms02_[ism-1] ) mepnprms02_[ism-1]->Reset();

    if ( me_hs01_[ism-1] ) me_hs01_[ism-1]->Reset();
    if ( me_hs02_[ism-1] ) me_hs02_[ism-1]->Reset();

    float meanAmplL1, meanAmplL2;

    int nCryL1, nCryL2;

    meanAmplL1 = meanAmplL2 = 0.;

    nCryL1 = nCryL2 = 0;

    for ( int ix = 1; ix <= 50; ix++ ) {
      for ( int iy = 1; iy <= 50; iy++ ) {

        bool update01;
        bool update02;

        float num01, num02;
        float mean01, mean02;
        float rms01, rms02;

        update01 = UtilsClient::getBinStatistics(h01_[ism-1], ix, iy, num01, mean01, rms01);
        update02 = UtilsClient::getBinStatistics(h03_[ism-1], ix, iy, num02, mean02, rms02);

        if ( update01 ) {
          meanAmplL1 += mean01;
          nCryL1++;
        }

        if ( update02 ) {
          meanAmplL2 += mean02;
          nCryL2++;
        }

      }
    }

    if ( nCryL1 > 0 ) meanAmplL1 /= float (nCryL1);
    if ( nCryL2 > 0 ) meanAmplL2 /= float (nCryL2);

    for ( int ix = 1; ix <= 50; ix++ ) {
      for ( int iy = 1; iy <= 50; iy++ ) {

        if ( meg01_[ism-1] ) meg01_[ism-1]->setBinContent( ix, iy, 6.);
        if ( meg02_[ism-1] ) meg02_[ism-1]->setBinContent( ix, iy, 6.);

        int jx = ix + Numbers::ix0EE(ism);
        int jy = iy + Numbers::iy0EE(ism);

        if ( ism >= 1 && ism <= 9 ) jx = 101 - jx;

        if ( Numbers::validEE(ism, jx, jy) ) {
          if ( meg01_[ism-1] ) meg01_[ism-1]->setBinContent( ix, iy, 2.);
          if ( meg02_[ism-1] ) meg02_[ism-1]->setBinContent( ix, iy, 2.);
        }

        bool update01;
        bool update02;
        bool update03;
        bool update04;

        bool update09;
        bool update10;

        float num01, num02, num03, num04;
        float num09, num10;
        float mean01, mean02, mean03, mean04;
        float mean09, mean10;
        float rms01, rms02, rms03, rms04;
        float rms09, rms10;

        update01 = UtilsClient::getBinStatistics(h01_[ism-1], ix, iy, num01, mean01, rms01);
        update02 = UtilsClient::getBinStatistics(h02_[ism-1], ix, iy, num02, mean02, rms02);
        update03 = UtilsClient::getBinStatistics(h03_[ism-1], ix, iy, num03, mean03, rms03);
        update04 = UtilsClient::getBinStatistics(h04_[ism-1], ix, iy, num04, mean04, rms04);

        update09 = UtilsClient::getBinStatistics(h09_[ism-1], ix, iy, num09, mean09, rms09);
        update10 = UtilsClient::getBinStatistics(h10_[ism-1], ix, iy, num10, mean10, rms10);

        if ( update01 ) {

          float val;

          val = 1.;
          if ( mean01 < amplitudeThreshold_ || rms01 > rmsThresholdRelative_ * mean01 )
            val = 0.;
          if ( meg01_[ism-1] ) meg01_[ism-1]->setBinContent( ix, iy, val );

          int ic = Numbers::icEE(ism, jx, jy);

          if ( ic != -1 ) {
            if ( mea01_[ism-1] ) {
              if ( mean01 > 0. ) {
                mea01_[ism-1]->setBinContent( ic, mean01 );
                mea01_[ism-1]->setBinError( ic, rms01 );
              } else {
                mea01_[ism-1]->setEntries( 1.+mea01_[ism-1]->getEntries() );
              }
            }
          }

        }

        if ( update03 ) {

          float val;

          val = 1.;
          if ( mean03 < amplitudeThreshold_ || rms03 > rmsThresholdRelative_ * mean03 )
            val = 0.;
          if ( meg02_[ism-1] ) meg02_[ism-1]->setBinContent( ix, iy, val);

          int ic = Numbers::icEE(ism, jx, jy);

          if ( ic != -1 ) {
            if ( mea02_[ism-1] ) {
              if ( mean03 > 0. ) {
                mea02_[ism-1]->setBinContent( ic, mean03 );
                mea02_[ism-1]->setBinError( ic, rms03 );
              } else {
                mea02_[ism-1]->setEntries( 1.+mea02_[ism-1]->getEntries() );
              }
            }
          }

        }

        if ( update02 ) {

          int ic = Numbers::icEE(ism, jx, jy);

          if ( ic != -1 ) {
            if ( meaopn01_[ism-1] ) {
              if ( mean02 > 0. ) {
                meaopn01_[ism-1]->setBinContent( ic, mean02 );
                meaopn01_[ism-1]->setBinError( ic, rms02 );
              } else {
                meaopn01_[ism-1]->setEntries( 1.+meaopn01_[ism-1]->getEntries() );
              }
            }
          }

        }

        if ( update04 ) {

          int ic = Numbers::icEE(ism, jx, jy);

          if ( ic != -1 ) {
            if ( meaopn02_[ism-1] ) {
              if ( mean04 > 0. ) {
                meaopn02_[ism-1]->setBinContent( ic, mean04 );
                meaopn02_[ism-1]->setBinError( ic, rms04 );
              } else {
                meaopn02_[ism-1]->setEntries( 1.+meaopn02_[ism-1]->getEntries() );
              }
            }
          }

        }

        if ( update09 ) {

          int ic = Numbers::icEE(ism, jx, jy);

          if ( ic != -1 ) {
            if ( met01_[ism-1] ) {
              if ( mean09 > 0. ) {
                met01_[ism-1]->setBinContent( ic, mean09 );
                met01_[ism-1]->setBinError( ic, rms09 );
              } else {
                met01_[ism-1]->setEntries(1.+met01_[ism-1]->getEntries());
              }
            }

            if ( metav01_[ism-1] ) metav01_[ism-1] ->Fill(mean09);
            if ( metrms01_[ism-1] ) metrms01_[ism-1]->Fill(rms09);
          }

        }

        if ( update10 ) {

          int ic = Numbers::icEE(ism, jx, jy);

          if ( ic != -1 ) {
            if ( met02_[ism-1] ) {
              if ( mean10 > 0. ) {
                met02_[ism-1]->setBinContent( ic, mean10 );
                met02_[ism-1]->setBinError( ic, rms10 );
              } else {
                met02_[ism-1]->setEntries(1.+met02_[ism-1]->getEntries());
              }
            }

            if ( metav02_[ism-1] ) metav02_[ism-1] ->Fill(mean10);
            if ( metrms02_[ism-1] ) metrms02_[ism-1]->Fill(rms10);
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
                UtilsClient::maskBinContent( meg01_[ism-1], ix, iy );
                UtilsClient::maskBinContent( meg02_[ism-1], ix, iy );
              }
            }

          }
        }

        // TT masking

        if ( mask3.size() != 0 ) {
          map<EcalLogicID, RunTTErrorsDat>::const_iterator m;
          for (m = mask3.begin(); m != mask3.end(); m++) {

            EcalLogicID ecid = m->first;

            int itt = Numbers::iTT(ism, EcalEndcap, ix, iy);

            if ( ecid.getLogicID() == LogicID::getEcalLogicID("EE_readout_tower", Numbers::iSM(ism, EcalEndcap), itt).getLogicID() ) {
              if ( (m->second).getErrorBits() & bits01 ) {
                UtilsClient::maskBinContent( meg01_[ism-1], ix, iy );                           UtilsClient::maskBinContent( meg02_[ism-1], ix, iy );                         }
            }

          }
        }

      }
    }

    for ( int i = 1; i <= 10; i++ ) {

      if ( meg05_[ism-1] ) meg05_[ism-1]->setBinContent( i, 1, 2. );
      if ( meg06_[ism-1] ) meg06_[ism-1]->setBinContent( i, 1, 2. );

      if ( meg09_[ism-1] ) meg09_[ism-1]->setBinContent( i, 1, 2. );
      if ( meg10_[ism-1] ) meg10_[ism-1]->setBinContent( i, 1, 2. );

      bool update01;
      bool update02;

      bool update05;
      bool update06;

      bool update09;
      bool update10;

      bool update13;
      bool update14;

      float num01, num02, num05, num06;
      float num09, num10, num13, num14;
      float mean01, mean02, mean05, mean06;
      float mean09, mean10, mean13, mean14;
      float rms01, rms02, rms05, rms06;
      float rms09, rms10, rms13, rms14;

      update01 = UtilsClient::getBinStatistics(i01_[ism-1], i, 0, num01, mean01, rms01);
      update02 = UtilsClient::getBinStatistics(i02_[ism-1], i, 0, num02, mean02, rms02);

      update05 = UtilsClient::getBinStatistics(i05_[ism-1], i, 0, num05, mean05, rms05);
      update06 = UtilsClient::getBinStatistics(i06_[ism-1], i, 0, num06, mean06, rms06);

      update09 = UtilsClient::getBinStatistics(i09_[ism-1], i, 0, num09, mean09, rms09);
      update10 = UtilsClient::getBinStatistics(i10_[ism-1], i, 0, num10, mean10, rms10);

      update13 = UtilsClient::getBinStatistics(i13_[ism-1], i, 0, num13, mean13, rms13);
      update14 = UtilsClient::getBinStatistics(i14_[ism-1], i, 0, num14, mean14, rms14);

      if ( update01 && update05 ) {

        float val;

        val = 1.;
        if ( mean01 < amplitudeThresholdPnG01_ )
          val = 0.;
        if ( mean05 <  pedPnExpectedMean_[0] - pedPnDiscrepancyMean_[0] ||
             pedPnExpectedMean_[0] + pedPnDiscrepancyMean_[0] < mean05)
          val = 0.;
        if ( rms05 > pedPnRMSThreshold_[0] )
          val = 0.;

        if ( meg05_[ism-1] ) meg05_[ism-1]->setBinContent(i, 1, val);
        if ( mepnprms01_[ism-1] ) mepnprms01_[ism-1]->Fill(rms05);

      }

      if ( update02 && update06 ) {

        float val;

        val = 1.;
        if ( mean02 < amplitudeThresholdPnG01_ )
          val = 0.;
        if ( mean06 <  pedPnExpectedMean_[0] - pedPnDiscrepancyMean_[0] ||
             pedPnExpectedMean_[0] + pedPnDiscrepancyMean_[0] < mean06)
          val = 0.;
        if ( rms06 > pedPnRMSThreshold_[0] )
          val = 0.;

        if ( meg06_[ism-1] )           meg06_[ism-1]->setBinContent(i, 1, val);
        if ( mepnprms02_[ism-1] ) mepnprms02_[ism-1]->Fill(rms06);

      }

      if ( update09 && update13 ) {

        float val;

        val = 1.;
        if ( mean09 < amplitudeThresholdPnG16_ )
          val = 0.;
        if ( mean13 <  pedPnExpectedMean_[1] - pedPnDiscrepancyMean_[1] ||
             pedPnExpectedMean_[1] + pedPnDiscrepancyMean_[1] < mean13)
          val = 0.;
        if ( rms13 > pedPnRMSThreshold_[1] )
          val = 0.;

        if ( meg09_[ism-1] )           meg09_[ism-1]->setBinContent(i, 1, val);
        if ( mepnprms05_[ism-1] ) mepnprms05_[ism-1]->Fill(rms13);

      }

      if ( update10 && update14 ) {

        float val;

        val = 1.;
        if ( mean10 < amplitudeThresholdPnG16_ )
          val = 0.;
        //        if ( mean14 < pedestalThresholdPn_ )
       if ( mean14 <  pedPnExpectedMean_[1] - pedPnDiscrepancyMean_[1] ||
            pedPnExpectedMean_[1] + pedPnDiscrepancyMean_[1] < mean14)
           val = 0.;
       if ( rms14 > pedPnRMSThreshold_[1] )
          val = 0.;

       if ( meg10_[ism-1] )           meg10_[ism-1]->setBinContent(i, 1, val);
       if ( mepnprms06_[ism-1] ) mepnprms06_[ism-1]->Fill(rms14);

      }

      // masking

      if ( mask2.size() != 0 ) {
        map<EcalLogicID, RunPNErrorsDat>::const_iterator m;
        for (m = mask2.begin(); m != mask2.end(); m++) {

          EcalLogicID ecid = m->first;

          if ( ecid.getLogicID() == LogicID::getEcalLogicID("EE_LM_PN", Numbers::iSM(ism, EcalEndcap), i-1).getLogicID() ) {
            if ( (m->second).getErrorBits() & (bits01|bits02) ) {
              UtilsClient::maskBinContent( meg05_[ism-1], i, 1 );
              UtilsClient::maskBinContent( meg06_[ism-1], i, 1 );
            }
            if ( (m->second).getErrorBits() & (bits01|bits04) ) {
              UtilsClient::maskBinContent( meg09_[ism-1], i, 1 );
              UtilsClient::maskBinContent( meg10_[ism-1], i, 1 );
            }
          }

        }
      }

    }

    for ( int i = 1; i <= 10; i++ ) {

      if ( hs01_[ism-1] ) {
        int ic = UtilsClient::getFirstNonEmptyChannel( hs01_[ism-1] );
        if ( me_hs01_[ism-1] ) {
          me_hs01_[ism-1]->setBinContent( i, hs01_[ism-1]->GetBinContent(ic, i) );
          me_hs01_[ism-1]->setBinError( i, hs01_[ism-1]->GetBinError(ic, i) );
        }
      }

      if ( hs02_[ism-1] ) {
        int ic = UtilsClient::getFirstNonEmptyChannel( hs02_[ism-1] );
        if ( me_hs02_[ism-1] ) {
          me_hs02_[ism-1]->setBinContent( i, hs02_[ism-1]->GetBinContent(ic, i) );
          me_hs02_[ism-1]->setBinError( i, hs02_[ism-1]->GetBinError(ic, i) );
        }
      }

    }

  }

}

void EELedClient::softReset(bool flag) {

}

