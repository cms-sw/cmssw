/*
 * \file EELedClient.cc
 *
 * $Date: 2012/04/27 13:46:07 $
 * $Revision: 1.133 $
 * \author G. Della Ricca
 * \author G. Franzoni
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
#include "DQM/EcalCommon/interface/LogicID.h"
#endif

#include "DQM/EcalCommon/interface/Masks.h"

#include "DQM/EcalCommon/interface/UtilsClient.h"
#include "DQM/EcalCommon/interface/Numbers.h"

#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "DQM/EcalEndcapMonitorClient/interface/EELedClient.h"

EELedClient::EELedClient(const edm::ParameterSet& ps) {

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

  ledWavelengths_.reserve(2);
  for ( unsigned int i = 1; i <= 2; i++ ) ledWavelengths_.push_back(i);
  ledWavelengths_ = ps.getUntrackedParameter<std::vector<int> >("ledWavelengths", ledWavelengths_);

  if ( verbose_ ) {
    std::cout << " Led wavelengths:" << std::endl;
    for ( unsigned int i = 0; i < ledWavelengths_.size(); i++ ) {
      std::cout << " " << ledWavelengths_[i];
    }
    std::cout << std::endl;
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

  percentVariation01_ = 999.; // not used nor not normalized VPTs
  percentVariation03_ = 999.; // not used nor not normalized VPTs

  amplitudeThreshold01_ = 2.;
  amplitudeThreshold03_ = 2.;

  rmsThreshold01_ = 10.;
  rmsThreshold03_ = 10.;

  amplitudeThresholdPnG01_ = 100.;
  amplitudeThresholdPnG16_ = 100.;

  pedPnExpectedMean_[0] = 750.0;
  pedPnExpectedMean_[1] = 750.0;

  pedPnDiscrepancyMean_[0] = 100.0;
  pedPnDiscrepancyMean_[1] = 100.0;

  pedPnRMSThreshold_[0] = 10.;
  pedPnRMSThreshold_[1] = 10.;

}

EELedClient::~EELedClient() {

}

void EELedClient::beginJob(void) {

  dqmStore_ = edm::Service<DQMStore>().operator->();

  if ( debug_ ) std::cout << "EELedClient: beginJob" << std::endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EELedClient::beginRun(void) {

  if ( debug_ ) std::cout << "EELedClient: beginRun" << std::endl;

  jevt_ = 0;

  this->setup();

}

void EELedClient::endJob(void) {

  if ( debug_ ) std::cout << "EELedClient: endJob, ievt = " << ievt_ << std::endl;

  this->cleanup();

}

void EELedClient::endRun(void) {

  if ( debug_ ) std::cout << "EELedClient: endRun, jevt = " << jevt_ << std::endl;

  this->cleanup();

}

void EELedClient::setup(void) {

  std::string name;

  dqmStore_->setCurrentFolder( prefixME_ + "/EELedClient" );

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 1) != ledWavelengths_.end() ) {
      if ( meg01_[ism-1] ) dqmStore_->removeElement( meg01_[ism-1]->getName() );
      name = "EELDT led quality L1 " + Numbers::sEE(ism);
      meg01_[ism-1] = dqmStore_->book2D(name, name, 50, Numbers::ix0EE(ism)+0., Numbers::ix0EE(ism)+50., 50, Numbers::iy0EE(ism)+0., Numbers::iy0EE(ism)+50.);
      meg01_[ism-1]->setAxisTitle("ix", 1);
      if ( ism >= 1 && ism <= 9 ) meg01_[ism-1]->setAxisTitle("101-ix", 1);
      meg01_[ism-1]->setAxisTitle("iy", 2);
    }
    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 2) != ledWavelengths_.end() ) {
      if ( meg02_[ism-1] ) dqmStore_->removeElement( meg02_[ism-1]->getName() );
      name = "EELDT led quality L2 " + Numbers::sEE(ism);
      meg02_[ism-1] = dqmStore_->book2D(name, name, 50, Numbers::ix0EE(ism)+0., Numbers::ix0EE(ism)+50., 50, Numbers::iy0EE(ism)+0., Numbers::iy0EE(ism)+50.);
      meg02_[ism-1]->setAxisTitle("ix", 1);
      if ( ism >= 1 && ism <= 9 ) meg02_[ism-1]->setAxisTitle("101-ix", 1);
      meg02_[ism-1]->setAxisTitle("iy", 2);
    }

    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 1) != ledWavelengths_.end() ) {
      if ( meg05_[ism-1] ) dqmStore_->removeElement( meg05_[ism-1]->getName() );
      name = "EELDT led quality L1 PNs G01 " + Numbers::sEE(ism);
      meg05_[ism-1] = dqmStore_->book2D(name, name, 10, 0., 10., 1, 0., 5.);
      meg05_[ism-1]->setAxisTitle("pseudo-strip", 1);
      meg05_[ism-1]->setAxisTitle("channel", 2);
    }
    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 2) != ledWavelengths_.end() ) {
      if ( meg06_[ism-1] ) dqmStore_->removeElement( meg06_[ism-1]->getName() );
      name = "EELDT led quality L2 PNs G01 " + Numbers::sEE(ism);
      meg06_[ism-1] = dqmStore_->book2D(name, name, 10, 0., 10., 1, 0., 5.);
      meg06_[ism-1]->setAxisTitle("pseudo-strip", 1);
      meg06_[ism-1]->setAxisTitle("channel", 2);
    }

    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 1) != ledWavelengths_.end() ) {
      if ( meg09_[ism-1] ) dqmStore_->removeElement( meg09_[ism-1]->getName() );
      name = "EELDT led quality L1 PNs G16 " + Numbers::sEE(ism);
      meg09_[ism-1] = dqmStore_->book2D(name, name, 10, 0., 10., 1, 0., 5.);
      meg09_[ism-1]->setAxisTitle("pseudo-strip", 1);
      meg09_[ism-1]->setAxisTitle("channel", 2);
    }
    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 2) != ledWavelengths_.end() ) {
      if ( meg10_[ism-1] ) dqmStore_->removeElement( meg10_[ism-1]->getName() );
      name = "EELDT led quality L2 PNs G16 " + Numbers::sEE(ism);
      meg10_[ism-1] = dqmStore_->book2D(name, name, 10, 0., 10., 1, 0., 5.);
      meg10_[ism-1]->setAxisTitle("pseudo-strip", 1);
      meg10_[ism-1]->setAxisTitle("channel", 2);
    }

    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 1) != ledWavelengths_.end() ) {
      if ( mea01_[ism-1] ) dqmStore_->removeElement( mea01_[ism-1]->getName() );;
      name = "EELDT amplitude L1 " + Numbers::sEE(ism);
      mea01_[ism-1] = dqmStore_->book1D(name, name, 850, 0., 850.);
      mea01_[ism-1]->setAxisTitle("channel", 1);
      mea01_[ism-1]->setAxisTitle("amplitude", 2);
    }
    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 2) != ledWavelengths_.end() ) {
      if ( mea02_[ism-1] ) dqmStore_->removeElement( mea02_[ism-1]->getName() );
      name = "EELDT amplitude L2 " + Numbers::sEE(ism);
      mea02_[ism-1] = dqmStore_->book1D(name, name, 850, 0., 850.);
      mea02_[ism-1]->setAxisTitle("channel", 1);
      mea02_[ism-1]->setAxisTitle("amplitude", 2);
    }

    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 1) != ledWavelengths_.end() ) {
      if ( met01_[ism-1] ) dqmStore_->removeElement( met01_[ism-1]->getName() );
      name = "EELDT led timing L1 " + Numbers::sEE(ism);
      met01_[ism-1] = dqmStore_->book1D(name, name, 850, 0., 850.);
      met01_[ism-1]->setAxisTitle("channel", 1);
      met01_[ism-1]->setAxisTitle("jitter", 2);
    }
    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 2) != ledWavelengths_.end() ) {
      if ( met02_[ism-1] ) dqmStore_->removeElement( met02_[ism-1]->getName() );
      name = "EELDT led timing L2 " + Numbers::sEE(ism);
      met02_[ism-1] = dqmStore_->book1D(name, name, 850, 0., 850.);
      met02_[ism-1]->setAxisTitle("channel", 1);
      met02_[ism-1]->setAxisTitle("jitter", 2);
    }

    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 1) != ledWavelengths_.end() ) {
      if ( metav01_[ism-1] ) dqmStore_->removeElement( metav01_[ism-1]->getName() );
      name = "EELDT led timing mean L1 " + Numbers::sEE(ism);
      metav01_[ism-1] = dqmStore_->book1D(name, name, 100, 0., 10.);
      metav01_[ism-1]->setAxisTitle("mean", 1);
    }
    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 2) != ledWavelengths_.end() ) {
      if ( metav02_[ism-1] ) dqmStore_->removeElement( metav02_[ism-1]->getName() );
      name = "EELDT led timing mean L2 " + Numbers::sEE(ism);
      metav02_[ism-1] = dqmStore_->book1D(name, name, 100, 0., 10.);
      metav02_[ism-1]->setAxisTitle("mean", 1);
    }

    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 1) != ledWavelengths_.end() ) {
      if ( metrms01_[ism-1] ) dqmStore_->removeElement( metrms01_[ism-1]->getName() );
      name = "EELDT led timing rms L1 " + Numbers::sEE(ism);
      metrms01_[ism-1] = dqmStore_->book1D(name, name, 100, 0., 0.5);
      metrms01_[ism-1]->setAxisTitle("rms", 1);
    }
    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 2) != ledWavelengths_.end() ) {
      if ( metrms02_[ism-1] ) dqmStore_->removeElement( metrms02_[ism-1]->getName() );
      name = "EELDT led timing rms L2 " + Numbers::sEE(ism);
      metrms02_[ism-1] = dqmStore_->book1D(name, name, 100, 0., 0.5);
      metrms02_[ism-1]->setAxisTitle("rms", 1);
    }

    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 1) != ledWavelengths_.end() ) {
      if ( meaopn01_[ism-1] ) dqmStore_->removeElement( meaopn01_[ism-1]->getName() );
      name = "EELDT amplitude over PN L1 " + Numbers::sEE(ism);
      meaopn01_[ism-1] = dqmStore_->book1D(name, name, 850, 0., 850.);
      meaopn01_[ism-1]->setAxisTitle("channel", 1);
      meaopn01_[ism-1]->setAxisTitle("amplitude/PN", 2);
    }
    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 2) != ledWavelengths_.end() ) {
      if ( meaopn02_[ism-1] ) dqmStore_->removeElement( meaopn02_[ism-1]->getName() );
      name = "EELDT amplitude over PN L2 " + Numbers::sEE(ism);
      meaopn02_[ism-1] = dqmStore_->book1D(name, name, 850, 0., 850.);
      meaopn02_[ism-1]->setAxisTitle("channel", 1);
      meaopn02_[ism-1]->setAxisTitle("amplitude/PN", 2);
    }

    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 1) != ledWavelengths_.end() ) {
      if ( mepnprms01_[ism-1] ) dqmStore_->removeElement( mepnprms01_[ism-1]->getName() );
      name = "EELDT PNs pedestal rms " + Numbers::sEE(ism) + " G01 L1";
      mepnprms01_[ism-1] = dqmStore_->book1D(name, name, 100, 0., 10.);
      mepnprms01_[ism-1]->setAxisTitle("rms", 1);
    }
    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 2) != ledWavelengths_.end() ) {
      if ( mepnprms02_[ism-1] ) dqmStore_->removeElement( mepnprms02_[ism-1]->getName() );
      name = "EELDT PNs pedestal rms " + Numbers::sEE(ism) + " G01 L2";
      mepnprms02_[ism-1] = dqmStore_->book1D(name, name, 100, 0., 10.);
      mepnprms02_[ism-1]->setAxisTitle("rms", 1);
    }

    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 1) != ledWavelengths_.end() ) {
      if ( mepnprms05_[ism-1] ) dqmStore_->removeElement( mepnprms05_[ism-1]->getName() );
      name = "EELDT PNs pedestal rms " + Numbers::sEE(ism) + " G16 L1";
      mepnprms05_[ism-1] = dqmStore_->book1D(name, name, 100, 0., 10.);
      mepnprms05_[ism-1]->setAxisTitle("rms", 1);
    }
    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 2) != ledWavelengths_.end() ) {
      if ( mepnprms06_[ism-1] ) dqmStore_->removeElement( mepnprms06_[ism-1]->getName() );
      name = "EELDT PNs pedestal rms " + Numbers::sEE(ism) + " G16 L2";
      mepnprms06_[ism-1] = dqmStore_->book1D(name, name, 100, 0., 10.);
      mepnprms06_[ism-1]->setAxisTitle("rms", 1);
    }

    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 1) != ledWavelengths_.end() ) {
      if ( me_hs01_[ism-1] ) dqmStore_->removeElement( me_hs01_[ism-1]->getName() );
      name = "EELDT led shape L1 " + Numbers::sEE(ism);
      me_hs01_[ism-1] = dqmStore_->book1D(name, name, 10, 0., 10.);
      me_hs01_[ism-1]->setAxisTitle("sample", 1);
      me_hs01_[ism-1]->setAxisTitle("amplitude", 2);
    }
    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 2) != ledWavelengths_.end() ) {
      if ( me_hs02_[ism-1] ) dqmStore_->removeElement( me_hs02_[ism-1]->getName() );
      name = "EELDT led shape L2 " + Numbers::sEE(ism);
      me_hs02_[ism-1] = dqmStore_->book1D(name, name, 10, 0., 10.);
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

        if ( meg05_[ism-1] ) meg05_[ism-1]->setBinContent( i, 1, 6. );
        if ( meg06_[ism-1] ) meg06_[ism-1]->setBinContent( i, 1, 6. );

        if ( meg09_[ism-1] ) meg09_[ism-1]->setBinContent( i, 1, 6. );
        if ( meg10_[ism-1] ) meg10_[ism-1]->setBinContent( i, 1, 6. );

        // non-existing mem
        if ( (ism >=  3 && ism <=  4) || (ism >=  7 && ism <=  9) ) continue;
        if ( (ism >= 12 && ism <= 13) || (ism >= 16 && ism <= 18) ) continue;

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

#ifdef WITH_ECAL_COND_DB
bool EELedClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov, bool& status) {

  status = true;

  EcalLogicID ecid;

  MonLed1Dat vpt_l1;
  std::map<EcalLogicID, MonLed1Dat> dataset1_l1;
  MonLed2Dat vpt_l2;
  std::map<EcalLogicID, MonLed2Dat> dataset1_l2;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( verbose_ ) {
      std::cout << " " << Numbers::sEE(ism) << " (ism=" << ism << ")" << std::endl;
      std::cout << std::endl;
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
              std::cout << "Preparing dataset for " << Numbers::sEE(ism) << " (ism=" << ism << ")" << std::endl;
              std::cout << "L1 (" << Numbers::ix0EE(i+1)+ix << "," << Numbers::iy0EE(i+1)+iy << ") " << num01 << " " << mean01 << " " << rms01 << std::endl;
              std::cout << std::endl;
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
              std::cout << "Preparing dataset for " << Numbers::sEE(ism) << " (ism=" << ism << ")" << std::endl;
              std::cout << "L2 (" << Numbers::ix0EE(i+1)+ix << "," << Numbers::iy0EE(i+1)+iy << ") " << num03 << " " << mean03 << " " << rms03 << std::endl;
              std::cout << std::endl;
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
      if ( verbose_ ) std::cout << "Inserting MonLedDat ..." << std::endl;
      if ( dataset1_l1.size() != 0 ) econn->insertDataArraySet(&dataset1_l1, moniov);
      if ( dataset1_l2.size() != 0 ) econn->insertDataArraySet(&dataset1_l2, moniov);
      if ( verbose_ ) std::cout << "done." << std::endl;
    } catch (std::runtime_error &e) {
      std::cerr << e.what() << std::endl;
    }
  }

  if ( verbose_ ) std::cout << std::endl;

  MonPNLed1Dat pn_l1;
  std::map<EcalLogicID, MonPNLed1Dat> dataset2_l1;
  MonPNLed2Dat pn_l2;
  std::map<EcalLogicID, MonPNLed2Dat> dataset2_l2;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( verbose_ ) {
      std::cout << " " << Numbers::sEE(ism) << " (ism=" << ism << ")" << std::endl;
      std::cout << std::endl;
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
            std::cout << "Preparing dataset for " << Numbers::sEE(ism) << " (ism=" << ism << ")" << std::endl;
            std::cout << "PNs (" << i << ") L1 G01 " << num01  << " " << mean01 << " " << rms01  << std::endl;
            std::cout << "PNs (" << i << ") L1 G16 " << num09  << " " << mean09 << " " << rms09  << std::endl;
            std::cout << std::endl;
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
            std::cout << "Preparing dataset for " << Numbers::sEE(ism) << " (ism=" << ism << ")" << std::endl;
            std::cout << "PNs (" << i << ") L2 G01 " << num02  << " " << mean02 << " " << rms02  << std::endl;
            std::cout << "PNs (" << i << ") L2 G16 " << num10  << " " << mean10 << " " << rms10  << std::endl;
            std::cout << std::endl;
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
      if ( verbose_ ) std::cout << "Inserting MonPnDat ..." << std::endl;
      if ( dataset2_l1.size() != 0 ) econn->insertDataArraySet(&dataset2_l1, moniov);
      if ( dataset2_l2.size() != 0 ) econn->insertDataArraySet(&dataset2_l2, moniov);
      if ( verbose_ ) std::cout << "done." << std::endl;
    } catch (std::runtime_error &e) {
      std::cerr << e.what() << std::endl;
    }
  }

  if ( verbose_ ) std::cout << std::endl;

  MonTimingLed1CrystalDat t_l1;
  std::map<EcalLogicID, MonTimingLed1CrystalDat> dataset3_l1;
  MonTimingLed2CrystalDat t_l2;
  std::map<EcalLogicID, MonTimingLed2CrystalDat> dataset3_l2;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( verbose_ ) {
      std::cout << " " << Numbers::sEE(ism) << " (ism=" << ism << ")" << std::endl;
      std::cout << std::endl;
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

          if ( Numbers::icEE(ism, jx, jy) == 1 ) {

            if ( verbose_ ) {
              std::cout << "Preparing dataset for " << Numbers::sEE(ism) << " (ism=" << ism << ")" << std::endl;
              std::cout << "L1 crystal (" << Numbers::ix0EE(i+1)+ix << "," << Numbers::iy0EE(i+1)+iy << ") " << num01  << " " << mean01 << " " << rms01  << std::endl;
              std::cout << std::endl;
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

          int ic = Numbers::indexEE(ism, jx, jy);

          if ( ic == -1 ) continue;

          if ( econn ) {
            ecid = LogicID::getEcalLogicID("EE_crystal_number", Numbers::iSM(ism, EcalEndcap), ic);
            dataset3_l1[ecid] = t_l1;
          }

        }

        if ( update02 ) {

          if ( Numbers::icEE(ism, jx, jy) == 1 ) {

            if ( verbose_ ) {
              std::cout << "Preparing dataset for " << Numbers::sEE(ism) << " (ism=" << ism << ")" << std::endl;
              std::cout << "L2 crystal (" << Numbers::ix0EE(i+1)+ix << "," << Numbers::iy0EE(i+1)+iy << ") " << num02  << " " << mean02 << " " << rms02  << std::endl;
              std::cout << std::endl;
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

          int ic = Numbers::indexEE(ism, jx, jy);

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
      if ( verbose_ ) std::cout << "Inserting MonTimingLaserCrystalDat ..." << std::endl;
      if ( dataset3_l1.size() != 0 ) econn->insertDataArraySet(&dataset3_l1, moniov);
      if ( dataset3_l2.size() != 0 ) econn->insertDataArraySet(&dataset3_l2, moniov);
      if ( verbose_ ) std::cout << "done." << std::endl;
    } catch (std::runtime_error &e) {
      std::cerr << e.what() << std::endl;
    }
  }

  return true;

}
#endif

void EELedClient::analyze(void) {

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( debug_ ) std::cout << "EELedClient: ievt/jevt = " << ievt_ << "/" << jevt_ << std::endl;
  }

  uint32_t bits01 = 0;
  bits01 |= 1 << EcalDQMStatusHelper::LED_MEAN_ERROR;
  bits01 |= 1 << EcalDQMStatusHelper::LED_RMS_ERROR;

  uint32_t bits02 = 0;
  bits02 |= 1 << EcalDQMStatusHelper::LED_TIMING_MEAN_ERROR;
  bits02 |= 1 << EcalDQMStatusHelper::LED_TIMING_RMS_ERROR;

  uint32_t bits03 = 0;
  bits03 |= 1 << EcalDQMStatusHelper::PEDESTAL_LOW_GAIN_MEAN_ERROR;
  bits03 |= 1 << EcalDQMStatusHelper::PEDESTAL_LOW_GAIN_RMS_ERROR;

  uint32_t bits04 = 0;
  bits04 |= 1 << EcalDQMStatusHelper::PEDESTAL_HIGH_GAIN_MEAN_ERROR;
  bits04 |= 1 << EcalDQMStatusHelper::PEDESTAL_HIGH_GAIN_RMS_ERROR;

  MonitorElement* me;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 1) != ledWavelengths_.end() ) {

      me = dqmStore_->get( prefixME_ + "/EELedTask/Led1/EELDT amplitude " + Numbers::sEE(ism) + " L1" );
      h01_[ism-1] = UtilsClient::getHisto( me, cloneME_, h01_[ism-1] );

      me = dqmStore_->get( prefixME_ + "/EELedTask/Led1/EELDT amplitude over PN " + Numbers::sEE(ism) + " L1" );
      h02_[ism-1] = UtilsClient::getHisto( me, cloneME_, h02_[ism-1] );

    }

    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 2) != ledWavelengths_.end() ) {

      me = dqmStore_->get( prefixME_ + "/EELedTask/Led2/EELDT amplitude " + Numbers::sEE(ism) + " L2" );
      h03_[ism-1] = UtilsClient::getHisto( me, cloneME_, h03_[ism-1] );

      me = dqmStore_->get( prefixME_ + "/EELedTask/Led2/EELDT amplitude over PN " + Numbers::sEE(ism) + " L2" );
      h04_[ism-1] = UtilsClient::getHisto( me, cloneME_, h04_[ism-1] );

    }

    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 1) != ledWavelengths_.end() ) {

      me = dqmStore_->get( prefixME_ + "/EELedTask/Led1/EELDT timing " + Numbers::sEE(ism) + " L1" );
      h09_[ism-1] = UtilsClient::getHisto( me, cloneME_, h09_[ism-1] );

    }

    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 2) != ledWavelengths_.end() ) {

      me = dqmStore_->get( prefixME_ + "/EELedTask/Led2/EELDT timing " + Numbers::sEE(ism) + " L2" );
      h10_[ism-1] = UtilsClient::getHisto( me, cloneME_, h10_[ism-1] );

    }

    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 1) != ledWavelengths_.end() ) {

      me = dqmStore_->get( prefixME_ + "/EELedTask/Led1/EELDT shape " + Numbers::sEE(ism) + " L1" );
      hs01_[ism-1] = UtilsClient::getHisto( me, cloneME_, hs01_[ism-1] );

    }

    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 2) != ledWavelengths_.end() ) {

      me = dqmStore_->get( prefixME_ + "/EELedTask/Led2/EELDT shape " + Numbers::sEE(ism) + " L2" );
      hs02_[ism-1] = UtilsClient::getHisto( me, cloneME_, hs02_[ism-1] );

    }

    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 1) != ledWavelengths_.end() ) {

      me = dqmStore_->get( prefixME_ + "/EELedTask/Led1/PN/Gain01/EELDT PNs amplitude " + Numbers::sEE(ism) + " G01 L1" );
      i01_[ism-1] = UtilsClient::getHisto( me, cloneME_, i01_[ism-1] );

    }

    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 2) != ledWavelengths_.end() ) {

      me = dqmStore_->get( prefixME_ + "/EELedTask/Led2/PN/Gain01/EELDT PNs amplitude " + Numbers::sEE(ism) + " G01 L2" );
      i02_[ism-1] = UtilsClient::getHisto( me, cloneME_, i02_[ism-1] );

    }

    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 1) != ledWavelengths_.end() ) {

      me = dqmStore_->get( prefixME_ + "/EELedTask/Led1/PN/Gain01/EELDT PNs pedestal " + Numbers::sEE(ism) + " G01 L1" );
      i05_[ism-1] = UtilsClient::getHisto( me, cloneME_, i05_[ism-1] );

    }

    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 2) != ledWavelengths_.end() ) {

      me = dqmStore_->get( prefixME_ + "/EELedTask/Led2/PN/Gain01/EELDT PNs pedestal " + Numbers::sEE(ism) + " G01 L2" );
      i06_[ism-1] = UtilsClient::getHisto( me, cloneME_, i06_[ism-1] );

    }

    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 1) != ledWavelengths_.end() ) {

      me = dqmStore_->get( prefixME_ + "/EELedTask/Led1/PN/Gain16/EELDT PNs amplitude " + Numbers::sEE(ism) + " G16 L1" );
      i09_[ism-1] = UtilsClient::getHisto( me, cloneME_, i09_[ism-1] );

    }

    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 2) != ledWavelengths_.end() ) {

      me = dqmStore_->get( prefixME_ + "/EELedTask/Led2/PN/Gain16/EELDT PNs amplitude " + Numbers::sEE(ism) + " G16 L2" );
      i10_[ism-1] = UtilsClient::getHisto( me, cloneME_, i10_[ism-1] );

    }

    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 1) != ledWavelengths_.end() ) {

      me = dqmStore_->get( prefixME_ + "/EELedTask/Led1/PN/Gain16/EELDT PNs pedestal " + Numbers::sEE(ism) + " G16 L1" );
      i13_[ism-1] = UtilsClient::getHisto( me, cloneME_, i13_[ism-1] );

    }

    if ( find(ledWavelengths_.begin(), ledWavelengths_.end(), 2) != ledWavelengths_.end() ) {

      me = dqmStore_->get( prefixME_ + "/EELedTask/Led2/PN/Gain16/EELDT PNs pedestal " + Numbers::sEE(ism) + " G16 L2" );
      i14_[ism-1] = UtilsClient::getHisto( me, cloneME_, i14_[ism-1] );

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
          if ( mean01 < amplitudeThreshold01_ || rms01 > rmsThreshold01_ )
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
          if (  mean03 < amplitudeThreshold03_ || rms03 > rmsThreshold03_ )
            val = 0.;
          if ( meg02_[ism-1] ) meg02_[ism-1]->setBinContent( ix, iy, val );

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

        if ( Masks::maskChannel(ism, ix, iy, bits01, EcalEndcap) ) {
          UtilsClient::maskBinContent( meg01_[ism-1], ix, iy );
          UtilsClient::maskBinContent( meg02_[ism-1], ix, iy );
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

      }
    }

    for ( int i = 1; i <= 10; i++ ) {

      if ( meg05_[ism-1] ) meg05_[ism-1]->setBinContent( i, 1, 6. );
      if ( meg06_[ism-1] ) meg06_[ism-1]->setBinContent( i, 1, 6. );

      if ( meg09_[ism-1] ) meg09_[ism-1]->setBinContent( i, 1, 6. );
      if ( meg10_[ism-1] ) meg10_[ism-1]->setBinContent( i, 1, 6. );

      // non-existing mem
      if ( (ism >=  3 && ism <=  4) || (ism >=  7 && ism <=  9) ) continue;
      if ( (ism >= 12 && ism <= 13) || (ism >= 16 && ism <= 18) ) continue;

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

      if ( Masks::maskPn(ism, i, bits01|bits03, EcalEndcap) ) {
        UtilsClient::maskBinContent( meg05_[ism-1], i, 1 );
        UtilsClient::maskBinContent( meg06_[ism-1], i, 1 );
      }

      if ( Masks::maskPn(ism, i, bits01|bits04, EcalEndcap) ) {
        UtilsClient::maskBinContent( meg09_[ism-1], i, 1 );
        UtilsClient::maskBinContent( meg10_[ism-1], i, 1 );
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

