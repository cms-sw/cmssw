/*
 * \file EELedClient.cc
 *
 * $Date: 2008/10/10 16:51:52 $
 * $Revision: 1.91 $
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

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    h01_[ism-1] = 0;
    h02_[ism-1] = 0;
    h03_[ism-1] = 0;
    h04_[ism-1] = 0;

    h09_[ism-1] = 0;
    h10_[ism-1] = 0;

    h13_[ism-1] = 0;
    h14_[ism-1] = 0;
    h15_[ism-1] = 0;
    h16_[ism-1] = 0;

    h21_[ism-1] = 0;
    h22_[ism-1] = 0;

    hs01_[ism-1] = 0;
    hs02_[ism-1] = 0;

    hs05_[ism-1] = 0;
    hs06_[ism-1] = 0;

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

    mea05_[ism-1] = 0;
    mea06_[ism-1] = 0;

    met01_[ism-1] = 0;
    met02_[ism-1] = 0;

    met05_[ism-1] = 0;
    met06_[ism-1] = 0;

    metav01_[ism-1] = 0;
    metav02_[ism-1] = 0;

    metav05_[ism-1] = 0;
    metav06_[ism-1] = 0;

    metrms01_[ism-1] = 0;
    metrms02_[ism-1] = 0;

    metrms05_[ism-1] = 0;
    metrms06_[ism-1] = 0;

    meaopn01_[ism-1] = 0;
    meaopn02_[ism-1] = 0;

    meaopn05_[ism-1] = 0;
    meaopn06_[ism-1] = 0;

    mepnprms01_[ism-1] = 0;
    mepnprms02_[ism-1] = 0;

    mepnprms05_[ism-1] = 0;
    mepnprms06_[ism-1] = 0;

    me_hs01_[ism-1] = 0;
    me_hs02_[ism-1] = 0;

    me_hs05_[ism-1] = 0;
    me_hs06_[ism-1] = 0;

  }

  percentVariation_ = 0.4;
  
  amplitudeThreshold_ = 10.;

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

    if ( meg01_[ism-1] ) dqmStore_->removeElement( meg01_[ism-1]->getName() );
    sprintf(histo, "EELDT led quality L1 %s", Numbers::sEE(ism).c_str());
    meg01_[ism-1] = dqmStore_->book2D(histo, histo, 50, Numbers::ix0EE(ism)+0., Numbers::ix0EE(ism)+50., 50, Numbers::iy0EE(ism)+0., Numbers::iy0EE(ism)+50.);
    meg01_[ism-1]->setAxisTitle("jx", 1);
    meg01_[ism-1]->setAxisTitle("jy", 2);
    if ( meg02_[ism-1] ) dqmStore_->removeElement( meg02_[ism-1]->getName() );
    sprintf(histo, "EELDT led quality L2 %s", Numbers::sEE(ism).c_str());
    meg02_[ism-1] = dqmStore_->book2D(histo, histo, 50, Numbers::ix0EE(ism)+0., Numbers::ix0EE(ism)+50., 50, Numbers::iy0EE(ism)+0., Numbers::iy0EE(ism)+50.);
    meg02_[ism-1]->setAxisTitle("jx", 1);
    meg02_[ism-1]->setAxisTitle("jy", 2);

    if ( meg05_[ism-1] ) dqmStore_->removeElement( meg05_[ism-1]->getName() );
    sprintf(histo, "EELDT led quality L1 PNs G01 %s", Numbers::sEE(ism).c_str());
    meg05_[ism-1] = dqmStore_->book2D(histo, histo, 10, 0., 10., 1, 0., 5.);
    meg05_[ism-1]->setAxisTitle("pseudo-strip", 1);
    meg05_[ism-1]->setAxisTitle("channel", 2);
    if ( meg06_[ism-1] ) dqmStore_->removeElement( meg06_[ism-1]->getName() );
    sprintf(histo, "EELDT led quality L2 PNs G01 %s", Numbers::sEE(ism).c_str());
    meg06_[ism-1] = dqmStore_->book2D(histo, histo, 10, 0., 10., 1, 0., 5.);
    meg06_[ism-1]->setAxisTitle("pseudo-strip", 1);
    meg06_[ism-1]->setAxisTitle("channel", 2);

    if ( meg09_[ism-1] ) dqmStore_->removeElement( meg09_[ism-1]->getName() );
    sprintf(histo, "EELDT led quality L1 PNs G16 %s", Numbers::sEE(ism).c_str());
    meg09_[ism-1] = dqmStore_->book2D(histo, histo, 10, 0., 10., 1, 0., 5.);
    meg09_[ism-1]->setAxisTitle("pseudo-strip", 1);
    meg09_[ism-1]->setAxisTitle("channel", 2);
    if ( meg10_[ism-1] ) dqmStore_->removeElement( meg10_[ism-1]->getName() );
    sprintf(histo, "EELDT led quality L2 PNs G16 %s", Numbers::sEE(ism).c_str());
    meg10_[ism-1] = dqmStore_->book2D(histo, histo, 10, 0., 10., 1, 0., 5.);
    meg10_[ism-1]->setAxisTitle("pseudo-strip", 1);
    meg10_[ism-1]->setAxisTitle("channel", 2);

    if ( mea01_[ism-1] ) dqmStore_->removeElement( mea01_[ism-1]->getName() );;
    sprintf(histo, "EELDT amplitude L1A %s", Numbers::sEE(ism).c_str());
    mea01_[ism-1] = dqmStore_->book1D(histo, histo, 850, 0., 850.);
    mea01_[ism-1]->setAxisTitle("channel", 1);
    mea01_[ism-1]->setAxisTitle("amplitude", 2);
    if ( mea02_[ism-1] ) dqmStore_->removeElement( mea02_[ism-1]->getName() );
    sprintf(histo, "EELDT amplitude L2A %s", Numbers::sEE(ism).c_str());
    mea02_[ism-1] = dqmStore_->book1D(histo, histo, 850, 0., 850.);
    mea02_[ism-1]->setAxisTitle("channel", 1);
    mea02_[ism-1]->setAxisTitle("amplitude", 2);

    if ( mea05_[ism-1] ) dqmStore_->removeElement( mea05_[ism-1]->getName() );;
    sprintf(histo, "EELDT amplitude L1B %s", Numbers::sEE(ism).c_str());
    mea05_[ism-1] = dqmStore_->book1D(histo, histo, 850, 0., 850.);
    mea05_[ism-1]->setAxisTitle("channel", 1);
    mea05_[ism-1]->setAxisTitle("amplitude", 2);
    if ( mea06_[ism-1] ) dqmStore_->removeElement( mea06_[ism-1]->getName() );
    sprintf(histo, "EELDT amplitude L2B %s", Numbers::sEE(ism).c_str());
    mea06_[ism-1] = dqmStore_->book1D(histo, histo, 850, 0., 850.);
    mea06_[ism-1]->setAxisTitle("channel", 1);
    mea06_[ism-1]->setAxisTitle("amplitude", 2);

    if ( met01_[ism-1] ) dqmStore_->removeElement( met01_[ism-1]->getName() );
    sprintf(histo, "EELDT led timing L1A %s", Numbers::sEE(ism).c_str());
    met01_[ism-1] = dqmStore_->book1D(histo, histo, 850, 0., 850.);
    met01_[ism-1]->setAxisTitle("channel", 1);
    met01_[ism-1]->setAxisTitle("jitter", 2);
    if ( met02_[ism-1] ) dqmStore_->removeElement( met02_[ism-1]->getName() );
    sprintf(histo, "EELDT led timing L2A %s", Numbers::sEE(ism).c_str());
    met02_[ism-1] = dqmStore_->book1D(histo, histo, 850, 0., 850.);
    met02_[ism-1]->setAxisTitle("channel", 1);
    met02_[ism-1]->setAxisTitle("jitter", 2);

    if ( met05_[ism-1] ) dqmStore_->removeElement( met05_[ism-1]->getName() );
    sprintf(histo, "EELDT led timing L1B %s", Numbers::sEE(ism).c_str());
    met05_[ism-1] = dqmStore_->book1D(histo, histo, 850, 0., 850.);
    met05_[ism-1]->setAxisTitle("channel", 1);
    met05_[ism-1]->setAxisTitle("jitter", 2);
    if ( met06_[ism-1] ) dqmStore_->removeElement( met06_[ism-1]->getName() );
    sprintf(histo, "EELDT led timing L2B %s", Numbers::sEE(ism).c_str());
    met06_[ism-1] = dqmStore_->book1D(histo, histo, 850, 0., 850.);
    met06_[ism-1]->setAxisTitle("channel", 1);
    met06_[ism-1]->setAxisTitle("jitter", 2);

    if ( metav01_[ism-1] ) dqmStore_->removeElement( metav01_[ism-1]->getName() );
    sprintf(histo, "EELDT led timing mean L1A %s", Numbers::sEE(ism).c_str());
    metav01_[ism-1] = dqmStore_->book1D(histo, histo, 100, 0., 10.);
    metav01_[ism-1]->setAxisTitle("mean", 1);
    if ( metav02_[ism-1] ) dqmStore_->removeElement( metav02_[ism-1]->getName() );
    sprintf(histo, "EELDT led timing mean L2A %s", Numbers::sEE(ism).c_str());
    metav02_[ism-1] = dqmStore_->book1D(histo, histo, 100, 0., 10.);
    metav02_[ism-1]->setAxisTitle("mean", 1);

    if ( metav05_[ism-1] ) dqmStore_->removeElement( metav05_[ism-1]->getName() );
    sprintf(histo, "EELDT led timing mean L1B %s", Numbers::sEE(ism).c_str());
    metav05_[ism-1] = dqmStore_->book1D(histo, histo, 100, 0., 10.);
    metav05_[ism-1]->setAxisTitle("mean", 1);
    if ( metav06_[ism-1] ) dqmStore_->removeElement( metav06_[ism-1]->getName() );
    sprintf(histo, "EELDT led timing mean L2B %s", Numbers::sEE(ism).c_str());
    metav06_[ism-1] = dqmStore_->book1D(histo, histo, 100, 0., 10.);
    metav06_[ism-1]->setAxisTitle("mean", 1);

    if ( metrms01_[ism-1] ) dqmStore_->removeElement( metrms01_[ism-1]->getName() );
    sprintf(histo, "EELDT led timing rms L1A %s", Numbers::sEE(ism).c_str());
    metrms01_[ism-1] = dqmStore_->book1D(histo, histo, 100, 0., 0.5);
    metrms01_[ism-1]->setAxisTitle("rms", 1);
    if ( metrms02_[ism-1] ) dqmStore_->removeElement( metrms02_[ism-1]->getName() );
    sprintf(histo, "EELDT led timing rms L2A %s", Numbers::sEE(ism).c_str());
    metrms02_[ism-1] = dqmStore_->book1D(histo, histo, 100, 0., 0.5);
    metrms02_[ism-1]->setAxisTitle("rms", 1);

    if ( metrms05_[ism-1] ) dqmStore_->removeElement( metrms05_[ism-1]->getName() );
    sprintf(histo, "EELDT led timing rms L1B %s", Numbers::sEE(ism).c_str());
    metrms05_[ism-1] = dqmStore_->book1D(histo, histo, 100, 0., 0.5);
    metrms05_[ism-1]->setAxisTitle("rms", 1);
    if ( metrms06_[ism-1] ) dqmStore_->removeElement( metrms06_[ism-1]->getName() );
    sprintf(histo, "EELDT led timing rms L2B %s", Numbers::sEE(ism).c_str());
    metrms06_[ism-1] = dqmStore_->book1D(histo, histo, 100, 0., 0.5);
    metrms06_[ism-1]->setAxisTitle("rms", 1);

    if ( meaopn01_[ism-1] ) dqmStore_->removeElement( meaopn01_[ism-1]->getName() );
    sprintf(histo, "EELDT amplitude over PN L1A %s", Numbers::sEE(ism).c_str());
    meaopn01_[ism-1] = dqmStore_->book1D(histo, histo, 850, 0., 850.);
    meaopn01_[ism-1]->setAxisTitle("channel", 1);
    meaopn01_[ism-1]->setAxisTitle("amplitude/PN", 2);
    if ( meaopn02_[ism-1] ) dqmStore_->removeElement( meaopn02_[ism-1]->getName() );
    sprintf(histo, "EELDT amplitude over PN L2A %s", Numbers::sEE(ism).c_str());
    meaopn02_[ism-1] = dqmStore_->book1D(histo, histo, 850, 0., 850.);
    meaopn02_[ism-1]->setAxisTitle("channel", 1);
    meaopn02_[ism-1]->setAxisTitle("amplitude/PN", 2);

    if ( meaopn05_[ism-1] ) dqmStore_->removeElement( meaopn05_[ism-1]->getName() );
    sprintf(histo, "EELDT amplitude over PN L1B %s", Numbers::sEE(ism).c_str());
    meaopn05_[ism-1] = dqmStore_->book1D(histo, histo, 850, 0., 850.);
    meaopn05_[ism-1]->setAxisTitle("channel", 1);
    meaopn05_[ism-1]->setAxisTitle("amplitude/PN", 2);
    if ( meaopn06_[ism-1] ) dqmStore_->removeElement( meaopn06_[ism-1]->getName() );
    sprintf(histo, "EELDT amplitude over PN L2B %s", Numbers::sEE(ism).c_str());
    meaopn06_[ism-1] = dqmStore_->book1D(histo, histo, 850, 0., 850.);
    meaopn06_[ism-1]->setAxisTitle("channel", 1);
    meaopn06_[ism-1]->setAxisTitle("amplitude/PN", 2);

    if ( mepnprms01_[ism-1] ) dqmStore_->removeElement( mepnprms01_[ism-1]->getName() );
    sprintf(histo, "EEPDT PNs pedestal rms %s G01 L1", Numbers::sEE(ism).c_str());
    mepnprms01_[ism-1] = dqmStore_->book1D(histo, histo, 100, 0., 10.);
    mepnprms01_[ism-1]->setAxisTitle("rms", 1);
    if ( mepnprms02_[ism-1] ) dqmStore_->removeElement( mepnprms02_[ism-1]->getName() );
    sprintf(histo, "EEPDT PNs pedestal rms %s G01 L2", Numbers::sEE(ism).c_str());
    mepnprms02_[ism-1] = dqmStore_->book1D(histo, histo, 100, 0., 10.);
    mepnprms02_[ism-1]->setAxisTitle("rms", 1);

    if ( mepnprms05_[ism-1] ) dqmStore_->removeElement( mepnprms05_[ism-1]->getName() );
    sprintf(histo, "EEPDT PNs pedestal rms %s G16 L1", Numbers::sEE(ism).c_str());
    mepnprms05_[ism-1] = dqmStore_->book1D(histo, histo, 100, 0., 10.);
    mepnprms05_[ism-1]->setAxisTitle("rms", 1);
    if ( mepnprms06_[ism-1] ) dqmStore_->removeElement( mepnprms06_[ism-1]->getName() );
    sprintf(histo, "EEPDT PNs pedestal rms %s G16 L2", Numbers::sEE(ism).c_str());
    mepnprms06_[ism-1] = dqmStore_->book1D(histo, histo, 100, 0., 10.);
    mepnprms06_[ism-1]->setAxisTitle("rms", 1);

    if ( me_hs01_[ism-1] ) dqmStore_->removeElement( me_hs01_[ism-1]->getName() );
    sprintf(histo, "EELDT led shape L1A %s", Numbers::sEE(ism).c_str());
    me_hs01_[ism-1] = dqmStore_->book1D(histo, histo, 10, 0., 10.);
    me_hs01_[ism-1]->setAxisTitle("sample", 1);
    me_hs01_[ism-1]->setAxisTitle("amplitude", 2);
    if ( me_hs02_[ism-1] ) dqmStore_->removeElement( me_hs02_[ism-1]->getName() );
    sprintf(histo, "EELDT led shape L2A %s", Numbers::sEE(ism).c_str());
    me_hs02_[ism-1] = dqmStore_->book1D(histo, histo, 10, 0., 10.);
    me_hs02_[ism-1]->setAxisTitle("sample", 1);
    me_hs02_[ism-1]->setAxisTitle("amplitude", 2);

    if ( me_hs05_[ism-1] ) dqmStore_->removeElement( me_hs05_[ism-1]->getName() );
    sprintf(histo, "EELDT led shape L1B %s", Numbers::sEE(ism).c_str());
    me_hs05_[ism-1] = dqmStore_->book1D(histo, histo, 10, 0., 10.);
    me_hs05_[ism-1]->setAxisTitle("sample", 1);
    me_hs05_[ism-1]->setAxisTitle("amplitude", 2);
    if ( me_hs06_[ism-1] ) dqmStore_->removeElement( me_hs06_[ism-1]->getName() );
    sprintf(histo, "EELDT led shape L2B %s", Numbers::sEE(ism).c_str());
    me_hs06_[ism-1] = dqmStore_->book1D(histo, histo, 10, 0., 10.);
    me_hs06_[ism-1]->setAxisTitle("sample", 1);
    me_hs06_[ism-1]->setAxisTitle("amplitude", 2);

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

    if ( mea05_[ism-1] ) mea05_[ism-1]->Reset();
    if ( mea06_[ism-1] ) mea06_[ism-1]->Reset();

    if ( met01_[ism-1] ) met01_[ism-1]->Reset();
    if ( met02_[ism-1] ) met02_[ism-1]->Reset();

    if ( met05_[ism-1] ) met05_[ism-1]->Reset();
    if ( met06_[ism-1] ) met06_[ism-1]->Reset();

    if ( metav01_[ism-1] ) metav01_[ism-1]->Reset();
    if ( metav02_[ism-1] ) metav02_[ism-1]->Reset();

    if ( metav05_[ism-1] ) metav05_[ism-1]->Reset();
    if ( metav06_[ism-1] ) metav06_[ism-1]->Reset();

    if ( metrms01_[ism-1] ) metrms01_[ism-1]->Reset();
    if ( metrms02_[ism-1] ) metrms02_[ism-1]->Reset();

    if ( metrms05_[ism-1] ) metrms05_[ism-1]->Reset();
    if ( metrms06_[ism-1] ) metrms06_[ism-1]->Reset();

    if ( meaopn01_[ism-1] ) meaopn01_[ism-1]->Reset();
    if ( meaopn02_[ism-1] ) meaopn02_[ism-1]->Reset();

    if ( meaopn05_[ism-1] ) meaopn05_[ism-1]->Reset();
    if ( meaopn06_[ism-1] ) meaopn06_[ism-1]->Reset();

    if ( mepnprms01_[ism-1] ) mepnprms01_[ism-1]->Reset();
    if ( mepnprms02_[ism-1] ) mepnprms02_[ism-1]->Reset();

    if ( mepnprms05_[ism-1] ) mepnprms05_[ism-1]->Reset();
    if ( mepnprms06_[ism-1] ) mepnprms06_[ism-1]->Reset();

    if ( me_hs01_[ism-1] ) me_hs01_[ism-1]->Reset();
    if ( me_hs02_[ism-1] ) me_hs02_[ism-1]->Reset();

    if ( me_hs05_[ism-1] ) me_hs05_[ism-1]->Reset();
    if ( me_hs06_[ism-1] ) me_hs06_[ism-1]->Reset();

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

      if ( h13_[ism-1] ) delete h13_[ism-1];
      if ( h14_[ism-1] ) delete h14_[ism-1];
      if ( h15_[ism-1] ) delete h15_[ism-1];
      if ( h16_[ism-1] ) delete h16_[ism-1];

      if ( h21_[ism-1] ) delete h21_[ism-1];
      if ( h22_[ism-1] ) delete h22_[ism-1];

      if ( hs01_[ism-1] ) delete hs01_[ism-1];
      if ( hs02_[ism-1] ) delete hs02_[ism-1];

      if ( hs05_[ism-1] ) delete hs05_[ism-1];
      if ( hs06_[ism-1] ) delete hs06_[ism-1];

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

    h13_[ism-1] = 0;
    h14_[ism-1] = 0;
    h15_[ism-1] = 0;
    h16_[ism-1] = 0;

    h21_[ism-1] = 0;
    h22_[ism-1] = 0;

    hs01_[ism-1] = 0;
    hs02_[ism-1] = 0;

    hs05_[ism-1] = 0;
    hs06_[ism-1] = 0;

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

    if ( mea05_[ism-1] ) dqmStore_->removeElement( mea05_[ism-1]->getName() );
    mea05_[ism-1] = 0;
    if ( mea06_[ism-1] ) dqmStore_->removeElement( mea06_[ism-1]->getName() );
    mea06_[ism-1] = 0;

    if ( met01_[ism-1] ) dqmStore_->removeElement( met01_[ism-1]->getName() );
    met01_[ism-1] = 0;
    if ( met02_[ism-1] ) dqmStore_->removeElement( met02_[ism-1]->getName() );
    met02_[ism-1] = 0;

    if ( met05_[ism-1] ) dqmStore_->removeElement( met05_[ism-1]->getName() );
    met05_[ism-1] = 0;
    if ( met06_[ism-1] ) dqmStore_->removeElement( met06_[ism-1]->getName() );
    met06_[ism-1] = 0;

    if ( metav01_[ism-1] ) dqmStore_->removeElement( metav01_[ism-1]->getName() );
    metav01_[ism-1] = 0;
    if ( metav02_[ism-1] ) dqmStore_->removeElement( metav02_[ism-1]->getName() );
    metav02_[ism-1] = 0;

    if ( metav05_[ism-1] ) dqmStore_->removeElement( metav05_[ism-1]->getName() );
    metav05_[ism-1] = 0;
    if ( metav06_[ism-1] ) dqmStore_->removeElement( metav06_[ism-1]->getName() );
    metav06_[ism-1] = 0;

    if ( metrms01_[ism-1] ) dqmStore_->removeElement( metrms01_[ism-1]->getName() );
    metrms01_[ism-1] = 0;
    if ( metrms02_[ism-1] ) dqmStore_->removeElement( metrms02_[ism-1]->getName() );
    metrms02_[ism-1] = 0;

    if ( metrms05_[ism-1] ) dqmStore_->removeElement( metrms05_[ism-1]->getName() );
    metrms05_[ism-1] = 0;
    if ( metrms06_[ism-1] ) dqmStore_->removeElement( metrms06_[ism-1]->getName() );
    metrms06_[ism-1] = 0;

    if ( meaopn01_[ism-1] ) dqmStore_->removeElement( meaopn01_[ism-1]->getName() );
    meaopn01_[ism-1] = 0;
    if ( meaopn02_[ism-1] ) dqmStore_->removeElement( meaopn02_[ism-1]->getName() );
    meaopn02_[ism-1] = 0;

    if ( meaopn05_[ism-1] ) dqmStore_->removeElement( meaopn05_[ism-1]->getName() );
    meaopn05_[ism-1] = 0;
    if ( meaopn06_[ism-1] ) dqmStore_->removeElement( meaopn06_[ism-1]->getName() );
    meaopn06_[ism-1] = 0;

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

    if ( me_hs05_[ism-1] ) dqmStore_->removeElement( me_hs05_[ism-1]->getName() );
    me_hs05_[ism-1] = 0;
    if ( me_hs06_[ism-1] ) dqmStore_->removeElement( me_hs06_[ism-1]->getName() );
    me_hs06_[ism-1] = 0;

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
      UtilsClient::printBadChannels(meg01_[ism-1], h01_[ism-1]);
      UtilsClient::printBadChannels(meg01_[ism-1], h13_[ism-1]);
      UtilsClient::printBadChannels(meg02_[ism-1], h03_[ism-1]);
      UtilsClient::printBadChannels(meg02_[ism-1], h15_[ism-1]);
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

        bool update09;
        bool update10;
        bool update11;
        bool update12;

        float num01, num02, num03, num04;
        float mean01, mean02, mean03, mean04;
        float rms01, rms02, rms03, rms04;

        float num09, num10, num11, num12;
        float mean09, mean10, mean11, mean12;
        float rms09, rms10, rms11, rms12;

        update01 = UtilsClient::getBinStatistics(h01_[ism-1], ix, iy, num01, mean01, rms01);
        update02 = UtilsClient::getBinStatistics(h02_[ism-1], ix, iy, num02, mean02, rms02);
        update03 = UtilsClient::getBinStatistics(h03_[ism-1], ix, iy, num03, mean03, rms03);
        update04 = UtilsClient::getBinStatistics(h04_[ism-1], ix, iy, num04, mean04, rms04);

        update09 = UtilsClient::getBinStatistics(h13_[ism-1], ix, iy, num09, mean09, rms09);
        update10 = UtilsClient::getBinStatistics(h14_[ism-1], ix, iy, num10, mean10, rms10);
        update11 = UtilsClient::getBinStatistics(h15_[ism-1], ix, iy, num11, mean11, rms11);
        update12 = UtilsClient::getBinStatistics(h16_[ism-1], ix, iy, num12, mean12, rms12);

        if ( update01 || update02 ) {

          if ( Numbers::icEE(ism, jx, jy) == 1 ) {

            if ( verbose_ ) {
              cout << "Preparing dataset for " << Numbers::sEE(ism) << " (ism=" << ism << ")" << endl;
              cout << "L1A (" << Numbers::ix0EE(i+1)+ix << "," << Numbers::iy0EE(i+1)+iy << ") " << num01 << " " << mean01 << " " << rms01 << endl;
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

        if ( update09 || update10 ) {

          if ( Numbers::icEE(ism, jx, jy) == 1 ) {

            if ( verbose_ ) {
              cout << "Preparing dataset for " << Numbers::sEE(ism) << " (ism=" << ism << ")" << endl;
              cout << "L1B (" << Numbers::ix0EE(i+1)+ix << "," << Numbers::iy0EE(i+1)+iy << ") " << num09 << " " << mean09 << " " << rms09 << endl;
              cout << endl;
            }

          }

          vpt_l1.setVPTMean(mean09);
          vpt_l1.setVPTRMS(rms09);

          vpt_l1.setVPTOverPNMean(mean10);
          vpt_l1.setVPTOverPNRMS(rms10);

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
              cout << "L2A (" << Numbers::ix0EE(i+1)+ix << "," << Numbers::iy0EE(i+1)+iy << ") " << num03 << " " << mean03 << " " << rms03 << endl;
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

        if ( update11 || update12 ) {

          if ( Numbers::icEE(ism, jx, jy) == 1 ) {

            if ( verbose_ ) {
              cout << "Preparing dataset for " << Numbers::sEE(ism) << " (ism=" << ism << ")" << endl;
              cout << "L2B (" << Numbers::ix0EE(i+1)+ix << "," << Numbers::iy0EE(i+1)+iy << ") " << num11 << " " << mean11 << " " << rms11 << endl;
              cout << endl;
            }

          }

          vpt_l2.setVPTMean(mean11);
          vpt_l2.setVPTRMS(rms11);

          vpt_l2.setVPTOverPNMean(mean12);
          vpt_l2.setVPTOverPNRMS(rms12);

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
      UtilsClient::printBadChannels(meg05_[ism-1], i01_[ism-1]);
      UtilsClient::printBadChannels(meg05_[ism-1], i05_[ism-1]);
      UtilsClient::printBadChannels(meg06_[ism-1], i02_[ism-1]);
      UtilsClient::printBadChannels(meg06_[ism-1], i06_[ism-1]);

      UtilsClient::printBadChannels(meg09_[ism-1], i09_[ism-1]);
      UtilsClient::printBadChannels(meg09_[ism-1], i13_[ism-1]);
      UtilsClient::printBadChannels(meg10_[ism-1], i10_[ism-1]);
      UtilsClient::printBadChannels(meg10_[ism-1], i14_[ism-1]);
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

        bool update05;
        bool update06;

        float num01, num02, num05, num06;
        float mean01, mean02, mean05, mean06;
        float rms01, rms02, rms05, rms06;

        update01 = UtilsClient::getBinStatistics(h09_[ism-1], ix, iy, num01, mean01, rms01);
        update02 = UtilsClient::getBinStatistics(h10_[ism-1], ix, iy, num02, mean02, rms02);

        update05 = UtilsClient::getBinStatistics(h21_[ism-1], ix, iy, num05, mean05, rms05);
        update06 = UtilsClient::getBinStatistics(h22_[ism-1], ix, iy, num06, mean06, rms06);

        if ( update01 ) {

          if ( Numbers::icEE(ism, ix, iy) == 1 ) {

            if ( verbose_ ) {
              cout << "Preparing dataset for " << Numbers::sEE(ism) << " (ism=" << ism << ")" << endl;
              cout << "L1A crystal (" << Numbers::ix0EE(i+1)+ix << "," << Numbers::iy0EE(i+1)+iy << ") " << num01  << " " << mean01 << " " << rms01  << endl;
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

        if ( update05 ) {

          if ( Numbers::icEE(ism, ix, iy) == 1 ) {

            if ( verbose_ ) {
              cout << "Preparing dataset for " << Numbers::sEE(ism) << " (ism=" << ism << ")" << endl;
              cout << "L1B crystal (" << Numbers::ix0EE(i+1)+ix << "," << Numbers::iy0EE(i+1)+iy << ") " << num05  << " " << mean05 << " " << rms05  << endl;
              cout << endl;
            }

          }

          t_l1.setTimingMean(mean05);
          t_l1.setTimingRMS(rms05);

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
              cout << "L2A crystal (" << Numbers::ix0EE(i+1)+ix << "," << Numbers::iy0EE(i+1)+iy << ") " << num02  << " " << mean02 << " " << rms02  << endl;
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

        if ( update06 ) {

          if ( Numbers::icEE(ism, ix, iy) == 1 ) {

            if ( verbose_ ) {
              cout << "Preparing dataset for " << Numbers::sEE(ism) << " (ism=" << ism << ")" << endl;
              cout << "L2B crystal (" << Numbers::ix0EE(i+1)+ix << "," << Numbers::iy0EE(i+1)+iy << ") " << num06  << " " << mean06 << " " << rms06  << endl;
              cout << endl;
            }

          }

          t_l2.setTimingMean(mean06);
          t_l2.setTimingRMS(rms06);

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

    sprintf(histo, (prefixME_ + "/EELedTask/Led1/EELDT amplitude %s L1A").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    h01_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h01_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELedTask/Led1/EELDT amplitude over PN %s L1A").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    h02_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h02_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELedTask/Led2/EELDT amplitude %s L2A").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    h03_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h03_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELedTask/Led2/EELDT amplitude over PN %s L2A").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    h04_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h04_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELedTask/Led1/EELDT timing %s L1A").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    h09_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h09_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELedTask/Led2/EELDT timing %s L2A").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    h10_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h10_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELedTask/Led1/EELDT amplitude %s L1B").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    h13_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h13_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELedTask/Led1/EELDT amplitude over PN %s L1B").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    h14_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h14_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELedTask/Led2/EELDT amplitude %s L2B").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    h15_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h15_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELedTask/Led2/EELDT amplitude over PN %s L2B").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    h16_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h16_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELedTask/Led1/EELDT timing %s L1B").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    h21_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h21_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELedTask/Led2/EELDT timing %s L2B").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    h22_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h22_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELedTask/Led1/EELDT shape %s L1A").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    hs01_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, hs01_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELedTask/Led2/EELDT shape %s L2A").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    hs02_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, hs02_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELedTask/Led1/EELDT shape %s L1B").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    hs05_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, hs05_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELedTask/Led2/EELDT shape %s L2B").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    hs06_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, hs06_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELedTask/Led1/PN/Gain01/EEPDT PNs amplitude %s G01 L1").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    i01_[ism-1] = UtilsClient::getHisto<TProfile*>( me, cloneME_, i01_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELedTask/Led2/PN/Gain01/EEPDT PNs amplitude %s G01 L2").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    i02_[ism-1] = UtilsClient::getHisto<TProfile*>( me, cloneME_, i02_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELedTask/Led1/PN/Gain01/EEPDT PNs pedestal %s G01 L1").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    i05_[ism-1] = UtilsClient::getHisto<TProfile*>( me, cloneME_, i05_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELedTask/Led2/PN/Gain01/EEPDT PNs pedestal %s G01 L2").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    i06_[ism-1] = UtilsClient::getHisto<TProfile*>( me, cloneME_, i06_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELedTask/Led1/PN/Gain16/EEPDT PNs amplitude %s G16 L1").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    i09_[ism-1] = UtilsClient::getHisto<TProfile*>( me, cloneME_, i09_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELedTask/Led2/PN/Gain16/EEPDT PNs amplitude %s G16 L2").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    i10_[ism-1] = UtilsClient::getHisto<TProfile*>( me, cloneME_, i10_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELedTask/Led1/PN/Gain16/EEPDT PNs pedestal %s G16 L1").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    i13_[ism-1] = UtilsClient::getHisto<TProfile*>( me, cloneME_, i13_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELedTask/Led2/PN/Gain16/EEPDT PNs pedestal %s G16 L2").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    i14_[ism-1] = UtilsClient::getHisto<TProfile*>( me, cloneME_, i14_[ism-1] );

    if ( meg01_[ism-1] ) meg01_[ism-1]->Reset();
    if ( meg02_[ism-1] ) meg02_[ism-1]->Reset();

    if ( meg05_[ism-1] ) meg05_[ism-1]->Reset();
    if ( meg06_[ism-1] ) meg06_[ism-1]->Reset();

    if ( meg09_[ism-1] ) meg09_[ism-1]->Reset();
    if ( meg10_[ism-1] ) meg10_[ism-1]->Reset();

    if ( mea01_[ism-1] ) mea01_[ism-1]->Reset();
    if ( mea02_[ism-1] ) mea02_[ism-1]->Reset();

    if ( mea05_[ism-1] ) mea05_[ism-1]->Reset();
    if ( mea06_[ism-1] ) mea06_[ism-1]->Reset();

    if ( met01_[ism-1] ) met01_[ism-1]->Reset();
    if ( met02_[ism-1] ) met02_[ism-1]->Reset();

    if ( met05_[ism-1] ) met05_[ism-1]->Reset();
    if ( met06_[ism-1] ) met06_[ism-1]->Reset();

    if ( metav01_[ism-1] ) metav01_[ism-1]->Reset();
    if ( metav02_[ism-1] ) metav02_[ism-1]->Reset();

    if ( metav05_[ism-1] ) metav05_[ism-1]->Reset();
    if ( metav06_[ism-1] ) metav06_[ism-1]->Reset();

    if ( metrms01_[ism-1] ) metrms01_[ism-1]->Reset();
    if ( metrms02_[ism-1] ) metrms02_[ism-1]->Reset();

    if ( metrms05_[ism-1] ) metrms05_[ism-1]->Reset();
    if ( metrms06_[ism-1] ) metrms06_[ism-1]->Reset();

    if ( meaopn01_[ism-1] ) meaopn01_[ism-1]->Reset();
    if ( meaopn02_[ism-1] ) meaopn02_[ism-1]->Reset();

    if ( meaopn05_[ism-1] ) meaopn05_[ism-1]->Reset();
    if ( meaopn06_[ism-1] ) meaopn06_[ism-1]->Reset();

    if ( mepnprms01_[ism-1] ) mepnprms01_[ism-1]->Reset();
    if ( mepnprms02_[ism-1] ) mepnprms02_[ism-1]->Reset();

    if ( mepnprms05_[ism-1] ) mepnprms05_[ism-1]->Reset();
    if ( mepnprms06_[ism-1] ) mepnprms06_[ism-1]->Reset();

    if ( me_hs01_[ism-1] ) me_hs01_[ism-1]->Reset();
    if ( me_hs02_[ism-1] ) me_hs02_[ism-1]->Reset();

    if ( me_hs05_[ism-1] ) me_hs05_[ism-1]->Reset();
    if ( me_hs06_[ism-1] ) me_hs06_[ism-1]->Reset();

    float meanAmplL1A, meanAmplL2A;
    float meanAmplL1B, meanAmplL2B;

    int nCryL1A, nCryL2A;
    int nCryL1B, nCryL2B;

    meanAmplL1A = meanAmplL2A = 0.;
    meanAmplL1B = meanAmplL2B = 0.;

    nCryL1A = nCryL2A = 0;
    nCryL1B = nCryL2B = 0;

    for ( int ix = 1; ix <= 50; ix++ ) {
      for ( int iy = 1; iy <= 50; iy++ ) {

        bool update01;
        bool update02;

        bool update05;
        bool update06;

        float num01, num02, num05, num06;
        float mean01, mean02, mean05, mean06;
        float rms01, rms02, rms05, rms06;

        update01 = UtilsClient::getBinStatistics(h01_[ism-1], ix, iy, num01, mean01, rms01);
        update02 = UtilsClient::getBinStatistics(h03_[ism-1], ix, iy, num02, mean02, rms02);

        update05 = UtilsClient::getBinStatistics(h13_[ism-1], ix, iy, num05, mean05, rms05);
        update06 = UtilsClient::getBinStatistics(h15_[ism-1], ix, iy, num06, mean06, rms06);

        if ( update01 ) {
          meanAmplL1A += mean01;
          nCryL1A++;
        }

        if ( update02 ) {
          meanAmplL2A += mean02;
          nCryL2A++;
        }

        if ( update05 ) {
          meanAmplL1B += mean05;
          nCryL1B++;
        }

        if ( update06 ) {
          meanAmplL2B += mean06;
          nCryL2B++;
        }

      }
    }

    if ( nCryL1A > 0 ) meanAmplL1A /= float (nCryL1A);
    if ( nCryL2A > 0 ) meanAmplL2A /= float (nCryL2A);

    if ( nCryL1B > 0 ) meanAmplL1B /= float (nCryL1B);
    if ( nCryL2B > 0 ) meanAmplL2B /= float (nCryL2B);

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

        bool update13;
        bool update14;
        bool update15;
        bool update16;

        bool update21;
        bool update22;

        float num01, num02, num03, num04;
        float num09, num10;
        float mean01, mean02, mean03, mean04;
        float mean09, mean10;
        float rms01, rms02, rms03, rms04;
        float rms09, rms10;

        float num13, num14, num15, num16;
        float num21, num22;
        float mean13, mean14, mean15, mean16;
        float mean21, mean22;
        float rms13, rms14, rms15, rms16;
        float rms21, rms22;

        update01 = UtilsClient::getBinStatistics(h01_[ism-1], ix, iy, num01, mean01, rms01);
        update02 = UtilsClient::getBinStatistics(h02_[ism-1], ix, iy, num02, mean02, rms02);
        update03 = UtilsClient::getBinStatistics(h03_[ism-1], ix, iy, num03, mean03, rms03);
        update04 = UtilsClient::getBinStatistics(h04_[ism-1], ix, iy, num04, mean04, rms04);

        update09 = UtilsClient::getBinStatistics(h09_[ism-1], ix, iy, num09, mean09, rms09);
        update10 = UtilsClient::getBinStatistics(h10_[ism-1], ix, iy, num10, mean10, rms10);

        // other SM half

        update13 = UtilsClient::getBinStatistics(h13_[ism-1], ix, iy, num13, mean13, rms13);
        update14 = UtilsClient::getBinStatistics(h14_[ism-1], ix, iy, num14, mean14, rms14);
        update15 = UtilsClient::getBinStatistics(h15_[ism-1], ix, iy, num15, mean15, rms15);
        update16 = UtilsClient::getBinStatistics(h16_[ism-1], ix, iy, num16, mean16, rms16);

        update21 = UtilsClient::getBinStatistics(h21_[ism-1], ix, iy, num21, mean21, rms21);
        update22 = UtilsClient::getBinStatistics(h22_[ism-1], ix, iy, num22, mean22, rms22);

        if ( update01 ) {

          float val;

          val = 1.;
          if ( fabs(mean01 - meanAmplL1A) > fabs(percentVariation_ * meanAmplL1A) || mean01 < amplitudeThreshold_ )
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

        if ( update13 ) {

          float val;

          val = 1.;
          if ( fabs(mean13 - meanAmplL1B) > fabs(percentVariation_ * meanAmplL1B) || mean13 < amplitudeThreshold_ )
           val = 0.;
          if ( meg01_[ism-1] ) meg01_[ism-1]->setBinContent( ix, iy, val );

          int ic = Numbers::icEE(ism, jx, jy);

          if ( ic != -1 ) {
            if ( mea05_[ism-1] ) {
              if ( mean13 > 0. ) {
                mea05_[ism-1]->setBinContent( ic, mean13 );
                mea05_[ism-1]->setBinError( ic, rms13 );
            } else {
                mea05_[ism-1]->setEntries( 1.+mea05_[ism-1]->getEntries() );
              }
            }
          }

        }

        if ( update03 ) {

          float val;

          val = 1.;
          if ( fabs(mean03 - meanAmplL2A) > fabs(percentVariation_ * meanAmplL2A) || mean03 < amplitudeThreshold_ )
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

        if ( update15 ) {

          float val;

          val = 1.;
          if ( fabs(mean15 - meanAmplL2B) > fabs(percentVariation_ * meanAmplL2B) || mean15 < amplitudeThreshold_ )
            val = 0.;
          if ( meg02_[ism-1] ) meg02_[ism-1]->setBinContent( ix, iy, val);

          int ic = Numbers::icEE(ism, jx, jy);

          if ( ic != -1 ) {
            if ( mea06_[ism-1] ) {
              if ( mean15 > 0. ) {
                mea06_[ism-1]->setBinContent( ic, mean15 );
                mea06_[ism-1]->setBinError( ic, rms15 );
              } else {
                mea06_[ism-1]->setEntries( 1.+mea06_[ism-1]->getEntries() );
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

        if ( update14 ) {

          int ic = Numbers::icEE(ism, jx, jy);

          if ( ic != -1 ) {
            if ( meaopn05_[ism-1] ) {
              if ( mean14 > 0. ) {
                meaopn05_[ism-1]->setBinContent( ic, mean14 );
                meaopn05_[ism-1]->setBinError( ic, rms14 );
              } else {
                meaopn05_[ism-1]->setEntries( 1.+meaopn05_[ism-1]->getEntries() );
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

        if ( update16 ) {

          int ic = Numbers::icEE(ism, jx, jy);

          if ( ic != -1 ) {
            if ( meaopn06_[ism-1] ) {
              if ( mean16 > 0. ) {
                meaopn06_[ism-1]->setBinContent( ic, mean16 );
                meaopn06_[ism-1]->setBinError( ic, rms16 );
              } else {
                meaopn06_[ism-1]->setEntries( 1.+meaopn06_[ism-1]->getEntries() );
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

        if ( update21 ) {

          int ic = Numbers::icEE(ism, jx, jy);

          if ( ic != -1 ) {
            if ( met05_[ism-1] ) {
              if ( mean21 > 0. ) {
                met05_[ism-1]->setBinContent( ic, mean21 );
                met05_[ism-1]->setBinError( ic, rms21 );
              } else {
                met05_[ism-1]->setEntries(1.+met05_[ism-1]->getEntries());
              }
            }

            if ( metav05_[ism-1] ) metav05_[ism-1] ->Fill(mean21);
            if ( metrms05_[ism-1] ) metrms05_[ism-1]->Fill(rms21);
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

        if ( update22 ) {

          int ic = Numbers::icEE(ism, jx, jy);

          if ( ic != -1 ) {
            if ( met06_[ism-1] ) {
              if ( mean22 > 0. ) {
                met06_[ism-1]->setBinContent( ic, mean22 );
                met06_[ism-1]->setBinError( ic, rms22 );
              } else {
                met06_[ism-1]->setEntries(1.+met06_[ism-1]->getEntries());
              }
            }

            if ( metav06_[ism-1] ) metav06_[ism-1] ->Fill(mean22);
            if ( metrms06_[ism-1] ) metrms06_[ism-1]->Fill(rms22);
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

      if ( hs05_[ism-1] ) {
        int ic = UtilsClient::getFirstNonEmptyChannel( hs05_[ism-1] );
        if ( me_hs05_[ism-1] ) {
          me_hs05_[ism-1]->setBinContent( i, hs05_[ism-1]->GetBinContent(ic, i) );
          me_hs05_[ism-1]->setBinError( i, hs05_[ism-1]->GetBinError(ic, i) );
        }
      }

      if ( hs06_[ism-1] ) {
        int ic = UtilsClient::getFirstNonEmptyChannel( hs06_[ism-1] );
        if ( me_hs06_[ism-1] ) {
          me_hs06_[ism-1]->setBinContent( i, hs06_[ism-1]->GetBinContent(ic, i) );
          me_hs06_[ism-1]->setBinError( i, hs06_[ism-1]->GetBinError(ic, i) );
        }
      }

    }

  }

}

void EELedClient::softReset(bool flag) {

}

