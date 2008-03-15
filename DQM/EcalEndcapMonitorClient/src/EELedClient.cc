/*
 * \file EELedClient.cc
 *
 * $Date: 2008/03/14 14:38:58 $
 * $Revision: 1.70 $
 * \author G. Della Ricca
 * \author G. Franzoni
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

#include "OnlineDB/EcalCondDB/interface/MonLed1Dat.h"
#include "OnlineDB/EcalCondDB/interface/MonLed2Dat.h"
#include "OnlineDB/EcalCondDB/interface/MonPNLed1Dat.h"
#include "OnlineDB/EcalCondDB/interface/MonPNLed2Dat.h"
#include "OnlineDB/EcalCondDB/interface/RunCrystalErrorsDat.h"
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

EELedClient::EELedClient(const ParameterSet& ps){

  // cloneME switch
  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);

  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

  // enableCleanup_ switch
  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  // prefix to ME paths
  prefixME_ = ps.getUntrackedParameter<string>("prefixME", "");

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

  amplitudeThresholdPnG01_ = 50.;
  amplitudeThresholdPnG16_ = 50.;

  pedPnExpectedMean_[0] = 750.0;
  pedPnExpectedMean_[1] = 750.0;

  pedPnDiscrepancyMean_[0] = 100.0;
  pedPnDiscrepancyMean_[1] = 100.0;

  pedPnRMSThreshold_[0] = 1.0; // value at h4; expected nominal: 0.5
  pedPnRMSThreshold_[1] = 3.0; // value at h4; expected nominal: 1.6

}

EELedClient::~EELedClient(){

}

void EELedClient::beginJob(DQMStore* dbe){

  dbe_ = dbe;

  if ( verbose_ ) cout << "EELedClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EELedClient::beginRun(void){

  if ( verbose_ ) cout << "EELedClient: beginRun" << endl;

  jevt_ = 0;

  this->setup();

}

void EELedClient::endJob(void) {

  if ( verbose_ ) cout << "EELedClient: endJob, ievt = " << ievt_ << endl;

  this->cleanup();

}

void EELedClient::endRun(void) {

  if ( verbose_ ) cout << "EELedClient: endRun, jevt = " << jevt_ << endl;

  this->cleanup();

}

void EELedClient::setup(void) {

  char histo[200];

  dbe_->setCurrentFolder( "EcalEndcap/EELedClient" );

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( meg01_[ism-1] ) dbe_->removeElement( meg01_[ism-1]->getName() );
    sprintf(histo, "EELDT led quality L1 %s", Numbers::sEE(ism).c_str());
    meg01_[ism-1] = dbe_->book2D(histo, histo, 50, Numbers::ix0EE(ism)+0., Numbers::ix0EE(ism)+50., 50, Numbers::iy0EE(ism)+0., Numbers::iy0EE(ism)+50.);
    meg01_[ism-1]->setAxisTitle("jx", 1);
    meg01_[ism-1]->setAxisTitle("jy", 2);
    if ( meg02_[ism-1] ) dbe_->removeElement( meg02_[ism-1]->getName() );
    sprintf(histo, "EELDT led quality L2 %s", Numbers::sEE(ism).c_str());
    meg02_[ism-1] = dbe_->book2D(histo, histo, 50, Numbers::ix0EE(ism)+0., Numbers::ix0EE(ism)+50., 50, Numbers::iy0EE(ism)+0., Numbers::iy0EE(ism)+50.);
    meg02_[ism-1]->setAxisTitle("jx", 1);
    meg02_[ism-1]->setAxisTitle("jy", 2);

    if ( meg05_[ism-1] ) dbe_->removeElement( meg05_[ism-1]->getName() );
    sprintf(histo, "EELDT led quality L1 PNs G01 %s", Numbers::sEE(ism).c_str());
    meg05_[ism-1] = dbe_->book2D(histo, histo, 10, 0., 10., 1, 0., 5.);
    meg05_[ism-1]->setAxisTitle("pseudo-strip", 1);
    meg05_[ism-1]->setAxisTitle("channel", 2);
    if ( meg06_[ism-1] ) dbe_->removeElement( meg06_[ism-1]->getName() );
    sprintf(histo, "EELDT led quality L2 PNs G01 %s", Numbers::sEE(ism).c_str());
    meg06_[ism-1] = dbe_->book2D(histo, histo, 10, 0., 10., 1, 0., 5.);
    meg06_[ism-1]->setAxisTitle("pseudo-strip", 1);
    meg06_[ism-1]->setAxisTitle("channel", 2);

    if ( meg09_[ism-1] ) dbe_->removeElement( meg09_[ism-1]->getName() );
    sprintf(histo, "EELDT led quality L1 PNs G16 %s", Numbers::sEE(ism).c_str());
    meg09_[ism-1] = dbe_->book2D(histo, histo, 10, 0., 10., 1, 0., 5.);
    meg09_[ism-1]->setAxisTitle("pseudo-strip", 1);
    meg09_[ism-1]->setAxisTitle("channel", 2);
    if ( meg10_[ism-1] ) dbe_->removeElement( meg10_[ism-1]->getName() );
    sprintf(histo, "EELDT led quality L2 PNs G16 %s", Numbers::sEE(ism).c_str());
    meg10_[ism-1] = dbe_->book2D(histo, histo, 10, 0., 10., 1, 0., 5.);
    meg10_[ism-1]->setAxisTitle("pseudo-strip", 1);
    meg10_[ism-1]->setAxisTitle("channel", 2);

    if ( mea01_[ism-1] ) dbe_->removeElement( mea01_[ism-1]->getName() );;
    sprintf(histo, "EELDT amplitude L1A %s", Numbers::sEE(ism).c_str());
    mea01_[ism-1] = dbe_->book1D(histo, histo, 850, 0., 850.);
    mea01_[ism-1]->setAxisTitle("channel", 1);
    mea01_[ism-1]->setAxisTitle("amplitude", 2);
    if ( mea02_[ism-1] ) dbe_->removeElement( mea02_[ism-1]->getName() );
    sprintf(histo, "EELDT amplitude L2A %s", Numbers::sEE(ism).c_str());
    mea02_[ism-1] = dbe_->book1D(histo, histo, 850, 0., 850.);
    mea02_[ism-1]->setAxisTitle("channel", 1);
    mea02_[ism-1]->setAxisTitle("amplitude", 2);

    if ( mea05_[ism-1] ) dbe_->removeElement( mea05_[ism-1]->getName() );;
    sprintf(histo, "EELDT amplitude L1B %s", Numbers::sEE(ism).c_str());
    mea05_[ism-1] = dbe_->book1D(histo, histo, 850, 0., 850.);
    mea05_[ism-1]->setAxisTitle("channel", 1);
    mea05_[ism-1]->setAxisTitle("amplitude", 2);
    if ( mea06_[ism-1] ) dbe_->removeElement( mea06_[ism-1]->getName() );
    sprintf(histo, "EELDT amplitude L2B %s", Numbers::sEE(ism).c_str());
    mea06_[ism-1] = dbe_->book1D(histo, histo, 850, 0., 850.);
    mea06_[ism-1]->setAxisTitle("channel", 1);
    mea06_[ism-1]->setAxisTitle("amplitude", 2);

    if ( met01_[ism-1] ) dbe_->removeElement( met01_[ism-1]->getName() );
    sprintf(histo, "EELDT led timing L1A %s", Numbers::sEE(ism).c_str());
    met01_[ism-1] = dbe_->book1D(histo, histo, 850, 0., 850.);
    met01_[ism-1]->setAxisTitle("channel", 1);
    met01_[ism-1]->setAxisTitle("jitter", 2);
    if ( met02_[ism-1] ) dbe_->removeElement( met02_[ism-1]->getName() );
    sprintf(histo, "EELDT led timing L2A %s", Numbers::sEE(ism).c_str());
    met02_[ism-1] = dbe_->book1D(histo, histo, 850, 0., 850.);
    met02_[ism-1]->setAxisTitle("channel", 1);
    met02_[ism-1]->setAxisTitle("jitter", 2);

    if ( met05_[ism-1] ) dbe_->removeElement( met05_[ism-1]->getName() );
    sprintf(histo, "EELDT led timing L1B %s", Numbers::sEE(ism).c_str());
    met05_[ism-1] = dbe_->book1D(histo, histo, 850, 0., 850.);
    met05_[ism-1]->setAxisTitle("channel", 1);
    met05_[ism-1]->setAxisTitle("jitter", 2);
    if ( met06_[ism-1] ) dbe_->removeElement( met06_[ism-1]->getName() );
    sprintf(histo, "EELDT led timing L2B %s", Numbers::sEE(ism).c_str());
    met06_[ism-1] = dbe_->book1D(histo, histo, 850, 0., 850.);
    met06_[ism-1]->setAxisTitle("channel", 1);
    met06_[ism-1]->setAxisTitle("jitter", 2);

    if ( metav01_[ism-1] ) dbe_->removeElement( metav01_[ism-1]->getName() );
    sprintf(histo, "EELDT led timing mean L1A %s", Numbers::sEE(ism).c_str());
    metav01_[ism-1] = dbe_->book1D(histo, histo, 100, 0., 10.);
    metav01_[ism-1]->setAxisTitle("mean", 1);
    if ( metav02_[ism-1] ) dbe_->removeElement( metav02_[ism-1]->getName() );
    sprintf(histo, "EELDT led timing mean L2A %s", Numbers::sEE(ism).c_str());
    metav02_[ism-1] = dbe_->book1D(histo, histo, 100, 0., 10.);
    metav02_[ism-1]->setAxisTitle("mean", 1);

    if ( metav05_[ism-1] ) dbe_->removeElement( metav05_[ism-1]->getName() );
    sprintf(histo, "EELDT led timing mean L1B %s", Numbers::sEE(ism).c_str());
    metav05_[ism-1] = dbe_->book1D(histo, histo, 100, 0., 10.);
    metav05_[ism-1]->setAxisTitle("mean", 1);
    if ( metav06_[ism-1] ) dbe_->removeElement( metav06_[ism-1]->getName() );
    sprintf(histo, "EELDT led timing mean L2B %s", Numbers::sEE(ism).c_str());
    metav06_[ism-1] = dbe_->book1D(histo, histo, 100, 0., 10.);
    metav06_[ism-1]->setAxisTitle("mean", 1);

    if ( metrms01_[ism-1] ) dbe_->removeElement( metrms01_[ism-1]->getName() );
    sprintf(histo, "EELDT led timing rms L1A %s", Numbers::sEE(ism).c_str());
    metrms01_[ism-1] = dbe_->book1D(histo, histo, 100, 0., 0.5);
    metrms01_[ism-1]->setAxisTitle("rms", 1);
    if ( metrms02_[ism-1] ) dbe_->removeElement( metrms02_[ism-1]->getName() );
    sprintf(histo, "EELDT led timing rms L2A %s", Numbers::sEE(ism).c_str());
    metrms02_[ism-1] = dbe_->book1D(histo, histo, 100, 0., 0.5);
    metrms02_[ism-1]->setAxisTitle("rms", 1);

    if ( metrms05_[ism-1] ) dbe_->removeElement( metrms05_[ism-1]->getName() );
    sprintf(histo, "EELDT led timing rms L1B %s", Numbers::sEE(ism).c_str());
    metrms05_[ism-1] = dbe_->book1D(histo, histo, 100, 0., 0.5);
    metrms05_[ism-1]->setAxisTitle("rms", 1);
    if ( metrms06_[ism-1] ) dbe_->removeElement( metrms06_[ism-1]->getName() );
    sprintf(histo, "EELDT led timing rms L2B %s", Numbers::sEE(ism).c_str());
    metrms06_[ism-1] = dbe_->book1D(histo, histo, 100, 0., 0.5);
    metrms06_[ism-1]->setAxisTitle("rms", 1);

    if ( meaopn01_[ism-1] ) dbe_->removeElement( meaopn01_[ism-1]->getName() );
    sprintf(histo, "EELDT amplitude over PN L1A %s", Numbers::sEE(ism).c_str());
    meaopn01_[ism-1] = dbe_->book1D(histo, histo, 850, 0., 850.);
    meaopn01_[ism-1]->setAxisTitle("channel", 1);
    meaopn01_[ism-1]->setAxisTitle("amplitude/PN", 2);
    if ( meaopn02_[ism-1] ) dbe_->removeElement( meaopn02_[ism-1]->getName() );
    sprintf(histo, "EELDT amplitude over PN L2A %s", Numbers::sEE(ism).c_str());
    meaopn02_[ism-1] = dbe_->book1D(histo, histo, 850, 0., 850.);
    meaopn02_[ism-1]->setAxisTitle("channel", 1);
    meaopn02_[ism-1]->setAxisTitle("amplitude/PN", 2);

    if ( meaopn05_[ism-1] ) dbe_->removeElement( meaopn05_[ism-1]->getName() );
    sprintf(histo, "EELDT amplitude over PN L1B %s", Numbers::sEE(ism).c_str());
    meaopn05_[ism-1] = dbe_->book1D(histo, histo, 850, 0., 850.);
    meaopn05_[ism-1]->setAxisTitle("channel", 1);
    meaopn05_[ism-1]->setAxisTitle("amplitude/PN", 2);
    if ( meaopn06_[ism-1] ) dbe_->removeElement( meaopn06_[ism-1]->getName() );
    sprintf(histo, "EELDT amplitude over PN L2B %s", Numbers::sEE(ism).c_str());
    meaopn06_[ism-1] = dbe_->book1D(histo, histo, 850, 0., 850.);
    meaopn06_[ism-1]->setAxisTitle("channel", 1);
    meaopn06_[ism-1]->setAxisTitle("amplitude/PN", 2);

    if ( mepnprms01_[ism-1] ) dbe_->removeElement( mepnprms01_[ism-1]->getName() );
    sprintf(histo, "EEPDT PNs pedestal rms %s G01 L1", Numbers::sEE(ism).c_str());
    mepnprms01_[ism-1] = dbe_->book1D(histo, histo, 100, 0., 10.);
    mepnprms01_[ism-1]->setAxisTitle("rms", 1);
    if ( mepnprms02_[ism-1] ) dbe_->removeElement( mepnprms02_[ism-1]->getName() );
    sprintf(histo, "EEPDT PNs pedestal rms %s G01 L2", Numbers::sEE(ism).c_str());
    mepnprms02_[ism-1] = dbe_->book1D(histo, histo, 100, 0., 10.);
    mepnprms02_[ism-1]->setAxisTitle("rms", 1);

    if ( mepnprms05_[ism-1] ) dbe_->removeElement( mepnprms05_[ism-1]->getName() );
    sprintf(histo, "EEPDT PNs pedestal rms %s G16 L1", Numbers::sEE(ism).c_str());
    mepnprms05_[ism-1] = dbe_->book1D(histo, histo, 100, 0., 10.);
    mepnprms05_[ism-1]->setAxisTitle("rms", 1);
    if ( mepnprms06_[ism-1] ) dbe_->removeElement( mepnprms06_[ism-1]->getName() );
    sprintf(histo, "EEPDT PNs pedestal rms %s G16 L2", Numbers::sEE(ism).c_str());
    mepnprms06_[ism-1] = dbe_->book1D(histo, histo, 100, 0., 10.);
    mepnprms06_[ism-1]->setAxisTitle("rms", 1);

    if ( me_hs01_[ism-1] ) dbe_->removeElement( me_hs01_[ism-1]->getName() );
    sprintf(histo, "EELDT led shape L1A %s", Numbers::sEE(ism).c_str());
    me_hs01_[ism-1] = dbe_->book1D(histo, histo, 10, 0., 10.);
    me_hs01_[ism-1]->setAxisTitle("sample", 1);
    me_hs01_[ism-1]->setAxisTitle("amplitude", 2);
    if ( me_hs02_[ism-1] ) dbe_->removeElement( me_hs02_[ism-1]->getName() );
    sprintf(histo, "EELDT led shape L2A %s", Numbers::sEE(ism).c_str());
    me_hs02_[ism-1] = dbe_->book1D(histo, histo, 10, 0., 10.);
    me_hs02_[ism-1]->setAxisTitle("sample", 1);
    me_hs02_[ism-1]->setAxisTitle("amplitude", 2);

    if ( me_hs05_[ism-1] ) dbe_->removeElement( me_hs05_[ism-1]->getName() );
    sprintf(histo, "EELDT led shape L1B %s", Numbers::sEE(ism).c_str());
    me_hs05_[ism-1] = dbe_->book1D(histo, histo, 10, 0., 10.);
    me_hs05_[ism-1]->setAxisTitle("sample", 1);
    me_hs05_[ism-1]->setAxisTitle("amplitude", 2);
    if ( me_hs06_[ism-1] ) dbe_->removeElement( me_hs06_[ism-1]->getName() );
    sprintf(histo, "EELDT led shape L2B %s", Numbers::sEE(ism).c_str());
    me_hs06_[ism-1] = dbe_->book1D(histo, histo, 10, 0., 10.);
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

        if ( meg01_[ism-1] ) meg01_[ism-1]->setBinContent( ix, iy, -1. );
        if ( meg02_[ism-1] ) meg02_[ism-1]->setBinContent( ix, iy, -1. );

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

    dbe_->setCurrentFolder( "EcalEndcap/EELedClient" );

    if ( meg01_[ism-1] ) dbe_->removeElement( meg01_[ism-1]->getName() );
    meg01_[ism-1] = 0;
    if ( meg02_[ism-1] ) dbe_->removeElement( meg02_[ism-1]->getName() );
    meg02_[ism-1] = 0;

    if ( meg05_[ism-1] ) dbe_->removeElement( meg05_[ism-1]->getName() );
    meg05_[ism-1] = 0;
    if ( meg06_[ism-1] ) dbe_->removeElement( meg06_[ism-1]->getName() );
    meg06_[ism-1] = 0;

    if ( meg09_[ism-1] ) dbe_->removeElement( meg09_[ism-1]->getName() );
    meg09_[ism-1] = 0;
    if ( meg10_[ism-1] ) dbe_->removeElement( meg10_[ism-1]->getName() );
    meg10_[ism-1] = 0;

    if ( mea01_[ism-1] ) dbe_->removeElement( mea01_[ism-1]->getName() );
    mea01_[ism-1] = 0;
    if ( mea02_[ism-1] ) dbe_->removeElement( mea02_[ism-1]->getName() );
    mea02_[ism-1] = 0;

    if ( mea05_[ism-1] ) dbe_->removeElement( mea05_[ism-1]->getName() );
    mea05_[ism-1] = 0;
    if ( mea06_[ism-1] ) dbe_->removeElement( mea06_[ism-1]->getName() );
    mea06_[ism-1] = 0;

    if ( met01_[ism-1] ) dbe_->removeElement( met01_[ism-1]->getName() );
    met01_[ism-1] = 0;
    if ( met02_[ism-1] ) dbe_->removeElement( met02_[ism-1]->getName() );
    met02_[ism-1] = 0;

    if ( met05_[ism-1] ) dbe_->removeElement( met05_[ism-1]->getName() );
    met05_[ism-1] = 0;
    if ( met06_[ism-1] ) dbe_->removeElement( met06_[ism-1]->getName() );
    met06_[ism-1] = 0;

    if ( metav01_[ism-1] ) dbe_->removeElement( metav01_[ism-1]->getName() );
    metav01_[ism-1] = 0;
    if ( metav02_[ism-1] ) dbe_->removeElement( metav02_[ism-1]->getName() );
    metav02_[ism-1] = 0;

    if ( metav05_[ism-1] ) dbe_->removeElement( metav05_[ism-1]->getName() );
    metav05_[ism-1] = 0;
    if ( metav06_[ism-1] ) dbe_->removeElement( metav06_[ism-1]->getName() );
    metav06_[ism-1] = 0;

    if ( metrms01_[ism-1] ) dbe_->removeElement( metrms01_[ism-1]->getName() );
    metrms01_[ism-1] = 0;
    if ( metrms02_[ism-1] ) dbe_->removeElement( metrms02_[ism-1]->getName() );
    metrms02_[ism-1] = 0;

    if ( metrms05_[ism-1] ) dbe_->removeElement( metrms05_[ism-1]->getName() );
    metrms05_[ism-1] = 0;
    if ( metrms06_[ism-1] ) dbe_->removeElement( metrms06_[ism-1]->getName() );
    metrms06_[ism-1] = 0;

    if ( meaopn01_[ism-1] ) dbe_->removeElement( meaopn01_[ism-1]->getName() );
    meaopn01_[ism-1] = 0;
    if ( meaopn02_[ism-1] ) dbe_->removeElement( meaopn02_[ism-1]->getName() );
    meaopn02_[ism-1] = 0;

    if ( meaopn05_[ism-1] ) dbe_->removeElement( meaopn05_[ism-1]->getName() );
    meaopn05_[ism-1] = 0;
    if ( meaopn06_[ism-1] ) dbe_->removeElement( meaopn06_[ism-1]->getName() );
    meaopn06_[ism-1] = 0;

    if ( mepnprms01_[ism-1] ) dbe_->removeElement( mepnprms01_[ism-1]->getName() );
    mepnprms01_[ism-1] = 0;
    if ( mepnprms02_[ism-1] ) dbe_->removeElement( mepnprms02_[ism-1]->getName() );
    mepnprms02_[ism-1] = 0;

    if ( mepnprms05_[ism-1] ) dbe_->removeElement( mepnprms05_[ism-1]->getName() );
    mepnprms05_[ism-1] = 0;
    if ( mepnprms06_[ism-1] ) dbe_->removeElement( mepnprms06_[ism-1]->getName() );
    mepnprms06_[ism-1] = 0;

    if ( me_hs01_[ism-1] ) dbe_->removeElement( me_hs01_[ism-1]->getName() );
    me_hs01_[ism-1] = 0;
    if ( me_hs02_[ism-1] ) dbe_->removeElement( me_hs02_[ism-1]->getName() );
    me_hs02_[ism-1] = 0;

    if ( me_hs05_[ism-1] ) dbe_->removeElement( me_hs05_[ism-1]->getName() );
    me_hs05_[ism-1] = 0;
    if ( me_hs06_[ism-1] ) dbe_->removeElement( me_hs06_[ism-1]->getName() );
    me_hs06_[ism-1] = 0;

  }

}

bool EELedClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov) {

  bool status = true;

  EcalLogicID ecid;

  MonLed1Dat vpt_l1;
  map<EcalLogicID, MonLed1Dat> dataset1_l1;
  MonLed2Dat vpt_l2;
  map<EcalLogicID, MonLed2Dat> dataset1_l2;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    cout << " " << Numbers::sEE(ism) << " (ism=" << ism << ")" << endl;
    cout << endl;

    UtilsClient::printBadChannels(meg01_[ism-1], h01_[ism-1]);
    UtilsClient::printBadChannels(meg01_[ism-1], h13_[ism-1]);
    UtilsClient::printBadChannels(meg02_[ism-1], h03_[ism-1]);
    UtilsClient::printBadChannels(meg02_[ism-1], h15_[ism-1]);

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

        update01 = UtilsClient::getBinStats(h01_[ism-1], ix, iy, num01, mean01, rms01);
        update02 = UtilsClient::getBinStats(h02_[ism-1], ix, iy, num02, mean02, rms02);
        update03 = UtilsClient::getBinStats(h03_[ism-1], ix, iy, num03, mean03, rms03);
        update04 = UtilsClient::getBinStats(h04_[ism-1], ix, iy, num04, mean04, rms04);

        update09 = UtilsClient::getBinStats(h13_[ism-1], ix, iy, num09, mean09, rms09);
        update10 = UtilsClient::getBinStats(h14_[ism-1], ix, iy, num10, mean10, rms10);
        update11 = UtilsClient::getBinStats(h15_[ism-1], ix, iy, num11, mean11, rms11);
        update12 = UtilsClient::getBinStats(h16_[ism-1], ix, iy, num12, mean12, rms12);

        if ( update01 || update02 ) {

          if ( Numbers::icEE(ism, jx, jy) == 1 ) {

            cout << "Preparing dataset for " << Numbers::sEE(ism) << " (ism=" << ism << ")" << endl;

            cout << "L1A (" << Numbers::ix0EE(i+1)+ix << "," << Numbers::iy0EE(i+1)+iy << ") " << num01 << " " << mean01 << " " << rms01 << endl;

            cout << endl;

          }

          vpt_l1.setVPTMean(mean01);
          vpt_l1.setVPTRMS(rms01);

          vpt_l1.setVPTOverPNMean(mean02);
          vpt_l1.setVPTOverPNRMS(rms02);

          if ( meg01_[ism-1] && int(meg01_[ism-1]->getBinContent( ix, iy )) % 3 == 1. ) {
            vpt_l1.setTaskStatus(true);
          } else {
            vpt_l1.setTaskStatus(false);
          }

          status = status && UtilsClient::getBinQual(meg01_[ism-1], ix, iy);

          int ic = Numbers::indexEE(ism, jx, jy);

          if ( ic == -1 ) continue;

          if ( econn ) {
            ecid = LogicID::getEcalLogicID("EE_crystal_number", Numbers::iSM(ism, EcalEndcap), ic);
            dataset1_l1[ecid] = vpt_l1;
          }

        }

        if ( update09 || update10 ) {

          if ( Numbers::icEE(ism, jx, jy) == 1 ) {

            cout << "Preparing dataset for " << Numbers::sEE(ism) << " (ism=" << ism << ")" << endl;

            cout << "L1B (" << Numbers::ix0EE(i+1)+ix << "," << Numbers::iy0EE(i+1)+iy << ") " << num09 << " " << mean09 << " " << rms09 << endl;

            cout << endl;

          }

          vpt_l1.setVPTMean(mean09);
          vpt_l1.setVPTRMS(rms09);

          vpt_l1.setVPTOverPNMean(mean10);
          vpt_l1.setVPTOverPNRMS(rms10);

          if ( meg01_[ism-1] && int(meg01_[ism-1]->getBinContent( ix, iy )) % 3 == 1. ) {
            vpt_l1.setTaskStatus(true);
          } else {
            vpt_l1.setTaskStatus(false);
          }

          status = status && UtilsClient::getBinQual(meg01_[ism-1], ix, iy);

          int ic = Numbers::indexEE(ism, jx, jy);

          if ( ic == -1 ) continue;

          if ( econn ) {
            ecid = LogicID::getEcalLogicID("EE_crystal_number", Numbers::iSM(ism, EcalEndcap), ic);
            dataset1_l1[ecid] = vpt_l1;
          }

        }

        if ( update03 || update04 ) {

          if ( Numbers::icEE(ism, jx, jy) == 1 ) {

            cout << "Preparing dataset for " << Numbers::sEE(ism) << " (ism=" << ism << ")" << endl;

            cout << "L2A (" << Numbers::ix0EE(i+1)+ix << "," << Numbers::iy0EE(i+1)+iy << ") " << num03 << " " << mean03 << " " << rms03 << endl;

            cout << endl;

          }

          vpt_l2.setVPTMean(mean03);
          vpt_l2.setVPTRMS(rms03);

          vpt_l2.setVPTOverPNMean(mean04);
          vpt_l2.setVPTOverPNRMS(rms04);

          if ( meg02_[ism-1] && int(meg02_[ism-1]->getBinContent( ix, iy )) % 3 == 1. ) {
            vpt_l2.setTaskStatus(true);
          } else {
            vpt_l2.setTaskStatus(false);
          }

          status = status && UtilsClient::getBinQual(meg02_[ism-1], ix, iy);

          int ic = Numbers::indexEE(ism, jx, jy);

          if ( ic == -1 ) continue;

          if ( econn ) {
            ecid = LogicID::getEcalLogicID("EE_crystal_number", Numbers::iSM(ism, EcalEndcap), ic);
            dataset1_l2[ecid] = vpt_l2;
          }

        }

        if ( update11 || update12 ) {

          if ( Numbers::icEE(ism, jx, jy) == 1 ) {

            cout << "Preparing dataset for " << Numbers::sEE(ism) << " (ism=" << ism << ")" << endl;

            cout << "L2B (" << Numbers::ix0EE(i+1)+ix << "," << Numbers::iy0EE(i+1)+iy << ") " << num11 << " " << mean11 << " " << rms11 << endl;

            cout << endl;

          }

          vpt_l2.setVPTMean(mean11);
          vpt_l2.setVPTRMS(rms11);

          vpt_l2.setVPTOverPNMean(mean12);
          vpt_l2.setVPTOverPNRMS(rms12);

          if ( meg02_[ism-1] && int(meg02_[ism-1]->getBinContent( ix, iy )) % 3 == 1. ) {
            vpt_l2.setTaskStatus(true);
          } else {
            vpt_l2.setTaskStatus(false);
          }

          status = status && UtilsClient::getBinQual(meg02_[ism-1], ix, iy);

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
      cout << "Inserting MonLedDat ..." << endl;
      if ( dataset1_l1.size() != 0 ) econn->insertDataArraySet(&dataset1_l1, moniov);
      if ( dataset1_l2.size() != 0 ) econn->insertDataArraySet(&dataset1_l2, moniov);
      cout << "done." << endl;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

  cout << endl;

  MonPNLed1Dat pn_l1;
  map<EcalLogicID, MonPNLed1Dat> dataset2_l1;
  MonPNLed2Dat pn_l2;
  map<EcalLogicID, MonPNLed2Dat> dataset2_l2;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    cout << " " << Numbers::sEE(ism) << " (ism=" << ism << ")" << endl;
    cout << endl;

    UtilsClient::printBadChannels(meg05_[ism-1], i01_[ism-1]);
    UtilsClient::printBadChannels(meg05_[ism-1], i05_[ism-1]);
    UtilsClient::printBadChannels(meg06_[ism-1], i02_[ism-1]);
    UtilsClient::printBadChannels(meg06_[ism-1], i06_[ism-1]);

    UtilsClient::printBadChannels(meg09_[ism-1], i09_[ism-1]);
    UtilsClient::printBadChannels(meg09_[ism-1], i13_[ism-1]);
    UtilsClient::printBadChannels(meg10_[ism-1], i10_[ism-1]);
    UtilsClient::printBadChannels(meg10_[ism-1], i14_[ism-1]);

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

      update01 = UtilsClient::getBinStats(i01_[ism-1], i, 0, num01, mean01, rms01);
      update02 = UtilsClient::getBinStats(i02_[ism-1], i, 0, num02, mean02, rms02);

      update05 = UtilsClient::getBinStats(i05_[ism-1], i, 0, num05, mean05, rms05);
      update06 = UtilsClient::getBinStats(i06_[ism-1], i, 0, num06, mean06, rms06);

      update09 = UtilsClient::getBinStats(i09_[ism-1], i, 0, num09, mean09, rms09);
      update10 = UtilsClient::getBinStats(i10_[ism-1], i, 0, num10, mean10, rms10);

      update13 = UtilsClient::getBinStats(i13_[ism-1], i, 0, num13, mean13, rms13);
      update14 = UtilsClient::getBinStats(i14_[ism-1], i, 0, num14, mean14, rms14);

      if ( update01 || update05 || update09 || update13 ) {

        if ( i == 1 ) {

          cout << "Preparing dataset for " << Numbers::sEE(ism) << " (ism=" << ism << ")" << endl;

          cout << "PNs (" << i << ") L1 G01 " << num01  << " " << mean01 << " " << rms01  << endl;
          cout << "PNs (" << i << ") L1 G16 " << num09  << " " << mean09 << " " << rms09  << endl;

          cout << endl;

        }

        pn_l1.setADCMeanG1(mean01);
        pn_l1.setADCRMSG1(rms01);

        pn_l1.setPedMeanG1(mean05);
        pn_l1.setPedRMSG1(rms05);

        pn_l1.setADCMeanG16(mean09);
        pn_l1.setADCRMSG16(rms09);

        pn_l1.setPedMeanG16(mean13);
        pn_l1.setPedRMSG16(rms13);

        if ( meg05_[ism-1] && int(meg05_[ism-1]->getBinContent( i, 1 )) % 3 == 1. ||
             meg09_[ism-1] && int(meg09_[ism-1]->getBinContent( i, 1 )) % 3 == 1. ) {
          pn_l1.setTaskStatus(true);
        } else {
          pn_l1.setTaskStatus(false);
        }

        status = status && ( UtilsClient::getBinQual(meg05_[ism-1], i, 1) ||
                             UtilsClient::getBinQual(meg09_[ism-1], i, 1) );

        if ( econn ) {
          ecid = LogicID::getEcalLogicID("EE_LM_PN", Numbers::iSM(ism, EcalEndcap), i-1);
          dataset2_l1[ecid] = pn_l1;
        }

      }

      if ( update02 || update06 || update10 || update14 ) {

        if ( i == 1 ) {

          cout << "Preparing dataset for " << Numbers::sEE(ism) << " (ism=" << ism << ")" << endl;

          cout << "PNs (" << i << ") L2 G01 " << num02  << " " << mean02 << " " << rms02  << endl;
          cout << "PNs (" << i << ") L2 G16 " << num10  << " " << mean10 << " " << rms10  << endl;

          cout << endl;

        }

        pn_l2.setADCMeanG1(mean02);
        pn_l2.setADCRMSG1(rms02);

        pn_l2.setPedMeanG1(mean06);
        pn_l2.setPedRMSG1(rms06);

        pn_l2.setADCMeanG16(mean10);
        pn_l2.setADCRMSG16(rms10);

        pn_l2.setPedMeanG16(mean14);
        pn_l2.setPedRMSG16(rms14);

        if ( meg06_[ism-1] && int(meg06_[ism-1]->getBinContent( i, 1 )) % 3 == 1. ||
             meg10_[ism-1] && int(meg10_[ism-1]->getBinContent( i, 1 )) % 3 == 1. ) {
          pn_l2.setTaskStatus(true);
        } else {
          pn_l2.setTaskStatus(false);
        }

        status = status && ( UtilsClient::getBinQual(meg06_[ism-1], i, 1) ||
                             UtilsClient::getBinQual(meg10_[ism-1], i, 1) );

        if ( econn ) {
          ecid = LogicID::getEcalLogicID("EE_LM_PN", Numbers::iSM(ism, EcalEndcap), i-1);
          dataset2_l2[ecid] = pn_l2;
        }

      }

    }

  }

  if ( econn ) {
    try {
      cout << "Inserting MonPnDat ..." << endl;
      if ( dataset2_l1.size() != 0 ) econn->insertDataArraySet(&dataset2_l1, moniov);
      if ( dataset2_l2.size() != 0 ) econn->insertDataArraySet(&dataset2_l2, moniov);
      cout << "done." << endl;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

  return status;

}

void EELedClient::analyze(void){

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( verbose_ ) cout << "EELedClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;
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

  EcalErrorMask::fetchDataSet(&mask1);
  EcalErrorMask::fetchDataSet(&mask2);

  char histo[200];

  MonitorElement* me;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    sprintf(histo, (prefixME_+"EcalEndcap/EELedTask/Led1/EELDT amplitude %s L1A").c_str(), Numbers::sEE(ism).c_str());
    me = dbe_->get(histo);
    h01_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h01_[ism-1] );

    sprintf(histo, (prefixME_+"EcalEndcap/EELedTask/Led1/EELDT amplitude over PN %s L1A").c_str(), Numbers::sEE(ism).c_str());
    me = dbe_->get(histo);
    h02_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h02_[ism-1] );

    sprintf(histo, (prefixME_+"EcalEndcap/EELedTask/Led2/EELDT amplitude %s L2A").c_str(), Numbers::sEE(ism).c_str());
    me = dbe_->get(histo);
    h03_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h03_[ism-1] );

    sprintf(histo, (prefixME_+"EcalEndcap/EELedTask/Led2/EELDT amplitude over PN %s L2A").c_str(), Numbers::sEE(ism).c_str());
    me = dbe_->get(histo);
    h04_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h04_[ism-1] );

    sprintf(histo, (prefixME_+"EcalEndcap/EELedTask/Led1/EELDT timing %s L1A").c_str(), Numbers::sEE(ism).c_str());
    me = dbe_->get(histo);
    h09_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h09_[ism-1] );

    sprintf(histo, (prefixME_+"EcalEndcap/EELedTask/Led2/EELDT timing %s L2A").c_str(), Numbers::sEE(ism).c_str());
    me = dbe_->get(histo);
    h10_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h10_[ism-1] );

    sprintf(histo, (prefixME_+"EcalEndcap/EELedTask/Led1/EELDT amplitude %s L1B").c_str(), Numbers::sEE(ism).c_str());
    me = dbe_->get(histo);
    h13_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h13_[ism-1] );

    sprintf(histo, (prefixME_+"EcalEndcap/EELedTask/Led1/EELDT amplitude over PN %s L1B").c_str(), Numbers::sEE(ism).c_str());
    me = dbe_->get(histo);
    h14_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h14_[ism-1] );

    sprintf(histo, (prefixME_+"EcalEndcap/EELedTask/Led2/EELDT amplitude %s L2B").c_str(), Numbers::sEE(ism).c_str());
    me = dbe_->get(histo);
    h15_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h15_[ism-1] );

    sprintf(histo, (prefixME_+"EcalEndcap/EELedTask/Led2/EELDT amplitude over PN %s L2B").c_str(), Numbers::sEE(ism).c_str());
    me = dbe_->get(histo);
    h16_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h16_[ism-1] );

    sprintf(histo, (prefixME_+"EcalEndcap/EELedTask/Led1/EELDT timing %s L1B").c_str(), Numbers::sEE(ism).c_str());
    me = dbe_->get(histo);
    h21_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h21_[ism-1] );

    sprintf(histo, (prefixME_+"EcalEndcap/EELedTask/Led2/EELDT timing %s L2B").c_str(), Numbers::sEE(ism).c_str());
    me = dbe_->get(histo);
    h22_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h22_[ism-1] );

    sprintf(histo, (prefixME_+"EcalEndcap/EELedTask/Led1/EELDT shape %s L1A").c_str(), Numbers::sEE(ism).c_str());
    me = dbe_->get(histo);
    hs01_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, hs01_[ism-1] );

    sprintf(histo, (prefixME_+"EcalEndcap/EELedTask/Led2/EELDT shape %s L2A").c_str(), Numbers::sEE(ism).c_str());
    me = dbe_->get(histo);
    hs02_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, hs02_[ism-1] );

    sprintf(histo, (prefixME_+"EcalEndcap/EELedTask/Led1/EELDT shape %s L1B").c_str(), Numbers::sEE(ism).c_str());
    me = dbe_->get(histo);
    hs05_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, hs05_[ism-1] );

    sprintf(histo, (prefixME_+"EcalEndcap/EELedTask/Led2/EELDT shape %s L2B").c_str(), Numbers::sEE(ism).c_str());
    me = dbe_->get(histo);
    hs06_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, hs06_[ism-1] );

    sprintf(histo, (prefixME_+"EcalEndcap/EELedTask/Led1/PN/Gain01/EEPDT PNs amplitude %s G01 L1").c_str(), Numbers::sEE(ism).c_str());
    me = dbe_->get(histo);
    i01_[ism-1] = UtilsClient::getHisto<TProfile*>( me, cloneME_, i01_[ism-1] );

    sprintf(histo, (prefixME_+"EcalEndcap/EELedTask/Led2/PN/Gain01/EEPDT PNs amplitude %s G01 L2").c_str(), Numbers::sEE(ism).c_str());
    me = dbe_->get(histo);
    i02_[ism-1] = UtilsClient::getHisto<TProfile*>( me, cloneME_, i02_[ism-1] );

    sprintf(histo, (prefixME_+"EcalEndcap/EELedTask/Led1/PN/Gain01/EEPDT PNs pedestal %s G01 L1").c_str(), Numbers::sEE(ism).c_str());
    me = dbe_->get(histo);
    i05_[ism-1] = UtilsClient::getHisto<TProfile*>( me, cloneME_, i05_[ism-1] );

    sprintf(histo, (prefixME_+"EcalEndcap/EELedTask/Led2/PN/Gain01/EEPDT PNs pedestal %s G01 L2").c_str(), Numbers::sEE(ism).c_str());
    me = dbe_->get(histo);
    i06_[ism-1] = UtilsClient::getHisto<TProfile*>( me, cloneME_, i06_[ism-1] );

    sprintf(histo, (prefixME_+"EcalEndcap/EELedTask/Led1/PN/Gain16/EEPDT PNs amplitude %s G16 L1").c_str(), Numbers::sEE(ism).c_str());
    me = dbe_->get(histo);
    i09_[ism-1] = UtilsClient::getHisto<TProfile*>( me, cloneME_, i09_[ism-1] );

    sprintf(histo, (prefixME_+"EcalEndcap/EELedTask/Led2/PN/Gain16/EEPDT PNs amplitude %s G16 L2").c_str(), Numbers::sEE(ism).c_str());
    me = dbe_->get(histo);
    i10_[ism-1] = UtilsClient::getHisto<TProfile*>( me, cloneME_, i10_[ism-1] );

    sprintf(histo, (prefixME_+"EcalEndcap/EELedTask/Led1/PN/Gain16/EEPDT PNs pedestal %s G16 L1").c_str(), Numbers::sEE(ism).c_str());
    me = dbe_->get(histo);
    i13_[ism-1] = UtilsClient::getHisto<TProfile*>( me, cloneME_, i13_[ism-1] );

    sprintf(histo, (prefixME_+"EcalEndcap/EELedTask/Led2/PN/Gain16/EEPDT PNs pedestal %s G16 L2").c_str(), Numbers::sEE(ism).c_str());
    me = dbe_->get(histo);
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

        update01 = UtilsClient::getBinStats(h01_[ism-1], ix, iy, num01, mean01, rms01);
        update02 = UtilsClient::getBinStats(h03_[ism-1], ix, iy, num02, mean02, rms02);

        update05 = UtilsClient::getBinStats(h13_[ism-1], ix, iy, num05, mean05, rms05);
        update06 = UtilsClient::getBinStats(h15_[ism-1], ix, iy, num06, mean06, rms06);

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

        if ( meg01_[ism-1] ) meg01_[ism-1]->setBinContent( ix, iy, -1.);
        if ( meg02_[ism-1] ) meg02_[ism-1]->setBinContent( ix, iy, -1.);

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

        update01 = UtilsClient::getBinStats(h01_[ism-1], ix, iy, num01, mean01, rms01);
        update02 = UtilsClient::getBinStats(h02_[ism-1], ix, iy, num02, mean02, rms02);
        update03 = UtilsClient::getBinStats(h03_[ism-1], ix, iy, num03, mean03, rms03);
        update04 = UtilsClient::getBinStats(h04_[ism-1], ix, iy, num04, mean04, rms04);

        update09 = UtilsClient::getBinStats(h09_[ism-1], ix, iy, num09, mean09, rms09);
        update10 = UtilsClient::getBinStats(h10_[ism-1], ix, iy, num10, mean10, rms10);

        // other SM half

        update13 = UtilsClient::getBinStats(h13_[ism-1], ix, iy, num13, mean13, rms13);
        update14 = UtilsClient::getBinStats(h14_[ism-1], ix, iy, num14, mean14, rms14);
        update15 = UtilsClient::getBinStats(h15_[ism-1], ix, iy, num15, mean15, rms15);
        update16 = UtilsClient::getBinStats(h16_[ism-1], ix, iy, num16, mean16, rms16);

        update21 = UtilsClient::getBinStats(h21_[ism-1], ix, iy, num21, mean21, rms21);
        update22 = UtilsClient::getBinStats(h22_[ism-1], ix, iy, num22, mean22, rms22);

        if ( update01 ) {

          float val;

          val = 1.;
          if ( fabs(mean01 - meanAmplL1A) > fabs(percentVariation_ * meanAmplL1A) )
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
          if ( fabs(mean13 - meanAmplL1B) > fabs(percentVariation_ * meanAmplL1B) )
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
          if ( fabs(mean03 - meanAmplL2A) > fabs(percentVariation_ * meanAmplL2A) )
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
          if ( fabs(mean15 - meanAmplL2B) > fabs(percentVariation_ * meanAmplL2B) )
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
                if ( meg01_[ism-1] ) {
                  float val = int(meg01_[ism-1]->getBinContent(ix, iy)) % 3;
                  meg01_[ism-1]->setBinContent( ix, iy, val+3 );
                }
                if ( meg02_[ism-1] ) {
                  float val = int(meg02_[ism-1]->getBinContent(ix, iy)) % 3;
                  meg02_[ism-1]->setBinContent( ix, iy, val+3 );
                }
              }
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

      update01 = UtilsClient::getBinStats(i01_[ism-1], i, 0, num01, mean01, rms01);
      update02 = UtilsClient::getBinStats(i02_[ism-1], i, 0, num02, mean02, rms02);

      update05 = UtilsClient::getBinStats(i05_[ism-1], i, 0, num05, mean05, rms05);
      update06 = UtilsClient::getBinStats(i06_[ism-1], i, 0, num06, mean06, rms06);

      update09 = UtilsClient::getBinStats(i09_[ism-1], i, 0, num09, mean09, rms09);
      update10 = UtilsClient::getBinStats(i10_[ism-1], i, 0, num10, mean10, rms10);

      update13 = UtilsClient::getBinStats(i13_[ism-1], i, 0, num13, mean13, rms13);
      update14 = UtilsClient::getBinStats(i14_[ism-1], i, 0, num14, mean14, rms14);

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
              if ( meg05_[ism-1] ) {
                float val = int(meg05_[ism-1]->getBinContent(i, 1)) % 3;
                meg05_[ism-1]->setBinContent( i, 1, val+3 );
              }
            }
            if ( (m->second).getErrorBits() & (bits01|bits02) ) {
              if ( meg06_[ism-1] ) {
                float val = int(meg06_[ism-1]->getBinContent(i, 1)) % 3;
                meg06_[ism-1]->setBinContent( i, 1, val+3 );
              }
            }
            if ( (m->second).getErrorBits() & (bits01|bits04) ) {
              if ( meg09_[ism-1] ) {
                float val = int(meg09_[ism-1]->getBinContent(i, 1)) % 3;
                meg09_[ism-1]->setBinContent( i, 1, val+3 );
              }
            }
            if ( (m->second).getErrorBits() & (bits01|bits04) ) {
              if ( meg10_[ism-1] ) {
                float val = int(meg10_[ism-1]->getBinContent(i, 1)) % 3;
                meg10_[ism-1]->setBinContent( i, 1, val+3 );
              }
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

void EELedClient::htmlOutput(int run, string& htmlDir, string& htmlName){

  cout << "Preparing EELedClient html output ..." << endl;

  ofstream htmlFile;

  htmlFile.open((htmlDir + htmlName).c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlFile << "<html>  " << endl;
  htmlFile << "<head>  " << endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlFile << " http-equiv=\"content-type\">  " << endl;
  htmlFile << "  <title>Monitor:LedTask output</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlFile << "<body>  " << endl;
  //htmlFile << "<br>  " << endl;
  htmlFile << "<a name=""top""></a>" << endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << run << "</span></h2>" << endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">LED</span></h2> " << endl;
  htmlFile << "<hr>" << endl;
  htmlFile << "<table border=1><tr><td bgcolor=red>channel has problems in this task</td>" << endl;
  htmlFile << "<td bgcolor=lime>channel has NO problems</td>" << endl;
  htmlFile << "<td bgcolor=yellow>channel is missing</td></table>" << endl;
  htmlFile << "<hr>" << endl;
  //   htmlFile << "<table border=1><tr><td>L1 = Led 1</td>" << endl;
  //   htmlFile << "<td>L2 = Led 2</td>" << endl;
  htmlFile << "<table style=\"width: 600px;\" border=\"0\">" << endl;
  htmlFile << "<tbody>" << endl;
  htmlFile << "<tr>" << endl;
  htmlFile << "<td style=\"text-align: center;\">" << endl;
  htmlFile << "<div style=\"text-align: center;\"> </div>" << endl;
  htmlFile << "<table style=\"width: 482px; height: 35px;\" border=\"1\">" << endl;
  htmlFile << "<tbody>" << endl;
  htmlFile << "<tr>" << endl;
  htmlFile << "<td style=\"text-align: center;\">L1 = Led 1 </td>" << endl;
  htmlFile << "<td style=\"vertical-align: top; text-align: center;\">L2 = Led 2</td>" << endl;
  htmlFile << "</tr>" << endl;
  htmlFile << "</tbody>" << endl;
  htmlFile << "</table>" << endl;
  htmlFile << "</td>" << endl;
  htmlFile << "<td align=\"center\">" << endl;
  htmlFile << "<div style=\"text-align: center;\"> </div>" << endl;
  htmlFile << "<table style=\"width: 255px; height: 35px;\" border=\"1\">" << endl;
  htmlFile << "<tbody>" << endl;
  htmlFile << "<tr>" << endl;
  htmlFile << "<td style=\"text-align: center;\">A=first half</td>" << endl;
  htmlFile << "<td style=\"vertical-align: top; text-align: center;\">B=second half<br>" << endl;
  htmlFile << "</td>" << endl;
  htmlFile << "</tr>" << endl;
  htmlFile << "</tbody>" << endl;
  htmlFile << "</table>" << endl;
  htmlFile << "</td>" << endl;
  htmlFile << "</tr>" << endl;
  htmlFile << "</tbody>" << endl;
  htmlFile << "</table>" << endl;
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

  string imgNameQual[8], imgNameAmp[8], imgNameTim[8], imgNameTimav[8], imgNameTimrms[8], imgNameShape[8], imgNameAmpoPN[8], imgNameMEPnQualG01[8], imgNameMEPnG01[8], imgNameMEPnPedG01[8], imgNameMEPnRmsPedG01[8], imgNameMEPnQualG16[8], imgNameMEPnG16[8], imgNameMEPnPedG16[8], imgNameMEPnRmsPedG16[8], imgName, meName;

  TCanvas* cQual   = new TCanvas("cQual", "Temp", 2*csize, 2*csize);
  TCanvas* cQualPN = new TCanvas("cQualPN", "Temp", 2*csize, csize);
  TCanvas* cAmp    = new TCanvas("cAmp", "Temp", csize, csize);
  TCanvas* cTim    = new TCanvas("cTim", "Temp", csize, csize);
  TCanvas* cTimav  = new TCanvas("cTimav", "Temp", csize, csize);
  TCanvas* cTimrms = new TCanvas("cTimrms", "Temp", csize, csize);
  TCanvas* cShape  = new TCanvas("cShape", "Temp", csize, csize);
  TCanvas* cAmpoPN = new TCanvas("cAmpoPN", "Temp", csize, csize);
  TCanvas* cPed    = new TCanvas("cPed", "Temp", csize, csize);

  TH2F* obj2f;
  TH1F* obj1f;
  TProfile* objp;

  // Loop on endcap sectors

  for ( unsigned int i=0; i<superModules_.size(); i ++ ) {

    int ism = superModules_[i];

    // Loop on wavelength times 2 'sides'

    for ( int iCanvas = 1 ; iCanvas <= 4 * 2 ; iCanvas++ ) {

      // skip unused wavelengths
      if ( iCanvas == 3 || iCanvas == 4 ) continue;
      if ( iCanvas == 4+3 || iCanvas == 4+4 ) continue;

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
        case 4:
        case 5:
        case 6:
        case 7:
        case 8:
          obj2f = 0;
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

      // Amplitude distributions

      imgNameAmp[iCanvas-1] = "";

      obj1f = 0;
      switch ( iCanvas ) {
        case 1:
          obj1f = UtilsClient::getHisto<TH1F*>( mea01_[ism-1] );
          break;
        case 2:
          obj1f = UtilsClient::getHisto<TH1F*>( mea02_[ism-1] );
          break;
        case 3:
        case 4:
          obj1f = 0;
          break;
        case 5:
          obj1f = UtilsClient::getHisto<TH1F*>( mea05_[ism-1] );
          break;
        case 6:
          obj1f = UtilsClient::getHisto<TH1F*>( mea06_[ism-1] );
          break;
        case 7:
        case 8:
          obj1f = 0;
          break;
        default:
          break;
      }

      if ( obj1f ) {

        meName = obj1f->GetName();

        replace(meName.begin(), meName.end(), ' ', '_');
        imgNameAmp[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgNameAmp[iCanvas-1];

        cAmp->cd();
        gStyle->SetOptStat("euo");
        obj1f->SetStats(kTRUE);
//        if ( obj1f->GetMaximum(histMax) > 0. ) {
//          gPad->SetLogy(1);
//        } else {
//          gPad->SetLogy(0);
//        }
        obj1f->SetMinimum(0.0);
        obj1f->Draw();
        cAmp->Update();
        cAmp->SaveAs(imgName.c_str());
        gPad->SetLogy(0);

      }

      // Timing distributions

      imgNameTim[iCanvas-1] = "";

      obj1f = 0;
      switch ( iCanvas ) {
        case 1:
          obj1f = UtilsClient::getHisto<TH1F*>( met01_[ism-1] );
          break;
        case 2:
          obj1f = UtilsClient::getHisto<TH1F*>( met02_[ism-1] );
          break;
        case 3:
        case 4:
          obj1f = 0;
          break;
        case 5:
          obj1f = UtilsClient::getHisto<TH1F*>( met05_[ism-1] );
          break;
        case 6:
          obj1f = UtilsClient::getHisto<TH1F*>( met06_[ism-1] );
          break;
        case 7:
        case 8:
          obj1f = 0;
          break;
        default:
          break;
      }

      if ( obj1f ) {

        meName = obj1f->GetName();

        replace(meName.begin(), meName.end(), ' ', '_');
        imgNameTim[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgNameTim[iCanvas-1];

        cTim->cd();
        gStyle->SetOptStat("euo");
        obj1f->SetStats(kTRUE);
        obj1f->SetMinimum(0.0);
        obj1f->SetMaximum(10.0);
        obj1f->Draw();
        cTim->Update();
        cTim->SaveAs(imgName.c_str());
        gPad->SetLogy(0);

      }

      // Timing mean distributions

      imgNameTimav[iCanvas-1] = "";

      obj1f = 0;
      switch ( iCanvas ) {
        case 1:
          obj1f = UtilsClient::getHisto<TH1F*>( metav01_[ism-1] );
          break;
        case 2:
          obj1f = UtilsClient::getHisto<TH1F*>( metav02_[ism-1] );
          break;
        case 3:
        case 4:
          obj1f = 0;
          break;
        case 5:
          obj1f = UtilsClient::getHisto<TH1F*>( metav05_[ism-1] );
          break;
        case 6:
          obj1f = UtilsClient::getHisto<TH1F*>( metav06_[ism-1] );
          break;
        case 7:
        case 8:
          obj1f = 0;
          break;
        default:
          break;
      }

      if ( obj1f ) {

        meName = obj1f->GetName();

        replace(meName.begin(), meName.end(), ' ', '_');
        imgNameTimav[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgNameTimav[iCanvas-1];

        cTimav->cd();
        gStyle->SetOptStat("euomr");
        obj1f->SetStats(kTRUE);
        if ( obj1f->GetMaximum(histMax) > 0. ) {
          gPad->SetLogy(1);
        } else {
          gPad->SetLogy(0);
        }
        obj1f->Draw();
        cTimav->Update();
        cTimav->SaveAs(imgName.c_str());
        gPad->SetLogy(0);

      }

      // Timing rms distributions

      imgNameTimrms[iCanvas-1] = "";

      obj1f = 0;
      switch ( iCanvas ) {
        case 1:
          obj1f = UtilsClient::getHisto<TH1F*>( metrms01_[ism-1] );
          break;
        case 2:
          obj1f = UtilsClient::getHisto<TH1F*>( metrms02_[ism-1] );
          break;
        case 3:
        case 4:
          obj1f = 0;
          break;
        case 5:
          obj1f = UtilsClient::getHisto<TH1F*>( metrms05_[ism-1] );
          break;
        case 6:
          obj1f = UtilsClient::getHisto<TH1F*>( metrms06_[ism-1] );
          break;
        case 7:
        case 8:
          obj1f = 0;
          break;
        default:
          break;
      }

      if ( obj1f ) {

        meName = obj1f->GetName();

        replace(meName.begin(), meName.end(), ' ', '_');
        imgNameTimrms[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgNameTimrms[iCanvas-1];

        cTimrms->cd();
        gStyle->SetOptStat("euomr");
        obj1f->SetStats(kTRUE);
        if ( obj1f->GetMaximum(histMax) > 0. ) {
          gPad->SetLogy(1);
        } else {
          gPad->SetLogy(0);
        }
        obj1f->Draw();
        cTimrms->Update();
        cTimrms->SaveAs(imgName.c_str());
        gPad->SetLogy(0);

      }

      // Shape distributions

      imgNameShape[iCanvas-1] = "";

      obj1f = 0;
      switch ( iCanvas ) {
        case 1:
          obj1f = UtilsClient::getHisto<TH1F*>( me_hs01_[ism-1] );
          break;
        case 2:
          obj1f = UtilsClient::getHisto<TH1F*>( me_hs02_[ism-1] );
          break;
        case 3:
        case 4:
          obj1f = 0;
          break;
        case 5:
          obj1f = UtilsClient::getHisto<TH1F*>( me_hs05_[ism-1] );
          break;
        case 6:
          obj1f = UtilsClient::getHisto<TH1F*>( me_hs06_[ism-1] );
          break;
        case 7:
        case 8:
          obj1f = 0;
          break;
        default:
          break;
      }

      if ( obj1f ) {
        meName = obj1f->GetName();

        replace(meName.begin(), meName.end(), ' ', '_');
        imgNameShape[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgNameShape[iCanvas-1];

        cShape->cd();
        gStyle->SetOptStat("euo");
        obj1f->SetStats(kTRUE);
//        if ( obj1f->GetMaximum(histMax) > 0. ) {
//          gPad->SetLogy(1);
//        } else {
//          gPad->SetLogy(0);
//        }
        obj1f->Draw();
        cShape->Update();
        cShape->SaveAs(imgName.c_str());
        gPad->SetLogy(0);

      }

      // Amplitude over PN distributions

      imgNameAmpoPN[iCanvas-1] = "";

      obj1f = 0;
      switch ( iCanvas ) {
        case 1:
          obj1f = UtilsClient::getHisto<TH1F*>( meaopn01_[ism-1] );
          break;
        case 2:
          obj1f = UtilsClient::getHisto<TH1F*>( meaopn02_[ism-1] );
          break;
        case 3:
        case 4:
          obj1f = 0;
          break;
        case 5:
          obj1f = UtilsClient::getHisto<TH1F*>( meaopn05_[ism-1] );
          break;
        case 6:
          obj1f = UtilsClient::getHisto<TH1F*>( meaopn06_[ism-1] );
          break;
        case 7:
        case 8:
          obj1f = 0;
          break;
        default:
          break;
      }

      if ( obj1f ) {

        meName = obj1f->GetName();

        replace(meName.begin(), meName.end(), ' ', '_');
        imgNameAmpoPN[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgNameAmpoPN[iCanvas-1];

        cAmpoPN->cd();
        gStyle->SetOptStat("euo");
        obj1f->SetStats(kTRUE);
//        if ( obj1f->GetMaximum(histMax) > 0. ) {
//          gPad->SetLogy(1);
//        } else {
//          gPad->SetLogy(0);
//        }
        obj1f->SetMinimum(0.0);
        obj1f->SetMaximum(20.0);
        obj1f->Draw();
        cAmpoPN->Update();
        cAmpoPN->SaveAs(imgName.c_str());
        gPad->SetLogy(0);

      }

      // Monitoring elements plots

      imgNameMEPnQualG01[iCanvas-1] = "";

      obj2f = 0;
      switch ( iCanvas ) {
      case 1:
        obj2f = UtilsClient::getHisto<TH2F*>( meg05_[ism-1] );
        break;
      case 2:
        obj2f = UtilsClient::getHisto<TH2F*>( meg06_[ism-1] );
        break;
      case 3:
      case 4:
      case 5:
      case 6:
      case 7:
      case 8:
        obj2f = 0;
        break;
      default:
        break;
      }

      if ( obj2f ) {

        meName = obj2f->GetName();

        replace(meName.begin(), meName.end(), ' ', '_');
        imgNameMEPnQualG01[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgNameMEPnQualG01[iCanvas-1];

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

      imgNameMEPnQualG16[iCanvas-1] = "";

      obj2f = 0;
      switch ( iCanvas ) {
      case 1:
        obj2f = UtilsClient::getHisto<TH2F*>( meg09_[ism-1] );
        break;
      case 2:
        obj2f = UtilsClient::getHisto<TH2F*>( meg10_[ism-1] );
        break;
      case 3:
      case 4:
      case 5:
      case 6:
      case 7:
      case 8:
        obj2f = 0;
        break;
      default:
        break;
      }

      if ( obj2f ) {

        meName = obj2f->GetName();

        replace(meName.begin(), meName.end(), ' ', '_');
        imgNameMEPnQualG16[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgNameMEPnQualG16[iCanvas-1];

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

      imgNameMEPnG01[iCanvas-1] = "";

      objp = 0;
      switch ( iCanvas ) {
        case 1:
          objp = i01_[ism-1];
          break;
        case 2:
          objp = i02_[ism-1];
          break;
        case 3:
        case 4:
        case 5:
        case 6:
        case 7:
        case 8:
          objp = 0;
          break;
        default:
          break;
      }

      if ( objp ) {

        meName = objp->GetName();

        replace(meName.begin(), meName.end(), ' ', '_');
        imgNameMEPnG01[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgNameMEPnG01[iCanvas-1];

        cAmp->cd();
        gStyle->SetOptStat("euo");
        objp->SetStats(kTRUE);
//        if ( objp->GetMaximum(histMax) > 0. ) {
//          gPad->SetLogy(1);
//        } else {
//          gPad->SetLogy(0);
//        }
        objp->SetMinimum(0.0);
        objp->Draw();
        cAmp->Update();
        cAmp->SaveAs(imgName.c_str());
        gPad->SetLogy(0);

      }

      imgNameMEPnG16[iCanvas-1] = "";

      objp = 0;
      switch ( iCanvas ) {
        case 1:
          objp = i09_[ism-1];
          break;
        case 2:
          objp = i10_[ism-1];
          break;
        case 3:
        case 4:
        case 5:
        case 6:
        case 7:
        case 8:
          objp = 0;
          break;
        default:
          break;
      }

      if ( objp ) {

        meName = objp->GetName();

        replace(meName.begin(), meName.end(), ' ', '_');
        imgNameMEPnG16[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgNameMEPnG16[iCanvas-1];

        cAmp->cd();
        gStyle->SetOptStat("euo");
        objp->SetStats(kTRUE);
//        if ( objp->GetMaximum(histMax) > 0. ) {
//          gPad->SetLogy(1);
//        } else {
//          gPad->SetLogy(0);
//        }
        objp->SetMinimum(0.0);
        objp->Draw();
        cAmp->Update();
        cAmp->SaveAs(imgName.c_str());
        gPad->SetLogy(0);

      }

      // Monitoring elements plots

      imgNameMEPnPedG01[iCanvas-1] = "";

      objp = 0;
      switch ( iCanvas ) {
        case 1:
          objp = i05_[ism-1];
          break;
        case 2:
          objp = i06_[ism-1];
          break;
        case 3:
        case 4:
        case 5:
        case 6:
        case 7:
        case 8:
          objp = 0;
          break;
        default:
          break;
      }

      if ( objp ) {

        meName = objp->GetName();

        replace(meName.begin(), meName.end(), ' ', '_');
        imgNameMEPnPedG01[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgNameMEPnPedG01[iCanvas-1];

        cPed->cd();
        gStyle->SetOptStat("euo");
        objp->SetStats(kTRUE);
//        if ( objp->GetMaximum(histMax) > 0. ) {
//          gPad->SetLogy(1);
//        } else {
//          gPad->SetLogy(0);
//        }
        objp->SetMinimum(0.0);
        objp->Draw();
        cPed->Update();
        cPed->SaveAs(imgName.c_str());
        gPad->SetLogy(0);

      }

      imgNameMEPnPedG16[iCanvas-1] = "";

      objp = 0;
      switch ( iCanvas ) {
      case 1:
        objp = i13_[ism-1];
        break;
      case 2:
        objp = i14_[ism-1];
        break;
      case 3:
      case 4:
      case 5:
      case 6:
      case 7:
      case 8:
        objp = 0;
        break;
      default:
        break;
      }

      if ( objp ) {

        meName = objp->GetName();

        replace(meName.begin(), meName.end(), ' ', '_');
        imgNameMEPnPedG16[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgNameMEPnPedG16[iCanvas-1];

        cPed->cd();
        gStyle->SetOptStat("euo");
        objp->SetStats(kTRUE);
//        if ( objp->GetMaximum(histMax) > 0. ) {
//          gPad->SetLogy(1);
//        } else {
//          gPad->SetLogy(0);
//        }
        objp->SetMinimum(0.0);
        objp->Draw();
        cPed->Update();
        cPed->SaveAs(imgName.c_str());
        gPad->SetLogy(0);

      }

      imgNameMEPnRmsPedG01[iCanvas-1] = "";

      obj1f = 0;
      switch ( iCanvas ) {
      case 1:
        if ( mepnprms01_[ism-1] ) obj1f =  UtilsClient::getHisto<TH1F*>(mepnprms01_[ism-1]);
        break;
      case 2:
        if ( mepnprms02_[ism-1] ) obj1f =  UtilsClient::getHisto<TH1F*>(mepnprms02_[ism-1]);
        break;
      case 3:
      case 4:
      case 5:
      case 6:
      case 7:
      case 8:
        obj1f = 0;
        break;
      default:
        break;
      }

      if ( obj1f ) {

        meName = obj1f->GetName();

        replace(meName.begin(), meName.end(), ' ', '_');
        imgNameMEPnRmsPedG01[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgNameMEPnRmsPedG01[iCanvas-1];

        cPed->cd();
        gStyle->SetOptStat("euomr");
        obj1f->SetStats(kTRUE);
//        if ( obj1f->GetMaximum(histMax) > 0. ) {
//          gPad->SetLogy(1);
//        } else {
//          gPad->SetLogy(0);
//        }
        obj1f->SetMinimum(0.0);
        obj1f->Draw();
        cPed->Update();
        cPed->SaveAs(imgName.c_str());
        gPad->SetLogy(0);

      }

      imgNameMEPnRmsPedG16[iCanvas-1] = "";

      obj1f = 0;
      switch ( iCanvas ) {
      case 1:
        if ( mepnprms05_[ism-1] ) obj1f =  UtilsClient::getHisto<TH1F*>(mepnprms05_[ism-1]);
        break;
      case 2:
        if ( mepnprms06_[ism-1] ) obj1f =  UtilsClient::getHisto<TH1F*>(mepnprms06_[ism-1]);
        break;
      case 3:
      case 4:
      case 5:
      case 6:
      case 7:
      case 8:
        obj1f = 0;
        break;
      default:
        break;
      }

      if ( obj1f ) {

        meName = obj1f->GetName();

        replace(meName.begin(), meName.end(), ' ', '_');
        imgNameMEPnRmsPedG16[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgNameMEPnRmsPedG16[iCanvas-1];

        cPed->cd();
        gStyle->SetOptStat("euomr");
        obj1f->SetStats(kTRUE);
//        if ( obj1f->GetMaximum(histMax) > 0. ) {
//          gPad->SetLogy(1);
//        } else {
//          gPad->SetLogy(0);
//        }
        obj1f->SetMinimum(0.0);
        obj1f->Draw();
        cPed->Update();
        cPed->SaveAs(imgName.c_str());
        gPad->SetLogy(0);

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

    for ( int iCanvas = 1 ; iCanvas <= 4 ; iCanvas++ ) {

      // skip unused wavelengths
      if ( iCanvas == 3 || iCanvas == 4 ) continue;

      if ( imgNameQual[iCanvas-1].size() != 0 )
        htmlFile << "<td colspan=\"2\"><img src=\"" << imgNameQual[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td colspan=\"2\"><img src=\"" << " " << "\"></td>" << endl;

    }

    htmlFile << "</tr>" << endl;

    htmlFile << "<tr>" << endl;

    for ( int iCanvas = 1 ; iCanvas <= 4 ; iCanvas++ ) {

      // skip unused wavelengths
      if ( iCanvas == 3 || iCanvas == 4 ) continue;

      if ( imgNameAmp[iCanvas-1].size() != 0 )
        htmlFile << "<td><img src=\"" << imgNameAmp[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

      if ( imgNameAmpoPN[iCanvas-1].size() != 0 )
        htmlFile << "<td><img src=\"" << imgNameAmpoPN[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

    }

    htmlFile << "</tr>" << endl;

    htmlFile << "<tr>" << endl;

    for ( int iCanvas = 1 ; iCanvas <= 4 ; iCanvas++ ) {

      // skip unused wavelengths
      if ( iCanvas == 3 || iCanvas == 4 ) continue;

      if ( imgNameAmp[4+iCanvas-1].size() != 0 )
        htmlFile << "<td><img src=\"" << imgNameAmp[4+iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

      if ( imgNameAmpoPN[4+iCanvas-1].size() != 0 )
        htmlFile << "<td><img src=\"" << imgNameAmpoPN[4+iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

    }

    htmlFile << "</tr>" << endl;

    htmlFile << "<tr>" << endl;

    for ( int iCanvas = 1 ; iCanvas <= 4 ; iCanvas++ ) {

      // skip unused wavelengths
      if ( iCanvas == 3 || iCanvas == 4 ) continue;

      if ( imgNameTim[iCanvas-1].size() != 0 )
        htmlFile << "<td><img src=\"" << imgNameTim[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

      if ( imgNameShape[iCanvas-1].size() != 0 )
        htmlFile << "<td><img src=\"" << imgNameShape[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

    }

    htmlFile << "</tr>" << endl;

    htmlFile << "<tr>" << endl;

    for ( int iCanvas = 1 ; iCanvas <= 4 ; iCanvas++ ) {

      // skip unused wavelengths
      if ( iCanvas == 3 || iCanvas == 4 ) continue;

      if ( imgNameTimav[iCanvas-1].size() != 0 )
        htmlFile << "<td><img src=\"" << imgNameTimav[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

      if ( imgNameTimrms[iCanvas-1].size() != 0 )
        htmlFile << "<td><img src=\"" << imgNameTimrms[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

    }

    htmlFile << "</tr>" << endl;

    htmlFile << "<tr>" << endl;

    for ( int iCanvas = 1 ; iCanvas <= 4 ; iCanvas++ ) {

      // skip unused wavelengths
      if ( iCanvas == 3 || iCanvas == 4 ) continue;

      if ( imgNameTim[4+iCanvas-1].size() != 0 )
        htmlFile << "<td><img src=\"" << imgNameTim[4+iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

      if ( imgNameShape[4+iCanvas-1].size() != 0 )
        htmlFile << "<td><img src=\"" << imgNameShape[4+iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

    }

    htmlFile << "</tr>" << endl;

    htmlFile << "<tr>" << endl;

    for ( int iCanvas = 1 ; iCanvas <= 4 ; iCanvas++ ) {

      // skip unused wavelengths
      if ( iCanvas == 3 || iCanvas == 4 ) continue;

      if ( imgNameTimav[4+iCanvas-1].size() != 0 )
        htmlFile << "<td><img src=\"" << imgNameTimav[4+iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

      if ( imgNameTimrms[4+iCanvas-1].size() != 0 )
        htmlFile << "<td><img src=\"" << imgNameTimrms[4+iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

    }

    htmlFile << "</tr>" << endl;

    htmlFile << "<tr align=\"center\">" << endl;

    for ( int iCanvas = 1 ; iCanvas <= 4 ; iCanvas++ ) {

      // skip unused wavelengths
      if ( iCanvas == 3 || iCanvas == 4 ) continue;

      htmlFile << "<td colspan=\"2\">Led " << iCanvas << "</td>" << endl;

    }

    htmlFile << "</tr>" << endl;
    htmlFile << "</table>" << endl;
    htmlFile << "<br>" << endl;

    htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
    htmlFile << "<tr align=\"center\">" << endl;

    for ( int iCanvas = 1 ; iCanvas <= 4 ; iCanvas++ ) {

      // skip unused wavelengths
      if ( iCanvas == 3 || iCanvas == 4 ) continue;

      if ( imgNameMEPnQualG01[iCanvas-1].size() != 0 )
        htmlFile << "<td colspan=\"2\"><img src=\"" << imgNameMEPnQualG01[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td colspan=\"2\"><img src=\"" << " " << "\"></td>" << endl;

    }

    htmlFile << "</tr>" << endl;

    htmlFile << "<tr>" << endl;

    for ( int iCanvas = 1 ; iCanvas <= 4 ; iCanvas++ ) {

      // skip unused wavelengths
      if ( iCanvas == 3 || iCanvas == 4 ) continue;

      if ( imgNameMEPnPedG01[iCanvas-1].size() != 0 )
        htmlFile << "<td><img src=\"" << imgNameMEPnPedG01[iCanvas-1] << "\"></td>" << endl;
      else
       htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

      if ( imgNameMEPnRmsPedG01[iCanvas-1].size() != 0 )
        htmlFile << "<td><img src=\"" << imgNameMEPnRmsPedG01[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

      if ( imgNameMEPnG01[iCanvas-1].size() != 0 )
        htmlFile << "<td><img src=\"" << imgNameMEPnG01[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

    }

    htmlFile << "</tr>" << endl;

    htmlFile << "<tr align=\"center\">" << endl;

    for ( int iCanvas = 1 ; iCanvas <= 4 ; iCanvas++ ) {

      // skip unused wavelengths
      if ( iCanvas == 3 || iCanvas == 4 ) continue;

      htmlFile << "<td colspan=\"1\"> </td> <td colspan=\"1\">Led " << iCanvas << " - PN Gain 1</td> <td colspan=\"1\">" << endl;

    }

    htmlFile << "</tr>" << endl;

    htmlFile << "<tr align=\"center\">" << endl;

    for ( int iCanvas = 1 ; iCanvas <= 4 ; iCanvas++ ) {

      // skip unused wavelengths
      if ( iCanvas == 3 || iCanvas == 4 ) continue;

      if ( imgNameMEPnQualG16[iCanvas-1].size() != 0 )
        htmlFile << "<td colspan=\"2\"><img src=\"" << imgNameMEPnQualG16[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td colspan=\"2\"><img src=\"" << " " << "\"></td>" << endl;

    }

    htmlFile << "</tr>" << endl;

    htmlFile << "<tr>" << endl;

    for ( int iCanvas = 1 ; iCanvas <= 4 ; iCanvas++ ) {

      // skip unused wavelengths
      if ( iCanvas == 3 || iCanvas == 4 ) continue;

      if ( imgNameMEPnPedG16[iCanvas-1].size() != 0 )
        htmlFile << "<td><img src=\"" << imgNameMEPnPedG16[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

      if ( imgNameMEPnRmsPedG16[iCanvas-1].size() != 0 )
        htmlFile << "<td><img src=\"" << imgNameMEPnRmsPedG16[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

      if ( imgNameMEPnG16[iCanvas-1].size() != 0 )
        htmlFile << "<td><img src=\"" << imgNameMEPnG16[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

    }

    htmlFile << "<tr align=\"center\">" << endl;

    for ( int iCanvas = 1 ; iCanvas <= 4 ; iCanvas++ ) {

      // skip unused wavelengths
      if ( iCanvas == 3 || iCanvas == 4 ) continue;

      htmlFile << "<td colspan=\"1\"> </td> <td colspan=\"1\">Led " << iCanvas << " - PN Gain 16</td> <td colspan=\"1\"> </td>" << endl;

    }

    htmlFile << "</tr>" << endl;

    htmlFile << "</tr>" << endl;
    htmlFile << "</table>" << endl;

    htmlFile << "<br>" << endl;

  }

  delete cQual;
  delete cQualPN;
  delete cAmp;
  delete cTim;
  delete cTimav;
  delete cTimrms;
  delete cShape;
  delete cAmpoPN;
  delete cPed;

  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  htmlFile.close();

}

