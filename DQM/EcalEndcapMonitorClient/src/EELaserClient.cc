/*
 * \file EELaserClient.cc
 *
 * $Date: 2008/10/07 18:34:23 $
 * $Revision: 1.106 $
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

#include "OnlineDB/EcalCondDB/interface/MonLaserBlueDat.h"
#include "OnlineDB/EcalCondDB/interface/MonLaserGreenDat.h"
#include "OnlineDB/EcalCondDB/interface/MonLaserIRedDat.h"
#include "OnlineDB/EcalCondDB/interface/MonLaserRedDat.h"
#include "OnlineDB/EcalCondDB/interface/MonPNBlueDat.h"
#include "OnlineDB/EcalCondDB/interface/MonPNGreenDat.h"
#include "OnlineDB/EcalCondDB/interface/MonPNIRedDat.h"
#include "OnlineDB/EcalCondDB/interface/MonPNRedDat.h"
#include "OnlineDB/EcalCondDB/interface/MonTimingLaserBlueCrystalDat.h"
#include "OnlineDB/EcalCondDB/interface/MonTimingLaserGreenCrystalDat.h"
#include "OnlineDB/EcalCondDB/interface/MonTimingLaserIRedCrystalDat.h"
#include "OnlineDB/EcalCondDB/interface/MonTimingLaserRedCrystalDat.h"
#include "OnlineDB/EcalCondDB/interface/RunCrystalErrorsDat.h"
#include "OnlineDB/EcalCondDB/interface/RunTTErrorsDat.h"
#include "OnlineDB/EcalCondDB/interface/RunPNErrorsDat.h"

#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"

#include "CondTools/Ecal/interface/EcalErrorDictionary.h"

#include "DQM/EcalCommon/interface/EcalErrorMask.h"
#include "DQM/EcalCommon/interface/UtilsClient.h"
#include "DQM/EcalCommon/interface/LogicID.h"
#include "DQM/EcalCommon/interface/Numbers.h"

#include <DQM/EcalEndcapMonitorClient/interface/EELaserClient.h>

using namespace cms;
using namespace edm;
using namespace std;

EELaserClient::EELaserClient(const ParameterSet& ps) {

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
    h05_[ism-1] = 0;
    h06_[ism-1] = 0;
    h07_[ism-1] = 0;
    h08_[ism-1] = 0;

    h09_[ism-1] = 0;
    h10_[ism-1] = 0;
    h11_[ism-1] = 0;
    h12_[ism-1] = 0;

    h13_[ism-1] = 0;
    h14_[ism-1] = 0;
    h15_[ism-1] = 0;
    h16_[ism-1] = 0;
    h17_[ism-1] = 0;
    h18_[ism-1] = 0;
    h19_[ism-1] = 0;
    h20_[ism-1] = 0;

    h21_[ism-1] = 0;
    h22_[ism-1] = 0;
    h23_[ism-1] = 0;
    h24_[ism-1] = 0;

    hs01_[ism-1] = 0;
    hs02_[ism-1] = 0;
    hs03_[ism-1] = 0;
    hs04_[ism-1] = 0;

    hs05_[ism-1] = 0;
    hs06_[ism-1] = 0;
    hs07_[ism-1] = 0;
    hs08_[ism-1] = 0;

    i01_[ism-1] = 0;
    i02_[ism-1] = 0;
    i03_[ism-1] = 0;
    i04_[ism-1] = 0;
    i05_[ism-1] = 0;
    i06_[ism-1] = 0;
    i07_[ism-1] = 0;
    i08_[ism-1] = 0;

    i09_[ism-1] = 0;
    i10_[ism-1] = 0;
    i11_[ism-1] = 0;
    i12_[ism-1] = 0;
    i13_[ism-1] = 0;
    i14_[ism-1] = 0;
    i15_[ism-1] = 0;
    i16_[ism-1] = 0;

  }

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    meg01_[ism-1] = 0;
    meg02_[ism-1] = 0;
    meg03_[ism-1] = 0;
    meg04_[ism-1] = 0;

    meg05_[ism-1] = 0;
    meg06_[ism-1] = 0;
    meg07_[ism-1] = 0;
    meg08_[ism-1] = 0;
    meg09_[ism-1] = 0;
    meg10_[ism-1] = 0;
    meg11_[ism-1] = 0;
    meg12_[ism-1] = 0;

    mea01_[ism-1] = 0;
    mea02_[ism-1] = 0;
    mea03_[ism-1] = 0;
    mea04_[ism-1] = 0;
    mea05_[ism-1] = 0;
    mea06_[ism-1] = 0;
    mea07_[ism-1] = 0;
    mea08_[ism-1] = 0;

    met01_[ism-1] = 0;
    met02_[ism-1] = 0;
    met03_[ism-1] = 0;
    met04_[ism-1] = 0;
    met05_[ism-1] = 0;
    met06_[ism-1] = 0;
    met07_[ism-1] = 0;
    met08_[ism-1] = 0;

    metav01_[ism-1] = 0;
    metav02_[ism-1] = 0;
    metav03_[ism-1] = 0;
    metav04_[ism-1] = 0;
    metav05_[ism-1] = 0;
    metav06_[ism-1] = 0;
    metav07_[ism-1] = 0;
    metav08_[ism-1] = 0;

    metrms01_[ism-1] = 0;
    metrms02_[ism-1] = 0;
    metrms03_[ism-1] = 0;
    metrms04_[ism-1] = 0;
    metrms05_[ism-1] = 0;
    metrms06_[ism-1] = 0;
    metrms07_[ism-1] = 0;
    metrms08_[ism-1] = 0;

    meaopn01_[ism-1] = 0;
    meaopn02_[ism-1] = 0;
    meaopn03_[ism-1] = 0;
    meaopn04_[ism-1] = 0;
    meaopn05_[ism-1] = 0;
    meaopn06_[ism-1] = 0;
    meaopn07_[ism-1] = 0;
    meaopn08_[ism-1] = 0;

    mepnprms01_[ism-1] = 0;
    mepnprms02_[ism-1] = 0;
    mepnprms03_[ism-1] = 0;
    mepnprms04_[ism-1] = 0;
    mepnprms05_[ism-1] = 0;
    mepnprms06_[ism-1] = 0;
    mepnprms07_[ism-1] = 0;
    mepnprms08_[ism-1] = 0;

    me_hs01_[ism-1] = 0;
    me_hs02_[ism-1] = 0;
    me_hs03_[ism-1] = 0;
    me_hs04_[ism-1] = 0;
    me_hs05_[ism-1] = 0;
    me_hs06_[ism-1] = 0;
    me_hs07_[ism-1] = 0;
    me_hs08_[ism-1] = 0;

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

EELaserClient::~EELaserClient() {

}

void EELaserClient::beginJob(DQMStore* dqmStore) {

  dqmStore_ = dqmStore;

  if ( debug_ ) cout << "EELaserClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EELaserClient::beginRun(void) {

  if ( debug_ ) cout << "EELaserClient: beginRun" << endl;

  jevt_ = 0;

  this->setup();

}

void EELaserClient::endJob(void) {

  if ( debug_ ) cout << "EELaserClient: endJob, ievt = " << ievt_ << endl;

  this->cleanup();

}

void EELaserClient::endRun(void) {

  if ( debug_ ) cout << "EELaserClient: endRun, jevt = " << jevt_ << endl;

  this->cleanup();

}

void EELaserClient::setup(void) {

  char histo[200];

  dqmStore_->setCurrentFolder( prefixME_ + "/EELaserClient" );

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( meg01_[ism-1] ) dqmStore_->removeElement( meg01_[ism-1]->getName() );
    sprintf(histo, "EELT laser quality L1 %s", Numbers::sEE(ism).c_str());
    meg01_[ism-1] = dqmStore_->book2D(histo, histo, 50, Numbers::ix0EE(ism)+0., Numbers::ix0EE(ism)+50., 50, Numbers::iy0EE(ism)+0., Numbers::iy0EE(ism)+50.);
    meg01_[ism-1]->setAxisTitle("jx", 1);
    meg01_[ism-1]->setAxisTitle("jy", 2);
    if ( meg02_[ism-1] ) dqmStore_->removeElement( meg02_[ism-1]->getName() );
    sprintf(histo, "EELT laser quality L2 %s", Numbers::sEE(ism).c_str());
    meg02_[ism-1] = dqmStore_->book2D(histo, histo, 50, Numbers::ix0EE(ism)+0., Numbers::ix0EE(ism)+50., 50, Numbers::iy0EE(ism)+0., Numbers::iy0EE(ism)+50.);
    meg02_[ism-1]->setAxisTitle("jx", 1);
    meg02_[ism-1]->setAxisTitle("jy", 2);
    if ( meg03_[ism-1] ) dqmStore_->removeElement( meg03_[ism-1]->getName() );
    sprintf(histo, "EELT laser quality L3 %s", Numbers::sEE(ism).c_str());
    meg03_[ism-1] = dqmStore_->book2D(histo, histo, 50, Numbers::ix0EE(ism)+0., Numbers::ix0EE(ism)+50., 50, Numbers::iy0EE(ism)+0., Numbers::iy0EE(ism)+50.);
    meg03_[ism-1]->setAxisTitle("jx", 1);
    meg03_[ism-1]->setAxisTitle("jy", 2);
    if ( meg04_[ism-1] ) dqmStore_->removeElement( meg04_[ism-1]->getName() );
    sprintf(histo, "EELT laser quality L4 %s", Numbers::sEE(ism).c_str());
    meg04_[ism-1] = dqmStore_->book2D(histo, histo, 50, Numbers::ix0EE(ism)+0., Numbers::ix0EE(ism)+50., 50, Numbers::iy0EE(ism)+0., Numbers::iy0EE(ism)+50.);
    meg04_[ism-1]->setAxisTitle("jx", 1);
    meg04_[ism-1]->setAxisTitle("jy", 2);

    if ( meg05_[ism-1] ) dqmStore_->removeElement( meg05_[ism-1]->getName() );
    sprintf(histo, "EELT laser quality L1 PNs G01 %s", Numbers::sEE(ism).c_str());
    meg05_[ism-1] = dqmStore_->book2D(histo, histo, 10, 0., 10., 1, 0., 5.);
    meg05_[ism-1]->setAxisTitle("pseudo-strip", 1);
    meg05_[ism-1]->setAxisTitle("channel", 2);
    if ( meg06_[ism-1] ) dqmStore_->removeElement( meg06_[ism-1]->getName() );
    sprintf(histo, "EELT laser quality L2 PNs G01 %s", Numbers::sEE(ism).c_str());
    meg06_[ism-1] = dqmStore_->book2D(histo, histo, 10, 0., 10., 1, 0., 5.);
    meg06_[ism-1]->setAxisTitle("pseudo-strip", 1);
    meg06_[ism-1]->setAxisTitle("channel", 2);
    if ( meg07_[ism-1] ) dqmStore_->removeElement( meg07_[ism-1]->getName() );
    sprintf(histo, "EELT laser quality L3 PNs G01 %s", Numbers::sEE(ism).c_str());
    meg07_[ism-1] = dqmStore_->book2D(histo, histo, 10, 0., 10., 1, 0., 5.);
    meg07_[ism-1]->setAxisTitle("pseudo-strip", 1);
    meg07_[ism-1]->setAxisTitle("channel", 2);
    if ( meg08_[ism-1] ) dqmStore_->removeElement( meg08_[ism-1]->getName() );
    sprintf(histo, "EELT laser quality L4 PNs G01 %s", Numbers::sEE(ism).c_str());
    meg08_[ism-1] = dqmStore_->book2D(histo, histo, 10, 0., 10., 1, 0., 5.);
    meg08_[ism-1]->setAxisTitle("pseudo-strip", 1);
    meg08_[ism-1]->setAxisTitle("channel", 2);
    if ( meg09_[ism-1] ) dqmStore_->removeElement( meg09_[ism-1]->getName() );
    sprintf(histo, "EELT laser quality L1 PNs G16 %s", Numbers::sEE(ism).c_str());
    meg09_[ism-1] = dqmStore_->book2D(histo, histo, 10, 0., 10., 1, 0., 5.);
    meg09_[ism-1]->setAxisTitle("pseudo-strip", 1);
    meg09_[ism-1]->setAxisTitle("channel", 2);
    if ( meg10_[ism-1] ) dqmStore_->removeElement( meg10_[ism-1]->getName() );
    sprintf(histo, "EELT laser quality L2 PNs G16 %s", Numbers::sEE(ism).c_str());
    meg10_[ism-1] = dqmStore_->book2D(histo, histo, 10, 0., 10., 1, 0., 5.);
    meg10_[ism-1]->setAxisTitle("pseudo-strip", 1);
    meg10_[ism-1]->setAxisTitle("channel", 2);
    if ( meg11_[ism-1] ) dqmStore_->removeElement( meg11_[ism-1]->getName() );
    sprintf(histo, "EELT laser quality L3 PNs G16 %s", Numbers::sEE(ism).c_str());
    meg11_[ism-1] = dqmStore_->book2D(histo, histo, 10, 0., 10., 1, 0., 5.);
    meg11_[ism-1]->setAxisTitle("pseudo-strip", 1);
    meg11_[ism-1]->setAxisTitle("channel", 2);
    if ( meg12_[ism-1] ) dqmStore_->removeElement( meg12_[ism-1]->getName() );
    sprintf(histo, "EELT laser quality L4 PNs G16 %s", Numbers::sEE(ism).c_str());
    meg12_[ism-1] = dqmStore_->book2D(histo, histo, 10, 0., 10., 1, 0., 5.);
    meg12_[ism-1]->setAxisTitle("pseudo-strip", 1);
    meg12_[ism-1]->setAxisTitle("channel", 2);

    if ( mea01_[ism-1] ) dqmStore_->removeElement( mea01_[ism-1]->getName() );;
    sprintf(histo, "EELT amplitude L1A %s", Numbers::sEE(ism).c_str());
    mea01_[ism-1] = dqmStore_->book1D(histo, histo, 850, 0., 850.);
    mea01_[ism-1]->setAxisTitle("channel", 1);
    mea01_[ism-1]->setAxisTitle("amplitude", 2);
    if ( mea02_[ism-1] ) dqmStore_->removeElement( mea02_[ism-1]->getName() );
    sprintf(histo, "EELT amplitude L2A %s", Numbers::sEE(ism).c_str());
    mea02_[ism-1] = dqmStore_->book1D(histo, histo, 850, 0., 850.);
    mea02_[ism-1]->setAxisTitle("channel", 1);
    mea02_[ism-1]->setAxisTitle("amplitude", 2);
    if ( mea03_[ism-1] ) dqmStore_->removeElement( mea03_[ism-1]->getName() );
    sprintf(histo, "EELT amplitude L3A %s", Numbers::sEE(ism).c_str());
    mea03_[ism-1] = dqmStore_->book1D(histo, histo, 850, 0., 850.);
    mea03_[ism-1]->setAxisTitle("channel", 1);
    mea03_[ism-1]->setAxisTitle("amplitude", 2);
    if ( mea04_[ism-1] ) dqmStore_->removeElement( mea04_[ism-1]->getName() );
    sprintf(histo, "EELT amplitude L4A %s", Numbers::sEE(ism).c_str());
    mea04_[ism-1] = dqmStore_->book1D(histo, histo, 850, 0., 850.);
    mea04_[ism-1]->setAxisTitle("channel", 1);
    mea04_[ism-1]->setAxisTitle("amplitude", 2);
    if ( mea05_[ism-1] ) dqmStore_->removeElement( mea05_[ism-1]->getName() );;
    sprintf(histo, "EELT amplitude L1B %s", Numbers::sEE(ism).c_str());
    mea05_[ism-1] = dqmStore_->book1D(histo, histo, 850, 0., 850.);
    mea05_[ism-1]->setAxisTitle("channel", 1);
    mea05_[ism-1]->setAxisTitle("amplitude", 2);
    if ( mea06_[ism-1] ) dqmStore_->removeElement( mea06_[ism-1]->getName() );
    sprintf(histo, "EELT amplitude L2B %s", Numbers::sEE(ism).c_str());
    mea06_[ism-1] = dqmStore_->book1D(histo, histo, 850, 0., 850.);
    mea06_[ism-1]->setAxisTitle("channel", 1);
    mea06_[ism-1]->setAxisTitle("amplitude", 2);
    if ( mea07_[ism-1] ) dqmStore_->removeElement( mea07_[ism-1]->getName() );
    sprintf(histo, "EELT amplitude L3B %s", Numbers::sEE(ism).c_str());
    mea07_[ism-1] = dqmStore_->book1D(histo, histo, 850, 0., 850.);
    mea07_[ism-1]->setAxisTitle("channel", 1);
    mea07_[ism-1]->setAxisTitle("amplitude", 2);
    if ( mea08_[ism-1] ) dqmStore_->removeElement( mea08_[ism-1]->getName() );
    sprintf(histo, "EELT amplitude L4B %s", Numbers::sEE(ism).c_str());
    mea08_[ism-1] = dqmStore_->book1D(histo, histo, 850, 0., 850.);
    mea08_[ism-1]->setAxisTitle("channel", 1);
    mea08_[ism-1]->setAxisTitle("amplitude", 2);

    if ( met01_[ism-1] ) dqmStore_->removeElement( met01_[ism-1]->getName() );
    sprintf(histo, "EELT laser timing L1A %s", Numbers::sEE(ism).c_str());
    met01_[ism-1] = dqmStore_->book1D(histo, histo, 850, 0., 850.);
    met01_[ism-1]->setAxisTitle("channel", 1);
    met01_[ism-1]->setAxisTitle("jitter", 2);
    if ( met02_[ism-1] ) dqmStore_->removeElement( met02_[ism-1]->getName() );
    sprintf(histo, "EELT laser timing L2A %s", Numbers::sEE(ism).c_str());
    met02_[ism-1] = dqmStore_->book1D(histo, histo, 850, 0., 850.);
    met02_[ism-1]->setAxisTitle("channel", 1);
    met02_[ism-1]->setAxisTitle("jitter", 2);
    if ( met03_[ism-1] ) dqmStore_->removeElement( met03_[ism-1]->getName() );
    sprintf(histo, "EELT laser timing L3A %s", Numbers::sEE(ism).c_str());
    met03_[ism-1] = dqmStore_->book1D(histo, histo, 850, 0., 850.);
    met03_[ism-1]->setAxisTitle("channel", 1);
    met03_[ism-1]->setAxisTitle("jitter", 2);
    if ( met04_[ism-1] ) dqmStore_->removeElement( met04_[ism-1]->getName() );
    sprintf(histo, "EELT laser timing L4A %s", Numbers::sEE(ism).c_str());
    met04_[ism-1] = dqmStore_->book1D(histo, histo, 850, 0., 850.);
    met04_[ism-1]->setAxisTitle("channel", 1);
    met04_[ism-1]->setAxisTitle("jitter", 2);
    if ( met05_[ism-1] ) dqmStore_->removeElement( met05_[ism-1]->getName() );
    sprintf(histo, "EELT laser timing L1B %s", Numbers::sEE(ism).c_str());
    met05_[ism-1] = dqmStore_->book1D(histo, histo, 850, 0., 850.);
    met05_[ism-1]->setAxisTitle("channel", 1);
    met05_[ism-1]->setAxisTitle("jitter", 2);
    if ( met06_[ism-1] ) dqmStore_->removeElement( met06_[ism-1]->getName() );
    sprintf(histo, "EELT laser timing L2B %s", Numbers::sEE(ism).c_str());
    met06_[ism-1] = dqmStore_->book1D(histo, histo, 850, 0., 850.);
    met06_[ism-1]->setAxisTitle("channel", 1);
    met06_[ism-1]->setAxisTitle("jitter", 2);
    if ( met07_[ism-1] ) dqmStore_->removeElement( met07_[ism-1]->getName() );
    sprintf(histo, "EELT laser timing L3B %s", Numbers::sEE(ism).c_str());
    met07_[ism-1] = dqmStore_->book1D(histo, histo, 850, 0., 850.);
    met07_[ism-1]->setAxisTitle("channel", 1);
    met07_[ism-1]->setAxisTitle("jitter", 2);
    if ( met08_[ism-1] ) dqmStore_->removeElement( met08_[ism-1]->getName() );
    sprintf(histo, "EELT laser timing L4B %s", Numbers::sEE(ism).c_str());
    met08_[ism-1] = dqmStore_->book1D(histo, histo, 850, 0., 850.);
    met08_[ism-1]->setAxisTitle("channel", 1);
    met08_[ism-1]->setAxisTitle("jitter", 2);

    if ( metav01_[ism-1] ) dqmStore_->removeElement( metav01_[ism-1]->getName() );
    sprintf(histo, "EELT laser timing mean L1A %s", Numbers::sEE(ism).c_str());
    metav01_[ism-1] = dqmStore_->book1D(histo, histo, 100, 0., 10.);
    metav01_[ism-1]->setAxisTitle("mean", 1);
    if ( metav02_[ism-1] ) dqmStore_->removeElement( metav02_[ism-1]->getName() );
    sprintf(histo, "EELT laser timing mean L2A %s", Numbers::sEE(ism).c_str());
    metav02_[ism-1] = dqmStore_->book1D(histo, histo, 100, 0., 10.);
    metav02_[ism-1]->setAxisTitle("mean", 1);
    if ( metav03_[ism-1] ) dqmStore_->removeElement( metav03_[ism-1]->getName() );
    sprintf(histo, "EELT laser timing mean L3A %s", Numbers::sEE(ism).c_str());
    metav03_[ism-1] = dqmStore_->book1D(histo, histo, 100, 0., 10.);
    metav03_[ism-1]->setAxisTitle("mean", 1);
    if ( metav04_[ism-1] ) dqmStore_->removeElement( metav04_[ism-1]->getName() );
    sprintf(histo, "EELT laser timing mean L4A %s", Numbers::sEE(ism).c_str());
    metav04_[ism-1] = dqmStore_->book1D(histo, histo, 100, 0., 10.);
    metav04_[ism-1]->setAxisTitle("mean", 1);
    if ( metav05_[ism-1] ) dqmStore_->removeElement( metav05_[ism-1]->getName() );
    sprintf(histo, "EELT laser timing mean L1B %s", Numbers::sEE(ism).c_str());
    metav05_[ism-1] = dqmStore_->book1D(histo, histo, 100, 0., 10.);
    metav05_[ism-1]->setAxisTitle("mean", 1);
    if ( metav06_[ism-1] ) dqmStore_->removeElement( metav06_[ism-1]->getName() );
    sprintf(histo, "EELT laser timing mean L2B %s", Numbers::sEE(ism).c_str());
    metav06_[ism-1] = dqmStore_->book1D(histo, histo, 100, 0., 10.);
    metav06_[ism-1]->setAxisTitle("mean", 1);
    if ( metav07_[ism-1] ) dqmStore_->removeElement( metav07_[ism-1]->getName() );
    sprintf(histo, "EELT laser timing mean L3B %s", Numbers::sEE(ism).c_str());
    metav07_[ism-1] = dqmStore_->book1D(histo, histo, 100, 0., 10.);
    metav07_[ism-1]->setAxisTitle("mean", 1);
    if ( metav08_[ism-1] ) dqmStore_->removeElement( metav08_[ism-1]->getName() );
    sprintf(histo, "EELT laser timing mean L4B %s", Numbers::sEE(ism).c_str());
    metav08_[ism-1] = dqmStore_->book1D(histo, histo, 100, 0., 10.);
    metav08_[ism-1]->setAxisTitle("mean", 1);

    if ( metrms01_[ism-1] ) dqmStore_->removeElement( metrms01_[ism-1]->getName() );
    sprintf(histo, "EELT laser timing rms L1A %s", Numbers::sEE(ism).c_str());
    metrms01_[ism-1] = dqmStore_->book1D(histo, histo, 100, 0., 0.5);
    metrms01_[ism-1]->setAxisTitle("rms", 1);
    if ( metrms02_[ism-1] ) dqmStore_->removeElement( metrms02_[ism-1]->getName() );
    sprintf(histo, "EELT laser timing rms L2A %s", Numbers::sEE(ism).c_str());
    metrms02_[ism-1] = dqmStore_->book1D(histo, histo, 100, 0., 0.5);
    metrms02_[ism-1]->setAxisTitle("rms", 1);
    if ( metrms03_[ism-1] ) dqmStore_->removeElement( metrms03_[ism-1]->getName() );
    sprintf(histo, "EELT laser timing rms L3A %s", Numbers::sEE(ism).c_str());
    metrms03_[ism-1] = dqmStore_->book1D(histo, histo, 100, 0., 0.5);
    metrms03_[ism-1]->setAxisTitle("rms", 1);
    if ( metrms04_[ism-1] ) dqmStore_->removeElement( metrms04_[ism-1]->getName() );
    sprintf(histo, "EELT laser timing rms L4A %s", Numbers::sEE(ism).c_str());
    metrms04_[ism-1] = dqmStore_->book1D(histo, histo, 100, 0., 0.5);
    metrms04_[ism-1]->setAxisTitle("rms", 1);
    if ( metrms05_[ism-1] ) dqmStore_->removeElement( metrms05_[ism-1]->getName() );
    sprintf(histo, "EELT laser timing rms L1B %s", Numbers::sEE(ism).c_str());
    metrms05_[ism-1] = dqmStore_->book1D(histo, histo, 100, 0., 0.5);
    metrms05_[ism-1]->setAxisTitle("rms", 1);
    if ( metrms06_[ism-1] ) dqmStore_->removeElement( metrms06_[ism-1]->getName() );
    sprintf(histo, "EELT laser timing rms L2B %s", Numbers::sEE(ism).c_str());
    metrms06_[ism-1] = dqmStore_->book1D(histo, histo, 100, 0., 0.5);
    metrms06_[ism-1]->setAxisTitle("rms", 1);
    if ( metrms07_[ism-1] ) dqmStore_->removeElement( metrms07_[ism-1]->getName() );
    sprintf(histo, "EELT laser timing rms L3B %s", Numbers::sEE(ism).c_str());
    metrms07_[ism-1] = dqmStore_->book1D(histo, histo, 100, 0., 0.5);
    metrms07_[ism-1]->setAxisTitle("rms", 1);
    if ( metrms08_[ism-1] ) dqmStore_->removeElement( metrms08_[ism-1]->getName() );
    sprintf(histo, "EELT laser timing rms L4B %s", Numbers::sEE(ism).c_str());
    metrms08_[ism-1] = dqmStore_->book1D(histo, histo, 100, 0., 0.5);
    metrms08_[ism-1]->setAxisTitle("rms", 1);

    if ( meaopn01_[ism-1] ) dqmStore_->removeElement( meaopn01_[ism-1]->getName() );
    sprintf(histo, "EELT amplitude over PN L1A %s", Numbers::sEE(ism).c_str());
    meaopn01_[ism-1] = dqmStore_->book1D(histo, histo, 850, 0., 850.);
    meaopn01_[ism-1]->setAxisTitle("channel", 1);
    meaopn01_[ism-1]->setAxisTitle("amplitude/PN", 2);
    if ( meaopn02_[ism-1] ) dqmStore_->removeElement( meaopn02_[ism-1]->getName() );
    sprintf(histo, "EELT amplitude over PN L2A %s", Numbers::sEE(ism).c_str());
    meaopn02_[ism-1] = dqmStore_->book1D(histo, histo, 850, 0., 850.);
    meaopn02_[ism-1]->setAxisTitle("channel", 1);
    meaopn02_[ism-1]->setAxisTitle("amplitude/PN", 2);
    if ( meaopn03_[ism-1] ) dqmStore_->removeElement( meaopn03_[ism-1]->getName() );
    sprintf(histo, "EELT amplitude over PN L3A %s", Numbers::sEE(ism).c_str());
    meaopn03_[ism-1] = dqmStore_->book1D(histo, histo, 850, 0., 850.);
    meaopn03_[ism-1]->setAxisTitle("channel", 1);
    meaopn03_[ism-1]->setAxisTitle("amplitude/PN", 2);
    if ( meaopn04_[ism-1] ) dqmStore_->removeElement( meaopn04_[ism-1]->getName() );
    sprintf(histo, "EELT amplitude over PN L4A %s", Numbers::sEE(ism).c_str());
    meaopn04_[ism-1] = dqmStore_->book1D(histo, histo, 850, 0., 850.);
    meaopn04_[ism-1]->setAxisTitle("channel", 1);
    meaopn04_[ism-1]->setAxisTitle("amplitude/PN", 2);
    if ( meaopn05_[ism-1] ) dqmStore_->removeElement( meaopn05_[ism-1]->getName() );
    sprintf(histo, "EELT amplitude over PN L1B %s", Numbers::sEE(ism).c_str());
    meaopn05_[ism-1] = dqmStore_->book1D(histo, histo, 850, 0., 850.);
    meaopn05_[ism-1]->setAxisTitle("channel", 1);
    meaopn05_[ism-1]->setAxisTitle("amplitude/PN", 2);
    if ( meaopn06_[ism-1] ) dqmStore_->removeElement( meaopn06_[ism-1]->getName() );
    sprintf(histo, "EELT amplitude over PN L2B %s", Numbers::sEE(ism).c_str());
    meaopn06_[ism-1] = dqmStore_->book1D(histo, histo, 850, 0., 850.);
    meaopn06_[ism-1]->setAxisTitle("channel", 1);
    meaopn06_[ism-1]->setAxisTitle("amplitude/PN", 2);
    if ( meaopn07_[ism-1] ) dqmStore_->removeElement( meaopn07_[ism-1]->getName() );
    sprintf(histo, "EELT amplitude over PN L3B %s", Numbers::sEE(ism).c_str());
    meaopn07_[ism-1] = dqmStore_->book1D(histo, histo, 850, 0., 850.);
    meaopn07_[ism-1]->setAxisTitle("channel", 1);
    meaopn07_[ism-1]->setAxisTitle("amplitude/PN", 2);
    if ( meaopn08_[ism-1] ) dqmStore_->removeElement( meaopn08_[ism-1]->getName() );
    sprintf(histo, "EELT amplitude over PN L4B %s", Numbers::sEE(ism).c_str());
    meaopn08_[ism-1] = dqmStore_->book1D(histo, histo, 850, 0., 850.);
    meaopn08_[ism-1]->setAxisTitle("channel", 1);
    meaopn08_[ism-1]->setAxisTitle("amplitude/PN", 2);

    if ( mepnprms01_[ism-1] ) dqmStore_->removeElement( mepnprms01_[ism-1]->getName() );
    sprintf(histo, "EEPDT PNs pedestal rms %s G01 L1", Numbers::sEE(ism).c_str());
    mepnprms01_[ism-1] = dqmStore_->book1D(histo, histo, 100, 0., 10.);
    mepnprms01_[ism-1]->setAxisTitle("rms", 1);
    if ( mepnprms02_[ism-1] ) dqmStore_->removeElement( mepnprms02_[ism-1]->getName() );
    sprintf(histo, "EEPDT PNs pedestal rms %s G01 L2", Numbers::sEE(ism).c_str());
    mepnprms02_[ism-1] = dqmStore_->book1D(histo, histo, 100, 0., 10.);
    mepnprms02_[ism-1]->setAxisTitle("rms", 1);
    if ( mepnprms03_[ism-1] ) dqmStore_->removeElement( mepnprms03_[ism-1]->getName() );
    sprintf(histo, "EEPDT PNs pedestal rms %s G01 L3", Numbers::sEE(ism).c_str());
    mepnprms03_[ism-1] = dqmStore_->book1D(histo, histo, 100, 0., 10.);
    mepnprms03_[ism-1]->setAxisTitle("rms", 1);
    if ( mepnprms04_[ism-1] ) dqmStore_->removeElement( mepnprms04_[ism-1]->getName() );
    sprintf(histo, "EEPDT PNs pedestal rms %s G01 L4", Numbers::sEE(ism).c_str());
    mepnprms04_[ism-1] = dqmStore_->book1D(histo, histo, 100, 0., 10.);
    mepnprms04_[ism-1]->setAxisTitle("rms", 1);
    if ( mepnprms05_[ism-1] ) dqmStore_->removeElement( mepnprms05_[ism-1]->getName() );
    sprintf(histo, "EEPDT PNs pedestal rms %s G16 L1", Numbers::sEE(ism).c_str());
    mepnprms05_[ism-1] = dqmStore_->book1D(histo, histo, 100, 0., 10.);
    mepnprms05_[ism-1]->setAxisTitle("rms", 1);
    if ( mepnprms06_[ism-1] ) dqmStore_->removeElement( mepnprms06_[ism-1]->getName() );
    sprintf(histo, "EEPDT PNs pedestal rms %s G16 L2", Numbers::sEE(ism).c_str());
    mepnprms06_[ism-1] = dqmStore_->book1D(histo, histo, 100, 0., 10.);
    mepnprms06_[ism-1]->setAxisTitle("rms", 1);
    if ( mepnprms07_[ism-1] ) dqmStore_->removeElement( mepnprms07_[ism-1]->getName() );
    sprintf(histo, "EEPDT PNs pedestal rms %s G16 L3", Numbers::sEE(ism).c_str());
    mepnprms07_[ism-1] = dqmStore_->book1D(histo, histo, 100, 0., 10.);
    mepnprms07_[ism-1]->setAxisTitle("rms", 1);
    if ( mepnprms08_[ism-1] ) dqmStore_->removeElement( mepnprms08_[ism-1]->getName() );
    sprintf(histo, "EEPDT PNs pedestal rms %s G16 L4", Numbers::sEE(ism).c_str());
    mepnprms08_[ism-1] = dqmStore_->book1D(histo, histo, 100, 0., 10.);
    mepnprms08_[ism-1]->setAxisTitle("rms", 1);

    if ( me_hs01_[ism-1] ) dqmStore_->removeElement( me_hs01_[ism-1]->getName() );
    sprintf(histo, "EELT laser shape L1A %s", Numbers::sEE(ism).c_str());
    me_hs01_[ism-1] = dqmStore_->book1D(histo, histo, 10, 0., 10.);
    me_hs01_[ism-1]->setAxisTitle("sample", 1);
    me_hs01_[ism-1]->setAxisTitle("amplitude", 2);
    if ( me_hs02_[ism-1] ) dqmStore_->removeElement( me_hs02_[ism-1]->getName() );
    sprintf(histo, "EELT laser shape L2A %s", Numbers::sEE(ism).c_str());
    me_hs02_[ism-1] = dqmStore_->book1D(histo, histo, 10, 0., 10.);
    me_hs02_[ism-1]->setAxisTitle("sample", 1);
    me_hs02_[ism-1]->setAxisTitle("amplitude", 2);
    if ( me_hs03_[ism-1] ) dqmStore_->removeElement( me_hs03_[ism-1]->getName() );
    sprintf(histo, "EELT laser shape L3A %s", Numbers::sEE(ism).c_str());
    me_hs03_[ism-1] = dqmStore_->book1D(histo, histo, 10, 0., 10.);
    me_hs03_[ism-1]->setAxisTitle("sample", 1);
    me_hs03_[ism-1]->setAxisTitle("amplitude", 2);
    if ( me_hs04_[ism-1] ) dqmStore_->removeElement( me_hs04_[ism-1]->getName() );
    sprintf(histo, "EELT laser shape L4A %s", Numbers::sEE(ism).c_str());
    me_hs04_[ism-1] = dqmStore_->book1D(histo, histo, 10, 0., 10.);
    me_hs04_[ism-1]->setAxisTitle("sample", 1);
    me_hs04_[ism-1]->setAxisTitle("amplitude", 2);
    if ( me_hs05_[ism-1] ) dqmStore_->removeElement( me_hs05_[ism-1]->getName() );
    sprintf(histo, "EELT laser shape L1B %s", Numbers::sEE(ism).c_str());
    me_hs05_[ism-1] = dqmStore_->book1D(histo, histo, 10, 0., 10.);
    me_hs05_[ism-1]->setAxisTitle("sample", 1);
    me_hs05_[ism-1]->setAxisTitle("amplitude", 2);
    if ( me_hs06_[ism-1] ) dqmStore_->removeElement( me_hs06_[ism-1]->getName() );
    sprintf(histo, "EELT laser shape L2B %s", Numbers::sEE(ism).c_str());
    me_hs06_[ism-1] = dqmStore_->book1D(histo, histo, 10, 0., 10.);
    me_hs06_[ism-1]->setAxisTitle("sample", 1);
    me_hs06_[ism-1]->setAxisTitle("amplitude", 2);
    if ( me_hs07_[ism-1] ) dqmStore_->removeElement( me_hs07_[ism-1]->getName() );
    sprintf(histo, "EELT laser shape L3B %s", Numbers::sEE(ism).c_str());
    me_hs07_[ism-1] = dqmStore_->book1D(histo, histo, 10, 0., 10.);
    me_hs07_[ism-1]->setAxisTitle("sample", 1);
    me_hs07_[ism-1]->setAxisTitle("amplitude", 2);
    if ( me_hs08_[ism-1] ) dqmStore_->removeElement( me_hs08_[ism-1]->getName() );
    sprintf(histo, "EELT laser shape L4B %s", Numbers::sEE(ism).c_str());
    me_hs08_[ism-1] = dqmStore_->book1D(histo, histo, 10, 0., 10.);
    me_hs08_[ism-1]->setAxisTitle("sample", 1);
    me_hs08_[ism-1]->setAxisTitle("amplitude", 2);

  }

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( meg01_[ism-1] ) meg01_[ism-1]->Reset();
    if ( meg02_[ism-1] ) meg02_[ism-1]->Reset();
    if ( meg03_[ism-1] ) meg03_[ism-1]->Reset();
    if ( meg04_[ism-1] ) meg04_[ism-1]->Reset();

    if ( meg05_[ism-1] ) meg05_[ism-1]->Reset();
    if ( meg06_[ism-1] ) meg06_[ism-1]->Reset();
    if ( meg07_[ism-1] ) meg07_[ism-1]->Reset();
    if ( meg08_[ism-1] ) meg08_[ism-1]->Reset();
    if ( meg09_[ism-1] ) meg09_[ism-1]->Reset();
    if ( meg10_[ism-1] ) meg10_[ism-1]->Reset();
    if ( meg11_[ism-1] ) meg11_[ism-1]->Reset();
    if ( meg12_[ism-1] ) meg12_[ism-1]->Reset();

    for ( int ix = 1; ix <= 50; ix++ ) {
      for ( int iy = 1; iy <= 50; iy++ ) {

        if ( meg01_[ism-1] ) meg01_[ism-1]->setBinContent( ix, iy, 6. );
        if ( meg02_[ism-1] ) meg02_[ism-1]->setBinContent( ix, iy, 6. );
        if ( meg03_[ism-1] ) meg03_[ism-1]->setBinContent( ix, iy, 6. );
        if ( meg04_[ism-1] ) meg04_[ism-1]->setBinContent( ix, iy, 6. );

        int jx = ix + Numbers::ix0EE(ism);
        int jy = iy + Numbers::iy0EE(ism);

        if ( ism >= 1 && ism <= 9 ) jx = 101 - jx;

        if ( Numbers::validEE(ism, jx, jy) ) {
          if ( meg01_[ism-1] ) meg01_[ism-1]->setBinContent( ix, iy, 2. );
          if ( meg02_[ism-1] ) meg02_[ism-1]->setBinContent( ix, iy, 2. );
          if ( meg03_[ism-1] ) meg03_[ism-1]->setBinContent( ix, iy, 2. );
          if ( meg04_[ism-1] ) meg04_[ism-1]->setBinContent( ix, iy, 2. );
        }

      }
    }

    for ( int i = 1; i <= 10; i++ ) {

        if ( meg05_[ism-1] ) meg05_[ism-1]->setBinContent( i, 1, 2. );
        if ( meg06_[ism-1] ) meg06_[ism-1]->setBinContent( i, 1, 2. );
        if ( meg07_[ism-1] ) meg07_[ism-1]->setBinContent( i, 1, 2. );
        if ( meg08_[ism-1] ) meg08_[ism-1]->setBinContent( i, 1, 2. );
        if ( meg09_[ism-1] ) meg09_[ism-1]->setBinContent( i, 1, 2. );
        if ( meg10_[ism-1] ) meg10_[ism-1]->setBinContent( i, 1, 2. );
        if ( meg11_[ism-1] ) meg11_[ism-1]->setBinContent( i, 1, 2. );
        if ( meg12_[ism-1] ) meg12_[ism-1]->setBinContent( i, 1, 2. );

    }

    if ( mea01_[ism-1] ) mea01_[ism-1]->Reset();
    if ( mea02_[ism-1] ) mea02_[ism-1]->Reset();
    if ( mea03_[ism-1] ) mea03_[ism-1]->Reset();
    if ( mea04_[ism-1] ) mea04_[ism-1]->Reset();
    if ( mea05_[ism-1] ) mea05_[ism-1]->Reset();
    if ( mea06_[ism-1] ) mea06_[ism-1]->Reset();
    if ( mea07_[ism-1] ) mea07_[ism-1]->Reset();
    if ( mea08_[ism-1] ) mea08_[ism-1]->Reset();

    if ( met01_[ism-1] ) met01_[ism-1]->Reset();
    if ( met02_[ism-1] ) met02_[ism-1]->Reset();
    if ( met03_[ism-1] ) met03_[ism-1]->Reset();
    if ( met04_[ism-1] ) met04_[ism-1]->Reset();
    if ( met05_[ism-1] ) met05_[ism-1]->Reset();
    if ( met06_[ism-1] ) met06_[ism-1]->Reset();
    if ( met07_[ism-1] ) met07_[ism-1]->Reset();
    if ( met08_[ism-1] ) met08_[ism-1]->Reset();

    if ( metav01_[ism-1] ) metav01_[ism-1]->Reset();
    if ( metav02_[ism-1] ) metav02_[ism-1]->Reset();
    if ( metav03_[ism-1] ) metav03_[ism-1]->Reset();
    if ( metav04_[ism-1] ) metav04_[ism-1]->Reset();
    if ( metav05_[ism-1] ) metav05_[ism-1]->Reset();
    if ( metav06_[ism-1] ) metav06_[ism-1]->Reset();
    if ( metav07_[ism-1] ) metav07_[ism-1]->Reset();
    if ( metav08_[ism-1] ) metav08_[ism-1]->Reset();

    if ( metrms01_[ism-1] ) metrms01_[ism-1]->Reset();
    if ( metrms02_[ism-1] ) metrms02_[ism-1]->Reset();
    if ( metrms03_[ism-1] ) metrms03_[ism-1]->Reset();
    if ( metrms04_[ism-1] ) metrms04_[ism-1]->Reset();
    if ( metrms05_[ism-1] ) metrms05_[ism-1]->Reset();
    if ( metrms06_[ism-1] ) metrms06_[ism-1]->Reset();
    if ( metrms07_[ism-1] ) metrms07_[ism-1]->Reset();
    if ( metrms08_[ism-1] ) metrms08_[ism-1]->Reset();

    if ( meaopn01_[ism-1] ) meaopn01_[ism-1]->Reset();
    if ( meaopn02_[ism-1] ) meaopn02_[ism-1]->Reset();
    if ( meaopn03_[ism-1] ) meaopn03_[ism-1]->Reset();
    if ( meaopn04_[ism-1] ) meaopn04_[ism-1]->Reset();
    if ( meaopn05_[ism-1] ) meaopn05_[ism-1]->Reset();
    if ( meaopn06_[ism-1] ) meaopn06_[ism-1]->Reset();
    if ( meaopn07_[ism-1] ) meaopn07_[ism-1]->Reset();
    if ( meaopn08_[ism-1] ) meaopn08_[ism-1]->Reset();

    if ( mepnprms01_[ism-1] ) mepnprms01_[ism-1]->Reset();
    if ( mepnprms02_[ism-1] ) mepnprms02_[ism-1]->Reset();
    if ( mepnprms03_[ism-1] ) mepnprms03_[ism-1]->Reset();
    if ( mepnprms04_[ism-1] ) mepnprms04_[ism-1]->Reset();
    if ( mepnprms05_[ism-1] ) mepnprms05_[ism-1]->Reset();
    if ( mepnprms06_[ism-1] ) mepnprms06_[ism-1]->Reset();
    if ( mepnprms07_[ism-1] ) mepnprms07_[ism-1]->Reset();
    if ( mepnprms08_[ism-1] ) mepnprms08_[ism-1]->Reset();

    if ( me_hs01_[ism-1] ) me_hs01_[ism-1]->Reset();
    if ( me_hs02_[ism-1] ) me_hs02_[ism-1]->Reset();
    if ( me_hs03_[ism-1] ) me_hs03_[ism-1]->Reset();
    if ( me_hs04_[ism-1] ) me_hs04_[ism-1]->Reset();
    if ( me_hs05_[ism-1] ) me_hs05_[ism-1]->Reset();
    if ( me_hs06_[ism-1] ) me_hs06_[ism-1]->Reset();
    if ( me_hs07_[ism-1] ) me_hs07_[ism-1]->Reset();
    if ( me_hs08_[ism-1] ) me_hs08_[ism-1]->Reset();

  }

}

void EELaserClient::cleanup(void) {

  if ( ! enableCleanup_ ) return;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( cloneME_ ) {
      if ( h01_[ism-1] ) delete h01_[ism-1];
      if ( h02_[ism-1] ) delete h02_[ism-1];
      if ( h03_[ism-1] ) delete h03_[ism-1];
      if ( h04_[ism-1] ) delete h04_[ism-1];
      if ( h05_[ism-1] ) delete h05_[ism-1];
      if ( h06_[ism-1] ) delete h06_[ism-1];
      if ( h07_[ism-1] ) delete h07_[ism-1];
      if ( h08_[ism-1] ) delete h08_[ism-1];

      if ( h09_[ism-1] ) delete h09_[ism-1];
      if ( h10_[ism-1] ) delete h10_[ism-1];
      if ( h11_[ism-1] ) delete h11_[ism-1];
      if ( h12_[ism-1] ) delete h12_[ism-1];

      if ( h13_[ism-1] ) delete h13_[ism-1];
      if ( h14_[ism-1] ) delete h14_[ism-1];
      if ( h15_[ism-1] ) delete h15_[ism-1];
      if ( h16_[ism-1] ) delete h16_[ism-1];
      if ( h17_[ism-1] ) delete h17_[ism-1];
      if ( h18_[ism-1] ) delete h18_[ism-1];
      if ( h19_[ism-1] ) delete h19_[ism-1];
      if ( h20_[ism-1] ) delete h20_[ism-1];

      if ( h21_[ism-1] ) delete h21_[ism-1];
      if ( h22_[ism-1] ) delete h22_[ism-1];
      if ( h23_[ism-1] ) delete h23_[ism-1];
      if ( h24_[ism-1] ) delete h24_[ism-1];

      if ( hs01_[ism-1] ) delete hs01_[ism-1];
      if ( hs02_[ism-1] ) delete hs02_[ism-1];
      if ( hs03_[ism-1] ) delete hs03_[ism-1];
      if ( hs04_[ism-1] ) delete hs04_[ism-1];

      if ( hs05_[ism-1] ) delete hs05_[ism-1];
      if ( hs06_[ism-1] ) delete hs06_[ism-1];
      if ( hs07_[ism-1] ) delete hs07_[ism-1];
      if ( hs08_[ism-1] ) delete hs08_[ism-1];

      if ( i01_[ism-1] ) delete i01_[ism-1];
      if ( i02_[ism-1] ) delete i02_[ism-1];
      if ( i03_[ism-1] ) delete i03_[ism-1];
      if ( i04_[ism-1] ) delete i04_[ism-1];
      if ( i05_[ism-1] ) delete i05_[ism-1];
      if ( i06_[ism-1] ) delete i06_[ism-1];
      if ( i07_[ism-1] ) delete i07_[ism-1];
      if ( i08_[ism-1] ) delete i08_[ism-1];

      if ( i09_[ism-1] ) delete i09_[ism-1];
      if ( i10_[ism-1] ) delete i10_[ism-1];
      if ( i11_[ism-1] ) delete i11_[ism-1];
      if ( i12_[ism-1] ) delete i12_[ism-1];
      if ( i13_[ism-1] ) delete i13_[ism-1];
      if ( i14_[ism-1] ) delete i14_[ism-1];
      if ( i15_[ism-1] ) delete i15_[ism-1];
      if ( i16_[ism-1] ) delete i16_[ism-1];
    }

    h01_[ism-1] = 0;
    h02_[ism-1] = 0;
    h03_[ism-1] = 0;
    h04_[ism-1] = 0;
    h05_[ism-1] = 0;
    h06_[ism-1] = 0;
    h07_[ism-1] = 0;
    h08_[ism-1] = 0;

    h09_[ism-1] = 0;
    h10_[ism-1] = 0;
    h11_[ism-1] = 0;
    h12_[ism-1] = 0;

    h13_[ism-1] = 0;
    h14_[ism-1] = 0;
    h15_[ism-1] = 0;
    h16_[ism-1] = 0;
    h17_[ism-1] = 0;
    h18_[ism-1] = 0;
    h19_[ism-1] = 0;
    h20_[ism-1] = 0;

    h21_[ism-1] = 0;
    h22_[ism-1] = 0;
    h23_[ism-1] = 0;
    h24_[ism-1] = 0;

    hs01_[ism-1] = 0;
    hs02_[ism-1] = 0;
    hs03_[ism-1] = 0;
    hs04_[ism-1] = 0;

    hs05_[ism-1] = 0;
    hs06_[ism-1] = 0;
    hs07_[ism-1] = 0;
    hs08_[ism-1] = 0;

    i01_[ism-1] = 0;
    i02_[ism-1] = 0;
    i03_[ism-1] = 0;
    i04_[ism-1] = 0;
    i05_[ism-1] = 0;
    i06_[ism-1] = 0;
    i07_[ism-1] = 0;
    i08_[ism-1] = 0;

    i09_[ism-1] = 0;
    i10_[ism-1] = 0;
    i11_[ism-1] = 0;
    i12_[ism-1] = 0;
    i13_[ism-1] = 0;
    i14_[ism-1] = 0;
    i15_[ism-1] = 0;
    i16_[ism-1] = 0;

  }

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    dqmStore_->setCurrentFolder( prefixME_ + "/EELaserClient" );

    if ( meg01_[ism-1] ) dqmStore_->removeElement( meg01_[ism-1]->getName() );
    meg01_[ism-1] = 0;
    if ( meg02_[ism-1] ) dqmStore_->removeElement( meg02_[ism-1]->getName() );
    meg02_[ism-1] = 0;
    if ( meg03_[ism-1] ) dqmStore_->removeElement( meg03_[ism-1]->getName() );
    meg03_[ism-1] = 0;
    if ( meg04_[ism-1] ) dqmStore_->removeElement( meg04_[ism-1]->getName() );
    meg04_[ism-1] = 0;

    if ( meg05_[ism-1] ) dqmStore_->removeElement( meg05_[ism-1]->getName() );
    meg05_[ism-1] = 0;
    if ( meg06_[ism-1] ) dqmStore_->removeElement( meg06_[ism-1]->getName() );
    meg06_[ism-1] = 0;
    if ( meg07_[ism-1] ) dqmStore_->removeElement( meg07_[ism-1]->getName() );
    meg07_[ism-1] = 0;
    if ( meg08_[ism-1] ) dqmStore_->removeElement( meg08_[ism-1]->getName() );
    meg08_[ism-1] = 0;
    if ( meg09_[ism-1] ) dqmStore_->removeElement( meg09_[ism-1]->getName() );
    meg09_[ism-1] = 0;
    if ( meg10_[ism-1] ) dqmStore_->removeElement( meg10_[ism-1]->getName() );
    meg10_[ism-1] = 0;
    if ( meg11_[ism-1] ) dqmStore_->removeElement( meg11_[ism-1]->getName() );
    meg11_[ism-1] = 0;
    if ( meg12_[ism-1] ) dqmStore_->removeElement( meg12_[ism-1]->getName() );
    meg12_[ism-1] = 0;

    if ( mea01_[ism-1] ) dqmStore_->removeElement( mea01_[ism-1]->getName() );
    mea01_[ism-1] = 0;
    if ( mea02_[ism-1] ) dqmStore_->removeElement( mea02_[ism-1]->getName() );
    mea02_[ism-1] = 0;
    if ( mea03_[ism-1] ) dqmStore_->removeElement( mea03_[ism-1]->getName() );
    mea03_[ism-1] = 0;
    if ( mea04_[ism-1] ) dqmStore_->removeElement( mea04_[ism-1]->getName() );
    mea04_[ism-1] = 0;
    if ( mea05_[ism-1] ) dqmStore_->removeElement( mea05_[ism-1]->getName() );
    mea05_[ism-1] = 0;
    if ( mea06_[ism-1] ) dqmStore_->removeElement( mea06_[ism-1]->getName() );
    mea06_[ism-1] = 0;
    if ( mea07_[ism-1] ) dqmStore_->removeElement( mea07_[ism-1]->getName() );
    mea07_[ism-1] = 0;
    if ( mea08_[ism-1] ) dqmStore_->removeElement( mea08_[ism-1]->getName() );
    mea08_[ism-1] = 0;

    if ( met01_[ism-1] ) dqmStore_->removeElement( met01_[ism-1]->getName() );
    met01_[ism-1] = 0;
    if ( met02_[ism-1] ) dqmStore_->removeElement( met02_[ism-1]->getName() );
    met02_[ism-1] = 0;
    if ( met03_[ism-1] ) dqmStore_->removeElement( met03_[ism-1]->getName() );
    met03_[ism-1] = 0;
    if ( met04_[ism-1] ) dqmStore_->removeElement( met04_[ism-1]->getName() );
    met04_[ism-1] = 0;
    if ( met05_[ism-1] ) dqmStore_->removeElement( met05_[ism-1]->getName() );
    met05_[ism-1] = 0;
    if ( met06_[ism-1] ) dqmStore_->removeElement( met06_[ism-1]->getName() );
    met06_[ism-1] = 0;
    if ( met07_[ism-1] ) dqmStore_->removeElement( met07_[ism-1]->getName() );
    met07_[ism-1] = 0;
    if ( met08_[ism-1] ) dqmStore_->removeElement( met08_[ism-1]->getName() );
    met08_[ism-1] = 0;

    if ( metav01_[ism-1] ) dqmStore_->removeElement( metav01_[ism-1]->getName() );
    metav01_[ism-1] = 0;
    if ( metav02_[ism-1] ) dqmStore_->removeElement( metav02_[ism-1]->getName() );
    metav02_[ism-1] = 0;
    if ( metav03_[ism-1] ) dqmStore_->removeElement( metav03_[ism-1]->getName() );
    metav03_[ism-1] = 0;
    if ( metav04_[ism-1] ) dqmStore_->removeElement( metav04_[ism-1]->getName() );
    metav04_[ism-1] = 0;
    if ( metav05_[ism-1] ) dqmStore_->removeElement( metav05_[ism-1]->getName() );
    metav05_[ism-1] = 0;
    if ( metav06_[ism-1] ) dqmStore_->removeElement( metav06_[ism-1]->getName() );
    metav06_[ism-1] = 0;
    if ( metav07_[ism-1] ) dqmStore_->removeElement( metav07_[ism-1]->getName() );
    metav07_[ism-1] = 0;
    if ( metav08_[ism-1] ) dqmStore_->removeElement( metav08_[ism-1]->getName() );
    metav08_[ism-1] = 0;

    if ( metrms01_[ism-1] ) dqmStore_->removeElement( metrms01_[ism-1]->getName() );
    metrms01_[ism-1] = 0;
    if ( metrms02_[ism-1] ) dqmStore_->removeElement( metrms02_[ism-1]->getName() );
    metrms02_[ism-1] = 0;
    if ( metrms03_[ism-1] ) dqmStore_->removeElement( metrms03_[ism-1]->getName() );
    metrms03_[ism-1] = 0;
    if ( metrms04_[ism-1] ) dqmStore_->removeElement( metrms04_[ism-1]->getName() );
    metrms04_[ism-1] = 0;
    if ( metrms05_[ism-1] ) dqmStore_->removeElement( metrms05_[ism-1]->getName() );
    metrms05_[ism-1] = 0;
    if ( metrms06_[ism-1] ) dqmStore_->removeElement( metrms06_[ism-1]->getName() );
    metrms06_[ism-1] = 0;
    if ( metrms07_[ism-1] ) dqmStore_->removeElement( metrms07_[ism-1]->getName() );
    metrms07_[ism-1] = 0;
    if ( metrms08_[ism-1] ) dqmStore_->removeElement( metrms08_[ism-1]->getName() );
    metrms08_[ism-1] = 0;

    if ( meaopn01_[ism-1] ) dqmStore_->removeElement( meaopn01_[ism-1]->getName() );
    meaopn01_[ism-1] = 0;
    if ( meaopn02_[ism-1] ) dqmStore_->removeElement( meaopn02_[ism-1]->getName() );
    meaopn02_[ism-1] = 0;
    if ( meaopn03_[ism-1] ) dqmStore_->removeElement( meaopn03_[ism-1]->getName() );
    meaopn03_[ism-1] = 0;
    if ( meaopn04_[ism-1] ) dqmStore_->removeElement( meaopn04_[ism-1]->getName() );
    meaopn04_[ism-1] = 0;
    if ( meaopn05_[ism-1] ) dqmStore_->removeElement( meaopn05_[ism-1]->getName() );
    meaopn05_[ism-1] = 0;
    if ( meaopn06_[ism-1] ) dqmStore_->removeElement( meaopn06_[ism-1]->getName() );
    meaopn06_[ism-1] = 0;
    if ( meaopn07_[ism-1] ) dqmStore_->removeElement( meaopn07_[ism-1]->getName() );
    meaopn07_[ism-1] = 0;
    if ( meaopn08_[ism-1] ) dqmStore_->removeElement( meaopn08_[ism-1]->getName() );
    meaopn08_[ism-1] = 0;

    if ( mepnprms01_[ism-1] ) dqmStore_->removeElement( mepnprms01_[ism-1]->getName() );
    mepnprms01_[ism-1] = 0;
    if ( mepnprms02_[ism-1] ) dqmStore_->removeElement( mepnprms02_[ism-1]->getName() );
    mepnprms02_[ism-1] = 0;
    if ( mepnprms03_[ism-1] ) dqmStore_->removeElement( mepnprms03_[ism-1]->getName() );
    mepnprms03_[ism-1] = 0;
    if ( mepnprms04_[ism-1] ) dqmStore_->removeElement( mepnprms04_[ism-1]->getName() );
    mepnprms04_[ism-1] = 0;
    if ( mepnprms05_[ism-1] ) dqmStore_->removeElement( mepnprms05_[ism-1]->getName() );
    mepnprms05_[ism-1] = 0;
    if ( mepnprms06_[ism-1] ) dqmStore_->removeElement( mepnprms06_[ism-1]->getName() );
    mepnprms06_[ism-1] = 0;
    if ( mepnprms07_[ism-1] ) dqmStore_->removeElement( mepnprms07_[ism-1]->getName() );
    mepnprms07_[ism-1] = 0;
    if ( mepnprms08_[ism-1] ) dqmStore_->removeElement( mepnprms08_[ism-1]->getName() );
    mepnprms08_[ism-1] = 0;

    if ( me_hs01_[ism-1] ) dqmStore_->removeElement( me_hs01_[ism-1]->getName() );
    me_hs01_[ism-1] = 0;
    if ( me_hs02_[ism-1] ) dqmStore_->removeElement( me_hs02_[ism-1]->getName() );
    me_hs02_[ism-1] = 0;
    if ( me_hs03_[ism-1] ) dqmStore_->removeElement( me_hs03_[ism-1]->getName() );
    me_hs03_[ism-1] = 0;
    if ( me_hs04_[ism-1] ) dqmStore_->removeElement( me_hs04_[ism-1]->getName() );
    me_hs04_[ism-1] = 0;
    if ( me_hs05_[ism-1] ) dqmStore_->removeElement( me_hs05_[ism-1]->getName() );
    me_hs05_[ism-1] = 0;
    if ( me_hs06_[ism-1] ) dqmStore_->removeElement( me_hs06_[ism-1]->getName() );
    me_hs06_[ism-1] = 0;
    if ( me_hs07_[ism-1] ) dqmStore_->removeElement( me_hs07_[ism-1]->getName() );
    me_hs07_[ism-1] = 0;
    if ( me_hs08_[ism-1] ) dqmStore_->removeElement( me_hs08_[ism-1]->getName() );
    me_hs08_[ism-1] = 0;

  }

}

bool EELaserClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov, bool& status, bool flag) {

  status = true;

  if ( ! flag ) return false;

  EcalLogicID ecid;

  MonLaserBlueDat apd_bl;
  map<EcalLogicID, MonLaserBlueDat> dataset1_bl;
  MonLaserGreenDat apd_gr;
  map<EcalLogicID, MonLaserGreenDat> dataset1_gr;
  MonLaserIRedDat apd_ir;
  map<EcalLogicID, MonLaserIRedDat> dataset1_ir;
  MonLaserRedDat apd_rd;
  map<EcalLogicID, MonLaserRedDat> dataset1_rd;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( verbose_ ) {
      cout << " " << Numbers::sEE(ism) << " (ism=" << ism << ")" << endl;
      cout << endl;
      UtilsClient::printBadChannels(meg01_[ism-1], h01_[ism-1]);
      UtilsClient::printBadChannels(meg01_[ism-1], h13_[ism-1]);
      UtilsClient::printBadChannels(meg02_[ism-1], h03_[ism-1]);
      UtilsClient::printBadChannels(meg02_[ism-1], h15_[ism-1]);
      UtilsClient::printBadChannels(meg03_[ism-1], h05_[ism-1]);
      UtilsClient::printBadChannels(meg03_[ism-1], h17_[ism-1]);
      UtilsClient::printBadChannels(meg04_[ism-1], h07_[ism-1]);
      UtilsClient::printBadChannels(meg04_[ism-1], h19_[ism-1]);
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
        bool update05;
        bool update06;
        bool update07;
        bool update08;

        bool update09;
        bool update10;
        bool update11;
        bool update12;
        bool update13;
        bool update14;
        bool update15;
        bool update16;

        float num01, num02, num03, num04, num05, num06, num07, num08;
        float mean01, mean02, mean03, mean04, mean05, mean06, mean07, mean08;
        float rms01, rms02, rms03, rms04, rms05, rms06, rms07, rms08;

        float num09, num10, num11, num12, num13, num14, num15, num16;
        float mean09, mean10, mean11, mean12, mean13, mean14, mean15, mean16;
        float rms09, rms10, rms11, rms12, rms13, rms14, rms15, rms16;

        update01 = UtilsClient::getBinStatistics(h01_[ism-1], ix, iy, num01, mean01, rms01);
        update02 = UtilsClient::getBinStatistics(h02_[ism-1], ix, iy, num02, mean02, rms02);
        update03 = UtilsClient::getBinStatistics(h03_[ism-1], ix, iy, num03, mean03, rms03);
        update04 = UtilsClient::getBinStatistics(h04_[ism-1], ix, iy, num04, mean04, rms04);
        update05 = UtilsClient::getBinStatistics(h05_[ism-1], ix, iy, num05, mean05, rms05);
        update06 = UtilsClient::getBinStatistics(h06_[ism-1], ix, iy, num06, mean06, rms06);
        update07 = UtilsClient::getBinStatistics(h07_[ism-1], ix, iy, num07, mean07, rms07);
        update08 = UtilsClient::getBinStatistics(h08_[ism-1], ix, iy, num08, mean08, rms08);

        update09 = UtilsClient::getBinStatistics(h13_[ism-1], ix, iy, num09, mean09, rms09);
        update10 = UtilsClient::getBinStatistics(h14_[ism-1], ix, iy, num10, mean10, rms10);
        update11 = UtilsClient::getBinStatistics(h15_[ism-1], ix, iy, num11, mean11, rms11);
        update12 = UtilsClient::getBinStatistics(h16_[ism-1], ix, iy, num12, mean12, rms12);
        update13 = UtilsClient::getBinStatistics(h17_[ism-1], ix, iy, num13, mean13, rms13);
        update14 = UtilsClient::getBinStatistics(h18_[ism-1], ix, iy, num14, mean14, rms14);
        update15 = UtilsClient::getBinStatistics(h19_[ism-1], ix, iy, num15, mean15, rms15);
        update16 = UtilsClient::getBinStatistics(h20_[ism-1], ix, iy, num16, mean16, rms16);

        if ( update01 || update02 ) {

          if ( Numbers::icEE(ism, jx, jy) == 1 ) {

            if ( verbose_ ) {
              cout << "Preparing dataset for " << Numbers::sEE(ism) << " (ism=" << ism << ")" << endl;
              cout << "L1A (" << Numbers::ix0EE(i+1)+ix << "," << Numbers::iy0EE(i+1)+iy << ") " << num01 << " " << mean01 << " " << rms01 << endl;
              cout << endl;
            }

          }

          apd_bl.setAPDMean(mean01);
          apd_bl.setAPDRMS(rms01);

          apd_bl.setAPDOverPNMean(mean02);
          apd_bl.setAPDOverPNRMS(rms02);

          if ( UtilsClient::getBinStatus(meg01_[ism-1], ix, iy) ) {
            apd_bl.setTaskStatus(true);
          } else {
            apd_bl.setTaskStatus(false);
          }

          status = status && UtilsClient::getBinQuality(meg01_[ism-1], ix, iy);

          int ic = Numbers::indexEE(ism, jx, jy);

          if ( ic == -1 ) continue;

          if ( econn ) {
            ecid = LogicID::getEcalLogicID("EE_crystal_number", Numbers::iSM(ism, EcalEndcap), ic);
            dataset1_bl[ecid] = apd_bl;
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

          apd_bl.setAPDMean(mean09);
          apd_bl.setAPDRMS(rms09);

          apd_bl.setAPDOverPNMean(mean10);
          apd_bl.setAPDOverPNRMS(rms10);

          if ( UtilsClient::getBinStatus(meg01_[ism-1], ix, iy) ) {
            apd_bl.setTaskStatus(true);
          } else {
            apd_bl.setTaskStatus(false);
          }

          status = status && UtilsClient::getBinQuality(meg01_[ism-1], ix, iy);

          int ic = Numbers::indexEE(ism, jx, jy);

          if ( ic == -1 ) continue;

          if ( econn ) {
            ecid = LogicID::getEcalLogicID("EE_crystal_number", Numbers::iSM(ism, EcalEndcap), ic);
            dataset1_bl[ecid] = apd_bl;
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

          apd_ir.setAPDMean(mean03);
          apd_ir.setAPDRMS(rms03);

          apd_ir.setAPDOverPNMean(mean04);
          apd_ir.setAPDOverPNRMS(rms04);

          if ( UtilsClient::getBinStatus(meg02_[ism-1], ix, iy) ) {
            apd_ir.setTaskStatus(true);
          } else {
            apd_ir.setTaskStatus(false);
          }

          status = status && UtilsClient::getBinQuality(meg02_[ism-1], ix, iy);

          int ic = Numbers::indexEE(ism, jx, jy);

          if ( ic == -1 ) continue;

          if ( econn ) {
            ecid = LogicID::getEcalLogicID("EE_crystal_number", Numbers::iSM(ism, EcalEndcap), ic);
            dataset1_ir[ecid] = apd_ir;
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

          apd_ir.setAPDMean(mean11);
          apd_ir.setAPDRMS(rms11);

          apd_ir.setAPDOverPNMean(mean12);
          apd_ir.setAPDOverPNRMS(rms12);

          if ( UtilsClient::getBinStatus(meg02_[ism-1], ix, iy) ) {
            apd_ir.setTaskStatus(true);
          } else {
            apd_ir.setTaskStatus(false);
          }

          status = status && UtilsClient::getBinQuality(meg02_[ism-1], ix, iy);

          int ic = Numbers::indexEE(ism, jx, jy);

          if ( ic == -1 ) continue;

          if ( econn ) {
            ecid = LogicID::getEcalLogicID("EE_crystal_number", Numbers::iSM(ism, EcalEndcap), ic);
            dataset1_ir[ecid] = apd_ir;
          }

        }

        if ( update05 || update06 ) {

          if ( Numbers::icEE(ism, jx, jy) == 1 ) {

            if ( verbose_ ) {
              cout << "Preparing dataset for " << Numbers::sEE(ism) << " (ism=" << ism << ")" << endl;
              cout << "L3A (" << Numbers::ix0EE(i+1)+ix << "," << Numbers::iy0EE(i+1)+iy << ") " << num05 << " " << mean05 << " " << rms05 << endl;
              cout << endl;
            }

          }

          apd_gr.setAPDMean(mean05);
          apd_gr.setAPDRMS(rms05);

          apd_gr.setAPDOverPNMean(mean06);
          apd_gr.setAPDOverPNRMS(rms06);

          if ( UtilsClient::getBinStatus(meg03_[ism-1], ix, iy) ) {
            apd_gr.setTaskStatus(true);
          } else {
            apd_gr.setTaskStatus(false);
          }

          status = status && UtilsClient::getBinQuality(meg03_[ism-1], ix, iy);

          int ic = Numbers::indexEE(ism, jx, jy);

          if ( ic == -1 ) continue;

          if ( econn ) {
            ecid = LogicID::getEcalLogicID("EE_crystal_number", Numbers::iSM(ism, EcalEndcap), ic);
            dataset1_gr[ecid] = apd_gr;
          }

        }

        if ( update13 || update14 ) {

          if ( Numbers::icEE(ism, jx, jy) == 1 ) {

            if ( verbose_ ) {
              cout << "Preparing dataset for " << Numbers::sEE(ism) << " (ism=" << ism << ")" << endl;
              cout << "L3B (" << Numbers::ix0EE(i+1)+ix << "," << Numbers::iy0EE(i+1)+iy << ") " << num13 << " " << mean13 << " " << rms13 << endl;
              cout << endl;
            }

          }

          apd_gr.setAPDMean(mean13);
          apd_gr.setAPDRMS(rms13);

          apd_gr.setAPDOverPNMean(mean14);
          apd_gr.setAPDOverPNRMS(rms14);

          if ( UtilsClient::getBinStatus(meg03_[ism-1], ix, iy) ) {
            apd_gr.setTaskStatus(true);
          } else {
            apd_gr.setTaskStatus(false);
          }

          status = status && UtilsClient::getBinQuality(meg03_[ism-1], ix, iy);

          int ic = Numbers::indexEE(ism, jx, jy);

          if ( ic == -1 ) continue;

          if ( econn ) {
            ecid = LogicID::getEcalLogicID("EE_crystal_number", Numbers::iSM(ism, EcalEndcap), ic);
            dataset1_gr[ecid] = apd_gr;
          }

        }

        if ( update07 || update08 ) {

          if ( Numbers::icEE(ism, jx, jy) == 1 ) {

            if ( verbose_ ) {
              cout << "Preparing dataset for " << Numbers::sEE(ism) << " (ism=" << ism << ")" << endl;
              cout << "L4A (" << Numbers::ix0EE(i+1)+ix << "," << Numbers::iy0EE(i+1)+iy << ") " << num07 << " " << mean07 << " " << rms07 << endl;
              cout << endl;
            }

          }

          apd_rd.setAPDMean(mean07);
          apd_rd.setAPDRMS(rms07);

          apd_rd.setAPDOverPNMean(mean08);
          apd_rd.setAPDOverPNRMS(rms08);

          if ( UtilsClient::getBinStatus(meg04_[ism-1], ix, iy) ) {
            apd_rd.setTaskStatus(true);
          } else {
            apd_rd.setTaskStatus(false);
          }

          status = status && UtilsClient::getBinQuality(meg04_[ism-1], ix, iy);

          int ic = Numbers::indexEE(ism, jx, jy);

          if ( ic == -1 ) continue;

          if ( econn ) {
            ecid = LogicID::getEcalLogicID("EE_crystal_number", Numbers::iSM(ism, EcalEndcap), ic);
            dataset1_rd[ecid] = apd_rd;
          }

        }

        if ( update15 || update16 ) {

          if ( Numbers::icEE(ism, jx, jy) == 1 ) {

            if ( verbose_ ) {
              cout << "Preparing dataset for " << Numbers::sEE(ism) << " (ism=" << ism << ")" << endl;
              cout << "L4B (" << Numbers::ix0EE(i+1)+ix << "," << Numbers::iy0EE(i+1)+iy << ") " << num15 << " " << mean15 << " " << rms15 << endl;
              cout << endl;
            }

          }

          apd_rd.setAPDMean(mean15);
          apd_rd.setAPDRMS(rms15);

          apd_rd.setAPDOverPNMean(mean16);
          apd_rd.setAPDOverPNRMS(rms16);

          if ( UtilsClient::getBinStatus(meg04_[ism-1], ix, iy) ) {
            apd_rd.setTaskStatus(true);
          } else {
            apd_rd.setTaskStatus(false);
          }

          status = status && UtilsClient::getBinQuality(meg04_[ism-1], ix, iy);

          int ic = Numbers::indexEE(ism, jx, jy);

          if ( ic == -1 ) continue;

          if ( econn ) {
            ecid = LogicID::getEcalLogicID("EE_crystal_number", Numbers::iSM(ism, EcalEndcap), ic);
            dataset1_rd[ecid] = apd_rd;
          }

        }

      }
    }

  }

  if ( econn ) {
    try {
      if ( verbose_ ) cout << "Inserting MonLaserDat ..." << endl;
      if ( dataset1_bl.size() != 0 ) econn->insertDataArraySet(&dataset1_bl, moniov);
      if ( dataset1_ir.size() != 0 ) econn->insertDataArraySet(&dataset1_ir, moniov);
      if ( dataset1_gr.size() != 0 ) econn->insertDataArraySet(&dataset1_gr, moniov);
      if ( dataset1_rd.size() != 0 ) econn->insertDataArraySet(&dataset1_rd, moniov);
      if ( verbose_ ) cout << "done." << endl;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

  if ( verbose_ ) cout << endl;

  MonPNBlueDat pn_bl;
  map<EcalLogicID, MonPNBlueDat> dataset2_bl;
  MonPNGreenDat pn_gr;
  map<EcalLogicID, MonPNGreenDat> dataset2_gr;
  MonPNIRedDat pn_ir;
  map<EcalLogicID, MonPNIRedDat> dataset2_ir;
  MonPNRedDat pn_rd;
  map<EcalLogicID, MonPNRedDat> dataset2_rd;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( verbose_ ) {
      cout << " " << Numbers::sEE(ism) << " (ism=" << ism << ")" << endl;
      cout << endl;
      UtilsClient::printBadChannels(meg05_[ism-1], i01_[ism-1]);
      UtilsClient::printBadChannels(meg05_[ism-1], i05_[ism-1]);
      UtilsClient::printBadChannels(meg06_[ism-1], i02_[ism-1]);
      UtilsClient::printBadChannels(meg06_[ism-1], i06_[ism-1]);
      UtilsClient::printBadChannels(meg07_[ism-1], i03_[ism-1]);
      UtilsClient::printBadChannels(meg07_[ism-1], i07_[ism-1]);
      UtilsClient::printBadChannels(meg08_[ism-1], i04_[ism-1]);
      UtilsClient::printBadChannels(meg08_[ism-1], i08_[ism-1]);
      UtilsClient::printBadChannels(meg09_[ism-1], i09_[ism-1]);
      UtilsClient::printBadChannels(meg09_[ism-1], i13_[ism-1]);
      UtilsClient::printBadChannels(meg10_[ism-1], i10_[ism-1]);
      UtilsClient::printBadChannels(meg10_[ism-1], i14_[ism-1]);
      UtilsClient::printBadChannels(meg11_[ism-1], i11_[ism-1]);
      UtilsClient::printBadChannels(meg11_[ism-1], i15_[ism-1]);
      UtilsClient::printBadChannels(meg12_[ism-1], i12_[ism-1]);
      UtilsClient::printBadChannels(meg12_[ism-1], i16_[ism-1]);
    }

    for ( int i = 1; i <= 10; i++ ) {

      bool update01;
      bool update02;
      bool update03;
      bool update04;
      bool update05;
      bool update06;
      bool update07;
      bool update08;
      bool update09;
      bool update10;
      bool update11;
      bool update12;
      bool update13;
      bool update14;
      bool update15;
      bool update16;

      float num01, num02, num03, num04, num05, num06, num07, num08;
      float num09, num10, num11, num12, num13, num14, num15, num16;
      float mean01, mean02, mean03, mean04, mean05, mean06, mean07, mean08;
      float mean09, mean10, mean11, mean12, mean13, mean14, mean15, mean16;
      float rms01, rms02, rms03, rms04, rms05, rms06, rms07, rms08;
      float rms09, rms10, rms11, rms12, rms13, rms14, rms15, rms16;

      update01 = UtilsClient::getBinStatistics(i01_[ism-1], i, 0, num01, mean01, rms01);
      update02 = UtilsClient::getBinStatistics(i02_[ism-1], i, 0, num02, mean02, rms02);
      update03 = UtilsClient::getBinStatistics(i03_[ism-1], i, 0, num03, mean03, rms03);
      update04 = UtilsClient::getBinStatistics(i04_[ism-1], i, 0, num04, mean04, rms04);
      update05 = UtilsClient::getBinStatistics(i05_[ism-1], i, 0, num05, mean05, rms05);
      update06 = UtilsClient::getBinStatistics(i06_[ism-1], i, 0, num06, mean06, rms06);
      update07 = UtilsClient::getBinStatistics(i07_[ism-1], i, 0, num07, mean07, rms07);
      update08 = UtilsClient::getBinStatistics(i08_[ism-1], i, 0, num08, mean08, rms08);
      update09 = UtilsClient::getBinStatistics(i09_[ism-1], i, 0, num09, mean09, rms09);
      update10 = UtilsClient::getBinStatistics(i10_[ism-1], i, 0, num10, mean10, rms10);
      update11 = UtilsClient::getBinStatistics(i11_[ism-1], i, 0, num11, mean11, rms11);
      update12 = UtilsClient::getBinStatistics(i12_[ism-1], i, 0, num12, mean12, rms12);
      update13 = UtilsClient::getBinStatistics(i13_[ism-1], i, 0, num13, mean13, rms13);
      update14 = UtilsClient::getBinStatistics(i14_[ism-1], i, 0, num14, mean14, rms14);
      update15 = UtilsClient::getBinStatistics(i15_[ism-1], i, 0, num15, mean15, rms15);
      update16 = UtilsClient::getBinStatistics(i16_[ism-1], i, 0, num16, mean16, rms16);

      if ( update01 || update05 || update09 || update13 ) {

        if ( i == 1 ) {

          if ( verbose_ ) {
            cout << "Preparing dataset for " << Numbers::sEE(ism) << " (ism=" << ism << ")" << endl;
            cout << "PNs (" << i << ") L1 G01 " << num01  << " " << mean01 << " " << rms01  << endl;
            cout << "PNs (" << i << ") L1 G16 " << num09  << " " << mean09 << " " << rms09  << endl;
            cout << endl;
          }

        }

        pn_bl.setADCMeanG1(mean01);
        pn_bl.setADCRMSG1(rms01);

        pn_bl.setPedMeanG1(mean05);
        pn_bl.setPedRMSG1(rms05);

        pn_bl.setADCMeanG16(mean09);
        pn_bl.setADCRMSG16(rms09);

        pn_bl.setPedMeanG16(mean13);
        pn_bl.setPedRMSG16(rms13);

        if ( UtilsClient::getBinStatus(meg05_[ism-1], i, 1) ||
             UtilsClient::getBinStatus(meg09_[ism-1], i, 1) ) {
          pn_bl.setTaskStatus(true);
        } else {
          pn_bl.setTaskStatus(false);
        }

        status = status && ( UtilsClient::getBinQuality(meg05_[ism-1], i, 1) ||
                             UtilsClient::getBinQuality(meg09_[ism-1], i, 1) );

        if ( econn ) {
          ecid = LogicID::getEcalLogicID("EE_LM_PN", Numbers::iSM(ism, EcalEndcap), i-1);
          dataset2_bl[ecid] = pn_bl;
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

        pn_ir.setADCMeanG1(mean02);
        pn_ir.setADCRMSG1(rms02);

        pn_ir.setPedMeanG1(mean06);
        pn_ir.setPedRMSG1(rms06);

        pn_ir.setADCMeanG16(mean10);
        pn_ir.setADCRMSG16(rms10);

        pn_ir.setPedMeanG16(mean14);
        pn_ir.setPedRMSG16(rms14);

        if ( UtilsClient::getBinStatus(meg06_[ism-1], i, 1) ||
             UtilsClient::getBinStatus(meg10_[ism-1], i, 1) ) {
          pn_ir.setTaskStatus(true);
        } else {
          pn_ir.setTaskStatus(false);
        }

        status = status && ( UtilsClient::getBinQuality(meg06_[ism-1], i, 1) ||
                             UtilsClient::getBinQuality(meg10_[ism-1], i, 1) );

        if ( econn ) {
          ecid = LogicID::getEcalLogicID("EE_LM_PN", Numbers::iSM(ism, EcalEndcap), i-1);
          dataset2_ir[ecid] = pn_ir;
        }

      }

      if ( update03 || update07 || update11 || update15 ) {

        if ( i == 1 ) {

          if ( verbose_ ) {
            cout << "Preparing dataset for " << Numbers::sEE(ism) << " (ism=" << ism << ")" << endl;
            cout << "PNs (" << i << ") L3 G01 " << num03  << " " << mean03 << " " << rms03  << endl;
            cout << "PNs (" << i << ") L3 G16 " << num11  << " " << mean11 << " " << rms11  << endl;
            cout << endl;
          }

        }

        pn_gr.setADCMeanG1(mean03);
        pn_gr.setADCRMSG1(rms03);

        pn_gr.setPedMeanG1(mean07);
        pn_gr.setPedRMSG1(rms07);

        pn_gr.setADCMeanG16(mean11);
        pn_gr.setADCRMSG16(rms11);

        pn_gr.setPedMeanG16(mean15);
        pn_gr.setPedRMSG16(rms15);

        if ( UtilsClient::getBinStatus(meg07_[ism-1], i, 1) ||
             UtilsClient::getBinStatus(meg11_[ism-1], i, 1) ) {
          pn_gr.setTaskStatus(true);
        } else {
          pn_gr.setTaskStatus(false);
        }

        status = status && ( UtilsClient::getBinQuality(meg07_[ism-1], i, 1) ||
                             UtilsClient::getBinQuality(meg11_[ism-1], i, 1) );

        if ( econn ) {
          ecid = LogicID::getEcalLogicID("EE_LM_PN", Numbers::iSM(ism, EcalEndcap), i-1);
          dataset2_gr[ecid] = pn_gr;
        }

      }

      if ( update04 || update08 || update12 || update16 ) {

        if ( i == 1 ) {

          if ( verbose_ ) {
            cout << "Preparing dataset for " << Numbers::sEE(ism) << " (ism=" << ism << ")" << endl;
            cout << "PNs (" << i << ") L4 G01 " << num04  << " " << mean04 << " " << rms04  << endl;
            cout << "PNs (" << i << ") L4 G16 " << num12  << " " << mean12 << " " << rms12  << endl;
            cout << endl;
          }

        }

        pn_rd.setADCMeanG1(mean04);
        pn_rd.setADCRMSG1(rms04);

        pn_rd.setPedMeanG1(mean08);
        pn_rd.setPedRMSG1(mean08);

        pn_rd.setADCMeanG16(mean12);
        pn_rd.setADCRMSG16(rms12);

        pn_rd.setPedMeanG16(mean16);
        pn_rd.setPedRMSG16(rms16);

        if ( UtilsClient::getBinStatus(meg08_[ism-1], i, 1) ||
             UtilsClient::getBinStatus(meg12_[ism-1], i, 1) ) {
          pn_rd.setTaskStatus(true);
        } else {
          pn_rd.setTaskStatus(false);
        }

        status = status && ( UtilsClient::getBinQuality(meg08_[ism-1], i, 1) ||
                             UtilsClient::getBinQuality(meg12_[ism-1], i, 1) );

        if ( econn ) {
          ecid = LogicID::getEcalLogicID("EE_LM_PN", Numbers::iSM(ism, EcalEndcap), i-1);
          dataset2_rd[ecid] = pn_rd;
        }

      }

    }

  }

  if ( econn ) {
    try {
      if ( verbose_ ) cout << "Inserting MonPnDat ..." << endl;
      if ( dataset2_bl.size() != 0 ) econn->insertDataArraySet(&dataset2_bl, moniov);
      if ( dataset2_ir.size() != 0 ) econn->insertDataArraySet(&dataset2_ir, moniov);
      if ( dataset2_gr.size() != 0 ) econn->insertDataArraySet(&dataset2_gr, moniov);
      if ( dataset2_rd.size() != 0 ) econn->insertDataArraySet(&dataset2_rd, moniov);
      if ( verbose_ ) cout << "done." << endl;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

  if ( verbose_ ) cout << endl;

  MonTimingLaserBlueCrystalDat t_bl;
  map<EcalLogicID, MonTimingLaserBlueCrystalDat> dataset3_bl;
  MonTimingLaserGreenCrystalDat t_gr;
  map<EcalLogicID, MonTimingLaserGreenCrystalDat> dataset3_gr;
  MonTimingLaserIRedCrystalDat t_ir;
  map<EcalLogicID, MonTimingLaserIRedCrystalDat> dataset3_ir;
  MonTimingLaserRedCrystalDat t_rd;
  map<EcalLogicID, MonTimingLaserRedCrystalDat> dataset3_rd;

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
        bool update03;
        bool update04;
        bool update05;
        bool update06;
        bool update07;
        bool update08;

        float num01, num02, num03, num04, num05, num06, num07, num08;
        float mean01, mean02, mean03, mean04, mean05, mean06, mean07, mean08;
        float rms01, rms02, rms03, rms04, rms05, rms06, rms07, rms08;

        update01 = UtilsClient::getBinStatistics(h09_[ism-1], ix, iy, num01, mean01, rms01);
        update02 = UtilsClient::getBinStatistics(h10_[ism-1], ix, iy, num02, mean02, rms02);
        update03 = UtilsClient::getBinStatistics(h11_[ism-1], ix, iy, num03, mean03, rms03);
        update04 = UtilsClient::getBinStatistics(h12_[ism-1], ix, iy, num04, mean04, rms04);
        update05 = UtilsClient::getBinStatistics(h21_[ism-1], ix, iy, num05, mean05, rms05);
        update06 = UtilsClient::getBinStatistics(h22_[ism-1], ix, iy, num06, mean06, rms06);
        update07 = UtilsClient::getBinStatistics(h23_[ism-1], ix, iy, num07, mean07, rms07);
        update08 = UtilsClient::getBinStatistics(h24_[ism-1], ix, iy, num08, mean08, rms08);

        if ( update01 ) {

          if ( Numbers::icEE(ism, ix, iy) == 1 ) {

            if ( verbose_ ) {
              cout << "Preparing dataset for " << Numbers::sEE(ism) << " (ism=" << ism << ")" << endl;
              cout << "L1A crystal (" << Numbers::ix0EE(i+1)+ix << "," << Numbers::iy0EE(i+1)+iy << ") " << num01  << " " << mean01 << " " << rms01  << endl;
              cout << endl;
            }

          }

          t_bl.setTimingMean(mean01);
          t_bl.setTimingRMS(rms01);

          if ( UtilsClient::getBinStatus(meg01_[ism-1], ix, iy) ) {
            t_bl.setTaskStatus(true);
          } else {
            t_bl.setTaskStatus(false);
          }

          status = status && UtilsClient::getBinQuality(meg01_[ism-1], ix, iy);

          int ic = Numbers::indexEE(ism, ix, iy);

          if ( ic == -1 ) continue;

          if ( econn ) {
            ecid = LogicID::getEcalLogicID("EE_crystal_number", Numbers::iSM(ism, EcalEndcap), ic);
            dataset3_bl[ecid] = t_bl;
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

          t_bl.setTimingMean(mean05);
          t_bl.setTimingRMS(rms05);

          if ( UtilsClient::getBinStatus(meg01_[ism-1], ix, iy) ) {
            t_bl.setTaskStatus(true);
          } else {
            t_bl.setTaskStatus(false);
          }

          status = status && UtilsClient::getBinQuality(meg01_[ism-1], ix, iy);

          int ic = Numbers::indexEE(ism, ix, iy);

          if ( ic == -1 ) continue;

          if ( econn ) {
            ecid = LogicID::getEcalLogicID("EE_crystal_number", Numbers::iSM(ism, EcalEndcap), ic);
            dataset3_bl[ecid] = t_bl;
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

          t_gr.setTimingMean(mean02);
          t_gr.setTimingRMS(rms02);

          if ( UtilsClient::getBinStatus(meg02_[ism-1], ix, iy) ) {
            t_gr.setTaskStatus(true);
          } else {
            t_gr.setTaskStatus(false);
          }

          status = status && UtilsClient::getBinQuality(meg02_[ism-1], ix, iy);

          int ic = Numbers::indexEE(ism, ix, iy);

          if ( ic == -1 ) continue;

          if ( econn ) {
            ecid = LogicID::getEcalLogicID("EE_crystal_number", Numbers::iSM(ism, EcalEndcap), ic);
            dataset3_gr[ecid] = t_gr;
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

          t_gr.setTimingMean(mean06);
          t_gr.setTimingRMS(rms06);

          if ( UtilsClient::getBinStatus(meg02_[ism-1], ix, iy) ) {
            t_gr.setTaskStatus(true);
          } else {
            t_gr.setTaskStatus(false);
          }

          status = status && UtilsClient::getBinQuality(meg02_[ism-1], ix, iy);

          int ic = Numbers::indexEE(ism, ix, iy);

          if ( ic == -1 ) continue;

          if ( econn ) {
            ecid = LogicID::getEcalLogicID("EE_crystal_number", Numbers::iSM(ism, EcalEndcap), ic);
            dataset3_gr[ecid] = t_gr;
          }

        }

        if ( update03 ) {

          if ( Numbers::icEE(ism, ix, iy) == 1 ) {

            if ( verbose_ ) {
              cout << "Preparing dataset for " << Numbers::sEE(ism) << " (ism=" << ism << ")" << endl;
              cout << "L3A crystal (" << Numbers::ix0EE(i+1)+ix << "," << Numbers::iy0EE(i+1)+iy << ") " << num03  << " " << mean03 << " " << rms03  << endl;
              cout << endl;
            }

          }

          t_ir.setTimingMean(mean03);
          t_ir.setTimingRMS(rms03);

          if ( UtilsClient::getBinStatus(meg03_[ism-1], ix, iy) ) {
            t_ir.setTaskStatus(true);
          } else {
            t_ir.setTaskStatus(false);
          }

          status = status && UtilsClient::getBinQuality(meg03_[ism-1], ix, iy);

          int ic = Numbers::indexEE(ism, ix, iy);

          if ( ic == -1 ) continue;

          if ( econn ) {
            ecid = LogicID::getEcalLogicID("EE_crystal_number", Numbers::iSM(ism, EcalEndcap), ic);
            dataset3_ir[ecid] = t_ir;
          }

        }

        if ( update07 ) {

          if ( Numbers::icEE(ism, ix, iy) == 1 ) {

            if ( verbose_ ) {
              cout << "Preparing dataset for " << Numbers::sEE(ism) << " (ism=" << ism << ")" << endl;
              cout << "L3B crystal (" << Numbers::ix0EE(i+1)+ix << "," << Numbers::iy0EE(i+1)+iy << ") " << num07  << " " << mean07 << " " << rms07  << endl;
              cout << endl;
            }

          }

          t_ir.setTimingMean(mean07);
          t_ir.setTimingRMS(rms07);

          if ( UtilsClient::getBinStatus(meg03_[ism-1], ix, iy) ) {
            t_ir.setTaskStatus(true);
          } else {
            t_ir.setTaskStatus(false);
          }

          status = status && UtilsClient::getBinQuality(meg03_[ism-1], ix, iy);

          int ic = Numbers::indexEE(ism, ix, iy);

          if ( ic == -1 ) continue;

          if ( econn ) {
            ecid = LogicID::getEcalLogicID("EE_crystal_number", Numbers::iSM(ism, EcalEndcap), ic);
            dataset3_ir[ecid] = t_ir;
          }

        }

        if ( update04 ) {

          if ( Numbers::icEE(ism, ix, iy) == 1 ) {

            if ( verbose_ ) {
              cout << "Preparing dataset for " << Numbers::sEE(ism) << " (ism=" << ism << ")" << endl;
              cout << "L4A crystal (" << Numbers::ix0EE(i+1)+ix << "," << Numbers::iy0EE(i+1)+iy << ") " << num04  << " " << mean04 << " " << rms04  << endl;
              cout << endl;
            }

          }

          t_rd.setTimingMean(mean04);
          t_rd.setTimingRMS(rms04);

          if ( UtilsClient::getBinStatus(meg04_[ism-1], ix, iy) ) {
            t_rd.setTaskStatus(true);
          } else {
            t_rd.setTaskStatus(false);
          }

          status = status && UtilsClient::getBinQuality(meg04_[ism-1], ix, iy);

          int ic = Numbers::indexEE(ism, ix, iy);

          if ( ic == -1 ) continue;

          if ( econn ) {
            ecid = LogicID::getEcalLogicID("EE_crystal_number", Numbers::iSM(ism, EcalEndcap), ic);
            dataset3_rd[ecid] = t_rd;
          }

        }

        if ( update08 ) {

          if ( Numbers::icEE(ism, ix, iy) == 1 ) {

            if ( verbose_ ) {
              cout << "Preparing dataset for " << Numbers::sEE(ism) << " (ism=" << ism << ")" << endl;
              cout << "L4B crystal (" << Numbers::ix0EE(i+1)+ix << "," << Numbers::iy0EE(i+1)+iy << ") " << num08  << " " << mean08 << " " << rms08  << endl;
              cout << endl;
            }

          }

          t_bl.setTimingMean(mean08);
          t_bl.setTimingRMS(rms08);

          if ( UtilsClient::getBinStatus(meg04_[ism-1], ix, iy) ) {
            t_rd.setTaskStatus(true);
          } else {
            t_rd.setTaskStatus(false);
          }

          status = status && UtilsClient::getBinQuality(meg04_[ism-1], ix, iy);

          int ic = Numbers::indexEE(ism, ix, iy);

          if ( ic == -1 ) continue;

          if ( econn ) {
            ecid = LogicID::getEcalLogicID("EE_crystal_number", Numbers::iSM(ism, EcalEndcap), ic);
            dataset3_rd[ecid] = t_rd;
          }

        }

      }
    }

  }

  if ( econn ) {
    try {
      if ( verbose_ ) cout << "Inserting MonTimingLaserCrystalDat ..." << endl;
      if ( dataset3_bl.size() != 0 ) econn->insertDataArraySet(&dataset3_bl, moniov);
      if ( dataset3_ir.size() != 0 ) econn->insertDataArraySet(&dataset3_ir, moniov);
      if ( dataset3_gr.size() != 0 ) econn->insertDataArraySet(&dataset3_gr, moniov);
      if ( dataset3_rd.size() != 0 ) econn->insertDataArraySet(&dataset3_rd, moniov);
      if ( verbose_ ) cout << "done." << endl;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

  return true;

}

void EELaserClient::analyze(void) {

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( debug_ ) cout << "EELaserClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;
  }

  uint64_t bits01 = 0;
  bits01 |= EcalErrorDictionary::getMask("LASER_MEAN_WARNING");
  bits01 |= EcalErrorDictionary::getMask("LASER_RMS_WARNING");
  bits01 |= EcalErrorDictionary::getMask("LASER_MEAN_OVER_PN_WARNING");
  bits01 |= EcalErrorDictionary::getMask("LASER_RMS_OVER_PN_WARNING");

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

    sprintf(histo, (prefixME_ + "/EELaserTask/Laser1/EELT amplitude %s L1A").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    h01_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h01_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELaserTask/Laser1/EELT amplitude over PN %s L1A").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    h02_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h02_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELaserTask/Laser2/EELT amplitude %s L2A").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    h03_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h03_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELaserTask/Laser2/EELT amplitude over PN %s L2A").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    h04_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h04_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELaserTask/Laser3/EELT amplitude %s L3A").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    h05_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h05_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELaserTask/Laser3/EELT amplitude over PN %s L3A").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    h06_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h06_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELaserTask/Laser4/EELT amplitude %s L4A").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    h07_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h07_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELaserTask/Laser4/EELT amplitude over PN %s L4A").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    h08_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h08_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELaserTask/Laser1/EELT timing %s L1A").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    h09_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h09_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELaserTask/Laser2/EELT timing %s L2A").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    h10_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h10_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELaserTask/Laser3/EELT timing %s L3A").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    h11_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h11_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELaserTask/Laser4/EELT timing %s L4A").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    h12_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h12_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELaserTask/Laser1/EELT amplitude %s L1B").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    h13_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h13_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELaserTask/Laser1/EELT amplitude over PN %s L1B").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    h14_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h14_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELaserTask/Laser2/EELT amplitude %s L2B").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    h15_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h15_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELaserTask/Laser2/EELT amplitude over PN %s L2B").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    h16_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h16_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELaserTask/Laser3/EELT amplitude %s L3B").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    h17_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h17_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELaserTask/Laser3/EELT amplitude over PN %s L3B").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    h18_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h18_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELaserTask/Laser4/EELT amplitude %s L4B").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    h19_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h19_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELaserTask/Laser4/EELT amplitude over PN %s L4B").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    h20_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h20_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELaserTask/Laser1/EELT timing %s L1B").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    h21_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h21_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELaserTask/Laser2/EELT timing %s L2B").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    h22_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h22_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELaserTask/Laser3/EELT timing %s L3B").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    h23_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h23_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELaserTask/Laser4/EELT timing %s L4B").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    h24_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h24_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELaserTask/Laser1/EELT shape %s L1A").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    hs01_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, hs01_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELaserTask/Laser2/EELT shape %s L2A").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    hs02_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, hs02_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELaserTask/Laser3/EELT shape %s L3A").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    hs03_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, hs03_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELaserTask/Laser4/EELT shape %s L4A").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    hs04_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, hs04_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELaserTask/Laser1/EELT shape %s L1B").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    hs05_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, hs05_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELaserTask/Laser2/EELT shape %s L2B").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    hs06_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, hs06_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELaserTask/Laser3/EELT shape %s L3B").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    hs07_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, hs07_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELaserTask/Laser4/EELT shape %s L4B").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    hs08_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, hs08_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELaserTask/Laser1/PN/Gain01/EEPDT PNs amplitude %s G01 L1").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    i01_[ism-1] = UtilsClient::getHisto<TProfile*>( me, cloneME_, i01_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELaserTask/Laser2/PN/Gain01/EEPDT PNs amplitude %s G01 L2").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    i02_[ism-1] = UtilsClient::getHisto<TProfile*>( me, cloneME_, i02_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELaserTask/Laser3/PN/Gain01/EEPDT PNs amplitude %s G01 L3").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    i03_[ism-1] = UtilsClient::getHisto<TProfile*>( me, cloneME_, i03_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELaserTask/Laser4/PN/Gain01/EEPDT PNs amplitude %s G01 L4").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    i04_[ism-1] = UtilsClient::getHisto<TProfile*>( me, cloneME_, i04_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELaserTask/Laser1/PN/Gain01/EEPDT PNs pedestal %s G01 L1").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    i05_[ism-1] = UtilsClient::getHisto<TProfile*>( me, cloneME_, i05_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELaserTask/Laser2/PN/Gain01/EEPDT PNs pedestal %s G01 L2").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    i06_[ism-1] = UtilsClient::getHisto<TProfile*>( me, cloneME_, i06_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELaserTask/Laser3/PN/Gain01/EEPDT PNs pedestal %s G01 L3").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    i07_[ism-1] = UtilsClient::getHisto<TProfile*>( me, cloneME_, i07_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELaserTask/Laser4/PN/Gain01/EEPDT PNs pedestal %s G01 L4").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    i08_[ism-1] = UtilsClient::getHisto<TProfile*>( me, cloneME_, i08_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELaserTask/Laser1/PN/Gain16/EEPDT PNs amplitude %s G16 L1").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    i09_[ism-1] = UtilsClient::getHisto<TProfile*>( me, cloneME_, i09_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELaserTask/Laser2/PN/Gain16/EEPDT PNs amplitude %s G16 L2").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    i10_[ism-1] = UtilsClient::getHisto<TProfile*>( me, cloneME_, i10_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELaserTask/Laser3/PN/Gain16/EEPDT PNs amplitude %s G16 L3").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    i11_[ism-1] = UtilsClient::getHisto<TProfile*>( me, cloneME_, i11_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELaserTask/Laser4/PN/Gain16/EEPDT PNs amplitude %s G16 L4").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    i12_[ism-1] = UtilsClient::getHisto<TProfile*>( me, cloneME_, i12_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELaserTask/Laser1/PN/Gain16/EEPDT PNs pedestal %s G16 L1").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    i13_[ism-1] = UtilsClient::getHisto<TProfile*>( me, cloneME_, i13_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELaserTask/Laser2/PN/Gain16/EEPDT PNs pedestal %s G16 L2").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    i14_[ism-1] = UtilsClient::getHisto<TProfile*>( me, cloneME_, i14_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELaserTask/Laser3/PN/Gain16/EEPDT PNs pedestal %s G16 L3").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    i15_[ism-1] = UtilsClient::getHisto<TProfile*>( me, cloneME_, i15_[ism-1] );

    sprintf(histo, (prefixME_ + "/EELaserTask/Laser4/PN/Gain16/EEPDT PNs pedestal %s G16 L4").c_str(), Numbers::sEE(ism).c_str());
    me = dqmStore_->get(histo);
    i16_[ism-1] = UtilsClient::getHisto<TProfile*>( me, cloneME_, i16_[ism-1] );

    if ( meg01_[ism-1] ) meg01_[ism-1]->Reset();
    if ( meg02_[ism-1] ) meg02_[ism-1]->Reset();
    if ( meg03_[ism-1] ) meg03_[ism-1]->Reset();
    if ( meg04_[ism-1] ) meg04_[ism-1]->Reset();

    if ( meg05_[ism-1] ) meg05_[ism-1]->Reset();
    if ( meg06_[ism-1] ) meg06_[ism-1]->Reset();
    if ( meg07_[ism-1] ) meg07_[ism-1]->Reset();
    if ( meg08_[ism-1] ) meg08_[ism-1]->Reset();
    if ( meg09_[ism-1] ) meg09_[ism-1]->Reset();
    if ( meg10_[ism-1] ) meg10_[ism-1]->Reset();
    if ( meg11_[ism-1] ) meg11_[ism-1]->Reset();
    if ( meg12_[ism-1] ) meg12_[ism-1]->Reset();

    if ( mea01_[ism-1] ) mea01_[ism-1]->Reset();
    if ( mea02_[ism-1] ) mea02_[ism-1]->Reset();
    if ( mea03_[ism-1] ) mea03_[ism-1]->Reset();
    if ( mea04_[ism-1] ) mea04_[ism-1]->Reset();
    if ( mea05_[ism-1] ) mea05_[ism-1]->Reset();
    if ( mea06_[ism-1] ) mea06_[ism-1]->Reset();
    if ( mea07_[ism-1] ) mea07_[ism-1]->Reset();
    if ( mea08_[ism-1] ) mea08_[ism-1]->Reset();

    if ( met01_[ism-1] ) met01_[ism-1]->Reset();
    if ( met02_[ism-1] ) met02_[ism-1]->Reset();
    if ( met03_[ism-1] ) met03_[ism-1]->Reset();
    if ( met04_[ism-1] ) met04_[ism-1]->Reset();
    if ( met05_[ism-1] ) met05_[ism-1]->Reset();
    if ( met06_[ism-1] ) met06_[ism-1]->Reset();
    if ( met07_[ism-1] ) met07_[ism-1]->Reset();
    if ( met08_[ism-1] ) met08_[ism-1]->Reset();

    if ( metav01_[ism-1] ) metav01_[ism-1]->Reset();
    if ( metav02_[ism-1] ) metav02_[ism-1]->Reset();
    if ( metav03_[ism-1] ) metav03_[ism-1]->Reset();
    if ( metav04_[ism-1] ) metav04_[ism-1]->Reset();
    if ( metav05_[ism-1] ) metav05_[ism-1]->Reset();
    if ( metav06_[ism-1] ) metav06_[ism-1]->Reset();
    if ( metav07_[ism-1] ) metav07_[ism-1]->Reset();
    if ( metav08_[ism-1] ) metav08_[ism-1]->Reset();

    if ( metrms01_[ism-1] ) metrms01_[ism-1]->Reset();
    if ( metrms02_[ism-1] ) metrms02_[ism-1]->Reset();
    if ( metrms03_[ism-1] ) metrms03_[ism-1]->Reset();
    if ( metrms04_[ism-1] ) metrms04_[ism-1]->Reset();
    if ( metrms05_[ism-1] ) metrms05_[ism-1]->Reset();
    if ( metrms06_[ism-1] ) metrms06_[ism-1]->Reset();
    if ( metrms07_[ism-1] ) metrms07_[ism-1]->Reset();
    if ( metrms08_[ism-1] ) metrms08_[ism-1]->Reset();

    if ( meaopn01_[ism-1] ) meaopn01_[ism-1]->Reset();
    if ( meaopn02_[ism-1] ) meaopn02_[ism-1]->Reset();
    if ( meaopn03_[ism-1] ) meaopn03_[ism-1]->Reset();
    if ( meaopn04_[ism-1] ) meaopn04_[ism-1]->Reset();
    if ( meaopn05_[ism-1] ) meaopn05_[ism-1]->Reset();
    if ( meaopn06_[ism-1] ) meaopn06_[ism-1]->Reset();
    if ( meaopn07_[ism-1] ) meaopn07_[ism-1]->Reset();
    if ( meaopn08_[ism-1] ) meaopn08_[ism-1]->Reset();

    if ( mepnprms01_[ism-1] ) mepnprms01_[ism-1]->Reset();
    if ( mepnprms02_[ism-1] ) mepnprms02_[ism-1]->Reset();
    if ( mepnprms03_[ism-1] ) mepnprms03_[ism-1]->Reset();
    if ( mepnprms04_[ism-1] ) mepnprms04_[ism-1]->Reset();
    if ( mepnprms05_[ism-1] ) mepnprms05_[ism-1]->Reset();
    if ( mepnprms06_[ism-1] ) mepnprms06_[ism-1]->Reset();
    if ( mepnprms07_[ism-1] ) mepnprms07_[ism-1]->Reset();
    if ( mepnprms08_[ism-1] ) mepnprms08_[ism-1]->Reset();

    if ( me_hs01_[ism-1] ) me_hs01_[ism-1]->Reset();
    if ( me_hs02_[ism-1] ) me_hs02_[ism-1]->Reset();
    if ( me_hs03_[ism-1] ) me_hs03_[ism-1]->Reset();
    if ( me_hs04_[ism-1] ) me_hs04_[ism-1]->Reset();
    if ( me_hs05_[ism-1] ) me_hs05_[ism-1]->Reset();
    if ( me_hs06_[ism-1] ) me_hs06_[ism-1]->Reset();
    if ( me_hs07_[ism-1] ) me_hs07_[ism-1]->Reset();
    if ( me_hs08_[ism-1] ) me_hs08_[ism-1]->Reset();

    float meanAmplL1A, meanAmplL2A, meanAmplL3A, meanAmplL4A;
    float meanAmplL1B, meanAmplL2B, meanAmplL3B, meanAmplL4B;

    int nCryL1A, nCryL2A, nCryL3A, nCryL4A;
    int nCryL1B, nCryL2B, nCryL3B, nCryL4B;

    meanAmplL1A = meanAmplL2A = meanAmplL3A = meanAmplL4A = 0.;
    meanAmplL1B = meanAmplL2B = meanAmplL3B = meanAmplL4B = 0.;

    nCryL1A = nCryL2A = nCryL3A = nCryL4A = 0;
    nCryL1B = nCryL2B = nCryL3B = nCryL4B = 0;

    for ( int ix = 1; ix <= 50; ix++ ) {
      for ( int iy = 1; iy <= 50; iy++ ) {

        bool update01;
        bool update02;
        bool update03;
        bool update04;
        bool update05;
        bool update06;
        bool update07;
        bool update08;

        float num01, num02, num03, num04, num05, num06, num07, num08;
        float mean01, mean02, mean03, mean04, mean05, mean06, mean07, mean08;
        float rms01, rms02, rms03, rms04, rms05, rms06, rms07, rms08;

        update01 = UtilsClient::getBinStatistics(h01_[ism-1], ix, iy, num01, mean01, rms01);
        update02 = UtilsClient::getBinStatistics(h03_[ism-1], ix, iy, num02, mean02, rms02);
        update03 = UtilsClient::getBinStatistics(h05_[ism-1], ix, iy, num03, mean03, rms03);
        update04 = UtilsClient::getBinStatistics(h07_[ism-1], ix, iy, num04, mean04, rms04);
        update05 = UtilsClient::getBinStatistics(h13_[ism-1], ix, iy, num05, mean05, rms05);
        update06 = UtilsClient::getBinStatistics(h15_[ism-1], ix, iy, num06, mean06, rms06);
        update07 = UtilsClient::getBinStatistics(h17_[ism-1], ix, iy, num07, mean07, rms07);
        update08 = UtilsClient::getBinStatistics(h19_[ism-1], ix, iy, num08, mean08, rms08);

        if ( update01 ) {
          meanAmplL1A += mean01;
          nCryL1A++;
        }

        if ( update02 ) {
          meanAmplL2A += mean02;
          nCryL2A++;
        }

        if ( update03 ) {
          meanAmplL3A += mean03;
          nCryL3A++;
        }

        if ( update04 ) {
          meanAmplL4A += mean04;
          nCryL4A++;
        }

        if ( update05 ) {
          meanAmplL1B += mean05;
          nCryL1B++;
        }

        if ( update06 ) {
          meanAmplL2B += mean06;
          nCryL2B++;
        }

        if ( update07 ) {
          meanAmplL3B += mean07;
          nCryL3B++;
        }

        if ( update08 ) {
          meanAmplL4B += mean08;
          nCryL4B++;
        }

      }
    }

    if ( nCryL1A > 0 ) meanAmplL1A /= float (nCryL1A);
    if ( nCryL2A > 0 ) meanAmplL2A /= float (nCryL2A);
    if ( nCryL3A > 0 ) meanAmplL3A /= float (nCryL3A);
    if ( nCryL4A > 0 ) meanAmplL4A /= float (nCryL4A);
    if ( nCryL1B > 0 ) meanAmplL1B /= float (nCryL1B);
    if ( nCryL2B > 0 ) meanAmplL2B /= float (nCryL2B);
    if ( nCryL3B > 0 ) meanAmplL3B /= float (nCryL3B);
    if ( nCryL4B > 0 ) meanAmplL4B /= float (nCryL4B);

    for ( int ix = 1; ix <= 50; ix++ ) {
      for ( int iy = 1; iy <= 50; iy++ ) {

        if ( meg01_[ism-1] ) meg01_[ism-1]->setBinContent( ix, iy, 6.);
        if ( meg02_[ism-1] ) meg02_[ism-1]->setBinContent( ix, iy, 6.);
        if ( meg03_[ism-1] ) meg03_[ism-1]->setBinContent( ix, iy, 6.);
        if ( meg04_[ism-1] ) meg04_[ism-1]->setBinContent( ix, iy, 6.);

        int jx = ix + Numbers::ix0EE(ism);
        int jy = iy + Numbers::iy0EE(ism);

        if ( ism >= 1 && ism <= 9 ) jx = 101 - jx;

        if ( Numbers::validEE(ism, jx, jy) ) {
          if ( meg01_[ism-1] ) meg01_[ism-1]->setBinContent( ix, iy, 2.);
          if ( meg02_[ism-1] ) meg02_[ism-1]->setBinContent( ix, iy, 2.);
          if ( meg03_[ism-1] ) meg03_[ism-1]->setBinContent( ix, iy, 2.);
          if ( meg04_[ism-1] ) meg04_[ism-1]->setBinContent( ix, iy, 2.);
        }

        bool update01;
        bool update02;
        bool update03;
        bool update04;
        bool update05;
        bool update06;
        bool update07;
        bool update08;
        bool update09;
        bool update10;
        bool update11;
        bool update12;

        bool update13;
        bool update14;
        bool update15;
        bool update16;
        bool update17;
        bool update18;
        bool update19;
        bool update20;
        bool update21;
        bool update22;
        bool update23;
        bool update24;

        float num01, num02, num03, num04, num05, num06, num07, num08;
        float num09, num10, num11, num12;
        float mean01, mean02, mean03, mean04, mean05, mean06, mean07, mean08;
        float mean09, mean10, mean11, mean12;
        float rms01, rms02, rms03, rms04, rms05, rms06, rms07, rms08;
        float rms09, rms10, rms11, rms12;

        float num13, num14, num15, num16, num17, num18, num19, num20;
        float num21, num22, num23, num24;
        float mean13, mean14, mean15, mean16, mean17, mean18, mean19, mean20;
        float mean21, mean22, mean23, mean24;
        float rms13, rms14, rms15, rms16, rms17, rms18, rms19, rms20;
        float rms21, rms22, rms23, rms24;

        update01 = UtilsClient::getBinStatistics(h01_[ism-1], ix, iy, num01, mean01, rms01);
        update02 = UtilsClient::getBinStatistics(h02_[ism-1], ix, iy, num02, mean02, rms02);
        update03 = UtilsClient::getBinStatistics(h03_[ism-1], ix, iy, num03, mean03, rms03);
        update04 = UtilsClient::getBinStatistics(h04_[ism-1], ix, iy, num04, mean04, rms04);
        update05 = UtilsClient::getBinStatistics(h05_[ism-1], ix, iy, num05, mean05, rms05);
        update06 = UtilsClient::getBinStatistics(h06_[ism-1], ix, iy, num06, mean06, rms06);
        update07 = UtilsClient::getBinStatistics(h07_[ism-1], ix, iy, num07, mean07, rms07);
        update08 = UtilsClient::getBinStatistics(h08_[ism-1], ix, iy, num08, mean08, rms08);
        update09 = UtilsClient::getBinStatistics(h09_[ism-1], ix, iy, num09, mean09, rms09);
        update10 = UtilsClient::getBinStatistics(h10_[ism-1], ix, iy, num10, mean10, rms10);
        update11 = UtilsClient::getBinStatistics(h11_[ism-1], ix, iy, num11, mean11, rms11);
        update12 = UtilsClient::getBinStatistics(h12_[ism-1], ix, iy, num12, mean12, rms12);

        // other SM half

        update13 = UtilsClient::getBinStatistics(h13_[ism-1], ix, iy, num13, mean13, rms13);
        update14 = UtilsClient::getBinStatistics(h14_[ism-1], ix, iy, num14, mean14, rms14);
        update15 = UtilsClient::getBinStatistics(h15_[ism-1], ix, iy, num15, mean15, rms15);
        update16 = UtilsClient::getBinStatistics(h16_[ism-1], ix, iy, num16, mean16, rms16);
        update17 = UtilsClient::getBinStatistics(h17_[ism-1], ix, iy, num17, mean17, rms17);
        update18 = UtilsClient::getBinStatistics(h18_[ism-1], ix, iy, num18, mean18, rms18);
        update19 = UtilsClient::getBinStatistics(h19_[ism-1], ix, iy, num19, mean19, rms19);
        update20 = UtilsClient::getBinStatistics(h20_[ism-1], ix, iy, num20, mean20, rms20);
        update21 = UtilsClient::getBinStatistics(h21_[ism-1], ix, iy, num21, mean21, rms21);
        update22 = UtilsClient::getBinStatistics(h22_[ism-1], ix, iy, num22, mean22, rms22);
        update23 = UtilsClient::getBinStatistics(h23_[ism-1], ix, iy, num23, mean23, rms23);
        update24 = UtilsClient::getBinStatistics(h24_[ism-1], ix, iy, num24, mean24, rms24);

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

        if ( update05 ) {

          float val;

          val = 1.;
          if ( fabs(mean05 - meanAmplL3A) > fabs(percentVariation_ * meanAmplL3A) || mean05 < amplitudeThreshold_ )
            val = 0.;
          if ( meg03_[ism-1] ) meg03_[ism-1]->setBinContent( ix, iy, val );

          int ic = Numbers::icEE(ism, jx, jy);

          if ( ic != -1 ) {
            if ( mea03_[ism-1] ) {
              if ( mean05 > 0. ) {
                mea03_[ism-1]->setBinContent( ic, mean05 );
                mea03_[ism-1]->setBinError( ic, rms05 );
              } else {
                mea03_[ism-1]->setEntries( 1.+mea03_[ism-1]->getEntries() );
              }
            }
          }

        }

        if ( update17 ) {

          float val;

          val = 1.;
          if ( fabs(mean17 - meanAmplL3B) > fabs(percentVariation_ * meanAmplL3B) || mean17 < amplitudeThreshold_ )
            val = 0.;
          if ( meg03_[ism-1] ) meg03_[ism-1]->setBinContent( ix, iy, val );

          int ic = Numbers::icEE(ism, jx, jy);

          if ( ic != -1 ) {
            if ( mea07_[ism-1] ) {
              if ( mean17 > 0. ) {
                mea07_[ism-1]->setBinContent( ic, mean17 );
                mea07_[ism-1]->setBinError( ic, rms17 );
              } else {
                mea07_[ism-1]->setEntries( 1.+mea07_[ism-1]->getEntries() );
              }
            }
          }

        }

        if ( update07 ) {

          float val;

          val = 1.;
          if ( fabs(mean07 - meanAmplL4A) > fabs(percentVariation_ * meanAmplL4A) || mean07 < amplitudeThreshold_ )
            val = 0.;
          if ( meg04_[ism-1] ) meg04_[ism-1]->setBinContent( ix, iy, val );

          int ic = Numbers::icEE(ism, jx, jy);

          if ( ic != -1 ) {
            if ( mea04_[ism-1] ) {
              if ( mean07 > 0. ) {
                mea04_[ism-1]->setBinContent( ic, mean07 );
                mea04_[ism-1]->setBinError( ic, rms07 );
              } else {
                mea04_[ism-1]->setEntries( 1.+mea04_[ism-1]->getEntries() );
              }
            }
          }

        }

        if ( update19 ) {

          float val;

          val = 1.;
          if ( fabs(mean19 - meanAmplL4B) > fabs(percentVariation_ * meanAmplL4B) || mean19 < amplitudeThreshold_ )
            val = 0.;
          if ( meg04_[ism-1] ) meg04_[ism-1]->setBinContent( ix, iy, val );

          int ic = Numbers::icEE(ism, jx, jy);

          if ( ic != -1 ) {
            if ( mea08_[ism-1] ) {
              if ( mean19 > 0. ) {
                mea08_[ism-1]->setBinContent( ic, mean19 );
                mea08_[ism-1]->setBinError( ic, rms19 );
              } else {
                mea08_[ism-1]->setEntries( 1.+mea08_[ism-1]->getEntries() );
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

        if ( update06 ) {

          int ic = Numbers::icEE(ism, jx, jy);

          if ( ic != -1 ) {
            if ( meaopn03_[ism-1] ) {
              if ( mean06 > 0. ) {
                meaopn03_[ism-1]->setBinContent( ic, mean06 );
                meaopn03_[ism-1]->setBinError( ic, rms06 );
              } else {
                meaopn03_[ism-1]->setEntries( 1.+meaopn03_[ism-1]->getEntries() );
              }
            }
          }

        }

        if ( update18 ) {

          int ic = Numbers::icEE(ism, jx, jy);

          if ( ic != -1 ) {
            if ( meaopn07_[ism-1] ) {
              if ( mean18 > 0. ) {
                meaopn07_[ism-1]->setBinContent( ic, mean18 );
                meaopn07_[ism-1]->setBinError( ic, rms18 );
              } else {
                meaopn07_[ism-1]->setEntries( 1.+meaopn07_[ism-1]->getEntries() );
              }
            }
          }

        }

        if ( update08 ) {

          int ic = Numbers::icEE(ism, jx, jy);

          if ( ic != -1 ) {
            if ( meaopn04_[ism-1] ) {
              if ( mean08 > 0. ) {
                meaopn04_[ism-1]->setBinContent( ic, mean08 );
                meaopn04_[ism-1]->setBinError( ic, rms08 );
              } else {
                meaopn04_[ism-1]->setEntries( 1.+meaopn04_[ism-1]->getEntries() );
              }
            }
          }

        }

        if ( update20 ) {

          int ic = Numbers::icEE(ism, jx, jy);

          if ( ic != -1 ) {
            if ( meaopn08_[ism-1] ) {
              if ( mean20 > 0. ) {
                meaopn08_[ism-1]->setBinContent( ic, mean20 );
                meaopn08_[ism-1]->setBinError( ic, rms20 );
              } else {
                meaopn08_[ism-1]->setEntries( 1.+meaopn08_[ism-1]->getEntries() );
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

        if ( update11 ) {

          int ic = Numbers::icEE(ism, jx, jy);

          if ( ic != -1 ) {
            if ( met03_[ism-1] ) {
              if ( mean11 > 0. ) {
                met03_[ism-1]->setBinContent( ic, mean11 );
                met03_[ism-1]->setBinError( ic, rms11 );
              } else {
                met03_[ism-1]->setEntries(1.+met03_[ism-1]->getEntries());
              }
            }

            if ( metav03_[ism-1] ) metav03_[ism-1] ->Fill(mean11);
            if ( metrms03_[ism-1] ) metrms03_[ism-1]->Fill(rms11);
          }

        }

        if ( update23 ) {

          int ic = Numbers::icEE(ism, jx, jy);

          if ( ic != -1 ) {
            if ( met07_[ism-1] ) {
              if ( mean23 > 0. ) {
                met07_[ism-1]->setBinContent( ic, mean23 );
                met07_[ism-1]->setBinError( ic, rms23 );
              } else {
                met07_[ism-1]->setEntries(1.+met07_[ism-1]->getEntries());
              }
            }

            if ( metav07_[ism-1] ) metav07_[ism-1] ->Fill(mean23);
            if ( metrms07_[ism-1] ) metrms07_[ism-1]->Fill(rms23);
          }

        }

        if ( update12 ) {

          int ic = Numbers::icEE(ism, jx, jy);

          if ( ic != -1 ) {
            if ( met04_[ism-1] ) {
              if ( mean12 > 0. ) {
                met04_[ism-1]->setBinContent( ic, mean12 );
                met04_[ism-1]->setBinError( ic, rms12 );
              } else {
                met04_[ism-1]->setEntries(1.+met04_[ism-1]->getEntries());
              }
            }

            if ( metav04_[ism-1] ) metav04_[ism-1] ->Fill(mean12);
            if ( metrms04_[ism-1] ) metrms04_[ism-1]->Fill(rms12);
          }

        }

        if ( update24 ) {

          int ic = Numbers::icEE(ism, jx, jy);

          if ( ic != -1 ) {
            if ( met08_[ism-1] ) {
              if ( mean24 > 0. ) {
                met08_[ism-1]->setBinContent( ic, mean24 );
                met08_[ism-1]->setBinError( ic, rms24 );
              } else {
                met08_[ism-1]->setEntries(1.+met08_[ism-1]->getEntries());
              }
            }

            if ( metav08_[ism-1] ) metav08_[ism-1] ->Fill(mean24);
            if ( metrms08_[ism-1] ) metrms08_[ism-1]->Fill(rms24);
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
                UtilsClient::maskBinContent( meg03_[ism-1], ix, iy );
                UtilsClient::maskBinContent( meg04_[ism-1], ix, iy );
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
                UtilsClient::maskBinContent( meg01_[ism-1], ix, iy );
                UtilsClient::maskBinContent( meg02_[ism-1], ix, iy );
                UtilsClient::maskBinContent( meg03_[ism-1], ix, iy );
                UtilsClient::maskBinContent( meg04_[ism-1], ix, iy );
              }
            }

          }
        }

      }
    }

    for ( int i = 1; i <= 10; i++ ) {

      if ( meg05_[ism-1] ) meg05_[ism-1]->setBinContent( i, 1, 2. );
      if ( meg06_[ism-1] ) meg06_[ism-1]->setBinContent( i, 1, 2. );
      if ( meg07_[ism-1] ) meg07_[ism-1]->setBinContent( i, 1, 2. );
      if ( meg08_[ism-1] ) meg08_[ism-1]->setBinContent( i, 1, 2. );
      if ( meg09_[ism-1] ) meg09_[ism-1]->setBinContent( i, 1, 2. );
      if ( meg10_[ism-1] ) meg10_[ism-1]->setBinContent( i, 1, 2. );
      if ( meg11_[ism-1] ) meg11_[ism-1]->setBinContent( i, 1, 2. );
      if ( meg12_[ism-1] ) meg12_[ism-1]->setBinContent( i, 1, 2. );

      bool update01;
      bool update02;
      bool update03;
      bool update04;
      bool update05;
      bool update06;
      bool update07;
      bool update08;
      bool update09;
      bool update10;
      bool update11;
      bool update12;
      bool update13;
      bool update14;
      bool update15;
      bool update16;

      float num01, num02, num03, num04, num05, num06, num07, num08;
      float num09, num10, num11, num12, num13, num14, num15, num16;
      float mean01, mean02, mean03, mean04, mean05, mean06, mean07, mean08;
      float mean09, mean10, mean11, mean12, mean13, mean14, mean15, mean16;
      float rms01, rms02, rms03, rms04, rms05, rms06, rms07, rms08;
      float rms09, rms10, rms11, rms12, rms13, rms14, rms15, rms16;

      update01 = UtilsClient::getBinStatistics(i01_[ism-1], i, 0, num01, mean01, rms01);
      update02 = UtilsClient::getBinStatistics(i02_[ism-1], i, 0, num02, mean02, rms02);
      update03 = UtilsClient::getBinStatistics(i03_[ism-1], i, 0, num03, mean03, rms03);
      update04 = UtilsClient::getBinStatistics(i04_[ism-1], i, 0, num04, mean04, rms04);
      update05 = UtilsClient::getBinStatistics(i05_[ism-1], i, 0, num05, mean05, rms05);
      update06 = UtilsClient::getBinStatistics(i06_[ism-1], i, 0, num06, mean06, rms06);
      update07 = UtilsClient::getBinStatistics(i07_[ism-1], i, 0, num07, mean07, rms07);
      update08 = UtilsClient::getBinStatistics(i08_[ism-1], i, 0, num08, mean08, rms08);
      update09 = UtilsClient::getBinStatistics(i09_[ism-1], i, 0, num09, mean09, rms09);
      update10 = UtilsClient::getBinStatistics(i10_[ism-1], i, 0, num10, mean10, rms10);
      update11 = UtilsClient::getBinStatistics(i11_[ism-1], i, 0, num11, mean11, rms11);
      update12 = UtilsClient::getBinStatistics(i12_[ism-1], i, 0, num12, mean12, rms12);
      update13 = UtilsClient::getBinStatistics(i13_[ism-1], i, 0, num13, mean13, rms13);
      update14 = UtilsClient::getBinStatistics(i14_[ism-1], i, 0, num14, mean14, rms14);
      update15 = UtilsClient::getBinStatistics(i15_[ism-1], i, 0, num15, mean15, rms15);
      update16 = UtilsClient::getBinStatistics(i16_[ism-1], i, 0, num16, mean16, rms16);

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

      if ( update03 && update07 ) {

        float val;

        val = 1.;
        if ( mean03 < amplitudeThresholdPnG01_ )
          val = 0.;
      if ( mean07 <  pedPnExpectedMean_[0] - pedPnDiscrepancyMean_[0] ||
           pedPnExpectedMean_[0] + pedPnDiscrepancyMean_[0] < mean07)
          val = 0.;
      if ( rms07 > pedPnRMSThreshold_[0] )
          val = 0.;

      if ( meg07_[ism-1] )           meg07_[ism-1]->setBinContent(i, 1, val);
      if ( mepnprms03_[ism-1] ) mepnprms03_[ism-1]->Fill(rms07);

      }

      if ( update04 && update08 ) {

        float val;

        val = 1.;
        if ( mean04 < amplitudeThresholdPnG01_ )
          val = 0.;
       if ( mean08 <  pedPnExpectedMean_[0] - pedPnDiscrepancyMean_[0] ||
            pedPnExpectedMean_[0] + pedPnDiscrepancyMean_[0] < mean08)
           val = 0.;
       if ( rms08 > pedPnRMSThreshold_[0] )
           val = 0.;

       if ( meg08_[ism-1] )           meg08_[ism-1]->setBinContent(i, 1, val);
       if ( mepnprms04_[ism-1] ) mepnprms04_[ism-1]->Fill(rms08);

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

      if ( update11 && update15 ) {

        float val;

        val = 1.;
        if ( mean11 < amplitudeThresholdPnG16_ )
          val = 0.;
        //        if ( mean15 < pedestalThresholdPn_ )
        if ( mean15 <  pedPnExpectedMean_[1] - pedPnDiscrepancyMean_[1] ||
             pedPnExpectedMean_[1] + pedPnDiscrepancyMean_[1] < mean15)
          val = 0.;
        if ( rms15 > pedPnRMSThreshold_[1] )
          val = 0.;

        if ( meg11_[ism-1] ) meg11_[ism-1]->setBinContent(i, 1, val);
        if ( mepnprms07_[ism-1] ) mepnprms07_[ism-1]->Fill(rms15);

      }

      if ( update12 && update16 ) {

        float val;

        val = 1.;
        if ( mean12 < amplitudeThresholdPnG16_ )
          val = 0.;
        //        if ( mean16 < pedestalThresholdPn_ )
        if ( mean16 <  pedPnExpectedMean_[1] - pedPnDiscrepancyMean_[1] ||
                        pedPnExpectedMean_[1] + pedPnDiscrepancyMean_[1] < mean16)
          val = 0.;
        if ( rms16 > pedPnRMSThreshold_[1] )
          val = 0.;

        if ( meg12_[ism-1] ) meg12_[ism-1]->setBinContent(i, 1, val);
        if ( mepnprms08_[ism-1] ) mepnprms08_[ism-1]->Fill(rms16);

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
              UtilsClient::maskBinContent( meg07_[ism-1], i, 1 );
              UtilsClient::maskBinContent( meg08_[ism-1], i, 1 );
            }
            if ( (m->second).getErrorBits() & (bits01|bits04) ) {
              UtilsClient::maskBinContent( meg09_[ism-1], i, 1 );
              UtilsClient::maskBinContent( meg10_[ism-1], i, 1 );
              UtilsClient::maskBinContent( meg11_[ism-1], i, 1 );
              UtilsClient::maskBinContent( meg12_[ism-1], i, 1 );
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

      if ( hs03_[ism-1] ) {
        int ic = UtilsClient::getFirstNonEmptyChannel( hs03_[ism-1] );
        if ( me_hs03_[ism-1] ) {
          me_hs03_[ism-1]->setBinContent( i, hs03_[ism-1]->GetBinContent(ic, i) );
          me_hs03_[ism-1]->setBinError( i, hs03_[ism-1]->GetBinError(ic, i) );
        }
      }

      if ( hs04_[ism-1] ) {
        int ic = UtilsClient::getFirstNonEmptyChannel( hs04_[ism-1] );
        if ( me_hs04_[ism-1] ) {
          me_hs04_[ism-1]->setBinContent( i, hs04_[ism-1]->GetBinContent(ic, i) );
          me_hs04_[ism-1]->setBinError( i, hs04_[ism-1]->GetBinError(ic, i) );
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

      if ( hs07_[ism-1] ) {
        int ic = UtilsClient::getFirstNonEmptyChannel( hs07_[ism-1] );
        if ( me_hs07_[ism-1] ) {
          me_hs07_[ism-1]->setBinContent( i, hs07_[ism-1]->GetBinContent(ic, i) );
          me_hs07_[ism-1]->setBinError( i, hs07_[ism-1]->GetBinError(ic, i) );
        }
      }

      if ( hs08_[ism-1] ) {
        int ic = UtilsClient::getFirstNonEmptyChannel( hs08_[ism-1] );
        if ( me_hs08_[ism-1] ) {
          me_hs08_[ism-1]->setBinContent( i, hs08_[ism-1]->GetBinContent(ic, i) );
          me_hs08_[ism-1]->setBinError( i, hs08_[ism-1]->GetBinError(ic, i) );
        }
      }

    }

  }

}

void EELaserClient::softReset(bool flag) {

}

