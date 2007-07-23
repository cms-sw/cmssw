/*
 * \file EBLaserClient.cc
 *
 * $Date: 2007/02/19 07:14:22 $
 * $Revision: 1.141 $
 * \author G. Della Ricca
 *
*/

#include <memory>
#include <iostream>
#include <fstream>

#include "TStyle.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"

#include "DQMServices/UI/interface/MonitorUIRoot.h"

#include "OnlineDB/EcalCondDB/interface/RunTag.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"

#include "OnlineDB/EcalCondDB/interface/MonLaserBlueDat.h"
#include "OnlineDB/EcalCondDB/interface/MonLaserGreenDat.h"
#include "OnlineDB/EcalCondDB/interface/MonLaserIRedDat.h"
#include "OnlineDB/EcalCondDB/interface/MonLaserRedDat.h"

#include "OnlineDB/EcalCondDB/interface/MonPNBlueDat.h"
#include "OnlineDB/EcalCondDB/interface/MonPNGreenDat.h"
#include "OnlineDB/EcalCondDB/interface/MonPNIRedDat.h"
#include "OnlineDB/EcalCondDB/interface/MonPNRedDat.h"

#include <DQM/EcalBarrelMonitorClient/interface/EBLaserClient.h>
#include <DQM/EcalBarrelMonitorClient/interface/EBMUtilsClient.h>

#include "DQM/EcalBarrelMonitorClient/interface/EcalErrorMask.h"
#include "CondTools/Ecal/interface/EcalErrorDictionary.h"
#include "OnlineDB/EcalCondDB/interface/RunCrystalErrorsDat.h"
#include "OnlineDB/EcalCondDB/interface/RunPNErrorsDat.h"

using namespace cms;
using namespace edm;
using namespace std;

EBLaserClient::EBLaserClient(const ParameterSet& ps){

  // collateSources switch
  collateSources_ = ps.getUntrackedParameter<bool>("collateSources", false);

  // cloneME switch
  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);

  // enableQT switch
  enableQT_ = ps.getUntrackedParameter<bool>("enableQT", true);

  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

  // MonitorDaemon switch
  enableMonitorDaemon_ = ps.getUntrackedParameter<bool>("enableMonitorDaemon", true);

  // prefix to ME paths
  prefixME_ = ps.getUntrackedParameter<string>("prefixME", "");

  // vector of selected Super Modules (Defaults to all 36).
  superModules_.reserve(36);
  for ( unsigned int i = 1; i < 37; i++ ) superModules_.push_back(i);
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

    meaopn01_[ism-1] = 0;
    meaopn02_[ism-1] = 0;
    meaopn03_[ism-1] = 0;
    meaopn04_[ism-1] = 0;
    meaopn05_[ism-1] = 0;
    meaopn06_[ism-1] = 0;
    meaopn07_[ism-1] = 0;
    meaopn08_[ism-1] = 0;

    qth01_[ism-1] = 0;
    qth02_[ism-1] = 0;
    qth03_[ism-1] = 0;
    qth04_[ism-1] = 0;
    qth05_[ism-1] = 0;
    qth06_[ism-1] = 0;
    qth07_[ism-1] = 0;
    qth08_[ism-1] = 0;

    qth09_[ism-1] = 0;
    qth10_[ism-1] = 0;
    qth11_[ism-1] = 0;
    qth12_[ism-1] = 0;
    qth13_[ism-1] = 0;
    qth14_[ism-1] = 0;
    qth15_[ism-1] = 0;
    qth16_[ism-1] = 0;
    qth17_[ism-1] = 0;
    qth18_[ism-1] = 0;
    qth19_[ism-1] = 0;
    qth20_[ism-1] = 0;
    qth21_[ism-1] = 0;
    qth22_[ism-1] = 0;
    qth23_[ism-1] = 0;
    qth24_[ism-1] = 0;

  }

  percentVariation_ = 0.4;

  amplitudeThresholdPN_ = 50.;
  meanThresholdPN_ = 200.;

}

EBLaserClient::~EBLaserClient(){

}

void EBLaserClient::beginJob(MonitorUserInterface* mui){

  mui_ = mui;

  if ( verbose_ ) cout << "EBLaserClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

  if ( enableQT_ ) {

    Char_t qtname[200];

    for ( unsigned int i=0; i<superModules_.size(); i++ ) {

      int ism = superModules_[i];

      sprintf(qtname, "EBLT laser quality SM%02d L1A", ism);
      qth01_[ism-1] = dynamic_cast<MEContentsProf2DWithinRangeROOT*> (mui_->createQTest(ContentsProf2DWithinRangeROOT::getAlgoName(), qtname));

      sprintf(qtname, "EBLT laser quality SM%02d L2A", ism);
      qth02_[ism-1] = dynamic_cast<MEContentsProf2DWithinRangeROOT*> (mui_->createQTest(ContentsProf2DWithinRangeROOT::getAlgoName(), qtname));

      sprintf(qtname, "EBLT laser quality SM%02d L3A", ism);
      qth03_[ism-1] = dynamic_cast<MEContentsProf2DWithinRangeROOT*> (mui_->createQTest(ContentsProf2DWithinRangeROOT::getAlgoName(), qtname));

      sprintf(qtname, "EBLT laser quality SM%02d L4A", ism);
      qth04_[ism-1] = dynamic_cast<MEContentsProf2DWithinRangeROOT*> (mui_->createQTest(ContentsProf2DWithinRangeROOT::getAlgoName(), qtname));

      sprintf(qtname, "EBLT laser quality SM%02d L1B", ism);
      qth05_[ism-1] = dynamic_cast<MEContentsProf2DWithinRangeROOT*> (mui_->createQTest(ContentsProf2DWithinRangeROOT::getAlgoName(), qtname));

      sprintf(qtname, "EBLT laser quality SM%02d L2B", ism);
      qth06_[ism-1] = dynamic_cast<MEContentsProf2DWithinRangeROOT*> (mui_->createQTest(ContentsProf2DWithinRangeROOT::getAlgoName(), qtname));

      sprintf(qtname, "EBLT laser quality SM%02d L3B", ism);
      qth07_[ism-1] = dynamic_cast<MEContentsProf2DWithinRangeROOT*> (mui_->createQTest(ContentsProf2DWithinRangeROOT::getAlgoName(), qtname));

      sprintf(qtname, "EBLT laser quality SM%02d L4B", ism);
      qth08_[ism-1] = dynamic_cast<MEContentsProf2DWithinRangeROOT*> (mui_->createQTest(ContentsProf2DWithinRangeROOT::getAlgoName(), qtname));

      sprintf(qtname, "EBLT laser amplitude quality PNs SM%02d L1 G01", ism);
      qth09_[ism-1] = dynamic_cast<MEContentsProf2DWithinRangeROOT*> (mui_->createQTest(ContentsProf2DWithinRangeROOT::getAlgoName(), qtname));

      sprintf(qtname, "EBLT laser amplitude quality PNs SM%02d L2 G10", ism);
      qth10_[ism-1] = dynamic_cast<MEContentsProf2DWithinRangeROOT*> (mui_->createQTest(ContentsProf2DWithinRangeROOT::getAlgoName(), qtname));

      sprintf(qtname, "EBLT laser amplitude quality PNs SM%02d L3 G01", ism);
      qth11_[ism-1] = dynamic_cast<MEContentsProf2DWithinRangeROOT*> (mui_->createQTest(ContentsProf2DWithinRangeROOT::getAlgoName(), qtname));

      sprintf(qtname, "EBLT laser amplitude quality PNs SM%02d L4 G01", ism);
      qth12_[ism-1] = dynamic_cast<MEContentsProf2DWithinRangeROOT*> (mui_->createQTest(ContentsProf2DWithinRangeROOT::getAlgoName(), qtname));

      sprintf(qtname, "EBLT laser pedestal quality PNs SM%02d L1 G01", ism);
      qth13_[ism-1] = dynamic_cast<MEContentsProf2DWithinRangeROOT*> (mui_->createQTest(ContentsProf2DWithinRangeROOT::getAlgoName(), qtname));

      sprintf(qtname, "EBLT laser pedestal quality PNs SM%02d L2 G01", ism);
      qth14_[ism-1] = dynamic_cast<MEContentsProf2DWithinRangeROOT*> (mui_->createQTest(ContentsProf2DWithinRangeROOT::getAlgoName(), qtname));

      sprintf(qtname, "EBLT laser pedestal quality PNs SM%02d L3 G01", ism);
      qth15_[ism-1] = dynamic_cast<MEContentsProf2DWithinRangeROOT*> (mui_->createQTest(ContentsProf2DWithinRangeROOT::getAlgoName(), qtname));

      sprintf(qtname, "EBLT laser pedestal quality PNs SM%02d L4 G01", ism);
      qth16_[ism-1] = dynamic_cast<MEContentsProf2DWithinRangeROOT*> (mui_->createQTest(ContentsProf2DWithinRangeROOT::getAlgoName(), qtname));

      sprintf(qtname, "EBLT laser amplitude quality PNs SM%02d L1 G16", ism);
      qth17_[ism-1] = dynamic_cast<MEContentsProf2DWithinRangeROOT*> (mui_->createQTest(ContentsProf2DWithinRangeROOT::getAlgoName(), qtname));

      sprintf(qtname, "EBLT laser amplitude quality PNs SM%02d L2 G16", ism);
      qth18_[ism-1] = dynamic_cast<MEContentsProf2DWithinRangeROOT*> (mui_->createQTest(ContentsProf2DWithinRangeROOT::getAlgoName(), qtname));

      sprintf(qtname, "EBLT laser amplitude quality PNs SM%02d L3 G16", ism);
      qth19_[ism-1] = dynamic_cast<MEContentsProf2DWithinRangeROOT*> (mui_->createQTest(ContentsProf2DWithinRangeROOT::getAlgoName(), qtname));

      sprintf(qtname, "EBLT laser amplitude quality PNs SM%02d L4 G16", ism);
      qth20_[ism-1] = dynamic_cast<MEContentsProf2DWithinRangeROOT*> (mui_->createQTest(ContentsProf2DWithinRangeROOT::getAlgoName(), qtname));

      sprintf(qtname, "EBLT laser pedestal quality PNs SM%02d L1 G16", ism);
      qth21_[ism-1] = dynamic_cast<MEContentsProf2DWithinRangeROOT*> (mui_->createQTest(ContentsProf2DWithinRangeROOT::getAlgoName(), qtname));

      sprintf(qtname, "EBLT laser pedestal quality PNs SM%02d L2 G16", ism);
      qth22_[ism-1] = dynamic_cast<MEContentsProf2DWithinRangeROOT*> (mui_->createQTest(ContentsProf2DWithinRangeROOT::getAlgoName(), qtname));

      sprintf(qtname, "EBLT laser pedestal quality PNs SM%02d L3 G16", ism);
      qth23_[ism-1] = dynamic_cast<MEContentsProf2DWithinRangeROOT*> (mui_->createQTest(ContentsProf2DWithinRangeROOT::getAlgoName(), qtname));

      sprintf(qtname, "EBLT laser pedestal quality PNs SM%02d L4 G16", ism);
      qth24_[ism-1] = dynamic_cast<MEContentsProf2DWithinRangeROOT*> (mui_->createQTest(ContentsProf2DWithinRangeROOT::getAlgoName(), qtname));

      qth01_[ism-1]->setMeanRange(100.0, 4096.0*12.);
      qth02_[ism-1]->setMeanRange(100.0, 4096.0*12.);
      qth03_[ism-1]->setMeanRange(100.0, 4096.0*12.);
      qth04_[ism-1]->setMeanRange(100.0, 4096.0*12.);
      qth05_[ism-1]->setMeanRange(100.0, 4096.0*12.);
      qth06_[ism-1]->setMeanRange(100.0, 4096.0*12.);
      qth07_[ism-1]->setMeanRange(100.0, 4096.0*12.);
      qth08_[ism-1]->setMeanRange(100.0, 4096.0*12.);

      qth09_[ism-1]->setMeanRange(amplitudeThresholdPN_, 4096.0);
      qth10_[ism-1]->setMeanRange(amplitudeThresholdPN_, 4096.0);
      qth11_[ism-1]->setMeanRange(amplitudeThresholdPN_, 4096.0);
      qth12_[ism-1]->setMeanRange(amplitudeThresholdPN_, 4096.0);
      qth13_[ism-1]->setMeanRange(meanThresholdPN_, 4096.0);
      qth14_[ism-1]->setMeanRange(meanThresholdPN_, 4096.0);
      qth15_[ism-1]->setMeanRange(meanThresholdPN_, 4096.0);
      qth16_[ism-1]->setMeanRange(meanThresholdPN_, 4096.0);
      qth17_[ism-1]->setMeanRange(amplitudeThresholdPN_, 4096.0);
      qth18_[ism-1]->setMeanRange(amplitudeThresholdPN_, 4096.0);
      qth19_[ism-1]->setMeanRange(amplitudeThresholdPN_, 4096.0);
      qth20_[ism-1]->setMeanRange(amplitudeThresholdPN_, 4096.0);
      qth21_[ism-1]->setMeanRange(meanThresholdPN_, 4096.0);
      qth22_[ism-1]->setMeanRange(meanThresholdPN_, 4096.0);
      qth23_[ism-1]->setMeanRange(meanThresholdPN_, 4096.0);
      qth24_[ism-1]->setMeanRange(meanThresholdPN_, 4096.0);

      qth01_[ism-1]->setMeanTolerance(percentVariation_);
      qth02_[ism-1]->setMeanTolerance(percentVariation_);
      qth03_[ism-1]->setMeanTolerance(percentVariation_);
      qth04_[ism-1]->setMeanTolerance(percentVariation_);
      qth05_[ism-1]->setMeanTolerance(percentVariation_);
      qth06_[ism-1]->setMeanTolerance(percentVariation_);
      qth07_[ism-1]->setMeanTolerance(percentVariation_);
      qth08_[ism-1]->setMeanTolerance(percentVariation_);

      qth09_[ism-1]->setRMSRange(0.0, 4096.0);
      qth10_[ism-1]->setRMSRange(0.0, 4096.0);
      qth11_[ism-1]->setRMSRange(0.0, 4096.0);
      qth12_[ism-1]->setRMSRange(0.0, 4096.0);
      qth13_[ism-1]->setRMSRange(0.0, 4096.0);
      qth14_[ism-1]->setRMSRange(0.0, 4096.0);
      qth15_[ism-1]->setRMSRange(0.0, 4096.0);
      qth16_[ism-1]->setRMSRange(0.0, 4096.0);
      qth17_[ism-1]->setRMSRange(0.0, 4096.0);
      qth18_[ism-1]->setRMSRange(0.0, 4096.0);
      qth19_[ism-1]->setRMSRange(0.0, 4096.0);
      qth20_[ism-1]->setRMSRange(0.0, 4096.0);
      qth21_[ism-1]->setRMSRange(0.0, 4096.0);
      qth22_[ism-1]->setRMSRange(0.0, 4096.0);
      qth23_[ism-1]->setRMSRange(0.0, 4096.0);
      qth24_[ism-1]->setRMSRange(0.0, 4096.0);

      qth01_[ism-1]->setMinimumEntries(10*1700);
      qth02_[ism-1]->setMinimumEntries(10*1700);
      qth03_[ism-1]->setMinimumEntries(10*1700);
      qth04_[ism-1]->setMinimumEntries(10*1700);
      qth05_[ism-1]->setMinimumEntries(10*1700);
      qth06_[ism-1]->setMinimumEntries(10*1700);
      qth07_[ism-1]->setMinimumEntries(10*1700);
      qth08_[ism-1]->setMinimumEntries(10*1700);

      qth09_[ism-1]->setMinimumEntries(10*10);
      qth10_[ism-1]->setMinimumEntries(10*10);
      qth11_[ism-1]->setMinimumEntries(10*10);
      qth12_[ism-1]->setMinimumEntries(10*10);
      qth13_[ism-1]->setMinimumEntries(10*10);
      qth14_[ism-1]->setMinimumEntries(10*10);
      qth15_[ism-1]->setMinimumEntries(10*10);
      qth16_[ism-1]->setMinimumEntries(10*10);
      qth17_[ism-1]->setMinimumEntries(10*10);
      qth18_[ism-1]->setMinimumEntries(10*10);
      qth19_[ism-1]->setMinimumEntries(10*10);
      qth20_[ism-1]->setMinimumEntries(10*10);
      qth21_[ism-1]->setMinimumEntries(10*10);
      qth22_[ism-1]->setMinimumEntries(10*10);
      qth23_[ism-1]->setMinimumEntries(10*10);
      qth24_[ism-1]->setMinimumEntries(10*10);

      qth01_[ism-1]->setErrorProb(1.00);
      qth02_[ism-1]->setErrorProb(1.00);
      qth03_[ism-1]->setErrorProb(1.00);
      qth04_[ism-1]->setErrorProb(1.00);
      qth05_[ism-1]->setErrorProb(1.00);
      qth06_[ism-1]->setErrorProb(1.00);
      qth07_[ism-1]->setErrorProb(1.00);
      qth08_[ism-1]->setErrorProb(1.00);

      qth09_[ism-1]->setErrorProb(1.00);
      qth10_[ism-1]->setErrorProb(1.00);
      qth11_[ism-1]->setErrorProb(1.00);
      qth12_[ism-1]->setErrorProb(1.00);
      qth13_[ism-1]->setErrorProb(1.00);
      qth14_[ism-1]->setErrorProb(1.00);
      qth15_[ism-1]->setErrorProb(1.00);
      qth16_[ism-1]->setErrorProb(1.00);
      qth17_[ism-1]->setErrorProb(1.00);
      qth18_[ism-1]->setErrorProb(1.00);
      qth19_[ism-1]->setErrorProb(1.00);
      qth20_[ism-1]->setErrorProb(1.00);
      qth21_[ism-1]->setErrorProb(1.00);
      qth22_[ism-1]->setErrorProb(1.00);
      qth23_[ism-1]->setErrorProb(1.00);
      qth24_[ism-1]->setErrorProb(1.00);

    }

  }

}

void EBLaserClient::beginRun(void){

  if ( verbose_ ) cout << "EBLaserClient: beginRun" << endl;

  jevt_ = 0;

  this->setup();

  this->subscribe();

}

void EBLaserClient::endJob(void) {

  if ( verbose_ ) cout << "EBLaserClient: endJob, ievt = " << ievt_ << endl;

  this->unsubscribe();

  this->cleanup();

}

void EBLaserClient::endRun(void) {

  if ( verbose_ ) cout << "EBLaserClient: endRun, jevt = " << jevt_ << endl;

  this->unsubscribe();

  this->cleanup();

}

void EBLaserClient::setup(void) {

  Char_t histo[200];

  mui_->setCurrentFolder( "EcalBarrel/EBLaserClient" );
  
  DaqMonitorBEInterface* bei = mui_->getBEInterface();

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( meg01_[ism-1] ) bei->removeElement( meg01_[ism-1]->getName() );
    sprintf(histo, "EBLT laser quality L1 SM%02d", ism);
    meg01_[ism-1] = bei->book2D(histo, histo, 85, 0., 85., 20, 0., 20.);
    if ( meg02_[ism-1] ) bei->removeElement( meg02_[ism-1]->getName() );
    sprintf(histo, "EBLT laser quality L2 SM%02d", ism);
    meg02_[ism-1] = bei->book2D(histo, histo, 85, 0., 85., 20, 0., 20.);
    if ( meg03_[ism-1] ) bei->removeElement( meg03_[ism-1]->getName() );
    sprintf(histo, "EBLT laser quality L3 SM%02d", ism);
    meg03_[ism-1] = bei->book2D(histo, histo, 85, 0., 85., 20, 0., 20.);
    if ( meg04_[ism-1] ) bei->removeElement( meg04_[ism-1]->getName() );
    sprintf(histo, "EBLT laser quality L4 SM%02d", ism);
    meg04_[ism-1] = bei->book2D(histo, histo, 85, 0., 85., 20, 0., 20.);

    if ( meg05_[ism-1] ) bei->removeElement( meg05_[ism-1]->getName() );
    sprintf(histo, "EBLT laser quality L1 PNs SM%02d G01", ism);
    meg05_[ism-1] = bei->book2D(histo, histo, 10, 0., 10., 1, 0., 5.);
    if ( meg06_[ism-1] ) bei->removeElement( meg06_[ism-1]->getName() );
    sprintf(histo, "EBLT laser quality L2 PNs SM%02d G01", ism);
    meg06_[ism-1] = bei->book2D(histo, histo, 10, 0., 10., 1, 0., 5.);
    if ( meg07_[ism-1] ) bei->removeElement( meg07_[ism-1]->getName() );
    sprintf(histo, "EBLT laser quality L3 PNs SM%02d G01", ism);
    meg07_[ism-1] = bei->book2D(histo, histo, 10, 0., 10., 1, 0., 5.);
    if ( meg08_[ism-1] ) bei->removeElement( meg08_[ism-1]->getName() );
    sprintf(histo, "EBLT laser quality L4 PNs SM%02d G01", ism);
    meg08_[ism-1] = bei->book2D(histo, histo, 10, 0., 10., 1, 0., 5.);
    if ( meg09_[ism-1] ) bei->removeElement( meg09_[ism-1]->getName() );
    sprintf(histo, "EBLT laser quality L1 PNs SM%02d G16", ism);
    meg09_[ism-1] = bei->book2D(histo, histo, 10, 0., 10., 1, 0., 5.);
    if ( meg10_[ism-1] ) bei->removeElement( meg10_[ism-1]->getName() );
    sprintf(histo, "EBLT laser quality L2 PNs SM%02d G16", ism);
    meg10_[ism-1] = bei->book2D(histo, histo, 10, 0., 10., 1, 0., 5.);
    if ( meg11_[ism-1] ) bei->removeElement( meg11_[ism-1]->getName() );
    sprintf(histo, "EBLT laser quality L3 PNs SM%02d G16", ism);
    meg11_[ism-1] = bei->book2D(histo, histo, 10, 0., 10., 1, 0., 5.);
    if ( meg12_[ism-1] ) bei->removeElement( meg12_[ism-1]->getName() );
    sprintf(histo, "EBLT laser quality L4 PNs SM%02d G16", ism);
    meg12_[ism-1] = bei->book2D(histo, histo, 10, 0., 10., 1, 0., 5.);

    if ( mea01_[ism-1] ) bei->removeElement( mea01_[ism-1]->getName() );;
    sprintf(histo, "EBLT amplitude L1A SM%02d", ism);
    mea01_[ism-1] = bei->book1D(histo, histo, 1700, 0., 1700.);
    if ( mea02_[ism-1] ) bei->removeElement( mea02_[ism-1]->getName() );
    sprintf(histo, "EBLT amplitude L2A SM%02d", ism);
    mea02_[ism-1] = bei->book1D(histo, histo, 1700, 0., 1700.);
    if ( mea03_[ism-1] ) bei->removeElement( mea03_[ism-1]->getName() );
    sprintf(histo, "EBLT amplitude L3A SM%02d", ism);
    mea03_[ism-1] = bei->book1D(histo, histo, 1700, 0., 1700.);
    if ( mea04_[ism-1] ) bei->removeElement( mea04_[ism-1]->getName() );
    sprintf(histo, "EBLT amplitude L4A SM%02d", ism);
    mea04_[ism-1] = bei->book1D(histo, histo, 1700, 0., 1700.);
    if ( mea05_[ism-1] ) bei->removeElement( mea05_[ism-1]->getName() );;
    sprintf(histo, "EBLT amplitude L1B SM%02d", ism);
    mea05_[ism-1] = bei->book1D(histo, histo, 1700, 0., 1700.);
    if ( mea06_[ism-1] ) bei->removeElement( mea06_[ism-1]->getName() );
    sprintf(histo, "EBLT amplitude L2B SM%02d", ism);
    mea06_[ism-1] = bei->book1D(histo, histo, 1700, 0., 1700.);
    if ( mea07_[ism-1] ) bei->removeElement( mea07_[ism-1]->getName() );
    sprintf(histo, "EBLT amplitude L3B SM%02d", ism);
    mea07_[ism-1] = bei->book1D(histo, histo, 1700, 0., 1700.);
    if ( mea08_[ism-1] ) bei->removeElement( mea08_[ism-1]->getName() );
    sprintf(histo, "EBLT amplitude L4B SM%02d", ism);
    mea08_[ism-1] = bei->book1D(histo, histo, 1700, 0., 1700.);

    if ( met01_[ism-1] ) bei->removeElement( met01_[ism-1]->getName() );
    sprintf(histo, "EBLT timing L1A SM%02d", ism);
    met01_[ism-1] = bei->book1D(histo, histo, 1700, 0., 1700.);
    if ( met02_[ism-1] ) bei->removeElement( met02_[ism-1]->getName() );
    sprintf(histo, "EBLT timing L2A SM%02d", ism);
    met02_[ism-1] = bei->book1D(histo, histo, 1700, 0., 1700.);
    if ( met03_[ism-1] ) bei->removeElement( met03_[ism-1]->getName() );
    sprintf(histo, "EBLT timing L3A SM%02d", ism);
    met03_[ism-1] = bei->book1D(histo, histo, 1700, 0., 1700.);
    if ( met04_[ism-1] ) bei->removeElement( met04_[ism-1]->getName() );
    sprintf(histo, "EBLT timing L4A SM%02d", ism);
    met04_[ism-1] = bei->book1D(histo, histo, 1700, 0., 1700.);
    if ( met05_[ism-1] ) bei->removeElement( met05_[ism-1]->getName() );
    sprintf(histo, "EBLT timing L1B SM%02d", ism);
    met05_[ism-1] = bei->book1D(histo, histo, 1700, 0., 1700.);
    if ( met06_[ism-1] ) bei->removeElement( met06_[ism-1]->getName() );
    sprintf(histo, "EBLT timing L2B SM%02d", ism);
    met06_[ism-1] = bei->book1D(histo, histo, 1700, 0., 1700.);
    if ( met07_[ism-1] ) bei->removeElement( met07_[ism-1]->getName() );
    sprintf(histo, "EBLT timing L3B SM%02d", ism);
    met07_[ism-1] = bei->book1D(histo, histo, 1700, 0., 1700.);
    if ( met08_[ism-1] ) bei->removeElement( met08_[ism-1]->getName() );
    sprintf(histo, "EBLT timing L4B SM%02d", ism);
    met08_[ism-1] = bei->book1D(histo, histo, 1700, 0., 1700.);

    if ( meaopn01_[ism-1] ) bei->removeElement( meaopn01_[ism-1]->getName() );
    sprintf(histo, "EBLT amplitude over PN L1A SM%02d", ism);
    meaopn01_[ism-1] = bei->book1D(histo, histo, 1700, 0., 1700.);
    if ( meaopn02_[ism-1] ) bei->removeElement( meaopn02_[ism-1]->getName() );
    sprintf(histo, "EBLT amplitude over PN L2A SM%02d", ism);
    meaopn02_[ism-1] = bei->book1D(histo, histo, 1700, 0., 1700.);
    if ( meaopn03_[ism-1] ) bei->removeElement( meaopn03_[ism-1]->getName() );
    sprintf(histo, "EBLT amplitude over PN L3A SM%02d", ism);
    meaopn03_[ism-1] = bei->book1D(histo, histo, 1700, 0., 1700.);
    if ( meaopn04_[ism-1] ) bei->removeElement( meaopn04_[ism-1]->getName() );
    sprintf(histo, "EBLT amplitude over PN L4A SM%02d", ism);
    meaopn04_[ism-1] = bei->book1D(histo, histo, 1700, 0., 1700.);
    if ( meaopn05_[ism-1] ) bei->removeElement( meaopn05_[ism-1]->getName() );
    sprintf(histo, "EBLT amplitude over PN L1B SM%02d", ism);
    meaopn05_[ism-1] = bei->book1D(histo, histo, 1700, 0., 1700.);
    if ( meaopn06_[ism-1] ) bei->removeElement( meaopn06_[ism-1]->getName() );
    sprintf(histo, "EBLT amplitude over PN L2B SM%02d", ism);
    meaopn06_[ism-1] = bei->book1D(histo, histo, 1700, 0., 1700.);
    if ( meaopn07_[ism-1] ) bei->removeElement( meaopn07_[ism-1]->getName() );
    sprintf(histo, "EBLT amplitude over PN L3B SM%02d", ism);
    meaopn07_[ism-1] = bei->book1D(histo, histo, 1700, 0., 1700.);
    if ( meaopn08_[ism-1] ) bei->removeElement( meaopn08_[ism-1]->getName() );
    sprintf(histo, "EBLT amplitude over PN L4B SM%02d", ism);
    meaopn08_[ism-1] = bei->book1D(histo, histo, 1700, 0., 1700.);

  }

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    EBMUtilsClient::resetHisto( meg01_[ism-1] );
    EBMUtilsClient::resetHisto( meg02_[ism-1] );
    EBMUtilsClient::resetHisto( meg03_[ism-1] );
    EBMUtilsClient::resetHisto( meg04_[ism-1] );

    EBMUtilsClient::resetHisto( meg05_[ism-1] );
    EBMUtilsClient::resetHisto( meg06_[ism-1] );
    EBMUtilsClient::resetHisto( meg07_[ism-1] );
    EBMUtilsClient::resetHisto( meg08_[ism-1] );
    EBMUtilsClient::resetHisto( meg09_[ism-1] );
    EBMUtilsClient::resetHisto( meg10_[ism-1] );
    EBMUtilsClient::resetHisto( meg11_[ism-1] );
    EBMUtilsClient::resetHisto( meg12_[ism-1] );

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

        meg01_[ism-1]->setBinContent( ie, ip, 2. );
        meg02_[ism-1]->setBinContent( ie, ip, 2. );
        meg03_[ism-1]->setBinContent( ie, ip, 2. );
        meg04_[ism-1]->setBinContent( ie, ip, 2. );

      }
    }

    for ( int i = 1; i <= 10; i++ ) {

        meg05_[ism-1]->setBinContent( i, 1, 2. );
        meg06_[ism-1]->setBinContent( i, 1, 2. );
        meg07_[ism-1]->setBinContent( i, 1, 2. );
        meg08_[ism-1]->setBinContent( i, 1, 2. );
        meg09_[ism-1]->setBinContent( i, 1, 2. );
        meg10_[ism-1]->setBinContent( i, 1, 2. );
        meg11_[ism-1]->setBinContent( i, 1, 2. );
        meg12_[ism-1]->setBinContent( i, 1, 2. );

    }
 
    EBMUtilsClient::resetHisto( mea01_[ism-1] );
    EBMUtilsClient::resetHisto( mea02_[ism-1] );
    EBMUtilsClient::resetHisto( mea03_[ism-1] );
    EBMUtilsClient::resetHisto( mea04_[ism-1] );
    EBMUtilsClient::resetHisto( mea05_[ism-1] );
    EBMUtilsClient::resetHisto( mea06_[ism-1] );
    EBMUtilsClient::resetHisto( mea07_[ism-1] );
    EBMUtilsClient::resetHisto( mea08_[ism-1] );

    EBMUtilsClient::resetHisto( met01_[ism-1] );
    EBMUtilsClient::resetHisto( met02_[ism-1] );
    EBMUtilsClient::resetHisto( met03_[ism-1] );
    EBMUtilsClient::resetHisto( met04_[ism-1] );
    EBMUtilsClient::resetHisto( met05_[ism-1] );
    EBMUtilsClient::resetHisto( met06_[ism-1] );
    EBMUtilsClient::resetHisto( met07_[ism-1] );
    EBMUtilsClient::resetHisto( met08_[ism-1] );

    EBMUtilsClient::resetHisto( meaopn01_[ism-1] );
    EBMUtilsClient::resetHisto( meaopn02_[ism-1] );
    EBMUtilsClient::resetHisto( meaopn03_[ism-1] );
    EBMUtilsClient::resetHisto( meaopn04_[ism-1] );
    EBMUtilsClient::resetHisto( meaopn05_[ism-1] );
    EBMUtilsClient::resetHisto( meaopn06_[ism-1] );
    EBMUtilsClient::resetHisto( meaopn07_[ism-1] );
    EBMUtilsClient::resetHisto( meaopn08_[ism-1] );

  }

}

void EBLaserClient::cleanup(void) {

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

    mui_->setCurrentFolder( "EcalBarrel/EBLaserClient" );
    DaqMonitorBEInterface* bei = mui_->getBEInterface();

    if ( meg01_[ism-1] ) bei->removeElement( meg01_[ism-1]->getName() );
    meg01_[ism-1] = 0;
    if ( meg02_[ism-1] ) bei->removeElement( meg02_[ism-1]->getName() );
    meg02_[ism-1] = 0;
    if ( meg03_[ism-1] ) bei->removeElement( meg03_[ism-1]->getName() );
    meg03_[ism-1] = 0;
    if ( meg04_[ism-1] ) bei->removeElement( meg04_[ism-1]->getName() );
    meg04_[ism-1] = 0;

    if ( meg05_[ism-1] ) bei->removeElement( meg05_[ism-1]->getName() );
    meg05_[ism-1] = 0;
    if ( meg06_[ism-1] ) bei->removeElement( meg06_[ism-1]->getName() );
    meg06_[ism-1] = 0;
    if ( meg07_[ism-1] ) bei->removeElement( meg07_[ism-1]->getName() );
    meg07_[ism-1] = 0;
    if ( meg08_[ism-1] ) bei->removeElement( meg08_[ism-1]->getName() );
    meg08_[ism-1] = 0;
    if ( meg09_[ism-1] ) bei->removeElement( meg09_[ism-1]->getName() );
    meg09_[ism-1] = 0;
    if ( meg10_[ism-1] ) bei->removeElement( meg10_[ism-1]->getName() );
    meg10_[ism-1] = 0;
    if ( meg11_[ism-1] ) bei->removeElement( meg11_[ism-1]->getName() );
    meg11_[ism-1] = 0;
    if ( meg12_[ism-1] ) bei->removeElement( meg12_[ism-1]->getName() );
    meg12_[ism-1] = 0;

    if ( mea01_[ism-1] ) bei->removeElement( mea01_[ism-1]->getName() );
    mea01_[ism-1] = 0;
    if ( mea02_[ism-1] ) bei->removeElement( mea02_[ism-1]->getName() );
    mea02_[ism-1] = 0;
    if ( mea03_[ism-1] ) bei->removeElement( mea03_[ism-1]->getName() );
    mea03_[ism-1] = 0;
    if ( mea04_[ism-1] ) bei->removeElement( mea04_[ism-1]->getName() );
    mea04_[ism-1] = 0;
    if ( mea05_[ism-1] ) bei->removeElement( mea05_[ism-1]->getName() );
    mea05_[ism-1] = 0;
    if ( mea06_[ism-1] ) bei->removeElement( mea06_[ism-1]->getName() );
    mea06_[ism-1] = 0; 
    if ( mea07_[ism-1] ) bei->removeElement( mea07_[ism-1]->getName() );
    mea07_[ism-1] = 0; 
    if ( mea08_[ism-1] ) bei->removeElement( mea08_[ism-1]->getName() );
    mea08_[ism-1] = 0;

    if ( met01_[ism-1] ) bei->removeElement( met01_[ism-1]->getName() );
    met01_[ism-1] = 0;
    if ( met02_[ism-1] ) bei->removeElement( met02_[ism-1]->getName() );
    met02_[ism-1] = 0;
    if ( met03_[ism-1] ) bei->removeElement( met03_[ism-1]->getName() );
    met03_[ism-1] = 0;
    if ( met04_[ism-1] ) bei->removeElement( met04_[ism-1]->getName() );
    met04_[ism-1] = 0;
    if ( met05_[ism-1] ) bei->removeElement( met05_[ism-1]->getName() );
    met05_[ism-1] = 0;
    if ( met06_[ism-1] ) bei->removeElement( met06_[ism-1]->getName() );
    met06_[ism-1] = 0;
    if ( met07_[ism-1] ) bei->removeElement( met07_[ism-1]->getName() );
    met07_[ism-1] = 0;
    if ( met08_[ism-1] ) bei->removeElement( met08_[ism-1]->getName() );
    met08_[ism-1] = 0;

    if ( meaopn01_[ism-1] ) bei->removeElement( meaopn01_[ism-1]->getName() );
    meaopn01_[ism-1] = 0;
    if ( meaopn02_[ism-1] ) bei->removeElement( meaopn02_[ism-1]->getName() );
    meaopn02_[ism-1] = 0;
    if ( meaopn03_[ism-1] ) bei->removeElement( meaopn03_[ism-1]->getName() );
    meaopn03_[ism-1] = 0;
    if ( meaopn04_[ism-1] ) bei->removeElement( meaopn04_[ism-1]->getName() );
    meaopn04_[ism-1] = 0;
    if ( meaopn05_[ism-1] ) bei->removeElement( meaopn05_[ism-1]->getName() );
    meaopn05_[ism-1] = 0;
    if ( meaopn06_[ism-1] ) bei->removeElement( meaopn06_[ism-1]->getName() );
    meaopn06_[ism-1] = 0;
    if ( meaopn07_[ism-1] ) bei->removeElement( meaopn07_[ism-1]->getName() );
    meaopn07_[ism-1] = 0;
    if ( meaopn08_[ism-1] ) bei->removeElement( meaopn08_[ism-1]->getName() );
    meaopn08_[ism-1] = 0;

  }

}

bool EBLaserClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov, int ism) {

  bool status = true;

  EBMUtilsClient::printBadChannels(qth01_[ism-1]);
  EBMUtilsClient::printBadChannels(qth05_[ism-1]);
  EBMUtilsClient::printBadChannels(qth02_[ism-1]);
  EBMUtilsClient::printBadChannels(qth06_[ism-1]);
  EBMUtilsClient::printBadChannels(qth03_[ism-1]);
  EBMUtilsClient::printBadChannels(qth07_[ism-1]);
  EBMUtilsClient::printBadChannels(qth04_[ism-1]);
  EBMUtilsClient::printBadChannels(qth08_[ism-1]);

  EBMUtilsClient::printBadChannels(qth09_[ism-1]);
  EBMUtilsClient::printBadChannels(qth10_[ism-1]);
  EBMUtilsClient::printBadChannels(qth11_[ism-1]);
  EBMUtilsClient::printBadChannels(qth12_[ism-1]);
  EBMUtilsClient::printBadChannels(qth13_[ism-1]);
  EBMUtilsClient::printBadChannels(qth14_[ism-1]);
  EBMUtilsClient::printBadChannels(qth15_[ism-1]);
  EBMUtilsClient::printBadChannels(qth16_[ism-1]);
  EBMUtilsClient::printBadChannels(qth17_[ism-1]);
  EBMUtilsClient::printBadChannels(qth18_[ism-1]);
  EBMUtilsClient::printBadChannels(qth19_[ism-1]);
  EBMUtilsClient::printBadChannels(qth20_[ism-1]);
  EBMUtilsClient::printBadChannels(qth21_[ism-1]);
  EBMUtilsClient::printBadChannels(qth22_[ism-1]);
  EBMUtilsClient::printBadChannels(qth23_[ism-1]);
  EBMUtilsClient::printBadChannels(qth24_[ism-1]);

  EcalLogicID ecid;
  MonLaserBlueDat apd_bl;
  map<EcalLogicID, MonLaserBlueDat> dataset1_bl;
  MonLaserGreenDat apd_gr;
  map<EcalLogicID, MonLaserGreenDat> dataset1_gr;
  MonLaserIRedDat apd_ir;
  map<EcalLogicID, MonLaserIRedDat> dataset1_ir;
  MonLaserRedDat apd_rd;
  map<EcalLogicID, MonLaserRedDat> dataset1_rd;

  for ( int ie = 1; ie <= 85; ie++ ) {
    for ( int ip = 1; ip <= 20; ip++ ) {

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

      update01 = EBMUtilsClient::getBinStats(h01_[ism-1], ie, ip, num01, mean01, rms01);
      update02 = EBMUtilsClient::getBinStats(h02_[ism-1], ie, ip, num02, mean02, rms02);
      update03 = EBMUtilsClient::getBinStats(h03_[ism-1], ie, ip, num03, mean03, rms03);
      update04 = EBMUtilsClient::getBinStats(h04_[ism-1], ie, ip, num04, mean04, rms04);
      update05 = EBMUtilsClient::getBinStats(h05_[ism-1], ie, ip, num05, mean05, rms05);
      update06 = EBMUtilsClient::getBinStats(h06_[ism-1], ie, ip, num06, mean06, rms06);
      update07 = EBMUtilsClient::getBinStats(h07_[ism-1], ie, ip, num07, mean07, rms07);
      update08 = EBMUtilsClient::getBinStats(h08_[ism-1], ie, ip, num08, mean08, rms08);

      if ( ! update01 )
        update01 = EBMUtilsClient::getBinStats(h13_[ism-1], ie, ip, num01, mean01, rms01);
      if ( ! update02 )
        update02 = EBMUtilsClient::getBinStats(h14_[ism-1], ie, ip, num02, mean02, rms02);
      if ( ! update03 )
        update03 = EBMUtilsClient::getBinStats(h15_[ism-1], ie, ip, num03, mean03, rms03);
      if ( ! update04 )
        update04 = EBMUtilsClient::getBinStats(h16_[ism-1], ie, ip, num04, mean04, rms04);
      if ( ! update05 )
        update05 = EBMUtilsClient::getBinStats(h17_[ism-1], ie, ip, num05, mean05, rms05);
      if ( ! update06 )
        update06 = EBMUtilsClient::getBinStats(h18_[ism-1], ie, ip, num06, mean06, rms06);
      if ( ! update07 )
        update07 = EBMUtilsClient::getBinStats(h19_[ism-1], ie, ip, num07, mean07, rms07);
      if ( ! update08 )
        update08 = EBMUtilsClient::getBinStats(h20_[ism-1], ie, ip, num08, mean08, rms08);

      if ( update01 || update02 ) {

        if ( ie == 1 && ip == 1 ) {

          cout << "Preparing dataset for SM=" << ism << endl;

          cout << "L1 (" << ie << "," << ip << ") " << num01 << " " << mean01 << " " << rms01 << endl;

          cout << endl;

        }

        apd_bl.setAPDMean(mean01);
        apd_bl.setAPDRMS(rms01);

        apd_bl.setAPDOverPNMean(mean02);
        apd_bl.setAPDOverPNRMS(rms02);

        if ( meg01_[ism-1] && int(meg01_[ism-1]->getBinContent( ie, ip )) % 3 == 1. ) {
          apd_bl.setTaskStatus(true);
        } else {
          apd_bl.setTaskStatus(false);
        }

        status = status && EBMUtilsClient::getBinQual(meg01_[ism-1], ie, ip);

        int ic = (ip-1) + 20*(ie-1) + 1;

        if ( econn ) {
          try {
            ecid = econn->getEcalLogicID("EB_crystal_number", ism, ic);
            dataset1_bl[ecid] = apd_bl;
          } catch (runtime_error &e) {
            cerr << e.what() << endl;
          }
        }

      }

      if ( update03 || update04 ) {

        if ( ie == 1 && ip == 1 ) {

          cout << "Preparing dataset for SM=" << ism << endl;

          cout << "L2 (" << ie << "," << ip << ") " << num03 << " " << mean03 << " " << rms03 << endl;

          cout << endl;

        }

        apd_ir.setAPDMean(mean03);
        apd_ir.setAPDRMS(rms03);

        apd_ir.setAPDOverPNMean(mean04);
        apd_ir.setAPDOverPNRMS(rms04);

        if ( meg02_[ism-1] && int(meg02_[ism-1]->getBinContent( ie, ip )) % 3 == 1. ) {
          apd_ir.setTaskStatus(true);
        } else {
          apd_ir.setTaskStatus(false);
        }

        status = status && EBMUtilsClient::getBinQual(meg02_[ism-1], ie, ip);

        int ic = (ip-1) + 20*(ie-1) + 1;

        if ( econn ) {
          try {
            ecid = econn->getEcalLogicID("EB_crystal_number", ism, ic);
            dataset1_ir[ecid] = apd_ir;
          } catch (runtime_error &e) {
            cerr << e.what() << endl;
          }
        }

      }

      if ( update05 || update06 ) {

        if ( ie == 1 && ip == 1 ) {

          cout << "Preparing dataset for SM=" << ism << endl;

          cout << "L3 (" << ie << "," << ip << ") " << num05 << " " << mean05 << " " << rms05 << endl;

          cout << endl;

        }

        apd_gr.setAPDMean(mean05);
        apd_gr.setAPDRMS(rms05);

        apd_gr.setAPDOverPNMean(mean06);
        apd_gr.setAPDOverPNRMS(rms06);

        if ( meg03_[ism-1] && int(meg03_[ism-1]->getBinContent( ie, ip )) % 3 == 1. ) {
          apd_gr.setTaskStatus(true);
        } else {
          apd_gr.setTaskStatus(false);
        }

        status = status && EBMUtilsClient::getBinQual(meg03_[ism-1], ie, ip);

        int ic = (ip-1) + 20*(ie-1) + 1;

        if ( econn ) {
          try {
            ecid = econn->getEcalLogicID("EB_crystal_number", ism, ic);
            dataset1_gr[ecid] = apd_gr;
          } catch (runtime_error &e) {
            cerr << e.what() << endl;
          }
        }

      }

      if ( update07 || update08 ) {

        if ( ie == 1 && ip == 1 ) {

          cout << "Preparing dataset for SM=" << ism << endl;

          cout << "L4 (" << ie << "," << ip << ") " << num07 << " " << mean07 << " " << rms07 << endl;

          cout << endl;

        }

        apd_rd.setAPDMean(mean07);
        apd_rd.setAPDRMS(rms07);

        apd_rd.setAPDOverPNMean(mean08);
        apd_rd.setAPDOverPNRMS(rms08);

        if ( meg04_[ism-1] && int(meg04_[ism-1]->getBinContent( ie, ip )) % 3 == 1. ) {
          apd_rd.setTaskStatus(true);
        } else {
          apd_rd.setTaskStatus(false);
        }

        status = status && EBMUtilsClient::getBinQual(meg04_[ism-1], ie, ip);

        int ic = (ip-1) + 20*(ie-1) + 1;

        if ( econn ) {
          try {
            ecid = econn->getEcalLogicID("EB_crystal_number", ism, ic);
            dataset1_rd[ecid] = apd_rd;
          } catch (runtime_error &e) {
            cerr << e.what() << endl;
          }
        }

      }

    }
  }

  if ( econn ) {
    try {
      cout << "Inserting MonLaserDat ... " << flush;
      if ( dataset1_bl.size() != 0 ) econn->insertDataSet(&dataset1_bl, moniov);
      if ( dataset1_ir.size() != 0 ) econn->insertDataSet(&dataset1_ir, moniov);
      if ( dataset1_gr.size() != 0 ) econn->insertDataSet(&dataset1_gr, moniov);
      if ( dataset1_rd.size() != 0 ) econn->insertDataSet(&dataset1_rd, moniov);
      cout << "done." << endl;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

  MonPNBlueDat pn_bl;
  map<EcalLogicID, MonPNBlueDat> dataset2_bl;
  MonPNGreenDat pn_gr;
  map<EcalLogicID, MonPNGreenDat> dataset2_gr;
  MonPNIRedDat pn_ir;
  map<EcalLogicID, MonPNIRedDat> dataset2_ir;
  MonPNRedDat pn_rd;
  map<EcalLogicID, MonPNRedDat> dataset2_rd;

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

    update01 = EBMUtilsClient::getBinStats(i01_[ism-1], 1, i, num01, mean01, rms01);
    update02 = EBMUtilsClient::getBinStats(i02_[ism-1], 1, i, num02, mean02, rms02);
    update03 = EBMUtilsClient::getBinStats(i03_[ism-1], 1, i, num03, mean03, rms03);
    update04 = EBMUtilsClient::getBinStats(i04_[ism-1], 1, i, num04, mean04, rms04);
    update05 = EBMUtilsClient::getBinStats(i05_[ism-1], 1, i, num05, mean05, rms05);
    update06 = EBMUtilsClient::getBinStats(i06_[ism-1], 1, i, num06, mean06, rms06);
    update07 = EBMUtilsClient::getBinStats(i07_[ism-1], 1, i, num07, mean07, rms07);
    update08 = EBMUtilsClient::getBinStats(i08_[ism-1], 1, i, num08, mean08, rms08);
    update09 = EBMUtilsClient::getBinStats(i09_[ism-1], 1, i, num09, mean09, rms09);
    update10 = EBMUtilsClient::getBinStats(i10_[ism-1], 1, i, num10, mean10, rms10);
    update11 = EBMUtilsClient::getBinStats(i11_[ism-1], 1, i, num11, mean11, rms11);
    update12 = EBMUtilsClient::getBinStats(i12_[ism-1], 1, i, num12, mean12, rms12);
    update13 = EBMUtilsClient::getBinStats(i13_[ism-1], 1, i, num13, mean13, rms13);
    update14 = EBMUtilsClient::getBinStats(i14_[ism-1], 1, i, num14, mean14, rms14);
    update15 = EBMUtilsClient::getBinStats(i15_[ism-1], 1, i, num15, mean15, rms15);
    update16 = EBMUtilsClient::getBinStats(i16_[ism-1], 1, i, num16, mean16, rms16);

    if ( update01 || update05 || update09 || update13 ) {

      if ( i == 1 ) {

        cout << "Preparing dataset for SM=" << ism << endl;

        cout << "PNs (" << i << ") L1 G01 " << num01  << " " << mean01 << " " << rms01  << endl;
        cout << "PNs (" << i << ") L1 G16 " << num09  << " " << mean09 << " " << rms09  << endl;

        cout << endl;

      }

      pn_bl.setADCMeanG1(mean01);
      pn_bl.setADCRMSG1(rms01);

      pn_bl.setPedMeanG1(mean05);
      pn_bl.setPedRMSG1(rms05);

      pn_bl.setADCMeanG16(mean09);
      pn_bl.setADCRMSG16(rms09);

      pn_bl.setPedMeanG16(mean13);
      pn_bl.setPedRMSG16(rms13);

      if ( meg05_[ism-1] && int(meg05_[ism-1]->getBinContent( i, 1 )) % 3 == 1. ||
           meg09_[ism-1] && int(meg09_[ism-1]->getBinContent( i, 1 )) % 3 == 1. ) {
        pn_bl.setTaskStatus(true);
      } else {
        pn_bl.setTaskStatus(false);
      }

      status = status && ( EBMUtilsClient::getBinQual(meg05_[ism-1], i, 1) ||
                           EBMUtilsClient::getBinQual(meg09_[ism-1], i, 1) );

      if ( econn ) {
        try {
          ecid = econn->getEcalLogicID("EB_LM_PN", ism, i-1);
          dataset2_bl[ecid] = pn_bl;
        } catch (runtime_error &e) {
          cerr << e.what() << endl;
        }
      }

    }

    if ( update02 || update06 || update10 || update14 ) {

      if ( i == 1 ) {

        cout << "Preparing dataset for SM=" << ism << endl;

        cout << "PNs (" << i << ") L2 G01 " << num02  << " " << mean02 << " " << rms02  << endl;
        cout << "PNs (" << i << ") L2 G16 " << num10  << " " << mean10 << " " << rms10  << endl;

        cout << endl;

      }

      pn_ir.setADCMeanG1(mean02);
      pn_ir.setADCRMSG1(rms02);

      pn_ir.setPedMeanG1(mean06);
      pn_ir.setPedRMSG1(rms06);

      pn_ir.setADCMeanG16(mean10);
      pn_ir.setADCRMSG16(rms10);

      pn_ir.setPedMeanG16(mean14);
      pn_ir.setPedRMSG16(rms14);

      if ( meg06_[ism-1] && int(meg06_[ism-1]->getBinContent( i, 1 )) % 3 == 1. ||
           meg10_[ism-1] && int(meg10_[ism-1]->getBinContent( i, 1 )) % 3 == 1. ) {
        pn_ir.setTaskStatus(true);
      } else {
        pn_ir.setTaskStatus(false);
      }

      status = status && ( EBMUtilsClient::getBinQual(meg06_[ism-1], i, 1) ||
                           EBMUtilsClient::getBinQual(meg10_[ism-1], i, 1) );

      if ( econn ) {
        try {
          ecid = econn->getEcalLogicID("EB_LM_PN", ism, i-1);
          dataset2_ir[ecid] = pn_ir;
        } catch (runtime_error &e) {
          cerr << e.what() << endl;
        }
      }

    }

    if ( update03 || update07 || update11 || update15 ) {

      if ( i == 1 ) {

        cout << "Preparing dataset for SM=" << ism << endl;

        cout << "PNs (" << i << ") L3 G01 " << num03  << " " << mean03 << " " << rms03  << endl;
        cout << "PNs (" << i << ") L3 G16 " << num11  << " " << mean11 << " " << rms11  << endl;

        cout << endl;

      }

      pn_gr.setADCMeanG1(mean03);
      pn_gr.setADCRMSG1(rms03);

      pn_gr.setPedMeanG1(mean07);
      pn_gr.setPedRMSG1(rms07);

      pn_gr.setADCMeanG16(mean11);
      pn_gr.setADCRMSG16(rms11);

      pn_gr.setPedMeanG16(mean15);
      pn_gr.setPedRMSG16(rms15);

      if ( meg07_[ism-1] && int(meg07_[ism-1]->getBinContent( i, 1 )) % 3 == 1. ||
           meg11_[ism-1] && int(meg11_[ism-1]->getBinContent( i, 1 )) % 3 == 1. ) {
        pn_gr.setTaskStatus(true);
      } else {
        pn_gr.setTaskStatus(false);
      }

      status = status && ( EBMUtilsClient::getBinQual(meg07_[ism-1], i, 1) ||
                           EBMUtilsClient::getBinQual(meg11_[ism-1], i, 1) );

      if ( econn ) {
        try {
          ecid = econn->getEcalLogicID("EB_LM_PN", ism, i-1);
          dataset2_gr[ecid] = pn_gr;
        } catch (runtime_error &e) {
          cerr << e.what() << endl;
        }
      }

    }

    if ( update04 || update08 || update12 || update16 ) {

      if ( i == 1 ) {

        cout << "Preparing dataset for SM=" << ism << endl;

        cout << "PNs (" << i << ") L4 G01 " << num04  << " " << mean04 << " " << rms04  << endl;
        cout << "PNs (" << i << ") L4 G16 " << num12  << " " << mean12 << " " << rms12  << endl;

        cout << endl;

      }

      pn_rd.setADCMeanG1(mean04);
      pn_rd.setADCRMSG1(rms04);

      pn_rd.setPedMeanG1(mean08);
      pn_rd.setPedRMSG1(mean08);

      pn_rd.setADCMeanG16(mean12);
      pn_rd.setADCRMSG16(rms12);

      pn_rd.setPedMeanG16(mean16);
      pn_rd.setPedRMSG16(rms16);

      if ( meg08_[ism-1] && int(meg08_[ism-1]->getBinContent( i, 1 )) % 3 == 1. ||
           meg12_[ism-1] && int(meg12_[ism-1]->getBinContent( i, 1 )) % 3 == 1. ) {
        pn_rd.setTaskStatus(true);
      } else {
        pn_rd.setTaskStatus(false);
      }

      status = status && ( EBMUtilsClient::getBinQual(meg08_[ism-1], i, 1) ||
                           EBMUtilsClient::getBinQual(meg12_[ism-1], i, 1) );

      if ( econn ) {
        try {
          ecid = econn->getEcalLogicID("EB_LM_PN", ism, i-1);
          dataset2_rd[ecid] = pn_rd;
        } catch (runtime_error &e) {
          cerr << e.what() << endl;
        }
      }

    }

  }

  if ( econn ) {
    try {
      cout << "Inserting MonPnDat ... " << flush;
      if ( dataset2_bl.size() != 0 ) econn->insertDataSet(&dataset2_bl, moniov);
      if ( dataset2_ir.size() != 0 ) econn->insertDataSet(&dataset2_ir, moniov);
      if ( dataset2_gr.size() != 0 ) econn->insertDataSet(&dataset2_gr, moniov);
      if ( dataset2_rd.size() != 0 ) econn->insertDataSet(&dataset2_rd, moniov);
      cout << "done." << endl;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

  return status;

}

void EBLaserClient::subscribe(void){

  if ( verbose_ ) cout << "EBLaserClient: subscribe" << endl;

  Char_t histo[200];

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    unsigned int ism = superModules_[i];

    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser1/EBLT amplitude SM%02d L1A", ism);
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser1/EBLT timing SM%02d L1A", ism);
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser1/EBLT amplitude over PN SM%02d L1A", ism);
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser2/EBLT amplitude SM%02d L2A", ism);
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser2/EBLT timing SM%02d L2A", ism);
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser2/EBLT amplitude over PN SM%02d L2A", ism);
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser3/EBLT amplitude SM%02d L3A", ism);
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser3/EBLT timing SM%02d L3A", ism);
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser3/EBLT amplitude over PN SM%02d L3A", ism);
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser4/EBLT amplitude SM%02d L4A", ism);
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser4/EBLT timing SM%02d L4A", ism);
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser4/EBLT amplitude over PN SM%02d L4A", ism);
    mui_->subscribe(histo, ism);

    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser1/EBLT amplitude SM%02d L1B", ism);
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser1/EBLT timing SM%02d L1B", ism);
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser1/EBLT amplitude over PN SM%02d L1B", ism);
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser2/EBLT amplitude SM%02d L2B", ism);
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser2/EBLT timing SM%02d L2B", ism);
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser2/EBLT amplitude over PN SM%02d L2B", ism);
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser3/EBLT amplitude SM%02d L3B", ism);
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser3/EBLT timing SM%02d L3B", ism);
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser3/EBLT amplitude over PN SM%02d L3B", ism);
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser4/EBLT amplitude SM%02d L4B", ism);
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser4/EBLT timing SM%02d L4B", ism);
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser4/EBLT amplitude over PN SM%02d L4B", ism);
    mui_->subscribe(histo, ism);

    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser1/EBLT shape SM%02d L1A", ism);
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser2/EBLT shape SM%02d L2A", ism);
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser3/EBLT shape SM%02d L3A", ism);
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser4/EBLT shape SM%02d L4A", ism);
    mui_->subscribe(histo, ism);

    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser1/EBLT shape SM%02d L1B", ism);
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser2/EBLT shape SM%02d L2B", ism);
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser3/EBLT shape SM%02d L3B", ism);
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser4/EBLT shape SM%02d L4B", ism);
    mui_->subscribe(histo, ism);

    sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser1/Gain01/EBPDT PNs amplitude SM%02d G01 L1", ism);
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser2/Gain01/EBPDT PNs amplitude SM%02d G01 L2", ism);
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser3/Gain01/EBPDT PNs amplitude SM%02d G01 L3", ism);
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser4/Gain01/EBPDT PNs amplitude SM%02d G01 L4", ism);
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser1/Gain01/EBPDT PNs pedestal SM%02d G01 L1", ism);
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser2/Gain01/EBPDT PNs pedestal SM%02d G01 L2", ism);
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser3/Gain01/EBPDT PNs pedestal SM%02d G01 L3", ism);
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser4/Gain01/EBPDT PNs pedestal SM%02d G01 L4", ism);
    mui_->subscribe(histo, ism);

    sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser1/Gain16/EBPDT PNs amplitude SM%02d G16 L1", ism);
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser2/Gain16/EBPDT PNs amplitude SM%02d G16 L2", ism);
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser3/Gain16/EBPDT PNs amplitude SM%02d G16 L3", ism);
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser4/Gain16/EBPDT PNs amplitude SM%02d G16 L4", ism);
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser1/Gain16/EBPDT PNs pedestal SM%02d G16 L1", ism);
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser2/Gain16/EBPDT PNs pedestal SM%02d G16 L2", ism);
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser3/Gain16/EBPDT PNs pedestal SM%02d G16 L3", ism);
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser4/Gain16/EBPDT PNs pedestal SM%02d G16 L4", ism);
    mui_->subscribe(histo, ism);

  }

  if ( collateSources_ ) {

    if ( verbose_ ) cout << "EBLaserClient: collate" << endl;

    for ( unsigned int i=0; i<superModules_.size(); i++ ) {

      int ism = superModules_[i];

      sprintf(histo, "EBLT amplitude SM%02d L1A", ism);
      me_h01_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBLaserTask/Laser1");
      sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser1/EBLT amplitude SM%02d L1A", ism);
      mui_->add(me_h01_[ism-1], histo);

      sprintf(histo, "EBLT amplitude over PN SM%02d L1A", ism);
      me_h02_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBLaserTask/Laser1");
      sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser1/EBLT amplitude over PN SM%02d L1A", ism);
      mui_->add(me_h02_[ism-1], histo);

      sprintf(histo, "EBLT amplitude SM%02d L2A", ism);
      me_h03_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBLaserTask/Laser2");
      sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser2/EBLT amplitude SM%02d L2A", ism);
      mui_->add(me_h03_[ism-1], histo);

      sprintf(histo, "EBLT amplitude over PN SM%02d L2A", ism);
      me_h04_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBLaserTask/Laser2");
      sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser2/EBLT amplitude over PN SM%02d L2A", ism);
      mui_->add(me_h04_[ism-1], histo);

      sprintf(histo, "EBLT amplitude SM%02d L3A", ism);
      me_h05_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBLaserTask/Laser3");
      sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser3/EBLT amplitude SM%02d L3A", ism);
      mui_->add(me_h05_[ism-1], histo);

      sprintf(histo, "EBLT amplitude over PN SM%02d L3A", ism);
      me_h06_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBLaserTask/Laser3");
      sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser3/EBLT amplitude over PN SM%02d L3A", ism);
      mui_->add(me_h06_[ism-1], histo);

      sprintf(histo, "EBLT amplitude SM%02d L4A", ism);
      me_h07_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBLaserTask/Laser4");
      sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser4/EBLT amplitude SM%02d L4A", ism);
      mui_->add(me_h07_[ism-1], histo);

      sprintf(histo, "EBLT amplitude over PN SM%02d L4A", ism);
      me_h08_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBLaserTask/Laser4");
      sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser4/EBLT amplitude over PN SM%02d L4A", ism);
      mui_->add(me_h08_[ism-1], histo);

      sprintf(histo, "EBLT timing SM%02d L1A", ism);
      me_h09_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBLaserTask/Laser1");
      sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser1/EBLT timing SM%02d L1A", ism);
      mui_->add(me_h09_[ism-1], histo);

      sprintf(histo, "EBLT timing SM%02d L2A", ism);
      me_h10_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBLaserTask/Laser2");
      sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser2/EBLT timing SM%02d L2A", ism);
      mui_->add(me_h10_[ism-1], histo);

      sprintf(histo, "EBLT timing SM%02d L3A", ism);
      me_h11_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBLaserTask/Laser3");
      sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser3/EBLT timing SM%02d L3A", ism);
      mui_->add(me_h11_[ism-1], histo);

      sprintf(histo, "EBLT timing SM%02d L4A", ism);
      me_h12_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBLaserTask/Laser4");
      sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser4/EBLT timing SM%02d L4A", ism);
      mui_->add(me_h12_[ism-1], histo);

      sprintf(histo, "EBLT amplitude SM%02d L1B", ism);
      me_h13_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBLaserTask/Laser1");
      sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser1/EBLT amplitude SM%02d L1B", ism);
      mui_->add(me_h13_[ism-1], histo);

      sprintf(histo, "EBLT amplitude over PN SM%02d L1B", ism);
      me_h14_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBLaserTask/Laser1");
      sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser1/EBLT amplitude over PN SM%02d L1B", ism);
      mui_->add(me_h14_[ism-1], histo);

      sprintf(histo, "EBLT amplitude SM%02d L2B", ism);
      me_h15_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBLaserTask/Laser2");
      sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser2/EBLT amplitude SM%02d L2B", ism);
      mui_->add(me_h15_[ism-1], histo);

      sprintf(histo, "EBLT amplitude over PN SM%02d L2B", ism);
      me_h16_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBLaserTask/Laser2");
      sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser2/EBLT amplitude over PN SM%02d L2B", ism);
      mui_->add(me_h16_[ism-1], histo);

      sprintf(histo, "EBLT amplitude SM%02d L3B", ism);
      me_h17_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBLaserTask/Laser3");
      sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser3/EBLT amplitude SM%02d L3B", ism);
      mui_->add(me_h17_[ism-1], histo);

      sprintf(histo, "EBLT amplitude over PN SM%02d L3B", ism);
      me_h18_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBLaserTask/Laser3");
      sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser3/EBLT amplitude over PN SM%02d L3B", ism);
      mui_->add(me_h18_[ism-1], histo);

      sprintf(histo, "EBLT amplitude SM%02d L4B", ism);
      me_h19_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBLaserTask/Laser4");
      sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser4/EBLT amplitude SM%02d L4B", ism);
      mui_->add(me_h19_[ism-1], histo);

      sprintf(histo, "EBLT amplitude over PN SM%02d L4B", ism);
      me_h20_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBLaserTask/Laser4");
      sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser4/EBLT amplitude over PN SM%02d L4B", ism);
      mui_->add(me_h20_[ism-1], histo);

      sprintf(histo, "EBLT timing SM%02d L1B", ism);
      me_h21_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBLaserTask/Laser1");
      sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser1/EBLT timing SM%02d L1B", ism);
      mui_->add(me_h21_[ism-1], histo);

      sprintf(histo, "EBLT timing SM%02d L2B", ism);
      me_h22_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBLaserTask/Laser2");
      sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser2/EBLT timing SM%02d L2B", ism);
      mui_->add(me_h22_[ism-1], histo);

      sprintf(histo, "EBLT timing SM%02d L3B", ism);
      me_h23_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBLaserTask/Laser3");
      sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser3/EBLT timing SM%02d L3B", ism);
      mui_->add(me_h23_[ism-1], histo);

      sprintf(histo, "EBLT timing SM%02d L4B", ism);
      me_h24_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBLaserTask/Laser4");
      sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser4/EBLT timing SM%02d L4B", ism);
      mui_->add(me_h24_[ism-1], histo);

      sprintf(histo, "EBLT shape SM%02d L1A", ism);
      me_hs01_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBLaserTask/Laser1");
      sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser1/EBLT shape SM%02d L1A", ism);
      mui_->add(me_hs01_[ism-1], histo);

      sprintf(histo, "EBLT shape SM%02d L2A", ism);
      me_hs02_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBLaserTask/Laser2");
      sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser2/EBLT shape SM%02d L2A", ism);
      mui_->add(me_hs02_[ism-1], histo);

      sprintf(histo, "EBLT shape SM%02d L3A", ism);
      me_hs03_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBLaserTask/Laser3");
      sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser3/EBLT shape SM%02d L3A", ism);
      mui_->add(me_hs03_[ism-1], histo);

      sprintf(histo, "EBLT shape SM%02d L4A", ism);
      me_hs04_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBLaserTask/Laser4");
      sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser4/EBLT shape SM%02d L4A", ism);
      mui_->add(me_hs04_[ism-1], histo);

      sprintf(histo, "EBLT shape SM%02d L1B", ism);
      me_hs05_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBLaserTask/Laser1");
      sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser1/EBLT shape SM%02d L1B", ism);
      mui_->add(me_hs05_[ism-1], histo);

      sprintf(histo, "EBLT shape SM%02d L2B", ism);
      me_hs06_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBLaserTask/Laser2");
      sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser2/EBLT shape SM%02d L2B", ism);
      mui_->add(me_hs06_[ism-1], histo);

      sprintf(histo, "EBLT shape SM%02d L3B", ism);
      me_hs07_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBLaserTask/Laser3");
      sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser3/EBLT shape SM%02d L3B", ism);
      mui_->add(me_hs07_[ism-1], histo);

      sprintf(histo, "EBLT shape SM%02d L4B", ism);
      me_hs08_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBLaserTask/Laser4");
      sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser4/EBLT shape SM%02d L4B", ism);
      mui_->add(me_hs08_[ism-1], histo);

      sprintf(histo, "EBPDT PNs amplitude SM%02d G01 L1", ism);
      me_i01_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser1/Gain01");
      sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser1/Gain01/EBPDT PNs amplitude SM%02d G01 L1", ism);
      mui_->add(me_i01_[ism-1], histo);

      sprintf(histo, "EBPDT PNs amplitude SM%02d G01 L2", ism);
      me_i02_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser2/Gain01");
      sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser2/Gain01/EBPDT PNs amplitude SM%02d G01 L2", ism);
      mui_->add(me_i02_[ism-1], histo);

      sprintf(histo, "EBPDT PNs amplitude SM%02d G01 L3", ism);
      me_i03_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser3/Gain01");
      sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser3/Gain01/EBPDT PNs amplitude SM%02d G01 L3", ism);
      mui_->add(me_i03_[ism-1], histo);

      sprintf(histo, "EBPDT PNs amplitude SM%02d G01 L4", ism);
      me_i04_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser4/Gain01");
      sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser4/Gain01/EBPDT PNs amplitude SM%02d G01 L4", ism);
      mui_->add(me_i04_[ism-1], histo);

      sprintf(histo, "EBPDT PNs pedestal SM%02d G01 L1", ism);
      me_i05_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser1/Gain01");
      sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser1/Gain01/EBPDT PNs pedestal SM%02d G01 L1", ism);
      mui_->add(me_i05_[ism-1], histo);

      sprintf(histo, "EBPDT PNs pedestal SM%02d G01 L2", ism);
      me_i06_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser2/Gain01");
      sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser2/Gain01/EBPDT PNs pedestal SM%02d G01 L2", ism);
      mui_->add(me_i06_[ism-1], histo);

      sprintf(histo, "EBPDT PNs pedestal SM%02d G01 L3", ism);
      me_i07_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser3/Gain01");
      sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser3/Gain01/EBPDT PNs pedestal SM%02d G01 L3", ism);
      mui_->add(me_i07_[ism-1], histo);

      sprintf(histo, "EBPDT PNs pedestal SM%02d G01 L4", ism);
      me_i08_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser4/Gain01");
      sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser4/Gain01/EBPDT PNs pedestal SM%02d G01 L4", ism);
      mui_->add(me_i08_[ism-1], histo);

      sprintf(histo, "EBPDT PNs amplitude SM%02d G16 L1", ism);
      me_i09_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser1/Gain16");
      sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser1/Gain16/EBPDT PNs amplitude SM%02d G16 L1", ism);
      mui_->add(me_i09_[ism-1], histo);

      sprintf(histo, "EBPDT PNs amplitude SM%02d G16 L2", ism);
      me_i10_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser2/Gain16");
      sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser2/Gain16/EBPDT PNs amplitude SM%02d G16 L2", ism);
      mui_->add(me_i10_[ism-1], histo);

      sprintf(histo, "EBPDT PNs amplitude SM%02d G16 L3", ism);
      me_i11_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser3/Gain16");
      sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser3/Gain16/EBPDT PNs amplitude SM%02d G16 L3", ism);
      mui_->add(me_i11_[ism-1], histo);

      sprintf(histo, "EBPDT PNs amplitude SM%02d G16 L4", ism);
      me_i12_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser4/Gain16");
      sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser4/Gain16/EBPDT PNs amplitude SM%02d G16 L4", ism);
      mui_->add(me_i12_[ism-1], histo);

      sprintf(histo, "EBPDT PNs pedestal SM%02d G16 L1", ism);
      me_i13_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser1/Gain16");
      sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser1/Gain16/EBPDT PNs pedestal SM%02d G16 L1", ism);
      mui_->add(me_i13_[ism-1], histo);

      sprintf(histo, "EBPDT PNs pedestal SM%02d G16 L2", ism);
      me_i14_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser2/Gain16");
      sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser2/Gain16/EBPDT PNs pedestal SM%02d G16 L2", ism);
      mui_->add(me_i14_[ism-1], histo);

      sprintf(histo, "EBPDT PNs pedestal SM%02d G16 L3", ism);
      me_i15_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser3/Gain16");
      sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser3/Gain16/EBPDT PNs pedestal SM%02d G16 L3", ism);
      mui_->add(me_i15_[ism-1], histo);

      sprintf(histo, "EBPDT PNs pedestal SM%02d G16 L4", ism);
      me_i16_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser4/Gain16");
      sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser4/Gain16/EBPDT PNs pedestal SM%02d G16 L4", ism);
      mui_->add(me_i16_[ism-1], histo);

    }

  }

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBLaserTask/Laser1/EBLT amplitude SM%02d L1A", ism);
      if ( qth01_[ism-1] ) mui_->useQTest(histo, qth01_[ism-1]->getName());
      sprintf(histo, "EcalBarrel/Sums/EBLaserTask/Laser2/EBLT amplitude SM%02d L2A", ism);
      if ( qth02_[ism-1] ) mui_->useQTest(histo, qth02_[ism-1]->getName());
      sprintf(histo, "EcalBarrel/Sums/EBLaserTask/Laser3/EBLT amplitude SM%02d L3A", ism);
      if ( qth03_[ism-1] ) mui_->useQTest(histo, qth03_[ism-1]->getName());
      sprintf(histo, "EcalBarrel/Sums/EBLaserTask/Laser4/EBLT amplitude SM%02d L4A", ism);
      if ( qth04_[ism-1] ) mui_->useQTest(histo, qth04_[ism-1]->getName());
      sprintf(histo, "EcalBarrel/Sums/EBLaserTask/Laser1/EBLT amplitude SM%02d L1B", ism);
      if ( qth05_[ism-1] ) mui_->useQTest(histo, qth05_[ism-1]->getName());
      sprintf(histo, "EcalBarrel/Sums/EBLaserTask/Laser2/EBLT amplitude SM%02d L2B", ism);
      if ( qth06_[ism-1] ) mui_->useQTest(histo, qth06_[ism-1]->getName());
      sprintf(histo, "EcalBarrel/Sums/EBLaserTask/Laser3/EBLT amplitude SM%02d L3B", ism);
      if ( qth07_[ism-1] ) mui_->useQTest(histo, qth07_[ism-1]->getName());
      sprintf(histo, "EcalBarrel/Sums/EBLaserTask/Laser4/EBLT amplitude SM%02d L4B", ism);
      if ( qth08_[ism-1] ) mui_->useQTest(histo, qth08_[ism-1]->getName());
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser1/Gain01/EBPDT PNs amplitude SM%02d G01 L1", ism);
      if ( qth09_[ism-1] ) mui_->useQTest(histo, qth09_[ism-1]->getName());
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser2/Gain01/EBPDT PNs amplitude SM%02d G01 L2", ism);
      if ( qth10_[ism-1] ) mui_->useQTest(histo, qth10_[ism-1]->getName());
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser3/Gain01/EBPDT PNs amplitude SM%02d G01 L3", ism);
      if ( qth11_[ism-1] ) mui_->useQTest(histo, qth11_[ism-1]->getName());
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser4/Gain01/EBPDT PNs amplitude SM%02d G01 L4", ism);
      if ( qth12_[ism-1] ) mui_->useQTest(histo, qth12_[ism-1]->getName());
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser1/Gain01/EBPDT PNs pedestal SM%02d G01 L1", ism);
      if ( qth13_[ism-1] ) mui_->useQTest(histo, qth13_[ism-1]->getName());
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser2/Gain01/EBPDT PNs pedestal SM%02d G01 L2", ism);
      if ( qth14_[ism-1] ) mui_->useQTest(histo, qth14_[ism-1]->getName());
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser3/Gain01/EBPDT PNs pedestal SM%02d G01 L3", ism);
      if ( qth15_[ism-1] ) mui_->useQTest(histo, qth15_[ism-1]->getName());
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser4/Gain01/EBPDT PNs pedestal SM%02d G01 L4", ism);
      if ( qth16_[ism-1] ) mui_->useQTest(histo, qth16_[ism-1]->getName());
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser1/Gain16/EBPDT PNs amplitude SM%02d G16 L1", ism);
      if ( qth17_[ism-1] ) mui_->useQTest(histo, qth17_[ism-1]->getName());
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser2/Gain16/EBPDT PNs amplitude SM%02d G16 L2", ism);
      if ( qth18_[ism-1] ) mui_->useQTest(histo, qth18_[ism-1]->getName());
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser3/Gain16/EBPDT PNs amplitude SM%02d G16 L3", ism);
      if ( qth19_[ism-1] ) mui_->useQTest(histo, qth19_[ism-1]->getName());
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser4/Gain16/EBPDT PNs amplitude SM%02d G16 L4", ism);
      if ( qth20_[ism-1] ) mui_->useQTest(histo, qth20_[ism-1]->getName());
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser1/Gain16/EBPDT PNs pedestal SM%02d G16 L1", ism);
      if ( qth21_[ism-1] ) mui_->useQTest(histo, qth21_[ism-1]->getName());
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser2/Gain16/EBPDT PNs pedestal SM%02d G16 L2", ism);
      if ( qth22_[ism-1] ) mui_->useQTest(histo, qth22_[ism-1]->getName());
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser3/Gain16/EBPDT PNs pedestal SM%02d G16 L3", ism);
      if ( qth23_[ism-1] ) mui_->useQTest(histo, qth23_[ism-1]->getName());
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser4/Gain16/EBPDT PNs pedestal SM%02d G16 L4", ism);
      if ( qth24_[ism-1] ) mui_->useQTest(histo, qth24_[ism-1]->getName());
    } else {
      if ( enableMonitorDaemon_ ) {
        sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser1/EBLT amplitude SM%02d L1A", ism);
        if ( qth01_[ism-1] ) mui_->useQTest(histo, qth01_[ism-1]->getName());
        sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser2/EBLT amplitude SM%02d L2A", ism);
        if ( qth02_[ism-1] ) mui_->useQTest(histo, qth02_[ism-1]->getName());
        sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser3/EBLT amplitude SM%02d L3A", ism);
        if ( qth03_[ism-1] ) mui_->useQTest(histo, qth03_[ism-1]->getName());
        sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser4/EBLT amplitude SM%02d L4A", ism);
        if ( qth04_[ism-1] ) mui_->useQTest(histo, qth04_[ism-1]->getName());
        sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser1/EBLT amplitude SM%02d L1B", ism);
        if ( qth05_[ism-1] ) mui_->useQTest(histo, qth05_[ism-1]->getName());
        sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser2/EBLT amplitude SM%02d L2B", ism);
        if ( qth06_[ism-1] ) mui_->useQTest(histo, qth06_[ism-1]->getName());
        sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser3/EBLT amplitude SM%02d L3B", ism);
        if ( qth07_[ism-1] ) mui_->useQTest(histo, qth07_[ism-1]->getName());
        sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser4/EBLT amplitude SM%02d L4B", ism);
        if ( qth08_[ism-1] ) mui_->useQTest(histo, qth08_[ism-1]->getName());
        sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser1/Gain01/EBPDT PNs amplitude SM%02d G01 L1", ism);
        if ( qth09_[ism-1] ) mui_->useQTest(histo, qth09_[ism-1]->getName());
        sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser2/Gain01/EBPDT PNs amplitude SM%02d G01 L2", ism);
        if ( qth10_[ism-1] ) mui_->useQTest(histo, qth10_[ism-1]->getName());
        sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser3/Gain01/EBPDT PNs amplitude SM%02d G01 L3", ism);
        if ( qth11_[ism-1] ) mui_->useQTest(histo, qth11_[ism-1]->getName());
        sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser4/Gain01/EBPDT PNs amplitude SM%02d G01 L4", ism);
        if ( qth12_[ism-1] ) mui_->useQTest(histo, qth12_[ism-1]->getName());
        sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser1/Gain01/EBPDT PNs pedestal SM%02d G01 L1", ism);
        if ( qth13_[ism-1] ) mui_->useQTest(histo, qth13_[ism-1]->getName());
        sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser2/Gain01/EBPDT PNs pedestal SM%02d G01 L2", ism);
        if ( qth14_[ism-1] ) mui_->useQTest(histo, qth14_[ism-1]->getName());
        sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser3/Gain01/EBPDT PNs pedestal SM%02d G01 L3", ism);
        if ( qth15_[ism-1] ) mui_->useQTest(histo, qth15_[ism-1]->getName());
        sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser4/Gain01/EBPDT PNs pedestal SM%02d G01 L4", ism);
        if ( qth16_[ism-1] ) mui_->useQTest(histo, qth16_[ism-1]->getName());
        sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser1/Gain16/EBPDT PNs amplitude SM%02d G16 L1", ism);
        if ( qth17_[ism-1] ) mui_->useQTest(histo, qth17_[ism-1]->getName());
        sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser2/Gain16/EBPDT PNs amplitude SM%02d G16 L2", ism);
        if ( qth18_[ism-1] ) mui_->useQTest(histo, qth18_[ism-1]->getName());
        sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser3/Gain16/EBPDT PNs amplitude SM%02d G16 L3", ism);
        if ( qth19_[ism-1] ) mui_->useQTest(histo, qth19_[ism-1]->getName());
        sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser4/Gain16/EBPDT PNs amplitude SM%02d G16 L4", ism);
        if ( qth20_[ism-1] ) mui_->useQTest(histo, qth20_[ism-1]->getName());
        sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser1/Gain16/EBPDT PNs pedestal SM%02d G16 L1", ism);
        if ( qth21_[ism-1] ) mui_->useQTest(histo, qth21_[ism-1]->getName());
        sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser2/Gain16/EBPDT PNs pedestal SM%02d G16 L2", ism);
        if ( qth22_[ism-1] ) mui_->useQTest(histo, qth22_[ism-1]->getName());
        sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser3/Gain16/EBPDT PNs pedestal SM%02d G16 L3", ism);
        if ( qth23_[ism-1] ) mui_->useQTest(histo, qth23_[ism-1]->getName());
        sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser4/Gain16/EBPDT PNs pedestal SM%02d G16 L4", ism);
        if ( qth24_[ism-1] ) mui_->useQTest(histo, qth24_[ism-1]->getName());
      } else {
        sprintf(histo, "EcalBarrel/EBLaserTask/Laser1/EBLT amplitude SM%02d L1A", ism);
        if ( qth01_[ism-1] ) mui_->useQTest(histo, qth01_[ism-1]->getName());
        sprintf(histo, "EcalBarrel/EBLaserTask/Laser2/EBLT amplitude SM%02d L2A", ism);
        if ( qth02_[ism-1] ) mui_->useQTest(histo, qth02_[ism-1]->getName());
        sprintf(histo, "EcalBarrel/EBLaserTask/Laser3/EBLT amplitude SM%02d L3A", ism);
        if ( qth03_[ism-1] ) mui_->useQTest(histo, qth03_[ism-1]->getName());
        sprintf(histo, "EcalBarrel/EBLaserTask/Laser4/EBLT amplitude SM%02d L4A", ism);
        if ( qth04_[ism-1] ) mui_->useQTest(histo, qth04_[ism-1]->getName());
        sprintf(histo, "EcalBarrel/EBLaserTask/Laser1/EBLT amplitude SM%02d L1B", ism);
        if ( qth05_[ism-1] ) mui_->useQTest(histo, qth05_[ism-1]->getName());
        sprintf(histo, "EcalBarrel/EBLaserTask/Laser2/EBLT amplitude SM%02d L2B", ism);
        if ( qth06_[ism-1] ) mui_->useQTest(histo, qth06_[ism-1]->getName());
        sprintf(histo, "EcalBarrel/EBLaserTask/Laser3/EBLT amplitude SM%02d L3B", ism);
        if ( qth07_[ism-1] ) mui_->useQTest(histo, qth07_[ism-1]->getName());
        sprintf(histo, "EcalBarrel/EBLaserTask/Laser4/EBLT amplitude SM%02d L4B", ism);
        if ( qth08_[ism-1] ) mui_->useQTest(histo, qth08_[ism-1]->getName());
        sprintf(histo, "EcalBarrel/EBPnDiodeTask/Laser1/Gain01/EBPDT PNs amplitude SM%02d G01 L1", ism);
        if ( qth09_[ism-1] ) mui_->useQTest(histo, qth09_[ism-1]->getName());
        sprintf(histo, "EcalBarrel/EBPnDiodeTask/Laser2/Gain01/EBPDT PNs amplitude SM%02d G01 L2", ism);
        if ( qth10_[ism-1] ) mui_->useQTest(histo, qth10_[ism-1]->getName());
        sprintf(histo, "EcalBarrel/EBPnDiodeTask/Laser3/Gain01/EBPDT PNs amplitude SM%02d G01 L3", ism);
        if ( qth11_[ism-1] ) mui_->useQTest(histo, qth11_[ism-1]->getName());
        sprintf(histo, "EcalBarrel/EBPnDiodeTask/Laser4/Gain01/EBPDT PNs amplitude SM%02d G01 L4", ism);
        if ( qth12_[ism-1] ) mui_->useQTest(histo, qth12_[ism-1]->getName());
        sprintf(histo, "EcalBarrel/EBPnDiodeTask/Laser1/Gain01/EBPDT PNs pedestal SM%02d G01 L1", ism);
        if ( qth13_[ism-1] ) mui_->useQTest(histo, qth13_[ism-1]->getName());
        sprintf(histo, "EcalBarrel/EBPnDiodeTask/Laser2/Gain01/EBPDT PNs pedestal SM%02d G01 L2", ism);
        if ( qth14_[ism-1] ) mui_->useQTest(histo, qth14_[ism-1]->getName());
        sprintf(histo, "EcalBarrel/EBPnDiodeTask/Laser3/Gain01/EBPDT PNs pedestal SM%02d G01 L3", ism);
        if ( qth15_[ism-1] ) mui_->useQTest(histo, qth15_[ism-1]->getName());
        sprintf(histo, "EcalBarrel/EBPnDiodeTask/Laser4/Gain01/EBPDT PNs pedestal SM%02d G01 L4", ism);
        if ( qth16_[ism-1] ) mui_->useQTest(histo, qth16_[ism-1]->getName());
        sprintf(histo, "EcalBarrel/EBPnDiodeTask/Laser1/Gain16/EBPDT PNs amplitude SM%02d G16 L1", ism);
        if ( qth17_[ism-1] ) mui_->useQTest(histo, qth17_[ism-1]->getName());
        sprintf(histo, "EcalBarrel/EBPnDiodeTask/Laser2/Gain16/EBPDT PNs amplitude SM%02d G16 L2", ism);
        if ( qth18_[ism-1] ) mui_->useQTest(histo, qth18_[ism-1]->getName());
        sprintf(histo, "EcalBarrel/EBPnDiodeTask/Laser3/Gain16/EBPDT PNs amplitude SM%02d G16 L3", ism);
        if ( qth19_[ism-1] ) mui_->useQTest(histo, qth19_[ism-1]->getName());
        sprintf(histo, "EcalBarrel/EBPnDiodeTask/Laser4/Gain16/EBPDT PNs amplitude SM%02d G16 L4", ism);
        if ( qth20_[ism-1] ) mui_->useQTest(histo, qth20_[ism-1]->getName());
        sprintf(histo, "EcalBarrel/EBPnDiodeTask/Laser1/Gain16/EBPDT PNs pedestal SM%02d G16 L1", ism);
        if ( qth21_[ism-1] ) mui_->useQTest(histo, qth21_[ism-1]->getName());
        sprintf(histo, "EcalBarrel/EBPnDiodeTask/Laser2/Gain16/EBPDT PNs pedestal SM%02d G16 L2", ism);
        if ( qth22_[ism-1] ) mui_->useQTest(histo, qth22_[ism-1]->getName());
        sprintf(histo, "EcalBarrel/EBPnDiodeTask/Laser3/Gain16/EBPDT PNs pedestal SM%02d G16 L3", ism);
        if ( qth23_[ism-1] ) mui_->useQTest(histo, qth23_[ism-1]->getName());
        sprintf(histo, "EcalBarrel/EBPnDiodeTask/Laser4/Gain16/EBPDT PNs pedestal SM%02d G16 L4", ism);
        if ( qth24_[ism-1] ) mui_->useQTest(histo, qth24_[ism-1]->getName());
      }
    }

  }

}

void EBLaserClient::subscribeNew(void){

  Char_t histo[200];

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    unsigned int ism = superModules_[i];

    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser1/EBLT amplitude SM%02d L1A", ism);
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser1/EBLT timing SM%02d L1A", ism);
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser1/EBLT amplitude over PN SM%02d L1A", ism);
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser2/EBLT amplitude SM%02d L2A", ism);
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser2/EBLT timing SM%02d L2A", ism);
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser2/EBLT amplitude over PN SM%02d L2A", ism);
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser3/EBLT amplitude SM%02d L3A", ism);
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser3/EBLT timing SM%02d L3A", ism);
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser3/EBLT amplitude over PN SM%02d L3A", ism);
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser4/EBLT amplitude SM%02d L4A", ism);
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser4/EBLT timing SM%02d L4A", ism);
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser4/EBLT amplitude over PN SM%02d L4A", ism);
    mui_->subscribeNew(histo, ism);

    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser1/EBLT amplitude SM%02d L1B", ism);
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser1/EBLT timing SM%02d L1B", ism);
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser1/EBLT amplitude over PN SM%02d L1B", ism);
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser2/EBLT amplitude SM%02d L2B", ism);
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser2/EBLT timing SM%02d L2B", ism); 
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser2/EBLT amplitude over PN SM%02d L2B", ism);
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser3/EBLT amplitude SM%02d L3B", ism);
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser3/EBLT timing SM%02d L3B", ism);
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser3/EBLT amplitude over PN SM%02d L3B", ism);
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser4/EBLT amplitude SM%02d L4B", ism);
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser4/EBLT timing SM%02d L4B", ism);
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser4/EBLT amplitude over PN SM%02d L4B", ism);
    mui_->subscribeNew(histo, ism);

    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser1/EBLT shape SM%02d L1A", ism);
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser2/EBLT shape SM%02d L2A", ism);
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser3/EBLT shape SM%02d L3A", ism);
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser4/EBLT shape SM%02d L4A", ism);
    mui_->subscribeNew(histo, ism);

    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser1/EBLT shape SM%02d L1B", ism);
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser2/EBLT shape SM%02d L2B", ism);
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser3/EBLT shape SM%02d L3B", ism);
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser4/EBLT shape SM%02d L4B", ism);
    mui_->subscribeNew(histo, ism);

    sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser1/Gain01/EBPDT PNs amplitude SM%02d G01 L1", ism);
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser2/Gain01/EBPDT PNs amplitude SM%02d G01 L2", ism);
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser3/Gain01/EBPDT PNs amplitude SM%02d G01 L3", ism);
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser4/Gain01/EBPDT PNs amplitude SM%02d G01 L4", ism);
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser1/Gain01/EBPDT PNs pedestal SM%02d G01 L1", ism);
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser2/Gain01/EBPDT PNs pedestal SM%02d G01 L2", ism);
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser3/Gain01/EBPDT PNs pedestal SM%02d G01 L3", ism);
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser4/Gain01/EBPDT PNs pedestal SM%02d G01 L4", ism);
    mui_->subscribeNew(histo, ism);

    sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser1/Gain16/EBPDT PNs amplitude SM%02d G16 L1", ism);
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser2/Gain16/EBPDT PNs amplitude SM%02d G16 L2", ism);
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser3/Gain16/EBPDT PNs amplitude SM%02d G16 L3", ism);
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser4/Gain16/EBPDT PNs amplitude SM%02d G16 L4", ism);
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser1/Gain16/EBPDT PNs pedestal SM%02d G16 L1", ism);
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser2/Gain16/EBPDT PNs pedestal SM%02d G16 L2", ism);
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser3/Gain16/EBPDT PNs pedestal SM%02d G16 L3", ism);
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser4/Gain16/EBPDT PNs pedestal SM%02d G16 L4", ism);
    mui_->subscribeNew(histo, ism);

  }

}

void EBLaserClient::unsubscribe(void){

  if ( verbose_ ) cout << "EBLaserClient: unsubscribe" << endl;

  if ( collateSources_ ) {

    if ( verbose_ ) cout << "EBLaserClient: uncollate" << endl;

    if ( mui_ ) {

      for ( unsigned int i=0; i<superModules_.size(); i++ ) {

        int ism = superModules_[i];

        mui_->removeCollate(me_h01_[ism-1]);
        mui_->removeCollate(me_h02_[ism-1]);
        mui_->removeCollate(me_h03_[ism-1]);
        mui_->removeCollate(me_h04_[ism-1]);
        mui_->removeCollate(me_h05_[ism-1]);
        mui_->removeCollate(me_h06_[ism-1]);
        mui_->removeCollate(me_h07_[ism-1]);
        mui_->removeCollate(me_h08_[ism-1]);

        mui_->removeCollate(me_h09_[ism-1]);
        mui_->removeCollate(me_h10_[ism-1]);
        mui_->removeCollate(me_h11_[ism-1]);
        mui_->removeCollate(me_h12_[ism-1]);

        mui_->removeCollate(me_h13_[ism-1]);
        mui_->removeCollate(me_h14_[ism-1]);
        mui_->removeCollate(me_h15_[ism-1]);
        mui_->removeCollate(me_h16_[ism-1]);
        mui_->removeCollate(me_h17_[ism-1]);
        mui_->removeCollate(me_h18_[ism-1]);
        mui_->removeCollate(me_h19_[ism-1]);
        mui_->removeCollate(me_h20_[ism-1]);

        mui_->removeCollate(me_h21_[ism-1]);
        mui_->removeCollate(me_h22_[ism-1]);
        mui_->removeCollate(me_h23_[ism-1]);
        mui_->removeCollate(me_h24_[ism-1]);

	mui_->removeCollate(me_hs01_[ism-1]);
        mui_->removeCollate(me_hs02_[ism-1]);
        mui_->removeCollate(me_hs03_[ism-1]);
        mui_->removeCollate(me_hs04_[ism-1]);

        mui_->removeCollate(me_hs05_[ism-1]);
        mui_->removeCollate(me_hs06_[ism-1]);
        mui_->removeCollate(me_hs07_[ism-1]);
        mui_->removeCollate(me_hs08_[ism-1]);

        mui_->removeCollate(me_i01_[ism-1]);
        mui_->removeCollate(me_i02_[ism-1]);
        mui_->removeCollate(me_i03_[ism-1]);
        mui_->removeCollate(me_i04_[ism-1]);
        mui_->removeCollate(me_i05_[ism-1]);
        mui_->removeCollate(me_i06_[ism-1]);
        mui_->removeCollate(me_i07_[ism-1]);
        mui_->removeCollate(me_i08_[ism-1]);

        mui_->removeCollate(me_i09_[ism-1]);
        mui_->removeCollate(me_i10_[ism-1]);
        mui_->removeCollate(me_i11_[ism-1]);
        mui_->removeCollate(me_i12_[ism-1]);
        mui_->removeCollate(me_i13_[ism-1]);
        mui_->removeCollate(me_i14_[ism-1]);
        mui_->removeCollate(me_i15_[ism-1]);
        mui_->removeCollate(me_i16_[ism-1]);

      }

    }

  }

  Char_t histo[200];

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    unsigned int ism = superModules_[i];

    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser1/EBLT amplitude SM%02d L1A", ism);
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser1/EBLT timing SM%02d L1A", ism);
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser1/EBLT amplitude over PN SM%02d L1A", ism);
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser2/EBLT amplitude SM%02d L2A", ism);
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser2/EBLT timing SM%02d L2A", ism);
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser2/EBLT amplitude over PN SM%02d L2A", ism);
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser3/EBLT amplitude SM%02d L3A", ism);
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser3/EBLT timing SM%02d L3A", ism);
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser3/EBLT amplitude over PN SM%02d L3A", ism);
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser4/EBLT amplitude SM%02d L4A", ism);
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser4/EBLT timing SM%02d L4A", ism);
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser4/EBLT amplitude over PN SM%02d L4A", ism);
    mui_->unsubscribe(histo, ism);

    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser1/EBLT amplitude SM%02d L1B", ism);
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser1/EBLT timing SM%02d L1B", ism);
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser1/EBLT amplitude over PN SM%02d L1B", ism);
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser2/EBLT amplitude SM%02d L2B", ism);
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser2/EBLT timing SM%02d L2B", ism);
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser2/EBLT amplitude over PN SM%02d L2B", ism);
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser3/EBLT amplitude SM%02d L3B", ism);
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser3/EBLT timing SM%02d L3B", ism);
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser3/EBLT amplitude over PN SM%02d L3B", ism);
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser4/EBLT amplitude SM%02d L4B", ism);
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser4/EBLT timing SM%02d L4B", ism);
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser4/EBLT amplitude over PN SM%02d L4B", ism);
    mui_->unsubscribe(histo, ism);

    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser1/EBLT shape SM%02d L1A", ism);
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser2/EBLT shape SM%02d L2A", ism);
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser3/EBLT shape SM%02d L3A", ism);
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser4/EBLT shape SM%02d L4A", ism);
    mui_->unsubscribe(histo, ism);

    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser1/EBLT shape SM%02d L1B", ism);
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser2/EBLT shape SM%02d L2B", ism);
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser3/EBLT shape SM%02d L3B", ism);
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser4/EBLT shape SM%02d L4B", ism);
    mui_->unsubscribe(histo, ism);

    sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser1/Gain01/EBPDT PNs amplitude SM%02d G01 L1", ism);
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser2/Gain01/EBPDT PNs amplitude SM%02d G01 L2", ism);
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser3/Gain01/EBPDT PNs amplitude SM%02d G01 L3", ism);
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser4/Gain01/EBPDT PNs amplitude SM%02d G01 L4", ism);
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser1/Gain01/EBPDT PNs pedestal SM%02d G01 L1", ism);
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser2/Gain01/EBPDT PNs pedestal SM%02d G01 L2", ism);
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser3/Gain01/EBPDT PNs pedestal SM%02d G01 L3", ism);
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser4/Gain01/EBPDT PNs pedestal SM%02d G01 L4", ism);
    mui_->unsubscribe(histo, ism);

    sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser1/Gain16/EBPDT PNs amplitude SM%02d G16 L1", ism);
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser2/Gain16/EBPDT PNs amplitude SM%02d G16 L2", ism);
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser3/Gain16/EBPDT PNs amplitude SM%02d G16 L3", ism);
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser4/Gain16/EBPDT PNs amplitude SM%02d G16 L4", ism);
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser1/Gain16/EBPDT PNs pedestal SM%02d G16 L1", ism);
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser2/Gain16/EBPDT PNs pedestal SM%02d G16 L2", ism);
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser3/Gain16/EBPDT PNs pedestal SM%02d G16 L3", ism);
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser4/Gain16/EBPDT PNs pedestal SM%02d G16 L4", ism);
    mui_->unsubscribe(histo, ism);

  }

}

void EBLaserClient::softReset(void){

}

void EBLaserClient::analyze(void){

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( verbose_ ) cout << "EBLaserClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;
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

  EcalErrorMask::fetchDataSet(&mask1);
  EcalErrorMask::fetchDataSet(&mask2);

  Char_t histo[200];

  MonitorElement* me;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBLaserTask/Laser1/EBLT amplitude SM%02d L1A", ism);
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBLaserTask/Laser1/EBLT amplitude SM%02d L1A").c_str(), ism);
    }
    me = mui_->get(histo);
    h01_[ism-1] = EBMUtilsClient::getHisto<TProfile2D*>( me, cloneME_, h01_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBLaserTask/Laser1/EBLT amplitude over PN SM%02d L1A", ism);
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBLaserTask/Laser1/EBLT amplitude over PN SM%02d L1A").c_str(), ism);
    }
    me = mui_->get(histo);
    h02_[ism-1] = EBMUtilsClient::getHisto<TProfile2D*>( me, cloneME_, h02_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBLaserTask/Laser2/EBLT amplitude SM%02d L2A", ism);
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBLaserTask/Laser2/EBLT amplitude SM%02d L2A").c_str(), ism);
    }
    me = mui_->get(histo);
    h03_[ism-1] = EBMUtilsClient::getHisto<TProfile2D*>( me, cloneME_, h03_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBLaserTask/Laser2/EBLT amplitude over PN SM%02d L2A", ism);
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBLaserTask/Laser2/EBLT amplitude over PN SM%02d L2A").c_str(), ism);
    }
    me = mui_->get(histo);
    h04_[ism-1] = EBMUtilsClient::getHisto<TProfile2D*>( me, cloneME_, h04_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBLaserTask/Laser3/EBLT amplitude SM%02d L3A", ism);
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBLaserTask/Laser3/EBLT amplitude SM%02d L3A").c_str(), ism);
    }
    me = mui_->get(histo);
    h05_[ism-1] = EBMUtilsClient::getHisto<TProfile2D*>( me, cloneME_, h05_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBLaserTask/Laser3/EBLT amplitude over PN SM%02d L3A", ism);
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBLaserTask/Laser3/EBLT amplitude over PN SM%02d L3A").c_str(), ism);
    }
    me = mui_->get(histo);
    h06_[ism-1] = EBMUtilsClient::getHisto<TProfile2D*>( me, cloneME_, h06_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBLaserTask/Laser4/EBLT amplitude SM%02d L4A", ism);
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBLaserTask/Laser4/EBLT amplitude SM%02d L4A").c_str(), ism);
    }
    me = mui_->get(histo);
    h07_[ism-1] = EBMUtilsClient::getHisto<TProfile2D*>( me, cloneME_, h07_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBLaserTask/Laser4/EBLT amplitude over PN SM%02d L4A", ism);
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBLaserTask/Laser4/EBLT amplitude over PN SM%02d L4A").c_str(), ism);
    }
    me = mui_->get(histo);
    h08_[ism-1] = EBMUtilsClient::getHisto<TProfile2D*>( me, cloneME_, h08_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBLaserTask/Laser1/EBLT timing SM%02d L1A", ism);
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBLaserTask/Laser1/EBLT timing SM%02d L1A").c_str(), ism);
    }
    me = mui_->get(histo);
    h09_[ism-1] = EBMUtilsClient::getHisto<TProfile2D*>( me, cloneME_, h09_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBLaserTask/Laser2/EBLT timing SM%02d L2A", ism);
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBLaserTask/Laser2/EBLT timing SM%02d L2A").c_str(), ism);
    }
    me = mui_->get(histo);
    h10_[ism-1] = EBMUtilsClient::getHisto<TProfile2D*>( me, cloneME_, h10_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBLaserTask/Laser3/EBLT timing SM%02d L3A", ism);
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBLaserTask/Laser3/EBLT timing SM%02d L3A").c_str(), ism);
    }
    me = mui_->get(histo);
    h11_[ism-1] = EBMUtilsClient::getHisto<TProfile2D*>( me, cloneME_, h11_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBLaserTask/Laser4/EBLT timing SM%02d L4A", ism);
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBLaserTask/Laser4/EBLT timing SM%02d L4A").c_str(), ism);
    }
    me = mui_->get(histo);
    h12_[ism-1] = EBMUtilsClient::getHisto<TProfile2D*>( me, cloneME_, h12_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBLaserTask/Laser1/EBLT amplitude SM%02d L1B", ism);
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBLaserTask/Laser1/EBLT amplitude SM%02d L1B").c_str(), ism);
    }
    me = mui_->get(histo);
    h13_[ism-1] = EBMUtilsClient::getHisto<TProfile2D*>( me, cloneME_, h13_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBLaserTask/Laser1/EBLT amplitude over PN SM%02d L1B", ism);
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBLaserTask/Laser1/EBLT amplitude over PN SM%02d L1B").c_str(), ism);
    }
    me = mui_->get(histo);
    h14_[ism-1] = EBMUtilsClient::getHisto<TProfile2D*>( me, cloneME_, h14_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBLaserTask/Laser2/EBLT amplitude SM%02d L2B", ism);
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBLaserTask/Laser2/EBLT amplitude SM%02d L2B").c_str(), ism);
    }
    me = mui_->get(histo);
    h15_[ism-1] = EBMUtilsClient::getHisto<TProfile2D*>( me, cloneME_, h15_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBLaserTask/Laser2/EBLT amplitude over PN SM%02d L2B", ism);
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBLaserTask/Laser2/EBLT amplitude over PN SM%02d L2B").c_str(), ism);
    }
    me = mui_->get(histo);
    h16_[ism-1] = EBMUtilsClient::getHisto<TProfile2D*>( me, cloneME_, h16_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBLaserTask/Laser3/EBLT amplitude SM%02d L3B", ism);
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBLaserTask/Laser3/EBLT amplitude SM%02d L3B").c_str(), ism);
    }
    me = mui_->get(histo);
    h17_[ism-1] = EBMUtilsClient::getHisto<TProfile2D*>( me, cloneME_, h17_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBLaserTask/Laser3/EBLT amplitude over PN SM%02d L3B", ism);
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBLaserTask/Laser3/EBLT amplitude over PN SM%02d L3B").c_str(), ism);
    }
    me = mui_->get(histo);
    h18_[ism-1] = EBMUtilsClient::getHisto<TProfile2D*>( me, cloneME_, h18_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBLaserTask/Laser4/EBLT amplitude SM%02d L4B", ism);
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBLaserTask/Laser4/EBLT amplitude SM%02d L4B").c_str(), ism);
    }
    me = mui_->get(histo);
    h19_[ism-1] = EBMUtilsClient::getHisto<TProfile2D*>( me, cloneME_, h19_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBLaserTask/Laser4/EBLT amplitude over PN SM%02d L4B", ism);
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBLaserTask/Laser4/EBLT amplitude over PN SM%02d L4B").c_str(), ism);
    }
    me = mui_->get(histo);
    h20_[ism-1] = EBMUtilsClient::getHisto<TProfile2D*>( me, cloneME_, h20_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBLaserTask/Laser1/EBLT timing SM%02d L1B", ism);
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBLaserTask/Laser1/EBLT timing SM%02d L1B").c_str(), ism);
    }
    me = mui_->get(histo);
    h21_[ism-1] = EBMUtilsClient::getHisto<TProfile2D*>( me, cloneME_, h21_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBLaserTask/Laser2/EBLT timing SM%02d L2B", ism);
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBLaserTask/Laser2/EBLT timing SM%02d L2B").c_str(), ism);
    }
    me = mui_->get(histo);
    h22_[ism-1] = EBMUtilsClient::getHisto<TProfile2D*>( me, cloneME_, h22_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBLaserTask/Laser3/EBLT timing SM%02d L3B", ism);
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBLaserTask/Laser3/EBLT timing SM%02d L3B").c_str(), ism);
    }
    me = mui_->get(histo);
    h23_[ism-1] = EBMUtilsClient::getHisto<TProfile2D*>( me, cloneME_, h23_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBLaserTask/Laser4/EBLT timing SM%02d L4B", ism);
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBLaserTask/Laser4/EBLT timing SM%02d L4B").c_str(), ism);
    }
    me = mui_->get(histo);
    h24_[ism-1] = EBMUtilsClient::getHisto<TProfile2D*>( me, cloneME_, h24_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBLaserTask/Laser1/EBLT shape SM%02d L1A", ism);
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBLaserTask/Laser1/EBLT shape SM%02d L1A").c_str(), ism);
    }
    me = mui_->get(histo);
    hs01_[ism-1] = EBMUtilsClient::getHisto<TProfile2D*>( me, cloneME_, hs01_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBLaserTask/Laser2/EBLT shape SM%02d L2A", ism);
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBLaserTask/Laser2/EBLT shape SM%02d L2A").c_str(), ism);
    }
    me = mui_->get(histo);
    hs02_[ism-1] = EBMUtilsClient::getHisto<TProfile2D*>( me, cloneME_, hs02_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBLaserTask/Laser3/EBLT shape SM%02d L3A", ism);
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBLaserTask/Laser3/EBLT shape SM%02d L3A").c_str(), ism);
    }
    me = mui_->get(histo);
    hs03_[ism-1] = EBMUtilsClient::getHisto<TProfile2D*>( me, cloneME_, hs03_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBLaserTask/Laser4/EBLT shape SM%02d L4A", ism);
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBLaserTask/Laser4/EBLT shape SM%02d L4A").c_str(), ism);
    }
    me = mui_->get(histo);
    hs04_[ism-1] = EBMUtilsClient::getHisto<TProfile2D*>( me, cloneME_, hs04_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBLaserTask/Laser1/EBLT shape SM%02d L1B", ism);
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBLaserTask/Laser1/EBLT shape SM%02d L1B").c_str(), ism);
    }
    me = mui_->get(histo);
    hs05_[ism-1] = EBMUtilsClient::getHisto<TProfile2D*>( me, cloneME_, hs05_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBLaserTask/Laser2/EBLT shape SM%02d L2B", ism);
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBLaserTask/Laser2/EBLT shape SM%02d L2B").c_str(), ism);
    }
    me = mui_->get(histo);
    hs06_[ism-1] = EBMUtilsClient::getHisto<TProfile2D*>( me, cloneME_, hs06_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBLaserTask/Laser3/EBLT shape SM%02d L3B", ism);
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBLaserTask/Laser3/EBLT shape SM%02d L3B").c_str(), ism);
    }
    me = mui_->get(histo);
    hs07_[ism-1] = EBMUtilsClient::getHisto<TProfile2D*>( me, cloneME_, hs07_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBLaserTask/Laser4/EBLT shape SM%02d L4B", ism);
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBLaserTask/Laser4/EBLT shape SM%02d L4B").c_str(), ism);
    }
    me = mui_->get(histo);
    hs08_[ism-1] = EBMUtilsClient::getHisto<TProfile2D*>( me, cloneME_, hs08_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser1/Gain01/EBPDT PNs amplitude SM%02d G01 L1", ism);
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBPnDiodeTask/Laser1/Gain01/EBPDT PNs amplitude SM%02d G01 L1").c_str(), ism);
    }
    me = mui_->get(histo);
    i01_[ism-1] = EBMUtilsClient::getHisto<TProfile2D*>( me, cloneME_, i01_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser2/Gain01/EBPDT PNs amplitude SM%02d G01 L2", ism);
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBPnDiodeTask/Laser2/Gain01/EBPDT PNs amplitude SM%02d G01 L2").c_str(), ism);
    }
    me = mui_->get(histo);
    i02_[ism-1] = EBMUtilsClient::getHisto<TProfile2D*>( me, cloneME_, i02_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser3/Gain01/EBPDT PNs amplitude SM%02d G01 L3", ism);
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBPnDiodeTask/Laser3/Gain01/EBPDT PNs amplitude SM%02d G01 L3").c_str(), ism);
    }
    me = mui_->get(histo);
    i03_[ism-1] = EBMUtilsClient::getHisto<TProfile2D*>( me, cloneME_, i03_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser4/Gain01/EBPDT PNs amplitude SM%02d G01 L4", ism);
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBPnDiodeTask/Laser4/Gain01/EBPDT PNs amplitude SM%02d G01 L4").c_str(), ism);
    }
    me = mui_->get(histo);
    i04_[ism-1] = EBMUtilsClient::getHisto<TProfile2D*>( me, cloneME_, i04_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser1/Gain01/EBPDT PNs pedestal SM%02d G01 L1", ism);
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBPnDiodeTask/Laser1/Gain01/EBPDT PNs pedestal SM%02d G01 L1").c_str(), ism);
    }
    me = mui_->get(histo);
    i05_[ism-1] = EBMUtilsClient::getHisto<TProfile2D*>( me, cloneME_, i05_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser2/Gain01/EBPDT PNs pedestal SM%02d G01 L2", ism);
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBPnDiodeTask/Laser2/Gain01/EBPDT PNs pedestal SM%02d G01 L2").c_str(), ism);
    }
    me = mui_->get(histo);
    i06_[ism-1] = EBMUtilsClient::getHisto<TProfile2D*>( me, cloneME_, i06_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser3/Gain01/EBPDT PNs pedestal SM%02d G01 L3", ism);
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBPnDiodeTask/Laser3/Gain01/EBPDT PNs pedestal SM%02d G01 L3").c_str(), ism);
    }
    me = mui_->get(histo);
    i07_[ism-1] = EBMUtilsClient::getHisto<TProfile2D*>( me, cloneME_, i07_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser4/Gain01/EBPDT PNs pedestal SM%02d G01 L4", ism);
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBPnDiodeTask/Laser4/Gain01/EBPDT PNs pedestal SM%02d G01 L4").c_str(), ism);
    }
    me = mui_->get(histo);
    i08_[ism-1] = EBMUtilsClient::getHisto<TProfile2D*>( me, cloneME_, i08_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser1/Gain16/EBPDT PNs amplitude SM%02d G16 L1", ism);
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBPnDiodeTask/Laser1/Gain16/EBPDT PNs amplitude SM%02d G16 L1").c_str(), ism);
    }
    me = mui_->get(histo);
    i09_[ism-1] = EBMUtilsClient::getHisto<TProfile2D*>( me, cloneME_, i09_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser2/Gain16/EBPDT PNs amplitude SM%02d G16 L2", ism);
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBPnDiodeTask/Laser2/Gain16/EBPDT PNs amplitude SM%02d G16 L2").c_str(), ism);
    }
    me = mui_->get(histo);
    i10_[ism-1] = EBMUtilsClient::getHisto<TProfile2D*>( me, cloneME_, i10_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser3/Gain16/EBPDT PNs amplitude SM%02d G16 L3", ism);
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBPnDiodeTask/Laser3/Gain16/EBPDT PNs amplitude SM%02d G16 L3").c_str(), ism);
    }
    me = mui_->get(histo);
    i11_[ism-1] = EBMUtilsClient::getHisto<TProfile2D*>( me, cloneME_, i11_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser4/Gain16/EBPDT PNs amplitude SM%02d G16 L4", ism);
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBPnDiodeTask/Laser4/Gain16/EBPDT PNs amplitude SM%02d G16 L4").c_str(), ism);
    }
    me = mui_->get(histo);
    i12_[ism-1] = EBMUtilsClient::getHisto<TProfile2D*>( me, cloneME_, i12_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser1/Gain16/EBPDT PNs pedestal SM%02d G16 L1", ism);
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBPnDiodeTask/Laser1/Gain16/EBPDT PNs pedestal SM%02d G16 L1").c_str(), ism);
    }
    me = mui_->get(histo);
    i13_[ism-1] = EBMUtilsClient::getHisto<TProfile2D*>( me, cloneME_, i13_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser2/Gain16/EBPDT PNs pedestal SM%02d G16 L2", ism);
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBPnDiodeTask/Laser2/Gain16/EBPDT PNs pedestal SM%02d G16 L2").c_str(), ism);
    }
    me = mui_->get(histo);
    i14_[ism-1] = EBMUtilsClient::getHisto<TProfile2D*>( me, cloneME_, i14_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser3/Gain16/EBPDT PNs pedestal SM%02d G16 L3", ism);
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBPnDiodeTask/Laser3/Gain16/EBPDT PNs pedestal SM%02d G16 L3").c_str(), ism);
    }
    me = mui_->get(histo);
    i15_[ism-1] = EBMUtilsClient::getHisto<TProfile2D*>( me, cloneME_, i15_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser4/Gain16/EBPDT PNs pedestal SM%02d G16 L4", ism);
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBPnDiodeTask/Laser4/Gain16/EBPDT PNs pedestal SM%02d G16 L4").c_str(), ism);
    }
    me = mui_->get(histo);
    i16_[ism-1] = EBMUtilsClient::getHisto<TProfile2D*>( me, cloneME_, i16_[ism-1] );

    EBMUtilsClient::resetHisto( meg01_[ism-1] );
    EBMUtilsClient::resetHisto( meg02_[ism-1] );
    EBMUtilsClient::resetHisto( meg03_[ism-1] );
    EBMUtilsClient::resetHisto( meg04_[ism-1] );

    EBMUtilsClient::resetHisto( meg05_[ism-1] );
    EBMUtilsClient::resetHisto( meg06_[ism-1] );
    EBMUtilsClient::resetHisto( meg07_[ism-1] );
    EBMUtilsClient::resetHisto( meg08_[ism-1] );
    EBMUtilsClient::resetHisto( meg09_[ism-1] );
    EBMUtilsClient::resetHisto( meg10_[ism-1] );
    EBMUtilsClient::resetHisto( meg11_[ism-1] );
    EBMUtilsClient::resetHisto( meg12_[ism-1] );

    EBMUtilsClient::resetHisto( mea01_[ism-1] );
    EBMUtilsClient::resetHisto( mea02_[ism-1] );
    EBMUtilsClient::resetHisto( mea03_[ism-1] );
    EBMUtilsClient::resetHisto( mea04_[ism-1] );
    EBMUtilsClient::resetHisto( mea05_[ism-1] );
    EBMUtilsClient::resetHisto( mea06_[ism-1] );
    EBMUtilsClient::resetHisto( mea07_[ism-1] );
    EBMUtilsClient::resetHisto( mea08_[ism-1] );

    EBMUtilsClient::resetHisto( met01_[ism-1] );
    EBMUtilsClient::resetHisto( met02_[ism-1] );
    EBMUtilsClient::resetHisto( met03_[ism-1] );
    EBMUtilsClient::resetHisto( met04_[ism-1] );
    EBMUtilsClient::resetHisto( met05_[ism-1] );
    EBMUtilsClient::resetHisto( met06_[ism-1] );
    EBMUtilsClient::resetHisto( met07_[ism-1] );
    EBMUtilsClient::resetHisto( met08_[ism-1] );

    EBMUtilsClient::resetHisto( meaopn01_[ism-1] );
    EBMUtilsClient::resetHisto( meaopn02_[ism-1] );
    EBMUtilsClient::resetHisto( meaopn03_[ism-1] );
    EBMUtilsClient::resetHisto( meaopn04_[ism-1] );
    EBMUtilsClient::resetHisto( meaopn05_[ism-1] );
    EBMUtilsClient::resetHisto( meaopn06_[ism-1] );
    EBMUtilsClient::resetHisto( meaopn07_[ism-1] );
    EBMUtilsClient::resetHisto( meaopn08_[ism-1] );

    float meanAmplL1A, meanAmplL2A, meanAmplL3A, meanAmplL4A;
    float meanAmplL1B, meanAmplL2B, meanAmplL3B, meanAmplL4B;

    int nCryL1A, nCryL2A, nCryL3A, nCryL4A;
    int nCryL1B, nCryL2B, nCryL3B, nCryL4B;

    meanAmplL1A = meanAmplL2A = meanAmplL3A = meanAmplL4A = 0.;
    meanAmplL1B = meanAmplL2B = meanAmplL3B = meanAmplL4B = 0.;

    nCryL1A = nCryL2A = nCryL3A = nCryL4A = 0;
    nCryL1B = nCryL2B = nCryL3B = nCryL4B = 0;

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

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

        update01 = EBMUtilsClient::getBinStats(h01_[ism-1], ie, ip, num01, mean01, rms01);
        update02 = EBMUtilsClient::getBinStats(h03_[ism-1], ie, ip, num02, mean02, rms02);
        update03 = EBMUtilsClient::getBinStats(h05_[ism-1], ie, ip, num03, mean03, rms03);
        update04 = EBMUtilsClient::getBinStats(h07_[ism-1], ie, ip, num04, mean04, rms04);
        update05 = EBMUtilsClient::getBinStats(h13_[ism-1], ie, ip, num05, mean05, rms05);
        update06 = EBMUtilsClient::getBinStats(h15_[ism-1], ie, ip, num06, mean06, rms06);
        update07 = EBMUtilsClient::getBinStats(h17_[ism-1], ie, ip, num07, mean07, rms07);
        update08 = EBMUtilsClient::getBinStats(h19_[ism-1], ie, ip, num08, mean08, rms08);

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

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

        if ( meg01_[ism-1] ) meg01_[ism-1]->setBinContent( ie, ip, 2.);
        if ( meg02_[ism-1] ) meg02_[ism-1]->setBinContent( ie, ip, 2.);
        if ( meg03_[ism-1] ) meg03_[ism-1]->setBinContent( ie, ip, 2.);
        if ( meg04_[ism-1] ) meg04_[ism-1]->setBinContent( ie, ip, 2.);

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

        float num01, num02, num03, num04, num05, num06, num07, num08;
        float num09, num10, num11, num12;
        float mean01, mean02, mean03, mean04, mean05, mean06, mean07, mean08;
        float mean09, mean10, mean11, mean12;
        float rms01, rms02, rms03, rms04, rms05, rms06, rms07, rms08;
        float rms09, rms10, rms11, rms12;

        update01 = EBMUtilsClient::getBinStats(h01_[ism-1], ie, ip, num01, mean01, rms01);
        update02 = EBMUtilsClient::getBinStats(h02_[ism-1], ie, ip, num02, mean02, rms02);
        update03 = EBMUtilsClient::getBinStats(h03_[ism-1], ie, ip, num03, mean03, rms03);
        update04 = EBMUtilsClient::getBinStats(h04_[ism-1], ie, ip, num04, mean04, rms04);
        update05 = EBMUtilsClient::getBinStats(h05_[ism-1], ie, ip, num05, mean05, rms05);
        update06 = EBMUtilsClient::getBinStats(h06_[ism-1], ie, ip, num06, mean06, rms06);
        update07 = EBMUtilsClient::getBinStats(h07_[ism-1], ie, ip, num07, mean07, rms07);
        update08 = EBMUtilsClient::getBinStats(h08_[ism-1], ie, ip, num08, mean08, rms08);
        update09 = EBMUtilsClient::getBinStats(h09_[ism-1], ie, ip, num09, mean09, rms09);
        update10 = EBMUtilsClient::getBinStats(h10_[ism-1], ie, ip, num10, mean10, rms10);
        update11 = EBMUtilsClient::getBinStats(h11_[ism-1], ie, ip, num11, mean11, rms11);
        update12 = EBMUtilsClient::getBinStats(h12_[ism-1], ie, ip, num12, mean12, rms12);

        if ( ! update01 )
          update01 = EBMUtilsClient::getBinStats(h13_[ism-1], ie, ip, num01, mean01, rms01);
        if ( ! update02 )
          update02 = EBMUtilsClient::getBinStats(h14_[ism-1], ie, ip, num02, mean02, rms02);
        if ( ! update03 )
          update03 = EBMUtilsClient::getBinStats(h15_[ism-1], ie, ip, num03, mean03, rms03);
        if ( ! update04 )
          update04 = EBMUtilsClient::getBinStats(h16_[ism-1], ie, ip, num04, mean04, rms04);
        if ( ! update05 )
          update05 = EBMUtilsClient::getBinStats(h17_[ism-1], ie, ip, num05, mean05, rms05);
        if ( ! update06 )
          update06 = EBMUtilsClient::getBinStats(h18_[ism-1], ie, ip, num06, mean06, rms06);
        if ( ! update07 )
          update07 = EBMUtilsClient::getBinStats(h19_[ism-1], ie, ip, num07, mean07, rms07);
        if ( ! update08 )
          update08 = EBMUtilsClient::getBinStats(h20_[ism-1], ie, ip, num08, mean08, rms08);
        if ( ! update09 )
          update09 = EBMUtilsClient::getBinStats(h21_[ism-1], ie, ip, num09, mean09, rms09);
        if ( ! update10 )
          update10 = EBMUtilsClient::getBinStats(h22_[ism-1], ie, ip, num10, mean10, rms10);
        if ( ! update11 )
          update11 = EBMUtilsClient::getBinStats(h23_[ism-1], ie, ip, num11, mean11, rms11);
        if ( ! update12 )
          update12 = EBMUtilsClient::getBinStats(h24_[ism-1], ie, ip, num12, mean12, rms12);

        if ( update01 ) {

          float val;

          if ( ie < 6 || ip > 10 ) {

            val = 1.;
            if ( fabs(mean01 - meanAmplL1A) > fabs(percentVariation_ * meanAmplL1A) )
              val = 0.;
            if ( meg01_[ism-1] ) meg01_[ism-1]->setBinContent( ie, ip, val );

            if ( mea01_[ism-1] ) {
              if ( mean01 > 0. ) {
                mea01_[ism-1]->setBinContent( ip+20*(ie-1), mean01 );
                mea01_[ism-1]->setBinError( ip+20*(ie-1), rms01 );
              } else {
                mea01_[ism-1]->setEntries( 1.+mea01_[ism-1]->getEntries() );
              }
            }

          } else {

            val = 1.;
            if ( fabs(mean01 - meanAmplL1B) > fabs(percentVariation_ * meanAmplL1B) )
              val = 0.;
            if ( meg01_[ism-1] ) meg01_[ism-1]->setBinContent( ie, ip, val );

            if ( mea05_[ism-1] ) {
              if ( mean01 > 0. ) {
                mea05_[ism-1]->setBinContent( ip+20*(ie-1), mean01 );
                mea05_[ism-1]->setBinError( ip+20*(ie-1), rms01 );
              } else {
                mea05_[ism-1]->setEntries( 1.+mea05_[ism-1]->getEntries() );
              }
            }

          }

        }

        if ( update03 ) {

          float val;

          if ( ie < 6 || ip > 10 ) {

            val = 1.;
            if ( fabs(mean03 - meanAmplL2A) > fabs(percentVariation_ * meanAmplL2A) )
              val = 0.;
            if ( meg02_[ism-1] ) meg02_[ism-1]->setBinContent( ie, ip, val);

            if ( mea02_[ism-1] ) {
              if ( mean03 > 0. ) {
                mea02_[ism-1]->setBinContent( ip+20*(ie-1), mean03 );
                mea02_[ism-1]->setBinError( ip+20*(ie-1), rms03 );
              } else {
                mea02_[ism-1]->setEntries( 1.+mea02_[ism-1]->getEntries() );
              }
            }

          } else {

            val = 1.;
            if ( fabs(mean03 - meanAmplL2B) > fabs(percentVariation_ * meanAmplL2B) )
              val = 0.;
            if ( meg02_[ism-1] ) meg02_[ism-1]->setBinContent( ie, ip, val);

            if ( mea06_[ism-1] ) {
              if ( mean03 > 0. ) {
                mea06_[ism-1]->setBinContent( ip+20*(ie-1), mean03 );
                mea06_[ism-1]->setBinError( ip+20*(ie-1), rms03 ); 
              } else {
                mea06_[ism-1]->setEntries( 1.+mea06_[ism-1]->getEntries() );
              }
            }

          }

        }

        if ( update05 ) {

          float val;

          if ( ie < 6 || ip > 10 ) {

            val = 1.;
            if ( fabs(mean05 - meanAmplL3A) > fabs(percentVariation_ * meanAmplL3A) )
              val = 0.;
            if ( meg03_[ism-1] ) meg03_[ism-1]->setBinContent( ie, ip, val );

            if ( mea03_[ism-1] ) {
              if ( mean05 > 0. ) {
                mea03_[ism-1]->setBinContent( ip+20*(ie-1), mean05 );
                mea03_[ism-1]->setBinError( ip+20*(ie-1), rms05 ); 
              } else {
                mea03_[ism-1]->setEntries( 1.+mea03_[ism-1]->getEntries() );
              }
            }

          } else {

            val = 1.;
            if ( fabs(mean05 - meanAmplL3B) > fabs(percentVariation_ * meanAmplL3B) )
              val = 0.;
            if ( meg03_[ism-1] ) meg03_[ism-1]->setBinContent( ie, ip, val );

            if ( mea07_[ism-1] ) {
              if ( mean05 > 0. ) {
                mea07_[ism-1]->setBinContent( ip+20*(ie-1), mean05 );
                mea07_[ism-1]->setBinError( ip+20*(ie-1), rms05 );
              } else {
                mea07_[ism-1]->setEntries( 1.+mea07_[ism-1]->getEntries() );
              }
            }

          }

        }

        if ( update07 ) {

          float val;

          if ( ie < 6 || ip > 10 ) {

            val = 1.;
            if ( fabs(mean07 - meanAmplL4A) > fabs(percentVariation_ * meanAmplL4A) )
              val = 0.;
            if ( meg04_[ism-1] ) meg04_[ism-1]->setBinContent( ie, ip, val );

            if ( mea04_[ism-1] ) mea04_[ism-1]->setBinContent( ip+20*(ie-1), mean07 );
            if ( mea04_[ism-1] ) mea04_[ism-1]->setBinError( ip+20*(ie-1), rms07 );

          } else {

            val = 1.;
            if ( fabs(mean07 - meanAmplL4B) > fabs(percentVariation_ * meanAmplL4B) )
              val = 0.;
            if ( meg04_[ism-1] ) meg04_[ism-1]->setBinContent( ie, ip, val );

            if ( mea08_[ism-1] ) {
              if ( mean07 > 0. ) {
                mea08_[ism-1]->setBinContent( ip+20*(ie-1), mean07 );
                mea08_[ism-1]->setBinError( ip+20*(ie-1), rms07 );
              } else {
                mea08_[ism-1]->setEntries( 1.+mea08_[ism-1]->getEntries() );
              }
            }

          }

        }

        if ( update02 ) {

          if ( ie < 6 || ip > 10 ) {

            if ( meaopn01_[ism-1] ) {
              if ( mean02 > 0. ) {
                meaopn01_[ism-1]->setBinContent( ip+20*(ie-1), mean02 );
                meaopn01_[ism-1]->setBinError( ip+20*(ie-1), rms02 );
              } else {
                meaopn01_[ism-1]->setEntries( 1.+meaopn01_[ism-1]->getEntries() );
              }
            }

          } else {


            if ( meaopn05_[ism-1] ) {
              if ( mean02 > 0. ) {
                meaopn05_[ism-1]->setBinContent( ip+20*(ie-1), mean02 );
                meaopn05_[ism-1]->setBinError( ip+20*(ie-1), rms02 );
              } else {
                meaopn05_[ism-1]->setEntries( 1.+meaopn05_[ism-1]->getEntries() );
              }
            }

          }

        }

        if ( update04 ) {

          if ( ie < 6 || ip > 10 ) {

            if ( meaopn02_[ism-1] ) {
              if ( mean04 > 0. ) {
                meaopn02_[ism-1]->setBinContent( ip+20*(ie-1), mean04 );
                meaopn02_[ism-1]->setBinError( ip+20*(ie-1), rms04 );
              } else {
                meaopn02_[ism-1]->setEntries( 1.+meaopn02_[ism-1]->getEntries() );
              }
            }

          } else {

            if ( meaopn06_[ism-1] ) {
              if ( mean04 > 0. ) {
                meaopn06_[ism-1]->setBinContent( ip+20*(ie-1), mean04 );
                meaopn06_[ism-1]->setBinError( ip+20*(ie-1), rms04 );
              } else {
                meaopn06_[ism-1]->setEntries( 1.+meaopn06_[ism-1]->getEntries() );
              }
            }

          }

        }

        if ( update06 ) {

          if ( ie < 6 || ip > 10 ) {

            if ( meaopn03_[ism-1] ) {
              if ( mean06 > 0. ) {
                meaopn03_[ism-1]->setBinContent( ip+20*(ie-1), mean06 );
                meaopn03_[ism-1]->setBinError( ip+20*(ie-1), rms06 );
              } else {
                meaopn03_[ism-1]->setEntries( 1.+meaopn03_[ism-1]->getEntries() );
              }
            }

          } else {

            if ( meaopn07_[ism-1] ) {
              if ( mean06 > 0. ) {
                meaopn07_[ism-1]->setBinContent( ip+20*(ie-1), mean06 );
                meaopn07_[ism-1]->setBinError( ip+20*(ie-1), rms06 );
              } else {
                meaopn07_[ism-1]->setEntries( 1.+meaopn07_[ism-1]->getEntries() );
              }
            }

          }

        }

        if ( update08 ) {

          if ( ie < 6 || ip > 10 ) {

            if ( meaopn04_[ism-1] ) {
              if ( mean08 > 0. ) {
                meaopn04_[ism-1]->setBinContent( ip+20*(ie-1), mean08 );
                meaopn04_[ism-1]->setBinError( ip+20*(ie-1), rms08 );
              } else {
                meaopn04_[ism-1]->setEntries( 1.+meaopn04_[ism-1]->getEntries() );
              }
            }

          } else {

            if ( meaopn08_[ism-1] ) {
              if ( mean08 > 0. ) {
                meaopn08_[ism-1]->setBinContent( ip+20*(ie-1), mean08 );
                meaopn08_[ism-1]->setBinError( ip+20*(ie-1), rms08 );
              } else {
                meaopn08_[ism-1]->setEntries( 1.+meaopn08_[ism-1]->getEntries() );
              }
            }

          }

        }

        if ( update09 ) {

          if ( ie < 6 || ip > 10 ) {

            if ( met01_[ism-1] ) {
              if ( mean09 > 0. ) {
                met01_[ism-1]->setBinContent( ip+20*(ie-1), mean09 );
                met01_[ism-1]->setBinError( ip+20*(ie-1), rms09 );
              } else {
                met01_[ism-1]->setEntries(1.+met01_[ism-1]->getEntries());
              }
            }

          } else {

            if ( met05_[ism-1] ) {
              if ( mean09 > 0. ) {
                met05_[ism-1]->setBinContent( ip+20*(ie-1), mean09 );
                met05_[ism-1]->setBinError( ip+20*(ie-1), rms09 );
              } else {
                met05_[ism-1]->setEntries(1.+met05_[ism-1]->getEntries());
              }
            }

          }

        }

        if ( update10 ) {

          if ( ie < 6 || ip > 10 ) {

            if ( met02_[ism-1] ) {
              if ( mean10 > 0. ) {
                met02_[ism-1]->setBinContent( ip+20*(ie-1), mean10 );
                met02_[ism-1]->setBinError( ip+20*(ie-1), rms10 );
              } else {
                met02_[ism-1]->setEntries(1.+met02_[ism-1]->getEntries());
              }
            }

          } else {

            if ( met06_[ism-1] ) {
              if ( mean10 > 0. ) {
                met06_[ism-1]->setBinContent( ip+20*(ie-1), mean10 );
                met06_[ism-1]->setBinError( ip+20*(ie-1), rms10 );
              } else {
                met06_[ism-1]->setEntries(1.+met06_[ism-1]->getEntries());
              }
            }

          }

        }

        if ( update11 ) {

          if ( ie < 6 || ip > 10 ) {

            if ( met03_[ism-1] ) {
              if ( mean11 > 0. ) {
                met03_[ism-1]->setBinContent( ip+20*(ie-1), mean11 );
                met03_[ism-1]->setBinError( ip+20*(ie-1), rms11 );
              } else {
                met03_[ism-1]->setEntries(1.+met03_[ism-1]->getEntries());
              }
            }

          } else {

            if ( met07_[ism-1] ) {
              if ( mean11 > 0. ) {
                met07_[ism-1]->setBinContent( ip+20*(ie-1), mean11 );
                met07_[ism-1]->setBinError( ip+20*(ie-1), rms11 );
              } else {
                met07_[ism-1]->setEntries(1.+met07_[ism-1]->getEntries());
              }
            }

          }

        }

        if ( update12 ) {

          if ( ie < 6 || ip > 10 ) {

            if ( met04_[ism-1] ) {
              if ( mean12 > 0. ) {
                met04_[ism-1]->setBinContent( ip+20*(ie-1), mean12 );
                met04_[ism-1]->setBinError( ip+20*(ie-1), rms12 );
              } else {
                met04_[ism-1]->setEntries(1.+met04_[ism-1]->getEntries());
              }
            }

          } else {

            if ( met08_[ism-1] ) {
              if ( mean12 > 0. ) {
                met08_[ism-1]->setBinContent( ip+20*(ie-1), mean12 );
                met08_[ism-1]->setBinError( ip+20*(ie-1), rms12 );
              } else {
                met08_[ism-1]->setEntries(1.+met08_[ism-1]->getEntries());
              }
            }

          }

        }

        // masking

        if ( mask1.size() != 0 ) {
          map<EcalLogicID, RunCrystalErrorsDat>::const_iterator m;
          for (m = mask1.begin(); m != mask1.end(); m++) {

            EcalLogicID ecid = m->first;

            int ic = (ip-1) + 20*(ie-1) + 1;

            if ( ecid.getID1() == ism && ecid.getID2() == ic ) {
              if ( (m->second).getErrorBits() & bits01 ) {
                if ( meg01_[ism-1] ) {
                  float val = int(meg01_[ism-1]->getBinContent(ie, ip)) % 3;
                  meg01_[ism-1]->setBinContent( ie, ip, val+3 );
                }
                if ( meg02_[ism-1] ) {
                  float val = int(meg02_[ism-1]->getBinContent(ie, ip)) % 3;
                  meg02_[ism-1]->setBinContent( ie, ip, val+3 );
                }
                if ( meg03_[ism-1] ) {
                  float val = int(meg03_[ism-1]->getBinContent(ie, ip)) % 3;
                  meg03_[ism-1]->setBinContent( ie, ip, val+3 );
                }
                if ( meg04_[ism-1] ) {
                  float val = int(meg04_[ism-1]->getBinContent(ie, ip)) % 3;
                  meg04_[ism-1]->setBinContent( ie, ip, val+3 );
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

      update01 = EBMUtilsClient::getBinStats(i01_[ism-1], 1, i, num01, mean01, rms01);
      update02 = EBMUtilsClient::getBinStats(i02_[ism-1], 1, i, num02, mean02, rms02);
      update03 = EBMUtilsClient::getBinStats(i03_[ism-1], 1, i, num03, mean03, rms03);
      update04 = EBMUtilsClient::getBinStats(i04_[ism-1], 1, i, num04, mean04, rms04);
      update05 = EBMUtilsClient::getBinStats(i05_[ism-1], 1, i, num05, mean05, rms05);
      update06 = EBMUtilsClient::getBinStats(i06_[ism-1], 1, i, num06, mean06, rms06);
      update07 = EBMUtilsClient::getBinStats(i07_[ism-1], 1, i, num07, mean07, rms07);
      update08 = EBMUtilsClient::getBinStats(i08_[ism-1], 1, i, num08, mean08, rms08);
      update09 = EBMUtilsClient::getBinStats(i09_[ism-1], 1, i, num09, mean09, rms09);
      update10 = EBMUtilsClient::getBinStats(i10_[ism-1], 1, i, num10, mean10, rms10);
      update11 = EBMUtilsClient::getBinStats(i11_[ism-1], 1, i, num11, mean11, rms11);
      update12 = EBMUtilsClient::getBinStats(i12_[ism-1], 1, i, num12, mean12, rms12);
      update13 = EBMUtilsClient::getBinStats(i13_[ism-1], 1, i, num13, mean13, rms13);
      update14 = EBMUtilsClient::getBinStats(i14_[ism-1], 1, i, num14, mean14, rms14);
      update15 = EBMUtilsClient::getBinStats(i15_[ism-1], 1, i, num15, mean15, rms15);
      update16 = EBMUtilsClient::getBinStats(i16_[ism-1], 1, i, num16, mean16, rms16);

      if ( update01 && update05 ) {

        float val;

        val = 1.;
        if ( mean01 < amplitudeThresholdPN_ )
          val = 0.;
        if ( mean05 < meanThresholdPN_ )
          val = 0.;
        if ( meg05_[ism-1] ) meg05_[ism-1]->setBinContent(i, 1, val);

      }

      if ( update02 && update06 ) {

        float val; 

        val = 1.;
        if ( mean02 < amplitudeThresholdPN_ )
          val = 0.;
        if ( mean06 < meanThresholdPN_ )
          val = 0.;
        if ( meg06_[ism-1] ) meg06_[ism-1]->setBinContent(i, 1, val); 

      }

      if ( update03 && update07 ) {

        float val;

        val = 1.;
        if ( mean03 < amplitudeThresholdPN_ )
          val = 0.;
        if ( mean07 < meanThresholdPN_ )
          val = 0.;
        if ( meg07_[ism-1] ) meg07_[ism-1]->setBinContent(i, 1, val);

      }

      if ( update04 && update08 ) {

        float val;

        val = 1.;
        if ( mean04 < amplitudeThresholdPN_ )
          val = 0.;
        if ( mean08 < meanThresholdPN_ )
          val = 0.;
        if ( meg08_[ism-1] ) meg08_[ism-1]->setBinContent(i, 1, val);

      }

      if ( update09 && update13 ) {

        float val;

        val = 1.;
        if ( mean09 < amplitudeThresholdPN_ )
          val = 0.;
        if ( mean13 < meanThresholdPN_ )
          val = 0.;
        if ( meg09_[ism-1] ) meg09_[ism-1]->setBinContent(i, 1, val); 

      }

      if ( update10 && update14 ) {

        float val;

        val = 1.;
        if ( mean10 < amplitudeThresholdPN_ )
          val = 0.;
        if ( mean14 < meanThresholdPN_ )
          val = 0.;
        if ( meg10_[ism-1] ) meg10_[ism-1]->setBinContent(i, 1, val); 

      }

      if ( update11 && update15 ) {

        float val;

        val = 1.;
        if ( mean11 < amplitudeThresholdPN_ )
          val = 0.;
        if ( mean15 < meanThresholdPN_ )
          val = 0.;
        if ( meg11_[ism-1] ) meg11_[ism-1]->setBinContent(i, 1, val);

      }

      if ( update12 && update16 ) {

        float val;

        val = 1.;
        if ( mean12 < amplitudeThresholdPN_ )
          val = 0.;
        if ( mean16 < meanThresholdPN_ )
          val = 0.;
        if ( meg12_[ism-1] ) meg12_[ism-1]->setBinContent(i, 1, val);

      }

      // masking

      if ( mask2.size() != 0 ) {
        map<EcalLogicID, RunPNErrorsDat>::const_iterator m;
        for (m = mask2.begin(); m != mask2.end(); m++) {

          EcalLogicID ecid = m->first;

          if ( ecid.getID1() == ism && ecid.getID2() == i-1 ) {
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
            if ( (m->second).getErrorBits() & (bits01|bits02) ) {
              if ( meg07_[ism-1] ) {
                float val = int(meg07_[ism-1]->getBinContent(i, 1)) % 3;
                meg07_[ism-1]->setBinContent( i, 1, val+3 );
              }
            }
            if ( (m->second).getErrorBits() & (bits01|bits02) ) {
              if ( meg08_[ism-1] ) {
                float val = int(meg08_[ism-1]->getBinContent(i, 1)) % 3;
                meg08_[ism-1]->setBinContent( i, 1, val+3 );
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
            if ( (m->second).getErrorBits() & (bits01|bits04) ) {
              if ( meg11_[ism-1] ) {
                float val = int(meg11_[ism-1]->getBinContent(i, 1)) % 3;
                meg11_[ism-1]->setBinContent( i, 1, val+3 );
              }
            }
            if ( (m->second).getErrorBits() & (bits01|bits04) ) {
              if ( meg12_[ism-1] ) {
                float val = int(meg12_[ism-1]->getBinContent(i, 1)) % 3;
                meg12_[ism-1]->setBinContent( i, 1, val+3 );
              }
            }
          }

        }
      }

    }

    vector<dqm::me_util::Channel> badChannels01;
    vector<dqm::me_util::Channel> badChannels02;
    vector<dqm::me_util::Channel> badChannels03;
    vector<dqm::me_util::Channel> badChannels04;
    vector<dqm::me_util::Channel> badChannels05;
    vector<dqm::me_util::Channel> badChannels06;
    vector<dqm::me_util::Channel> badChannels07;
    vector<dqm::me_util::Channel> badChannels08;

    if ( qth01_[ism-1] ) badChannels01 = qth01_[ism-1]->getBadChannels();
    if ( qth02_[ism-1] ) badChannels02 = qth02_[ism-1]->getBadChannels();
    if ( qth03_[ism-1] ) badChannels03 = qth03_[ism-1]->getBadChannels();
    if ( qth04_[ism-1] ) badChannels04 = qth04_[ism-1]->getBadChannels();
    if ( qth05_[ism-1] ) badChannels05 = qth05_[ism-1]->getBadChannels();
    if ( qth06_[ism-1] ) badChannels06 = qth06_[ism-1]->getBadChannels();
    if ( qth07_[ism-1] ) badChannels07 = qth07_[ism-1]->getBadChannels();
    if ( qth08_[ism-1] ) badChannels08 = qth08_[ism-1]->getBadChannels();

    vector<dqm::me_util::Channel> badChannels09;
    vector<dqm::me_util::Channel> badChannels10;
    vector<dqm::me_util::Channel> badChannels11;
    vector<dqm::me_util::Channel> badChannels12;
    vector<dqm::me_util::Channel> badChannels13;
    vector<dqm::me_util::Channel> badChannels14;
    vector<dqm::me_util::Channel> badChannels15;
    vector<dqm::me_util::Channel> badChannels16;
    vector<dqm::me_util::Channel> badChannels17;
    vector<dqm::me_util::Channel> badChannels18;
    vector<dqm::me_util::Channel> badChannels19;
    vector<dqm::me_util::Channel> badChannels20;
    vector<dqm::me_util::Channel> badChannels21;
    vector<dqm::me_util::Channel> badChannels22;
    vector<dqm::me_util::Channel> badChannels23;
    vector<dqm::me_util::Channel> badChannels24;

    if ( qth09_[ism-1] ) badChannels09 = qth09_[ism-1]->getBadChannels();
    if ( qth10_[ism-1] ) badChannels10 = qth10_[ism-1]->getBadChannels();
    if ( qth11_[ism-1] ) badChannels11 = qth11_[ism-1]->getBadChannels();
    if ( qth12_[ism-1] ) badChannels12 = qth12_[ism-1]->getBadChannels();
    if ( qth13_[ism-1] ) badChannels13 = qth13_[ism-1]->getBadChannels();
    if ( qth14_[ism-1] ) badChannels14 = qth14_[ism-1]->getBadChannels();
    if ( qth15_[ism-1] ) badChannels15 = qth15_[ism-1]->getBadChannels();
    if ( qth16_[ism-1] ) badChannels16 = qth16_[ism-1]->getBadChannels();
    if ( qth17_[ism-1] ) badChannels09 = qth17_[ism-1]->getBadChannels();
    if ( qth18_[ism-1] ) badChannels10 = qth18_[ism-1]->getBadChannels();
    if ( qth19_[ism-1] ) badChannels11 = qth19_[ism-1]->getBadChannels();
    if ( qth20_[ism-1] ) badChannels12 = qth20_[ism-1]->getBadChannels();
    if ( qth21_[ism-1] ) badChannels13 = qth21_[ism-1]->getBadChannels();
    if ( qth22_[ism-1] ) badChannels14 = qth22_[ism-1]->getBadChannels();
    if ( qth23_[ism-1] ) badChannels15 = qth23_[ism-1]->getBadChannels();
    if ( qth24_[ism-1] ) badChannels16 = qth24_[ism-1]->getBadChannels();

  }

}

void EBLaserClient::htmlOutput(int run, string htmlDir, string htmlName){

  cout << "Preparing EBLaserClient html output ..." << endl;

  ofstream htmlFile;

  htmlFile.open((htmlDir + htmlName).c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlFile << "<html>  " << endl;
  htmlFile << "<head>  " << endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlFile << " http-equiv=\"content-type\">  " << endl;
  htmlFile << "  <title>Monitor:LaserTask output</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlFile << "<body>  " << endl;
  htmlFile << "<br>  " << endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << run << "</span></h2>" << endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">LASER</span></h2> " << endl;
  htmlFile << "<hr>" << endl;
  htmlFile << "<table border=1><tr><td bgcolor=red>channel has problems in this task</td>" << endl;
  htmlFile << "<td bgcolor=lime>channel has NO problems</td>" << endl;
  htmlFile << "<td bgcolor=yellow>channel is missing</td></table>" << endl;
  htmlFile << "<hr>" << endl;
  //   htmlFile << "<table border=1><tr><td>L1 = blu laser</td>" << endl;
  //   htmlFile << "<td>L2 = green laser</td>" << endl;
  //   htmlFile << "<td>L3 = red laser</td>" << endl;
  //   htmlFile << "<td>L4 = infrared laser</td></table>" << endl;
  htmlFile << "<table style=\"width: 600px;\" border=\"0\">" << endl;
  htmlFile << "<tbody>" << endl;
  htmlFile << "<tr>" << endl;
  htmlFile << "<td style=\"text-align: center;\">" << endl;
  htmlFile << "<div style=\"text-align: center;\"> </div>" << endl;
  htmlFile << "<table style=\"width: 482px; height: 35px;\" border=\"1\">" << endl;
  htmlFile << "<tbody>" << endl;
  htmlFile << "<tr>" << endl;
  htmlFile << "<td style=\"text-align: center;\">L1 = blue laser </td>" << endl;
  htmlFile << "<td style=\"vertical-align: top; text-align: center;\">L2 =green laser </td>" << endl;
  htmlFile << "<td style=\"vertical-align: top; text-align: center;\">L3 =red laser </td>" << endl;
  htmlFile << "<td style=\"vertical-align: top; text-align: center;\">L4 =infrared laser </td>" << endl;
  htmlFile << "</tr>" << endl;
  htmlFile << "</tbody>" << endl;
  htmlFile << "</table>" << endl;
  htmlFile << "</td>" << endl;
  htmlFile << "<td align=\"center\">" << endl;
  htmlFile << "<div style=\"text-align: center;\"> </div>" << endl;
  htmlFile << "<table style=\"width: 255px; height: 35px;\" border=\"1\">" << endl;
  htmlFile << "<tbody>" << endl;
  htmlFile << "<tr>" << endl;
  htmlFile << "<td style=\"text-align: center;\">A=L-shaped half</td>" << endl;
  htmlFile << "<td style=\"vertical-align: top; text-align: center;\">B=notL-shaped half<br>" << endl;
  htmlFile << "</td>" << endl;
  htmlFile << "</tr>" << endl;
  htmlFile << "</tbody>" << endl;
  htmlFile << "</table>" << endl;
  htmlFile << "</td>" << endl;
  htmlFile << "</tr>" << endl;
  htmlFile << "</tbody>" << endl;
  htmlFile << "</table>" << endl;
  htmlFile << "<hr>" << endl;

  // Produce the plots to be shown as .png files from existing histograms

  const int csize = 250;

//  const double histMax = 1.e15;

  int pCol3[6] = { 301, 302, 303, 304, 305, 306 };

  TH2C dummy( "dummy", "dummy for sm", 85, 0., 85., 20, 0., 20. );
  for ( int i = 0; i < 68; i++ ) {
    int a = 2 + ( i/4 ) * 5;
    int b = 2 + ( i%4 ) * 5;
    dummy.Fill( a, b, i+1 );
  }
  dummy.SetMarkerSize(2);
  dummy.SetMinimum(0.1);

  TH2C dummy1( "dummy1", "dummy1 for sm mem", 10, 0, 10, 5, 0, 5 );
  for ( short i=0; i<2; i++ ) {
    int a = 2 + i*5; 
    int b = 2;
    dummy1.Fill( a, b, i+1+68 );
  }
  dummy1.SetMarkerSize(2);
  dummy1.SetMinimum(0.1);

  string imgNameQual[8], imgNameAmp[8], imgNameTim[8], imgNameShape[8], imgNameAmpoPN[8], imgNameMEPnQualG01[8], imgNameMEPnG01[8], imgNameMEPnPedG01[8], imgNameMEPnQualG16[8], imgNameMEPnG16[8], imgNameMEPnPedG16[8], imgName, meName;

  TCanvas* cQual = new TCanvas("cQual", "Temp", 2*csize, csize);
  TCanvas* cAmp = new TCanvas("cAmp", "Temp", csize, csize);
  TCanvas* cTim = new TCanvas("cTim", "Temp", csize, csize);
  TCanvas* cShape = new TCanvas("cShape", "Temp", csize, csize);
  TCanvas* cAmpoPN = new TCanvas("cAmpoPN", "Temp", csize, csize);
  TCanvas* cPed = new TCanvas("cPed", "Temp", csize, csize);

  TH2F* obj2f;
  TH1F* obj1f;
  TH1D* obj1d;

  // Loop on barrel supermodules

  for ( unsigned int i=0; i<superModules_.size(); i ++ ) {

    int ism = superModules_[i];

    // Loop on wavelength times 2 'sides'

    for ( int iCanvas = 1 ; iCanvas <= 4 * 2 ; iCanvas++ ) {

      // skip unused wavelengths
      if ( iCanvas == 2 || iCanvas == 3 ) continue;
      if ( iCanvas == 4+2 || iCanvas == 4+3 ) continue;

      // Quality plots

      imgNameQual[iCanvas-1] = "";

      obj2f = 0;
      switch ( iCanvas ) {
        case 1:
          obj2f = EBMUtilsClient::getHisto<TH2F*>( meg01_[ism-1] );
          break;
        case 2:
          obj2f = EBMUtilsClient::getHisto<TH2F*>( meg02_[ism-1] );
          break;
        case 3:
          obj2f = EBMUtilsClient::getHisto<TH2F*>( meg03_[ism-1] );
          break;
        case 4:
          obj2f = EBMUtilsClient::getHisto<TH2F*>( meg04_[ism-1] );
          break;
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

        for ( unsigned int i = 0; i < meName.size(); i++ ) {
          if ( meName.substr(i, 1) == " " )  {
            meName.replace(i, 1, "_");
          }
        }
        imgNameQual[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgNameQual[iCanvas-1];

        cQual->cd();
        gStyle->SetOptStat(" ");
        gStyle->SetPalette(6, pCol3);
        obj2f->GetXaxis()->SetNdivisions(17);
        obj2f->GetYaxis()->SetNdivisions(4);
        cQual->SetGridx();
        cQual->SetGridy();
        obj2f->SetMinimum(-0.00000001);
        obj2f->SetMaximum(6.0);
        obj2f->Draw("col");
        dummy.Draw("text,same");
        cQual->Update();
        cQual->SaveAs(imgName.c_str());

      }

      // Amplitude distributions

      imgNameAmp[iCanvas-1] = "";

      obj1f = 0;
      switch ( iCanvas ) {
        case 1:
          obj1f = EBMUtilsClient::getHisto<TH1F*>( mea01_[ism-1] );
          break;
        case 2:
          obj1f = EBMUtilsClient::getHisto<TH1F*>( mea02_[ism-1] );
          break;
        case 3:
          obj1f = EBMUtilsClient::getHisto<TH1F*>( mea03_[ism-1] );
          break;
        case 4:
          obj1f = EBMUtilsClient::getHisto<TH1F*>( mea04_[ism-1] );
          break;
        case 5:
          obj1f = EBMUtilsClient::getHisto<TH1F*>( mea05_[ism-1] );
          break;
        case 6:
          obj1f = EBMUtilsClient::getHisto<TH1F*>( mea06_[ism-1] );
          break;
        case 7:
          obj1f = EBMUtilsClient::getHisto<TH1F*>( mea07_[ism-1] );
          break;
        case 8:
          obj1f = EBMUtilsClient::getHisto<TH1F*>( mea08_[ism-1] );
          break;
        default:
          break;
      }

      if ( obj1f ) {

        meName = obj1f->GetName();

        for ( unsigned int i = 0; i < meName.size(); i++ ) {
          if ( meName.substr(i, 1) == " " )  {
            meName.replace(i, 1 ,"_" );
          }
        }
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
          obj1f = EBMUtilsClient::getHisto<TH1F*>( met01_[ism-1] );
          break;
        case 2:
          obj1f = EBMUtilsClient::getHisto<TH1F*>( met02_[ism-1] );
          break;
        case 3:
          obj1f = EBMUtilsClient::getHisto<TH1F*>( met03_[ism-1] );
          break;
        case 4:
          obj1f = EBMUtilsClient::getHisto<TH1F*>( met04_[ism-1] );
          break;
        case 5:
          obj1f = EBMUtilsClient::getHisto<TH1F*>( met05_[ism-1] );
          break;
        case 6:
          obj1f = EBMUtilsClient::getHisto<TH1F*>( met06_[ism-1] );
          break;
        case 7:
          obj1f = EBMUtilsClient::getHisto<TH1F*>( met07_[ism-1] );
          break;
        case 8:
          obj1f = EBMUtilsClient::getHisto<TH1F*>( met08_[ism-1] );
          break;
        default:
          break;
      }

      if ( obj1f ) {

        meName = obj1f->GetName();

        for ( unsigned int i = 0; i < meName.size(); i++ ) {
          if ( meName.substr(i, 1) == " " )  {
            meName.replace(i, 1 ,"_" );
          }
        }
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

      // Shape distributions

      imgNameShape[iCanvas-1] = "";

      obj1d = 0;
      switch ( iCanvas ) {
        case 1:
          if ( hs01_[ism-1] ) obj1d = hs01_[ism-1]->ProjectionY("_py", 1, 1, "e");
          break;
        case 2:
          if ( hs02_[ism-1] ) obj1d = hs02_[ism-1]->ProjectionY("_py", 1, 1, "e");
          break;
        case 3:
          if ( hs03_[ism-1] ) obj1d = hs03_[ism-1]->ProjectionY("_py", 1, 1, "e");
          break;
        case 4: 
          if ( hs04_[ism-1] ) obj1d = hs04_[ism-1]->ProjectionY("_py", 1, 1, "e");
          break;
        case 5:
          if ( hs05_[ism-1] ) obj1d = hs05_[ism-1]->ProjectionY("_py", 1681, 1681, "e");
          break;
        case 6:
          if ( hs06_[ism-1] ) obj1d = hs06_[ism-1]->ProjectionY("_py", 1681, 1681, "e");
          break;
        case 7:
          if ( hs07_[ism-1] ) obj1d = hs07_[ism-1]->ProjectionY("_py", 1681, 1681, "e");
          break;
        case 8:
          if ( hs08_[ism-1] ) obj1d = hs08_[ism-1]->ProjectionY("_py", 1681, 1681, "e");
          break;
        default:
          break;
      }

      if ( obj1d ) {
        meName = obj1d->GetName();

        for ( unsigned int i = 0; i < meName.size(); i++ ) {
          if ( meName.substr(i, 1) == " " )  {
            meName.replace(i, 1, "_");
          }
        }
        imgNameShape[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgNameShape[iCanvas-1];

        cShape->cd();
        gStyle->SetOptStat("euo");
        obj1d->SetStats(kTRUE);
//        if ( obj1d->GetMaximum(histMax) > 0. ) {
//          gPad->SetLogy(1);
//        } else {
//          gPad->SetLogy(0);
//        }
        obj1d->Draw();
        cShape->Update();
        cShape->SaveAs(imgName.c_str());
        gPad->SetLogy(0);

        delete obj1d;

      }

      // Amplitude over PN distributions

      imgNameAmpoPN[iCanvas-1] = "";

      obj1f = 0;
      switch ( iCanvas ) {
        case 1:
          obj1f = EBMUtilsClient::getHisto<TH1F*>( meaopn01_[ism-1] );
          break;
        case 2:
          obj1f = EBMUtilsClient::getHisto<TH1F*>( meaopn02_[ism-1] );
          break;
        case 3:
          obj1f = EBMUtilsClient::getHisto<TH1F*>( meaopn03_[ism-1] );
          break;
        case 4:
          obj1f = EBMUtilsClient::getHisto<TH1F*>( meaopn04_[ism-1] );
          break;
        case 5:
          obj1f = EBMUtilsClient::getHisto<TH1F*>( meaopn05_[ism-1] );
          break;
        case 6:
          obj1f = EBMUtilsClient::getHisto<TH1F*>( meaopn06_[ism-1] );
          break;
        case 7:
          obj1f = EBMUtilsClient::getHisto<TH1F*>( meaopn07_[ism-1] );
          break;
        case 8:
          obj1f = EBMUtilsClient::getHisto<TH1F*>( meaopn08_[ism-1] );
          break;
        default:
          break;
      }

      if ( obj1f ) {

        meName = obj1f->GetName();

        for ( unsigned int i = 0; i < meName.size(); i++ ) {
          if ( meName.substr(i, 1) == " " )  {
            meName.replace(i, 1, "_");
          }
        }
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
        obj2f = EBMUtilsClient::getHisto<TH2F*>( meg05_[ism-1] );
        break;
      case 2:
        obj2f = EBMUtilsClient::getHisto<TH2F*>( meg06_[ism-1] );
        break;
      case 3:
        obj2f = EBMUtilsClient::getHisto<TH2F*>( meg07_[ism-1] );
        break;
      case 4:
        obj2f = EBMUtilsClient::getHisto<TH2F*>( meg08_[ism-1] );
        break;
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

        for ( unsigned int i = 0; i < meName.size(); i++ ) {
          if ( meName.substr(i, 1) == " " )  {
            meName.replace(i, 1, "_");
          }
        }
        imgNameMEPnQualG01[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgNameMEPnQualG01[iCanvas-1];

        cQual->cd();
        gStyle->SetOptStat(" ");
        gStyle->SetPalette(6, pCol3);
        obj2f->GetXaxis()->SetNdivisions(10);
        obj2f->GetYaxis()->SetNdivisions(5);
        cQual->SetGridx();
        cQual->SetGridy(0);
        obj2f->SetMinimum(-0.00000001);
        obj2f->SetMaximum(6.0);
        obj2f->Draw("col");
        dummy1.Draw("text,same");
        cQual->Update();
        cQual->SaveAs(imgName.c_str());

      }

      imgNameMEPnQualG16[iCanvas-1] = "";

      obj2f = 0;
      switch ( iCanvas ) {
      case 1:
        obj2f = EBMUtilsClient::getHisto<TH2F*>( meg09_[ism-1] );
        break;
      case 2:
        obj2f = EBMUtilsClient::getHisto<TH2F*>( meg10_[ism-1] );
        break;
      case 3:
        obj2f = EBMUtilsClient::getHisto<TH2F*>( meg11_[ism-1] );
        break;
      case 4:
        obj2f = EBMUtilsClient::getHisto<TH2F*>( meg12_[ism-1] );
        break;
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

        for ( unsigned int i = 0; i < meName.size(); i++ ) {
          if ( meName.substr(i, 1) == " " )  {
            meName.replace(i, 1, "_");
          }
        }
        imgNameMEPnQualG16[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgNameMEPnQualG16[iCanvas-1];

        cQual->cd();
        gStyle->SetOptStat(" ");
        gStyle->SetPalette(6, pCol3);
        obj2f->GetXaxis()->SetNdivisions(10);
        obj2f->GetYaxis()->SetNdivisions(5);
        cQual->SetGridx();
        cQual->SetGridy(0);
        obj2f->SetMinimum(-0.00000001);
        obj2f->SetMaximum(6.0);
        obj2f->Draw("col");
        dummy1.Draw("text,same");
        cQual->Update();
        cQual->SaveAs(imgName.c_str());

      }

      imgNameMEPnG01[iCanvas-1] = "";

      obj1d = 0;
      switch ( iCanvas ) {
        case 1:
          if ( i01_[ism-1] ) obj1d = i01_[ism-1]->ProjectionY("_py", 1, 1, "e");
          break;
        case 2:
          if ( i02_[ism-1] ) obj1d = i02_[ism-1]->ProjectionY("_py", 1, 1, "e");
          break;
        case 3:
          if ( i03_[ism-1] ) obj1d = i03_[ism-1]->ProjectionY("_py", 1, 1, "e");
          break;
        case 4:
          if ( i04_[ism-1] ) obj1d = i04_[ism-1]->ProjectionY("_py", 1, 1, "e");
          break;
        case 5:
        case 6:
        case 7:
        case 8:
          obj2f = 0;
          break;
        default:
          break;
      }

      if ( obj1d ) {

        meName = obj1d->GetName();

        for ( unsigned int i = 0; i < meName.size(); i++ ) {
          if ( meName.substr(i, 1) == " " )  {
            meName.replace(i, 1 ,"_" );
          }
        }
        imgNameMEPnG01[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgNameMEPnG01[iCanvas-1];

        cAmp->cd();
        gStyle->SetOptStat("euo");
        obj1d->SetStats(kTRUE);
//        if ( obj1d->GetMaximum(histMax) > 0. ) {
//          gPad->SetLogy(1);
//        } else {
//          gPad->SetLogy(0);
//        }
        obj1d->SetMinimum(0.0);
        obj1d->Draw();
        cAmp->Update();
        cAmp->SaveAs(imgName.c_str());
        gPad->SetLogy(0);

        delete obj1d;

      }

      imgNameMEPnG16[iCanvas-1] = "";

      obj1d = 0;
      switch ( iCanvas ) {
        case 1:
          if ( i09_[ism-1] ) obj1d = i09_[ism-1]->ProjectionY("_py", 1, 1, "e");
          break;
        case 2:
          if ( i10_[ism-1] ) obj1d = i10_[ism-1]->ProjectionY("_py", 1, 1, "e");
          break;
        case 3:
          if ( i11_[ism-1] ) obj1d = i11_[ism-1]->ProjectionY("_py", 1, 1, "e");
          break;
        case 4:
          if ( i12_[ism-1] ) obj1d = i12_[ism-1]->ProjectionY("_py", 1, 1, "e");
          break;
        case 5:
        case 6:
        case 7:
        case 8:
          obj2f = 0;
          break;
        default:
          break;
      }

      if ( obj1d ) {

        meName = obj1d->GetName();

        for ( unsigned int i = 0; i < meName.size(); i++ ) {
          if ( meName.substr(i, 1) == " " )  {
            meName.replace(i, 1 ,"_" );
          }
        }
        imgNameMEPnG16[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgNameMEPnG16[iCanvas-1];

        cAmp->cd();
        gStyle->SetOptStat("euo");
        obj1d->SetStats(kTRUE);
//        if ( obj1d->GetMaximum(histMax) > 0. ) {
//          gPad->SetLogy(1);
//        } else {
//          gPad->SetLogy(0);
//        }
        obj1d->SetMinimum(0.0);
        obj1d->Draw();
        cAmp->Update();
        cAmp->SaveAs(imgName.c_str());
        gPad->SetLogy(0);

        delete obj1d;

      }

      // Monitoring elements plots

      imgNameMEPnPedG01[iCanvas-1] = "";

      obj1d = 0;
      switch ( iCanvas ) {
        case 1:
          if ( i05_[ism-1] ) obj1d = i05_[ism-1]->ProjectionY("_py", 1, 1, "e");
          break;
        case 2:
          if ( i06_[ism-1] ) obj1d = i06_[ism-1]->ProjectionY("_py", 1, 1, "e");
          break;
        case 3:
          if ( i07_[ism-1] ) obj1d = i07_[ism-1]->ProjectionY("_py", 1, 1, "e");
          break;
        case 4:
          if ( i08_[ism-1] ) obj1d = i08_[ism-1]->ProjectionY("_py", 1, 1, "e");
          break;
        case 5:
        case 6:
        case 7:
        case 8:
          obj2f = 0;
          break;
        default:
          break;
      }

      if ( obj1d ) {

        meName = obj1d->GetName();

        for ( unsigned int i = 0; i < meName.size(); i++ ) {
          if ( meName.substr(i, 1) == " " )  {
            meName.replace(i, 1 ,"_" );
          }
        }
        imgNameMEPnPedG01[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgNameMEPnPedG01[iCanvas-1];

        cPed->cd();
        gStyle->SetOptStat("euo");
        obj1d->SetStats(kTRUE);
//        if ( obj1d->GetMaximum(histMax) > 0. ) {
//          gPad->SetLogy(1);
//        } else {
//          gPad->SetLogy(0);
//        }
        obj1d->SetMinimum(0.0);
        obj1d->Draw();
        cPed->Update();
        cPed->SaveAs(imgName.c_str());
        gPad->SetLogy(0);

        delete obj1d;

      }

      imgNameMEPnPedG16[iCanvas-1] = "";

      obj1d = 0;
      switch ( iCanvas ) {
        case 1:
          if ( i13_[ism-1] ) obj1d = i13_[ism-1]->ProjectionY("_py", 1, 1, "e");
          break;
        case 2:
          if ( i14_[ism-1] ) obj1d = i14_[ism-1]->ProjectionY("_py", 1, 1, "e");
          break;
        case 3:
          if ( i15_[ism-1] ) obj1d = i15_[ism-1]->ProjectionY("_py", 1, 1, "e");
          break;
        case 4:
          if ( i16_[ism-1] ) obj1d = i16_[ism-1]->ProjectionY("_py", 1, 1, "e");
          break;
        case 5:
        case 6:
        case 7:
        case 8:
          obj2f = 0;
          break;
        default:
          break;
      }

      if ( obj1d ) {

        meName = obj1d->GetName();

        for ( unsigned int i = 0; i < meName.size(); i++ ) {
          if ( meName.substr(i, 1) == " " )  {
            meName.replace(i, 1 ,"_" );
          }
        }
        imgNameMEPnPedG16[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgNameMEPnPedG16[iCanvas-1];

        cPed->cd();
        gStyle->SetOptStat("euo");
        obj1d->SetStats(kTRUE);
//        if ( obj1d->GetMaximum(histMax) > 0. ) {
//          gPad->SetLogy(1);
//        } else {
//          gPad->SetLogy(0);
//        }
        obj1d->SetMinimum(0.0);
        obj1d->Draw();
        cPed->Update();
        cPed->SaveAs(imgName.c_str());
        gPad->SetLogy(0);

        delete obj1d;

      }

    }

    htmlFile << "<h3><strong>Supermodule&nbsp;&nbsp;" << ism << "</strong></h3>" << endl;
    htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
    htmlFile << "<tr align=\"center\">" << endl;

    for ( int iCanvas = 1 ; iCanvas <= 4 ; iCanvas++ ) {

      // skip unused wavelengths
      if ( iCanvas == 2 || iCanvas == 3 ) continue;

      if ( imgNameQual[iCanvas-1].size() != 0 )
        htmlFile << "<td colspan=\"2\"><img src=\"" << imgNameQual[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td colspan=\"2\"><img src=\"" << " " << "\"></td>" << endl;

    }

    htmlFile << "</tr>" << endl;

    htmlFile << "<tr>" << endl;

    for ( int iCanvas = 1 ; iCanvas <= 4 ; iCanvas++ ) {

      // skip unused wavelengths
      if ( iCanvas == 2 || iCanvas == 3 ) continue;

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
      if ( iCanvas == 2 || iCanvas == 3 ) continue;

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

    htmlFile << "<tr align=\"center\">" << endl;

    for ( int iCanvas = 1 ; iCanvas <= 4 ; iCanvas++ ) {

      // skip unused wavelengths
      if ( iCanvas == 2 || iCanvas == 3 ) continue;

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

    htmlFile << "<tr align=\"center\">" << endl;

    for ( int iCanvas = 1 ; iCanvas <= 4 ; iCanvas++ ) {

      // skip unused wavelengths
      if ( iCanvas == 2 || iCanvas == 3 ) continue;

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

    htmlFile << "<tr align=\"center\">" << endl;

    for ( int iCanvas = 1 ; iCanvas <= 4 ; iCanvas++ ) {

      // skip unused wavelengths
      if ( iCanvas == 2 || iCanvas == 3 ) continue;

      htmlFile << "<td colspan=\"2\">Laser " << iCanvas << "</td>" << endl;

    }

    htmlFile << "</tr>" << endl;
    htmlFile << "</table>" << endl;
    htmlFile << "<br>" << endl;

    htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
    htmlFile << "<tr align=\"center\">" << endl;

    for ( int iCanvas = 1 ; iCanvas <= 4 ; iCanvas++ ) {

      // skip unused wavelengths
      if ( iCanvas == 2 || iCanvas == 3 ) continue;

      if ( imgNameMEPnQualG01[iCanvas-1].size() != 0 )
        htmlFile << "<td colspan=\"2\"><img src=\"" << imgNameMEPnQualG01[iCanvas-1] << "\"></td>" << endl;
      else 
        htmlFile << "<td colspan=\"2\"><img src=\"" << " " << "\"></td>" << endl;

    }

    htmlFile << "</tr>" << endl;

    htmlFile << "<tr align=\"center\">" << endl;

    for ( int iCanvas = 1 ; iCanvas <= 4 ; iCanvas++ ) {

      // skip unused wavelengths
      if ( iCanvas == 2 || iCanvas == 3 ) continue;

      if ( imgNameMEPnPedG01[iCanvas-1].size() != 0 )
        htmlFile << "<td><img src=\"" << imgNameMEPnPedG01[iCanvas-1] << "\"></td>" << endl;
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
      if ( iCanvas == 2 || iCanvas == 3 ) continue;

      htmlFile << "<td colspan=\"2\">Laser " << iCanvas << " - PN Gain 1</td>" << endl;

    }

    htmlFile << "</tr>" << endl;

    htmlFile << "<tr align=\"center\">" << endl;

    for ( int iCanvas = 1 ; iCanvas <= 4 ; iCanvas++ ) {

      // skip unused wavelengths
      if ( iCanvas == 2 || iCanvas == 3 ) continue;

      if ( imgNameMEPnQualG16[iCanvas-1].size() != 0 )
        htmlFile << "<td colspan=\"2\"><img src=\"" << imgNameMEPnQualG16[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td colspan=\"2\"><img src=\"" << " " << "\"></td>" << endl;

    }

    htmlFile << "</tr>" << endl;

    htmlFile << "<tr align=\"center\">" << endl;

    for ( int iCanvas = 1 ; iCanvas <= 4 ; iCanvas++ ) {

      // skip unused wavelengths
      if ( iCanvas == 2 || iCanvas == 3 ) continue;

      if ( imgNameMEPnPedG16[iCanvas-1].size() != 0 )
        htmlFile << "<td><img src=\"" << imgNameMEPnPedG16[iCanvas-1] << "\"></td>" << endl;
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
      if ( iCanvas == 2 || iCanvas == 3 ) continue;

      htmlFile << "<td colspan=\"2\">Laser " << iCanvas << " - PN Gain 16</td>" << endl;

    }

    htmlFile << "</tr>" << endl;

    htmlFile << "</tr>" << endl;
    htmlFile << "</table>" << endl;

    htmlFile << "<br>" << endl;

  }

  delete cQual;
  delete cAmp;
  delete cTim;
  delete cShape;
  delete cAmpoPN;
  delete cPed;

  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  htmlFile.close();

}

