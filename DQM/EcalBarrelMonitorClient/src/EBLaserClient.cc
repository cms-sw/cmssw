/*
 * \file EBLaserClient.cc
 *
 * $Date: 2007/01/27 18:00:06 $
 * $Revision: 1.125 $
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

    j01_[ism-1] = 0;
    j02_[ism-1] = 0;
    j03_[ism-1] = 0;
    j04_[ism-1] = 0;
    j05_[ism-1] = 0;
    j06_[ism-1] = 0;
    j07_[ism-1] = 0;
    j08_[ism-1] = 0;

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

      sprintf(qtname, "EBLT laser amplitude quality PNs SM%02d L1 G16", ism);
      qth09_[ism-1] = dynamic_cast<MEContentsProf2DWithinRangeROOT*> (mui_->createQTest(ContentsProf2DWithinRangeROOT::getAlgoName(), qtname));

      sprintf(qtname, "EBLT laser amplitude quality PNs SM%02d L2 G16", ism);
      qth10_[ism-1] = dynamic_cast<MEContentsProf2DWithinRangeROOT*> (mui_->createQTest(ContentsProf2DWithinRangeROOT::getAlgoName(), qtname));

      sprintf(qtname, "EBLT laser amplitude quality PNs SM%02d L3 G16", ism);
      qth11_[ism-1] = dynamic_cast<MEContentsProf2DWithinRangeROOT*> (mui_->createQTest(ContentsProf2DWithinRangeROOT::getAlgoName(), qtname));

      sprintf(qtname, "EBLT laser amplitude quality PNs SM%02d L4 G16", ism);
      qth12_[ism-1] = dynamic_cast<MEContentsProf2DWithinRangeROOT*> (mui_->createQTest(ContentsProf2DWithinRangeROOT::getAlgoName(), qtname));

      sprintf(qtname, "EBLT laser pedestal quality PNs SM%02d L1 G16", ism);
      qth13_[ism-1] = dynamic_cast<MEContentsProf2DWithinRangeROOT*> (mui_->createQTest(ContentsProf2DWithinRangeROOT::getAlgoName(), qtname));

      sprintf(qtname, "EBLT laser pedestal quality PNs SM%02d L2 G16", ism);
      qth14_[ism-1] = dynamic_cast<MEContentsProf2DWithinRangeROOT*> (mui_->createQTest(ContentsProf2DWithinRangeROOT::getAlgoName(), qtname));

      sprintf(qtname, "EBLT laser pedestal quality PNs SM%02d L3 G16", ism);
      qth15_[ism-1] = dynamic_cast<MEContentsProf2DWithinRangeROOT*> (mui_->createQTest(ContentsProf2DWithinRangeROOT::getAlgoName(), qtname));

      sprintf(qtname, "EBLT laser pedestal quality PNs SM%02d L4 G16", ism);
      qth16_[ism-1] = dynamic_cast<MEContentsProf2DWithinRangeROOT*> (mui_->createQTest(ContentsProf2DWithinRangeROOT::getAlgoName(), qtname));

      qth01_[ism-1]->setMeanRange(100.0, 4096.0);
      qth02_[ism-1]->setMeanRange(100.0, 4096.0);
      qth03_[ism-1]->setMeanRange(100.0, 4096.0);
      qth04_[ism-1]->setMeanRange(100.0, 4096.0);
      qth05_[ism-1]->setMeanRange(100.0, 4096.0);
      qth06_[ism-1]->setMeanRange(100.0, 4096.0);
      qth07_[ism-1]->setMeanRange(100.0, 4096.0);
      qth08_[ism-1]->setMeanRange(100.0, 4096.0);

      qth09_[ism-1]->setMeanRange(amplitudeThresholdPN_, 4096.0);
      qth10_[ism-1]->setMeanRange(amplitudeThresholdPN_, 4096.0);
      qth11_[ism-1]->setMeanRange(amplitudeThresholdPN_, 4096.0);
      qth12_[ism-1]->setMeanRange(amplitudeThresholdPN_, 4096.0);
      qth13_[ism-1]->setMeanRange(meanThresholdPN_, 4096.0);
      qth14_[ism-1]->setMeanRange(meanThresholdPN_, 4096.0);
      qth15_[ism-1]->setMeanRange(meanThresholdPN_, 4096.0);
      qth16_[ism-1]->setMeanRange(meanThresholdPN_, 4096.0);

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
    sprintf(histo, "EBLT laser quality L1 PNs SM%02d G16", ism);
    meg05_[ism-1] = bei->book2D(histo, histo, 10, 0., 10., 1, 0., 5.);
    if ( meg06_[ism-1] ) bei->removeElement( meg06_[ism-1]->getName() );
    sprintf(histo, "EBLT laser quality L2 PNs SM%02d G16", ism);
    meg06_[ism-1] = bei->book2D(histo, histo, 10, 0., 10., 1, 0., 5.);
    if ( meg07_[ism-1] ) bei->removeElement( meg07_[ism-1]->getName() );
    sprintf(histo, "EBLT laser quality L3 PNs SM%02d G16", ism);
    meg07_[ism-1] = bei->book2D(histo, histo, 10, 0., 10., 1, 0., 5.);
    if ( meg08_[ism-1] ) bei->removeElement( meg08_[ism-1]->getName() );
    sprintf(histo, "EBLT laser quality L4 PNs SM%02d G16", ism);
    meg08_[ism-1] = bei->book2D(histo, histo, 10, 0., 10., 1, 0., 5.);

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

      if ( j01_[ism-1] ) delete j01_[ism-1];
      if ( j02_[ism-1] ) delete j02_[ism-1];
      if ( j03_[ism-1] ) delete j03_[ism-1];
      if ( j04_[ism-1] ) delete j04_[ism-1];
      if ( j05_[ism-1] ) delete j05_[ism-1];
      if ( j06_[ism-1] ) delete j06_[ism-1];
      if ( j07_[ism-1] ) delete j07_[ism-1];
      if ( j08_[ism-1] ) delete j08_[ism-1];
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

    j01_[ism-1] = 0;
    j02_[ism-1] = 0;
    j03_[ism-1] = 0;
    j04_[ism-1] = 0;
    j05_[ism-1] = 0;
    j06_[ism-1] = 0;
    j07_[ism-1] = 0;
    j08_[ism-1] = 0;

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

  vector<dqm::me_util::Channel> badChannels;

  if ( qth01_[ism-1] ) badChannels = qth01_[ism-1]->getBadChannels();

  if ( ! badChannels.empty() ) {

    cout << endl;
    cout << " Channels that failed \""
         << qth01_[ism-1]->getName() << "\" "
         << "(Algorithm: "
         << qth01_[ism-1]->getAlgoName()
         << ")" << endl;

    cout << endl;
    for ( vector<dqm::me_util::Channel>::iterator it = badChannels.begin(); it != badChannels.end(); ++it ) {
      cout << " (" << it->getBinX()
           << ", " << it->getBinY()
           << ", " << it->getBinZ()
           << ") = " << it->getContents()
           << " +- " << it->getRMS()
           << endl;
    }
    cout << endl;

  }

  if ( qth05_[ism-1] ) badChannels = qth05_[ism-1]->getBadChannels();

  if ( ! badChannels.empty() ) {

    cout << endl;
    cout << " Channels that failed \""
         << qth05_[ism-1]->getName() << "\" "
         << "(Algorithm: "
         << qth05_[ism-1]->getAlgoName()
         << ")" << endl;

    cout << endl;
    for ( vector<dqm::me_util::Channel>::iterator it = badChannels.begin(); it != badChannels.end(); ++it ) {
      cout << " (" << it->getBinX()
           << ", " << it->getBinY()
           << ", " << it->getBinZ()
           << ") = " << it->getContents()
           << " +- " << it->getRMS()
           << endl;
    }
    cout << endl;

  }

  if ( qth02_[ism-1] ) badChannels = qth02_[ism-1]->getBadChannels();
 
  if ( ! badChannels.empty() ) {

    cout << endl;
    cout << " Channels that failed \""
         << qth02_[ism-1]->getName() << "\" "
         << "(Algorithm: "
         << qth02_[ism-1]->getAlgoName()
         << ")" << endl;

    cout << endl;
    for ( vector<dqm::me_util::Channel>::iterator it = badChannels.begin(); it != badChannels.end(); ++it ) {
      cout << " (" << it->getBinX()
           << ", " << it->getBinY()
           << ", " << it->getBinZ()
           << ") = " << it->getContents()
           << " +- " << it->getRMS()
           << endl;
    }
    cout << endl;

  }

  if ( qth06_[ism-1] ) badChannels = qth06_[ism-1]->getBadChannels();

  if ( ! badChannels.empty() ) {

    cout << endl;
    cout << " Channels that failed \""
         << qth06_[ism-1]->getName() << "\" "
         << "(Algorithm: "
         << qth06_[ism-1]->getAlgoName()
         << ")" << endl;

    cout << endl;
    for ( vector<dqm::me_util::Channel>::iterator it = badChannels.begin(); it != badChannels.end(); ++it ) {
      cout << " (" << it->getBinX()
           << ", " << it->getBinY()
           << ", " << it->getBinZ()
           << ") = " << it->getContents()
           << " +- " << it->getRMS()
           << endl;
    }
    cout << endl;

  }

  if ( qth03_[ism-1] ) badChannels = qth03_[ism-1]->getBadChannels();

  if ( ! badChannels.empty() ) {

    cout << endl;
    cout << " Channels that failed \""
         << qth03_[ism-1]->getName() << "\" "
         << "(Algorithm: "
         << qth03_[ism-1]->getAlgoName()
         << ")" << endl;

    cout << endl;
    for ( vector<dqm::me_util::Channel>::iterator it = badChannels.begin(); it != badChannels.end(); ++it ) {
      cout << " (" << it->getBinX()
           << ", " << it->getBinY()
           << ", " << it->getBinZ()
           << ") = " << it->getContents()
           << " +- " << it->getRMS()
           << endl;
    }
    cout << endl;

  }

  if ( qth07_[ism-1] ) badChannels = qth07_[ism-1]->getBadChannels();

  if ( ! badChannels.empty() ) {

    cout << endl;
    cout << " Channels that failed \""
         << qth07_[ism-1]->getName() << "\" "
         << "(Algorithm: "
         << qth07_[ism-1]->getAlgoName()
         << ")" << endl;

    cout << endl;
    for ( vector<dqm::me_util::Channel>::iterator it = badChannels.begin(); it != badChannels.end(); ++it ) {
      cout << " (" << it->getBinX()
           << ", " << it->getBinY()
           << ", " << it->getBinZ()
           << ") = " << it->getContents()
           << " +- " << it->getRMS()
           << endl;
    }
    cout << endl;

  }

  if ( qth04_[ism-1] ) badChannels = qth04_[ism-1]->getBadChannels();

  if ( ! badChannels.empty() ) {

    cout << endl;
    cout << " Channels that failed \""
         << qth04_[ism-1]->getName() << "\" "
         << "(Algorithm: "
         << qth04_[ism-1]->getAlgoName()
         << ")" << endl;

    cout << endl;
    for ( vector<dqm::me_util::Channel>::iterator it = badChannels.begin(); it != badChannels.end(); ++it ) {
      cout << " (" << it->getBinX()
           << ", " << it->getBinY()
           << ", " << it->getBinZ()
           << ") = " << it->getContents()
           << " +- " << it->getRMS()
           << endl;
    }
    cout << endl;

  }

  if ( qth08_[ism-1] ) badChannels = qth08_[ism-1]->getBadChannels();

  if ( ! badChannels.empty() ) {

    cout << endl;
    cout << " Channels that failed \""
         << qth08_[ism-1]->getName() << "\" "
         << "(Algorithm: "
         << qth08_[ism-1]->getAlgoName()
         << ")" << endl;

    cout << endl;
    for ( vector<dqm::me_util::Channel>::iterator it = badChannels.begin(); it != badChannels.end(); ++it ) {
      cout << " (" << it->getBinX()
           << ", " << it->getBinY()
           << ", " << it->getBinZ()
           << ") = " << it->getContents()
           << " +- " << it->getRMS()
           << endl;
    }
    cout << endl;

  }

  if ( qth09_[ism-1] ) badChannels = qth09_[ism-1]->getBadChannels();

  if ( ! badChannels.empty() ) {

    cout << endl;
    cout << " Channels that failed \""
         << qth09_[ism-1]->getName() << "\" "
         << "(Algorithm: "
         << qth09_[ism-1]->getAlgoName()
         << ")" << endl;

    cout << endl;
    for ( vector<dqm::me_util::Channel>::iterator it = badChannels.begin(); it != badChannels.end(); ++it ) {
      cout << " (" << it->getBinX()
           << ", " << it->getBinY()
           << ", " << it->getBinZ()
           << ") = " << it->getContents()
           << " +- " << it->getRMS()
           << endl;
    }
    cout << endl;

  }

  if ( qth10_[ism-1] ) badChannels = qth10_[ism-1]->getBadChannels();

  if ( ! badChannels.empty() ) {

    cout << endl;
    cout << " Channels that failed \""
         << qth10_[ism-1]->getName() << "\" "
         << "(Algorithm: "
         << qth10_[ism-1]->getAlgoName()
         << ")" << endl;

    cout << endl;
    for ( vector<dqm::me_util::Channel>::iterator it = badChannels.begin(); it != badChannels.end(); ++it ) {
      cout << " (" << it->getBinX()
           << ", " << it->getBinY()
           << ", " << it->getBinZ()
           << ") = " << it->getContents()
           << " +- " << it->getRMS()
           << endl;
    }
    cout << endl;

  }

  if ( qth11_[ism-1] ) badChannels = qth11_[ism-1]->getBadChannels();

  if ( ! badChannels.empty() ) {

    cout << endl;
    cout << " Channels that failed \""
         << qth11_[ism-1]->getName() << "\" "
         << "(Algorithm: "
         << qth11_[ism-1]->getAlgoName()
         << ")" << endl;

    cout << endl;
    for ( vector<dqm::me_util::Channel>::iterator it = badChannels.begin(); it != badChannels.end(); ++it ) {
      cout << " (" << it->getBinX()
           << ", " << it->getBinY()
           << ", " << it->getBinZ()
           << ") = " << it->getContents()
           << " +- " << it->getRMS()
           << endl;
    }
    cout << endl;

  }

  if ( qth12_[ism-1] ) badChannels = qth12_[ism-1]->getBadChannels();

  if ( ! badChannels.empty() ) {

    cout << endl;
    cout << " Channels that failed \""
         << qth12_[ism-1]->getName() << "\" "
         << "(Algorithm: "
         << qth12_[ism-1]->getAlgoName()
         << ")" << endl;

    cout << endl;
    for ( vector<dqm::me_util::Channel>::iterator it = badChannels.begin(); it != badChannels.end(); ++it ) {
      cout << " (" << it->getBinX()
           << ", " << it->getBinY()
           << ", " << it->getBinZ()
           << ") = " << it->getContents()
           << " +- " << it->getRMS()
           << endl;
    }
    cout << endl;

  }

  if ( qth13_[ism-1] ) badChannels = qth13_[ism-1]->getBadChannels();

  if ( ! badChannels.empty() ) {

    cout << endl;
    cout << " Channels that failed \""
         << qth13_[ism-1]->getName() << "\" "
         << "(Algorithm: "
         << qth13_[ism-1]->getAlgoName()
         << ")" << endl;

    cout << endl;
    for ( vector<dqm::me_util::Channel>::iterator it = badChannels.begin(); it != badChannels.end(); ++it ) {
      cout << " (" << it->getBinX()
           << ", " << it->getBinY()
           << ", " << it->getBinZ()
           << ") = " << it->getContents()
           << " +- " << it->getRMS()
           << endl;
    }
    cout << endl;

  }

  if ( qth14_[ism-1] ) badChannels = qth14_[ism-1]->getBadChannels();

  if ( ! badChannels.empty() ) {

    cout << endl;
    cout << " Channels that failed \""
         << qth14_[ism-1]->getName() << "\" "
         << "(Algorithm: "
         << qth14_[ism-1]->getAlgoName()
         << ")" << endl;

    cout << endl;
    for ( vector<dqm::me_util::Channel>::iterator it = badChannels.begin(); it != badChannels.end(); ++it ) {
      cout << " (" << it->getBinX()
           << ", " << it->getBinY()
           << ", " << it->getBinZ()
           << ") = " << it->getContents()
           << " +- " << it->getRMS()
           << endl;
    }
    cout << endl;

  }

  if ( qth15_[ism-1] ) badChannels = qth15_[ism-1]->getBadChannels();

  if ( ! badChannels.empty() ) {

    cout << endl;
    cout << " Channels that failed \""
         << qth15_[ism-1]->getName() << "\" "
         << "(Algorithm: "
         << qth15_[ism-1]->getAlgoName()
         << ")" << endl;

    cout << endl;
    for ( vector<dqm::me_util::Channel>::iterator it = badChannels.begin(); it != badChannels.end(); ++it ) {
      cout << " (" << it->getBinX()
           << ", " << it->getBinY()
           << ", " << it->getBinZ()
           << ") = " << it->getContents()
           << " +- " << it->getRMS()
           << endl;
    }
    cout << endl;

  }

  if ( qth16_[ism-1] ) badChannels = qth16_[ism-1]->getBadChannels();

  if ( ! badChannels.empty() ) {

    cout << endl;
    cout << " Channels that failed \""
         << qth16_[ism-1]->getName() << "\" "
         << "(Algorithm: "
         << qth16_[ism-1]->getAlgoName()
         << ")" << endl;

    cout << endl;
    for ( vector<dqm::me_util::Channel>::iterator it = badChannels.begin(); it != badChannels.end(); ++it ) {
      cout << " (" << it->getBinX()
           << ", " << it->getBinY()
           << ", " << it->getBinZ()
           << ") = " << it->getContents()
           << " +- " << it->getRMS()
           << endl;
    }
    cout << endl;

  }

  EcalLogicID ecid;
  MonLaserBlueDat apd_bl;
  map<EcalLogicID, MonLaserBlueDat> dataset1_bl;
  MonLaserGreenDat apd_gr;
  map<EcalLogicID, MonLaserGreenDat> dataset1_gr;
  MonLaserIRedDat apd_ir;
  map<EcalLogicID, MonLaserIRedDat> dataset1_ir;
  MonLaserRedDat apd_rd;
  map<EcalLogicID, MonLaserRedDat> dataset1_rd;

  const float n_min_tot = 1000.;
  const float n_min_bin = 50.;

  float num01, num02, num03, num04, num05, num06, num07, num08;
  float mean01, mean02, mean03, mean04, mean05, mean06, mean07, mean08;
  float rms01, rms02, rms03, rms04, rms05, rms06, rms07, rms08;

  for ( int ie = 1; ie <= 85; ie++ ) {
    for ( int ip = 1; ip <= 20; ip++ ) {

      num01  = num02  = num03  = num04  = num05  = num06  = num07  = num08  = -1.;
      mean01 = mean02 = mean03 = mean04 = mean05 = mean06 = mean07 = mean08 = -1.;
      rms01  = rms02  = rms03  = rms04  = rms05  = rms06  = rms07  = rms08  = -1.;

      bool update_channel1 = false;
      bool update_channel2 = false;
      bool update_channel3 = false;
      bool update_channel4 = false;

      if ( h01_[ism-1] && h01_[ism-1]->GetEntries() >= n_min_tot ) {
        num01 = h01_[ism-1]->GetBinEntries(h01_[ism-1]->GetBin(ie, ip));
        if ( num01 >= n_min_bin ) {
          mean01 = h01_[ism-1]->GetBinContent(h01_[ism-1]->GetBin(ie, ip));
          rms01  = h01_[ism-1]->GetBinError(h01_[ism-1]->GetBin(ie, ip));
          update_channel1 = true;
        }
      }

      if ( h02_[ism-1] && h02_[ism-1]->GetEntries() >= n_min_tot ) {
        num02 = h02_[ism-1]->GetBinEntries(h02_[ism-1]->GetBin(ie, ip));
        if ( num02 >= n_min_bin ) {
          mean02 = h02_[ism-1]->GetBinContent(h02_[ism-1]->GetBin(ie, ip));
          rms02  = h02_[ism-1]->GetBinError(h02_[ism-1]->GetBin(ie, ip));
          update_channel1 = true;
        }
      }

      if ( h03_[ism-1] && h03_[ism-1]->GetEntries() >= n_min_tot ) {
        num03 = h03_[ism-1]->GetBinEntries(h03_[ism-1]->GetBin(ie, ip));
        if ( num03 >= n_min_bin ) {
          mean03 = h03_[ism-1]->GetBinContent(h03_[ism-1]->GetBin(ie, ip));
          rms03  = h03_[ism-1]->GetBinError(h03_[ism-1]->GetBin(ie, ip));
          update_channel2 = true;
        }
      }

      if ( h04_[ism-1] && h04_[ism-1]->GetEntries() >= n_min_tot ) {
        num04 = h04_[ism-1]->GetBinEntries(h04_[ism-1]->GetBin(ie, ip));
        if ( num04 >= n_min_bin ) {
          mean04 = h04_[ism-1]->GetBinContent(h04_[ism-1]->GetBin(ie, ip));
          rms04  = h04_[ism-1]->GetBinError(h04_[ism-1]->GetBin(ie, ip));
          update_channel2 = true;
        }
      }

      if ( h05_[ism-1] && h05_[ism-1]->GetEntries() >= n_min_tot ) {
        num05 = h05_[ism-1]->GetBinEntries(h05_[ism-1]->GetBin(ie, ip));
        if ( num05 >= n_min_bin ) {
          mean05 = h05_[ism-1]->GetBinContent(h05_[ism-1]->GetBin(ie, ip));
          rms05  = h05_[ism-1]->GetBinError(h05_[ism-1]->GetBin(ie, ip));
          update_channel3 = true;
        }
      }

      if ( h06_[ism-1] && h06_[ism-1]->GetEntries() >= n_min_tot ) {
        num06 = h06_[ism-1]->GetBinEntries(h06_[ism-1]->GetBin(ie, ip));
        if ( num06 >= n_min_bin ) {
          mean06 = h06_[ism-1]->GetBinContent(h06_[ism-1]->GetBin(ie, ip));
          rms06  = h06_[ism-1]->GetBinError(h06_[ism-1]->GetBin(ie, ip));
          update_channel3 = true;
        }
      }

      if ( h07_[ism-1] && h07_[ism-1]->GetEntries() >= n_min_tot ) {
        num07 = h07_[ism-1]->GetBinEntries(h07_[ism-1]->GetBin(ie, ip));
        if ( num07 >= n_min_bin ) {
          mean07 = h07_[ism-1]->GetBinContent(h07_[ism-1]->GetBin(ie, ip));
          rms07  = h07_[ism-1]->GetBinError(h07_[ism-1]->GetBin(ie, ip));
          update_channel4 = true;
        }
      }

      if ( h08_[ism-1] && h08_[ism-1]->GetEntries() >= n_min_tot ) {
        num08 = h08_[ism-1]->GetBinEntries(h08_[ism-1]->GetBin(ie, ip));
        if ( num08 >= n_min_bin ) {
          mean08 = h08_[ism-1]->GetBinContent(h08_[ism-1]->GetBin(ie, ip));
          rms08  = h08_[ism-1]->GetBinError(h08_[ism-1]->GetBin(ie, ip));
          update_channel4 = true;
        }
      }

      if ( h13_[ism-1] && h13_[ism-1]->GetEntries() >= n_min_tot ) {
        num01 = h13_[ism-1]->GetBinEntries(h13_[ism-1]->GetBin(ie, ip));
        if ( num01 >= n_min_bin ) {
          mean01 = h13_[ism-1]->GetBinContent(h13_[ism-1]->GetBin(ie, ip));
          rms01  = h13_[ism-1]->GetBinError(h13_[ism-1]->GetBin(ie, ip));
          update_channel1 = true;
        }
      }

      if ( h14_[ism-1] && h14_[ism-1]->GetEntries() >= n_min_tot ) {
        num02 = h14_[ism-1]->GetBinEntries(h14_[ism-1]->GetBin(ie, ip));
        if ( num02 >= n_min_bin ) {
          mean02 = h14_[ism-1]->GetBinContent(h14_[ism-1]->GetBin(ie, ip));
          rms02  = h14_[ism-1]->GetBinError(h14_[ism-1]->GetBin(ie, ip));
          update_channel1 = true;
        }
      }

      if ( h15_[ism-1] && h15_[ism-1]->GetEntries() >= n_min_tot ) {
        num03 = h15_[ism-1]->GetBinEntries(h15_[ism-1]->GetBin(ie, ip));
        if ( num03 >= n_min_bin ) {
          mean03 = h15_[ism-1]->GetBinContent(h15_[ism-1]->GetBin(ie, ip));
          rms03  = h15_[ism-1]->GetBinError(h15_[ism-1]->GetBin(ie, ip));
          update_channel2 = true;
        }
      }

      if ( h16_[ism-1] && h16_[ism-1]->GetEntries() >= n_min_tot ) {
        num04 = h16_[ism-1]->GetBinEntries(h16_[ism-1]->GetBin(ie, ip));
        if ( num04 >= n_min_bin ) {
          mean04 = h16_[ism-1]->GetBinContent(h16_[ism-1]->GetBin(ie, ip));
          rms04  = h16_[ism-1]->GetBinError(h16_[ism-1]->GetBin(ie, ip));
          update_channel2 = true;
        }
      }

      if ( h17_[ism-1] && h05_[ism-1]->GetEntries() >= n_min_tot ) {
        num05 = h17_[ism-1]->GetBinEntries(h17_[ism-1]->GetBin(ie, ip));
        if ( num05 >= n_min_bin ) {
          mean05 = h17_[ism-1]->GetBinContent(h17_[ism-1]->GetBin(ie, ip));
          rms05  = h17_[ism-1]->GetBinError(h17_[ism-1]->GetBin(ie, ip));
          update_channel3 = true;
        }
      }

      if ( h18_[ism-1] && h18_[ism-1]->GetEntries() >= n_min_tot ) {
        num06 = h18_[ism-1]->GetBinEntries(h18_[ism-1]->GetBin(ie, ip));
        if ( num06 >= n_min_bin ) {
          mean06 = h18_[ism-1]->GetBinContent(h18_[ism-1]->GetBin(ie, ip));
          rms06  = h18_[ism-1]->GetBinError(h18_[ism-1]->GetBin(ie, ip));
          update_channel3 = true;
        }
      }

      if ( h19_[ism-1] && h19_[ism-1]->GetEntries() >= n_min_tot ) {
        num07 = h19_[ism-1]->GetBinEntries(h19_[ism-1]->GetBin(ie, ip));
        if ( num07 >= n_min_bin ) {
          mean07 = h19_[ism-1]->GetBinContent(h19_[ism-1]->GetBin(ie, ip));
          rms07  = h19_[ism-1]->GetBinError(h19_[ism-1]->GetBin(ie, ip));
          update_channel4 = true;
        }
      }

      if ( h20_[ism-1] && h20_[ism-1]->GetEntries() >= n_min_tot ) {
        num08 = h20_[ism-1]->GetBinEntries(h20_[ism-1]->GetBin(ie, ip));
        if ( num08 >= n_min_bin ) {
          mean08 = h20_[ism-1]->GetBinContent(h20_[ism-1]->GetBin(ie, ip));
          rms08  = h20_[ism-1]->GetBinError(h20_[ism-1]->GetBin(ie, ip));
          update_channel4 = true;
        }
      }

      if ( update_channel1 ) {

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
          status = status && false;
        }

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

      if ( update_channel2 ) {

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
          status = status && false;
        }

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

      if ( update_channel3 ) {

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
          status = status && false;
        }

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

      if ( update_channel4 ) {

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
          status = status && false;
        }

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

  const float m_min_tot = 1000.;
  const float m_min_bin = 50.;

//  float num01, num02, num03, num04, num05, num06, num07, num08;
  float num09, num10, num11, num12, num13, num14, num15, num16;
//  float mean01, mean02, mean03, mean04, mean05, mean06, mean07, mean08;
  float mean09, mean10, mean11, mean12, mean13, mean14, mean15, mean16;
//  float rms01, rms02, rms03, rms04, rms05, rms06, rms07, rms08;
  float rms09, rms10, rms11, rms12, rms13, rms14, rms15, rms16;

  for ( int i = 1; i <= 10; i++ ) {

    num01  = num02  = num03  = num04  = num05  = num06  = num07  = num08  = -1.;
    num09  = num10  = num11  = num12  = num13  = num14  = num15  = num16  = -1.;
    mean01 = mean02 = mean03 = mean04 = mean05 = mean06 = mean07 = mean08 = -1.;
    mean09 = mean10 = mean11 = mean12 = mean13 = mean14 = mean15 = mean16 = -1.;
    rms01  = rms02  = rms03  = rms04  = rms05  = rms06  = rms07  = rms08  = -1.;
    rms09  = rms10  = rms11  = rms12  = rms13  = rms14  = rms15  = rms16  = -1.;

    bool update_channel1 = false;
    bool update_channel2 = false;
    bool update_channel3 = false;
    bool update_channel4 = false;

    if ( i01_[ism-1] && i01_[ism-1]->GetEntries() >= m_min_tot ) {
      num01 = i01_[ism-1]->GetBinEntries(i01_[ism-1]->GetBin(1, i));
      if ( num01 >= m_min_bin ) {
        mean01 = i01_[ism-1]->GetBinContent(i01_[ism-1]->GetBin(1, i));
        rms01  = i01_[ism-1]->GetBinError(i01_[ism-1]->GetBin(1, i));
        update_channel1 = true;
      }
    }

    if ( i02_[ism-1] && i02_[ism-1]->GetEntries() >= m_min_tot ) {
      num02 = i02_[ism-1]->GetBinEntries(i02_[ism-1]->GetBin(1, i));
      if ( num02 >= m_min_bin ) {
        mean02 = i02_[ism-1]->GetBinContent(i02_[ism-1]->GetBin(1, i));
        rms02  = i02_[ism-1]->GetBinError(i02_[ism-1]->GetBin(1, i));
        update_channel2 = true;
      }
    }

    if ( i03_[ism-1] && i03_[ism-1]->GetEntries() >= m_min_tot ) {
      num03 = i03_[ism-1]->GetBinEntries(i03_[ism-1]->GetBin(i));
      if ( num03 >= m_min_bin ) {
        mean03 = i03_[ism-1]->GetBinContent(i03_[ism-1]->GetBin(1, i));
        rms03  = i03_[ism-1]->GetBinError(i03_[ism-1]->GetBin(1, i));
        update_channel3 = true;
      }
    }

    if ( i04_[ism-1] && i04_[ism-1]->GetEntries() >= m_min_tot ) {
      num04 = i04_[ism-1]->GetBinEntries(i04_[ism-1]->GetBin(1, i));
      if ( num04 >= m_min_bin ) {
        mean04 = i04_[ism-1]->GetBinContent(i04_[ism-1]->GetBin(1, i));
        rms04  = i04_[ism-1]->GetBinError(i04_[ism-1]->GetBin(1, i));
        update_channel4 = true;
      }
    }

    if ( i05_[ism-1] && i05_[ism-1]->GetEntries() >= m_min_tot ) {
      num05 = i05_[ism-1]->GetBinEntries(i05_[ism-1]->GetBin(1, i));
      if ( num05 >= m_min_bin ) {
        mean05 = i05_[ism-1]->GetBinContent(i05_[ism-1]->GetBin(1, i));
        rms05  = i05_[ism-1]->GetBinError(i05_[ism-1]->GetBin(1, i));
        update_channel1 = true;
      }
    }

    if ( i06_[ism-1] && i06_[ism-1]->GetEntries() >= m_min_tot ) {
      num06 = i06_[ism-1]->GetBinEntries(i06_[ism-1]->GetBin(1, i));
      if ( num06 >= m_min_bin ) {
        mean06 = i06_[ism-1]->GetBinContent(i06_[ism-1]->GetBin(1, i));
        rms06  = i06_[ism-1]->GetBinError(i06_[ism-1]->GetBin(1, i));
        update_channel2 = true;
      }
    }

    if ( i07_[ism-1] && i07_[ism-1]->GetEntries() >= m_min_tot ) {
      num07 = i07_[ism-1]->GetBinEntries(i07_[ism-1]->GetBin(1, i));
      if ( num07 >= m_min_bin ) {
        mean07 = i07_[ism-1]->GetBinContent(i07_[ism-1]->GetBin(1, i));
        rms07  = i07_[ism-1]->GetBinError(i07_[ism-1]->GetBin(1, i));
        update_channel3 = true;
      }
    }

    if ( i08_[ism-1] && i08_[ism-1]->GetEntries() >= m_min_tot ) {
      num08 = i08_[ism-1]->GetBinEntries(i08_[ism-1]->GetBin(1, i));
      if ( num08 >= m_min_bin ) {
        mean08 = i08_[ism-1]->GetBinContent(i08_[ism-1]->GetBin(1, i));
        rms08  = i08_[ism-1]->GetBinError(i08_[ism-1]->GetBin(1, i));
        update_channel4 = true;
      }
    }

    if ( j01_[ism-1] && j01_[ism-1]->GetEntries() >= m_min_tot ) {
      num09 = j01_[ism-1]->GetBinEntries(j01_[ism-1]->GetBin(1, i));
      if ( num09 >= m_min_bin ) {
        mean09 = j01_[ism-1]->GetBinContent(j01_[ism-1]->GetBin(1, i));
        rms09  = j01_[ism-1]->GetBinError(j01_[ism-1]->GetBin(1, i));
        update_channel1 = true;
      }
    }

    if ( j02_[ism-1] && j02_[ism-1]->GetEntries() >= m_min_tot ) {
      num10 = j02_[ism-1]->GetBinEntries(j02_[ism-1]->GetBin(1, i));
      if ( num10 >= m_min_bin ) {
        mean10 = j02_[ism-1]->GetBinContent(j02_[ism-1]->GetBin(1, i));
        rms10  = j02_[ism-1]->GetBinError(j02_[ism-1]->GetBin(1, i));
        update_channel2 = true;
      }
    }

    if ( j03_[ism-1] && j03_[ism-1]->GetEntries() >= m_min_tot ) {
      num11 = j03_[ism-1]->GetBinEntries(j03_[ism-1]->GetBin(i));
      if ( num11 >= m_min_bin ) {
        mean11 = j03_[ism-1]->GetBinContent(j03_[ism-1]->GetBin(1, i));
        rms11  = j03_[ism-1]->GetBinError(j03_[ism-1]->GetBin(1, i));
        update_channel3 = true;
      }
    }

    if ( j04_[ism-1] && j04_[ism-1]->GetEntries() >= m_min_tot ) {
      num12 = j04_[ism-1]->GetBinEntries(j04_[ism-1]->GetBin(1, i));
      if ( num12 >= m_min_bin ) {
        mean12 = j04_[ism-1]->GetBinContent(j04_[ism-1]->GetBin(1, i));
        rms12  = j04_[ism-1]->GetBinError(j04_[ism-1]->GetBin(1, i));
        update_channel4 = true;
      }
    }

    if ( j05_[ism-1] && j05_[ism-1]->GetEntries() >= m_min_tot ) {
      num13 = j05_[ism-1]->GetBinEntries(j05_[ism-1]->GetBin(1, i));
      if ( num13 >= m_min_bin ) {
        mean13 = j05_[ism-1]->GetBinContent(j05_[ism-1]->GetBin(1, i));
        rms13  = j05_[ism-1]->GetBinError(j05_[ism-1]->GetBin(1, i));
        update_channel1 = true;
      }
    }

    if ( j06_[ism-1] && j06_[ism-1]->GetEntries() >= m_min_tot ) {
      num14 = j06_[ism-1]->GetBinEntries(j06_[ism-1]->GetBin(1, i));
      if ( num14 >= m_min_bin ) {
        mean14 = j06_[ism-1]->GetBinContent(j06_[ism-1]->GetBin(1, i));
        rms14  = j06_[ism-1]->GetBinError(j06_[ism-1]->GetBin(1, i));
        update_channel2 = true;
      }
    }

    if ( j07_[ism-1] && j07_[ism-1]->GetEntries() >= m_min_tot ) {
      num15 = j07_[ism-1]->GetBinEntries(j07_[ism-1]->GetBin(1, i));
      if ( num15 >= m_min_bin ) {
        mean15 = j07_[ism-1]->GetBinContent(j07_[ism-1]->GetBin(1, i));
        rms15  = j07_[ism-1]->GetBinError(j07_[ism-1]->GetBin(1, i));
        update_channel3 = true;
      }
    }

    if ( j08_[ism-1] && j08_[ism-1]->GetEntries() >= m_min_tot ) {
      num16 = j08_[ism-1]->GetBinEntries(j08_[ism-1]->GetBin(1, i));
      if ( num16 >= m_min_bin ) {
        mean16 = j08_[ism-1]->GetBinContent(j08_[ism-1]->GetBin(1, i));
        rms16  = j08_[ism-1]->GetBinError(j08_[ism-1]->GetBin(1, i));
        update_channel4 = true;
      }
    }

    if ( update_channel1 ) {

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

      if ( meg05_[ism-1] && int(meg05_[ism-1]->getBinContent( i, 1 )) % 3 == 1. ) {
        pn_bl.setTaskStatus(true);
      } else {
        pn_bl.setTaskStatus(false);
        status = status && false;
      }

      if ( econn ) {
        try {
          ecid = econn->getEcalLogicID("EB_LM_PN", ism, i-1);
          dataset2_bl[ecid] = pn_bl;
        } catch (runtime_error &e) {
          cerr << e.what() << endl;
        }
      }

    }

    if ( update_channel2 ) {

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

      if ( meg06_[ism-1] && int(meg06_[ism-1]->getBinContent( i, 1 )) % 3 == 1. ) {
        pn_ir.setTaskStatus(true);
      } else {
        pn_ir.setTaskStatus(false);
        status = status && false;
      }

      if ( econn ) {
        try {
          ecid = econn->getEcalLogicID("EB_LM_PN", ism, i-1);
          dataset2_ir[ecid] = pn_ir;
        } catch (runtime_error &e) {
          cerr << e.what() << endl;
        }
      }

    }

    if ( update_channel3 ) {

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

      if ( meg07_[ism-1] && int(meg07_[ism-1]->getBinContent( i, 1 )) % 3 == 1. ) {
        pn_gr.setTaskStatus(true);
      } else {
        pn_gr.setTaskStatus(false);
        status = status && false;
      }

      if ( econn ) {
        try {
          ecid = econn->getEcalLogicID("EB_LM_PN", ism, i-1);
          dataset2_gr[ecid] = pn_gr;
        } catch (runtime_error &e) {
          cerr << e.what() << endl;
        }
      }

    }

    if ( update_channel4 ) {

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

      if ( meg08_[ism-1] && int(meg08_[ism-1]->getBinContent( i, 1 )) % 3 == 1. ) {
        pn_rd.setTaskStatus(true);
      } else {
        pn_rd.setTaskStatus(false);
        status = status && false;
      }

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
      me_j01_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser1/Gain16");
      sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser1/Gain16/EBPDT PNs amplitude SM%02d G16 L1", ism);
      mui_->add(me_j01_[ism-1], histo);

      sprintf(histo, "EBPDT PNs amplitude SM%02d G16 L2", ism);
      me_j02_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser2/Gain16");
      sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser2/Gain16/EBPDT PNs amplitude SM%02d G16 L2", ism);
      mui_->add(me_j02_[ism-1], histo);

      sprintf(histo, "EBPDT PNs amplitude SM%02d G16 L3", ism);
      me_j03_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser3/Gain16");
      sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser3/Gain16/EBPDT PNs amplitude SM%02d G16 L3", ism);
      mui_->add(me_j03_[ism-1], histo);

      sprintf(histo, "EBPDT PNs amplitude SM%02d G16 L4", ism);
      me_j04_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser4/Gain16");
      sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser4/Gain16/EBPDT PNs amplitude SM%02d G16 L4", ism);
      mui_->add(me_j04_[ism-1], histo);

      sprintf(histo, "EBPDT PNs pedestal SM%02d G16 L1", ism);
      me_j05_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser1/Gain16");
      sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser1/Gain16/EBPDT PNs pedestal SM%02d G16 L1", ism);
      mui_->add(me_j05_[ism-1], histo);

      sprintf(histo, "EBPDT PNs pedestal SM%02d G16 L2", ism);
      me_j06_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser2/Gain16");
      sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser2/Gain16/EBPDT PNs pedestal SM%02d G16 L2", ism);
      mui_->add(me_j06_[ism-1], histo);

      sprintf(histo, "EBPDT PNs pedestal SM%02d G16 L3", ism);
      me_j07_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser3/Gain16");
      sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser3/Gain16/EBPDT PNs pedestal SM%02d G16 L3", ism);
      mui_->add(me_j07_[ism-1], histo);

      sprintf(histo, "EBPDT PNs pedestal SM%02d G16 L4", ism);
      me_j08_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser4/Gain16");
      sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser4/Gain16/EBPDT PNs pedestal SM%02d G16 L4", ism);
      mui_->add(me_j08_[ism-1], histo);

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
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser1/Gain16/EBPDT PNs amplitude SM%02d G16 L1", ism);
      if ( qth09_[ism-1] ) mui_->useQTest(histo, qth09_[ism-1]->getName());
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser2/Gain16/EBPDT PNs amplitude SM%02d G16 L2", ism);
      if ( qth10_[ism-1] ) mui_->useQTest(histo, qth10_[ism-1]->getName());
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser3/Gain16/EBPDT PNs amplitude SM%02d G16 L3", ism);
      if ( qth11_[ism-1] ) mui_->useQTest(histo, qth11_[ism-1]->getName());
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser4/Gain16/EBPDT PNs amplitude SM%02d G16 L4", ism);
      if ( qth12_[ism-1] ) mui_->useQTest(histo, qth12_[ism-1]->getName());
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser1/Gain16/EBPDT PNs pedestal SM%02d G16 L1", ism);
      if ( qth13_[ism-1] ) mui_->useQTest(histo, qth13_[ism-1]->getName());
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser2/Gain16/EBPDT PNs pedestal SM%02d G16 L2", ism);
      if ( qth14_[ism-1] ) mui_->useQTest(histo, qth14_[ism-1]->getName());
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser3/Gain16/EBPDT PNs pedestal SM%02d G16 L3", ism);
      if ( qth15_[ism-1] ) mui_->useQTest(histo, qth15_[ism-1]->getName());
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser4/Gain16/EBPDT PNs pedestal SM%02d G16 L4", ism);
      if ( qth16_[ism-1] ) mui_->useQTest(histo, qth16_[ism-1]->getName());
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
        sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser1/Gain16/EBPDT PNs amplitude SM%02d G16 L1", ism);
        if ( qth09_[ism-1] ) mui_->useQTest(histo, qth09_[ism-1]->getName());
        sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser2/Gain16/EBPDT PNs amplitude SM%02d G16 L2", ism);
        if ( qth10_[ism-1] ) mui_->useQTest(histo, qth10_[ism-1]->getName());
        sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser3/Gain16/EBPDT PNs amplitude SM%02d G16 L3", ism);
        if ( qth11_[ism-1] ) mui_->useQTest(histo, qth11_[ism-1]->getName());
        sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser4/Gain16/EBPDT PNs amplitude SM%02d G16 L4", ism);
        if ( qth12_[ism-1] ) mui_->useQTest(histo, qth12_[ism-1]->getName());
        sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser1/Gain16/EBPDT PNs pedestal SM%02d G16 L1", ism);
        if ( qth13_[ism-1] ) mui_->useQTest(histo, qth13_[ism-1]->getName());
        sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser2/Gain16/EBPDT PNs pedestal SM%02d G16 L2", ism);
        if ( qth14_[ism-1] ) mui_->useQTest(histo, qth14_[ism-1]->getName());
        sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser3/Gain16/EBPDT PNs pedestal SM%02d G16 L3", ism);
        if ( qth15_[ism-1] ) mui_->useQTest(histo, qth15_[ism-1]->getName());
        sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser4/Gain16/EBPDT PNs pedestal SM%02d G16 L4", ism);
        if ( qth16_[ism-1] ) mui_->useQTest(histo, qth16_[ism-1]->getName());
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
        sprintf(histo, "EcalBarrel/EBPnDiodeTask/Laser1/Gain16/EBPDT PNs amplitude SM%02d G16 L1", ism);
        if ( qth09_[ism-1] ) mui_->useQTest(histo, qth09_[ism-1]->getName());
        sprintf(histo, "EcalBarrel/EBPnDiodeTask/Laser2/Gain16/EBPDT PNs amplitude SM%02d G16 L2", ism);
        if ( qth10_[ism-1] ) mui_->useQTest(histo, qth10_[ism-1]->getName());
        sprintf(histo, "EcalBarrel/EBPnDiodeTask/Laser3/Gain16/EBPDT PNs amplitude SM%02d G16 L3", ism);
        if ( qth11_[ism-1] ) mui_->useQTest(histo, qth11_[ism-1]->getName());
        sprintf(histo, "EcalBarrel/EBPnDiodeTask/Laser4/Gain16/EBPDT PNs amplitude SM%02d G16 L4", ism);
        if ( qth12_[ism-1] ) mui_->useQTest(histo, qth12_[ism-1]->getName());
        sprintf(histo, "EcalBarrel/EBPnDiodeTask/Laser1/Gain16/EBPDT PNs pedestal SM%02d G16 L1", ism);
        if ( qth13_[ism-1] ) mui_->useQTest(histo, qth13_[ism-1]->getName());
        sprintf(histo, "EcalBarrel/EBPnDiodeTask/Laser2/Gain16/EBPDT PNs pedestal SM%02d G16 L2", ism);
        if ( qth14_[ism-1] ) mui_->useQTest(histo, qth14_[ism-1]->getName());
        sprintf(histo, "EcalBarrel/EBPnDiodeTask/Laser3/Gain16/EBPDT PNs pedestal SM%02d G16 L3", ism);
        if ( qth15_[ism-1] ) mui_->useQTest(histo, qth15_[ism-1]->getName());
        sprintf(histo, "EcalBarrel/EBPnDiodeTask/Laser4/Gain16/EBPDT PNs pedestal SM%02d G16 L4", ism);
        if ( qth16_[ism-1] ) mui_->useQTest(histo, qth16_[ism-1]->getName());
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

        mui_->removeCollate(me_j01_[ism-1]);
        mui_->removeCollate(me_j02_[ism-1]);
        mui_->removeCollate(me_j03_[ism-1]);
        mui_->removeCollate(me_j04_[ism-1]);
        mui_->removeCollate(me_j05_[ism-1]);
        mui_->removeCollate(me_j06_[ism-1]);
        mui_->removeCollate(me_j07_[ism-1]);
        mui_->removeCollate(me_j08_[ism-1]);

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
  bits02 |= EcalErrorDictionary::getMask("PEDESTAL_HIGH_GAIN_MEAN_WARNING");
  bits02 |= EcalErrorDictionary::getMask("PEDESTAL_HIGH_GAIN_RMS_WARNING");
  bits02 |= EcalErrorDictionary::getMask("PEDESTAL_HIGH_GAIN_MEAN_ERROR");
  bits02 |= EcalErrorDictionary::getMask("PEDESTAL_HIGH_GAIN_RMS_ERROR");

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
    j01_[ism-1] = EBMUtilsClient::getHisto<TProfile2D*>( me, cloneME_, j01_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser2/Gain16/EBPDT PNs amplitude SM%02d G16 L2", ism);
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBPnDiodeTask/Laser2/Gain16/EBPDT PNs amplitude SM%02d G16 L2").c_str(), ism);
    }
    me = mui_->get(histo);
    j02_[ism-1] = EBMUtilsClient::getHisto<TProfile2D*>( me, cloneME_, j02_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser3/Gain16/EBPDT PNs amplitude SM%02d G16 L3", ism);
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBPnDiodeTask/Laser3/Gain16/EBPDT PNs amplitude SM%02d G16 L3").c_str(), ism);
    }
    me = mui_->get(histo);
    j03_[ism-1] = EBMUtilsClient::getHisto<TProfile2D*>( me, cloneME_, j03_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser4/Gain16/EBPDT PNs amplitude SM%02d G16 L4", ism);
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBPnDiodeTask/Laser4/Gain16/EBPDT PNs amplitude SM%02d G16 L4").c_str(), ism);
    }
    me = mui_->get(histo);
    j04_[ism-1] = EBMUtilsClient::getHisto<TProfile2D*>( me, cloneME_, j04_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser1/Gain16/EBPDT PNs pedestal SM%02d G16 L1", ism);
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBPnDiodeTask/Laser1/Gain16/EBPDT PNs pedestal SM%02d G16 L1").c_str(), ism);
    }
    me = mui_->get(histo);
    j05_[ism-1] = EBMUtilsClient::getHisto<TProfile2D*>( me, cloneME_, j05_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser2/Gain16/EBPDT PNs pedestal SM%02d G16 L2", ism);
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBPnDiodeTask/Laser2/Gain16/EBPDT PNs pedestal SM%02d G16 L2").c_str(), ism);
    }
    me = mui_->get(histo);
    j06_[ism-1] = EBMUtilsClient::getHisto<TProfile2D*>( me, cloneME_, j06_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser3/Gain16/EBPDT PNs pedestal SM%02d G16 L3", ism);
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBPnDiodeTask/Laser3/Gain16/EBPDT PNs pedestal SM%02d G16 L3").c_str(), ism);
    }
    me = mui_->get(histo);
    j07_[ism-1] = EBMUtilsClient::getHisto<TProfile2D*>( me, cloneME_, j07_[ism-1] );

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser4/Gain16/EBPDT PNs pedestal SM%02d G16 L4", ism);
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBPnDiodeTask/Laser4/Gain16/EBPDT PNs pedestal SM%02d G16 L4").c_str(), ism);
    }
    me = mui_->get(histo);
    j08_[ism-1] = EBMUtilsClient::getHisto<TProfile2D*>( me, cloneME_, j08_[ism-1] );

    const float n_min_tot = 1000.;
    const float n_min_bin = 50.;

    float num01, num02, num03, num04, num05, num06, num07, num08;
    float num09, num10, num11, num12;
    float mean01, mean02, mean03, mean04, mean05, mean06, mean07, mean08;
    float mean09, mean10, mean11, mean12;
    float rms01, rms02, rms03, rms04, rms05, rms06, rms07, rms08;
    float rms09, rms10, rms11, rms12;

    EBMUtilsClient::resetHisto( meg01_[ism-1] );
    EBMUtilsClient::resetHisto( meg02_[ism-1] );
    EBMUtilsClient::resetHisto( meg03_[ism-1] );
    EBMUtilsClient::resetHisto( meg04_[ism-1] );

    EBMUtilsClient::resetHisto( meg05_[ism-1] );
    EBMUtilsClient::resetHisto( meg06_[ism-1] );
    EBMUtilsClient::resetHisto( meg07_[ism-1] );
    EBMUtilsClient::resetHisto( meg08_[ism-1] );

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
    meanAmplL1A = meanAmplL2A = meanAmplL3A = meanAmplL4A = -1.;
    meanAmplL1B = meanAmplL2B = meanAmplL3B = meanAmplL4B = -1.;
    nCryL1A = nCryL2A = nCryL3A = nCryL4A = 0;
    nCryL1B = nCryL2B = nCryL3B = nCryL4B = 0;

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

        num01 = num03 = num05 = num07 = -1;

        if ( h01_[ism-1] && h01_[ism-1]->GetEntries() >= n_min_tot ) {
          num01 = h01_[ism-1]->GetBinEntries(h01_[ism-1]->GetBin(ie, ip));
          if ( num01 >= n_min_bin ) {
            meanAmplL1A += h01_[ism-1]->GetBinContent(h01_[ism-1]->GetBin(ie, ip));
            nCryL1A++;
          }
        }

        if ( h03_[ism-1] && h03_[ism-1]->GetEntries() >= n_min_tot ) {
          num03 = h03_[ism-1]->GetBinEntries(h03_[ism-1]->GetBin(ie, ip));
          if ( num03 >= n_min_bin ) {
            meanAmplL2A += h03_[ism-1]->GetBinContent(h03_[ism-1]->GetBin(ie, ip));
            nCryL2A++;
          }
        }

        if ( h05_[ism-1] && h05_[ism-1]->GetEntries() >= n_min_tot ) {
          num05 = h05_[ism-1]->GetBinEntries(h05_[ism-1]->GetBin(ie, ip));
          if ( num05 >= n_min_bin ) {
            meanAmplL3A += h05_[ism-1]->GetBinContent(h05_[ism-1]->GetBin(ie, ip));
            nCryL3A++;
          }
        }

        if ( h07_[ism-1] && h07_[ism-1]->GetEntries() >= n_min_tot ) {
          num07 = h07_[ism-1]->GetBinEntries(h07_[ism-1]->GetBin(ie, ip));
          if ( num07 >= n_min_bin ) {
            meanAmplL4A += h07_[ism-1]->GetBinContent(h07_[ism-1]->GetBin(ie, ip));
            nCryL4A++;
          }
        }

        num01 = num03 = num05 = num07 = -1;

        if ( h13_[ism-1] && h13_[ism-1]->GetEntries() >= n_min_tot ) {
          num01 = h13_[ism-1]->GetBinEntries(h13_[ism-1]->GetBin(ie, ip));
          if ( num01 >= n_min_bin ) {
            meanAmplL1B += h13_[ism-1]->GetBinContent(h13_[ism-1]->GetBin(ie, ip));
            nCryL1B++;
          }
        }

        if ( h15_[ism-1] && h15_[ism-1]->GetEntries() >= n_min_tot ) {
          num03 = h15_[ism-1]->GetBinEntries(h15_[ism-1]->GetBin(ie, ip));
          if ( num03 >= n_min_bin ) {
            meanAmplL2B += h15_[ism-1]->GetBinContent(h15_[ism-1]->GetBin(ie, ip));
            nCryL2B++;
          }
        }
    
        if ( h17_[ism-1] && h17_[ism-1]->GetEntries() >= n_min_tot ) {
          num05 = h17_[ism-1]->GetBinEntries(h17_[ism-1]->GetBin(ie, ip));
          if ( num05 >= n_min_bin ) {
            meanAmplL3B += h17_[ism-1]->GetBinContent(h17_[ism-1]->GetBin(ie, ip));
            nCryL3B++;
          }
        }
    
        if ( h19_[ism-1] && h19_[ism-1]->GetEntries() >= n_min_tot ) {
          num07 = h19_[ism-1]->GetBinEntries(h19_[ism-1]->GetBin(ie, ip));
          if ( num07 >= n_min_bin ) {
            meanAmplL4B += h19_[ism-1]->GetBinContent(h19_[ism-1]->GetBin(ie, ip));
            nCryL4B++;
          }
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

        num01  = num02  = num03  = num04  = num05  = num06  = num07  = num08  = -1.;
        num09  = num10  = num11  = num12  = -1.;
        mean01 = mean02 = mean03 = mean04 = mean05 = mean06 = mean07 = mean08 = -1.;
        mean09 = mean10 = mean11 = mean12 = -1.;
        rms01  = rms02  = rms03  = rms04  = rms05  = rms06  = rms07  = rms08  = -1.;
        rms09  = rms10  = rms11  = rms12  = -1.;

        if ( meg01_[ism-1] ) meg01_[ism-1]->setBinContent( ie, ip, 2.);
        if ( meg02_[ism-1] ) meg02_[ism-1]->setBinContent( ie, ip, 2.);
        if ( meg03_[ism-1] ) meg03_[ism-1]->setBinContent( ie, ip, 2.);
        if ( meg04_[ism-1] ) meg04_[ism-1]->setBinContent( ie, ip, 2.);

        bool update_channel1 = false;
        bool update_channel2 = false;
        bool update_channel3 = false;
        bool update_channel4 = false;

        if ( h01_[ism-1] && h01_[ism-1]->GetEntries() >= n_min_tot ) {
          num01 = h01_[ism-1]->GetBinEntries(h01_[ism-1]->GetBin(ie, ip));
          if ( num01 >= n_min_bin ) {
            mean01 = h01_[ism-1]->GetBinContent(h01_[ism-1]->GetBin(ie, ip));
            rms01  = h01_[ism-1]->GetBinError(h01_[ism-1]->GetBin(ie, ip));
            update_channel1 = true;
          }
        }

        if ( h02_[ism-1] && h02_[ism-1]->GetEntries() >= n_min_tot ) {
          num02 = h02_[ism-1]->GetBinEntries(h02_[ism-1]->GetBin(ie, ip));
          if ( num02 >= n_min_bin ) {
            mean02 = h02_[ism-1]->GetBinContent(h02_[ism-1]->GetBin(ie, ip));
            rms02  = h02_[ism-1]->GetBinError(h02_[ism-1]->GetBin(ie, ip));
            update_channel1 = true;
          }
        }

        if ( h03_[ism-1] && h03_[ism-1]->GetEntries() >= n_min_tot ) {
          num03 = h03_[ism-1]->GetBinEntries(h03_[ism-1]->GetBin(ie, ip));
          if ( num03 >= n_min_bin ) {
            mean03 = h03_[ism-1]->GetBinContent(h03_[ism-1]->GetBin(ie, ip));
            rms03  = h03_[ism-1]->GetBinError(h03_[ism-1]->GetBin(ie, ip));
            update_channel2 = true;
          }
        }

        if ( h04_[ism-1] && h04_[ism-1]->GetEntries() >= n_min_tot ) {
          num04 = h04_[ism-1]->GetBinEntries(h04_[ism-1]->GetBin(ie, ip));
          if ( num04 >= n_min_bin ) {
            mean04 = h04_[ism-1]->GetBinContent(h04_[ism-1]->GetBin(ie, ip));
            rms04  = h04_[ism-1]->GetBinError(h04_[ism-1]->GetBin(ie, ip));
            update_channel2 = true;
          }
        }

        if ( h05_[ism-1] && h05_[ism-1]->GetEntries() >= n_min_tot ) {
          num05 = h05_[ism-1]->GetBinEntries(h05_[ism-1]->GetBin(ie, ip));
          if ( num05 >= n_min_bin ) {
            mean05 = h05_[ism-1]->GetBinContent(h05_[ism-1]->GetBin(ie, ip));
            rms05  = h05_[ism-1]->GetBinError(h05_[ism-1]->GetBin(ie, ip));
            update_channel3 = true;
          }
        }

        if ( h06_[ism-1] && h06_[ism-1]->GetEntries() >= n_min_tot ) {
          num06 = h06_[ism-1]->GetBinEntries(h06_[ism-1]->GetBin(ie, ip));
          if ( num06 >= n_min_bin ) {
            mean06 = h06_[ism-1]->GetBinContent(h06_[ism-1]->GetBin(ie, ip));
            rms06  = h06_[ism-1]->GetBinError(h06_[ism-1]->GetBin(ie, ip));
            update_channel3 = true;
          }
        }

        if ( h07_[ism-1] && h07_[ism-1]->GetEntries() >= n_min_tot ) {
          num07 = h07_[ism-1]->GetBinEntries(h07_[ism-1]->GetBin(ie, ip));
          if ( num07 >= n_min_bin ) {
            mean07 = h07_[ism-1]->GetBinContent(h07_[ism-1]->GetBin(ie, ip));
            rms07  = h07_[ism-1]->GetBinError(h07_[ism-1]->GetBin(ie, ip));
            update_channel4 = true;
          }
        }

        if ( h08_[ism-1] && h08_[ism-1]->GetEntries() >= n_min_tot ) {
          num08 = h08_[ism-1]->GetBinEntries(h08_[ism-1]->GetBin(ie, ip));
          if ( num08 >= n_min_bin ) {
            mean08 = h08_[ism-1]->GetBinContent(h08_[ism-1]->GetBin(ie, ip));
            rms08  = h08_[ism-1]->GetBinError(h08_[ism-1]->GetBin(ie, ip));
            update_channel4 = true;
          }
        }

        if ( h09_[ism-1] && h09_[ism-1]->GetEntries() >= n_min_tot ) {
          num09 = h09_[ism-1]->GetBinEntries(h09_[ism-1]->GetBin(ie, ip));
          if ( num09 >= n_min_bin ) {
            mean09 = h09_[ism-1]->GetBinContent(h09_[ism-1]->GetBin(ie, ip));
            rms09  = h09_[ism-1]->GetBinError(h09_[ism-1]->GetBin(ie, ip));
            update_channel1 = true;
          }
        }

        if ( h10_[ism-1] && h10_[ism-1]->GetEntries() >= n_min_tot ) {
          num10 = h10_[ism-1]->GetBinEntries(h10_[ism-1]->GetBin(ie, ip));
          if ( num10 >= n_min_bin ) {
            mean10 = h10_[ism-1]->GetBinContent(h10_[ism-1]->GetBin(ie, ip));
            rms10  = h10_[ism-1]->GetBinError(h10_[ism-1]->GetBin(ie, ip));
            update_channel2 = true;
          }
        }

        if ( h11_[ism-1] && h11_[ism-1]->GetEntries() >= n_min_tot ) {
          num11 = h11_[ism-1]->GetBinEntries(h11_[ism-1]->GetBin(ie, ip));
          if ( num11 >= n_min_bin ) {
            mean11 = h11_[ism-1]->GetBinContent(h11_[ism-1]->GetBin(ie, ip));
            rms11  = h11_[ism-1]->GetBinError(h11_[ism-1]->GetBin(ie, ip));
            update_channel3 = true;
          }
        }

        if ( h12_[ism-1] && h12_[ism-1]->GetEntries() >= n_min_tot ) {
          num12 = h12_[ism-1]->GetBinEntries(h12_[ism-1]->GetBin(ie, ip));
          if ( num12 >= n_min_bin ) {
            mean12 = h12_[ism-1]->GetBinContent(h12_[ism-1]->GetBin(ie, ip));
            rms12  = h12_[ism-1]->GetBinError(h12_[ism-1]->GetBin(ie, ip));
            update_channel4 = true;
          }
        }

        if ( h13_[ism-1] && h13_[ism-1]->GetEntries() >= n_min_tot ) {
          num01 = h13_[ism-1]->GetBinEntries(h13_[ism-1]->GetBin(ie, ip));
          if ( num01 >= n_min_bin ) {
            mean01 = h13_[ism-1]->GetBinContent(h13_[ism-1]->GetBin(ie, ip));
            rms01  = h13_[ism-1]->GetBinError(h13_[ism-1]->GetBin(ie, ip));
            update_channel1 = true;
          }
        }

        if ( h14_[ism-1] && h14_[ism-1]->GetEntries() >= n_min_tot ) {
          num02 = h14_[ism-1]->GetBinEntries(h14_[ism-1]->GetBin(ie, ip));
          if ( num02 >= n_min_bin ) {
            mean02 = h14_[ism-1]->GetBinContent(h14_[ism-1]->GetBin(ie, ip));
            rms02  = h14_[ism-1]->GetBinError(h14_[ism-1]->GetBin(ie, ip));
            update_channel1 = true;
          }
        }

        if ( h15_[ism-1] && h15_[ism-1]->GetEntries() >= n_min_tot ) {
          num03 = h15_[ism-1]->GetBinEntries(h15_[ism-1]->GetBin(ie, ip));
          if ( num03 >= n_min_bin ) {
            mean03 = h15_[ism-1]->GetBinContent(h15_[ism-1]->GetBin(ie, ip));
            rms03  = h15_[ism-1]->GetBinError(h15_[ism-1]->GetBin(ie, ip));
            update_channel2 = true;
          }
        }

        if ( h16_[ism-1] && h16_[ism-1]->GetEntries() >= n_min_tot ) {
          num04 = h16_[ism-1]->GetBinEntries(h16_[ism-1]->GetBin(ie, ip));
          if ( num04 >= n_min_bin ) {
            mean04 = h16_[ism-1]->GetBinContent(h16_[ism-1]->GetBin(ie, ip));
            rms04  = h16_[ism-1]->GetBinError(h16_[ism-1]->GetBin(ie, ip));
            update_channel2 = true;
          }
        }

        if ( h17_[ism-1] && h17_[ism-1]->GetEntries() >= n_min_tot ) {
          num05 = h17_[ism-1]->GetBinEntries(h17_[ism-1]->GetBin(ie, ip));
          if ( num05 >= n_min_bin ) {
            mean05 = h17_[ism-1]->GetBinContent(h17_[ism-1]->GetBin(ie, ip));
            rms05  = h17_[ism-1]->GetBinError(h17_[ism-1]->GetBin(ie, ip));
            update_channel3 = true;
          }
        }

        if ( h18_[ism-1] && h18_[ism-1]->GetEntries() >= n_min_tot ) {
          num06 = h18_[ism-1]->GetBinEntries(h18_[ism-1]->GetBin(ie, ip));
          if ( num06 >= n_min_bin ) {
            mean06 = h18_[ism-1]->GetBinContent(h18_[ism-1]->GetBin(ie, ip));
            rms06  = h18_[ism-1]->GetBinError(h18_[ism-1]->GetBin(ie, ip));
            update_channel3 = true;
          }
        }

        if ( h19_[ism-1] && h19_[ism-1]->GetEntries() >= n_min_tot ) {
          num07 = h19_[ism-1]->GetBinEntries(h19_[ism-1]->GetBin(ie, ip));
          if ( num07 >= n_min_bin ) {
            mean07 = h19_[ism-1]->GetBinContent(h19_[ism-1]->GetBin(ie, ip));
            rms07  = h19_[ism-1]->GetBinError(h19_[ism-1]->GetBin(ie, ip));
            update_channel4 = true;
          }
        }

        if ( h20_[ism-1] && h20_[ism-1]->GetEntries() >= n_min_tot ) {
          num08 = h20_[ism-1]->GetBinEntries(h20_[ism-1]->GetBin(ie, ip));
          if ( num08 >= n_min_bin ) {
            mean08 = h20_[ism-1]->GetBinContent(h20_[ism-1]->GetBin(ie, ip));
            rms08  = h20_[ism-1]->GetBinError(h20_[ism-1]->GetBin(ie, ip));
            update_channel4 = true;
          }
        }

        if ( h21_[ism-1] && h21_[ism-1]->GetEntries() >= n_min_tot ) {
          num09 = h21_[ism-1]->GetBinEntries(h21_[ism-1]->GetBin(ie, ip));
          if ( num09 >= n_min_bin ) {
            mean09 = h21_[ism-1]->GetBinContent(h21_[ism-1]->GetBin(ie, ip));
            rms09  = h21_[ism-1]->GetBinError(h21_[ism-1]->GetBin(ie, ip));
            update_channel1 = true;
          }
        }

        if ( h22_[ism-1] && h22_[ism-1]->GetEntries() >= n_min_tot ) {
          num10 = h22_[ism-1]->GetBinEntries(h22_[ism-1]->GetBin(ie, ip));
          if ( num10 >= n_min_bin ) {
            mean10 = h22_[ism-1]->GetBinContent(h22_[ism-1]->GetBin(ie, ip));
            rms10  = h22_[ism-1]->GetBinError(h22_[ism-1]->GetBin(ie, ip));
            update_channel2 = true;
          }
        }

        if ( h23_[ism-1] && h23_[ism-1]->GetEntries() >= n_min_tot ) {
          num11 = h23_[ism-1]->GetBinEntries(h23_[ism-1]->GetBin(ie, ip));
          if ( num11 >= n_min_bin ) {
            mean11 = h23_[ism-1]->GetBinContent(h23_[ism-1]->GetBin(ie, ip));
            rms11  = h23_[ism-1]->GetBinError(h23_[ism-1]->GetBin(ie, ip));
            update_channel3 = true;
          }
        }

        if ( h24_[ism-1] && h24_[ism-1]->GetEntries() >= n_min_tot ) {
          num12 = h24_[ism-1]->GetBinEntries(h24_[ism-1]->GetBin(ie, ip));
          if ( num12 >= n_min_bin ) {
            mean12 = h24_[ism-1]->GetBinContent(h24_[ism-1]->GetBin(ie, ip));
            rms12  = h24_[ism-1]->GetBinError(h24_[ism-1]->GetBin(ie, ip));
            update_channel4 = true;
          }
        }

        if ( update_channel1 ) {

          float val;

          if ( ie < 6 || ip > 10 ) {

            val = 1.;
            if ( fabs(mean01 - meanAmplL1A) > fabs(percentVariation_ * meanAmplL1A) )
              val = 0.;
            if ( meg01_[ism-1] ) meg01_[ism-1]->setBinContent( ie, ip, val );

            if ( mea01_[ism-1] ) mea01_[ism-1]->setBinContent( ip+20*(ie-1), mean01 );
            if ( mea01_[ism-1] ) mea01_[ism-1]->setBinError( ip+20*(ie-1), rms01 );

            if ( met01_[ism-1] ) met01_[ism-1]->setBinContent( ip+20*(ie-1), mean09 );
            if ( met01_[ism-1] ) met01_[ism-1]->setBinError( ip+20*(ie-1), rms09 );

            if ( meaopn01_[ism-1] ) meaopn01_[ism-1]->setBinContent( ip+20*(ie-1), mean02 );
            if ( meaopn01_[ism-1] ) meaopn01_[ism-1]->setBinError( ip+20*(ie-1), rms02 );

          } else {

            val = 1.;
            if ( fabs(mean01 - meanAmplL1B) > fabs(percentVariation_ * meanAmplL1B) )
              val = 0.;
            if ( meg01_[ism-1] ) meg01_[ism-1]->setBinContent( ie, ip, val );

            if ( mea05_[ism-1] ) mea05_[ism-1]->setBinContent( ip+20*(ie-1), mean01 );
            if ( mea05_[ism-1] ) mea05_[ism-1]->setBinError( ip+20*(ie-1), rms01 );

            if ( met05_[ism-1] ) met05_[ism-1]->setBinContent( ip+20*(ie-1), mean09 );
            if ( met05_[ism-1] ) met05_[ism-1]->setBinError( ip+20*(ie-1), rms09 );

            if ( meaopn05_[ism-1] ) meaopn05_[ism-1]->setBinContent( ip+20*(ie-1), mean02 );
            if ( meaopn05_[ism-1] ) meaopn05_[ism-1]->setBinError( ip+20*(ie-1), rms02 );

          }

        }

        if ( update_channel2 ) {

          float val;

          if ( ie < 6 || ip > 10 ) {

            val = 1.;
            if ( fabs(mean03 - meanAmplL2A) > fabs(percentVariation_ * meanAmplL2A) )
              val = 0.;
            if ( meg02_[ism-1] ) meg02_[ism-1]->setBinContent( ie, ip, val);

            if ( mea02_[ism-1] ) mea02_[ism-1]->setBinContent( ip+20*(ie-1), mean03 );
            if ( mea02_[ism-1] ) mea02_[ism-1]->setBinError( ip+20*(ie-1), rms03 );

            if ( met02_[ism-1] ) met02_[ism-1]->setBinContent( ip+20*(ie-1), mean10 );
            if ( met02_[ism-1] ) met02_[ism-1]->setBinError( ip+20*(ie-1), rms10 );

            if ( meaopn02_[ism-1] ) meaopn02_[ism-1]->setBinContent( ip+20*(ie-1), mean04 );
            if ( meaopn02_[ism-1] ) meaopn02_[ism-1]->setBinError( ip+20*(ie-1), rms04 );

          } else {

            val = 1.;
            if ( fabs(mean03 - meanAmplL2B) > fabs(percentVariation_ * meanAmplL2B) )
              val = 0.;
            if ( meg02_[ism-1] ) meg02_[ism-1]->setBinContent( ie, ip, val);

            if ( mea06_[ism-1] ) mea06_[ism-1]->setBinContent( ip+20*(ie-1), mean03 );
            if ( mea06_[ism-1] ) mea06_[ism-1]->setBinError( ip+20*(ie-1), rms03 );

            if ( met06_[ism-1] ) met06_[ism-1]->setBinContent( ip+20*(ie-1), mean10 );
            if ( met06_[ism-1] ) met06_[ism-1]->setBinError( ip+20*(ie-1), rms10 );

            if ( meaopn06_[ism-1] ) meaopn06_[ism-1]->setBinContent( ip+20*(ie-1), mean04 );
            if ( meaopn06_[ism-1] ) meaopn06_[ism-1]->setBinError( ip+20*(ie-1), rms04 );

          }

        }

        if ( update_channel3 ) {

          float val;

          if ( ie < 6 || ip > 10 ) {

            val = 1.;
            if ( fabs(mean05 - meanAmplL3A) > fabs(percentVariation_ * meanAmplL3A) )
              val = 0.;
            if ( meg03_[ism-1] ) meg03_[ism-1]->setBinContent( ie, ip, val );

            if ( mea03_[ism-1] ) mea03_[ism-1]->setBinContent( ip+20*(ie-1), mean05 );
            if ( mea03_[ism-1] ) mea03_[ism-1]->setBinError( ip+20*(ie-1), rms05 );

            if ( met03_[ism-1] ) met03_[ism-1]->setBinContent( ip+20*(ie-1), mean11 );
            if ( met03_[ism-1] ) met03_[ism-1]->setBinError( ip+20*(ie-1), rms11 );

            if ( meaopn03_[ism-1] ) meaopn03_[ism-1]->setBinContent( ip+20*(ie-1), mean06 );
            if ( meaopn03_[ism-1] ) meaopn03_[ism-1]->setBinError( ip+20*(ie-1), rms06 );

          } else {

            val = 1.;
            if ( fabs(mean05 - meanAmplL3B) > fabs(percentVariation_ * meanAmplL3B) )
              val = 0.;
            if ( meg03_[ism-1] ) meg03_[ism-1]->setBinContent( ie, ip, val );

            if ( mea07_[ism-1] ) mea07_[ism-1]->setBinContent( ip+20*(ie-1), mean05 );
            if ( mea07_[ism-1] ) mea07_[ism-1]->setBinError( ip+20*(ie-1), rms05 );

            if ( met07_[ism-1] ) met07_[ism-1]->setBinContent( ip+20*(ie-1), mean11 );
            if ( met07_[ism-1] ) met07_[ism-1]->setBinError( ip+20*(ie-1), rms11 );

            if ( meaopn07_[ism-1] ) meaopn07_[ism-1]->setBinContent( ip+20*(ie-1), mean06 );
            if ( meaopn07_[ism-1] ) meaopn07_[ism-1]->setBinError( ip+20*(ie-1), rms06 );

          }

        }

        if ( update_channel4 ) {

          float val;

          if ( ie < 6 || ip > 10 ) {

            val = 1.;
            if ( fabs(mean07 - meanAmplL4A) > fabs(percentVariation_ * meanAmplL4A) )
              val = 0.;
            if ( meg04_[ism-1] ) meg04_[ism-1]->setBinContent( ie, ip, val );

            if ( mea04_[ism-1] ) mea04_[ism-1]->setBinContent( ip+20*(ie-1), mean07 );
            if ( mea04_[ism-1] ) mea04_[ism-1]->setBinError( ip+20*(ie-1), rms07 );

            if ( met04_[ism-1] ) met04_[ism-1]->setBinContent( ip+20*(ie-1), mean12 );
            if ( met04_[ism-1] ) met04_[ism-1]->setBinError( ip+20*(ie-1), rms12 );

            if ( meaopn04_[ism-1] ) meaopn04_[ism-1]->setBinContent( ip+20*(ie-1), mean08 );
            if ( meaopn04_[ism-1] ) meaopn04_[ism-1]->setBinError( ip+20*(ie-1), rms08 );

          } else {

            val = 1.;
            if ( fabs(mean07 - meanAmplL4B) > fabs(percentVariation_ * meanAmplL4B) )
              val = 0.;
            if ( meg04_[ism-1] ) meg04_[ism-1]->setBinContent( ie, ip, val );

            if ( mea08_[ism-1] ) mea08_[ism-1]->setBinContent( ip+20*(ie-1), mean07 );
            if ( mea08_[ism-1] ) mea08_[ism-1]->setBinError( ip+20*(ie-1), rms07 );

            if ( met08_[ism-1] ) met08_[ism-1]->setBinContent( ip+20*(ie-1), mean12 );
            if ( met08_[ism-1] ) met08_[ism-1]->setBinError( ip+20*(ie-1), rms12 );

            if ( meaopn08_[ism-1] ) meaopn08_[ism-1]->setBinContent( ip+20*(ie-1), mean08 );
            if ( meaopn08_[ism-1] ) meaopn08_[ism-1]->setBinError( ip+20*(ie-1), rms08 );

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

    const float m_min_tot = 1000.;
    const float m_min_bin = 50.;

//    float num01, num02, num03, num04, num05, num06, num07, num08;
//    float num09, num10, num11, num12;
    float num13, num14, num15, num16;
//    float mean01, mean02, mean03, mean04, mean05, mean06, mean07, mean08;
//    float mean09, mean10, mean11, mean12;
    float mean13, mean14, mean15, mean16;
//    float rms01, rms02, rms03, rms04, rms05, rms06, rms07, rms08;
//    float rms09, rms10, rms11, rms12;
    float rms13, rms14, rms15, rms16;

    for ( int i = 1; i <= 10; i++ ) {
  
      num01  = num02  = num03  = num04  = num05  = num06  = num07  = num08  = -1.;
      num09  = num10  = num11  = num12  = num13  = num14  = num15  = num16  = -1.;
      mean01 = mean02 = mean03 = mean04 = mean05 = mean06 = mean07 = mean08 = -1.;
      mean09 = mean10 = mean11 = mean12 = mean13 = mean14 = mean15 = mean16 = -1.;
      rms01  = rms02  = rms03  = rms04  = rms05  = rms06  = rms07  = rms08  = -1.;
      rms09  = rms10  = rms11  = rms12  = rms13  = rms14  = rms15  = rms16  = -1.;

      if ( meg05_[ism-1] ) meg05_[ism-1]->setBinContent( i, 1, 2. );
      if ( meg06_[ism-1] ) meg06_[ism-1]->setBinContent( i, 1, 2. );
      if ( meg07_[ism-1] ) meg07_[ism-1]->setBinContent( i, 1, 2. );
      if ( meg08_[ism-1] ) meg08_[ism-1]->setBinContent( i, 1, 2. );

      bool update_channel1 = false;
      bool update_channel2 = false;
      bool update_channel3 = false;
      bool update_channel4 = false;

      if ( i01_[ism-1] && i01_[ism-1]->GetEntries() >= m_min_tot ) {
        num01 = i01_[ism-1]->GetBinEntries(i01_[ism-1]->GetBin(1, i));
        if ( num01 >= m_min_bin ) {
          mean01 = i01_[ism-1]->GetBinContent(i01_[ism-1]->GetBin(1, i));
          rms01  = i01_[ism-1]->GetBinError(i01_[ism-1]->GetBin(1, i));
          update_channel1 = true;
        }
      }

      if ( i02_[ism-1] && i02_[ism-1]->GetEntries() >= m_min_tot ) {
        num02 = i02_[ism-1]->GetBinEntries(i02_[ism-1]->GetBin(1, i));
        if ( num02 >= m_min_bin ) {
          mean02 = i02_[ism-1]->GetBinContent(i02_[ism-1]->GetBin(1, i));
          rms02  = i02_[ism-1]->GetBinError(i02_[ism-1]->GetBin(1, i));
          update_channel2 = true;
        }
      }

      if ( i03_[ism-1] && i03_[ism-1]->GetEntries() >= m_min_tot ) {
        num03 = i03_[ism-1]->GetBinEntries(i03_[ism-1]->GetBin(i));
        if ( num03 >= m_min_bin ) {
          mean03 = i03_[ism-1]->GetBinContent(i03_[ism-1]->GetBin(1, i));
          rms03  = i03_[ism-1]->GetBinError(i03_[ism-1]->GetBin(1, i));
          update_channel3 = true;
        }
      }

      if ( i04_[ism-1] && i04_[ism-1]->GetEntries() >= m_min_tot ) {
        num04 = i04_[ism-1]->GetBinEntries(i04_[ism-1]->GetBin(1, i));
        if ( num04 >= m_min_bin ) {
          mean04 = i04_[ism-1]->GetBinContent(i04_[ism-1]->GetBin(1, i));
          rms04  = i04_[ism-1]->GetBinError(i04_[ism-1]->GetBin(1, i));
          update_channel4 = true;
        }
      }

      if ( i05_[ism-1] && i05_[ism-1]->GetEntries() >= m_min_tot ) {
        num05 = i05_[ism-1]->GetBinEntries(i05_[ism-1]->GetBin(1, i));
        if ( num05 >= m_min_bin ) {
          mean05 = i05_[ism-1]->GetBinContent(i05_[ism-1]->GetBin(1, i));
          rms05  = i05_[ism-1]->GetBinError(i05_[ism-1]->GetBin(1, i));
          update_channel1 = true;
        }
      }

      if ( i06_[ism-1] && i06_[ism-1]->GetEntries() >= m_min_tot ) {
        num06 = i06_[ism-1]->GetBinEntries(i06_[ism-1]->GetBin(1, i));
        if ( num06 >= m_min_bin ) {
          mean06 = i06_[ism-1]->GetBinContent(i06_[ism-1]->GetBin(1, i));
          rms06  = i06_[ism-1]->GetBinError(i06_[ism-1]->GetBin(1, i));
          update_channel2 = true;
        }
      }

      if ( i07_[ism-1] && i07_[ism-1]->GetEntries() >= m_min_tot ) {
        num07 = i07_[ism-1]->GetBinEntries(i07_[ism-1]->GetBin(1, i));
        if ( num07 >= m_min_bin ) {
          mean07 = i07_[ism-1]->GetBinContent(i07_[ism-1]->GetBin(1, i));
          rms07  = i07_[ism-1]->GetBinError(i07_[ism-1]->GetBin(1, i));
          update_channel3 = true;
        }
      }

      if ( i08_[ism-1] && i08_[ism-1]->GetEntries() >= m_min_tot ) {
        num08 = i08_[ism-1]->GetBinEntries(i08_[ism-1]->GetBin(1, i));
        if ( num08 >= m_min_bin ) {
          mean08 = i08_[ism-1]->GetBinContent(i08_[ism-1]->GetBin(1, i));
          rms08  = i08_[ism-1]->GetBinError(i08_[ism-1]->GetBin(1, i));
          update_channel4 = true;
        }
      }

      if ( j01_[ism-1] && j01_[ism-1]->GetEntries() >= m_min_tot ) {
        num09 = j01_[ism-1]->GetBinEntries(j01_[ism-1]->GetBin(1, i));
        if ( num09 >= m_min_bin ) {
          mean09 = j01_[ism-1]->GetBinContent(j01_[ism-1]->GetBin(1, i));
          rms09  = j01_[ism-1]->GetBinError(j01_[ism-1]->GetBin(1, i));
          update_channel1 = true;
        }
      }

      if ( j02_[ism-1] && j02_[ism-1]->GetEntries() >= m_min_tot ) {
        num10 = j02_[ism-1]->GetBinEntries(j02_[ism-1]->GetBin(1, i));
        if ( num10 >= m_min_bin ) {
          mean10 = j02_[ism-1]->GetBinContent(j02_[ism-1]->GetBin(1, i));
          rms10  = j02_[ism-1]->GetBinError(j02_[ism-1]->GetBin(1, i));
          update_channel2 = true;
        }
      }

      if ( j03_[ism-1] && j03_[ism-1]->GetEntries() >= m_min_tot ) {
        num11 = j03_[ism-1]->GetBinEntries(j03_[ism-1]->GetBin(i));
        if ( num11 >= m_min_bin ) {
          mean11 = j03_[ism-1]->GetBinContent(j03_[ism-1]->GetBin(1, i));
          rms11  = j03_[ism-1]->GetBinError(j03_[ism-1]->GetBin(1, i));
          update_channel3 = true;
        }
      }

      if ( j04_[ism-1] && j04_[ism-1]->GetEntries() >= m_min_tot ) {
        num12 = j04_[ism-1]->GetBinEntries(j04_[ism-1]->GetBin(1, i));
        if ( num12 >= m_min_bin ) {
          mean12 = j04_[ism-1]->GetBinContent(j04_[ism-1]->GetBin(1, i));
          rms12  = j04_[ism-1]->GetBinError(j04_[ism-1]->GetBin(1, i));
          update_channel4 = true;
        }
      }

      if ( j05_[ism-1] && j05_[ism-1]->GetEntries() >= m_min_tot ) {
        num13 = j05_[ism-1]->GetBinEntries(j05_[ism-1]->GetBin(1, i));
        if ( num13 >= m_min_bin ) {
          mean13 = j05_[ism-1]->GetBinContent(j05_[ism-1]->GetBin(1, i));
          rms13  = j05_[ism-1]->GetBinError(j05_[ism-1]->GetBin(1, i));
          update_channel1 = true;
        }
      }

      if ( j06_[ism-1] && j06_[ism-1]->GetEntries() >= m_min_tot ) {
        num14 = j06_[ism-1]->GetBinEntries(j06_[ism-1]->GetBin(1, i));
        if ( num14 >= m_min_bin ) {
          mean14 = j06_[ism-1]->GetBinContent(j06_[ism-1]->GetBin(1, i));
          rms14  = j06_[ism-1]->GetBinError(j06_[ism-1]->GetBin(1, i));
          update_channel2 = true;
        }
      }

      if ( j07_[ism-1] && j07_[ism-1]->GetEntries() >= m_min_tot ) {
      num15 = j07_[ism-1]->GetBinEntries(j07_[ism-1]->GetBin(1, i));
        if ( num15 >= m_min_bin ) {
          mean15 = j07_[ism-1]->GetBinContent(j07_[ism-1]->GetBin(1, i));
          rms15  = j07_[ism-1]->GetBinError(j07_[ism-1]->GetBin(1, i));
          update_channel3 = true;
        }
      }

      if ( j08_[ism-1] && j08_[ism-1]->GetEntries() >= m_min_tot ) {
        num16 = j08_[ism-1]->GetBinEntries(j08_[ism-1]->GetBin(1, i));
        if ( num16 >= m_min_bin ) {
          mean16 = j08_[ism-1]->GetBinContent(j08_[ism-1]->GetBin(1, i));
          rms16  = j08_[ism-1]->GetBinError(j08_[ism-1]->GetBin(1, i));
          update_channel4 = true;
        }
      }

      if ( update_channel1 ) {

        float val;

        val = 1.;
        if ( mean01 < amplitudeThresholdPN_ )
          val = 0.;
        if ( mean05 < meanThresholdPN_ )
          val = 0.;
        if ( meg05_[ism-1] ) meg05_[ism-1]->setBinContent(i, 1, val);

        val = 1.;
        if ( mean09 < amplitudeThresholdPN_ )
          val = 0.;
        if ( mean13 < meanThresholdPN_ )
          val = 0.;
        if ( meg05_[ism-1] ) meg05_[ism-1]->setBinContent(i, 1, val); 

      }

      if ( update_channel2 ) {

        float val; 

        val = 1.;
        if ( mean02 < amplitudeThresholdPN_ )
          val = 0.;
        if ( mean06 < meanThresholdPN_ )
          val = 0.;
        if ( meg06_[ism-1] ) meg06_[ism-1]->setBinContent(i, 1, val); 

        val = 1.;
        if ( mean10 < amplitudeThresholdPN_ )
          val = 0.;
        if ( mean14 < meanThresholdPN_ )
          val = 0.;
        if ( meg06_[ism-1] ) meg06_[ism-1]->setBinContent(i, 1, val); 

      }

      if ( update_channel3 ) {

        float val;

        val = 1.;
        if ( mean03 < amplitudeThresholdPN_ )
          val = 0.;
        if ( mean07 < meanThresholdPN_ )
          val = 0.;
        if ( meg07_[ism-1] ) meg07_[ism-1]->setBinContent(i, 1, val);

        val = 1.;
        if ( mean11 < amplitudeThresholdPN_ )
          val = 0.;
        if ( mean15 < meanThresholdPN_ )
          val = 0.;
        if ( meg07_[ism-1] ) meg07_[ism-1]->setBinContent(i, 1, val);

      }

      if ( update_channel4 ) {

        float val;

        val = 1.;
        if ( mean04 < amplitudeThresholdPN_ )
          val = 0.;
        if ( mean08 < meanThresholdPN_ )
          val = 0.;
        if ( meg08_[ism-1] ) meg08_[ism-1]->setBinContent(i, 1, val);

        val = 1.;
        if ( mean12 < amplitudeThresholdPN_ )
          val = 0.;
        if ( mean16 < meanThresholdPN_ )
          val = 0.;
        if ( meg08_[ism-1] ) meg08_[ism-1]->setBinContent(i, 1, val);

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

    if ( qth09_[ism-1] ) badChannels09 = qth09_[ism-1]->getBadChannels();
    if ( qth10_[ism-1] ) badChannels10 = qth10_[ism-1]->getBadChannels();
    if ( qth11_[ism-1] ) badChannels11 = qth11_[ism-1]->getBadChannels();
    if ( qth12_[ism-1] ) badChannels12 = qth12_[ism-1]->getBadChannels();
    if ( qth13_[ism-1] ) badChannels13 = qth13_[ism-1]->getBadChannels();
    if ( qth14_[ism-1] ) badChannels14 = qth14_[ism-1]->getBadChannels();
    if ( qth15_[ism-1] ) badChannels15 = qth15_[ism-1]->getBadChannels();
    if ( qth16_[ism-1] ) badChannels16 = qth16_[ism-1]->getBadChannels();

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

  const double histMax = 1.e15;

  int pCol3[6] = { 2, 3, 5, 1, 1, 1 };

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

  string imgNameQual[8], imgNameAmp[8], imgNameTim[8], imgNameShape[8], imgNameAmpoPN[8], imgNameMEPnQual[8], imgNameMEPnG01[8], imgNameMEPnPedG01[8], imgNameMEPnG16[8], imgNameMEPnPedG16[8], imgName, meName;

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
        obj2f->SetMaximum(5.0);
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
        gStyle->SetOptStat("euomr");
        obj1f->SetStats(kTRUE);
        if ( obj1f->GetMaximum(histMax) > 0. ) {
          gPad->SetLogy(1);
        } else {
          gPad->SetLogy(0);
        }
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
        gStyle->SetOptStat("euomr");
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
        gStyle->SetOptStat("euomr");
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
        gStyle->SetOptStat("euomr");
        obj1f->SetStats(kTRUE);
//        if ( obj1f->GetMaximum(histMax) > 0. ) {
//          gPad->SetLogy(1);
//        } else {
//          gPad->SetLogy(0);
//        }
        obj1f->Draw();
        cAmpoPN->Update();
        cAmpoPN->SaveAs(imgName.c_str());
        gPad->SetLogy(0);

      }

      // Monitoring elements plots

      imgNameMEPnQual[iCanvas-1] = "";

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
        imgNameMEPnQual[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgNameMEPnQual[iCanvas-1];

        cQual->cd();
        gStyle->SetOptStat(" ");
        gStyle->SetPalette(6, pCol3);
        obj2f->GetXaxis()->SetNdivisions(10);
        obj2f->GetYaxis()->SetNdivisions(5);
        cQual->SetGridx();
        cQual->SetGridy(0);
        obj2f->SetMinimum(-0.00000001);
        obj2f->SetMaximum(5.0);
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
        gStyle->SetOptStat("euomr");
        obj1d->SetStats(kTRUE);
//        if ( obj1d->GetMaximum(histMax) > 0. ) {
//          gPad->SetLogy(1);
//        } else {
//          gPad->SetLogy(0);
//        }
        obj1d->SetMinimum(0.);
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
          if ( j01_[ism-1] ) obj1d = j01_[ism-1]->ProjectionY("_py", 1, 1, "e");
          break;
        case 2:
          if ( j02_[ism-1] ) obj1d = j02_[ism-1]->ProjectionY("_py", 1, 1, "e");
          break;
        case 3:
          if ( j03_[ism-1] ) obj1d = j03_[ism-1]->ProjectionY("_py", 1, 1, "e");
          break;
        case 4:
          if ( j04_[ism-1] ) obj1d = j04_[ism-1]->ProjectionY("_py", 1, 1, "e");
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
        gStyle->SetOptStat("euomr");
        obj1d->SetStats(kTRUE);
//        if ( obj1d->GetMaximum(histMax) > 0. ) {
//          gPad->SetLogy(1);
//        } else {
//          gPad->SetLogy(0);
//        }
        obj1d->SetMinimum(0.);
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
        gStyle->SetOptStat("euomr");
        obj1d->SetStats(kTRUE);
//        if ( obj1d->GetMaximum(histMax) > 0. ) {
//          gPad->SetLogy(1);
//        } else {
//          gPad->SetLogy(0);
//        }
        obj1d->SetMinimum(0.);
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
          if ( j05_[ism-1] ) obj1d = j05_[ism-1]->ProjectionY("_py", 1, 1, "e");
          break;
        case 2:
          if ( j06_[ism-1] ) obj1d = j06_[ism-1]->ProjectionY("_py", 1, 1, "e");
          break;
        case 3:
          if ( j07_[ism-1] ) obj1d = j07_[ism-1]->ProjectionY("_py", 1, 1, "e");
          break;
        case 4:
          if ( j08_[ism-1] ) obj1d = j08_[ism-1]->ProjectionY("_py", 1, 1, "e");
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
        gStyle->SetOptStat("euomr");
        obj1d->SetStats(kTRUE);
//        if ( obj1d->GetMaximum(histMax) > 0. ) {
//          gPad->SetLogy(1);
//        } else {
//          gPad->SetLogy(0);
//        }
        obj1d->SetMinimum(0.);
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

      if ( imgNameMEPnQual[iCanvas-1].size() != 0 )
        htmlFile << "<td colspan=\"2\"><img src=\"" << imgNameMEPnQual[iCanvas-1] << "\"></td>" << endl;
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

