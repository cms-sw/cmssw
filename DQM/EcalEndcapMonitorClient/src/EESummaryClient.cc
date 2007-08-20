/*
 * \file EESummaryClient.cc
 *
 * $Date: 2007/08/20 21:23:28 $
 * $Revision: 1.22 $
 * \author G. Della Ricca
 *
*/

#include <memory>
#include <iostream>
#include <iomanip>
#include <map>

#include "TStyle.h"
#include "TGraph.h"
#include "TLine.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/UI/interface/MonitorUIRoot.h"
#include "DQMServices/Core/interface/QTestStatus.h"
#include "DQMServices/QualityTests/interface/QCriterionRoot.h"

#include "OnlineDB/EcalCondDB/interface/RunTag.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"

#include <DQM/EcalCommon/interface/UtilsClient.h>
#include <DQM/EcalCommon/interface/Numbers.h>

#include <DQM/EcalEndcapMonitorClient/interface/EECosmicClient.h>
#include <DQM/EcalEndcapMonitorClient/interface/EEIntegrityClient.h>
#include <DQM/EcalEndcapMonitorClient/interface/EELaserClient.h>
#include <DQM/EcalEndcapMonitorClient/interface/EELedClient.h>
#include <DQM/EcalEndcapMonitorClient/interface/EEPedestalClient.h>
#include <DQM/EcalEndcapMonitorClient/interface/EEPedestalOnlineClient.h>
#include <DQM/EcalEndcapMonitorClient/interface/EETestPulseClient.h>
#include <DQM/EcalEndcapMonitorClient/interface/EEBeamCaloClient.h>
#include <DQM/EcalEndcapMonitorClient/interface/EEBeamHodoClient.h>
#include <DQM/EcalEndcapMonitorClient/interface/EETriggerTowerClient.h>
#include <DQM/EcalEndcapMonitorClient/interface/EEClusterClient.h>
#include <DQM/EcalEndcapMonitorClient/interface/EETimingClient.h>

#include <DQM/EcalEndcapMonitorClient/interface/EESummaryClient.h>

using namespace cms;
using namespace edm;
using namespace std;

EESummaryClient::EESummaryClient(const ParameterSet& ps){

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

  // vector of selected Super Modules (Defaults to all 18).
  superModules_.reserve(18);
  for ( unsigned int i = 1; i < 19; i++ ) superModules_.push_back(i);
  superModules_ = ps.getUntrackedParameter<vector<int> >("superModules", superModules_);

  meIntegrity_[0]      = 0;
  meIntegrity_[1]      = 0;
  meOccupancy_[0]      = 0;
  meOccupancy_[1]      = 0;
  mePedestalOnline_[0] = 0;
  mePedestalOnline_[1] = 0;
  meLaserL1_[0]        = 0;
  meLaserL1_[1]        = 0;
  meLaserL1PN_[0]      = 0;
  meLaserL1PN_[1]      = 0;
  meLed_[0]            = 0;
  meLed_[1]            = 0;
  meLedPN_[0]          = 0;
  meLedPN_[1]          = 0;
  mePedestal_[0]       = 0;
  mePedestal_[1]       = 0;
  mePedestalPN_[0]     = 0;
  mePedestalPN_[1]     = 0;
  meTestPulse_[0]      = 0;
  meTestPulse_[1]      = 0;
  meTestPulsePN_[0]    = 0;
  meTestPulsePN_[1]    = 0;
  meGlobalSummary_[0]  = 0;
  meGlobalSummary_[1]  = 0;

  qtg01_[0] = 0;
  qtg01_[1] = 0;
  qtg02_[0] = 0;
  qtg02_[1] = 0;
  qtg03_[0] = 0;
  qtg03_[1] = 0;
  qtg04_[0] = 0;
  qtg04_[1] = 0;
  qtg04PN_[0] = 0;
  qtg04PN_[1] = 0;
  qtg05_[0] = 0;
  qtg05_[1] = 0;
  qtg05PN_[0] = 0;
  qtg05PN_[1] = 0;
  qtg06_[0] = 0;
  qtg06_[1] = 0;
  qtg06PN_[0] = 0;
  qtg06PN_[1] = 0;
  qtg07_[0] = 0;
  qtg07_[1] = 0;
  qtg07PN_[0] = 0;
  qtg07PN_[1] = 0;
  qtg08_[0] = 0;
  qtg08_[1] = 0;

}

EESummaryClient::~EESummaryClient(){

}

void EESummaryClient::beginJob(MonitorUserInterface* mui){

  mui_ = mui;
  dbe_ = mui->getBEInterface();

  if ( verbose_ ) cout << "EESummaryClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

  if ( enableQT_ ) {

    Char_t qtname[200];

    sprintf(qtname, "EEIT EE - summary quality test");
    qtg01_[0] = dynamic_cast<MEContentsTH2FWithinRangeROOT*> (dbe_->createQTest(ContentsTH2FWithinRangeROOT::getAlgoName(), qtname));

    sprintf(qtname, "EEIT EE + summary quality test");
    qtg01_[1] = dynamic_cast<MEContentsTH2FWithinRangeROOT*> (dbe_->createQTest(ContentsTH2FWithinRangeROOT::getAlgoName(), qtname));

    sprintf(qtname, "EEOT EE - summary quality test");
    qtg02_[0] = dynamic_cast<MEContentsTH2FWithinRangeROOT*> (dbe_->createQTest(ContentsTH2FWithinRangeROOT::getAlgoName(), qtname));

    sprintf(qtname, "EEOT EE + summary quality test");
    qtg02_[1] = dynamic_cast<MEContentsTH2FWithinRangeROOT*> (dbe_->createQTest(ContentsTH2FWithinRangeROOT::getAlgoName(), qtname));

    sprintf(qtname, "EEPOT EE - summary quality test");
    qtg03_[0] = dynamic_cast<MEContentsTH2FWithinRangeROOT*> (dbe_->createQTest(ContentsTH2FWithinRangeROOT::getAlgoName(), qtname));

    sprintf(qtname, "EEPOT EE + summary quality test");
    qtg03_[1] = dynamic_cast<MEContentsTH2FWithinRangeROOT*> (dbe_->createQTest(ContentsTH2FWithinRangeROOT::getAlgoName(), qtname));

    sprintf(qtname, "EELT EE - summary quality test L1");
    qtg04_[0] = dynamic_cast<MEContentsTH2FWithinRangeROOT*> (dbe_->createQTest(ContentsTH2FWithinRangeROOT::getAlgoName(), qtname));

    sprintf(qtname, "EELT EE - PN summary quality test L1");
    qtg04PN_[0] = dynamic_cast<MEContentsTH2FWithinRangeROOT*> (dbe_->createQTest(ContentsTH2FWithinRangeROOT::getAlgoName(), qtname));

    sprintf(qtname, "EELT EE + summary quality test L1");
    qtg04_[1] = dynamic_cast<MEContentsTH2FWithinRangeROOT*> (dbe_->createQTest(ContentsTH2FWithinRangeROOT::getAlgoName(), qtname));

    sprintf(qtname, "EELT EE + PN summary quality test L1");
    qtg04PN_[1] = dynamic_cast<MEContentsTH2FWithinRangeROOT*> (dbe_->createQTest(ContentsTH2FWithinRangeROOT::getAlgoName(), qtname));

    sprintf(qtname, "EELDT EE - summary quality test");
    qtg05_[0] = dynamic_cast<MEContentsTH2FWithinRangeROOT*> (dbe_->createQTest(ContentsTH2FWithinRangeROOT::getAlgoName(), qtname));

    sprintf(qtname, "EELDT EE - PN summary quality test");
    qtg05PN_[0] = dynamic_cast<MEContentsTH2FWithinRangeROOT*> (dbe_->createQTest(ContentsTH2FWithinRangeROOT::getAlgoName(), qtname));

    sprintf(qtname, "EELDT EE + summary quality test");
    qtg05_[1] = dynamic_cast<MEContentsTH2FWithinRangeROOT*> (dbe_->createQTest(ContentsTH2FWithinRangeROOT::getAlgoName(), qtname));

    sprintf(qtname, "EELDT EE + PN summary quality test");
    qtg05PN_[1] = dynamic_cast<MEContentsTH2FWithinRangeROOT*> (dbe_->createQTest(ContentsTH2FWithinRangeROOT::getAlgoName(), qtname));

    sprintf(qtname, "EEPT EE - summary quality test");
    qtg06_[0] = dynamic_cast<MEContentsTH2FWithinRangeROOT*> (dbe_->createQTest(ContentsTH2FWithinRangeROOT::getAlgoName(), qtname));

    sprintf(qtname, "EEPT EE - PN summary quality test");
    qtg06PN_[0] = dynamic_cast<MEContentsTH2FWithinRangeROOT*> (dbe_->createQTest(ContentsTH2FWithinRangeROOT::getAlgoName(), qtname));

    sprintf(qtname, "EEPT EE + summary quality test");
    qtg06_[1] = dynamic_cast<MEContentsTH2FWithinRangeROOT*> (dbe_->createQTest(ContentsTH2FWithinRangeROOT::getAlgoName(), qtname));

    sprintf(qtname, "EEPT EE + PN summary quality test");
    qtg06PN_[1] = dynamic_cast<MEContentsTH2FWithinRangeROOT*> (dbe_->createQTest(ContentsTH2FWithinRangeROOT::getAlgoName(), qtname));

    sprintf(qtname, "EETPT EE - summary quality test");
    qtg07_[0] = dynamic_cast<MEContentsTH2FWithinRangeROOT*> (dbe_->createQTest(ContentsTH2FWithinRangeROOT::getAlgoName(), qtname));

    sprintf(qtname, "EETPT EE - PN summary quality test");
    qtg07PN_[0] = dynamic_cast<MEContentsTH2FWithinRangeROOT*> (dbe_->createQTest(ContentsTH2FWithinRangeROOT::getAlgoName(), qtname));

    sprintf(qtname, "EETPT EE + summary quality test");
    qtg07_[1] = dynamic_cast<MEContentsTH2FWithinRangeROOT*> (dbe_->createQTest(ContentsTH2FWithinRangeROOT::getAlgoName(), qtname));

    sprintf(qtname, "EETPT EE + PN summary quality test");
    qtg07PN_[1] = dynamic_cast<MEContentsTH2FWithinRangeROOT*> (dbe_->createQTest(ContentsTH2FWithinRangeROOT::getAlgoName(), qtname));

    sprintf(qtname, "EE global summary quality test EE -");
    qtg08_[0] = dynamic_cast<MEContentsTH2FWithinRangeROOT*> (dbe_->createQTest(ContentsTH2FWithinRangeROOT::getAlgoName(), qtname));

    sprintf(qtname, "EE global summary quality test EE +");
    qtg08_[1] = dynamic_cast<MEContentsTH2FWithinRangeROOT*> (dbe_->createQTest(ContentsTH2FWithinRangeROOT::getAlgoName(), qtname));

    qtg01_[0]->setMeanRange(1., 6.);
    qtg01_[1]->setMeanRange(1., 6.);
    qtg02_[0]->setMeanRange(1., 6.);
    qtg02_[1]->setMeanRange(1., 6.);
    qtg03_[0]->setMeanRange(1., 6.);
    qtg03_[1]->setMeanRange(1., 6.);
    qtg04_[0]->setMeanRange(1., 6.);
    qtg04PN_[0]->setMeanRange(1., 6.);
    qtg04_[1]->setMeanRange(1., 6.);
    qtg04PN_[1]->setMeanRange(1., 6.);
    qtg05_[0]->setMeanRange(1., 6.);
    qtg05PN_[0]->setMeanRange(1., 6.);
    qtg05_[1]->setMeanRange(1., 6.);
    qtg05PN_[1]->setMeanRange(1., 6.);
    qtg06_[0]->setMeanRange(1., 6.);
    qtg06PN_[0]->setMeanRange(1., 6.);
    qtg06_[1]->setMeanRange(1., 6.);
    qtg06PN_[1]->setMeanRange(1., 6.);
    qtg07_[0]->setMeanRange(1., 6.);
    qtg07PN_[0]->setMeanRange(1., 6.);
    qtg07_[1]->setMeanRange(1., 6.);
    qtg07PN_[1]->setMeanRange(1., 6.);
    qtg08_[0]->setMeanRange(1., 6.);
    qtg08_[1]->setMeanRange(1., 6.);

    qtg01_[0]->setErrorProb(1.00);
    qtg01_[1]->setErrorProb(1.00);
    qtg02_[0]->setErrorProb(1.00);
    qtg02_[1]->setErrorProb(1.00);
    qtg03_[0]->setErrorProb(1.00);
    qtg03_[1]->setErrorProb(1.00);
    qtg04_[0]->setErrorProb(1.00);
    qtg04PN_[0]->setErrorProb(1.00);
    qtg04_[1]->setErrorProb(1.00);
    qtg04PN_[1]->setErrorProb(1.00);
    qtg05_[0]->setErrorProb(1.00);
    qtg05PN_[0]->setErrorProb(1.00);
    qtg05_[1]->setErrorProb(1.00);
    qtg05PN_[1]->setErrorProb(1.00);
    qtg06_[0]->setErrorProb(1.00);
    qtg06PN_[0]->setErrorProb(1.00);
    qtg06_[1]->setErrorProb(1.00);
    qtg06PN_[1]->setErrorProb(1.00);
    qtg07_[0]->setErrorProb(1.00);
    qtg07PN_[0]->setErrorProb(1.00);
    qtg07_[1]->setErrorProb(1.00);
    qtg07PN_[1]->setErrorProb(1.00);
    qtg08_[0]->setErrorProb(1.00);
    qtg08_[1]->setErrorProb(1.00);

  }

}

void EESummaryClient::beginRun(void){

  if ( verbose_ ) cout << "EESummaryClient: beginRun" << endl;

  jevt_ = 0;

  this->setup();

  this->subscribe();

}

void EESummaryClient::endJob(void) {

  if ( verbose_ ) cout << "EESummaryClient: endJob, ievt = " << ievt_ << endl;

  this->unsubscribe();

  this->cleanup();

}

void EESummaryClient::endRun(void) {

  if ( verbose_ ) cout << "EESummaryClient: endRun, jevt = " << jevt_ << endl;

  this->unsubscribe();

  this->cleanup();

}

void EESummaryClient::setup(void) {

  Char_t histo[200];

  dbe_->setCurrentFolder( "EcalEndcap/EESummaryClient" );

  if ( meIntegrity_[0] ) dbe_->removeElement( meIntegrity_[0]->getName() );
  sprintf(histo, "EEIT EE - integrity quality summary");
  meIntegrity_[0] = dbe_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);

  if ( meIntegrity_[1] ) dbe_->removeElement( meIntegrity_[0]->getName() );
  sprintf(histo, "EEIT EE + integrity quality summary");
  meIntegrity_[1] = dbe_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);

  if ( meOccupancy_[0] ) dbe_->removeElement( meOccupancy_[0]->getName() );
  sprintf(histo, "EEOT EE - occupancy summary");
  meOccupancy_[0] = dbe_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);

  if ( meOccupancy_[1] ) dbe_->removeElement( meOccupancy_[1]->getName() );
  sprintf(histo, "EEOT EE + occupancy summary");
  meOccupancy_[1] = dbe_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);

  if ( mePedestalOnline_[0] ) dbe_->removeElement( mePedestalOnline_[0]->getName() );
  sprintf(histo, "EEPOT EE - pedestal quality summary G12");
  mePedestalOnline_[0] = dbe_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);

  if ( mePedestalOnline_[1] ) dbe_->removeElement( mePedestalOnline_[1]->getName() );
  sprintf(histo, "EEPOT EE + pedestal quality summary G12");
  mePedestalOnline_[1] = dbe_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);

  if ( meLaserL1_[0] ) dbe_->removeElement( meLaserL1_[0]->getName() );
  sprintf(histo, "EELT EE - laser quality summary L1");
  meLaserL1_[0] = dbe_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);

  if ( meLaserL1PN_[0] ) dbe_->removeElement( meLaserL1PN_[0]->getName() );
  sprintf(histo, "EELT EE - PN laser quality summary L1");
  meLaserL1PN_[0] = dbe_->book2D(histo, histo, 90, 0., 90., 20, -10., 10.);

  if ( meLaserL1_[1] ) dbe_->removeElement( meLaserL1_[1]->getName() );
  sprintf(histo, "EELT EE + laser quality summary L1");
  meLaserL1_[1] = dbe_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
  
  if ( meLaserL1PN_[1] ) dbe_->removeElement( meLaserL1PN_[1]->getName() );
  sprintf(histo, "EELT EE + PN laser quality summary L1");
  meLaserL1PN_[1] = dbe_->book2D(histo, histo, 90, 0., 90., 20, -10., 10.);

  if ( meLed_[0] ) dbe_->removeElement( meLed_[0]->getName() );
  sprintf(histo, "EELDT EE - led quality summary");
  meLed_[0] = dbe_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
  
  if ( meLedPN_[0] ) dbe_->removeElement( meLedPN_[0]->getName() );
  sprintf(histo, "EELDT EE - PN led quality summary");
  meLedPN_[0] = dbe_->book2D(histo, histo, 90, 0., 90., 20, -10., 10.);
  
  if ( meLed_[1] ) dbe_->removeElement( meLed_[1]->getName() );
  sprintf(histo, "EELDT EE + led quality summary");
  meLed_[1] = dbe_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
  
  if ( meLedPN_[1] ) dbe_->removeElement( meLedPN_[1]->getName() );
  sprintf(histo, "EELDT EE + PN led quality summary");
  meLedPN_[1] = dbe_->book2D(histo, histo, 90, 0., 90., 20, -10., 10.);

  if( mePedestal_[0] ) dbe_->removeElement( mePedestal_[0]->getName() );
  sprintf(histo, "EEPT EE - pedestal quality summary");
  mePedestal_[0] = dbe_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
  
  if( mePedestalPN_[0] ) dbe_->removeElement( mePedestalPN_[0]->getName() );
  sprintf(histo, "EEPT EE - PN pedestal quality summary");
  mePedestalPN_[0] = dbe_->book2D(histo, histo, 90, 0., 90., 20, -10, 10.);

  if( mePedestal_[1] ) dbe_->removeElement( mePedestal_[1]->getName() );
  sprintf(histo, "EEPT EE + pedestal quality summary");
  mePedestal_[1] = dbe_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
  
  if( mePedestalPN_[1] ) dbe_->removeElement( mePedestalPN_[1]->getName() );
  sprintf(histo, "EEPT EE + PN pedestal quality summary");
  mePedestalPN_[1] = dbe_->book2D(histo, histo, 90, 0., 90., 20, -10, 10.);

  if( meTestPulse_[0] ) dbe_->removeElement( meTestPulse_[0]->getName() );
  sprintf(histo, "EETPT EE - test pulse quality summary");
  meTestPulse_[0] = dbe_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);

  if( meTestPulsePN_[0] ) dbe_->removeElement( meTestPulsePN_[0]->getName() );
  sprintf(histo, "EETPT EE - PN test pulse quality summary");
  meTestPulsePN_[0] = dbe_->book2D(histo, histo, 90, 0., 90., 20, -10., 10.);

  if( meTestPulse_[1] ) dbe_->removeElement( meTestPulse_[1]->getName() );
  sprintf(histo, "EETPT EE + test pulse quality summary");
  meTestPulse_[1] = dbe_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);

  if( meTestPulsePN_[1] ) dbe_->removeElement( meTestPulsePN_[1]->getName() );
  sprintf(histo, "EETPT EE + PN test pulse quality summary");
  meTestPulsePN_[1] = dbe_->book2D(histo, histo, 90, 0., 90., 20, -10., 10.);

  if( meGlobalSummary_[0] ) dbe_->removeElement( meGlobalSummary_[0]->getName() );
  sprintf(histo, "EE global summary EE -");
  meGlobalSummary_[0] = dbe_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);

  if( meGlobalSummary_[1] ) dbe_->removeElement( meGlobalSummary_[1]->getName() );
  sprintf(histo, "EE global summary EE +");
  meGlobalSummary_[1] = dbe_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);

}

void EESummaryClient::cleanup(void) {

  dbe_->setCurrentFolder( "EcalEndcap/EESummaryClient" );

  if ( meIntegrity_[0] ) dbe_->removeElement( meIntegrity_[0]->getName() );
  meIntegrity_[0] = 0;

  if ( meIntegrity_[1] ) dbe_->removeElement( meIntegrity_[1]->getName() );
  meIntegrity_[1] = 0;

  if ( meOccupancy_[0] ) dbe_->removeElement( meOccupancy_[0]->getName() );
  meOccupancy_[0] = 0;

  if ( meOccupancy_[1] ) dbe_->removeElement( meOccupancy_[1]->getName() );
  meOccupancy_[1] = 0;

  if ( mePedestalOnline_[0] ) dbe_->removeElement( mePedestalOnline_[0]->getName() );
  mePedestalOnline_[0] = 0;

  if ( mePedestalOnline_[1] ) dbe_->removeElement( mePedestalOnline_[1]->getName() );
  mePedestalOnline_[1] = 0;

  if ( meLaserL1_[0] ) dbe_->removeElement( meLaserL1_[0]->getName() );
  meLaserL1_[0] = 0;

  if ( meLaserL1_[1] ) dbe_->removeElement( meLaserL1_[1]->getName() );
  meLaserL1_[1] = 0;

  if ( meLaserL1PN_[0] ) dbe_->removeElement( meLaserL1PN_[0]->getName() );
  meLaserL1PN_[0] = 0;

  if ( meLaserL1PN_[1] ) dbe_->removeElement( meLaserL1PN_[1]->getName() );
  meLaserL1PN_[1] = 0;

  if ( meLed_[0] ) dbe_->removeElement( meLed_[0]->getName() );
  meLed_[0] = 0;
 
  if ( meLed_[1] ) dbe_->removeElement( meLed_[1]->getName() );
  meLed_[1] = 0;
 
  if ( meLedPN_[0] ) dbe_->removeElement( meLedPN_[0]->getName() );
  meLedPN_[0] = 0;

  if ( meLedPN_[1] ) dbe_->removeElement( meLedPN_[1]->getName() );
  meLedPN_[1] = 0;

  if ( mePedestal_[0] ) dbe_->removeElement( mePedestal_[0]->getName() );
  mePedestal_[0] = 0;

  if ( mePedestal_[1] ) dbe_->removeElement( mePedestal_[1]->getName() );
  mePedestal_[1] = 0;

  if ( mePedestalPN_[0] ) dbe_->removeElement( mePedestalPN_[0]->getName() );
  mePedestalPN_[0] = 0;

  if ( mePedestalPN_[1] ) dbe_->removeElement( mePedestalPN_[1]->getName() );
  mePedestalPN_[1] = 0;

  if ( meTestPulse_[0] ) dbe_->removeElement( meTestPulse_[0]->getName() );
  meTestPulse_[0] = 0;

  if ( meTestPulse_[1] ) dbe_->removeElement( meTestPulse_[1]->getName() );
  meTestPulse_[1] = 0;

  if ( meTestPulsePN_[0] ) dbe_->removeElement( meTestPulsePN_[0]->getName() );
  meTestPulsePN_[0] = 0;

  if ( meTestPulsePN_[1] ) dbe_->removeElement( meTestPulsePN_[1]->getName() );
  meTestPulsePN_[1] = 0;

  if ( meGlobalSummary_[0] ) dbe_->removeElement( meGlobalSummary_[0]->getName() );
  meGlobalSummary_[0] = 0;

  if ( meGlobalSummary_[1] ) dbe_->removeElement( meGlobalSummary_[1]->getName() );
  meGlobalSummary_[1] = 0;

}

bool EESummaryClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov) {

  bool status = true;

//  UtilsClient::printBadChannels(qtg01_[0]);
//  UtilsClient::printBadChannels(qtg01_[1]);
//  UtilsClient::printBadChannels(qtg02_[0]);
//  UtilsClient::printBadChannels(qtg02_[1]);
//  UtilsClient::printBadChannels(qtg03_[0]);
//  UtilsClient::printBadChannels(qtg03_[1]);
//  UtilsClient::printBadChannels(qtg04_[0]);
//  UtilsClient::printBadChannels(qtg04_[1]);

  return status;

}

void EESummaryClient::subscribe(void){

  if ( verbose_ ) cout << "EESummaryClient: subscribe" << endl;

  Char_t histo[200];

  sprintf(histo, "EcalEndcap/EESummaryClient/EEIT EE - integrity quality summary");
  if ( qtg01_[0] ) mui_->useQTest(histo, qtg01_[0]->getName());
  sprintf(histo, "EcalEndcap/EESummaryClient/EEIT EE + integrity quality summary");
  if ( qtg01_[1] ) mui_->useQTest(histo, qtg01_[1]->getName());
  sprintf(histo, "EcalEndcap/EESummaryClient/EEOT EE - occupancy summary");
  if ( qtg02_[0] ) mui_->useQTest(histo, qtg02_[0]->getName());
  sprintf(histo, "EcalEndcap/EESummaryClient/EEOT EE + occupancy summary");
  if ( qtg02_[1] ) mui_->useQTest(histo, qtg02_[1]->getName());
  sprintf(histo, "EcalEndcap/EESummaryClient/EEPOT EE - pedestal quality summary G12");
  if ( qtg03_[0] ) mui_->useQTest(histo, qtg03_[0]->getName());
  sprintf(histo, "EcalEndcap/EESummaryClient/EEPOT EE + pedestal quality summary G12");
  if ( qtg03_[1] ) mui_->useQTest(histo, qtg03_[1]->getName());
  sprintf(histo, "EcalBarrel/EESummaryClient/EELT EE - laser quality summary L1");
  if ( qtg04_[0] ) mui_->useQTest(histo, qtg04_[0]->getName());
  sprintf(histo, "EcalBarrel/EESummaryClient/EELT EE + laser quality summary L1");
  if ( qtg04_[1] ) mui_->useQTest(histo, qtg04_[1]->getName());
  sprintf(histo, "EcalBarrel/EESummaryClient/EELT EE - PN laser quality summary L1");
  if ( qtg04PN_[0] ) mui_->useQTest(histo, qtg04PN_[0]->getName());
  sprintf(histo, "EcalBarrel/EESummaryClient/EELT EE + PN laser quality summary L1");
  if ( qtg04PN_[1] ) mui_->useQTest(histo, qtg04PN_[1]->getName());
  sprintf(histo, "EcalBarrel/EESummaryClient/EELDT EE - led quality summary");
  if ( qtg05_[0] ) mui_->useQTest(histo, qtg05_[1]->getName());
  sprintf(histo, "EcalBarrel/EESummaryClient/EELDT EE + led quality summary");
  if ( qtg05_[1] ) mui_->useQTest(histo, qtg05_[1]->getName());
  sprintf(histo, "EcalBarrel/EESummaryClient/EELDT EE - PN led quality summary");
  if ( qtg05PN_[0] ) mui_->useQTest(histo, qtg05PN_[0]->getName());
  sprintf(histo, "EcalBarrel/EESummaryClient/EELDT EE + PN led quality summary");
  if ( qtg05PN_[1] ) mui_->useQTest(histo, qtg05PN_[1]->getName());
  sprintf(histo, "EcalBarrel/EESummaryClient/EEPT EE - PN pedestal quality summary");
  if ( qtg06_[0] ) mui_->useQTest(histo, qtg06_[0]->getName());
  sprintf(histo, "EcalBarrel/EESummaryClient/EEPT EE + PN pedestal quality summary");
  if ( qtg06_[1] ) mui_->useQTest(histo, qtg06_[1]->getName());
  sprintf(histo, "EcalBarrel/EESummaryClient/EEPT EE - PN pedestal quality summary");
  if ( qtg06PN_[0] ) mui_->useQTest(histo, qtg06PN_[0]->getName());
  sprintf(histo, "EcalBarrel/EESummaryClient/EEPT EE + PN pedestal quality summary");
  if ( qtg06PN_[1] ) mui_->useQTest(histo, qtg06PN_[1]->getName());
  sprintf(histo, "EcalBarrel/EESummaryClient/EETPT EE - test pulse quality summary");
  if ( qtg07_[0] ) mui_->useQTest(histo, qtg07_[0]->getName());
  sprintf(histo, "EcalBarrel/EESummaryClient/EETPT EE + test pulse quality summary");
  if ( qtg07_[1] ) mui_->useQTest(histo, qtg07_[1]->getName());
  sprintf(histo, "EcalBarrel/EESummaryClient/EETPT EE - PN test pulse quality summary");
  if ( qtg07PN_[0] ) mui_->useQTest(histo, qtg07PN_[0]->getName());
  sprintf(histo, "EcalBarrel/EESummaryClient/EETPT EE + PN test pulse quality summary");
  if ( qtg07PN_[1] ) mui_->useQTest(histo, qtg07PN_[1]->getName());
  sprintf(histo, "EcalEndcap/EESummaryClient/EE global summary EE -");
  if ( qtg08_[0] ) mui_->useQTest(histo, qtg08_[0]->getName());
  sprintf(histo, "EcalEndcap/EESummaryClient/EE global summary EE +");
  if ( qtg08_[1] ) mui_->useQTest(histo, qtg08_[1]->getName());

}

void EESummaryClient::subscribeNew(void){

}

void EESummaryClient::unsubscribe(void){

  if ( verbose_ ) cout << "EESummaryClient: unsubscribe" << endl;

}

void EESummaryClient::softReset(void){

}

void EESummaryClient::analyze(void){

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( verbose_ ) cout << "EESummaryClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;
  }

  for ( int ix = 1; ix <= 100; ix++ ) {
    for ( int iy = 1; iy <= 100; iy++ ) {

      meIntegrity_[0]->setBinContent( ix, iy, -1. );
      meIntegrity_[1]->setBinContent( ix, iy, -1. );
      meOccupancy_[0]->setBinContent( ix, iy, -1. );
      meOccupancy_[1]->setBinContent( ix, iy, -1. );
      mePedestalOnline_[0]->setBinContent( ix, iy, -1. );
      mePedestalOnline_[1]->setBinContent( ix, iy, -1. );

      meLaserL1_[0]->setBinContent( ix, iy, -1. );
      meLaserL1_[1]->setBinContent( ix, iy, -1. );
      meLed_[0]->setBinContent( ix, iy, -1. );
      meLed_[1]->setBinContent( ix, iy, -1. );
      mePedestal_[0]->setBinContent( ix, iy, -1. );
      mePedestal_[1]->setBinContent( ix, iy, -1. );
      meTestPulse_[0]->setBinContent( ix, iy, -1. );
      meTestPulse_[1]->setBinContent( ix, iy, -1. );

      meGlobalSummary_[0]->setBinContent( ix, iy, -1. );
      meGlobalSummary_[1]->setBinContent( ix, iy, -1. );

    }
  }

  for ( unsigned int i=0; i<clients_.size(); i++ ) {

    EEIntegrityClient* eeic = dynamic_cast<EEIntegrityClient*>(clients_[i]);
    EEPedestalOnlineClient* eepoc = dynamic_cast<EEPedestalOnlineClient*>(clients_[i]);

    EELaserClient* eelc = dynamic_cast<EELaserClient*>(clients_[i]);
    EELedClient* eeldc = dynamic_cast<EELedClient*>(clients_[i]);
    EEPedestalClient* eepc = dynamic_cast<EEPedestalClient*>(clients_[i]);
    EETestPulseClient* eetpc = dynamic_cast<EETestPulseClient*>(clients_[i]);

    MonitorElement* me;
    MonitorElement *me_01, *me_02, *me_03;
//    MonitorElement *me_04, *me_05;
    TH2F* h2;

    std::map<float,float> priority;
    priority.insert( make_pair(0,3) );
    priority.insert( make_pair(1,1) );
    priority.insert( make_pair(2,2) );
    priority.insert( make_pair(3,2) );
    priority.insert( make_pair(4,3) );
    priority.insert( make_pair(5,1) );

    for ( unsigned int i=0; i<superModules_.size(); i++ ) {

      int ism = superModules_[i];

      for ( int ix = 1; ix <= 50; ix++ ) {
        for ( int iy = 1; iy <= 50; iy++ ) {

          int jx = ix + Numbers::ix0EE(ism);
          int jy = iy + Numbers::iy0EE(ism);

          if ( eeic ) {

            me = eeic->meg01_[ism-1];

            if ( me ) {

              float xval = me->getBinContent( ix, iy );

              if ( Numbers::validEE(ism, 101 - jx, jy) ) {
                if ( ism <= 9 ) {
                  meIntegrity_[0]->setBinContent( jx, jy, xval );
                } else {
                  meIntegrity_[1]->setBinContent( jx, jy, xval );
                }
              }

            }

            h2 = eeic->h_[ism-1];

            if ( h2 ) {

              float xval = h2->GetBinContent( ix, iy );

              if ( Numbers::validEE(ism, 101 - jx, jy) ) {
                if ( ism <= 9 ) {
                  meOccupancy_[0]->setBinContent( jx, jy, xval );
                } else {
                  meOccupancy_[1]->setBinContent( jx, jy, xval );
                }
              }

            }

          }

          if ( eepoc ) {

            me = eepoc->meg03_[ism-1];

            if ( me ) {

              float xval = me->getBinContent( ix, iy );

              if ( Numbers::validEE(ism, 101 - jx, jy) ) {
                if ( ism <= 9 ) {
                  mePedestalOnline_[0]->setBinContent( jx, jy, xval );
                } else {
                  mePedestalOnline_[1]->setBinContent( jx, jy, xval );
                }
              }

            }

          }

          if ( eelc ) {

            me = eelc->meg01_[ism-1];

            if ( me ) {

              float xval = me->getBinContent( ix, iy );

              if ( Numbers::validEE(ism, 101 - jx, jy) ) {
                if ( ism <= 9 ) {
                  if ( me->getEntries() != 0 ) {
                    meLaserL1_[0]->setBinContent( jx, jy, xval );
                  }
                } else {
                  if ( me->getEntries() != 0 ) {
                    meLaserL1_[1]->setBinContent( jx, jy, xval );
                  }
                }
              }

            }

          }

          if ( eeldc ) {

            me = eeldc->meg01_[ism-1];

            if ( me ) {

              float xval = me->getBinContent( ix, iy );

              if ( Numbers::validEE(ism, 101 - jx, jy) ) {
                if ( ism <= 9 ) {
                  if ( me->getEntries() != 0 ) {
                    meLed_[0]->setBinContent( jx, jy, xval );
                  }
                } else {
                  if ( me->getEntries() != 0 ) {
                    meLed_[1]->setBinContent( jx, jy, xval );
                  }
                }
              }

            }

          }

          if ( eepc ) {

            me_01 = eepc->meg01_[ism-1];
            me_02 = eepc->meg02_[ism-1];
            me_03 = eepc->meg03_[ism-1];

            if (me_01 && me_02 && me_03 ) {
              float xval=2;
              float val_01=me_01->getBinContent(ix,iy);
              float val_02=me_02->getBinContent(ix,iy);
              float val_03=me_03->getBinContent(ix,iy);

              std::vector<float> maskedVal, unmaskedVal;
              (val_01>2) ? maskedVal.push_back(val_01) : unmaskedVal.push_back(val_01);
              (val_02>2) ? maskedVal.push_back(val_02) : unmaskedVal.push_back(val_02);
              (val_03>2) ? maskedVal.push_back(val_03) : unmaskedVal.push_back(val_03);

              float brightColor=-1, darkColor=-1;
              float maxPriority=-1;
              std::vector<float>::const_iterator Val;
              for(Val=unmaskedVal.begin(); Val<unmaskedVal.end(); Val++) {
                if(priority[*Val]>maxPriority) brightColor=*Val;
              }
              maxPriority=-1;
              for(Val=maskedVal.begin(); Val<maskedVal.end(); Val++) {
                if(priority[*Val]>maxPriority) darkColor=*Val;
              }
              if(unmaskedVal.size()==3)  xval = brightColor;
              else if(maskedVal.size()==3)  xval = darkColor;
              else {
                if(brightColor==1 && darkColor==5) xval = 5;
                else xval = brightColor;
              }

              if ( Numbers::validEE(ism, 101 - jx, jy) ) {
                if ( ism <= 9 ) {
                  if ( me_01->getEntries() != 0 && me_02->getEntries() != 0 && me_03->getEntries() != 0 ) {
                    mePedestal_[0]->setBinContent( jx, jy, xval );
                  }
                } else {
                  if ( me_01->getEntries() != 0 && me_02->getEntries() != 0 && me_03->getEntries() != 0 ) {
                    mePedestal_[1]->setBinContent( jx, jy, xval );
                  }
                }
              }

            }

          }

          if ( eetpc ) {

            me_01 = eetpc->meg01_[ism-1];
            me_02 = eetpc->meg02_[ism-1];
            me_03 = eetpc->meg03_[ism-1];

            if (me_01 && me_02 && me_03 ) {
              float xval=2;
              float val_01=me_01->getBinContent(ix,iy);
              float val_02=me_02->getBinContent(ix,iy);
              float val_03=me_03->getBinContent(ix,iy);

              std::vector<float> maskedVal, unmaskedVal;
              (val_01>2) ? maskedVal.push_back(val_01) : unmaskedVal.push_back(val_01);
              (val_02>2) ? maskedVal.push_back(val_02) : unmaskedVal.push_back(val_02);
              (val_03>2) ? maskedVal.push_back(val_03) : unmaskedVal.push_back(val_03);

              float brightColor=-1, darkColor=-1;
              float maxPriority=-1;
              std::vector<float>::const_iterator Val;
              for(Val=unmaskedVal.begin(); Val<unmaskedVal.end(); Val++) {
                if(priority[*Val]>maxPriority) brightColor=*Val;
              }
              maxPriority=-1;
              for(Val=maskedVal.begin(); Val<maskedVal.end(); Val++) {
                if(priority[*Val]>maxPriority) darkColor=*Val;
              }
              if(unmaskedVal.size()==3) xval = brightColor;
              else if(maskedVal.size()==3) xval = darkColor;
              else {
                if(brightColor==1 && darkColor==5) xval = 5;
                else xval = brightColor;
              }

              if ( Numbers::validEE(ism, 101 - jx, jy) ) {
                if ( ism <= 9 ) {
                  if ( me_01->getEntries() != 0 && me_02->getEntries() != 0 && me_03->getEntries() != 0 ) {
                    meTestPulse_[0]->setBinContent( jx, jy, xval );
                  }
                } else {
                  if ( me_01->getEntries() != 0 && me_02->getEntries() != 0 && me_03->getEntries() != 0 ) {
                    meTestPulse_[1]->setBinContent( jx, jy, xval );
                  }
                }
              }

            }

          }

        }
      }

    }

  }

  // The global-summary
  // right now a summary of Integrity and PO
  for ( int jx = 1; jx <= 100; jx++ ) {
    for ( int jy = 1; jy <= 100; jy++ ) {

      if(meIntegrity_[0] && mePedestalOnline_[0]) {

        float xval = 2;
        float val_in = meIntegrity_[0]->getBinContent(jx,jy); 
        float val_po = mePedestalOnline_[0]->getBinContent(jx,jy);

        // turn each dark color to bright green
        if(val_in>2) val_in=1;
        if(val_po>2) val_po=1;

        if(val_in==0) xval=0;
        else if(val_in==2) xval=2;
        else xval=val_po;

        meGlobalSummary_[0]->setBinContent( jx, jy, xval );

      }

      if(meIntegrity_[1] && mePedestalOnline_[1]) {

        float xval = 2;
        float val_in = meIntegrity_[1]->getBinContent(jx,jy);  
        float val_po = mePedestalOnline_[1]->getBinContent(jx,jy);

        // turn each dark color to bright green
        if(val_in>2) val_in=1;
        if(val_po>2) val_po=1;

        if(val_in==0) xval=0;
        else if(val_in==2) xval=2;
        else xval=val_po;

        meGlobalSummary_[1]->setBinContent( jx, jy, xval );

      }

    }
  }

}

void EESummaryClient::htmlOutput(int run, string htmlDir, string htmlName){

  cout << "Preparing EESummaryClient html output ..." << endl;

  ofstream htmlFile;

  htmlFile.open((htmlDir + htmlName).c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlFile << "<html>  " << endl;
  htmlFile << "<head>  " << endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlFile << " http-equiv=\"content-type\">  " << endl;
  htmlFile << "  <title>Monitor:Summary output</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlFile << "<body>  " << endl;
  //htmlFile << "<br>  " << endl;
  htmlFile << "<a name=""top""></a>" << endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << run << "</span></h2>" << endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">SUMMARY</span></h2> " << endl;
  htmlFile << "<hr>" << endl;
  htmlFile << "<table border=1><tr><td bgcolor=red>channel has problems in this task</td>" << endl;
  htmlFile << "<td bgcolor=lime>channel has NO problems</td>" << endl;
  htmlFile << "<td bgcolor=yellow>channel is missing</td></table>" << endl;
  htmlFile << "<br>" << endl;

  // Produce the plots to be shown as .png files from existing histograms

  const int csize = 250;

//  const double histMax = 1.e15;

  int pCol3[6] = { 301, 302, 303, 304, 305, 306 };
  int pCol4[10];
  for ( int i = 0; i < 10; i++ ) pCol4[i] = 401+i;

  // dummy histogram labelling the SM's
  TH2C labelGrid1("labelGrid1","label grid for EE -", 10, 0., 100., 10, 0., 100.);

  for ( int i=1; i<=10; i++) {
    for ( int j=1; j<=10; j++) {
      labelGrid1.SetBinContent(i, j, -10);
    }
  }

  labelGrid1.SetBinContent(2, 5, -7);
  labelGrid1.SetBinContent(2, 7, -8);
  labelGrid1.SetBinContent(4, 9, -9);
  labelGrid1.SetBinContent(7, 9, -1);
  labelGrid1.SetBinContent(9, 7, -2);
  labelGrid1.SetBinContent(9, 5, -3);
  labelGrid1.SetBinContent(8, 3, -4);
  labelGrid1.SetBinContent(6, 2, -5);
  labelGrid1.SetBinContent(3, 3, -6);

  labelGrid1.SetMarkerSize(2);
  labelGrid1.SetMinimum(-9.01);
  labelGrid1.SetMaximum(-0.01);

  TH2C labelGrid2("labelGrid2","label grid for EE +", 10, 0., 100., 10, 0., 100.);

  for ( int i=1; i<=10; i++) {
    for ( int j=1; j<=10; j++) {
      labelGrid2.SetBinContent(i, j, -10);
    }
  }

  labelGrid2.SetBinContent(2, 5, +3);
  labelGrid2.SetBinContent(2, 7, +2);
  labelGrid2.SetBinContent(4, 9, +1);
  labelGrid2.SetBinContent(7, 9, +9);
  labelGrid2.SetBinContent(9, 7, +8);
  labelGrid2.SetBinContent(9, 5, +7);
  labelGrid2.SetBinContent(8, 3, +6);
  labelGrid2.SetBinContent(5, 2, +5);
  labelGrid2.SetBinContent(3, 3, +4);

  labelGrid2.SetMarkerSize(2);
  labelGrid2.SetMinimum(+0.01);
  labelGrid2.SetMaximum(+9.01);

  string imgNameMapI[2], imgNameMapO[2], imgNameMapPO[2], imgNameMapLL1[2], imgNameMapLL1_PN[2], imgNameMapLD[2], imgNameMapLD_PN[2], imgNameMapP[2], imgNameMapP_PN[2], imgNameMapTP[2], imgNameMapTP_PN[2], imgName, meName;
  string imgNameMapGS[2];

  TCanvas* cMap = new TCanvas("cMap", "Temp", int(1.5*csize), int(1.5*csize));

  float saveHeigth = gStyle->GetTitleH();
  gStyle->SetTitleH(0.07);
  float saveFontSize = gStyle->GetTitleFontSize();
  gStyle->SetTitleFontSize(15);

  TH2F* obj2f;

  gStyle->SetPaintTextFormat("+g");

  imgNameMapI[0] = "";

  obj2f = 0;
  obj2f = UtilsClient::getHisto<TH2F*>( meIntegrity_[0] );

  if ( obj2f ) {

    meName = obj2f->GetName();

    for ( unsigned int i = 0; i < meName.size(); i++ ) {
      if ( meName.substr(i, 1) == " " )  {
        meName.replace(i, 1 ,"_" );
      }
    }
    imgNameMapI[0] = meName + ".png";
    imgName = htmlDir + imgNameMapI[0];

    cMap->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(6, pCol3);
    cMap->SetGridx();
    cMap->SetGridy();
    obj2f->SetMinimum(-0.00000001);
    obj2f->SetMaximum(6.0);
    obj2f->SetTitleSize(0.5);
    obj2f->Draw("col");
    labelGrid1.Draw("text,same");
    cMap->SetBit(TGraph::kClipFrame);
    TLine l;
    l.SetLineWidth(1);
    for ( int i=0; i<201; i=i+1){
      if ( (Numbers::ixSectorsEE[i]!=0 || Numbers::iySectorsEE[i]!=0) && (Numbers::ixSectorsEE[i+1]!=0 || Numbers::iySectorsEE[i+1]!=0) ) {
        l.DrawLine(Numbers::ixSectorsEE[i], Numbers::iySectorsEE[i], Numbers::ixSectorsEE[i+1], Numbers::iySectorsEE[i+1]);
      }
    }
    cMap->Update();
    cMap->SaveAs(imgName.c_str());

  }

  imgNameMapI[1] = "";

  obj2f = 0;
  obj2f = UtilsClient::getHisto<TH2F*>( meIntegrity_[1] );

  if ( obj2f ) {

    meName = obj2f->GetName();

    for ( unsigned int i = 0; i < meName.size(); i++ ) {
      if ( meName.substr(i, 1) == " " )  {
        meName.replace(i, 1 ,"_" );
      }
    }
    imgNameMapI[1] = meName + ".png";
    imgName = htmlDir + imgNameMapI[1];

    cMap->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(6, pCol3);
    cMap->SetGridx();
    cMap->SetGridy();
    obj2f->SetMinimum(-0.00000001);
    obj2f->SetMaximum(6.0);
    obj2f->SetTitleSize(0.5);
    obj2f->Draw("col");
    labelGrid2.Draw("text,same");
    cMap->SetBit(TGraph::kClipFrame);
    TLine l;
    l.SetLineWidth(1);
    for ( int i=0; i<201; i=i+1){
      if ( (Numbers::ixSectorsEE[i]!=0 || Numbers::iySectorsEE[i]!=0) && (Numbers::ixSectorsEE[i+1]!=0 || Numbers::iySectorsEE[i+1]!=0) ) {
        l.DrawLine(Numbers::ixSectorsEE[i], Numbers::iySectorsEE[i], Numbers::ixSectorsEE[i+1], Numbers::iySectorsEE[i+1]);
      }
    }
    cMap->Update();
    cMap->SaveAs(imgName.c_str());

  }

  imgNameMapO[0] = "";

  obj2f = 0;
  obj2f = UtilsClient::getHisto<TH2F*>( meOccupancy_[0] );

  if ( obj2f ) {

    meName = obj2f->GetName();

    for ( unsigned int i = 0; i < meName.size(); i++ ) {
      if ( meName.substr(i, 1) == " " )  {
        meName.replace(i, 1 ,"_" );
      }
    }
    imgNameMapO[0] = meName + ".png";
    imgName = htmlDir + imgNameMapO[0];

    cMap->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(10, pCol4);
    cMap->SetGridx();
    cMap->SetGridy();
    obj2f->SetMinimum(-0.00000001);
    obj2f->SetMaximum(6.0);
    obj2f->Draw("colz");
    labelGrid1.Draw("text,same");
    cMap->SetBit(TGraph::kClipFrame);
    TLine l;
    l.SetLineWidth(1);
    for ( int i=0; i<201; i=i+1){
      if ( (Numbers::ixSectorsEE[i]!=0 || Numbers::iySectorsEE[i]!=0) && (Numbers::ixSectorsEE[i+1]!=0 || Numbers::iySectorsEE[i+1]!=0) ) {
        l.DrawLine(Numbers::ixSectorsEE[i], Numbers::iySectorsEE[i], Numbers::ixSectorsEE[i+1], Numbers::iySectorsEE[i+1]);
      }
    }
    cMap->Update();
    cMap->SaveAs(imgName.c_str());

  }

  imgNameMapO[1] = "";

  obj2f = 0;
  obj2f = UtilsClient::getHisto<TH2F*>( meOccupancy_[1] );

  if ( obj2f ) {

    meName = obj2f->GetName();

    for ( unsigned int i = 0; i < meName.size(); i++ ) {
      if ( meName.substr(i, 1) == " " )  {
        meName.replace(i, 1 ,"_" );
      }
    }
    imgNameMapO[1] = meName + ".png";
    imgName = htmlDir + imgNameMapO[1];

    cMap->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(10, pCol4);
    cMap->SetGridx();
    cMap->SetGridy();
    obj2f->SetMinimum(-0.00000001);
    obj2f->SetMaximum(6.0);
    obj2f->Draw("colz");
    labelGrid2.Draw("text,same");
    cMap->SetBit(TGraph::kClipFrame);
    TLine l;
    l.SetLineWidth(1);
    for ( int i=0; i<201; i=i+1){
      if ( (Numbers::ixSectorsEE[i]!=0 || Numbers::iySectorsEE[i]!=0) && (Numbers::ixSectorsEE[i+1]!=0 || Numbers::iySectorsEE[i+1]!=0) ) {
        l.DrawLine(Numbers::ixSectorsEE[i], Numbers::iySectorsEE[i], Numbers::ixSectorsEE[i+1], Numbers::iySectorsEE[i+1]);
      }
    }
    cMap->Update();
    cMap->SaveAs(imgName.c_str());

  }

  imgNameMapPO[0] = "";

  obj2f = 0;
  obj2f = UtilsClient::getHisto<TH2F*>( mePedestalOnline_[0] );

  if ( obj2f ) {

    meName = obj2f->GetName();

    for ( unsigned int i = 0; i < meName.size(); i++ ) {
      if ( meName.substr(i, 1) == " " )  {
        meName.replace(i, 1 ,"_" );
      }
    }
    imgNameMapPO[0] = meName + ".png";
    imgName = htmlDir + imgNameMapPO[0];

    cMap->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(6, pCol3);
    cMap->SetGridx();
    cMap->SetGridy();
    obj2f->SetMinimum(-0.00000001);
    obj2f->SetMaximum(6.0);
    obj2f->Draw("col");
    labelGrid1.Draw("text,same");
    cMap->SetBit(TGraph::kClipFrame);
    TLine l;
    l.SetLineWidth(1);
    for ( int i=0; i<201; i=i+1){
      if ( (Numbers::ixSectorsEE[i]!=0 || Numbers::iySectorsEE[i]!=0) && (Numbers::ixSectorsEE[i+1]!=0 || Numbers::iySectorsEE[i+1]!=0) ) {
        l.DrawLine(Numbers::ixSectorsEE[i], Numbers::iySectorsEE[i], Numbers::ixSectorsEE[i+1], Numbers::iySectorsEE[i+1]);
      }
    }
    cMap->Update();
    cMap->SaveAs(imgName.c_str());

  }

  imgNameMapPO[1] = "";

  obj2f = 0;
  obj2f = UtilsClient::getHisto<TH2F*>( mePedestalOnline_[1] );

  if ( obj2f ) {

    meName = obj2f->GetName();

    for ( unsigned int i = 0; i < meName.size(); i++ ) {
      if ( meName.substr(i, 1) == " " )  {
        meName.replace(i, 1 ,"_" );
      }
    }
    imgNameMapPO[1] = meName + ".png";
    imgName = htmlDir + imgNameMapPO[1];

    cMap->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(6, pCol3);
    cMap->SetGridx();
    cMap->SetGridy();
    obj2f->SetMinimum(-0.00000001);
    obj2f->SetMaximum(6.0);
    obj2f->Draw("col");
    labelGrid2.Draw("text,same");
    cMap->SetBit(TGraph::kClipFrame);
    TLine l;
    l.SetLineWidth(1);
    for ( int i=0; i<201; i=i+1){
      if ( (Numbers::ixSectorsEE[i]!=0 || Numbers::iySectorsEE[i]!=0) && (Numbers::ixSectorsEE[i+1]!=0 || Numbers::iySectorsEE[i+1]!=0) ) {
        l.DrawLine(Numbers::ixSectorsEE[i], Numbers::iySectorsEE[i], Numbers::ixSectorsEE[i+1], Numbers::iySectorsEE[i+1]);
      }
    }
    cMap->Update();
    cMap->SaveAs(imgName.c_str());

  }

  imgNameMapLL1[0] = "";

  obj2f = 0;
  obj2f = UtilsClient::getHisto<TH2F*>( meLaserL1_[0] );

  if ( obj2f ) {

    meName = obj2f->GetName();

    for ( unsigned int i = 0; i < meName.size(); i++ ) {
      if ( meName.substr(i, 1) == " " )  {
        meName.replace(i, 1 ,"_" );
      }
    }
    imgNameMapLL1[0] = meName + ".png";
    imgName = htmlDir + imgNameMapLL1[0];

    cMap->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(6, pCol3);
    cMap->SetGridx();
    cMap->SetGridy();
    obj2f->SetMinimum(-0.00000001);
    obj2f->SetMaximum(6.0);
    obj2f->SetTitleSize(0.5);
    obj2f->Draw("col");
    labelGrid1.Draw("text,same");
    cMap->SetBit(TGraph::kClipFrame);
    TLine l;
    l.SetLineWidth(1);
    for ( int i=0; i<201; i=i+1){
      if ( (Numbers::ixSectorsEE[i]!=0 || Numbers::iySectorsEE[i]!=0) && (Numbers::ixSectorsEE[i+1]!=0 || Numbers::iySectorsEE[i+1]!=0) ) {
        l.DrawLine(Numbers::ixSectorsEE[i], Numbers::iySectorsEE[i], Numbers::ixSectorsEE[i+1], Numbers::iySectorsEE[i+1]);
      }
    }
    cMap->Update();
    cMap->SaveAs(imgName.c_str());

  }

  imgNameMapLL1[1] = "";

  obj2f = 0;
  obj2f = UtilsClient::getHisto<TH2F*>( meLaserL1_[1] );

  if ( obj2f ) {

    meName = obj2f->GetName();

    for ( unsigned int i = 0; i < meName.size(); i++ ) {
      if ( meName.substr(i, 1) == " " )  {
        meName.replace(i, 1 ,"_" );
      }
    }
    imgNameMapLL1[1] = meName + ".png";
    imgName = htmlDir + imgNameMapLL1[1];

    cMap->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(6, pCol3);
    cMap->SetGridx();
    cMap->SetGridy();
    obj2f->SetMinimum(-0.00000001);
    obj2f->SetMaximum(6.0);
    obj2f->SetTitleSize(0.5);
    obj2f->Draw("col");
    labelGrid2.Draw("text,same");
    cMap->SetBit(TGraph::kClipFrame);
    TLine l;
    l.SetLineWidth(1);
    for ( int i=0; i<201; i=i+1){
      if ( (Numbers::ixSectorsEE[i]!=0 || Numbers::iySectorsEE[i]!=0) && (Numbers::ixSectorsEE[i+1]!=0 || Numbers::iySectorsEE[i+1]!=0) ) {
        l.DrawLine(Numbers::ixSectorsEE[i], Numbers::iySectorsEE[i], Numbers::ixSectorsEE[i+1], Numbers::iySectorsEE[i+1]);
      }
    }
    cMap->Update();
    cMap->SaveAs(imgName.c_str());
  
  }

//
  imgNameMapLL1_PN[0] = "";
  imgNameMapLL1_PN[1] = "";
//

  imgNameMapLD[0] = "";

  obj2f = 0;
  obj2f = UtilsClient::getHisto<TH2F*>( meLed_[0] );

  if ( obj2f ) {

    meName = obj2f->GetName();

    for ( unsigned int i = 0; i < meName.size(); i++ ) {
      if ( meName.substr(i, 1) == " " )  {
        meName.replace(i, 1 ,"_" );
      }
    }
    imgNameMapLD[0] = meName + ".png";
    imgName = htmlDir + imgNameMapLD[0];

    cMap->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(6, pCol3);
    cMap->SetGridx();
    cMap->SetGridy();
    obj2f->SetMinimum(-0.00000001);
    obj2f->SetMaximum(6.0);
    obj2f->SetTitleSize(0.5);
    obj2f->Draw("col");
    labelGrid1.Draw("text,same");
    cMap->SetBit(TGraph::kClipFrame);
    TLine l;
    l.SetLineWidth(1);
    for ( int i=0; i<201; i=i+1){
      if ( (Numbers::ixSectorsEE[i]!=0 || Numbers::iySectorsEE[i]!=0) && (Numbers::ixSectorsEE[i+1]!=0 || Numbers::iySectorsEE[i+1]!=0) ) {
        l.DrawLine(Numbers::ixSectorsEE[i], Numbers::iySectorsEE[i], Numbers::ixSectorsEE[i+1], Numbers::iySectorsEE[i+1]);
      }
    }
    cMap->Update();
    cMap->SaveAs(imgName.c_str());

  }

  imgNameMapLD[1] = "";

  obj2f = 0;
  obj2f = UtilsClient::getHisto<TH2F*>( meLed_[1] );
  
  if ( obj2f ) {
  
    meName = obj2f->GetName();
  
    for ( unsigned int i = 0; i < meName.size(); i++ ) {
      if ( meName.substr(i, 1) == " " )  {
        meName.replace(i, 1 ,"_" );
      }
    }
    imgNameMapLD[1] = meName + ".png";
    imgName = htmlDir + imgNameMapLD[1];

    cMap->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(6, pCol3);
    cMap->SetGridx();
    cMap->SetGridy();
    obj2f->SetMinimum(-0.00000001);
    obj2f->SetMaximum(6.0);
    obj2f->SetTitleSize(0.5);
    obj2f->Draw("col");
    labelGrid2.Draw("text,same");
    cMap->SetBit(TGraph::kClipFrame);
    TLine l;
    l.SetLineWidth(1);
    for ( int i=0; i<201; i=i+1){
      if ( (Numbers::ixSectorsEE[i]!=0 || Numbers::iySectorsEE[i]!=0) && (Numbers::ixSectorsEE[i+1]!=0 || Numbers::iySectorsEE[i+1]!=0) ) {
        l.DrawLine(Numbers::ixSectorsEE[i], Numbers::iySectorsEE[i], Numbers::ixSectorsEE[i+1], Numbers::iySectorsEE[i+1]);
      }
    }
    cMap->Update();
    cMap->SaveAs(imgName.c_str());

  }

//
  imgNameMapLD_PN[0] = "";
  imgNameMapLD_PN[1] = "";
//

  imgNameMapP[0] = "";

  obj2f = 0;
  obj2f = UtilsClient::getHisto<TH2F*>( mePedestal_[0] );

  if ( obj2f ) {

    meName = obj2f->GetName();

    for ( unsigned int i = 0; i < meName.size(); i++ ) {
      if ( meName.substr(i, 1) == " " )  {
        meName.replace(i, 1 ,"_" );
      }
    }
    imgNameMapP[0] = meName + ".png";
    imgName = htmlDir + imgNameMapP[0];

    cMap->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(6, pCol3);
    cMap->SetGridx();
    cMap->SetGridy();
    obj2f->SetMinimum(-0.00000001);
    obj2f->SetMaximum(6.0);
    obj2f->SetTitleSize(0.5);
    obj2f->Draw("col");
    labelGrid1.Draw("text,same");
    cMap->SetBit(TGraph::kClipFrame);
    TLine l;
    l.SetLineWidth(1);
    for ( int i=0; i<201; i=i+1){
      if ( (Numbers::ixSectorsEE[i]!=0 || Numbers::iySectorsEE[i]!=0) && (Numbers::ixSectorsEE[i+1]!=0 || Numbers::iySectorsEE[i+1]!=0) ) {
        l.DrawLine(Numbers::ixSectorsEE[i], Numbers::iySectorsEE[i], Numbers::ixSectorsEE[i+1], Numbers::iySectorsEE[i+1]);
      }
    }
    cMap->Update();
    cMap->SaveAs(imgName.c_str());

  }

  imgNameMapP[1] = "";

  obj2f = 0;
  obj2f = UtilsClient::getHisto<TH2F*>( mePedestal_[1] );
  
  if ( obj2f ) {
  
    meName = obj2f->GetName();
  
    for ( unsigned int i = 0; i < meName.size(); i++ ) {
      if ( meName.substr(i, 1) == " " )  {
        meName.replace(i, 1 ,"_" );
      }
    }
    imgNameMapP[1] = meName + ".png";
    imgName = htmlDir + imgNameMapP[1];

    cMap->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(6, pCol3);
    cMap->SetGridx();
    cMap->SetGridy();
    obj2f->SetMinimum(-0.00000001);
    obj2f->SetMaximum(6.0);
    obj2f->SetTitleSize(0.5);
    obj2f->Draw("col");
    labelGrid2.Draw("text,same");
    cMap->SetBit(TGraph::kClipFrame);
    TLine l;
    l.SetLineWidth(1);
    for ( int i=0; i<201; i=i+1){
      if ( (Numbers::ixSectorsEE[i]!=0 || Numbers::iySectorsEE[i]!=0) && (Numbers::ixSectorsEE[i+1]!=0 || Numbers::iySectorsEE[i+1]!=0) ) {
        l.DrawLine(Numbers::ixSectorsEE[i], Numbers::iySectorsEE[i], Numbers::ixSectorsEE[i+1], Numbers::iySectorsEE[i+1]);
      }
    }
    cMap->Update();
    cMap->SaveAs(imgName.c_str());

  }

//
  imgNameMapP_PN[0] = "";
  imgNameMapP_PN[1] = "";
//

  imgNameMapTP[0] = "";

  obj2f = 0;
  obj2f = UtilsClient::getHisto<TH2F*>( meTestPulse_[0] );

  if ( obj2f ) {

    meName = obj2f->GetName();

    for ( unsigned int i = 0; i < meName.size(); i++ ) {
      if ( meName.substr(i, 1) == " " )  {
        meName.replace(i, 1 ,"_" );
      }
    }
    imgNameMapTP[0] = meName + ".png";
    imgName = htmlDir + imgNameMapTP[0];

    cMap->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(6, pCol3);
    cMap->SetGridx();
    cMap->SetGridy();
    obj2f->SetMinimum(-0.00000001);
    obj2f->SetMaximum(6.0);
    obj2f->SetTitleSize(0.5);
    obj2f->Draw("col");
    labelGrid1.Draw("text,same");
    cMap->SetBit(TGraph::kClipFrame);
    TLine l;
    l.SetLineWidth(1);
    for ( int i=0; i<201; i=i+1){
      if ( (Numbers::ixSectorsEE[i]!=0 || Numbers::iySectorsEE[i]!=0) && (Numbers::ixSectorsEE[i+1]!=0 || Numbers::iySectorsEE[i+1]!=0) ) {
        l.DrawLine(Numbers::ixSectorsEE[i], Numbers::iySectorsEE[i], Numbers::ixSectorsEE[i+1], Numbers::iySectorsEE[i+1]);
      }
    }
    cMap->Update();
    cMap->SaveAs(imgName.c_str());

  }

  imgNameMapTP[1] = "";

  obj2f = 0;
  obj2f = UtilsClient::getHisto<TH2F*>( meTestPulse_[1] );
  
  if ( obj2f ) {
  
    meName = obj2f->GetName();
  
    for ( unsigned int i = 0; i < meName.size(); i++ ) {
      if ( meName.substr(i, 1) == " " )  {
        meName.replace(i, 1 ,"_" );
      }
    }
    imgNameMapTP[1] = meName + ".png";
    imgName = htmlDir + imgNameMapTP[1];

    cMap->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(6, pCol3);
    cMap->SetGridx();
    cMap->SetGridy();
    obj2f->SetMinimum(-0.00000001);
    obj2f->SetMaximum(6.0);
    obj2f->SetTitleSize(0.5);
    obj2f->Draw("col");
    labelGrid2.Draw("text,same");
    cMap->SetBit(TGraph::kClipFrame);
    TLine l;
    l.SetLineWidth(1);
    for ( int i=0; i<201; i=i+1){
      if ( (Numbers::ixSectorsEE[i]!=0 || Numbers::iySectorsEE[i]!=0) && (Numbers::ixSectorsEE[i+1]!=0 || Numbers::iySectorsEE[i+1]!=0) ) {
        l.DrawLine(Numbers::ixSectorsEE[i], Numbers::iySectorsEE[i], Numbers::ixSectorsEE[i+1], Numbers::iySectorsEE[i+1]);
      }
    }
    cMap->Update();
    cMap->SaveAs(imgName.c_str());

  }

//
  imgNameMapTP_PN[0] = "";
  imgNameMapTP_PN[1] = "";
//

  imgNameMapGS[0] = "";

  obj2f = 0;
  obj2f = UtilsClient::getHisto<TH2F*>( meGlobalSummary_[0] );

  if ( obj2f ) {

    meName = obj2f->GetName();

    for ( unsigned int i = 0; i < meName.size(); i++ ) {
      if ( meName.substr(i, 1) == " " )  {
        meName.replace(i, 1 ,"_" );
      }
    }
    imgNameMapGS[0] = meName + ".png";
    imgName = htmlDir + imgNameMapGS[0];

    cMap->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(6, pCol3);
    cMap->SetGridx();
    cMap->SetGridy();
    obj2f->SetMinimum(-0.00000001);
    obj2f->SetMaximum(6.0);
    obj2f->SetTitleSize(0.5);
    obj2f->Draw("col");
    labelGrid1.Draw("text,same");
    cMap->SetBit(TGraph::kClipFrame);
    TLine l;
    l.SetLineWidth(1);
    for ( int i=0; i<201; i=i+1){
      if ( (Numbers::ixSectorsEE[i]!=0 || Numbers::iySectorsEE[i]!=0) && (Numbers::ixSectorsEE[i+1]!=0 || Numbers::iySectorsEE[i+1]!=0) ) {
        l.DrawLine(Numbers::ixSectorsEE[i], Numbers::iySectorsEE[i], Numbers::ixSectorsEE[i+1], Numbers::iySectorsEE[i+1]);
      }
    }
    cMap->Update();
    cMap->SaveAs(imgName.c_str());

  }

  imgNameMapGS[1] = "";

  obj2f = 0;
  obj2f = UtilsClient::getHisto<TH2F*>( meGlobalSummary_[1] );
  
  if ( obj2f ) {

    meName = obj2f->GetName();
    
    for ( unsigned int i = 0; i < meName.size(); i++ ) {
      if ( meName.substr(i, 1) == " " )  {
        meName.replace(i, 1 ,"_" );
      }
    }
    imgNameMapGS[1] = meName + ".png";
    imgName = htmlDir + imgNameMapGS[1];

    cMap->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(6, pCol3);
    cMap->SetGridx();
    cMap->SetGridy();
    obj2f->SetMinimum(-0.00000001);
    obj2f->SetMaximum(6.0);
    obj2f->SetTitleSize(0.5);
    obj2f->Draw("col");
    labelGrid2.Draw("text,same");
    cMap->SetBit(TGraph::kClipFrame);
    TLine l;
    l.SetLineWidth(1);
    for ( int i=0; i<201; i=i+1){
      if ( (Numbers::ixSectorsEE[i]!=0 || Numbers::iySectorsEE[i]!=0) && (Numbers::ixSectorsEE[i+1]!=0 || Numbers::iySectorsEE[i+1]!=0) ) {
        l.DrawLine(Numbers::ixSectorsEE[i], Numbers::iySectorsEE[i], Numbers::ixSectorsEE[i+1], Numbers::iySectorsEE[i+1]);
      }
    }
    cMap->Update();
    cMap->SaveAs(imgName.c_str());

  }

  gStyle->SetPaintTextFormat();

  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
  htmlFile << "<tr align=\"center\">" << endl;

  if ( imgNameMapI[0].size() != 0 )
    htmlFile << "<td><img src=\"" << imgNameMapI[0] << "\" usemap=\"#Integrity_0\" border=0></td>" << endl;
  else
    htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

  if ( imgNameMapI[1].size() != 0 )
    htmlFile << "<td><img src=\"" << imgNameMapI[1] << "\" usemap=\"#Integrity_1\" border=0></td>" << endl;
  else
    htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

  htmlFile << "</tr>" << endl;
  htmlFile << "</table>" << endl;
  htmlFile << "<br>" << endl;

  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
  htmlFile << "<tr align=\"center\">" << endl;

  if ( imgNameMapO[0].size() != 0 )
    htmlFile << "<td><img src=\"" << imgNameMapO[0] << "\" usemap=\"#Occupancy_0\" border=0></td>" << endl;
  else
    htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

  if ( imgNameMapO[1].size() != 0 )
    htmlFile << "<td><img src=\"" << imgNameMapO[1] << "\" usemap=\"#Occupancy_1\" border=0></td>" << endl;
  else
    htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

  htmlFile << "</tr>" << endl;
  htmlFile << "</table>" << endl;
  htmlFile << "<br>" << endl;

  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
  htmlFile << "<tr align=\"center\">" << endl;

  if ( imgNameMapPO[0].size() != 0 )
    htmlFile << "<td><img src=\"" << imgNameMapPO[0] << "\" usemap=\"#PedestalOnline_0\" border=0></td>" << endl;
  else
    htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

  if ( imgNameMapPO[1].size() != 0 )
    htmlFile << "<td><img src=\"" << imgNameMapPO[1] << "\" usemap=\"#PedestalOnline_1\" border=0></td>" << endl;
  else
    htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

  htmlFile << "</tr>" << endl;
  htmlFile << "</table>" << endl;
  htmlFile << "<br>" << endl;

  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
  htmlFile << "<tr align=\"center\">" << endl;

  if ( imgNameMapLL1[0].size() != 0 )
    htmlFile << "<td><img src=\"" << imgNameMapLL1[0] << "\" usemap=\"#LaserL1_0\" border=0></td>" << endl;
  else
    htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

  if ( imgNameMapLL1[1].size() != 0 )
    htmlFile << "<td><img src=\"" << imgNameMapLL1[1] << "\" usemap=\"#LaserL1_1\" border=0></td>" << endl;
  else
    htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

  htmlFile << "</tr>" << endl;
  htmlFile << "</table>" << endl;
  htmlFile << "<br>" << endl;

  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
  htmlFile << "<tr align=\"center\">" << endl;

  if ( imgNameMapLD[0].size() != 0 )
    htmlFile << "<td><img src=\"" << imgNameMapLD[0] << "\" usemap=\"#Led_0\" border=0></td>" << endl;
  else
    htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

  if ( imgNameMapLD[1].size() != 0 )
    htmlFile << "<td><img src=\"" << imgNameMapLD[1] << "\" usemap=\"#Led_1\" border=0></td>" << endl;
  else
    htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

  htmlFile << "</tr>" << endl;
  htmlFile << "</table>" << endl;
  htmlFile << "<br>" << endl;

  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
  htmlFile << "<tr align=\"center\">" << endl;

  if ( imgNameMapP[0].size() != 0 )
    htmlFile << "<td><img src=\"" << imgNameMapP[0] << "\" usemap=\"#Pedestal_0\" border=0></td>" << endl;
  else
    htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

  if ( imgNameMapP[1].size() != 0 )
    htmlFile << "<td><img src=\"" << imgNameMapP[1] << "\" usemap=\"#Pedestal_1\" border=0></td>" << endl;
  else
    htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

  htmlFile << "</tr>" << endl;
  htmlFile << "</table>" << endl;
  htmlFile << "<br>" << endl;

  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
  htmlFile << "<tr align=\"center\">" << endl;

  if ( imgNameMapTP[0].size() != 0 )
    htmlFile << "<td><img src=\"" << imgNameMapTP[0] << "\" usemap=\"#TestPulse_0\" border=0></td>" << endl;
  else
    htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

  if ( imgNameMapTP[1].size() != 0 )
    htmlFile << "<td><img src=\"" << imgNameMapTP[1] << "\" usemap=\"#TestPulse_1\" border=0></td>" << endl;
  else
    htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

  htmlFile << "</tr>" << endl;
  htmlFile << "</table>" << endl;
  htmlFile << "<br>" << endl;

  delete cMap;

  gStyle->SetPaintTextFormat();

  if ( imgNameMapI[0].size() != 0 || imgNameMapI[1].size() != 0 ) this->writeMap( htmlFile, "Integrity" );
  if ( imgNameMapO[0].size() != 0 || imgNameMapO[1].size() != 0 ) this->writeMap( htmlFile, "Occupancy" );
  if ( imgNameMapPO[0].size() != 0 || imgNameMapPO[1].size() != 0 ) this->writeMap( htmlFile, "PedestalOnline" );
  if ( imgNameMapLL1[0].size() != 0 || imgNameMapLL1[1].size() != 0 ) this->writeMap( htmlFile, "LaserL1" );
  if ( imgNameMapLD[0].size() != 0 || imgNameMapLD[1].size() != 0 ) this->writeMap( htmlFile, "Led" );
  if ( imgNameMapP[0].size() != 0  || imgNameMapP[1].size() != 0 ) this->writeMap( htmlFile, "Pedestal" );
  if ( imgNameMapTP[0].size() != 0 || imgNameMapTP[1].size() != 0 ) this->writeMap( htmlFile, "TestPulse" );

  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  htmlFile.close();

  gStyle->SetTitleH( saveHeigth );
  gStyle->SetTitleFontSize( saveFontSize );

}

void EESummaryClient::writeMap( std::ofstream& hf, std::string mapname ) {

  std::map<std::string, std::string> refhtml;
  refhtml["Integrity"] = "EEIntegrityClient.html";
  refhtml["Occupancy"] = "EEIntegrityClient.html";
  refhtml["PedestalOnline"] = "EEPedestalOnlineClient.html";
  refhtml["LaserL1"] = "EELaserClient.html";
  refhtml["Led"] = "EELedClient.html";
  refhtml["Pedestal"] = "EEPedestalClient.html";
  refhtml["TestPulse"] = "EETestPulseClient.html";

  const int A0 =  38;
  const int A1 = 334;
  const int B0 =  33;
  const int B1 = 312;

  const int C0 = 34;
  const int C1 = 148;

  hf << "<map name=\"" << mapname << "_0\">" << std::endl;
  for( unsigned int sm=0; sm<superModules_.size(); sm++ ) {
    if( superModules_[sm] >= 1 && superModules_[sm] <= 9 ) {
      int i=superModules_[sm]-1;
      int j=superModules_[sm];
      int x0 = (A0+A1)/2 + int(C0*cos(M_PI/2+3*2*M_PI/9-i*2*M_PI/9));
      int x1 = (A0+A1)/2 + int(C0*cos(M_PI/2+3*2*M_PI/9-j*2*M_PI/9));
      int x2 = (A0+A1)/2 + int(C1*cos(M_PI/2+3*2*M_PI/9-j*2*M_PI/9));
      int x3 = (A0+A1)/2 + int(C1*cos(M_PI/2+3*2*M_PI/9-i*2*M_PI/9));
      int y0 = (B0+B1)/2 - int(C0*sin(M_PI/2+3*2*M_PI/9-i*2*M_PI/9));
      int y1 = (B0+B1)/2 - int(C0*sin(M_PI/2+3*2*M_PI/9-j*2*M_PI/9));
      int y2 = (B0+B1)/2 - int(C1*sin(M_PI/2+3*2*M_PI/9-j*2*M_PI/9));
      int y3 = (B0+B1)/2 - int(C1*sin(M_PI/2+3*2*M_PI/9-i*2*M_PI/9));
      hf << "<area title=\"" << Numbers::sEE(superModules_[sm]).c_str()
         << "\" shape=\"poly\" href=\"" << refhtml[mapname]
         << "#" << Numbers::sEE(superModules_[sm]).c_str()
         << "\" coords=\"" << x0 << ", " << y0 << ", "
                           << x1 << ", " << y1 << ", "
                           << x2 << ", " << y2 << ", "
                           << x3 << ", " << y3 << "\">"
         << std::endl;
    }
  }
  hf << "</map>" << std::endl;

  hf << "<map name=\"" << mapname << "_1\">" << std::endl;
  for( unsigned int sm=0; sm<superModules_.size(); sm++ ) {
    if( superModules_[sm] >= 10 && superModules_[sm] <= 18 ) {
      int i=superModules_[sm]-9-1;
      int j=superModules_[sm]-9;
      int x0 = (A0+A1)/2 + int(C0*cos(M_PI/2-3*2*M_PI/9+i*2*M_PI/9));
      int x1 = (A0+A1)/2 + int(C0*cos(M_PI/2-3*2*M_PI/9+j*2*M_PI/9));
      int x2 = (A0+A1)/2 + int(C1*cos(M_PI/2-3*2*M_PI/9+j*2*M_PI/9));
      int x3 = (A0+A1)/2 + int(C1*cos(M_PI/2-3*2*M_PI/9+i*2*M_PI/9));
      int y0 = (B0+B1)/2 - int(C0*sin(M_PI/2-3*2*M_PI/9+i*2*M_PI/9));
      int y1 = (B0+B1)/2 - int(C0*sin(M_PI/2-3*2*M_PI/9+j*2*M_PI/9));
      int y2 = (B0+B1)/2 - int(C1*sin(M_PI/2-3*2*M_PI/9+j*2*M_PI/9));
      int y3 = (B0+B1)/2 - int(C1*sin(M_PI/2-3*2*M_PI/9+i*2*M_PI/9));
      hf << "<area title=\"" << Numbers::sEE(superModules_[sm]).c_str()
         << "\" shape=\"poly\" href=\"" << refhtml[mapname]
         << "#" << Numbers::sEE(superModules_[sm]).c_str()
         << "\" coords=\"" << x0 << ", " << y0 << ", "
                           << x1 << ", " << y1 << ", "
                           << x2 << ", " << y2 << ", "
                           << x3 << ", " << y3 << "\">"
         << std::endl;
    }
  }
  hf << "</map>" << std::endl;

}

