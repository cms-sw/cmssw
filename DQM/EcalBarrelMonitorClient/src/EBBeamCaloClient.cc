/*
 * \file EBBeamCaloClient.cc
 *
 * $Date: 2006/06/29 22:03:25 $
 * $Revision: 1.5 $
 * \author G. Della Ricca
 * \author A. Ghezzi
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

#include "OnlineDB/EcalCondDB/interface/MonOccupancyDat.h"

#include <DQM/EcalBarrelMonitorClient/interface/EBBeamCaloClient.h>
#include <DQM/EcalBarrelMonitorClient/interface/EBMUtilsClient.h>

EBBeamCaloClient::EBBeamCaloClient(const ParameterSet& ps){

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

  ///////// task specific histos 
  for(int u=0;u<cryInArray_;u++){
    meBBCaloGains_[u] = 0;
    meBBCaloGainsMoving_[u] = 0;
  }
  meBBCaloEne1_ = 0;
  meBBCaloEne1Moving_ = 0;
  meBBCaloAllNeededCry_ = 0;
  meBBCaloE3x3_ = 0;
  meBBCaloE3x3Moving_ = 0;
  meBBCaloCryOnBeam_ = 0;
  meBBCaloMaxEneCry_ = 0;
  meEBBCaloReadCryErrors_ = 0;
  meEBBCaloE1vsCry_ = 0;
  meEBBCaloE3x3vsCry_ = 0;

  meEBBCaloRedGreen_ = 0;
}

EBBeamCaloClient::~EBBeamCaloClient(){

}

void EBBeamCaloClient::beginJob(MonitorUserInterface* mui){

  mui_ = mui;

  if ( verbose_ ) cout << "EBBeamCaloClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

  if ( enableQT_ ) {

  }

}

void EBBeamCaloClient::beginRun(void){

  if ( verbose_ ) cout << "EBBeamCaloClient: beginRun" << endl;

  jevt_ = 0;

  this->setup();

  this->subscribe();

}

void EBBeamCaloClient::endJob(void) {

  if ( verbose_ ) cout << "EBBeamCaloClient: endJob, ievt = " << ievt_ << endl;

  this->unsubscribe();

  this->cleanup();

}

void EBBeamCaloClient::endRun(void) {

  if ( verbose_ ) cout << "EBBeamCaloClient: endRun, jevt = " << jevt_ << endl;

  this->unsubscribe();

  this->cleanup();

}

void EBBeamCaloClient::setup(void) {

  Char_t histo[200];

  mui_->setCurrentFolder( "EcalBarrel/EBBeamCaloTask" );
  DaqMonitorBEInterface* bei = mui_->getBEInterface();
  if ( meEBBCaloRedGreen_) bei->removeElement( meEBBCaloRedGreen_->getName() );
  sprintf(histo, "EBBCT quality");
  meEBBCaloRedGreen_ = bei->book2D(histo, histo, 85, 0., 85., 20, 0., 20.);

  EBMUtilsClient::resetHisto( meEBBCaloRedGreen_ );

  for ( int ie = 1; ie <= 85; ie++ ) {
    for ( int ip = 1; ip <= 20; ip++ ) {

      meEBBCaloRedGreen_ ->setBinContent( ie, ip, 2. );

    }
  }

}

void EBBeamCaloClient::cleanup(void) {
  if ( cloneME_ ) {
    for(int u=0;u<cryInArray_;u++){
      if(meBBCaloGains_[u]) delete meBBCaloGains_[u];
      if(meBBCaloGainsMoving_[u])delete meBBCaloGainsMoving_[u];
    }
    if(meBBCaloEne1_) delete meBBCaloEne1_;
    if(meBBCaloEne1Moving_) delete meBBCaloEne1Moving_;
    if(meBBCaloAllNeededCry_) delete meBBCaloAllNeededCry_;
    if(meBBCaloE3x3_) delete meBBCaloE3x3_;
    if(meBBCaloE3x3Moving_) delete meBBCaloE3x3Moving_;
    if(meBBCaloCryOnBeam_) delete meBBCaloCryOnBeam_;
    if(meBBCaloMaxEneCry_) delete meBBCaloMaxEneCry_;
    if(meEBBCaloReadCryErrors_) delete meEBBCaloReadCryErrors_;
    if(meEBBCaloE1vsCry_) delete meEBBCaloE1vsCry_;
    if(meEBBCaloE3x3vsCry_) delete meEBBCaloE3x3vsCry_;
  }
  
  for(int u=0;u<cryInArray_;u++){
    meBBCaloGains_[u] = 0;
    meBBCaloGainsMoving_[u] = 0;
  }
  meBBCaloEne1_ = 0;
  meBBCaloEne1Moving_ = 0;
  meBBCaloAllNeededCry_ = 0;
  meBBCaloE3x3_ = 0;
  meBBCaloE3x3Moving_ = 0;
  meBBCaloCryOnBeam_ = 0;
  meBBCaloMaxEneCry_ = 0;
  meEBBCaloReadCryErrors_ = 0;
  meEBBCaloE1vsCry_ = 0;
  meEBBCaloE3x3vsCry_ = 0;

  mui_->setCurrentFolder( "EcalBarrel/EBBeamCaloTask" );
  DaqMonitorBEInterface* bei = mui_->getBEInterface();
  if ( meEBBCaloRedGreen_) bei->removeElement( meEBBCaloRedGreen_->getName() );
  meEBBCaloRedGreen_ = 0;

}


void EBBeamCaloClient::writeDb(EcalCondDBInterface* econn, MonRunIOV* moniov, int ism) {

  EcalLogicID ecid;
  MonOccupancyDat o;
  map<EcalLogicID, MonOccupancyDat> dataset;

  if ( econn ) {
    try {
      cout << "Inserting MonOccupancyDat ..." << flush;
      if ( dataset.size() != 0 ) econn->insertDataSet(&dataset, moniov);
      cout << "done." << endl;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

}

void EBBeamCaloClient::subscribe(void){

  if ( verbose_ ) cout << "EBBeamCaloClient: subscribe" << endl;

  Char_t histo[200];

  for (int i = 0; i < cryInArray_ ; i++){
    sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT pulse profile cry: %01d", i+1);
    mui_->subscribe(histo);
    sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT pulse profile in G12 cry: %01d", i+1);
    mui_->subscribe(histo);
    sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT found gains cry: %01d", i+1);
    mui_->subscribe(histo);
    sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT rec energy cry: %01d", i+1);
    mui_->subscribe(histo);
    sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT pulse profile moving table cry: %01d", i+1);
    mui_->subscribe(histo);
    sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT pulse profile in G12 moving table cry: %01d", i+1);
    mui_->subscribe(histo);
    sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT found gains moving table cry: %01d", i+1);
    mui_->subscribe(histo);
    sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT rec energy moving table cry: %01d", i+1);

  }
    
  //     for(int u=0; u< 1701;u++){
  //       sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EnergyHistos/EBBCT rec Ene sum 3x3 cry: %04d",u);
  //          mui_->subscribe(histo);
  //       sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EnergyHistos/EBBCT rec Energy1 cry: %04d",u);
  //          mui_->subscribe(histo);
  //     }
    
   
  sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT readout crystals");
  mui_->subscribe(histo);
  sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT readout crystals table moving");
  mui_->subscribe(histo);
  sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT all needed crystals readout");
  mui_->subscribe(histo);
  sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT number of readout crystals");
  mui_->subscribe(histo);
  sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT rec Ene sum 3x3");
  mui_->subscribe(histo);
  sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT rec Ene sum 3x3 table moving");
  mui_->subscribe(histo);
  sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT crystal on beam");
  mui_->subscribe(histo);
  sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT crystal with maximum rec energy");
  mui_->subscribe(histo);
  sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT table is moving");
  mui_->subscribe(histo);
  sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT crystals done");
  mui_->subscribe(histo);
  sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT crystal in beam vs event");
  mui_->subscribe(histo);
  sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT errors in the number of readout crystals");
  mui_->subscribe(histo);
  sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT average rec energy in the single cristal");
  mui_->subscribe(histo);
  sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT average rec energy in the 3x3 array");
  mui_->subscribe(histo);
  sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT energy deposition in the 3x3");
  mui_->subscribe(histo);

  if ( collateSources_ ) {

    if ( verbose_ ) cout << "EBBeamCaloClient: collate" << endl;

  }

}

void EBBeamCaloClient::subscribeNew(void){

  Char_t histo[200];
  
  for (int i = 0; i < cryInArray_ ; i++){
    sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT pulse profile cry: %01d", i+1);
    mui_->subscribe(histo);
    sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT pulse profile in G12 cry: %01d", i+1);
    mui_->subscribe(histo);
    sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT found gains cry: %01d", i+1);
    mui_->subscribe(histo);
    sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT rec energy cry: %01d", i+1);
    mui_->subscribe(histo);
    sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT pulse profile moving table cry: %01d", i+1);
    mui_->subscribe(histo);
    sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT pulse profile in G12 moving table cry: %01d", i+1);
    mui_->subscribe(histo);
    sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT found gains moving table cry: %01d", i+1);
    mui_->subscribe(histo);
    sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT rec energy moving table cry: %01d", i+1);

  }
    
  //     for(int u=0; u< 1701;u++){
  //       sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EnergyHistos/EBBCT rec Ene sum 3x3 cry: %04d",u);
  //          mui_->subscribe(histo);
  //       sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EnergyHistos/EBBCT rec Energy1 cry: %04d",u);
  //          mui_->subscribe(histo);
  //     }
    
   
  sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT readout crystals");
  mui_->subscribe(histo);
  sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT readout crystals table moving");
  mui_->subscribe(histo);
  sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT all needed crystals readout");
  mui_->subscribe(histo);
  sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT number of readout crystals");
  mui_->subscribe(histo);
  sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT rec Ene sum 3x3");
  mui_->subscribe(histo);
  sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT rec Ene sum 3x3 table moving");
  mui_->subscribe(histo);
  sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT crystal on beam");
  mui_->subscribe(histo);
  sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT crystal with maximum rec energy");
  mui_->subscribe(histo);
  sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT table is moving");
  mui_->subscribe(histo);
  sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT crystals done");
  mui_->subscribe(histo);
  sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT crystal in beam vs event");
  mui_->subscribe(histo);
  sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT errors in the number of readout crystals");
  mui_->subscribe(histo);
  sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT average rec energy in the single cristal");
  mui_->subscribe(histo);
  sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT average rec energy in the 3x3 array");
  mui_->subscribe(histo);
  sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT energy deposition in the 3x3");
  mui_->subscribe(histo);
}

void EBBeamCaloClient::unsubscribe(void){

  if ( verbose_ ) cout << "EBBeamCaloClient: unsubscribe" << endl;

  Char_t histo[200];

  for (int i = 0; i < cryInArray_ ; i++){
    sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT pulse profile cry: %01d", i+1);
    mui_->unsubscribe(histo);
    sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT pulse profile in G12 cry: %01d", i+1);
    mui_->unsubscribe(histo);
    sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT found gains cry: %01d", i+1);
    mui_->unsubscribe(histo);
    sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT rec energy cry: %01d", i+1);
    mui_->unsubscribe(histo);
    sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT pulse profile moving table cry: %01d", i+1);
    mui_->unsubscribe(histo);
    sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT pulse profile in G12 moving table cry: %01d", i+1);
    mui_->unsubscribe(histo);
    sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT found gains moving table cry: %01d", i+1);
    mui_->unsubscribe(histo);
    sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT rec energy moving table cry: %01d", i+1);

  }
    
  //     for(int u=0; u< 1701;u++){
  //       sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EnergyHistos/EBBCT rec Ene sum 3x3 cry: %04d",u);
  //          mui_->unsubscribe(histo);
  //       sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EnergyHistos/EBBCT rec Energy1 cry: %04d",u);
  //          mui_->unsubscribe(histo);
  //     }
    
   
  sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT readout crystals");
  mui_->unsubscribe(histo);
  sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT readout crystals table moving");
  mui_->unsubscribe(histo);
  sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT all needed crystals readout");
  mui_->unsubscribe(histo);
  sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT number of readout crystals");
  mui_->unsubscribe(histo);
  sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT rec Ene sum 3x3");
  mui_->unsubscribe(histo);
  sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT rec Ene sum 3x3 table moving");
  mui_->unsubscribe(histo);
  sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT crystal on beam");
  mui_->unsubscribe(histo);
  sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT crystal with maximum rec energy");
  mui_->unsubscribe(histo);
  sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT table is moving");
  mui_->unsubscribe(histo);
  sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT crystals done");
  mui_->unsubscribe(histo);
  sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT crystal in beam vs event");
  mui_->unsubscribe(histo);
  sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT errors in the number of readout crystals");
  mui_->unsubscribe(histo);
  sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT average rec energy in the single cristal");
  mui_->unsubscribe(histo);
  sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT average rec energy in the 3x3 array");
  mui_->unsubscribe(histo);
  sprintf(histo, "*/EcalBarrel/EBBeamCaloTask/EBBCT energy deposition in the 3x3");
  mui_->unsubscribe(histo);


  if ( collateSources_ ) {

    if ( verbose_ ) cout << "EBBeamCaloClient: uncollate" << endl;

  }

}

void EBBeamCaloClient::analyze(void){

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( verbose_ ) cout << "EBBeamCaloClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;
  }

}

void EBBeamCaloClient::htmlOutput(int run, string htmlDir, string htmlName){

  cout << "Preparing EBBeamCaloClient html output ..." << endl;

  ofstream htmlFile;

  htmlFile.open((htmlDir + htmlName).c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlFile << "<html>  " << endl;
  htmlFile << "<head>  " << endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlFile << " http-equiv=\"content-type\">  " << endl;
  htmlFile << "  <title>Monitor:BeamTask output</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlFile << "<body>  " << endl;
  htmlFile << "<br>  " << endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << run << "</span></h2>" << endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">BeamCalo</span></h2> " << endl;
  htmlFile << "<hr>" << endl;
//  htmlFile << "<table border=1><tr><td bgcolor=red>channel has problems in this task</td>" << endl;
//  htmlFile << "<td bgcolor=lime>channel has NO problems</td>" << endl;
//  htmlFile << "<td bgcolor=yellow>channel is missing</td></table>" << endl;
//  htmlFile << "<hr>" << endl;

  // Produce the plots to be shown as .png files from existing histograms

  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  htmlFile.close();

}

