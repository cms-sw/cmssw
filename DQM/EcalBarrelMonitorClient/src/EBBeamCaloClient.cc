/*
 * \file EBBeamCaloClient.cc
 *
 * $Date: 2006/06/30 10:33:28 $
 * $Revision: 1.6 $
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

  checkedCry_.reserve(86);
  // there should be not more than a eta row in an autoscan
  minEvtNum_ = 2000;//FIX ME, change in case of prescaling
  //FIX ME, this should be configurable and change with the beam energy
  aveEne1_    = 1500;  E1Th_   = 500;
  aveEne3x3_  = 2000;  E3x3Th_ = 500;
  RMSEne3x3_  = 150;

  ReadCryErrThr_ = 0.01;// 1%
  //FIX ME, this should follow the prescaling in the monitoring
  prescaling_ = 1;
  
  ///////// task specific histos 
  for(int u=0;u<cryInArray_;u++){
    hBGains_[u] = 0;
    hBGainsMoving_[u] = 0;
  }
  hBEne1_ = 0;
  hBEne1Moving_ = 0;
  hBAllNeededCry_ = 0;
  hBNumReadCry_ = 0;
  hBE3x3_ = 0;
  hBE3x3Moving_ = 0;
  hBCryOnBeam_ = 0;
  hBMaxEneCry_ = 0;
  hBReadCryErrors_ = 0;
  hBE1vsCry_ = 0;
  hBE3x3vsCry_ = 0;
  hBcryDone_ = 0;
  hBBeamCentered_ = 0;

  meEBBCaloRedGreen_ = 0;
  meEBBCaloRedGreenReadCry_ = 0;
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

  if ( meEBBCaloRedGreenReadCry_) bei->removeElement( meEBBCaloRedGreenReadCry_->getName() );
  sprintf(histo, "EBBCT read crystal errors");
  meEBBCaloRedGreenReadCry_ = bei->book2D(histo, histo, 1, 0., 1., 1, 0., 1.);
  EBMUtilsClient::resetHisto( meEBBCaloRedGreenReadCry_ );
  meEBBCaloRedGreenReadCry_ ->setBinContent( 1, 1, 2. );

}

void EBBeamCaloClient::cleanup(void) {
  if ( cloneME_ ) {
    for(int u=0;u<cryInArray_;u++){
      if(hBGains_[u]) delete hBGains_[u];
      if(hBGainsMoving_[u])delete hBGainsMoving_[u];
    }
    if(hBEne1_) delete hBEne1_;
    if(hBEne1Moving_) delete hBEne1Moving_;
    if(hBAllNeededCry_) delete hBAllNeededCry_;
    if(hBNumReadCry_) delete hBNumReadCry_;
    if(hBE3x3_) delete hBE3x3_;
    if(hBE3x3Moving_) delete hBE3x3Moving_;
    if(hBCryOnBeam_) delete hBCryOnBeam_;
    if(hBMaxEneCry_) delete hBMaxEneCry_;
    if(hBReadCryErrors_) delete hBReadCryErrors_;
    if(hBE1vsCry_) delete hBE1vsCry_;
    if(hBE3x3vsCry_) delete hBE3x3vsCry_;
    if(hBcryDone_) delete hBcryDone_;
    if(hBBeamCentered_) delete hBBeamCentered_;
  }
  
  for(int u=0;u<cryInArray_;u++){
    hBGains_[u] = 0;
    hBGainsMoving_[u] = 0;
  }
  hBEne1_ = 0;
  hBEne1Moving_ = 0;
  hBAllNeededCry_ = 0;
  hBNumReadCry_ = 0;
  hBE3x3_ = 0;
  hBE3x3Moving_ = 0;
  hBCryOnBeam_ = 0;
  hBMaxEneCry_ = 0;
  hBReadCryErrors_ = 0;
  hBE1vsCry_ = 0;
  hBE3x3vsCry_ = 0;
  hBcryDone_ = 0;
  hBBeamCentered_ = 0;

  mui_->setCurrentFolder( "EcalBarrel/EBBeamCaloTask" );
  DaqMonitorBEInterface* bei = mui_->getBEInterface();
  if ( meEBBCaloRedGreen_) bei->removeElement( meEBBCaloRedGreen_->getName() );
  meEBBCaloRedGreen_ = 0;
  if ( meEBBCaloRedGreenReadCry_) bei->removeElement( meEBBCaloRedGreenReadCry_->getName() );
  meEBBCaloRedGreenReadCry_ = 0;
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

  Char_t histo[200];
  
  MonitorElement* me = 0;

  // MonitorElement* meCD;
  if ( collateSources_ ) {;}
  else{ sprintf(histo, (prefixME_+"EcalBarrel/EBBeamCaloTask/EBBCT crystals done").c_str() ); }
  //meCD = mui_->get(histo);
  me = mui_->get(histo);
  hBcryDone_ = EBMUtilsClient::getHisto<TH1F*>( me, cloneME_, hBcryDone_ );

  //MonitorElement* meCryInBeam;
  if ( collateSources_ ) {;}
  else { sprintf(histo, (prefixME_+"EcalBarrel/EBBeamCaloTask/EBBCT crystal on beam").c_str() ); }
  //meCryInBeam = mui_->get(histo);
  me = mui_->get(histo);
  hBCryOnBeam_ = EBMUtilsClient::getHisto<TH2F*>( me, cloneME_, hBCryOnBeam_);

  //MonitorElement* allNeededCry;
  if ( collateSources_ ) {;}
  else {sprintf(histo, (prefixME_+"EcalBarrel/EBBeamCaloTask/EBBCT all needed crystals readout").c_str() ); }
  //allNeededCry= mui_->get(histo);
  me = mui_->get(histo);
  hBAllNeededCry_ = EBMUtilsClient::getHisto<TH1F*>( me, cloneME_, hBAllNeededCry_);
 
  if ( collateSources_ ) {;}
  else {sprintf(histo, (prefixME_+"EcalBarrel/EBBeamCaloTask/EBBCT number of readout crystals").c_str() ); }
  //allNeededCry= mui_->get(histo);
  me = mui_->get(histo);
  hBNumReadCry_ = EBMUtilsClient::getHisto<TH1F*>( me, cloneME_, hBNumReadCry_);

  //MonitorElement* RecEne3x3;
  if ( collateSources_ ) {;}
  else { sprintf(histo, (prefixME_+"EcalBarrel/EBBeamCaloTask/EBBCT rec Ene sum 3x3").c_str() ); }
  //RecEne3x3= mui_->get(histo);
  me = mui_->get(histo);
  hBE3x3_ = EBMUtilsClient::getHisto<TH1F*>( me, cloneME_, hBE3x3_);

  //MonitorElement* ErrRedCry;
  if ( collateSources_ ) {;}
  else { sprintf(histo, (prefixME_+"EcalBarrel/EBBeamCaloTask/EBBCT errors in the number of readout crystals").c_str() ); }
  //ErrRedCry = mui_->get(histo);
  me = mui_->get(histo);
  hBReadCryErrors_ = EBMUtilsClient::getHisto<TH1F*>( me, cloneME_, hBReadCryErrors_);
   
  //  MonitorElement* RecEne1;
  if ( collateSources_ ) {;}
  else { sprintf(histo, (prefixME_+"EcalBarrel/EBBeamCaloTask/EBBCT rec energy cry: 5").c_str() ); }
  //RecEne1= mui_->get(histo);
  me = mui_->get(histo);
  hBEne1_ = EBMUtilsClient::getHisto<TH1F*>( me, cloneME_, hBEne1_);

  if ( collateSources_ ) {;}
  else { sprintf(histo, (prefixME_+"EcalBarrel/EBBeamCaloTask/EBBCT crystal with maximum rec energy").c_str() ); }
  me = mui_->get(histo);
  hBMaxEneCry_ = EBMUtilsClient::getHisto<TH2F*>( me, cloneME_, hBMaxEneCry_);

  if ( collateSources_ ) {;}
  else { sprintf(histo, (prefixME_+"EcalBarrel/EBBeamCaloTask/EBBCT average rec energy in the 3x3 array").c_str() ); }
  me = mui_->get(histo);
  hBE3x3vsCry_ = EBMUtilsClient::getHisto<TH1F*>( me, cloneME_, hBE3x3vsCry_);

  if ( collateSources_ ) {;}
  else {  sprintf(histo, (prefixME_+"EcalBarrel/EBBeamCaloTask/EBBCT average rec energy in the single cristal").c_str() ); }
  me = mui_->get(histo);
  hBE1vsCry_ = EBMUtilsClient::getHisto<TH1F*>( me, cloneME_, hBE1vsCry_);

  if ( collateSources_ ) {;}
  else { sprintf(histo, (prefixME_+"EcalBarrel/EBBeamCaloTask/EBBCT energy deposition in the 3x3").c_str() ); }
  me = mui_->get(histo);
  hBBeamCentered_ = EBMUtilsClient::getHisto<TH2F*>( me, cloneME_, hBBeamCentered_);



  int DoneCry = 0;//if it stays 0 the run is not an autoscan
  if (hBcryDone_){
    for(int cry=1 ; cry<1701 ; cry ++){
      int step = (int) hBcryDone_->GetBinContent(cry);
      if( step>0 ){//this crystal has been scanned 
	DoneCry++;
	//activate check for this cristal int the step
      }
    }
  }
  if(DoneCry == 0){//this is probably not an auotscan
    float nEvt = 0;
    if(hBE3x3_){nEvt = hBE3x3_->GetEntries();}
    if(nEvt > 1*prescaling_ && hBE3x3_ && hBEne1_ && hBCryOnBeam_ && meEBBCaloRedGreen_){//check for mean and RMS
      bool RMS3x3  =  ( hBE3x3_->GetRMS() < RMSEne3x3_ );
      bool Mean3x3 =  ( (hBE3x3_->GetMean() - aveEne3x3_) < E3x3Th_);
      bool Mean1   =  ( (hBEne1_->GetMean() < aveEne1_) < E1Th_ );
      //fill the RedGreen histo
      int ieta=0,iphi=0;
      float found =0; //there should be just one bin filled but...
      for (int b_eta =1; b_eta<86; b_eta++){
	for (int b_phi =1; b_phi<21; b_phi++){
	  float bc = hBCryOnBeam_->GetBinContent(hBCryOnBeam_->GetBin(b_eta,b_phi));//FIX ME check if this is the correct binning 
	  if(bc > found){ found =bc; ieta = b_eta; iphi= b_phi;}
	}
      }
      if(ieta >0 && iphi >0 ){
	if(RMS3x3 && Mean3x3 && Mean1) {meEBBCaloRedGreen_->setBinContent(ieta,iphi,1.);}
	else {meEBBCaloRedGreen_->setBinContent(ieta,iphi,0.);}
      }
    }
    if(hBReadCryErrors_){
      float nErr = hBReadCryErrors_->GetBinContent(1);// for a non autoscan just the first bin should be filled
      if( nErr > nEvt*ReadCryErrThr_ ){ meEBBCaloRedGreenReadCry_->setBinContent(1,1,0.);}
      else { meEBBCaloRedGreenReadCry_->setBinContent(1,1,1.);}
    }
  }

  //   // was done using me instead of histos
  //   if(DoneCry == 0){//this is probably not an auotscan
  //     float nEvt = RecEne3x3->getEntries();
  //     if(nEvt > 1000*prescaling_){//check for mean and RMS
  //       bool RMS3x3  =  ( RecEne3x3->getRMS() < RMSEne3x3_ );
  //       bool Mean3x3 =  ( (RecEne3x3->getMean() - aveEne3x3_) < E3x3Th_);
  //       bool Mean1   =  ( (RecEne1->getMean() < aveEne1_) < E1Th_ );
  //       //fill the RedGreen histo
  //       int ieta=0,iphi=0;
  //       float found =0; //there should be just one bin filled but...
  //       for (int b_eta =1; b_eta<86; b_eta++){
  // 	for (int b_phi =1; b_phi<21; b_phi++){
  // 	  float bc = meCryInBeam->getBinContent(b_eta,b_phi);//FIX ME check if this is the correct binning 
  // 	  if(bc > found){ found =bc; ieta = b_eta; iphi= b_phi;}
  // 	}
  //       }
  //       if(ieta >0 && iphi >0 ){
  // 	if(RMS3x3 && Mean3x3 && Mean1) {meEBBCaloRedGreen_->setBinContent(ieta,iphi,1.);}
  // 	else {meEBBCaloRedGreen_->setBinContent(ieta,iphi,0.);}
  //       }
  //     }
  //     float nErr = ErrRedCry->getBinContent(1);// for a non autoscan just the first bin should be filled
  //     if( nErr > nEvt*ReadCryErrThr_ ){ meEBBCaloRedGreenReadCry_->setBinContent(1,1,0.);}
  //     else { meEBBCaloRedGreenReadCry_->setBinContent(1,1,1.);}
  //   }


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


  const int csize = 250;
  const double histMax = 1.e15;

  int pCol3[3] = { 2, 3, 5 };
  
  TH2C dummy( "dummy", "dummy for sm", 85, 0., 85., 20, 0., 20. );
  for ( int i = 0; i < 68; i++ ) {
    int a = 2 + ( i/4 ) * 5;
    int b = 2 + ( i%4 ) * 5;
    dummy.Fill( a, b, i+1 );
  }
  dummy.SetMarkerSize(2);
  
  //useful for both autoscan and non autoscan
  string RedGreenSMImg,RedGreenImg,numCryReadImg, cryReadErrImg;
  string cryOnBeamImg, cryMaxEneImg, ratioImg;
  // useful for non autoscan
  string ene1Img, ene3x3Img, EBBeamCentered;
  //useful for autoscan
  string TBmoving, cryDoneImg, E1vsCryImg, E3x3vsCryImg;  
  
  string meName,imgName1,imgName2,imgName3;
  TH2F* obj2f = 0;
  TH1F* obj1f = 0;
  
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
  htmlFile << "<tr align=\"center\">" << endl;
   ////////////////////////////////////////////////////////////////////////////////
  // quality red-green histo
  obj2f = EBMUtilsClient::getHisto<TH2F*>( meEBBCaloRedGreen_ );
  if ( obj2f ) {
    
    TCanvas* can = new TCanvas("can", "Temp", 2*csize, csize);
    meName = obj2f->GetName();
    for ( unsigned int i = 0; i < meName.size(); i++ ) {
      if ( meName.substr(i, 1) == " " )  {
	meName.replace(i, 1, "_");
      }
    }
    RedGreenSMImg = meName + ".png";
    imgName1 = htmlDir + RedGreenSMImg;
      
    can->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(3, pCol3);
    obj2f->GetXaxis()->SetNdivisions(17);
    obj2f->GetYaxis()->SetNdivisions(4);
    can->SetGridx();
    can->SetGridy();
    obj2f->SetMinimum(-0.00000001);
    obj2f->SetMaximum(2.0);
    obj2f->Draw("col");
    dummy.Draw("text,same");
    can->Update();
    can->SaveAs(imgName1.c_str());
    delete can;
  }
  if ( imgName1.size() != 0 )
    htmlFile << "<td><img src=\"" << RedGreenSMImg  << "\"></td>" << endl;
  else
    htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

  //red-green pad 
  obj2f = EBMUtilsClient::getHisto<TH2F*>( meEBBCaloRedGreenReadCry_ );
  if ( obj2f ) {
    
    TCanvas* can = new TCanvas("can", "Temp", csize, csize);
    meName = obj2f->GetName();
    
    for ( unsigned int i = 0; i < meName.size(); i++ ) {
      if ( meName.substr(i, 1) == " " )  {
	meName.replace(i, 1, "_");
      }
    }
    RedGreenImg = meName + ".png";
    imgName1 = htmlDir + RedGreenImg;
      
    can->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(3, pCol3);
    obj2f->GetXaxis()->SetNdivisions(0);
    obj2f->GetYaxis()->SetNdivisions(0);
    obj2f->SetMinimum(-0.00000001);
    obj2f->SetMaximum(2.0);
    obj2f->Draw("col");
    can->Update();
    can->SaveAs(imgName1.c_str());
    delete can;
  }
  if ( imgName1.size() != 0 )
    htmlFile << "<td><img src=\"" << RedGreenImg << "\"></td>" << endl;
  else
    htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

  htmlFile << "</tr>" << endl;

  htmlFile << "<tr align=\"center\">" << endl;
 ////////////////////////////////////////////////////////////////////////////////
  // number of readout crystals : numCryReadImg
  obj1f = hBNumReadCry_;
  if ( obj1f ) {
    TCanvas* can = new TCanvas("can", "Temp", csize, csize);
    meName = obj1f->GetName();
    
    for ( unsigned int i = 0; i < meName.size(); i++ ) {
      if ( meName.substr(i, 1) == " " )  {
	meName.replace(i, 1, "_");
      }
    }
    numCryReadImg = meName + ".png";
    imgName1 = htmlDir + numCryReadImg;
      
    can->cd();
    obj1f->SetStats(kTRUE);
    gStyle->SetOptStat(1110);
    obj1f->Draw();
    can->Update();
    can->SaveAs(imgName1.c_str());
    delete can;
  }
  if ( imgName1.size() != 0 )
    htmlFile << "<td><img src=\"" << numCryReadImg << "\"></td>" << endl;
  else
    htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

  // error: reading less than the 7x7: cryReadErrImg;
  obj1f = hBReadCryErrors_ ;
  if ( obj1f ) {
    TCanvas* can = new TCanvas("can", "Temp", csize, csize);
    meName = obj1f->GetName();
    
    for ( unsigned int i = 0; i < meName.size(); i++ ) {
      if ( meName.substr(i, 1) == " " )  {
	meName.replace(i, 1, "_");
      }
    }
    cryReadErrImg = meName + ".png";
    imgName1 = htmlDir + cryReadErrImg;
      
    can->cd();
    obj1f->SetStats(kTRUE);
    gStyle->SetOptStat(10);
    obj1f->Draw();
    can->Update();
    can->SaveAs(imgName1.c_str());
    delete can;
  }
  if ( imgName1.size() != 0 )
    htmlFile << "<td><img src=\"" << cryReadErrImg << "\"></td>" << endl;
  else
    htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

  
  htmlFile << "</tr>" << endl;

  htmlFile << "<tr align=\"center\">" << endl;
  ////////////////////////////////////////////////////////////////////////////////
  //  crystal on beam: cryOnBeamImg
  obj2f =  hBCryOnBeam_;
  if ( obj2f ) {
    
    TCanvas* can = new TCanvas("can", "Temp", 2*csize, csize);
    meName = obj2f->GetName();
    
    for ( unsigned int i = 0; i < meName.size(); i++ ) {
      if ( meName.substr(i, 1) == " " )  {
	meName.replace(i, 1, "_");
      }
    }
    cryOnBeamImg = meName + ".png";
    imgName1 = htmlDir + cryOnBeamImg;
      
    can->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(1);
    obj2f->GetXaxis()->SetNdivisions(17);
    obj2f->GetYaxis()->SetNdivisions(4);
    can->SetGridx();
    can->SetGridy();
    obj2f->SetMinimum(-0.00000001);
    //obj2f->SetMaximum(2.0);
    obj2f->Draw("colz");
    dummy.Draw("text,same");
    can->Update();
    can->SaveAs(imgName1.c_str());
    delete can;
  }
  if ( imgName1.size() != 0 )
    htmlFile << "<td><img src=\"" << cryOnBeamImg << "\"></td>" << endl;
  else
    htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

  //cryMaxEneImg

  obj2f = hBMaxEneCry_;
  if ( obj2f ) {
    
    TCanvas* can = new TCanvas("can", "Temp", 2*csize, csize);
    meName = obj2f->GetName();
    
    for ( unsigned int i = 0; i < meName.size(); i++ ) {
      if ( meName.substr(i, 1) == " " )  {
	meName.replace(i, 1, "_");
      }
    }
    cryMaxEneImg = meName + ".png";
    imgName1 = htmlDir + cryMaxEneImg;
      
    can->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(1);
    obj2f->GetXaxis()->SetNdivisions(17);
    obj2f->GetYaxis()->SetNdivisions(4);
    can->SetGridx();
    can->SetGridy();
    obj2f->SetMinimum(-0.00000001);
    obj2f->Draw("colz");
    dummy.Draw("text,same");
    can->Update();
    can->SaveAs(imgName1.c_str());
    delete can;
  }
  if ( imgName1.size() != 0 )
    htmlFile << "<td><img src=\"" << cryMaxEneImg << "\"></td>" << endl;
  else
    htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

  // ratioImg still to be done

  htmlFile << "</tr>" << endl;
  ////////////////////////////////////////////////////////////////////////////////////////////
  htmlFile << "<tr align=\"center\">" << endl;
  //ene1Img, ene3x3Img, EBBeamCentered;
  obj1f = hBEne1_;
  if ( obj1f ) {
    TCanvas* can = new TCanvas("can", "Temp", csize, csize);
    meName = obj1f->GetName();
    
    for ( unsigned int i = 0; i < meName.size(); i++ ) {
      if ( meName.substr(i, 1) == " " )  {
	meName.replace(i, 1, "_");
      }
    }
    ene1Img = meName + ".png";
    imgName1 = htmlDir + ene1Img;
      
    can->cd();
    obj1f->SetStats(kTRUE);
    gStyle->SetOptStat(1110);
    obj1f->Draw();
    can->Update();
    can->SaveAs(imgName1.c_str());
    delete can;
  }
  if ( imgName1.size() != 0 )
    htmlFile << "<td><img src=\"" << ene1Img << "\"></td>" << endl;
  else
    htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

  //ene3x3Img
  obj1f = hBE3x3_;
  if ( obj1f ) {
    TCanvas* can = new TCanvas("can", "Temp", csize, csize);
    meName = obj1f->GetName();
    
    for ( unsigned int i = 0; i < meName.size(); i++ ) {
      if ( meName.substr(i, 1) == " " )  {
	meName.replace(i, 1, "_");
      }
    }
    ene3x3Img = meName + ".png";
    imgName1 = htmlDir + ene3x3Img;
      
    can->cd();
    obj1f->SetStats(kTRUE);
    gStyle->SetOptStat(1110);
    obj1f->Draw();
    can->Update();
    can->SaveAs(imgName1.c_str());
    delete can;
  }
  if ( imgName1.size() != 0 )
    htmlFile << "<td><img src=\"" << ene3x3Img << "\"></td>" << endl;
  else
    htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

  //EBBeamCentered
  obj2f = hBBeamCentered_;
  if ( obj2f ) {
    
    TCanvas* can = new TCanvas("can", "Temp", csize, csize);
    meName = obj2f->GetName();
    
    for ( unsigned int i = 0; i < meName.size(); i++ ) {
      if ( meName.substr(i, 1) == " " )  {
	meName.replace(i, 1, "_");
      }
    }
    EBBeamCentered = meName + ".png";
    imgName1 = htmlDir + EBBeamCentered;
      
    can->cd();
    gStyle->SetOptStat(" ");
    obj2f->GetXaxis()->SetNdivisions(0);
    obj2f->GetYaxis()->SetNdivisions(0);
    obj2f->SetMinimum(-0.00000001);
    //obj2f->SetMaximum(2.0);
    obj2f->Draw("box");
    can->Update();
    can->SaveAs(imgName1.c_str());
    delete can;
  }
  if ( imgName1.size() != 0 )
    htmlFile << "<td><img src=\"" << EBBeamCentered << "\"></td>" << endl;
  else
    htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

  htmlFile << "</tr>" << endl;
  //////////////////////////////////////////////////////////
  htmlFile << "</table>" << endl;
  htmlFile << "<br>" << endl;


  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  htmlFile.close();

}

