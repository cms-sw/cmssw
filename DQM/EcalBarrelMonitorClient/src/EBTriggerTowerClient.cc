/*
 * \file EBTriggerTowerClient.cc
 *
 * $Date: 2007/11/14 11:19:45 $
 * $Revision: 1.65 $
 * \author G. Della Ricca
 * \author F. Cossutti
 *
*/

#include <memory>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <bitset>

#include "TStyle.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/UI/interface/MonitorUIRoot.h"

#include "OnlineDB/EcalCondDB/interface/RunTag.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"

#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"

#include "CondTools/Ecal/interface/EcalErrorDictionary.h"

#include "DQM/EcalCommon/interface/EcalErrorMask.h"
#include "DQM/EcalCommon/interface/UtilsClient.h"
#include "DQM/EcalCommon/interface/LogicID.h"
#include "DQM/EcalCommon/interface/Numbers.h"

#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

#include <DQM/EcalBarrelMonitorClient/interface/EBTriggerTowerClient.h>

using namespace cms;
using namespace edm;
using namespace std;

EBTriggerTowerClient::EBTriggerTowerClient(const ParameterSet& ps){

  // cloneME switch
  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);

  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

  // MonitorDaemon switch
  enableMonitorDaemon_ = ps.getUntrackedParameter<bool>("enableMonitorDaemon", true);

  // prefix to ME paths
  prefixME_ = ps.getUntrackedParameter<string>("prefixME", "");

  // vector of selected Super Modules (Defaults to all 36).
  superModules_.reserve(36);
  for ( unsigned int i = 1; i <= 36; i++ ) superModules_.push_back(i);
  superModules_ = ps.getUntrackedParameter<vector<int> >("superModules", superModules_);

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    h01_[ism-1] = 0;
    h02_[ism-1] = 0;
    i01_[ism-1] = 0;
    i02_[ism-1] = 0;
    j01_[ism-1] = 0;
    j02_[ism-1] = 0;

    l01_[ism-1] = 0;
    m01_[ism-1] = 0;
    n01_[ism-1] = 0;

    meh01_[ism-1] = 0;
    meh02_[ism-1] = 0;
    mei01_[ism-1] = 0;
    mei02_[ism-1] = 0;
    mej01_[ism-1] = 0;
    mej02_[ism-1] = 0;

    mel01_[ism-1] = 0;
    mem01_[ism-1] = 0;
    men01_[ism-1] = 0;

//     for (int j=0; j<68; j++) {
//
//       k01_[ism-1][j] = 0;
//       k02_[ism-1][j] = 0;
//
//       mek01_[ism-1][j] = 0;
//       mek02_[ism-1][j] = 0;
//
//     }

    me_h01_[ism-1] = 0;
    me_h02_[ism-1] = 0;
    for (int j=0; j<2; j++) {
      me_i01_[ism-1][j] = 0;
      me_i02_[ism-1][j] = 0;
    }
    for (int j=0; j<6; j++) {
      me_j01_[ism-1][j] = 0;
      me_j02_[ism-1][j] = 0;
    }

  }

}

EBTriggerTowerClient::~EBTriggerTowerClient(){

}

void EBTriggerTowerClient::beginJob(MonitorUserInterface* mui){

  mui_ = mui;
  dbe_ = mui->getBEInterface();

  if ( verbose_ ) cout << "EBTriggerTowerClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EBTriggerTowerClient::beginRun(void) {

  if ( verbose_ ) cout << "EBTriggerTowerClient: beginRun" << endl;

  jevt_ = 0;

  this->setup();

  this->subscribe();

}

void EBTriggerTowerClient::endJob(void) {

  if ( verbose_ ) cout << "EBTriggerTowerClient: endJob, ievt = " << ievt_ << endl;

  this->unsubscribe();

  this->cleanup();

}

void EBTriggerTowerClient::endRun(void) {

  if ( verbose_ ) cout << "EBTriggerTowerClient: endRun, jevt = " << jevt_ << endl;

  this->unsubscribe();

  this->cleanup();

}

void EBTriggerTowerClient::setup(void) {

  Char_t histo[200];

  dbe_->setCurrentFolder( "EcalBarrel/EBTriggerTowerClient" );

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( me_h01_[ism-1] ) dbe_->removeElement( me_h01_[ism-1]->getName() );
    sprintf(histo, "EBTTT Et map Real Digis %s", Numbers::sEB(ism).c_str());
    me_h01_[ism-1] = dbe_->bookProfile2D(histo, histo, 17, 0., 17., 4, 0., 4., 256, 0., 256., "s");
    me_h01_[ism-1]->setAxisTitle("jeta", 1);
    me_h01_[ism-1]->setAxisTitle("jphi", 2);
    if ( me_h02_[ism-1] ) dbe_->removeElement( me_h02_[ism-1]->getName() );
    sprintf(histo, "EBTTT Et map Emulated Digis %s", Numbers::sEB(ism).c_str());
    me_h02_[ism-1] = dbe_->bookProfile2D(histo, histo, 17, 0., 17., 4, 0., 4., 256, 0., 256., "s");
    me_h02_[ism-1]->setAxisTitle("jeta", 1);
    me_h02_[ism-1]->setAxisTitle("jphi", 2);
    for (int j=0; j<2; j++) {
      if ( me_i01_[ism-1][j] ) dbe_->removeElement( me_i01_[ism-1][j]->getName() );
      sprintf(histo, "EBTTT FineGrainVeto Real Digis Flag %d %s", j, Numbers::sEB(ism).c_str());
      me_i01_[ism-1][j] = dbe_->book2D(histo, histo, 17, 0., 17., 4, 0., 4.);
      me_i01_[ism-1][j]->setAxisTitle("jeta", 1);
      me_i01_[ism-1][j]->setAxisTitle("jphi", 2);
      if ( me_i02_[ism-1][j] ) dbe_->removeElement( me_i02_[ism-1][j]->getName() );
      sprintf(histo, "EBTTT FineGrainVeto Emulated Digis Flag %d %s", j, Numbers::sEB(ism).c_str());
      me_i02_[ism-1][j] = dbe_->book2D(histo, histo, 17, 0., 17., 4, 0., 4.);
      me_i02_[ism-1][j]->setAxisTitle("jeta", 1);
      me_i02_[ism-1][j]->setAxisTitle("jphi", 2);
    }
    for (int j=0; j<6; j++) {
      string bits;
      if ( j == 0 ) bits = "Bit 000";
      if ( j == 1 ) bits = "Bit 001";
      if ( j == 2 ) bits = "Bit 011";
      if ( j == 3 ) bits = "Bit 100";
      if ( j == 4 ) bits = "Bit 101";
      if ( j == 5 ) bits = "Bits 110+111";
      if ( me_j01_[ism-1][j] ) dbe_->removeElement( me_j01_[ism-1][j]->getName() );
      sprintf(histo, "EBTTT Flags Real Digis %s %s", bits.c_str(), Numbers::sEB(ism).c_str());
      me_j01_[ism-1][j] = dbe_->book2D(histo, histo, 17, 0., 17., 4, 0., 4.);
      me_j01_[ism-1][j]->setAxisTitle("jeta", 1);
      me_j01_[ism-1][j]->setAxisTitle("jphi", 2);
      if ( me_j02_[ism-1][j] ) dbe_->removeElement( me_j02_[ism-1][j]->getName() );
      sprintf(histo, "EBTTT Flags Emulated Digis %s %s", bits.c_str(), Numbers::sEB(ism).c_str());
      me_j02_[ism-1][j] = dbe_->book2D(histo, histo, 17, 0., 17., 4, 0., 4.);
      me_j02_[ism-1][j]->setAxisTitle("jeta", 1);
      me_j02_[ism-1][j]->setAxisTitle("jphi", 2);
    }

  }

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    me_h01_[ism-1]->Reset();
    me_h02_[ism-1]->Reset();
    for (int j=0; j<2; j++) {
      me_i01_[ism-1][j]->Reset();
      me_i02_[ism-1][j]->Reset();
    } 
    for (int j=0; j<6; j++) {
      me_j01_[ism-1][j]->Reset();
      me_j02_[ism-1][j]->Reset();
    }

  }

}

void EBTriggerTowerClient::cleanup(void) {

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( cloneME_ ) {
      if ( h01_[ism-1] ) delete h01_[ism-1];
      if ( h02_[ism-1] ) delete h02_[ism-1];
      if ( i01_[ism-1] ) delete i01_[ism-1];
      if ( i02_[ism-1] ) delete i02_[ism-1];
      if ( j01_[ism-1] ) delete j01_[ism-1];
      if ( j02_[ism-1] ) delete j02_[ism-1];
      if ( l01_[ism-1] ) delete l01_[ism-1];
      if ( m01_[ism-1] ) delete m01_[ism-1];
      if ( n01_[ism-1] ) delete n01_[ism-1];
    }

    h01_[ism-1] = 0;
    h02_[ism-1] = 0;
    i01_[ism-1] = 0;
    i02_[ism-1] = 0;
    j01_[ism-1] = 0;
    j02_[ism-1] = 0;

    l01_[ism-1] = 0;
    m01_[ism-1] = 0;
    n01_[ism-1] = 0;

    meh01_[ism-1] = 0;
    meh02_[ism-1] = 0;
    mei01_[ism-1] = 0;
    mei02_[ism-1] = 0;
    mej01_[ism-1] = 0;
    mej02_[ism-1] = 0;

    mel01_[ism-1] = 0;
    mem01_[ism-1] = 0;
    men01_[ism-1] = 0;

//     for (int j=0; j<68; j++) {
//
//       if ( cloneME_ ) {
//         if ( k01_[ism-1][j] ) delete k01_[ism-1][j];
//         if ( k02_[ism-1][j] ) delete k02_[ism-1][j];
//       }
//
//       k01_[ism-1][j] = 0;
//       k02_[ism-1][j] = 0;
//
//       mek01_[ism-1][j] = 0;
//       mek02_[ism-1][j] = 0;
//
//     }

  }

  dbe_->setCurrentFolder( "EcalBarrel/EBTriggerTowerClient" );

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( me_h01_[ism-1] ) dbe_->removeElement( me_h01_[ism-1]->getName() );
    me_h01_[ism-1] = 0; 
    if ( me_h02_[ism-1] ) dbe_->removeElement( me_h02_[ism-1]->getName() );
    me_h02_[ism-1] = 0; 
    for (int j=0; j<2; j++) {
      if ( me_i01_[ism-1][j] ) dbe_->removeElement( me_i01_[ism-1][j]->getName() );
      me_i01_[ism-1][j] = 0;
      if ( me_i02_[ism-1][j] ) dbe_->removeElement( me_i02_[ism-1][j]->getName() );
      me_i02_[ism-1][j] = 0;
    }
    for (int j=0; j<6; j++) {
      if ( me_j01_[ism-1][j] ) dbe_->removeElement( me_j01_[ism-1][j]->getName() );
      me_j01_[ism-1][j] = 0;
      if ( me_j02_[ism-1][j] ) dbe_->removeElement( me_j02_[ism-1][j]->getName() );
      me_j02_[ism-1][j] = 0;
    }

  }

}

bool EBTriggerTowerClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov) {

  bool status = true;

  return status;

}

void EBTriggerTowerClient::subscribe(void){

  if ( verbose_ ) cout << "EBTriggerTowerClient: subscribe" << endl;

  subscribe( "Real Digis",
             "EBTriggerTowerTask", false);

  subscribe( "Emulated Digis",
             "EBTriggerTowerTask/Emulated", true);

}

void EBTriggerTowerClient::subscribe( const char* nameext,
                                      const char* folder,
                                      bool emulated ) {

  Char_t histo[200];

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    unsigned int ism = superModules_[i];

    sprintf(histo, "*/EcalBarrel/%s/EBTTT Et map %s %s", folder, nameext, Numbers::sEB(ism).c_str());
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/%s/EBTTT FineGrainVeto %s %s", folder, nameext, Numbers::sEB(ism).c_str());
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/%s/EBTTT Flags %s %s", folder, nameext, Numbers::sEB(ism).c_str());
    mui_->subscribe(histo, ism);

    if(!emulated) {
      sprintf(histo, "*/EcalBarrel/%s/EBTTT EmulError %s", folder, Numbers::sEB(ism).c_str());
      mui_->subscribe(histo, ism);
      sprintf(histo, "*/EcalBarrel/%s/EBTTT EmulFlagError %s", folder, Numbers::sEB(ism).c_str());
      mui_->subscribe(histo, ism);
      sprintf(histo, "*/EcalBarrel/%s/EBTTT EmulFineGrainVetoError %s", folder, Numbers::sEB(ism).c_str());
      mui_->subscribe(histo, ism);
    }

//     for (int j=0; j<68; j++) {
//       sprintf(histo, "*/EcalBarrel/EBTriggerTowerTask/EnergyMaps/EBTTT Et R %s TT%02d", ism, j+1);
//       mui_->subscribe(histo, ism);
//       sprintf(histo, "*/EcalBarrel/EBTriggerTowerTask/EnergyMaps/EBTTT Et T %s TT%02d", ism, j+1);
//       mui_->subscribe(histo, ism);
//     }

  }

}

void EBTriggerTowerClient::subscribeNew(void){

  subscribeNew( "Real Digis",
                "EBTriggerTowerTask", false );

  subscribeNew( "Emulated Digis",
                "EBTriggerTowerTask/Emulated", true );

}

void EBTriggerTowerClient::subscribeNew( const char* nameext,
                                         const char* folder,
                                         bool emulated ) {
  Char_t histo[200];

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    unsigned int ism = superModules_[i];

    sprintf(histo, "*/EcalBarrel/%s/EBTTT Et map %s %s", folder, nameext, Numbers::sEB(ism).c_str());
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalBarrel/%s/EBTTT FineGrainVeto %s %s", folder, nameext, Numbers::sEB(ism).c_str());
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalBarrel/%s/EBTTT Flags %s %s", folder, nameext, Numbers::sEB(ism).c_str());
    mui_->subscribeNew(histo, ism);

    if(!emulated) {
      sprintf(histo, "*/EcalBarrel/%s/EBTTT EmulError %s", folder, Numbers::sEB(ism).c_str());
      mui_->subscribeNew(histo, ism);
      sprintf(histo, "*/EcalBarrel/%s/EBTTT EmulFlagError %s", folder, Numbers::sEB(ism).c_str());
      mui_->subscribeNew(histo, ism);
      sprintf(histo, "*/EcalBarrel/%s/EBTTT EmulFineGrainVetoError %s", folder, Numbers::sEB(ism).c_str());
      mui_->subscribeNew(histo, ism);
    }

//     for (int j=0; j<68; j++) {
//       sprintf(histo, "*/EcalBarrel/EBTriggerTowerTask/EnergyMaps/EBTTT Et T %s TT%02d", ism, j+1);
//       mui_->subscribeNew(histo, ism);
//       sprintf(histo, "*/EcalBarrel/EBTriggerTowerTask/EnergyMaps/EBTTT Et R %s TT%02d", ism, j+1);
//       mui_->subscribeNew(histo, ism);
//     }

  }

}

void EBTriggerTowerClient::unsubscribe(void){

  if ( verbose_ ) cout << "EBTriggerTowerClient: unsubscribe" << endl;

  unsubscribe( "Real Digis",
               "EBTriggerTowerTask", false );

  unsubscribe( "Emulated Digis",
               "EBTriggerTowerTask/Emulated", true);

}

void EBTriggerTowerClient::unsubscribe( const char* nameext,
                                        const char* folder,
                                        bool emulated ) {

  Char_t histo[200];

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    unsigned int ism = superModules_[i];

    sprintf(histo, "*/EcalBarrel/%s/EBTTT Et map %s %s", folder, nameext, Numbers::sEB(ism).c_str());
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/%s/EBTTT FineGrainVeto %s %s", folder, nameext, Numbers::sEB(ism).c_str());
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/%s/EBTTT Flags %s %s", folder, nameext, Numbers::sEB(ism).c_str());
    mui_->unsubscribe(histo, ism);

    if(!emulated) {
      sprintf(histo, "*/EcalBarrel/%s/EBTTT EmulError %s", folder, Numbers::sEB(ism).c_str());
      mui_->unsubscribe(histo, ism);
      sprintf(histo, "*/EcalBarrel/%s/EBTTT EmulFlagError %s", folder, Numbers::sEB(ism).c_str());
      mui_->unsubscribe(histo, ism);
      sprintf(histo, "*/EcalBarrel/%s/EBTTT EmulFineGrainVetoError %s", folder, Numbers::sEB(ism).c_str());
      mui_->unsubscribe(histo, ism);
    }

//     for (int j=0; j<68; j++) {
//       sprintf(histo, "*/EcalBarrel/EBTriggerTowerTask/EnergyMaps/EBTTT Et T %s TT%02d", ism, j+1);
//       mui_->subscribe(histo, ism);
//       sprintf(histo, "*/EcalBarrel/EBTriggerTowerTask/EnergyMaps/EBTTT Et R %s TT%02d", ism, j+1);
//       mui_->subscribe(histo, ism);
//     }

  }

}

void EBTriggerTowerClient::softReset(void){

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( meh01_[ism-1] ) dbe_->softReset(meh01_[ism-1]);
    if ( meh02_[ism-1] ) dbe_->softReset(meh02_[ism-1]);
    if ( mei01_[ism-1] ) dbe_->softReset(mei01_[ism-1]);
    if ( mei02_[ism-1] ) dbe_->softReset(mei02_[ism-1]);
    if ( mej01_[ism-1] ) dbe_->softReset(mej01_[ism-1]);
    if ( mej02_[ism-1] ) dbe_->softReset(mej02_[ism-1]);

    if ( mel01_[ism-1] ) dbe_->softReset(mel01_[ism-1]);
    if ( mem01_[ism-1] ) dbe_->softReset(mem01_[ism-1]);
    if ( men01_[ism-1] ) dbe_->softReset(men01_[ism-1]);

//     for (int j=0; j<68; j++) {
//
//       if ( mek01_[ism-1][j] ) dbe_->softReset(mek01_[ism-1][j]);
//       if ( mek02_[ism-1][j] ) dbe_->softReset(mek02_[ism-1][j]);
//
//     }

  }

}

void EBTriggerTowerClient::analyze(void){

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( verbose_ ) cout << "EBTriggerTowerClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;
  }

  analyze("Real Digis",
          "EBTriggerTowerTask", false );

  analyze("Emulated Digis",
          "EBTriggerTowerTask/Emulated", true );

}

void EBTriggerTowerClient::analyze(const char* nameext,
                                   const char* folder,
                                   bool emulated) {
  Char_t histo[200];

  MonitorElement* me;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    sprintf(histo, (prefixME_+"EcalBarrel/%s/EBTTT Et map %s %s").c_str(), folder, nameext, Numbers::sEB(ism).c_str());
    me = dbe_->get(histo);
    if(!emulated) {
      h01_[ism-1] = UtilsClient::getHisto<TH3F*>( me, cloneME_, h01_[ism-1] );
      if(h01_[ism-1]) h01_[ism-1]->SetEntries(1.+h01_[ism-1]->GetEntries());
      meh01_[ism-1] = me;
    }
    else {
      h02_[ism-1] = UtilsClient::getHisto<TH3F*>( me, cloneME_, h02_[ism-1] );
      if(h02_[ism-1]) h02_[ism-1]->SetEntries(1.+h02_[ism-1]->GetEntries());
      meh02_[ism-1] = me;
    }

    sprintf(histo, (prefixME_+"EcalBarrel/%s/EBTTT FineGrainVeto %s %s").c_str(), folder, nameext, Numbers::sEB(ism).c_str());
    me = dbe_->get(histo);
    if(!emulated) {
      i01_[ism-1] = UtilsClient::getHisto<TH3F*>( me, cloneME_, i01_[ism-1] );
      if(i01_[ism-1]) i01_[ism-1]->SetEntries(1.+i01_[ism-1]->GetEntries());
      mei01_[ism-1] = me;
    }
    else {
      i02_[ism-1] = UtilsClient::getHisto<TH3F*>( me, cloneME_, i02_[ism-1] );
      if(i02_[ism-1]) i02_[ism-1]->SetEntries(1.+i02_[ism-1]->GetEntries());
      mei02_[ism-1] = me;
    }

    sprintf(histo, (prefixME_+"EcalBarrel/%s/EBTTT Flags %s %s").c_str(), folder, nameext, Numbers::sEB(ism).c_str());
    me = dbe_->get(histo);
    if(!emulated) {
      j01_[ism-1] = UtilsClient::getHisto<TH3F*>( me, cloneME_, j01_[ism-1] );
      if(j01_[ism-1]) j01_[ism-1]->SetEntries(1.+j01_[ism-1]->GetEntries());
      mej01_[ism-1] = me;
    }
    else {
      j02_[ism-1] = UtilsClient::getHisto<TH3F*>( me, cloneME_, j02_[ism-1] );
      if(j02_[ism-1]) j02_[ism-1]->SetEntries(1.+j02_[ism-1]->GetEntries());
      mej02_[ism-1] = me;
    }

    if(!emulated) {
      sprintf(histo, (prefixME_+"EcalBarrel/%s/EBTTT EmulError %s").c_str(), folder, Numbers::sEB(ism).c_str());
      me = dbe_->get(histo);
      l01_[ism-1] = UtilsClient::getHisto<TH2F*>( me, cloneME_, l01_[ism-1] );
      if(l01_[ism-1]) l01_[ism-1]->SetEntries(1.+l01_[ism-1]->GetEntries());
      mel01_[ism-1] = me;

      sprintf(histo, (prefixME_+"EcalBarrel/%s/EBTTT EmulFlagError %s").c_str(), folder, Numbers::sEB(ism).c_str());
      me = dbe_->get(histo);
      m01_[ism-1] = UtilsClient::getHisto<TH3F*>( me, cloneME_, m01_[ism-1] );
      if(m01_[ism-1]) m01_[ism-1]->SetEntries(1.+m01_[ism-1]->GetEntries());
      mem01_[ism-1] = me;

      sprintf(histo, (prefixME_+"EcalBarrel/%s/EBTTT EmulFineGrainVetoError %s").c_str(), folder, Numbers::sEB(ism).c_str());
      me = dbe_->get(histo);
      n01_[ism-1] = UtilsClient::getHisto<TH3F*>( me, cloneME_, n01_[ism-1] );
      if(n01_[ism-1]) n01_[ism-1]->SetEntries(1.+n01_[ism-1]->GetEntries());
      men01_[ism-1] = me;

    }

//     for (int j=0; j<68; j++) {
//
//       sprintf(histo, (prefixME_+"EcalBarrel/EBTriggerTowerTask/EnergyMaps/EBTTT Et T %s TT%02d").c_str(), ism, j+1);
//       me = dbe_->get(histo);
//       k01_[ism-1][j] = UtilsClient::getHisto<TH1F*>( me, cloneME_, k01_[ism-1][j] );
//       mek01_[ism-1][j] = me;
//
//       sprintf(histo, (prefixME_+"EcalBarrel/EBTriggerTowerTask/EnergyMaps/EBTTT Et R %s TT%02d").c_str(), ism, j+1);
//       me = dbe_->get(histo);
//       k02_[ism-1][j] = UtilsClient::getHisto<TH1F*>( me, cloneME_, k02_[ism-1][j] );
//       mek02_[ism-1][j] = me;
//
//     }

    me_h01_[ism-1]->Reset();
    me_h02_[ism-1]->Reset();
    for (int j=0; j<2; j++) {
      me_i01_[ism-1][j]->Reset();
      me_i02_[ism-1][j]->Reset();
    } 
    for (int j=0; j<6; j++) {
      me_j01_[ism-1][j]->Reset();
      me_j02_[ism-1][j]->Reset();
    }

    for (int i1 = 1; i1 <= 17; i1++) {
      for (int i2 = 1; i2 <= 4; i2++) {

        for (int i3 = 1; i3 <= 256; i3++) {
          if ( h01_[ism-1] ) me_h01_[ism-1]->Fill(i1, i2, i3, h01_[ism-1]->GetBinContent(i1, i2));
          if ( h02_[ism-1] ) me_h02_[ism-1]->Fill(i1, i2, i3, h02_[ism-1]->GetBinContent(i1, i2));
        }
        for (int j=0; j<2; j++) {
          if ( i01_[ism-1] ) me_i01_[ism-1][j]->Fill(i1, i2, i01_[ism-1]->GetBinContent(i1, i2, j+1));
          if ( i02_[ism-1] ) me_i02_[ism-1][j]->Fill(i1, i2, i02_[ism-1]->GetBinContent(i1, i2, j+1));
        }
        for (int j=0; j<6; j++) {
          if ( j == 0 ) {
            if ( j01_[ism-1] ) me_j01_[ism-1][j]->Fill(i1, i2, j01_[ism-1]->GetBinContent(i1, i2, j+1));
            if ( j02_[ism-1] ) me_j02_[ism-1][j]->Fill(i1, i2, j02_[ism-1]->GetBinContent(i1, i2, j+1));
          }
          if ( j == 1 ) {
            if ( j01_[ism-1] ) me_j01_[ism-1][j]->Fill(i1, i2, j01_[ism-1]->GetBinContent(i1, i2, j+1));
            if ( j02_[ism-1] ) me_j02_[ism-1][j]->Fill(i1, i2, j02_[ism-1]->GetBinContent(i1, i2, j+1));
          }
          if ( j == 2 ) {
            if ( j01_[ism-1] ) me_j01_[ism-1][j]->Fill(i1, i2, j01_[ism-1]->GetBinContent(i1, i2, j+2));
            if ( j02_[ism-1] ) me_j02_[ism-1][j]->Fill(i1, i2, j02_[ism-1]->GetBinContent(i1, i2, j+2));
          }
          if ( j == 3 ) {
            if ( j01_[ism-1] ) me_j01_[ism-1][j]->Fill(i1, i2, j01_[ism-1]->GetBinContent(i1, i2, j+2));
            if ( j02_[ism-1] ) me_j02_[ism-1][j]->Fill(i1, i2, j02_[ism-1]->GetBinContent(i1, i2, j+2));
          }
          if ( j == 4 ) {
            if ( j01_[ism-1] ) me_j01_[ism-1][j]->Fill(i1, i2, j01_[ism-1]->GetBinContent(i1, i2, j+2));
            if ( j02_[ism-1] ) me_j02_[ism-1][j]->Fill(i1, i2, j02_[ism-1]->GetBinContent(i1, i2, j+2));
          }
          if ( j == 5 ) {
            if ( j01_[ism-1] ) me_j01_[ism-1][j]->Fill(i1, i2, j01_[ism-1]->GetBinContent(i1, i2, j+2));
            if ( j01_[ism-1] ) me_j01_[ism-1][j]->Fill(i1, i2, j01_[ism-1]->GetBinContent(i1, i2, j+3));
            if ( j02_[ism-1] ) me_j02_[ism-1][j]->Fill(i1, i2, j02_[ism-1]->GetBinContent(i1, i2, j+2));
            if ( j02_[ism-1] ) me_j02_[ism-1][j]->Fill(i1, i2, j02_[ism-1]->GetBinContent(i1, i2, j+3));
          }
        }

      }
    }

  }

}

void EBTriggerTowerClient::htmlOutput(int run, string htmlDir, string htmlName){

  cout << "Preparing EBTriggerTowerClient html output ..." << std::endl;

  std::ofstream htmlFile[37];

  htmlFile[0].open((htmlDir + htmlName).c_str());

  // html page header
  htmlFile[0] << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << std::endl;
  htmlFile[0] << "<html>  " << std::endl;
  htmlFile[0] << "<head>  " << std::endl;
  htmlFile[0] << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << std::endl;
  htmlFile[0] << " http-equiv=\"content-type\">  " << std::endl;
  htmlFile[0] << "  <title>Monitor:TriggerTowerTask output</title> " << std::endl;
  htmlFile[0] << "</head>  " << std::endl;
  htmlFile[0] << "<style type=\"text/css\"> td { font-weight: bold } </style>" << std::endl;
  htmlFile[0] << "<body>  " << std::endl;
  //htmlFile[0] << "<br>  " << std::endl;
  htmlFile[0] << "<a name=""top""></a>" << endl;
  htmlFile[0] << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << std::endl;
  htmlFile[0] << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << std::endl;
  htmlFile[0] << " style=\"color: rgb(0, 0, 153);\">" << run << "</span></h2>" << std::endl;
  htmlFile[0] << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << std::endl;
  htmlFile[0] << " style=\"color: rgb(0, 0, 153);\">TRIGGER TOWER</span></h2> " << std::endl;
  htmlFile[0] << "<br>" << std::endl;
  //htmlFile[0] << "<table border=1><tr><td bgcolor=red>channel has problems in this task</td>" << std::endl;
  //htmlFile[0] << "<td bgcolor=lime>channel has NO problems</td>" << std::endl;
  //htmlFile[0] << "<td bgcolor=yellow>channel is missing</td></table>" << std::endl;
  htmlFile[0] << "<hr>" << std::endl;
  htmlFile[0] << "<table border=1>" << std::endl;
  for ( unsigned int i=0; i<superModules_.size(); i ++ ) {
    htmlFile[0] << "<td bgcolor=white><a href=""#"
                << Numbers::sEB(superModules_[i]).c_str() << ">"
                << setfill( '0' ) << setw(2) << superModules_[i] << "</a></td>";
  }
  htmlFile[0] << std::endl << "</table>" << std::endl;

  // Produce the plots to be shown as .png files from existing histograms

  const int csize = 250;

  //const double histMax = 1.e15;

  int pCol4[10];
  for ( int i = 0; i < 10; i++ ) pCol4[i] = 401+i;

  TH2C dummy( "dummy", "dummy for sm", 17, 0., 17., 4, 0., 4. );
  for ( int i = 0; i < 68; i++ ) {
    dummy.Fill( i/4, i%4, i+1 );
  }
  dummy.SetMarkerSize(2);
  dummy.SetMinimum(0.1);

  string imgName[3], meName[3], imgMeName[3];

  TCanvas* cMe1 = new TCanvas("cMe1", "Temp", 3*csize, csize);
  //  TCanvas* cMe2 = new TCanvas("cMe2", "Temp", int(1.2*csize), int(0.4*csize));
  //  TCanvas* cMe3 = new TCanvas("cMe3", "Temp", int(0.4*csize), int(0.4*csize));
  TCanvas* cMe2 = new TCanvas("cMe2", "Temp", int(1.8*csize), int(0.9*csize));
  TCanvas* cMe3 = new TCanvas("cMe3", "Temp", int(0.9*csize), int(0.9*csize));

  TH2F* obj2f;
  TH3F* obj3f;
  TProfile2D* obj2p;

  // Loop on barrel supermodules

  for ( unsigned int i=0; i<superModules_.size(); i ++ ) {

    int ism = superModules_[i];

    if ( i>0 ) htmlFile[0] << "<a href=""#top"">Top</a>" << std::endl;
    htmlFile[0] << "<hr>" << std::endl;
    htmlFile[0] << "<h3><a name="""
                << Numbers::sEB(ism).c_str() << """></a><strong>"
                << Numbers::sEB(ism).c_str() << "</strong></h3>" << endl;


    // ---------------------------  Emulator Error

    htmlFile[0] << "<h3><strong>Emulator Error</strong></h3>" << std::endl;
    htmlFile[0] << "<table border=\"0\" cellspacing=\"0\" " << std::endl;
    htmlFile[0] << "cellpadding=\"10\" align=\"center\"> " << std::endl;
    htmlFile[0] << "<tr align=\"center\">" << std::endl;


    imgName[0] = "";

    obj2f = l01_[ism-1];

    if ( obj2f ) {

      meName[0] = obj2f->GetName();

      for ( unsigned int i = 0; i < meName[0].size(); i++ ) {
        if ( meName[0].substr(i, 1) == " " )  {
          meName[0].replace(i, 1 ,"_" );
        }
      }

      imgName[0] = meName[0] + ".png";
      imgMeName[0] = htmlDir + imgName[0];

      cMe2->cd();
      gStyle->SetOptStat(" ");
      gStyle->SetPalette(10, pCol4);
      obj2f->GetXaxis()->SetNdivisions(17);
      obj2f->GetYaxis()->SetNdivisions(4);
      obj2f->SetMinimum(0);
      cMe2->SetGridx();
      cMe2->SetGridy();
      obj2f->Draw("colz");
      dummy.Draw("text,same");
      cMe2->Update();
      cMe2->SaveAs(imgMeName[0].c_str());
    }


    htmlFile[0] << "<img src=\"" << imgName[0] << "\"><br>" << std::endl;
    htmlFile[0] << "</tr>" << std::endl;
    htmlFile[0] << "<br><br>" << std::endl;


    // ---------------------------  Et plots

    for(int iemu=0;iemu<2;iemu++) {

      imgName[iemu] = "";

      obj2p = 0;
      switch ( iemu ) {
        case 0:
          obj2p = UtilsClient::getHisto<TProfile2D*>( me_h01_[ism-1] );
          break;
        case 1:
          obj2p = UtilsClient::getHisto<TProfile2D*>( me_h02_[ism-1] );
          break;
        default:
          break;
      }

      if ( obj2p ) {
        meName[iemu] = obj2p->GetName();

        for ( unsigned int i = 0; i < meName[iemu].size(); i++ ) {
          if ( meName[iemu].substr(i, 1) == " " )  {
            meName[iemu].replace(i, 1 ,"_" );
          }
        }

        imgName[iemu] = meName[iemu] + ".png";
        imgMeName[iemu] = htmlDir + imgName[iemu];

        cMe1->cd();
        gStyle->SetOptStat(" ");
        gStyle->SetPalette(10, pCol4);
        obj2p->GetXaxis()->SetNdivisions(17);
        obj2p->GetYaxis()->SetNdivisions(4);
        cMe1->SetGridx();
        cMe1->SetGridy();
        obj2p->Draw("colz");
        dummy.Draw("text,same");
        cMe1->Update();
        cMe1->SaveAs(imgMeName[iemu].c_str());
        delete obj2p;
      }
    }

    htmlFile[0] << "<td><img src=\"" << imgName[0] << "\"></td>" << std::endl;
    htmlFile[0] << "<td><img src=\"" << imgName[1] << "\"></td>" << std::endl;
    htmlFile[0] << "</table>" << std::endl;
    htmlFile[0] << "<br>" << std::endl;

    std::stringstream subpage;
    subpage << htmlName.substr( 0, htmlName.find( ".html" ) ) << "_" << Numbers::sEB(ism).c_str() << ".html" << std::ends;
    htmlFile[0] << "<a href=\"" << subpage.str().c_str() << "\">" << Numbers::sEB(ism).c_str() << " details</a><br>" << std::endl;
    htmlFile[0] << "<hr>" << std::endl;

    htmlFile[ism].open((htmlDir + subpage.str()).c_str());

    // html page header
    htmlFile[ism] << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << std::endl;
    htmlFile[ism] << "<html>  " << std::endl;
    htmlFile[ism] << "<head>  " << std::endl;
    htmlFile[ism] << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << std::endl;
    htmlFile[ism] << " http-equiv=\"content-type\">  " << std::endl;
    htmlFile[ism] << "  <title>Monitor:TriggerTowerTask output " << Numbers::sEB(ism).c_str()  << "</title> " << std::endl;
    htmlFile[ism] << "</head>  " << std::endl;
    htmlFile[ism] << "<style type=\"text/css\"> td { font-weight: bold } </style>" << std::endl;
    htmlFile[ism] << "<body>  " << std::endl;
    htmlFile[ism] << "<br>  " << std::endl;
    htmlFile[ism] << "<h3>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << std::endl;
    htmlFile[ism] << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << std::endl;
    htmlFile[ism] << " style=\"color: rgb(0, 0, 153);\">" << run << "</span></h3>" << std::endl;
    htmlFile[ism] << "<h3>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << std::endl;
    htmlFile[ism] << " style=\"color: rgb(0, 0, 153);\">TRIGGER TOWER</span></h3> " << std::endl;
    htmlFile[ism] << "<h3>SM:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << std::endl;
    htmlFile[ism] << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << std::endl;
    htmlFile[ism] << " style=\"color: rgb(0, 0, 153);\">" << Numbers::sEB(ism).c_str() << "</span></h3>" << std::endl;
    htmlFile[ism] << "<hr>" << std::endl;

    // ---------------------------  Flag bits plots

    htmlFile[ism] << "<h3><strong>Trigger Tower Flags</strong></h3>" << std::endl;
    htmlFile[ism] << "<table border=\"0\" cellspacing=\"0\" " << std::endl;
    htmlFile[ism] << "cellpadding=\"10\" align=\"center\"> " << std::endl;
    htmlFile[ism] << "<tr align=\"center\">" << std::endl;

    int counter = 0;

    for ( int j=1; j<8; j++ ) {

      for(int iemu=0;iemu<3;iemu++) {

        imgName[iemu] = "";

        if( iemu==0 ) obj3f = m01_[ism-1];
        else if( iemu==1 ) obj3f = j01_[ism-1];
        else if( iemu==2 ) obj3f = j02_[ism-1];

        if ( obj3f ) {

          meName[iemu] = obj3f->GetName();

          for ( unsigned int i = 0; i < meName[iemu].size(); i++ ) {
            if ( meName[iemu].substr(i, 1) == " " )  {
              meName[iemu].replace(i, 1 ,"_" );
            }
          }

          if ( j == 3 ) continue;   //  010 bits combination is not used
          counter++;
          if ( j < 7 ) {
            imgName[iemu] = meName[iemu] + "_" + char(47+j) + ".png";
            obj3f->GetZaxis()->SetRange( j, j );
          }
          else {
            imgName[iemu] = meName[iemu] + "_6-7.png";
            obj3f->GetZaxis()->SetRange( j, j+1 );
          }
          imgMeName[iemu] = htmlDir + imgName[iemu];

          obj2f = (TH2F*) obj3f->Project3D( "yx" );

          cMe2->cd();
          gStyle->SetOptStat(" ");
          gStyle->SetPalette(10, pCol4);
          obj2f->GetXaxis()->SetNdivisions(17);
          obj2f->GetYaxis()->SetNdivisions(4);
          obj2f->SetMinimum(0);
          cMe2->SetGridx();
          cMe2->SetGridy();

          std::stringstream title;
          std::string emustring;
          if (iemu==0) emustring = "Errors ";
          else if(iemu==1) emustring = "Real Digis ";
          else if(iemu==2) emustring = "Emulated Digis ";
          if ( j < 7 ) {
            title << "EBTTT Flags " << emustring << " Bit " << bitset<3>(j-1) << " " << Numbers::sEB(ism).c_str();
          } else {
            title << "EBTTT Flags " << emustring << " Bits 110+111 " << Numbers::sEB(ism).c_str();
          }
          obj2f->SetTitle( title.str().c_str() );

          obj2f->Draw("colz");
          dummy.Draw("text,same");
          cMe2->Update();
          cMe2->SaveAs(imgMeName[iemu].c_str());

          delete obj2f;

          htmlFile[ism] << "<td><img src=\"" << imgName[iemu] << "\"></td>" << std::endl;
          if ( counter%3 == 0 ) htmlFile[ism] << "</tr><tr>" << std::endl;
        }
      }
    }

    htmlFile[ism] << "</tr>" << std::endl << "</table>" << std::endl;



    // ---------------------------  Fine Grain Veto

    htmlFile[ism] << "<h3><strong>Fine Grain Veto</strong></h3>" << std::endl;
    htmlFile[ism] << "<table border=\"0\" cellspacing=\"0\" " << std::endl;
    htmlFile[ism] << "cellpadding=\"10\" align=\"center\"> " << std::endl;
    htmlFile[ism] << "<tr align=\"center\">" << std::endl;


    for ( int j=1; j<=2; j++ ) {

      for(int iemu=0;iemu<3;iemu++) {

        imgName[iemu] = "";

        if( iemu==0 ) obj3f = n01_[ism-1];
        else if( iemu==1 ) obj3f = i01_[ism-1];
        else if( iemu==2 ) obj3f = i02_[ism-1];

        if ( obj3f ) {

          meName[iemu] = obj3f->GetName();

          for ( unsigned int i = 0; i < meName[iemu].size(); i++ ) {
            if ( meName[iemu].substr(i, 1) == " " )  {
              meName[iemu].replace(i, 1 ,"_" );
            }
          }

          imgName[iemu] = meName[iemu] + "_" + char(47+j) + ".png";
          imgMeName[iemu] = htmlDir + imgName[iemu];

          obj3f->GetZaxis()->SetRange( j, j );

          obj2f = (TH2F*) obj3f->Project3D( "yx" );

          cMe2->cd();
          gStyle->SetOptStat(" ");
          gStyle->SetPalette(10, pCol4);
          obj2f->GetXaxis()->SetNdivisions(17);
          obj2f->GetYaxis()->SetNdivisions(4);
          obj2f->SetMinimum(0);
          cMe2->SetGridx();
          cMe2->SetGridy();

          std::stringstream title;
          std::string emustring;
          if (iemu==0) emustring = "Errors ";
          else if(iemu==1) emustring = "Real Digis ";
          else if(iemu==2) emustring = "Emulated Digis ";
          title << "EBTTT FineGrainVeto " << emustring << " Flag " << j-1 << " " << Numbers::sEB(ism).c_str();
          obj2f->SetTitle( title.str().c_str() );

          obj2f->Draw("colz");
          dummy.Draw("text,same");
          cMe2->Update();
          cMe2->SaveAs(imgMeName[iemu].c_str());
          delete obj2f;

          htmlFile[ism] << "<td><img src=\"" << imgName[iemu] << "\"></td>" << std::endl;
        }
      }
      htmlFile[ism] << "</tr><tr>" << std::endl;
    }

    htmlFile[ism] << "</tr>" << std::endl << "</table>" << std::endl;

    // ---------------------------  Et plots per Tower

    //     htmlFile[ism] << "<h3><strong>Et</strong></h3>" << std::endl;
    //     htmlFile[ism] << "<table border=\"0\" cellspacing=\"0\" " << std::endl;
    //     htmlFile[ism] << "cellpadding=\"10\" align=\"center\"> " << std::endl;
    //     htmlFile[ism] << "<tr align=\"center\">" << std::endl;
    //
    //     for (int j=0; j<68; j++) {
    //
    //       TH1F* obj1f1 = k01_[ism-1][j];
    //       TH1F* obj1f2 = k02_[ism-1][j];
    //
    //       if ( obj1f1 ) {
    //
    //         imgName[iemu] = "";
    //
    //         meName[iemu] = obj1f1->GetName();
    //
    //         for ( unsigned int i = 0; i < meName[iemu].size(); i++ ) {
    //           if ( meName[iemu].substr(i, 1) == " " )  {
    //             meName[iemu].replace(i, 1 ,"_" );
    //           }
    //         }
    //
    //         imgName[iemu] = meName[iemu] + ".png";
    //         imgMeName[iemu] = htmlDir + imgName[iemu];
    //
    //         cMe3->cd();
    //         gStyle->SetOptStat("euomr");
    //         if ( obj1f2 ) {
    //           float m = TMath::Max( obj1f1->GetMaximum(), obj1f2->GetMaximum() );
    //           obj1f1->SetMaximum( m + 1. );
    //         }
    //         obj1f1->SetStats(kTRUE);
    //         gStyle->SetStatW( gStyle->GetStatW() * 1.5 );
    //         obj1f1->Draw();
    //         cMe3->Update();
    //
    //         if ( obj1f2 ) {
    //           gStyle->SetStatY( gStyle->GetStatY() - 1.25*gStyle->GetStatH() );
    //           gStyle->SetStatTextColor( kRed );
    //           obj1f2->SetStats(kTRUE);
    //           obj1f2->SetLineColor( kRed );
    //           obj1f2->Draw( "sames" );
    //           cMe3->Update();
    //           gStyle->SetStatY( gStyle->GetStatY() + 1.25*gStyle->GetStatH() );
    //           gStyle->SetStatTextColor( kBlack );
    //         }
    //
    //         gStyle->SetStatW( gStyle->GetStatW() / 1.5 );
    //         cMe3->SaveAs(imgMeName[iemu].c_str());
    //
    //         htmlFile[ism] << "<td><img src=\"" << imgName[iemu] << "\"></td>" << std::endl;
    //
    //       }
    //
    //       if ( (j+1)%4 == 0 ) htmlFile[ism] << "</tr><tr>" << std::endl;
    //
    //     }

    htmlFile[ism] << "</tr>" << std::endl << "</table>" << std::endl;

    // html page footer
    htmlFile[ism] << "</body> " << std::endl;
    htmlFile[ism] << "</html> " << std::endl;
    htmlFile[ism].close();
  }

  delete cMe1;
  delete cMe2;
  delete cMe3;

  // html page footer
  htmlFile[0] << "</body> " << std::endl;
  htmlFile[0] << "</html> " << std::endl;
  htmlFile[0].close();

}

