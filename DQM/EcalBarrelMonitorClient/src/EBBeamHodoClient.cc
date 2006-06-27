/*
 * \file EBBeamHodoClient.cc
 *
 * $Date: 2006/06/27 14:03:04 $
 * $Revision: 1.5 $
 * \author G. Della Ricca
 * \author G. Franzoni
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

#include <DQM/EcalBarrelMonitorClient/interface/EBBeamHodoClient.h>
#include <DQM/EcalBarrelMonitorClient/interface/EBMUtilsClient.h>

EBBeamHodoClient::EBBeamHodoClient(const ParameterSet& ps, MonitorUserInterface* mui){

  mui_ = mui;

  // collateSources switch
  collateSources_ = ps.getUntrackedParameter<bool>("collateSources", false);

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
  for ( unsigned int i = 1; i < 37; i++ ) superModules_.push_back(i);
  superModules_ = ps.getUntrackedParameter<vector<int> >("superModules", superModules_);

  for (int i=0; i<4; i++) {

    ho01_[i] = 0;
    hr01_[i] = 0;

  }

  hp01_[0] = 0;
  hp01_[1] = 0;

  hp02_ = 0;

  hs01_[0] = 0;
  hs01_[1] = 0;

  hq01_[0] = 0;
  hq01_[1] = 0;

  ht01_ = 0;

  hc01_[0] = 0;
  hc01_[1] = 0;
  hc01_[2] = 0;

  he01_[0] = 0;
  he01_[1] = 0;

  he02_[0] = 0;
  he02_[1] = 0;

  he03_[0] = 0;
  he03_[1] = 0;
  he03_[2] = 0;

}

EBBeamHodoClient::~EBBeamHodoClient(){

}

void EBBeamHodoClient::beginJob(void){

  if ( verbose_ ) cout << "EBBeamHodoClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EBBeamHodoClient::beginRun(void){

  if ( verbose_ ) cout << "EBBeamHodoClient: beginRun" << endl;

  jevt_ = 0;

  this->setup();

  this->subscribe();

}

void EBBeamHodoClient::endJob(void) {

  if ( verbose_ ) cout << "EBBeamHodoClient: endJob, ievt = " << ievt_ << endl;

  this->unsubscribe();

  this->cleanup();

  if ( cloneME_ ) {

    for (int i=0; i<4; i++) {

      if ( ho01_[i] ) delete ho01_[i];
      if ( hr01_[i] ) delete hr01_[i];

    }

    if ( hp01_[0] ) delete hp01_[0];
    if ( hp01_[1] ) delete hp01_[1];

    if ( hp02_ ) delete hp02_;

    if ( hs01_[0] ) delete hs01_[0];
    if ( hs01_[1] ) delete hs01_[1];

    if ( hq01_[0] ) delete hq01_[0];
    if ( hq01_[1] ) delete hq01_[1];

    if ( ht01_ ) delete ht01_;

    if ( hc01_[0] ) delete hc01_[0];
    if ( hc01_[1] ) delete hc01_[1];
    if ( hc01_[2] ) delete hc01_[2];

    if ( he01_[0] ) delete he01_[0];
    if ( he01_[1] ) delete he01_[1];

    if ( he02_[0] ) delete he02_[0];
    if ( he02_[1] ) delete he02_[1];

    if ( he03_[0] ) delete he03_[0];
    if ( he03_[1] ) delete he03_[1];
    if ( he03_[2] ) delete he03_[2];

  }

  for (int i=0; i<4; i++) {

    ho01_[i] = 0;
    hr01_[i] = 0;

  }

  hp01_[0] = 0;
  hp01_[1] = 0;

  hp02_ = 0;

  hs01_[0] = 0;
  hs01_[1] = 0;

  hq01_[0] = 0;
  hq01_[1] = 0;

  ht01_ = 0;

  hc01_[0] = 0;
  hc01_[1] = 0;
  hc01_[2] = 0;

  he01_[0] = 0;
  he01_[1] = 0;

  he02_[0] = 0;
  he02_[1] = 0;

  he03_[0] = 0;
  he03_[1] = 0;
  he03_[2] = 0;

}

void EBBeamHodoClient::endRun(void) {

  if ( verbose_ ) cout << "EBBeamHodoClient: endRun, jevt = " << jevt_ << endl;

  this->unsubscribe();

  this->cleanup();

}

void EBBeamHodoClient::setup(void) {

}

void EBBeamHodoClient::cleanup(void) {

}

void EBBeamHodoClient::writeDb(EcalCondDBInterface* econn, MonRunIOV* moniov, int ism) {

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

void EBBeamHodoClient::subscribe(void){

  if ( verbose_ ) cout << "EBBeamHodoClient: subscribe" << endl;

  int smId = 1;

  Char_t histo[80];

  for (int i=0; i<4; i++) {

    sprintf(histo, "*/EcalBarrel/EBBeamHodoTask/EBBHT occup SM%02d, %02d", smId, i+1);
    mui_->subscribe(histo);
    sprintf(histo, "*/EcalBarrel/EBBeamHodoTask/EBBHT raw SM%02d, %02d", smId, i+1);
    mui_->subscribe(histo);

  }

  sprintf(histo, "*/EcalBarrel/EBBeamHodoTask/EBBHT PosX rec SM%02d", smId);
  mui_->subscribe(histo);

  sprintf(histo, "*/EcalBarrel/EBBeamHodoTask/EBBHT PosY rec SM%02d", smId);
  mui_->subscribe(histo);

  sprintf(histo, "*/EcalBarrel/EBBeamHodoTask/EBBHT PosYX rec SM%02d", smId);
  mui_->subscribe(histo);

  sprintf(histo, "*/EcalBarrel/EBBeamHodoTask/EBBHT SloX SM%02d", smId);
  mui_->subscribe(histo);

  sprintf(histo, "*/EcalBarrel/EBBeamHodoTask/EBBHT SloY SM%02d", smId);
  mui_->subscribe(histo);

  sprintf(histo, "*/EcalBarrel/EBBeamHodoTask/EBBHT QualX SM%02d", smId);
  mui_->subscribe(histo);

  sprintf(histo, "*/EcalBarrel/EBBeamHodoTask/EBBHT QualY SM%02d", smId);
  mui_->subscribe(histo);

  sprintf(histo, "*/EcalBarrel/EBBeamHodoTask/EBBHT TDC rec SM%02d", smId);
  mui_->subscribe(histo);

  sprintf(histo, "*/EcalBarrel/EBBeamHodoTask/EBBHT (Hodo-Calo)XVsCry SM%02d", smId);
  mui_->subscribe(histo);

  sprintf(histo, "*/EcalBarrel/EBBeamHodoTask/EBBHT (Hodo-Calo)YVsCry SM%02d", smId);
  mui_->subscribe(histo);

  sprintf(histo, "*/EcalBarrel/EBBeamHodoTask/EBBHT (TDC-Calo)VsCry SM%02d", smId);
  mui_->subscribe(histo);

  sprintf(histo, "*/EcalBarrel/EBBeamHodoTask/EBBHT prof E1 vs X SM%02d", smId);
  mui_->subscribe(histo);

  sprintf(histo, "*/EcalBarrel/EBBeamHodoTask/EBBHT prof E1 vs Y SM%02d", smId);
  mui_->subscribe(histo);

  sprintf(histo, "*/EcalBarrel/EBBeamHodoTask/EBBHT his E1 vs X SM%02d", smId);
  mui_->subscribe(histo);

  sprintf(histo, "*/EcalBarrel/EBBeamHodoTask/EBBHT his E1 vs Y SM%02d", smId);
  mui_->subscribe(histo);

  sprintf(histo, "*/EcalBarrel/EBBeamHodoTask/EBBHT PosX: Hodo-Calo SM%02d", smId);
  mui_->subscribe(histo);

  sprintf(histo, "*/EcalBarrel/EBBeamHodoTask/EBBHT PosY: Hodo-Calo SM%02d", smId);
  mui_->subscribe(histo);

  sprintf(histo, "*/EcalBarrel/EBBeamHodoTask/EBBHT TimeMax: TDC-Calo SM%02d", smId);
  mui_->subscribe(histo);

  if ( collateSources_ ) {

    if ( verbose_ ) cout << "EBBeamHodoClient: collate" << endl;

  }

}

void EBBeamHodoClient::subscribeNew(void){

  Char_t histo[80];
  
  int smId = 1;
  
  for (int i=0; i<4; i++) {

    sprintf(histo, "*/EcalBarrel/EBBeamHodoTask/EBBHT occup SM%02d, %02d", smId, i+1);
    mui_->subscribeNew(histo);
    sprintf(histo, "*/EcalBarrel/EBBeamHodoTask/EBBHT raw SM%02d, %02d", smId, i+1);
    mui_->subscribeNew(histo);

  }

  sprintf(histo, "*/EcalBarrel/EBBeamHodoTask/EBBHT PosX rec SM%02d", smId);
  mui_->subscribeNew(histo);

  sprintf(histo, "*/EcalBarrel/EBBeamHodoTask/EBBHT PosY rec SM%02d", smId);
  mui_->subscribeNew(histo);

  sprintf(histo, "*/EcalBarrel/EBBeamHodoTask/EBBHT PosYX rec SM%02d", smId);
  mui_->subscribeNew(histo);

  sprintf(histo, "*/EcalBarrel/EBBeamHodoTask/EBBHT SloX SM%02d", smId);
  mui_->subscribeNew(histo);
  
  sprintf(histo, "*/EcalBarrel/EBBeamHodoTask/EBBHT SloY SM%02d", smId);
  mui_->subscribeNew(histo);

  sprintf(histo, "*/EcalBarrel/EBBeamHodoTask/EBBHT QualX SM%02d", smId);
  mui_->subscribeNew(histo);
  
  sprintf(histo, "*/EcalBarrel/EBBeamHodoTask/EBBHT QualY SM%02d", smId);
  mui_->subscribeNew(histo);

  sprintf(histo, "*/EcalBarrel/EBBeamHodoTask/EBBHT TDC rec SM%02d", smId);
  mui_->subscribeNew(histo);

  sprintf(histo, "*/EcalBarrel/EBBeamHodoTask/EBBHT (Hodo-Calo)XVsCry SM%02d", smId);
  mui_->subscribeNew(histo);

  sprintf(histo, "*/EcalBarrel/EBBeamHodoTask/EBBHT (Hodo-Calo)YVsCry SM%02d", smId);
  mui_->subscribeNew(histo);

  sprintf(histo, "*/EcalBarrel/EBBeamHodoTask/EBBHT (TDC-Calo)VsCry SM%02d", smId);
  mui_->subscribeNew(histo);

  sprintf(histo, "*/EcalBarrel/EBBeamHodoTask/EBBHT prof E1 vs X SM%02d", smId);
  mui_->subscribeNew(histo);

  sprintf(histo, "*/EcalBarrel/EBBeamHodoTask/EBBHT prof E1 vs Y SM%02d", smId);
  mui_->subscribeNew(histo);
  
  sprintf(histo, "*/EcalBarrel/EBBeamHodoTask/EBBHT his E1 vs X SM%02d", smId);
  mui_->subscribeNew(histo);

  sprintf(histo, "*/EcalBarrel/EBBeamHodoTask/EBBHT his E1 vs Y SM%02d", smId);
  mui_->subscribeNew(histo);

  sprintf(histo, "*/EcalBarrel/EBBeamHodoTask/EBBHT PosX: Hodo-Calo SM%02d", smId);
  mui_->subscribeNew(histo);

  sprintf(histo, "*/EcalBarrel/EBBeamHodoTask/EBBHT PosY: Hodo-Calo SM%02d", smId);
  mui_->subscribeNew(histo);

  sprintf(histo, "*/EcalBarrel/EBBeamHodoTask/EBBHT TimeMax: TDC-Calo SM%02d", smId);
  mui_->subscribeNew(histo);

}

void EBBeamHodoClient::unsubscribe(void){

  if ( verbose_ ) cout << "EBBeamHodoClient: unsubscribe" << endl;

  Char_t histo[80];

  int smId = 1;

  for (int i=0; i<4; i++) {

    sprintf(histo, "*/EcalBarrel/EBBeamHodoTask/EBBHT occup SM%02d, %02d", smId, i+1);
    mui_->unsubscribe(histo);
    sprintf(histo, "*/EcalBarrel/EBBeamHodoTask/EBBHT raw SM%02d, %02d", smId, i+1);
    mui_->unsubscribe(histo);

  }

  sprintf(histo, "*/EcalBarrel/EBBeamHodoTask/EBBHT PosX rec SM%02d", smId);
  mui_->unsubscribe(histo);
  
  sprintf(histo, "*/EcalBarrel/EBBeamHodoTask/EBBHT PosY rec SM%02d", smId);
  mui_->unsubscribe(histo);

  sprintf(histo, "*/EcalBarrel/EBBeamHodoTask/EBBHT PosYX rec SM%02d", smId);
  mui_->unsubscribe(histo);

  sprintf(histo, "*/EcalBarrel/EBBeamHodoTask/EBBHT SloX SM%02d", smId);
  mui_->unsubscribe(histo);

  sprintf(histo, "*/EcalBarrel/EBBeamHodoTask/EBBHT SloY SM%02d", smId);
  mui_->unsubscribe(histo);

  sprintf(histo, "*/EcalBarrel/EBBeamHodoTask/EBBHT QualX SM%02d", smId);
  mui_->unsubscribe(histo);
  
  sprintf(histo, "*/EcalBarrel/EBBeamHodoTask/EBBHT QualY SM%02d", smId);
  mui_->unsubscribe(histo);

  sprintf(histo, "*/EcalBarrel/EBBeamHodoTask/EBBHT TDC rec SM%02d", smId);
  mui_->unsubscribe(histo);

  sprintf(histo, "*/EcalBarrel/EBBeamHodoTask/EBBHT (Hodo-Calo)XVsCry SM%02d", smId);
  mui_->unsubscribe(histo);

  sprintf(histo, "*/EcalBarrel/EBBeamHodoTask/EBBHT (Hodo-Calo)YVsCry SM%02d", smId);
  mui_->unsubscribe(histo);

  sprintf(histo, "*/EcalBarrel/EBBeamHodoTask/EBBHT (TDC-Calo)VsCry SM%02d", smId);
  mui_->unsubscribe(histo);

  sprintf(histo, "*/EcalBarrel/EBBeamHodoTask/EBBHT prof E1 vs X SM%02d", smId);
  mui_->unsubscribe(histo);
    
  sprintf(histo, "*/EcalBarrel/EBBeamHodoTask/EBBHT prof E1 vs Y SM%02d", smId);
  mui_->unsubscribe(histo);
  
  sprintf(histo, "*/EcalBarrel/EBBeamHodoTask/EBBHT his E1 vs X SM%02d", smId);
  mui_->unsubscribe(histo);

  sprintf(histo, "*/EcalBarrel/EBBeamHodoTask/EBBHT his E1 vs Y SM%02d", smId);
  mui_->unsubscribe(histo);
  
  sprintf(histo, "*/EcalBarrel/EBBeamHodoTask/EBBHT PosX: Hodo-Calo SM%02d", smId);
  mui_->unsubscribe(histo);

  sprintf(histo, "*/EcalBarrel/EBBeamHodoTask/EBBHT PosY: Hodo-Calo SM%02d", smId);
  mui_->unsubscribe(histo);

  sprintf(histo, "*/EcalBarrel/EBBeamHodoTask/EBBHT TimeMax: TDC-Calo SM%02d", smId);
  mui_->unsubscribe(histo);

  if ( collateSources_ ) {

    if ( verbose_ ) cout << "EBBeamHodoClient: uncollate" << endl;

  }

}

void EBBeamHodoClient::analyze(void){

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( verbose_ ) cout << "EBBeamHodoClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;
  }

  int smId = 1;

  Char_t histo[150];

  MonitorElement* me;

  for (int i=0; i<4; i++) {

    if ( collateSources_ ) {
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBBeamHodoTask/EBBHT occup SM%02d, %02d").c_str(), smId, i+1);
    }
    me = mui_->get(histo);
    ho01_[i] = EBMUtilsClient::getHisto<TH1F*>( me, cloneME_, ho01_[i] );

    if ( collateSources_ ) {
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBBeamHodoTask/EBBHT raw SM%02d, %02d").c_str(), smId, i+1);
    }
    me = mui_->get(histo);
    hr01_[i] = EBMUtilsClient::getHisto<TH1F*>( me, cloneME_, hr01_[i] );

  }

  if ( collateSources_ ) {
  } else {
    sprintf(histo, (prefixME_+"EcalBarrel/EBBeamHodoTask/EcalBarrel/EBBeamHodoTask/EBBHT PosX rec SM%02d").c_str(), smId);
  }
  me = mui_->get(histo);
  hp01_[0] = EBMUtilsClient::getHisto<TH1F*>( me, cloneME_, hp01_[0] );

  if ( collateSources_ ) {
  } else {
    sprintf(histo, (prefixME_+"EcalBarrel/EBBeamHodoTask/EBBHT PosY rec SM%02d").c_str(), smId);
  }
  me = mui_->get(histo);
  hp01_[1] = EBMUtilsClient::getHisto<TH1F*>( me, cloneME_, hp01_[1] );

  if ( collateSources_ ) {
  } else {
    sprintf(histo, (prefixME_+"EcalBarrel/EBBeamHodoTask/EBBHT PosYX rec SM%02d").c_str(), smId);
  }
  me = mui_->get(histo);
  hp02_ = EBMUtilsClient::getHisto<TH2F*>( me, cloneME_, hp02_ );

  if ( collateSources_ ) {
  } else {
    sprintf(histo, (prefixME_+"EcalBarrel/EBBeamHodoTask/EBBHT SloX SM%02d").c_str(), smId);
  }
  me = mui_->get(histo);
  hs01_[0] = EBMUtilsClient::getHisto<TH1F*>( me, cloneME_, hs01_[0] );

  if ( collateSources_ ) {
  } else {
    sprintf(histo, (prefixME_+"EcalBarrel/EBBeamHodoTask/EBBHT SloY SM%02d").c_str(), smId);
  }
  me = mui_->get(histo);
  hs01_[1] = EBMUtilsClient::getHisto<TH1F*>( me, cloneME_, hs01_[1] );

  if ( collateSources_ ) {
  } else {
    sprintf(histo, (prefixME_+"EcalBarrel/EBBeamHodoTask/EBBHT QualX SM%02d").c_str(), smId);
  }
  me = mui_->get(histo);
  hq01_[0] = EBMUtilsClient::getHisto<TH1F*>( me, cloneME_, hq01_[0] );

  if ( collateSources_ ) {
  } else {
    sprintf(histo, (prefixME_+"EcalBarrel/EBBeamHodoTask/EBBHT QualY SM%02d").c_str(), smId);
  }
  me = mui_->get(histo);
  hq01_[1] = EBMUtilsClient::getHisto<TH1F*>( me, cloneME_, hq01_[1] );

  if ( collateSources_ ) {
  } else {
    sprintf(histo, (prefixME_+"EcalBarrel/EBBeamHodoTask/EBBHT TDC rec SM%02d").c_str(), smId);
  }
  me = mui_->get(histo);
  ht01_ = EBMUtilsClient::getHisto<TH1F*>( me, cloneME_, ht01_ );

  if ( collateSources_ ) {
  } else {
    sprintf(histo, (prefixME_+"EcalBarrel/EBBeamHodoTask/EBBHT (Hodo-Calo)XVsCry SM%02d").c_str(), smId);
  }
  me = mui_->get(histo);
  hc01_[0] = EBMUtilsClient::getHisto<TH1F*>( me, cloneME_, hc01_[0] );

  if ( collateSources_ ) {
  } else {
    sprintf(histo, (prefixME_+"EcalBarrel/EBBeamHodoTask/EBBHT (Hodo-Calo)YVsCry SM%02d").c_str(), smId);
  }
  me = mui_->get(histo);
  hc01_[1] = EBMUtilsClient::getHisto<TH1F*>( me, cloneME_, hc01_[1] );

  if ( collateSources_ ) {
  } else {
    sprintf(histo, (prefixME_+"EcalBarrel/EBBeamHodoTask/EBBHT (TDC-Calo)VsCry SM%02d").c_str(), smId);
  }
  me = mui_->get(histo);
  hc01_[2] = EBMUtilsClient::getHisto<TH1F*>( me, cloneME_, hc01_[2] );

  if ( collateSources_ ) {
  } else {
    sprintf(histo, (prefixME_+"EcalBarrel/EBBeamHodoTask/EBBHT prof E1 vs X SM%02d").c_str(), smId);
  }
  me = mui_->get(histo);
  he01_[0] = EBMUtilsClient::getHisto<TProfile*>( me, cloneME_, he01_[0] );

  if ( collateSources_ ) {
  } else {
    sprintf(histo, (prefixME_+"EcalBarrel/EBBeamHodoTask/EBBHT prof E1 vs Y SM%02d").c_str(), smId);
  }
  me = mui_->get(histo);
  he01_[1] = EBMUtilsClient::getHisto<TProfile*>( me, cloneME_, he01_[1] );

  if ( collateSources_ ) {
  } else {
    sprintf(histo, (prefixME_+"EcalBarrel/EBBeamHodoTask/EBBHT his E1 vs X SM%02d").c_str(), smId);
  }
  me = mui_->get(histo);
  he02_[0] = EBMUtilsClient::getHisto<TProfile2D*>( me, cloneME_, he02_[0] );

  if ( collateSources_ ) {
  } else {
    sprintf(histo, (prefixME_+"EcalBarrel/EBBeamHodoTask/EBBHT his E1 vs Y SM%02d").c_str(), smId);
  }
  me = mui_->get(histo);
  he02_[1] = EBMUtilsClient::getHisto<TProfile2D*>( me, cloneME_, he02_[1] );

  if ( collateSources_ ) {
  } else {
    sprintf(histo, (prefixME_+"EcalBarrel/EBBeamHodoTask/EBBHT PosX: Hodo-Calo SM%02d").c_str(), smId);
  }
  me = mui_->get(histo);
  he03_[0] = EBMUtilsClient::getHisto<TProfile*>( me, cloneME_, he03_[0] );

  if ( collateSources_ ) {
  } else {
    sprintf(histo, (prefixME_+"EcalBarrel/EBBeamHodoTask/EBBHT PosY: Hodo-Calo SM%02d").c_str(), smId);
  }
  me = mui_->get(histo);
  he03_[1] = EBMUtilsClient::getHisto<TProfile*>( me, cloneME_, he03_[1] );

  if ( collateSources_ ) {
  } else {
    sprintf(histo, (prefixME_+"EcalBarrel/EBBeamHodoTask/EBBHT TimeMax: TDC-Calo SM%02d").c_str(), smId);
  }
  me = mui_->get(histo);
  he03_[2] = EBMUtilsClient::getHisto<TProfile*>( me, cloneME_, he03_[2] );

}

void EBBeamHodoClient::htmlOutput(int run, string htmlDir, string htmlName){

  cout << "Preparing EBBeamHodoClient html output ..." << endl;

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
  htmlFile << " style=\"color: rgb(0, 0, 153);\">BeamHodo</span></h2> " << endl;
  htmlFile << "<hr>" << endl;
//  htmlFile << "<table border=1><tr><td bgcolor=red>channel has problems in this task</td>" << endl;
//  htmlFile << "<td bgcolor=lime>channel has NO problems</td>" << endl;
//  htmlFile << "<td bgcolor=yellow>channel is missing</td></table>" << endl;
//  htmlFile << "<hr>" << endl;

  // Produce the plots to be shown as .png files from existing histograms

  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

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

  string imgNameP, imgNameR, imgName, meName;

  TCanvas* cP = new TCanvas("cP", "Temp", csize, csize);

  TH2F* obj2f;
  TH1F* obj1f;
  TProfile2D* objp;

  for (int i=0; i<4; i++) {

    imgNameP = "";

    obj1f = ho01_[i];

    if ( obj1f ) {

      meName = obj1f->GetName();

      for ( unsigned int j = 0; j < meName.size(); j++ ) {
        if ( meName.substr(j, 1) == " " )  {
          meName.replace(j, 1, "_");
        }
      }
      imgNameP = meName + ".png";
      imgName = htmlDir + imgNameP;

      cP->cd();
      gStyle->SetOptStat("euomr");
      obj1f->SetStats(kTRUE);
      if ( obj1f->GetMaximum(histMax) > 0. ) {
        gPad->SetLogy(1);
      } else {
        gPad->SetLogy(0);
      }
      obj1f->Draw();
      cP->Update();
      cP->SaveAs(imgName.c_str());
      gPad->SetLogy(0);

    }

    obj1f = hr01_[i];

    if ( obj1f ) {

      meName = obj1f->GetName();
    
      for ( unsigned int j = 0; j < meName.size(); j++ ) {
        if ( meName.substr(j, 1) == " " )  {
          meName.replace(j, 1, "_");
        }
      }
      imgNameR = meName + ".png";
      imgName = htmlDir + imgNameR;

      cP->cd();
      gStyle->SetOptStat("euomr");
      obj1f->SetStats(kTRUE);
      if ( obj1f->GetMaximum(histMax) > 0. ) {
        gPad->SetLogy(1);
      } else {
        gPad->SetLogy(0);
      }
      obj1f->Draw();
      cP->Update();
      cP->SaveAs(imgName.c_str());
      gPad->SetLogy(0);

    }

    htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
    htmlFile << "<tr align=\"center\">" << endl;

    if ( imgNameP.size() != 0 )
      htmlFile << "<td colspan=\"2\"><img src=\"" << imgNameP << "\"></td>" << endl;
    else
      htmlFile << "<td colspan=\"2\"><img src=\"" << " " << "\"></td>" << endl;

    if ( imgNameR.size() != 0 )
      htmlFile << "<td colspan=\"2\"><img src=\"" << imgNameR << "\"></td>" << endl;
    else
      htmlFile << "<td colspan=\"2\"><img src=\"" << " " << "\"></td>" << endl;

    htmlFile << "</tr>" << endl;

    htmlFile << "</table>" << endl;
    htmlFile << "<br>" << endl;

  }

  htmlFile.close();

}

