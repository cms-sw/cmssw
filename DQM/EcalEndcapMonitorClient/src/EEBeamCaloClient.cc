/*
 * \file EEBeamCaloClient.cc
 *
 * $Date: 2007/07/19 12:02:02 $
 * $Revision: 1.10 $
 * \author G. Della Ricca
 * \author A. Ghezzi
 *
 */

#include <memory>
#include <iostream>
#include <fstream>
#include <algorithm>

#include "TStyle.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Core/interface/QTestStatus.h"
#include "DQMServices/QualityTests/interface/QCriterionRoot.h"
#include "DQMServices/UI/interface/MonitorUIRoot.h"

#include "OnlineDB/EcalCondDB/interface/RunTag.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"

#include "OnlineDB/EcalCondDB/interface/MonOccupancyDat.h"

#include "DQM/EcalCommon/interface/EcalErrorMask.h"
#include <DQM/EcalCommon/interface/UtilsClient.h>
#include <DQM/EcalCommon/interface/LogicID.h>
#include <DQM/EcalCommon/interface/Numbers.h>

#include <DQM/EcalEndcapMonitorClient/interface/EEBeamCaloClient.h>

using namespace cms;
using namespace edm;
using namespace std;

EEBeamCaloClient::EEBeamCaloClient(const ParameterSet& ps){

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

  checkedSteps_.reserve(86);
  // there should be not more than a eta row in an autoscan
  minEvtNum_ = 1800;//
  //FIX ME, this should be configurable and change with the beam energy
  aveEne1_    = 1850;  E1Th_   = 900;
  aveEne3x3_  = 2600;  E3x3Th_ = 2600;
  RMSEne3x3_  = 800;

  ReadCryErrThr_ = 0.01;// 1%
  //FIX ME, this should follow the prescaling in the monitoring
  prescaling_ = 20;

  ///////// task specific histos
  for(int u=0;u<cryInArray_;u++){
    hBGains_[u] = 0;
    hBpulse_[u] = 0;
    //hBGainsMoving_[u] = 0;
  }
  hBEne1_ = 0;
  //hBEne1Moving_ = 0;
  hBAllNeededCry_ = 0;
  hBNumReadCry_ = 0;
  hBE3x3_ = 0;
  hBE3x3Moving_ = 0;
  hBCryOnBeam_ = 0;
  hBMaxEneCry_ = 0;
  hBReadCryErrors_ = 0;
  hBE1vsCry_ = 0;
  hBE3x3vsCry_ = 0;
  hBEntriesvsCry_ = 0;
  hBcryDone_ = 0;
  hBBeamCentered_ = 0;
  hbTBmoving_ = 0;
  hbE1MaxCry_ = 0;
  hbDesync_ = 0;
  pBCriInBeamEvents_ = 0;

  meEEBCaloRedGreen_ = 0;
  meEEBCaloRedGreenReadCry_ = 0;
  meEEBCaloRedGreenSteps_ = 0;
}

EEBeamCaloClient::~EEBeamCaloClient(){

}

void EEBeamCaloClient::beginJob(MonitorUserInterface* mui){

  mui_ = mui;

  if ( verbose_ ) cout << "EEBeamCaloClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

  if ( enableQT_ ) {

  }

}

void EEBeamCaloClient::beginRun(void){

  if ( verbose_ ) cout << "EEBeamCaloClient: beginRun" << endl;

  jevt_ = 0;

  this->setup();

  this->subscribe();

}

void EEBeamCaloClient::endJob(void) {

  if ( verbose_ ) cout << "EEBeamCaloClient: endJob, ievt = " << ievt_ << endl;

  this->unsubscribe();

  this->cleanup();

}

void EEBeamCaloClient::endRun(void) {

  if ( verbose_ ) cout << "EEBeamCaloClient: endRun, jevt = " << jevt_ << endl;

  this->unsubscribe();

  this->cleanup();

}

void EEBeamCaloClient::setup(void) {

  Char_t histo[200];

  mui_->setCurrentFolder( "EcalEndcap/EEBeamCaloClient" );
  DaqMonitorBEInterface* dbe = mui_->getBEInterface();
  if ( meEEBCaloRedGreen_ ) dbe->removeElement( meEEBCaloRedGreen_->getName() );
  sprintf(histo, "EEBCT quality");
  meEEBCaloRedGreen_ = dbe->book2D(histo, histo, 85, 0., 85., 20, 0., 20.);

  UtilsClient::resetHisto( meEEBCaloRedGreen_ );

  for ( int ie = 1; ie <= 85; ie++ ) {
    for ( int ip = 1; ip <= 20; ip++ ) {

      meEEBCaloRedGreen_ ->setBinContent( ie, ip, 2. );

    }
  }

  if ( meEEBCaloRedGreenReadCry_ ) dbe->removeElement( meEEBCaloRedGreenReadCry_->getName() );
  sprintf(histo, "EEBCT quality read crystal errors");
  meEEBCaloRedGreenReadCry_ = dbe->book2D(histo, histo, 1, 0., 1., 1, 0., 1.);
  UtilsClient::resetHisto( meEEBCaloRedGreenReadCry_ );
  meEEBCaloRedGreenReadCry_ ->setBinContent( 1, 1, 2. );

  if( meEEBCaloRedGreenSteps_ )  dbe->removeElement( meEEBCaloRedGreenSteps_->getName() );
  sprintf(histo, "EEBCT quality entries or read crystals errors");
  meEEBCaloRedGreenSteps_ = dbe->book2D(histo, histo, 86, 1., 87., 1, 0., 1.);
  UtilsClient::resetHisto( meEEBCaloRedGreenSteps_ );
  for( int bin=1; bin <87; bin++){ meEEBCaloRedGreenSteps_->setBinContent( bin, 1, 2. );}

}

void EEBeamCaloClient::cleanup(void) {
  if ( cloneME_ ) {
    for(int u=0;u<cryInArray_;u++){
      if(hBGains_[u]) delete hBGains_[u];
      if(hBpulse_[u]) delete hBpulse_[u];
      //if(hBGainsMoving_[u])delete hBGainsMoving_[u];
    }
    if(hBEne1_) delete hBEne1_;
    //    if(hBEne1Moving_) delete hBEne1Moving_;
    if(hBAllNeededCry_) delete hBAllNeededCry_;
    if(hBNumReadCry_) delete hBNumReadCry_;
    if(hBE3x3_) delete hBE3x3_;
    if(hBE3x3Moving_) delete hBE3x3Moving_;
    if(hBCryOnBeam_) delete hBCryOnBeam_;
    if(hBMaxEneCry_) delete hBMaxEneCry_;
    if(hBReadCryErrors_) delete hBReadCryErrors_;
    if(hBE1vsCry_) delete hBE1vsCry_;
    if(hBE3x3vsCry_) delete hBE3x3vsCry_;
    if(hBEntriesvsCry_) delete hBEntriesvsCry_;
    if(hBcryDone_) delete hBcryDone_;
    if(hBBeamCentered_) delete hBBeamCentered_;
    if(hbTBmoving_) delete hbTBmoving_;
    if(hbE1MaxCry_) delete hbE1MaxCry_;
    if(hbDesync_) delete hbDesync_;
    if(pBCriInBeamEvents_) delete pBCriInBeamEvents_;
  }

  for(int u=0;u<cryInArray_;u++){
    hBGains_[u] = 0;
    hBpulse_[u] = 0;
    //hBGainsMoving_[u] = 0;
  }
  hBEne1_ = 0;
  //hBEne1Moving_ = 0;
  hBAllNeededCry_ = 0;
  hBNumReadCry_ = 0;
  hBE3x3_ = 0;
  hBE3x3Moving_ = 0;
  hBCryOnBeam_ = 0;
  hBMaxEneCry_ = 0;
  hBReadCryErrors_ = 0;
  hBE1vsCry_ = 0;
  hBE3x3vsCry_ = 0;
  hBEntriesvsCry_ = 0;
  hBcryDone_ = 0;
  hBBeamCentered_ = 0;
  hbTBmoving_ = 0;
  hbE1MaxCry_ = 0;
  hbDesync_ = 0;
  pBCriInBeamEvents_ =0;

  mui_->setCurrentFolder( "EcalEndcap/EEBeamCaloClient" );
  DaqMonitorBEInterface* dbe = mui_->getBEInterface();
  if ( meEEBCaloRedGreen_) dbe->removeElement( meEEBCaloRedGreen_->getName() );
  meEEBCaloRedGreen_ = 0;
  if ( meEEBCaloRedGreenReadCry_) dbe->removeElement( meEEBCaloRedGreenReadCry_->getName() );
  meEEBCaloRedGreenReadCry_ = 0;
  if( meEEBCaloRedGreenSteps_ ) dbe->removeElement (  meEEBCaloRedGreenSteps_->getName() );
  meEEBCaloRedGreenSteps_ = 0;
}

bool EEBeamCaloClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov) {

  bool status = true;

  EcalLogicID ecid;

  MonOccupancyDat o;
  map<EcalLogicID, MonOccupancyDat> dataset;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    cout << " SM=" << ism << endl;

    const float n_min_tot = 1000.;

    float num01, num02;
    float mean01;

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

        num01 = num02 = -1.;
        mean01 = -1.;

        bool update_channel = false;

        if ( hBCryOnBeam_ && hBCryOnBeam_->GetEntries() >= n_min_tot ) {
          num01 = hBCryOnBeam_->GetBinContent(ie, ip);
          update_channel = true;
        }

        if ( hBMaxEneCry_ && hBMaxEneCry_->GetEntries() >= n_min_tot ) {
          num02 = hBMaxEneCry_->GetBinContent(ie, ip);
          update_channel = true;
        }

        mean01 = 0.;
        //int cry = ip+20*(ie-1);
        int ic = (ip-1) + 20*(ie-1) + 1;
        int step = 0;
        if (hBcryDone_){ step = (int) hBcryDone_->GetBinContent(ic);}
        if( step > 0 && step < 86){
	//if(hBE3x3vsCry_){mean01 = hBE3x3vsCry_->GetBinContent(step);}// E in the 3x3
	if( hBE1vsCry_ ){mean01 = hBE1vsCry_->GetBinContent(ic);} // E1
        }
        //if(mean01 >0){cout<<"cry: "<<ic<<" ie: "<<ie<<" ip: "<<ip<<" mean: "<< mean01<<endl;}

        if ( update_channel ) {

          if ( ie == 1 && ip == 1 ) {
	//if ( mean01 !=0) {

            cout << "Preparing dataset for SM=" << ism << endl;

            cout << "CryOnBeam (" << ie << "," << ip << ") " << num01  << endl;
            cout << "MaxEneCry (" << ie << "," << ip << ") " << num02  << endl;
	  cout << "E1 ("        << ie << "," << ip << ") " << mean01 << endl;

            cout << endl;

          }

          o.setEventsOverHighThreshold(int(num01));
          o.setEventsOverLowThreshold(int(num02));

          o.setAvgEnergy(mean01);

          if ( econn ) {
            try {
              ecid = LogicID::getEcalLogicID("EB_crystal_number", ism, ic);
              dataset[ecid] = o;
            } catch (runtime_error &e) {
              cerr << e.what() << endl;
            }
          }

        }

      }
    }

  }

  if ( econn ) {
    try {
      cout << "Inserting MonOccupancyDat ..." << flush;
      if ( dataset.size() != 0 ) econn->insertDataArraySet(&dataset, moniov);
      cout << "done." << endl;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

  return status;

}

void EEBeamCaloClient::subscribe(void){

  if ( verbose_ ) cout << "EEBeamCaloClient: subscribe" << endl;

  Char_t histo[200];

  for (int i = 0; i < cryInArray_ ; i++){
    sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT pulse profile cry %01d", i+1);
    mui_->subscribe(histo);
    sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT pulse profile in G12 cry %01d", i+1);
    mui_->subscribe(histo);
    sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT found gains cry %01d", i+1);
    mui_->subscribe(histo);
    sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT rec energy cry %01d", i+1);
    mui_->subscribe(histo);
    // sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT pulse profile moving table cry %01d", i+1);
    //mui_->subscribe(histo);
    //sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT pulse profile in G12 moving table cry %01d", i+1);
    //mui_->subscribe(histo);
    //sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT found gains moving table cry %01d", i+1);
    //mui_->subscribe(histo);
    //sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT rec energy moving table cry %01d", i+1);
    //mui_->subscribe(histo);
  }

  //     for(int u=0; u< 1701;u++){
  //       sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EnergyHistos/EEBCT rec Ene sum 3x3 cry: %04d",u);
  //          mui_->subscribe(histo);
  //       sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EnergyHistos/EEBCT rec Energy1 cry: %04d",u);
  //          mui_->subscribe(histo);
  //     }


  sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT readout crystals");
  mui_->subscribe(histo);
  //  sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT readout crystals table moving");
  //mui_->subscribe(histo);
  sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT all needed crystals readout");
  mui_->subscribe(histo);
  sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT readout crystals number");
  mui_->subscribe(histo);
  sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT rec Ene sum 3x3");
  mui_->subscribe(histo);
  sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT rec Ene sum 3x3 table moving");
  mui_->subscribe(histo);
  sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT crystal on beam");
  mui_->subscribe(histo);
  sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT crystal with maximum rec energy");
  mui_->subscribe(histo);
  sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT table is moving");
  mui_->subscribe(histo);
  sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT crystals done");
  mui_->subscribe(histo);
  sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT crystal in beam vs event");
  mui_->subscribe(histo);
  sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT readout crystals errors");
  mui_->subscribe(histo);
  sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT average rec energy in the single crystal");
  mui_->subscribe(histo);
  sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT average rec energy in the 3x3 array");
  mui_->subscribe(histo);
  sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT number of entries");
  mui_->subscribe(histo);
  sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT energy deposition in the 3x3");
  mui_->subscribe(histo);
  sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT E1 in the max cry");
  mui_->subscribe(histo);
  sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT Desynchronization vs step");
  mui_->subscribe(histo);

  if ( collateSources_ ) {

    if ( verbose_ ) cout << "EEBeamCaloClient: collate" << endl;

  }

}

void EEBeamCaloClient::subscribeNew(void){

  Char_t histo[200];

  for (int i = 0; i < cryInArray_ ; i++){
    sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT pulse profile cry %01d", i+1);
    mui_->subscribe(histo);
    sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT pulse profile in G12 cry %01d", i+1);
    mui_->subscribe(histo);
    sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT found gains cry %01d", i+1);
    mui_->subscribe(histo);
    sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT rec energy cry %01d", i+1);
    mui_->subscribe(histo);
    //sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT pulse profile moving table cry %01d", i+1);
    //mui_->subscribe(histo);
    //sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT pulse profile in G12 moving table cry %01d", i+1);
    //mui_->subscribe(histo);
    //sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT found gains moving table cry %01d", i+1);
    //mui_->subscribe(histo);
    //sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT rec energy moving table cry %01d", i+1);
    //mui_->subscribe(histo);
  }

  //     for(int u=0; u< 1701;u++){
  //       sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EnergyHistos/EEBCT rec Ene sum 3x3 cry: %04d",u);
  //          mui_->subscribe(histo);
  //       sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EnergyHistos/EEBCT rec Energy1 cry: %04d",u);
  //          mui_->subscribe(histo);
  //     }


  sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT readout crystals");
  mui_->subscribe(histo);
  // sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT readout crystals table moving");
  //mui_->subscribe(histo);
  sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT all needed crystals readout");
  mui_->subscribe(histo);
  sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT readout crystals number");
  mui_->subscribe(histo);
  sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT rec Ene sum 3x3");
  mui_->subscribe(histo);
  sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT rec Ene sum 3x3 table moving");
  mui_->subscribe(histo);
  sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT crystal on beam");
  mui_->subscribe(histo);
  sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT crystal with maximum rec energy");
  mui_->subscribe(histo);
  sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT table is moving");
  mui_->subscribe(histo);
  sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT crystals done");
  mui_->subscribe(histo);
  sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT crystal in beam vs event");
  mui_->subscribe(histo);
  sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT readout crystals errors");
  mui_->subscribe(histo);
  sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT average rec energy in the single crystal");
  mui_->subscribe(histo);
  sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT average rec energy in the 3x3 array");
  mui_->subscribe(histo);
  sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT number of entries");
  mui_->subscribe(histo);
  sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT energy deposition in the 3x3");
  mui_->subscribe(histo);
  sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT E1 in the max cry");
  mui_->subscribe(histo);
  sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT Desynchronization vs step");
  mui_->subscribe(histo);
}

void EEBeamCaloClient::unsubscribe(void){

  if ( verbose_ ) cout << "EEBeamCaloClient: unsubscribe" << endl;

  Char_t histo[200];

  for (int i = 0; i < cryInArray_ ; i++){
    sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT pulse profile cry %01d", i+1);
    mui_->unsubscribe(histo);
    sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT pulse profile in G12 cry %01d", i+1);
    mui_->unsubscribe(histo);
    sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT found gains cry %01d", i+1);
    mui_->unsubscribe(histo);
    sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT rec energy cry %01d", i+1);
    mui_->unsubscribe(histo);
    // sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT pulse profile moving table cry %01d", i+1);
    //mui_->unsubscribe(histo);
    //sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT pulse profile in G12 moving table cry %01d", i+1);
    //mui_->unsubscribe(histo);
    //sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT found gains moving table cry %01d", i+1);
    //mui_->unsubscribe(histo);
    //sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT rec energy moving table cry %01d", i+1);
    //mui_->unsubscribe(histo);
  }

  //     for(int u=0; u< 1701;u++){
  //       sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EnergyHistos/EEBCT rec Ene sum 3x3 cry: %04d",u);
  //          mui_->unsubscribe(histo);
  //       sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EnergyHistos/EEBCT rec Energy1 cry: %04d",u);
  //          mui_->unsubscribe(histo);
  //     }


  sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT readout crystals");
  mui_->unsubscribe(histo);
  //sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT readout crystals table moving");
  //mui_->unsubscribe(histo);
  sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT all needed crystals readout");
  mui_->unsubscribe(histo);
  sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT readout crystals number");
  mui_->unsubscribe(histo);
  sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT rec Ene sum 3x3");
  mui_->unsubscribe(histo);
  sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT rec Ene sum 3x3 table moving");
  mui_->unsubscribe(histo);
  sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT crystal on beam");
  mui_->unsubscribe(histo);
  sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT crystal with maximum rec energy");
  mui_->unsubscribe(histo);
  sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT table is moving");
  mui_->unsubscribe(histo);
  sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT crystals done");
  mui_->unsubscribe(histo);
  sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT crystal in beam vs event");
  mui_->unsubscribe(histo);
  sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT readout crystals errors");
  mui_->unsubscribe(histo);
  sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT average rec energy in the single crystal");
  mui_->unsubscribe(histo);
  sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT average rec energy in the 3x3 array");
  mui_->unsubscribe(histo);
  sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT number of entries");
  mui_->unsubscribe(histo);
  sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT energy deposition in the 3x3");
  mui_->unsubscribe(histo);
  sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT E1 in the max cry");
  mui_->unsubscribe(histo);
  sprintf(histo, "*/EcalEndcap/EEBeamCaloTask/EEBCT Desynchronization vs step");
  mui_->unsubscribe(histo);
  if ( collateSources_ ) {

    if ( verbose_ ) cout << "EEBeamCaloClient: uncollate" << endl;

  }

}

void EEBeamCaloClient::softReset(void){

}

void EEBeamCaloClient::analyze(void){

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( verbose_ ) cout << "EEBeamCaloClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;
  }

  Char_t histo[200];

  MonitorElement* me = 0;

  // MonitorElement* meCD;
  if ( collateSources_ ) {;}
  else{ sprintf(histo, (prefixME_+"EcalEndcap/EEBeamCaloTask/EEBCT crystals done").c_str() ); }
  //meCD = mui_->get(histo);
  me = mui_->get(histo);
  hBcryDone_ = UtilsClient::getHisto<TH1F*>( me, cloneME_, hBcryDone_ );

  //MonitorElement* meCryInBeam;
  if ( collateSources_ ) {;}
  else { sprintf(histo, (prefixME_+"EcalEndcap/EEBeamCaloTask/EEBCT crystal on beam").c_str() ); }
  //meCryInBeam = mui_->get(histo);
  me = mui_->get(histo);
  hBCryOnBeam_ = UtilsClient::getHisto<TH2F*>( me, cloneME_, hBCryOnBeam_);

  //MonitorElement* allNeededCry;
  if ( collateSources_ ) {;}
  else {sprintf(histo, (prefixME_+"EcalEndcap/EEBeamCaloTask/EEBCT all needed crystals readout").c_str() ); }
  //allNeededCry= mui_->get(histo);
  me = mui_->get(histo);
  hBAllNeededCry_ = UtilsClient::getHisto<TH1F*>( me, cloneME_, hBAllNeededCry_);

  if ( collateSources_ ) {;}
  else {sprintf(histo, (prefixME_+"EcalEndcap/EEBeamCaloTask/EEBCT readout crystals number").c_str() ); }
  //allNeededCry= mui_->get(histo);
  me = mui_->get(histo);
  hBNumReadCry_ = UtilsClient::getHisto<TH1F*>( me, cloneME_, hBNumReadCry_);

  //MonitorElement* RecEne3x3;
  if ( collateSources_ ) {;}
  else { sprintf(histo, (prefixME_+"EcalEndcap/EEBeamCaloTask/EEBCT rec Ene sum 3x3").c_str() ); }
  //RecEne3x3= mui_->get(histo);
  me = mui_->get(histo);
  hBE3x3_ = UtilsClient::getHisto<TH1F*>( me, cloneME_, hBE3x3_);

  //MonitorElement* ErrRedCry;
  if ( collateSources_ ) {;}
  else { sprintf(histo, (prefixME_+"EcalEndcap/EEBeamCaloTask/EEBCT readout crystals errors").c_str() ); }
  //ErrRedCry = mui_->get(histo);
  me = mui_->get(histo);
  hBReadCryErrors_ = UtilsClient::getHisto<TH1F*>( me, cloneME_, hBReadCryErrors_);

  //  MonitorElement* RecEne1;
  if ( collateSources_ ) {;}
  else { sprintf(histo, (prefixME_+"EcalEndcap/EEBeamCaloTask/EEBCT rec energy cry 5").c_str() ); }
  //RecEne1= mui_->get(histo);
  me = mui_->get(histo);
  hBEne1_ = UtilsClient::getHisto<TH1F*>( me, cloneME_, hBEne1_);

  if ( collateSources_ ) {;}
  else { sprintf(histo, (prefixME_+"EcalEndcap/EEBeamCaloTask/EEBCT crystal with maximum rec energy").c_str() ); }
  me = mui_->get(histo);
  hBMaxEneCry_ = UtilsClient::getHisto<TH2F*>( me, cloneME_, hBMaxEneCry_);

  if ( collateSources_ ) {;}
  else { sprintf(histo, (prefixME_+"EcalEndcap/EEBeamCaloTask/EEBCT average rec energy in the 3x3 array").c_str() ); }
  me = mui_->get(histo);
  hBE3x3vsCry_ = UtilsClient::getHisto<TProfile*>( me, cloneME_, hBE3x3vsCry_);

  if ( collateSources_ ) {;}
  else {  sprintf(histo, (prefixME_+"EcalEndcap/EEBeamCaloTask/EEBCT average rec energy in the single crystal").c_str() ); }
  me = mui_->get(histo);
  hBE1vsCry_ = UtilsClient::getHisto<TProfile*>( me, cloneME_, hBE1vsCry_);

  if ( collateSources_ ) {;}
  else {  sprintf(histo, (prefixME_+"EcalEndcap/EEBeamCaloTask/EEBCT number of entries").c_str() ); }
  me = mui_->get(histo);
  hBEntriesvsCry_ = UtilsClient::getHisto<TH1F*>( me, cloneME_, hBEntriesvsCry_);

  if ( collateSources_ ) {;}
  else { sprintf(histo, (prefixME_+"EcalEndcap/EEBeamCaloTask/EEBCT energy deposition in the 3x3").c_str() ); }
  me = mui_->get(histo);
  hBBeamCentered_ = UtilsClient::getHisto<TH2F*>( me, cloneME_, hBBeamCentered_);

  if ( collateSources_ ) {;}
  else { sprintf(histo, (prefixME_+"EcalEndcap/EEBeamCaloTask/EEBCT table is moving").c_str() ); }
  me = mui_->get(histo);
  hbTBmoving_ = UtilsClient::getHisto<TH1F*>( me, cloneME_, hbTBmoving_);

  if ( collateSources_ ) {;}
  else {sprintf(histo, (prefixME_+"EcalEndcap/EEBeamCaloTask/EEBCT crystal in beam vs event").c_str() );}
  me = mui_->get(histo);
  pBCriInBeamEvents_ =  UtilsClient::getHisto<TProfile*>( me, cloneME_, pBCriInBeamEvents_);

  if ( collateSources_ ) {;}
  else {sprintf(histo, (prefixME_+"EcalEndcap/EEBeamCaloTask/EEBCT E1 in the max cry").c_str() );}
  me = mui_->get(histo);
  hbE1MaxCry_ =  UtilsClient::getHisto<TH1F*>( me, cloneME_, hbE1MaxCry_);

  if ( collateSources_ ) {;}
  else {sprintf(histo, (prefixME_+"EcalEndcap/EEBeamCaloTask/EEBCT Desynchronization vs step").c_str() );}
  me = mui_->get(histo);
  hbDesync_ =  UtilsClient::getHisto<TH1F*>( me, cloneME_, hbDesync_);

  if ( collateSources_ ){;}
  else {
    char me_name[200];
    for(int ind = 0; ind < cryInArray_; ind ++){
      sprintf(me_name,"EcalEndcap/EEBeamCaloTask/EEBCT pulse profile in G12 cry %01d", ind+1);
      sprintf(histo, (prefixME_ + me_name).c_str() );
      me = mui_->get(histo);
      hBpulse_[ind] = UtilsClient::getHisto<TProfile*>( me, cloneME_, hBpulse_[ind]);

      sprintf(me_name,"EcalEndcap/EEBeamCaloTask/EEBCT found gains cry %01d", ind+1);
      sprintf(histo, (prefixME_ + me_name).c_str() );
      me = mui_->get(histo);
      hBGains_[ind] = UtilsClient::getHisto<TH1F*>( me, cloneME_, hBGains_[ind]);
    }
  }

  int DoneCry = 0;//if it stays 1 the run is not an autoscan
  if (hBcryDone_){
    for(int cry=1 ; cry<1701 ; cry ++){
      int step = (int) hBcryDone_->GetBinContent(cry);
      if( step>0 ){//this crystal has been scanned or is dbeng scanned
	DoneCry++;
	float E3x3RMS = -1, E3x3 =-1, E1=-1;
	if(hBE3x3vsCry_){
	  //E3x3RMS = hBE3x3vsCry_->GetBinError(step);
	  //E3x3 = hBE3x3vsCry_->GetBinContent(step);
	  E3x3RMS = hBE3x3vsCry_->GetBinError(cry);
	  E3x3 = hBE3x3vsCry_->GetBinContent(cry);
	}
	//if( hBE1vsCry_){E1=hBE1vsCry_->GetBinContent(step);}
	if( hBE1vsCry_){E1=hBE1vsCry_->GetBinContent(cry);}
	bool RMS3x3  =  (  E3x3RMS < RMSEne3x3_ && E3x3RMS >= 0 );
	bool Mean3x3 =  ( fabs( E3x3 - aveEne3x3_ ) < E3x3Th_);
	bool Mean1   =  ( fabs( E1 - aveEne1_ ) < E1Th_ );
	//cout<<"E1: "<<E1<<" E3x3: "<<E3x3<<" E3x3RMS: "<<E3x3RMS<<endl;
	int ieta = ( cry - 1)/20 + 1 ;//+1 for the bin
	int iphi = ( cry - 1)%20 + 1 ;//+1 for the bin
	//fill the RedGreen histo
	if(ieta >0 && iphi >0 ){
	  if(RMS3x3 && Mean3x3 && Mean1) {meEEBCaloRedGreen_->setBinContent(ieta,iphi,1.);}
	  else {meEEBCaloRedGreen_->setBinContent(ieta,iphi,0.);}
	}

	float Entries = -1;
	//if ( hBEntriesvsCry_ ){Entries = hBEntriesvsCry_->GetBinContent(step);}
	if ( hBEntriesvsCry_ ){Entries = hBEntriesvsCry_->GetBinContent(cry);}
	bool Nent = ( Entries * prescaling_  > minEvtNum_ );
	//cout<<"step: "<<step<<" entries: "<<Entries<<endl;
	//cout<<"step -1 entries: "<<hBEntriesvsCry_->GetBinContent(step-1)<<endl;
	//cout<<"step +1 entries: "<<hBEntriesvsCry_->GetBinContent(step+1)<<endl;
	bool readCryOk = true;
	if( hBReadCryErrors_ ) {
	  int step_bin = hBReadCryErrors_->GetXaxis()->FindBin(step);
	  if ( step_bin > 0 && step_bin < hBReadCryErrors_->GetNbinsX() ){
	    if ( hBReadCryErrors_->GetBinContent(step_bin) <= Entries*ReadCryErrThr_ ){readCryOk = true;}
	    else {readCryOk = false;}
	  }
	}

	if(Nent && readCryOk ){ meEEBCaloRedGreenSteps_->setBinContent(step,1,1.);}
	else{ meEEBCaloRedGreenSteps_->setBinContent(step,1,0.);}

	if (readCryOk &&  meEEBCaloRedGreenReadCry_->getBinContent(1,1) != 0.){ meEEBCaloRedGreenReadCry_->setBinContent(1,1, 1.);}
	else if ( !readCryOk ){ meEEBCaloRedGreenReadCry_->setBinContent(1,1, 0.);}
      }// end of if (step>0)
    }//end of loop over cry
  }//end of if(hBcryDone_)

  if(DoneCry == 1){//this is probably not an auotscan or it is the first crystal
    float nEvt = 0;
    if(hBE3x3_){nEvt = hBE3x3_->GetEntries();}
    if(nEvt > 1*prescaling_ && hBE3x3_ && hBEne1_ && hBCryOnBeam_ && meEEBCaloRedGreen_){//check for mean and RMS
      bool RMS3x3  =  ( hBE3x3_->GetRMS() < RMSEne3x3_ );
      bool Mean3x3 =  ( fabs( hBE3x3_->GetMean() - aveEne3x3_ ) < E3x3Th_ );
      bool Mean1   =  ( fabs( hBEne1_->GetMean() - aveEne1_ ) < E1Th_ );
      //fill the RedGreen histo
      int ieta=0,iphi=0;
      float found =0; //there should be just one bin filled but...
      for (int b_eta =1; b_eta<86; b_eta++){
	for (int b_phi =1; b_phi<21; b_phi++){
	  float bc = hBCryOnBeam_->GetBinContent(b_eta,b_phi);//FIX ME check if this is the correct binning
	  if(bc > found){ found =bc; ieta = b_eta; iphi= b_phi;}
	}
      }
      if(ieta >0 && iphi >0 ){
	if(RMS3x3 && Mean3x3 && Mean1) {meEEBCaloRedGreen_->setBinContent(ieta,iphi,1.);}
	else {meEEBCaloRedGreen_->setBinContent(ieta,iphi,0.);}
      }
    }
    if(hBReadCryErrors_){
      float nErr = hBReadCryErrors_->GetBinContent(1);// for a non autoscan just the first bin should be filled
      if( nErr > nEvt*ReadCryErrThr_ ){ meEEBCaloRedGreenReadCry_->setBinContent(1,1,0.);}
      else { meEEBCaloRedGreenReadCry_->setBinContent(1,1,1.);}
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
  // 	if(RMS3x3 && Mean3x3 && Mean1) {meEEBCaloRedGreen_->setBinContent(ieta,iphi,1.);}
  // 	else {meEEBCaloRedGreen_->setBinContent(ieta,iphi,0.);}
  //       }
  //     }
  //     float nErr = ErrRedCry->getBinContent(1);// for a non autoscan just the first bin should be filled
  //     if( nErr > nEvt*ReadCryErrThr_ ){ meEEBCaloRedGreenReadCry_->setBinContent(1,1,0.);}
  //     else { meEEBCaloRedGreenReadCry_->setBinContent(1,1,1.);}
  //   }


}

void EEBeamCaloClient::htmlOutput(int run, string htmlDir, string htmlName){

  cout << "Preparing EEBeamCaloClient html output ..." << endl;

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


  htmlFile <<  "<a href=\"#qua_plots\"> Quality plots </a>" << endl;
  htmlFile << "<p>" << endl;
  htmlFile <<  "<a href=\"#gen_plots\"> General plots </a>" << endl;
  htmlFile << "<p>" << endl;
  htmlFile <<  "<a href=\"#fix_plots\"> Plots for the fixed crystal runs </a>" << endl;
  htmlFile << "<p>" << endl;
  htmlFile <<  "<a href=\"#aut_plots\"> Plots for the autoscan runs </a>" << endl;
  htmlFile << "<p>" << endl;

  htmlFile << "<hr>" << endl;
  htmlFile << "<p>" << endl;



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

  TH2I dummyStep( "dummyStep", "dummy2 for sm", 86, 1., 87., 1, 0., 1. );
  if(hBcryDone_){
    for(int cry=1 ; cry<1701 ; cry ++){
      int step = (int) hBcryDone_->GetBinContent(cry);
      if (step >0 ){dummyStep.SetBinContent( step+1, 1, cry );}
      //cout<<"cry: "<<cry<<" step: "<<step <<"  histo: "<<dummyStep.GetBinContent(step+1,1)<<endl;}
    }
  }
  //dummyStep.SetBinContent( 6, 1, 1699 );
  //dummyStep.SetBinContent( 85, 1, 1698 );
  dummyStep.SetMarkerSize(2);
  dummyStep.SetMinimum(0.1);

  //useful for both autoscan and non autoscan
  string RedGreenSMImg,RedGreenImg,RedGreenAutoImg, numCryReadImg, cryReadErrImg, E1MaxCryImg, DesyncImg;
  string cryOnBeamImg, cryMaxEneImg, ratioImg;
  // useful for non autoscan
  string ene1Img, ene3x3Img, EEBeamCentered;
  string pulseshapeImg, gainImg;
  //useful for autoscan
  string cryDoneImg, EntriesVScryImg, E1vsCryImg, E3x3vsCryImg;
  string cryVSeventImg, TBmoving;

  string meName,imgName1,imgName2,imgName3;
  TH2F* obj2f = 0;
  TH1F* obj1f = 0;
  TProfile* objp1 = 0;

  ///*****************************************************************************///
  htmlFile << "<br>" << endl;
  htmlFile <<  "<a name=\"qua_plots\"> <B> Quality plots </B> </a> " << endl;
  htmlFile << "</br>" << endl;
  ///*****************************************************************************///

  ////////////////////////////////////////////////////////////////////////////////
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
  htmlFile << "<tr align=\"center\">" << endl;
  // quality red-green histo
  obj2f = UtilsClient::getHisto<TH2F*>( meEEBCaloRedGreen_ );
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
    gStyle->SetPalette(6, pCol3);
    obj2f->GetXaxis()->SetNdivisions(17);
    obj2f->GetYaxis()->SetNdivisions(4);
    can->SetGridx();
    can->SetGridy();
    obj2f->SetMinimum(-0.00000001);
    obj2f->SetMaximum(6.0);
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
  obj2f = UtilsClient::getHisto<TH2F*>( meEEBCaloRedGreenReadCry_ );
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
    gStyle->SetPalette(6, pCol3);
    obj2f->GetXaxis()->SetNdivisions(0);
    obj2f->GetYaxis()->SetNdivisions(0);
    obj2f->SetMinimum(-0.00000001);
    obj2f->SetMaximum(6.0);
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
  htmlFile << "</table>" << endl;
  htmlFile << "<br>" << endl;
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
  htmlFile << "<tr align=\"center\">" << endl;
  // quality entries and read cry red-green histo for autoscan
  obj2f = UtilsClient::getHisto<TH2F*>( meEEBCaloRedGreenSteps_ );
  if ( obj2f ) {

    TCanvas* can = new TCanvas("can", "Temp", 5*csize, csize);
    meName = obj2f->GetName();
    for ( unsigned int i = 0; i < meName.size(); i++ ) {
      if ( meName.substr(i, 1) == " " )  {
	meName.replace(i, 1, "_");
      }
    }
    RedGreenAutoImg = meName + ".png";
    imgName1 = htmlDir + RedGreenAutoImg;

    can->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(6, pCol3);
    obj2f->GetXaxis()->SetNdivisions(86);
    obj2f->GetYaxis()->SetNdivisions(0);
    //obj2f->SetTitle("");
    can->SetGridx();
    //can->SetGridy();
    obj2f->SetMinimum(-0.00000001);
    obj2f->SetMaximum(6.0);
    obj2f->GetXaxis()->SetTitle("step in the scan");
    obj2f->Draw("col");
    dummyStep.Draw("text90,same");
    can->Update();
    can->SaveAs(imgName1.c_str());
    delete can;
  }
  if ( imgName1.size() != 0 )
    htmlFile << "<td><img src=\"" << RedGreenAutoImg  << "\"></td>" << endl;
  else
    htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

  htmlFile << "</tr>" << endl;
  htmlFile << "</table>" << endl;
  htmlFile << "<br>" << endl;

  ////////////////////////////////////////////////////////////////////////////////

  ///*****************************************************************************///
  htmlFile << "<br>" << endl;
  htmlFile <<  "<a name=\"gen_plots\"> <B> General plots </B> </a>" << endl;
  htmlFile << "</br>" << endl;
  ///*****************************************************************************///

  ////////////////////////////////////////////////////////////////////////////////
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
  htmlFile << "<tr align=\"center\">" << endl;
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
    AdjustRange(obj1f);
    obj1f->GetXaxis()->SetTitle("number of read crystals");
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
    gStyle->SetOptStat("e");
    obj1f->GetXaxis()->SetTitle("step in the scan");
    AdjustRange(obj1f);
    obj1f->Draw();
    can->Update();
    can->SaveAs(imgName1.c_str());
    delete can;
  }
  if ( imgName1.size() != 0 )
    htmlFile << "<td><img src=\"" << cryReadErrImg << "\"></td>" << endl;
  else
    htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;


 // E1 of the max cry: cryReadErrImg;
  obj1f = hbE1MaxCry_;
  if ( obj1f ) {
    TCanvas* can = new TCanvas("can", "Temp", csize, csize);
    meName = obj1f->GetName();

    for ( unsigned int i = 0; i < meName.size(); i++ ) {
      if ( meName.substr(i, 1) == " " )  {
	meName.replace(i, 1, "_");
      }
    }
    E1MaxCryImg = meName + ".png";
    imgName1 = htmlDir +  E1MaxCryImg;

    can->cd();
    obj1f->SetStats(kTRUE);
    gStyle->SetOptStat("e");
    AdjustRange(obj1f);
    obj1f->GetXaxis()->SetTitle("rec Ene (ADC)");
    obj1f->Draw();
    can->Update();
    can->SaveAs(imgName1.c_str());
    delete can;
  }
  if ( imgName1.size() != 0 )
    htmlFile << "<td><img src=\"" <<  E1MaxCryImg << "\"></td>" << endl;
  else
    htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

 // Desynchronization
  obj1f = hbDesync_;
  if ( obj1f ) {
    TCanvas* can = new TCanvas("can", "Temp", csize, csize);
    meName = obj1f->GetName();

    for ( unsigned int i = 0; i < meName.size(); i++ ) {
      if ( meName.substr(i, 1) == " " )  {
	meName.replace(i, 1, "_");
      }
    }
    DesyncImg = meName + ".png";
    imgName1 = htmlDir +  DesyncImg;

    can->cd();
    obj1f->SetStats(kTRUE);
    gStyle->SetOptStat("e");
    AdjustRange(obj1f);
    obj1f->GetXaxis()->SetTitle("step");
    obj1f->GetYaxis()->SetTitle("Desynchronized events");
    obj1f->Draw();
    can->Update();
    can->SaveAs(imgName1.c_str());
    delete can;
  }
  if ( imgName1.size() != 0 )
    htmlFile << "<td><img src=\"" << DesyncImg  << "\"></td>" << endl;
  else
    htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

  htmlFile << "</tr>" << endl;
  htmlFile << "</table>" << endl;
  htmlFile << "<br>" << endl;
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
  htmlFile << "<tr align=\"center\">" << endl;
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
    obj2f->SetMinimum(0.00000001);
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
    obj2f->SetMinimum(0.00000001);
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
  htmlFile << "</table>" << endl;
  htmlFile << "<br>" << endl;
  ////////////////////////////////////////////////////////////////////////////////

  ///*****************************************************************************///
  htmlFile << "<br>" << endl;
  htmlFile <<  "<a name=\"fix_plots\"> <B> Plots for the fixed crystal runs </B> </a>" << endl;
  htmlFile << "</br>" << endl;
  ///*****************************************************************************///

  ////////////////////////////////////////////////////////////////////////////////////////////
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
  htmlFile << "<tr align=\"center\">" << endl;
  //ene1Img, ene3x3Img, EEBeamCentered;
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
    AdjustRange(obj1f);
    obj1f->GetXaxis()->SetTitle("rec ene (ADC)");
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
    AdjustRange(obj1f);
    obj1f->GetXaxis()->SetTitle("rec ene (ADC)");
    obj1f->Draw();
    can->Update();
    can->SaveAs(imgName1.c_str());
    delete can;
  }
  if ( imgName1.size() != 0 )
    htmlFile << "<td><img src=\"" << ene3x3Img << "\"></td>" << endl;
  else
    htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

  //EEBeamCentered
  obj2f = hBBeamCentered_;
  if ( obj2f ) {

    TCanvas* can = new TCanvas("can", "Temp", csize, csize);
    meName = obj2f->GetName();

    for ( unsigned int i = 0; i < meName.size(); i++ ) {
      if ( meName.substr(i, 1) == " " )  {
	meName.replace(i, 1, "_");
      }
    }
    EEBeamCentered = meName + ".png";
    imgName1 = htmlDir + EEBeamCentered;

    can->cd();
    gStyle->SetOptStat(" ");
    obj2f->SetLineColor(kRed);
    obj2f->SetFillColor(kRed);
    obj2f->GetXaxis()->SetTitle("\\Delta \\eta");
    obj2f->GetYaxis()->SetTitle("\\Delta \\phi");

    obj2f->Draw("box");
    can->Update();
    can->SaveAs(imgName1.c_str());
    delete can;
  }
  if ( imgName1.size() != 0 )
    htmlFile << "<td><img src=\"" << EEBeamCentered << "\"></td>" << endl;
  else
    htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

  htmlFile << "</tr>" << endl;
  htmlFile << "</table>" << endl;
  htmlFile << "<br>" << endl;
  ////////////////////////////////////////////////////////////////////////////////
  string pulseImg[cryInArray_], gainsImg[cryInArray_], pulseImgF[cryInArray_], gainsImgF[cryInArray_];
  for(int ind = 0; ind < cryInArray_; ind ++){
    objp1 = hBpulse_[ind];
    if ( objp1 ) {
      TCanvas* can = new TCanvas("can", "Temp", csize, csize);
      meName = objp1->GetName();

      for ( unsigned int i = 0; i < meName.size(); i++ ) {
	if ( meName.substr(i, 1) == " " )  {
	  meName.replace(i, 1, "_");
	}
      }
      pulseImg[ind] = meName + ".png";
      pulseImgF[ind] = htmlDir + pulseImg[ind] ;

      can->cd();
      objp1->SetStats(kTRUE);
      gStyle->SetOptStat("e");
      objp1->GetXaxis()->SetTitle("#sample");
      objp1->GetYaxis()->SetTitle("ADC");
      objp1->Draw();
      can->Update();
      can->SaveAs( pulseImgF[ind].c_str());
      delete can;
    }

    obj1f = hBGains_[ind];
    if ( obj1f ) {
      TCanvas* can = new TCanvas("can", "Temp", csize, csize);
      meName = obj1f->GetName();

      for ( unsigned int i = 0; i < meName.size(); i++ ) {
	if ( meName.substr(i, 1) == " " )  {
	  meName.replace(i, 1, "_");
	}
      }
      gainsImg[ind] = meName + ".png";
      gainsImgF[ind] = htmlDir + gainsImg[ind];

      can->cd();
      obj1f->SetStats(kTRUE);
      gStyle->SetOptStat(1110);
      obj1f->GetXaxis()->SetTitle("gain");
      if(obj1f->GetEntries() != 0 ){gStyle->SetOptLogy(1);}
      obj1f->Draw();
      can->Update();
      can->SaveAs(gainsImgF[ind].c_str());
      gStyle->SetOptLogy(0);
      delete can;
    }

  }

  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
  htmlFile << "<tr align=\"center\">" << endl;
  htmlFile << "<td>" << endl;
  int row = (int) sqrt(float(cryInArray_));
  ///// sub table /////////////////
  htmlFile << "<table border=\"4\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
  htmlFile << "<tr align=\"center\">" << endl;

  for(int ind=0; ind < cryInArray_; ind++){
    if ( pulseImgF[ind].size() != 0 )
      htmlFile << "<td><img src=\"" << pulseImg[ind] << "\"></td>" << endl;
    else
      htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;
    if ( (ind+1) % row == 0){htmlFile << "</tr>" << endl;}
  }
  htmlFile << "</table>" << endl;
  htmlFile << "<br>" << endl;
  /////
  htmlFile << "</td>" << endl;
  htmlFile << "<td>" << endl;
  ////sub table /////////////////
  htmlFile << "<table border=\"4\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
  htmlFile << "<tr align=\"center\">" << endl;

  for(int ind=0; ind < cryInArray_; ind++){
    if ( gainsImgF[ind].size() != 0 )
      htmlFile << "<td><img src=\"" << gainsImg[ind] << "\"></td>" << endl;
    else
      htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;
    if ( (ind+1) % row == 0){htmlFile << "</tr>" << endl;}
  }
  htmlFile << "</table>" << endl;
  htmlFile << "<br>" << endl;
  ///////////
  htmlFile << "</td>" << endl;
  htmlFile << "</tr>" << endl;
  htmlFile << "</table>" << endl;
  htmlFile << "<br>" << endl;
  //////////////////////////////////////////////////////////

  ///*****************************************************************************///
  htmlFile << "<br>" << endl;
  htmlFile <<  "<a name=\"aut_plots\"> <B> Plots for the autoscan runs </B> </a>" << endl;
  htmlFile << "</br>" << endl;
  ///*****************************************************************************///

  //////////////////////////////////////////////////////////
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
  htmlFile << "<tr align=\"center\">" << endl;
  // cryDoneImg,EntriesVScryImg E1vsCryImg, E3x3vsCryImg
  //cryDoneImg
  obj1f = hBcryDone_  ;
  if ( obj1f ) {
    TCanvas* can = new TCanvas("can", "Temp", int(1.618*csize), csize);
    meName = obj1f->GetName();

    for ( unsigned int i = 0; i < meName.size(); i++ ) {
      if ( meName.substr(i, 1) == " " )  {
	meName.replace(i, 1, "_");
      }
    }
    cryDoneImg = meName + ".png";
    imgName1 = htmlDir + cryDoneImg;

    can->cd();
    obj1f->SetStats(kTRUE);
    gStyle->SetOptStat("e");
    obj1f->GetXaxis()->SetTitle("crystal");
    obj1f->GetYaxis()->SetTitle("step in the scan");
    AdjustRange(obj1f);
    obj1f->Draw();
    can->Update();
    can->SaveAs(imgName1.c_str());
    delete can;
  }
  if ( imgName1.size() != 0 )
    htmlFile << "<td><img src=\"" << cryDoneImg << "\"></td>" << endl;
  else
    htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

  //EntriesVScryImg
  obj1f = hBEntriesvsCry_ ;
  if ( obj1f ) {
    TCanvas* can = new TCanvas("can", "Temp", int(1.618*csize), csize);
    meName = obj1f->GetName();

    for ( unsigned int i = 0; i < meName.size(); i++ ) {
      if ( meName.substr(i, 1) == " " )  {
	meName.replace(i, 1, "_");
      }
    }
    EntriesVScryImg = meName + ".png";
    imgName1 = htmlDir + EntriesVScryImg;

    can->cd();
    obj1f->SetStats(kTRUE);
    gStyle->SetOptStat("e");
    obj1f->GetXaxis()->SetTitle("crystal");
    obj1f->GetYaxis()->SetTitle("number of events (prescaled)");

    if(obj1f->GetEntries() != 0 ){gStyle->SetOptLogy(1);}
    AdjustRange(obj1f);
    obj1f->Draw();
    can->Update();
    can->SaveAs(imgName1.c_str());
    delete can;
    gStyle->SetOptLogy(0);
  }
  if ( imgName1.size() != 0 )
    htmlFile << "<td><img src=\"" << EntriesVScryImg << "\"></td>" << endl;
  else
    htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;


  htmlFile << "</tr>" << endl;
 ////////////////////////////////////////////////////////////////////////////////
  htmlFile << "<tr align=\"center\">" << endl;
  //E1vsCryImg
  objp1 = hBE1vsCry_ ;
  if ( objp1 ) {
    TCanvas* can = new TCanvas("can", "Temp", int(1.618*csize), csize);
    meName = objp1->GetName();

    for ( unsigned int i = 0; i < meName.size(); i++ ) {
      if ( meName.substr(i, 1) == " " )  {
	meName.replace(i, 1, "_");
      }
    }
     E1vsCryImg = meName + ".png";
    imgName1 = htmlDir +  E1vsCryImg;

    can->cd();
    objp1->SetStats(kTRUE);
    objp1->GetXaxis()->SetTitle("crystal");
    objp1->GetYaxis()->SetTitle("rec energy (ADC)");
    gStyle->SetOptStat("e");
    AdjustRange(objp1);
    objp1->Draw();
    can->Update();
    can->SaveAs(imgName1.c_str());
    delete can;
  }
  if ( imgName1.size() != 0 )
    htmlFile << "<td><img src=\"" <<  E1vsCryImg << "\"></td>" << endl;
  else
    htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

  //E3x3vsCryImg
  objp1 = hBE3x3vsCry_ ;
  if ( objp1 ) {
    TCanvas* can = new TCanvas("can", "Temp", int(1.618*csize), csize);
    meName = objp1->GetName();

    for ( unsigned int i = 0; i < meName.size(); i++ ) {
      if ( meName.substr(i, 1) == " " )  {
	meName.replace(i, 1, "_");
      }
    }
    E3x3vsCryImg = meName + ".png";
    imgName1 = htmlDir +  E3x3vsCryImg;

    can->cd();
    objp1->SetStats(kTRUE);
    gStyle->SetOptStat("e");
    objp1->GetXaxis()->SetTitle("crystal");
    objp1->GetYaxis()->SetTitle("rec energy (ADC)");
    AdjustRange(objp1);
    objp1->Draw();
    can->Update();
    can->SaveAs(imgName1.c_str());
    delete can;
  }
  if ( imgName1.size() != 0 )
    htmlFile << "<td><img src=\"" <<  E3x3vsCryImg << "\"></td>" << endl;
  else
    htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;


  htmlFile << "</tr>" << endl;
  htmlFile << "</table>" << endl;
  htmlFile << "<br>" << endl;
    ////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
  htmlFile << "<tr align=\"center\">" << endl;
  // cryVSeventImg, TBmoving;
  objp1 = pBCriInBeamEvents_;
  if ( objp1 ) {
    TCanvas* can = new TCanvas("can", "Temp", 3*csize, csize);
    meName = objp1->GetName();

    for ( unsigned int i = 0; i < meName.size(); i++ ) {
      if ( meName.substr(i, 1) == " " )  {
	meName.replace(i, 1, "_");
      }
    }
    cryVSeventImg = meName + ".png";
    imgName1 = htmlDir +  cryVSeventImg;

    can->cd();
    //objp1->SetStats(kTRUE);
    //gStyle->SetOptStat("e");

    float dd=0;
    int mbin =0;
    for( int bin=1; bin < objp1->GetNbinsX()+1; bin++ ){
      float temp = objp1->GetBinContent(bin);
      if(temp>0){ dd= temp+0.01; mbin=bin; break;}
    }
    if(mbin >0) { objp1->Fill(20*mbin-1,dd);}

    objp1->GetXaxis()->SetTitle("event");
    objp1->GetYaxis()->SetTitle("crystal in beam");

    AdjustRange(objp1);
    float Ymin = 1701, Ymax =0;
    for( int bin=1; bin < objp1->GetNbinsX()+1; bin++ ){
      float temp = objp1->GetBinContent(bin);
      if(temp >0){
	if(temp < Ymin){Ymin=temp;}
	if(temp > Ymax){Ymax=temp;}
      }
    }
    //cout<<"Ym: "<<Ymin<< " YM: "<<Ymax<<endl;
    if( Ymin < Ymax+1 ){
       for( int bin=1; bin < objp1->GetNbinsX()+1; bin++ ){
	 if( objp1->GetBinError(bin) >0 ){
	   objp1->SetBinContent(bin, (Ymin+Ymax)/2.*objp1->GetBinEntries(bin) );
	   // cout<<"bin: "<<bin<<" rms: "<< objp1->GetBinError(bin) <<"  "<<(Ymin+Ymax)/2<<endl;
	 }
       }
       objp1->GetYaxis()->SetRangeUser(Ymin-1. , Ymax+1.);
    }

    objp1->Draw("e");
    can->Update();
    can->SaveAs(imgName1.c_str());
    delete can;
  }
  if ( imgName1.size() != 0 )
    htmlFile << "<td><img src=\"" <<  cryVSeventImg << "\"></td>" << endl;
  else
    htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

  // TBmoving;
  obj1f = hbTBmoving_;
  if ( obj1f ) {
    TCanvas* can = new TCanvas("can", "Temp", csize, csize);
    meName = obj1f->GetName();

    for ( unsigned int i = 0; i < meName.size(); i++ ) {
      if ( meName.substr(i, 1) == " " )  {
	meName.replace(i, 1, "_");
      }
    }
    TBmoving = meName + ".png";
    imgName1 = htmlDir +  TBmoving;

    can->cd();
    obj1f->SetStats(kTRUE);
    gStyle->SetOptStat("e");
    obj1f->GetXaxis()->SetTitle("table status (0=stable, 1=moving)");
    obj1f->Draw();
    can->Update();
    can->SaveAs(imgName1.c_str());
    delete can;
  }
  if ( imgName1.size() != 0 )
    htmlFile << "<td><img src=\"" <<  TBmoving << "\"></td>" << endl;
  else
    htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;



  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  htmlFile.close();

}

template<class T> void EEBeamCaloClient::AdjustRange( T obj){
  if (obj->GetEntries() == 0) {return;}
  int first_bin = -1, last_bin=-1;
  for( int bin=1; bin < obj->GetNbinsX()+1; bin++ ){
    if( obj->GetBinContent(bin) > 0){
      if(first_bin == -1){first_bin = bin;}
      last_bin = bin;
    }
  }

  if(first_bin < 1 || last_bin < 1){return;}
  if(first_bin > 3){first_bin -= 3;}
  if(last_bin < obj->GetNbinsX() ){last_bin += 3;}

  obj->GetXaxis()->SetRange(first_bin, last_bin);
}

