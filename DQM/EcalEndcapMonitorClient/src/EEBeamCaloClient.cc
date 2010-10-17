/*
 * \file EEBeamCaloClient.cc
 *
 * $Date: 2010/09/07 20:57:53 $
 * $Revision: 1.64 $
 * \author G. Della Ricca
 * \author A. Ghezzi
 *
 */

#include <memory>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <math.h>

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"

#ifdef WITH_ECAL_COND_DB
#include "OnlineDB/EcalCondDB/interface/MonOccupancyDat.h"
#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"
#include "DQM/EcalCommon/interface/LogicID.h"
#endif

#include "DQM/EcalCommon/interface/UtilsClient.h"
#include "DQM/EcalCommon/interface/Numbers.h"

#include "DQM/EcalEndcapMonitorClient/interface/EEBeamCaloClient.h"

EEBeamCaloClient::EEBeamCaloClient(const edm::ParameterSet& ps) {

  // cloneME switch
  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);

  // verbose switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", true);

  // debug switch
  debug_ = ps.getUntrackedParameter<bool>("debug", false);

  // prefixME path
  prefixME_ = ps.getUntrackedParameter<std::string>("prefixME", "");

  // enableCleanup_ switch
  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  // vector of selected Super Modules (Defaults to all 18).
  superModules_.reserve(18);
  for ( unsigned int i = 1; i <= 18; i++ ) superModules_.push_back(i);
  superModules_ = ps.getUntrackedParameter<std::vector<int> >("superModules", superModules_);

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
  for(int u=0;u<cryInArray_;u++) {
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

EEBeamCaloClient::~EEBeamCaloClient() {

}

void EEBeamCaloClient::beginJob(void) {

  dqmStore_ = edm::Service<DQMStore>().operator->();

  if ( debug_ ) std::cout << "EEBeamCaloClient: beginJob" << std::endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EEBeamCaloClient::beginRun(void) {

  if ( debug_ ) std::cout << "EEBeamCaloClient: beginRun" << std::endl;

  jevt_ = 0;

  this->setup();

}

void EEBeamCaloClient::endJob(void) {

  if ( debug_ ) std::cout << "EEBeamCaloClient: endJob, ievt = " << ievt_ << std::endl;

  this->cleanup();

}

void EEBeamCaloClient::endRun(void) {

  if ( debug_ ) std::cout << "EEBeamCaloClient: endRun, jevt = " << jevt_ << std::endl;

  this->cleanup();

}

void EEBeamCaloClient::setup(void) {

  char histo[200];

  dqmStore_->setCurrentFolder( prefixME_ + "/EEBeamCaloClient" );

  if ( meEEBCaloRedGreen_ ) dqmStore_->removeElement( meEEBCaloRedGreen_->getName() );
  sprintf(histo, "EEBCT quality");
  meEEBCaloRedGreen_ = dqmStore_->book2D(histo, histo, 85, 0., 85., 20, 0., 20.);

  meEEBCaloRedGreen_->Reset();

  for ( int ie = 1; ie <= 85; ie++ ) {
    for ( int ip = 1; ip <= 20; ip++ ) {

      meEEBCaloRedGreen_ ->setBinContent( ie, ip, 2. );

    }
  }

  if ( meEEBCaloRedGreenReadCry_ ) dqmStore_->removeElement( meEEBCaloRedGreenReadCry_->getName() );
  sprintf(histo, "EEBCT quality read crystal errors");
  meEEBCaloRedGreenReadCry_ = dqmStore_->book2D(histo, histo, 1, 0., 1., 1, 0., 1.);
  meEEBCaloRedGreenReadCry_->Reset();
  meEEBCaloRedGreenReadCry_ ->setBinContent( 1, 1, 2. );

  if( meEEBCaloRedGreenSteps_ )  dqmStore_->removeElement( meEEBCaloRedGreenSteps_->getName() );
  sprintf(histo, "EEBCT quality entries or read crystals errors");
  meEEBCaloRedGreenSteps_ = dqmStore_->book2D(histo, histo, 86, 1., 87., 1, 0., 1.);
  meEEBCaloRedGreenSteps_->setAxisTitle("step in the scan");
  meEEBCaloRedGreenSteps_->Reset();
  for( int bin=1; bin <87; bin++) { meEEBCaloRedGreenSteps_->setBinContent( bin, 1, 2. );}

}

void EEBeamCaloClient::cleanup(void) {
  if ( ! enableCleanup_ ) return;
  if ( cloneME_ ) {
    for(int u=0;u<cryInArray_;u++) {
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

  for(int u=0;u<cryInArray_;u++) {
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

  dqmStore_->setCurrentFolder( prefixME_ + "/EEBeamCaloClient" );

  if ( meEEBCaloRedGreen_) dqmStore_->removeElement( meEEBCaloRedGreen_->getName() );
  meEEBCaloRedGreen_ = 0;
  if ( meEEBCaloRedGreenReadCry_) dqmStore_->removeElement( meEEBCaloRedGreenReadCry_->getName() );
  meEEBCaloRedGreenReadCry_ = 0;
  if( meEEBCaloRedGreenSteps_ ) dqmStore_->removeElement (  meEEBCaloRedGreenSteps_->getName() );
  meEEBCaloRedGreenSteps_ = 0;
}

#ifdef WITH_ECAL_COND_DB
bool EEBeamCaloClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov, bool& status) {

  status = true;

  EcalLogicID ecid;

  MonOccupancyDat o;
  std::map<EcalLogicID, MonOccupancyDat> dataset;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( verbose_ ) {
      std::cout << " " << Numbers::sEE(ism) << " (ism=" << ism << ")" << std::endl;
      std::cout << std::endl;
    }

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
        if (hBcryDone_) { step = (int) hBcryDone_->GetBinContent(ic);}
        if( step > 0 && step < 86) {
        //if(hBE3x3vsCry_) {mean01 = hBE3x3vsCry_->GetBinContent(step);}// E in the 3x3
        if( hBE1vsCry_ ) {mean01 = hBE1vsCry_->GetBinContent(ic);} // E1
        }

        if ( update_channel ) {

          if ( Numbers::icEB(ism, ie, ip) == 1 ) {

            if ( verbose_ ) {
              std::cout << "Preparing dataset for " << Numbers::sEE(ism) << " (ism=" << ism << ")" << std::endl;
              std::cout << "CryOnBeam (" << ie << "," << ip << ") " << num01  << std::endl;
              std::cout << "MaxEneCry (" << ie << "," << ip << ") " << num02  << std::endl;
              std::cout << "E1 ("        << ie << "," << ip << ") " << mean01 << std::endl;
              std::cout << std::endl;
            }

          }

          o.setEventsOverHighThreshold(int(num01));
          o.setEventsOverLowThreshold(int(num02));

          o.setAvgEnergy(mean01);

          if ( econn ) {
            ecid = LogicID::getEcalLogicID("EE_crystal_number", ism, ic);
            dataset[ecid] = o;
          }

        }

      }
    }

  }

  if ( econn ) {
    try {
      if ( verbose_ ) std::cout << "Inserting MonOccupancyDat ..." << std::endl;
      if ( dataset.size() != 0 ) econn->insertDataArraySet(&dataset, moniov);
      if ( verbose_ ) std::cout << "done." << std::endl;
    } catch (std::runtime_error &e) {
      std::cerr << e.what() << std::endl;
    }
  }

  return true;

}
#endif

void EEBeamCaloClient::analyze(void) {

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( debug_ ) std::cout << "EEBeamCaloClient: ievt/jevt = " << ievt_ << "/" << jevt_ << std::endl;
  }

  char histo[200];

  MonitorElement* me = 0;

  // MonitorElement* meCD;
  sprintf(histo, (prefixME_ + "/EEBeamCaloTask/EEBCT crystals done").c_str());
  //meCD = dqmStore_->get(histo);
  me = dqmStore_->get(histo);
  hBcryDone_ = UtilsClient::getHisto<TH1F*>( me, cloneME_, hBcryDone_ );

  //MonitorElement* meCryInBeam;
  sprintf(histo, (prefixME_ + "/EEBeamCaloTask/EEBCT crystal on beam").c_str());
  //meCryInBeam = dqmStore_->get(histo);
  me = dqmStore_->get(histo);
  hBCryOnBeam_ = UtilsClient::getHisto<TH2F*>( me, cloneME_, hBCryOnBeam_);

  //MonitorElement* allNeededCry;
  sprintf(histo, (prefixME_ + "/EEBeamCaloTask/EEBCT all needed crystals readout").c_str());
  //allNeededCry= dqmStore_->get(histo);
  me = dqmStore_->get(histo);
  hBAllNeededCry_ = UtilsClient::getHisto<TH1F*>( me, cloneME_, hBAllNeededCry_);

  sprintf(histo, (prefixME_ + "/EEBeamCaloTask/EEBCT readout crystals number").c_str());
  //allNeededCry= dqmStore_->get(histo);
  me = dqmStore_->get(histo);
  hBNumReadCry_ = UtilsClient::getHisto<TH1F*>( me, cloneME_, hBNumReadCry_);

  //MonitorElement* RecEne3x3;
  sprintf(histo, (prefixME_ + "/EEBeamCaloTask/EEBCT rec Ene sum 3x3").c_str());
  //RecEne3x3= dqmStore_->get(histo);
  me = dqmStore_->get(histo);
  hBE3x3_ = UtilsClient::getHisto<TH1F*>( me, cloneME_, hBE3x3_);

  //MonitorElement* ErrRedCry;
  sprintf(histo, (prefixME_ + "/EEBeamCaloTask/EEBCT readout crystals errors").c_str());
  //ErrRedCry = dqmStore_->get(histo);
  me = dqmStore_->get(histo);
  hBReadCryErrors_ = UtilsClient::getHisto<TH1F*>( me, cloneME_, hBReadCryErrors_);

  //  MonitorElement* RecEne1;
  sprintf(histo, (prefixME_ + "/EEBeamCaloTask/EEBCT rec energy cry 5").c_str());
  //RecEne1= dqmStore_->get(histo);
  me = dqmStore_->get(histo);
  hBEne1_ = UtilsClient::getHisto<TH1F*>( me, cloneME_, hBEne1_);

  sprintf(histo, (prefixME_ + "/EEBeamCaloTask/EEBCT crystal with maximum rec energy").c_str());
  me = dqmStore_->get(histo);
  hBMaxEneCry_ = UtilsClient::getHisto<TH2F*>( me, cloneME_, hBMaxEneCry_);

  sprintf(histo, (prefixME_ + "/EEBeamCaloTask/EEBCT average rec energy in the 3x3 array").c_str());
  me = dqmStore_->get(histo);
  hBE3x3vsCry_ = UtilsClient::getHisto<TProfile*>( me, cloneME_, hBE3x3vsCry_);

  sprintf(histo, (prefixME_ + "/EEBeamCaloTask/EEBCT average rec energy in the single crystal").c_str());
  me = dqmStore_->get(histo);
  hBE1vsCry_ = UtilsClient::getHisto<TProfile*>( me, cloneME_, hBE1vsCry_);

  sprintf(histo, (prefixME_ + "/EEBeamCaloTask/EEBCT number of entries").c_str());
  me = dqmStore_->get(histo);
  hBEntriesvsCry_ = UtilsClient::getHisto<TH1F*>( me, cloneME_, hBEntriesvsCry_);

  sprintf(histo, (prefixME_ + "/EEBeamCaloTask/EEBCT energy deposition in the 3x3").c_str());
  me = dqmStore_->get(histo);
  hBBeamCentered_ = UtilsClient::getHisto<TH2F*>( me, cloneME_, hBBeamCentered_);

  sprintf(histo, (prefixME_ + "/EEBeamCaloTask/EEBCT table is moving").c_str());
  me = dqmStore_->get(histo);
  hbTBmoving_ = UtilsClient::getHisto<TH1F*>( me, cloneME_, hbTBmoving_);

  sprintf(histo, (prefixME_ + "/EEBeamCaloTask/EEBCT crystal in beam vs event").c_str());
  me = dqmStore_->get(histo);
  pBCriInBeamEvents_ =  UtilsClient::getHisto<TProfile*>( me, cloneME_, pBCriInBeamEvents_);

  sprintf(histo, (prefixME_ + "/EEBeamCaloTask/EEBCT E1 in the max cry").c_str());
  me = dqmStore_->get(histo);
  hbE1MaxCry_ =  UtilsClient::getHisto<TH1F*>( me, cloneME_, hbE1MaxCry_);

  sprintf(histo, (prefixME_ + "/EEBeamCaloTask/EEBCT Desynchronization vs step").c_str());
  me = dqmStore_->get(histo);
  hbDesync_ =  UtilsClient::getHisto<TH1F*>( me, cloneME_, hbDesync_);

  for(int ind = 0; ind < cryInArray_; ind ++) {
    sprintf(histo, (prefixME_ + "/EEBeamCaloTask/EEBCT pulse profile in G12 cry %01d").c_str(), ind+1);
    me = dqmStore_->get(histo);
    hBpulse_[ind] = UtilsClient::getHisto<TProfile*>( me, cloneME_, hBpulse_[ind]);

    sprintf(histo, (prefixME_ + "/EEBeamCaloTask/EEBCT found gains cry %01d").c_str(), ind+1);
    me = dqmStore_->get(histo);
    hBGains_[ind] = UtilsClient::getHisto<TH1F*>( me, cloneME_, hBGains_[ind]);
  }

  int DoneCry = 0;//if it stays 1 the run is not an autoscan
  if (hBcryDone_) {
    for(int cry=1 ; cry<1701 ; cry ++) {
      int step = (int) hBcryDone_->GetBinContent(cry);
      if( step>0 ) {//this crystal has been scanned or is being scanned
        DoneCry++;
        float E3x3RMS = -1, E3x3 =-1, E1=-1;
        if(hBE3x3vsCry_) {
          //E3x3RMS = hBE3x3vsCry_->GetBinError(step);
          //E3x3 = hBE3x3vsCry_->GetBinContent(step);
          E3x3RMS = hBE3x3vsCry_->GetBinError(cry);
          E3x3 = hBE3x3vsCry_->GetBinContent(cry);
        }
        //if( hBE1vsCry_) {E1=hBE1vsCry_->GetBinContent(step);}
        if( hBE1vsCry_) {E1=hBE1vsCry_->GetBinContent(cry);}
        bool RMS3x3  =  (  E3x3RMS < RMSEne3x3_ && E3x3RMS >= 0 );
        bool Mean3x3 =  ( std::abs( E3x3 - aveEne3x3_ ) < E3x3Th_);
        bool Mean1   =  ( std::abs( E1 - aveEne1_ ) < E1Th_ );
        int ieta = ( cry - 1)/20 + 1 ;//+1 for the bin
        int iphi = ( cry - 1)%20 + 1 ;//+1 for the bin
        //fill the RedGreen histo
        if(ieta >0 && iphi >0 ) {
          if(RMS3x3 && Mean3x3 && Mean1) {meEEBCaloRedGreen_->setBinContent(ieta,iphi,1.);}
          else {meEEBCaloRedGreen_->setBinContent(ieta,iphi,0.);}
        }

        float Entries = -1;
        //if ( hBEntriesvsCry_ ) {Entries = hBEntriesvsCry_->GetBinContent(step);}
        if ( hBEntriesvsCry_ ) {Entries = hBEntriesvsCry_->GetBinContent(cry);}
        bool Nent = ( Entries * prescaling_  > minEvtNum_ );
        bool readCryOk = true;
        if( hBReadCryErrors_ ) {
          int step_bin = hBReadCryErrors_->GetXaxis()->FindFixBin(step);
          if ( step_bin > 0 && step_bin < hBReadCryErrors_->GetNbinsX() ) {
            if ( hBReadCryErrors_->GetBinContent(step_bin) <= Entries*ReadCryErrThr_ ) {readCryOk = true;}
            else {readCryOk = false;}
          }
        }

        if(Nent && readCryOk ) { meEEBCaloRedGreenSteps_->setBinContent(step,1,1.);}
        else{ meEEBCaloRedGreenSteps_->setBinContent(step,1,0.);}

        if (readCryOk &&  meEEBCaloRedGreenReadCry_->getBinContent(1,1) != 0.) { meEEBCaloRedGreenReadCry_->setBinContent(1,1, 1.);}
        else if ( !readCryOk ) { meEEBCaloRedGreenReadCry_->setBinContent(1,1, 0.);}
      }// end of if (step>0)
    }//end of loop over cry
  }//end of if(hBcryDone_)

  if(DoneCry == 1) {//this is probably not an auotscan or it is the first crystal
    float nEvt = 0;
    if(hBE3x3_) {nEvt = hBE3x3_->GetEntries();}
    if(nEvt > 1*prescaling_ && hBE3x3_ && hBEne1_ && hBCryOnBeam_ && meEEBCaloRedGreen_) {//check for mean and RMS
      bool RMS3x3  =  ( hBE3x3_->GetRMS() < RMSEne3x3_ );
      bool Mean3x3 =  ( std::abs( hBE3x3_->GetMean() - aveEne3x3_ ) < E3x3Th_ );
      bool Mean1   =  ( std::abs( hBEne1_->GetMean() - aveEne1_ ) < E1Th_ );
      //fill the RedGreen histo
      int ieta=0,iphi=0;
      float found =0; //there should be just one bin filled but...
      for (int b_eta =1; b_eta<86; b_eta++) {
        for (int b_phi =1; b_phi<21; b_phi++) {
          float bc = hBCryOnBeam_->GetBinContent(b_eta,b_phi);//FIX ME check if this is the correct binning
          if(bc > found) { found =bc; ieta = b_eta; iphi= b_phi;}
        }
      }
      if(ieta >0 && iphi >0 ) {
        if(RMS3x3 && Mean3x3 && Mean1) {meEEBCaloRedGreen_->setBinContent(ieta,iphi,1.);}
        else {meEEBCaloRedGreen_->setBinContent(ieta,iphi,0.);}
      }
    }
    if(hBReadCryErrors_) {
      float nErr = hBReadCryErrors_->GetBinContent(1);// for a non autoscan just the first bin should be filled
      if( nErr > nEvt*ReadCryErrThr_ ) { meEEBCaloRedGreenReadCry_->setBinContent(1,1,0.);}
      else { meEEBCaloRedGreenReadCry_->setBinContent(1,1,1.);}
    }
  }

  //   // was done using me instead of histos
  //   if(DoneCry == 0) {//this is probably not an auotscan
  //     float nEvt = RecEne3x3->getEntries();
  //     if(nEvt > 1000*prescaling_) {//check for mean and RMS
  //       bool RMS3x3  =  ( RecEne3x3->getRMS() < RMSEne3x3_ );
  //       bool Mean3x3 =  ( (RecEne3x3->getMean() - aveEne3x3_) < E3x3Th_);
  //       bool Mean1   =  ( (RecEne1->getMean() < aveEne1_) < E1Th_ );
  //       //fill the RedGreen histo
  //       int ieta=0,iphi=0;
  //       float found =0; //there should be just one bin filled but...
  //       for (int b_eta =1; b_eta<86; b_eta++) {
  //         for (int b_phi =1; b_phi<21; b_phi++) {
  //           float bc = meCryInBeam->getBinContent(b_eta,b_phi);//FIX ME check if this is the correct binning
  //           if(bc > found) { found =bc; ieta = b_eta; iphi= b_phi;}
  //         }
  //       }
  //       if(ieta >0 && iphi >0 ) {
  //         if(RMS3x3 && Mean3x3 && Mean1) {meEEBCaloRedGreen_->setBinContent(ieta,iphi,1.);}
  //         else {meEEBCaloRedGreen_->setBinContent(ieta,iphi,0.);}
  //       }
  //     }
  //     float nErr = ErrRedCry->getBinContent(1);// for a non autoscan just the first bin should be filled
  //     if( nErr > nEvt*ReadCryErrThr_ ) { meEEBCaloRedGreenReadCry_->setBinContent(1,1,0.);}
  //     else { meEEBCaloRedGreenReadCry_->setBinContent(1,1,1.);}
  //   }


}

