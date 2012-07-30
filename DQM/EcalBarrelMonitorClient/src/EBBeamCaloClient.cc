/*
 * \file EBBeamCaloClient.cc
 *
 * $Date: 2011/08/30 09:33:51 $
 * $Revision: 1.101 $
 * \author G. Della Ricca
 * \author A. Ghezzi
 *
 */

#include <memory>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <math.h>

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#ifdef WITH_ECAL_COND_DB
#include "OnlineDB/EcalCondDB/interface/MonOccupancyDat.h"
#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"
#include "DQM/EcalCommon/interface/LogicID.h"
#endif

#include "DQM/EcalCommon/interface/UtilsClient.h"
#include "DQM/EcalCommon/interface/Numbers.h"

#include "DQM/EcalBarrelMonitorClient/interface/EBBeamCaloClient.h"

EBBeamCaloClient::EBBeamCaloClient(const edm::ParameterSet& ps) {

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

  // vector of selected Super Modules (Defaults to all 36).
  superModules_.reserve(36);
  for ( unsigned int i = 1; i <= 36; i++ ) superModules_.push_back(i);
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

  meEBBCaloRedGreen_ = 0;
  meEBBCaloRedGreenReadCry_ = 0;
  meEBBCaloRedGreenSteps_ = 0;
}

EBBeamCaloClient::~EBBeamCaloClient() {

}

void EBBeamCaloClient::beginJob(void) {

  dqmStore_ = edm::Service<DQMStore>().operator->();

  if ( debug_ ) std::cout << "EBBeamCaloClient: beginJob" << std::endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EBBeamCaloClient::beginRun(void) {

  if ( debug_ ) std::cout << "EBBeamCaloClient: beginRun" << std::endl;

  jevt_ = 0;

  this->setup();

}

void EBBeamCaloClient::endJob(void) {

  if ( debug_ ) std::cout << "EBBeamCaloClient: endJob, ievt = " << ievt_ << std::endl;

  this->cleanup();

}

void EBBeamCaloClient::endRun(void) {

  if ( debug_ ) std::cout << "EBBeamCaloClient: endRun, jevt = " << jevt_ << std::endl;

  this->cleanup();

}

void EBBeamCaloClient::setup(void) {

  std::string name;

  dqmStore_->setCurrentFolder( prefixME_ + "/EBBeamCaloClient" );

  if ( meEBBCaloRedGreen_ ) dqmStore_->removeElement( meEBBCaloRedGreen_->getName() );
  name = "EBBCT quality";
  meEBBCaloRedGreen_ = dqmStore_->book2D(name, name, 85, 0., 85., 20, 0., 20.);

  meEBBCaloRedGreen_->Reset();

  for ( int ie = 1; ie <= 85; ie++ ) {
    for ( int ip = 1; ip <= 20; ip++ ) {

      meEBBCaloRedGreen_ ->setBinContent( ie, ip, 2. );

    }
  }

  if ( meEBBCaloRedGreenReadCry_ ) dqmStore_->removeElement( meEBBCaloRedGreenReadCry_->getName() );
  name = "EBBCT quality read crystal errors";
  meEBBCaloRedGreenReadCry_ = dqmStore_->book2D(name, name, 1, 0., 1., 1, 0., 1.);
  meEBBCaloRedGreenReadCry_->Reset();
  meEBBCaloRedGreenReadCry_ ->setBinContent( 1, 1, 2. );

  if( meEBBCaloRedGreenSteps_ ) dqmStore_->removeElement( meEBBCaloRedGreenSteps_->getName() );
  name = "EBBCT quality entries or read crystals errors";
  meEBBCaloRedGreenSteps_ = dqmStore_->book2D(name, name, 86, 1., 87., 1, 0., 1.);
  meEBBCaloRedGreenSteps_->setAxisTitle("step in the scan");
  meEBBCaloRedGreenSteps_->Reset();
  for( int bin=1; bin <87; bin++) { meEBBCaloRedGreenSteps_->setBinContent( bin, 1, 2. );}

}

void EBBeamCaloClient::cleanup(void) {
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

  dqmStore_->setCurrentFolder( prefixME_ + "/EBBeamCaloClient" );

  if ( meEBBCaloRedGreen_) dqmStore_->removeElement( meEBBCaloRedGreen_->getName() );
  meEBBCaloRedGreen_ = 0;
  if ( meEBBCaloRedGreenReadCry_) dqmStore_->removeElement( meEBBCaloRedGreenReadCry_->getName() );
  meEBBCaloRedGreenReadCry_ = 0;
  if( meEBBCaloRedGreenSteps_ ) dqmStore_->removeElement (  meEBBCaloRedGreenSteps_->getName() );
  meEBBCaloRedGreenSteps_ = 0;
}

#ifdef WITH_ECAL_COND_DB
bool EBBeamCaloClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov, bool& status) {

  status = true;

  EcalLogicID ecid;

  MonOccupancyDat o;
  std::map<EcalLogicID, MonOccupancyDat> dataset;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( verbose_ ) {
      std::cout << " " << Numbers::sEB(ism) << " (ism=" << ism << ")" << std::endl;
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
              std::cout << "Preparing dataset for " << Numbers::sEB(ism) << " (ism=" << ism << ")" << std::endl;
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
            ecid = LogicID::getEcalLogicID("EB_crystal_number", Numbers::iSM(ism, EcalBarrel), ic);
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

void EBBeamCaloClient::analyze(void) {

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( debug_ ) std::cout << "EBBeamCaloClient: ievt/jevt = " << ievt_ << "/" << jevt_ << std::endl;
  }

  MonitorElement* me = 0;

  me = dqmStore_->get(prefixME_ + "/EBBeamCaloTask/EBBCT crystals done");
  hBcryDone_ = UtilsClient::getHisto( me, cloneME_, hBcryDone_ );

  me = dqmStore_->get(prefixME_ + "/EBBeamCaloTask/EBBCT crystal on beam");
  hBCryOnBeam_ = UtilsClient::getHisto( me, cloneME_, hBCryOnBeam_);

  me = dqmStore_->get(prefixME_ + "/EBBeamCaloTask/EBBCT all needed crystals readout");
  hBAllNeededCry_ = UtilsClient::getHisto( me, cloneME_, hBAllNeededCry_);

  me = dqmStore_->get(prefixME_ + "/EBBeamCaloTask/EBBCT readout crystals number");
  hBNumReadCry_ = UtilsClient::getHisto( me, cloneME_, hBNumReadCry_);

  me = dqmStore_->get(prefixME_ + "/EBBeamCaloTask/EBBCT rec Ene sum 3x3");
  hBE3x3_ = UtilsClient::getHisto( me, cloneME_, hBE3x3_);

  me = dqmStore_->get(prefixME_ + "/EBBeamCaloTask/EBBCT readout crystals errors");
  hBReadCryErrors_ = UtilsClient::getHisto( me, cloneME_, hBReadCryErrors_);

  me = dqmStore_->get(prefixME_ + "/EBBeamCaloTask/EBBCT rec energy cry 5");
  hBEne1_ = UtilsClient::getHisto( me, cloneME_, hBEne1_);

  me = dqmStore_->get(prefixME_ + "/EBBeamCaloTask/EBBCT crystal with maximum rec energy");
  hBMaxEneCry_ = UtilsClient::getHisto( me, cloneME_, hBMaxEneCry_);

  me = dqmStore_->get(prefixME_ + "/EBBeamCaloTask/EBBCT average rec energy in the 3x3 array");
  hBE3x3vsCry_ = UtilsClient::getHisto( me, cloneME_, hBE3x3vsCry_);

  me = dqmStore_->get(prefixME_ + "/EBBeamCaloTask/EBBCT average rec energy in the single crystal");
  hBE1vsCry_ = UtilsClient::getHisto( me, cloneME_, hBE1vsCry_);

  me = dqmStore_->get(prefixME_ + "/EBBeamCaloTask/EBBCT number of entries");
  hBEntriesvsCry_ = UtilsClient::getHisto( me, cloneME_, hBEntriesvsCry_);

  me = dqmStore_->get(prefixME_ + "/EBBeamCaloTask/EBBCT energy deposition in the 3x3");
  hBBeamCentered_ = UtilsClient::getHisto( me, cloneME_, hBBeamCentered_);

  me = dqmStore_->get(prefixME_ + "/EBBeamCaloTask/EBBCT table is moving");
  hbTBmoving_ = UtilsClient::getHisto( me, cloneME_, hbTBmoving_);

  me = dqmStore_->get(prefixME_ + "/EBBeamCaloTask/EBBCT crystal in beam vs event");
  pBCriInBeamEvents_ =  UtilsClient::getHisto( me, cloneME_, pBCriInBeamEvents_);

  me = dqmStore_->get(prefixME_ + "/EBBeamCaloTask/EBBCT E1 in the max cry");
  hbE1MaxCry_ =  UtilsClient::getHisto( me, cloneME_, hbE1MaxCry_);

  me = dqmStore_->get(prefixME_ + "/EBBeamCaloTask/EBBCT Desynchronization vs step");
  hbDesync_ =  UtilsClient::getHisto( me, cloneME_, hbDesync_);

  std::stringstream ss;
  for(int ind = 0; ind < cryInArray_; ind ++) {
    ss.str("");
    ss << prefixME_ << "/EBBeamCaloTask/EBBCT pulse profile in G12 cry " << std::setfill('0') << std::setw(1) << ind+1;
    me = dqmStore_->get(ss.str());
    hBpulse_[ind] = UtilsClient::getHisto( me, cloneME_, hBpulse_[ind]);

    ss.str("");
    ss << prefixME_ << "/EBBeamCaloTask/EBBCT found gains cry " << std::setfill('0') << std::setw(1) << ind+1;
    me = dqmStore_->get(ss.str());
    hBGains_[ind] = UtilsClient::getHisto( me, cloneME_, hBGains_[ind]);
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
          if(RMS3x3 && Mean3x3 && Mean1) {meEBBCaloRedGreen_->setBinContent(ieta,iphi,1.);}
          else {meEBBCaloRedGreen_->setBinContent(ieta,iphi,0.);}
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

        if(Nent && readCryOk ) { meEBBCaloRedGreenSteps_->setBinContent(step,1,1.);}
        else{ meEBBCaloRedGreenSteps_->setBinContent(step,1,0.);}

        if (readCryOk &&  meEBBCaloRedGreenReadCry_->getBinContent(1,1) != 0.) { meEBBCaloRedGreenReadCry_->setBinContent(1,1, 1.);}
        else if ( !readCryOk ) { meEBBCaloRedGreenReadCry_->setBinContent(1,1, 0.);}
      }// end of if (step>0)
    }//end of loop over cry
  }//end of if(hBcryDone_)

  if(DoneCry == 1) {//this is probably not an auotscan or it is the first crystal
    float nEvt = 0;
    if(hBE3x3_) {nEvt = hBE3x3_->GetEntries();}
    if(nEvt > 1*prescaling_ && hBE3x3_ && hBEne1_ && hBCryOnBeam_ && meEBBCaloRedGreen_) {//check for mean and RMS
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
        if(RMS3x3 && Mean3x3 && Mean1) {meEBBCaloRedGreen_->setBinContent(ieta,iphi,1.);}
        else {meEBBCaloRedGreen_->setBinContent(ieta,iphi,0.);}
      }
    }
    if(hBReadCryErrors_) {
      float nErr = hBReadCryErrors_->GetBinContent(1);// for a non autoscan just the first bin should be filled
      if( nErr > nEvt*ReadCryErrThr_ ) { meEBBCaloRedGreenReadCry_->setBinContent(1,1,0.);}
      else { meEBBCaloRedGreenReadCry_->setBinContent(1,1,1.);}
    }
  }
}

