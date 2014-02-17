/*
 * \file EBTrendClient.cc
 *
 * $Date: 2012/04/27 13:45:59 $
 * $Revision: 1.10 $
 * \author Dongwook Jang, Soon Yung Jun
 *
*/

#include <iostream>
#include <fstream>
#include <math.h>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DQM/EcalBarrelMonitorClient/interface/EBTrendClient.h"
#include "DQM/EcalCommon/interface/UtilFunctions.h"
#include "DQM/EcalCommon/interface/UtilsClient.h"

EBTrendClient::EBTrendClient(const edm::ParameterSet& ps){

  init_ = false;

  dqmStore_ = edm::Service<DQMStore>().operator->();

  prefixME_ = ps.getUntrackedParameter<std::string>("prefixME", "");

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  mergeRuns_ = ps.getUntrackedParameter<bool>("mergeRuns", false);

  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

  // histograms...
  for(int i=0; i<nHists_; i++) {
    meanMinutely_[i] = 0;
    sigmaMinutely_[i] = 0;

    meanHourly_[i] = 0;
    sigmaHourly_[i] = 0;

    previousHist_[i] = 0;
    currentHist_[i] = 0;
  } // for

}


EBTrendClient::~EBTrendClient(){
}


void EBTrendClient::beginJob(void){

  ievt_ = 0;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EBTrendClient");
    dqmStore_->rmdir(prefixME_ + "/EBTrendClient");
  }

  // noise,
  // entries of EBOT rec hit thr occupancy
  // entries of EBOT tp digi occupancy
  // rec hit energy
  // ebtmt timing mean ID summary
  // ebtmt timing RMS ID summary
  //

  int index = 0;

  moduleNames_[index] = "EBClusterTask"; // TH1
  histTitles_[index]  = "EBCLT BC energy";
  index++;

  moduleNames_[index] = "EBClusterTask"; // TH1
  histTitles_[index]  = "EBCLT SC energy";
  index++;

  moduleNames_[index] = "EBSummaryClient"; // TProfile
  histTitles_[index]  = "EBPOT pedestal G12 mean";
  index++;

  moduleNames_[index] = "EBSummaryClient"; // TProfile
  histTitles_[index]  = "EBPOT pedestal G12 rms";
  index++;

  moduleNames_[index] = "EBOccupancyTask"; // TH2
  histTitles_[index]  = "EBOT rec hit thr occupancy";
  index++;

  moduleNames_[index] = "EBOccupancyTask"; // TH2
  histTitles_[index]  = "EBOT TP digi thr occupancy";
  index++;

}


void EBTrendClient::beginRun(const edm::Run& r, const edm::EventSetup& c) {

  if ( ! mergeRuns_ ) this->reset();

  start_time_ = time(NULL);

}


void EBTrendClient::endRun(const edm::Run& r, const edm::EventSetup& c) {

}


void EBTrendClient::reset(void) {

  for(int i=0; i<nHists_; i++) {
    if(meanMinutely_[i]) meanMinutely_[i]->Reset();
    if(sigmaMinutely_[i]) sigmaMinutely_[i]->Reset();

    if(meanHourly_[i]) meanHourly_[i]->Reset();
    if(sigmaHourly_[i]) sigmaHourly_[i]->Reset();
  }// for

}


void EBTrendClient::setup(void){

  init_ = true;

  std::string histo;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EBTrendClient");

    for(int i=0; i<nHists_; i++) {

      // minutely

      histo = "Average of " + histTitles_[i] + " Vs 5Minutes";
      meanMinutely_[i] = dqmStore_->bookProfile(histo.c_str(), histo.c_str(), 12, 0.0, 60.0, 100, 0.0, 1.0e6, "s");
      meanMinutely_[i]->setAxisTitle("Minutes", 1);
      histo = "Average of " + histTitles_[i] + " / 5 minutes";
      meanMinutely_[i]->setAxisTitle(histo.c_str(), 2);

      histo = "RMS of " + histTitles_[i] + " Vs 5Minutes";
      sigmaMinutely_[i] = dqmStore_->bookProfile(histo.c_str(), histo.c_str(), 12, 0.0, 60.0, 100, 0.0, 1.0e6, "s");
      sigmaMinutely_[i]->setAxisTitle("Minutes", 1);
      histo = "RMS of " + histTitles_[i] + " / 5 minutes";
      sigmaMinutely_[i]->setAxisTitle(histo.c_str(), 2);


      // hourly

      histo = "Average of " + histTitles_[i] + " Vs 1Hour";
      meanHourly_[i] = dqmStore_->bookProfile(histo.c_str(), histo.c_str(), 24, 0.0, 24.0, 100, 0.0, 1.0e6, "s");
      meanHourly_[i]->setAxisTitle("Hours", 1);
      histo = "Average of " + histTitles_[i] + " / hour";
      meanHourly_[i]->setAxisTitle(histo.c_str(), 2);

      histo = "RMS of " + histTitles_[i] + " Vs 1Hour";
      sigmaHourly_[i] = dqmStore_->bookProfile(histo.c_str(), histo.c_str(), 24, 0.0, 24.0, 100, 0.0, 1.0e6, "s");
      sigmaHourly_[i]->setAxisTitle("Hours", 1);
      histo = "RMS of " + histTitles_[i] + " / hour";
      sigmaHourly_[i]->setAxisTitle(histo.c_str(), 2);

    }// for i

  }// if

}


void EBTrendClient::cleanup(void){

  if ( ! init_ ) return;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EBTrendClient");

    for(int i=0; i<nHists_; i++) {
      if(meanMinutely_[i]) dqmStore_->removeElement( meanMinutely_[i]->getName());
      meanMinutely_[i] = 0;
      if(sigmaMinutely_[i]) dqmStore_->removeElement( sigmaMinutely_[i]->getName());
      sigmaMinutely_[i] = 0;

      if(meanHourly_[i]) dqmStore_->removeElement( meanHourly_[i]->getName());
      meanHourly_[i] = 0;
      if(sigmaHourly_[i]) dqmStore_->removeElement( sigmaHourly_[i]->getName());
      sigmaHourly_[i] = 0;

      if(previousHist_[i]) delete previousHist_[i];
      previousHist_[i] = 0;
      if(currentHist_[i]) delete currentHist_[i];
      currentHist_[i] = 0;

    }// for

  }

  init_ = false;

}


void EBTrendClient::endJob(void){

  edm::LogInfo("EBTrendClient") << "analyzed " << ievt_ << " events";

  if ( enableCleanup_ ) this->cleanup();

}


void EBTrendClient::analyze(const edm::Event& e, const edm::EventSetup& c){

  if ( ! init_ ) this->setup();

  ievt_++;

  // --------------------------------------------------
  // Collect time information
  // --------------------------------------------------

  updateTime();

  //  long int diff_time = (current_time_ - start_time_)/60;

  long int minuteBinDiff = -1;
  long int minuteDiff = -1;
  //  ecaldqm::calcBins(5,1,start_time_,last_time_,current_time_,minuteBinDiff,minuteDiff);
  ecaldqm::calcBins(5,60,start_time_,last_time_,current_time_,minuteBinDiff,minuteDiff);

  if(minuteBinDiff <= 0) return;

  long int hourBinDiff = -1;
  long int hourDiff = -1;
  ecaldqm::calcBins(1,3600,start_time_,last_time_,current_time_,hourBinDiff,hourDiff);


  for(int i=0; i<nHists_; i++){

    mean_[i] = rms_[i] = 0;

    // delete previous hists if any
    if(previousHist_[i]) delete previousHist_[i];

    // assign currentHists to previousHists
    previousHist_[i] = currentHist_[i];

  } // for i


  std::string histo;

  MonitorElement* me;

  // get clones of histograms from other tasks or clients
  // assign cloned histogrmas to currentHist_[i]

  for(int i=0; i<nHists_; i++) {
    histo = prefixME_ + "/" + moduleNames_[i] + "/" + histTitles_[i];
    me = dqmStore_->get(histo.c_str());
    currentHist_[i] = ecaldqm::cloneIt(me,histo);
  }


  // Get mean and rms and fill Profile

  for(int i=0; i<nHists_; i++){

    ecaldqm::getMeanRms(previousHist_[i],currentHist_[i],mean_[i],rms_[i]);

    if(verbose_) {
      std::cout << std::scientific;
      std::cout << "EBTrendClient mean["<<i<<"] = " << mean_[i] << ", \t rms["<<i<<"] = " << rms_[i] << std::endl;
    }

    ecaldqm::shift2Right(meanMinutely_[i]->getTProfile(), minuteBinDiff);
    meanMinutely_[i]->Fill(minuteDiff,mean_[i]);

    ecaldqm::shift2Right(sigmaMinutely_[i]->getTProfile(), minuteBinDiff);
    sigmaMinutely_[i]->Fill(minuteDiff,rms_[i]);

    ecaldqm::shift2Right(meanHourly_[i]->getTProfile(), hourBinDiff);
    meanHourly_[i]->Fill(hourDiff,mean_[i]);

    ecaldqm::shift2Right(sigmaHourly_[i]->getTProfile(), hourBinDiff);
    sigmaHourly_[i]->Fill(hourDiff,rms_[i]);
  } // for i


}


void EBTrendClient::updateTime(){

  last_time_ = current_time_;
  current_time_ = time(NULL);

}

