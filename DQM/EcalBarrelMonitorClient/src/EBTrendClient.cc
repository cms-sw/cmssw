/*
 * \file EBTrendClient.cc
 *
 * $Date: 2010/08/11 15:01:48 $
 * $Revision: 1.7 $
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

    moduleNames_[i] = "";
    histTitles_[i] = "";
    previousHist_[i] = 0;
    currentHist_[i] = 0;
    mean_[i] = 0.;
    rms_[i] = 0.;
  } // for

  ievt_ = 0;

  start_time_ = 0;
  current_time_ = 0;
  last_time_ = 0;
}


EBTrendClient::~EBTrendClient(){
}


void EBTrendClient::beginJob(void){

  ievt_ = 0;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/Trend");
    dqmStore_->rmdir(prefixME_ + "/Trend");
  }

  // noise,
  // entries of EBOT rec hit thr occupancy
  // entries of EBOT tp digi occupancy
  // rec hit energy
  // ebtmt timing mean ID summary
  // ebtmt timing RMS ID summary
  //

  int index = 0;

  moduleNames_[index] = "Cluster"; // TH1
  histTitles_[index]  = "BasicClusters/ClusterTask BC energy EB";
  index++;

  moduleNames_[index] = "Cluster"; // TH1
  histTitles_[index]  = "SuperClusters/ClusterTask SC energy EB";
  index++;

  moduleNames_[index] = "Summary"; // TProfile
  histTitles_[index]  = "SummaryClient presample G12";
  index++;

//   moduleNames_[index] = "Summary"; // TProfile
//   histTitles_[index]  = "SummaryClient presample rms G12";
  index++;

  moduleNames_[index] = "Occupancy"; // TH2
  histTitles_[index]  = "RecHit/OccupancyTask rec hit occupancy EB";
  index++;

  moduleNames_[index] = "Occupancy"; // TH2
  histTitles_[index]  = "TPDigi/OccupancyTask TP digi occupancy EB";
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
  std::string binning;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/Trend");

    int i(0);

    // minutely
    dqmStore_->setCurrentFolder(prefixME_ + "/Trend/ShortTerm");

    binning = "5min bin EB";

    histo = "TrendClient BC energy mean " + binning;
    meanMinutely_[i] = dqmStore_->bookProfile(histo.c_str(), histo.c_str(), 24, 0.0, 120.0, 0.0, 1.0e6, "s");
    meanMinutely_[i]->setAxisTitle("Minutes", 1);
    meanMinutely_[i]->setAxisTitle("BC energy (GeV)", 2);

    histo = "TrendClient BC energy fluctuation " + binning;
    sigmaMinutely_[i] = dqmStore_->bookProfile(histo.c_str(), histo.c_str(), 24, 0.0, 120.0, 0.0, 1.0e6, "s");
    sigmaMinutely_[i]->setAxisTitle("Minutes", 1);
    sigmaMinutely_[i]->setAxisTitle("BC energy rms (GeV)", 2);

    i++;

    histo = "TrendClient SC energy mean " + binning;
    meanMinutely_[i] = dqmStore_->bookProfile(histo.c_str(), histo.c_str(), 24, 0.0, 120.0, 0.0, 1.0e6, "s");
    meanMinutely_[i]->setAxisTitle("Minutes", 1);
    meanMinutely_[i]->setAxisTitle("SC energy (GeV)", 2);

    histo = "TrendClient SC energy fluctuation " + binning;
    sigmaMinutely_[i] = dqmStore_->bookProfile(histo.c_str(), histo.c_str(), 24, 0.0, 120.0, 0.0, 1.0e6, "s");
    sigmaMinutely_[i]->setAxisTitle("Minutes", 1);
    sigmaMinutely_[i]->setAxisTitle("SC energy rms (GeV)", 2);

    i++;

    histo = "TrendClient presample mean " + binning;
    meanMinutely_[i] = dqmStore_->bookProfile(histo.c_str(), histo.c_str(), 24, 0.0, 120.0, 0.0, 1.0e6, "s");
    meanMinutely_[i]->setAxisTitle("Minutes", 1);
    meanMinutely_[i]->setAxisTitle("pedestal", 2);

    histo = "TrendClient presample fluctuation " + binning;
    sigmaMinutely_[i] = dqmStore_->bookProfile(histo.c_str(), histo.c_str(), 24, 0.0, 120.0, 0.0, 1.0e6, "s");
    sigmaMinutely_[i]->setAxisTitle("Minutes", 1);
    sigmaMinutely_[i]->setAxisTitle("pedestal rms", 2);

    i++;
    i++;

    histo = "TrendClient rec hit thr occupancy mean " + binning;
    meanMinutely_[i] = dqmStore_->bookProfile(histo.c_str(), histo.c_str(), 24, 0.0, 120.0, 0.0, 1.0e6, "s");
    meanMinutely_[i]->setAxisTitle("Minutes", 1);
    meanMinutely_[i]->setAxisTitle("occupancy", 2);

    histo = "TrendClient rec hit thr occupancy fluctuation " + binning;
    sigmaMinutely_[i] = dqmStore_->bookProfile(histo.c_str(), histo.c_str(), 24, 0.0, 120.0, 0.0, 1.0e6, "s");
    sigmaMinutely_[i]->setAxisTitle("Minutes", 1);
    sigmaMinutely_[i]->setAxisTitle("occupancy rms", 2);

    i++;

    histo = "TrendClient TP digi thr occupancy mean " + binning;
    meanMinutely_[i] = dqmStore_->bookProfile(histo.c_str(), histo.c_str(), 24, 0.0, 120.0, 0.0, 1.0e6, "s");
    meanMinutely_[i]->setAxisTitle("Minutes", 1);
    meanMinutely_[i]->setAxisTitle("occupancy", 2);

    histo = "TrendClient TP digi thr occupancy fluctuation " + binning;
    sigmaMinutely_[i] = dqmStore_->bookProfile(histo.c_str(), histo.c_str(), 24, 0.0, 120.0, 0.0, 1.0e6, "s");
    sigmaMinutely_[i]->setAxisTitle("Minutes", 1);
    sigmaMinutely_[i]->setAxisTitle("occupancy rms", 2);

    
    // hourly
    dqmStore_->setCurrentFolder(prefixME_ + "/Trend/LongTerm");

    binning = "20min bin EB";

    i = 0;

    histo = "TrendClient BC energy mean " + binning;
    meanHourly_[i] = dqmStore_->bookProfile(histo.c_str(), histo.c_str(), 24, 0.0, 480.0, 0.0, 1.0e6, "s");
    meanHourly_[i]->setAxisTitle("Minutes", 1);
    meanHourly_[i]->setAxisTitle("BC energy (GeV)", 2);

    histo = "TrendClient BC energy fluctuation " + binning;
    sigmaHourly_[i] = dqmStore_->bookProfile(histo.c_str(), histo.c_str(), 24, 0.0, 480.0, 0.0, 1.0e6, "s");
    sigmaHourly_[i]->setAxisTitle("Minutes", 1);
    sigmaHourly_[i]->setAxisTitle("BC energy rms (GeV)", 2);

    i++;

    histo = "TrendClient SC energy mean " + binning;
    meanHourly_[i] = dqmStore_->bookProfile(histo.c_str(), histo.c_str(), 24, 0.0, 480.0, 0.0, 1.0e6, "s");
    meanHourly_[i]->setAxisTitle("Minutes", 1);
    meanHourly_[i]->setAxisTitle("SC energy (GeV)", 2);

    histo = "TrendClient SC energy fluctuation " + binning;
    sigmaHourly_[i] = dqmStore_->bookProfile(histo.c_str(), histo.c_str(), 24, 0.0, 480.0, 0.0, 1.0e6, "s");
    sigmaHourly_[i]->setAxisTitle("Minutes", 1);
    sigmaHourly_[i]->setAxisTitle("SC energy rms (GeV)", 2);

    i++;

    histo = "TrendClient presample mean " + binning;
    meanHourly_[i] = dqmStore_->bookProfile(histo.c_str(), histo.c_str(), 24, 0.0, 480.0, 0.0, 1.0e6, "s");
    meanHourly_[i]->setAxisTitle("Minutes", 1);
    meanHourly_[i]->setAxisTitle("pedestal", 2);

    histo = "TrendClient presample fluctuation " + binning;
    sigmaHourly_[i] = dqmStore_->bookProfile(histo.c_str(), histo.c_str(), 24, 0.0, 480.0, 0.0, 1.0e6, "s");
    sigmaHourly_[i]->setAxisTitle("Minutes", 1);
    sigmaHourly_[i]->setAxisTitle("pedestal rms", 2);

    i++;
    i++;

    histo = "TrendClient rec hit thr occupancy mean " + binning;
    meanHourly_[i] = dqmStore_->bookProfile(histo.c_str(), histo.c_str(), 24, 0.0, 480.0, 0.0, 1.0e6, "s");
    meanHourly_[i]->setAxisTitle("Minutes", 1);
    meanHourly_[i]->setAxisTitle("occupancy", 2);

    histo = "TrendClient rec hit thr occupancy fluctuation " + binning;
    sigmaHourly_[i] = dqmStore_->bookProfile(histo.c_str(), histo.c_str(), 24, 0.0, 480.0, 0.0, 1.0e6, "s");
    sigmaHourly_[i]->setAxisTitle("Minutes", 1);
    sigmaHourly_[i]->setAxisTitle("occupancy rms", 2);

    i++;

    histo = "TrendClient TP digi thr occupancy mean " + binning;
    meanHourly_[i] = dqmStore_->bookProfile(histo.c_str(), histo.c_str(), 24, 0.0, 480.0, 0.0, 1.0e6, "s");
    meanHourly_[i]->setAxisTitle("Minutes", 1);
    meanHourly_[i]->setAxisTitle("occupancy", 2);

    histo = "TrendClient TP digi thr occupancy fluctuation " + binning;
    sigmaHourly_[i] = dqmStore_->bookProfile(histo.c_str(), histo.c_str(), 24, 0.0, 480.0, 0.0, 1.0e6, "s");
    sigmaHourly_[i]->setAxisTitle("Minutes", 1);
    sigmaHourly_[i]->setAxisTitle("occupancy rms", 2);


  }// if

}


void EBTrendClient::cleanup(void){

  if ( ! init_ ) return;

  if ( dqmStore_ ) {

    for(int i=0; i<nHists_; i++) {
      if(meanMinutely_[i]) dqmStore_->removeElement( meanMinutely_[i]->getFullname());
      meanMinutely_[i] = 0;
      if(sigmaMinutely_[i]) dqmStore_->removeElement( sigmaMinutely_[i]->getFullname());
      sigmaMinutely_[i] = 0;

      if(meanHourly_[i]) dqmStore_->removeElement( meanHourly_[i]->getFullname());
      meanHourly_[i] = 0;
      if(sigmaHourly_[i]) dqmStore_->removeElement( sigmaHourly_[i]->getFullname());
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
  ecaldqm::calcBins(20,60,start_time_,last_time_,current_time_,hourBinDiff,hourDiff);


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
    if(moduleNames_[i] == "") continue;
    histo = prefixME_ + "/" + moduleNames_[i] + "/" + histTitles_[i];
    me = dqmStore_->get(histo.c_str());
    currentHist_[i] = ecaldqm::cloneIt(me,histo);
  }


  // Get mean and rms and fill Profile

  for(int i=0; i<nHists_; i++){

    if(!meanMinutely_[i] || !sigmaMinutely_[i] || !meanHourly_[i] || !sigmaHourly_[i]) continue;

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

