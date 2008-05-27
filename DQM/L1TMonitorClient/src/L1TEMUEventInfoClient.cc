#include "DQM/L1TMonitorClient/interface/L1TEMUEventInfoClient.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/Core/interface/QReport.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "TRandom.h"
#include <TF1.h>
#include <stdio.h>
#include <sstream>
#include <math.h>
#include <TProfile.h>
#include <TProfile2D.h>
#include <memory>
#include <iostream>
#include <iomanip>
#include <map>
#include <vector>
#include <string>
#include <fstream>
#include "TROOT.h"

using namespace edm;
using namespace std;

L1TEMUEventInfoClient::L1TEMUEventInfoClient(const edm::ParameterSet& ps)
{
  parameters_=ps;
  initialize();
}

L1TEMUEventInfoClient::~L1TEMUEventInfoClient(){
 if(verbose_) cout <<"[TriggerDQM]: ending... " << endl;
}

//--------------------------------------------------------
void L1TEMUEventInfoClient::initialize(){ 

  counterLS_=0; 
  counterEvt_=0; 
  
  // get back-end interface
  dbe_ = Service<DQMStore>().operator->();
  
  // base folder for the contents of this job
  verbose_ = parameters_.getUntrackedParameter<bool>("verbose", false);
  
  monitorDir_ = parameters_.getUntrackedParameter<string>("monitorDir","");
  if(verbose_) cout << "Monitor dir = " << monitorDir_ << endl;
    
  prescaleLS_ = parameters_.getUntrackedParameter<int>("prescaleLS", -1);
  if(verbose_) cout << "DQM lumi section prescale = " << prescaleLS_ << " lumi section(s)"<< endl;
  
  prescaleEvt_ = parameters_.getUntrackedParameter<int>("prescaleEvt", -1);
  if(verbose_) cout << "DQM event prescale = " << prescaleEvt_ << " events(s)"<< endl;
  

      
}

//--------------------------------------------------------
void L1TEMUEventInfoClient::beginJob(const EventSetup& context){

  if(verbose_) cout <<"[TriggerDQM]: Begin Job" << endl;
  // get backendinterface  
  dbe_ = Service<DQMStore>().operator->();

  dbe_->setCurrentFolder("L1TEMU/EventInfo");

//  sprintf(histo, "reportSummary");
  if ( reportSummary_ = dbe_->get("L1TEMU/EventInfo/reportSumamry") ) {
      dbe_->removeElement(reportSummary_->getName()); 
   }
  
  reportSummary_ = dbe_->bookFloat("reportSummary");

  dbe_->setCurrentFolder("L1TEMU/EventInfo/reportSummaryContents");

  int nSubsystems = 10;
  
  char histo[100];
  
  for (int i = 0; i < nSubsystems; i++) {    

  sprintf(histo,"status sub-system %d",i);
  
//  if( reportSummaryContent_[i] = dbe_->get("L1T/EventInfo/reportSummaryContents/" + histo) ) 
//  {
//       dbe_->removeElement(reportSummaryContent_[i]->getName());
//   }
  
   reportSummaryContent_[i] = dbe_->bookFloat(histo);
  }


  dbe_->setCurrentFolder("L1TEMU/EventInfo");

  if ( reportSummaryMap_ = dbe_->get("L1TEMU/EventInfo/reportSummaryMap") ) {
  dbe_->removeElement(reportSummaryMap_->getName());
  }

  reportSummaryMap_ = dbe_->book2D("reportSummaryMap", "reportSummaryMap", 4, 0., 4., 3, 0., 3);
			    reportSummaryMap_->setAxisTitle("XXXX", 1);
			      reportSummaryMap_->setAxisTitle("YYYY", 2);

}

//--------------------------------------------------------
void L1TEMUEventInfoClient::beginRun(const Run& r, const EventSetup& context) {
}

//--------------------------------------------------------
void L1TEMUEventInfoClient::beginLuminosityBlock(const LuminosityBlock& lumiSeg, const EventSetup& context) {
   // optionally reset histograms here
}

void L1TEMUEventInfoClient::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
                          const edm::EventSetup& c){


}

//--------------------------------------------------------
void L1TEMUEventInfoClient::analyze(const Event& e, const EventSetup& context){
   
   counterEvt_++;
   if (prescaleEvt_<1) return;
   if (prescaleEvt_>0 && counterEvt_%prescaleEvt_ != 0) return;

   if(verbose_) cout << "L1TEMUEventInfoClient::analyze" << endl;

/*
  MonitorElement *NonIsoEmDeadEtaChannels = dbe_->get("L1T/L1TGCT/NonIsoEmOccEta");
  int nXChannels = NonIsoEmDeadEtaChannels->getNbinsX();
  int nYChannels = NonIsoEmDeadEtaChannels->getNbinsY();
  if(nYChannels) nChannels = nXChannels*nYChannels;
  
  if (NonIsoEmDeadEtaChannels){
    const QReport *NonIsoEmDeadEtaQReport = NonIsoEmDeadEtaChannels->getQReport("DeadChannels");
    if (NonIsoEmDeadEtaQReport) {
      int nBadChannels = NonIsoEmDeadEtaQReport->getBadChannels().size();
      reportSummary = nBadChannels/nChannels;
    } 
  }   
*/ 
  reportSummary = 1.;
  if (reportSummary_) reportSummary_->Fill(reportSummary);


  int nSubsystems = 10;
  for (int i = 0; i < nSubsystems; i++) {    

     reportSummaryContent_[i]->Fill(1.);
  }

  for (int i = 0; i < 4; i++) {    
    for (int j = 0; j < 3; j++) {    

     reportSummaryMap_->setBinContent( i, j, 1. );
    }
  }

}

//--------------------------------------------------------
void L1TEMUEventInfoClient::endRun(const Run& r, const EventSetup& context){
}

//--------------------------------------------------------
void L1TEMUEventInfoClient::endJob(){
}



TH1F * L1TEMUEventInfoClient::get1DHisto(string meName, DQMStore * dbi)
{

  MonitorElement * me_ = dbi->get(meName);

  if (!me_) { 
    if(verbose_) cout << "ME NOT FOUND." << endl;
    return NULL;
  }

  return me_->getTH1F();
}

TH2F * L1TEMUEventInfoClient::get2DHisto(string meName, DQMStore * dbi)
{


  MonitorElement * me_ = dbi->get(meName);

  if (!me_) { 
    if(verbose_) cout << "ME NOT FOUND." << endl;
    return NULL;
  }

  return me_->getTH2F();
}



TProfile2D *  L1TEMUEventInfoClient::get2DProfile(string meName, DQMStore * dbi)
{


  MonitorElement * me_ = dbi->get(meName);

  if (!me_) { 
     if(verbose_) cout << "ME NOT FOUND." << endl;
   return NULL;
  }

  return me_->getTProfile2D();
}


TProfile *  L1TEMUEventInfoClient::get1DProfile(string meName, DQMStore * dbi)
{


  MonitorElement * me_ = dbi->get(meName);

  if (!me_) { 
    if(verbose_) cout << "ME NOT FOUND." << endl;
    return NULL;
  }

  return me_->getTProfile();
}








