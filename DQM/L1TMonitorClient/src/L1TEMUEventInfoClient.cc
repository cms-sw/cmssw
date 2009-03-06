#include "DQM/L1TMonitorClient/interface/L1TEMUEventInfoClient.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/Core/interface/QReport.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "TRandom.h"
#include <TF1.h>
#include <stdio.h>
#include <sstream>
#include <math.h>
#include <TProfile.h>
#include <TProfile2D.h>
#include <memory>
#include <iostream>
#include <vector>
#include <iomanip>
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
  
  //tbd should revert to regular order as defined in hardwarevalidation
  // + use std labels defined in traits therein
  std::string syslabel   [nsysmon_]=
    {"DTTF","DTTPG","CSCTF","CSCTPG","RPC","GMT", "ECAL","HCAL","RCT","GCT","GT"};
  std::string syslabelext[nsysmon_]=
    {"DTF","DTP","CTF","CTP","RPC","GMT", "ETP","HTP","RCT","GCT","GLT"};
  std::vector<unsigned int> sysmask(0,nsysmon_); 
  sysmask = parameters_.getUntrackedParameter<std::vector<unsigned int> >("maskedSystems", sysmask);

  for(int i=0; i<nsysmon_; i++) {
    syslabel_[i] = syslabel[i];
    syslabelext_[i] = syslabelext[i];
    sysmask_[i] = sysmask[i];
  }
  
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

  //initialize reportSummary to 1
  if (reportSummary_) reportSummary_->Fill(1);

  dbe_->setCurrentFolder("L1TEMU/EventInfo/reportSummaryContents");

  char lbl[100];  

  for (int i=0; i<nsys_; i++) {    
    
    if(i<nsysmon_)
      sprintf(lbl,"L1TEMU_%s",syslabelext_[i].data());
    else 
      sprintf(lbl,"L1TEMU_dummy%d",i-nsysmon_+1);

    reportSummaryContent_[i] = dbe_->bookFloat(lbl);
    //if(reportSummaryContent_[i] = dbe_->get("L1T/EventInfo/reportSummaryContents/" + histo)) dbe_->removeElement(reportSummaryContent_[i]->getName());
  }

  //initialize reportSummaryContents to 1
  for (int k=0; k<nsys_; k++) {
    summaryContent[k] = 1;
    reportSummaryContent_[k]->Fill(1.);
  }  

  dbe_->setCurrentFolder("L1TEMU/EventInfo");

  if ( reportSummaryMap_ = dbe_->get("L1TEMU/EventInfo/reportSummaryMap") ) {
    dbe_->removeElement(reportSummaryMap_->getName());
  }

  reportSummaryMap_ = dbe_->book2D("reportSummaryMap", "reportSummaryMap", 1, 1, 2, 11, 1, nsysmon_+1);
  for(int i=0; i<nsysmon_; i++) {
    reportSummaryMap_->setBinLabel(i+1,syslabelext_[i],2);
  }
  reportSummaryMap_->setAxisTitle("", 1);
  reportSummaryMap_->setAxisTitle("", 2);
  reportSummaryMap_->setBinLabel(1," ",1);

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

  for (int i = 0; i < nsys_; i++) {
    summaryContent[i] = 1;    
    reportSummaryContent_[i]->Fill(1.);
  }
  summarySum = 0;

  MonitorElement* QHist[nsysmon_];   
  std::string lbl("");  
  for(int i=0; i<nsysmon_; i++) {
    lbl.clear();
    lbl+="L1TEMU/"; lbl+=syslabel_[i]; lbl+="/"; 
    lbl+=syslabelext_[i]; lbl+="ErrorFlag";
    QHist[i]=dbe_->get(lbl.data());
    float pv = -1.;
    if(!sysmask_[i])
      pv = setSummary(QHist[i]);
    summaryContent[i] = pv;
    reportSummaryContent_[i]->Fill(pv);
  }

  for (int i = 0; i < nsys_; i++) {    
    if(summaryContent[i] != -1)  summarySum += summaryContent[i];
  }
  
  reportSummary = summarySum / nsys_;
  if (reportSummary_) reportSummary_->Fill(reportSummary);

   //12x1 summary map
  for (int i=0; i< 11; i++)
    reportSummaryMap_->setBinContent(1,i+1,summaryContent[i]);

}

//--------------------------------------------------------
void L1TEMUEventInfoClient::analyze(const Event& e, const EventSetup& context){
   
   counterEvt_++;
   if (prescaleEvt_<1) return;
   if (prescaleEvt_>0 && counterEvt_%prescaleEvt_ != 0) return;

   if(verbose_) cout << "L1TEMUEventInfoClient::analyze" << endl;
}

//--------------------------------------------------------
void L1TEMUEventInfoClient::endRun(const Run& r, const EventSetup& context){
}

//--------------------------------------------------------
void L1TEMUEventInfoClient::endJob(){
}

//set subsystem pv in summary map
Float_t L1TEMUEventInfoClient::setSummary(MonitorElement* QHist) {
  bool isempty = QHist->getEntries()==0;
  //errflag bins: agree, loc agree, loc disagree, data only, emul only
  if(!isempty)
    for(int i=1; i<5; i++) 
      if(QHist->getBinContent(i)>0) 
	{isempty=false;continue;}
  return isempty ? -1. : 
    (QHist->getBinContent(1)) / (QHist->getEntries());
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








