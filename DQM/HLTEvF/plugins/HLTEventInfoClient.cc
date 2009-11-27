#include "DQM/HLTEvF/interface/HLTEventInfoClient.h"

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

HLTEventInfoClient::HLTEventInfoClient(const edm::ParameterSet& ps)
{
  parameters_=ps;
  initialize();
}

HLTEventInfoClient::~HLTEventInfoClient(){
 if(verbose_) cout <<"[TriggerDQM]: ending... " << endl;
}

//--------------------------------------------------------
void HLTEventInfoClient::initialize(){ 

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
void HLTEventInfoClient::beginJob(){

  if(verbose_) cout <<"[TriggerDQM]: Begin Job" << endl;
  // get backendinterface  
  dbe_ = Service<DQMStore>().operator->();

  dbe_->setCurrentFolder("HLT/EventInfo");

//  sprintf(histo, "reportSummary");
  reportSummary_ = dbe_->get("HLT/EventInfo/reportSumamry");
  if ( reportSummary_ ) {
      dbe_->removeElement(reportSummary_->getName()); 
   }
  
  reportSummary_ = dbe_->bookFloat("reportSummary");

  int nSubsystems = 20;

 //initialize reportSummary to 1
  if (reportSummary_) reportSummary_->Fill(1);

  dbe_->setCurrentFolder("HLT/EventInfo/reportSummaryContents");

  
  char histo[100];
  
  for (int n = 0; n < nSubsystems; n++) {    

  switch(n){
  case 0 :   sprintf(histo,"hlt_dqm_EGamma");	break;
  case 1 :   sprintf(histo,"hlt_dqm_Muon");	break;
  case 2 :   sprintf(histo,"hlt_dqm_JetMet");	break;
  case 3 :   sprintf(histo,"hlt_dqm_BJets");	break;
  case 4 :   sprintf(histo,"hlt_dqm_Tau");	break;
  case 5 :   sprintf(histo,"hlt_dqm_Test1");	break;
  case 6 :   sprintf(histo,"hlt_dqm_Test2");	break;
  case 7 :   sprintf(histo,"hlt_dqm_Test3");	break;
  case 8 :   sprintf(histo,"hlt_dqm_Test4");	break;
  case 9 :   sprintf(histo,"hlt_dqm_Test5");	break;
  case 10 :  sprintf(histo,"hlt_dqm_Test6");	break;
  case 11 :  sprintf(histo,"hlt_dqm_Test7");	break;
  case 12 :  sprintf(histo,"hlt_dqm_Test8");	break;
  case 13 :  sprintf(histo,"hlt_dqm_Test9");	break;
  case 14 :  sprintf(histo,"hlt_dqm_Test10");	break;
  case 15 :  sprintf(histo,"hlt_dqm_Test11");	break;
  case 16 :  sprintf(histo,"hlt_dqm_Test12");	break;
  case 17 :  sprintf(histo,"hlt_dqm_Test13");	break;
  case 18 :  sprintf(histo,"hlt_dqm_Test14");	break;
  case 19 :  sprintf(histo,"hlt_dqm_Test15");	break;
  }
  





















  
//  if( reportSummaryContent_[i] = dbe_->get("HLT/EventInfo/reportSummaryContents/" + histo) ) 
//  {
//       dbe_->removeElement(reportSummaryContent_[i]->getName());
//   }
  
   reportSummaryContent_[n] = dbe_->bookFloat(histo);
  }

  //initialize reportSummaryContents to 1
  for (int k = 0; k < nSubsystems; k++) {
    summaryContent[k] = 1;
    reportSummaryContent_[k]->Fill(1.);
  }  


  dbe_->setCurrentFolder("HLT/EventInfo");

  reportSummaryMap_ = dbe_->get("HLT/EventInfo/reportSummaryMap");
  if ( reportSummaryMap_ ) {
  dbe_->removeElement(reportSummaryMap_->getName());
  }


  reportSummaryMap_ = dbe_->book2D("reportSummaryMap", "reportSummaryMap", 1, 1, 2, 5, 1, 6);
  reportSummaryMap_->setAxisTitle("", 1);
  reportSummaryMap_->setAxisTitle("", 2);
  reportSummaryMap_->setBinLabel(1,"EGAMMA",2);
  reportSummaryMap_->setBinLabel(2,"MUON",2);
  reportSummaryMap_->setBinLabel(3,"JETMET",2);
  reportSummaryMap_->setBinLabel(4,"BJETS",2);
  reportSummaryMap_->setBinLabel(5,"TAU",2);
  reportSummaryMap_->setBinLabel(1," ",1);

}

//--------------------------------------------------------
void HLTEventInfoClient::beginRun(const Run& r, const EventSetup& context) {
}

//--------------------------------------------------------
void HLTEventInfoClient::beginLuminosityBlock(const LuminosityBlock& lumiSeg, const EventSetup& context) {
   // optionally reset histograms here
}

void HLTEventInfoClient::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
                          const edm::EventSetup& c){

  MonitorElement *Muon_QHist = dbe_->get("HLT/HLTMonMuon/Summary/Ratio_HLT_L1MuOpen");

  float muonResult = 0;

  if(Muon_QHist){
    const QReport *Muon_QReport = Muon_QHist->getQReport("CompareHist_Shape");
    if(Muon_QReport) muonResult = Muon_QReport->getQTresult();
  }

  int nSubsystems = 20;
  for (int k = 0; k < nSubsystems; k++) {
    // mask all HLT applications
    //if(k == 1 && muonResult != -1){
    //  summaryContent[k] = muonResult;
    //  reportSummaryContent_[k]->Fill(muonResult);
    //}else{
      summaryContent[k] = 1;
      reportSummaryContent_[k]->Fill(1.);
    //}
  }
  summarySum = 0;

  for (int m = 0; m < nSubsystems; m++) {    
    summarySum += summaryContent[m];
  }


  reportSummary = summarySum / nSubsystems;;
  if (reportSummary_) reportSummary_->Fill(reportSummary);


  reportSummaryMap_->setBinContent(1,1,summaryContent[0]);//Egamma
  reportSummaryMap_->setBinContent(1,2,summaryContent[1]);//Muon
  reportSummaryMap_->setBinContent(1,3,summaryContent[2]);//JetMet
  reportSummaryMap_->setBinContent(1,4,summaryContent[3]);//BJets
  reportSummaryMap_->setBinContent(1,5,summaryContent[4]);//Taus

}

//--------------------------------------------------------
void HLTEventInfoClient::analyze(const Event& e, const EventSetup& context){
   
   counterEvt_++;
   if (prescaleEvt_<1) return;
   if (prescaleEvt_>0 && counterEvt_%prescaleEvt_ != 0) return;

   if(verbose_) cout << "HLTEventInfoClient::analyze" << endl;


}

//--------------------------------------------------------
void HLTEventInfoClient::endRun(const Run& r, const EventSetup& context){
}

//--------------------------------------------------------
void HLTEventInfoClient::endJob(){
}



TH1F * HLTEventInfoClient::get1DHisto(string meName, DQMStore * dbi)
{

  MonitorElement * me_ = dbi->get(meName);

  if (!me_) { 
    if(verbose_) cout << "ME NOT FOUND." << endl;
    return NULL;
  }

  return me_->getTH1F();
}

TH2F * HLTEventInfoClient::get2DHisto(string meName, DQMStore * dbi)
{


  MonitorElement * me_ = dbi->get(meName);

  if (!me_) { 
    if(verbose_) cout << "ME NOT FOUND." << endl;
    return NULL;
  }

  return me_->getTH2F();
}



TProfile2D *  HLTEventInfoClient::get2DProfile(string meName, DQMStore * dbi)
{


  MonitorElement * me_ = dbi->get(meName);

  if (!me_) { 
     if(verbose_) cout << "ME NOT FOUND." << endl;
   return NULL;
  }

  return me_->getTProfile2D();
}


TProfile *  HLTEventInfoClient::get1DProfile(string meName, DQMStore * dbi)
{


  MonitorElement * me_ = dbi->get(meName);

  if (!me_) { 
    if(verbose_) cout << "ME NOT FOUND." << endl;
    return NULL;
  }

  return me_->getTProfile();
}








