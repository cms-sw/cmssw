#include "DQM/HLTEvF/interface/HLTEventInfoClient.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"

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
 if(verbose_) std::cout <<"[TriggerDQM]: ending... " << std::endl;
}

//--------------------------------------------------------
void HLTEventInfoClient::initialize(){ 

  counterLS_=0; 
  counterEvt_=0; 
  
  // get back-end interface
  dbe_ = Service<DQMStore>().operator->();
  
  // base folder for the contents of this job
  verbose_ = parameters_.getUntrackedParameter<bool>("verbose", false);
  
  monitorDir_ = parameters_.getUntrackedParameter<std::string>("monitorDir","");
  if(verbose_) std::cout << "Monitor dir = " << monitorDir_ << std::endl;
    
  prescaleLS_ = parameters_.getUntrackedParameter<int>("prescaleLS", -1);
  if(verbose_) std::cout << "DQM lumi section prescale = " << prescaleLS_ << " lumi section(s)"<< std::endl;
  
  prescaleEvt_ = parameters_.getUntrackedParameter<int>("prescaleEvt", -1);
  if(verbose_) std::cout << "DQM event prescale = " << prescaleEvt_ << " events(s)"<< std::endl;
  

      
}

//--------------------------------------------------------
void HLTEventInfoClient::beginJob(){

  if(verbose_) std::cout <<"[TriggerDQM]: Begin Job" << std::endl;
  // get backendinterface  
  dbe_ = Service<DQMStore>().operator->();

  dbe_->setCurrentFolder("HLT/EventInfo");

//  sprintf(histo, "reportSummary");
  reportSummary_ = dbe_->get("HLT/EventInfo/reportSumamry");
  if ( reportSummary_ ) {
      dbe_->removeElement(reportSummary_->getName()); 
   }
  
  reportSummary_ = dbe_->bookFloat("reportSummary");

  int nPDs = 20;

 //initialize reportSummary to 1
  if (reportSummary_) reportSummary_->Fill(1);

  dbe_->setCurrentFolder("HLT/EventInfo/reportSummaryContents");

  
  char histo[100];
  
  for (int n = 0; n < nPDs; n++) {    

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
  for (int k = 0; k < nPDs; k++) {
    summaryContent[k] = 1;
    reportSummaryContent_[k]->Fill(1.);
  }  


  dbe_->setCurrentFolder("HLT/EventInfo");

  reportSummaryMap_ = dbe_->get("HLT/EventInfo/reportSummaryMap");
  if ( reportSummaryMap_ ) {
  dbe_->removeElement(reportSummaryMap_->getName());
  }


  reportSummaryMap_ = dbe_->book2D("reportSummaryMap", "reportSummaryMap", 1, 1, 2, 10, 1, 11);
  reportSummaryMap_->setAxisTitle("", 1);
  reportSummaryMap_->setAxisTitle("", 2);
  reportSummaryMap_->setBinLabel(1,"SingleElectron",2);
  reportSummaryMap_->setBinLabel(2,"DoubleElectron",2);
  reportSummaryMap_->setBinLabel(3,"SingleMu",2);
  reportSummaryMap_->setBinLabel(4,"DoubleMu",2);
  reportSummaryMap_->setBinLabel(5,"Photon",2);
  reportSummaryMap_->setBinLabel(6,"Tau",2);
  reportSummaryMap_->setBinLabel(7,"BTag",2);
  reportSummaryMap_->setBinLabel(8,"HT",2);
  reportSummaryMap_->setBinLabel(9,"Jet",2);
  reportSummaryMap_->setBinLabel(10,"MET",2);
  reportSummaryMap_->setBinLabel(1," ",1);

}

//--------------------------------------------------------
void HLTEventInfoClient::beginRun(const Run& r, const EventSetup& context) {
}

//--------------------------------------------------------
void HLTEventInfoClient::beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& context) {
   // optionally reset histograms here
}

void HLTEventInfoClient::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& c) {


  int ilumi =  int(lumiSeg.id().luminosityBlock());

  const int nPDs = 10;
  MonitorElement* Pass_Hists[nPDs];
  int nPathsPD[nPDs];
  double PDResult[nPDs];
  int nTotPD[nPDs];
  for( int i = 0; i < nPDs; i++ ) {
    PDResult[i] = 1.0;
    nTotPD[i] = 0.0;
  }
  bool isCollision = true;

  for( int i = 0; i < nPDs; i++ ) {
    if( i == 0 ) Pass_Hists[i] = dbe_->get("HLT/FourVector/PathsSummary/HLT_SingleElectron_Pass_Any"); // SingleElectron
    if( i == 1 ) Pass_Hists[i] = dbe_->get("HLT/FourVector/PathsSummary/HLT_DoubleElectron_Pass_Any"); // DoubleElectron
    if( i == 2 ) Pass_Hists[i] = dbe_->get("HLT/FourVector/PathsSummary/HLT_SingleMu_Pass_Any"); // SingleMu
    if( i == 3 ) Pass_Hists[i] = dbe_->get("HLT/FourVector/PathsSummary/HLT_DoubleMu_Pass_Any"); // DoubleMu
    if( i == 4 ) Pass_Hists[i] = dbe_->get("HLT/FourVector/PathsSummary/HLT_Photon_Pass_Any"); // Photon
    if( i == 5 ) Pass_Hists[i] = dbe_->get("HLT/FourVector/PathsSummary/HLT_Tau_Pass_Any"); // Tau
    if( i == 6 ) Pass_Hists[i] = dbe_->get("HLT/FourVector/PathsSummary/HLT_BTag_Pass_Any"); // BTag
    if( i == 7 ) Pass_Hists[i] = dbe_->get("HLT/FourVector/PathsSummary/HLT_HT_Pass_Any"); // HT
    if( i == 8 ) Pass_Hists[i] = dbe_->get("HLT/FourVector/PathsSummary/HLT_Jet_Pass_Any"); // Jet
    if( i == 9 ) Pass_Hists[i] = dbe_->get("HLT/FourVector/PathsSummary/HLT_MET_Pass_Any"); // MET

    if( Pass_Hists[i] ) {
      if( i == 5 && !isCollision ) continue;
      nPathsPD[i] = Pass_Hists[i]->getNbinsX();
      int noBins = 2;
      if( i == 1 ) noBins = 3; // the last trigger is low rate
      if( i == 8 ) noBins = 4; // two last triggers are low rate

      for( int j = 0; j < nPathsPD[i]-noBins; j++ ) {
	// if triggers in each PD are too much prescaled (or low rate), skip in the summary
	
	if( i == 1 && (j == 0) ) continue; // DoubleElectron
	if( i == 3 && (j == 1 || j == 4) ) continue; // DoubleMu
	if( i == 4 && (j > 1) ) continue; // Photon
	if( i == 5 && (j > 4) ) continue; // Tau
	if( i == 7 && (j == 7) ) continue; // HT
	if( i == 8 && (j == 8) ) continue; // Jet
	if( i == 9 && (j == 8 || j == 13 || j == 15) ) continue; // MET

        double val = Pass_Hists[i]->getBinContent(j+1);
        if( val == 0 ) {
          if( ilumi > 5 ) PDResult[i] = 0.5;
        }
        nTotPD[i] += val;
      }
      if( nTotPD[i] == 0 ) {
        if( ilumi > 5 ) PDResult[i] = 0.0; 
      }
    }
    else {
      isCollision = false;
    }
  }
  
  for (int k = 0; k < nPDs; k++) {
    if( k < 10 ) {
      summaryContent[k] = PDResult[k];
      reportSummaryContent_[k]->Fill(PDResult[k]);
    }
    else {
      summaryContent[k] = 1;
      reportSummaryContent_[k]->Fill(1.);
    }
  }
  summarySum = 0;

  for (int m = 0; m < nPDs; m++) {    
    summarySum += summaryContent[m];
  }


  reportSummary = summarySum / nPDs;;
  if (reportSummary_) reportSummary_->Fill(reportSummary);


  reportSummaryMap_->setBinContent(1,1,summaryContent[0]);//SingleElectron
  reportSummaryMap_->setBinContent(1,2,summaryContent[1]);//DoubleElectron
  reportSummaryMap_->setBinContent(1,3,summaryContent[2]);//SingleMu
  reportSummaryMap_->setBinContent(1,4,summaryContent[3]);//DoubleMu
  reportSummaryMap_->setBinContent(1,5,summaryContent[4]);//Photon
  reportSummaryMap_->setBinContent(1,6,summaryContent[5]);//Tau
  reportSummaryMap_->setBinContent(1,7,summaryContent[6]);//BTag
  reportSummaryMap_->setBinContent(1,8,summaryContent[7]);//HT
  reportSummaryMap_->setBinContent(1,9,summaryContent[8]);//Jet
  reportSummaryMap_->setBinContent(1,10,summaryContent[9]);//MET

}

//--------------------------------------------------------
void HLTEventInfoClient::analyze(const Event& e, const EventSetup& context){
   
   counterEvt_++;
   if (prescaleEvt_<1) return;
   if (prescaleEvt_>0 && counterEvt_%prescaleEvt_ != 0) return;

   if(verbose_) std::cout << "HLTEventInfoClient::analyze" << std::endl;


}

//--------------------------------------------------------
void HLTEventInfoClient::endRun(const Run& r, const EventSetup& context){
}

//--------------------------------------------------------
void HLTEventInfoClient::endJob(){
}



TH1F * HLTEventInfoClient::get1DHisto(std::string meName, DQMStore * dbi)
{

  MonitorElement * me_ = dbi->get(meName);

  if (!me_) { 
    if(verbose_) std::cout << "ME NOT FOUND." << std::endl;
    return NULL;
  }

  return me_->getTH1F();
}

TH2F * HLTEventInfoClient::get2DHisto(std::string meName, DQMStore * dbi)
{


  MonitorElement * me_ = dbi->get(meName);

  if (!me_) { 
    if(verbose_) std::cout << "ME NOT FOUND." << std::endl;
    return NULL;
  }

  return me_->getTH2F();
}



TProfile2D *  HLTEventInfoClient::get2DProfile(std::string meName, DQMStore * dbi)
{


  MonitorElement * me_ = dbi->get(meName);

  if (!me_) { 
     if(verbose_) std::cout << "ME NOT FOUND." << std::endl;
   return NULL;
  }

  return me_->getTProfile2D();
}


TProfile *  HLTEventInfoClient::get1DProfile(std::string meName, DQMStore * dbi)
{


  MonitorElement * me_ = dbi->get(meName);

  if (!me_) { 
    if(verbose_) std::cout << "ME NOT FOUND." << std::endl;
    return NULL;
  }

  return me_->getTProfile();
}








