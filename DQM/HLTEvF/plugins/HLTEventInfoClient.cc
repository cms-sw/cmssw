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
void HLTEventInfoClient::beginJob(const EventSetup& context){

  if(verbose_) cout <<"[TriggerDQM]: Begin Job" << endl;
  // get backendinterface  
  dbe_ = Service<DQMStore>().operator->();

  dbe_->setCurrentFolder("HLT/EventInfo");

//  sprintf(histo, "reportSummary");
  if ( reportSummary_ = dbe_->get("HLT/EventInfo/reportSumamry") ) {
      dbe_->removeElement(reportSummary_->getName()); 
   }
  
  reportSummary_ = dbe_->bookFloat("reportSummary");

  dbe_->setCurrentFolder("HLT/EventInfo/reportSummaryContents");

  int nSubsystems = 20;
  
  char histo[100];
  
  for (int i = 0; i < nSubsystems; i++) {    

// ugly hack for cruzet2
  if(i==0)  sprintf(histo,"hlt_dqm_EGamma");
  if(i==1)  sprintf(histo,"hlt_dqm_Muon");
  if(i==2)  sprintf(histo,"hlt_dqm_JetMet");
  if(i==3)  sprintf(histo,"hlt_dqm_BJets");
  if(i==4)  sprintf(histo,"hlt_dqm_Tau");
  if(i==5)  sprintf(histo,"hlt_dqm_Test1");
  if(i==6)  sprintf(histo,"hlt_dqm_Test2");
  if(i==7)  sprintf(histo,"hlt_dqm_Test3");
  if(i==8)  sprintf(histo,"hlt_dqm_Test4");
  if(i==9)  sprintf(histo,"hlt_dqm_Test5");
  if(i==10) sprintf(histo,"hlt_dqm_Test6");
  if(i==11) sprintf(histo,"hlt_dqm_Test7");
  if(i==12) sprintf(histo,"hlt_dqm_Test8");
  if(i==13) sprintf(histo,"hlt_dqm_Test9");
  if(i==14) sprintf(histo,"hlt_dqm_Test10");
  if(i==15) sprintf(histo,"hlt_dqm_Test11");
  if(i==16) sprintf(histo,"hlt_dqm_Test12");
  if(i==17) sprintf(histo,"hlt_dqm_Test13");
  if(i==18) sprintf(histo,"hlt_dqm_Test14");
  if(i==19) sprintf(histo,"hlt_dqm_Test15");
  


  
//  if( reportSummaryContent_[i] = dbe_->get("HLT/EventInfo/reportSummaryContents/" + histo) ) 
//  {
//       dbe_->removeElement(reportSummaryContent_[i]->getName());
//   }
  
   reportSummaryContent_[i] = dbe_->bookFloat(histo);
  }


  dbe_->setCurrentFolder("HLT/EventInfo");

  if ( reportSummaryMap_ = dbe_->get("HLT/EventInfo/reportSummaryMap") ) {
  dbe_->removeElement(reportSummaryMap_->getName());
  }

  reportSummaryMap_ = dbe_->book2D("reportSummaryMap", "reportSummaryMap", 5, 0., 5., 4, 0., 4);
			    reportSummaryMap_->setAxisTitle("XXXX", 1);
			      reportSummaryMap_->setAxisTitle("YYYY", 2);

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


}

//--------------------------------------------------------
void HLTEventInfoClient::analyze(const Event& e, const EventSetup& context){
   
   counterEvt_++;
   if (prescaleEvt_<1) return;
   if (prescaleEvt_>0 && counterEvt_%prescaleEvt_ != 0) return;

   if(verbose_) cout << "HLTEventInfoClient::analyze" << endl;
/*
// check GCT
  MonitorElement *NonIsoEmDeadEtaChannels = dbe_->get("HLT/HLTGCT/NonIsoEmOccEta");
  if(!NonIsoEmDeadEtaChannels) gctFloat = 0.;
  
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


  int nSubsystems = 20;
  for (int i = 0; i < nSubsystems; i++) {    

     reportSummaryContent_[i]->Fill(1.);
  }

  for (int i = 0; i < 5; i++) {    
    for (int j = 0; j < 4; j++) {    

     reportSummaryMap_->setBinContent( i, j, 1. );
    }
  }

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








