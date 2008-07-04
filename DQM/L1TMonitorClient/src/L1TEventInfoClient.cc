#include "DQM/L1TMonitorClient/interface/L1TEventInfoClient.h"

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

L1TEventInfoClient::L1TEventInfoClient(const edm::ParameterSet& ps)
{
  parameters_=ps;
  initialize();
}

L1TEventInfoClient::~L1TEventInfoClient(){
 if(verbose_) cout <<"[TriggerDQM]: ending... " << endl;
}

//--------------------------------------------------------
void L1TEventInfoClient::initialize(){ 

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
void L1TEventInfoClient::beginJob(const EventSetup& context){

  if(verbose_) cout <<"[TriggerDQM]: Begin Job" << endl;
  // get backendinterface  
  dbe_ = Service<DQMStore>().operator->();

  dbe_->setCurrentFolder("L1T/EventInfo");

//  sprintf(histo, "reportSummary");
  if ( reportSummary_ = dbe_->get("L1T/EventInfo/reportSumamry") ) {
      dbe_->removeElement(reportSummary_->getName()); 
   }
  
  reportSummary_ = dbe_->bookFloat("reportSummary");

  dbe_->setCurrentFolder("L1T/EventInfo/reportSummaryContents");

  int nSubsystems = 20;
  
  char histo[100];
  
  for (int n = 0; n < nSubsystems; n++) {    

    
    switch(n){
    case 0 :   sprintf(histo,"L1T_ECAL");    break;
    case 1 :   sprintf(histo,"L1T_HCAL");    break;
    case 2 :   sprintf(histo,"L1T_RCT");     break;
    case 3 :   sprintf(histo,"L1T_GCT");     break;
    case 4 :   sprintf(histo,"L1T_DTTPG");   break;
    case 5 :   sprintf(histo,"L1T_DTTF");    break;
    case 6 :   sprintf(histo,"L1T_CSCTPG");  break;
    case 7 :   sprintf(histo,"L1T_CSCTF");   break;
    case 8 :   sprintf(histo,"L1T_RPC");     break;
    case 9 :   sprintf(histo,"L1T_GMT");     break;
    case 10 :  sprintf(histo,"L1T_GT");      break;
    case 11 :  sprintf(histo,"L1T_RPCTG");   break;
    case 12 :  sprintf(histo,"L1T_EMUL");    break;
    case 13 :  sprintf(histo,"L1T_Timing");  break;
    case 14 :  sprintf(histo,"L1T_Test1");   break;
    case 15 :  sprintf(histo,"L1T_Test2");   break;
    case 16 :  sprintf(histo,"L1T_Test3");   break;
    case 17 :  sprintf(histo,"L1T_Test4");   break;
    case 18 :  sprintf(histo,"L1T_Test5");   break;
    case 19 :  sprintf(histo,"L1T_Test6");   break;
    }  
    
  
    //    if( reportSummaryContent_[i] = dbe_->get("L1T/EventInfo/reportSummaryContents/" + histo) ) 
    //  {
    //	dbe_->removeElement(reportSummaryContent_[i]->getName());
    //  }
    
    reportSummaryContent_[n] = dbe_->bookFloat(histo);
  }
  

  dbe_->setCurrentFolder("L1T/EventInfo");

  if ( reportSummaryMap_ = dbe_->get("L1T/EventInfo/reportSummaryMap") ) {
  dbe_->removeElement(reportSummaryMap_->getName());
  }

  reportSummaryMap_ = dbe_->book2D("reportSummaryMap", "reportSummaryMap", 5, 0., 5., 4, 0., 4);
  reportSummaryMap_->setAxisTitle("Subsystem Index", 1);
  reportSummaryMap_->setAxisTitle("Subsystem Index", 2);

}

//--------------------------------------------------------
void L1TEventInfoClient::beginRun(const Run& r, const EventSetup& context) {
}

//--------------------------------------------------------
void L1TEventInfoClient::beginLuminosityBlock(const LuminosityBlock& lumiSeg, const EventSetup& context) {
   // optionally reset histograms here
}

void L1TEventInfoClient::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
                          const edm::EventSetup& c){   


  MonitorElement *GCT_QHist = dbe_->get("L1T/L1TGCT/NonIsoEmOccEtaPhi");
  MonitorElement *RCT_QHist = dbe_->get("L1T/L1TRCT/RctNonIsoEmOccEtaPhi");
  MonitorElement *GMT_QHist = dbe_->get("L1T/L1TGMT/GMT_etaphi");
  MonitorElement *CSCTF_QHist = dbe_->get("L1T/L1TCSCTF/CSCTF_occupancies");
  MonitorElement *DTTF_QHist = dbe_->get("L1T/L1TDTTF/DTTF_TRACKS/INTEG/dttf_p_phi_eta");
  
  int nSubsystems = 20;
  for (int k = 0; k < nSubsystems; k++) {
    summaryContent[k] = 1;
    reportSummaryContent_[k]->Fill(1.);
  }
  summarySum = 0;
  int GCT_nXCh = 0,GCT_nYCh=0,RCT_nXCh=0,RCT_nYCh=0,GMT_nXCh=0,GMT_nYCh=0,CSCTF_nXCh=0,CSCTF_nYCh=0,DTTF_nXCh=0,DTTF_nYCh=0;

  if(GCT_QHist){GCT_nXCh = GCT_QHist->getNbinsX(); GCT_nYCh = GCT_QHist->getNbinsY();} 
  if(RCT_QHist){RCT_nXCh = RCT_QHist->getNbinsX(); RCT_nYCh = RCT_QHist->getNbinsY();}
  if(GMT_QHist){GMT_nXCh = GMT_QHist->getNbinsX(); GMT_nYCh = GMT_QHist->getNbinsY();}
  if(CSCTF_QHist){CSCTF_nXCh = CSCTF_QHist->getNbinsX(); CSCTF_nYCh = CSCTF_QHist->getNbinsY();}
  if(DTTF_QHist){DTTF_nXCh = DTTF_QHist->getNbinsX();  DTTF_nYCh = DTTF_QHist->getNbinsY();}

  int GCT_nCh=0,RCT_nCh=0,GMT_nCh=0,CSCTF_nCh=0,DTTF_nCh=0;
  
  //cout << "CSCTF_nXCh = " << CSCTF_nXCh << ", CSCTF_nYCh = " << CSCTF_nYCh << endl;
  if(GCT_nYCh) 
    GCT_nCh = GCT_nXCh*GCT_nYCh;
  if(RCT_nYCh) 
    RCT_nCh = RCT_nXCh*RCT_nYCh;
  if(GMT_nYCh) 
    GMT_nCh = GMT_nXCh*GMT_nYCh;
  if(CSCTF_nYCh) 
    CSCTF_nCh = CSCTF_nXCh*CSCTF_nYCh;
  if(DTTF_nYCh)
    DTTF_nCh = DTTF_nXCh*DTTF_nYCh;
  

  if (GCT_QHist){
    const QReport *GCT_QReport = GCT_QHist->getQReport("HotChannels");
    if (GCT_QReport) {
      int GCT_nBadCh = GCT_QReport->getBadChannels().size();
      //cout << "nBadCh(GCT): "  << GCT_nBadCh << endl;
      summaryContent[3] =  1 - GCT_nBadCh/GCT_nCh;
      //cout << "summaryContent[0]-GCT=" << summaryContent[0] << endl;
      reportSummaryContent_[3]->Fill( summaryContent[3] );
    } 
   
    // //get list of quality tests
   //     std::vector<QReport *> Qtest_map = NonIsoEmDeadEtaPhiChannels->getQReports();
   //     cout << "Qtest_map.size() = " << Qtest_map.size() << endl;
   //     if(Qtest_map.size() > 0) {
   //       for (std::vector<QReport *>::const_iterator it=Qtest_map.begin(); it!=Qtest_map.end(); it++)
   // 	{
   // 	  //cout << endl;
   // 	  string qt_name = (*it)->getQRName();
   // 	  int qt_status = (*it)->getStatus();
   
   // 	  cout << "qt_name = " << qt_name << endl;
   // 	  cout << "qt_status = " << qt_status << endl;
   //	  
   //	}
 }

  
  if (RCT_QHist){
    const QReport *RCT_QReport = RCT_QHist->getQReport("HotChannels");
    if (RCT_QReport) {
      int RCT_nBadCh = RCT_QReport->getBadChannels().size();
      //cout << "nBadCh(RCT): "  << RCT_nBadCh << endl;
      summaryContent[2]=1-RCT_nBadCh/RCT_nCh;
      //cout << "summaryContent[1]-RCT=" << summaryContent[1] << endl;
      reportSummaryContent_[2]->Fill( summaryContent[2] );
    } 
  }

  if (GMT_QHist){
    const QReport *GMT_QReport = GMT_QHist->getQReport("HotChannels");
    //cout << "GMT_QReport: " << GMT_QReport << endl;
    if (GMT_QReport) {
      int GMT_nBadCh = GMT_QReport->getBadChannels().size();
      //cout << "nBadCh(GMT): "  << GMT_nBadCh << endl;
      //cout << "GMT_nCh: " << GMT_nCh << endl;
      summaryContent[9] = 1 - GMT_nBadCh/GMT_nCh;
      //cout << "summaryContent[2]-GMT=" << summaryContent[2] << endl;
      reportSummaryContent_[9]->Fill( summaryContent[9] );
    } 
  }

  if (CSCTF_QHist){
    const QReport *CSCTF_QReport = CSCTF_QHist->getQReport("HotChannels");
    //cout << "CSCTF_QReport: " << CSCTF_QReport << endl;
    if (CSCTF_QReport) {
      int CSCTF_nBadCh = CSCTF_QReport->getBadChannels().size();
      //cout << "nBadCh(CSCTF): "  << CSCTF_nBadCh << endl;
      summaryContent[7] = 1 - CSCTF_nBadCh/CSCTF_nCh;
      //cout << "summaryContent[3]-CSCTF=" << summaryContent[3] << endl;
      reportSummaryContent_[7]->Fill( summaryContent[7]);
    } 
  }

  if (DTTF_QHist){
    const QReport *DTTF_QReport = DTTF_QHist->getQReport("HotChannels");
    //cout << "DTTF_QReport: " << DTTF_QReport << endl;
    if (DTTF_QReport) {
      int DTTF_nBadCh = DTTF_QReport->getBadChannels().size();
      //cout << "nBadCh(DTTF): "  << DTTF_nBadCh << endl;
      summaryContent[5] = 1 - DTTF_nBadCh/DTTF_nCh;
      //cout << "summaryContent[4]-DTTF=" << summaryContent[4] << endl;
      reportSummaryContent_[5]->Fill( summaryContent[5] );
    } 
  }

  
  
  for (int m = 0; m < nSubsystems; m++) {    
    summarySum += summaryContent[m];
  }
  
  reportSummary = summarySum / nSubsystems;
//  cout << "reportSummary " << reportSummary << endl;
  if (reportSummary_) reportSummary_->Fill(reportSummary);
  

 int jcount=0;
   //fill the known systems
  for (int i = 0; i < nSubsystems; i++) {
//    cout << "summaryContent[" << i << "]" << summaryContent[i] << endl;
    if(!(i%5))jcount++;
    reportSummaryMap_->setBinContent(i%5+1,jcount, summaryContent[i]);
  }

//   //fill for known systems
//   for(int i = 1; i < 6; i++){
    
//     cout << "summaryContent[" << i-1 << "] = " << summaryContent[i-1] << endl;
//     reportSummaryMap_->setBinContent( i, 1, summaryContent[i-1] );
//   }

//   //fill the rest
//   for (int i = 1; i < 6; i++) {    
//     for (int j = 2; j < 5; j++) {    
      
//       reportSummaryMap_->setBinContent( i, j, 1. );
//     }
//   }


} 



//--------------------------------------------------------
void L1TEventInfoClient::analyze(const Event& e, const EventSetup& context){
   
  counterEvt_++;
  if (prescaleEvt_<1) return;
  if (prescaleEvt_>0 && counterEvt_%prescaleEvt_ != 0) return;
  
  if(verbose_) cout << "L1TEventInfoClient::analyze" << endl;

  


  //reportSummary = average of report summaries of each system
  
 
}

//--------------------------------------------------------
void L1TEventInfoClient::endRun(const Run& r, const EventSetup& context){
}

//--------------------------------------------------------
void L1TEventInfoClient::endJob(){
}



TH1F * L1TEventInfoClient::get1DHisto(string meName, DQMStore * dbi)
{

  MonitorElement * me_ = dbi->get(meName);

  if (!me_) { 
    if(verbose_) cout << "ME NOT FOUND." << endl;
    return NULL;
  }

  return me_->getTH1F();
}

TH2F * L1TEventInfoClient::get2DHisto(string meName, DQMStore * dbi)
{


  MonitorElement * me_ = dbi->get(meName);

  if (!me_) { 
    if(verbose_) cout << "ME NOT FOUND." << endl;
    return NULL;
  }

  return me_->getTH2F();
}



TProfile2D *  L1TEventInfoClient::get2DProfile(string meName, DQMStore * dbi)
{


  MonitorElement * me_ = dbi->get(meName);

  if (!me_) { 
     if(verbose_) cout << "ME NOT FOUND." << endl;
   return NULL;
  }

  return me_->getTProfile2D();
}


TProfile *  L1TEventInfoClient::get1DProfile(string meName, DQMStore * dbi)
{


  MonitorElement * me_ = dbi->get(meName);

  if (!me_) { 
    if(verbose_) cout << "ME NOT FOUND." << endl;
    return NULL;
  }

  return me_->getTProfile();
}








