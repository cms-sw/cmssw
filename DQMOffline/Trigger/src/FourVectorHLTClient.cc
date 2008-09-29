#include "DQMOffline/Trigger/interface/FourVectorHLTClient.h"

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

FourVectorHLTClient::FourVectorHLTClient(const edm::ParameterSet& ps)
{
  parameters_=ps;
  initialize();
}

FourVectorHLTClient::~FourVectorHLTClient(){
  LogDebug("FourVectorHLTClient")<< "FourVectorHLTClient: ending...." ;
}

//--------------------------------------------------------
void FourVectorHLTClient::initialize(){ 

  counterLS_=0; 
  counterEvt_=0; 
  
  // get back-end interface
  dbe_ = Service<DQMStore>().operator->();
  
  // base folder for the contents of this job
  monitorDir_ = parameters_.getUntrackedParameter<string>("monitorDir","");
  LogDebug("FourVectorHLTClient")<< "Monitor dir = " << monitorDir_ << endl;
    
  prescaleLS_ = parameters_.getUntrackedParameter<int>("prescaleLS", -1);
  LogDebug("FourVectorHLTClient")<< "DQM lumi section prescale = " << prescaleLS_ << " lumi section(s)"<< endl;
  
  prescaleEvt_ = parameters_.getUntrackedParameter<int>("prescaleEvt", -1);
  LogDebug("FourVectorHLTClient")<< "DQM event prescale = " << prescaleEvt_ << " events(s)"<< endl;
  

      
}

//--------------------------------------------------------
void FourVectorHLTClient::beginJob(const EventSetup& context){


  LogDebug("FourVectorHLTClient")<<"[FourVectorHLTClient]: Begin Job" << endl;
  // get backendinterface  
  dbe_ = Service<DQMStore>().operator->();

  dbe_->setCurrentFolder("HLTOffline/FourVectorClient/reportSummaryContents");
  //dbe_->setCurrentFolder("HLTClient");

//  sprintf(histo, "reportSummary");
  if ( reportSummary_ = dbe_->get("HLTOffline/FourVectorClient/reportSumaryContents/reportSummary") ) {
      dbe_->removeElement(reportSummary_->getName()); 
   }
  
  reportSummary_ = dbe_->bookFloat("reportSummary");

  //initialize reportSummary to 1
  if (reportSummary_) reportSummary_->Fill(1);

  
  int nSubsystems = 20;

  char histo[100];
  
  for (int n = 0; n < nSubsystems; n++) {    

    
    switch(n){
    case 0 :   sprintf(histo,"HLT_ECAL");    break;
    case 1 :   sprintf(histo,"HLT_HCAL");    break;
    case 2 :   sprintf(histo,"HLT_MUON");     break;
    case 3 :   sprintf(histo,"HLT_JET");     break;
    case 4 :   sprintf(histo,"HLT_DTTPG");   break;
    case 5 :   sprintf(histo,"HLT_DTTF");    break;
    case 6 :   sprintf(histo,"HLT_CSCTPG");  break;
    case 7 :   sprintf(histo,"HLT_PHOTON");   break;
    case 8 :   sprintf(histo,"HLT_RPC");     break;
    case 9 :   sprintf(histo,"HLT_ELECTRON");     break;
    case 10 :  sprintf(histo,"HLT_GT");      break;
    case 11 :  sprintf(histo,"HLT_RPCTG");   break;
    case 12 :  sprintf(histo,"HLT_EMUL");    break;
    case 13 :  sprintf(histo,"HLT_Timing");  break;
    case 14 :  sprintf(histo,"HLT_Test1");   break;
    case 15 :  sprintf(histo,"HLT_Test2");   break;
    case 16 :  sprintf(histo,"HLT_Test3");   break;
    case 17 :  sprintf(histo,"HLT_Test4");   break;
    case 18 :  sprintf(histo,"HLT_Test5");   break;
    case 19 :  sprintf(histo,"HLT_Test6");   break;
    }  
    
  
    //    if( reportSummaryContent_[i] = dbe_->get("L1T/EventInfo/reportSummaryContents/" + histo) ) 
    //  {
    //	dbe_->removeElement(reportSummaryContent_[i]->getName());
    //  }
    
    reportSummaryContent_[n] = dbe_->bookFloat(histo);
  }

  //initialize reportSummaryContents to 1
  for (int k = 0; k < nSubsystems; k++) {
    summaryContent[k] = 1;
    reportSummaryContent_[k]->Fill(1.);
  }  


  dbe_->setCurrentFolder("HLTOffline/FourVectorClient/EventInfo");

  if ( reportSummaryMap_ = dbe_->get("HLTOffline/FourVectorClient/EventInfo/reportSummaryMap") ) {
  dbe_->removeElement(reportSummaryMap_->getName());
  }

  //reportSummaryMap_ = dbe_->book2D("reportSummaryMap", "reportSummaryMap", 5, 1, 6, 4, 1, 5);
  reportSummaryMap_ = dbe_->book2D("reportSummaryMap", "reportSummaryMap", 1, 1, 2, 8, 1, 9);
  reportSummaryMap_->setAxisTitle("", 1);
  reportSummaryMap_->setAxisTitle("", 2);
  reportSummaryMap_->setBinLabel(1,"DTTF",2);
  reportSummaryMap_->setBinLabel(2,"PHO",2);
  reportSummaryMap_->setBinLabel(4,"ELE",2);
  reportSummaryMap_->setBinLabel(5,"MUO",2);
  reportSummaryMap_->setBinLabel(6,"JET",2);
  reportSummaryMap_->setBinLabel(8,"Timing",2);
  reportSummaryMap_->setBinLabel(1," ",1);

}

//--------------------------------------------------------
void FourVectorHLTClient::beginRun(const Run& r, const EventSetup& context) {
}

//--------------------------------------------------------
void FourVectorHLTClient::beginLuminosityBlock(const LuminosityBlock& lumiSeg, const EventSetup& context) {
   // optionally reset histograms here
}

void FourVectorHLTClient::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
                          const edm::EventSetup& c){   


  MonitorElement *JET_QHist = dbe_->get("HLTOffline/FourVectorHLTOfflinehltResults/HLTJet30_etaphiOff");
  MonitorElement *MUO_QHist = dbe_->get("HLTOffline/FourVectorHLTOfflinehltResults/HLT_L1MuOpen_etaphiOff");
  MonitorElement *ELE_QHist = dbe_->get("HLTOffline/FourVectorHLTOfflinehltResults/HLT_IsoEle18_L1R_etaphiOff");
  MonitorElement *PHO_QHist = dbe_->get("HLTOffline/FourVectorHLTOfflinehltResults/HLT_IsoPhoton10_L1R_etaphiOff");
  MonitorElement *DTTF_QHist = dbe_->get("L1T/L1TDTTF/DTTF_TRACKS/INTEG");

  vector<string> hltMEs;
  if(dbe_->containsAnyMonitorable("L1T/L1TDTTF/DTTF_TRACKS/INTEG/Occupancy Summary")) cout << "Interesting dir has MEs " << endl;
  
  //MonitorElement *DTTF_QHist_phi = dbe_->get("L1T/L1TDTTF/DTTF_TRACKS/INTEG/Integrated Packed Phi");
  //MonitorElement *DTTF_QHist_pt = dbe_->get("L1T/L1TDTTF/DTTF_TRACKS/INTEG/Integrated Packed Pt");
  //MonitorElement *DTTF_QHist_qual = dbe_->get("L1T/L1TDTTF/DTTF_TRACKS/INTEG/Integrated Packed Quality");


  int nSubsystems = 20;
  for (int k = 0; k < nSubsystems; k++) {
    summaryContent[k] = 1;
    reportSummaryContent_[k]->Fill(1.);
  }
  summarySum = 0;

  
  int JET_nXCh = 0,JET_nYCh=0,MUO_nXCh=0,MUO_nYCh=0,ELE_nXCh=0,ELE_nYCh=0,PHO_nXCh=0,PHO_nYCh=0,DTTF_nXCh=0,DTTF_nYCh=0;

  if(JET_QHist){
    JET_nXCh = JET_QHist->getNbinsX(); 
    JET_nYCh = JET_QHist->getNbinsY();
  }
  if(MUO_QHist){
    MUO_nXCh = MUO_QHist->getNbinsX(); 
    MUO_nYCh = MUO_QHist->getNbinsY();
  }
  if(ELE_QHist){
    ELE_nXCh = ELE_QHist->getNbinsX(); 
    ELE_nYCh = ELE_QHist->getNbinsY();
  }
  if(PHO_QHist){
    PHO_nXCh = PHO_QHist->getNbinsX(); 
    PHO_nYCh = PHO_QHist->getNbinsY();
  }
  if(DTTF_QHist){
    DTTF_nXCh = DTTF_QHist->getNbinsX();  
    DTTF_nYCh = DTTF_QHist->getNbinsY();
  } 


  int JET_nCh=0,MUO_nCh=0,ELE_nCh=0,PHO_nCh=0,DTTF_nCh=0;
  
  if(JET_nYCh) 
    JET_nCh = JET_nXCh*JET_nYCh;
  if(MUO_nYCh) 
    MUO_nCh = MUO_nXCh*MUO_nYCh;
  if(ELE_nYCh) 
    ELE_nCh = ELE_nXCh*ELE_nYCh;
  if(PHO_nYCh) 
    PHO_nCh = PHO_nXCh*PHO_nYCh;
  if(DTTF_nYCh)
    DTTF_nCh = DTTF_nXCh*DTTF_nYCh;
  

  if (JET_QHist){
    const QReport *JET_QReport = JET_QHist->getQReport("HotChannels_JET");
    if (JET_QReport) {
      int JET_nBadCh = JET_QReport->getBadChannels().size();
      //cout << "nBadCh(JET): "  << JET_nBadCh << endl;
      summaryContent[3] =  1 - JET_nBadCh/JET_nCh;
      //cout << "summaryContent[0]-JET=" << summaryContent[0] << endl;
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

  
  if (MUO_QHist){
    const QReport *MUO_QReport = MUO_QHist->getQReport("HotChannels_MUO");
    if (MUO_QReport) {
      int MUO_nBadCh = MUO_QReport->getBadChannels().size();
      summaryContent[2]=1-MUO_nBadCh/MUO_nCh;
      reportSummaryContent_[2]->Fill( summaryContent[2] );
    } 
  }

  if (ELE_QHist){
    const QReport *ELE_QReport = ELE_QHist->getQReport("HotChannels_ELE");
    if (ELE_QReport) {
      int ELE_nBadCh = ELE_QReport->getBadChannels().size();
      summaryContent[9] = 1 - ELE_nBadCh/ELE_nCh;
      reportSummaryContent_[9]->Fill( summaryContent[9] );
    } 
  }

  if (PHO_QHist){
    const QReport *PHO_QReport = PHO_QHist->getQReport("HotChannels_PHO");
    if (PHO_QReport) {
      int PHO_nBadCh = PHO_QReport->getBadChannels().size();
      summaryContent[7] = 1 - PHO_nBadCh/PHO_nCh;
      reportSummaryContent_[7]->Fill( summaryContent[7]);
    } 
  }

// place holder and an exmaple from L1TEventInfoClient
  if (DTTF_QHist){
//      const QReport *DTTF_QReport = DTTF_QHist->getQReport("HotChannels_DTTF_2D");
//      cout << "DTTF_QReport: " << DTTF_QReport << endl;

//      if (DTTF_QReport) {
//        int DTTF_nBadCh = DTTF_QReport->getBadChannels().size();
//        cout << "nBadCh(DTTF): "  << DTTF_nBadCh << endl;
//        cout << "hotchannel: " << DTTF_QReport->getQRName() << endl;
//        cout << "hotchannel: " << DTTF_QReport->getMessage() << endl;
//        cout << "getStatus: " << DTTF_QReport->getStatus() << endl;
//      } 

//     if (DTTF_QReport) {
//       int DTTF_nBadCh = DTTF_QReport->getBadChannels().size();
//       //cout << "nBadCh(DTTF): "  << DTTF_nBadCh << endl;
//       summaryContent[5] = 1 - DTTF_nBadCh/DTTF_nCh;
//       //cout << "summaryContent[4]-DTTF=" << summaryContent[4] << endl;
//       reportSummaryContent_[5]->Fill( summaryContent[5] );
//    } 

    //summaryContent = fraction of filled bins
    int nFilledBins = 0;
    int nTotalBins  = 72;
    for(int i=1; i<7; i++)//6 logical wheels
      for(int j=1; j<13;j++){ //12 sectors
	if(DTTF_QHist->getBinContent(i,j)) nFilledBins++;
      }
    summaryContent[5] = (float)nFilledBins / (float)nTotalBins;
    reportSummaryContent_[5]->Fill( summaryContent[5] );
  }

  //obtain results from Comp2RefChi2 test
//   if(DTTF_QHist_phi){
//     const QReport *DTTF_QReport_phi = DTTF_QHist_phi->getQReport("CompareHist");
//     if (DTTF_QReport_phi){
//       cout << "phi: " << DTTF_QReport_phi->getQRName() << endl;
//       cout << "phi: " << DTTF_QReport_phi->getMessage() << endl;
//       cout << "getStatus: " << DTTF_QReport_phi->getStatus() << endl;
//     }
    
//    const QReport *DTTF_QReport_phi2 = DTTF_QHist_phi->getQReport("HotChannels_DTTF_phi");
//     cout << "DTTF_QReport_phi2: " << DTTF_QReport_phi2 << endl;
//     if (DTTF_QReport_phi2) {
//       int DTTF_nBadCh = DTTF_QReport_phi2->getBadChannels().size();
//       cout << "nBadCh(DTTF): "  << DTTF_nBadCh << endl;
//       cout << "hotchannel: " << DTTF_QReport_phi2->getQRName() << endl;
//       cout << "hotchannel: " << DTTF_QReport_phi2->getMessage() << endl;
//       cout << "getStatus: " << DTTF_QReport_phi2->getStatus() << endl;
//     } 

// }
  
//   if(DTTF_QHist_pt){
//     const QReport *DTTF_QReport_pt = DTTF_QHist_pt->getQReport("CompareHist");
//     if (DTTF_QReport_pt){
//       cout << "pt: " << DTTF_QReport_pt->getQRName() << endl;
//       cout << "pt: " << DTTF_QReport_pt->getMessage() << endl;
//       cout << "getStatus: " << DTTF_QReport_pt->getStatus() << endl;
//     }
  
//     const QReport *DTTF_QReport_pt2 = DTTF_QHist_pt->getQReport("HotChannels_DTTF_pt");
//     cout << "DTTF_QReport_pt2: " << DTTF_QReport_pt2 << endl;
//     if (DTTF_QReport_pt2) {
//       int DTTF_nBadCh = DTTF_QReport_pt2->getBadChannels().size();
//       cout << "nBadCh(DTTF): "  << DTTF_nBadCh << endl;
//       cout << "hotchannel: " << DTTF_QReport_pt2->getQRName() << endl;
//       cout << "hotchannel: " << DTTF_QReport_pt2->getMessage() << endl;
//       cout << "getStatus: " << DTTF_QReport_pt2->getStatus() << endl;
//     }

//  }
  
//   if(DTTF_QHist_qual){
//     const QReport *DTTF_QReport_qual = DTTF_QHist_qual->getQReport("CompareHist");
//     if (DTTF_QReport_qual){
//       cout << "qual: " << DTTF_QReport_qual->getQRName() << endl;
//       cout << "qual: " << DTTF_QReport_qual->getMessage() << endl;
//       cout << "getStatus: " << DTTF_QReport_qual->getStatus() << endl;
//     }
//   }
  
  
  for (int m = 0; m < nSubsystems; m++) {    
    summarySum += summaryContent[m];
  }
  
  reportSummary = summarySum / nSubsystems;
  //cout << "reportSummary " << reportSummary << endl;
  if (reportSummary_) reportSummary_->Fill(reportSummary);
  

    //5x4 summary map
//  int jcount=0;
//    //fill the known systems
//   for (int i = 0; i < nSubsystems; i++) {
//     cout << "summaryContent[" << i << "]" << summaryContent[i] << endl;
//     if(!(i%5))jcount++;
//     reportSummaryMap_->setBinContent(i%5+1,jcount, summaryContent[i]);
//   }

   //8x1 summary map
  reportSummaryMap_->setBinContent(1,1,summaryContent[5]);//DTTF
  reportSummaryMap_->setBinContent(1,2,summaryContent[7]);//PHO
  reportSummaryMap_->setBinContent(1,4,summaryContent[1]);//ELE
  reportSummaryMap_->setBinContent(1,5,summaryContent[9]);//MUO
  reportSummaryMap_->setBinContent(1,6,summaryContent[3]);//JET
  reportSummaryMap_->setBinContent(1,8,summaryContent[13]);//Timing

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
void FourVectorHLTClient::analyze(const Event& e, const EventSetup& context){
   
  counterEvt_++;
  if (prescaleEvt_<1) return;
  if (prescaleEvt_>0 && counterEvt_%prescaleEvt_ != 0) return;
  
  LogDebug("FourVectorHLTClient")<<"analyze..." << endl;

  


  //reportSummary = average of report summaries of each system
  
 
}

//--------------------------------------------------------
void FourVectorHLTClient::endRun(const Run& r, const EventSetup& context){
}

//--------------------------------------------------------
void FourVectorHLTClient::endJob(){
}



TH1F * FourVectorHLTClient::get1DHisto(string meName, DQMStore * dbi)
{

  MonitorElement * me_ = dbi->get(meName);

  if (!me_) { 
    LogDebug("FourVectorHLTClient")<< "ME NOT FOUND." << endl;
    return NULL;
  }

  return me_->getTH1F();
}

TH2F * FourVectorHLTClient::get2DHisto(string meName, DQMStore * dbi)
{


  MonitorElement * me_ = dbi->get(meName);

  if (!me_) { 
    LogDebug("FourVectorHLTClient")<< "ME NOT FOUND." << endl;
    return NULL;
  }

  return me_->getTH2F();
}



TProfile2D *  FourVectorHLTClient::get2DProfile(string meName, DQMStore * dbi)
{


  MonitorElement * me_ = dbi->get(meName);

  if (!me_) { 
    LogDebug("FourVectorHLTClient")<< "ME NOT FOUND." << endl;
   return NULL;
  }

  return me_->getTProfile2D();
}


TProfile *  FourVectorHLTClient::get1DProfile(string meName, DQMStore * dbi)
{


  MonitorElement * me_ = dbi->get(meName);

  if (!me_) { 
    LogDebug("FourVectorHLTClient")<< "ME NOT FOUND." << endl;
    return NULL;
  }

  return me_->getTProfile();
}








