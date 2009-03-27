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

  int nSubsystems = 13;

  //initialize reportSummary to 1
  if (reportSummary_) reportSummary_->Fill(1);

  dbe_->setCurrentFolder("L1T/EventInfo/reportSummaryContents");

  
  char histo[100];
  
  for (int n = 0; n < nSubsystems; n++) {    

    
    switch(n){
    case 0 :   sprintf(histo,"L1T_MET");      break;
    case 1 :   sprintf(histo,"L1T_NonIsoEM"); break;
    case 2 :   sprintf(histo,"L1T_IsoEM");    break;
    case 3 :   sprintf(histo,"L1T_TauJets");  break;
    case 4 :   sprintf(histo,"L1T_Jets");     break;
    case 5 :   sprintf(histo,"L1T_Muons");    break;
    case 6 :   sprintf(histo,"L1T_GT");       break;
    case 7 :   sprintf(histo,"L1T_Test1");    break;
    case 8 :   sprintf(histo,"L1T_Test2");    break;
    case 9 :   sprintf(histo,"L1T_Test3");    break;
    case 10 :  sprintf(histo,"L1T_Test4");    break;
    case 11 :  sprintf(histo,"L1T_Test5");    break;
    case 12 :  sprintf(histo,"L1T_Test6");    break;
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


  dbe_->setCurrentFolder("L1T/EventInfo");

  if ( reportSummaryMap_ = dbe_->get("L1T/EventInfo/reportSummaryMap") ) {
  dbe_->removeElement(reportSummaryMap_->getName());
  }

  //reportSummaryMap_ = dbe_->book2D("reportSummaryMap", "reportSummaryMap", 5, 1, 6, 4, 1, 5);
  reportSummaryMap_ = dbe_->book2D("reportSummaryMap", "reportSummaryMap", 1, 1, 2, 8, 1, 9);
  reportSummaryMap_->setAxisTitle("", 1);
  reportSummaryMap_->setAxisTitle("", 2);
  reportSummaryMap_->setBinLabel(1,"MET",2);
  reportSummaryMap_->setBinLabel(2,"NonIsoEM",2);
  reportSummaryMap_->setBinLabel(3,"IsoEM",2);
  reportSummaryMap_->setBinLabel(4,"TauJets",2);
  reportSummaryMap_->setBinLabel(5,"Jets",2);
  reportSummaryMap_->setBinLabel(6,"Muons",2);
  reportSummaryMap_->setBinLabel(7,"GT",2);
  reportSummaryMap_->setBinLabel(8,"Empty",2);
  reportSummaryMap_->setBinLabel(1," ",1);

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


  MonitorElement *GMT_QHist = dbe_->get("L1T/L1TGMT/GMT_etaphi");
  MonitorElement *GCT_IsoEm_QHist = dbe_->get("L1T/L1TGCT/IsoEmRankEtaPhi");
  MonitorElement *GCT_NonIsoEm_QHist = dbe_->get("L1T/L1TGCT/NonIsoEmRankEtaPhi");
  MonitorElement *GCT_AllJets_QHist = dbe_->get("L1T/L1TGCT/AllJetsEtEtaPhi");
  MonitorElement *GCT_TauJets_QHist = dbe_->get("L1T/L1TGCT/TauJetsEtEtaPhi");
  MonitorElement *GT_AlgoBits_QHist = dbe_->get("L1T/L1TGT/algo_bits");
  MonitorElement *GT_TechBits_QHist = dbe_->get("L1T/L1TGT/tt_bits");

  // MET ME


  bool verbose = verbose_;
  int nSubsystems = 13;
  for (int k = 0; k < nSubsystems; k++) {
    summaryContent[k] = 1;
    reportSummaryContent_[k]->Fill(1.);
  }
  summarySum = 0;

  
  int GCT_IsoEm_nXCh = 0,GCT_IsoEm_nYCh=0,GCT_NonIsoEm_nXCh = 0,GCT_NonIsoEm_nYCh=0,GCT_AllJets_nXCh = 0,GCT_AllJets_nYCh=0,GCT_TauJets_nXCh = 0,GCT_TauJets_nYCh=0,GMT_nXCh=0,GMT_nYCh=0;


  if(GCT_IsoEm_QHist){
    GCT_IsoEm_nXCh = GCT_IsoEm_QHist->getNbinsX(); 
    GCT_IsoEm_nYCh = GCT_IsoEm_QHist->getNbinsY();
  }
  if(GCT_NonIsoEm_QHist){
    GCT_NonIsoEm_nXCh = GCT_NonIsoEm_QHist->getNbinsX(); 
    GCT_NonIsoEm_nYCh = GCT_NonIsoEm_QHist->getNbinsY();
  }
  if(GCT_AllJets_QHist){
    GCT_AllJets_nXCh = GCT_AllJets_QHist->getNbinsX(); 
    GCT_AllJets_nYCh = GCT_AllJets_QHist->getNbinsY();
  }
  if(GCT_TauJets_QHist){
    GCT_TauJets_nXCh = GCT_TauJets_QHist->getNbinsX(); 
    GCT_TauJets_nYCh = GCT_TauJets_QHist->getNbinsY();
  }
  if(GMT_QHist){
    GMT_nXCh = GMT_QHist->getNbinsX(); 
    GMT_nYCh = GMT_QHist->getNbinsY();
  }


  int GCT_IsoEm_nCh=0,GCT_NonIsoEm_nCh=0,GCT_AllJets_nCh=0,GCT_TauJets_nCh=0,GMT_nCh=0,GT_AlgoBits_nCh=0,GT_TechBits_nCh=0;
  
  if(GCT_IsoEm_nYCh) 
    GCT_IsoEm_nCh = GCT_IsoEm_nXCh*GCT_IsoEm_nYCh;
  if(GCT_NonIsoEm_nYCh) 
    GCT_NonIsoEm_nCh = GCT_NonIsoEm_nXCh*GCT_NonIsoEm_nYCh;
  if(GCT_AllJets_nYCh) 
    GCT_AllJets_nCh = GCT_AllJets_nXCh*GCT_AllJets_nYCh;
  if(GCT_TauJets_nYCh) 
    GCT_TauJets_nCh = GCT_TauJets_nXCh*GCT_TauJets_nYCh;
  if(GMT_nYCh) 
    GMT_nCh = GMT_nXCh*GMT_nYCh;
  if(GT_AlgoBits_QHist) GT_AlgoBits_nCh = GT_AlgoBits_QHist->getNbinsX(); 
  if(GT_TechBits_QHist) GT_TechBits_nCh = GT_TechBits_QHist->getNbinsX(); 


  GCT_IsoEm_nCh-=252;
  GCT_NonIsoEm_nCh-=252;





  //
  // 00  MET Quality Tests
  //



  //
  // 01  NonIsoEM Quality Tests
  //
  if (GCT_NonIsoEm_QHist){
    const QReport *GCT_NonIsoEm_DeadCh_QReport = GCT_NonIsoEm_QHist->getQReport("DeadChannels_GCT_2D");
    const QReport *GCT_NonIsoEm_HotCh_QReport = GCT_NonIsoEm_QHist->getQReport("HotChannels_GCT_2D");

    int GCT_NonIsoEm_nBadCh = 0;

    if (GCT_NonIsoEm_DeadCh_QReport) {
      int GCT_NonIsoEm_nDeadCh = GCT_NonIsoEm_DeadCh_QReport->getBadChannels().size();
      GCT_NonIsoEm_nDeadCh-=252;     // Remove uninstrumented regions

      if( verbose ) cout << "  GCT_NonIsoEm_nDeadCh: "  << GCT_NonIsoEm_nDeadCh 
			 << ", GCT_NonIsoEm_nCh: " << GCT_NonIsoEm_nCh 
			 << ", GCT_NonIsoEm_DeadCh_efficiency: " << 1 - (float)GCT_NonIsoEm_nDeadCh/(float)GCT_NonIsoEm_nCh << endl;
      //if( verbose ) std::cout << " GCT_NonIsoEm_DeadCh QTResult = " << GCT_NonIsoEm_DeadCh_QReport->getQTresult() << std::endl;

      GCT_NonIsoEm_nBadCh+=GCT_NonIsoEm_nDeadCh;
    } 
//    else std::cout << "      GCT_NonIsoEm_DeadCh_QReport = False  !! " << std::endl;

    if (GCT_NonIsoEm_HotCh_QReport) {
      int GCT_NonIsoEm_nHotCh = GCT_NonIsoEm_HotCh_QReport->getBadChannels().size();
      if( verbose ) cout << "  GCT_NonIsoEm_nHotCh: "  << GCT_NonIsoEm_nHotCh 
			 << ", GCT_NonIsoEm_nCh: " << GCT_NonIsoEm_nCh 
			 << ", GCT_NonIsoEm_HotCh_efficiency: " << 1 - (float)GCT_NonIsoEm_nHotCh/(float)GCT_NonIsoEm_nCh << endl;
      //if( verbose ) std::cout << " GCT_NonIsoEm_HotCh QTResult = " << GCT_NonIsoEm_HotCh_QReport->getQTresult() << std::endl;

      GCT_NonIsoEm_nBadCh+=GCT_NonIsoEm_nHotCh;
    }
//    else std::cout << "      GCT_NonIsoEm_HotCh_QReport = False  !!" << std::endl;

    if( verbose ) std::cout << "    GCT_NonIsoEm total efficiency = " << 1 - (float)GCT_NonIsoEm_nBadCh/(float)GCT_NonIsoEm_nCh << std::endl;

    summaryContent[1] = 1 - (float)GCT_NonIsoEm_nBadCh/(float)GCT_NonIsoEm_nCh;
    reportSummaryContent_[1]->Fill( summaryContent[1] );
  }
//  else std::cout << "      GCT_NonIsoEm_QHist = False  !! " << std::endl;




  //
  // 02  IsoEM Quality Tests
  //
  if (GCT_IsoEm_QHist){
    const QReport *GCT_IsoEm_DeadCh_QReport = GCT_IsoEm_QHist->getQReport("DeadChannels_GCT_2D");
    const QReport *GCT_IsoEm_HotCh_QReport = GCT_IsoEm_QHist->getQReport("HotChannels_GCT_2D");

    int GCT_IsoEm_nBadCh = 0;

    if (GCT_IsoEm_DeadCh_QReport) {
      int GCT_IsoEm_nDeadCh = GCT_IsoEm_DeadCh_QReport->getBadChannels().size();
      GCT_IsoEm_nDeadCh-=252;     // Remove uninstrumented regions

      if( verbose ) cout << "  GCT_IsoEm_nDeadCh: "  << GCT_IsoEm_nDeadCh 
			 << ", GCT_IsoEm_nCh: " << GCT_IsoEm_nCh 
			 << ", GCT_IsoEm_DeadCh_efficiency: " << 1 - (float)GCT_IsoEm_nDeadCh/(float)GCT_IsoEm_nCh << endl;
      //if( verbose ) std::cout << " GCT_IsoEm_DeadCh QTResult = " << GCT_IsoEm_DeadCh_QReport->getQTresult() << std::endl;

      GCT_IsoEm_nBadCh+=GCT_IsoEm_nDeadCh;
    } 
//    else std::cout << "      GCT_IsoEm_DeadCh_QReport = False  !! " << std::endl;

    if (GCT_IsoEm_HotCh_QReport) {
      int GCT_IsoEm_nHotCh = GCT_IsoEm_HotCh_QReport->getBadChannels().size();
      if( verbose ) cout << "  GCT_IsoEm_nHotCh: "  << GCT_IsoEm_nHotCh 
			 << ", GCT_IsoEm_nCh: " << GCT_IsoEm_nCh 
			 << ", GCT_IsoEm_HotCh_efficiency: " << 1 - (float)GCT_IsoEm_nHotCh/(float)GCT_IsoEm_nCh << endl;
      //if( verbose ) std::cout << " GCT_IsoEm_HotCh QTResult = " << GCT_IsoEm_HotCh_QReport->getQTresult() << std::endl;

      GCT_IsoEm_nBadCh+=GCT_IsoEm_nHotCh;
    }
//    else std::cout << "      GCT_IsoEm_HotCh_QReport = False  !!" << std::endl;

    if( verbose ) std::cout << "    GCT_IsoEm total efficiency = " << 1 - (float)GCT_IsoEm_nBadCh/(float)GCT_IsoEm_nCh << std::endl;

    summaryContent[2] = 1 - (float)GCT_IsoEm_nBadCh/(float)GCT_IsoEm_nCh;
    reportSummaryContent_[2]->Fill( summaryContent[2] );
  }
//  else std::cout << "      GCT_IsoEm_QHist = False  !! " << std::endl;




  //
  // 03  TauJets Quality Tests
  //
  if (GCT_TauJets_QHist){
    const QReport *GCT_TauJets_DeadCh_QReport = GCT_TauJets_QHist->getQReport("DeadChannels_GCT_2D");
    const QReport *GCT_TauJets_HotCh_QReport = GCT_TauJets_QHist->getQReport("HotChannels_GCT_2D");

    int GCT_TauJets_nBadCh = 0;

    if (GCT_TauJets_DeadCh_QReport) {
      int GCT_TauJets_nDeadCh = GCT_TauJets_DeadCh_QReport->getBadChannels().size();
      if( verbose ) cout << "  GCT_TauJets_nDeadCh: "  << GCT_TauJets_nDeadCh 
			 << ", GCT_TauJets_nCh: " << GCT_TauJets_nCh 
			 << ", GCT_TauJets_DeadCh_efficiency: " << 1 - (float)GCT_TauJets_nDeadCh/(float)GCT_TauJets_nCh << endl;
      //if( verbose ) std::cout << " GCT_TauJets_DeadCh QTResult = " << GCT_TauJets_DeadCh_QReport->getQTresult() << std::endl;

      GCT_TauJets_nBadCh+=GCT_TauJets_nDeadCh;
    } 
//    else std::cout << "      GCT_TauJets_DeadCh_QReport = False  !! " << std::endl;

    if (GCT_TauJets_HotCh_QReport) {
      int GCT_TauJets_nHotCh = GCT_TauJets_HotCh_QReport->getBadChannels().size();
      if( verbose ) cout << "  GCT_TauJets_nHotCh: "  << GCT_TauJets_nHotCh 
			 << ", GCT_TauJets_nCh: " << GCT_TauJets_nCh 
			 << ", GCT_TauJets_HotCh_efficiency: " << 1 - (float)GCT_TauJets_nHotCh/(float)GCT_TauJets_nCh << endl;
      //if( verbose ) std::cout << " GCT_TauJets_HotCh QTResult = " << GCT_TauJets_HotCh_QReport->getQTresult() << std::endl;

      GCT_TauJets_nBadCh+=GCT_TauJets_nHotCh;
    }
//    else std::cout << "      GCT_TauJets_HotCh_QReport = False  !!" << std::endl;

    if( verbose ) std::cout << "    GCT_TauJets total efficiency = " << 1 - (float)GCT_TauJets_nBadCh/(float)GCT_TauJets_nCh << std::endl;

    summaryContent[3] = 1 - (float)GCT_TauJets_nBadCh/(float)GCT_TauJets_nCh;
    reportSummaryContent_[3]->Fill( summaryContent[3] );
  }
//  else std::cout << "      GCT_TauJets_QHist = False  !! " << std::endl;




  //
  // 04  Jets Quality Tests
  //
  if (GCT_AllJets_QHist){
    const QReport *GCT_AllJets_DeadCh_QReport = GCT_AllJets_QHist->getQReport("DeadChannels_GCT_2D");
    const QReport *GCT_AllJets_HotCh_QReport = GCT_AllJets_QHist->getQReport("HotChannels_GCT_2D");

    int GCT_AllJets_nBadCh = 0;

    if (GCT_AllJets_DeadCh_QReport) {
      int GCT_AllJets_nDeadCh = GCT_AllJets_DeadCh_QReport->getBadChannels().size();
      if( verbose ) cout << "  GCT_AllJets_nDeadCh: "  << GCT_AllJets_nDeadCh 
			 << ", GCT_AllJets_nCh: " << GCT_AllJets_nCh 
			 << ", GCT_AllJets_DeadCh_efficiency: " << 1 - (float)GCT_AllJets_nDeadCh/(float)GCT_AllJets_nCh << endl;
      //if( verbose ) std::cout << " GCT_AllJets_DeadCh QTResult = " << GCT_AllJets_DeadCh_QReport->getQTresult() << std::endl;

      GCT_AllJets_nBadCh+=GCT_AllJets_nDeadCh;
    } 
//    else std::cout << "      GCT_AllJets_DeadCh_QReport = False  !! " << std::endl;

    if (GCT_AllJets_HotCh_QReport) {
      int GCT_AllJets_nHotCh = GCT_AllJets_HotCh_QReport->getBadChannels().size();
      if( verbose ) cout << "  GCT_AllJets_nHotCh: "  << GCT_AllJets_nHotCh 
			 << ", GCT_AllJets_nCh: " << GCT_AllJets_nCh 
			 << ", GCT_AllJets_HotCh_efficiency: " << 1 - (float)GCT_AllJets_nHotCh/(float)GCT_AllJets_nCh << endl;
      //if( verbose ) std::cout << " GCT_AllJets_HotCh QTResult = " << GCT_AllJets_HotCh_QReport->getQTresult() << std::endl;

      GCT_AllJets_nBadCh+=GCT_AllJets_nHotCh;
    }
//    else std::cout << "      GCT_AllJets_HotCh_QReport = False  !!" << std::endl;

    if( verbose ) std::cout << "    GCT_AllJets total efficiency = " << 1 - (float)GCT_AllJets_nBadCh/(float)GCT_AllJets_nCh << std::endl;

    summaryContent[4] = 1 - (float)GCT_AllJets_nBadCh/(float)GCT_AllJets_nCh;
    reportSummaryContent_[4]->Fill( summaryContent[4] );
  }
//  else std::cout << "      GCT_AllJets_QHist = False  !! " << std::endl;




  //
  // 05  Muon Quality Tests
  //
  if (GMT_QHist){
    const QReport *GMT_DeadCh_QReport = GMT_QHist->getQReport("DeadChannels_GMT_2D");
    const QReport *GMT_HotCh_QReport  = GMT_QHist->getQReport("HotChannels_GMT_2D");

    int GMT_nBadCh = 0;

    if (GMT_DeadCh_QReport) {
      int GMT_nDeadCh = GMT_DeadCh_QReport->getBadChannels().size();
      if( verbose ) cout << "  GMT_nDeadCh: "  << GMT_nDeadCh 
			 << ", GMT_nCh: " << GMT_nCh 
			 << ", GMT_DeadCh_efficiency: " << 1 - (float)GMT_nDeadCh/(float)GMT_nCh << endl;
      //if( verbose ) std::cout << " GMT_DeadCh QTResult = " << GMT_DeadCh_QReport->getQTresult() << std::endl;

      GMT_nBadCh+=GMT_nDeadCh;
    } 
//    else std::cout << "      GMT_DeadCh_QReport = False  !! " << std::endl;

    if (GMT_HotCh_QReport) {
      int GMT_nHotCh = GMT_HotCh_QReport->getBadChannels().size();
      if( verbose ) cout << "  GMT_nHotCh: "  << GMT_nHotCh 
			 << ", GMT_nCh: " << GMT_nCh 
			 << ", GMT_HotCh_efficiency: " << 1 - (float)GMT_nHotCh/(float)GMT_nCh << endl;
      //if( verbose ) std::cout << " GMT_HotCh QTResult = " << GMT_HotCh_QReport->getQTresult() << std::endl;

      GMT_nBadCh+=GMT_nHotCh;
    }
//    else std::cout << "      GMT_HotCh_QReport = False  !!" << std::endl;

    if( verbose ) std::cout << "    GMT total efficiency = " << 1 - (float)GMT_nBadCh/(float)GMT_nCh << std::endl;

    summaryContent[5] = 1 - (float)GMT_nBadCh/(float)GMT_nCh;
    reportSummaryContent_[5]->Fill( summaryContent[5] );
  }
//  else std::cout << "      GMT_QHist = False  !! " << std::endl;




  //
  // 06  GT Quality Tests
  //
  int GT_AlgoBits_nBadCh = -1;
  int GT_TechBits_nBadCh = -1;
  if (GT_AlgoBits_QHist){
    const QReport *GT_AlgoBits_QReport = GT_AlgoBits_QHist->getQReport("CompareHist_GT");

    double gt_algobits_prob = -1;
    if (GT_AlgoBits_QReport) {
      GT_AlgoBits_nBadCh = GT_AlgoBits_QReport->getBadChannels().size();
      if( verbose ) cout << "  GT_AlgoBits_nBadCh: "  << GT_AlgoBits_nBadCh
			 << ", GT_AlgoBits_nCh: " << GT_AlgoBits_nCh 
			 << ", GT_AlgoBits_efficiency: " << 1 - (float)GT_AlgoBits_nBadCh/(float)GT_AlgoBits_nCh << endl;

      //if( verbose ) std::cout << " GT_AlgoBits QTResult = " << GT_AlgoBits_QReport->getQTresult() << std::endl;
      //gt_algobits_prob = GT_AlgoBits_QReport->getQTresult();
    } 
//    else std::cout << "      GT_AlgoBits_QReport = False  !! " << std::endl;

  }
//  else std::cout << "      GT_AlgoBits_QHist = False  !! " << std::endl;

  if (GT_TechBits_QHist){
    const QReport *GT_TechBits_QReport = GT_TechBits_QHist->getQReport("CompareHist_GT");

    double gt_techbits_prob = -1;
    if (GT_TechBits_QReport) {
      GT_TechBits_nBadCh = GT_TechBits_QReport->getBadChannels().size();
      if( verbose ) cout << "  GT_TechBits_nBadCh: "  << GT_TechBits_nBadCh
			 << ", GT_TechBits_nCh: " << GT_TechBits_nCh 
			 << ", GT_TechBits_efficiency: " << 1 - (float)GT_TechBits_nBadCh/(float)GT_TechBits_nCh << endl;

      //if( verbose ) std::cout << " GT_TechBits QTResult = " << GT_TechBits_QReport->getQTresult() << std::endl;
      //gt_techbits_prob = GT_TechBits_QReport->getQTresult();
    } 
//    else std::cout << "      GT_TechBits_QReport = False  !! " << std::endl;
  }
//  else std::cout << "      GT_TechBits_QHist = False  !! " << std::endl;

  if( GT_AlgoBits_nBadCh!=-1 && GT_AlgoBits_nBadCh!=-1 ){
    if( verbose ) 
      std::cout << "    GT total efficiency = " << 1-(float)(GT_AlgoBits_nBadCh+GT_AlgoBits_nBadCh)/(float)(GT_AlgoBits_nCh+GT_AlgoBits_nCh) << std::endl;

    summaryContent[6] = 1-(float)(GT_AlgoBits_nBadCh+GT_TechBits_nBadCh)/(float)(GT_AlgoBits_nCh+GT_TechBits_nCh);
    reportSummaryContent_[6]->Fill( summaryContent[6] );
  }
//  else std::cout << "      QT Results for the GT not found  !! " << std::endl;






  for (int m = 0; m < nSubsystems; m++) {    
    summarySum += summaryContent[m];
  }
  
  reportSummary = summarySum / nSubsystems;
  if (reportSummary_) reportSummary_->Fill(reportSummary);
  

   //8x1 summary map
  reportSummaryMap_->setBinContent(1,1,summaryContent[0]);//MET
  reportSummaryMap_->setBinContent(1,2,summaryContent[1]);//NonIsoEM
  reportSummaryMap_->setBinContent(1,3,summaryContent[2]);//IsoEM
  reportSummaryMap_->setBinContent(1,4,summaryContent[3]);//TauJets
  reportSummaryMap_->setBinContent(1,5,summaryContent[4]);//Jets
  reportSummaryMap_->setBinContent(1,6,summaryContent[5]);//Muons
  reportSummaryMap_->setBinContent(1,7,summaryContent[6]);//GT
  reportSummaryMap_->setBinContent(1,8,summaryContent[7]);//Empty


  if( verbose ){
    std::cout << "     summary content[0] = MET      = " << summaryContent[0] << std::endl;
    std::cout << "     summary content[1] = NonIsoEM = " << summaryContent[1] << std::endl;
    std::cout << "     summary content[2] = IsoEM    = " << summaryContent[2] << std::endl;
    std::cout << "     summary content[3] = TauJets  = " << summaryContent[3] << std::endl;
    std::cout << "     summary content[4] = Jets     = " << summaryContent[4] << std::endl;
    std::cout << "     summary content[5] = Muons    = " << summaryContent[5] << std::endl;
    std::cout << "     summary content[6] = GT       = " << summaryContent[6] << std::endl;
    std::cout << "     summary content[7] = Empty    = " << summaryContent[7] << std::endl;
  }

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








