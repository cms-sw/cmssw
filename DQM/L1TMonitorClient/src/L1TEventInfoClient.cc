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
    
  thresholdLS_ = parameters_.getUntrackedParameter<int>("thresholdLS", 1);
  if(verbose_) cout << "Minimum LS required to perform QTests = " << thresholdLS_ << " lumi section(s)"<< endl;

  GCT_NonIsoEm_threshold_ = parameters_.getUntrackedParameter<double>("GCT_NonIsoEm_threshold",100000);
  GCT_IsoEm_threshold_ = parameters_.getUntrackedParameter<double>("GCT_IsoEm_threshold",1000000);
  GCT_TauJets_threshold_ = parameters_.getUntrackedParameter<double>("GCT_TauJets_threshold",100000);
  GCT_AllJets_threshold_ = parameters_.getUntrackedParameter<double>("GCT_AllJets_threshold",100000);
  GMT_Muons_threshold_ = parameters_.getUntrackedParameter<double>("GMT_Muons_threshold",100000);

  if(verbose_){
    cout << " Thresholds are as follows:" << endl;
    cout << " \t GCT_NonIsoEm_threshold_: " << GCT_NonIsoEm_threshold_ << endl;
    cout << " \t GCT_IsoEm_threshold_:    " << GCT_IsoEm_threshold_ << endl;
    cout << " \t GCT_TauJets_threshold_:  " << GCT_TauJets_threshold_ << endl;
    cout << " \t GCT_AllJets_threshold_:  " << GCT_AllJets_threshold_ << endl;
    cout << " \t GMT_Muons_threshold_:    " << GMT_Muons_threshold_ << endl;
  }

  std::vector<string> emptyMask;

  dataMask = parameters_.getUntrackedParameter<std::vector<string> >("dataMaskedSystems", emptyMask);
  emulMask = parameters_.getUntrackedParameter<std::vector<string> >("emulatorMaskedSystems", emptyMask);

  s_mapDataValues["EMPTY"]    = data_empty;
  s_mapDataValues["ALL"]      = data_all;
  s_mapDataValues["GT"]       = data_gt;
  s_mapDataValues["MUONS"]    = data_muons;
  s_mapDataValues["JETS"]     = data_jets;
  s_mapDataValues["TAUJETS"]  = data_taujets;
  s_mapDataValues["ISOEM"]    = data_isoem;
  s_mapDataValues["NONISOEM"] = data_nonisoem;
  s_mapDataValues["MET"]      = data_met;

  s_mapEmulValues["EMPTY"]  = emul_empty;
  s_mapEmulValues["ALL"]    = emul_all;
  s_mapEmulValues["DTTF"]   = emul_dtf;
  s_mapEmulValues["DTTPG"]  = emul_dtp;
  s_mapEmulValues["CSCTF"]  = emul_ctf;
  s_mapEmulValues["CSCTPG"] = emul_ctp;
  s_mapEmulValues["RPC"]    = emul_rpc;
  s_mapEmulValues["GMT"]    = emul_gmt;
  s_mapEmulValues["ECAL"]   = emul_etp;
  s_mapEmulValues["HCAL"]   = emul_htp;
  s_mapEmulValues["RCT"]    = emul_rct;
  s_mapEmulValues["GCT"]    = emul_gct;
  s_mapEmulValues["GLT"]    = emul_glt;

}

//--------------------------------------------------------
void L1TEventInfoClient::beginJob(void){

  if(verbose_) cout <<"[TriggerDQM]: Begin Job" << endl;
  // get backendinterface  
  dbe_ = Service<DQMStore>().operator->();

  dbe_->setCurrentFolder("L1T/EventInfo");

  if( (reportSummary_ = dbe_->get("L1T/EventInfo/reportSumamry")) ) {
      dbe_->removeElement(reportSummary_->getName()); 
   }
  
  reportSummary_ = dbe_->bookFloat("reportSummary");

  //initialize reportSummary to 1
  if (reportSummary_) reportSummary_->Fill(1);

  dbe_->setCurrentFolder("L1T/EventInfo/reportSummaryContents");

  
  char histo[100];
  
  for (int n = 0; n < nsys_; n++) {    

    switch(n){
    case 0 :   sprintf(histo,"L1T_MET");      break;
    case 1 :   sprintf(histo,"L1T_NonIsoEM"); break;
    case 2 :   sprintf(histo,"L1T_IsoEM");    break;
    case 3 :   sprintf(histo,"L1T_TauJets");  break;
    case 4 :   sprintf(histo,"L1T_Jets");     break;
    case 5 :   sprintf(histo,"L1T_Muons");    break;
    case 6 :   sprintf(histo,"L1T_GT");       break;
    case 7 :   sprintf(histo,"L1TEMU_GLT");   break;
    case 8 :   sprintf(histo,"L1TEMU_GMT");   break;
    case 9 :   sprintf(histo,"L1TEMU_RPC");   break;
    case 10:   sprintf(histo,"L1TEMU_CTP");   break;
    case 11:   sprintf(histo,"L1TEMU_CTF");   break;
    case 12:   sprintf(histo,"L1TEMU_DTP");   break;
    case 13:   sprintf(histo,"L1TEMU_DTF");   break;
    case 14:   sprintf(histo,"L1TEMU_HTP");   break;
    case 15:   sprintf(histo,"L1TEMU_ETP");   break;
    case 16:   sprintf(histo,"L1TEMU_GCT");   break;
    case 17:   sprintf(histo,"L1TEMU_RCT");   break;
    }  
    
    reportSummaryContent_[n] = dbe_->bookFloat(histo);
  }

  //initialize reportSummaryContents to 0
  for (int k = 0; k < nsys_; k++) {
    summaryContent[k] = 0;
    reportSummaryContent_[k]->Fill(0.);
  }  


  dbe_->setCurrentFolder("L1T/EventInfo");

  if( (reportSummaryMap_ = dbe_->get("L1T/EventInfo/reportSummaryMap")) ){
    dbe_->removeElement(reportSummaryMap_->getName());
  }

  reportSummaryMap_ = dbe_->book2D("reportSummaryMap", "reportSummaryMap", 2, 1, 3, 11, 1, 12);
  reportSummaryMap_->setAxisTitle("", 1);
  reportSummaryMap_->setAxisTitle("", 2);

  reportSummaryMap_->setBinLabel(1," ",1);
  reportSummaryMap_->setBinLabel(2," ",1);

  reportSummaryMap_->setBinLabel(1," ",2);
  reportSummaryMap_->setBinLabel(2," ",2);
  reportSummaryMap_->setBinLabel(3," ",2);
  reportSummaryMap_->setBinLabel(4," ",2);
  reportSummaryMap_->setBinLabel(5," ",2);
  reportSummaryMap_->setBinLabel(6," ",2);
  reportSummaryMap_->setBinLabel(7," ",2);
  reportSummaryMap_->setBinLabel(8," ",2);
  reportSummaryMap_->setBinLabel(9," ",2);
  reportSummaryMap_->setBinLabel(10," ",2);
  reportSummaryMap_->setBinLabel(11," ",2);

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


  counterLS_++;

  MonitorElement *GMT_QHist = dbe_->get("L1T/L1TGMT/GMT_etaphi");
  MonitorElement *GCT_IsoEm_QHist = dbe_->get("L1T/L1TGCT/IsoEmRankEtaPhi");
  MonitorElement *GCT_NonIsoEm_QHist = dbe_->get("L1T/L1TGCT/NonIsoEmRankEtaPhi");
  MonitorElement *GCT_AllJets_QHist = dbe_->get("L1T/L1TGCT/AllJetsEtEtaPhi");
  MonitorElement *GCT_TauJets_QHist = dbe_->get("L1T/L1TGCT/TauJetsEtEtaPhi");
  MonitorElement *GT_AlgoBits_QHist = dbe_->get("L1T/L1TGT/algo_bits");
  MonitorElement *GT_TechBits_QHist = dbe_->get("L1T/L1TGT/tt_bits");



  for (int k = 0; k < nsys_; k++) {
    summaryContent[k] = 0;
    reportSummaryContent_[k]->Fill(0.);
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


  int GCT_IsoEm_nCh=0,GCT_NonIsoEm_nCh=0,GCT_AllJets_nCh=0,GCT_TauJets_nCh=0,GMT_nCh=0;
  
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





  //
  // 00  MET Quality Tests
  //



  // GCT uninstrumented regions for IsoEm, NonIsoEm, and TauJets
  int nCh_no_inst = 144;

  //
  // 01  NonIsoEM Quality Tests
  //
  if (GCT_NonIsoEm_QHist){
    const QReport *GCT_NonIsoEm_DeadCh_QReport = GCT_NonIsoEm_QHist->getQReport("DeadChannels_GCT_2D_loose");
    const QReport *GCT_NonIsoEm_HotCh_QReport  = GCT_NonIsoEm_QHist->getQReport("HotChannels_GCT_2D");

    int GCT_NonIsoEm_nBadCh = 0;

    if (GCT_NonIsoEm_DeadCh_QReport) {
      int GCT_NonIsoEm_nDeadCh = GCT_NonIsoEm_DeadCh_QReport->getBadChannels().size();
      if( verbose_ ) cout << "  GCT_NonIsoEm_nDeadCh: "  << GCT_NonIsoEm_nDeadCh 
			  << ", GCT_NonIsoEm_nCh: " << GCT_NonIsoEm_nCh 
			  << ", GCT_NonIsoEm_DeadCh_efficiency: " << 1 - (float)GCT_NonIsoEm_nDeadCh/(float)GCT_NonIsoEm_nCh 
			  << " GCT_NonIsoEm_DeadCh QTResult = " << GCT_NonIsoEm_DeadCh_QReport->getQTresult() << std::endl;

      GCT_NonIsoEm_nBadCh+=GCT_NonIsoEm_nDeadCh;
    } 

    if (GCT_NonIsoEm_HotCh_QReport) {
      int GCT_NonIsoEm_nHotCh = GCT_NonIsoEm_HotCh_QReport->getBadChannels().size();
      if( verbose_ ) cout << "  GCT_NonIsoEm_nHotCh: "  << GCT_NonIsoEm_nHotCh 
			  << ", GCT_NonIsoEm_nCh: " << GCT_NonIsoEm_nCh 
			  << ", GCT_NonIsoEm_HotCh_efficiency: " << 1 - (float)GCT_NonIsoEm_nHotCh/(float)GCT_NonIsoEm_nCh 
			  << " GCT_NonIsoEm_HotCh QTResult = " << GCT_NonIsoEm_HotCh_QReport->getQTresult() << std::endl;

      GCT_NonIsoEm_nBadCh+=GCT_NonIsoEm_nHotCh;
    }

    if( verbose_ ) std::cout << "    GCT_NonIsoEm total efficiency = " << 1 - (float)GCT_NonIsoEm_nBadCh/(float)GCT_NonIsoEm_nCh << std::endl;

    double GCT_NonIsoEm_nentries = GCT_NonIsoEm_QHist->getEntries();
    float nonisoResult = -1;
    if( (counterLS_>=1000 && GCT_NonIsoEm_nentries==0) ) nonisoResult = 0;
    if( (GCT_NonIsoEm_nentries>GCT_NonIsoEm_threshold_) ) nonisoResult = 1 - (float)(GCT_NonIsoEm_nBadCh-nCh_no_inst)/(float)(GCT_NonIsoEm_nCh-nCh_no_inst);
    summaryContent[1] = ( nonisoResult < (1.0+1e-10) ) ? nonisoResult : 1.0;
  }




  //
  // 02  IsoEM Quality Tests
  //
  if (GCT_IsoEm_QHist){
    const QReport *GCT_IsoEm_DeadCh_QReport = GCT_IsoEm_QHist->getQReport("DeadChannels_GCT_2D_loose");
    const QReport *GCT_IsoEm_HotCh_QReport = GCT_IsoEm_QHist->getQReport("HotChannels_GCT_2D");

    int GCT_IsoEm_nBadCh = 0;

    if (GCT_IsoEm_DeadCh_QReport) {
      int GCT_IsoEm_nDeadCh = GCT_IsoEm_DeadCh_QReport->getBadChannels().size();
      if( verbose_ ) cout << "  GCT_IsoEm_nDeadCh: "  << GCT_IsoEm_nDeadCh 
			  << ", GCT_IsoEm_nCh: " << GCT_IsoEm_nCh 
			  << ", GCT_IsoEm_DeadCh_efficiency: " << 1 - (float)GCT_IsoEm_nDeadCh/(float)GCT_IsoEm_nCh 
			  << " GCT_IsoEm_DeadCh QTResult = " << GCT_IsoEm_DeadCh_QReport->getQTresult() << std::endl;

      GCT_IsoEm_nBadCh+=GCT_IsoEm_nDeadCh;
    } 

    if (GCT_IsoEm_HotCh_QReport) {
      int GCT_IsoEm_nHotCh = GCT_IsoEm_HotCh_QReport->getBadChannels().size();
      if( verbose_ ) cout << "  GCT_IsoEm_nHotCh: "  << GCT_IsoEm_nHotCh 
			  << ", GCT_IsoEm_nCh: " << GCT_IsoEm_nCh 
			  << ", GCT_IsoEm_HotCh_efficiency: " << 1 - (float)GCT_IsoEm_nHotCh/(float)GCT_IsoEm_nCh 
			  << " GCT_IsoEm_HotCh QTResult = " << GCT_IsoEm_HotCh_QReport->getQTresult() << std::endl;

      GCT_IsoEm_nBadCh+=GCT_IsoEm_nHotCh;
    }

    if( verbose_ ) std::cout << "    GCT_IsoEm total efficiency = " << 1 - (float)GCT_IsoEm_nBadCh/(float)GCT_IsoEm_nCh << std::endl;

    double GCT_IsoEm_nentries = GCT_IsoEm_QHist->getEntries();
    float isoResult = -1;
    if( (counterLS_>=thresholdLS_ && GCT_IsoEm_nentries==0) ) isoResult = 0;
    if( (GCT_IsoEm_nentries>GCT_IsoEm_threshold_) ) isoResult = 1 - (float)(GCT_IsoEm_nBadCh-nCh_no_inst)/(float)(GCT_IsoEm_nCh-nCh_no_inst);
    summaryContent[2] = ( isoResult < (1.0+1e-10) ) ? isoResult : 1.0;
  }




  //
  // 03  TauJets Quality Tests
  //
  if (GCT_TauJets_QHist){
    const QReport *GCT_TauJets_DeadCh_QReport = GCT_TauJets_QHist->getQReport("DeadChannels_GCT_2D_loose");
    const QReport *GCT_TauJets_HotCh_QReport = GCT_TauJets_QHist->getQReport("HotChannels_GCT_2D");

    int GCT_TauJets_nBadCh = 0;

    if (GCT_TauJets_DeadCh_QReport) {
      int GCT_TauJets_nDeadCh = GCT_TauJets_DeadCh_QReport->getBadChannels().size();
      if( verbose_ ) cout << "  GCT_TauJets_nDeadCh: "  << GCT_TauJets_nDeadCh 
			  << ", GCT_TauJets_nCh: " << GCT_TauJets_nCh 
			  << ", GCT_TauJets_DeadCh_efficiency: " << 1 - (float)GCT_TauJets_nDeadCh/(float)GCT_TauJets_nCh 
			  << " GCT_TauJets_DeadCh QTResult = " << GCT_TauJets_DeadCh_QReport->getQTresult() << std::endl;

      GCT_TauJets_nBadCh+=GCT_TauJets_nDeadCh;
    } 

    if (GCT_TauJets_HotCh_QReport) {
      int GCT_TauJets_nHotCh = GCT_TauJets_HotCh_QReport->getBadChannels().size();
      if( verbose_ ) cout << "  GCT_TauJets_nHotCh: "  << GCT_TauJets_nHotCh 
			  << ", GCT_TauJets_nCh: " << GCT_TauJets_nCh 
			  << ", GCT_TauJets_HotCh_efficiency: " << 1 - (float)GCT_TauJets_nHotCh/(float)GCT_TauJets_nCh 
			  << " GCT_TauJets_HotCh QTResult = " << GCT_TauJets_HotCh_QReport->getQTresult() << std::endl;

      GCT_TauJets_nBadCh+=GCT_TauJets_nHotCh;
    }

    if( verbose_ ) std::cout << "    GCT_TauJets total efficiency = " << 1 - (float)GCT_TauJets_nBadCh/(float)GCT_TauJets_nCh << std::endl;

    double GCT_TauJets_nentries = GCT_TauJets_QHist->getEntries();
    float taujetsResult = -1;
    if( (counterLS_>=thresholdLS_ && GCT_TauJets_nentries==0) ) taujetsResult = 0;
    if( (GCT_TauJets_nentries>GCT_TauJets_threshold_) ) taujetsResult = 1 - (float)(GCT_TauJets_nBadCh-nCh_no_inst)/(float)(GCT_TauJets_nCh-nCh_no_inst);
    summaryContent[3] = ( taujetsResult < (1.0+1e-10) ) ? taujetsResult : 1.0;
  }




  //
  // 04  Jets Quality Tests
  //
  if (GCT_AllJets_QHist){
    const QReport *GCT_AllJets_DeadCh_QReport = GCT_AllJets_QHist->getQReport("DeadChannels_GCT_2D_tight");
    const QReport *GCT_AllJets_HotCh_QReport = GCT_AllJets_QHist->getQReport("HotChannels_GCT_2D");

    int GCT_AllJets_nBadCh = 0;

    if (GCT_AllJets_DeadCh_QReport) {
      int GCT_AllJets_nDeadCh = GCT_AllJets_DeadCh_QReport->getBadChannels().size();
      if( verbose_ ) cout << "  GCT_AllJets_nDeadCh: "  << GCT_AllJets_nDeadCh 
			  << ", GCT_AllJets_nCh: " << GCT_AllJets_nCh 
			  << ", GCT_AllJets_DeadCh_efficiency: " << 1 - (float)GCT_AllJets_nDeadCh/(float)GCT_AllJets_nCh 
			  << " GCT_AllJets_DeadCh QTResult = " << GCT_AllJets_DeadCh_QReport->getQTresult() << std::endl;

      GCT_AllJets_nBadCh+=GCT_AllJets_nDeadCh;
    } 

    if (GCT_AllJets_HotCh_QReport) {
      int GCT_AllJets_nHotCh = GCT_AllJets_HotCh_QReport->getBadChannels().size();
      if( verbose_ ) cout << "  GCT_AllJets_nHotCh: "  << GCT_AllJets_nHotCh 
			  << ", GCT_AllJets_nCh: " << GCT_AllJets_nCh 
			  << ", GCT_AllJets_HotCh_efficiency: " << 1 - (float)GCT_AllJets_nHotCh/(float)GCT_AllJets_nCh 
			  << " GCT_AllJets_HotCh QTResult = " << GCT_AllJets_HotCh_QReport->getQTresult() << std::endl;

      GCT_AllJets_nBadCh+=GCT_AllJets_nHotCh;
    }

    if( verbose_ ) std::cout << "    GCT_AllJets total efficiency = " << 1 - (float)GCT_AllJets_nBadCh/(float)GCT_AllJets_nCh << std::endl;

    double GCT_AllJets_nentries = GCT_AllJets_QHist->getEntries();
    float jetsResult = -1;
    if( (counterLS_>=thresholdLS_ && GCT_AllJets_nentries==0) ) jetsResult = 0;
    if( (GCT_AllJets_nentries>GCT_AllJets_threshold_) ) jetsResult = 1 - (float)GCT_AllJets_nBadCh/(float)GCT_AllJets_nCh;
    summaryContent[4] = ( jetsResult < (1.0+1e-10) ) ? jetsResult : 1.0;
  }




  //
  // 05  Muon Quality Tests
  //

  if (GMT_QHist){
    const QReport *GMT_DeadCh_QReport = GMT_QHist->getQReport("DeadChannels_GMT_2D");
    const QReport *GMT_HotCh_QReport  = GMT_QHist->getQReport("HotChannels_GMT_2D");

    int GMT_nBadCh = 0;

    if (GMT_DeadCh_QReport) {
      int GMT_nDeadCh = GMT_DeadCh_QReport->getBadChannels().size();
      if( verbose_ ) cout << "  GMT_nDeadCh: "  << GMT_nDeadCh 
			  << ", GMT_nCh: " << GMT_nCh 
			  << ", GMT_DeadCh_efficiency: " << 1 - (float)GMT_nDeadCh/(float)GMT_nCh 
			  << " GMT_DeadCh QTResult = " << GMT_DeadCh_QReport->getQTresult() << std::endl;

      GMT_nBadCh+=GMT_nDeadCh;
    } 

    if (GMT_HotCh_QReport) {
      int GMT_nHotCh = GMT_HotCh_QReport->getBadChannels().size();
      if( verbose_ ) cout << "  GMT_nHotCh: "  << GMT_nHotCh 
			  << ", GMT_nCh: " << GMT_nCh 
			  << ", GMT_HotCh_efficiency: " << 1 - (float)GMT_nHotCh/(float)GMT_nCh 
			  << " GMT_HotCh QTResult = " << GMT_HotCh_QReport->getQTresult() << std::endl;

      GMT_nBadCh+=GMT_nHotCh;
    }

    if( verbose_ ) std::cout << "    GMT total efficiency = " << 1 - (float)GMT_nBadCh/(float)GMT_nCh << std::endl;

    double GMT_nentries = GMT_QHist->getEntries();
    float muonResult = -1;
    if( (counterLS_>=thresholdLS_ && GMT_nentries==0) ) muonResult = 0;
    if( (GMT_nentries>GMT_Muons_threshold_) ) muonResult = 1.5*(1 - (float)GMT_nBadCh/(float)GMT_nCh);
    summaryContent[5] = ( muonResult < (1.0+1e-10) ) ? muonResult : 1.0;
  }



  //
  // 06  GT Quality Tests
  //
  double gt_algobits_prob = 0;
  double gt_techbits_prob = 0;

  if (GT_AlgoBits_QHist){
    gt_algobits_prob = 1;
    const QReport *GT_AlgoBits_QReport = GT_AlgoBits_QHist->getQReport("CompareHist_GT");
    if (GT_AlgoBits_QReport) gt_algobits_prob = GT_AlgoBits_QReport->getQTresult();
  }
  if (GT_TechBits_QHist){
    gt_techbits_prob = 1;
    const QReport *GT_TechBits_QReport = GT_TechBits_QHist->getQReport("CompareHist_GT");
    if (GT_TechBits_QReport) gt_techbits_prob = GT_TechBits_QReport->getQTresult();
  }

  if( gt_algobits_prob!=-1 && gt_techbits_prob!=-1 ) summaryContent[6] = 0.5*( gt_algobits_prob + gt_techbits_prob );
  else if( GT_AlgoBits_QHist && GT_TechBits_QHist  ) summaryContent[6] = 1;
  else summaryContent[6] = 0;




  //
  // 07 - 17  L1T EMU Quality Tests
  //



  //
  // Apply masks for data and emulator
  //

  //  Data Mask
  unsigned int NumDataMask = dataMask.size();
  std::vector<string> maskedData;
  for( unsigned int i=0; i<NumDataMask; i++ ){
    std::string mask_sys_tmp  = dataMask[i];
    std::string mask_sys = StringToUpper(mask_sys_tmp);
    switch(s_mapDataValues[mask_sys])
      {
      case data_empty:
	break;
      case data_all:
	for( int m=0; m<7; m++ ) summaryContent[m] = -2;
	maskedData.push_back(mask_sys_tmp);
	break;
      case data_gt:
	summaryContent[6]=-2;
	maskedData.push_back(mask_sys_tmp);
	break;
      case data_muons:
	summaryContent[5]=-2;
	maskedData.push_back(mask_sys_tmp);
	break;
      case data_jets:
	summaryContent[4]=-2;
	maskedData.push_back(mask_sys_tmp);
	break;
      case data_taujets:
	summaryContent[3]=-2;
	maskedData.push_back(mask_sys_tmp);
	break;
      case data_isoem:
	summaryContent[2]=-2;
	maskedData.push_back(mask_sys_tmp);
	break;
      case data_nonisoem:
	summaryContent[1]=-2;
	maskedData.push_back(mask_sys_tmp);
	break;
      case data_met:
	summaryContent[0]=-2;
	maskedData.push_back(mask_sys_tmp);
	break;
      default:
	if( verbose_ ) cout << "   User input mask '" << mask_sys_tmp << "' is not recognized." << endl;
	break;
      }
  }

  //  Emulator Mask
  unsigned int NumEmulMask = emulMask.size();
  std::vector<string> maskedEmul;
  for( unsigned int i=0; i<NumEmulMask; i++ ){
    std::string mask_sys_tmp  = emulMask[i];
    std::string mask_sys = StringToUpper(mask_sys_tmp);
    switch(s_mapEmulValues[mask_sys])
      {
      case emul_empty:
	break;
      case emul_all:
	for( int m=7; m<18; m++ ) summaryContent[m] = -2;
	maskedEmul.push_back(mask_sys_tmp);
	break;
      case emul_glt:
	summaryContent[7]=-2;
	maskedEmul.push_back(mask_sys_tmp);
	break;
      case emul_gmt:
	summaryContent[8]=-2;
	maskedEmul.push_back(mask_sys_tmp);
	break;
      case emul_rpc:
	summaryContent[9]=-2;
	maskedEmul.push_back(mask_sys_tmp);
	break;
      case emul_ctp:
	summaryContent[10]=-2;
	maskedEmul.push_back(mask_sys_tmp);
	break;
      case emul_ctf:
	summaryContent[11]=-2;
	maskedEmul.push_back(mask_sys_tmp);
	break;
      case emul_dtp:
	summaryContent[12]=-2;
	maskedEmul.push_back(mask_sys_tmp);
	break;
      case emul_dtf:
	summaryContent[13]=-2;
	maskedEmul.push_back(mask_sys_tmp);
	break;
      case emul_htp:
	summaryContent[14]=-2;
	maskedEmul.push_back(mask_sys_tmp);
	break;
      case emul_etp:
	summaryContent[15]=-2;
	maskedEmul.push_back(mask_sys_tmp);
	break;
      case emul_gct:
	summaryContent[16]=-2;
	maskedEmul.push_back(mask_sys_tmp);
	break;
      case emul_rct:
	summaryContent[17]=-2;
	maskedEmul.push_back(mask_sys_tmp);
	break;
      default:
	if( verbose_ ) cout << "   User input mask '" << mask_sys_tmp << "' is not recognized." << endl;
	break;
      }
  }


  int numUnMaskedSystems = 0;
  for( int m=0; m<nsys_; m++ ){
    if( summaryContent[m]>-1e-5){
      if( m<7 ){
	summarySum += summaryContent[m];
	numUnMaskedSystems++;
      }

      reportSummaryContent_[m]->Fill( summaryContent[m] );
    }
  }



  // For now, only use L1T for reportSummary value
  reportSummary = summarySum/float(numUnMaskedSystems);
  if (reportSummary_) reportSummary_->Fill(reportSummary);
  

  //L1T summary map
  reportSummaryMap_->setBinContent(1,11,summaryContent[6]); // GT
  reportSummaryMap_->setBinContent(1,10,summaryContent[5]); // Muons
  reportSummaryMap_->setBinContent(1,9, summaryContent[4]); // Jets
  reportSummaryMap_->setBinContent(1,8, summaryContent[3]); // TauJets
  reportSummaryMap_->setBinContent(1,7, summaryContent[2]); // IsoEM
  reportSummaryMap_->setBinContent(1,6, summaryContent[1]); // NonIsoEM
  reportSummaryMap_->setBinContent(1,5, summaryContent[0]); // MET

  //L1TEMU summary map
  reportSummaryMap_->setBinContent(2,11,summaryContent[7]); // GLT
  reportSummaryMap_->setBinContent(2,10,summaryContent[8]); // GMT
  reportSummaryMap_->setBinContent(2,9, summaryContent[9]); // RPC
  reportSummaryMap_->setBinContent(2,8, summaryContent[10]);// CTP
  reportSummaryMap_->setBinContent(2,7, summaryContent[11]);// CTF
  reportSummaryMap_->setBinContent(2,6, summaryContent[12]);// DTP
  reportSummaryMap_->setBinContent(2,5, summaryContent[13]);// DTF
  reportSummaryMap_->setBinContent(2,4, summaryContent[14]);// HTP
  reportSummaryMap_->setBinContent(2,3, summaryContent[15]);// ETP
  reportSummaryMap_->setBinContent(2,2, summaryContent[16]);// GCT
  reportSummaryMap_->setBinContent(2,1, summaryContent[17]);// RCT


  if( verbose_ ){
    if( maskedData.size()>0 ){
      std::cout << "  Masked Data Systems = ";
      for( unsigned int i=0; i<maskedData.size(); i++ ){
	if( i!=maskedData.size()-1 ){
	  std::cout << maskedData[i] << ", ";
	}
	else {
	  std::cout << maskedData[i] << std::endl;
	}
      }
    }
    if( maskedEmul.size()>0 ){
      std::cout << "  Masked Emul Systems = ";
      for( unsigned int i=0; i<maskedEmul.size(); i++ ){
	if( i!=maskedEmul.size()-1 ){
	  std::cout << maskedEmul[i] << ", ";
	}
	else {
	  std::cout << maskedEmul[i] << std::endl;
	}
      }
    }

    std::cout << "  L1T " << std::endl;
    std::cout << "     summaryContent[0]  = MET      = " << summaryContent[0] << std::endl;
    std::cout << "     summaryContent[1]  = NonIsoEM = " << summaryContent[1] << std::endl;
    std::cout << "     summaryContent[2]  = IsoEM    = " << summaryContent[2] << std::endl;
    std::cout << "     summaryContent[3]  = TauJets  = " << summaryContent[3] << std::endl;
    std::cout << "     summaryContent[4]  = Jets     = " << summaryContent[4] << std::endl;
    std::cout << "     summaryContent[5]  = Muons    = " << summaryContent[5] << std::endl;
    std::cout << "     summaryContent[6]  = GT       = " << summaryContent[6] << std::endl;
    std::cout << "  L1T EMU" << std::endl;
    std::cout << "     summaryContent[7]  = GLT      = " << summaryContent[7] << std::endl;
    std::cout << "     summaryContent[8]  = GMT      = " << summaryContent[8] << std::endl;
    std::cout << "     summaryContent[9]  = RPC      = " << summaryContent[9] << std::endl;
    std::cout << "     summaryContent[10] = CTP      = " << summaryContent[10] << std::endl;
    std::cout << "     summaryContent[11] = CTF      = " << summaryContent[11] << std::endl;
    std::cout << "     summaryContent[12] = DTP      = " << summaryContent[12] << std::endl;
    std::cout << "     summaryContent[13] = DTF      = " << summaryContent[13] << std::endl;
    std::cout << "     summaryContent[14] = HTP      = " << summaryContent[14] << std::endl;
    std::cout << "     summaryContent[15] = ETP      = " << summaryContent[15] << std::endl;
    std::cout << "     summaryContent[16] = GCT      = " << summaryContent[16] << std::endl;
    std::cout << "     summaryContent[17] = RCT      = " << summaryContent[17] << std::endl;
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

string L1TEventInfoClient::StringToUpper(string strToConvert)
{//change each element of the string to upper case
   for(unsigned int i=0;i<strToConvert.length();i++)
   {
      strToConvert[i] = toupper(strToConvert[i]);
   }
   return strToConvert;//return the converted string
}







