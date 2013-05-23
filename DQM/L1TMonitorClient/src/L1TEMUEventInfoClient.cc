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

  emulatorMap[0]  = 13;
  emulatorMap[1]  = 12;
  emulatorMap[2]  = 11;
  emulatorMap[3]  = 10;
  emulatorMap[4]  = 9;
  emulatorMap[5]  = 8;
  emulatorMap[6]  = 15;
  emulatorMap[7]  = 14;
  emulatorMap[8]  = 17;
  emulatorMap[9]  = 16;
  emulatorMap[10] = 7;

}

//--------------------------------------------------------
void L1TEMUEventInfoClient::beginJob(void){

  if(verbose_) cout <<"[TriggerDQM]: Begin Job" << endl;
  // get backendinterface  
  dbe_ = Service<DQMStore>().operator->();

  dbe_->setCurrentFolder("L1TEMU/EventInfo");

//  sprintf(histo, "reportSummary");
  if( (reportSummary_ = dbe_->get("L1TEMU/EventInfo/reportSumamry")) ){
    dbe_->removeElement(reportSummary_->getName()); 
   }
  
  reportSummary_ = dbe_->bookFloat("reportSummary");

  //initialize reportSummary to 1
  if (reportSummary_) reportSummary_->Fill(1);

  dbe_->setCurrentFolder("L1TEMU/EventInfo/reportSummaryContents");

  
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


  dbe_->setCurrentFolder("L1TEMU/EventInfo");

  if( (reportSummaryMap_ = dbe_->get("L1TEMU/EventInfo/reportSummaryMap")) ){
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
void L1TEMUEventInfoClient::beginRun(const Run& r, const EventSetup& context) {
}

//--------------------------------------------------------
void L1TEMUEventInfoClient::beginLuminosityBlock(const LuminosityBlock& lumiSeg, const EventSetup& context) {
   // optionally reset histograms here
}

void L1TEMUEventInfoClient::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
                          const edm::EventSetup& c){


  for (int k = 0; k < nsys_; k++) {
    summaryContent[k] = 0;
    reportSummaryContent_[k]->Fill(0.);
  }
  summarySum = 0;


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
	for( int m=0; m<7; m++ ) summaryContent[m] = -1;
	maskedData.push_back(mask_sys_tmp);
	break;
      case data_gt:
	summaryContent[6]=-1;
	maskedData.push_back(mask_sys_tmp);
	break;
      case data_muons:
	summaryContent[5]=-1;
	maskedData.push_back(mask_sys_tmp);
	break;
      case data_jets:
	summaryContent[4]=-1;
	maskedData.push_back(mask_sys_tmp);
	break;
      case data_taujets:
	summaryContent[3]=-1;
	maskedData.push_back(mask_sys_tmp);
	break;
      case data_isoem:
	summaryContent[2]=-1;
	maskedData.push_back(mask_sys_tmp);
	break;
      case data_nonisoem:
	summaryContent[1]=-1;
	maskedData.push_back(mask_sys_tmp);
	break;
      case data_met:
	summaryContent[0]=-1;
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
	for( int m=7; m<18; m++ ) summaryContent[m] = -1;
	maskedEmul.push_back(mask_sys_tmp);
	break;
      case emul_glt:
	summaryContent[7]=-1;
	maskedEmul.push_back(mask_sys_tmp);
	break;
      case emul_gmt:
	summaryContent[8]=-1;
	maskedEmul.push_back(mask_sys_tmp);
	break;
      case emul_rpc:
	summaryContent[9]=-1;
	maskedEmul.push_back(mask_sys_tmp);
	break;
      case emul_ctp:
	summaryContent[10]=-1;
	maskedEmul.push_back(mask_sys_tmp);
	break;
      case emul_ctf:
	summaryContent[11]=-1;
	maskedEmul.push_back(mask_sys_tmp);
	break;
      case emul_dtp:
	summaryContent[12]=-1;
	maskedEmul.push_back(mask_sys_tmp);
	break;
      case emul_dtf:
	summaryContent[13]=-1;
	maskedEmul.push_back(mask_sys_tmp);
	break;
      case emul_htp:
	summaryContent[14]=-1;
	maskedEmul.push_back(mask_sys_tmp);
	break;
      case emul_etp:
	summaryContent[15]=-1;
	maskedEmul.push_back(mask_sys_tmp);
	break;
      case emul_gct:
	summaryContent[16]=-1;
	maskedEmul.push_back(mask_sys_tmp);
	break;
      case emul_rct:
	summaryContent[17]=-1;
	maskedEmul.push_back(mask_sys_tmp);
	break;
      default:
	if( verbose_ ) cout << "   User input mask '" << mask_sys_tmp << "' is not recognized." << endl;
	break;
      }
  }

  for( int i=0; i<nsysmon_; i++ ){
    if( summaryContent[emulatorMap[i]]==-1 ) sysmask_[i] = 1;
  }


  MonitorElement* QHist[nsysmon_];   
  std::string lbl("");  
  for(int i=0; i<nsysmon_; i++) {
    lbl.clear();
    lbl+="L1TEMU/"; lbl+=syslabel_[i]; lbl+="/"; 
    lbl+=syslabelext_[i]; lbl+="ErrorFlag";
    QHist[i]=dbe_->get(lbl.data());
    float pv = -1.;
    if(!sysmask_[i]){
      pv = setSummary(QHist[i]);
    }
    summaryContent[emulatorMap[i]] = pv;
  }



  int numUnMaskedSystems = 0;
  for( int m=0; m<nsys_; m++ ){
    if( summaryContent[m]!=-1){
      if( m>6 ){
	summarySum += summaryContent[m];
	numUnMaskedSystems++;
      }

      reportSummaryContent_[m]->Fill( summaryContent[m] );
    }
  }



  // For now, only use L1TEMU for reportSummary value
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

string L1TEMUEventInfoClient::StringToUpper(string strToConvert)
{//change each element of the string to upper case
   for(unsigned int i=0;i<strToConvert.length();i++)
   {
      strToConvert[i] = toupper(strToConvert[i]);
   }
   return strToConvert;//return the converted string
}








