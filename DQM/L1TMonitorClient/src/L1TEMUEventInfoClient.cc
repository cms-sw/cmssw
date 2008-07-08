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

  int nSubsystems = 20;
  
  char histo[100];
  
  for (int i = 0; i < nSubsystems; i++) {    

    switch(i){
    case 0 :   sprintf(histo,"L1TEMU_ECAL");    break;
    case 1 :   sprintf(histo,"L1TEMU_HCAL");    break;
    case 2 :   sprintf(histo,"L1TEMU_RCT");     break;
    case 3 :   sprintf(histo,"L1TEMU_GCT");     break;
    case 4 :   sprintf(histo,"L1TEMU_DTTPG");   break;
    case 5 :   sprintf(histo,"L1TEMU_DTTF");    break;
    case 6 :   sprintf(histo,"L1TEMU_CSCTPG");  break;
    case 7 :   sprintf(histo,"L1TEMU_CSCTF");   break;
    case 8 :   sprintf(histo,"L1TEMU_RPC");     break;
    case 9 :   sprintf(histo,"L1TEMU_GMT");     break;
    case 10 :  sprintf(histo,"L1TEMU_GT");      break;
    case 11 :  sprintf(histo,"L1TEMU_RPCTG");   break;
    case 12 :  sprintf(histo,"L1TEMU_EMUL");    break;
    case 13 :  sprintf(histo,"L1TEMU_Test1");   break;
    case 14 :  sprintf(histo,"L1TEMU_Test2");   break;
    case 15 :  sprintf(histo,"L1TEMU_Test3");   break;
    case 16 :  sprintf(histo,"L1TEMU_Test4");   break;
    case 17 :  sprintf(histo,"L1TEMU_Test5");   break;
    case 18 :  sprintf(histo,"L1TEMU_Test6");   break;
    case 19 :  sprintf(histo,"L1TEMU_Test7");   break;
    }  
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

  //reportSummaryMap_ = dbe_->book2D("reportSummaryMap", "reportSummaryMap", 5, 0.,5., 4, 0., 4.);
  //reportSummaryMap_->setAxisTitle("Subsystem Index", 1);
  //reportSummaryMap_->setAxisTitle("Subsystem Index", 2);
  reportSummaryMap_ = dbe_->book2D("reportSummaryMap", "reportSummaryMap", 1, 1, 2, 12, 1, 13);
  reportSummaryMap_->setAxisTitle("", 1);
  reportSummaryMap_->setAxisTitle("", 2);
  reportSummaryMap_->setBinLabel(1,"DTTF",2);
  reportSummaryMap_->setBinLabel(2,"DTTPG",2);
  reportSummaryMap_->setBinLabel(3,"CSCTF",2);
  reportSummaryMap_->setBinLabel(4,"CSCTPG",2);
  reportSummaryMap_->setBinLabel(5,"RPC",2);
  reportSummaryMap_->setBinLabel(6,"RPCTG",2);
  reportSummaryMap_->setBinLabel(7,"GMT",2);
  reportSummaryMap_->setBinLabel(8,"ECAL",2);
  reportSummaryMap_->setBinLabel(9,"HCAL",2);
  reportSummaryMap_->setBinLabel(10,"RCT",2);
  reportSummaryMap_->setBinLabel(11,"GCT",2);
  reportSummaryMap_->setBinLabel(12,"GT",2);
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

  int nSubsystems = 20; 
  for (int i = 0; i < nSubsystems; i++) {
    summaryContent[i] = 1;    
    reportSummaryContent_[i]->Fill(1.);
  }
  summarySum = 0;
  

  MonitorElement *ECAL_QHist = dbe_->get("L1TEMU/ECAL/ETPErrorFlag");
  MonitorElement *HCAL_QHist = dbe_->get("L1TEMU/HCAL/HTPErrorFlag");
  MonitorElement *RCT_QHist = dbe_->get("L1TEMU/RCT/RCTErrorFlag");
  MonitorElement *GCT_QHist = dbe_->get("L1TEMU/GCT/GCTErrorFlag");
  MonitorElement *DTTPG_QHist = dbe_->get("L1TEMU/DTTPG/DTPErrorFlag");
  MonitorElement *DTTF_QHist = dbe_->get("L1TEMU/DTTF/DTFErrorFlag");
  MonitorElement *CSCTPG_QHist = dbe_->get("L1TEMU/CSCTPG/CTPErrorFlag");
  MonitorElement *CSCTF_QHist = dbe_->get("L1TEMU/CSCTF/CTFErrorFlag");
  MonitorElement *RPC_QHist = dbe_->get("L1TEMU/RPC/RPCErrorFlag");
  MonitorElement *GMT_QHist = dbe_->get("L1TEMU/GMT/GMTErrorFlag");
  MonitorElement *GT_QHist = dbe_->get("L1TEMU/GT/GLTErrorFlag");

  if(ECAL_QHist){
    if(ECAL_QHist->getEntries())
      summaryContent[0] = (ECAL_QHist->getBinContent(1)) / (ECAL_QHist->getEntries());
    else summaryContent[0] = 1;
    reportSummaryContent_[0]->Fill( summaryContent[0] );
  }
 
 //double ECAL_nEnt = ECAL_QHist->;
//  if (ECAL_QHist){
//    const QReport *ECAL_QReport = ECAL_QHist->getQReport("deDiffInXRange_ErrorFlag");
//    int ECAL_nBadCh = ECAL_QReport->getBadChannels().size();
//    cout << "RPC_nBadCh = " << ECAL_nBadCh << endl;
//  }

 
  if(HCAL_QHist){
    if(HCAL_QHist->getEntries())
       summaryContent[1] = (HCAL_QHist->getBinContent(1)) / (HCAL_QHist->getEntries());
    else summaryContent[1] = 1;
       reportSummaryContent_[1]->Fill( summaryContent[1] );
  }
  
  if(RCT_QHist){
    if(RCT_QHist->getEntries())
      summaryContent[2] = (RCT_QHist->getBinContent(1)) / (RCT_QHist->getEntries());
    else summaryContent[2] = 1;
    reportSummaryContent_[2]->Fill( summaryContent[2] );
  }
  
  if(GCT_QHist){
    if(GCT_QHist->getEntries())
      summaryContent[3] = (GCT_QHist->getBinContent(1)) / (GCT_QHist->getEntries());
    else summaryContent[3] = 1;
    reportSummaryContent_[3]->Fill( summaryContent[3] );
  }

  if(DTTPG_QHist){
    if(DTTPG_QHist->getEntries())
      summaryContent[4] = (DTTPG_QHist->getBinContent(1)) / (DTTPG_QHist->getEntries());
    else summaryContent[4] = 1;
    reportSummaryContent_[4]->Fill( summaryContent[4] );
  }
  
  if(DTTF_QHist){
    if(DTTF_QHist->getEntries())
      summaryContent[5] = (DTTF_QHist->getBinContent(1)) / (DTTF_QHist->getEntries());
    else summaryContent[5] = 1;
    reportSummaryContent_[5]->Fill( summaryContent[5] );
  }
  
  if(CSCTPG_QHist){
    if(CSCTPG_QHist->getEntries())
      summaryContent[6] = (CSCTPG_QHist->getBinContent(1)) / (CSCTPG_QHist->getEntries());
    else summaryContent[6] = 1;
    reportSummaryContent_[6]->Fill( summaryContent[6] );
  }

  if(CSCTF_QHist){
    if(CSCTF_QHist->getEntries())
      summaryContent[7] = (CSCTF_QHist->getBinContent(1)) / (CSCTF_QHist->getEntries());
    else summaryContent[7] = 1;
    reportSummaryContent_[7]->Fill( summaryContent[7] );
  }
  
  if(RPC_QHist){
    if(RPC_QHist->getEntries())
      summaryContent[8] = (RPC_QHist->getBinContent(1)) / (RPC_QHist->getEntries());
    else summaryContent[8] = 1;
    reportSummaryContent_[8]->Fill( summaryContent[8] );
  }
  
  if(GMT_QHist){
    if(GMT_QHist->getEntries())
      summaryContent[9] = (GMT_QHist->getBinContent(1)) / (GMT_QHist->getEntries());
    else summaryContent[9] = 1;
    reportSummaryContent_[9]->Fill( summaryContent[9] );
  }
  
  if(GT_QHist){
    if(GT_QHist->getEntries())
      summaryContent[10] = (GT_QHist->getBinContent(1)) / (GT_QHist->getEntries());
    else summaryContent[10] = 1;
    reportSummaryContent_[10]->Fill( summaryContent[10] );
  }
  

  for (int i = 0; i < nSubsystems; i++) {    
    summarySum += summaryContent[i];
  }
  
  reportSummary = summarySum / nSubsystems;
  cout << "reportSummary " << reportSummary << endl;
  if (reportSummary_) reportSummary_->Fill(reportSummary);

  //5x4 map
//   int jcount = 0;

//   //fill the known systems
//   for (int i = 0; i < nSubsystems; i++) {
//     cout << "summaryContent[" << i << "]" << summaryContent[i] << endl;
//     if((i%5)==0)jcount++;
//     reportSummaryMap_->setBinContent(i%5+1,jcount, summaryContent[i]);
//   }




   //12x1 summary map
  reportSummaryMap_->setBinContent(1,1,summaryContent[5]);//DTTF
  reportSummaryMap_->setBinContent(1,2,summaryContent[4]);//DTTPG
  reportSummaryMap_->setBinContent(1,3,summaryContent[7]);//CSCTF
  reportSummaryMap_->setBinContent(1,4,summaryContent[6]);//CSCTPG
  reportSummaryMap_->setBinContent(1,5,summaryContent[8]);//RPC
  reportSummaryMap_->setBinContent(1,6,summaryContent[11]);//RPCTG
  reportSummaryMap_->setBinContent(1,7,summaryContent[9]);//GMT
  reportSummaryMap_->setBinContent(1,8,summaryContent[0]);//ECAL
  reportSummaryMap_->setBinContent(1,9,summaryContent[1]);//HCAL
  reportSummaryMap_->setBinContent(1,10,summaryContent[2]);//RCT
  reportSummaryMap_->setBinContent(1,11,summaryContent[3]);//GCT
  reportSummaryMap_->setBinContent(1,12,summaryContent[10]);//GT



//   //fill the rest
//   for (int i = 0; i < 5; i++) {    
//     for (int j = 0; j < 4; j++) {    

//      reportSummaryMap_->setBinContent( i, j, 1. );
//     }
//   }


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








