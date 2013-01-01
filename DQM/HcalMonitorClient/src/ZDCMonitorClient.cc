#include "DQM/HcalMonitorClient/interface/ZDCMonitorClient.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "TROOT.h"
#include "TTree.h"
#include "TGaxis.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TFile.h"

#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sys/time.h>

#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"

#include "DQM/HcalMonitorClient/interface/HcalDQMDbInterface.h"
// Use to hold/get channel status
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"
#include "CondFormats/HcalObjects/interface/HcalChannelStatus.h"

#include "DQM/HcalMonitorClient/interface/HcalClientUtils.h"
#include "DQM/HcalMonitorClient/interface/HcalHistoUtils.h"
#include "DQM/HcalMonitorTasks/interface/HcalEtaPhiHists.h"

#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"


//--------------------------------------------------------
ZDCMonitorClient::ZDCMonitorClient(const edm::ParameterSet& ps){
  
  inputFile_ = ps.getUntrackedParameter<std::string>("inputFile","");
  mergeRuns_ = ps.getUntrackedParameter<bool>("mergeRuns", false);
  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);
  prescaleFactor_ = ps.getUntrackedParameter<int>("prescaleFactor", -1);
  prefixME_ = ps.getUntrackedParameter<std::string>("subSystemFolder", "Hcal/");
  if (prefixME_.substr(prefixME_.size()-1,prefixME_.size())!="/")
    prefixME_.append("/");
  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);
  
  updateTime_ = ps.getUntrackedParameter<int>("UpdateTime",0);
  baseHtmlDir_ = ps.getUntrackedParameter<std::string>("baseHtmlDir", "");
  htmlUpdateTime_ = ps.getUntrackedParameter<int>("htmlUpdateTime", 0);
  htmlFirstUpdate_ = ps.getUntrackedParameter<int>("htmlFirstUpdate",20);
  databasedir_   = ps.getUntrackedParameter<std::string>("databaseDir","");
  databaseUpdateTime_ = ps.getUntrackedParameter<int>("databaseUpdateTime",0);
  databaseFirstUpdate_ = ps.getUntrackedParameter<int>("databaseFirstUpdate",10);

  saveByLumiSection_  = ps.getUntrackedParameter<bool>("saveByLumiSection",false);
  Online_             = ps.getUntrackedParameter<bool>("online",false);
 
  subdir_             = ps.getUntrackedParameter<std::string>("ZDCFolder","ZDCMonitor_Hcal/");  
  if (subdir_.substr(subdir_.size()-1,subdir_.size())!="/")
    subdir_.append("/");
  subdir_=prefixME_+subdir_; // prefixME = "Hcal/", subdir = "ZDCMonitor_Hcal/"

  debug_              = ps.getUntrackedParameter<int>("debug",0);
  ZDCGoodLumi_        = ps.getUntrackedParameter<std::vector<double> > ("ZDC_QIValueForGoodLS");

}


//--------------------------------------------------------
ZDCMonitorClient::~ZDCMonitorClient(){

  if (debug_>0) std::cout << "ZDCMonitorClient: Exit ..." << std::endl;
}


//--------------------------------------------------------
void ZDCMonitorClient::beginJob(){

  if( debug_>0 ) std::cout << "ZDCMonitorClient: beginJob" << std::endl;
  
  ievt_ = 0;

  begin_run_ = false;
  end_run_   = false;

  run_=-1;
  evt_=-1;
  ievt_=0;
  jevt_=0;

  current_time_ = time(NULL);
  last_time_html_ = 0; 
  last_time_db_ = 0;   

  // get hold of back-end interface

  dqmStore_ = edm::Service<DQMStore>().operator->();

  if ( inputFile_.size() != 0 ) 
    {
      if ( dqmStore_ )    dqmStore_->open(inputFile_);
    }

 
  return;
}

//--------------------------------------------------------
void ZDCMonitorClient::beginRun(const edm::Run& r, const edm::EventSetup& c) {

  begin_run_ = true;
  end_run_   = false;

  run_=r.id().run();
  evt_=0;
  jevt_=0;
  htmlcounter_=0;
  /*  if (!dqmStore_) 
    {
      if (debug_>0) std::cout <<"<ZDCMonitorClient::beginRun> dqmStore does not exist!"<<std::endl;
      return;
      }*/
 //subdir_="Hcal/";
  dqmStore_->setCurrentFolder(subdir_); // what is Hcal/ZDCMonitor/EventInfoDUMMY folder
  
  // Add new histograms; remove those created in previous runs
  // prefixMe = Hcal/
  
  ZDCChannelSummary_=dqmStore_->get(subdir_ + "/ZDC_Channel_Summary");
  if (ZDCChannelSummary_) dqmStore_->removeElement(ZDCChannelSummary_->getName());
  ZDCChannelSummary_= dqmStore_->book2D("ZDC_Channel_Summary", "Fraction of Events where ZDC Channels had no Errors" , 2, 0, 2, 9, 0, 9); //This is the histo which will show the health of each ZDC Channel
  ZDCChannelSummary_->setBinLabel(1,"ZDC+",1);
  ZDCChannelSummary_->setBinLabel(2,"ZDC-",1);
  ZDCChannelSummary_->setBinLabel(1,"EM1",2);
  ZDCChannelSummary_->setBinLabel(2,"EM2",2);
  ZDCChannelSummary_->setBinLabel(3,"EM3",2);
  ZDCChannelSummary_->setBinLabel(4,"EM4",2);
  ZDCChannelSummary_->setBinLabel(5,"EM5",2);
  ZDCChannelSummary_->setBinLabel(6,"HAD1",2);
  ZDCChannelSummary_->setBinLabel(7,"HAD2",2);
  ZDCChannelSummary_->setBinLabel(8,"HAD3",2);
  ZDCChannelSummary_->setBinLabel(9,"HAD4",2);
  ZDCChannelSummary_->getTH2F()->SetOption("coltext");  
  
  
  ZDCReportSummary_ = dqmStore_->get(subdir_ + "ZDC_ReportSummary");
  if (ZDCReportSummary_) dqmStore_->removeElement(ZDCReportSummary_->getName());
  ZDCReportSummary_= dqmStore_->book2D("ZDC_ReportSummary","Fraction of Good Lumis for either ZDC",2,0,2,1,0,1);
  ZDCReportSummary_->setBinLabel(1,"ZDC+",1);
  ZDCReportSummary_->setBinLabel(2,"ZDC-",1);
  ZDCReportSummary_->getTH2F()->SetOption("coltext");

  ZDCHotChannelFraction_ = dqmStore_->get(subdir_+"/Errors/HotChannel/ZDC_Hot_Channel_Fraction");
  if (ZDCHotChannelFraction_) dqmStore_->removeElement(ZDCHotChannelFraction_->getName());
  dqmStore_->setCurrentFolder(subdir_ + "/Errors/HotChannel");
  ZDCHotChannelFraction_ = dqmStore_->book2D("ZDC_Hot_Channel_Fraction", "Hot Channel Rates in the ZDC Channels", 2, 0, 2, 9, 0, 9); //Hot channel checker for ZDC
  ZDCHotChannelFraction_->setBinLabel(1,"ZDC+",1);
  ZDCHotChannelFraction_->setBinLabel(2,"ZDC-",1);
  ZDCHotChannelFraction_->setBinLabel(1,"EM1",2);
  ZDCHotChannelFraction_->setBinLabel(2,"EM2",2);
  ZDCHotChannelFraction_->setBinLabel(3,"EM3",2);
  ZDCHotChannelFraction_->setBinLabel(4,"EM4",2);
  ZDCHotChannelFraction_->setBinLabel(5,"EM5",2);
  ZDCHotChannelFraction_->setBinLabel(6,"HAD1",2);
  ZDCHotChannelFraction_->setBinLabel(7,"HAD2",2);
  ZDCHotChannelFraction_->setBinLabel(8,"HAD3",2);
  ZDCHotChannelFraction_->setBinLabel(9,"HAD4",2);
  ZDCHotChannelFraction_->getTH2F()->SetOption("coltext");
 
  ZDCColdChannelFraction_ = dqmStore_->get(subdir_ + "/Errors/ColdChannel/ZDC_Cold_Channel_Fraction");
  if (ZDCColdChannelFraction_) dqmStore_->removeElement(ZDCColdChannelFraction_->getName());
  dqmStore_->setCurrentFolder(subdir_ + "/Errors/ColdChannel");
  ZDCColdChannelFraction_=dqmStore_->book2D("ZDC_Cold_Channel_Fraction", "Cold Channel Rates in the ZDC Channels", 2, 0, 2,9, 0, 9); //Cold channel checker for ZDC                    
  ZDCColdChannelFraction_->setBinLabel(1,"ZDC+",1);
  ZDCColdChannelFraction_->setBinLabel(2,"ZDC-",1);
  ZDCColdChannelFraction_->setBinLabel(1,"EM1",2);
  ZDCColdChannelFraction_->setBinLabel(2,"EM2",2);
  ZDCColdChannelFraction_->setBinLabel(3,"EM3",2);
  ZDCColdChannelFraction_->setBinLabel(4,"EM4",2);
  ZDCColdChannelFraction_->setBinLabel(5,"EM5",2);
  ZDCColdChannelFraction_->setBinLabel(6,"HAD1",2);
  ZDCColdChannelFraction_->setBinLabel(7,"HAD2",2);
  ZDCColdChannelFraction_->setBinLabel(8,"HAD3",2);
  ZDCColdChannelFraction_->setBinLabel(9,"HAD4",2);
  ZDCColdChannelFraction_->getTH2F()->SetOption("coltext");

 
  ZDCDeadChannelFraction_ = dqmStore_->get(subdir_ + "/Errors/DeadChannel/ZDC_Dead_Channel_Fraction");
  if ( ZDCDeadChannelFraction_) dqmStore_->removeElement(ZDCDeadChannelFraction_->getName());
  dqmStore_->setCurrentFolder(subdir_+ "/Errors/DeadChannel");
  ZDCDeadChannelFraction_=dqmStore_->book2D("ZDC_Dead_Channel_Fraction","Dead Channel Rates in the ZDC Channels",2,0,2,9,0,9);
  ZDCDeadChannelFraction_->setBinLabel(1,"ZDC+",1);
  ZDCDeadChannelFraction_->setBinLabel(2,"ZDC-",1);
  ZDCDeadChannelFraction_->setBinLabel(1,"EM1",2);
  ZDCDeadChannelFraction_->setBinLabel(2,"EM2",2);
  ZDCDeadChannelFraction_->setBinLabel(3,"EM3",2);
  ZDCDeadChannelFraction_->setBinLabel(4,"EM4",2);
  ZDCDeadChannelFraction_->setBinLabel(5,"EM5",2);
  ZDCDeadChannelFraction_->setBinLabel(6,"HAD1",2);
  ZDCDeadChannelFraction_->setBinLabel(7,"HAD2",2);
  ZDCDeadChannelFraction_->setBinLabel(8,"HAD3",2);
  ZDCDeadChannelFraction_->setBinLabel(9,"HAD4",2);
  ZDCDeadChannelFraction_->getTH2F()->SetOption("coltext");

  ZDCDigiErrorFraction_ = dqmStore_->get(subdir_ + "/Errors/Digis/ZDC_Digi_Error_Fraction");
  if (ZDCDigiErrorFraction_) dqmStore_->removeElement(ZDCDigiErrorFraction_->getName());
  dqmStore_->setCurrentFolder(subdir_ + "/Errors/Digis");
  ZDCDigiErrorFraction_=dqmStore_->book2D("ZDC_Digi_Error_Fraction", "Digi Error Rates in the ZDC Channels", 2, 0, 2,9, 0, 9); //Hot channel checker for ZDC                    
  ZDCDigiErrorFraction_->setBinLabel(1,"ZDC+",1);
  ZDCDigiErrorFraction_->setBinLabel(2,"ZDC-",1);
  ZDCDigiErrorFraction_->setBinLabel(1,"EM1",2);
  ZDCDigiErrorFraction_->setBinLabel(2,"EM2",2);
  ZDCDigiErrorFraction_->setBinLabel(3,"EM3",2);
  ZDCDigiErrorFraction_->setBinLabel(4,"EM4",2);
  ZDCDigiErrorFraction_->setBinLabel(5,"EM5",2);
  ZDCDigiErrorFraction_->setBinLabel(6,"HAD1",2);
  ZDCDigiErrorFraction_->setBinLabel(7,"HAD2",2);
  ZDCDigiErrorFraction_->setBinLabel(8,"HAD3",2);
  ZDCDigiErrorFraction_->setBinLabel(9,"HAD4",2);
  ZDCDigiErrorFraction_->getTH2F()->SetOption("coltext");
  
}


//--------------------------------------------------------
void ZDCMonitorClient::endJob(void) {

  if( debug_>0 ) 
    std::cout << "ZDCMonitorClient: endJob, ievt = " << ievt_ << std::endl;

  return;
}

//--------------------------------------------------------
void ZDCMonitorClient::endRun(const edm::Run& r, const edm::EventSetup& c) {

  if (debug_>0)
    std::cout << std::endl<<"<ZDCMonitorClient> Standard endRun() for run " << r.id().run() << std::endl<<std::endl;

  begin_run_ = false;
  end_run_   = true;

  if( debug_ >0) std::cout <<"ZDCMonitorClient: processed events: "<<ievt_<<std::endl;

  // analyze at least once (for offline?)
  this->analyze();
  
  return;
}


//--------------------------------------------------------
void ZDCMonitorClient::beginLuminosityBlock(const edm::LuminosityBlock &l, const edm::EventSetup &c) 
{
  // don't allow 'backsliding' across lumi blocks in online running
  // This still won't prevent some lumi blocks from being evaluated multiple times.  Need to think about this.
  //if (Online_ && (int)l.luminosityBlock()<ilumisec_) return;
  if (debug_>0) std::cout <<"Entered Monitor Client beginLuminosityBlock for LS = "<<l.luminosityBlock()<<std::endl;
}

//--------------------------------------------------------
void ZDCMonitorClient::endLuminosityBlock(const edm::LuminosityBlock &l, const edm::EventSetup &c) {

  // don't allow backsliding in online running
  //if (Online_ && (int)l.luminosityBlock()<ilumisec_) return;
  if( debug_>0 ) std::cout << "ZDCMonitorClient: std::endluminosityBlock" << std::endl;

  current_time_ = time(NULL);
  if (updateTime_>0)
    {
      if ((current_time_-last_time_update_)<60*updateTime_)
	return;
      last_time_update_ = current_time_;
    }
  this->analyze(l.luminosityBlock());
  return;
}

//--------------------------------------------------------
void ZDCMonitorClient::analyze(const edm::Event& e, const edm::EventSetup& eventSetup){

  if (debug_>1)
    std::cout <<"Entered ZDCMonitorClient::analyze(const Evt...)"<<std::endl;
  
  ievt_++;
  jevt_++;

  run_=e.id().run();
  evt_=e.id().event();
   if (prescaleFactor_>0 && jevt_%prescaleFactor_==0) 
    this->analyze(e.luminosityBlock());
} //end EDAnalyzer analyze method

void ZDCMonitorClient::analyze(int LS)
{
  if (debug_>0) 
    std::cout <<"<ZDCMonitorClient> Entered ZDCMonitorClient::analyze()"<<std::endl;
  if(debug_>1) std::cout<<"\nZDC Monitor Client heartbeat...."<<std::endl;
  
  float ChannelRatio[18]={0};
   //dqmStore_->runQTests();


  
  // Make a rate plot, by first getting plots from tasks
  MonitorElement* me;
  std::string s;

  //////////////////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////  1)   DIGI ERROR RATE PLOT     /////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////

  s=subdir_+"/Errors/Digis/ZDC_Digi_Errors";  // prefixME_ = "Hcal/"
  me=dqmStore_->get(s.c_str());
  TH2F* numplot=0;
  if (me!=0)
    {
      numplot=HcalUtilsClient::getHisto<TH2F*>(me,false,0,0);
      if (numplot!=0)
	{
	  int nevents3 = numplot->GetBinContent(-1,-1);
	  if(nevents3 != 0)
	    {
	      for (int i=0;i<18;++i)
		{
		  ZDCDigiErrorFraction_->setBinContent((i/9)+1,(i%9)+1,(numplot->GetBinContent((i/9)+1,(i%9)+1)*1./nevents3));
		}
	    }
	}
    }


  //////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////// 2)  HOT CHANNEL RATE PLOT     /////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////

  // Now get Hot Channel plot, used normalized values (num hot/event) in ZDCHotChannelFraction_
  s=subdir_+"/Errors/HotChannel/ZDC_Hot_Channel_Errors";
  me=dqmStore_->get(s.c_str());
  TH2F* myhist=0;
  if (me!=0)
    {
      myhist=HcalUtilsClient::getHisto<TH2F*>(me,false,0,0);
      if (myhist!=0)
	{
	  int nevents = myhist->GetBinContent(-1,-1);
	  if(nevents!=0)
	    {
	      for (int i=0;i<18;++i)
		{
		  ZDCHotChannelFraction_->setBinContent((i/9)+1,(i%9)+1,((myhist->GetBinContent((i/9)+1,(i%9)+1))*1./nevents));
		}
	    }			 
	}     
    } // if (me!=0)


  //////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////// 3)  Cold CHANNEL RATE PLOT     /////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////

  //Now get Cold Channel plot, used normalized values (num Cold/event) in ZDCColdChannelFraction_
  s=subdir_+"/Errors/ColdChannel/ZDC_Cold_Channel_Errors";
  me=dqmStore_->get(s.c_str());
  TH2F* myhist1=0;
  if (me!=0)
    {
      myhist1=HcalUtilsClient::getHisto<TH2F*>(me,false,0,0);
      if ((myhist1)!=0)
	{
	  int normalization = myhist1->GetBinContent(-1,-1);
	  if(normalization!=0)
	    {
	      for (int i=0;i<18;++i)
		{
		  ZDCColdChannelFraction_->setBinContent((i/9)+1,(i%9)+1,((myhist1->GetBinContent((i/9)+1,(i%9)+1))*1./normalization));
		}
	    }
	}
    }



  //////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////// 4)  Dead CHANNEL RATE PLOT     /////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////

  //Now get Cold Channel plot, used normalized values (num Cold/event) in ZDCHotChannelFraction_
  s=subdir_+"/Errors/DeadChannel/ZDC_Dead_Channel_Errors";
  me=dqmStore_->get(s.c_str());
  TH2F* myhist6=0;
  if (me!=0)
    {
      myhist6=HcalUtilsClient::getHisto<TH2F*>(me,false,0,0);
      if ((myhist6)!=0)
	{
	  int normalizer = myhist6->GetBinContent(-1,-1);
	  if(normalizer!=0)
	    {
	      for (int i=0;i<18;++i)
		{
		  ZDCDeadChannelFraction_->setBinContent((i/9)+1,(i%9)+1,(myhist6->GetBinContent((i/9)+1,(i%9)+1))*1./normalizer);
		}
	    }
	}
    }





  //////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////// 5)  CHANNEL SUMMARY PLOT     /////////////////////////////////////////
  //     This simply takes the total errors that each channel got (Cold,hot, digi error) sums them up from the 
  //  totalchannelerrors plot and turns them into a rate. 1-errorrate be the number displayed.            //
  /////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////


  ///now we will make the channel summary map
  s=subdir_+"/Errors/ZDC_TotalChannelErrors";
  me=dqmStore_->get(s.c_str());
  TH2F* myhist2=0;
  if (me!=0)
    {
      myhist2=HcalUtilsClient::getHisto<TH2F*>(me,false,0,0);
      int nevents2 = myhist2->GetBinContent(-1,-1);
      if(nevents2!=0)
	{
	  for (int i=0;i<18;++i)
	    {
	      ChannelRatio[i]=(myhist2->GetBinContent((i/9)+1,(i%9)+1))*1./nevents2;
	      ZDCChannelSummary_->setBinContent((i/9)+1,(i%9)+1,1-ChannelRatio[i]);
	    }
	}
    }



  //////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////// 6)  ZDC REPORT SUMMARY PLOT     /////////////////////////////////////////
  //     This is a ratio of GoodLumis/TotalLumis. The determination of which is made by the Quality Index plots.            //
  /////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////

  LumiCounter=0;
  PZDC_GoodLumiCounter=0;
  PZDC_LumiRatio=0.;
  NZDC_GoodLumiCounter=0;
  NZDC_LumiRatio=0.;
  s=subdir_+"EventsVsLS";
  me=dqmStore_->get(s.c_str());
  TH1F* myhist3=0;
  if (me!=0)
    myhist3=HcalUtilsClient::getHisto<TH1F*>(me,false,0,0);
  s=subdir_+"PZDC_QualityIndexVSLB";
  me=dqmStore_->get(s.c_str());
  TH1F* myhist4=0;
  if (me!=0)
    myhist4=HcalUtilsClient::getHisto<TH1F*>(me,false,0,0);
  s=subdir_+"NZDC_QualityIndexVSLB";
  me=dqmStore_->get(s.c_str());
  TH1F* myhist5=0;
  if (me!=0)
    myhist5=HcalUtilsClient::getHisto<TH1F*>(me,false,0,0);
  for (int i=1;i<=myhist3->GetNbinsX();++i)
    {
      if (myhist3->GetBinContent(i)==0)
	continue;
      LumiCounter+=1;
      if(myhist4->GetBinContent(i)>ZDCGoodLumi_[0])
	PZDC_GoodLumiCounter+=1;
      if(myhist5->GetBinContent(i)>ZDCGoodLumi_[1])
	NZDC_GoodLumiCounter+=1;
      PZDC_LumiRatio=PZDC_GoodLumiCounter*(1./LumiCounter);
      NZDC_LumiRatio=NZDC_GoodLumiCounter*(1./LumiCounter);
    }
  
  ZDCReportSummary_->setBinContent(1,1,PZDC_LumiRatio);
  ZDCReportSummary_->setBinContent(2,1,NZDC_LumiRatio);

  return;
}


DEFINE_FWK_MODULE(ZDCMonitorClient);
