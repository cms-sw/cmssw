/*
 * \file HcalMonitorClient.cc
 * 
 * $Date: 2012/11/02 14:23:43 $
 * $Revision: 1.103 $
 * \author J. Temple
 * 
 */

#include "DQM/HcalMonitorClient/interface/HcalMonitorClient.h"
#include "DQM/HcalMonitorClient/interface/HcalDeadCellClient.h"
#include "DQM/HcalMonitorClient/interface/HcalHotCellClient.h"
#include "DQM/HcalMonitorClient/interface/HcalRecHitClient.h"
#include "DQM/HcalMonitorClient/interface/HcalRawDataClient.h"
#include "DQM/HcalMonitorClient/interface/HcalDigiClient.h"
#include "DQM/HcalMonitorClient/interface/HcalTrigPrimClient.h"
#include "DQM/HcalMonitorClient/interface/HcalBeamClient.h"
#include "DQM/HcalMonitorClient/interface/HcalNZSClient.h"
#include "DQM/HcalMonitorClient/interface/HcalSummaryClient.h"
#include "DQM/HcalMonitorClient/interface/HcalDetDiagPedestalClient.h"
#include "DQM/HcalMonitorClient/interface/HcalDetDiagLaserClient.h"
#include "DQM/HcalMonitorClient/interface/HcalDetDiagLEDClient.h"
#include "DQM/HcalMonitorClient/interface/HcalDetDiagNoiseMonitorClient.h"
#include "DQM/HcalMonitorClient/interface/HcalDetDiagTimingClient.h"
#include "DQM/HcalMonitorClient/interface/HcalCoarsePedestalClient.h"

#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "CondFormats/HcalObjects/interface/HcalChannelStatus.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"
#include "CondFormats/DataRecord/interface/HcalChannelQualityRcd.h"

#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"

#include <iostream>
#include <iomanip>
#include <fstream>

#include "TROOT.h"
#include "TH1.h"

//'using' declarations should only be used within classes/functions, and 'using namespace std;' should not be used,
// according to Bill Tanenbaum  (DQM development hypernews, 25 March 2010)

HcalMonitorClient::HcalMonitorClient(const edm::ParameterSet& ps)
{
  debug_ = ps.getUntrackedParameter<int>("debug",0);
  inputFile_ = ps.getUntrackedParameter<std::string>("inputFile","");
  mergeRuns_ = ps.getUntrackedParameter<bool>("mergeRuns", false);
  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);
  prescaleFactor_ = ps.getUntrackedParameter<int>("prescaleFactor", -1);
  prefixME_ = ps.getUntrackedParameter<std::string>("subSystemFolder", "Hcal/");
  if (prefixME_.substr(prefixME_.size()-1,prefixME_.size())!="/")
    prefixME_.append("/");
  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);
  enabledClients_ = ps.getUntrackedParameter<std::vector<std::string> >("enabledClients", enabledClients_);

  updateTime_ = ps.getUntrackedParameter<int>("UpdateTime",0);
  baseHtmlDir_ = ps.getUntrackedParameter<std::string>("baseHtmlDir", "");
  htmlUpdateTime_ = ps.getUntrackedParameter<int>("htmlUpdateTime", 0);
  htmlFirstUpdate_ = ps.getUntrackedParameter<int>("htmlFirstUpdate",20);
  databasedir_   = ps.getUntrackedParameter<std::string>("databaseDir","");
  databaseUpdateTime_ = ps.getUntrackedParameter<int>("databaseUpdateTime",0);
  databaseFirstUpdate_ = ps.getUntrackedParameter<int>("databaseFirstUpdate",10);

  saveByLumiSection_  = ps.getUntrackedParameter<bool>("saveByLumiSection",false);
  Online_                = ps.getUntrackedParameter<bool>("online",false);


  if (debug_>0)
    {
      std::cout <<"HcalMonitorClient:: The following clients are enabled:"<<std::endl;
      for (unsigned int i=0;i<enabledClients_.size();++i)
	  std::cout <<enabledClients_[i]<<std::endl;
    } // if (debug_>0)

  // Set all EtaPhiHists pointers to 0 to start
  ChannelStatus=0; 
  ADC_PedestalFromDBByDepth=0;
  ADC_WidthFromDBByDepth=0;
  fC_PedestalFromDBByDepth=0;
  fC_WidthFromDBByDepth=0;

  // Add all relevant clients
  clients_.clear();
  clients_.reserve(14); // any reason to reserve ahead of time?
  summaryClient_=0;

  clients_.push_back(new HcalBaseDQClient((std::string)"HcalMonitorModule",ps));
  if (find(enabledClients_.begin(), enabledClients_.end(),"DeadCellMonitor")!=enabledClients_.end())
    clients_.push_back(new HcalDeadCellClient((std::string)"DeadCellMonitor",ps));
  if (find(enabledClients_.begin(), enabledClients_.end(),"HotCellMonitor")!=enabledClients_.end())
    clients_.push_back(new HcalHotCellClient((std::string)"HotCellMonitor",ps));
  if (find(enabledClients_.begin(), enabledClients_.end(),"RecHitMonitor")!=enabledClients_.end())
    clients_.push_back(new HcalRecHitClient((std::string)"RecHitMonitor",ps));
  if (find(enabledClients_.begin(), enabledClients_.end(),"DigiMonitor")!=enabledClients_.end())
    clients_.push_back(new HcalDigiClient((std::string)"DigiMonitor",ps));
  if (find(enabledClients_.begin(), enabledClients_.end(),"RawDataMonitor")!=enabledClients_.end())
    clients_.push_back(new HcalRawDataClient((std::string)"RawDataMonitor",ps));
  if (find(enabledClients_.begin(), enabledClients_.end(),"TrigPrimMonitor")!=enabledClients_.end())
    clients_.push_back(new HcalTrigPrimClient((std::string)"TrigPrimMonitor",ps));
  if (find(enabledClients_.begin(), enabledClients_.end(),"NZSMonitor")!=enabledClients_.end())
    clients_.push_back(new HcalNZSClient((std::string)"NZSMonitor",ps));
  if (find(enabledClients_.begin(), enabledClients_.end(),"BeamMonitor")!=enabledClients_.end())
    clients_.push_back(new HcalBeamClient((std::string)"BeamMonitor",ps));
  if (find(enabledClients_.begin(), enabledClients_.end(),"DetDiagPedestalMonitor")!=enabledClients_.end())
    clients_.push_back(new HcalDetDiagPedestalClient((std::string)"DetDiagPedestalMonitor",ps));
  if (find(enabledClients_.begin(), enabledClients_.end(),"DetDiagLaserMonitor")!=enabledClients_.end())
    clients_.push_back(new HcalDetDiagLaserClient((std::string)"DetDiagLaserMonitor",ps));
  if (find(enabledClients_.begin(), enabledClients_.end(),"DetDiagLEDMonitor")!=enabledClients_.end())
    clients_.push_back(new HcalDetDiagLEDClient((std::string)"DetDiagLEDMonitor",ps));
  if (find(enabledClients_.begin(), enabledClients_.end(),"DetDiagNoiseMonitor")!=enabledClients_.end())
    clients_.push_back(new HcalDetDiagNoiseMonitorClient((std::string)"DetDiagNoiseMonitor",ps));
  if (find(enabledClients_.begin(), enabledClients_.end(),"DetDiagTimingMonitor")!=enabledClients_.end())
    clients_.push_back(new HcalDetDiagTimingClient((std::string)"DetDiagTimingMonitor",ps));
 if (find(enabledClients_.begin(), enabledClients_.end(),"CoarsePedestalMonitor")!=enabledClients_.end())
    clients_.push_back(new HcalCoarsePedestalClient((std::string)"CoarsePedestalMonitor",ps));

  if (find(enabledClients_.begin(), enabledClients_.end(),"Summary")!=enabledClients_.end())
    summaryClient_ = new HcalSummaryClient((std::string)"ReportSummaryClient",ps);
  
} // HcalMonitorClient constructor


HcalMonitorClient::~HcalMonitorClient()
{
  if (debug_>0) std::cout <<"<HcalMonitorClient>  Exiting..."<<std::endl;
  for (unsigned int i=0;i<clients_.size();++i)
    delete clients_[i];
  //if (summaryClient_) delete summaryClient_;

}

void HcalMonitorClient::beginJob(void)
{

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

  for ( unsigned int i=0; i<clients_.size();++i ) 
    clients_[i]->beginJob();

  if ( summaryClient_ ) summaryClient_->beginJob();
  

} // void HcalMonitorClient::beginJob(void)


void HcalMonitorClient::beginRun(const edm::Run& r, const edm::EventSetup& c) 
{
  if (debug_>0) std::cout <<"<HcalMonitorClient::beginRun(r,c)>"<<std::endl;
  begin_run_ = true;
  end_run_   = false;

  run_=r.id().run();
  evt_=0;
  jevt_=0;
  htmlcounter_=0;

  // Store list of bad channels and their values
  std::map <HcalDetId, unsigned int> badchannelmap; 
  badchannelmap.clear();

  // Let's get the channel status quality
  edm::ESHandle<HcalTopology> topo;
  c.get<IdealGeometryRecord>().get(topo);

  edm::ESHandle<HcalChannelQuality> p;
  c.get<HcalChannelQualityRcd>().get(p);
  chanquality_= new HcalChannelQuality(*p.product());
  if (!chanquality_->topo()) chanquality_->setTopo(topo.product());
 
  if (dqmStore_ && ChannelStatus==0)
    {
      dqmStore_->setCurrentFolder(prefixME_+"HcalInfo/ChannelStatus");
      ChannelStatus=new EtaPhiHists;
      ChannelStatus->setup(dqmStore_,"ChannelStatus");
      std::stringstream x;
      for (unsigned int d=0;d<ChannelStatus->depth.size();++d)
	{
	  ChannelStatus->depth[d]->Reset();
	  x<<"1+log2(status) for HCAL depth "<<d+1;
	  if (ChannelStatus->depth[d]) ChannelStatus->depth[d]->setTitle(x.str().c_str());
	  x.str("");
	}
    }

  edm::ESHandle<HcalDbService> conditions;
  c.get<HcalDbRecord>().get(conditions);
  // Now let's setup pedestals
  if (dqmStore_ )
    {
      dqmStore_->setCurrentFolder(prefixME_+"HcalInfo/PedestalsFromCondDB");
      if (ADC_PedestalFromDBByDepth==0)
	{
	  ADC_PedestalFromDBByDepth = new EtaPhiHists;
	  ADC_PedestalFromDBByDepth->setup(dqmStore_,"ADC Pedestals From Conditions DB");
	}
      if (ADC_WidthFromDBByDepth==0)
	{
	  ADC_WidthFromDBByDepth = new EtaPhiHists;
	  ADC_WidthFromDBByDepth->setup(dqmStore_,"ADC Widths From Conditions DB");
	}
      if (fC_PedestalFromDBByDepth==0)
	{
	  fC_PedestalFromDBByDepth = new EtaPhiHists;
	  fC_PedestalFromDBByDepth->setup(dqmStore_,"fC Pedestals From Conditions DB");
	}
      if (fC_WidthFromDBByDepth==0)
	{
	  fC_WidthFromDBByDepth = new EtaPhiHists;
	  fC_WidthFromDBByDepth->setup(dqmStore_,"fC Widths From Conditions DB");
	}
      PlotPedestalValues(*conditions);
    }

  // Find only channels with non-zero quality, and add them to badchannelmap
  std::vector<DetId> mydetids = chanquality_->getAllChannels();
  for (std::vector<DetId>::const_iterator i = mydetids.begin();i!=mydetids.end();++i)
    {
      if (i->det()!=DetId::Hcal) continue; // not an hcal cell
      HcalDetId id=HcalDetId(*i);
      int status=(chanquality_->getValues(id))->getValue();
      //if (status!=status) status=-1;  // protects against NaN values
      // The above line doesn't seem to work in identifying NaNs;  ints for bad values come back as negative numbers (at least in run 146501)
      if (status==0) continue;
      badchannelmap[id]=status;

      // Fill Channel Status histogram
      if (dqmStore_==0) continue;
      int depth=id.depth();
      if (depth<1 || depth>4) continue;
      int ieta=id.ieta();
      int iphi=id.iphi();
      if (id.subdet()==HcalForward)
	ieta>0 ? ++ieta: --ieta;

      double logstatus = 0;
      // Fill ChannelStatus value with '-1' when a 'NaN' occurs
      if (status<0)
	logstatus=-1*(log2(-1.*status)+1);
      else
	logstatus=log2(1.*status)+1;
      if (ChannelStatus->depth[depth-1]) ChannelStatus->depth[depth-1]->Fill(ieta,iphi,logstatus);
    }
    
  for (unsigned int i=0;i<clients_.size();++i)
    {
      if (clients_[i]->name()=="RawDataMonitor") clients_[i]->setEventSetup(c);
      clients_[i]->beginRun();
      clients_[i]->setStatusMap(badchannelmap);
    }
  
  if (summaryClient_!=0)
    {
      summaryClient_->getFriends(clients_);
      summaryClient_->beginRun();
    }

} // void HcalMonitorClient::beginRun(const Run& r, const EventSetup& c)

void HcalMonitorClient::beginRun()
{
  // What is the difference between this and beginRun above?
  // When would this be called?
  begin_run_ = true;
  end_run_   = false;
  jevt_ = 0;
  htmlcounter_=0;

  if (dqmStore_==0 || ChannelStatus!=0) return;
  dqmStore_->setCurrentFolder(prefixME_+"HcalInfo");
  ChannelStatus=new EtaPhiHists;
  ChannelStatus->setup(dqmStore_,"ChannelStatus");
  std::stringstream x;
  for (unsigned int d=0;d<ChannelStatus->depth.size();++d)
    {
      x<<"1+log2(status) for HCAL depth "<<d+1;
      if (ChannelStatus->depth[d]) ChannelStatus->depth[d]->setTitle(x.str().c_str());
      x.str("");
    }
} // void HcalMonitorClient::beginRun()

void HcalMonitorClient::setup(void)
{
  // no setup required
}

void HcalMonitorClient::beginLuminosityBlock(const edm::LuminosityBlock &l, const edm::EventSetup &c) 
{
  if (debug_>0) std::cout <<"<HcalMonitorClient::beginLuminosityBlock>"<<std::endl;
} // void HcalMonitorClient::beginLuminosityBlock

void HcalMonitorClient::analyze(const edm::Event & e, const edm::EventSetup & c)
{
  if (debug_>4) 
    std::cout <<"HcalMonitorClient::analyze(const edm::Event&, const edm::EventSetup&) ievt_ = "<<ievt_<<std::endl;
  ievt_++;
  jevt_++;

  run_=e.id().run();
  evt_=e.id().event();
  if (prescaleFactor_>0 && jevt_%prescaleFactor_==0) {

    for (unsigned int i=0;i<clients_.size();++i)
      clients_[i]->getLogicalMap(c); // actually runs just once internally

    this->analyze(e.luminosityBlock());
  }

} // void HcalMonitorClient::analyze(const edm::Event & e, const edm::EventSetup & c)

void HcalMonitorClient::analyze(int LS)
{
  if (debug_>0)
    std::cout <<"HcalMonitorClient::analyze() "<<std::endl;
  current_time_ = time(NULL);
  // no ievt_, jevt_ counters needed here:  this function gets called at endlumiblock, after default analyze function runs
  for (unsigned int i=0;i<clients_.size();++i)
    clients_[i]->analyze();
  if (summaryClient_!=0)
    {
      // Always call basic analyze to form histograms for each task
      summaryClient_->analyze(LS);
      // Call this if LS-by-LS enabling is set to true
      if (saveByLumiSection_==true)
	summaryClient_->fillReportSummaryLSbyLS(LS);
    }
} // void HcalMonitorClient::analyze()


void HcalMonitorClient::endLuminosityBlock(const edm::LuminosityBlock &l, const edm::EventSetup &c) 
{
  if (debug_>0) std::cout <<"<HcalMonitorClient::endLuminosityBlock>"<<std::endl;
  current_time_ = time(NULL);
  if (updateTime_>0)
    {
      if ((current_time_-last_time_update_)<60*updateTime_)
	return;
      last_time_update_ = current_time_;
    }
  this->analyze(l.luminosityBlock());

  if (databaseUpdateTime_>0)
    {
      if (
	  // first update occurs at after databaseFirstUpdate_ minutes
	  (last_time_db_==0 && (current_time_-last_time_db_)>=60*databaseFirstUpdate_)
	  ||
	  // following updates follow once every databaseUpdateTime_ minutes
	  ((current_time_-last_time_db_)>=60*databaseUpdateTime_)
	  )
	{
	  this->writeChannelStatus();
	  last_time_db_=current_time_;
	}
    }

  if (htmlUpdateTime_>0)
    {
      if (
	  (last_time_html_==0 && (current_time_-last_time_html_)>=60*htmlFirstUpdate_)
	  // 
	  ||((current_time_-last_time_html_)>=60*htmlUpdateTime_)
	  ) // htmlUpdateTime_ in minutes
	{
	  this->writeHtml();
	  last_time_html_=current_time_;
	}
    }

} // void HcalMonitorClient::endLuminosityBlock

void HcalMonitorClient::endRun(void)
{
  begin_run_ = false;
  end_run_   = true;

  // Always fill summaryClient at end of run (as opposed to the end-lumi fills, which may just contain info for a single LS)
  // At the end of this run, set LS=-1  (LS-based plotting in doesn't work yet anyway)
  if (summaryClient_)
    summaryClient_->analyze(-1);

  if (databasedir_.size()>0)
    this->writeChannelStatus();
  // writeHtml takes longer; run it last 
  // Also, don't run it if htmlUpdateTime_>0 -- it should have already been run
  if (baseHtmlDir_.size()>0 && htmlUpdateTime_==0)
    this->writeHtml();
}

void HcalMonitorClient::endRun(const edm::Run& r, const edm::EventSetup& c) 
{
  // Set values here, because the "analyze" method occasionally times out, 
  // which keeps the endRun() call from being made.  This causes endJob to
  // crash, since end_run_ is still set to false at that point.
  begin_run_ = false;
  end_run_   = true;

  this->analyze();
  this->endRun();
}

void HcalMonitorClient::endJob(void)
{
  // Temporary fix for crash of April 2011 in online DQM
  if (Online_==true)
    return;

  if (! end_run_)
    {
      this->analyze();
      this->endRun();
    }
  this->cleanup(); // currently does nothing

  for ( unsigned int i=0; i<clients_.size(); i++ ) 
    clients_[i]->endJob();
  //if ( summaryClient_ ) summaryClient_->endJob();

} // void HcalMonitorClient::endJob(void)

void HcalMonitorClient::cleanup(void)
{
  if (!enableCleanup_) return;
  // other cleanup?
} // void HcalMonitorClient::cleanup(void)


void HcalMonitorClient::writeHtml()
{
  if (debug_>0) std::cout << "Preparing HcalMonitorClient html output ..." << std::endl;
  

  // global ROOT style
  gStyle->Reset("Default");
  gStyle->SetCanvasColor(0);
  gStyle->SetPadColor(0);
  gStyle->SetFillColor(0);
  gStyle->SetTitleFillColor(10);
  //  gStyle->SetOptStat(0);
  gStyle->SetOptStat("ouemr");
  gStyle->SetPalette(1);

  char tmp[20];

  if(run_!=-1) sprintf(tmp, "DQM_%s_R%09d_%i", prefixME_.substr(0,prefixME_.size()-1).c_str(),run_,htmlcounter_);
  else sprintf(tmp, "DQM_%s_R%09d_%i", prefixME_.substr(0,prefixME_.size()-1).c_str(),0,htmlcounter_);
  std::string htmlDir = baseHtmlDir_ + "/" + tmp + "/";
  system(("/bin/mkdir -p " + htmlDir).c_str());

  ++htmlcounter_;

  ofstream htmlFile;
  htmlFile.open((htmlDir + "index.html").c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << std::endl;
  htmlFile << "<html>  " << std::endl;
  htmlFile << "<head>  " << std::endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << std::endl;
  htmlFile << " http-equiv=\"content-type\">  " << std::endl;
  htmlFile << "  <title>Hcal Data Quality Monitor</title> " << std::endl;
  htmlFile << "</head>  " << std::endl;
  htmlFile << "<body>  " << std::endl;
  htmlFile << "<br>  " << std::endl;
 htmlFile << "<center><h1>Hcal Data Quality Monitor</h1></center>" << std::endl;
  htmlFile << "<h2>Run Number:&nbsp;&nbsp;&nbsp;" << std::endl;
  htmlFile << "<span style=\"color: rgb(0, 0, 153);\">" << run_ <<"</span></h2> " << std::endl;
  htmlFile << "<h2>Events processed:&nbsp;&nbsp;&nbsp;" << std::endl;
  htmlFile << "<span style=\"color: rgb(0, 0, 153);\">" << ievt_ <<"</span></h2> " << std::endl;
  htmlFile << "<hr>" << std::endl;
  htmlFile << "<ul>" << std::endl;

  for (unsigned int i=0;i<clients_.size();++i)
    {
      if (clients_[i]->validHtmlOutput()==true)
	{
	  clients_[i]->htmlOutput(htmlDir);
	  // Always print this out?  Or only when validHtmlOutput is true? 
	  htmlFile << "<table border=0 WIDTH=\"50%\"><tr>" << std::endl;
	  htmlFile << "<td WIDTH=\"35%\"><a href=\"" << clients_[i]->name_ << ".html"<<"\">"<<clients_[i]->name_<<"</a></td>" << std::endl;
	  if(clients_[i]->hasErrors_Temp()) htmlFile << "<td bgcolor=red align=center>This monitor task has errors.</td>" << std::endl;
	  else if(clients_[i]->hasWarnings_Temp()) htmlFile << "<td bgcolor=yellow align=center>This monitor task has warnings.</td>" << std::endl;
	  else if(clients_[i]->hasOther_Temp()) htmlFile << "<td bgcolor=aqua align=center>This monitor task has messages.</td>" << std::endl;
	  else htmlFile << "<td bgcolor=lime align=center>This monitor task has no problems</td>" << std::endl;
	  htmlFile << "</tr></table>" << std::endl;
	}
    }

  // Add call to reportSummary html output
  if (summaryClient_)
    {
      summaryClient_->htmlOutput(htmlDir);
      htmlFile << "<table border=0 WIDTH=\"50%\"><tr>" << std::endl;
      htmlFile << "<td WIDTH=\"35%\"><a href=\"" << summaryClient_->name_ << ".html"<<"\">"<<summaryClient_->name_<<"</a></td>" << std::endl;
      if(summaryClient_->hasErrors_Temp()) htmlFile << "<td bgcolor=red align=center>This monitor task has errors.</td>" << std::endl;
      else if(summaryClient_->hasWarnings_Temp()) htmlFile << "<td bgcolor=yellow align=center>This monitor task has warnings.</td>" << std::endl;
      else if(summaryClient_->hasOther_Temp()) htmlFile << "<td bgcolor=aqua align=center>This monitor task has messages.</td>" << std::endl;
      else htmlFile << "<td bgcolor=lime align=center>This monitor task has no problems</td>" << std::endl;
      htmlFile << "</tr></table>" << std::endl;
    }

  htmlFile << "</ul>" << std::endl;

  // html page footer
  htmlFile << "</body> " << std::endl;
  htmlFile << "</html> " << std::endl;

  htmlFile.close();
  if (debug_>0) std::cout << "HcalMonitorClient html output done..." << std::endl;
  
} // void HcalMonitorClient::writeHtml()

void HcalMonitorClient::writeChannelStatus()
{
  if (databasedir_.size()==0) return;
  if (debug_>0) std::cout <<"<HcalMonitorClient::writeDBfile>  Writing file for database"<<std::endl;

  std::map<HcalDetId, unsigned int> myquality; //map of quality flags as reported by each client
  // Get status from all channels (we need to store all channels in case a bad channel suddenly becomes good)
  for (std::vector<HcalBaseDQClient*>::size_type i=0;i<clients_.size();++i)
    clients_[i]->updateChannelStatus(myquality);

  if (debug_>0) std::cout <<"<HcalMonitorClient::writeChannelStatus()>  myquality size = "<<myquality.size()<<std::endl;

  std::vector<DetId> mydetids = chanquality_->getAllChannels();
  HcalChannelQuality* newChanQual = new HcalChannelQuality(chanquality_->topo());

  for (unsigned int i=0;i<mydetids.size();++i)
    {
      if (mydetids[i].det()!=DetId::Hcal) continue; // not hcal
      
      HcalDetId id=mydetids[i];
      // get original channel status item
      const HcalChannelStatus* origstatus=chanquality_->getValues(mydetids[i]);
      // make copy of status
      HcalChannelStatus* mystatus=new HcalChannelStatus(origstatus->rawId(),origstatus->getValue());
      // loop over myquality flags
      if (myquality.find(id)!=myquality.end())
	{
	  
	  // check dead cells
	  if ((myquality[id]>>HcalChannelStatus::HcalCellDead)&0x1)
	    mystatus->setBit(HcalChannelStatus::HcalCellDead);
	  else
	    mystatus->unsetBit(HcalChannelStatus::HcalCellDead);
	  // check hot cells
	  if ((myquality[id]>>HcalChannelStatus::HcalCellHot)&0x1)
	    mystatus->setBit(HcalChannelStatus::HcalCellHot);
	  else
	    mystatus->unsetBit(HcalChannelStatus::HcalCellHot);
	} // if (myquality.find_...)
      newChanQual->addValues(*mystatus);
    } // for (unsigned int i=0;...)
  
  //Now dump out to text file
  std::ostringstream file;
  databasedir_=databasedir_+"/"; // add extra slash, just in case
  //file <<databasedir_<<"HcalDQMstatus_"<<run_<<".txt";
  file <<databasedir_<<"HcalDQMstatus.txt";
  std::ofstream outStream(file.str().c_str());
  outStream<<"###  Run # "<<run_<<std::endl;
  HcalDbASCIIIO::dumpObject (outStream, (*newChanQual));
  return;
} // void HcalMonitorClient::writeChannelStatus()


void HcalMonitorClient::PlotPedestalValues(const HcalDbService& cond)
{

  double ADC_ped=0;
  double ADC_width=0;
  double fC_ped=0;
  double fC_width=0;
  double temp_ADC=0;
  double temp_fC=0;

  int ieta=-9999;
  int iphi=-9999;
  HcalCalibrations calibs_;

  ADC_PedestalFromDBByDepth->Reset();
  ADC_WidthFromDBByDepth->Reset();
  fC_PedestalFromDBByDepth->Reset();
  fC_WidthFromDBByDepth->Reset();


  for (int subdet=1; subdet<=4;++subdet)
    {
      for (int depth=0;depth<4;++depth)
	{
	  int etabins= ADC_PedestalFromDBByDepth->depth[depth]->getNbinsX();
	  int phibins = ADC_PedestalFromDBByDepth->depth[depth]->getNbinsY();
	  for (int eta=0;eta<etabins;++eta)
	    {
	      ieta=CalcIeta(subdet,eta,depth+1);
	      if (ieta==-9999) continue;
	      for (int phi=0;phi<phibins;++phi)
		{
		  iphi=phi+1;
		  if (!validDetId((HcalSubdetector)(subdet), ieta, iphi, depth+1)) continue;
		  HcalDetId detid((HcalSubdetector)(subdet), ieta, iphi, depth+1);
		  ADC_ped=0;
		  ADC_width=0;
		  fC_ped=0;
		  fC_width=0;
		  calibs_= cond.getHcalCalibrations(detid);  
		  const HcalPedestalWidth* pedw = cond.getPedestalWidth(detid);
		  const HcalQIECoder* channelCoder_ = cond.getHcalCoder(detid);
		  const HcalQIEShape* shape_ = cond.getHcalShape(channelCoder_); 

		  // Loop over capIDs
		  for (unsigned int capid=0;capid<4;++capid)
		    {
		      // Still need to determine how to convert widths to ADC or fC
		      // calibs_.pedestal value is always in fC, according to Radek
		      temp_fC = calibs_.pedestal(capid);
		      fC_ped+= temp_fC;
		      // convert to ADC from fC
		      temp_ADC=channelCoder_->adc(*shape_,
						  (float)calibs_.pedestal(capid),
						  capid);
		      ADC_ped+=temp_ADC;
		      // Pedestals assumed to be read out in fC
		      temp_fC=pedw->getSigma(capid,capid);
		      fC_width+=temp_fC;
		      temp_ADC=pedw->getSigma(capid,capid)*pow(1.*channelCoder_->adc(*shape_,(float)calibs_.pedestal(capid),capid)/calibs_.pedestal(capid),2);
		      ADC_width+=temp_ADC;
		    }//capid loop

		  // Pedestal values are average over four cap IDs
		  // widths are sqrt(SUM [sigma_ii^2])/4.
		  fC_ped/=4.;
		  ADC_ped/=4.;

		  // Divide width by 2, or by four?
		  // Dividing by 2 gives subtracted results closer to zero -- estimate of variance?
		  fC_width=pow(fC_width,0.5)/2.;
		  ADC_width=pow(ADC_width,0.5)/2.;

		  if (debug_>1)
		    {
		      std::cout <<"<HcalMonitorClient::PlotPedestalValues> HcalDet ID = "<<(HcalSubdetector)subdet<<": ("<<ieta<<", "<<iphi<<", "<<depth<<")"<<std::endl;
		      std::cout <<"\tADC pedestal = "<<ADC_ped<<" +/- "<<ADC_width<<std::endl;
		      std::cout <<"\tfC pedestal = "<<fC_ped<<" +/- "<<fC_width<<std::endl;
		    }
		  // Shift HF by -/+1 when filling eta-phi histograms
		  int zside=0;
		  if (subdet==4)
		    {
		      if (ieta<0) zside=-1;
		      else zside=1;
		    }
		  ADC_PedestalFromDBByDepth->depth[depth]->Fill(ieta+zside,iphi,ADC_ped);
		  ADC_WidthFromDBByDepth->depth[depth]->Fill(ieta+zside, iphi, ADC_width);
		  fC_PedestalFromDBByDepth->depth[depth]->Fill(ieta+zside,iphi,fC_ped);
		  fC_WidthFromDBByDepth->depth[depth]->Fill(ieta+zside, iphi, fC_width);
		} // phi loop
	    } // eta loop
	} //depth loop

    } // subdet loop
  FillUnphysicalHEHFBins(*ADC_PedestalFromDBByDepth);
  FillUnphysicalHEHFBins(*ADC_WidthFromDBByDepth);
  FillUnphysicalHEHFBins(*fC_PedestalFromDBByDepth);
  FillUnphysicalHEHFBins(*fC_WidthFromDBByDepth);

  // Center ADC pedestal values near 3 +/- 1
  for (unsigned int i=0;i<ADC_PedestalFromDBByDepth->depth.size();++i)
  {
    ADC_PedestalFromDBByDepth->depth[i]->getTH2F()->SetMinimum(0);
    if (ADC_PedestalFromDBByDepth->depth[i]->getTH2F()->GetMaximum()<6)
      ADC_PedestalFromDBByDepth->depth[i]->getTH2F()->SetMaximum(6);
  }

  for (unsigned int i=0;i<ADC_WidthFromDBByDepth->depth.size();++i)
  {
    ADC_WidthFromDBByDepth->depth[i]->getTH2F()->SetMinimum(0);
    if (ADC_WidthFromDBByDepth->depth[i]->getTH2F()->GetMaximum()<2)
      ADC_WidthFromDBByDepth->depth[i]->getTH2F()->SetMaximum(2);
  }

}

DEFINE_FWK_MODULE(HcalMonitorClient);
