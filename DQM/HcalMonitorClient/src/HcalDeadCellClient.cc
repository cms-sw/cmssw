#include <DQM/HcalMonitorClient/interface/HcalDeadCellClient.h>
#include <DQM/HcalMonitorClient/interface/HcalClientUtils.h>
#include <DQM/HcalMonitorClient/interface/HcalHistoUtils.h>
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

HcalDeadCellClient::HcalDeadCellClient(){}


void HcalDeadCellClient::init(const ParameterSet& ps, DQMStore* dbe,string clientName){
  //Call the base class first
  HcalBaseClient::init(ps,dbe,clientName);

  if (debug_)
    cout <<"Initializing HcalDeadCellClient from ParameterSet"<<endl;

  hbhists.type=1;
  hehists.type=2;
  hohists.type=3;
  hfhists.type=4;
  hcalhists.type=10; // sum of other histograms

  clearHists(hbhists);
  clearHists(hehists);
  clearHists(hohists);
  clearHists(hfhists);
  clearHists(hcalhists);
  
  errorFrac_=ps.getUntrackedParameter<double>("deadcellErrorFrac",0.05);

}

HcalDeadCellClient::~HcalDeadCellClient(){
  this->cleanup();
}

void HcalDeadCellClient::beginJob(void){
  
  if ( debug_ ) cout << "HcalDeadCellClient: beginJob" << endl;
  
  ievt_ = 0;
  jevt_ = 0;

  this->setup();

  return;
}

void HcalDeadCellClient::beginRun(void){

  if ( debug_ ) cout << "HcalDeadCellClient: beginRun" << endl;

  jevt_ = 0;
  this->setup();
  this->resetAllME();
  return;
}

void HcalDeadCellClient::endJob(void) {

  if ( debug_ ) cout << "HcalDeadCellClient: endJob, ievt = " << ievt_ << endl;

  this->cleanup(); 
  return;
}

void HcalDeadCellClient::endRun(void) {

  if ( debug_ ) cout << "HcalDeadCellClient: endRun, jevt = " << jevt_ << endl;

  this->cleanup();  
  return;
}

void HcalDeadCellClient::setup(void) {
  
  return;
}

void HcalDeadCellClient::cleanup(void) {

  if (debug_)
    cout <<"HcalDeadCellClient::cleanup"<<endl;
  if ( cloneME_ ) 
    {
      deleteHists(hbhists);
      deleteHists(hehists);
      deleteHists(hohists);
      deleteHists(hfhists);
      deleteHists(hcalhists);
    }    

  clearHists(hbhists);
  clearHists(hehists);
  clearHists(hohists);
  clearHists(hfhists);
  clearHists(hcalhists);

  dqmReportMapErr_.clear(); dqmReportMapWarn_.clear(); dqmReportMapOther_.clear();
  dqmQtests_.clear();

  return;
}


void HcalDeadCellClient::report()
{

  if ( debug_ ) cout << "HcalDeadCellClient: report" << endl;
  
  char name[256];
  sprintf(name, "%sHcal/DeadCellMonitor/DeadCell Task Event Number",process_.c_str());
  MonitorElement* me = 0;
  if(dbe_) me = dbe_->get(name);
  if ( me ) 
    {
      string s = me->valueString();
      ievt_ = -1;
      sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &ievt_);
      if ( debug_ ) cout << "Found '" << name << "'" << endl;
  
      sprintf(name,"%sHcal/DeadCellMonitor/CheckNevents",process_.c_str());
      me = dbe_->get(name);
      s=me->valueString();
      checkNevents_ = -1;
      sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &checkNevents_);
      if ( debug_) cout <<"checkNevents_ = "<<checkNevents_<<endl;
    }

  getHistograms();
  
  return;
}

void HcalDeadCellClient::analyze(void){

  jevt_++;
  int updates = 0;
  if ( updates % 10 == 0 ) {
    if ( debug_ ) cout << "HcalDeadCellClient: " << updates << " updates" << endl;
  }
  
  return;
}

void HcalDeadCellClient::clearHists(DeadCellHists& hist)
{
  if(debug_) cout <<"Clearing HcalDeadCell histograms for HCAL type: "<<hist.type<<endl;

  hist.deadADC_map=0;
  hist.deadADC_eta=0;
  hist.ADCdist=0;
  hist.NADA_cool_cell_map=0;
  hist.coolcell_below_pedestal=0;
  hist.above_pedestal=0;
  
  hist.deadADC_map_depth.clear();
  hist.deadcapADC_map.clear();
  hist.NADA_cool_cell_map_depth.clear();
  hist.coolcell_below_pedestal_depth.clear();
  hist.above_pedestal_depth.clear();

  return;
}

void HcalDeadCellClient::deleteHists(DeadCellHists& hist)
{
  if (hist.deadADC_map) delete hist.deadADC_map;
  if (hist.deadADC_eta) delete hist.deadADC_eta;
  if (hist.ADCdist) delete hist.ADCdist;
  if (hist.NADA_cool_cell_map) delete hist.NADA_cool_cell_map;
  if (hist.coolcell_below_pedestal) delete hist.coolcell_below_pedestal;
  if (hist.above_pedestal) delete hist.above_pedestal;

  hist.deadADC_map_depth.clear();
  hist.deadcapADC_map.clear();
  hist.NADA_cool_cell_map_depth.clear();
  hist.coolcell_below_pedestal_depth.clear();
  hist.above_pedestal_depth.clear();

  return;
}



void HcalDeadCellClient::getHistograms(){
  if(!dbe_) return;


  if(subDetsOn_[0]) getSubDetHistograms(hbhists);
  if(subDetsOn_[1]) getSubDetHistograms(hehists);
  if(subDetsOn_[2]) getSubDetHistograms(hohists);
  if(subDetsOn_[3]) getSubDetHistograms(hfhists);
  getSubDetHistograms(hcalhists);
  return;
}

void HcalDeadCellClient::getSubDetHistograms(DeadCellHists& hist)
{
  if (debug_)
    cout <<"HcalDeadCellClient:: Getting subdetector histograms for subdetector "<<hist.type<<endl;
  
  char name[150];
  string type;
  if(hist.type==1) type= "HB";
  else if(hist.type==2) type = "HE"; 
  else if(hist.type==3) type = "HO"; 
  else if(hist.type==4) type = "HF"; 
  else if(hist.type==10) type = "HCAL";
  else {
    if (debug_)cout <<"<HcalDeadCellClient::getSubDetHistograms> Error:  unrecognized histogram type: "<<hist.type<<endl;
    return;
  }

  // Histograms related to ADC-counting method of finding dead cells
  sprintf(name,"DeadCellMonitor/%s/%sProblemDeadCells",type.c_str(),type.c_str());
  if (debug_) cout <<"Histogram name = "<<name<<endl;
  hist.problemDeadCells=getAnyHisto(new TH2F(),name, process_, dbe_, 
				    debug_, cloneME_);
  sprintf(name,"DeadCellMonitor/%s/%s_OccupancyMap_deadADC",type.c_str(),type.c_str());
  if (debug_) cout <<"Histogram name = "<<name<<endl;
  hist.deadADC_map = getAnyHisto(new TH2F(),name, 
				    process_, dbe_,debug_,cloneME_); 

  sprintf(name,"DeadCellMonitor/%s/expertPlots/%s_deadADCEta",type.c_str(),type.c_str());
  if (debug_) cout <<"Histogram name = "<<name<<endl;
  hist.deadADC_eta = getAnyHisto(new TH1F(),name,
				 process_, dbe_, debug_, cloneME_);
 
  sprintf(name,"DeadCellMonitor/%s/expertPlots/%s_ADCdist",type.c_str(),type.c_str());
  if (debug_) cout <<"Histogram name = "<<name<<endl;
  hist.ADCdist = getAnyHisto(new TH1F(), name, process_, dbe_,debug_,cloneME_);  
  
  for (int d=0;d<4;++d)
    {
      sprintf(name,"DeadCellMonitor/%s/depth%i/%s_DeadADCmap_Depth%i",type.c_str(),d+1,type.c_str(),d+1);
      if (debug_) cout <<"Histogram name = "<<name<<endl;
      hist.deadADC_map_depth.push_back(getAnyHisto(new TH2F(),name,
						   process_, dbe_, 
						   debug_, cloneME_));
    }
  for (int capid=0;capid<4;++capid)
    {
      sprintf(name,"DeadCellMonitor/%s/expertPlots/%s_DeadCap%i",type.c_str(),type.c_str(),capid);
      if (debug_) cout <<"Histogram name = "<<name<<endl;
      hist.deadcapADC_map.push_back(getAnyHisto(new TH2F(),name,
						process_, dbe_, 
						debug_, cloneME_));
    }

  // Dead Cell routine # 2:   cell cool compared to neighbors
  sprintf(name,"DeadCellMonitor/%s/%s_OccupancyMap_NADA_CoolCell",type.c_str(),type.c_str());
  if (debug_) cout <<"Histogram name = "<<name<<endl;
  hist.NADA_cool_cell_map=getAnyHisto(new TH2F(), name, process_, dbe_,debug_,cloneME_);

  for (int d=0;d<4;++d)
    {
      sprintf(name,"DeadCellMonitor/%s/depth%i/%s_NADACoolCell_Depth%i",type.c_str(),d+1,type.c_str(),d+1);
      if (debug_) cout <<"Histogram name = "<<name<<endl;
      hist.NADA_cool_cell_map_depth.push_back(getAnyHisto(new TH2F(),name,
							  process_, dbe_, 
							  debug_, 
							  cloneME_));
    }

  // Dead Cell routine #3:  comparison to pedestal + N sigma
  sprintf(name,"DeadCellMonitor/%s/expertPlots/%s_abovePed",type.c_str(),type.c_str());
  if (debug_) cout <<"Histogram name = "<<name<<endl;
  hist.above_pedestal=getAnyHisto(new TH2F(), name, process_, dbe_,debug_,cloneME_);
  
  sprintf(name,"DeadCellMonitor/%s/%s_OccupancyMap_belowPedestal",type.c_str(),type.c_str());
  if (debug_) cout <<"Histogram name = "<<name<<endl;
  hist.coolcell_below_pedestal=getAnyHisto(new TH2F(),
				    name, process_,
				    dbe_,debug_,cloneME_);
  for (int d=0;d<4;++d)
    {
      sprintf(name,"DeadCellMonitor/%s/depth%i/%s_coolcell_below_pedestal_Depth%i",type.c_str(),d+1,type.c_str(),d+1);
      hist.coolcell_below_pedestal_depth.push_back(getAnyHisto(new TH2F(),
							       name, process_,
							       dbe_,debug_,cloneME_));

      sprintf(name,"DeadCellMonitor/%s/depth%i/%s_cell_above_pedestal_Depth%i",type.c_str(),d+1,type.c_str(),d+1);
      hist.above_pedestal_depth.push_back(getAnyHisto(new TH2F(), name,
						      process_, dbe_,
						      debug_,cloneME_));
    }
  
  // Diagnostic histograms
  sprintf(name,"DeadCellMonitor/%s/expertPlots/%s_digiCheck",type.c_str(),type.c_str());
  if (debug_) cout <<"Histogram name = "<<name<<endl;
  hist.digiCheck=getAnyHisto(new TH2F(), name, process_, dbe_,debug_,cloneME_);

  sprintf(name,"DeadCellMonitor/%s/expertPlots/%s_cellCheck",type.c_str(),type.c_str());
  if (debug_) cout <<"Histogram name = "<<name<<endl;
  hist.cellCheck=getAnyHisto(new TH2F(), name, process_, dbe_,debug_,cloneME_);
  
  for (int d=0;d<4;++d)
    { 
      sprintf(name,"DeadCellMonitor/%s/depth%i/%s_digiCheck_Depth%i",
	      type.c_str(),d+1,type.c_str(),d+1);
      if (debug_) cout <<"Histogram name = "<<name<<endl;
      hist.digiCheck_depth.push_back(getAnyHisto(new TH2F(), name, process_, 
						 dbe_,debug_,cloneME_));
      
      sprintf(name,"DeadCellMonitor/%s/depth%i/%s_cellCheck_Depth%i",
	      type.c_str(),d+1,type.c_str(),d+1);
      if (debug_) cout <<"Histogram name = "<<name<<endl;
      hist.cellCheck_depth.push_back(getAnyHisto(new TH2F(), name, process_, 
						 dbe_,debug_,cloneME_));
    } // for (int d=0;d<4;++d)


  return;
} // get SubDetHistograms

  
void HcalDeadCellClient::getSubDetHistogramsFromFile(DeadCellHists& hist, TFile* infile)
{
  if (debug_)
    cout <<"HcalDeadCellClient:: Getting subdetector histograms from file for subdetector "<<hist.type<<endl;
  
  char name[150];
  string type;
  if(hist.type==1) type= "HB";
  else if(hist.type==2) type = "HE"; 
  else if(hist.type==3) type = "HO"; 
  else if(hist.type==4) type = "HF"; 
  else if(hist.type==10) type = "HCAL";
  else {
    if (debug_)cout <<"<HcalDeadCellClient::getSubDetHistograms> Error:  unrecognized histogram type: "<<hist.type<<endl;
    return;
  }
  

  // Histograms related to ADC-counting method of finding dead cells
  sprintf(name,"DeadCellMonitor/%s/%s_OccupancyMap_deadADC",type.c_str(),type.c_str());
  hist.deadADC_map =  (TH2F*)infile->Get(name); 

  sprintf(name,"DeadCellMonitor/%s/expertPlots/%s_deadADCEta",type.c_str(),type.c_str());
  hist.deadADC_eta = (TH1F*)infile->Get(name);
 
  sprintf(name,"DeadCellMonitor/%s/%s_ADCdist",type.c_str(),type.c_str());
  hist.ADCdist = (TH1F*)infile->Get(name);
  
  for (int d=0;d<4;++d)
    {
      sprintf(name,"DeadCellMonitor/%s/depth%i/%s_DeadADCmap_Depth%i",type.c_str(),d+1,type.c_str(),d+1);
      hist.deadADC_map_depth.push_back((TH2F*)infile->Get(name));
    }

  for (int capid=0;capid<4;++capid)
    {
      sprintf(name,"DeadCellMonitor/%s/expertPlots/%s_DeadCap%i",type.c_str(),type.c_str(),capid);
      hist.deadcapADC_map.push_back((TH2F*)infile->Get(name));
    }

  // Dead Cell routine # 2:   cell cool compared to neighbors
  sprintf(name,"DeadCellMonitor/%s/%s_NADA_CoolCellMap",type.c_str(),type.c_str());
  hist.NADA_cool_cell_map=(TH2F*)infile->Get(name);

  for (int d=0;d<4;++d)
    {
      sprintf(name,"DeadCellMonitor/%s/depth%i/%s_NADACoolCell_Depth%i",type.c_str(),d+1,type.c_str(),d+1);
      hist.NADA_cool_cell_map_depth.push_back((TH2F*)infile->Get(name));
    }

  // Dead Cell routine #3:  comparison to pedestal + N sigma
  sprintf(name,"DeadCellMonitor/%s/%s_abovePed",type.c_str(),type.c_str());
  hist.above_pedestal=(TH2F*)infile->Get(name);
  
  sprintf(name,"DeadCellMonitor/%s/%s_CoolCell_belowPed",type.c_str(),type.c_str());
  hist.coolcell_below_pedestal=(TH2F*)infile->Get(name);

  for (int d=0;d<4;++d)
    {
      sprintf(name,"DeadCellMonitor/%s/depth%i/%s_coolcell_below_pedestal_Depth%i",type.c_str(),d+1,type.c_str(),d+1);
      hist.coolcell_below_pedestal_depth.push_back((TH2F*)infile->Get(name));
      sprintf(name,"DeadCellMonitor/%s/depth%i/%s_cell_above_pedestal_Depth%i",type.c_str(),d+1,type.c_str(),d+1);
      hist.above_pedestal_depth.push_back((TH2F*)infile->Get(name));
    }
  
  // Diagnostic histograms
  sprintf(name,"DeadCellMonitor/%s/expertPlots/%s_digiCheck",type.c_str(),type.c_str());
  hist.digiCheck=(TH2F*)infile->Get(name);
  sprintf(name,"DeadCellMonitor/%s/expertPlots/%s_cellCheck",type.c_str(),type.c_str());
  hist.cellCheck=(TH2F*)infile->Get(name);
  for (int d=0;d<4;++d)
    {
      sprintf(name,"DeadCellMonitor/%s/depth%i/%s_digiCheck_Depth%i",type.c_str(),d+1,
	      type.c_str(),d+1);
      hist.digiCheck_depth.push_back((TH2F*)infile->Get(name));
      sprintf(name,"DeadCellMonitor/%s/depth%i/%s_cellCheck_Depth%i",
	      type.c_str(),d+1,type.c_str(),d+1);
      hist.cellCheck_depth.push_back((TH2F*)infile->Get(name));
    }
  return;
} // void HcalDeadCellClient::getSubDetHistogramsFromFile


void HcalDeadCellClient::resetSubDetHistograms(DeadCellHists& hist)
{
  if (debug_)
    cout <<"HcalDeadCellClient::Resetting subdetector histograms for subdetector "<<hist.type<<endl;
  
  char name[150];
  string type;
  if(hist.type==1) type= "HB";
  else if(hist.type==2) type = "HE"; 
  else if(hist.type==3) type = "HO"; 
  else if(hist.type==4) type = "HF"; 
  else if(hist.type==10) type = "HCAL";
  else {
    if (debug_)cout <<"<HcalDeadCellClient::resetSubDetHistograms> Error:  unrecognized histogram type: "<<hist.type<<endl;
    return;
  }
  cout <<"<HcalDeadCellClient> Reset histograms for subdet"<<type.c_str()<<endl;
  //printf("Reset subdet %s\n",type.c_str());

  sprintf(name,"DeadCellMonitor/%s/%sProblemDeadCells",type.c_str(),type.c_str());
  resetME(name,dbe_);

  // Dead Cell Routine #1:  ADC counts
  sprintf(name,"DeadCellMonitor/%s/%s_OccupancyMap_deadADC",type.c_str(),type.c_str());
  resetME(name,dbe_);

  sprintf(name,"DeadCellMonitor/%s/expertPlots/%s_deadADCEta",type.c_str(),type.c_str());
  resetME(name,dbe_);
  sprintf(name,"DeadCellMonitor/%s/expertPlots/%s_ADCdist",type.c_str(),type.c_str());
  resetME(name,dbe_);

  // Dead Cell Routine #2:  neighboring cells
  sprintf(name,"DeadCellMonitor/%s/%s_OccupancyMap_NADA_CoolCell",type.c_str(),type.c_str());
  resetME(name,dbe_);

  // Dead Cell Routine # 3:  below pedestal
  sprintf(name,"DeadCellMonitor/%s/%s_abovePed",type.c_str(),type.c_str());
  resetME(name,dbe_);
  sprintf(name,"DeadCellMonitor/%s/%s_CoolCell_belowPed",type.c_str(),type.c_str());
  resetME(name,dbe_);

  // Diagnostic plots
  sprintf(name,"DeadCellMonitor/%s/expertPlots/%s_digiCheck",type.c_str(),type.c_str());
  resetME(name,dbe_);
  sprintf(name,"DeadCellMonitor/%s/expertPlots/%s_cellCheck",type.c_str(),type.c_str());
  resetME(name,dbe_);
  
  // Loop over individual depths, capids
  for (int i=0;i<4;i++)
    {
      sprintf(name,"DeadCellMonitor/%s/expertPlots/%s_DeadCap%i",type.c_str(),type.c_str(),i);
      resetME(name,dbe_); 
      sprintf(name,"DeadCellMonitor/%s/depth%i/%s_DeadADCmap_Depth%i",type.c_str(),i+1,type.c_str(),i+1);
      resetME(name,dbe_);
      sprintf(name,"DeadCellMonitor/%s/depth%i/%s_NADACoolCell_Depth%i",type.c_str(),i+1,type.c_str(),i+1);
      resetME(name,dbe_);
      sprintf(name,"DeadCellMonitor/%s/depth%i/%s_cellCheck_Depth%i",type.c_str(),i+1,type.c_str(),i+1);
      resetME(name,dbe_);
      sprintf(name,"DeadCellMonitor/%s/depth%i/%s_cell_above_pedestal_Depth%i",type.c_str(),i+1,type.c_str(),i+1);
      resetME(name,dbe_);
      sprintf(name,"DeadCellMonitor/%s/depth%i/%s_coolcell_below_pedestal_Depth%i",type.c_str(),i+1,type.c_str(),i+1);
      resetME(name,dbe_);
      sprintf(name,"DeadCellMonitor/%s/depth%i/%s_digiCheck_Depth%i",type.c_str(),i+1,type.c_str(),i+1);
      resetME(name,dbe_);
    }

  return;
} // void HcalDeadCellClient::resetSubDetHistograms


void HcalDeadCellClient::resetAllME()
{
  if(!dbe_) return;
  if (subDetsOn_[0]) resetSubDetHistograms(hbhists);
  if (subDetsOn_[1]) resetSubDetHistograms(hehists);
  if (subDetsOn_[2]) resetSubDetHistograms(hohists);
  if (subDetsOn_[3]) resetSubDetHistograms(hfhists);
  resetSubDetHistograms(hcalhists);
  return;
} //void HcalDeadCellClient::resetAllME()


void HcalDeadCellClient::htmlOutput(int runNo, string htmlDir, string htmlName)
{
  if (debug_)
    cout << "Preparing HcalDeadCellClient html output ..." << endl;
  string client = "DeadCellMonitor";
  htmlErrors(runNo,htmlDir,client,process_,dbe_,dqmReportMapErr_,dqmReportMapWarn_,dqmReportMapOther_);
  
  //ofstream htmlFile;
  htmlFile.open((htmlDir + htmlName).c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlFile << "<html>  " << endl;
  htmlFile << "<head>  " << endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlFile << " http-equiv=\"content-type\">  " << endl;
  htmlFile << "  <title>Monitor: Hcal DeadCell Task output</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlFile << "<body>  " << endl;
  htmlFile << "<br>  " << endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << runNo << "</span></h2>" << endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">Hcal DeadCells</span></h2> " << endl;
  htmlFile << "<h2>Events processed:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;

  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << ievt_ << "</span></h2>" << endl;
  htmlFile << "<hr>" << endl;
  htmlFile << "<table  width=100% border=1><tr>" << endl;
  if(hasErrors())htmlFile << "<td bgcolor=red><a href=\"DeadCellMonitorErrors.html\">Errors in this task</a></td>" << endl;
  else htmlFile << "<td bgcolor=lime>No Errors</td>" << endl;
  if(hasWarnings()) htmlFile << "<td bgcolor=yellow><a href=\"DeadCellMonitorWarnings.html\">Warnings in this task</a></td>" << endl;
  else htmlFile << "<td bgcolor=lime>No Warnings</td>" << endl;
  if(hasOther()) htmlFile << "<td bgcolor=aqua><a href=\"DeadCellMonitorMessages.html\">Messages in this task</a></td>" << endl;
  else htmlFile << "<td bgcolor=lime>No Messages</td>" << endl;
  htmlFile << "</tr></table>" << endl;
  htmlFile << "<hr>" << endl;

  htmlFile<<"<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\"> " << endl;
  htmlFile << "<h3><tr><td><a href=\"index.html\"> Main DQM Page </a> </td>"<<endl;
  htmlFile << "<h3><tr><td>Detailed (expert-level) Plots:  </td>";
  htmlFile << "<td><a href=\"HcalDeadCellClient_HCAL_Plots.html\">HCAL Plots </a>  </td>" << endl;
  if(subDetsOn_[0]) htmlFile << "<td><a href=\"HcalDeadCellClient_HB_Plots.html\">HB Plots </a></br>  </td>" << endl;  
  if(subDetsOn_[1]) htmlFile << "<td><a href=\"HcalDeadCellClient_HE_Plots.html\">HE Plots </a></br>  </td>" << endl;
  if(subDetsOn_[2]) htmlFile << "<td><a href=\"HcalDeadCellClient_HO_Plots.html\">HO Plots </a></br>  </td>" << endl;
  if(subDetsOn_[3]) htmlFile << "<td><a href=\"HcalDeadCellClient_HF_Plots.html\">HF Plots </a></br></td>" << endl;
  htmlFile << "</h3></tr></table>" << endl;
  htmlFile << "<hr>" << endl;

  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\"> " << endl;
  //htmlFile << "<td>&nbsp;&nbsp;&nbsp;<h3>Global Histograms</h3></td></tr>" << endl;
  //htmlFile << "</table>" << endl;

  htmlSubDetOutput(hcalhists,runNo,htmlDir,htmlName);
  htmlSubDetOutput(hbhists,runNo,htmlDir,htmlName);
  htmlSubDetOutput(hehists,runNo,htmlDir,htmlName);
  htmlSubDetOutput(hohists,runNo,htmlDir,htmlName);
  htmlSubDetOutput(hfhists,runNo,htmlDir,htmlName);
  
  htmlADCSubDetOutput(hcalhists,runNo,htmlDir,htmlName);
  htmlADCSubDetOutput(hbhists,runNo,htmlDir,htmlName);
  htmlADCSubDetOutput(hehists,runNo,htmlDir,htmlName);
  htmlADCSubDetOutput(hohists,runNo,htmlDir,htmlName);
  htmlADCSubDetOutput(hfhists,runNo,htmlDir,htmlName);

  htmlBelowPedSubDetOutput(hcalhists,runNo,htmlDir,htmlName);
  htmlBelowPedSubDetOutput(hbhists,runNo,htmlDir,htmlName);
  htmlBelowPedSubDetOutput(hehists,runNo,htmlDir,htmlName);
  htmlBelowPedSubDetOutput(hohists,runNo,htmlDir,htmlName);
  htmlBelowPedSubDetOutput(hfhists,runNo,htmlDir,htmlName);



  htmlFile << "<br>" << endl;

  htmlFile << "<td align=\"center\">&nbsp;&nbsp;&nbsp;<h3>Cells matching dead conditions in  at least "<<(int)(errorFrac_*100)<<"% of events</h3></td>"<<endl;
  htmlFile << "</tr>"<<endl;

  htmlFile << "<tr align=\"center\">" << endl;

  
  hcalhists.problemDeadCells->Scale(1./ievt_);
  hcalhists.problemDeadCells->SetMinimum(errorFrac_);
  htmlAnyHisto(runNo,hcalhists.problemDeadCells,"iEta","iPhi", 92, htmlFile,htmlDir);
  htmlFile<<"</tr>"<<endl;

  htmlFile << "<tr align=\"left\">" << endl;
  htmlFile <<"<tr><td>This histograms shows cells that satisfy at least one dead cell condition in at least "<<(int)(errorFrac_*100)<<"% of events.  A cell is considered dead if its ADC count = 0 for an event or if it reads below pedestal for "<<checkNevents_<<" consecutive events.  Detailed plots for each type of dead cell are given in the links below."<<endl;
  
 
  htmlFile << "</tr></table><br>" << endl;

  // Add links here
 htmlFile <<"<table width = 75% align=\"center\"><tr align=\"center\">" <<endl;
 htmlFile << "<td><a href=\"HcalDeadCellClient_ADC_HCAL_Plots.html\">Dead ADC plots </a>  </td>" << endl;
htmlFile << "<td><a href=\"HcalDeadCellClient_BelowPed_HCAL_Plots.html\">Below-Pedestal plots </a>  </td>" << endl;

 
 htmlFile <<"</tr></table BORDER = \"3\" CELLPADDING = \"25\"><br>"<<endl;
 htmlFile <<"<hr>"<<endl;

  htmlFile <<"<table width=75%align = \"center\"><tr align=\"center\">" <<endl;
  htmlFile <<"<td>  List of Bad Cells</td><td align=\"center\"> Fraction of Events in which cells are bad</td></tr>"<<endl;

  int etabins = hcalhists.problemDeadCells->GetNbinsX();
  int phibins = hcalhists.problemDeadCells->GetNbinsY();
  float etaMin=hcalhists.problemDeadCells->GetXaxis()->GetXmin();
  float phiMin=hcalhists.problemDeadCells->GetYaxis()->GetXmin();
    

  int eta,phi;
  for (int ieta=1;ieta<=etabins;++ieta)
    {
      for (int iphi=1; iphi<=phibins;++iphi)
	{
	  eta=ieta+int(etaMin)-1;
	  phi=iphi+int(phiMin)-1;
	  /*
	  if (hcalhists.problemDeadCells->GetBinContent(ieta,iphi)>=errorFrac_)
	    cout<<" HCAL ("<<eta<<", "<<phi<<")  "<<hbhists.problemDeadCells->GetBinContent(ieta,iphi)<<""<<endl;
	  */
	  if (hbhists.problemDeadCells->GetBinContent(ieta,iphi)>=errorFrac_*ievt_)
	    {
	      htmlFile<<"<td align=\"center\"> HB ("<<eta<<", "<<phi<<") </td><td align=\"center\"> "<<hbhists.problemDeadCells->GetBinContent(ieta,iphi)/ievt_<<"</td></tr>"<<endl;
	    }
	  if (hehists.problemDeadCells->GetBinContent(ieta,iphi)>=errorFrac_*ievt_)
	    {
	      htmlFile<<"<td align=\"center\"> HE ("<<eta<<", "<<phi<<") </td><td align=\"center\"> "<<hehists.problemDeadCells->GetBinContent(ieta,iphi)/ievt_<<"</td></tr>"<<endl;
	    }
	  if (hohists.problemDeadCells->GetBinContent(ieta,iphi)>=errorFrac_*ievt_)
	    {
	      htmlFile<<"<td align=\"center\"> HO ("<<eta<<", "<<phi<<") </td><td align=\"center\"> "<<hohists.problemDeadCells->GetBinContent(ieta,iphi)/ievt_<<"</td></tr>"<<endl;
	    }
	  if (hfhists.problemDeadCells->GetBinContent(ieta,iphi)>=errorFrac_*ievt_)
	    {
	      htmlFile<<"<td align=\"center\"> HF ("<<eta<<", "<<phi<<") </td><td align=\"center\"> "<<hfhists.problemDeadCells->GetBinContent(ieta,iphi)/ievt_<<"</td></tr>"<<endl;
	    }

	} // for (int iphi =1...

    }// for (int ieta=1...


  htmlFile << "</table>" <<endl;
  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;
  htmlFile.close();

  return;
} //void HcalDeadCellClient::htmlOutput()



void HcalDeadCellClient::htmlADCSubDetOutput(DeadCellHists& hist, int runNo,
					     string htmlDir,
					     string htmlName)
{
if (debug_) cout <<"HcalDeadCellClient::Creating ADC html output for subdetector "<<hist.type<<endl;
  if(hist.type<5 && !subDetsOn_[hist.type-1]) return;
  
  string type;
  if(hist.type==1) type= "HB";
  else if(hist.type==2) type = "HE"; 
  else if(hist.type==3) type = "HO"; 
  else if(hist.type==4) type = "HF"; 
  else if(hist.type==10) type = "HCAL";
  else {
    if (debug_)cout <<"<HcalDeadCellClient::htmlADCSubDetOutput> Error:  unrecognized histogram type: "<<hist.type<<endl;
    return;
  }
  ofstream htmlSubFile;
  htmlSubFile.open((htmlDir + "HcalDeadCellClient_ADC_"+type+"_Plots.html").c_str());
  // html page header
  htmlSubFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlSubFile << "<html>  " << endl;
  htmlSubFile << "<head>  " << endl;
  htmlSubFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlSubFile << " http-equiv=\"content-type\">  " << endl;
  htmlSubFile << "  <title>Monitor: "<<type<<" ADC Dead Cell Detailed Plots</title> " << endl;
  htmlSubFile << "</head>  " << endl;
  htmlSubFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlSubFile << "<body>  " << endl;
  htmlSubFile << "<br>  " << endl;
  htmlSubFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlSubFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlSubFile << " style=\"color: rgb(0, 0, 153);\">" << runNo << "</span></h2>" << endl;
   htmlSubFile << "<h2>Events processed:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" <<   endl;
    
  htmlSubFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span " << endl;
  htmlSubFile << " style=\"color: rgb(0, 0, 153);\">" << ievt_ << "</span></h2>" << endl;
  htmlSubFile << "<hr>" << endl;
  htmlSubFile<<"<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlSubFile << "cellpadding=\"10\"> " << endl;
  htmlSubFile << "<h3><tr><td><a href=\"index.html\"> Main DQM Page </a> </td>"<<endl;
  htmlSubFile << "<h3><td><a href=\"HcalDeadCellClient.html\"> Main Dead Cell Page </a> </td>"<<endl;
  htmlSubFile << "<td><a href=\"HcalDeadCellClient_BelowPed_HCAL_Plots.html\"> Below-Pedestal Plots </a> </td></tr>"<<endl;
  htmlSubFile << "<h3><tr><td>Dead ADC plots:  </td>";
  htmlSubFile << "<td><a href=\"HcalDeadCellClient_ADC_HCAL_Plots.html\">HCAL Plots </a>  </td>" << endl;
  if(subDetsOn_[0]) htmlSubFile << "<td><a href=\"HcalDeadCellClient_ADC_HB_Plots.html\">HB Plots </a></br>  </td>" << endl;  
  if(subDetsOn_[1]) htmlSubFile << "<td><a href=\"HcalDeadCellClient_ADC_HE_Plots.html\">HE Plots </a></br>  </td>" << endl;
  if(subDetsOn_[2]) htmlSubFile << "<td><a href=\"HcalDeadCellClient_ADC_HO_Plots.html\">HO Plots </a></br>  </td>" << endl;
  if(subDetsOn_[3]) htmlSubFile << "<td><a href=\"HcalDeadCellClient_ADC_HF_Plots.html\">HF Plots </a></br></td>" << endl;
  htmlSubFile << "</h3></tr></table>" << endl;
  htmlSubFile << "<hr>" << endl;
    
  htmlSubFile << "<h2><strong>"<<type<<" Dead ADC Histograms by Depth</strong></h2>" << endl;
  htmlSubFile << "<h3>" << endl;
    
  htmlSubFile << "<table  width=100% border=1><tr>" << endl;
  htmlSubFile << "<tr align=\"left\">" << endl;	

  // Depth histograms
  for (int i=0;i<4;++i)
    {
      if (i%2==0)
	htmlSubFile << "<tr align=\"left\">" << endl;	
      if (hist.deadADC_map_depth[i]->GetMaximum()>0)
	htmlAnyHisto(runNo,hist.deadADC_map_depth[i],"iEta","iPhi", 92, htmlSubFile,htmlDir);
      else
	htmlSubFile<<"<td align=\"center\"><br><br>     No dead ADC cells for "<<type<<" Depth "<<i+1<<"     <br><br></td>"<<endl;
      if (i%2==1)
	htmlSubFile << "</tr>" << endl;
    }
  htmlSubFile <<"</table><br>"<<endl;
  htmlSubFile << "<h2><strong>"<<type<<" Expert-level ADC plots</strong></h2>" << endl;
  htmlSubFile << "<table  width=100% border=1><tr>" << endl;
  htmlSubFile << "<tr align=\"left\">" << endl;	
  htmlAnyHisto(runNo,hist.ADCdist,"ADC counts","#", 92, htmlSubFile,htmlDir);
  htmlAnyHisto(runNo,hist.deadADC_eta,"i#eta","ADC count < minimum", 92, htmlSubFile,htmlDir);
  htmlSubFile << "</tr></table>" << endl;
    
  // html page footer
  htmlSubFile << "</body> " << endl;
  htmlSubFile << "</html> " << endl;
    
  htmlSubFile.close();
  return;
} // void HcalDeadCellClient::htmlADCSubDetOutput(...)



void HcalDeadCellClient::htmlBelowPedSubDetOutput(DeadCellHists& hist, int runNo,
						  string htmlDir,
						  string htmlName)
{
if (debug_) cout <<"HcalDeadCellClient::Creating \"Below Pedestal\" html output for subdetector "<<hist.type<<endl;
  if(hist.type<5 && !subDetsOn_[hist.type-1]) return;
  
  string type;
  if(hist.type==1) type= "HB";
  else if(hist.type==2) type = "HE"; 
  else if(hist.type==3) type = "HO"; 
  else if(hist.type==4) type = "HF"; 
  else if(hist.type==10) type = "HCAL";
  else {
    if (debug_)cout <<"<HcalDeadCellClient::htmlBelowPedDetOutput> Error:  unrecognized histogram type: "<<hist.type<<endl;
    return;
  }
  ofstream htmlSubFile;
  htmlSubFile.open((htmlDir + "HcalDeadCellClient_BelowPed_"+type+"_Plots.html").c_str());
  // html page header
  htmlSubFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlSubFile << "<html>  " << endl;
  htmlSubFile << "<head>  " << endl;
  htmlSubFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlSubFile << " http-equiv=\"content-type\">  " << endl;
  htmlSubFile << "  <title>Monitor: "<<type<<" Below-Pedestal Dead Cell Detailed Plots</title> " << endl;
  htmlSubFile << "</head>  " << endl;
  htmlSubFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlSubFile << "<body>  " << endl;
  htmlSubFile << "<br>  " << endl;
  htmlSubFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlSubFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlSubFile << " style=\"color: rgb(0, 0, 153);\">" << runNo << "</span></h2>" << endl;
   htmlSubFile << "<h2>Events processed:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" <<   endl;
    
  htmlSubFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span " << endl;
  htmlSubFile << " style=\"color: rgb(0, 0, 153);\">" << ievt_ << "</span></h2>" << endl;
  htmlSubFile << "<hr>" << endl;
  htmlSubFile<<"<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlSubFile << "cellpadding=\"10\"> " << endl;
  htmlSubFile << "<h3><tr><td><a href=\"index.html\"> Main DQM Page </a> </td>"<<endl;
  htmlSubFile << "<h3><td><a href=\"HcalDeadCellClient.html\"> Main Dead Cell Page </a> </td>"<<endl;
  htmlSubFile << "<td><a href=\"HcalDeadCellClient_ADC_HCAL_Plots.html\"> Dead ADC Plots </a> </td></tr>"<<endl;
  htmlSubFile << "<h3><tr><td>Dead Cell (below pedestal) plots:  </td>";
  htmlSubFile << "<td><a href=\"HcalDeadCellClient_BelowPed_HCAL_Plots.html\">HCAL Plots </a>  </td>" << endl;
  if(subDetsOn_[0]) htmlSubFile << "<td><a href=\"HcalDeadCellClient_BelowPed_HB_Plots.html\">HB Plots </a></br>  </td>" << endl;  
  if(subDetsOn_[1]) htmlSubFile << "<td><a href=\"HcalDeadCellClient_BelowPed_HE_Plots.html\">HE Plots </a></br>  </td>" << endl;
  if(subDetsOn_[2]) htmlSubFile << "<td><a href=\"HcalDeadCellClient_BelowPed_HO_Plots.html\">HO Plots </a></br>  </td>" << endl;
  if(subDetsOn_[3]) htmlSubFile << "<td><a href=\"HcalDeadCellClient_BelowPed_HF_Plots.html\">HF Plots </a></br></td>" << endl;
  htmlSubFile << "</h3></tr></table>" << endl;
  htmlSubFile << "<hr>" << endl;
    
  htmlSubFile << "<h2><strong>"<<type<<" Dead Below-Pedestal Histograms by Depth</strong></h2>" << endl;
  htmlSubFile<<"(Histograms only filled if cells are below expected value for "<<checkNevents_<<" consecutive events)"<<endl;
  htmlSubFile << "<h3>" << endl;
    
  htmlSubFile << "<table  width=100% border=1><tr>" << endl;
  htmlSubFile << "<tr align=\"left\">" << endl;	

  // Depth histograms
  for (int i=0;i<4;++i)
    {
      if (i%2==0)
	htmlSubFile << "<tr align=\"left\">" << endl;	
      if (hist.coolcell_below_pedestal_depth[i]->GetMaximum()>0)
	htmlAnyHisto(runNo,hist.coolcell_below_pedestal_depth[i],"i#eta","i#phi", 92, htmlSubFile,htmlDir);
	else
	  htmlSubFile<<"<td align=\"center\"><br><br> No cells  below pedestal for "<<checkNevents_<<"<br>consecutive events in "<<type<<" Depth "<<i+1<<"     <br><br></td>"<<endl;
      if (i%2==1)
	htmlSubFile << "</tr>" << endl;
    }
  htmlSubFile <<"</table><br>"<<endl;

  htmlSubFile << "<h2><strong>"<<type<<" Expert-level Pedestal-based plots</strong></h2>" << endl;
  htmlSubFile << "<table  width=100% border=1><tr>" << endl;
  htmlSubFile << "<tr align=\"left\">" << endl;	
  htmlAnyHisto(runNo,hist.above_pedestal_depth[0],"i#eta","i#phi", 92, htmlSubFile,htmlDir);
  htmlAnyHisto(runNo,hist.above_pedestal_depth[1],"i#eta","i#phi", 92, htmlSubFile,htmlDir);
  htmlSubFile<<"</tr><tr>"<<endl;
  htmlAnyHisto(runNo,hist.above_pedestal_depth[2],"i#eta","i#phi", 92, htmlSubFile,htmlDir);
  htmlAnyHisto(runNo,hist.above_pedestal_depth[3],"i#eta","i#phi", 92, htmlSubFile,htmlDir);
  htmlSubFile << "</tr></table>" << endl;
    
  // html page footer
  htmlSubFile << "</body> " << endl;
  htmlSubFile << "</html> " << endl;
    
  htmlSubFile.close();
  return;
} // void HcalDeadCellClient::htmlBelowPedSubDetOutput(...)


void HcalDeadCellClient::htmlSubDetOutput(DeadCellHists& hist, int runNo,
					  string htmlDir,
					  string htmlName)
{
  if (debug_) cout <<"HcalDeadCellClient::Creating html output for subdetector "<<hist.type<<endl;
  if(hist.type<5 && !subDetsOn_[hist.type-1]) return;
  
  string type;
  if(hist.type==1) type= "HB";
  else if(hist.type==2) type = "HE"; 
  else if(hist.type==3) type = "HO"; 
  else if(hist.type==4) type = "HF"; 
  else if(hist.type==10) type = "HCAL";
  else {
    if (debug_)cout <<"<HcalDeadCellClient::htmlSubDetOutput> Error:  unrecognized histogram type: "<<hist.type<<endl;
    return;
  }

  ofstream htmlSubFile;
  htmlSubFile.open((htmlDir + "HcalDeadCellClient_"+type+"_Plots.html").c_str());

  // html page header
  htmlSubFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlSubFile << "<html>  " << endl;
  htmlSubFile << "<head>  " << endl;
  htmlSubFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlSubFile << " http-equiv=\"content-type\">  " << endl;
  htmlSubFile << "  <title>Monitor: Hcal "<<type<<" DeadCell Detailed Plots</title> " << endl;
  htmlSubFile << "</head>  " << endl;
  htmlSubFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlSubFile << "<body>  " << endl;
  htmlSubFile << "<br>  " << endl;
  htmlSubFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlSubFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlSubFile << " style=\"color: rgb(0, 0, 153);\">" << runNo << "</span></h2>" << endl;
  htmlSubFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlSubFile << " style=\"color: rgb(0, 0, 153);\">Hcal DeadCells</span></h2> " << endl;
  htmlSubFile << "<h2>Events processed:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" <<   endl;

  htmlSubFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span " << endl;
  htmlSubFile << " style=\"color: rgb(0, 0, 153);\">" << ievt_ << "</span></h2>" << endl;
  htmlSubFile << "<hr>" << endl;
  
  htmlSubFile << "<h2><strong>"<<type<<" Dead Cell Histograms</strong></h2>" << endl;
  htmlSubFile<<"<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlSubFile << "cellpadding=\"10\"> " << endl;
  htmlSubFile << "<h3><tr><td><a href=\"index.html\"> Main DQM Page </a> </td>"<<endl;
  htmlSubFile << "<h3><td><a href=\"HcalDeadCellClient.html\"> Main Dead Cell Page </a> </td>"<<endl;
  htmlSubFile << "<h3><tr><td>Detailed (expert-level) Plots:  </td>";
  htmlSubFile << "<td><a href=\"HcalDeadCellClient_HCAL_Plots.html\">HCAL Plots </a>  </td>" << endl;
  if(subDetsOn_[0]) htmlSubFile << "<td><a href=\"HcalDeadCellClient_HB_Plots.html\">HB Plots </a></br>  </td>" << endl;  
  if(subDetsOn_[1]) htmlSubFile << "<td><a href=\"HcalDeadCellClient_HE_Plots.html\">HE Plots </a></br>  </td>" << endl;
  if(subDetsOn_[2]) htmlSubFile << "<td><a href=\"HcalDeadCellClient_HO_Plots.html\">HO Plots </a></br>  </td>" << endl;
  if(subDetsOn_[3]) htmlSubFile << "<td><a href=\"HcalDeadCellClient_HF_Plots.html\">HF Plots </a></br></td>" << endl;
  htmlSubFile << "</h3></tr></table>" << endl;

  htmlSubFile <<"<hr>"<<endl;

  htmlSubFile << "<h3> Digi Occupancy Plots" << endl;
  htmlSubFile << "<table  width=100% border=1><tr>" << endl;
  htmlSubFile << "<tr align=\"left\">" << endl;	
  htmlAnyHisto(runNo,hist.digiCheck_depth[0],"i#eta","i#phi", 92, htmlSubFile,htmlDir);
  htmlAnyHisto(runNo,hist.digiCheck_depth[1],"i#eta","i#phi", 92, htmlSubFile,htmlDir);
  htmlSubFile << "</tr>" << endl;
  htmlAnyHisto(runNo,hist.digiCheck_depth[2],"i#eta","i#phi", 92, htmlSubFile,htmlDir);
  htmlAnyHisto(runNo,hist.digiCheck_depth[3],"i#eta","i#phi", 92, htmlSubFile,htmlDir);
  htmlSubFile << "</tr></table><br>" << endl;
  htmlSubFile <<"<hr>"<<endl;

  htmlSubFile << "<h3> RecHit Occupancy Plots" << endl;
  htmlSubFile << "<table  width=100% border=1><tr>" << endl;
  htmlSubFile << "<tr align=\"left\">" << endl;	
  htmlAnyHisto(runNo,hist.cellCheck_depth[0],"i#eta","i#phi", 92, htmlSubFile,htmlDir);
  htmlAnyHisto(runNo,hist.cellCheck_depth[1],"i#eta","i#phi", 92, htmlSubFile,htmlDir);
  htmlSubFile << "</tr>" << endl;
  htmlAnyHisto(runNo,hist.cellCheck_depth[2],"i#eta","i#phi", 92, htmlSubFile,htmlDir);
  htmlAnyHisto(runNo,hist.cellCheck_depth[3],"i#eta","i#phi", 92, htmlSubFile,htmlDir);
  htmlSubFile << "</tr></table><br>" << endl;
  htmlSubFile <<"<hr>"<<endl;

  htmlSubFile << "<h3> Plots for cells with low ADC counts" << endl;
  htmlSubFile << "<table  width=100% border=1><tr>" << endl;
  htmlSubFile << "<tr align=\"left\">" << endl;	
  htmlAnyHisto(runNo,hist.deadADC_map_depth[0],"i#eta","i#phi", 92, htmlSubFile,htmlDir);
  htmlAnyHisto(runNo,hist.deadADC_map_depth[1],"i#eta","i#phi", 92, htmlSubFile,htmlDir);
  htmlSubFile << "</tr>" << endl;
  htmlAnyHisto(runNo,hist.deadADC_map_depth[2],"i#eta","i#phi", 92, htmlSubFile,htmlDir);
  htmlAnyHisto(runNo,hist.deadADC_map_depth[3],"i#eta","i#phi", 92, htmlSubFile,htmlDir);
  htmlSubFile << "</tr>" << endl;
  htmlAnyHisto(runNo,hist.deadcapADC_map[0],"i#eta","i#phi", 92, htmlSubFile,htmlDir);
  htmlAnyHisto(runNo,hist.deadcapADC_map[1],"i#eta","i#phi", 92, htmlSubFile,htmlDir);
  htmlSubFile << "</tr>" << endl;
  htmlAnyHisto(runNo,hist.deadcapADC_map[2],"i#eta","i#phi", 92, htmlSubFile,htmlDir);
  htmlAnyHisto(runNo,hist.deadcapADC_map[3],"i#eta","i#phi", 92, htmlSubFile,htmlDir);
  htmlSubFile << "</tr>" << endl;
  htmlAnyHisto(runNo,hist.deadADC_eta,"i#eta","ADC counts", 92, htmlSubFile,htmlDir);
  htmlAnyHisto(runNo,hist.ADCdist,"# of ADC counts","", 92, htmlSubFile,htmlDir);
  htmlSubFile << "</tr></table><br>" << endl;
  htmlSubFile <<"<hr>"<<endl;

  htmlSubFile << "<h3> Plots for cells consistently below pedestal" << endl;
  htmlSubFile << "<table  width=100% border=1><tr>" << endl;
  htmlSubFile << "<tr align=\"left\">" << endl;	
  htmlAnyHisto(runNo,hist.coolcell_below_pedestal_depth[0],"i#eta","i#phi", 92, htmlSubFile,htmlDir);
  htmlAnyHisto(runNo,hist.coolcell_below_pedestal_depth[1],"i#eta","i#phi", 92, htmlSubFile,htmlDir);
  htmlSubFile << "</tr>" << endl;
  htmlAnyHisto(runNo,hist.coolcell_below_pedestal_depth[2],"i#eta","i#phi", 92, htmlSubFile,htmlDir);
  htmlAnyHisto(runNo,hist.coolcell_below_pedestal_depth[3],"i#eta","i#phi", 92, htmlSubFile,htmlDir);
  htmlSubFile << "</tr>" << endl;
  htmlAnyHisto(runNo,hist.above_pedestal_depth[0],"i#eta","i#phi", 92, htmlSubFile,htmlDir);
  htmlAnyHisto(runNo,hist.above_pedestal_depth[1],"i#eta","i#phi", 92, htmlSubFile,htmlDir);
  htmlSubFile << "</tr>" << endl;
  htmlAnyHisto(runNo,hist.above_pedestal_depth[2],"i#eta","i#phi", 92, htmlSubFile,htmlDir);
  htmlAnyHisto(runNo,hist.above_pedestal_depth[3],"i#eta","i#phi", 92, htmlSubFile,htmlDir);
  htmlSubFile << "</tr>" << endl;
  htmlAnyHisto(runNo,hist.coolcell_below_pedestal,"i#eta","i#phi", 92, htmlSubFile,htmlDir);
  htmlAnyHisto(runNo,hist.above_pedestal,"i#eta","i#phi", 92, htmlSubFile,htmlDir);
  htmlSubFile << "</tr></table><br>" << endl;
  htmlSubFile <<"<hr>"<<endl;

  htmlSubFile << "<h3> IN DEVELOPMENT:  Cells will low energies compared to their neighbors"<<endl;
  htmlSubFile << "<table  width=100% border=1><tr>" << endl;
  htmlSubFile << "<tr align=\"left\">" << endl;	
  htmlAnyHisto(runNo,hist.NADA_cool_cell_map_depth[0],"i#eta","i#phi", 92, htmlSubFile,htmlDir);
  htmlAnyHisto(runNo,hist.NADA_cool_cell_map_depth[1],"i#eta","i#phi", 92, htmlSubFile,htmlDir);
  htmlSubFile << "</tr>" << endl;
  htmlAnyHisto(runNo,hist.NADA_cool_cell_map_depth[2],"i#eta","i#phi", 92, htmlSubFile,htmlDir);
  htmlAnyHisto(runNo,hist.NADA_cool_cell_map_depth[3],"i#eta","i#phi", 92, htmlSubFile,htmlDir);

  htmlSubFile << "</tr></table><br>" << endl;
  htmlSubFile <<"<hr>"<<endl;

  // html page footer
  htmlSubFile << "</body> " << endl;
  htmlSubFile << "</html> " << endl;

  htmlSubFile.close();
  return;
} // void HcalDeadCellClient::htmlSubDetOutput



void HcalDeadCellClient::createTests()
{
  if(debug_) cout <<"Creating DeadCell tests..."<<endl;
  createSubDetTests(hbhists);
  createSubDetTests(hehists);
  createSubDetTests(hohists);
  createSubDetTests(hfhists);
  createSubDetTests(hcalhists); // redundant?  Or replaces individual subdetector tests?
  return;
} // void HcalDeadCellClient::createTests()


void HcalDeadCellClient::createSubDetTests(DeadCellHists& hist)
{  
  if(!dbe_) return;

  if(!subDetsOn_[hist.type]) return;
  if (debug_) 
    cout <<"Running HcalDeadCellClient::createSubDetTests for subdetector: "<<hist.type<<endl;
  char meTitle[250], name[250];
  vector<string> params;
  
  string type;
  if(hist.type==1) type= "HB";
  else if(hist.type==2) type = "HE"; 
  else if(hist.type==3) type = "HO"; 
  else if(hist.type==4) type = "HF"; 
  else if(hist.type==10) type = "HCAL";
  else {
    if (debug_)cout <<"<HcalDeadCellClient::createSubDetTests> Error:  unrecognized histogram type: "<<hist.type<<endl;
    return;
  }
  
  // Check for dead ADCs
  sprintf(name,"%sHcal/DeadCellMonitor/%s/%s_OccupancyMap_deadADC",process_.c_str(),type.c_str(), type.c_str());
  sprintf(meTitle,"%s No ADC Count Occupancy Map",type.c_str()); 
  if (debug_) cout <<"Checking for histogram named: "<<name<<endl;
  if(dqmQtests_.find(name)==dqmQtests_.end()){
    if (debug_) cout <<"Didn't find histogram; search for title: "<<meTitle<<endl;
    MonitorElement* me = dbe_->get(meTitle);
    if (me){
      if (debug_) cout <<"Got histogram with title "<<meTitle<<"\nChecking for content"<<endl;
      dqmQtests_[name]=meTitle;
      params.clear();
      params.push_back((string)meTitle);
      params.push_back((string)name);
      createH2ContentTest(dbe_,params);
    }
    else
      if (debug_) cout <<"Couldn't find histogram with title: "<<meTitle<<endl;
  }
  
  // Check NADA cool cells
  sprintf(name,"%sHcal/DeadCellMonitor/%s/%s_OccupancyMap_NADA_CoolCell",process_.c_str(),type.c_str(), type.c_str());
  sprintf(meTitle,"%s Cool Cells",type.c_str()); 
  if (debug_) cout <<"Checking for histogram named: "<<name<<endl;
  if(dqmQtests_.find(name)==dqmQtests_.end()){
    if (debug_) cout <<"Didn't find histogram; search for title: "<<meTitle<<endl;
    MonitorElement* me = dbe_->get(meTitle);
    if (me){
      if (debug_) cout <<"Got histogram with title "<<meTitle<<"\nChecking for content"<<endl;
      dqmQtests_[name]=meTitle;
      params.clear();
      params.push_back((string)meTitle);
      params.push_back((string)name);
      createH2ContentTest(dbe_,params);
    }
    else
      if (debug_) cout <<"Couldn't find histogram with title: "<<meTitle<<endl;
  }
  
  // Check for cells consistently below pedestal+nsigma
  sprintf(name,"%sHcal/DeadCellMonitor/%s/%s_OccupancyMap_belowPedestal",process_.c_str(),type.c_str(), type.c_str());
  
  //sprintf(meTitle,"%s cells below (pedestal+0sigma) for ",type.c_str()); 
  if (debug_) cout <<"Checking for histogram named: "<<name<<endl;
  /*
  // need to fix this -- name does not match title for CoolCell_belowPed
  // (name would be e.g., 'HB Cells below pedestal + N sigma for X consecutive events)
  // (And am I flipping name and title here?  Check when redoing alarms.)

  if(dqmQtests_.find(name)==dqmQtests_.end())
    {
      
      if (debug_) cout <<"Didn't find histogram; search for title: "<<meTitle<<endl;
      //return; 
      //MonitorElement* me = dbe_->get(meTitle);
      if (me)
	{
	  if (debug_) cout <<"Got histogram with title "<<meTitle<<"\nChecking for content"<<endl;
	  dqmQtests_[name]=meTitle;
	  params.clear();
	  params.push_back((string)meTitle);
	  params.push_back((string)name);
	  createH2ContentTest(dbe_,params);
	} //if (me)
      else
	if (debug_) cout <<"Couldn't find histogram with title: "<<meTitle<<endl;
    } // if (dqmQtests_.find(name)==dqmQtests_.end())
  */

  return;
} // void HcalDeadCellClient::createSubDetTests


void HcalDeadCellClient::loadHistograms(TFile* infile)
{
  TNamed* tnd = (TNamed*)infile->Get("DQMData/Hcal/DeadCellMonitor/DeadCell Task Event Number");
  if(tnd)
    {
      string s =tnd->GetTitle();
      ievt_ = -1;
      sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &ievt_);
    }
  
  tnd=(TNamed*)infile->Get("DQMData/Hcal/DeadCellMonitor/CheckNevents");
  if (tnd)
    {
      string s =tnd->GetTitle();
      checkNevents_ = -1;
      sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &checkNevents_);
    }

  getSubDetHistogramsFromFile(hbhists,infile);
  getSubDetHistogramsFromFile(hehists,infile);
  getSubDetHistogramsFromFile(hohists,infile);
  getSubDetHistogramsFromFile(hfhists,infile);
  getSubDetHistogramsFromFile(hcalhists,infile);
  return;
} // void HcalDeadCellClient::loadHistograms



