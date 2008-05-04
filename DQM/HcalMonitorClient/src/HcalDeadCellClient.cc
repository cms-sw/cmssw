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


void HcalDeadCellClient::report(){

  if ( debug_ ) cout << "HcalDeadCellClient: report" << endl;
  
  char name[256];
  sprintf(name, "%sHcal/DeadCellMonitor/DeadCell Task Event Number",process_.c_str());
  MonitorElement* me = 0;
  if(dbe_) me = dbe_->get(name);
  if ( me ) {
    string s = me->valueString();
    ievt_ = -1;
    sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &ievt_);
    if ( debug_ ) cout << "Found '" << name << "'" << endl;
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

  hist.deadADC_OccMap=0;
  hist.deadADC_Eta=0;
  hist.badCAPID_OccMap=0;
  hist.badCAPID_Eta=0;
  hist.ADCdist=0;
  hist.NADACoolCellMap=0;
  hist.digiCheck=0;
  hist.cellCheck=0;
  hist.AbovePed=0;
  hist.CoolCellBelowPed=0;
  hist.DeadCap.clear();

  return;
}

void HcalDeadCellClient::deleteHists(DeadCellHists& hist)
{
  if (hist.deadADC_OccMap) delete hist.deadADC_OccMap;
  if (hist.deadADC_Eta) delete hist.deadADC_Eta;
  if (hist.badCAPID_OccMap) delete hist.badCAPID_OccMap;
  if (hist.badCAPID_Eta) delete hist.badCAPID_Eta;
  if (hist.ADCdist) delete hist.ADCdist;
  if (hist.NADACoolCellMap) delete hist.NADACoolCellMap;
  if (hist.digiCheck) delete hist.digiCheck;
  if (hist.cellCheck) delete hist.cellCheck;
  if (hist.AbovePed) delete hist.AbovePed;
  if (hist.CoolCellBelowPed) delete hist.CoolCellBelowPed;
  hist.DeadCap.clear();
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

  sprintf(name,"DeadCellMonitor/%s/%s_deadADCOccupancyMap",type.c_str(),type.c_str());
  if (debug_) cout <<"Histogram name = "<<name<<endl;
  hist.deadADC_OccMap = getAnyHisto(new TH2F(),name, 
				    process_, dbe_,debug_,cloneME_); 

  sprintf(name,"DeadCellMonitor/%s/%s_deadADCEta",type.c_str(),type.c_str());
  if (debug_) cout <<"Histogram name = "<<name<<endl;
  hist.deadADC_Eta = getAnyHisto(new TH1F(),name,
				 process_, dbe_, debug_, cloneME_);

  sprintf(name,"DeadCellMonitor/%s/%s_noADCIDOccupancyMap",type.c_str(),type.c_str());
  if (debug_) cout <<"Histogram name = "<<name<<endl;
  hist.badCAPID_OccMap = getAnyHisto(new TH2F(), name, process_, dbe_,debug_,cloneME_); 

  sprintf(name,"DeadCellMonitor/%s/%s_noADCIDEta",type.c_str(),type.c_str());
  if (debug_) cout <<"Histogram name = "<<name<<endl;
  hist.badCAPID_Eta = getAnyHisto(new TH1F(), name, process_, dbe_,debug_,cloneME_);      
 
  sprintf(name,"DeadCellMonitor/%s/%s_ADCdist",type.c_str(),type.c_str());
  if (debug_) cout <<"Histogram name = "<<name<<endl;
  hist.ADCdist = getAnyHisto(new TH1F(), name, process_, dbe_,debug_,cloneME_);  
  
  sprintf(name,"DeadCellMonitor/%s/%s_NADA_CoolCellMap",type.c_str(),type.c_str());
  if (debug_) cout <<"Histogram name = "<<name<<endl;
  hist.NADACoolCellMap=getAnyHisto(new TH2F(), name, process_, dbe_,debug_,cloneME_);

  sprintf(name,"DeadCellMonitor/%s/%s_digiCheck",type.c_str(),type.c_str());
  if (debug_) cout <<"Histogram name = "<<name<<endl;
  hist.digiCheck=getAnyHisto(new TH2F(), name, process_, dbe_,debug_,cloneME_);

  sprintf(name,"DeadCellMonitor/%s/%s_cellCheck",type.c_str(),type.c_str());
  if (debug_) cout <<"Histogram name = "<<name<<endl;
  hist.cellCheck=getAnyHisto(new TH2F(), name, process_, dbe_,debug_,cloneME_);
  
  sprintf(name,"DeadCellMonitor/%s/%s_abovePed",type.c_str(),type.c_str());
  if (debug_) cout <<"Histogram name = "<<name<<endl;
  hist.AbovePed=getAnyHisto(new TH2F(), name, process_, dbe_,debug_,cloneME_);
  
  sprintf(name,"DeadCellMonitor/%s/%s_CoolCell_belowPed",type.c_str(),type.c_str());
  if (debug_) cout <<"Histogram name = "<<name<<endl;
  hist.CoolCellBelowPed=getAnyHisto(new TH2F(),
				    name, process_,
				    dbe_,debug_,cloneME_);

  for (int i=0;i<4;i++)
    {
      sprintf(name,"DeadCellMonitor/%s/%s_DeadCap%i",type.c_str(),type.c_str(),i);
      if (debug_) cout <<"Histogram name = "<<name<<endl;
      /*
      TH2F* temp=getAnyHisto(new TH2F(),name, process_, dbe_,debug_,cloneME_); 
      if (temp!=NULL)
	hist.DeadCap.push_back(temp);
      */
      hist.DeadCap.push_back(getAnyHisto(new TH2F(), name,process_,
					 dbe_,debug_,cloneME_));
    }
  return;
}

  
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
  
  sprintf(name,"DeadCellMonitor/%s/%s_deadADCOccupancyMap",type.c_str(),type.c_str());
  hist.deadADC_OccMap = (TH2F*)infile->Get(name); 
  sprintf(name,"DeadCellMonitor/%s/%s_deadADCEta",type.c_str(),type.c_str());
  hist.deadADC_Eta = (TH1F*)infile->Get(name);      
  sprintf(name,"DeadCellMonitor/%s/%s_noADCIDOccupancyMap",type.c_str(),type.c_str());
  hist.badCAPID_OccMap = (TH2F*)infile->Get(name); 
  sprintf(name,"DeadCellMonitor/%s/%s_noADCIDEta",type.c_str(),type.c_str());
  hist.badCAPID_Eta = (TH1F*)infile->Get(name);      
  sprintf(name,"DeadCellMonitor/%s/%s_ADCdist",type.c_str(),type.c_str());
  hist.ADCdist = (TH1F*)infile->Get(name);  
  
  sprintf(name,"DeadCellMonitor/%s/%s_NADA_CoolCellMap",type.c_str(),type.c_str());
  hist.NADACoolCellMap=(TH2F*)infile->Get(name);
  sprintf(name,"DeadCellMonitor/%s/%s_digiCheck",type.c_str(),type.c_str());
  hist.digiCheck=(TH2F*)infile->Get(name);
  sprintf(name,"DeadCellMonitor/%s/%s_cellCheck",type.c_str(),type.c_str());
  hist.cellCheck=(TH2F*)infile->Get(name);
  sprintf(name,"DeadCellMonitor/%s/%s_abovePed",type.c_str(),type.c_str());
  hist.AbovePed=(TH2F*)infile->Get(name);
  sprintf(name,"DeadCellMonitor/%s/%s_CoolCell_belowPed",type.c_str(),type.c_str());
  hist.CoolCellBelowPed=(TH2F*)infile->Get(name);

  for (int i=0;i<4;i++)
    {
      sprintf(name,"DeadCellMonitor/%s/%s_DeadCap%i",type.c_str(),type.c_str(),i);
      hist.DeadCap.push_back((TH2F*)infile->Get(name));
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
  cout <<"Reset subdet"<<type.c_str()<<endl;
  //printf("Reset subdet %s\n",type.c_str());
  sprintf(name,"DeadCellMonitor/%s/%s_deadADCOccupancyMap",type.c_str(),type.c_str());
  resetME(name,dbe_);
  sprintf(name,"DeadCellMonitor/%s/%s_deadADCEta",type.c_str(),type.c_str());
  resetME(name,dbe_);
  sprintf(name,"DeadCellMonitor/%s/%s_noADCIDOccupancyMap",type.c_str(),type.c_str());
  resetME(name,dbe_);
  sprintf(name,"DeadCellMonitor/%s/%s_noADCIDEta",type.c_str(),type.c_str());
  resetME(name,dbe_);
  sprintf(name,"DeadCellMonitor/%s/%s_ADCdist",type.c_str(),type.c_str());
  resetME(name,dbe_);
  sprintf(name,"DeadCellMonitor/%s/%s_NADA_CoolCellMap",type.c_str(),type.c_str());
  resetME(name,dbe_);
  sprintf(name,"DeadCellMonitor/%s/%s_digiCheck",type.c_str(),type.c_str());
  resetME(name,dbe_);
  sprintf(name,"DeadCellMonitor/%s/%s_cellCheck",type.c_str(),type.c_str());
  resetME(name,dbe_);
  sprintf(name,"DeadCellMonitor/%s/%s_abovePed",type.c_str(),type.c_str());
  resetME(name,dbe_);
  sprintf(name,"DeadCellMonitor/%s/%s_CoolCell_belowPed",type.c_str(),type.c_str());
  resetME(name,dbe_);
  
  for (int i=0;i<4;i++)
    {
      sprintf(name,"DeadCellMonitor/%s/%s_DeadCap%i",type.c_str(),type.c_str(),i);
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

  htmlFile << "<br>" << endl;

  htmlFile << "<td>&nbsp;&nbsp;&nbsp;<h3>Cells with no ADC hits</h3></td>"<<endl;
  htmlFile << "<td>&nbsp;&nbsp;&nbsp;<h3> Cells Consistently Below Pedestal Threshold</h3></td>"<<endl;
  htmlFile << "</tr>"<<endl;

  htmlFile << "<tr align=\"left\">" << endl;
  
  htmlAnyHisto(runNo,hcalhists.deadADC_OccMap,"iEta","iPhi", 92, htmlFile,htmlDir);
  htmlAnyHisto(runNo,hcalhists.CoolCellBelowPed,"iEta","iPhi", 92, htmlFile,htmlDir);
  htmlFile<<"</tr>"<<endl;

  htmlFile<< "<tr><td>This histogram shows cells with no ADC hits in an event.  We expect cells to almost always have at least one hit per event.  <BR>Warning messages are sent if a cell's ADC count is 0 for more than 1% of events.<BR> Error messages are sent if a cell's ADC count is 0 for more than 5% of events.</td>"<<endl;
  htmlFile<< "<td>This histogram shows cells with energy below (pedestal + N sigma) for a number of consecutive events.  (The value of N is given on the histogram.)  This histogram is expected to be empty, or nearly so.<BR>  No warnings or errors are sent yet for this histogram.</td>"<<endl;
 
  htmlFile << "</tr>" << endl;

  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;
  htmlFile.close();

  return;
} //void HcalDeadCellClient::htmlOutput()



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
  htmlSubFile << "<h3>" << endl;

  htmlSubFile << "<table  width=100% border=1><tr>" << endl;

  htmlSubFile << "<tr align=\"left\">" << endl;	

  htmlAnyHisto(runNo,hist.deadADC_OccMap,"iEta","iPhi", 92, htmlSubFile,htmlDir);
  htmlAnyHisto(runNo,hist.deadADC_Eta,"iEta","Evts", 100, htmlSubFile,htmlDir);
  htmlSubFile << "</tr>" << endl;
  
  htmlSubFile << "<tr align=\"left\">" << endl;	
  htmlAnyHisto(runNo,hist.digiCheck,"iEta","iPhi", 92, htmlSubFile,htmlDir);
  htmlAnyHisto(runNo,hist.cellCheck,"iEta","iPhi", 100, htmlSubFile,htmlDir);
  htmlSubFile << "</tr>" << endl;
  
  htmlSubFile << "<tr align=\"left\">" << endl;	
  htmlAnyHisto(runNo,hist.NADACoolCellMap,"iEta","iPhi", 92, htmlSubFile,htmlDir);
  htmlAnyHisto(runNo,hist.CoolCellBelowPed,"iEta","iPhi", 100, htmlSubFile,htmlDir);
  htmlSubFile << "</tr>" << endl;

  htmlSubFile << "<tr align=\"left\">" << endl;
  htmlAnyHisto(runNo,hist.DeadCap[0],"iEta","iPhi", 92, htmlSubFile,htmlDir);
  htmlAnyHisto(runNo,hist.DeadCap[1],"iEta","iPhi", 100, htmlSubFile,htmlDir);
  htmlSubFile << "</tr>" << endl;
  
  htmlSubFile << "<tr align=\"left\">" << endl;
  htmlAnyHisto(runNo,hist.DeadCap[2],"iEta","iPhi", 92, htmlSubFile,htmlDir);
  htmlAnyHisto(runNo,hist.DeadCap[3],"iEta","iPhi", 100, htmlSubFile,htmlDir);
  htmlSubFile << "</tr></table>" << endl;

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
  sprintf(meTitle,"%sHcal/DeadCellMonitor/%s/%s_deadADCOccupancyMap",process_.c_str(),type.c_str(), type.c_str());
  sprintf(name,"%s Dead ADC Map",type.c_str()); 
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
  sprintf(meTitle,"%sHcal/DeadCellMonitor/%s/%s_NADA_CoolCellMap",process_.c_str(),type.c_str(), type.c_str());
  sprintf(name,"%s NADA Cool Cell Map",type.c_str()); 
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
  sprintf(meTitle,"%sHcal/DeadCellMonitor/%s/%s_CoolCell_belowPed",process_.c_str(),type.c_str(), type.c_str());
  
  sprintf(name,"%s NADA Cool Cell Map",type.c_str()); 
  if (debug_) cout <<"Checking for histogram named: "<<name<<endl;
  /*
    // need to fix this -- name does not match title for CoolCell_belowPed
    // (name would be e.g., 'HB Cells below pedestal + N sigma for X consecutive events)
    // (And am I flipping name and title here?  Check when redoing alarms.)

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
  getSubDetHistogramsFromFile(hbhists,infile);
  getSubDetHistogramsFromFile(hehists,infile);
  getSubDetHistogramsFromFile(hohists,infile);
  getSubDetHistogramsFromFile(hfhists,infile);
  getSubDetHistogramsFromFile(hcalhists,infile);
  return;
} // void HcalDeadCellClient::loadHistograms


