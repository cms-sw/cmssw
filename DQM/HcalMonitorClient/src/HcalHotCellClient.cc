#include <DQM/HcalMonitorClient/interface/HcalHotCellClient.h>
#include <DQM/HcalMonitorClient/interface/HcalClientUtils.h>
#include <DQM/HcalMonitorClient/interface/HcalHistoUtils.h>
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

HcalHotCellClient::HcalHotCellClient(){}

void HcalHotCellClient::init(const ParameterSet& ps, DQMStore* dbe,string clientName){
  
  //Call the base class first
  HcalBaseClient::init(ps,dbe,clientName);

  if (debug_)
    cout <<"Initializing HcalHotCellClient from ParameterSet"<<endl;

  hbhists.type=1;
  hehists.type=2;
  hohists.type=3;
  hfhists.type=4;
  hcalhists.type=10; // sums over subdetector histograms

  clearHists(hbhists);
  clearHists(hehists);
  clearHists(hohists);
  clearHists(hfhists);
  clearHists(hcalhists);

 
  for(int i=0; i<4; i++) subDetsOn_[i] = false;

  vector<string> subdets = ps.getUntrackedParameter<vector<string> >("subDetsOn");
  for(unsigned int i=0; i<subdets.size(); i++){
    if(subdets[i]=="HB"){
      subDetsOn_[0]=true;
      getSubDetThresholds(hbhists);
    }
    else if(subdets[i]=="HE") {
      subDetsOn_[1]=true;
      getSubDetThresholds(hehists);
    }
    else if(subdets[i]=="HO") {
      subDetsOn_[2]=true;
      getSubDetThresholds(hohists);
    }
    else if(subdets[i]=="HF"){
      subDetsOn_[3]=true;
      getSubDetThresholds(hfhists);
    }
  } // for (unsigned int i=0; i<subdets.size();i++)
  hcalhists.thresholds=max(hbhists.thresholds,max(hehists.thresholds,max(hohists.thresholds,hfhists.thresholds)));

  // Until I can figure out what the $*!@$%@!%^ is wrong with saving floats in the code, hardcode the number of thresholds
  hcalhists.thresholds=5;

} // void HcalHotCellClient::init()


HcalHotCellClient::~HcalHotCellClient(){
  this->cleanup();
}

void HcalHotCellClient::beginJob(void){
  
  if ( debug_ ) cout << "HcalHotCellClient: beginJob" << endl;
  
  ievt_ = 0;
  jevt_ = 0;

  this->setup();
  return;
}

void HcalHotCellClient::beginRun(void){

  if ( debug_ ) cout << "HcalHotCellClient: beginRun" << endl;

  jevt_ = 0;
  this->setup();
  this->resetAllME();
  return;
}

void HcalHotCellClient::endJob(void) {

  if ( debug_ ) cout << "HcalHotCellClient: endJob, ievt = " << ievt_ << endl;

  this->cleanup(); 
  return;
}

void HcalHotCellClient::endRun(void) 
{

  if ( debug_ ) cout << "HcalHotCellClient: endRun, jevt = " << jevt_ << endl;

  cout <<"Cleaning up"<<endl;
  this->cleanup();  
  cout <<"Finished cleanup"<<endl;
  return;
}

void HcalHotCellClient::setup(void) {
  
  return;
}

void HcalHotCellClient::cleanup(void) {

  if (debug_)
    cout <<"HcalHotCellClient::cleanup"<<endl;
  if ( cloneME_ ) 
    {
      cout <<"Deleting histos"<<endl;
      deleteHists(hbhists);
      deleteHists(hehists);
      deleteHists(hohists);
      deleteHists(hfhists);
      deleteHists(hcalhists);
    }    
    
  cout <<"Clearing histos"<<endl;
  clearHists(hbhists);
  clearHists(hehists);
  clearHists(hohists);
  clearHists(hfhists);
  clearHists(hcalhists);

  cout <<"Clearing test"<<endl;
  dqmReportMapErr_.clear(); 
  dqmReportMapWarn_.clear(); 
  dqmReportMapOther_.clear();
  dqmQtests_.clear();
  
  if (debug_)
    cout <<"<HcalHotCellClient> finished cleanup"<<endl;
  return;
}


void HcalHotCellClient::report()
{

  if ( debug_ ) cout << "HcalHotCellClient: report" << endl;

  char name[256];
  sprintf(name, "%sHcal/HotCellMonitor/HotCell Task Event Number",process_.c_str());
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
}//void HcalHotCellClient::report()


void HcalHotCellClient::analyze(void)
{
  jevt_++;
  int updates = 0;

  if ( updates % 10 == 0 ) {
    if ( debug_ ) cout << "HcalHotCellClient: " << updates << " updates" << endl;
  }
  
  return;
}

void HcalHotCellClient::getHistograms(){
  if(!dbe_) return;
  
  if (debug_)
    cout <<"HcalHotCellClient::getHistograms()"<<endl;
  
  if(subDetsOn_[0]) getSubDetHistograms(hbhists);
  if(subDetsOn_[1]) getSubDetHistograms(hehists);
  if(subDetsOn_[2]) getSubDetHistograms(hohists);
  if(subDetsOn_[3]) getSubDetHistograms(hfhists);
  getSubDetHistograms(hcalhists);

  return;
}

void HcalHotCellClient::resetAllME(){
  if(!dbe_) return;

  Char_t name[150];    

  sprintf(name,"%sHcal/HotCellMonitor/HotCellEnergy",process_.c_str());
  resetME(name,dbe_);
  sprintf(name,"%sHcal/HotCellMonitor/HotCellTime",process_.c_str());
  resetME(name,dbe_);
  for(int i=1; i<5; i++){
    sprintf(name,"%sHcal/HotCellMonitor/HotCellDepth%dOccupancyMap",process_.c_str(),i);
    resetME(name,dbe_);
    sprintf(name,"%sHcal/HotCellMonitor/HotCellDepth%dEnergyMap",process_.c_str(),i);
    resetME(name,dbe_);
  }
  sprintf(name,"%sHcal/HotCellMonitor/HotCellOccupancyMap",process_.c_str());
  resetME(name,dbe_);
  sprintf(name,"%sHcal/HotCellMonitor/_HotCell_EnergyMap_",process_.c_str());
  resetME(name,dbe_);

  if (subDetsOn_[0]) resetSubDetHistograms(hbhists);
  if (subDetsOn_[1]) resetSubDetHistograms(hehists);
  if (subDetsOn_[2]) resetSubDetHistograms(hohists);
  if (subDetsOn_[3]) resetSubDetHistograms(hfhists);
  resetSubDetHistograms(hcalhists);
  return;
}

void HcalHotCellClient::htmlOutput(int runNo, string htmlDir, string htmlName)
{

  cout << "Preparing HcalHotCellClient html output ..." << endl;
  string client = "HotCellMonitor";
  htmlErrors(runNo,htmlDir,client,process_,dbe_,dqmReportMapErr_,dqmReportMapWarn_,dqmReportMapOther_);
  
  htmlFile.open((htmlDir + htmlName).c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlFile << "<html>  " << endl;
  htmlFile << "<head>  " << endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlFile << " http-equiv=\"content-type\">  " << endl;
  htmlFile << "  <title>Monitor: Hcal HotCell Task output</title> " << endl;
  // Add JavaScript for pop-up window
  // (java code taken from http://www.yourhtmlsource.com/javascript/popupwindows.html#openingnewwindows)
  htmlFile<<"<script language=JavaScript>"<<endl;
  htmlFile<<" var newwindow;"<<endl;
  htmlFile<<" function poptastic(url)"<<endl;
  htmlFile<<" {"<<endl;
  htmlFile<<" newwindow=window.open(url,'Energy Thresholds','height=400,width=800,scrollbars=yes,resizable=yes');"<<endl;
  htmlFile<<" if (window.focus) {newwindow.focus()}"<<endl;
  htmlFile<<" <a href=\"javascript:window.close()\">Close this window.</a>"<<endl;
  htmlFile<<" } "<<endl<<" </script>"<<endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlFile << "<body>  " << endl;
  htmlFile << "<br>  " << endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << runNo << "</span></h2>" << endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">Hcal HotCells</span></h2> " << endl;
  htmlFile << "<h2>Events processed:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;

  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << ievt_ << "</span></h2>" << endl;
  htmlFile << "<hr>" << endl;
  htmlFile << "<table  width=100% border=1><tr>" << endl;
  if(hasErrors())htmlFile << "<td bgcolor=red><a href=\"HotCellMonitorErrors.html\">Errors in this task</a></td>" << endl;
  else htmlFile << "<td bgcolor=lime>No Errors</td>" << endl;
  if(hasWarnings()) htmlFile << "<td bgcolor=yellow><a href=\"HotCellMonitorWarnings.html\">Warnings in this task</a></td>" << endl;
  else htmlFile << "<td bgcolor=lime>No Warnings</td>" << endl;
  if(hasOther()) htmlFile << "<td bgcolor=aqua><a href=\"HotCellMonitorMessages.html\">Messages in this task</a></td>" << endl;
  else htmlFile << "<td bgcolor=lime>No Messages</td>" << endl;
  htmlFile << "</tr></table>" << endl;
  htmlFile << "<hr>" << endl;

  htmlFile<<"<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\"> " << endl;
  htmlFile << "<h3><tr><td>Detailed (expert-level) Plots:  </td>";
  htmlFile << "<td><a href=\"HcalHotCellClient_HCAL_Plots.html\">HCAL Plots </a>  </td>" << endl;  
  if(subDetsOn_[0]) htmlFile << "<td><a href=\"HcalHotCellClient_HB_Plots.html\">HB Plots </a>  </td>" << endl;  
  if(subDetsOn_[1]) htmlFile << "<td><a href=\"HcalHotCellClient_HE_Plots.html\">HE Plots </a>  </td>" << endl;  
  if(subDetsOn_[2]) htmlFile << "<td><a href=\"HcalHotCellClient_HO_Plots.html\">HO Plots </a>  </td>" << endl;  
  if(subDetsOn_[3]) htmlFile << "<td><a href=\"HcalHotCellClient_HF_Plots.html\">HF Plots </a>  </td>" << endl; 
  htmlFile <<"</h3></tr></table>"<<endl;

  htmlFile << "<h2><strong>Hcal Hot Cell Histograms</strong></h2>" << endl;
  htmlFile <<"Occasional entries in the hot cell histograms are expected.<br>  Shifters should watch for persistent hot cells occurring at a high rate."<<endl;
  htmlFile <<"<hr>"<<endl;

  // Main histogram table
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\"> " << endl;
  
  // Label Row
  htmlFile <<"<tr><td>Hot Cells Above Threshold Energy</td>"<<endl;
  htmlFile <<"<td>Isolated Hot Cells</td>"<<endl;
  htmlFile <<"<td>Negative-Energy Cells</td></tr>"<<endl;
  htmlFile << "<tr align=\"left\">" << endl;



  for (unsigned int fu=0;fu<hcalhists.threshOccMap.size();++fu)
    {
      htmlAnyHisto(runNo,hcalhists.threshOccMap[fu],"iEta","iPhi",
		   92, htmlFile,htmlDir);
    }

 // Histogram Row
  htmlAnyHisto(runNo,hcalhists.threshOccMap[hcalhists.threshOccMap.size()-1],"iEta","iPhi", 92, htmlFile,htmlDir);
  htmlAnyHisto(runNo,hcalhists.nadaOccMap,"iEta","iPhi", 92, htmlFile,htmlDir);
  htmlAnyHisto(runNo,hcalhists.nadaNegOccMap,"iEta","iPhi", 92, htmlFile,htmlDir);
  htmlFile<<"</tr>"<<endl;


  // Description row

  htmlFile<<"<tr><td>"<<endl;
  htmlFile<<"<h4>Plot shows the fraction of events in which a cell's energy is greater than a threshold value.<BR>"<<endl;
 htmlFile <<"(<a href=\"javascript:poptastic('HotCell_Thresholds.html');\">Click here for threshold energy values.)</a>"<<endl;
  htmlFile<<"<BR>"<<endl;
  htmlFile<<" Warning messages are sent when a cell exceeds the largest-energy threshold in more than 10% of events.<BR>"<<endl;
  htmlFile<<"Error messages are sent when a cell exceeds the largest-energy threshold in more than 25% of events.<BR>(Warnings and Errors not yet implemented.)<BR></td>"<<endl;
  htmlFile<< "<td>This histogram shows cells with energy significantly higher than the surrounding cells.<BR>Warning messages are sent when an isolated hot cell is found in more than 10% of events.<BR>Error messages are sent when an isolated hot cell is found in more than 25% of events.<BR>(Warnings and Errors not yet implemented.)</td>"<<endl;
  htmlFile<< "<td>This histogram shows events in which cells reported -1 GeV of energy or less.<BR>Warning messages are sent when a cell reports such a low value more than 10% of the time.<BR>Error messages are sent when a cell reports such a value in more than 25% of events.<BR>(Warnings and Errors not yet implemented.)</td>"<<endl;
  htmlFile << "</tr>" << endl;
  htmlFile <<"</table>"<<endl;

  // Show all threshold histograms
  htmlFile<<"<hr><hr>"<<endl;
  htmlFile << "<h2><strong>Hcal Energy Threshold Histograms</strong></h2>" << endl;
  htmlFile <<"These histograms show cell occupancies for various energy thresholds (threshold values can be found <a href=\"javascript:poptastic('HotCell_Thresholds.html');\">here</a>).  These histograms are provided for shifter interest; they do not generate alarms."<<endl;
  
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\"> " << endl;
  // Divide histograms by number of events to express hot cells as fraction
  
  for (unsigned int i=hcalhists.threshOccMap.size();i>0;i--)
    {
      //hcalhists.threshOccMap[i-1]->Scale(1./ievt_);
      //cout <<"i = "<<i<<endl;
      htmlAnyHisto(runNo,hcalhists.threshOccMap[i-1],"iEta","iPhi", 92, htmlFile,htmlDir);
    }
  htmlFile << "</tr>" << endl;
  htmlFile << "</table>" << endl; // end of table
  htmlFile << "<br>" << endl;

  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  htmlFile.close();

  htmlSubDetOutput(hcalhists,runNo,htmlDir,htmlName);
  htmlSubDetOutput(hbhists,runNo,htmlDir,htmlName);
  htmlSubDetOutput(hehists,runNo,htmlDir,htmlName);
  htmlSubDetOutput(hohists,runNo,htmlDir,htmlName);
  htmlSubDetOutput(hfhists,runNo,htmlDir,htmlName);

  // Make page with histogram thresholds
  htmlFile.open((htmlDir + "HotCell_Thresholds.html").c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlFile << "<html>  " << endl;
  htmlFile << "<head>  " << endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlFile << " http-equiv=\"content-type\">  " << endl;
  htmlFile << "  <title>Monitor: Hcal HotCell Threshold Values</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlFile << "<body>  " << endl;
  htmlFile << "<br>  " << endl;
  htmlFile << "<h2>Hot Cell Histogram Energy Thresholds:" << endl;
  htmlFile <<"<br> "<<endl;
  htmlFile << "<table border=\"1\" cellspacing=\"0\" bgcolor=#FFCCFF" << endl;
  htmlFile << "cellpadding=\"10\"> " << endl;
  htmlFile <<"<tr><td></td>"<<endl;
  for (unsigned int i=hcalhists.threshOccMap.size();i>0;i--)
    {
      htmlFile<<"<td>Threshold "<<i-1<<"</td>"<<endl;
    }
  htmlFile<<"</tr>"<<endl;
  drawSubDetThresholds(hbhists);
  drawSubDetThresholds(hehists);
  drawSubDetThresholds(hohists);
  drawSubDetThresholds(hfhists);
  htmlFile<<"</table>"<<endl;
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  return;
}


void HcalHotCellClient::createTests()
{
  if(!dbe_) return;
  /*
  char meTitle[250], name[250];    
  vector<string> params;
  */

  if (debug_) cout <<"Creating HotCell tests..."<<endl;
  createSubDetTests(hbhists);
  createSubDetTests(hehists);
  createSubDetTests(hohists);
  createSubDetTests(hfhists);
  createSubDetTests(hcalhists); // unnecessary?  Or only use this test?
  return;
}


void HcalHotCellClient::createSubDetTests(HotCellHists& hist)
{
  if(!subDetsOn_[hist.type]) return;
  if (debug_) 
    cout <<"Running HcalHotCellClient::createSubDetTests for subdetector: "<<hist.type<<endl;
  char meTitle[250], name[250];
  vector<string> params;

  string type;
  if(hist.type==1)
    type="HB";
  else if(hist.type==2)
    type="HE";
  else if (hist.type==3)
    type="HO";
  else if (hist.type==4)
    type="HF";
  else if (hist.type==10)
    type="HCAL";

  // Check NADA Hot Cell
  sprintf(meTitle,"%sHcal/HotCellMonitor/%s/%s_OccupancyMap_NADA",process_.c_str(),type.c_str(), type.c_str());
  sprintf(name,"%s NADA Hot Cell Test",type.c_str()); 
  if (debug_) cout <<"Checking for histogram named: "<<name<<endl;
  if(dqmQtests_.find(name)==dqmQtests_.end())
    {
      if (debug_) cout <<"Didn't find histogram; search for title: "<<meTitle<<endl;
      MonitorElement* me = dbe_->get(meTitle);
      if (me)
	{
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

  // Check NADA Negative-energy cells
  sprintf(meTitle,"%sHcal/HotCellMonitor/%s/%snadaNegOccMap",process_.c_str(),type.c_str(), type.c_str());
  sprintf(name,"%s NADA Negative Energy Cell Test",type.c_str()); 
  if (debug_) cout <<"Checking for histogram named: "<<name<<endl;
  if(dqmQtests_.find(name)==dqmQtests_.end())
    {
      if (debug_) cout <<"Didn't find histogram; search for title: "<<meTitle<<endl;
      MonitorElement* me = dbe_->get(meTitle);
      if (me)
	{
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
  
  // Check Cell with Maximum Energy deposited
  // Jan. 2008 DQM Challenge -- instead of Maximum Energy histogram
  // Check for Cells > final (highest?) threshold
  //sprintf(meTitle,"%sHcal/HotCellMonitor/%s/%sHotCellOccMapMaxCell",process_.c_str(),type.c_str(), type.c_str());
  sprintf(meTitle,"%sHcal/HotCellMonitor/%s/%s_OccupancyMap_HotCell_Threshold%i",process_.c_str(),type.c_str(), type.c_str(),hist.thresholds);
  //sprintf(name,"%s Maximum Energy Cell",type.c_str()); 
  sprintf(name,"%s Threshold #%i Cell",type.c_str(),hist.thresholds);
  if (debug_) cout <<"Checking for histogram named: "<<name<<endl;
  if(dqmQtests_.find(name)==dqmQtests_.end())
    {
      if (debug_) cout <<"Didn't find histogram; search for title: "<<meTitle<<endl;
      MonitorElement* me = dbe_->get(meTitle);
      if (me)
	{
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

  // Check threshold tests  -- disable for now
  /*
  for (int i=0;i<hist.thresholds;i++)
    {
      sprintf(meTitle,"%sHcal/HotCellMonitor/%s/%sHotCellOccMapThresh%i",process_.c_str(),type.c_str(), type.c_str(),i);
      sprintf(name,"%s Hot Cells Above Threshold %i",type.c_str(),i); 
      if (debug_) cout <<"Checking for histogram named: "<<name<<endl;
      if(dqmQtests_.find(name)==dqmQtests_.end())
	{
	  if (debug_) cout <<"Didn't find histogram; search for title: "<<meTitle<<endl;
	  MonitorElement* me = dbe_->get(meTitle);
	  if (me)
	    {
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
    } // for (int i=0;i<thresholds; i++)
  */

  return;
}

void HcalHotCellClient::loadHistograms(TFile* infile){

  TNamed* tnd = (TNamed*)infile->Get("DQMData/Hcal/HotCellMonitor/HotCell Task Event Number");
  if(tnd){
    string s =tnd->GetTitle();
    ievt_ = -1;
    sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &ievt_);
  }
  getSubDetHistogramsFromFile(hcalhists,infile);
  getSubDetHistogramsFromFile(hbhists,infile);
  getSubDetHistogramsFromFile(hehists,infile);
  getSubDetHistogramsFromFile(hohists,infile);
  getSubDetHistogramsFromFile(hfhists,infile);

  return;
}


void HcalHotCellClient::clearHists(HotCellHists& hist)
{
  if(debug_) cout <<"Clearing HcalHotCell histograms for HCAL type: "<<hist.type<<endl;

  hist.maxCellOccMap=0;
  hist.maxCellEnergyMap=0;
  hist.maxCellEnergy=0;
  hist.maxCellTime=0;
  hist.maxCellID=0;

  /*
  unsigned int mysize=hist.threshOccMap.size();
  
  for (unsigned int i=0;i<mysize;i++)
    {
      hist.threshOccMap[i]=0;
      hist.threshEnergyMap[i]=0;
      hist.threshOccMapDepth1[i]=0;
      hist.threshEnergyMapDepth1[i]=0;
      hist.threshOccMapDepth2[i]=0;
      hist.threshEnergyMapDepth2[i]=0;
      hist.threshOccMapDepth3[i]=0;
      hist.threshEnergyMapDepth3[i]=0;
      hist.threshOccMapDepth4[i]=0;
      hist.threshEnergyMapDepth4[i]=0;
    }
  */

  hist.threshOccMap.clear();
  hist.threshEnergyMap.clear();
  hist.threshOccMapDepth1.clear();
  hist.threshEnergyMapDepth1.clear();
  hist.threshOccMapDepth2.clear();
  hist.threshEnergyMapDepth2.clear();
  hist.threshOccMapDepth3.clear();
  hist.threshEnergyMapDepth3.clear();
  hist.threshOccMapDepth4.clear();
  hist.threshEnergyMapDepth4.clear();

  // NADA histograms
  hist.nadaOccMap=0;
  hist.nadaEnergyMap=0;
  hist.nadaNumHotCells=0;
  hist.nadaTestCell=0;
  hist.nadaEnergy=0;
  hist.nadaNumNegCells=0;
  hist.nadaNegOccMap=0;
  hist.nadaNegEnergyMap=0;
  return;
}

void HcalHotCellClient::deleteHists(HotCellHists& hist)
{
  if (hist.maxCellOccMap) delete hist.maxCellOccMap;
  if (hist.maxCellEnergyMap) delete hist.maxCellEnergyMap;
  if (hist.maxCellEnergy) delete hist.maxCellEnergy;
  if (hist.maxCellTime) delete hist.maxCellTime;
  if (hist.maxCellID) delete hist.maxCellID;
  
  unsigned int mapsize=hist.threshOccMap.size();

  for (unsigned int i=0;i<mapsize;i++)
    {
      if (hist.threshOccMap[i]) delete hist.threshOccMap[i];
      if (hist.threshEnergyMap[i]) delete hist.threshEnergyMap[i];
      if (hist.threshOccMapDepth1[i]) delete hist.threshOccMapDepth1[i];
      if (hist.threshEnergyMapDepth1[i]) delete hist.threshEnergyMapDepth1[i];
      if (hist.threshOccMapDepth2[i]) delete hist.threshOccMapDepth2[i];
      //if (hist.threshEnergyMapDepth2[i]) delete hist.threshEnergyMapDepth2[i];
      if (hist.threshOccMapDepth3[i]) delete hist.threshOccMapDepth3[i];
      if (hist.threshEnergyMapDepth3[i]) delete hist.threshEnergyMapDepth3[i];
      if (hist.threshOccMapDepth4[i]) delete hist.threshOccMapDepth4[i];
      if (hist.threshEnergyMapDepth4[i]) delete hist.threshEnergyMapDepth4[i];

    }

  hist.threshOccMap.clear();
  hist.threshEnergyMap.clear();
  hist.threshOccMapDepth1.clear();
  hist.threshEnergyMapDepth1.clear();
  hist.threshOccMapDepth2.clear();
  //hist.threshEnergyMapDepth2.clear();
  hist.threshOccMapDepth3.clear();
  hist.threshEnergyMapDepth3.clear();
  hist.threshOccMapDepth4.clear();
  hist.threshEnergyMapDepth4.clear();

  // NADA histograms
  if (hist.nadaOccMap) delete hist.nadaOccMap;
  if (hist.nadaEnergyMap) delete hist.nadaEnergyMap;
  if (hist.nadaNumHotCells) delete hist.nadaNumHotCells;
  if (hist.nadaTestCell) delete hist.nadaTestCell;
  if (hist.nadaEnergy) delete hist.nadaEnergy;
  if (hist.nadaNumNegCells) delete hist.nadaNumNegCells;
  if (hist.nadaNegOccMap) delete hist.nadaNegOccMap;
  if (hist.nadaNegEnergyMap) delete hist.nadaNegEnergyMap;
  return;
}


void HcalHotCellClient::getSubDetHistograms(HotCellHists& hist)
{
  char name[150];
  string type;
  if(hist.type==1) type= "HB";
  else if(hist.type==2) type = "HE"; 
  else if(hist.type==3) type = "HO"; 
  else if(hist.type==4) type = "HF"; 
  else if(hist.type==10) type = "HCAL";
  else {
    if (debug_)cout <<"<HcalHotCellClient::getSubDetHistograms> Unrecognized subdetector type:  "<<hist.type<<endl;
    return;
  }
  
  hist.threshOccMap.clear();
  hist.threshEnergyMap.clear();
  hist.threshOccMapDepth1.clear();
  hist.threshEnergyMapDepth1.clear();
  hist.threshOccMapDepth2.clear();
  hist.threshEnergyMapDepth2.clear();
  hist.threshOccMapDepth3.clear();
  hist.threshEnergyMapDepth3.clear();
  hist.threshOccMapDepth4.clear();
  hist.threshEnergyMapDepth4.clear();

  if (debug_)
    cout <<"Getting HcalHotCell Subdetector Histograms for Subdetector:  "<<type<<endl;

  // Make dummy histogram that is used by template function getAnyHisto to determine hist type -- is there a better way to get this info?

  TH2F* dummy2D = new TH2F();
  
  for (int i=0;i<hist.thresholds;i++)
    {
      sprintf(name,"HotCellMonitor/%s/%s_OccupancyMap_HotCell_Threshold%i",type.c_str(),type.c_str(),i);
      // cout <<name<<endl;
      
      hist.threshOccMap.push_back(getAnyHisto(dummy2D, name,process_,dbe_,debug_,cloneME_));
      sprintf(name,"HotCellMonitor/%s/%s_HotCell_EnergyMap_Thresh%i",type.c_str(),type.c_str(),i);
      // cout <<name<<endl;
      hist.threshEnergyMap.push_back(getAnyHisto(dummy2D, name,process_,dbe_,debug_,cloneME_));
      
      sprintf(name,"HotCellMonitor/%s/Depth1/%s_OccupancyMap_HotCell_Threshold%iDepth1",type.c_str(),type.c_str(),i);
      // cout <<name<<endl;
      hist.threshOccMapDepth1.push_back(getAnyHisto(dummy2D, name,process_,dbe_,debug_,cloneME_));
      sprintf(name,"HotCellMonitor/%s/Depth1/%s_HotCell_EnergyMap_Thresh%iDepth1",type.c_str(),type.c_str(),i);
      // cout <<name<<endl;
      hist.threshEnergyMapDepth1.push_back(getAnyHisto(dummy2D, name,process_,dbe_,debug_,cloneME_));
      
      sprintf(name,"HotCellMonitor/%s/Depth2/%s_OccupancyMap_HotCell_Threshold%iDepth2",type.c_str(),type.c_str(),i);
      // cout <<name<<endl;
      hist.threshOccMapDepth2.push_back(getAnyHisto(dummy2D, name,process_,dbe_,debug_,cloneME_));
      sprintf(name,"HotCellMonitor/%s/Depth2/%s_HotCell_EnergyMap_Thresh%iDepth2",type.c_str(),type.c_str(),i);
      // cout <<name<<endl;
      // Something is wrong with this histogram -- causes seg fault in HE?
      //hist.threshEnergyMapDepth2.push_back(getAnyHisto(dummy2D, name,process_,dbe_,debug_,cloneME_));


      sprintf(name,"HotCellMonitor/%s/Depth3/%s_OccupancyMap_HotCell_Threshold%iDepth3",type.c_str(),type.c_str(),i);
      // cout <<name<<endl;
      hist.threshOccMapDepth3.push_back(getAnyHisto(dummy2D, name,process_,dbe_,debug_,cloneME_));
      sprintf(name,"HotCellMonitor/%s/Depth3/%s_HotCell_EnergyMap_Thresh%iDepth3",type.c_str(),type.c_str(),i);
      // cout <<name<<endl;
      hist.threshEnergyMapDepth3.push_back(getAnyHisto(dummy2D, name,process_,dbe_,debug_,cloneME_));
      
      sprintf(name,"HotCellMonitor/%s/Depth4/%s_OccupancyMap_HotCell_Threshold%iDepth4",type.c_str(),type.c_str(),i);
      // cout <<name<<endl;
      hist.threshOccMapDepth4.push_back(getAnyHisto(dummy2D, name,process_,dbe_,debug_,cloneME_));
      sprintf(name,"HotCellMonitor/%s/Depth4/%s_HotCell_EnergyMap_Thresh%iDepth4",type.c_str(),type.c_str(),i);
      // cout <<name<<endl;
      hist.threshEnergyMapDepth4.push_back(getAnyHisto(dummy2D, name,process_,dbe_,debug_,cloneME_));
    } //for (int i=0;i<hist.thresholds.size();++i)

  sprintf(name,"HotCellMonitor/%s/%sHotCellOccMapMaxCell",type.c_str(),type.c_str());
  // cout <<name<<endl;
  hist.maxCellOccMap = getAnyHisto(dummy2D, name, process_, dbe_,debug_,cloneME_);      
  sprintf(name,"HotCellMonitor/%s/%s_HotCell_EnergyMap_MaxCell",type.c_str(),type.c_str());
  // cout <<name<<endl;
  hist.maxCellEnergyMap = getAnyHisto(dummy2D, name, process_, dbe_,debug_,cloneME_);
  
  sprintf(name,"HotCellMonitor/%s/%sHotCellEnergyMaxCell",type.c_str(),type.c_str());
  hist.maxCellEnergy = getAnyHisto( new TH1F(), name, process_, dbe_,debug_,cloneME_);
  // cout <<name<<endl;
  sprintf(name,"HotCellMonitor/%s/%sHotCellTimeMaxCell",type.c_str(),type.c_str());
  hist.maxCellTime = getAnyHisto( new TH1F(), name, process_, dbe_,debug_,cloneME_);    
  // cout <<name<<endl;
  sprintf(name,"HotCellMonitor/%s/%sHotCellIDMaxCell",type.c_str(),type.c_str());
  hist.maxCellID = getAnyHisto( new TH1F(), name, process_, dbe_,debug_,cloneME_);    
  // cout <<name<<endl;

  // NADA histograms
  sprintf(name,"HotCellMonitor/%s/%s_OccupancyMap_NADA",type.c_str(),type.c_str());
  hist.nadaOccMap=getAnyHisto(dummy2D, name, process_, dbe_,debug_,cloneME_);
  //cout <<"NAME = "<<name<<endl;
  sprintf(name,"HotCellMonitor/%s/%snadaEnergyMap",type.c_str(),type.c_str());
  hist.nadaEnergyMap=getAnyHisto(dummy2D, name, process_, dbe_,debug_,cloneME_);
  // cout <<name<<endl;
  sprintf(name,"HotCellMonitor/%s/NADA_%s_NumHotCells",type.c_str(),type.c_str());
  hist.nadaNumHotCells = getAnyHisto( new TH1F(), name, process_, dbe_,debug_,cloneME_);
   // cout <<name<<endl;

  sprintf(name,"HotCellMonitor/%s/NADA_%s_testcell",type.c_str(),type.c_str());
  hist.nadaTestCell = getAnyHisto( new TH1F(), name, process_, dbe_,debug_,cloneME_); 
  // cout <<name<<endl;

  sprintf(name,"HotCellMonitor/%s/NADA_%s_Energy",type.c_str(),type.c_str());
  hist.nadaEnergy = getAnyHisto( new TH1F(), name, process_, dbe_,debug_,cloneME_); 
  // cout <<name<<endl;

  sprintf(name,"HotCellMonitor/%s/NADA_%s_NumNegCells",type.c_str(),type.c_str());
  hist.nadaNumNegCells = getAnyHisto( new TH1F(), name, process_, dbe_,debug_,cloneME_); 
  // cout <<name<<endl;

  sprintf(name,"HotCellMonitor/%s/%snadaNegOccMap",type.c_str(),type.c_str());
  hist.nadaNegOccMap = getAnyHisto(dummy2D, name, process_, dbe_,debug_,cloneME_); 
  // cout <<name<<endl;

  sprintf(name,"HotCellMonitor/%s/%snadaNegEnergyMap",type.c_str(),type.c_str());
  hist.nadaNegEnergyMap = getAnyHisto(dummy2D, name, process_, dbe_,debug_,cloneME_); 
  // cout <<name<<endl;

  return;
}

void HcalHotCellClient::getSubDetHistogramsFromFile(HotCellHists& hist, TFile* infile)
{
  char name[150];

  string type;
  if(hist.type==1) type= "HB";
  else if(hist.type==2) type = "HE"; 
  else if(hist.type==3) type = "HO"; 
  else if(hist.type==4) type = "HF"; 
  else if(hist.type==10) type = "HCAL";
  else {
    if (debug_)cout <<"<HcalHotCellClient::getSubDetHistogramsFromFile> Unrecognized subdetector type:  "<<hist.type<<endl;
    return;
  }

  hist.threshOccMap.clear();
  hist.threshEnergyMap.clear();
  hist.threshOccMapDepth1.clear();
  hist.threshEnergyMapDepth1.clear();
  hist.threshOccMapDepth2.clear();
  hist.threshEnergyMapDepth2.clear();
  hist.threshOccMapDepth3.clear();
  hist.threshEnergyMapDepth3.clear();
  hist.threshOccMapDepth4.clear();
  hist.threshEnergyMapDepth4.clear();
  

  for (int i=0;i<hist.thresholds;++i)
    {
      sprintf(name,"HotCellMonitor/%s/%s_OccupancyMap_HotCell_Threshold%i",type.c_str(),type.c_str(),i);
      hist.threshOccMap.push_back((TH2F*)infile->Get(name));
      sprintf(name,"HotCellMonitor/%s/%s_HotCell_EnergyMap_Thresh%i",type.c_str(),type.c_str(),i);
      hist.threshEnergyMap.push_back((TH2F*)infile->Get(name));
      sprintf(name,"HotCellMonitor/%s/Depth1/%s_OccupancyMap_HotCell_Threshold%iDepth1",type.c_str(),type.c_str(),i);
      hist.threshOccMapDepth1.push_back((TH2F*)infile->Get(name));
      sprintf(name,"HotCellMonitor/%s/Depth1/%s_HotCell_EnergyMap_Thresh%iDepth1",type.c_str(),type.c_str(),i);
      hist.threshEnergyMapDepth1.push_back((TH2F*)infile->Get(name));
      sprintf(name,"HotCellMonitor/%s/Depth2/%s_OccupancyMap_HotCell_Threshold%iDepth2",type.c_str(),type.c_str(),i);
      hist.threshOccMapDepth2.push_back((TH2F*)infile->Get(name));
      sprintf(name,"HotCellMonitor/%s/Depth2/%s_HotCell_EnergyMap_Thresh%iDepth2",type.c_str(),type.c_str(),i);
      hist.threshEnergyMapDepth2.push_back((TH2F*)infile->Get(name));
  sprintf(name,"HotCellMonitor/%s/Depth3/%s_OccupancyMap_HotCell_Threshold%iDepth3",type.c_str(),type.c_str(),i);
      hist.threshOccMapDepth3.push_back((TH2F*)infile->Get(name));
      sprintf(name,"HotCellMonitor/%s/Depth3/%s_HotCell_EnergyMap_Thresh%iDepth3",type.c_str(),type.c_str(),i);
      hist.threshEnergyMapDepth3.push_back((TH2F*)infile->Get(name));
      sprintf(name,"HotCellMonitor/%s/Depth4/%s_OccupancyMap_HotCell_Threshold%iDepth4",type.c_str(),type.c_str(),i);
      hist.threshOccMapDepth4.push_back((TH2F*)infile->Get(name));
      sprintf(name,"HotCellMonitor/%s/Depth4/%s_HotCell_EnergyMap_Thresh%iDepth4",type.c_str(),type.c_str(),i);
      hist.threshEnergyMapDepth4.push_back((TH2F*)infile->Get(name));
    }

  sprintf(name,"HotCellMonitor/%s/%sHotCellOccMapMaxCell",type.c_str(),type.c_str());
  hist.maxCellOccMap = (TH2F*)infile->Get(name);      
  sprintf(name,"HotCellMonitor/%s/%s_HotCell_EnergyMap_MaxCell",type.c_str(),type.c_str());
  hist.maxCellEnergyMap = (TH2F*)infile->Get(name);
  
  sprintf(name,"HotCellMonitor/%s/%sHotCellEnergyMaxCell",type.c_str(),type.c_str());
  hist.maxCellEnergy = (TH1F*)infile->Get(name);
  sprintf(name,"HotCellMonitor/%s/%sHotCellTimeMaxCell",type.c_str(),type.c_str());
  hist.maxCellTime = (TH1F*)infile->Get(name);    
  sprintf(name,"HotCellMonitor/%s/%sHotCellIDMaxCell",type.c_str(),type.c_str());
  hist.maxCellID = (TH1F*)infile->Get(name);    

  // NADA histograms
  sprintf(name,"HotCellMonitor/%s/%s_OccupancyMap_NADA",type.c_str(),type.c_str());
  hist.nadaOccMap=(TH2F*)infile->Get(name);
  sprintf(name,"HotCellMonitor/%s/%snadaEnergyMap",type.c_str(),type.c_str());
  hist.nadaEnergyMap=(TH2F*)infile->Get(name);
  sprintf(name,"HotCellMonitor/%s/NADA_%s_NumHotCells",type.c_str(),type.c_str());
  hist.nadaNumHotCells = (TH1F*)infile->Get(name); 
  sprintf(name,"HotCellMonitor/%s/NADA_%s_testcell",type.c_str(),type.c_str());
  hist.nadaTestCell = (TH1F*)infile->Get(name); 
  sprintf(name,"HotCellMonitor/%s/NADA_%s_Energy",type.c_str(),type.c_str());
  hist.nadaEnergy = (TH1F*)infile->Get(name); 
  sprintf(name,"HotCellMonitor/%s/NADA_%s_NumNegCells",type.c_str(),type.c_str());
  hist.nadaNumNegCells = (TH1F*)infile->Get(name); 
  sprintf(name,"HotCellMonitor/%s/%snadaNegOccMap",type.c_str(),type.c_str());
  hist.nadaNegOccMap = (TH2F*)infile->Get(name); 
  sprintf(name,"HotCellMonitor/%s/%snadaNegEnergyMap",type.c_str(),type.c_str());
  hist.nadaNegEnergyMap = (TH2F*)infile->Get(name); 

  return;
}



void HcalHotCellClient::resetSubDetHistograms(HotCellHists& hist){
  
  char name[150];
  string type;

  if(hist.type==1) type= "HB";
  else if(hist.type==2) type = "HE"; 
  else if(hist.type==3) type = "HO"; 
  else if(hist.type==4) type = "HF"; 
  else if(hist.type==10) type = "HCAL";
  else {
    if (debug_)cout <<"<HcalHotCellClient::resetSubDetHistograms> Unrecognized subdetector type:  "<<hist.type<<endl;
    return;
  }
  if (debug_) cout <<"Reset subdet "<<type.c_str()<<endl;
  for (int i=0;i<hist.thresholds;i++){
    sprintf(name,"HotCellMonitor/%s/%s_HotCell_EnergyMap_Thresh%i",type.c_str(),type.c_str(),i);
    //cout <<"NAME = "<<name<<endl;
    resetME(name,dbe_);
    sprintf(name,"HotCellMonitor/%s/%s_OccupancyMap_HotCell_Threshold%i",type.c_str(),type.c_str(),i);
    //cout <<"NAME = "<<name<<endl;
    resetME(name,dbe_);
    sprintf(name,"HotCellMonitor/%s/Depth1/%s_HotCell_EnergyMap_Thresh%iDepth1",type.c_str(),type.c_str(),i);
    //cout <<"NAME = "<<name<<endl;
    resetME(name,dbe_);
    sprintf(name,"HotCellMonitor/%s/Depth1/%s_OccupancyMap_HotCell_Threshold%iDepth1",type.c_str(),type.c_str(),i);
    //cout <<"NAME = "<<name<<endl;
    resetME(name,dbe_);
    sprintf(name,"HotCellMonitor/%s/Depth2/%s_HotCell_EnergyMap_Thresh%iDepth2",type.c_str(),type.c_str(),i);
    //cout <<"NAME = "<<name<<endl;
    resetME(name,dbe_);
    sprintf(name,"HotCellMonitor/%s/Depth2/%s_OccupancyMap_HotCell_Threshold%iDepth2",type.c_str(),type.c_str(),i);
    //cout <<"NAME = "<<name<<endl;
    resetME(name,dbe_);
    sprintf(name,"HotCellMonitor/%s/Depth3/%s_HotCell_EnergyMap_Thresh%iDepth3",type.c_str(),type.c_str(),i);
    //cout <<"NAME = "<<name<<endl;
    resetME(name,dbe_);
    sprintf(name,"HotCellMonitor/%s/Depth3/%s_OccupancyMap_HotCell_Threshold%iDepth3",type.c_str(),type.c_str(),i);
    //cout <<"NAME = "<<name<<endl;
    resetME(name,dbe_);
sprintf(name,"HotCellMonitor/%s/Depth4/%s_HotCell_EnergyMap_Thresh%iDepth4",type.c_str(),type.c_str(),i);
    //cout <<"NAME = "<<name<<endl;
    resetME(name,dbe_);
    sprintf(name,"HotCellMonitor/%s/Depth4/%s_OccupancyMap_HotCell_Threshold%iDepth4",type.c_str(),type.c_str(),i);
    //cout <<"NAME = "<<name<<endl;
    resetME(name,dbe_);


  }
  
  sprintf(name,"HotCellMonitor/%s/%s_HotCell_EnergyMap_MaxCell",type.c_str(),type.c_str());
  //// cout <<name<<endl;
  resetME(name,dbe_);
  
  sprintf(name,"HotCellMonitor/%s/%sHotCellOccMapMaxCell",type.c_str(),type.c_str());
  //// cout <<name<<endl;
  resetME(name,dbe_);
  
  sprintf(name,"HotCellMonitor/%s/%sHotCellEnergyMaxCell",type.c_str(),type.c_str());
  //// cout <<name<<endl;
  resetME(name,dbe_);

  sprintf(name,"HotCellMonitor/%s/%s HotCellTimeMaxCell",type.c_str(),type.c_str());
  //// cout <<name<<endl;
  resetME(name,dbe_);

  sprintf(name,"HotCellMonitor/%s/%sHotCellIDMaxCell",type.c_str(),type.c_str());
  //// cout <<name<<endl;
  resetME(name,dbe_);

  // NADA histograms
  sprintf(name,"HotCellMonitor/%s/%s_OccupancyMap_NADA",type.c_str(),type.c_str());
  //// cout <<name<<endl;
  resetME(name,dbe_);

  sprintf(name,"HotCellMonitor/%s/%snadaEnergyMap",type.c_str(),type.c_str());
  //// cout <<name<<endl;
  resetME(name,dbe_);

  sprintf(name,"HotCellMonitor/%s/#NADA_%s_NumHotCells",type.c_str(),type.c_str());
  //// cout <<name<<endl;
  resetME(name,dbe_);

  sprintf(name,"HotCellMonitor/%s/NADA_%s_testcell",type.c_str(),type.c_str());
  //// cout <<name<<endl;
  resetME(name,dbe_);
  
  sprintf(name,"HotCellMonitor/%s/NADA_%s_Energy",type.c_str(),type.c_str());
  //// cout <<name<<endl;
  resetME(name,dbe_);
  
  sprintf(name,"HotCellMonitor/%s/NADA_%s_NumNegCells",type.c_str(),type.c_str());
  //// cout <<name<<endl;
  resetME(name,dbe_);
  
  sprintf(name,"HotCellMonitor/%s/%snadaNegOccMap",type.c_str(),type.c_str());
  //// cout <<name<<endl;
  resetME(name,dbe_);
  
  sprintf(name,"HotCellMonitor/%s/%snadaNegEnergyMap",type.c_str(),type.c_str());
  //// cout <<name<<endl;
  resetME(name,dbe_);

  return;
}


void HcalHotCellClient::getSubDetThresholds(HotCellHists& hist)
{
  string type;
  if(hist.type==1)
    type="HB";
  else if(hist.type==2)
    type="HE";
  else if (hist.type==3)
    type="HO";
  else if (hist.type==4)
    type="HF";
  else if (hist.type==10)
    type="HCAL";
  else
    {
      if (debug_) cout <<"<HcalHotCellClient::getSubDetThresholds>  Unknown subdetector type: "<<hist.type<<endl;
      return;
    }

  /*
  int dummy=1; // counter to keep track of number of thresholds
  // for some !@#%* reason, the 0 counter isn't getting saved in the root file
  // investigate this later!  -- Jeff, 5/26/08
  
  while (1)
    {
      char name[256];
      sprintf(name, "%sHcal/HotCellMonitor/%s/%s_Threshold%i",process_.c_str(),type.c_str(),type.c_str(),dummy);
      MonitorElement* me = 0;
      if(dbe_) 
	{
	  me = dbe_->get(name);
	  if (me)
	      dummy++;

	  else break;
	}
      else break;
    }
  hist.thresholds=dummy;
  */
  // This isn't working anymore!  Hard code the stupid threshold values for now
  hist.thresholds=5;

  return;
} //void HcalHotCellClient::getSubDetThresholds(HotCellHists& hist)


void HcalHotCellClient::htmlSubDetOutput(HotCellHists& hist, int runNo, 
					 string htmlDir, 
					 string htmlName)
{

  ofstream htmlSubFile;
  if((hist.type<5) &&!subDetsOn_[hist.type-1]) return;
  
  string type;
  if(hist.type==1) 
    {
      type = "HB";
      htmlSubFile.open((htmlDir+"HcalHotCellClient_HB_Plots.html").c_str());
    }
  else if(hist.type==2) 
    {
      type = "HE";
      htmlSubFile.open((htmlDir+"HcalHotCellClient_HE_Plots.html").c_str());
    }
  else if(hist.type==3) 
    {
      type = "HO";
      htmlSubFile.open((htmlDir+"HcalHotCellClient_HO_Plots.html").c_str());
    }
  else if(hist.type==4) 
    {
      type = "HF";
      htmlSubFile.open((htmlDir+"HcalHotCellClient_HF_Plots.html").c_str());
    }
  else if(hist.type==10) 
    {
      type = "HCAL";
      htmlSubFile.open((htmlDir+"HcalHotCellClient_HCAL_Plots.html").c_str());
    }
  else
    {
      if (debug_)cout <<"<HcalHotCellClient::htmlSubDetOutput> Unrecognized detector type: "<<hist.type<<endl;
      return;
    }
  
  // html page header
  htmlSubFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlSubFile << "<html>  " << endl;
  htmlSubFile << "<head>  " << endl;
  htmlSubFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlSubFile << " http-equiv=\"content-type\">  " << endl;
  htmlSubFile << "  <title>Monitor: Hcal "<<type<<" HotCell Task output</title> " << endl;
  htmlSubFile << "</head>  " << endl;
  htmlSubFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlSubFile << "<body>  " << endl;
  htmlSubFile << "<br>  " << endl;
  htmlSubFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlSubFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlSubFile << " style=\"color: rgb(0, 0, 153);\">" << runNo << "</span></h2>" << endl;
  htmlSubFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlSubFile << " style=\"color: rgb(0, 0, 153);\">Hcal HotCells</span></h2> " << endl;
  htmlSubFile << "<h2>Events processed:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;

  htmlSubFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span " << endl;
  htmlSubFile << " style=\"color: rgb(0, 0, 153);\">" << ievt_ << "</span></h2>" << endl;
  htmlSubFile << "<hr>" << endl;

  htmlSubFile << "<h2><strong>"<<type<<" Hot Cell Histograms</strong></h2>" << endl;
  htmlSubFile << "<h3>" << endl;

  // Make main table
  htmlSubFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlSubFile << "cellpadding=\"10\"> " << endl;
  
  htmlSubFile << "<tr align=\"left\">" << endl;	
  htmlAnyHisto(runNo,hist.nadaOccMap,"iEta","iPhi", 92, htmlSubFile,htmlDir);
  htmlAnyHisto(runNo,hist.nadaEnergyMap,"iEta","iPhi", 100, htmlSubFile,htmlDir);
  htmlSubFile << "</tr>" << endl;

  htmlSubFile << "<tr align=\"left\">" << endl;	
  htmlAnyHisto(runNo,hist.nadaNegOccMap,"iEta","iPhi", 92, htmlSubFile,htmlDir);
  htmlAnyHisto(runNo,hist.nadaNegEnergyMap,"iEta","iPhi", 100, htmlSubFile,htmlDir);
  htmlSubFile << "</tr>" << endl;


  for (int i=0;i<hist.thresholds;i++){
    htmlSubFile << "<tr align=\"left\">" << endl;	
    htmlAnyHisto(runNo,hist.threshOccMap[i],"iEta","iPhi", 92, htmlSubFile,htmlDir);
    htmlAnyHisto(runNo,hist.threshEnergyMap[i],"iEta","iPhi", 100, htmlSubFile,htmlDir);
    htmlSubFile << "</tr>" << endl;
  }
  
  htmlSubFile << "<tr align=\"left\">" << endl;	
  htmlAnyHisto(runNo,hist.maxCellEnergy,"GeV","Evts", 92, htmlSubFile,htmlDir);
  htmlAnyHisto(runNo,hist.maxCellTime,"nS","Evts", 100, htmlSubFile,htmlDir);
  htmlSubFile << "</tr></table>" << endl;
  htmlSubFile << "<hr>" << endl;
  htmlSubFile.close();

  return;
} //void HcalHotCellClient::htmlSubDetOutput


void HcalHotCellClient::drawSubDetThresholds(HotCellHists& hist)
{
  if (!subDetsOn_[hist.type-1]) return;
  string type;
  if(hist.type==1)
    type="HB";
  else if(hist.type==2)
    type="HE";
  else if (hist.type==3)
    type="HO";
  else if (hist.type==4)
    type="HF";
  else if (hist.type==10)
    type="HCAL";
  else
    {
      if (debug_) cout <<"<HcalHotCellClient::drawSubDetThresholds>  Unknown subdetector type: "<<hist.type<<endl;
      return;
    }
  
  // new row
  htmlFile <<"<tr><td><a href=\"HcalHotCellClient_"<<type<<"_Plots.html\" target=\"_blank\">"<<type<<"</a></td>"<<endl;
  for (unsigned int i=hist.threshOccMap.size();i>0;i--)
    {
      char name[256];
      sprintf(name, "%sHcal/HotCellMonitor/%s/%sThreshold%i",process_.c_str(),type.c_str(),type.c_str(),i-1);
      MonitorElement* me = 0;
      if(dbe_) 
	{
	  me = dbe_->get(name);
	  if (me)
	    {
	     string s = me->valueString();
	     htmlFile<<"<td>"<<s.substr(2)<<" GeV</td>"<<endl;
	    }
	  else 
	    htmlFile<<"<td></td>"<<endl;
	}
    }
  htmlFile<<"</tr>"<<endl;
  return;

}// void HcalHotCellClient::drawSubDetThresholds(HotCellHists& hist)
