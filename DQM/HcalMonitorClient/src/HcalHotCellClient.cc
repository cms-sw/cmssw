#include <DQM/HcalMonitorClient/interface/HcalHotCellClient.h>

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

HcalHotCellClient::HcalHotCellClient(){}

void HcalHotCellClient::init(const ParameterSet& ps, DQMStore* dbe,string clientName){
  
  //Call the base class first
  HcalBaseClient::init(ps,dbe,clientName);

  if (debug_)
    cout <<"Initializing HcalHotCellClient from ParameterSet"<<endl;
  
  errorFrac_=ps.getUntrackedParameter<double>("hotcellErrorFrac",0.05);

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
  hcalhists.Nthresholds=max(hbhists.Nthresholds,max(hehists.Nthresholds,max(hohists.Nthresholds,hfhists.Nthresholds)));

} // void HcalHotCellClient::init()


HcalHotCellClient::~HcalHotCellClient(){
  if (debug_) cout <<"HcalHotCellClient:  Called destructor"<<endl;
  this->cleanup();
  if (debug_) cout <<"HcalHotCellClient: Finished Destructor"<<endl;
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
  if (debug_ ) cout <<" HcalHotCellClient:  Finished endJob"<<endl;
  return;
}

void HcalHotCellClient::endRun(void) 
{

  if ( debug_ ) cout << "HcalHotCellClient: endRun, jevt = " << jevt_ << endl;

  this->cleanup();  
  if (debug_ ) cout <<" HcalHotCellClient:  Finished endRun"<<endl;
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

  dqmReportMapErr_.clear(); 
  dqmReportMapWarn_.clear(); 
  dqmReportMapOther_.clear();
  dqmQtests_.clear();
  
  if (debug_)
    cout <<"HcalHotCellClient: finished cleanup"<<endl;
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
  if (debug_)
     cout <<"HcalHotCellClient::resetAllME()"<<endl;
  if(!dbe_) return;

  if (subDetsOn_[0]) resetSubDetHistograms(hbhists);
  if (subDetsOn_[1]) resetSubDetHistograms(hehists);
  if (subDetsOn_[2]) resetSubDetHistograms(hohists);
  if (subDetsOn_[3]) resetSubDetHistograms(hfhists);
  resetSubDetHistograms(hcalhists);
  return;
}

void HcalHotCellClient::htmlOutput(int runNo, string htmlDir, string htmlName)
{

  if (debug_)
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
  htmlFile << "<h3><tr><td><a href=\"index.html\"> Main DQM Page </a> </td>"<<endl;
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
  htmlFile << "<br>" << endl;

  // Main histogram table
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\"> " << endl;
  

htmlFile << "<td align=\"center\">&nbsp;&nbsp;&nbsp;<h3>Cells matching dead conditions in  at least "<<(int)(errorFrac_*100)<<"% of events</h3></td>"<<endl;
  htmlFile << "</tr>"<<endl;

  htmlFile << "<tr align=\"center\">" << endl;

  hcalhists.problemHotCells->Scale(1./ievt_);
  hcalhists.problemHotCells->SetMinimum(errorFrac_);
  htmlAnyHisto(runNo,hcalhists.problemHotCells,"i#eta","i#phi", 92, htmlFile,htmlDir);
  htmlFile<<"</tr>"<<endl;

  htmlFile << "<tr align=\"left\">" << endl;
  htmlFile <<"<tr><td>This histogram shows cells that satisfy at least one hot cell condition in at least "<<(int)(errorFrac_*100)<<"% of events.  A cell is considered hot if<br>:";
  htmlFile<<" (A) it is above a threshold energy;<br>";
  htmlFile<<" (B) if it is more than 3 sigma above its pedestal value; or <br>";
  htmlFile<<" (C) if its energy is especially large compared to its neighbors.  Detailed plots for each type of dead cell are given in the links below.<br>"<<endl;

  // Links to individual hot cell algorithms
  htmlFile <<"<table width = 90% align=\"center\"><tr align=\"center\">" <<endl;
  htmlFile <<"<td></td><td><a href=\"HcalHotCellClient_ProblemDepth_HCAL_Plots.html\">Problem Cells By Depth<br></td></tr><tr>"<<endl;
  htmlFile << "<td><a href=\"HcalHotCellClient_Threshold_HCAL_Plots.html\">RecHit Threshold Plots </a> </td>" << endl;
  htmlFile << "<td><a href=\"HcalHotCellClient_Digi_HCAL_Plots.html\">Digi Threshold Plots </a> </td>" << endl;
  htmlFile << "<td><a href=\"HcalHotCellClient_NADA_HCAL_Plots.html\">Neighboring Cell (NADA) Plots </a> </td>" << endl;
  htmlFile <<"</tr></table BORDER = \"3\" CELLPADDING = \"25\"><br>"<<endl;
  htmlFile <<"<hr>"<<endl;
  
  htmlFile <<"<h2>List of Hot Cells</h2>"<<endl;
  htmlFile <<" (Because the hot cell histogram above sums over all HCAL depths, not all entries in that histogram correspond to a single HCAL cell.  The list below shows all individual cells that are hot in at least "<<(int)(errorFrac_*100)<<"% of events.  Cells are labeled by (eta, phi, depth) index.<br><br>"<<endl;
  htmlFile <<"<table width=75%align = \"center\"><tr align=\"center\">" <<endl;
  htmlFile <<"<td> Hot Cells</td><td align=\"center\"> Fraction of Events in which cells are bad (%)</td></tr>"<<endl;

  // Dump out hot cell candidates
  for (unsigned int depth=0;depth<4;++depth)
    {
      if (hcalhists.problemHotCells_DEPTH.size()<=depth) continue;
      if (hcalhists.problemHotCells_DEPTH[depth]==NULL) continue;
      int etabins = hcalhists.problemHotCells_DEPTH[depth]->GetNbinsX();
      int phibins = hcalhists.problemHotCells_DEPTH[depth]->GetNbinsY();
      float etaMin=hcalhists.problemHotCells_DEPTH[depth]->GetXaxis()->GetXmin();
      float phiMin=hcalhists.problemHotCells_DEPTH[depth]->GetYaxis()->GetXmin();
      
      int eta,phi;
      for (int ieta=1;ieta<=etabins;++ieta)
	{
	  for (int iphi=1; iphi<=phibins;++iphi)
	    {
	      eta=ieta+int(etaMin)-1;
	      phi=iphi+int(phiMin)-1;

	      if (hbhists.problemHotCells_DEPTH[depth]->GetBinContent(ieta,iphi)>=errorFrac_*ievt_)
		{
		  htmlFile<<"<td align=\"center\"> HB ("<<eta<<", "<<phi<<", "<<depth+1<<") </td><td align=\"center\"> "<<100.*hbhists.problemHotCells_DEPTH[depth]->GetBinContent(ieta,iphi)/ievt_<<"%</td></tr>"<<endl;
		}
	      if (hehists.problemHotCells_DEPTH[depth]->GetBinContent(ieta,iphi)>=errorFrac_*ievt_)
		{
		  htmlFile<<"<td align=\"center\"> HE ("<<eta<<", "<<phi<<", "<<depth+1<<") </td><td align=\"center\"> "<<100.*hehists.problemHotCells_DEPTH[depth]->GetBinContent(ieta,iphi)/ievt_<<"%</td></tr>"<<endl;
		}
	      if (hohists.problemHotCells_DEPTH[depth]->GetBinContent(ieta,iphi)>=errorFrac_*ievt_)
		{
		  htmlFile<<"<td align=\"center\"> HO ("<<eta<<", "<<phi<<", "<<depth+1<<") </td><td align=\"center\"> "<<100.*hohists.problemHotCells_DEPTH[depth]->GetBinContent(ieta,iphi)/ievt_<<"%</td></tr>"<<endl;
		}
	      if (hfhists.problemHotCells_DEPTH[depth]->GetBinContent(ieta,iphi)>=errorFrac_*ievt_)
		{
		  htmlFile<<"<td align=\"center\"> HF ("<<eta<<", "<<phi<<", "<<depth+1<<") </td><td align=\"center\"> "<<100.*hfhists.problemHotCells_DEPTH[depth]->GetBinContent(ieta,iphi)/ievt_<<"%</td></tr>"<<endl;
		}
	      
	    } // for (int iphi =1...
	  
	}// for (int ieta=1...
    } // for (int depth=0...

  htmlFile << "</table>" <<endl;
  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;
  htmlFile.close();


  drawThresholdPlots(hcalhists, runNo, htmlDir);
  drawThresholdPlots(hbhists, runNo,htmlDir);
  drawThresholdPlots(hehists, runNo,htmlDir);
  drawThresholdPlots(hohists, runNo,htmlDir);
  drawThresholdPlots(hfhists, runNo,htmlDir);

  drawDigiPlots(hcalhists, runNo, htmlDir);
  drawDigiPlots(hbhists, runNo,htmlDir);
  drawDigiPlots(hehists, runNo,htmlDir);
  drawDigiPlots(hohists, runNo,htmlDir);
  drawDigiPlots(hfhists, runNo,htmlDir);

  drawNADAPlots(hcalhists, runNo, htmlDir);
  drawNADAPlots(hbhists, runNo,htmlDir);
  drawNADAPlots(hehists, runNo,htmlDir);
  drawNADAPlots(hohists, runNo,htmlDir);
  drawNADAPlots(hfhists, runNo,htmlDir);

  drawProblemPlots(hcalhists, runNo, htmlDir);
  drawProblemPlots(hbhists, runNo,htmlDir);
  drawProblemPlots(hehists, runNo,htmlDir);
  drawProblemPlots(hohists, runNo,htmlDir);
  drawProblemPlots(hfhists, runNo,htmlDir);

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
  for (unsigned int i=hcalhists.threshOccMap.size();i>0;--i)
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
  htmlFile.close();
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
  sprintf(name,"%sHcal/HotCellMonitor/%s/%s_OccupancyMap_NADA",process_.c_str(),type.c_str(), type.c_str());
  sprintf(meTitle,"%s NADA Occupancy",type.c_str()); 
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
  sprintf(meTitle,"%sHcal/HotCellMonitor/%s/%s_OccupancyMap_HotCell_Threshold%i",process_.c_str(),type.c_str(), type.c_str(),hist.Nthresholds);
  //sprintf(name,"%s Maximum Energy Cell",type.c_str()); 
  sprintf(name,"%s Threshold #%i Cell",type.c_str(),hist.Nthresholds);
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
  for (int i=0;i<hist.Nthresholds;i++)
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

void HcalHotCellClient::loadHistograms(TFile* infile)
{
 if (debug_)
   cout <<"HcalHotCellClient::loadHistograms()"<<endl;
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

  // problem cell histograms
  hist.problemHotCells=0;
  hist.problemHotCells_DEPTH.clear();

  // Digi histograms
  hist.abovePedSigma=0;

  // Threshold histograms
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
  hist.nadaEnergy=0;
  hist.nadaNumNegCells=0;
  hist.nadaNegOccMap=0;
  hist.nadaNegEnergyMap=0;

  hist.nadaOccMapDepth.clear();
  hist.nadaNegOccMapDepth.clear();
  hist.nadaEnergyMapDepth.clear();
  hist.nadaNegEnergyMapDepth.clear();

  return;
} // void HcalHotCellClient::clearHists(HotCellHists& hist)


void HcalHotCellClient::deleteHists(HotCellHists& hist)
{
  
  if(debug_) cout <<"Deleting HcalHotCell histograms for HCAL type: "<<hist.type<<endl;

  if (hist.problemHotCells) delete hist.problemHotCells;
  hist.problemHotCells_DEPTH.clear();

  if (hist.abovePedSigma) delete hist.abovePedSigma;

  if (hist.maxCellOccMap) delete hist.maxCellOccMap;
  if (hist.maxCellEnergyMap) delete hist.maxCellEnergyMap;
  if (hist.maxCellEnergy) delete hist.maxCellEnergy;
  if (hist.maxCellTime) delete hist.maxCellTime;
  if (hist.maxCellID) delete hist.maxCellID;
  
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
  if (hist.nadaOccMap) delete hist.nadaOccMap;
  if (hist.nadaEnergyMap) delete hist.nadaEnergyMap;
  if (hist.nadaNumHotCells) delete hist.nadaNumHotCells;
  if (hist.nadaEnergy) delete hist.nadaEnergy;
  if (hist.nadaNumNegCells) delete hist.nadaNumNegCells;
  if (hist.nadaNegOccMap) delete hist.nadaNegOccMap;
  if (hist.nadaNegEnergyMap) delete hist.nadaNegEnergyMap;

  hist.nadaOccMapDepth.clear();
  hist.nadaNegOccMapDepth.clear();
  hist.nadaEnergyMapDepth.clear();
  hist.nadaNegEnergyMapDepth.clear();

  if (debug_) cout <<"Finished deleting HcalHotCell histograms from HCAL type: "<<hist.type<<endl;
  return;
}


void HcalHotCellClient::getSubDetHistograms(HotCellHists& hist)
{
   if (debug_)
     cout <<"HcalHotCellClient::getSubDetHistograms()"<<endl;
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
  hist.problemHotCells_DEPTH.clear();
  //hist.digiPedestalPlots_depth.clear();

  if (debug_)
    cout <<"Getting HcalHotCell Subdetector Histograms for Subdetector:  "<<type<<endl;

  // Make dummy histogram that is used by template function getAnyHisto to determine hist type -- is there a better way to get this info?

  TH2F* dummy2D = new TH2F();
  

  for (int i=0;i<hist.Nthresholds;++i)
    {
      if (i==0)
	sprintf(name,"HotCellMonitor/%s/%s_OccupancyMap_HotCell_Threshold%i",type.c_str(),type.c_str(),i);
      else
	sprintf(name,"HotCellMonitor/%s/expertPlots/Thresholds/%s_OccupancyMap_HotCell_Threshold%i",type.c_str(),type.c_str(),i);
      // cout <<name<<endl;
      TH2F* temp1 = getAnyHisto(dummy2D, name,process_,dbe_,debug_,cloneME_);
      if (temp1!=NULL) hist.threshOccMap.push_back(temp1);
      
      sprintf(name,"HotCellMonitor/%s/expertPlots/Thresholds/%s_HotCell_EnergyMap_Thresh%i",type.c_str(),type.c_str(),i);
      // cout <<name<<endl;
      TH2F* temp2 = getAnyHisto(dummy2D, name,process_,dbe_,debug_,cloneME_);
      if (temp2!=NULL) hist.threshEnergyMap.push_back(temp2);
      
      sprintf(name,"HotCellMonitor/%s/Diagnostics/Depth1/%s_OccupancyMap_HotCell_Threshold%iDepth1",type.c_str(),type.c_str(),i);
      // cout <<name<<endl;
      TH2F* temp3 = getAnyHisto(dummy2D, name,process_,dbe_,debug_,cloneME_);
      if (temp3!=NULL) hist.threshOccMapDepth1.push_back(temp3);

      /*
	// For reasons passing my understanding, this particular set of histograms causes a crash in HE.  Everything else works fine.
      sprintf(name,"HotCellMonitor/%s/Diagnostics/Depth1/%s_HotCell_EnergyMap_Thresh%iDepth1",type.c_str(),type.c_str(),i);
      // cout <<name<<endl;
      TH2F* temp4 = getAnyHisto(dummy2D, name,process_,dbe_,debug_,cloneME_);
      if (temp4!=NULL) hist.threshEnergyMapDepth1.push_back(temp4);
      */

      sprintf(name,"HotCellMonitor/%s/Diagnostics/Depth2/%s_OccupancyMap_HotCell_Threshold%iDepth2",type.c_str(),type.c_str(),i);
      // cout <<name<<endl;
      TH2F* temp5 = getAnyHisto(dummy2D, name,process_,dbe_,debug_,cloneME_);
      if (temp5!=NULL) hist.threshOccMapDepth2.push_back(temp5);

      sprintf(name,"HotCellMonitor/%s/Diagnostics/Depth2/%s_HotCell_EnergyMap_Thresh%iDepth2",type.c_str(),type.c_str(),i);
      // cout <<name<<endl;
      // This used to cause seg fault, but now works fine -- problem passed to HEThresh1Depth1 -- memory issue?
      TH2F* temp6 = getAnyHisto(dummy2D, name,process_,dbe_,debug_,cloneME_);
      if (temp6!=NULL) hist.threshEnergyMapDepth2.push_back(temp6);


      sprintf(name,"HotCellMonitor/%s/Diagnostics/Depth3/%s_OccupancyMap_HotCell_Threshold%iDepth3",type.c_str(),type.c_str(),i);
      // cout <<name<<endl;
      TH2F* temp7 = getAnyHisto(dummy2D, name,process_,dbe_,debug_,cloneME_);
      if (temp7!=NULL) hist.threshOccMapDepth3.push_back(temp7);
     
      sprintf(name,"HotCellMonitor/%s/Diagnostics/Depth3/%s_HotCell_EnergyMap_Thresh%iDepth3",type.c_str(),type.c_str(),i);
      // cout <<name<<endl;
      TH2F* temp8 = getAnyHisto(dummy2D, name,process_,dbe_,debug_,cloneME_);
      if (temp8!=NULL) hist.threshEnergyMapDepth3.push_back(temp8);
      
      sprintf(name,"HotCellMonitor/%s/Diagnostics/Depth4/%s_OccupancyMap_HotCell_Threshold%iDepth4",type.c_str(),type.c_str(),i);
      // cout <<name<<endl;
      TH2F* temp9 = getAnyHisto(dummy2D, name,process_,dbe_,debug_,cloneME_);
      if (temp9!=NULL) hist.threshOccMapDepth4.push_back(temp9);

      sprintf(name,"HotCellMonitor/%s/Diagnostics/Depth4/%s_HotCell_EnergyMap_Thresh%iDepth4",type.c_str(),type.c_str(),i);
      TH2F* temp10 = getAnyHisto(dummy2D, name,process_,dbe_,debug_,cloneME_);
      // cout <<name<<endl;
      if (temp10!=NULL) hist.threshEnergyMapDepth4.push_back(temp10);
    } //for (int i=0;i<hist.Nthresholds;++i)
  
  sprintf(name,"HotCellMonitor/%s/%sProblemHotCells",type.c_str(),type.c_str());
  hist.problemHotCells = getAnyHisto(dummy2D, name, process_, dbe_,debug_,cloneME_);      
  // Trying to push back directly causes "target memory has disappeared error.  Do it the long way...
  sprintf(name,"HotCellMonitor/%s/expertPlots/%sProblemHotCells_depth1",type.c_str(),type.c_str());
  TH2F* temp11=getAnyHisto(new TH2F(),name,process_, dbe_, debug_, cloneME_);
  sprintf(name,"HotCellMonitor/%s/expertPlots/%sProblemHotCells_depth2",type.c_str(),type.c_str());
  TH2F* temp12=getAnyHisto(new TH2F(),name,process_, dbe_, debug_, cloneME_);
  sprintf(name,"HotCellMonitor/%s/expertPlots/%sProblemHotCells_depth3",type.c_str(),type.c_str());
  TH2F* temp13=getAnyHisto(new TH2F(),name,process_, dbe_, debug_, cloneME_);
  sprintf(name,"HotCellMonitor/%s/expertPlots/%sProblemHotCells_depth4",type.c_str(),type.c_str());
  TH2F* temp14=getAnyHisto(new TH2F(),name,process_, dbe_, debug_, cloneME_);

  hist.problemHotCells_DEPTH.push_back(temp11); 
  hist.problemHotCells_DEPTH.push_back(temp12);
  hist.problemHotCells_DEPTH.push_back(temp13);
  hist.problemHotCells_DEPTH.push_back(temp14);

  // Digi Histograms
  sprintf(name,"HotCellMonitor/%s/%s_OccupancyMap_HotCell_Digi",type.c_str(),type.c_str());
  hist.abovePedSigma=getAnyHisto(new TH2F(),name,process_, dbe_, debug_, cloneME_);
  // Add in diagnostic histograms at some point?  Or leave them in root file?

  sprintf(name,"HotCellMonitor/%s/expertPlots/%s_OccupancyMap_MaxCell",type.c_str(),type.c_str());
  // cout <<name<<endl;
  hist.maxCellOccMap = getAnyHisto(dummy2D, name, process_, dbe_,debug_,cloneME_);      
  sprintf(name,"HotCellMonitor/%s/expertPlots/MaxCell/%s_HotCell_EnergyMap_MaxCell",type.c_str(),type.c_str());
  // cout <<name<<endl;
  hist.maxCellEnergyMap = getAnyHisto(dummy2D, name, process_, dbe_,debug_,cloneME_);
  
  sprintf(name,"HotCellMonitor/%s/expertPlots/MaxCell/%sHotCellEnergyMaxCell",type.c_str(),type.c_str());
  hist.maxCellEnergy = getAnyHisto( new TH1F(), name, process_, dbe_,debug_,cloneME_);
  // cout <<name<<endl;
  sprintf(name,"HotCellMonitor/%s/expertPlots/MaxCell/%sHotCellTimeMaxCell",type.c_str(),type.c_str());
  hist.maxCellTime = getAnyHisto( new TH1F(), name, process_, dbe_,debug_,cloneME_);    
  // cout <<name<<endl;
  sprintf(name,"HotCellMonitor/%s/expertPlots/MaxCell/%sHotCellIDMaxCell",type.c_str(),type.c_str());
  hist.maxCellID = getAnyHisto( new TH1F(), name, process_, dbe_,debug_,cloneME_);    
  // cout <<name<<endl;

  // NADA histograms
  sprintf(name,"HotCellMonitor/%s/%s_OccupancyMap_NADA",type.c_str(),type.c_str());
  hist.nadaOccMap=getAnyHisto(dummy2D, name, process_, dbe_,debug_,cloneME_);
  //cout <<"NAME = "<<name<<endl;
  sprintf(name,"HotCellMonitor/%s/expertPlots/NADA/%snadaEnergyMap",type.c_str(),type.c_str());
  hist.nadaEnergyMap=getAnyHisto(dummy2D, name, process_, dbe_,debug_,cloneME_);
  // cout <<name<<endl;
  sprintf(name,"HotCellMonitor/%s/expertPlots/NADA/%snadaNumHotCells",type.c_str(),type.c_str());
  hist.nadaNumHotCells = getAnyHisto( new TH1F(), name, process_, dbe_,debug_,cloneME_);
   // cout <<name<<endl;

  sprintf(name,"HotCellMonitor/%s/expertPlots/NADA/%sEnergy",type.c_str(),type.c_str());
  hist.nadaEnergy = getAnyHisto( new TH1F(), name, process_, dbe_,debug_,cloneME_); 
  // cout <<name<<endl;

  sprintf(name,"HotCellMonitor/%s/expertPlots/NADA/%snadaNumNegCells",type.c_str(),type.c_str());
  hist.nadaNumNegCells = getAnyHisto( new TH1F(), name, process_, dbe_,debug_,cloneME_); 
  // cout <<name<<endl;

  sprintf(name,"HotCellMonitor/%s/expertPlots/NADA/%snadaNegOccMap",type.c_str(),type.c_str());
  hist.nadaNegOccMap = getAnyHisto(dummy2D, name, process_, dbe_,debug_,cloneME_); 
  // cout <<name<<endl;

  sprintf(name,"HotCellMonitor/%s/expertPlots/NADA/%snadaNegEnergyMap",type.c_str(),type.c_str());
  hist.nadaNegEnergyMap = getAnyHisto(dummy2D, name, process_, dbe_,debug_,cloneME_); 
  // cout <<name<<endl;


  // Get NADA histograms by depth
  for (int d=0;d<4;++d)
    {
      sprintf(name,"HotCellMonitor/%s/Diagnostics/Depth%i/%snadaOccMapDepth%i",
	      type.c_str(),d+1,type.c_str(),d+1);
      
      TH2F* temp15 = getAnyHisto(dummy2D, name, process_, 
				dbe_,debug_,cloneME_); 
      hist.nadaOccMapDepth.push_back(temp15);

      sprintf(name,"HotCellMonitor/%s/Diagnostics/Depth%i/%snadaNegOccMapDepth%i",
	      type.c_str(),d+1,type.c_str(),d+1);
      TH2F* temp16 = getAnyHisto(dummy2D, name, process_, 
				dbe_,debug_,cloneME_); 
      hist.nadaNegOccMapDepth.push_back(temp16);

      sprintf(name,"HotCellMonitor/%s/Diagnostics/Depth%i/%snadaEnergyMapDepth%i",
	      type.c_str(),d+1,type.c_str(),d+1);
      TH2F* temp17 = getAnyHisto(dummy2D, name, process_, 
				dbe_,debug_,cloneME_); 
      hist.nadaEnergyMapDepth.push_back(temp17);

      sprintf(name,"HotCellMonitor/%s/Diagnostics/Depth%i/%snadaNegEnergyMapDepth%i",
	      type.c_str(),d+1,type.c_str(),d+1);
      TH2F* temp18 = getAnyHisto(dummy2D, name, process_,
				dbe_,debug_,cloneME_); 
      hist.nadaNegEnergyMapDepth.push_back(temp18);
    }
  return;
} // void HcalHotCellClient::getSubDetHistograms(HotCellHists& hist)


void HcalHotCellClient::getSubDetHistogramsFromFile(HotCellHists& hist, TFile* infile)
{
   if (debug_)
     cout <<"HcalHotCellClient::getSubDetHistogramsFromFile()"<<endl;
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

  hist.nadaOccMapDepth.clear();
  hist.nadaNegOccMapDepth.clear();
  hist.nadaEnergyMapDepth.clear();
  hist.nadaNegEnergyMapDepth.clear();

  for (int i=0;i<hist.Nthresholds;++i)
    {
      if (i==0)
	sprintf(name,"HotCellMonitor/%s/%s_OccupancyMap_HotCell_Threshold%i",type.c_str(),type.c_str(),i);
      else
	sprintf(name,"HotCellMonitor/%s/expertPlots/Thresholds/%s_OccupancyMap_HotCell_Threshold%i",type.c_str(),type.c_str(),i);
      hist.threshOccMap.push_back((TH2F*)infile->Get(name));
      sprintf(name,"HotCellMonitor/%s/expertPlots/Thresholds/%s_HotCell_EnergyMap_Thresh%i",type.c_str(),type.c_str(),i);
      hist.threshEnergyMap.push_back((TH2F*)infile->Get(name));
      sprintf(name,"HotCellMonitor/%s/Diagnostics/Depth1/%s_OccupancyMap_HotCell_Threshold%iDepth1",type.c_str(),type.c_str(),i);
      hist.threshOccMapDepth1.push_back((TH2F*)infile->Get(name));
      sprintf(name,"HotCellMonitor/%s/Diagnostics/Depth1/%s_HotCell_EnergyMap_Thresh%iDepth1",type.c_str(),type.c_str(),i);
      hist.threshEnergyMapDepth1.push_back((TH2F*)infile->Get(name));
      sprintf(name,"HotCellMonitor/%s/Diagnostics/Depth2/%s_OccupancyMap_HotCell_Threshold%iDepth2",type.c_str(),type.c_str(),i);
      hist.threshOccMapDepth2.push_back((TH2F*)infile->Get(name));
      sprintf(name,"HotCellMonitor/%s/Diagnostics/Depth2/%s_HotCell_EnergyMap_Thresh%iDepth2",type.c_str(),type.c_str(),i);
      hist.threshEnergyMapDepth2.push_back((TH2F*)infile->Get(name));
  sprintf(name,"HotCellMonitor/%s/Diagnostics/Depth3/%s_OccupancyMap_HotCell_Threshold%iDepth3",type.c_str(),type.c_str(),i);
      hist.threshOccMapDepth3.push_back((TH2F*)infile->Get(name));
      sprintf(name,"HotCellMonitor/%s/Diagnostics/Depth3/%s_HotCell_EnergyMap_Thresh%iDepth3",type.c_str(),type.c_str(),i);
      hist.threshEnergyMapDepth3.push_back((TH2F*)infile->Get(name));
      sprintf(name,"HotCellMonitor/%s/Diagnostics/Depth4/%s_OccupancyMap_HotCell_Threshold%iDepth4",type.c_str(),type.c_str(),i);
      hist.threshOccMapDepth4.push_back((TH2F*)infile->Get(name));
      sprintf(name,"HotCellMonitor/%s/Diagnostics/Depth4/%s_HotCell_EnergyMap_Thresh%iDepth4",type.c_str(),type.c_str(),i);
      hist.threshEnergyMapDepth4.push_back((TH2F*)infile->Get(name));
    } //for (int i=0;i<hist.Nthresholds;++i)

 //List of problem cells
  sprintf(name,"HotCellMonitor/%s/%sProblemHotCells",type.c_str(),type.c_str());
  hist.problemHotCells = (TH2F*)infile->Get(name);

  sprintf(name,"HotCellMonitor/%s/expertPlots/%sProblemHotCells_depth1",type.c_str(),type.c_str());
  hist.problemHotCells_DEPTH.push_back((TH2F*)infile->Get(name));
  sprintf(name,"HotCellMonitor/%s/expertPlots/%sProblemHotCells_depth2",type.c_str(),type.c_str());
  hist.problemHotCells_DEPTH.push_back((TH2F*)infile->Get(name));
  sprintf(name,"HotCellMonitor/%s/expertPlots/%sProblemHotCells_depth3",type.c_str(),type.c_str());
  hist.problemHotCells_DEPTH.push_back((TH2F*)infile->Get(name));
  sprintf(name,"HotCellMonitor/%s/expertPlots/%sProblemHotCells_depth4",type.c_str(),type.c_str());
  hist.problemHotCells_DEPTH.push_back((TH2F*)infile->Get(name));


  sprintf(name,"HotCellMonitor/%s/expertPlots/%s_OccupancyMap_MaxCell",type.c_str(),type.c_str());
  hist.maxCellOccMap = (TH2F*)infile->Get(name);      
  sprintf(name,"HotCellMonitor/%s/expertPlots/MaxCell/%s_HotCell_EnergyMap_MaxCell",type.c_str(),type.c_str());
  hist.maxCellEnergyMap = (TH2F*)infile->Get(name);
  
  sprintf(name,"HotCellMonitor/%s/expertPlots/MaxCell/%sHotCellEnergyMaxCell",type.c_str(),type.c_str());
  hist.maxCellEnergy = (TH1F*)infile->Get(name);
  sprintf(name,"HotCellMonitor/%s/expertPlots/MaxCell/%sHotCellTimeMaxCell",type.c_str(),type.c_str());
  hist.maxCellTime = (TH1F*)infile->Get(name);    
  sprintf(name,"HotCellMonitor/%s/expertPlots/MaxCell/%sHotCellIDMaxCell",type.c_str(),type.c_str());
  hist.maxCellID = (TH1F*)infile->Get(name);    

  // NADA histograms
  sprintf(name,"HotCellMonitor/%s/%s_OccupancyMap_NADA",type.c_str(),type.c_str());
  hist.nadaOccMap=(TH2F*)infile->Get(name);
  sprintf(name,"HotCellMonitor/%s/%snadaEnergyMap",type.c_str(),type.c_str());
  hist.nadaEnergyMap=(TH2F*)infile->Get(name);
  sprintf(name,"HotCellMonitor/%s/NADA_%s_NumHotCells",type.c_str(),type.c_str());
  hist.nadaNumHotCells = (TH1F*)infile->Get(name); 

  sprintf(name,"HotCellMonitor/%s/NADA_%s_Energy",type.c_str(),type.c_str());
  hist.nadaEnergy = (TH1F*)infile->Get(name); 
  sprintf(name,"HotCellMonitor/%s/NADA_%s_NumNegCells",type.c_str(),type.c_str());
  hist.nadaNumNegCells = (TH1F*)infile->Get(name); 
  sprintf(name,"HotCellMonitor/%s/%snadaNegOccMap",type.c_str(),type.c_str());
  hist.nadaNegOccMap = (TH2F*)infile->Get(name); 
  sprintf(name,"HotCellMonitor/%s/%snadaNegEnergyMap",type.c_str(),type.c_str());
  hist.nadaNegEnergyMap = (TH2F*)infile->Get(name); 

  for (int d=0;d<4;++d)
    {
      sprintf(name,"HotCellMonitor/%s/Diagnostics/Depth%i/%snadaOccMapDepth%i",
	      type.c_str(),d+1,type.c_str(),d+1);
      hist.nadaOccMapDepth.push_back((TH2F*)infile->Get(name));
      sprintf(name,"HotCellMonitor/%s/Diagnostics/Depth%i/%snadaNegOccMapDepth%i",
	      type.c_str(),d+1,type.c_str(),d+1);
      hist.nadaNegOccMapDepth.push_back((TH2F*)infile->Get(name));
      sprintf(name,"HotCellMonitor/%s/Diagnostics/Depth%i/%snadaEnergyMapDepth%i",
	      type.c_str(),d+1,type.c_str(),d+1);
      hist.nadaEnergyMapDepth.push_back((TH2F*)infile->Get(name));
      sprintf(name,"HotCellMonitor/%s/Diagnostics/Depth%i/%snadaNegEnergyMapDepth%i",
	      type.c_str(),d+1,type.c_str(),d+1);
      hist.nadaNegEnergyMapDepth.push_back((TH2F*)infile->Get(name));
    }
  return;
}



void HcalHotCellClient::resetSubDetHistograms(HotCellHists& hist)
{
   if (debug_)
     cout <<"HcalHotCellClient::resetSubDetHistograms()"<<endl;
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

  sprintf(name,"HotCellMonitor/%s/%sProblemHotCells",type.c_str(),type.c_str());
  resetME(name,dbe_);

  for (int i=0;i<hist.Nthresholds;++i)
    {
      if (i==0)
	sprintf(name,"HotCellMonitor/%s/%s_HotCell_EnergyMap_Thresh%i",type.c_str(),type.c_str(),i);
      else
	sprintf(name,"HotCellMonitor/%s/expertPlots/Thresholds/%s_HotCell_EnergyMap_Thresh%i",type.c_str(),type.c_str(),i);
    //cout <<"NAME = "<<name<<endl;
    resetME(name,dbe_);
    sprintf(name,"HotCellMonitor/%s/expertPlots/Thresholds/%s_OccupancyMap_HotCell_Threshold%i",type.c_str(),type.c_str(),i);
    //cout <<"NAME = "<<name<<endl;
    resetME(name,dbe_);
    sprintf(name,"HotCellMonitor/%s/Diagnostics/Depth1/%s_HotCell_EnergyMap_Thresh%iDepth1",type.c_str(),type.c_str(),i);
    //cout <<"NAME = "<<name<<endl;
    resetME(name,dbe_);
    sprintf(name,"HotCellMonitor/%s/Diagnostics/Depth1/%s_OccupancyMap_HotCell_Threshold%iDepth1",type.c_str(),type.c_str(),i);
    //cout <<"NAME = "<<name<<endl;
    resetME(name,dbe_);
    sprintf(name,"HotCellMonitor/%s/Diagnostics/Depth2/%s_HotCell_EnergyMap_Thresh%iDepth2",type.c_str(),type.c_str(),i);
    //cout <<"NAME = "<<name<<endl;
    resetME(name,dbe_);
    sprintf(name,"HotCellMonitor/%s/Diagnostics/Depth2/%s_OccupancyMap_HotCell_Threshold%iDepth2",type.c_str(),type.c_str(),i);
    //cout <<"NAME = "<<name<<endl;
    resetME(name,dbe_);
    sprintf(name,"HotCellMonitor/%s/Diagnostics/Depth3/%s_HotCell_EnergyMap_Thresh%iDepth3",type.c_str(),type.c_str(),i);
    //cout <<"NAME = "<<name<<endl;
    resetME(name,dbe_);
    sprintf(name,"HotCellMonitor/%s/Diagnostics/Depth3/%s_OccupancyMap_HotCell_Threshold%iDepth3",type.c_str(),type.c_str(),i);
    //cout <<"NAME = "<<name<<endl;
    resetME(name,dbe_);
sprintf(name,"HotCellMonitor/%s/Diagnostics/Depth4/%s_HotCell_EnergyMap_Thresh%iDepth4",type.c_str(),type.c_str(),i);
    //cout <<"NAME = "<<name<<endl;
    resetME(name,dbe_);
    sprintf(name,"HotCellMonitor/%s/Diagnostics/Depth4/%s_OccupancyMap_HotCell_Threshold%iDepth4",type.c_str(),type.c_str(),i);
    //cout <<"NAME = "<<name<<endl;
    resetME(name,dbe_);


  }
  
  sprintf(name,"HotCellMonitor/%s/expertPlots/MaxCell/%s_HotCell_EnergyMap_MaxCell",type.c_str(),type.c_str());
  //// cout <<name<<endl;
  resetME(name,dbe_);
  
  sprintf(name,"HotCellMonitor/%s/expertPlots/%s_OccupancyMap_MaxCell",type.c_str(),type.c_str());
  //// cout <<name<<endl;
  resetME(name,dbe_);
  
  sprintf(name,"HotCellMonitor/%s/expertPlots/MaxCell/%sHotCellEnergyMaxCell",type.c_str(),type.c_str());
  //// cout <<name<<endl;
  resetME(name,dbe_);

  sprintf(name,"HotCellMonitor/%s/expertPlots/MaxCell/%sHotCellTimeMaxCell",type.c_str(),type.c_str());
  //// cout <<name<<endl;
  resetME(name,dbe_);

  sprintf(name,"HotCellMonitor/%s/expertPlots/%sHotCellIDMaxCell",type.c_str(),type.c_str());
  //// cout <<name<<endl;
  resetME(name,dbe_);

  // NADA histograms
  sprintf(name,"HotCellMonitor/%s/%s_OccupancyMap_NADA",type.c_str(),type.c_str());
  //// cout <<name<<endl;
  resetME(name,dbe_);

  sprintf(name,"HotCellMonitor/%s/expertPlots/NADA/%snadaEnergyMap",type.c_str(),type.c_str());
  //// cout <<name<<endl;
  resetME(name,dbe_);

  sprintf(name,"HotCellMonitor/%s/expertPlots/NADA/%snadaNumHotCells",type.c_str(),type.c_str());
  //// cout <<name<<endl;
  resetME(name,dbe_);
  
  sprintf(name,"HotCellMonitor/%s/expertPlots/NADA/%sEnergy",type.c_str(),type.c_str());
  //// cout <<name<<endl;
  resetME(name,dbe_);
  
  sprintf(name,"HotCellMonitor/%s/expertPlots/NADA/%snadaNumNegCells",type.c_str(),type.c_str());
  //// cout <<name<<endl;
  resetME(name,dbe_);
  
  sprintf(name,"HotCellMonitor/%s/expertPlots/NADA/%snadaNegOccMap",type.c_str(),type.c_str());
  //// cout <<name<<endl;
  resetME(name,dbe_);
  
  sprintf(name,"HotCellMonitor/%s/expertPlots/NADA/%snadaNegEnergyMap",type.c_str(),type.c_str());
  //// cout <<name<<endl;
  resetME(name,dbe_);

  for (int d=0;d<4;++d)
    {
      sprintf(name,"HotCellMonitor/%s/Diagnostics/Depth%i/%snadaOccMapDepth%i",
	      type.c_str(),d+1,type.c_str(),d+1);
      resetME(name,dbe_);
      sprintf(name,"HotCellMonitor/%s/Diagnostics/Depth%i/%snadaNegOccMapDepth%i",
	      type.c_str(),d+1,type.c_str(),d+1);
      resetME(name,dbe_);
      sprintf(name,"HotCellMonitor/%s/Diagnostics/Depth%i/%snadaEnergyMapDepth%i",
	      type.c_str(),d+1,type.c_str(),d+1);
      resetME(name,dbe_);
      sprintf(name,"HotCellMonitor/%s/Diagnostics/Depth%i/%snadaNegEnergyMapDepth%i",
	      type.c_str(),d+1,type.c_str(),d+1);
      resetME(name,dbe_);
    }
  return;
}


void HcalHotCellClient::getSubDetThresholds(HotCellHists& hist)
{
  if (debug_)
    cout <<"HcalHotCellClient::getSubDetThresholds()"<<endl;
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

  int dummy=0; // counter to keep track of number of thresholds
  while (1)
    {
      char name[256];
      sprintf(name, "%sHcal/HotCellMonitor/%s/%sHotCellThreshold%i",process_.c_str(),type.c_str(),type.c_str(),dummy);
      MonitorElement* me = 0;
      if(dbe_) 
	{
	  me = dbe_->get(name);
	  if (me)
	    {
	      ++dummy;
	      string s = me->valueString();
	      // Split string to get number value ( "f = 15" is split to just  "15") 
	      hist.thresholds.push_back(s.substr(2));
	    }
	  else break;
	}
      else break;
    }
  hist.Nthresholds=dummy;
  return;
} //void HcalHotCellClient::getSubDetThresholds(HotCellHists& hist)


void HcalHotCellClient::htmlSubDetOutput(HotCellHists& hist, int runNo, 
					 string htmlDir, 
					 string htmlName)
{
  if (debug_)
     cout <<"HcalHotCellClient::htmlSubDetOutput()"<<endl;

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
  htmlAnyHisto(runNo,hist.nadaOccMap,"i#eta","i#phi", 92, htmlSubFile,htmlDir);
  htmlAnyHisto(runNo,hist.nadaEnergyMap,"i#eta","i#phi", 100, htmlSubFile,htmlDir);
  htmlSubFile << "</tr>" << endl;

  htmlSubFile << "<tr align=\"left\">" << endl;	
  htmlAnyHisto(runNo,hist.nadaNegOccMap,"i#eta","i#phi", 92, htmlSubFile,htmlDir);
  htmlAnyHisto(runNo,hist.nadaNegEnergyMap,"i#eta","i#phi", 100, htmlSubFile,htmlDir);
  htmlSubFile << "</tr>" << endl;


  for (int i=0;i<hist.Nthresholds;++i)
    {
      htmlSubFile << "<tr align=\"left\">" << endl;	
      if ((int)(hist.threshOccMap.size())>i)
	htmlAnyHisto(runNo,hist.threshOccMap[i],"i#eta","i#phi", 92, htmlSubFile,htmlDir);
      if ((int)hist.threshEnergyMap.size()>i)
	 htmlAnyHisto(runNo,hist.threshEnergyMap[i],"i#eta","i#phi", 100, htmlSubFile,htmlDir);
      htmlSubFile << "</tr>" << endl;
    }
  
  htmlSubFile << "<tr align=\"left\">" << endl;	
  htmlAnyHisto(runNo,hist.abovePedSigma,"i#eta","i#phi", 92, htmlSubFile,htmlDir);
  htmlSubFile << "</tr><tr align=\"left\">" << endl;
  htmlAnyHisto(runNo,hist.maxCellEnergy,"GeV","Evts", 92, htmlSubFile,htmlDir);
  htmlAnyHisto(runNo,hist.maxCellTime,"nS","Evts", 100, htmlSubFile,htmlDir);
  htmlSubFile << "</tr></table>" << endl;
  htmlSubFile << "<hr>" << endl;

  // html page footer
  htmlSubFile << "</body> " << endl;
  htmlSubFile << "</html> " << endl;
  htmlSubFile.close();

  return;
} //void HcalHotCellClient::htmlSubDetOutput


void HcalHotCellClient::drawSubDetThresholds(HotCellHists& hist)
{
  // Should make an .html file containing threshold values that can be accessed as Java pop-up.  May run into problem with pop-up blockers on browsers, though.

  if (debug_)
    cout <<"HcalHotCellClient::drawSubDetThresholds()"<<endl;

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
  for (unsigned int i=hist.threshOccMap.size();i>0;--i)
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



void HcalHotCellClient::drawThresholdPlots(HotCellHists& hist, int run, string htmlDir)
{
if (debug_) cout <<"HcalHotCellClient::Creating \"Threshold Plot\" html output for subdetector "<<hist.type<<endl;
  if(hist.type<5 && !subDetsOn_[hist.type-1]) return;


  string type;
  if(hist.type==1) type= "HB";
  else if(hist.type==2) type = "HE"; 
  else if(hist.type==3) type = "HO"; 
  else if(hist.type==4) type = "HF"; 
  else if(hist.type==10) type = "HCAL";
  else 
    {
      if (debug_)cout <<"<HcalHotCellClient::drawThresholdPlots> Error:  unrecognized histogram type: "<<hist.type<<endl;
      return;
    }

  ofstream html;
  html.open((htmlDir + "HcalHotCellClient_Threshold_"+type+"_Plots.html").c_str());
  // html page header
  html << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  html << "<html>  " << endl;
  html << "<head>  " << endl;
  html << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  html << " http-equiv=\"content-type\">  " << endl;
  html << "  <title>Monitor: "<<type<<" Hot Cell Detailed Rec Hit Threshold Plots</title> " << endl;
  html << "</head>  " << endl;
  html << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  html << "<body>  " << endl;
  html << "<br>  " << endl;
  html << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  html << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  html << " style=\"color: rgb(0, 0, 153);\">" << run << "</span></h2>" << endl;
   html << "<h2>Events processed:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" <<   endl;
    
  html << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span " << endl;
  html << " style=\"color: rgb(0, 0, 153);\">" << ievt_ << "</span></h2>" << endl;
  html << "<hr>" << endl;
  html<<"<table border=\"0\" cellspacing=\"0\" " << endl;
  html << "cellpadding=\"10\"> " << endl;
  html << "<h3><tr><td><a href=\"index.html\"> Main DQM Page </a> </td>"<<endl;
  html << "<h3><td><a href=\"HcalHotCellClient.html\"> Main Hot Cell Page </a> </td>"<<endl;
  html << "<td><a href=\"HcalHotCellClient_NADA_HCAL_Plots.html\"> Hot NADA Plots </a> </td>"<<endl;
   html << "<td><a href=\"HcalHotCellClient_Digi_HCAL_Plots.html\"> Hot Digi Plots </a> </td>"<<endl;
  html << "<h3><tr><td>Hot Cell (Threshold) plots:  </td>";
  html << "<td><a href=\"HcalHotCellClient_Threshold_HCAL_Plots.html\">HCAL Plots </a>  </td>" << endl;
  if(subDetsOn_[0]) html << "<td><a href=\"HcalHotCellClient_Threshold_HB_Plots.html\">HB Plots </a></br>  </td>" << endl;  
  if(subDetsOn_[1]) html << "<td><a href=\"HcalHotCellClient_Threshold_HE_Plots.html\">HE Plots </a></br>  </td>" << endl;
  if(subDetsOn_[2]) html << "<td><a href=\"HcalHotCellClient_Threshold_HO_Plots.html\">HO Plots </a></br>  </td>" << endl;
  if(subDetsOn_[3]) html << "<td><a href=\"HcalHotCellClient_Threshold_HF_Plots.html\">HF Plots </a></br></td>" << endl;
  html << "</h3></tr></table>" << endl;
  html << "<hr>" << endl;

  html<<"<h2>Threshold Values:</h2><br>"<<endl;
  if (hist.type==10)
    html<<"(Threshold values are specified separately for individual subdetectors)"<<endl;
  else
    {
      html << "<table  width=100% border=1><tr>" << endl;
      html << "<tr align=\"left\">" << endl;	
      html << "<td>Threshold # </td><td>Threshold Value (GeV)</td></tr>"<<endl;
      for (int i=0;i<hist.Nthresholds;++i)
	{
	  html << "<tr><td>"<<i<<"</td>";
	  html <<"<td>"<<hist.thresholds[i]<<"</td></tr>"<<endl;
	}
    }
  html<<"</table><hr>"<<endl;

  html<<"<h2>Possible Hot Cells:</h2><br>"<<endl;
  html<< "<table  width=100% border=1><tr>" << endl;
  html<<"<tr align=\"center\">"<<endl;
  htmlAnyHisto(run,hist.threshOccMap[0],"i#eta","i#phi", 92, html,htmlDir);
  html<<"</tr>"<<endl;
  html<<"<tr><td>This first threshold level is used for determining whether hot cells are present.  The remaining thresholds below are used for diagnostic purposes</td>"<<endl;
  html<<"</tr></table><hr>"<<endl;

  html<<"<h2>Threshold Plots:</h2><br>"<<endl;
  html << "<table  width=100% border=1><tr>" << endl;
  html << "<td>Occupancy </td><td>Energy Distribution (GeV)</td><tr>"<<endl;
  for (int i=0;i<hist.Nthresholds;++i)
    {
      html << "<tr align=\"center\">" << endl;
      if (i<(int)hist.threshOccMap.size())
	htmlAnyHisto(run,hist.threshOccMap[i],"i#eta","i#phi", 92, html,htmlDir);
      if (i<(int)hist.threshEnergyMap.size())
	htmlAnyHisto(run,hist.threshEnergyMap[i],"i#eta","i#phi", 92, html,htmlDir);
      html <<"</tr>"<<endl;
    }
  html<<"</table><hr>"<<endl;

  // Individual depth plots
  html<<"<h2>Depth=1 Plots:</h2><br>"<<endl;
  html << "<table  width=100% border=1><tr>" << endl;
  html << "<td>Occupancy </td><td>Energy Distribution (GeV)</td><tr>"<<endl;

  for (int i=0;i<hist.Nthresholds;++i)
    {
      html << "<tr align=\"left\">" << endl;
      if (i<(int)hist.threshOccMapDepth1.size())
	htmlAnyHisto(run,hist.threshOccMapDepth1[i],"i#eta","i#phi", 92, html,htmlDir);
	else
	  html<<"<td align=\"center\"><br><br> Histogram does not exist in ROOT file!<br>Diagnostic flag may be off.<br>(This is normal in online running.)</td>"<<endl;
      if (i<(int)hist.threshEnergyMapDepth1.size())
	htmlAnyHisto(run,hist.threshEnergyMapDepth1[i],"i#eta","i#phi", 92, html,htmlDir);
      else
	html<<"<td align=\"center\"><br><br> Histogram does not exist in ROOT file!<br>Diagnostic flag may be off.<br>(This is normal in online running.)</td>"<<endl;
      html <<"</tr>"<<endl;
    }
  html<<"</table><hr>"<<endl;

  html<<"<h2>Depth=2 Plots:</h2><br>"<<endl;
  html << "<table  width=100% border=1><tr>" << endl;
  html << "<td>Occupancy </td><td>Energy Distribution (GeV)</td><tr>"<<endl;
  for (int i=0;i<hist.Nthresholds;++i)
    {
      html << "<tr align=\"left\">" << endl;
      if (i<(int)hist.threshOccMapDepth2.size())
	htmlAnyHisto(run,hist.threshOccMapDepth2[i],"i#eta","i#phi", 92, html,htmlDir);
      else
	html<<"<td align=\"center\"><br><br> Histogram does not exist in ROOT file!<br>Diagnostic flag may be off.<br>(This is normal in online running.)</td>"<<endl;
      if (i<(int)hist.threshEnergyMapDepth2.size())
	htmlAnyHisto(run,hist.threshEnergyMapDepth2[i],"i#eta","i#phi", 92, html,htmlDir);
      else
	html<<"<td align=\"center\"><br><br> Histogram does not exist in ROOT file!<br>Diagnostic flag may be off.<br>(This is normal in online running.)</td>"<<endl;
      html <<"</tr>"<<endl;
    }
  html<<"</table><hr>"<<endl;

  html<<"<h2>Depth=3 Plots:</h2><br>"<<endl;
  html << "<table  width=100% border=1><tr>" << endl;
  html << "<td>Occupancy </td><td>Energy Distribution (GeV)</td><tr>"<<endl;
  for (int i=0;i<hist.Nthresholds;++i)
    {
      html << "<tr align=\"left\">" << endl;
      if (i<(int)hist.threshOccMapDepth3.size())
	htmlAnyHisto(run,hist.threshOccMapDepth3[i],"i#eta","i#phi", 92, html,htmlDir);
      else
	html<<"<td align=\"center\"><br><br> Histogram does not exist in ROOT file!<br>Diagnostic flag may be off.<br>(This is normal in online running.)</td>"<<endl;
      if (i<(int)hist.threshEnergyMapDepth3.size())
	htmlAnyHisto(run,hist.threshEnergyMapDepth3[i],"i#eta","i#phi", 92, html,htmlDir);
      else
	html<<"<td align=\"center\"><br><br> Histogram does not exist in ROOT file!<br>Diagnostic flag may be off.<br>(This is normal in online running.)</td>"<<endl;
      html <<"</tr>"<<endl;
    }
  html<<"</table><hr>"<<endl;

  html<<"<h2>Depth=4 Plots:</h2><br>"<<endl;
  html << "<table  width=100% border=1><tr>" << endl;
  html << "<td>Occupancy </td><td>Energy Distribution (GeV)</td><tr>"<<endl;
  for (int i=0;i<hist.Nthresholds;++i)
    {
      html << "<tr align=\"left\">" << endl;
      if (i<(int)hist.threshOccMapDepth4.size())
	htmlAnyHisto(run,hist.threshOccMapDepth4[i],"i#eta","i#phi", 92, html,htmlDir);
      else
	html<<"<td align=\"center\"><br><br> Histogram does not exist in ROOT file!<br>Diagnostic flag may be off.<br>(This is normal in online running.)</td>"<<endl;
      if (i<(int)hist.threshEnergyMapDepth4.size())
	htmlAnyHisto(run,hist.threshEnergyMapDepth4[i],"i#eta","i#phi", 92, html,htmlDir);
      else html<<"<td align=\"center\"><br><br> Histogram does not exist in ROOT file!<br>Diagnostic flag may be off.<br>(This is normal in online running.)</td>"<<endl;
      html <<"</tr>"<<endl;
    }
  html<<"</table><hr>"<<endl;
  // html page footer
  html << "</body> " << endl;
  html << "</html> " << endl;
  html.close();


}// void HcalHotCellClient::drawThresholdPlots(HotCellHists& hist, int ...)



void HcalHotCellClient::drawDigiPlots(HotCellHists& hist, int run, string htmlDir)
{
if (debug_) cout <<"HcalHotCellClient::Creating \"Digi Plot\" html output for subdetector "<<hist.type<<endl;
  if(hist.type<5 && !subDetsOn_[hist.type-1]) return;


  string type;
  if(hist.type==1) type= "HB";
  else if(hist.type==2) type = "HE"; 
  else if(hist.type==3) type = "HO"; 
  else if(hist.type==4) type = "HF"; 
  else if(hist.type==10) type = "HCAL";
  else 
    {
      if (debug_)cout <<"<HcalHotCellClient::drawDigiPlots> Error:  unrecognized histogram type: "<<hist.type<<endl;
      return;
    }

  ofstream html;
  html.open((htmlDir + "HcalHotCellClient_Digi_"+type+"_Plots.html").c_str());
  // html page header
  html << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  html << "<html>  " << endl;
  html << "<head>  " << endl;
  html << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  html << " http-equiv=\"content-type\">  " << endl;
  html << "  <title>Monitor: "<<type<<" Hot Cell Detailed Digi Plots</title> " << endl;
  html << "</head>  " << endl;
  html << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  html << "<body>  " << endl;
  html << "<br>  " << endl;
  html << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  html << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  html << " style=\"color: rgb(0, 0, 153);\">" << run << "</span></h2>" << endl;
   html << "<h2>Events processed:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" <<   endl;
    
  html << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span " << endl;
  html << " style=\"color: rgb(0, 0, 153);\">" << ievt_ << "</span></h2>" << endl;
  html << "<hr>" << endl;
  html<<"<table border=\"0\" cellspacing=\"0\" " << endl;
  html << "cellpadding=\"10\"> " << endl;
  html << "<h3><tr><td><a href=\"index.html\"> Main DQM Page </a> </td>"<<endl;
  html << "<h3><td><a href=\"HcalHotCellClient.html\"> Main Hot Cell Page </a> </td>"<<endl;
  html << "<td><a href=\"HcalHotCellClient_NADA_HCAL_Plots.html\"> Hot NADA Plots </a> </td>"<<endl;
   html << "<td><a href=\"HcalHotCellClient_Threshold_HCAL_Plots.html\"> Hot Threshold Plots </a> </td>"<<endl;
  html << "<h3><tr><td>Hot Cell (Digi) plots:  </td>";
  html << "<td><a href=\"HcalHotCellClient_Digi_HCAL_Plots.html\">HCAL Plots </a>  </td>" << endl;
  if(subDetsOn_[0]) html << "<td><a href=\"HcalHotCellClient_Digi_HB_Plots.html\">HB Plots </a></br>  </td>" << endl;  
  if(subDetsOn_[1]) html << "<td><a href=\"HcalHotCellClient_Digi_HE_Plots.html\">HE Plots </a></br>  </td>" << endl;
  if(subDetsOn_[2]) html << "<td><a href=\"HcalHotCellClient_Digi_HO_Plots.html\">HO Plots </a></br>  </td>" << endl;
  if(subDetsOn_[3]) html << "<td><a href=\"HcalHotCellClient_Digi_HF_Plots.html\">HF Plots </a></br></td>" << endl;
  html << "</h3></tr></table>" << endl;
  html << "<hr>" << endl;


  html<<"<h2>Possible Hot Cells:</h2><br>"<<endl;
  html<< "<table  width=100% border=1><tr>" << endl;
  html<<"<tr align=\"center\">"<<endl;
  htmlAnyHisto(run,hist.abovePedSigma,"i#eta","i#phi", 92, html,htmlDir);
  html<<"</tr>"<<endl;
  html<<"<tr><td>Cells are considered hot if their digi value is > N sigma above pedestal for some fraction of events.</td>"<<endl;

  html<<"</tr></table><hr>"<<endl;
  
  // Add individual depth/sigma plots at some point?

  // html page footer
  html << "</body> " << endl;
  html << "</html> " << endl;
  html.close();


}// void HcalHotCellClient::drawDigiPlots(...)


void HcalHotCellClient::drawNADAPlots(HotCellHists& hist, int run, string htmlDir)
{
if (debug_) cout <<"HcalHotCellClient::Creating \"NADA Plot\" html output for subdetector "<<hist.type<<endl;
  if(hist.type<5 && !subDetsOn_[hist.type-1]) return;


  string type;
  if(hist.type==1) type= "HB";
  else if(hist.type==2) type = "HE"; 
  else if(hist.type==3) type = "HO"; 
  else if(hist.type==4) type = "HF"; 
  else if(hist.type==10) type = "HCAL";
  else 
    {
      if (debug_)cout <<"<HcalHotCellClient::drawNADAPlots> Error:  unrecognized histogram type: "<<hist.type<<endl;
      return;
    }

  ofstream html;
  html.open((htmlDir + "HcalHotCellClient_NADA_"+type+"_Plots.html").c_str());
  // html page header
  html << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  html << "<html>  " << endl;
  html << "<head>  " << endl;
  html << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  html << " http-equiv=\"content-type\">  " << endl;
  html << "  <title>Monitor: "<<type<<" Hot Cell Detailed NADA Plots</title> " << endl;
  html << "</head>  " << endl;
  html << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  html << "<body>  " << endl;
  html << "<br>  " << endl;
  html << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  html << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  html << " style=\"color: rgb(0, 0, 153);\">" << run << "</span></h2>" << endl;
   html << "<h2>Events processed:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" <<   endl;
    
  html << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span " << endl;
  html << " style=\"color: rgb(0, 0, 153);\">" << ievt_ << "</span></h2>" << endl;
  html << "<hr>" << endl;
  html<<"<table border=\"0\" cellspacing=\"0\" " << endl;
  html << "cellpadding=\"10\"> " << endl;
  html << "<h3><tr><td><a href=\"index.html\"> Main DQM Page </a> </td>"<<endl;
  html << "<h3><td><a href=\"HcalHotCellClient.html\"> Main Hot Cell Page </a> </td>"<<endl;
  html << "<td><a href=\"HcalHotCellClient_Digi_HCAL_Plots.html\"> Hot Digi Plots </a> </td>"<<endl;
   html << "<td><a href=\"HcalHotCellClient_Threshold_HCAL_Plots.html\"> Hot Threshold Plots </a> </td>"<<endl;
  html << "<h3><tr><td>Hot Cell (NADA) plots:  </td>";
  html << "<td><a href=\"HcalHotCellClient_NADA_HCAL_Plots.html\">HCAL Plots </a>  </td>" << endl;
  if(subDetsOn_[0]) html << "<td><a href=\"HcalHotCellClient_NADA_HB_Plots.html\">HB Plots </a></br>  </td>" << endl;  
  if(subDetsOn_[1]) html << "<td><a href=\"HcalHotCellClient_NADA_HE_Plots.html\">HE Plots </a></br>  </td>" << endl;
  if(subDetsOn_[2]) html << "<td><a href=\"HcalHotCellClient_NADA_HO_Plots.html\">HO Plots </a></br>  </td>" << endl;
  if(subDetsOn_[3]) html << "<td><a href=\"HcalHotCellClient_NADA_HF_Plots.html\">HF Plots </a></br></td>" << endl;
  html << "</h3></tr></table>" << endl;
  html << "<hr>" << endl;

  html<<"<h2>Possible Hot Cells:</h2><br>"<<endl;
  html<< "<table  width=100% border=1><tr>" << endl;
  html<<"<tr align=\"center\">"<<endl;
  html <<"<td>NADA cells with energy >> than neighbors</td>"<<endl;
  html <<"<td>NADA cell energies</td>"<<endl;
  html<<"</tr><tr>"<<endl;
  htmlAnyHisto(run,hist.nadaOccMap,"i#eta","i#phi", 92, html,htmlDir);
  htmlAnyHisto(run,hist.nadaEnergyMap,"i#eta","i#phi", 92, html,htmlDir);
  html<<"</tr><tr>"<<endl;
  htmlAnyHisto(run,hist.nadaNumHotCells,"# NADA hot cells","# occurrences", 92, html,htmlDir);
  htmlAnyHisto(run,hist.nadaEnergy,"NADA hot cell energy","# occurrences", 92, html,htmlDir);
  html<<"</tr>"<<endl;
  html <<"</table><hr>"<<endl;

  html<<"<h2>Negative-Energy Cells:</h2><br>"<<endl;
  html<< "<table  width=100% border=1><tr>" << endl;
  html<<"<tr align=\"center\">"<<endl;
  html <<"<td>NADA cells with significant negative energy</td>"<<endl;
  html <<"<td>NADA cell energies</td>"<<endl;
  html<<"</tr><tr>"<<endl;
  htmlAnyHisto(run,hist.nadaNegOccMap,"i#eta","i#phi", 92, html,htmlDir);
  htmlAnyHisto(run,hist.nadaNegEnergyMap,"i#eta","i#phi", 92, html,htmlDir);
  html<<"</tr><tr>"<<endl;
  htmlAnyHisto(run,hist.nadaNumNegCells,"# NADA negative-energy cells","# occurrences", 92, html,htmlDir);
  html<<"</tr>"<<endl;

  html <<"</table><hr><br>"<<endl;
  
  html<<"<h2>Possible Hot Cells by Depth:</h2><br>"<<endl;
  html<< "<table  width=100% border=1><tr>" << endl;
  html<<"<tr align=\"center\">"<<endl;
  html <<"<td>NADA cells with energy >> than neighbors</td>"<<endl;
  html <<"<td>NADA cell energies</td>"<<endl;

  TH2F* dummy=NULL;
  for (unsigned int d=0;d<4;++d)
    {
      html<<"<tr align=\"center\">"<<endl;
      hist.nadaOccMapDepth.size()>d ? htmlAnyHisto(run,hist.nadaOccMapDepth[d],"i#eta","i#phi",92,html,htmlDir) : htmlAnyHisto(run,dummy,"i#eta","i#phi",92,html,htmlDir);
      hist.nadaOccMapDepth.size()>d ? htmlAnyHisto(run,hist.nadaEnergyMapDepth[d],"i#eta","i#phi",92,html,htmlDir) : htmlAnyHisto(run,dummy,"i#eta","i#phi",92,html,htmlDir);
      html<<"</tr>"<<endl;
    }
  html <<"</table><hr><br>"<<endl;
  
  html<<"<h2>Negative-Energy Cells by Depth:</h2><br>"<<endl;
  html<< "<table  width=100% border=1><tr>" << endl;
  html<<"<tr align=\"center\">"<<endl;
  html <<"<td>NADA cells with significant negative energy</td>"<<endl;
  html <<"<td>NADA cell energies</td>"<<endl;
  for (unsigned int d=0;d<4;++d)
    {
      html<<"<tr align=\"center\">"<<endl;
      hist.nadaNegOccMapDepth.size()>d ? htmlAnyHisto(run,hist.nadaNegOccMapDepth[d],"i#eta","i#phi",92,html,htmlDir) : htmlAnyHisto(run,dummy,"i#eta","i#phi",92,html,htmlDir);
      hist.nadaNegOccMapDepth.size()>d ? htmlAnyHisto(run,hist.nadaNegEnergyMapDepth[d],"i#eta","i#phi",92,html,htmlDir) : htmlAnyHisto(run,dummy,"i#eta","i#phi",92,html,htmlDir);
      html<<"</tr>"<<endl;
    }
  html <<"</table><hr><br>"<<endl;

  // html page footer
  html << "</body> " << endl;
  html << "</html> " << endl;
  html.close();

  return;

}// void HcalHotCellClient::drawNADAPlots(...)




void HcalHotCellClient::drawProblemPlots(HotCellHists& hist, int run, string htmlDir)
{
if (debug_) cout <<"HcalHotCellClient::Creating \"Problem Plot\" html output for subdetector "<<hist.type<<endl;
  if(hist.type<5 && !subDetsOn_[hist.type-1]) return;


  string type;
  if(hist.type==1) type= "HB";
  else if(hist.type==2) type = "HE"; 
  else if(hist.type==3) type = "HO"; 
  else if(hist.type==4) type = "HF"; 
  else if(hist.type==10) type = "HCAL"; 
  else 
    {
      if (debug_)cout <<"<HcalHotCellClient::drawProblemPlots> Error:  unrecognized histogram type: "<<hist.type<<endl;
      return;
    }

  ofstream html;
  html.open((htmlDir + "HcalHotCellClient_ProblemDepth_"+type+"_Plots.html").c_str());
  // html page header
  html << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  html << "<html>  " << endl;
  html << "<head>  " << endl;
  html << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  html << " http-equiv=\"content-type\">  " << endl;
  html << "  <title>Monitor: "<<type<<" Hot Cell Problem Plots By Depth</title> " << endl;
  html << "</head>  " << endl;
  html << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  html << "<body>  " << endl;
  html << "<br>  " << endl;
  html << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  html << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  html << " style=\"color: rgb(0, 0, 153);\">" << run << "</span></h2>" << endl;
   html << "<h2>Events processed:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" <<   endl;
    
  html << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span " << endl;
  html << " style=\"color: rgb(0, 0, 153);\">" << ievt_ << "</span></h2>" << endl;
  html << "<hr>" << endl;
  html<<"<table border=\"0\" cellspacing=\"0\" " << endl;
  html << "cellpadding=\"10\"> " << endl;
  html << "<h3><tr><td><a href=\"index.html\"> Main DQM Page </a> </td>"<<endl;
  html << "<h3><td><a href=\"HcalHotCellClient.html\"> Main Hot Cell Page </a> </td>"<<endl;
  html << "<td><a href=\"HcalHotCellClient_Digi_HCAL_Plots.html\"> Hot Digi Plots </a> </td>"<<endl;
   html << "<td><a href=\"HcalHotCellClient_Threshold_HCAL_Plots.html\"> Hot Threshold Plots </a> </td>"<<endl;
  html << "<td><a href=\"HcalHotCellClient_NADA_HCAL_Plots.html\"> Hot NADA Plots </a> </td>"<<endl;
  html << "<h3><tr><td>Other Problem (by Depth) plots:  </td>";
  html << "<td><a href=\"HcalHotCellClient_ProblemDepth_HCAL_Plots.html\">HCAL Plots </a>  </td>" << endl;
  if(subDetsOn_[0]) html << "<td><a href=\"HcalHotCellClient_ProblemDepth_HB_Plots.html\">HB Plots </a></br>  </td>" << endl;  
  if(subDetsOn_[1]) html << "<td><a href=\"HcalHotCellClient_ProblemDepth_HE_Plots.html\">HE Plots </a></br>  </td>" << endl;
  if(subDetsOn_[2]) html << "<td><a href=\"HcalHotCellClient_ProblemDepth_HO_Plots.html\">HO Plots </a></br>  </td>" << endl;
  if(subDetsOn_[3]) html << "<td><a href=\"HcalHotCellClient_ProblemDepth_HF_Plots.html\">HF Plots </a></br></td>" << endl;
  html << "</h3></tr></table>" << endl;
  html << "<hr>" << endl;

  html<< "<table  width=100% border=1><tr>" << endl;
  html<<"<tr align=\"center\">"<<endl;
  htmlAnyHisto(run,hist.problemHotCells,"i#eta","i#phi", 92, html,htmlDir);
  html<<"</tr>"<<endl;
  html<<"<tr><td>Cell entries in this histogram are filled if:  <br> (a) Cell digi ADC is > 3 sigma above pedestal  (<a href=\"HcalHotCellClient_Digi_HCAL_Plots.html\">Hot Digi</a>);<br>"; 
  html<<" (b) Cell recHit energy is greater than some threshold energy (<a href=\"HcalHotCellClient_Threshold_HCAL_Plots.html\">Hot Threshold</a>); or <br> ";
  html<<" (c) Cell has significantly more energy than the sum of its neighboring cells (<a href=\"HcalHotCellClient_NADA_HCAL_Plots.html\">Hot NADA</a>) <br><br>.  ";
  html <<" Cells must be hot more than "<<(int)errorFrac_*100<<"% of the time to appear in the histogram.<br>";
  html <<"The plots below show all potentially problematic entries for each depth, without the "<<(int)errorFrac_*100<<"% cutoff.</td></tr><br>"<<endl;
  html <<"</table><hr><br>"<<endl;

  html<<"<h2>Potential Problem Plots by Depth:</h2><br>"<<endl;
  html<<"(Cells are identified as potential problems if they meet the requirements of any of the hot cell algorithms listed above)<br>"<<endl;
  html<< "<table  width=100% border=1><tr>" << endl;
  html<<"<tr align=\"center\">"<<endl;
  

  if (hist.problemHotCells_DEPTH.size()>0)
    htmlAnyHisto(run,hist.problemHotCells_DEPTH[0],"i#eta","i#phi", 92, html,htmlDir);
  else
    html<<"<td align=\"center\"><br><br> Histogram does not exist in ROOT file!<br>Diagnostic flag may be off.<br>(This is normal in online running.)</td>"<<endl;
  if (hist.problemHotCells_DEPTH.size()>1)
    htmlAnyHisto(run,hist.problemHotCells_DEPTH[1],"i#eta","i#phi", 92, html,htmlDir);
  else
    html<<"<td align=\"center\"><br><br> Histogram does not exist in ROOT file!<br>Diagnostic flag may be off.<br>(This is normal in online running.)</td>"<<endl;
  html<<"</tr><tr align=\"center\">"<<endl;
  if (hist.problemHotCells_DEPTH.size()>2)
    htmlAnyHisto(run,hist.problemHotCells_DEPTH[2],"i#eta","i#phi", 92, html,htmlDir);
  else 
    html<<"<td align=\"center\"><br><br> Histogram does not exist in ROOT file!<br>Diagnostic flag may be off.<br>(This is normal in online running.)</td>"<<endl;
  if (hist.problemHotCells_DEPTH.size()>3)
    htmlAnyHisto(run,hist.problemHotCells_DEPTH[3],"i#eta","i#phi", 92, html,htmlDir);
  else 
    html<<"<td align=\"center\"><br><br> Histogram does not exist in ROOT file!<br>Diagnostic flag may be off.<br>(This is normal in online running.)</td>"<<endl;
  html<<"</tr>"<<endl;
  html <<"</table><hr><br>"<<endl;

  
 
  html <<"<table width=75%align = \"center\"><tr align=\"center\">" <<endl;
  html <<"<td>  List of All Possibly Hot Cells</td><td align=\"center\"> Fraction of Events in which cells are hot (%)</td></tr>"<<endl;


  // Dump out hot cell candidates
  for (int depth=0;depth<4;++depth)
    {
      if (hist.problemHotCells_DEPTH[depth]==NULL) continue;
      int etabins = hist.problemHotCells_DEPTH[depth]->GetNbinsX();
      int phibins = hist.problemHotCells_DEPTH[depth]->GetNbinsY();
      float etaMin=hist.problemHotCells_DEPTH[depth]->GetXaxis()->GetXmin();
      float phiMin=hist.problemHotCells_DEPTH[depth]->GetYaxis()->GetXmin();
      
      int eta,phi;
      if (type=="HCAL")
	type=""; // don't display "HCAL" as a subdetector when dumping hot cells
      string color;
      for (int ieta=1;ieta<=etabins;++ieta)
	{
	  for (int iphi=1; iphi<=phibins;++iphi)
	    {
	      eta=ieta+int(etaMin)-1;
	      phi=iphi+int(phiMin)-1;

	      if (hist.problemHotCells_DEPTH[depth]->GetBinContent(ieta,iphi)>0)
		{
		  html<<"<td align=\"center\">";
		  // Mark cells as possibly problematic if >1/5 of (errorFrac)
		  hist.problemHotCells_DEPTH[depth]->GetBinContent(ieta,iphi)>errorFrac_*ievt_/5. ? color="orange": color="black";
		  // if > errorFrac, change color from orange to red
		  if (hist.problemHotCells_DEPTH[depth]->GetBinContent(ieta,iphi)>ievt_*errorFrac_) color="red";
		  html <<"<font color = '"<<color.c_str()<<"'>";
		  html<<type<<"  ("<<eta<<", "<<phi<<", "<<depth+1<<") </font></td>";
		  html<<"<td align=\"center\"><font color = '"<<color.c_str()<<"'>"<< 100.*hist.problemHotCells_DEPTH[depth]->GetBinContent(ieta,iphi)/ievt_<<"%</font></td>"<<endl;
		  if (color=="red")
		    html<<"<td align=\"center\"><font color = 'red' > POSSIBLY HOT </font></td>"<<endl;
		  html<<"</tr>"<<endl;
		}
	      
	    } // for (int iphi =1; iphi<=phibins;++iphi)
	  
	}// for (int ieta=1; ieta<=etabins; ++eta)
    } // for (int depth=0; depth<4;++depth)

  html << "</table>" <<endl;
  // html page footer
  html << "</body> " << endl;
  html << "</html> " << endl;
  html.close();

  return;
} //void HcalHotCellClient::drawProblemPlots(...)
