#include <DQM/HcalMonitorClient/interface/HcalHotCellClient.h>
#include <DQM/HcalMonitorClient/interface/HcalClientUtils.h>

using namespace cms;
using namespace edm;
using namespace std;


HcalHotCellClient::HcalHotCellClient(const ParameterSet& ps, DaqMonitorBEInterface* dbe){
  dqmReportMapErr_.clear(); dqmReportMapWarn_.clear(); dqmReportMapOther_.clear();
  dqmQtests_.clear();

  dbe_ = dbe;
  
  if (verbose_)
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

  // cloneME switch
  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);
  
  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

  // DQM default process name
  process_ = ps.getUntrackedParameter<string>("processName", "Hcal/");
  
  // Set # of histogram hot cell thresholds to 0 at start

  int dummy = 0;
  thresholds_=ps.getUntrackedParameter<int>("HotCellThresholds",dummy);
  hcalhists.thresholds=thresholds_;

  vector<string> subdets = ps.getUntrackedParameter<vector<string> >("subDetsOn");
  for(int i=0; i<4; i++) subDetsOn_[i] = false;
  
  for(unsigned int i=0; i<subdets.size(); i++){
    if(subdets[i]=="HB"){
      subDetsOn_[0] = true;
      hbhists.thresholds=ps.getUntrackedParameter<int>("HBHotCellThresholds",thresholds_);
    }
    else if(subdets[1]=="HE") {
      subDetsOn_[1] = true;
      hehists.thresholds=ps.getUntrackedParameter<int>("HEHotCellThresholds",thresholds_);
    }
    else if(subdets[i]=="HO") {
      subDetsOn_[2] = true;
      hohists.thresholds=ps.getUntrackedParameter<int>("HOHotCellThresholds",thresholds_);
    }
    else if(subdets[i]=="HF"){
      subDetsOn_[3] = true;
      hfhists.thresholds=ps.getUntrackedParameter<int>("HFHotCellThresholds",thresholds_);
    }
  }
}

HcalHotCellClient::HcalHotCellClient(){
  dqmReportMapErr_.clear(); dqmReportMapWarn_.clear(); dqmReportMapOther_.clear();
  dqmQtests_.clear();

  dbe_ = 0;

  if (verbose_)
    cout <<"Initializing HcalHotCellClient *without* ParameterSet"<<endl;
  clearHists(hbhists);
  clearHists(hehists);
  clearHists(hohists);
  clearHists(hfhists);
  clearHists(hfhists);

  hbhists.type=1;
  hehists.type=2;
  hohists.type=3;
  hfhists.type=4;
  hcalhists.type=10;

  hbhists.thresholds=1;
  hehists.thresholds=1;
  hohists.thresholds=1;
  hfhists.thresholds=1;
  hcalhists.thresholds=1;

  // verbosity switch
  verbose_ = false;
  for(int i=0; i<4; i++) subDetsOn_[i] = false;
}

HcalHotCellClient::~HcalHotCellClient(){

  this->cleanup();

}

void HcalHotCellClient::beginJob(void){
  
  if ( verbose_ ) cout << "HcalHotCellClient: beginJob" << endl;
  
  ievt_ = 0;
  jevt_ = 0;

  this->setup();
  this->resetAllME();
  return;
}

void HcalHotCellClient::beginRun(void){

  if ( verbose_ ) cout << "HcalHotCellClient: beginRun" << endl;

  jevt_ = 0;
  this->setup();
  this->resetAllME();
  return;
}

void HcalHotCellClient::endJob(void) {

  if ( verbose_ ) cout << "HcalHotCellClient: endJob, ievt = " << ievt_ << endl;

  this->cleanup(); 
  return;
}

void HcalHotCellClient::endRun(void) {

  if ( verbose_ ) cout << "HcalHotCellClient: endRun, jevt = " << jevt_ << endl;

  this->cleanup();  
  return;
}

void HcalHotCellClient::setup(void) {
  
  return;
}

void HcalHotCellClient::cleanup(void) {

  if (verbose_)
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

  dqmReportMapErr_.clear(); dqmReportMapWarn_.clear(); dqmReportMapOther_.clear();
  dqmQtests_.clear();

  return;
}

void HcalHotCellClient::errorOutput(){
  if(!dbe_) return;
  dqmReportMapErr_.clear(); dqmReportMapWarn_.clear(); dqmReportMapOther_.clear();
  
  for (map<string, string>::iterator testsMap=dqmQtests_.begin(); testsMap!=dqmQtests_.end();testsMap++){
    string testName = testsMap->first;
    string meName = testsMap->second;
    MonitorElement* me = dbe_->get(meName);
    if(me){
      if (me->hasError()){
	vector<QReport*> report =  me->getQErrors();
	dqmReportMapErr_[meName] = report;
      }
      if (me->hasWarning()){
	vector<QReport*> report =  me->getQWarnings();
	dqmReportMapWarn_[meName] = report;
      }
      if(me->hasOtherReport()){
	vector<QReport*> report= me->getQOthers();
	dqmReportMapOther_[meName] = report;
      }
    }
  }
  cout <<"HotCell Task: "<<dqmReportMapErr_.size()<<" errors, "<<dqmReportMapWarn_.size()<<" warnings, "<<dqmReportMapOther_.size()<<" others"<<endl;

  return;
}

void HcalHotCellClient::getErrors(map<string, vector<QReport*> > outE, map<string, vector<QReport*> > outW, map<string, vector<QReport*> > outO){

  this->errorOutput();
  outE.clear(); outW.clear(); outO.clear();

  for(map<string, vector<QReport*> >::iterator i=dqmReportMapErr_.begin(); i!=dqmReportMapErr_.end(); i++){
    outE[i->first] = i->second;
  }
  for(map<string, vector<QReport*> >::iterator i=dqmReportMapWarn_.begin(); i!=dqmReportMapWarn_.end(); i++){
    outW[i->first] = i->second;
  }
  for(map<string, vector<QReport*> >::iterator i=dqmReportMapOther_.begin(); i!=dqmReportMapOther_.end(); i++){
    outO[i->first] = i->second;
  }

  return;
}

void HcalHotCellClient::report(){

  if ( verbose_ ) cout << "HcalHotCellClient: report" << endl;
  //  this->setup();  
  
  char name[256];
  sprintf(name, "%sHcal/HotCellMonitor/HotCell Task Event Number",process_.c_str());
  MonitorElement* me = 0;
  if(dbe_) me = dbe_->get(name);
  if ( me ) {
    string s = me->valueString();
    ievt_ = -1;
    sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &ievt_);
    if ( verbose_ ) cout << "Found '" << name << "'" << endl;
  }

  getHistograms();

  return;
}

void HcalHotCellClient::analyze(void){

  jevt_++;
  int updates = 0;
  //  if(dbe_) dbe_->getNumUpdates();
  if ( updates % 10 == 0 ) {
    if ( verbose_ ) cout << "HcalHotCellClient: " << updates << " updates" << endl;
  }
  
  return;
}

void HcalHotCellClient::getHistograms(){
  if(!dbe_) return;
  char name[150];    
  
  if (verbose_)
    cout <<"HcalHotCellClient::getHistograms()"<<endl;
  for(int i=0; i<4; i++){

    sprintf(name,"HotCellMonitor/HotCellDepth%dOccupancyMap",i+1);
    gl_geo_[i] = getHisto2(name, process_, dbe_,verbose_,cloneME_);

    sprintf(name,"HotCellMonitor/HotCellDepth%dEnergyMap",i+1);
    gl_en_[i] = getHisto2(name, process_, dbe_,verbose_,cloneME_);    
  }
    
  
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
  sprintf(name,"%sHcal/HotCellMonitor/HotCellEnergyMap",process_.c_str());
  resetME(name,dbe_);

  if (subDetsOn_[0]) resetSubDetHistograms(hbhists);
  if (subDetsOn_[1]) resetSubDetHistograms(hehists);
  if (subDetsOn_[2]) resetSubDetHistograms(hohists);
  if (subDetsOn_[3]) resetSubDetHistograms(hfhists);
  resetSubDetHistograms(hcalhists);
  return;
}

void HcalHotCellClient::htmlOutput(int runNo, string htmlDir, string htmlName){

  cout << "Preparing HcalHotCellClient html output ..." << endl;
  string client = "HotCellMonitor";
  htmlErrors(runNo,htmlDir,client,process_,dbe_,dqmReportMapErr_,dqmReportMapWarn_,dqmReportMapOther_);
  
  //ofstream htmlFile;
  htmlFile.open((htmlDir + htmlName).c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlFile << "<html>  " << endl;
  htmlFile << "<head>  " << endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlFile << " http-equiv=\"content-type\">  " << endl;
  htmlFile << "  <title>Monitor: Hcal HotCell Task output</title> " << endl;
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

  htmlFile << "<h2><strong>Hcal Hot Cell Histograms</strong></h2>" << endl;
  htmlFile << "<h3>" << endl;
  htmlFile << "<a href=\"#HCAL_Plots\">Combined HCAL Plots </a></br>" << endl;  
  if(subDetsOn_[0]) htmlFile << "<a href=\"#HB_Plots\">HB Plots </a></br>" << endl;  
  if(subDetsOn_[1]) htmlFile << "<a href=\"#HE_Plots\">HE Plots </a></br>" << endl;
  if(subDetsOn_[2]) htmlFile << "<a href=\"#HO_Plots\">HO Plots </a></br>" << endl;
  if(subDetsOn_[3]) htmlFile << "<a href=\"#HF_Plots\">HF Plots </a></br>" << endl;
  htmlFile << "</h3>" << endl;
  htmlFile << "<hr>" << endl;

  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\"> " << endl;
  htmlFile << "<td>&nbsp;&nbsp;&nbsp;<h3>Global Histograms</h3></td></tr>" << endl;

  htmlFile << "<tr align=\"left\">" << endl;	
  histoHTML2(runNo,gl_geo_[0],"iEta","iPhi", 92, htmlFile,htmlDir);
  histoHTML2(runNo,gl_en_[0],"iEta","iPhi", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

  htmlFile << "<tr align=\"left\">" << endl;	
  histoHTML2(runNo,gl_geo_[1],"iEta","iPhi", 92, htmlFile,htmlDir);
  histoHTML2(runNo,gl_en_[1],"iEta","iPhi", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

  htmlFile << "<tr align=\"left\">" << endl;	
  histoHTML2(runNo,gl_geo_[2],"iEta","iPhi", 92, htmlFile,htmlDir);
  histoHTML2(runNo,gl_en_[2],"iEta","iPhi", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

  htmlFile << "<tr align=\"left\">" << endl;	
  histoHTML2(runNo,gl_geo_[3],"iEta","iPhi", 92, htmlFile,htmlDir);
  histoHTML2(runNo,gl_en_[3],"iEta","iPhi", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

  htmlSubDetOutput(hcalhists,runNo,htmlDir,htmlName);
  htmlSubDetOutput(hbhists,runNo,htmlDir,htmlName);
  htmlSubDetOutput(hehists,runNo,htmlDir,htmlName);
  htmlSubDetOutput(hohists,runNo,htmlDir,htmlName);
  htmlSubDetOutput(hfhists,runNo,htmlDir,htmlName);

  htmlFile << "</table>" << endl;
  htmlFile << "<br>" << endl;

  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  htmlFile.close();

  return;
}

void HcalHotCellClient::htmlSubDetOutput(HotCellHists& hist, int runNo, 
					 string htmlDir, 
					 string htmlName)
{
  if((hist.type<5) &&!subDetsOn_[hist.type-1]) return;
  
  string type;
  if(hist.type==1) type = "HB";
  else if(hist.type==2) type = "HE"; 
  else if(hist.type==3) type = "HO"; 
  else if(hist.type==4) type = "HF";
  else if(hist.type==10) type = "HCAL";
  else
    {
      cout <<"<HcalHotCellClient::htmlSubDetOutput> Unrecognized detector type: "<<hist.type<<endl;
      return;
    }

  htmlFile << "<tr align=\"left\">" << endl;
  htmlFile << "<td>&nbsp;&nbsp;&nbsp;<a name=\""<<type<<"_Plots\"><h3>" << type << " Histograms</h3></td></tr>" << endl;
  
  htmlFile << "<tr align=\"left\">" << endl;	
  histoHTML2(runNo,hist.NADA_OCC_MAP,"iEta","iPhi", 92, htmlFile,htmlDir);
  histoHTML2(runNo,hist.NADA_EN_MAP,"iEta","iPhi", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

  htmlFile << "<tr align=\"left\">" << endl;	
  histoHTML2(runNo,hist.NADA_NEG_OCC_MAP,"iEta","iPhi", 92, htmlFile,htmlDir);
  histoHTML2(runNo,hist.NADA_NEG_EN_MAP,"iEta","iPhi", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

  if (hist.type!=10)
    {
      for (int i=0;i<hist.thresholds;i++){
	htmlFile << "<tr align=\"left\">" << endl;	
	histoHTML2(runNo,hist.OCCmap[i],"iEta","iPhi", 92, htmlFile,htmlDir);
	histoHTML2(runNo,hist.ENERGYmap[i],"iEta","iPhi", 100, htmlFile,htmlDir);
	htmlFile << "</tr>" << endl;
      }
      
      htmlFile << "<tr align=\"left\">" << endl;	
      histoHTML(runNo,hist.MAX_E,"GeV","Evts", 92, htmlFile,htmlDir);
      histoHTML(runNo,hist.MAX_T,"nS","Evts", 100, htmlFile,htmlDir);
      htmlFile << "</tr>" << endl;
    }
  else
    {
      for (int i=0;i<hist.thresholds;i=i+2)
	// FIXME:  Max_E, Max_T histograms not needed (already displayed) --
	// restructure the code so that they're only displaye here? 
	// HCAL Energy histograms are showing up empty for some reason
	// (name error?  Check!)
	{	
	  htmlFile << "<tr align=\"left\">" << endl;	
	  histoHTML2(runNo,hist.OCCmap[i],"iEta","iPhi", 92, htmlFile,htmlDir);
	  
	  if ((i+1)<hist.thresholds)
	    histoHTML2(runNo,hist.OCCmap[i+1],"iEta","iPhi", 100, htmlFile,htmlDir);
	  htmlFile << "</tr>" << endl;
	}
    }
  return;
}

void HcalHotCellClient::createTests()
{
  if(!dbe_) return;
  /*
  char meTitle[250], name[250];    
  vector<string> params;
  */

  if (verbose_) cout <<"Creating HotCell tests..."<<endl;
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
  if (verbose_) 
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
  sprintf(meTitle,"%sHcal/HotCellMonitor/%s/NADA_%s_OCC_MAP",process_.c_str(),type.c_str(), type.c_str());
  sprintf(name,"%s NADA Hot Cell Test",type.c_str()); 
  if (verbose_) cout <<"Checking for histogram named: "<<name<<endl;
  if(dqmQtests_.find(name)==dqmQtests_.end())
    {
      if (verbose_) cout <<"Didn't find histogram; search for title: "<<meTitle<<endl;
      MonitorElement* me = dbe_->get(meTitle);
      if (me)
	{
	  if (verbose_) cout <<"Got histogram with title "<<meTitle<<"\nChecking for content"<<endl;
	  dqmQtests_[name]=meTitle;
	  params.clear();
	  params.push_back((string)meTitle);
	  params.push_back((string)name);
	  createH2ContentTest(dbe_,params);
	}
      else
	if (verbose_) cout <<"Couldn't find histogram with title: "<<meTitle<<endl;
    }

  // Check NADA Negative-energy cells
  sprintf(meTitle,"%sHcal/HotCellMonitor/%s/NADA_%s_NEG_OCC_MAP",process_.c_str(),type.c_str(), type.c_str());
  sprintf(name,"%s NADA Negative Energy Cell Test",type.c_str()); 
  if (verbose_) cout <<"Checking for histogram named: "<<name<<endl;
  if(dqmQtests_.find(name)==dqmQtests_.end())
    {
      if (verbose_) cout <<"Didn't find histogram; search for title: "<<meTitle<<endl;
      MonitorElement* me = dbe_->get(meTitle);
      if (me)
	{
	  if (verbose_) cout <<"Got histogram with title "<<meTitle<<"\nChecking for content"<<endl;
	  dqmQtests_[name]=meTitle;
	  params.clear();
	  params.push_back((string)meTitle);
	  params.push_back((string)name);
	  createH2ContentTest(dbe_,params);
	}
      else
	if (verbose_) cout <<"Couldn't find histogram with title: "<<meTitle<<endl;
    }
  
  // Check Cell with Maximum Energy deposited
  sprintf(meTitle,"%sHcal/HotCellMonitor/%s/%sHotCellGeoOccupancyMap_MaxCell",process_.c_str(),type.c_str(), type.c_str());
  sprintf(name,"%s Maximum Energy Cell",type.c_str()); 
  if (verbose_) cout <<"Checking for histogram named: "<<name<<endl;
  if(dqmQtests_.find(name)==dqmQtests_.end())
    {
      if (verbose_) cout <<"Didn't find histogram; search for title: "<<meTitle<<endl;
      MonitorElement* me = dbe_->get(meTitle);
      if (me)
	{
	  if (verbose_) cout <<"Got histogram with title "<<meTitle<<"\nChecking for content"<<endl;
	  dqmQtests_[name]=meTitle;
	  params.clear();
	  params.push_back((string)meTitle);
	  params.push_back((string)name);
	  createH2ContentTest(dbe_,params);
	}
      else
	if (verbose_) cout <<"Couldn't find histogram with title: "<<meTitle<<endl;
    }

  // Check threshold tests
  for (int i=0;i<hist.thresholds;i++)
    {
      sprintf(meTitle,"%sHcal/HotCellMonitor/%s/%sHotCellOCCmap_Thresh%i",process_.c_str(),type.c_str(), type.c_str(),i);
      sprintf(name,"%s Hot Cells Above Threshold %i",type.c_str(),i); 
      if (verbose_) cout <<"Checking for histogram named: "<<name<<endl;
      if(dqmQtests_.find(name)==dqmQtests_.end())
	{
	  if (verbose_) cout <<"Didn't find histogram; search for title: "<<meTitle<<endl;
	  MonitorElement* me = dbe_->get(meTitle);
	  if (me)
	    {
	      if (verbose_) cout <<"Got histogram with title "<<meTitle<<"\nChecking for content"<<endl;
	      dqmQtests_[name]=meTitle;
	      params.clear();
	      params.push_back((string)meTitle);
	      params.push_back((string)name);
	      createH2ContentTest(dbe_,params);
	    }
	  else
	    if (verbose_) cout <<"Couldn't find histogram with title: "<<meTitle<<endl;
	}
    } // for (int i=0;i<thresholds; i++)

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
  cout <<"Clearing HcalHotCell histograms for HCAL type: "<<hist.type<<endl;
  //hist.type=-1; //clear type?  Probably not?
  hist.OCC_MAP_GEO_Max=0;
  hist.EN_MAP_GEO_Max=0;
  hist.MAX_E=0;
  hist.MAX_T=0;
  hist.MAX_ID=0;
  hist.OCCmap.clear();
  hist.ENERGYmap.clear();
  // NADA histograms
  hist.NADA_OCC_MAP=0;
  hist.NADA_EN_MAP=0;
  hist.NADA_NumHotCells=0;
  hist.NADA_testcell=0;
  hist.NADA_Energy=0;
  hist.NADA_NumNegCells=0;
  hist.NADA_NEG_OCC_MAP=0;
  hist.NADA_NEG_EN_MAP=0;
  return;
}

void HcalHotCellClient::deleteHists(HotCellHists& hist)
{
  if (hist.OCC_MAP_GEO_Max) delete hist.OCC_MAP_GEO_Max;
  if (hist.EN_MAP_GEO_Max) delete hist.EN_MAP_GEO_Max;
  if (hist.MAX_E) delete hist.MAX_E;
  if (hist.MAX_T) delete hist.MAX_T;
  if (hist.MAX_ID) delete hist.MAX_ID;
  
  hist.OCCmap.clear();
  hist.ENERGYmap.clear();

  // NADA histograms
  if (hist.NADA_OCC_MAP) delete hist.NADA_OCC_MAP;
  if (hist.NADA_EN_MAP) delete hist.NADA_EN_MAP;
  if (hist.NADA_NumHotCells) delete hist.NADA_NumHotCells;
  if (hist.NADA_testcell) delete hist.NADA_testcell;
  if (hist.NADA_Energy) delete hist.NADA_Energy;
  if (hist.NADA_NumNegCells) delete hist.NADA_NumNegCells;
  if (hist.NADA_NEG_OCC_MAP) delete hist.NADA_NEG_OCC_MAP;
  if (hist.NADA_NEG_EN_MAP) delete hist.NADA_NEG_EN_MAP;
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
    cout <<"<HcalHotCellClient::getSubDetHistograms> Unrecognized subdetector type:  "<<hist.type<<endl;
    return;
  }
  // Why are these 2 lines here?
  //hist.OCCmap.clear();
  //hist.ENERGYmap.clear();
  
  if (verbose_)
  cout <<"Getting HcalHotCell Subdetector Histograms for Subdetector:  "<<type<<endl;

  for (int i=0;i<hist.thresholds;i++)
    {
      sprintf(name,"HotCellMonitor/%s/%sHotCellOCCmap_Thresh%i",type.c_str(),type.c_str(),i);
      //cout <<name<<endl;
      hist.OCCmap.push_back(getHisto2(name,process_,dbe_,verbose_,cloneME_));
      sprintf(name,"HotCellMonitor/%s/%sHotCellENERGYmap_Thresh%i",type.c_str(),type.c_str(),i);
      //cout <<name<<endl;
      hist.ENERGYmap.push_back(getHisto2(name,process_,dbe_,verbose_,cloneME_));
    }

  sprintf(name,"HotCellMonitor/%s/%sHotCellGeoOccupancyMap_MaxCell",type.c_str(),type.c_str());
  //cout <<name<<endl;
  hist.OCC_MAP_GEO_Max = getHisto2(name, process_, dbe_,verbose_,cloneME_);      
  sprintf(name,"HotCellMonitor/%s/%sHotCellGeoEnergyMap_MaxCell",type.c_str(),type.c_str());
  //cout <<name<<endl;
  hist.EN_MAP_GEO_Max = getHisto2(name, process_, dbe_,verbose_,cloneME_);
  
  sprintf(name,"HotCellMonitor/%s/%sHotCellEnergy",type.c_str(),type.c_str());
  hist.MAX_E = getHisto(name, process_, dbe_,verbose_,cloneME_);
  //cout <<name<<endl;
  sprintf(name,"HotCellMonitor/%s/%sHotCellTime",type.c_str(),type.c_str());
  hist.MAX_T = getHisto(name, process_, dbe_,verbose_,cloneME_);    
  //cout <<name<<endl;
  sprintf(name,"HotCellMonitor/%s/%sHotCellID",type.c_str(),type.c_str());
  hist.MAX_ID = getHisto(name, process_, dbe_,verbose_,cloneME_);    
  //cout <<name<<endl;

  // NADA histograms
  sprintf(name,"HotCellMonitor/%s/NADA_%s_OCC_MAP",type.c_str(),type.c_str());
  hist.NADA_OCC_MAP=getHisto2(name, process_, dbe_,verbose_,cloneME_);
  //cout <<"NAME = "<<name<<endl;
  sprintf(name,"HotCellMonitor/%s/NADA_%s_EN_MAP",type.c_str(),type.c_str());
  hist.NADA_EN_MAP=getHisto2(name, process_, dbe_,verbose_,cloneME_);
  //cout <<name<<endl;
  sprintf(name,"HotCellMonitor/%s/NADA_%s_NumHotCells",type.c_str(),type.c_str());
  hist.NADA_NumHotCells = getHisto(name, process_, dbe_,verbose_,cloneME_);
   //cout <<name<<endl;

  sprintf(name,"HotCellMonitor/%s/NADA_%s_testcell",type.c_str(),type.c_str());
  hist.NADA_testcell = getHisto(name, process_, dbe_,verbose_,cloneME_); 
  //cout <<name<<endl;

  sprintf(name,"HotCellMonitor/%s/NADA_%s_Energy",type.c_str(),type.c_str());
  hist.NADA_Energy = getHisto(name, process_, dbe_,verbose_,cloneME_); 
  //cout <<name<<endl;

  sprintf(name,"HotCellMonitor/%s/NADA_%s_NumNegCells",type.c_str(),type.c_str());
  hist.NADA_NumNegCells = getHisto(name, process_, dbe_,verbose_,cloneME_); 
  //cout <<name<<endl;

  sprintf(name,"HotCellMonitor/%s/NADA_%s_NEG_OCC_MAP",type.c_str(),type.c_str());
  hist.NADA_NEG_OCC_MAP = getHisto2(name, process_, dbe_,verbose_,cloneME_); 
  //cout <<name<<endl;

  sprintf(name,"HotCellMonitor/%s/NADA_%s_NEG_EN_MAP",type.c_str(),type.c_str());
  hist.NADA_NEG_EN_MAP = getHisto2(name, process_, dbe_,verbose_,cloneME_); 
  //cout <<name<<endl;

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
    cout <<"<HcalHotCellClient::getSubDetHistogramsFromFile> Unrecognized subdetector type:  "<<hist.type<<endl;
    return;
  }

  hist.OCCmap.clear();
  hist.ENERGYmap.clear();

  for (int i=0;i<hist.thresholds;i++)
    {
      sprintf(name,"HotCellMonitor/%s/%sHotCellOCCmap_Thresh%i",type.c_str(),type.c_str(),i);
      hist.OCCmap.push_back((TH2F*)infile->Get(name));
      sprintf(name,"HotCellMonitor/%s/%sHotCellENERGYmap_Thresh%i",type.c_str(),type.c_str(),i);
      hist.ENERGYmap.push_back((TH2F*)infile->Get(name));
    }

  sprintf(name,"HotCellMonitor/%s/%sHotCellGeoOccupancy Map_MaxCell",type.c_str(),type.c_str());
  hist.OCC_MAP_GEO_Max = (TH2F*)infile->Get(name);      
  sprintf(name,"HotCellMonitor/%s/%sHotCellGeoEnergyMap_MaxCell",type.c_str(),type.c_str());
  hist.EN_MAP_GEO_Max = (TH2F*)infile->Get(name);
  
  sprintf(name,"HotCellMonitor/%s/%sHotCellEnergy",type.c_str(),type.c_str());
  hist.MAX_E = (TH1F*)infile->Get(name);
  sprintf(name,"HotCellMonitor/%s/%sHotCellTime",type.c_str(),type.c_str());
  hist.MAX_T = (TH1F*)infile->Get(name);    
  sprintf(name,"HotCellMonitor/%s/%sHotCellID",type.c_str(),type.c_str());
  hist.MAX_ID = (TH1F*)infile->Get(name);    

  // NADA histograms
  sprintf(name,"HotCellMonitor/%s/NADA%s_OCC_MAP",type.c_str(),type.c_str());
  hist.NADA_OCC_MAP=(TH2F*)infile->Get(name);
  sprintf(name,"HotCellMonitor/%s/NADA%s_EN_MAP",type.c_str(),type.c_str());
  hist.NADA_EN_MAP=(TH2F*)infile->Get(name);
  sprintf(name,"HotCellMonitor/%s/NADA_%s_NumHotCells",type.c_str(),type.c_str());
  hist.NADA_NumHotCells = (TH1F*)infile->Get(name); 
  sprintf(name,"HotCellMonitor/%s/NADA_%s_testcell",type.c_str(),type.c_str());
  hist.NADA_testcell = (TH1F*)infile->Get(name); 
  sprintf(name,"HotCellMonitor/%s/NADA_%s_Energy",type.c_str(),type.c_str());
  hist.NADA_Energy = (TH1F*)infile->Get(name); 
  sprintf(name,"HotCellMonitor/%s/NADA_%s_NumNegCells",type.c_str(),type.c_str());
  hist.NADA_NumNegCells = (TH1F*)infile->Get(name); 
  sprintf(name,"HotCellMonitor/%s/NADA_%s_NEG_OCC_MAP",type.c_str(),type.c_str());
  hist.NADA_NEG_OCC_MAP = (TH2F*)infile->Get(name); 
  sprintf(name,"HotCellMonitor/%s/NADA_%s_NEG_EN_MAP",type.c_str(),type.c_str());
  hist.NADA_NEG_EN_MAP = (TH2F*)infile->Get(name); 

  return;
}



void HcalHotCellClient::resetSubDetHistograms(HotCellHists& hist)
{

  char name[150];
  string type;
  if(hist.type==1) type= "HB";
  else if(hist.type==2) type = "HE"; 
  else if(hist.type==3) type = "HO"; 
  else if(hist.type==4) type = "HF"; 
  else if(hist.type==10) type = "HCAL";
  else {
    cout <<"<HcalHotCellClient::resetSubDetHistograms> Unrecognized subdetector type:  "<<hist.type<<endl;
    return;
  }
  
  for (int i=0;i<hist.thresholds;i++){
      sprintf(name,"HotCellMonitor/%s/%sHotCellENERGYmap_Thresh%i",type.c_str(),type.c_str(),i);
      //cout <<"NAME = "<<name<<endl;
      resetME(name,dbe_);
      sprintf(name,"HotCellMonitor/%s/%sHotCellOCCmap_Thresh%i",type.c_str(),type.c_str(),i);
      //cout <<"NAME = "<<name<<endl;
      resetME(name,dbe_);
    }

  sprintf(name,"HotCellMonitor/%s/%sHotCellEnergyMap_MaxCell",type.c_str(),type.c_str());
  //cout <<name<<endl;
  resetME(name,dbe_);
  
  sprintf(name,"HotCellMonitor/%s/%sHotCellOccupancyMap_MaxCell",type.c_str(),type.c_str());
  //cout <<name<<endl;
  resetME(name,dbe_);
  
  sprintf(name,"HotCellMonitor/%s/%sHotCellEnergy",type.c_str(),type.c_str());
  //cout <<name<<endl;
  resetME(name,dbe_);

  sprintf(name,"HotCellMonitor/%s/%s HotCellTime",type.c_str(),type.c_str());
  //cout <<name<<endl;
  resetME(name,dbe_);

  sprintf(name,"HotCellMonitor/%s/%sHotCellID",type.c_str(),type.c_str());
  //cout <<name<<endl;
  resetME(name,dbe_);

  // NADA histograms
  sprintf(name,"HotCellMonitor/%s/NADA_%s_OCC_MAP",type.c_str(),type.c_str());
  //cout <<name<<endl;
  resetME(name,dbe_);

  sprintf(name,"HotCellMonitor/%s/NADA_%s_EN_MAP",type.c_str(),type.c_str());
  //cout <<name<<endl;
  resetME(name,dbe_);

  sprintf(name,"HotCellMonitor/%s/#NADA_%s_NumHotCells",type.c_str(),type.c_str());
  //cout <<name<<endl;
  resetME(name,dbe_);

  sprintf(name,"HotCellMonitor/%s/NADA_%s_testcell",type.c_str(),type.c_str());
  //cout <<name<<endl;
  resetME(name,dbe_);
  
  sprintf(name,"HotCellMonitor/%s/NADA_%s_Energy",type.c_str(),type.c_str());
  //cout <<name<<endl;
  resetME(name,dbe_);
  
  sprintf(name,"HotCellMonitor/%s/NADA_%s_NumNegCells",type.c_str(),type.c_str());
  //cout <<name<<endl;
  resetME(name,dbe_);
  
  sprintf(name,"HotCellMonitor/%s/NADA_%s_NEG_OCC_MAP",type.c_str(),type.c_str());
  //cout <<name<<endl;
  resetME(name,dbe_);
  
  sprintf(name,"HotCellMonitor/%s/NADA_%s_NEG_EN_MAP",type.c_str(),type.c_str());
  //cout <<name<<endl;
  resetME(name,dbe_);

  return;
}
