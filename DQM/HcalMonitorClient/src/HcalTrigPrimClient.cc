#include <DQM/HcalMonitorClient/interface/HcalTrigPrimClient.h>
#include <DQM/HcalMonitorClient/interface/HcalClientUtils.h>
#include <DQM/HcalMonitorClient/interface/HcalHistoUtils.h>

HcalTrigPrimClient::HcalTrigPrimClient(){
  // Summary
  //histo1d["TrigPrimMonitor/Summary HBHE"] = 0;
  //histo1d["TrigPrimMonitor/Summary HF"] = 0;
  histo2d["TrigPrimMonitor/Summary"] = 0;
  histo2d["TrigPrimMonitor/Summary for ZS run"] = 0;
  histo2d["TrigPrimMonitor/Error Flag"] = 0;
  histo2d["TrigPrimMonitor/Error Flag for ZS run"] = 0;
  histo2d["TrigPrimMonitor/EtCorr HBHE"] = 0;
  histo2d["TrigPrimMonitor/EtCorr HF"] = 0;

  // TP Occupancy
  histo2d["TrigPrimMonitor/TP Map/TP Occupancy"] = 0;
  histo1d["TrigPrimMonitor/TP Map/TPOccupancyVsEta"] = 0;
  histo1d["TrigPrimMonitor/TP Map/TPOccupancyVsPhi"] = 0;
  histo2d["TrigPrimMonitor/TP Map/Non Zero TP"] = 0;
  histo2d["TrigPrimMonitor/TP Map/Matched TP"] = 0;
  histo2d["TrigPrimMonitor/TP Map/Mismatched Et"] = 0;
  histo2d["TrigPrimMonitor/TP Map/Mismatched FG"] = 0;
  histo2d["TrigPrimMonitor/TP Map/Data Only"] = 0;
  histo2d["TrigPrimMonitor/TP Map/Emul Only"] = 0;
  histo2d["TrigPrimMonitor/TP Map/Missing Data"] = 0;
  histo2d["TrigPrimMonitor/TP Map/Missing Emul"] = 0;

  // Energy Plots
  histo1d["TrigPrimMonitor/Energy Plots/HBHE/Energy HBHE - All Data"] = 0;
  histo1d["TrigPrimMonitor/Energy Plots/HBHE/Energy HBHE - All Emul"] = 0;
  histo1d["TrigPrimMonitor/Energy Plots/HBHE/Energy HBHE - Mismatched FG"] = 0;
  histo1d["TrigPrimMonitor/Energy Plots/HBHE/Energy HBHE - Data Only"] = 0;
  histo1d["TrigPrimMonitor/Energy Plots/HBHE/Energy HBHE - Emul Only"] = 0;
  histo1d["TrigPrimMonitor/Energy Plots/HBHE/Energy HBHE - Missing Emul"] = 0;
  histo1d["TrigPrimMonitor/Energy Plots/HBHE/Energy HBHE - Missing Data"] = 0;

  histo1d["TrigPrimMonitor/Energy Plots/HF/Energy HF - All Data"] = 0;
  histo1d["TrigPrimMonitor/Energy Plots/HF/Energy HF - All Emul"] = 0;
  histo1d["TrigPrimMonitor/Energy Plots/HF/Energy HF - Mismatched FG"] = 0;
  histo1d["TrigPrimMonitor/Energy Plots/HF/Energy HF - Data Only"] = 0;
  histo1d["TrigPrimMonitor/Energy Plots/HF/Energy HF - Emul Only"] = 0;
  histo1d["TrigPrimMonitor/Energy Plots/HF/Energy HF - Missing Emul"] = 0;
  histo1d["TrigPrimMonitor/Energy Plots/HF/Energy HF - Missing Data"] = 0;
}

void HcalTrigPrimClient::init(const ParameterSet& ps, DQMStore* dbe, string clientName)
{
  //Call the base class first
  HcalBaseClient::init(ps,dbe,clientName);

 if (showTiming_)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  if (debug_>0) std::cout <<"<HcalTrigPrimClient> init(const ParameterSet& ps, DQMStore* dbe, string clientName)"<<std::endl;
 
 if (showTiming_)
   {
     cpu_timer.stop();  std::cout <<"TIMER:: HcalTrigPrimClient INIT  -> "<<cpu_timer.cpuTime()<<std::endl;
    }
  return;
} // void HcalTrigPrimClient::init(...)

HcalTrigPrimClient::~HcalTrigPrimClient(){
  this->cleanup();  
}

void HcalTrigPrimClient::beginJob(void){
  if ( debug_ >0) std::cout << "HcalTrigPrimClient: beginJob" << std::endl;

  ievt_ = 0; jevt_ = 0;
  return;
}

void HcalTrigPrimClient::beginRun(void){
  if ( debug_ >0) std::cout << "HcalTrigPrimClient: beginRun" << std::endl;

  jevt_ = 0;
  this->resetAllME();
  return;
}

void HcalTrigPrimClient::endJob(void) {
  if ( debug_ >0) std::cout << "HcalTrigPrimClient: endJob, ievt = " << ievt_ << std::endl;

  this->cleanup();

  return;
} //void HcalTrigPrimClient::endJob(void)

void HcalTrigPrimClient::endRun(void) {

  if ( debug_ >0) std::cout << "HcalTrigPrimClient: endRun, jevt = " << jevt_ << std::endl;

  this->cleanup();

  return;
} //void HcalTrigPrimClient::endRun(void)



void HcalTrigPrimClient::cleanup(void) {

  for (std::map< std::string, TH1* >::iterator h = histo1d.begin();
                                               h != histo1d.end();
                                               ++h){
    if (cloneME_ && h->second != 0) delete h->second;
    h->second = 0;
  }
  for (std::map< std::string, TH1* >::iterator h = histo2d.begin();
                                               h != histo2d.end();
                                               ++h){
    if (cloneME_ && h->second != 0) delete h->second;
    h->second = 0;
  }

  return;
} //void HcalTrigPrimClient::cleanup(void)



void HcalTrigPrimClient::analyze(void){
  jevt_++;

  int updates = 0;
  if ( updates % 10 == 0 ) {
    if ( debug_ >0) std::cout << "HcalTrigPrimClient: " << updates << " updates" << std::endl;
  }

  return;
} // void HcalTrigPrimClient::analyze(void)


void HcalTrigPrimClient::getHistograms()
{
  if(!dbe_) return;

  if (showTiming_)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  if (debug_>0) std::cout <<"<HcalTrigPrimClient> getHistograms()"<<std::endl;
  
  for (std::map< std::string, TH1* >::iterator h = histo1d.begin();
                                               h != histo1d.end();
                                               ++h){
    h->second = getHisto(h->first.c_str(), process_, dbe_, debug_, cloneME_);
  }
  for (std::map< std::string, TH1* >::iterator h = histo2d.begin();
                                               h != histo2d.end();
                                               ++h){
    h->second = getHisto2(h->first.c_str(), process_, dbe_, debug_, cloneME_);
  }

  if (showTiming_)
   {
     cpu_timer.stop();  std::cout <<"TIMER:: HcalTrigPrimClient GET HISTOGRAMS  -> "<<cpu_timer.cpuTime()<<std::endl;
    }
  return;
} //void HcalTrigPrimClient::getHistograms()


void HcalTrigPrimClient::report()
{
  if(!dbe_) return;

  if (showTiming_)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  if ( debug_ >0) std::cout << "<HcalTrigPrimClient> report()" << std::endl;
   getHistograms();

   stringstream name;
   name<<process_.c_str()<<rootFolder_.c_str()<<"/TrigPrimMonitor/TrigPrim Total Events Processed";
   MonitorElement* me = 0;
   if(dbe_) me = dbe_->get(name.str().c_str());
   if ( me ) 
     {
       string s = me->valueString();
       ievt_ = -1;
       sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &ievt_);
       if ( debug_ ) std::cout << "Found '" << name.str().c_str() << "'" << std::endl;
     }
   else
     std::cout <<"Didn't find "<<name.str().c_str()<<endl;
   name.str("");


  if (showTiming_)
    {
      cpu_timer.stop();  std::cout <<"TIMER:: HcalTrigPrimClient REPORT -> "<<cpu_timer.cpuTime()<<std::endl;
    }
  return;
} //void HcalTrigPrimClient::report()


void HcalTrigPrimClient::resetAllME()
{
  if (showTiming_)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  if(!dbe_) return;
  if ( debug_ >0) std::cout << "<HcalTrigPrimClient> resetAllME()" << std::endl;
  
  /*
  char name[150];
  sprintf(name, "%sHcal/TRigPrimMonitor/Summary/Summary HBHE",process_.c_str());
  resetME(name,dbe_);
  sprintf(name, "%sHcal/TRigPrimMonitor/Summary/Summary HF",process_.c_str());
  resetME(name,dbe_);
  */
  //sprintf(name,"%sHcal/TrigPrimMonitor/00 TP Occupancy",process_.c_str());
  //resetME(name,dbe_);  

  if (showTiming_)
   {
     cpu_timer.stop();  std::cout <<"TIMER:: HcalTrigPrimClient RESETALLME  -> "<<cpu_timer.cpuTime()<<std::endl;
    }
  return;
} //void HcalTrigPrimClient::resetAllME()


void HcalTrigPrimClient::htmlOutput(int runNo, string htmlDir, string htmlName)
{
  if (showTiming_)
    {
      cpu_timer.reset(); cpu_timer.start();
    }
  if (debug_>0) std::cout << "<HcalTrigPrimClient::htmlOutput> Preparing  html output ..." << std::endl;
  string client = "TrigPrimMonitor";
  htmlErrors(runNo,htmlDir,client,process_,dbe_,dqmReportMapErr_,dqmReportMapWarn_,dqmReportMapOther_);

  ofstream htmlFile;
  htmlFile.open((htmlDir + htmlName).c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << std::endl;
  htmlFile << "<html>  " << std::endl;
  htmlFile << "<head>  " << std::endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << std::endl;
  htmlFile << " http-equiv=\"content-type\">  " << std::endl;
  htmlFile << "  <title>Monitor: TP Task output</title> " << std::endl;
  htmlFile << "</head>  " << std::endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << std::endl;
  htmlFile << "<body>  " << std::endl;
  htmlFile << "<br>  " << std::endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << std::endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << std::endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << runNo << "</span></h2>" << std::endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << std::endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">TrigPrim Monitor</span></h2> " << std::endl;
  htmlFile << "<h2>Events processed:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << std::endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span " << std::endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << ievt_ << "</span></h2>" << std::endl;
  htmlFile << "<hr>" << std::endl;
  htmlFile << "<table width=100% border=1><tr>" << std::endl;
  if(hasErrors())htmlFile << "<td bgcolor=red><a href=\"TrigPrimMonitorErrors.html\">Errors in this task</a></td>" << std::endl;
  else htmlFile << "<td bgcolor=lime>No Errors</td>" << std::endl;
  if(hasWarnings()) htmlFile << "<td bgcolor=yellow><a href=\"TrigPrimMonitorWarnings.html\">Warnings in this task</a></td>" << std::endl;
  else htmlFile << "<td bgcolor=lime>No Warnings</td>" << std::endl;
  if(hasOther()) htmlFile << "<td bgcolor=aqua><a href=\"TrigPrimMonitorMessages.html\">Messages in this task</a></td>" << std::endl;
  else htmlFile << "<td bgcolor=lime>No Messages</td>" << std::endl;
  htmlFile << "</tr></table>" << std::endl;
  htmlFile << "<hr>" << std::endl;

  htmlFile << "<table width=100%>" << std::endl;

  //------------ Summary ------------------
  htmlFile << "<tr><td>&nbsp;&nbsp;&nbsp;<h3>Summary</h3></td></tr>" << std::endl;

  htmlFile << "<tr align=\"left\">" << std::endl;
  htmlAnyHisto(runNo,histo2d["TrigPrimMonitor/Summary"],"","", 10, htmlFile,htmlDir);  
  htmlAnyHisto(runNo,histo2d["TrigPrimMonitor/Summary for ZS run"],"","", 10, htmlFile,htmlDir);  
  htmlFile << "</tr>" << std::endl;

  htmlFile << "<tr align=\"left\">" << std::endl;
  htmlAnyHisto(runNo,histo2d["TrigPrimMonitor/Error Flag"],"","", 10, htmlFile,htmlDir);  
  htmlAnyHisto(runNo,histo2d["TrigPrimMonitor/Error Flag for ZS run"],"","", 10, htmlFile,htmlDir);  
  htmlFile << "</tr>" << std::endl;

  htmlFile << "<tr align=\"left\">" << std::endl;
  htmlAnyHisto(runNo,histo2d["TrigPrimMonitor/EtCorr HBHE"],"data","emul", 10, htmlFile,htmlDir);  
  htmlAnyHisto(runNo,histo2d["TrigPrimMonitor/EtCorr HF"],"data","emul", 10, htmlFile,htmlDir);  
  htmlFile << "</tr>" << std::endl;
  //----------------------------------------

  //------------- TP Occupancy --------------
  htmlFile << "<tr><td>&nbsp;&nbsp;&nbsp;<h3>TP Map</h3></td></tr>" << std::endl;

  htmlFile << "<tr align=\"left\">" << std::endl;
  htmlAnyHisto(runNo,histo2d["TrigPrimMonitor/TP Map/TP Occupancy"],"ieta","iphi", 10, htmlFile,htmlDir);  
  htmlAnyHisto(runNo,histo2d["TrigPrimMonitor/TP Map/Non Zero TP"],"ieta","iphi", 10, htmlFile,htmlDir);  
  htmlFile << "</tr>" << std::endl;

  htmlFile << "<tr align=\"left\">" << std::endl;
  htmlAnyHisto(runNo,histo1d["TrigPrimMonitor/TP Map/TPOccupancyVsEta"],"ieta","Triggers", 10, htmlFile,htmlDir);  
  htmlAnyHisto(runNo,histo1d["TrigPrimMonitor/TP Map/TPOccupancyVsPhi"],"iphi", "Triggers",10, htmlFile,htmlDir);  
  htmlFile << "</tr>" << std::endl;

htmlFile << "<tr align=\"left\">" << std::endl;
  htmlAnyHisto(runNo,histo2d["TrigPrimMonitor/TP Map/Matched TP"],"ieta","iphi", 10, htmlFile,htmlDir);  
  htmlFile << "</tr>" << std::endl;

htmlFile << "<tr align=\"left\">" << std::endl;
  htmlAnyHisto(runNo,histo2d["TrigPrimMonitor/TP Map/Mismatched Et"],"ieta","iphi", 10, htmlFile,htmlDir);  
  htmlAnyHisto(runNo,histo2d["TrigPrimMonitor/TP Map/Mismatched FG"],"ieta","iphi", 10, htmlFile,htmlDir);  
  htmlFile << "</tr>" << std::endl;

  htmlFile << "<tr align=\"left\">" << std::endl;
  htmlAnyHisto(runNo,histo2d["TrigPrimMonitor/TP Map/Data Only"],"ieta","iphi", 10, htmlFile,htmlDir);  
  htmlAnyHisto(runNo,histo2d["TrigPrimMonitor/TP Map/Emul Only"],"ieta","iphi", 10, htmlFile,htmlDir);  
  htmlFile << "</tr>" << std::endl;
  
  htmlFile << "<tr align=\"left\">" << std::endl;
  htmlAnyHisto(runNo,histo2d["TrigPrimMonitor/TP Map/Missing Data"],"ieta","iphi", 10, htmlFile,htmlDir);  
  htmlAnyHisto(runNo,histo2d["TrigPrimMonitor/TP Map/Missing Emul"],"ieta","iphi", 10, htmlFile,htmlDir);  
  htmlFile << "</tr>" << std::endl;

  //----------------------------------------

  //------------- Energy (HBHE) --------------
  htmlFile << "<tr><td>&nbsp;&nbsp;&nbsp;<h3>Energy Plots (HBHE)</h3></td></tr>" << std::endl;

  htmlFile << "<tr align=\"left\">" << std::endl;
  htmlAnyHisto(runNo,histo1d["TrigPrimMonitor/Energy Plots/HBHE/Energy HBHE - All Data"],"ieta","iphi", 10, htmlFile,htmlDir,true);  
  htmlAnyHisto(runNo,histo1d["TrigPrimMonitor/Energy Plots/HBHE/Energy HBHE - All Emul"],"ieta","iphi", 10, htmlFile,htmlDir,true);  
  htmlFile << "</tr>" << std::endl;

  htmlFile << "<tr align=\"left\">" << std::endl;
  htmlAnyHisto(runNo,histo1d["TrigPrimMonitor/Energy Plots/HBHE/Energy HBHE - Data Only"],"ieta","iphi", 10, htmlFile,htmlDir,true);  
  htmlAnyHisto(runNo,histo1d["TrigPrimMonitor/Energy Plots/HBHE/Energy HBHE - Emul Only"],"ieta","iphi", 10, htmlFile,htmlDir,true);  
  htmlFile << "</tr>" << std::endl;

  htmlFile << "<tr align=\"left\">" << std::endl;
  htmlAnyHisto(runNo,histo1d["TrigPrimMonitor/Energy Plots/HBHE/Energy HBHE - Missing Data"],"ieta","iphi", 10, htmlFile,htmlDir,true);  
  htmlAnyHisto(runNo,histo1d["TrigPrimMonitor/Energy Plots/HBHE/Energy HBHE - Missing Emul"],"ieta","iphi", 10, htmlFile,htmlDir,true);  
  htmlFile << "</tr>" << std::endl;

  htmlFile << "<tr align=\"left\">" << std::endl;
  htmlAnyHisto(runNo,histo1d["TrigPrimMonitor/Energy Plots/HBHE/Energy HBHE - Mismatched FG"],"ieta","iphi", 10, htmlFile,htmlDir,true);  
  htmlFile << "</tr>" << std::endl;
  //----------------------------------------

  //------------- Energy (HF) --------------
  htmlFile << "<tr><td>&nbsp;&nbsp;&nbsp;<h3>Energy Plots (HF)</h3></td></tr>" << std::endl;

  htmlFile << "<tr align=\"left\">" << std::endl;
  htmlAnyHisto(runNo,histo1d["TrigPrimMonitor/Energy Plots/HF/Energy HF - All Data"],"ieta","iphi", 10, htmlFile,htmlDir,true);  
  htmlAnyHisto(runNo,histo1d["TrigPrimMonitor/Energy Plots/HF/Energy HF - All Emul"],"ieta","iphi", 10, htmlFile,htmlDir,true);  
  htmlFile << "</tr>" << std::endl;

  htmlFile << "<tr align=\"left\">" << std::endl;
  htmlAnyHisto(runNo,histo1d["TrigPrimMonitor/Energy Plots/HF/Energy HF - Data Only"],"ieta","iphi", 10, htmlFile,htmlDir,true);  
  htmlAnyHisto(runNo,histo1d["TrigPrimMonitor/Energy Plots/HF/Energy HF - Emul Only"],"ieta","iphi", 10, htmlFile,htmlDir,true);  
  htmlFile << "</tr>" << std::endl;

  htmlFile << "<tr align=\"left\">" << std::endl;
  htmlAnyHisto(runNo,histo1d["TrigPrimMonitor/Energy Plots/HF/Energy HF - Missing Data"],"ieta","iphi", 10, htmlFile,htmlDir,true);  
  htmlAnyHisto(runNo,histo1d["TrigPrimMonitor/Energy Plots/HF/Energy HF - Missing Emul"],"ieta","iphi", 10, htmlFile,htmlDir,true);  
  htmlFile << "</tr>" << std::endl;

  htmlFile << "<tr align=\"left\">" << std::endl;
  htmlAnyHisto(runNo,histo1d["TrigPrimMonitor/Energy Plots/HF/Energy HF - Mismatched FG"],"ieta","iphi", 10, htmlFile,htmlDir,true);  
  htmlFile << "</tr>" << std::endl;

  //----------------------------------------

  htmlFile << "</table>" << std::endl;
  htmlFile << "<br>" << std::endl;   
  
  // html page footer
  htmlFile << "</body> " << std::endl;
  htmlFile << "</html> " << std::endl;

  htmlFile.close();

  if (showTiming_)
    {
      cpu_timer.stop();  std::cout <<"TIMER:: HcalTrigPrimClient HTML OUTPUT -> "<<cpu_timer.cpuTime()<<std::endl;
    }
  return;
} // void HcalTrigPrimClient::htmlOutput(...)


void HcalTrigPrimClient::createTests(){

  if(debug_>0) std::cout << "HcalTrigPrimClient: creating tests" << std::endl;

  if(!dbe_) return;

  return;
}

void HcalTrigPrimClient::loadHistograms(TFile* infile){

  TNamed* tnd = (TNamed*)infile->Get("DQMData/Hcal/TrigPrimMonitor/TrigPrim Event Number");
  if(tnd){
    string s =tnd->GetTitle();
    ievt_ = -1;
    sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &ievt_);
  }

  return;
}
