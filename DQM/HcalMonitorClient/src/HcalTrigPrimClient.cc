#include <DQM/HcalMonitorClient/interface/HcalTrigPrimClient.h>
#include <DQM/HcalMonitorClient/interface/HcalClientUtils.h>
#include <DQM/HcalMonitorClient/interface/HcalHistoUtils.h>

HcalTrigPrimClient::HcalTrigPrimClient(){}


void HcalTrigPrimClient::init(const ParameterSet& ps, DQMStore* dbe, string clientName)
{
  //Call the base class first
  HcalBaseClient::init(ps,dbe,clientName);

 if (showTiming_)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  if (debug_>0) std::cout <<"<HcalTrigPrimClient> init(const ParameterSet& ps, DQMStore* dbe, string clientName)"<<std::endl;

  for(int i=0; i<10; i++) tpSpectrum_[i] = 0;

  tpCount_ = 0;
  tpCountThr_ = 0;
  tpSize_ = 0;
  tpSpectrumAll_ = 0;
  tpETSumAll_ = 0;
  tpSOI_ET_ = 0;
  OCC_ETA_ = 0;
  OCC_PHI_ = 0;
  OCC_ELEC_VME_ = 0;
  OCC_ELEC_DCC_ = 0;
  OCC_MAP_GEO_ = 0;

  OCC_MAP_THR_ = 0;
  EN_ETA_ = 0;
  EN_PHI_ = 0;
  EN_ELEC_VME_ = 0;
  EN_ELEC_DCC_ = 0;
  EN_MAP_GEO_ = 0;

  TPTiming_ = 0;
  TPTimingTop_ = 0;
  TPTimingBot_ = 0; 
  TPOcc_ = 0;
  TP_ADC_ = 0;
  MAX_ADC_ = 0;
  TS_MAX_ = 0;
  TPvsDigi_ = 0;
  
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

  if ( cloneME_ ) {
    
    for(int i=0; i<10; i++) if(tpSpectrum_[i]) delete tpSpectrum_[i];
    
    if(tpCount_) delete tpCount_;
    if(tpCountThr_) delete tpCountThr_;
    if(tpSize_) delete tpSize_;
    if(tpSpectrumAll_) delete tpSpectrumAll_;
    if(tpETSumAll_) delete tpETSumAll_;
    if(tpSOI_ET_) delete tpSOI_ET_;
    if(OCC_ETA_) delete OCC_ETA_;
    if(OCC_PHI_) delete OCC_PHI_;
    if(OCC_ELEC_VME_) delete OCC_ELEC_VME_;
    if(OCC_ELEC_DCC_) delete OCC_ELEC_DCC_;
    if(OCC_MAP_GEO_) delete OCC_MAP_GEO_;

    if(OCC_MAP_THR_) delete OCC_MAP_THR_;
    if(EN_ETA_) delete EN_ETA_;
    if(EN_PHI_) delete EN_PHI_;
    if(EN_ELEC_VME_) delete EN_ELEC_VME_;
    if(EN_ELEC_DCC_) delete EN_ELEC_DCC_;
    if(EN_MAP_GEO_) delete EN_MAP_GEO_;

    if(TPTiming_) delete TPTiming_ ;
    if(TPTimingTop_) delete  TPTimingTop_;
    if(TPTimingBot_) delete  TPTimingBot_;
    if(TPOcc_) delete TPOcc_;
    if(TP_ADC_) delete  TP_ADC_;
    if(MAX_ADC_) delete  MAX_ADC_;
    if(TS_MAX_) delete  TS_MAX_;
    if(TPvsDigi_) delete TPvsDigi_;
  }  

  for(int i=0; i<10; i++) tpSpectrum_[i] = 0;

  tpCount_ = 0;
  tpCountThr_ = 0;
  tpSize_ = 0;
  tpSpectrumAll_ = 0;
  tpETSumAll_ = 0;
  tpSOI_ET_ = 0;
  OCC_ETA_ = 0;
  OCC_PHI_ = 0;
  OCC_ELEC_VME_ = 0;
  OCC_ELEC_DCC_ = 0;
  OCC_MAP_GEO_ = 0;
 
  OCC_MAP_THR_ = 0;
  EN_ETA_ = 0;
  EN_PHI_ = 0;
  EN_ELEC_VME_ = 0;
  EN_ELEC_DCC_ = 0;
  EN_MAP_GEO_ = 0;
  TPTiming_ = 0;
  TPTimingTop_ = 0;
  TPTimingBot_ = 0; 
  TPOcc_ = 0;
  TP_ADC_ = 0;
  MAX_ADC_ = 0;
  TS_MAX_ = 0;
  TPvsDigi_ = 0;

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


  tpCount_ = getHisto("TrigPrimMonitor/Energy Plots/# TP Digis", process_, dbe_, debug_,cloneME_);
  assert(tpCount_!=0);

  tpCountThr_ = getHisto("TrigPrimMonitor/Energy Plots/# TP Digis over Threshold", process_, dbe_, debug_,cloneME_);
  tpSize_ = getHisto("TrigPrimMonitor/Timing Plots/TP Size", process_, dbe_, debug_,cloneME_);  
  char name[150];      
  for (int i=0; i<10; i++) {
    sprintf(name,"TrigPrimMonitor/Energy Plots/TP Spectra by TS/TP Spectrum sample %d",i);
    tpSpectrum_[i]= getHisto(name, process_, dbe_, debug_,cloneME_);
  }
  tpSpectrumAll_ = getHisto("TrigPrimMonitor/Energy Plots/Full TP Spectrum", process_, dbe_, debug_,cloneME_);
  tpETSumAll_ = getHisto("TrigPrimMonitor/Energy Plots/TP ET Sum", process_, dbe_, debug_,cloneME_);
  tpSOI_ET_ = getHisto("TrigPrimMonitor/Energy Plots/TP SOI ET", process_, dbe_, debug_,cloneME_);  
  OCC_ETA_ = getHisto("TrigPrimMonitor/Geometry Plots/TrigPrim Eta Occupancy Map",process_, dbe_, debug_,cloneME_);  
  OCC_PHI_ = getHisto("TrigPrimMonitor/Geometry Plots/TrigPrim Phi Occupancy Map",process_, dbe_, debug_,cloneME_);  
  OCC_ELEC_VME_ = getHisto2("TrigPrimMonitor/Electronics Plots/TrigPrim VME Occupancy Map",process_, dbe_, debug_,cloneME_);  
  OCC_ELEC_DCC_ = getHisto2("TrigPrimMonitor/Electronics Plots/TrigPrim Spigot Occupancy Map",process_, dbe_, debug_,cloneME_);  
  OCC_MAP_GEO_ = getHisto2("TrigPrimMonitor/Geometry Plots/TrigPrim Geo Occupancy Map",process_, dbe_, debug_,cloneME_);  

  OCC_MAP_THR_ = getHisto2("TrigPrimMonitor/Geometry Plots/TrigPrim Geo Threshold Map",process_, dbe_, debug_,cloneME_);  
  EN_ETA_ = getHisto("TrigPrimMonitor/Geometry Plots/TrigPrim Eta Energy Map",process_, dbe_, debug_,cloneME_);  
  EN_PHI_ = getHisto("TrigPrimMonitor/Geometry Plots/TrigPrim Phi Energy Map",process_, dbe_, debug_,cloneME_);  
  EN_ELEC_VME_ = getHisto2("TrigPrimMonitor/Electronics Plots/TrigPrim VME Energy Map",process_, dbe_, debug_,cloneME_);  
  EN_ELEC_DCC_ = getHisto2("TrigPrimMonitor/Electronics Plots/TrigPrim Spigot Energy Map",process_, dbe_, debug_,cloneME_);  
  EN_MAP_GEO_ = getHisto2("TrigPrimMonitor/Geometry Plots/TrigPrim Geo Energy Map",process_, dbe_, debug_,cloneME_);  

  TPTiming_ = getHisto("TrigPrimMonitor/Timing Plots/TP Timing",process_, dbe_, debug_,cloneME_);  
  TPTimingTop_ = getHisto("TrigPrimMonitor/Timing Plots/TP Timing (Top wedges)",process_, dbe_, debug_,cloneME_);  
  TPTimingBot_ = getHisto("TrigPrimMonitor/Timing Plots/TP Timing (Bottom wedges)",process_, dbe_, debug_,cloneME_);  
  TP_ADC_ = getHisto("TrigPrimMonitor/Energy Plots/ADC spectrum positive TP",process_, dbe_, debug_,cloneME_);  
  MAX_ADC_ = getHisto("TrigPrimMonitor/Energy Plots/Max ADC in TP",process_, dbe_, debug_,cloneME_);  
  TS_MAX_ = getHisto("TrigPrimMonitor/Timing Plots/TS with max ADC",process_, dbe_, debug_,cloneME_);  
  TPOcc_ = getHisto2("TrigPrimMonitor/00 TP Occupancy",process_, dbe_, debug_,cloneME_);  
  TPvsDigi_ = getHisto2("TrigPrimMonitor/Electronics Plots/TP vs Digi",process_, dbe_, debug_,cloneME_);  

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
  
  char name[256];
  
  sprintf(name, "%sHcal/TrigPrimMonitor/ZZ Expert Plots/ZZ DQM Expert Plots/TrigPrim Event Number",process_.c_str());
  MonitorElement* me = dbe_->get(name);
  if ( me ) {
    string s = me->valueString();
    ievt_ = -1;
    sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &ievt_);
    if ( debug_ >0) std::cout << "Found '" << name << "'" << std::endl;
  }
  else printf("Didn't find %s\n",name);
  getHistograms();
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
  
  char name[150];     
  sprintf(name,"%sHcal/TrigPrimMonitor/Energy Plots/# TP Digis",process_.c_str());
  resetME(name, dbe_);
  sprintf(name,"%sHcal/TrigPrimMonitor/Energy Plots/# TP Digis over Threshold",process_.c_str());
  resetME(name, dbe_);
  sprintf(name,"%sHcal/TrigPrimMonitor/Timing Plots/TP Size",process_.c_str());
  resetME(name, dbe_);  
  for (int i=0; i<10; i++) {
    sprintf(name,"%sHcal/TrigPrimMonitor/Energy Plots/TP Spectra by TS/TP Spectrum sample %d",process_.c_str(),i);
    resetME(name, dbe_);
  }
  sprintf(name,"%sHcal/TrigPrimMonitor/Energy Plots/Full TP Spectrum",process_.c_str());
  resetME(name, dbe_);
  sprintf(name,"%sHcal/TrigPrimMonitor/Energy Plots/TP ET Sum",process_.c_str());
  resetME(name, dbe_);
  sprintf(name,"%sHcal/TrigPrimMonitor/Energy Plots/TP SOI ET",process_.c_str());
  resetME(name, dbe_);  
  sprintf(name,"%sHcal/TrigPrimMonitor/Geometry Plots/TrigPrim Eta Occupancy Map",process_.c_str());
  resetME(name,dbe_);  
  sprintf(name,"%sHcal/TrigPrimMonitor/Geometry Plots/TrigPrim Phi Occupancy Map",process_.c_str());
  resetME(name,dbe_);  
  sprintf(name,"%sHcal/TrigPrimMonitor/Electronics Plots/ZZ Expert Plots/TrigPrim VME Occupancy Map",process_.c_str());
  resetME(name,dbe_);  
  sprintf(name,"%sHcal/TrigPrimMonitor/Electronics Plots/ZZ Expert Plots/TrigPrim Spigot Occupancy Map",process_.c_str());
  resetME(name,dbe_);  
  sprintf(name,"%sHcal/TrigPrimMonitor/Geometry Plots/TrigPrim Geo Occupancy Map",process_.c_str());
  resetME(name,dbe_);  

  sprintf(name,"%sHcal/TrigPrimMonitor/Geometry Plots/TrigPrim Geo Threshold Map",process_.c_str());
  resetME(name,dbe_);  
  sprintf(name,"%sHcal/TrigPrimMonitor/Geometry Plots/TrigPrim Eta Energy Map",process_.c_str());
  resetME(name,dbe_);  
  sprintf(name,"%sHcal/TrigPrimMonitor/Geometry Plots/TrigPrim Phi Energy Map",process_.c_str());
  resetME(name,dbe_);  
  sprintf(name,"%sHcal/TrigPrimMonitor/Electronics Plots/ZZ Expert Plots/TrigPrim VME Energy Map",process_.c_str());
  resetME(name,dbe_);  
  sprintf(name,"%sHcal/TrigPrimMonitor/Electronics Plots/ZZ Expert Plots/TrigPrim Spigot Energy Map",process_.c_str());
  resetME(name,dbe_);  
  sprintf(name,"%sHcal/TrigPrimMonitor/Geometry Plots/TrigPrim Geo Energy Map",process_.c_str());
  resetME(name,dbe_);  

  sprintf(name,"%sHcal/TrigPrimMonitor/Timing Plots/TP Timing",process_.c_str());
  resetME(name,dbe_);  
  sprintf(name,"%sHcal/TrigPrimMonitor/Timing Plots/TP Timing (Top wedges)",process_.c_str());
  resetME(name,dbe_);  
  sprintf(name,"%sHcal/TrigPrimMonitor/Timing Plots/TP Timing (Bottom Wedges)",process_.c_str());
  resetME(name,dbe_);  
  sprintf(name,"%sHcal/TrigPrimMonitor/Energy Plots/ADC spectrum positive TP",process_.c_str());
  resetME(name,dbe_);  
  sprintf(name,"%sHcal/TrigPrimMonitor/Energy Plots/Max ADC in TP",process_.c_str());
  resetME(name,dbe_);  
  sprintf(name,"%sHcal/TrigPrimMonitor/Timing Plots/TS with max ADC",process_.c_str());
  resetME(name,dbe_);  
  sprintf(name,"%sHcal/TrigPrimMonitor/00 TP Occupancy",process_.c_str());
  resetME(name,dbe_);  
  sprintf(name,"%sHcal/TrigPrimMonitor/Electronics Plots/TP vs Digi",process_.c_str());
  resetME(name,dbe_);  

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
  htmlFile << "  <title>Monitor: Data Format Task output</title> " << std::endl;
  htmlFile << "</head>  " << std::endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << std::endl;
  htmlFile << "<body>  " << std::endl;
  htmlFile << "<br>  " << std::endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << std::endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << std::endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << runNo << "</span></h2>" << std::endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << std::endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">Data Format</span></h2> " << std::endl;
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

//Timing Plots
//Energy Plots
//Electronics Plots
//Geometry Plots
//ZZ Expert Plots
  htmlFile << "<td>&nbsp;&nbsp;&nbsp;<h3>Global Histograms</h3></td></tr>" << std::endl;
  htmlFile << "<tr align=\"left\">" << std::endl;
  htmlAnyHisto(runNo,tpCount_,"# TP Digis"," ", 92, htmlFile,htmlDir);
  htmlAnyHisto(runNo,tpCountThr_,"# TP Digis"," ", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << std::endl;

  htmlFile << "<tr align=\"left\">" << std::endl;
  htmlAnyHisto(runNo,tpSpectrumAll_ ,"TP Energy"," ", 92, htmlFile,htmlDir);
  htmlAnyHisto(runNo,tpETSumAll_,"TP ET Sum"," ", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << std::endl;
  
  htmlFile << "<tr align=\"left\">" << std::endl;
  htmlAnyHisto(runNo,tpSOI_ET_,"TP Energy"," ", 92, htmlFile,htmlDir);
  htmlAnyHisto(runNo,tpSize_,"# Samples"," ", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << std::endl;
  
  htmlFile << "<tr align=\"left\">" << std::endl;
  htmlAnyHisto(runNo,OCC_MAP_GEO_,"iEta","iPhi", 92, htmlFile,htmlDir);
  htmlAnyHisto(runNo,EN_MAP_GEO_,"iEta","iPhi", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << std::endl;

  htmlFile << "<tr align=\"left\">" << std::endl;
  htmlAnyHisto(runNo,OCC_ETA_,"iEta"," ", 92, htmlFile,htmlDir);
  htmlAnyHisto(runNo,EN_ETA_,"iEta"," ", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << std::endl;

  htmlFile << "<tr align=\"left\">" << std::endl;
  htmlAnyHisto(runNo,OCC_PHI_,"iPhi"," ", 92, htmlFile,htmlDir);
  htmlAnyHisto(runNo,EN_PHI_,"iPhi"," ", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << std::endl;

  htmlFile << "<tr align=\"left\">" << std::endl;
  htmlAnyHisto(runNo,OCC_ELEC_VME_,"Slot","Crate Id", 92, htmlFile,htmlDir);
  htmlAnyHisto(runNo,EN_ELEC_VME_,"Slot","Crate Id", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << std::endl;
  
  htmlFile << "<tr align=\"left\">" << std::endl;
  htmlAnyHisto(runNo,OCC_ELEC_DCC_,"Spigot","DCC Id", 92, htmlFile,htmlDir);
  htmlAnyHisto(runNo,EN_ELEC_DCC_,"Spigot","DCC Id", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << std::endl;


  htmlFile << "<tr align=\"left\">" << std::endl;
  htmlAnyHisto(runNo,TPTiming_,"","time", 92, htmlFile,htmlDir);
  htmlAnyHisto(runNo,TP_ADC_,"","raw ADC", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << std::endl;

  htmlFile << "<tr align=\"left\">" << std::endl;
  htmlAnyHisto(runNo,MAX_ADC_,"","raw ADC", 92, htmlFile,htmlDir);
  htmlAnyHisto(runNo,TS_MAX_,"TS","num at TS", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << std::endl;
  
  htmlFile << "<tr align=\"left\">" << std::endl;
  htmlAnyHisto(runNo,TPTimingTop_,"","time", 92, htmlFile,htmlDir);
  htmlAnyHisto(runNo,TPTimingBot_,"","time", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << std::endl;

  htmlFile << "<tr align=\"left\">" << std::endl;
  htmlAnyHisto(runNo,TPOcc_,"iEta","iPhi", 92, htmlFile,htmlDir);  
  htmlAnyHisto(runNo,TPvsDigi_,"Digi","TP", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << std::endl;

  htmlFile << "<tr align=\"left\">" << std::endl;
 
  htmlAnyHisto(runNo,OCC_MAP_THR_,"iEta","iPhi", 92, htmlFile,htmlDir);
  htmlFile << "</tr>" << std::endl;


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
