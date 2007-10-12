#include <DQM/HcalMonitorClient/interface/HcalMonitorClient.h>
#include "FWCore/ServiceRegistry/interface/Service.h"

//--------------------------------------------------------
HcalMonitorClient::HcalMonitorClient(const ParameterSet& ps): DQMAnalyzer(ps){
  initialize(ps);
}

HcalMonitorClient::HcalMonitorClient(): DQMAnalyzer(){}

//--------------------------------------------------------
HcalMonitorClient::~HcalMonitorClient(){

  cout << "HcalMonitorClient: Exit ..." << endl;

  if( dataformat_client_ ) delete dataformat_client_;
  if( digi_client_ )       delete digi_client_;
  if( rechit_client_ )     delete rechit_client_;
  if( pedestal_client_ )   delete pedestal_client_;
  if( led_client_ )        delete led_client_;
  if( hot_client_ )        delete hot_client_;
  if( mui_ )               delete mui_;
}

//--------------------------------------------------------
void HcalMonitorClient::initialize(const ParameterSet& ps){

  cout << endl;
  cout << " *** Hcal Monitor Client ***" << endl;
  cout << endl;

  dataformat_client_ = 0; digi_client_ = 0;
  rechit_client_ = 0; pedestal_client_ = 0;
  led_client_ = 0; hot_client_ = 0; 
  lastResetTime_=0;

  // MonitorDaemon switch
  enableMonitorDaemon_ = ps.getUntrackedParameter<bool>("enableMonitorDaemon", true);
  if ( enableMonitorDaemon_ ) cout << "-->enableMonitorDaemon switch is ON" << endl;
  else cout << "-->enableMonitorDaemon switch is OFF" << endl;

  mui_ = new MonitorUIRoot();

  // DQM ROOT input
  inputFile_ = ps.getUntrackedParameter<string>("inputFile", "");
  if(inputFile_.size()!=0 ) cout << "-->reading DQM input from " << inputFile_ << endl;
  
  if( ! enableMonitorDaemon_ ) {  
    if( inputFile_.size() != 0 && dbe_!=NULL){
      dbe_->open(inputFile_);
      dbe_->showDirStructure();     
    }
  }

  //histogram reset freqency, update frequency, timeout
  resetUpdate_ = ps.getUntrackedParameter<int>("resetFreqUpdates",-1);  //number of collector updates
  if(resetUpdate_!=-1) cout << "-->Will reset histograms every " << resetUpdate_ <<" collector updates." << endl;
  resetEvents_ = ps.getUntrackedParameter<int>("resetFreqEvents",-1);   //number of real events
  if(resetEvents_!=-1) cout << "-->Will reset histograms every " << resetEvents_ <<" events." << endl;
  resetTime_ = ps.getUntrackedParameter<int>("resetFreqTime",-1);       //number of minutes
  if(resetTime_!=-1) cout << "-->Will reset histograms every " << resetTime_ <<" minutes." << endl;
  resetLS_ = ps.getUntrackedParameter<int>("resetFreqLS",-1);       //number of lumisections
  if(resetLS_!=-1) cout << "-->Will reset histograms every " << resetLS_ <<" lumi sections." << endl;

  // base Html output directory
  baseHtmlDir_ = ps.getUntrackedParameter<string>("baseHtmlDir", "");
  if( baseHtmlDir_.size() != 0 ) 
    cout << "-->HTML output will go to baseHtmlDir = '" << baseHtmlDir_ << "'" << endl;
  else cout << "-->HTML output is disabled" << endl;
  
  // exit on end job switch
  enableExit_ = ps.getUntrackedParameter<bool>("enableExit", true);
  if( enableExit_ ) cout << "-->enableExit switch is ON" << endl;
  else cout << "-->enableExit switch is OFF" << endl;

  
  runningStandalone_ = ps.getUntrackedParameter<bool>("runningStandalone", false);
  if( runningStandalone_ ) cout << "-->standAlone switch is ON" << endl;
  else cout << "-->standAlone switch is OFF" << endl;

  // global ROOT style
  gStyle->Reset("Default");
  gStyle->SetCanvasColor(0);
  gStyle->SetPadColor(0);
  gStyle->SetFillColor(0);
  gStyle->SetTitleFillColor(10);
  //  gStyle->SetOptStat(0);
  gStyle->SetPalette(1);

  // clients' constructors
  if( ps.getUntrackedParameter<bool>("DataFormatClient", false) )
    dataformat_client_   = new HcalDataFormatClient(ps, dbe_);
  if( ps.getUntrackedParameter<bool>("DigiClient", false) )
    digi_client_         = new HcalDigiClient(ps, dbe_);
  if( ps.getUntrackedParameter<bool>("RecHitClient", false) )
    rechit_client_       = new HcalRecHitClient(ps, dbe_);
  if( ps.getUntrackedParameter<bool>("PedestalClient", false) )
    pedestal_client_     = new HcalPedestalClient(ps, dbe_);
  if( ps.getUntrackedParameter<bool>("LEDClient", false) )
    led_client_          = new HcalLEDClient(ps, dbe_);
  if( ps.getUntrackedParameter<bool>("HotCellClient", false) )
    hot_client_          = new HcalHotCellClient(ps, dbe_);

  return;
}

//--------------------------------------------------------
// remove all MonitorElements and directories
void HcalMonitorClient::removeAllME(){

  if(dbe_==NULL) return;

  // go to top directory
  dbe_->cd();
  // remove MEs at top directory
  dbe_->removeContents(); 
  // remove directory (including subdirectories recursively)
  if(dbe_->dirExists("Collector"))
     dbe_->rmdir("Collector");
  if(dbe_->dirExists("Summary"))
  dbe_->rmdir("Summary");
}

//--------------------------------------------------------
///do a reset of all monitor elements...
void HcalMonitorClient::resetAllME() {
  if( dataformat_client_ ) dataformat_client_->resetAllME();
  if( digi_client_ )       digi_client_->resetAllME();
  if( rechit_client_ )     rechit_client_->resetAllME();
  if( pedestal_client_ )   pedestal_client_->resetAllME();
  if( led_client_ )        led_client_->resetAllME();
  if( hot_client_ )         hot_client_->resetAllME();
  return;
}

//--------------------------------------------------------
void HcalMonitorClient::beginJob(const EventSetup& c){
  // call DQMAnalyzer in the beginning 
  DQMAnalyzer::beginJob(c);

  if( debug_ ) cout << "HcalMonitorClient: beginJob" << endl;
  
  ievt_ = 0;

  if( dataformat_client_ ) dataformat_client_->beginJob();
  if( digi_client_ )       digi_client_->beginJob();
  if( rechit_client_ )     rechit_client_->beginJob();
  if( pedestal_client_ )   pedestal_client_->beginJob(c);
  if( led_client_ )        led_client_->beginJob(c);
  if( hot_client_ )         hot_client_->beginJob();

  return;
}

//--------------------------------------------------------
void HcalMonitorClient::beginRun(const Run& r, const EventSetup& c) {
  // call DQMAnalyzer in the beginning 
  DQMAnalyzer::beginRun(r, c);

  cout << endl;
  cout << "HcalMonitorClient: Standard beginRun() for run " << r.id().run() << endl;
  cout << endl;
  if( dataformat_client_ ) dataformat_client_->beginRun();
  if( digi_client_ )       digi_client_->beginRun();
  if( rechit_client_ )     rechit_client_->beginRun();
  if( pedestal_client_ )   pedestal_client_->beginRun();
  if( led_client_ )        led_client_->beginRun();
  if( hot_client_ )        hot_client_->beginRun();

  return;
}

//--------------------------------------------------------
void HcalMonitorClient::endJob(void) {

  if( debug_ ) cout << "HcalMonitorClient: endJob, ievt = " << ievt_ << endl;

  if( dataformat_client_ ) dataformat_client_->endJob();
  if( digi_client_ )  digi_client_->endJob();
  if( rechit_client_ )  rechit_client_->endJob();
  if( hot_client_ ) hot_client_->endJob();
  if( pedestal_client_ ) pedestal_client_->endJob();
  if( led_client_ ) led_client_->endJob();
  
  DQMAnalyzer::endJob();
  return;
}

//--------------------------------------------------------
void HcalMonitorClient::endRun(const Run& r, const EventSetup& c) {

  cout << endl;
  cout << "Standard endRun() for run " << r.id().run() << endl;
  cout << endl;

  if( debug_ ) printf("HcalMonitorClient: processed events: %d\n",ievt_);

  printf("==>Creating report after run end condition\n");
  if(inputFile_.size()!=0) report(true);
  else report(false);
  
  if( hot_client_ )         hot_client_->endRun();
  if( dataformat_client_ ) dataformat_client_->endRun();
  if( digi_client_ )       digi_client_->endRun();
  if( rechit_client_ )     rechit_client_->endRun();
  if( pedestal_client_ )   pedestal_client_->endRun();
  if( led_client_ )        led_client_->endRun();

  // call DQMAnalyzer at the end
  DQMAnalyzer::endRun(r,c); 
  // this is an effective way to avoid ROOT memory leaks ...
  if( enableExit_ ) {
    cout << endl;
    cout << ">>> exit after End-Of-Run <<<" << endl;
    cout << endl;
    
    endJob();
    throw cms::Exception("End of Job")
      << "HcalMonitorClient: Done processing...\n";
  }
}

//--------------------------------------------------------
void HcalMonitorClient::beginLuminosityBlock(const LuminosityBlock &l, const EventSetup &c) {
  // call DQMAnalyzer in the beginning 
  DQMAnalyzer::beginLuminosityBlock(l,c);
  // then do your thing
  if(actonLS_ && !prescale()){
    // do scheduled tasks...
  }
}

//--------------------------------------------------------
void HcalMonitorClient::endLuminosityBlock(const LuminosityBlock &l, const EventSetup &c) {
  // then do your thing
  if(actonLS_ && !prescale()){
    // do scheduled tasks...
    analyze();
  }
  // call DQMAnalyzer at the end 
  DQMAnalyzer::endLuminosityBlock(l,c);
  return;
}

//--------------------------------------------------------
void HcalMonitorClient::analyze(const Event& e, const edm::EventSetup& eventSetup){
  DQMAnalyzer::analyze(e,eventSetup);

  if(resetEvents_>0 && (ievent_%resetEvents_)==0) resetAllME();
  if(resetLS_>0 && (ilumisec_%resetLS_)==0) resetAllME();
  int deltaT = itime_-lastResetTime_;
  if(resetTime_>0 && (deltaT>resetTime_)){
    resetAllME();
    lastResetTime_ = itime_;
  }
  //  if(nupdates%resetUpdate_)==0) resetAllME();

  if ( runningStandalone_ || prescale()) return;
  else analyze();
}


//--------------------------------------------------------
void HcalMonitorClient::analyze(){
  
  ievt_++;
  printf("\nHcal Monitor Client heartbeat....\n");

  createTests();  
  mui_->doMonitoring();
  dbe_->runQTests();

  if( dataformat_client_ ) dataformat_client_->analyze(); 	
  if( digi_client_ )       digi_client_->analyze(); 
  if( rechit_client_ )     rechit_client_->analyze(); 
  if( pedestal_client_ )   pedestal_client_->analyze();      
  if( led_client_ )        led_client_->analyze(); 
  if( hot_client_ )         hot_client_->analyze(); 

  return;
}

//--------------------------------------------------------
void HcalMonitorClient::createTests(void){

  if( dataformat_client_ ) dataformat_client_->createTests(); 
  if( digi_client_ )       digi_client_->createTests(); 
  if( rechit_client_ )     rechit_client_->createTests(); 
  if( pedestal_client_ )   pedestal_client_->createTests(); 
  if( led_client_ )        led_client_->createTests(); 
  if( hot_client_ )        hot_client_->createTests(); 

  return;
}

//--------------------------------------------------------
void HcalMonitorClient::report(bool doUpdate) {
  
  if( debug_ ) cout << "HcalMonitorClient: creating report, ievt = " << ievt_ << endl;
  
  if(doUpdate){
    createTests();  
    dbe_->runQTests();
  }

  if( dataformat_client_ ) dataformat_client_->report();
  if( digi_client_ ) digi_client_->report();
  if( led_client_ ) led_client_->report();
  if( pedestal_client_ ) pedestal_client_->report();
  if( rechit_client_ ) rechit_client_->report();
  if( hot_client_ ) hot_client_->report();
  
  map<string, vector<QReport*> > errE, errW, errO;
  if( hot_client_ ) hot_client_->getErrors(errE,errW,errO);
  if( led_client_ ) led_client_->getErrors(errE,errW,errO);
  if( pedestal_client_ ) pedestal_client_->getErrors(errE,errW,errO);
  if( digi_client_ ) digi_client_->getErrors(errE,errW,errO);
  if( rechit_client_ ) rechit_client_->getErrors(errE,errW,errO);
  if( dataformat_client_ ) dataformat_client_->getErrors(errE,errW,errO);


  save();
  if( baseHtmlDir_.size() != 0 ) htmlOutput();

  return;
}


void HcalMonitorClient::htmlOutput(void){

  cout << "Preparing HcalMonitorClient html output ..." << endl;

  char tmp[10];
  if(irun_!=-1) sprintf(tmp, "%09d", irun_);
  else sprintf(tmp, "%09d", 0);
  string htmlDir = baseHtmlDir_ + "/" + tmp + "/";
  system(("/bin/mkdir -p " + htmlDir).c_str());

  ofstream htmlFile;
  htmlFile.open((htmlDir + "index.html").c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlFile << "<html>  " << endl;
  htmlFile << "<head>  " << endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlFile << " http-equiv=\"content-type\">  " << endl;
  htmlFile << "  <title>Hcal Data Quality Monitor</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<body>  " << endl;
  htmlFile << "<br>  " << endl;
  htmlFile << "<center><h1>Hcal Data Quality Monitor</h1></center>" << endl;
  htmlFile << "<h2>Run Number:&nbsp&nbsp&nbsp" << endl;
  htmlFile << "<span style=\"color: rgb(0, 0, 153);\">" << irun_ <<"</span></h2> " << endl;
  htmlFile << "<h2>Events processed:&nbsp&nbsp&nbsp" << endl;
  htmlFile << "<span style=\"color: rgb(0, 0, 153);\">" << ievent_ <<"</span></h2> " << endl;
  htmlFile << "<hr>" << endl;
  htmlFile << "<ul>" << endl;

  string htmlName;
  if( dataformat_client_ ) {
    htmlName = "HcalDataFormatClient.html";
    dataformat_client_->htmlOutput(irun_, htmlDir, htmlName);
    htmlFile << "<table border=0 WIDTH=\"50%\"><tr>" << endl;
    htmlFile << "<td WIDTH=\"35%\"><a href=\"" << htmlName << "\">Data Format Monitor</a></td" << endl;
    if(dataformat_client_->hasErrors()) htmlFile << "<td bgcolor=red align=center>This monitor task has errors.</td>" << endl;
    else if(dataformat_client_->hasWarnings()) htmlFile << "<td bgcolor=yellow align=center>This monitor task has warnings.</td>" << endl;
    else if(dataformat_client_->hasOther()) htmlFile << "<td bgcolor=aqua align=center>This monitor task has messages.</td>" << endl;
    else htmlFile << "<td bgcolor=lime align=center>This monitor task has no problems</td>" << endl;
    htmlFile << "</table>" << endl;
  }
  if( digi_client_ ) {
    htmlName = "HcalDigiClient.html";
    digi_client_->htmlOutput(irun_, htmlDir, htmlName);
    htmlFile << "<table border=0 WIDTH=\"50%\"><tr>" << endl;
    htmlFile << "<td WIDTH=\"35%\"><a href=\"" << htmlName << "\">Digi Monitor</a></td" << endl;
    if(digi_client_->hasErrors()) htmlFile << "<td bgcolor=red align=center>This monitor task has errors.</td>" << endl;
    else if(digi_client_->hasWarnings()) htmlFile << "<td bgcolor=yellow align=center>This monitor task has warnings.</td>" << endl;
    else if(digi_client_->hasOther()) htmlFile << "<td bgcolor=aqua align=center>This monitor task has messages.</td>" << endl;
    else htmlFile << "<td bgcolor=lime align=center>This monitor task has no problems</td>" << endl;
    htmlFile << "</table>" << endl;
  }
  if( rechit_client_ ) {
    htmlName = "HcalRecHitClient.html";
    rechit_client_->htmlOutput(irun_, htmlDir, htmlName);
    htmlFile << "<table border=0 WIDTH=\"50%\"><tr>" << endl;
    htmlFile << "<td WIDTH=\"35%\"><a href=\"" << htmlName << "\">RecHit Monitor</a></td" << endl;
    if(rechit_client_->hasErrors()) htmlFile << "<td bgcolor=red align=center>This monitor task has errors.</td>" << endl;
    else if(rechit_client_->hasWarnings()) htmlFile << "<td bgcolor=yellow align=center>This monitor task has warnings.</td>" << endl;
    else if(rechit_client_->hasOther()) htmlFile << "<td bgcolor=aqua align=center>This monitor task has messages.</td>" << endl;
    else htmlFile << "<td bgcolor=lime align=center>This monitor task has no problems</td>" << endl;
    htmlFile << "</table>" << endl;
  }
  if( hot_client_ ) {
    htmlName = "HcalHotCellClient.html";
    hot_client_->htmlOutput(irun_, htmlDir, htmlName);
    htmlFile << "<table border=0 WIDTH=\"50%\"><tr>" << endl;
    htmlFile << "<td WIDTH=\"35%\"><a href=\"" << htmlName << "\">Hot Cell Monitor</a></td" << endl;
    if(hot_client_->hasErrors()) htmlFile << "<td bgcolor=red align=center>This monitor task has errors.</td>" << endl;
    else if(hot_client_->hasWarnings()) htmlFile << "<td bgcolor=yellow align=center>This monitor task has warnings.</td>" << endl;
    else if(hot_client_->hasOther()) htmlFile << "<td bgcolor=aqua align=center>This monitor task has messages.</td>" << endl;
    else htmlFile << "<td bgcolor=lime align=center>This monitor task has no problems</td>" << endl;
    htmlFile << "</table>" << endl;
  }
  if( pedestal_client_) {
    htmlName = "HcalPedestalClient.html";
    pedestal_client_->htmlOutput(irun_, htmlDir, htmlName);
    htmlFile << "<table border=0 WIDTH=\"50%\"><tr>" << endl;
    htmlFile << "<td WIDTH=\"35%\"><a href=\"" << htmlName << "\">Pedestal Monitor</a></td" << endl;
    
    if(pedestal_client_->hasErrors()) htmlFile << "<td bgcolor=red align=center>This monitor task has errors.</td>" << endl;
    else if(pedestal_client_->hasWarnings()) htmlFile << "<td bgcolor=yellow align=center>This monitor task has warnings.</td>" << endl;
    else if(pedestal_client_->hasOther()) htmlFile << "<td bgcolor=aqua align=center>This monitor task has messages.</td>" << endl;
    else htmlFile << "<td bgcolor=lime align=center>This monitor task has no problems</td>" << endl;
    
    htmlFile << "</table>" << endl;
  }

  if( led_client_) {
    htmlName = "HcalLEDClient.html";
    led_client_->htmlOutput(irun_, htmlDir, htmlName);
    htmlFile << "<table border=0 WIDTH=\"50%\"><tr>" << endl;
    htmlFile << "<td WIDTH=\"35%\"><a href=\"" << htmlName << "\">LED Monitor</a></td" << endl;
    
    if(led_client_->hasErrors()) htmlFile << "<td bgcolor=red align=center>This monitor task has errors.</td>" << endl;
    else if(led_client_->hasWarnings()) htmlFile << "<td bgcolor=yellow align=center>This monitor task has warnings.</td>" << endl;
    else if(led_client_->hasOther()) htmlFile << "<td bgcolor=aqua align=center>This monitor task has messages.</td>" << endl;
    else htmlFile << "<td bgcolor=lime align=center>This monitor task has no problems</td>" << endl;
    
    htmlFile << "</table>" << endl;
  }
  
  htmlFile << "</ul>" << endl;


  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  htmlFile.close();
  cout << "HcalMonitorClient html output done..." << endl;
  
  return;
}

void HcalMonitorClient::offlineSetup(){
  //  cout << endl;
  //  cout << " *** Hcal Generic Monitor Client, for offline operation***" << endl;
  //  cout << endl;

  dataformat_client_ = 0; digi_client_ = 0;
  rechit_client_ = 0; pedestal_client_ = 0;
  led_client_ = 0;  hot_client_ = 0;
  
  // base Html output directory
  baseHtmlDir_ = ".";
  
  // clients' constructors
  hot_client_          = new HcalHotCellClient();
  dataformat_client_   = new HcalDataFormatClient();
  rechit_client_       = new HcalRecHitClient();
  digi_client_         = new HcalDigiClient();
  pedestal_client_     = new HcalPedestalClient();
  led_client_          = new HcalLEDClient();

  return;
}

void HcalMonitorClient::loadHistograms(TFile* infile, const char* fname){
  
  if(!infile){
    throw cms::Exception("Incomplete configuration") << 
      "HcalMonitorClient: this histogram file is bad! " <<endl;
    return;
  }
  /*
  TNamed* tnd = (TNamed*)infile->Get("DQMData/HcalMonitor/RUN NUMBER");
  string s =tnd->GetTitle();
  irun_ = -1;
  sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &irun_);
  
  tnd = (TNamed*)infile->Get("DQMData/HcalMonitor/EVT NUMBER");
  s =tnd->GetTitle();
  mon_evt_ = -1;
  sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &mon_evt_);

  tnd = (TNamed*)infile->Get("DQMData/HcalMonitor/EVT MASK");
  s =tnd->GetTitle();
  int mask = -1;
  sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &mask);
  
  if(mask&HCAL_BEAM_TRIGGER) runtype_ = "BEAM RUN";
  if(mask&DO_HCAL_PED_CALIBMON){ runtype_ = "PEDESTAL RUN";
    if(mask&HCAL_BEAM_TRIGGER) runtype_ = "BEAM AND PEDESTALS";
  }
  if(mask&DO_HCAL_LED_CALIBMON) runtype_ = "LED RUN";
  if(mask&DO_HCAL_LASER_CALIBMON) runtype_ = "LASER RUN";
  
  status_="unknown";
  tnd = (TNamed*)infile->Get("DQMData/HcalMonitor/EVT MASK");
  s = tnd->GetTitle();
  status_ = "unknown";
  if( s.substr(2,1) == "0" ) status_ = "begin-of-run";
  if( s.substr(2,1) == "1" ) status_ = "running";
  if( s.substr(2,1) == "2" ) status_ = "end-of-run";
  

  if(hot_client_) hot_client_->loadHistograms(infile);
  if(dataformat_client_) dataformat_client_->loadHistograms(infile);
  if(rechit_client_) rechit_client_->loadHistograms(infile);
  if(digi_client_) digi_client_->loadHistograms(infile);
  if(pedestal_client_) pedestal_client_->loadHistograms(infile);
  if(led_client_) led_client_->loadHistograms(infile);
  */
  return;

}


void HcalMonitorClient::dumpHistograms(int& runNum, vector<TH1F*> &hist1d,vector<TH2F*> &hist2d){
  
  hist1d.clear(); 
  hist2d.clear(); 

  /*
  if(hot_client_) hot_client_->dumpHistograms(names,meanX,meanY,rmsX,rmsY);
  if(dataformat_client) dataformat_client_->dumpHistograms(names,meanX,meanY,rmsX,rmsY);
  if(rechit_client_) rechit_client_->dumpHistograms(names,meanX,meanY,rmsX,rmsY);
  if(digi_client_) digi_client_->dumpHistograms(names,meanX,meanY,rmsX,rmsY);
  if(pedestal_client_) pedestal_client_->dumpHistograms(names,meanX,meanY,rmsX,rmsY);
  if(led_client_) led_client_->dumpHistograms(names,meanX,meanY,rmsX,rmsY);
  */
 return;
}

#include "FWCore/Framework/interface/MakerMacros.h"
#include <DQM/HcalMonitorClient/interface/HcalMonitorClient.h>

DEFINE_FWK_MODULE(HcalMonitorClient);
