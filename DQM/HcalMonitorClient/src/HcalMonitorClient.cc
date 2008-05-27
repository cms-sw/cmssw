#include <DQM/HcalMonitorClient/interface/HcalMonitorClient.h>
#include "FWCore/ServiceRegistry/interface/Service.h"

//--------------------------------------------------------
HcalMonitorClient::HcalMonitorClient(const ParameterSet& ps){
  initialize(ps);
}

HcalMonitorClient::HcalMonitorClient(){}

//--------------------------------------------------------
HcalMonitorClient::~HcalMonitorClient(){

  cout << "HcalMonitorClient: Exit ..." << endl;
  if( summary_client_ )    delete summary_client_;
  if( dataformat_client_ ) delete dataformat_client_;
  if( digi_client_ )       delete digi_client_;
  if( rechit_client_ )     delete rechit_client_;
  if( pedestal_client_ )   delete pedestal_client_;
  if( led_client_ )        delete led_client_;
  if( hot_client_ )        delete hot_client_;
  if( dead_client_ )       delete dead_client_;
  if( tp_client_ )         delete tp_client_;
  if (ct_client_ )         delete ct_client_;
  if( mui_ )               delete mui_;
}

//--------------------------------------------------------
void HcalMonitorClient::initialize(const ParameterSet& ps){

  cout << endl;
  cout << " *** Hcal Monitor Client ***" << endl;
  cout << endl;

  irun_=0; ilumisec_=0; ievent_=0; itime_=0;
  actonLS_=false;

  summary_client_ = 0;
  dataformat_client_ = 0; digi_client_ = 0;
  rechit_client_ = 0; pedestal_client_ = 0;
  led_client_ = 0; hot_client_ = 0; dead_client_=0;
  tp_client_=0;
  ct_client_=0;
  lastResetTime_=0;

  debug_ = ps.getUntrackedParameter<bool>("debug", false);
  if(debug_) cout << "HcalMonitorClient: constructor...." << endl;



  // MonitorDaemon switch
  enableMonitorDaemon_ = ps.getUntrackedParameter<bool>("enableMonitorDaemon", true);
  if ( enableMonitorDaemon_ ) cout << "-->enableMonitorDaemon switch is ON" << endl;
  else cout << "-->enableMonitorDaemon switch is OFF" << endl;

  mui_ = new DQMOldReceiver();
  dbe_ = mui_->getBEInterface();

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
  gStyle->SetOptStat("ouemr");
  gStyle->SetPalette(1);

  // clients' constructors
  if( ps.getUntrackedParameter<bool>("SummaryClient", true) ){
    if(debug_) {;}
    cout << "===>DQM Summary Client is ON" << endl;
    summary_client_   = new HcalSummaryClient(ps);
    ///> No init() exists, and we may not need one....
    //summary_client_->init(ps, dbe_,"DataFormatClient");
  }
  if( ps.getUntrackedParameter<bool>("DataFormatClient", false) ){
    if(debug_)   cout << "===>DQM DataFormat Client is ON" << endl;
    dataformat_client_   = new HcalDataFormatClient();
    dataformat_client_->init(ps, dbe_,"DataFormatClient");
  }
  if( ps.getUntrackedParameter<bool>("DigiClient", false) ){
    if(debug_)   cout << "===>DQM Digi Client is ON" << endl;
    digi_client_         = new HcalDigiClient();
    digi_client_->init(ps, dbe_,"DigiClient");
  }
  if( ps.getUntrackedParameter<bool>("RecHitClient", false) ){
    if(debug_)   cout << "===>DQM RecHit Client is ON" << endl;
    rechit_client_       = new HcalRecHitClient();
    rechit_client_->init(ps, dbe_,"RecHitClient");
}
  if( ps.getUntrackedParameter<bool>("PedestalClient", false) ){
    if(debug_)   cout << "===>DQM Pedestal Client is ON" << endl;
    pedestal_client_     = new HcalPedestalClient();
    pedestal_client_->init(ps, dbe_,"PedestalClient"); 
  }
  if( ps.getUntrackedParameter<bool>("LEDClient", false) ){
    if(debug_)   cout << "===>DQM LED Client is ON" << endl;
    led_client_          = new HcalLEDClient();
    led_client_->init(ps, dbe_,"LEDClient"); 
  }
  if( ps.getUntrackedParameter<bool>("HotCellClient", false) ){
    if(debug_)   cout << "===>DQM HotCell Client is ON" << endl;
    hot_client_          = new HcalHotCellClient();
    hot_client_->init(ps, dbe_,"HotCellClient");
  }
  if( ps.getUntrackedParameter<bool>("DeadCellClient", false) ){
    if(debug_)   cout << "===>DQM DeadCell Client is ON" << endl;
    dead_client_          = new HcalDeadCellClient();
    dead_client_->init(ps, dbe_,"DeadCellClient");
  }
  if( ps.getUntrackedParameter<bool>("TrigPrimClient", false) ){
    if(debug_)   cout << "===>DQM TrigPim Client is ON" << endl;
    tp_client_          = new HcalTrigPrimClient();
    tp_client_->init(ps, dbe_,"TrigPrimClient");
  }
  if( ps.getUntrackedParameter<bool>("CaloTowerClient", false) ){
    if(debug_)   cout << "===>DQM TrigPim Client is ON" << endl;
    ct_client_          = new HcalCaloTowerClient();
    ct_client_->init(ps, dbe_,"CaloTowerClient");
  }
  dqm_db_ = new HcalHotCellDbInterface(); 

  // set parameters   
  prescaleEvt_ = ps.getUntrackedParameter<int>("diagnosticPrescaleEvt", -1);
  cout << "===>DQM event prescale = " << prescaleEvt_ << " event(s)"<< endl;

  prescaleLS_ = ps.getUntrackedParameter<int>("diagnosticPrescaleLS", -1);
  cout << "===>DQM lumi section prescale = " << prescaleLS_ << " lumi section(s)"<< endl;
  if (prescaleLS_>0) actonLS_=true;

  prescaleUpdate_ = ps.getUntrackedParameter<int>("diagnosticPrescaleUpdate", -1);
  cout << "===>DQM update prescale = " << prescaleUpdate_ << " update(s)"<< endl;

  prescaleTime_ = ps.getUntrackedParameter<int>("diagnosticPrescaleTime", -1);
  cout << "===>DQM time prescale = " << prescaleTime_ << " minute(s)"<< endl;
  

  // Base folder for the contents of this job
  string subsystemname = ps.getUntrackedParameter<string>("subSystemFolder", "Hcal") ;
  cout << "===>HcalMonitor name = " << subsystemname << endl;
  rootFolder_ = subsystemname + "/";

  
  gettimeofday(&psTime_.startTV,NULL);
  /// get time in milliseconds, convert to minutes
  psTime_.startTime = (psTime_.startTV.tv_sec*1000.0+psTime_.startTV.tv_usec/1000.0);
  psTime_.startTime /= (60.0*1000.0);
  psTime_.elapsedTime=0;
  psTime_.updateTime=0;

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
  if( hot_client_ )        hot_client_->resetAllME();
  if( dead_client_ )       dead_client_->resetAllME();
  if( tp_client_ )         tp_client_->resetAllME();
  if( ct_client_ )         ct_client_->resetAllME();

  return;
}

//--------------------------------------------------------
void HcalMonitorClient::beginJob(const EventSetup& c){

  if( debug_ ) cout << "HcalMonitorClient: beginJob" << endl;
  
  ievt_ = 0;
  if( summary_client_ )    summary_client_->beginJob(dbe_);
  if( dataformat_client_ ) dataformat_client_->beginJob();
  if( digi_client_ )       digi_client_->beginJob();
  if( rechit_client_ )     rechit_client_->beginJob();
  if( pedestal_client_ )   pedestal_client_->beginJob(c);
  if( led_client_ )        led_client_->beginJob(c);
  if( hot_client_ )        hot_client_->beginJob();
  if( dead_client_ )       dead_client_->beginJob();
  if( tp_client_ )         tp_client_->beginJob();
  if( ct_client_ )         ct_client_->beginJob();
  return;
}

//--------------------------------------------------------
void HcalMonitorClient::beginRun(const Run& r, const EventSetup& c) {

  cout << endl;
  cout << "HcalMonitorClient: Standard beginRun() for run " << r.id().run() << endl;
  cout << endl;
  if( summary_client_ )    summary_client_->beginRun();
  if( dataformat_client_ ) dataformat_client_->beginRun();
  if( digi_client_ )       digi_client_->beginRun();
  if( rechit_client_ )     rechit_client_->beginRun();
  if( pedestal_client_ )   pedestal_client_->beginRun();
  if( led_client_ )        led_client_->beginRun();
  if( hot_client_ )        hot_client_->beginRun();
  if( dead_client_ )       dead_client_->beginRun();
  if( tp_client_ )         tp_client_->beginRun();
  if( ct_client_ )         ct_client_->beginRun();
  return;
}

//--------------------------------------------------------
void HcalMonitorClient::endJob(void) {

  if( debug_ ) cout << "HcalMonitorClient: endJob, ievt = " << ievt_ << endl;

  if (summary_client_)         summary_client_->endJob();
  if( dataformat_client_ )     dataformat_client_->endJob();
  if( digi_client_ )           digi_client_->endJob();
  if( rechit_client_ )         rechit_client_->endJob();
  if( hot_client_ )            hot_client_->endJob();
  if( dead_client_ )           dead_client_->endJob();
  if( pedestal_client_ )       pedestal_client_->endJob();
  if( led_client_ )            led_client_->endJob();
  if( tp_client_ )             tp_client_->endJob();
  if( ct_client_ )             ct_client_->endJob();

  /*
  ///Don't leave this here!!!  FIX ME!
  ///Just a temporary example!!
  time_t rawtime;
  time(&rawtime);
  tm* ptm = gmtime(&rawtime);
  char ntime[256];
  sprintf(ntime,"%4d-%02d-%02d %02d:%02d:%02d.0",  
	  ptm->tm_year+1900, 
	  ptm->tm_mon,  ptm->tm_mday,
	  ptm->tm_hour, ptm->tm_min, ptm->tm_sec);
  
  const char* version = "V2test";
  const char* tag = "test3";
  const char* comment = "Test DQM Input";
  const char* detector = "HCAL";
  int iovStart = irun_;
  int iovEnd = irun_;

  try{
    XMLPlatformUtils::Initialize();
    DOMDocument* doc = dqm_db_->createDocument();
    //dqm_db_->createHeader(doc, irun_, getRunStartTime());
    dqm_db_->createHeader(doc,irun_,ntime);
    for(int i=0; i<4; i++){
      HcalSubdetector subdet = HcalBarrel;
      if(i==1) subdet =  HcalEndcap;
      else if(i==2) subdet = HcalForward;
      else if(i==3) subdet = HcalOuter;

      for(int ieta=-42; ieta<=42; ieta++){
	if(ieta==0) continue;
	for(int iphi=1; iphi<=73; iphi++){
	  for(int depth=1; depth<=4; depth++){	    
	    if(!isValidGeom(i, ieta, iphi,depth)) continue;

	    HcalDetId id(subdet,ieta,iphi,depth);
	    HcalDQMChannelQuality::Item item;
	    item.mId = id.rawId();
	    item.mAlgo = 0;
	    item.mMasked = 0;
	    item.mQuality = 2;
	    item.mComment = "First DQM Channel Quality test";
	    dqm_db_->createDataset(doc,item,ntime,version);
	  }
	}
      }
    }
    dqm_db_->createFooter(doc,iovStart, iovEnd,tag,detector,comment);
    dqm_db_->writeDocument(doc,"myTestXML.xml");
    doc->release();
  }
  catch (...){
    std::cerr << "Exception" << std::endl;
  }
  XMLPlatformUtils::Terminate();
  */

  return;
}

//--------------------------------------------------------
void HcalMonitorClient::endRun(const Run& r, const EventSetup& c) {

  cout << endl;
  cout << "Standard endRun() for run " << r.id().run() << endl;
  cout << endl;

  if( debug_ ) printf("HcalMonitorClient: processed events: %d\n",ievt_);

  printf("==>Creating report after run end condition\n");
  if(irun_>1){
    if(inputFile_.size()!=0) report(true);
    else report(false);
  }

  if( summary_client_)      summary_client_->endRun();
  if( hot_client_ )         hot_client_->endRun();
  if( dead_client_ )        dead_client_->endRun(); 
  if( dataformat_client_ )  dataformat_client_->endRun();
  if( digi_client_ )        digi_client_->endRun();
  if( rechit_client_ )      rechit_client_->endRun();
  if( pedestal_client_ )    pedestal_client_->endRun();
  if( led_client_ )         led_client_->endRun();
  if( tp_client_ )          tp_client_->endRun();
  if( ct_client_ )          ct_client_->endRun();

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

  return;
}

//--------------------------------------------------------
void HcalMonitorClient::analyze(const Event& e, const edm::EventSetup& eventSetup){

  if (debug_) cout <<"Entered HcalMonitorClient::analyze(const Evt...)"<<endl;
  
  if(resetEvents_>0 && (ievent_%resetEvents_)==0) resetAllME();
  if(resetLS_>0 && (ilumisec_%resetLS_)==0) resetAllME();
  int deltaT = itime_-lastResetTime_;
  if(resetTime_>0 && (deltaT>resetTime_)){
    resetAllME();
    lastResetTime_ = itime_;
  }


  // environment datamembers
  irun_     = e.id().run();
  ilumisec_ = e.luminosityBlock();
  ievent_   = e.id().event();
  itime_    = e.time().value();

  if (debug_) cout << "HcalMonitorClient: evts: "<< ievt_ << ", run: " << irun_ << ", LS: " << ilumisec_ << ", evt: " << ievent_ << ", time: " << itime_ << endl; 

  ievt_++; //I think we want our web pages, etc. to display this counter (the number of events used in the task) rather than nevt_ (the number of times the MonitorClient analyze function below is called) -- Jeff, 1/22/08
  if ( runningStandalone_ || prescale()) return;

  else analyze();
}


//--------------------------------------------------------
void HcalMonitorClient::analyze(){
  if (debug_) cout <<"Entered HcalMonitorClient::analyze()"<<endl;

  //nevt_++; // counter not currently displayed anywhere 
  if(debug_) printf("\nHcal Monitor Client heartbeat....\n");
  
  createTests();  
  mui_->doMonitoring();
  dbe_->runQTests();

  if( summary_client_ )    summary_client_->analyze(); 	
  if( dataformat_client_ ) dataformat_client_->analyze(); 	
  if( digi_client_ )       digi_client_->analyze(); 
  if( rechit_client_ )     rechit_client_->analyze(); 
  if( pedestal_client_ )   pedestal_client_->analyze();      
  if( led_client_ )        led_client_->analyze(); 
  if( hot_client_ )        hot_client_->analyze(); 
  if( dead_client_ )       dead_client_->analyze(); 
  if( tp_client_ )         tp_client_->analyze(); 
  if( ct_client_ )         ct_client_->analyze(); 

  errorSummary();

  return;
}

//--------------------------------------------------------
void HcalMonitorClient::createTests(void){
  
  if( debug_ ) cout << "HcalMonitorClient: creating all tests" << endl;

  if( dataformat_client_ ) dataformat_client_->createTests(); 
  if( digi_client_ )       digi_client_->createTests(); 
  if( rechit_client_ )     rechit_client_->createTests(); 
  if( pedestal_client_ )   pedestal_client_->createTests(); 
  if( led_client_ )        led_client_->createTests(); 
  if( hot_client_ )        hot_client_->createTests(); 
  if( dead_client_ )       dead_client_->createTests(); 
  if( tp_client_ )         tp_client_->createTests(); 
  if( ct_client_ )         ct_client_->createTests(); 

  return;
}

//--------------------------------------------------------
void HcalMonitorClient::report(bool doUpdate) {
  
  if( debug_ ) 
    cout << "HcalMonitorClient: creating report, ievt = " << ievt_ << endl;
  
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
  if( dead_client_ ) dead_client_->report();
  if( tp_client_ ) tp_client_->report();
  if( ct_client_ ) ct_client_->report();

  errorSummary();

  //create html output if specified...
  if( baseHtmlDir_.size() != 0 && ievt_>0) htmlOutput();

  return;
}

void HcalMonitorClient::errorSummary(){
  
  ///Collect test summary information
  int nTests=0;
  map<string, vector<QReport*> > errE, errW, errO;
  if( hot_client_ )        hot_client_->getTestResults(nTests,errE,errW,errO);
  if( dead_client_ )       dead_client_->getTestResults(nTests,errE,errW,errO);
  if( led_client_ )        led_client_->getTestResults(nTests,errE,errW,errO);
  if( tp_client_ )         tp_client_->getTestResults(nTests,errE,errW,errO);
  if( pedestal_client_ )   pedestal_client_->getTestResults(nTests,errE,errW,errO);
  if( digi_client_ )       digi_client_->getTestResults(nTests,errE,errW,errO);
  if( rechit_client_ )     rechit_client_->getTestResults(nTests,errE,errW,errO);
  if( dataformat_client_ ) dataformat_client_->getTestResults(nTests,errE,errW,errO);
  if( ct_client_ ) ct_client_->getTestResults(nTests,errE,errW,errO);
  //For now, report the fraction of good tests....
  float errorSummary = 1.0;
  if(nTests>0) errorSummary = 1.0 - (float(errE.size())+float(errW.size()))/float(nTests);
  
  cout << "Hcal DQM Error Summary ("<< errorSummary <<"): "<< nTests << " tests, "<<errE.size() << " errors, " <<errW.size() << " warnings, "<< errO.size() << " others" << endl;
  
  char meTitle[256];
  sprintf(meTitle,"%sEventInfo/errorSummary",rootFolder_.c_str() );
  MonitorElement* me = dbe_->get(meTitle);
  if(me) me->Fill(errorSummary);
  
  return;
}


void HcalMonitorClient::htmlOutput(void){

  cout << "Preparing HcalMonitorClient html output ..." << endl;

  char tmp[10];
  if(irun_!=-1) sprintf(tmp, "DQM_Hcal_R%09d", irun_);
  else sprintf(tmp, "DQM_Hcal_R%09d", 0);
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
  htmlFile << "<h2>Run Number:&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "<span style=\"color: rgb(0, 0, 153);\">" << irun_ <<"</span></h2> " << endl;
  htmlFile << "<h2>Events processed:&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "<span style=\"color: rgb(0, 0, 153);\">" << ievt_ <<"</span></h2> " << endl;
  htmlFile << "<hr>" << endl;
  htmlFile << "<ul>" << endl;

  string htmlName;
  if( dataformat_client_ ) {
    htmlName = "HcalDataFormatClient.html";
    dataformat_client_->htmlOutput(irun_, htmlDir, htmlName);
    htmlFile << "<table border=0 WIDTH=\"50%\"><tr>" << endl;
    htmlFile << "<td WIDTH=\"35%\"><a href=\"" << htmlName << "\">Data Format Monitor</a></td>" << endl;
    if(dataformat_client_->hasErrors()) htmlFile << "<td bgcolor=red align=center>This monitor task has errors.</td>" << endl;
    else if(dataformat_client_->hasWarnings()) htmlFile << "<td bgcolor=yellow align=center>This monitor task has warnings.</td>" << endl;
    else if(dataformat_client_->hasOther()) htmlFile << "<td bgcolor=aqua align=center>This monitor task has messages.</td>" << endl;
    else htmlFile << "<td bgcolor=lime align=center>This monitor task has no problems</td>" << endl;
    htmlFile << "</tr></table>" << endl;
  }
  if( digi_client_ ) {
    htmlName = "HcalDigiClient.html";
    digi_client_->htmlOutput(irun_, htmlDir, htmlName);
    htmlFile << "<table border=0 WIDTH=\"50%\"><tr>" << endl;
    htmlFile << "<td WIDTH=\"35%\"><a href=\"" << htmlName << "\">Digi Monitor</a></td>" << endl;
    if(digi_client_->hasErrors()) htmlFile << "<td bgcolor=red align=center>This monitor task has errors.</td>" << endl;
    else if(digi_client_->hasWarnings()) htmlFile << "<td bgcolor=yellow align=center>This monitor task has warnings.</td>" << endl;
    else if(digi_client_->hasOther()) htmlFile << "<td bgcolor=aqua align=center>This monitor task has messages.</td>" << endl;
    else htmlFile << "<td bgcolor=lime align=center>This monitor task has no problems</td>" << endl;
    htmlFile << "</tr></table>" << endl;
  }
  if( tp_client_ ) {
    htmlName = "HcalTrigPrimClient.html";
    tp_client_->htmlOutput(irun_, htmlDir, htmlName);
    htmlFile << "<table border=0 WIDTH=\"50%\"><tr>" << endl;
    htmlFile << "<td WIDTH=\"35%\"><a href=\"" << htmlName << "\">TrigPrim Monitor</a></td>" << endl;
    if(tp_client_->hasErrors()) htmlFile << "<td bgcolor=red align=center>This monitor task has errors.</td>" << endl;
    else if(tp_client_->hasWarnings()) htmlFile << "<td bgcolor=yellow align=center>This monitor task has warnings.</td>" << endl;
    else if(tp_client_->hasOther()) htmlFile << "<td bgcolor=aqua align=center>This monitor task has messages.</td>" << endl;
    else htmlFile << "<td bgcolor=lime align=center>This monitor task has no problems</td>" << endl;
    htmlFile << "</tr></table>" << endl;
  }
  if( rechit_client_ ) {
    htmlName = "HcalRecHitClient.html";
    rechit_client_->htmlOutput(irun_, htmlDir, htmlName);
    htmlFile << "<table border=0 WIDTH=\"50%\"><tr>" << endl;
    htmlFile << "<td WIDTH=\"35%\"><a href=\"" << htmlName << "\">RecHit Monitor</a></td>" << endl;
    if(rechit_client_->hasErrors()) htmlFile << "<td bgcolor=red align=center>This monitor task has errors.</td>" << endl;
    else if(rechit_client_->hasWarnings()) htmlFile << "<td bgcolor=yellow align=center>This monitor task has warnings.</td>" << endl;
    else if(rechit_client_->hasOther()) htmlFile << "<td bgcolor=aqua align=center>This monitor task has messages.</td>" << endl;
    else htmlFile << "<td bgcolor=lime align=center>This monitor task has no problems</td>" << endl;
    htmlFile << "</tr></table>" << endl;
  }
  if( ct_client_ ) {
    htmlName = "HcalCaloTowerClient.html";
    ct_client_->htmlOutput(irun_, htmlDir, htmlName);
    htmlFile << "<table border=0 WIDTH=\"50%\"><tr>" << endl;
    htmlFile << "<td WIDTH=\"35%\"><a href=\"" << htmlName << "\">CaloTower Monitor</a></td>" << endl;
    if(ct_client_->hasErrors()) htmlFile << "<td bgcolor=red align=center>This monitor task has errors.</td>" << endl;
    else if(ct_client_->hasWarnings()) htmlFile << "<td bgcolor=yellow align=center>This monitor task has warnings.</td>" << endl;
    else if(ct_client_->hasOther()) htmlFile << "<td bgcolor=aqua align=center>This monitor task has messages.</td>" << endl;
    else htmlFile << "<td bgcolor=lime align=center>This monitor task has no problems</td>" << endl;
    htmlFile << "</tr></table>" << endl;
  }
  if( hot_client_ ) {
    htmlName = "HcalHotCellClient.html";
    hot_client_->htmlOutput(irun_, htmlDir, htmlName);
    htmlFile << "<table border=0 WIDTH=\"50%\"><tr>" << endl;
    htmlFile << "<td WIDTH=\"35%\"><a href=\"" << htmlName << "\">Hot Cell Monitor</a></td>" << endl;
    if(hot_client_->hasErrors()) htmlFile << "<td bgcolor=red align=center>This monitor task has errors.</td>" << endl;
    else if(hot_client_->hasWarnings()) htmlFile << "<td bgcolor=yellow align=center>This monitor task has warnings.</td>" << endl;
    else if(hot_client_->hasOther()) htmlFile << "<td bgcolor=aqua align=center>This monitor task has messages.</td>" << endl;
    else htmlFile << "<td bgcolor=lime align=center>This monitor task has no problems</td>" << endl;
    htmlFile << "</tr></table>" << endl;
  }
  if( dead_client_) {
    htmlName = "HcalDeadCellClient.html";
    dead_client_->htmlOutput(irun_, htmlDir, htmlName);
    htmlFile << "<table border=0 WIDTH=\"50%\"><tr>" << endl;
    htmlFile << "<td WIDTH=\"35%\"><a href=\"" << htmlName << "\">Dead Cell Monitor</a></td>" << endl;
    if(dead_client_->hasErrors()) htmlFile << "<td bgcolor=red align=center>This monitor task has errors.</td>" << endl;
    else if(dead_client_->hasWarnings()) htmlFile << "<td bgcolor=yellow align=center>This monitor task has warnings.</td>" << endl;
    else if(dead_client_->hasOther()) htmlFile << "<td bgcolor=aqua align=center>This monitor task has messages.</td>" << endl;
    else htmlFile << "<td bgcolor=lime align=center>This monitor task has no problems</td>" << endl;
    htmlFile << "</tr></table>" << endl;
  }
  if( pedestal_client_) {
    htmlName = "HcalPedestalClient.html";
    pedestal_client_->htmlOutput(irun_, htmlDir, htmlName);
    htmlFile << "<table border=0 WIDTH=\"50%\"><tr>" << endl;
    htmlFile << "<td WIDTH=\"35%\"><a href=\"" << htmlName << "\">Pedestal Monitor</a></td>" << endl;
    
    if(pedestal_client_->hasErrors()) htmlFile << "<td bgcolor=red align=center>This monitor task has errors.</td>" << endl;
    else if(pedestal_client_->hasWarnings()) htmlFile << "<td bgcolor=yellow align=center>This monitor task has warnings.</td>" << endl;
    else if(pedestal_client_->hasOther()) htmlFile << "<td bgcolor=aqua align=center>This monitor task has messages.</td>" << endl;
    else htmlFile << "<td bgcolor=lime align=center>This monitor task has no problems</td>" << endl;
    
    htmlFile << "</tr></table>" << endl;
  }

  if( led_client_) {
    htmlName = "HcalLEDClient.html";
    led_client_->htmlOutput(irun_, htmlDir, htmlName);
    htmlFile << "<table border=0 WIDTH=\"50%\"><tr>" << endl;
    htmlFile << "<td WIDTH=\"35%\"><a href=\"" << htmlName << "\">LED Monitor</a></td>" << endl;
    
    if(led_client_->hasErrors()) htmlFile << "<td bgcolor=red align=center>This monitor task has errors.</td>" << endl;
    else if(led_client_->hasWarnings()) htmlFile << "<td bgcolor=yellow align=center>This monitor task has warnings.</td>" << endl;
    else if(led_client_->hasOther()) htmlFile << "<td bgcolor=aqua align=center>This monitor task has messages.</td>" << endl;
    else htmlFile << "<td bgcolor=lime align=center>This monitor task has no problems</td>" << endl;
    
    htmlFile << "</tr></table>" << endl;
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
  /*
  dataformat_client_ = 0; digi_client_ = 0;
  rechit_client_ = 0; pedestal_client_ = 0;
  led_client_ = 0;  hot_client_ = 0;
  dead_client_=0;

  // base Html output directory
  baseHtmlDir_ = ".";
  
  // clients' constructors
  hot_client_          = new HcalHotCellClient();
  dead_client_         = new HcalDeadCellClient();
  dataformat_client_   = new HcalDataFormatClient();
  rechit_client_       = new HcalRecHitClient();
  digi_client_         = new HcalDigiClient();
  pedestal_client_     = new HcalPedestalClient();
  led_client_          = new HcalLEDClient();
  */
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
  if(dead_client_) dead_client_->loadHistograms(infile);
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
  if(dead_client_) dead_client_->dumpHistograms(names,meanX,meanY,rmsX,rmsY);
  if(dataformat_client) dataformat_client_->dumpHistograms(names,meanX,meanY,rmsX,rmsY);
  if(rechit_client_) rechit_client_->dumpHistograms(names,meanX,meanY,rmsX,rmsY);
  if(digi_client_) digi_client_->dumpHistograms(names,meanX,meanY,rmsX,rmsY);
  if(pedestal_client_) pedestal_client_->dumpHistograms(names,meanX,meanY,rmsX,rmsY);
  if(led_client_) led_client_->dumpHistograms(names,meanX,meanY,rmsX,rmsY);
  */
 return;
}

//--------------------------------------------------------
bool HcalMonitorClient::prescale(){
  ///Return true if this event should be skipped according to the prescale condition...
  ///    Accommodate a logical "OR" of the possible tests
  if (debug_) cout <<"HcalMonitorClient::prescale"<<endl;
  
  //First determine if we care...
  bool evtPS =    prescaleEvt_>0;
  bool lsPS =     prescaleLS_>0;
  bool timePS =   prescaleTime_>0;
  bool updatePS = prescaleUpdate_>0;

  // If no prescales are set, keep the event
  if(!evtPS && !lsPS && !timePS && !updatePS) return false;

  //check each instance
  if(lsPS && (ilumisec_%prescaleLS_)!=0) lsPS = false; //LS veto
  if(evtPS && (ievent_%prescaleEvt_)!=0) evtPS = false; //evt # veto
  if(timePS){
    float time = psTime_.elapsedTime - psTime_.updateTime;
    if(time<prescaleTime_){
      timePS = false;  //timestamp veto
      psTime_.updateTime = psTime_.elapsedTime;
    }
  }
  //  if(prescaleUpdate_>0 && (nupdates_%prescaleUpdate_)==0) updatePS=false; ///need to define what "updates" means
  
  if (debug_) printf("HcalMonitorClient::prescale  evt: %d/%d, ls: %d/%d, time: %f/%d\n",
		     ievent_,evtPS,
		     ilumisec_,lsPS,
		     psTime_.elapsedTime - psTime_.updateTime,timePS);

  // if any criteria wants to keep the event, do so
  if(evtPS || lsPS || timePS) return false; //FIXME updatePS left out for now
  return true;
}

#include "FWCore/Framework/interface/MakerMacros.h"
#include <DQM/HcalMonitorClient/interface/HcalMonitorClient.h>
#include "DQMServices/Core/interface/MonitorElement.h"

DEFINE_FWK_MODULE(HcalMonitorClient);
