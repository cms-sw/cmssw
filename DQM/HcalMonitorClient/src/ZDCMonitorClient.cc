#include <DQM/HcalMonitorClient/interface/ZDCMonitorClient.h>
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "CondFormats/HcalObjects/interface/HcalChannelStatus.h"
#include "DataFormats/DetId/interface/DetId.h"

//--------------------------------------------------------
ZDCMonitorClient::ZDCMonitorClient(const ParameterSet& ps){
  initialize(ps);
}

ZDCMonitorClient::ZDCMonitorClient(){}

//--------------------------------------------------------
ZDCMonitorClient::~ZDCMonitorClient(){

  if (debug_>0) std::cout << "ZDCMonitorClient: Exit ..." << endl;
}

//--------------------------------------------------------
void ZDCMonitorClient::initialize(const ParameterSet& ps){

  irun_=0; ilumisec_=0; ievent_=0; itime_=0;

  maxlumisec_=0; minlumisec_=0;


  debug_ = ps.getUntrackedParameter<int>("debug", 0);
  if (debug_>0)
    std::cout << endl<<" *** ZDC Monitor Client ***" << endl<<endl;

  if(debug_>1) std::cout << "ZDCMonitorClient: constructor...." << endl;

  Online_ = ps.getUntrackedParameter<bool>("Online",false);
  // timing switch 
  showTiming_ = ps.getUntrackedParameter<bool>("showTiming",false);  

  // MonitorDaemon switch
  enableMonitorDaemon_ = ps.getUntrackedParameter<bool>("enableMonitorDaemon", true);
  if (debug_>0)
    {
      if ( enableMonitorDaemon_ ) std::cout << "-->enableMonitorDaemon switch is ON" << endl;
      else std::cout << "-->enableMonitorDaemon switch is OFF" << endl;
    }

  // get hold of back-end interface
  dbe_ = Service<DQMStore>().operator->();
  if (debug_>1) dbe_->showDirStructure();   

  // DQM ROOT input
  inputFile_ = ps.getUntrackedParameter<string>("inputFile", "");
  if(inputFile_.size()!=0 && debug_>0) std::cout << "-->reading DQM input from " << inputFile_ << endl;
  
  if( ! enableMonitorDaemon_ ) {  
    if( inputFile_.size() != 0 && dbe_!=NULL){
      dbe_->open(inputFile_);
      dbe_->showDirStructure();     
    }
  }

  //histogram reset freqency, update frequency, timeout
  resetEvents_ = ps.getUntrackedParameter<int>("resetFreqEvents",-1);   //number of real events
  if(resetEvents_!=-1 && debug_>0) std::cout << "-->Will reset histograms every " << resetEvents_ <<" events." << endl;
  resetLS_ = ps.getUntrackedParameter<int>("resetFreqLS",-1);       //number of lumisections
  if(resetLS_!=-1 && debug_>0) std::cout << "-->Will reset histograms every " << resetLS_ <<" lumi sections." << endl;

  // base Html output directory
  baseHtmlDir_ = ps.getUntrackedParameter<string>("baseHtmlDir", "");
  if (debug_>0)
    {
      if( baseHtmlDir_.size() != 0) 
	std::cout << "-->HTML output will go to baseHtmlDir = '" << baseHtmlDir_ << "'" << endl;
      else std::cout << "-->HTML output is disabled" << endl;
    }
  
  runningStandalone_ = ps.getUntrackedParameter<bool>("runningStandalone", false); // unnecessary? Or use for offline client processing?
  if (debug_>1)
    {
      if( runningStandalone_ ) std::cout << "-->standAlone switch is ON" << endl;
      else std::cout << "-->standAlone switch is OFF" << endl;
    }

  // set parameters   
  prescaleEvt_ = ps.getUntrackedParameter<int>("diagnosticPrescaleEvt", -1);
  if (debug_>0) 
    std::cout << "===>DQM event prescale = " << prescaleEvt_ << " event(s)"<< endl;

  prescaleLS_ = ps.getUntrackedParameter<int>("diagnosticPrescaleLS", -1);
  if (debug_>0) std::cout << "===>DQM lumi section prescale = " << prescaleLS_ << " lumi section(s)"<< endl;

  // Base folder for the contents of this job
  string subsystemname = ps.getUntrackedParameter<string>("subSystemFolder", "ZDC") ;
  if (debug_>0) std::cout << "===>ZDCMonitor name = " << subsystemname << endl;
  rootFolder_ = subsystemname + "/";

  return;
}

//--------------------------------------------------------
// remove all MonitorElements and directories
void ZDCMonitorClient::removeAllME(){
  if (debug_>0) std::cout <<"<ZDCMonitorClient>removeAllME()"<<endl;
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
  return;
}

//--------------------------------------------------------
///do a reset of all monitor elements...
void ZDCMonitorClient::resetAllME() {
  if (debug_>0) std::cout <<"<ZDCMonitorClient> resetAllME()"<<endl;
   return;
}

//--------------------------------------------------------
void ZDCMonitorClient::beginJob(){

  if( debug_>0 ) std::cout << "ZDCMonitorClient: beginJob" << endl;
  
  ievt_ = 0;
 
  return;
}

//--------------------------------------------------------
void ZDCMonitorClient::beginRun(const Run& r, const EventSetup& c) {

  if (debug_>0)
    std::cout << endl<<"ZDCMonitorClient: Standard beginRun() for run " << r.id().run() << endl<<endl;
 
  // Get current channel quality 
  /*
 edm::ESHandle<HcalChannelQuality> p;
  c.get<HcalChannelQualityRcd>().get(p);
  chanquality_= new HcalChannelQuality(*p.product());
  */

  string eventinfo="/EventInfo";
  if (rootFolder_!="ZDC")
    eventinfo+="DUMMY";

  // Setup histograms -- this is all we will do for ZDC Monitor at the moment!
  MonitorElement* me; //JEFF
  dbe_->setCurrentFolder(rootFolder_+eventinfo.c_str()+"/");
  me=dbe_->get(rootFolder_+eventinfo.c_str()+"/reportSummary");
  if (me)
     dbe_->removeElement(me->getName());
  me = dbe_->bookFloat("reportSummary");
  me->Fill(-1); // set status to unknown at startup
  me=dbe_->get(rootFolder_+eventinfo.c_str()+"/reportSummaryMap");
  if (me)
    dbe_->removeElement(me->getName());
  me = dbe_->book2D("reportSummaryMap","ZDC Report Summary Map",4,0,4,1,0,1);
  TH2F* myhist=me->getTH2F();
  myhist->GetXaxis()->SetBinLabel(1,"HAD-");
  myhist->GetXaxis()->SetBinLabel(2,"EM-");
  myhist->GetXaxis()->SetBinLabel(3,"EM+");
  myhist->GetXaxis()->SetBinLabel(4,"HAD+");
  // Set all values to -1
  myhist->SetBinContent(1,1,-1);
  myhist->SetBinContent(2,1,-1);
  myhist->SetBinContent(3,1,-1);
  myhist->SetBinContent(4,1,-1);
  

  dbe_->setCurrentFolder(rootFolder_+eventinfo.c_str()+"/reportSummaryContents/");
  me=dbe_->get(rootFolder_+eventinfo.c_str()+"/reportSummary/reportSummaryContents/ZDC_HADMinus");
  if (me)
     dbe_->removeElement(me->getName());
  me = dbe_->bookFloat("ZDC_HADMinus");
  me->Fill(-1); // set status to unknown at startup
  me=dbe_->get(rootFolder_+eventinfo.c_str()+"/reportSummary/reportSummaryContents/ZDC_EMMinus");
  if (me)
     dbe_->removeElement(me->getName());
  me = dbe_->bookFloat("ZDC_EMMinus");
  me->Fill(-1); // set status to unknown at startup
  me=dbe_->get(rootFolder_+eventinfo.c_str()+"/reportSummary/reportSummaryContents/ZDC_EMPlus");
  if (me)
     dbe_->removeElement(me->getName());
  me = dbe_->bookFloat("ZDC_EMPlus");
  me->Fill(-1); // set status to unknown at startup
  me=dbe_->get(rootFolder_+eventinfo.c_str()+"/reportSummary/reportSummaryContents/ZDC_HADPlus");
  if (me)
     dbe_->removeElement(me->getName());
  me = dbe_->bookFloat("ZDC_HADPlus");
  me->Fill(-1); // set status to unknown at startup

  // Add dummy DAQ Summary, DCS Summary
  dbe_->setCurrentFolder(rootFolder_+eventinfo.c_str());
  me=dbe_->get(rootFolder_+eventinfo.c_str()+"/DAQSummary");
  if (me)
    dbe_->removeElement(me->getName());
  me = dbe_->bookFloat("DAQSummary");
  me->Fill(-1); // set status to unknown at startup

  dbe_->setCurrentFolder(rootFolder_+eventinfo.c_str()+"/DAQSummaryContents");
  me=dbe_->get(rootFolder_+eventinfo.c_str()+"/DAQSummary/DAQSummaryContents/ZDC_HADPlus");
  if (me)
    dbe_->removeElement(me->getName());
  me = dbe_->bookFloat("ZDC_HADPlus");
  me->Fill(-1); // set status to unknown at startup

  me=dbe_->get(rootFolder_+eventinfo.c_str()+"/DAQSummary/DAQSummaryContents/ZDC_HADMinus");
  if (me)
    dbe_->removeElement(me->getName());
  me = dbe_->bookFloat("ZDC_HADMinus");
  me->Fill(-1); // set status to unknown at startup

  me=dbe_->get(rootFolder_+eventinfo.c_str()+"/DAQSummary/DAQSummaryContents/ZDC_EMPlus");
  if (me)
    dbe_->removeElement(me->getName());
  me = dbe_->bookFloat("ZDC_EMPlus");
  me->Fill(-1); // set status to unknown at startup

  me=dbe_->get(rootFolder_+eventinfo.c_str()+"/DAQSummary/DAQSummaryContents/ZDC_EMMinus");
  if (me)
    dbe_->removeElement(me->getName());
  me = dbe_->bookFloat("ZDC_EMMinus");
  me->Fill(-1); // set status to unknown at startup

  // DCS Summary 
  dbe_->setCurrentFolder(rootFolder_+eventinfo.c_str());
  me=dbe_->get(rootFolder_+eventinfo.c_str()+"/DCSSummary");
  if (me)
    dbe_->removeElement(me->getName());
  me = dbe_->bookFloat("DCSSummary");
  me->Fill(-1); // set status to unknown at startup

  dbe_->setCurrentFolder(rootFolder_+eventinfo.c_str()+"/DCSSummaryContents");
  me=dbe_->get(rootFolder_+eventinfo.c_str()+"/DCSSummary/DCSSummaryContents/ZDC_HADPlus");
  if (me)
    dbe_->removeElement(me->getName());
  me = dbe_->bookFloat("ZDC_HADPlus");
  me->Fill(-1); // set status to unknown at startup

  me=dbe_->get(rootFolder_+eventinfo.c_str()+"/DCSSummary/DCSSummaryContents/ZDC_HADMinus");
  if (me)
    dbe_->removeElement(me->getName());
  me = dbe_->bookFloat("ZDC_HADMinus");
  me->Fill(-1); // set status to unknown at startup

  me=dbe_->get(rootFolder_+eventinfo.c_str()+"/DCSSummary/DCSSummaryContents/ZDC_EMPlus");
  if (me)
    dbe_->removeElement(me->getName());
  me = dbe_->bookFloat("ZDC_EMPlus");
  me->Fill(-1); // set status to unknown at startup

  me=dbe_->get(rootFolder_+eventinfo.c_str()+"/DCSSummary/DCSSummaryContents/ZDC_EMMinus");
  if (me)
    dbe_->removeElement(me->getName());
  me = dbe_->bookFloat("ZDC_EMMinus");
  me->Fill(-1); // set status to unknown at startup

}


//--------------------------------------------------------
void ZDCMonitorClient::endJob(void) {

  if( debug_>0 ) 
    std::cout << "ZDCMonitorClient: endJob, ievt = " << ievt_ << endl;

  return;
}

//--------------------------------------------------------
void ZDCMonitorClient::endRun(const Run& r, const EventSetup& c) {

  if (debug_>0)
    std::cout << endl<<"<ZDCMonitorClient> Standard endRun() for run " << r.id().run() << endl<<endl;

  if (!Online_)
    analyze();

  if( debug_ >0) std::cout <<"ZDCMonitorClient: processed events: "<<ievt_<<endl;

  if (debug_>0) std::cout <<"==>Creating report after run end condition"<<endl;
  if(irun_>1){
    if(inputFile_.size()!=0) report(true);
    else report(false);
  }

  // dumping to database

  // need to add separate function to do this!!!

  return;
}

void ZDCMonitorClient::writeDBfile()

{
  return; // not used for ZDC

} // ZDCMonitorClient::writeDBfile()

//--------------------------------------------------------
void ZDCMonitorClient::beginLuminosityBlock(const LuminosityBlock &l, const EventSetup &c) 
{
  // don't allow 'backsliding' across lumi blocks in online running
  // This still won't prevent some lumi blocks from being evaluated multiple times.  Need to think about this.
  //if (Online_ && (int)l.luminosityBlock()<ilumisec_) return;
  if (debug_>0) std::cout <<"Entered Monitor Client beginLuminosityBlock for LS = "<<l.luminosityBlock()<<endl;
  ilumisec_ = l.luminosityBlock();
  if( debug_>0 ) std::cout << "ZDCMonitorClient: beginLuminosityBlock" << endl;
}

//--------------------------------------------------------
void ZDCMonitorClient::endLuminosityBlock(const LuminosityBlock &l, const EventSetup &c) {

  // don't allow backsliding in online running
  //if (Online_ && (int)l.luminosityBlock()<ilumisec_) return;
  if( debug_>0 ) std::cout << "ZDCMonitorClient: endLuminosityBlock" << endl;
  if(prescaleLS_>0 && !prescale()){
    // do scheduled tasks...
    if (Online_)
      analyze();
  }

  return;
}

//--------------------------------------------------------
void ZDCMonitorClient::analyze(const Event& e, const edm::EventSetup& eventSetup){

  if (debug_>1)
    std::cout <<"Entered ZDCMonitorClient::analyze(const Evt...)"<<endl;
  
  if(resetEvents_>0 && (ievt_%resetEvents_)==0) resetAllME(); // use ievt_ here, not ievent_, since ievent is the event #, not the # of events processed
  if(resetLS_>0 && (ilumisec_%resetLS_)==0) resetAllME();

  // environment datamembers

  // Don't process out-of-order lumi block information in online running
  //if (Online_ && (int)e.luminosityBlock()<ilumisec_) return;
  irun_     = e.id().run();
  ilumisec_ = e.luminosityBlock();
  ievent_   = e.id().event();
  itime_    = e.time().value();
  mytime_   = (e.time().value())>>32;

  if (minlumisec_==0)
    minlumisec_=ilumisec_;
  minlumisec_=min(minlumisec_,ilumisec_);
  maxlumisec_=max(maxlumisec_,ilumisec_);

  if (debug_>1) 
    std::cout << "ZDCMonitorClient: evts: "<< ievt_ << ", run: " << irun_ << ", LS: " << ilumisec_ << ", evt: " << ievent_ << ", time: " << itime_ << endl; 
  
  ievt_++; 

  if ( runningStandalone_) return;

  // run if we want to check individual events, and if this event isn't prescaled
  if (prescaleEvt_>0 && !prescale()) 
    analyze();
}


//--------------------------------------------------------
void ZDCMonitorClient::analyze(){
  if (debug_>0) 
    std::cout <<"<ZDCMonitorClient> Entered ZDCMonitorClient::analyze()"<<endl;
  if(debug_>1) std::cout<<"\nZDC Monitor Client heartbeat...."<<endl;
  
  createTests();  
  //mui_->doMonitoring();
  dbe_->runQTests();
  errorSummary();


  return;
}

//--------------------------------------------------------
void ZDCMonitorClient::createTests(void){
  
  if( debug_>0 ) std::cout << "ZDCMonitorClient: creating all tests" << endl;
  return;
}

//--------------------------------------------------------
void ZDCMonitorClient::report(bool doUpdate) {
  
  if( debug_>0 ) 
    std::cout << "ZDCMonitorClient: creating report, ievt = " << ievt_ << endl;
  
  if(doUpdate){
    createTests();  
    dbe_->runQTests();
  }
  errorSummary();

  //create html output if specified...
  if( baseHtmlDir_.size() != 0 && ievt_>0) 
    htmlOutput();
  return;
}

void ZDCMonitorClient::errorSummary(){
  

  float errorSummary = 1.0;
  
  char meTitle[256];
  sprintf(meTitle,"%sEventInfo/errorSummary",rootFolder_.c_str() );
  MonitorElement* me = dbe_->get(meTitle);
  if(me) me->Fill(errorSummary);
  
  return;
}


void ZDCMonitorClient::htmlOutput(void){
  return;
}

void ZDCMonitorClient::offlineSetup(){
  //  std::cout << endl;
  //  std::cout << " *** Hcal Generic Monitor Client, for offline operation***" << endl;
  //  std::cout << endl;
  return;
}

void ZDCMonitorClient::loadHistograms(TFile* infile, const char* fname)
{
  if(!infile){
    throw cms::Exception("Incomplete configuration") << 
      "ZDCMonitorClient: this histogram file is bad! " <<endl;
    return;
  }
  return;
}


void ZDCMonitorClient::dumpHistograms(int& runNum, vector<TH1F*> &hist1d,vector<TH2F*> &hist2d)
{
  hist1d.clear(); 
  hist2d.clear(); 
  return;
}

//--------------------------------------------------------
bool ZDCMonitorClient::prescale(){
  ///Return true if this event should be skipped according to the prescale condition...

  ///    Accommodate a logical "OR" of the possible tests
  if (debug_>1) std::cout <<"ZDCMonitorClient::prescale"<<endl;
  
  // If no prescales are set, return 'false'.  (This means that we should process the event.)
  if(prescaleEvt_<=0 && prescaleLS_<=0) return false;

  // Now check whether event should be kept.  Assume that it should not by default

  bool keepEvent=false;
  
  // Keep event if prescaleLS test is met or if prescaleEvt test is met
  if(prescaleLS_>0 && (ilumisec_%prescaleLS_)==0) keepEvent = true; // check on ls prescale; 
  if (prescaleEvt_>0 && (ievt_%prescaleEvt_)==0) keepEvent = true; // 
  
  // if any criteria wants to keep the event, do so
  if (keepEvent) return false;  // event should be kept; don't apply prescale
  return true; // apply prescale by default
}


DEFINE_FWK_MODULE(ZDCMonitorClient);
