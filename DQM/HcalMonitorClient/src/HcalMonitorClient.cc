#include <DQM/HcalMonitorClient/interface/HcalMonitorClient.h>
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include <DQM/HcalMonitorClient/interface/HcalMonitorClient.h>
#include "DQMServices/Core/interface/MonitorElement.h"

//--------------------------------------------------------
HcalMonitorClient::HcalMonitorClient(const ParameterSet& ps){
  initialize(ps);
}

HcalMonitorClient::HcalMonitorClient(){}

//--------------------------------------------------------
HcalMonitorClient::~HcalMonitorClient(){

  if (debug_>0) cout << "HcalMonitorClient: Exit ..." << endl;
  /*
    // leave deletions to code framework?
  if( summary_client_ )    delete summary_client_;
  if( dataformat_client_ ) delete dataformat_client_;
  if( digi_client_ )       delete digi_client_;
  if( rechit_client_ )     delete rechit_client_;
  if( pedestal_client_ )   delete pedestal_client_;
  if( led_client_ )        delete led_client_;
  if( laser_client_ )      delete laser_client_;
  if( hot_client_ )        delete hot_client_;
  if( dead_client_ )       delete dead_client_;
  if( tp_client_ )         delete tp_client_;
  if( ct_client_ )         delete ct_client_;
  if( beam_client_)        delete beam_client_;
  if (dqm_db_)             delete dqm_db_;
  //if( dbe_ )               delete dbe_;
  if( mui_ )               delete mui_;
  */
  if (debug_>1) std::cout <<"HcalMonitorClient: Finished destructor..."<<endl;
}

//--------------------------------------------------------
void HcalMonitorClient::initialize(const ParameterSet& ps){

  irun_=0; ilumisec_=0; ievent_=0; itime_=0;

  maxlumisec_=0; minlumisec_=0;

  actonLS_=false;

  summary_client_ = 0;
  dataformat_client_ = 0; digi_client_ = 0;
  rechit_client_ = 0; pedestal_client_ = 0;
  led_client_ = 0; laser_client_ = 0; hot_client_ = 0; dead_client_=0;
  tp_client_=0;
  ct_client_=0;
  beam_client_=0;
  lastResetTime_=0;
  //////////////////////////////////////////////////////////////////
  detdiagped_client_=0; 
  detdiagled_client_=0;
  detdiaglas_client_=0; 
  //////////////////////////////////////////////////////////////////
  debug_ = ps.getUntrackedParameter<int>("debug", 0);
  if (debug_>0)
    std::cout << endl<<" *** Hcal Monitor Client ***" << endl<<endl;

  if(debug_>1) std::cout << "HcalMonitorClient: constructor...." << endl;

  // timing switch 
  showTiming_ = ps.getUntrackedParameter<bool>("showTiming",false);  

  // MonitorDaemon switch
  enableMonitorDaemon_ = ps.getUntrackedParameter<bool>("enableMonitorDaemon", true);
  if (debug_>0)
    {
      if ( enableMonitorDaemon_ ) std::cout << "-->enableMonitorDaemon switch is ON" << endl;
      else std::cout << "-->enableMonitorDaemon switch is OFF" << endl;
    }

  mui_ = new DQMOldReceiver();
  dbe_ = mui_->getBEInterface();

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
  resetUpdate_ = ps.getUntrackedParameter<int>("resetFreqUpdates",-1);  //number of collector updates
  if(resetUpdate_!=-1 && debug_>0) std::cout << "-->Will reset histograms every " << resetUpdate_ <<" collector updates." << endl;
  resetEvents_ = ps.getUntrackedParameter<int>("resetFreqEvents",-1);   //number of real events
  if(resetEvents_!=-1 && debug_>0) std::cout << "-->Will reset histograms every " << resetEvents_ <<" events." << endl;
  resetTime_ = ps.getUntrackedParameter<int>("resetFreqTime",-1);       //number of minutes
  if(resetTime_!=-1 && debug_>0) std::cout << "-->Will reset histograms every " << resetTime_ <<" minutes." << endl;
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

  // exit on end job switch
  enableExit_ = ps.getUntrackedParameter<bool>("enableExit", true);
  if (debug_>1)
    {
      if( enableExit_ ) std::cout << "-->enableExit switch is ON" << endl;
      else std::cout << "-->enableExit switch is OFF" << endl;
    }
  
  runningStandalone_ = ps.getUntrackedParameter<bool>("runningStandalone", false);
  dump2database_ = false; // controls whether we write bad cells to database

  if (debug_>1)
    {
      if( runningStandalone_ ) std::cout << "-->standAlone switch is ON" << endl;
      else std::cout << "-->standAlone switch is OFF" << endl;
    }
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
  if( ps.getUntrackedParameter<bool>("SummaryClient", true) )
    {
      if(debug_>0) 
	std::cout << "===>DQM Summary Client is ON" << endl;
      summary_client_   = new HcalSummaryClient();
      summary_client_->init(ps, dbe_,"SummaryClient");
    }
  if( ps.getUntrackedParameter<bool>("DataFormatClient", false) ){
    if(debug_>0)   std::cout << "===>DQM DataFormat Client is ON" << endl;
    dataformat_client_   = new HcalDataFormatClient();
    dataformat_client_->init(ps, dbe_,"DataFormatClient");
  }
  if( ps.getUntrackedParameter<bool>("DigiClient", false) ){
    if(debug_>0)  
      std::cout << "===>DQM Digi Client is ON" << endl;
    digi_client_         = new HcalDigiClient();
    digi_client_->init(ps, dbe_,"DigiClient");
  }
  if( ps.getUntrackedParameter<bool>("RecHitClient", false) ){
    if(debug_>0)   std::cout << "===>DQM RecHit Client is ON" << endl;
    rechit_client_       = new HcalRecHitClient();
    rechit_client_->init(ps, dbe_,"RecHitClient");
}
  if( ps.getUntrackedParameter<bool>("PedestalClient", false) ){
    if(debug_>0)   std::cout << "===>DQM Pedestal Client is ON" << endl;
    pedestal_client_     = new HcalPedestalClient();
    pedestal_client_->init(ps, dbe_,"PedestalClient"); 
  }
  if( ps.getUntrackedParameter<bool>("LEDClient", false) ){
    if(debug_>0)   std::cout << "===>DQM LED Client is ON" << endl;
    led_client_          = new HcalLEDClient();
    led_client_->init(ps, dbe_,"LEDClient"); 
  }
  if( ps.getUntrackedParameter<bool>("LaserClient", false) ){
    if(debug_>0)   std::cout << "===>DQM Laser Client is ON" << endl;
    laser_client_          = new HcalLaserClient();
    laser_client_->init(ps, dbe_,"LaserClient"); 
  }
  if( ps.getUntrackedParameter<bool>("HotCellClient", false) ){
    if(debug_>0)   std::cout << "===>DQM HotCell Client is ON" << endl;
    hot_client_          = new HcalHotCellClient();
    hot_client_->init(ps, dbe_,"HotCellClient");
  }
  if( ps.getUntrackedParameter<bool>("DeadCellClient", false) ){
    if(debug_>0)   std::cout << "===>DQM DeadCell Client is ON" << endl;
    dead_client_          = new HcalDeadCellClient();
    dead_client_->init(ps, dbe_,"DeadCellClient");
  }
  if( ps.getUntrackedParameter<bool>("TrigPrimClient", false) ){
    if(debug_>0)   std::cout << "===>DQM TrigPrim Client is ON" << endl;
    tp_client_          = new HcalTrigPrimClient();
    tp_client_->init(ps, dbe_,"TrigPrimClient");
  }
  if( ps.getUntrackedParameter<bool>("CaloTowerClient", false) ){
    if(debug_>0)   std::cout << "===>DQM CaloTower Client is ON" << endl;
    ct_client_          = new HcalCaloTowerClient();
    ct_client_->init(ps, dbe_,"CaloTowerClient");
  }
  if( ps.getUntrackedParameter<bool>("BeamClient", false) ){
    if(debug_>0)   std::cout << "===>DQM Beam Client is ON" << endl;
    beam_client_          = new HcalBeamClient();
    beam_client_->init(ps, dbe_,"BeamClient");
  }
  ///////////////////////////////////////////////////////////////
  if( ps.getUntrackedParameter<bool>("DetDiagPedestalClient", false) ){
    if(debug_>0)   cout << "===>DQM DetDiagPedestal Client is ON" << endl;
    detdiagped_client_ = new HcalDetDiagPedestalClient();
    detdiagped_client_->init(ps, dbe_,"DetDiagPedestalClient");
  }
  if( ps.getUntrackedParameter<bool>("DetDiagLEDClient", false) ){
    if(debug_>0)   cout << "===>DQM DetDiagLED Client is ON" << endl;
    detdiagled_client_ = new HcalDetDiagLEDClient();
    detdiagled_client_->init(ps, dbe_,"DetDiagLEDClient");
  }
  if( ps.getUntrackedParameter<bool>("DetDiagLaserClient", false) ){
    if(debug_>0)   cout << "===>DQM DetDiagLaser Client is ON" << endl;
    detdiaglas_client_ = new HcalDetDiagLaserClient();
    detdiaglas_client_->init(ps, dbe_,"DetDiagLaserClient");
  }
  ///////////////////////////////////////////////////////////////

  dqm_db_ = new HcalHotCellDbInterface();  // Is this even necessary?

  
  // set parameters   
  prescaleEvt_ = ps.getUntrackedParameter<int>("diagnosticPrescaleEvt", -1);
  if (debug_>0) 
    std::cout << "===>DQM event prescale = " << prescaleEvt_ << " event(s)"<< endl;

  prescaleLS_ = ps.getUntrackedParameter<int>("diagnosticPrescaleLS", -1);
  if (debug_>0) std::cout << "===>DQM lumi section prescale = " << prescaleLS_ << " lumi section(s)"<< endl;
  if (prescaleLS_>0) actonLS_=true;

  prescaleUpdate_ = ps.getUntrackedParameter<int>("diagnosticPrescaleUpdate", -1);
  if (debug_>0) std::cout << "===>DQM update prescale = " << prescaleUpdate_ << " update(s)"<< endl;

  prescaleTime_ = ps.getUntrackedParameter<int>("diagnosticPrescaleTime", -1);
  if (debug_>0) std::cout << "===>DQM time prescale = " << prescaleTime_ << " minute(s)"<< endl;
  

  // Base folder for the contents of this job
  string subsystemname = ps.getUntrackedParameter<string>("subSystemFolder", "Hcal") ;
  if (debug_>0) std::cout << "===>HcalMonitor name = " << subsystemname << endl;
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
  if (debug_>0) std::cout <<"<HcalMonitorClient>removeAllME()"<<endl;
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
void HcalMonitorClient::resetAllME() {
  if (debug_>0) std::cout <<"<HcalMonitorClient> resetAllME()"<<endl;
  if( dataformat_client_ ) dataformat_client_->resetAllME();
  if( digi_client_ )       digi_client_->resetAllME();
  if( rechit_client_ )     rechit_client_->resetAllME();
  if( pedestal_client_ )   pedestal_client_->resetAllME();
  if( led_client_ )        led_client_->resetAllME();
  if( laser_client_ )      laser_client_->resetAllME();
  if( hot_client_ )        {
    cout <<"Resetting all ME!"<<endl;
    hot_client_->resetAllME();
  }
  if( dead_client_ )       dead_client_->resetAllME();
  if( tp_client_ )         tp_client_->resetAllME();
  if( ct_client_ )         ct_client_->resetAllME();
  if( beam_client_ )       beam_client_->resetAllME();
  /////////////////////////////////////////////////////////
  if( detdiagped_client_ ) detdiagped_client_->resetAllME();
  if( detdiagled_client_ ) detdiagled_client_->resetAllME();
  if( detdiaglas_client_ ) detdiaglas_client_->resetAllME();
  /////////////////////////////////////////////////////////

  return;
}

//--------------------------------------------------------
void HcalMonitorClient::beginJob(const EventSetup& c){

  if( debug_>0 ) std::cout << "HcalMonitorClient: beginJob" << endl;
  
  ievt_ = 0;
  if( summary_client_ )    summary_client_->beginJob(dbe_);
  if( dataformat_client_ ) dataformat_client_->beginJob();
  if( digi_client_ )       digi_client_->beginJob();
  if( rechit_client_ )     rechit_client_->beginJob();
  if( pedestal_client_ )   pedestal_client_->beginJob(c);
  if( led_client_ )        led_client_->beginJob(c);
  if( laser_client_ )      laser_client_->beginJob(c);
  if( hot_client_ )        hot_client_->beginJob(c);
  if( dead_client_ )       dead_client_->beginJob(c);
  if( tp_client_ )         tp_client_->beginJob();
  if( ct_client_ )         ct_client_->beginJob();
  if( beam_client_ )       beam_client_->beginJob();
  /////////////////////////////////////////////////////////
  if( detdiagped_client_ ) detdiagped_client_->beginJob();
  if( detdiagled_client_ ) detdiagled_client_->beginJob();
  if( detdiaglas_client_ ) detdiaglas_client_->beginJob();
  /////////////////////////////////////////////////////////
  return;
}

//--------------------------------------------------------
void HcalMonitorClient::beginRun(const Run& r, const EventSetup& c) {

  if (debug_>0)
    std::cout << endl<<"HcalMonitorClient: Standard beginRun() for run " << r.id().run() << endl<<endl;

  if( summary_client_ )    summary_client_->beginRun();
  if( dataformat_client_ ) dataformat_client_->beginRun();
  if( digi_client_ )       digi_client_->beginRun();
  if( rechit_client_ )     rechit_client_->beginRun();
  if( pedestal_client_ )   pedestal_client_->beginRun();
  if( led_client_ )        led_client_->beginRun();
  if( laser_client_ )      laser_client_->beginRun();
  if( hot_client_ )        hot_client_->beginRun();
  if( dead_client_ )       dead_client_->beginRun();
  if( tp_client_ )         tp_client_->beginRun();
  if( ct_client_ )         ct_client_->beginRun();
  if( beam_client_ )       beam_client_->beginRun();
  /////////////////////////////////////////////////////////
  if( detdiagped_client_ ) detdiagped_client_->beginRun();
  if( detdiagled_client_ ) detdiagled_client_->beginRun();
  if( detdiaglas_client_ ) detdiaglas_client_->beginRun();
  /////////////////////////////////////////////////////////
  return;
}

//--------------------------------------------------------
void HcalMonitorClient::endJob(void) {

  if( debug_>0 ) std::cout << "HcalMonitorClient: endJob, ievt = " << ievt_ << endl;

  if (summary_client_)         summary_client_->endJob();
  if( dataformat_client_ )     dataformat_client_->endJob();
  if( digi_client_ )           digi_client_->endJob();
  if( rechit_client_ )         rechit_client_->endJob();
  if( dead_client_ )           dead_client_->endJob(myquality_);
  if( hot_client_ )            hot_client_->endJob(myquality_);
  if( pedestal_client_ )       pedestal_client_->endJob();
  if( led_client_ )            led_client_->endJob();
  if( laser_client_ )          laser_client_->endJob();
  if( tp_client_ )             tp_client_->endJob();
  if( ct_client_ )             ct_client_->endJob();
  if( beam_client_ )           beam_client_->endJob();
  /////////////////////////////////////////////////////////
  if( detdiagped_client_ ) detdiagped_client_->endJob();
  if( detdiagled_client_ ) detdiagled_client_->endJob();
  if( detdiaglas_client_ ) detdiaglas_client_->endJob();
  /////////////////////////////////////////////////////////

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

  // dumping to database

  if (dump2database_)
    {
      if (debug_>0) std::cout <<"<HcalMonitorModule::endJob>  Writing file for database"<<endl;
      std::vector<DetId> mydetids = chanquality_->getAllChannels();
      HcalChannelQuality* newChanQual = new HcalChannelQuality();
      for (unsigned int i=0;i<mydetids.size();++i)
	{
	  if (mydetids[i].det()!=4) continue; // not hcal
	  //HcalDetId id(mydetids[i]);
	  HcalDetId id=mydetids[i];
	  // get original channel status item
	  const HcalChannelStatus* origstatus=chanquality_->getValues(mydetids[i]);
	  // make copy of status
	  HcalChannelStatus* mystatus=new HcalChannelStatus(origstatus->rawId(),origstatus->getValue());
	  if (myquality_.find(id)!=myquality_.end())
	    {
	      // Set bit 1 for cells which aren't present 	 
	      /*if ((id.subdet()==HcalBarrel &&!HBpresent_) || 	 
		  (id.subdet()==HcalEndcap &&!HEpresent_) || 	 
		  (id.subdet()==HcalOuter  &&!HOpresent_) || 	 
		  (id.subdet()==HcalForward&&!HFpresent_))*/
	      
	      // Update -- why do the check that subdetector is present?
	      // In normal running, if subdetector out of run, we'll still
	      // want to mark that as bad.
	      if (id.subdet()==HcalBarrel || id.subdet()==HcalEndcap ||
		  id.subdet()==HcalOuter || id.subdet()==HcalForward )
		{ 	 
		  mystatus->setBit(1); 	 
		} 	 
	      // Only perform these checks if bit 0 not set?
	      // check dead cells
	      if ((myquality_[id]>>5)&0x1)
		  mystatus->setBit(5);
	      else
		mystatus->unsetBit(5);
	      // check hot cells
	      if ((myquality_[id]>>6)&0x1)
		mystatus->setBit(6);
	      else
		mystatus->unsetBit(6);
	    } // if (myquality_.find_...)
	  newChanQual->addValues(*mystatus);
	  // Clean up pointers to avoid memory leaks
	  delete origstatus;
	  delete mystatus;
 	} // for (unsigned int i=0;...)
      // Now dump out to text file
      std::ostringstream file;
      file <<"HcalDQMstatus_"<<irun_<<".txt";
      std::ofstream outStream(file.str().c_str());
      HcalDbASCIIIO::dumpObject (outStream, (*newChanQual));

    } // if (dump2databse_)

  return;
}

//--------------------------------------------------------
void HcalMonitorClient::endRun(const Run& r, const EventSetup& c) {

  if (debug_>0)
    std::cout << endl<<"<HcalMonitorClient> Standard endRun() for run " << r.id().run() << endl<<endl;


  if( debug_ >0) std::cout <<"HcalMonitorClient: processed events: "<<ievt_<<endl;

  if (debug_>0) std::cout <<"==>Creating report after run end condition"<<endl;
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
  if( laser_client_ )       laser_client_->endRun();
  if( tp_client_ )          tp_client_->endRun();
  if( ct_client_ )          ct_client_->endRun();
  if( beam_client_ )        beam_client_->endRun();
  /////////////////////////////////////////////////////////
  if( detdiagped_client_ ) detdiagped_client_->endRun();
  if( detdiagled_client_ ) detdiagled_client_->endRun();
  if( detdiaglas_client_ ) detdiaglas_client_->endRun();
  /////////////////////////////////////////////////////////

  // this is an effective way to avoid ROOT memory leaks ...
  if( enableExit_ ) {
    if (debug_>0) std::cout << endl << ">>> exit after End-Of-Run <<<" << endl <<endl;
        
    endJob();
    throw cms::Exception("End of Job")
      << "HcalMonitorClient: Done processing...\n";
  }
}

//--------------------------------------------------------
void HcalMonitorClient::beginLuminosityBlock(const LuminosityBlock &l, const EventSetup &c) 
{
  if( debug_>0 ) std::cout << "HcalMonitorClient: beginLuminosityBlock" << endl;
  if(actonLS_ && !prescale()){
    // do scheduled tasks...
  }

}

//--------------------------------------------------------
void HcalMonitorClient::endLuminosityBlock(const LuminosityBlock &l, const EventSetup &c) {
  // then do your thing
  if( debug_>0 ) std::cout << "HcalMonitorClient: endLuminosityBlock" << endl;
  if(actonLS_ && !prescale()){
    // do scheduled tasks...
    analyze();
  }

  return;
}

//--------------------------------------------------------
void HcalMonitorClient::analyze(const Event& e, const edm::EventSetup& eventSetup){

  if (debug_>1)
    std::cout <<"Entered HcalMonitorClient::analyze(const Evt...)"<<endl;
  
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
  mytime_   = (e.time().value())>>32;

  if (minlumisec_==0)
    minlumisec_=ilumisec_;
  minlumisec_=min(minlumisec_,ilumisec_);
  maxlumisec_=max(maxlumisec_,ilumisec_);

  if (debug_>1) 
    std::cout << "HcalMonitorClient: evts: "<< ievt_ << ", run: " << irun_ << ", LS: " << ilumisec_ << ", evt: " << ievent_ << ", time: " << itime_ << endl; 

  ievt_++; //I think we want our web pages, etc. to display this counter (the number of events used in the task) rather than nevt_ (the number of times the MonitorClient analyze function below is called) -- Jeff, 1/22/08


  // Need to increment summary client on every event, not just when prescale is called, since summary_client_ plots error rates/event.
  if( summary_client_ ) 
    {
      summary_client_->incrementCounters(); // All this does is increment a counter.
      /*
      // No reason this has to be done on first event, right?
      // counters are initialized to -1 in setup
      if (ievt_ ==1) {
      summary_client_->analyze();}}        // Check if HBHE, HO, or HF is in the run at all.
      */
    }
  if ( runningStandalone_ || prescale()) return;
  
  else analyze();
}


//--------------------------------------------------------
void HcalMonitorClient::analyze(){
  if (debug_>0) 
    std::cout <<"<HcalMonitorClient> Entered HcalMonitorClient::analyze()"<<endl;

  //nevt_++; // counter not currently displayed anywhere 
  if(debug_>1) std::cout<<"\nHcal Monitor Client heartbeat...."<<endl;
  
  createTests();  
  mui_->doMonitoring();
  dbe_->runQTests();

  if (showTiming_) 
    { 
      cpu_timer.reset(); cpu_timer.start(); 
    } 
  if( dataformat_client_ ) dataformat_client_->analyze(); 	
  if (showTiming_) 
    { 
      cpu_timer.stop(); 
      if (dataformat_client_) std::cout <<"TIMER:: DATAFORMAT CLIENT ->"<<cpu_timer.cpuTime()<<endl; 
      cpu_timer.reset(); cpu_timer.start(); 
    } 

  if( digi_client_)       digi_client_->analyze(); 
  if (showTiming_) 
    { 
      cpu_timer.stop(); 
      if (digi_client_) std::cout <<"TIMER:: DIGI CLIENT ->"<<cpu_timer.cpuTime()<<endl; 
      cpu_timer.reset(); cpu_timer.start(); 
    } 

  if( rechit_client_ )     rechit_client_->analyze(); 
  if (showTiming_) 
    { 
      cpu_timer.stop(); 
      if (rechit_client_) std::cout <<"TIMER:: RECHIT CLIENT ->"<<cpu_timer.cpuTime()<<endl; 
      cpu_timer.reset(); cpu_timer.start(); 
    } 

  if( pedestal_client_ )   pedestal_client_->analyze();      
  if (showTiming_) 
    { 
      cpu_timer.stop(); 
      if (pedestal_client_) std::cout <<"TIMER:: PEDESTAL CLIENT ->"<<cpu_timer.cpuTime()<<endl; 
      cpu_timer.reset(); cpu_timer.start(); 
    } 

  if( led_client_ )        led_client_->analyze(); 
  if (showTiming_) 
    { 
      cpu_timer.stop(); 
      if (led_client_) std::cout <<"TIMER:: LED CLIENT ->"<<cpu_timer.cpuTime()<<endl; 
      cpu_timer.reset(); cpu_timer.start(); 
    } 

  if( laser_client_ )        laser_client_->analyze(); 
  if (showTiming_) 
    { 
      cpu_timer.stop(); 
      if (laser_client_) std::cout <<"TIMER:: LASER CLIENT ->"<<cpu_timer.cpuTime()<<endl; 
      cpu_timer.reset(); cpu_timer.start(); 
    } 

  if( hot_client_ )        hot_client_->analyze(); 
  if (showTiming_) 
    { 
      cpu_timer.stop(); 
      if (hot_client_) std::cout <<"TIMER:: HOT CLIENT ->"<<cpu_timer.cpuTime()<<endl; 
      cpu_timer.reset(); cpu_timer.start(); 
    } 

  if( dead_client_ )       dead_client_->analyze(); 
  if (showTiming_) 
    { 
      cpu_timer.stop(); 
      if (dead_client_) std::cout <<"TIMER:: DEAD CLIENT ->"<<cpu_timer.cpuTime()<<endl; 
      cpu_timer.reset(); cpu_timer.start(); 
    } 

  if( tp_client_ )         tp_client_->analyze(); 
  if (showTiming_) 
    { 
      cpu_timer.stop(); 
      if (tp_client_) std::cout <<"TIMER:: TP CLIENT ->"<<cpu_timer.cpuTime()<<endl; 
      cpu_timer.reset(); cpu_timer.start(); 
    } 

  if( ct_client_ )         ct_client_->analyze(); 
  if (showTiming_) 
    { 
      cpu_timer.stop(); 
      if (ct_client_) std::cout <<"TIMER:: CT CLIENT ->"<<cpu_timer.cpuTime()<<endl; 
      cpu_timer.reset(); cpu_timer.start(); 
    } 
  if( beam_client_ )         beam_client_->analyze(); 
  if (showTiming_) 
    { 
      cpu_timer.stop(); 
      if (beam_client_) std::cout <<"TIMER:: BEAM CLIENT ->"<<cpu_timer.cpuTime()<<endl; 
      cpu_timer.reset(); cpu_timer.start(); 
    } 

  if (showTiming_) 
    { 
      cpu_timer.stop(); 
      if (beam_client_) std::cout <<"TIMER:: BEAM CLIENT ->"<<cpu_timer.cpuTime()<<endl; 
      cpu_timer.reset(); cpu_timer.start(); 
    } 

  if (summary_client_ )    summary_client_->analyze();
  if (showTiming_) 
    { 
      cpu_timer.stop(); 
      if (summary_client_) std::cout <<"TIMER:: SUMMARY CLIENT ->"<<cpu_timer.cpuTime()<<endl; 
    } 

  errorSummary();

  return;
}

//--------------------------------------------------------
void HcalMonitorClient::createTests(void){
  
  if( debug_>0 ) std::cout << "HcalMonitorClient: creating all tests" << endl;

  if( dataformat_client_ ) dataformat_client_->createTests(); 
  if( digi_client_ )       digi_client_->createTests(); 
  if( rechit_client_ )     rechit_client_->createTests(); 
  if( pedestal_client_ )   pedestal_client_->createTests(); 
  if( led_client_ )        led_client_->createTests(); 
  if( laser_client_ )      laser_client_->createTests(); 
  if( hot_client_ )        hot_client_->createTests(); 
  if( dead_client_ )       dead_client_->createTests(); 
  if( tp_client_ )         tp_client_->createTests(); 
  if( ct_client_ )         ct_client_->createTests(); 
  if( beam_client_ )       beam_client_->createTests();
  /////////////////////////////////////////////////////////
  if( detdiagped_client_ ) detdiagped_client_->createTests();
  if( detdiagled_client_ ) detdiagled_client_->createTests();
  if( detdiaglas_client_ ) detdiaglas_client_->createTests();
  /////////////////////////////////////////////////////////
  return;
}

//--------------------------------------------------------
void HcalMonitorClient::report(bool doUpdate) {
  
  if( debug_>0 ) 
    std::cout << "HcalMonitorClient: creating report, ievt = " << ievt_ << endl;
  
  if(doUpdate){
    createTests();  
    dbe_->runQTests();
  }

  if( dataformat_client_ ) dataformat_client_->report();
  if( digi_client_ ) digi_client_->report();
  if( led_client_ ) led_client_->report();
  if( laser_client_ ) laser_client_->report();
  if( pedestal_client_ ) pedestal_client_->report();
  if( rechit_client_ ) rechit_client_->report();
  if( hot_client_ ) hot_client_->report();
  if( dead_client_ ) dead_client_->report();
  if( tp_client_ ) tp_client_->report();
  if( ct_client_ ) ct_client_->report();
  if( beam_client_ ) beam_client_->report();
  /////////////////////////////////////////////////////////
  if( detdiagped_client_ ) detdiagped_client_->report();
  if( detdiagled_client_ ) detdiagled_client_->report();
  if( detdiaglas_client_ ) detdiaglas_client_->report();
  /////////////////////////////////////////////////////////
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
  if( laser_client_ )      laser_client_->getTestResults(nTests,errE,errW,errO);
  if( tp_client_ )         tp_client_->getTestResults(nTests,errE,errW,errO);
  if( pedestal_client_ )   pedestal_client_->getTestResults(nTests,errE,errW,errO);
  if( digi_client_ )       digi_client_->getTestResults(nTests,errE,errW,errO);
  if( rechit_client_ )     rechit_client_->getTestResults(nTests,errE,errW,errO);
  if( dataformat_client_ ) dataformat_client_->getTestResults(nTests,errE,errW,errO);
  if( ct_client_ )         ct_client_->getTestResults(nTests,errE,errW,errO);
  if( beam_client_ )       beam_client_->getTestResults(nTests,errE,errW,errO);
  /////////////////////////////////////////////////////////
  if( detdiagped_client_ ) detdiagped_client_->getTestResults(nTests,errE,errW,errO);
  if( detdiagled_client_ ) detdiagled_client_->getTestResults(nTests,errE,errW,errO);
  if( detdiaglas_client_ ) detdiaglas_client_->getTestResults(nTests,errE,errW,errO);
  /////////////////////////////////////////////////////////
  //For now, report the fraction of good tests....
  float errorSummary = 1.0;
  if(nTests>0) errorSummary = 1.0 - (float(errE.size())+float(errW.size()))/float(nTests);
  
  if (debug_>0) std::cout << "Hcal DQM Error Summary ("<< errorSummary <<"): "<< nTests << " tests, "<<errE.size() << " errors, " <<errW.size() << " warnings, "<< errO.size() << " others" << endl;
  
  char meTitle[256];
  sprintf(meTitle,"%sEventInfo/errorSummary",rootFolder_.c_str() );
  MonitorElement* me = dbe_->get(meTitle);
  if(me) me->Fill(errorSummary);
  
  return;
}


void HcalMonitorClient::htmlOutput(void){

  if (debug_>0) std::cout << "Preparing HcalMonitorClient html output ..." << endl;

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
    if(dataformat_client_->hasErrors_Temp()) htmlFile << "<td bgcolor=red align=center>This monitor task has errors.</td>" << endl;
    else if(dataformat_client_->hasWarnings_Temp()) htmlFile << "<td bgcolor=yellow align=center>This monitor task has warnings.</td>" << endl;
    else if(dataformat_client_->hasOther_Temp()) htmlFile << "<td bgcolor=aqua align=center>This monitor task has messages.</td>" << endl;
    else htmlFile << "<td bgcolor=lime align=center>This monitor task has no problems</td>" << endl;
    htmlFile << "</tr></table>" << endl;
  }
  if( digi_client_ ) {
    htmlName = "HcalDigiClient.html";
    digi_client_->htmlOutput(irun_, htmlDir, htmlName);
    htmlFile << "<table border=0 WIDTH=\"50%\"><tr>" << endl;
    htmlFile << "<td WIDTH=\"35%\"><a href=\"" << htmlName << "\">Digi Monitor</a></td>" << endl;
    if(digi_client_->hasErrors_Temp()) htmlFile << "<td bgcolor=red align=center>This monitor task has errors.</td>" << endl;
    else if(digi_client_->hasWarnings_Temp()) htmlFile << "<td bgcolor=yellow align=center>This monitor task has warnings.</td>" << endl;
    else if(digi_client_->hasOther_Temp()) htmlFile << "<td bgcolor=aqua align=center>This monitor task has messages.</td>" << endl;
    else htmlFile << "<td bgcolor=lime align=center>This monitor task has no problems</td>" << endl;
    htmlFile << "</tr></table>" << endl;
  }
  if( tp_client_ ) {
    htmlName = "HcalTrigPrimClient.html";
    tp_client_->htmlOutput(irun_, htmlDir, htmlName);
    htmlFile << "<table border=0 WIDTH=\"50%\"><tr>" << endl;
    htmlFile << "<td WIDTH=\"35%\"><a href=\"" << htmlName << "\">TrigPrim Monitor</a></td>" << endl;
    if(tp_client_->hasErrors_Temp()) htmlFile << "<td bgcolor=red align=center>This monitor task has errors.</td>" << endl;
    else if(tp_client_->hasWarnings_Temp()) htmlFile << "<td bgcolor=yellow align=center>This monitor task has warnings.</td>" << endl;
    else if(tp_client_->hasOther_Temp()) htmlFile << "<td bgcolor=aqua align=center>This monitor task has messages.</td>" << endl;
    else htmlFile << "<td bgcolor=lime align=center>This monitor task has no problems</td>" << endl;
    htmlFile << "</tr></table>" << endl;
  }
  if( rechit_client_ ) {
    htmlName = "HcalRecHitClient.html";
    rechit_client_->htmlOutput(irun_, htmlDir, htmlName);
    htmlFile << "<table border=0 WIDTH=\"50%\"><tr>" << endl;
    htmlFile << "<td WIDTH=\"35%\"><a href=\"" << htmlName << "\">RecHit Monitor</a></td>" << endl;
    if(rechit_client_->hasErrors_Temp()) htmlFile << "<td bgcolor=red align=center>This monitor task has errors.</td>" << endl;
    else if(rechit_client_->hasWarnings_Temp()) htmlFile << "<td bgcolor=yellow align=center>This monitor task has warnings.</td>" << endl;
    else if(rechit_client_->hasOther_Temp()) htmlFile << "<td bgcolor=aqua align=center>This monitor task has messages.</td>" << endl;
    else htmlFile << "<td bgcolor=lime align=center>This monitor task has no problems</td>" << endl;
    htmlFile << "</tr></table>" << endl;
  }
  if( ct_client_ ) {
    htmlName = "HcalCaloTowerClient.html";
    ct_client_->htmlOutput(irun_, htmlDir, htmlName);
    htmlFile << "<table border=0 WIDTH=\"50%\"><tr>" << endl;
    htmlFile << "<td WIDTH=\"35%\"><a href=\"" << htmlName << "\">CaloTower Monitor</a></td>" << endl;
    if(ct_client_->hasErrors_Temp()) htmlFile << "<td bgcolor=red align=center>This monitor task has errors.</td>" << endl;
    else if(ct_client_->hasWarnings_Temp()) htmlFile << "<td bgcolor=yellow align=center>This monitor task has warnings.</td>" << endl;
    else if(ct_client_->hasOther_Temp()) htmlFile << "<td bgcolor=aqua align=center>This monitor task has messages.</td>" << endl;
    else htmlFile << "<td bgcolor=lime align=center>This monitor task has no problems</td>" << endl;
    htmlFile << "</tr></table>" << endl;
  }
  if( hot_client_ ) {
    htmlName = "HcalHotCellClient.html";
    hot_client_->htmlOutput(irun_, htmlDir, htmlName);
    htmlFile << "<table border=0 WIDTH=\"50%\"><tr>" << endl;
    htmlFile << "<td WIDTH=\"35%\"><a href=\"" << htmlName << "\">Hot Cell Monitor</a></td>" << endl;
    if(hot_client_->hasErrors_Temp()) htmlFile << "<td bgcolor=red align=center>This monitor task has errors.</td>" << endl;
    else if(hot_client_->hasWarnings_Temp()) htmlFile << "<td bgcolor=yellow align=center>This monitor task has warnings.</td>" << endl;
    else if(hot_client_->hasOther_Temp()) htmlFile << "<td bgcolor=aqua align=center>This monitor task has messages.</td>" << endl;
    else htmlFile << "<td bgcolor=lime align=center>This monitor task has no problems</td>" << endl;
    htmlFile << "</tr></table>" << endl;
  }

  if( dead_client_) {
    htmlName = "HcalDeadCellClient.html";
    dead_client_->htmlOutput(irun_, htmlDir, htmlName);
    htmlFile << "<table border=0 WIDTH=\"50%\"><tr>" << endl;
    htmlFile << "<td WIDTH=\"35%\"><a href=\"" << htmlName << "\">Dead Cell Monitor</a></td>" << endl;
    if(dead_client_->hasErrors_Temp()) htmlFile << "<td bgcolor=red align=center>This monitor task has errors.</td>" << endl;
    else if(dead_client_->hasWarnings_Temp()) htmlFile << "<td bgcolor=yellow align=center>This monitor task has warnings.</td>" << endl;
    else if(dead_client_->hasOther_Temp()) htmlFile << "<td bgcolor=aqua align=center>This monitor task has messages.</td>" << endl;
    else htmlFile << "<td bgcolor=lime align=center>This monitor task has no problems</td>" << endl;
    htmlFile << "</tr></table>" << endl;
  }
  if( pedestal_client_) {
    htmlName = "HcalPedestalClient.html";
    pedestal_client_->htmlOutput(irun_, htmlDir, htmlName);
    htmlFile << "<table border=0 WIDTH=\"50%\"><tr>" << endl;
    htmlFile << "<td WIDTH=\"35%\"><a href=\"" << htmlName << "\">Pedestal Monitor</a></td>" << endl;
    
    if(pedestal_client_->hasErrors_Temp()) htmlFile << "<td bgcolor=red align=center>This monitor task has errors.</td>" << endl;
    else if(pedestal_client_->hasWarnings_Temp()) htmlFile << "<td bgcolor=yellow align=center>This monitor task has warnings.</td>" << endl;
    else if(pedestal_client_->hasOther_Temp()) htmlFile << "<td bgcolor=aqua align=center>This monitor task has messages.</td>" << endl;
    else htmlFile << "<td bgcolor=lime align=center>This monitor task has no problems</td>" << endl;
    
    htmlFile << "</tr></table>" << endl;
  }

  if( led_client_) {
    htmlName = "HcalLEDClient.html";
    led_client_->htmlOutput(irun_, htmlDir, htmlName);
    htmlFile << "<table border=0 WIDTH=\"50%\"><tr>" << endl;
    htmlFile << "<td WIDTH=\"35%\"><a href=\"" << htmlName << "\">LED Monitor</a></td>" << endl;
    
    if(led_client_->hasErrors_Temp()) htmlFile << "<td bgcolor=red align=center>This monitor task has errors.</td>" << endl;
    else if(led_client_->hasWarnings_Temp()) htmlFile << "<td bgcolor=yellow align=center>This monitor task has warnings.</td>" << endl;
    else if(led_client_->hasOther_Temp()) htmlFile << "<td bgcolor=aqua align=center>This monitor task has messages.</td>" << endl;
    else htmlFile << "<td bgcolor=lime align=center>This monitor task has no problems</td>" << endl;
    
    htmlFile << "</tr></table>" << endl;
  }
  
  if( laser_client_) {
    htmlName = "HcalLaserClient.html";
    laser_client_->htmlOutput(irun_, htmlDir, htmlName);
    htmlFile << "<table border=0 WIDTH=\"50%\"><tr>" << endl;
    htmlFile << "<td WIDTH=\"35%\"><a href=\"" << htmlName << "\">Laser Monitor</a></td>" << endl;
    
    if(laser_client_->hasErrors_Temp()) htmlFile << "<td bgcolor=red align=center>This monitor task has errors.</td>" << endl;
    else if(laser_client_->hasWarnings_Temp()) htmlFile << "<td bgcolor=yellow align=center>This monitor task has warnings.</td>" << endl;
    else if(laser_client_->hasOther_Temp()) htmlFile << "<td bgcolor=aqua align=center>This monitor task has messages.</td>" << endl;
    else htmlFile << "<td bgcolor=lime align=center>This monitor task has no problems</td>" << endl;
    
    htmlFile << "</tr></table>" << endl;
  }

  if( beam_client_) {
    htmlName = "HcalBeamClient.html";
    beam_client_->htmlOutput(irun_, htmlDir, htmlName);
    htmlFile << "<table border=0 WIDTH=\"50%\"><tr>" << endl;
    htmlFile << "<td WIDTH=\"35%\"><a href=\"" << htmlName << "\">Beam Monitor</a></td>" << endl;
    
    if(beam_client_->hasErrors_Temp()) htmlFile << "<td bgcolor=red align=center>This monitor task has errors.</td>" << endl;
    else if(beam_client_->hasWarnings_Temp()) htmlFile << "<td bgcolor=yellow align=center>This monitor task has warnings.</td>" << endl;
    else if(beam_client_->hasOther_Temp()) htmlFile << "<td bgcolor=aqua align=center>This monitor task has messages.</td>" << endl;
    else htmlFile << "<td bgcolor=lime align=center>This monitor task has no problems</td>" << endl;
    
    htmlFile << "</tr></table>" << endl;
  }
  /////////////////////////////////////////////////////////
  if( detdiagped_client_ )if(detdiagped_client_->haveOutput()){
    htmlName = "HcalDetDiagPedestalClient.html";
    detdiagped_client_->htmlOutput(irun_, htmlDir, htmlName);
    htmlFile << "<table border=0 WIDTH=\"50%\"><tr>" << endl;
    htmlFile << "<td WIDTH=\"35%\"><a href=\"" << htmlName << "\">Pedestal Diagnostic Monitor</a></td>" << endl;
    int status=detdiagped_client_->SummaryStatus();
    if(detdiagped_client_->hasErrors() || status==2) htmlFile << "<td bgcolor=red align=center>This monitor task has errors.</td>" << endl;
    else 
    if(detdiagped_client_->hasWarnings()||status==1) htmlFile << "<td bgcolor=yellow align=center>This monitor task has warnings.</td>" << endl;
    else if(detdiagped_client_->hasOther()) htmlFile << "<td bgcolor=aqua align=center>This monitor task has messages.</td>" << endl;
    else htmlFile << "<td bgcolor=lime align=center>This monitor task has no problems</td>" << endl;
    
    htmlFile << "</tr></table>" << endl;
  }
  if( detdiagled_client_ )if(detdiagled_client_->haveOutput()){
    htmlName = "HcalDetDiagLEDClient.html";
    detdiagled_client_->htmlOutput(irun_, htmlDir, htmlName);
    htmlFile << "<table border=0 WIDTH=\"50%\"><tr>" << endl;
    htmlFile << "<td WIDTH=\"35%\"><a href=\"" << htmlName << "\">LED Diagnostic Monitor</a></td>" << endl;
    int status=detdiagled_client_->SummaryStatus();
    if(detdiagled_client_->hasErrors() || status==2) htmlFile << "<td bgcolor=red align=center>This monitor task has errors.</td>" << endl;
    else 
    if(detdiagled_client_->hasWarnings()||status==1) htmlFile << "<td bgcolor=yellow align=center>This monitor task has warnings.</td>" << endl;
    else if(detdiagled_client_->hasOther()) htmlFile << "<td bgcolor=aqua align=center>This monitor task has messages.</td>" << endl;
    else htmlFile << "<td bgcolor=lime align=center>This monitor task has no problems</td>" << endl;
    
    htmlFile << "</tr></table>" << endl;
  }
  if( detdiaglas_client_ )if(detdiaglas_client_->haveOutput()){
    htmlName = "HcalDetDiagLaserClient.html";
    detdiaglas_client_->htmlOutput(irun_, htmlDir, htmlName);
    htmlFile << "<table border=0 WIDTH=\"50%\"><tr>" << endl;
    htmlFile << "<td WIDTH=\"35%\"><a href=\"" << htmlName << "\">Laser Diagnostic Monitor</a></td>" << endl;
    int status=detdiaglas_client_->SummaryStatus();
    if(detdiaglas_client_->hasErrors() || status==2) htmlFile << "<td bgcolor=red align=center>This monitor task has errors.</td>" << endl;
    else 
    if(detdiaglas_client_->hasWarnings()||status==1) htmlFile << "<td bgcolor=yellow align=center>This monitor task has warnings.</td>" << endl;
    else if(detdiaglas_client_->hasOther()) htmlFile << "<td bgcolor=aqua align=center>This monitor task has messages.</td>" << endl;
    else htmlFile << "<td bgcolor=lime align=center>This monitor task has no problems</td>" << endl;
    
    htmlFile << "</tr></table>" << endl;
  }
  /////////////////////////////////////////////////////////
  if( summary_client_) 
    {
      htmlName = "HcalSummaryClient.html";
      summary_client_->htmlOutput(irun_, mytime_, minlumisec_, maxlumisec_, htmlDir, htmlName);
      htmlFile << "<table border=0 WIDTH=\"50%\"><tr>" << endl;
      htmlFile << "<td WIDTH=\"35%\"><a href=\"" << htmlName << "\">Summary Monitor</a></td>" << endl;
      if(summary_client_->hasErrors_Temp()) htmlFile << "<td bgcolor=red align=center>This monitor task has errors.</td>" << endl;
      else if(summary_client_->hasWarnings_Temp()) htmlFile << "<td bgcolor=yellow align=center>This monitor task has warnings.</td>" << endl;
      else if(summary_client_->hasOther_Temp()) htmlFile << "<td bgcolor=aqua align=center>This monitor task has messages.</td>" << endl;
      else
	htmlFile << "<td bgcolor=lime align=center>This monitor task has no problems</td>" << endl;
      htmlFile << "</tr></table>" << endl;
    }

  htmlFile << "</ul>" << endl;


  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  htmlFile.close();
  if (debug_>0) std::cout << "HcalMonitorClient html output done..." << endl;
  
  return;
}

void HcalMonitorClient::offlineSetup(){
  //  std::cout << endl;
  //  std::cout << " *** Hcal Generic Monitor Client, for offline operation***" << endl;
  //  std::cout << endl;
  /*
  dataformat_client_ = 0; digi_client_ = 0;
  rechit_client_ = 0; pedestal_client_ = 0;
  led_client_ = 0;  hot_client_ = 0; laser_client_ = 0;
  dead_client_=0;
  beam_client_=0;

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
  laser_client_        = new HcalLaserClient();
  beam_client_         = new HcalBeamClient();
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
  

  if(hot_client_)          hot_client_->loadHistograms(infile);
  if(dead_client_)         dead_client_->loadHistograms(infile);
  if(dataformat_client_)   dataformat_client_->loadHistograms(infile);
  if(rechit_client_)       rechit_client_->loadHistograms(infile);
  if(digi_client_)         digi_client_->loadHistograms(infile);
  if(pedestal_client_)     pedestal_client_->loadHistograms(infile);
  if(led_client_)          led_client_->loadHistograms(infile);
  if(laser_client_)        laser_client_->loadHistograms(infile);
  if(beam_client_)         beam_client_->loadHistograms(infile);
 */
  return;

}


void HcalMonitorClient::dumpHistograms(int& runNum, vector<TH1F*> &hist1d,vector<TH2F*> &hist2d){
  
  hist1d.clear(); 
  hist2d.clear(); 

  /*
  if(hot_client_)        hot_client_->dumpHistograms(names,meanX,meanY,rmsX,rmsY);
  if(dead_client_)       dead_client_->dumpHistograms(names,meanX,meanY,rmsX,rmsY);
  if(dataformat_client)  dataformat_client_->dumpHistograms(names,meanX,meanY,rmsX,rmsY);
  if(rechit_client_)     rechit_client_->dumpHistograms(names,meanX,meanY,rmsX,rmsY);
  if(digi_client_)       digi_client_->dumpHistograms(names,meanX,meanY,rmsX,rmsY);
  if(pedestal_client_)   pedestal_client_->dumpHistograms(names,meanX,meanY,rmsX,rmsY);
  if(led_client_)        led_client_->dumpHistograms(names,meanX,meanY,rmsX,rmsY);
  if(laser_client_)      laser_client_->dumpHistograms(names,meanX,meanY,rmsX,rmsY);
  if(beam_client_)       beam_client_->dumpHistograms(names,meanX,meanY,rmsX,rmsY);
  */
 return;
}

//--------------------------------------------------------
bool HcalMonitorClient::prescale(){
  ///Return true if this event should be skipped according to the prescale condition...
  ///    Accommodate a logical "OR" of the possible tests
  if (debug_>1) std::cout <<"HcalMonitorClient::prescale"<<endl;
  
  //First determine if we care...
  bool evtPS =    prescaleEvt_>0;
  bool lsPS =     prescaleLS_>0;
  bool timePS =   prescaleTime_>0;
  bool updatePS = prescaleUpdate_>0;

  // If no prescales are set, keep the event
  if(!evtPS && !lsPS && !timePS && !updatePS) return false;

  //check each instance
  if(lsPS && (ilumisec_%prescaleLS_)!=0) lsPS = false; //LS veto
  // BAH!  This doesn't work -- ievent is the raw event number, and doesn't have to be in strict numerical order.  Use ievt instead.
  //if(evtPS && (ievent_%prescaleEvt_)!=0) evtPS = false; //evt # veto
  if (evtPS && (ievt_%prescaleEvt_)!=0) evtPS = false;
  if(timePS){
    float time = psTime_.elapsedTime - psTime_.updateTime;
    if(time<prescaleTime_){
      timePS = false;  //timestamp veto
      psTime_.updateTime = psTime_.elapsedTime;
    }
  }
  //  if(prescaleUpdate_>0 && (nupdates_%prescaleUpdate_)==0) updatePS=false; ///need to define what "updates" means
  
  if (debug_>1) 
    std::cout<<"HcalMonitor::prescale  evt: "<<ievent_<<"/"<<evtPS<<", ls: "<<ilumisec_<<"/"<<lsPS<<", time: "<<(psTime_.elapsedTime - psTime_.updateTime)<<"/"<<timePS<<endl;
  /*
  printf("HcalMonitorClient::prescale  evt: %d/%d, ls: %d/%d, time: %f/%d\n",
	 ievent_,evtPS,
	 ilumisec_,lsPS,
	 psTime_.elapsedTime - psTime_.updateTime,timePS);
  */

  // if any criteria wants to keep the event, do so
  if(evtPS || lsPS || timePS) return false; //FIXME updatePS left out for now
  return true;
}


DEFINE_FWK_MODULE(HcalMonitorClient);
