#include <DQM/HcalMonitorClient/interface/HcalMonitorClient.h>

//--------------------------------------------------------
HcalMonitorClient::HcalMonitorClient(const ParameterSet& ps){
  initialize(ps);
}

HcalMonitorClient::HcalMonitorClient(){}

//--------------------------------------------------------
HcalMonitorClient::~HcalMonitorClient(){

  if (debug_>0) std::cout << "HcalMonitorClient: Exit ..." << endl;
}

//--------------------------------------------------------
void HcalMonitorClient::initialize(const ParameterSet& ps){

  irun_=0; ilumisec_=0; ievent_=0; itime_=0;

  maxlumisec_=0; minlumisec_=0;

  summary_client_ = 0;
  dataformat_client_ = 0; digi_client_ = 0;
  rechit_client_ = 0; pedestal_client_ = 0;

// #########################################################
  noise_client_ = 0;
// #########################################################

  led_client_ = 0; laser_client_ = 0; hot_client_ = 0; dead_client_=0;
  tp_client_=0;
  ct_client_=0;
  beam_client_=0;
  
  //////////////////////////////////////////////////////////////////
  detdiagped_client_=0; 
  detdiagled_client_=0;
  detdiaglas_client_=0; 
  //////////////////////////////////////////////////////////////////

  debug_ = ps.getUntrackedParameter<int>("debug", 0);
  if (debug_>0)
    std::cout << endl<<" *** Hcal Monitor Client ***" << endl<<endl;

  if(debug_>1) std::cout << "HcalMonitorClient: constructor...." << endl;

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

  databasedir_   = ps.getUntrackedParameter<std::string>("databasedir","");

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

// #########################################################
  if( ps.getUntrackedParameter<bool>("NoiseClient", false) ){
    if(debug_>0)   std::cout << "===>DQM Noise Client is ON" << endl;
    noise_client_       = new HcalDetDiagNoiseMonitorClient();
    noise_client_->init(ps, dbe_,"NoiseClient");
  }
// #########################################################

  if( ps.getUntrackedParameter<bool>("ReferencePedestalClient", false) ){
    if(debug_>0)   std::cout << "===>DQM Pedestal Client is ON" << endl;
    pedestal_client_     = new HcalPedestalClient();
    pedestal_client_->init(ps, dbe_,"ReferencePedestalClient"); 
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
    if(debug_>0)   std::cout << "===>DQM DetDiagPedestal Client is ON" << endl;
    detdiagped_client_ = new HcalDetDiagPedestalClient();
    detdiagped_client_->init(ps, dbe_,"DetDiagPedestalClient");
  }
  if( ps.getUntrackedParameter<bool>("DetDiagLEDClient", false) ){
    if(debug_>0)   std::cout << "===>DQM DetDiagLED Client is ON" << endl;
    detdiagled_client_ = new HcalDetDiagLEDClient();
    detdiagled_client_->init(ps, dbe_,"DetDiagLEDClient");
  }
  if( ps.getUntrackedParameter<bool>("DetDiagLaserClient", false) ){
    if(debug_>0)   std::cout << "===>DQM DetDiagLaser Client is ON" << endl;
    detdiaglas_client_ = new HcalDetDiagLaserClient();
    detdiaglas_client_->init(ps, dbe_,"DetDiagLaserClient");
  }
  ///////////////////////////////////////////////////////////////
  
  // set parameters   
  prescaleEvt_ = ps.getUntrackedParameter<int>("diagnosticPrescaleEvt", -1);
  if (debug_>0) 
    std::cout << "===>DQM event prescale = " << prescaleEvt_ << " event(s)"<< endl;

  prescaleLS_ = ps.getUntrackedParameter<int>("diagnosticPrescaleLS", -1);
  if (debug_>0) std::cout << "===>DQM lumi section prescale = " << prescaleLS_ << " lumi section(s)"<< endl;

  // Base folder for the contents of this job
  string subsystemname = ps.getUntrackedParameter<string>("subSystemFolder", "Hcal") ;
  if (debug_>0) std::cout << "===>HcalMonitor name = " << subsystemname << endl;
  rootFolder_ = subsystemname + "/";
  if (dbe_!=NULL)
    {
      dbe_->setCurrentFolder(rootFolder_+"DQM Job Status" );
      meProcessedEndLumi_=dbe_->bookInt("EndLumiBlockProcessed_MonitorClient");
      meProcessedEndLumi_->Fill(-1);
    }
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

// #########################################################
  if( noise_client_ )     noise_client_->resetAllME();
// #########################################################

  if( pedestal_client_ )   pedestal_client_->resetAllME();
  if( led_client_ )        led_client_->resetAllME();
  if( laser_client_ )      laser_client_->resetAllME();
  if( hot_client_ )        hot_client_->resetAllME();
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
void HcalMonitorClient::beginJob(){

  if( debug_>0 ) std::cout << "HcalMonitorClient: beginJob" << endl;
  
  ievt_ = 0;
  if( summary_client_ )    summary_client_->beginJob(dbe_);
  if( dataformat_client_ ) dataformat_client_->beginJob();
  if( digi_client_ )       digi_client_->beginJob();
  if( rechit_client_ )     rechit_client_->beginJob();

// #########################################################
  if( noise_client_ )     noise_client_->beginJob();
// #########################################################

  if( pedestal_client_ )   pedestal_client_->beginJob();
  if( led_client_ )        led_client_->beginJob();
  if( laser_client_ )      laser_client_->beginJob();
  if( hot_client_ )        hot_client_->beginJob();
  if( dead_client_ )       dead_client_->beginJob();
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
  myquality_.clear(); // remove old quality flag contents at the start of each run
  if( summary_client_ )    summary_client_->beginRun();
  if( dataformat_client_ ) dataformat_client_->beginRun();
  if( digi_client_ )       digi_client_->beginRun();
  if( rechit_client_ )     rechit_client_->beginRun();

// #########################################################
  if( noise_client_ )     noise_client_->beginRun();
// #########################################################

  if( pedestal_client_ )   pedestal_client_->beginRun(c);
  if( led_client_ )        led_client_->beginRun(c);
  if( laser_client_ )      laser_client_->beginRun(c);
  if( hot_client_ )        hot_client_->beginRun(c);
  if( dead_client_ )       dead_client_->beginRun(c);
  if( tp_client_ )         tp_client_->beginRun();
  if( ct_client_ )         ct_client_->beginRun();
  if( beam_client_ )       beam_client_->beginRun();
  /////////////////////////////////////////////////////////
  if( detdiagped_client_ ) detdiagped_client_->beginRun();
  if( detdiagled_client_ ) detdiagled_client_->beginRun();
  if( detdiaglas_client_ ) detdiaglas_client_->beginRun();
  /////////////////////////////////////////////////////////

  if (databasedir_.size()==0) return;
  // Get current channel quality 
  edm::ESHandle<HcalChannelQuality> p;
  c.get<HcalChannelQualityRcd>().get(p);
  chanquality_= new HcalChannelQuality(*p.product());
  return;
}

//--------------------------------------------------------
void HcalMonitorClient::endJob(void) {

  if( debug_>0 ) 
    std::cout << "HcalMonitorClient: endJob, ievt = " << ievt_ << endl;

  if (summary_client_)         summary_client_->endJob();
  if( dataformat_client_ )     dataformat_client_->endJob();
  if( digi_client_ )           digi_client_->endJob();
  if( rechit_client_ )         rechit_client_->endJob();

// #########################################################
  if( noise_client_ )         noise_client_->endJob();
// #########################################################

  if( dead_client_ )           dead_client_->endJob();
  if( hot_client_ )            hot_client_->endJob();
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

  return;
}

//--------------------------------------------------------
void HcalMonitorClient::endRun(const Run& r, const EventSetup& c) {

  if (debug_>0)
    std::cout << endl<<"<HcalMonitorClient> Standard endRun() for run " << r.id().run() << endl<<endl;

  if (!Online_)
    analyze();

  if( debug_ >0) std::cout <<"HcalMonitorClient: processed events: "<<ievt_<<endl;

  if (debug_>0) std::cout <<"==>Creating report after run end condition"<<endl;
  if(irun_>1){
    if(inputFile_.size()!=0) report(true);
    else report(false);
  }

  if( dead_client_ )        dead_client_->endRun(myquality_);
  if( hot_client_ )         hot_client_->endRun(myquality_);
  if( dataformat_client_ )  dataformat_client_->endRun();
  if( digi_client_ )        digi_client_->endRun();
  if( rechit_client_ )      rechit_client_->endRun();

// #########################################################
  if( noise_client_ )      noise_client_->endRun();
// #########################################################

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
  if( summary_client_)      summary_client_->endRun();


  // dumping to database

  // need to add separate function to do this!!!
  
  writeDBfile();
  return;
}

void HcalMonitorClient::writeDBfile()

{
  if (databasedir_.size()==0) return;
  if (debug_>0) std::cout <<"<HcalMonitorClient::writeDBfile>  Writing file for database"<<endl;
  std::vector<DetId> mydetids = chanquality_->getAllChannels();
  HcalChannelQuality* newChanQual = new HcalChannelQuality();
  for (unsigned int i=0;i<mydetids.size();++i)
    {
      if (mydetids[i].det()!=DetId::Hcal) continue; // not hcal
      
      HcalDetId id=mydetids[i];
      // get original channel status item
      const HcalChannelStatus* origstatus=chanquality_->getValues(mydetids[i]);
      // make copy of status
      HcalChannelStatus* mystatus=new HcalChannelStatus(origstatus->rawId(),origstatus->getValue());
      // loop over myquality flags
      if (myquality_.find(id)!=myquality_.end())
	{
	  
	  // check dead cells
	  if ((myquality_[id]>>HcalChannelStatus::HcalCellDead)&0x1)
	    mystatus->setBit(HcalChannelStatus::HcalCellDead);
	  else
	    mystatus->unsetBit(HcalChannelStatus::HcalCellDead);
	  // check hot cells
	  if ((myquality_[id]>>HcalChannelStatus::HcalCellHot)&0x1)
	    mystatus->setBit(HcalChannelStatus::HcalCellHot);
	  else
	    mystatus->unsetBit(HcalChannelStatus::HcalCellHot);
	} // if (myquality_.find_...)
      newChanQual->addValues(*mystatus);
    } // for (unsigned int i=0;...)
      // Now dump out to text file
  std::ostringstream file;
  databasedir_=databasedir_+"/"; // add extra slash, just in case
  file <<databasedir_<<"HcalDQMstatus_"<<irun_<<".txt";
  std::ofstream outStream(file.str().c_str());
  HcalDbASCIIIO::dumpObject (outStream, (*newChanQual));
  return;
} // HcalMonitorClient::writeDBfile()

//--------------------------------------------------------
void HcalMonitorClient::beginLuminosityBlock(const LuminosityBlock &l, const EventSetup &c) 
{
  // don't allow 'backsliding' across lumi blocks in online running
  // This still won't prevent some lumi blocks from being evaluated multiple times.  Need to think about this.
  //if (Online_ && (int)l.luminosityBlock()<ilumisec_) return;
  if (debug_>0) std::cout <<"Entered Monitor Client beginLuminosityBlock for LS = "<<l.luminosityBlock()<<endl;
  ilumisec_ = l.luminosityBlock();
  if( debug_>0 ) std::cout << "HcalMonitorClient: beginLuminosityBlock" << endl;
  if( summary_client_)      summary_client_->SetLS(ilumisec_);
  if( hot_client_ )         hot_client_->SetLS(ilumisec_);
  if( dead_client_ )        dead_client_->SetLS(ilumisec_); 
  if( dataformat_client_ )  dataformat_client_->SetLS(ilumisec_);
  if( digi_client_ )        digi_client_->SetLS(ilumisec_);
  if( rechit_client_ )      rechit_client_->SetLS(ilumisec_);

// #########################################################
  if( noise_client_ )      noise_client_->SetLS(l.luminosityBlock());
// #########################################################

  if( pedestal_client_ )    pedestal_client_->SetLS(ilumisec_);
  if( led_client_ )         led_client_->SetLS(ilumisec_);
  if( laser_client_ )       laser_client_->SetLS(ilumisec_);
  if( tp_client_ )          tp_client_->SetLS(ilumisec_);
  if( ct_client_ )          ct_client_->SetLS(ilumisec_);
  if( beam_client_ )        beam_client_->SetLS(ilumisec_);
  /////////////////////////////////////////////////////////
  if( detdiagped_client_ ) detdiagped_client_->SetLS(ilumisec_);
  if( detdiagled_client_ ) detdiagled_client_->SetLS(ilumisec_);
  if( detdiaglas_client_ ) detdiaglas_client_->SetLS(ilumisec_);
  /////////////////////////////////////////////////////////
}

//--------------------------------------------------------
void HcalMonitorClient::endLuminosityBlock(const LuminosityBlock &l, const EventSetup &c) {

  // don't allow backsliding in online running
  //if (Online_ && (int)l.luminosityBlock()<ilumisec_) return;
  meProcessedEndLumi_->Fill(l.luminosityBlock());
  if( debug_>0 ) std::cout << "HcalMonitorClient: endLuminosityBlock" << endl;
  if(prescaleLS_>0 && prescale()==false){
    // do scheduled tasks...
    if (Online_)
      analyze();
  }

  return;
}

//--------------------------------------------------------
void HcalMonitorClient::analyze(const Event& e, const edm::EventSetup& eventSetup){

  if (debug_>1)
    std::cout <<"Entered HcalMonitorClient::analyze(const Evt...)"<<endl;
  
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
    std::cout << "HcalMonitorClient: evts: "<< ievt_ << ", run: " << irun_ << ", LS: " << ilumisec_ << ", evt: " << ievent_ << ", time: " << itime_ << endl; 
  
  ievt_++; 

  // Need to increment summary client on every event, not just when prescale is called, since summary_client_ plots error rates/event.
  // Is this still true?  10 Nov 2009
  if( summary_client_ ) 
    summary_client_->incrementCounters(); // All this does is increment a counter.

  if (ievt_%50000==10000) writeDBfile(); // write to db every 50k events, starting with event 10000 -- add cfg values some day?
  if ( runningStandalone_) return;

  // run if we want to check individual events, and if this event isn't prescaled
  if (prescaleEvt_>0 && prescale()==false) 
    analyze();
}


//--------------------------------------------------------
void HcalMonitorClient::analyze(){
  if (debug_>0) 
    std::cout <<"<HcalMonitorClient> Entered HcalMonitorClient::analyze()"<<endl;
  if(debug_>1) std::cout<<"\nHcal Monitor Client heartbeat...."<<endl;
  
  createTests();  
  //mui_->doMonitoring();
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

// #########################################################
  if( noise_client_ )     noise_client_->analyze(); 
  if (showTiming_) 
    { 
      cpu_timer.stop(); 
      if (noise_client_) std::cout <<"TIMER:: RECHIT CLIENT ->"<<cpu_timer.cpuTime()<<endl; 
      cpu_timer.reset(); cpu_timer.start(); 
    } 
// #########################################################

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

// #########################################################
  if( noise_client_ )     noise_client_->createTests(); 
// #########################################################
 
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

// #########################################################
  if( noise_client_ ) noise_client_->report();
// #########################################################

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
  if( baseHtmlDir_.size() != 0 && ievt_>0) 
    htmlOutput();
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

// #########################################################
  if( noise_client_ )     noise_client_->getTestResults(nTests,errE,errW,errO);
// #########################################################

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
  
  // global ROOT style
  gStyle->Reset("Default");
  gStyle->SetCanvasColor(0);
  gStyle->SetPadColor(0);
  gStyle->SetFillColor(0);
  gStyle->SetTitleFillColor(10);
  //  gStyle->SetOptStat(0);
  gStyle->SetOptStat("ouemr");
  gStyle->SetPalette(1);

  char tmp[20];
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

// #########################################################
  if( noise_client_ ) {
    htmlName = "HcalDetDiagNoiseMonitorClient.html";
    noise_client_->htmlOutput(irun_, htmlDir, htmlName);
    htmlFile << "<table border=0 WIDTH=\"50%\"><tr>" << endl;
    htmlFile << "<td WIDTH=\"35%\"><a href=\"" << htmlName << "\">Noise Monitor</a></td>" << endl;
    if(noise_client_->hasErrors_Temp()) htmlFile << "<td bgcolor=red align=center>This monitor task has errors.</td>" << endl;
    else if(noise_client_->hasWarnings_Temp()) htmlFile << "<td bgcolor=yellow align=center>This monitor task has warnings.</td>" << endl;
    else if(noise_client_->hasOther_Temp()) htmlFile << "<td bgcolor=aqua align=center>This monitor task has messages.</td>" << endl;
    else htmlFile << "<td bgcolor=lime align=center>This monitor task has no problems</td>" << endl;
    htmlFile << "</tr></table>" << endl;
  }
// #########################################################

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
  return;
}

void HcalMonitorClient::loadHistograms(TFile* infile, const char* fname)
{
  if(!infile){
    throw cms::Exception("Incomplete configuration") << 
      "HcalMonitorClient: this histogram file is bad! " <<endl;
    return;
  }
  return;
}


void HcalMonitorClient::dumpHistograms(int& runNum, vector<TH1F*> &hist1d,vector<TH2F*> &hist2d)
{
  hist1d.clear(); 
  hist2d.clear(); 
  return;
}

//--------------------------------------------------------
bool HcalMonitorClient::prescale(){
  ///Return true if this event should be skipped according to the prescale condition...

  ///    Accommodate a logical "OR" of the possible tests
  if (debug_>1) std::cout <<"HcalMonitorClient::prescale"<<endl;
  
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


DEFINE_FWK_MODULE(HcalMonitorClient);
