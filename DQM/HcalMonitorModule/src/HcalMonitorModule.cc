#include <DQM/HcalMonitorModule/src/HcalMonitorModule.h>
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

/*
 * \file HcalMonitorModule.cc
 * 
 * $Date: 2009/12/14 17:45:16 $
 * $Revision: 1.158 $
 * \author W Fisher
 * \author J Temple
 *
*/

using namespace std;
using namespace edm;

//--------------------------------------------------------
HcalMonitorModule::HcalMonitorModule(const edm::ParameterSet& ps){

  irun_=0; ilumisec=0; ievent_=0; itime_=0;

  meStatus_=0;  
  meFEDS_=0;
  meLatency_=0; meQuality_=0;
  fedsListed_ = false;
  digiMon_ = 0;   dfMon_ = 0;
  diTask_ = 0;
  rhMon_ = 0;     pedMon_ = 0; 
  ledMon_ = 0;    mtccMon_ = 0;
  hotMon_ = 0;    tempAnalysis_ = 0;
  deadMon_ = 0;   tpMon_ = 0;
  ctMon_ = 0;     beamMon_ = 0;
  laserMon_ = 0;
  expertMon_ = 0;  eeusMon_ = 0;
  zdcMon_ = 0;

  nzsMon_=0; // need to initialize to 0!

  ////////////////////////////////////
  detDiagPed_ =0; detDiagLed_ =0; detDiagLas_ =0; detDiagNoise_ =0; detDiagTiming_=0;
  ///////////////////////////////////// 

  // initialize hcal quality object
  

  // All subdetectors assumed out of the run by default
  HBpresent_=0;
  HEpresent_=0;
  HOpresent_=0;
  HFpresent_=0;

  inputLabelDigi_        = ps.getParameter<edm::InputTag>("digiLabel");
  inputLabelRecHitHBHE_  = ps.getParameter<edm::InputTag>("hbheRecHitLabel");
  inputLabelRecHitHF_    = ps.getParameter<edm::InputTag>("hfRecHitLabel");
  inputLabelRecHitHO_    = ps.getParameter<edm::InputTag>("hoRecHitLabel");
  inputLabelRecHitZDC_   = ps.getParameter<edm::InputTag>("zdcRecHitLabel");
  inputLabelCaloTower_   = ps.getParameter<edm::InputTag>("caloTowerLabel");
  inputLabelLaser_       = ps.getParameter<edm::InputTag>("hcalLaserLabel");

  //for NZS monitor:
  inputHLTresultsTag_       = ps.getUntrackedParameter<edm::InputTag>("hltResultsTag",edm::InputTag("HLTriggerResults","","HLT"));

  // Check Online running
  Online_                = ps.getUntrackedParameter<bool>("Online",false);
  //Emulator
  inputLabelEmulDigi_  = ps.getParameter<edm::InputTag>("emulTPLabel");

  checkHB_=ps.getUntrackedParameter<bool>("checkHB", 1); 
  checkHE_=ps.getUntrackedParameter<bool>("checkHE", 1);  
  checkHO_=ps.getUntrackedParameter<bool>("checkHO", 1);  
  checkHF_=ps.getUntrackedParameter<bool>("checkHF", 1);   
  
  dbe_ = Service<DQMStore>().operator->();
  
  debug_ = ps.getUntrackedParameter<int>("debug", 0);
  
  periodicReset_ = ps.getUntrackedParameter<int>("periodicReset",-1);

  showTiming_ = ps.getUntrackedParameter<bool>("showTiming", false);
  dump2database_   = ps.getUntrackedParameter<bool>("dump2database",false); // dumps output to database file

  FEDRawDataCollection_ = ps.getUntrackedParameter<edm::InputTag>("FEDRawDataCollection",edm::InputTag("source",""));

  bool taskOn = ps.getUntrackedParameter<bool>("DataFormatMonitor", false);
  if (taskOn) {
    if(debug_>0) std::cout << "HcalMonitorModule: DataFormat monitor flag is on...." << std::endl;
    dfMon_ = new HcalDataFormatMonitor();
    dfMon_->setup(ps, dbe_);
  }

  taskOn = ps.getUntrackedParameter<bool>("DataIntegrityTask", false); 
  if (taskOn ) 
    {
      if (debug_>0) std::cout <<"HcalMonitorModule: DataIntegrity monitor flag is on...."<<endl;
      diTask_ = new HcalDataIntegrityTask();
      diTask_->setup(ps, dbe_);
    }

  if ( ps.getUntrackedParameter<bool>("DigiMonitor", false) ) {
    if(debug_>0) std::cout << "HcalMonitorModule: Digi monitor flag is on...." << std::endl;
    digiMon_ = new HcalDigiMonitor();
    digiMon_->setup(ps, dbe_);
  }
  
  if ( ps.getUntrackedParameter<bool>("RecHitMonitor", false) ) {
    if(debug_>0) std::cout << "HcalMonitorModule: RecHit monitor flag is on...." << std::endl;
    rhMon_ = new HcalRecHitMonitor();
    rhMon_->setup(ps, dbe_);
  }
  
  if ( ps.getUntrackedParameter<bool>("ReferencePedestalMonitor", false) ) {
    if(debug_>0) std::cout << "HcalMonitorModule: Pedestal monitor flag is on...." << std::endl;
    pedMon_ = new HcalPedestalMonitor();
    pedMon_->setup(ps, dbe_);
  }
  
  if ( ps.getUntrackedParameter<bool>("LEDMonitor", false) ) {
    if(debug_>0) std::cout << "HcalMonitorModule: LED monitor flag is on...." << std::endl;
    ledMon_ = new HcalLEDMonitor();
    ledMon_->setup(ps, dbe_);
  }
  
  if ( ps.getUntrackedParameter<bool>("LaserMonitor", false) ) {
    if(debug_>0) std::cout << "HcalMonitorModule: Laser monitor flag is on...." << std::endl;
    laserMon_ = new HcalLaserMonitor();
    laserMon_->setup(ps, dbe_);
  }

  if ( ps.getUntrackedParameter<bool>("MTCCMonitor", false) ) {
    if(debug_>0) std::cout << "HcalMonitorModule: MTCC monitor flag is on...." << std::endl;
    mtccMon_ = new HcalMTCCMonitor();
    mtccMon_->setup(ps, dbe_);
  }
  
  if ( ps.getUntrackedParameter<bool>("HotCellMonitor", false) ) {
    if(debug_>0) std::cout << "HcalMonitorModule: Hot Cell monitor flag is on...." << std::endl;
    hotMon_ = new HcalHotCellMonitor();
    hotMon_->setup(ps, dbe_);
  }
  
  if ( ps.getUntrackedParameter<bool>("DeadCellMonitor", false) ) {
    if(debug_>0) std::cout << "HcalMonitorModule: Dead Cell monitor flag is on...." << std::endl;
    deadMon_ = new HcalDeadCellMonitor();
    deadMon_->setup(ps, dbe_);
  }

  if ( ps.getUntrackedParameter<bool>("TrigPrimMonitor", false) ) { 	 
    if(debug_>0) std::cout << "HcalMonitorModule: TrigPrim monitor flag is on...." << std::endl; 	 
    tpMon_ = new HcalTrigPrimMonitor(); 	 
    tpMon_->setup(ps, dbe_); 	 
  }  

  if (ps.getUntrackedParameter<bool>("CaloTowerMonitor",false)){
    if(debug_>0) std::cout << "HcalMonitorModule: CaloTower monitor flag is on...." << std::endl; 	 
    ctMon_ = new HcalCaloTowerMonitor(); 	 
    ctMon_->setup(ps, dbe_); 	 
  }  

  if (ps.getUntrackedParameter<bool>("BeamMonitor",false)){
    if(debug_>0) std::cout << "HcalMonitorModule: Beam monitor flag is on...."<<endl;
    beamMon_ = new HcalBeamMonitor();
    beamMon_->setup(ps, dbe_);
  }

  if (ps.getUntrackedParameter<bool>("ZDCMonitor",false))
    {
      if (debug_>0) std::cout <<"HcalMonitorModule: ZDC monitor flag is on..."<<endl;
      zdcMon_ = new HcalZDCMonitor();
      zdcMon_->setup(ps, dbe_);
    }

  if (ps.getUntrackedParameter<bool>("ExpertMonitor",false)){
    if(debug_>0) std::cout << "HcalMonitorModule: Expert monitor flag is on...."<<endl;
    expertMon_ = new HcalExpertMonitor();
    expertMon_->setup(ps, dbe_);
  }
  
  if (ps.getUntrackedParameter<bool>("NZSMonitor",false)){
    if(debug_>0) std::cout << "HcalMonitorModule: NZS monitor flag is on...."<<endl;
    nzsMon_ = new HcalNZSMonitor();
    nzsMon_->setup(ps, dbe_);
  }


  //////////////////////////////////////////////////////
  if ( ps.getUntrackedParameter<bool>("DetDiagPedestalMonitor", false) ) {
    if(debug_>0) std::cout << "HcalDetDiagPedestalMonitor: Hcal Analysis flag is on...." << std::endl;
    detDiagPed_= new HcalDetDiagPedestalMonitor();
    detDiagPed_->setup(ps, dbe_);
  }
  if ( ps.getUntrackedParameter<bool>("DetDiagLEDMonitor", false) ) {
    if(debug_>0) std::cout << "HcalDetDiagLEDMonitor: Hcal Analysis flag is on...." << std::endl;
    detDiagLed_= new HcalDetDiagLEDMonitor();
    detDiagLed_->setup(ps, dbe_);
  }
  if ( ps.getUntrackedParameter<bool>("DetDiagLaserMonitor", false) ) {
    if(debug_>0) std::cout << "HcalDetDiagLaserMonitor: Hcal Analysis flag is on...." << std::endl;
    detDiagLas_= new HcalDetDiagLaserMonitor();
    detDiagLas_->setup(ps, dbe_);
  }
  if ( ps.getUntrackedParameter<bool>("DetDiagNoiseMonitor", false) ) {
    if(debug_>0) std::cout << "DetDiagNoiseMonitor: Hcal Analysis flag is on...." << std::endl;
    detDiagNoise_= new HcalDetDiagNoiseMonitor();
    detDiagNoise_->setup(ps, dbe_);
  }
  if ( ps.getUntrackedParameter<bool>("DetDiagTimingMonitor", false) ) {
    if(debug_>0) std::cout << "DetDiagTimingMonitor: Hcal Analysis flag is on...." << std::endl;
    detDiagTiming_= new HcalDetDiagTimingMonitor();
    detDiagTiming_->setup(ps, dbe_);
  }
  //////////////////////////////////////////////////////


  if ( ps.getUntrackedParameter<bool>("HcalAnalysis", false) ) {
    if(debug_>0) std::cout << "HcalMonitorModule: Hcal Analysis flag is on...." << std::endl;
    tempAnalysis_ = new HcalTemplateAnalysis();
    tempAnalysis_->setup(ps);
  }

  if (ps.getUntrackedParameter<bool>("EEUSMonitor",false))
    {
      if (debug_>0) std::cout <<"HcalMonitorModule:  Empty Event/Unsuppressed Moniotr is on..."<<endl;
      eeusMon_ = new HcalEEUSMonitor();
      eeusMon_->setup(ps, dbe_);
    }

  // set parameters   
  prescaleEvt_ = ps.getUntrackedParameter<int>("diagnosticPrescaleEvt", -1);
  if(debug_>1) std::cout << "===>HcalMonitor event prescale = " << prescaleEvt_ << " event(s)"<< std::endl;

  prescaleLS_ = ps.getUntrackedParameter<int>("diagnosticPrescaleLS", -1);
  if(debug_>1) std::cout << "===>HcalMonitor lumi section prescale = " << prescaleLS_ << " lumi section(s)"<< std::endl;
  
  // Base folder for the contents of this job
  string subsystemname = ps.getUntrackedParameter<string>("subSystemFolder", "Hcal") ;
  if(debug_>0) std::cout << "===>HcalMonitor name = " << subsystemname << std::endl;
  rootFolder_ = subsystemname + "/";
  
  gettimeofday(&psTime_.updateTV,NULL);
  /// get time in milliseconds, convert to minutes
  psTime_.updateTime = (psTime_.updateTV.tv_sec*1000.0+psTime_.updateTV.tv_usec/1000.0);
  psTime_.updateTime /= 1000.0;
  psTime_.elapsedTime=0;
  psTime_.vetoTime=psTime_.updateTime;
}

//--------------------------------------------------------
HcalMonitorModule::~HcalMonitorModule()
{
  if (dbe_!=0)
    {    
      if(digiMon_!=0)   {  digiMon_->clearME();}
      if(dfMon_!=0)     {  dfMon_->clearME();}
      if(diTask_!=0)    {  diTask_->clearME();}
      if(pedMon_!=0)    {  pedMon_->clearME();}
      if(ledMon_!=0)    {  ledMon_->clearME();}
      if(laserMon_!=0)  {  laserMon_->clearME();}
      if(hotMon_!=0)    {  hotMon_->clearME();}
      if(deadMon_!=0)   {  deadMon_->clearME();}
      if(mtccMon_!=0)   {  mtccMon_->clearME();}
      if(rhMon_!=0)     {  rhMon_->clearME();}
      if (zdcMon_!=0)   {  zdcMon_->clearME();}
      if (beamMon_!=0) { beamMon_->clearME();}
      if (nzsMon_!=0) { nzsMon_->clearME();}
      
      //////////////////////////////////////////////
      if(detDiagPed_!=0){  detDiagPed_->clearME();}
      if(detDiagLed_!=0){  detDiagLed_->clearME();}
      if(detDiagLas_!=0){  detDiagLas_->clearME();}
      if(detDiagNoise_!=0){  detDiagNoise_->clearME();}
      if(detDiagTiming_!=0){  detDiagTiming_->clearME();}
      /////////////////////////////////////////////
      
      dbe_->setCurrentFolder(rootFolder_);
      dbe_->removeContents();
    }
  
  // I think setting pointers to NULL (0) after delete is unnecessary here,
  // since we're in the destructor (and thus won't be using the pointers again.)
  if(digiMon_!=0) 
    { 
      delete digiMon_;  digiMon_=0; 
    }
  if(dfMon_!=0) 
    { delete dfMon_;     dfMon_=0; 
    }
  if(diTask_!=0) 
    { delete diTask_;   diTask_=0; 
    }
  if(pedMon_!=0) 
    {
      delete pedMon_;   pedMon_=0; 
    }
  if(ledMon_!=0) 
    { delete ledMon_;   ledMon_=0; 
    }
  if(laserMon_!=0) 
    { delete laserMon_;   laserMon_=0; 
    }
  if(hotMon_!=0) 
    { delete hotMon_;   hotMon_=0; 
    }
  if(deadMon_!=0) 
    { delete deadMon_; deadMon_=0; 
    }
  if (beamMon_!=0)
    { delete beamMon_;  beamMon_=0;
    }
  if(mtccMon_!=0) 
    { delete mtccMon_; mtccMon_=0; 
    }
  if(rhMon_!=0) 
    { delete rhMon_;     rhMon_=0; 
    }
  if (zdcMon_!=0)
    {
      delete zdcMon_; zdcMon_=0;
    }
  if (nzsMon_!=0)
    {
      delete nzsMon_; nzsMon_=0;
    }
  
  if(tempAnalysis_!=0) 
    { delete tempAnalysis_; 
    tempAnalysis_=0; 
    }
  /////////////////////////////////////////////
  if(detDiagPed_!=0) 
    { delete detDiagPed_; 
    detDiagPed_=0; 
    }
  if(detDiagLed_!=0) 
    { delete detDiagLed_; 
    detDiagLed_=0; 
    }
  if(detDiagLas_!=0) 
    { delete detDiagLas_; 
    detDiagLas_=0; 
    }
  if(detDiagNoise_!=0) 
    { delete detDiagNoise_; 
    detDiagNoise_=0; 
    }
  if(detDiagTiming_!=0) 
    { delete detDiagTiming_; 
    detDiagTiming_=0; 
    }
  ////////////////////////////////////////////  

} //void HcalMonitorModule::~HcalMonitorModule()

//--------------------------------------------------------
// beginJob no longer needed; trying setup within beginJob won't work -- IOV's not loaded
void HcalMonitorModule::beginJob()
{
  // should we reset these counters at the start of each run?
  ievt_ = 0;
  ievt_pre_=0;

  // Counters for rawdata, digi, and rechit
  ievt_rawdata_=0;
  ievt_digi_=0;
  ievt_rechit_=0;
  return;
}

//--------------------------------------------------------
void HcalMonitorModule::beginRun(const edm::Run& run, const edm::EventSetup& c) {
 
  fedsListed_ = false;

  // I think we want to reset these at 0 at the start of each run
  HBpresent_ = 0;
  HEpresent_ = 0;
  HOpresent_ = 0;
  HFpresent_ = 0;

  reset();

  if ( dbe_ != NULL ){
    dbe_->setCurrentFolder(rootFolder_+"DQM Job Status" );

    meIEVTALL_ = dbe_->bookInt("Events Processed");
    meIEVTRAW_ = dbe_->bookInt("Events with Raw Data");
    meIEVTDIGI_= dbe_->bookInt("Events with Digis");
    meIEVTRECHIT_ = dbe_->bookInt("Events with RecHits");
    meIEVTALL_->Fill(ievt_);
    meIEVTRAW_->Fill(ievt_rawdata_);
    meIEVTDIGI_->Fill(ievt_digi_);
    meIEVTRECHIT_->Fill(ievt_rechit_);
    meStatus_  = dbe_->bookInt("STATUS");
   
    meFEDS_    = dbe_->book1D("FEDs Unpacked","FEDs Unpacked",100,700,799);
    // process latency was (200,0,1), but that gave overflows
    meLatency_ = dbe_->book1D("Process Latency","Process Latency",2000,0,10);
    meQuality_ = dbe_->book1D("Quality Status","Quality Status",100,0,1);
    // Store whether or not subdetectors are present
    meHB_ = dbe_->bookInt("HBpresent");
    meHE_ = dbe_->bookInt("HEpresent");
    meHO_ = dbe_->bookInt("HOpresent");
    meHF_ = dbe_->bookInt("HFpresent");
    meStatus_->Fill(0);

    // Should fill with 0 to start
    meHB_->Fill(HBpresent_);
    meHE_->Fill(HEpresent_);
    meHO_->Fill(HOpresent_);
    meHF_->Fill(HFpresent_);
  }

  // Create histograms for individual Tasks
  if (digiMon_)   digiMon_->beginRun();
  if (pedMon_)    pedMon_->beginRun();
  if (rhMon_)     rhMon_->beginRun();

  if (beamMon_)   beamMon_->beginRun(c,run.run()); // pass in event setup to get list of bad channel quality cells 
  if (expertMon_) expertMon_->beginRun();
  if (pedMon_)    pedMon_->beginRun();
  if (ledMon_)    ledMon_->beginRun();
  if (laserMon_)  laserMon_->beginRun();

  if (mtccMon_)   mtccMon_->beginRun();
  if (hotMon_)    hotMon_->beginRun();
  if (deadMon_)   deadMon_->beginRun();
  if (ctMon_)     ctMon_->beginRun();

  if (tpMon_)     tpMon_->beginRun();
  if (zdcMon_)    zdcMon_->beginRun();
  if (eeusMon_)   eeusMon_->beginRun();

  if (nzsMon_!=NULL)     nzsMon_->beginRun();

  edm::ESHandle<HcalDbService> pSetup;
  c.get<HcalDbRecord>().get( pSetup );

  readoutMap_=pSetup->getHcalMapping();
  DetId detid_;
  HcalDetId hcaldetid_; 

  // Build a map of readout hardware unit to calorimeter channel
  std::vector <HcalElectronicsId> AllElIds = readoutMap_->allElectronicsIdPrecision();
  uint32_t itsdcc    =0;
  uint32_t itsspigot =0;
  uint32_t itshtrchan=0;
  
  // by looping over all precision (non-trigger) items.
  for (std::vector <HcalElectronicsId>::iterator eid = AllElIds.begin();
       eid != AllElIds.end();
       eid++) {

    //Get the HcalDetId from the HcalElectronicsId
    detid_ = readoutMap_->lookup(*eid);
    // NULL if illegal; ignore
    if (!detid_.null()) {
      if (detid_.det()!=4) continue; //not Hcal
      if (detid_.subdetId()!=HcalBarrel &&
	  detid_.subdetId()!=HcalEndcap &&
	  detid_.subdetId()!=HcalOuter  &&
	  detid_.subdetId()!=HcalForward) continue;

      itsdcc    =(uint32_t) eid->dccid(); 
      itsspigot =(uint32_t) eid->spigot();
      itshtrchan=(uint32_t) eid->htrChanId();
      hcaldetid_ = HcalDetId(detid_);

      if (dfMon_)
	dfMon_->stashHDI(dfMon_->hashup(itsdcc,itsspigot,itshtrchan),
			 hcaldetid_);
    } // if (!detid_.null()) 
  } 
  
  //get conditions
  c.get<HcalDbRecord>().get(conditions_);
  
  // fill reference pedestals with database values
  if (pedMon_!=NULL)
    pedMon_->fillDBValues(*conditions_);
  //if (deadMon_!=NULL)
  //  deadMon_->createMaps(*conditions_);
  if (hotMon_!=NULL)
    hotMon_->createMaps(*conditions_);

  // get channel quality
  edm::ESHandle<HcalChannelQuality> p;
  c.get<HcalChannelQualityRcd>().get(p);
  chanquality_= new HcalChannelQuality(*p.product());

  return;

}

//--------------------------------------------------------
void HcalMonitorModule::beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
     const edm::EventSetup& context) 
{
  /* Don't start a new luminosity block if it is less than the current value
     when running online.  This avoids the problem of getting events
     from mis-ordered lumi blocks, which screws up our lumi block 
     monitoring.
  */
  if (Online_ && lumiSeg.luminosityBlock()<ilumisec)
    return;

  // Otherwise, run normal startups
  ilumisec = lumiSeg.luminosityBlock();
  if(digiMon_!=0)   {  digiMon_->beginLuminosityBlock(ilumisec);}
  if(dfMon_!=0)     {  dfMon_->beginLuminosityBlock(ilumisec);}
  if(diTask_!=0)    {  diTask_->beginLuminosityBlock(ilumisec);}
  if(pedMon_!=0)    {  pedMon_->beginLuminosityBlock(ilumisec);}
  if(ledMon_!=0)    {  ledMon_->beginLuminosityBlock(ilumisec);}
  if(laserMon_!=0)  {  laserMon_->beginLuminosityBlock(ilumisec);}
  if(hotMon_!=0)    {  hotMon_->beginLuminosityBlock(ilumisec);}
  if(deadMon_!=0)   {  deadMon_->beginLuminosityBlock(ilumisec);}
  if(mtccMon_!=0)   {  mtccMon_->beginLuminosityBlock(ilumisec);}
  if(rhMon_!=0)     {  rhMon_->beginLuminosityBlock(ilumisec);}
  if (zdcMon_!=0)   {  zdcMon_->beginLuminosityBlock(ilumisec);}
  if (beamMon_!=0)  {  beamMon_->beginLuminosityBlock(ilumisec);}
  if (tpMon_!=0)    {  tpMon_->beginLuminosityBlock(ilumisec);}
  if (nzsMon_!=0)    {  nzsMon_->beginLuminosityBlock(ilumisec);}

  //////////////////////////////////////////////
  if(detDiagPed_!=0){  detDiagPed_->beginLuminosityBlock(ilumisec);}
  if(detDiagLed_!=0){  detDiagLed_->beginLuminosityBlock(ilumisec);}
  if(detDiagLas_!=0){  detDiagLas_->beginLuminosityBlock(ilumisec);}
  if(detDiagNoise_!=0){  detDiagNoise_->beginLuminosityBlock(ilumisec);}
  if(detDiagTiming_!=0){  detDiagTiming_->beginLuminosityBlock(ilumisec);}
  /////////////////////////////////////////////
}


//--------------------------------------------------------
void HcalMonitorModule::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
					   const edm::EventSetup& context) {
  
  // In online running, don't process events that occur before current luminosity block
  if (Online_ && lumiSeg.luminosityBlock()<ilumisec)
    return; 

  // Call these every luminosity block
  if(digiMon_!=0)   {  digiMon_->endLuminosityBlock();}
  if(dfMon_!=0)     {  dfMon_->endLuminosityBlock();}
  if(diTask_!=0)    {  diTask_->endLuminosityBlock();}
  if(pedMon_!=0)    {  pedMon_->endLuminosityBlock();}
  if(ledMon_!=0)    {  ledMon_->endLuminosityBlock();}
  if(laserMon_!=0)  {  laserMon_->endLuminosityBlock();}
  if(hotMon_!=0)    {  hotMon_->endLuminosityBlock();}
  if(deadMon_!=0)   {  deadMon_->endLuminosityBlock();}
  if(mtccMon_!=0)   {  mtccMon_->endLuminosityBlock();}
  if(rhMon_!=0)     {  rhMon_->endLuminosityBlock();}
  if (zdcMon_!=0)   {  zdcMon_->endLuminosityBlock();}
  if (beamMon_!=0)  {  beamMon_->endLuminosityBlock();}
  if (tpMon_!=0)    {  tpMon_->endLuminosityBlock();}
  if (nzsMon_!=0)    {  nzsMon_->endLuminosityBlock();}

  // Call these only if prescale set
  if (prescaleLS_>-1 && !prescale())
    {
    }
  return;
}

//--------------------------------------------------------
void HcalMonitorModule::endRun(const edm::Run& r, const edm::EventSetup& context)
{
  if (debug_>0)  
    std::cout <<"HcalMonitorModule::endRun(...) ievt = "<<ievt_<<endl;

  // We should probably call functions here to make a final fill of histograms at the end of a run.
  // This will ensure that all appropriate histograms get filled at least once.
  

  // Do final pedestal histogram filling
  if (pedMon_!=NULL)
    pedMon_->fillPedestalHistos();

  // These should be unnecessary; call them just in case, so that
  // we're sure we get at least one fill per run
  if(digiMon_!=0)   {  digiMon_->endLuminosityBlock();}
  if(dfMon_!=0)     {  dfMon_->endLuminosityBlock();}
  if(diTask_!=0)    {  diTask_->endLuminosityBlock();}
  if(pedMon_!=0)    {  pedMon_->endLuminosityBlock();}
  if(ledMon_!=0)    {  ledMon_->endLuminosityBlock();}
  if(laserMon_!=0)  {  laserMon_->endLuminosityBlock();}
  if(hotMon_!=0)    {  hotMon_->endLuminosityBlock();}
  if(deadMon_!=0)   {  deadMon_->endLuminosityBlock();}
  if(mtccMon_!=0)   {  mtccMon_->endLuminosityBlock();}
  if(rhMon_!=0)     {  rhMon_->endLuminosityBlock();}
  if (zdcMon_!=0)   {  zdcMon_->endLuminosityBlock();}
  if (beamMon_!=0)  {  beamMon_->endLuminosityBlock();}
  if (tpMon_!=0)    {  tpMon_->endLuminosityBlock();}
  if (nzsMon_!=0)    {  nzsMon_->endLuminosityBlock();} 

  if (dfMon_!=NULL) dfMon_->UpdateMEs(); // should no longer be necessary; called through endLuminoisityBlock()
  
  /////////////////////////////////////////////////////
  if(detDiagLas_!=NULL) detDiagLas_->fillHistos();
  /////////////////////////////////////////////////////

  return;
}


//--------------------------------------------------------
void HcalMonitorModule::endJob(void) {

  if ( dbe_ != NULL ){
    meStatus_  = dbe_->get(rootFolder_+"DQM Job Status/STATUS");
  }
  
  if ( meStatus_ ) meStatus_->Fill(2);


  return; // All of the rest of the endjob stuff (filling db, etc.) should be done in the client, right?

  if(rhMon_!=NULL)         rhMon_->done();
  if(digiMon_!=NULL)       digiMon_->done();
  if(dfMon_!=NULL)         dfMon_->done();
  if(diTask_!=NULL)        diTask_->done();
  if(pedMon_!=NULL)        pedMon_->done();
  if(ledMon_!=NULL)        ledMon_->done();
  if(laserMon_!=NULL)      laserMon_->done();
  if(hotMon_!=NULL)        hotMon_->done();
  if(deadMon_!=NULL)       deadMon_->done();
  if(mtccMon_!=NULL)       mtccMon_->done();
  if (tpMon_!=NULL)        tpMon_->done();
  if (ctMon_!=NULL)        ctMon_->done();
  if (beamMon_!=NULL)      beamMon_->done();
  if (zdcMon_!=NULL)       zdcMon_->done();
  if (expertMon_!=NULL)    expertMon_->done();
  if (eeusMon_!=NULL)      eeusMon_->done();
  if (nzsMon_!=NULL)       nzsMon_->done();
  if(tempAnalysis_!=NULL)  tempAnalysis_->done();
  ////////////////////////////////////////////////////
  if(detDiagPed_!=NULL)    detDiagPed_->done();
  if(detDiagLed_!=NULL)    detDiagLed_->done();
  if(detDiagLas_!=NULL)    detDiagLas_->done();
  if(detDiagNoise_!=NULL)  detDiagNoise_->done();
  if(detDiagNoise_!=NULL)  detDiagTiming_->done();
  /////////////////////////////////////////////////////

  return;
}

//--------------------------------------------------------
void HcalMonitorModule::reset(){

  if(rhMon_!=NULL)   rhMon_->reset();
  if(digiMon_!=NULL) digiMon_->reset();
  if(dfMon_!=NULL)   dfMon_->reset();
  if(diTask_!=NULL)  diTask_->reset();
  if(pedMon_!=NULL)  pedMon_->reset();
  if(ledMon_!=NULL)  ledMon_->reset();
  if(laserMon_!=NULL)  laserMon_->reset();
  if(hotMon_!=NULL)  hotMon_->reset();
  if(deadMon_!=NULL)  deadMon_->reset();
  if(mtccMon_!=NULL)   mtccMon_->reset();
  if(tempAnalysis_!=NULL) tempAnalysis_->reset();
  if(tpMon_!=NULL) tpMon_->reset();
  if(ctMon_!=NULL) ctMon_->reset();
  if (zdcMon_!=NULL) zdcMon_->reset();
  if(beamMon_!=NULL) beamMon_->reset();
  if(expertMon_!=NULL) expertMon_->reset();
  if(eeusMon_!=NULL) eeusMon_->reset();
  if(nzsMon_!=NULL) nzsMon_->reset();
  ////////////////////////////////////////////////////
  if(detDiagPed_!=0) detDiagPed_->reset();
  if(detDiagLed_!=0) detDiagLed_->reset();
  if(detDiagLas_!=0) detDiagLas_->reset();
  if(detDiagNoise_!=0) detDiagNoise_->reset();
  if(detDiagTiming_!=0) detDiagTiming_->reset();
  /////////////////////////////////////////////////////

}

//--------------------------------------------------------
void HcalMonitorModule::analyze(const edm::Event& e, const edm::EventSetup& eventSetup){
  
  // environment datamembers
  irun_     = e.id().run();

  ievent_   = e.id().event();
  itime_    = e.time().value();
  


  if (Online_ && e.luminosityBlock()<ilumisec)
    return;

  if (debug_>1) std::cout << "HcalMonitorModule: evts: "<< nevt_ << ", run: " << irun_ << ", LS: " << e.luminosityBlock() << ", evt: " << ievent_ << ", time: " << itime_ << std::endl <<"\t counter = "<<ievt_pre_<<"\t total count = "<<ievt_<<endl; 

  if ( meStatus_ ) meStatus_->Fill(1);
  meLatency_->Fill(psTime_.elapsedTime);

  ////////////////////////////////////////////////////
  if(detDiagPed_!=0) detDiagPed_->processEvent(e,eventSetup,*conditions_);
  if(detDiagLed_!=0) detDiagLed_->processEvent(e,eventSetup,*conditions_);
  if(detDiagLas_!=0) detDiagLas_->processEvent(e,eventSetup,*conditions_);
  if(detDiagNoise_!=0)  detDiagNoise_->processEvent(e,eventSetup,*conditions_);
  if(detDiagTiming_!=0) detDiagTiming_->processEvent(e,eventSetup,*conditions_);
  /////////////////////////////////////////////////////

  //TB-based HcalMonitorSelector is deprecated in favor of new calibtype selection
  
  ///See if our products are in the event...
  bool rawOK_    = true;
  bool digiOK_   = true;
  bool rechitOK_ = true;
  bool zdchitOK_ = true;
  bool trigOK_   = false;
  bool tpdOK_    = true;
  bool calotowerOK_ = true;
  bool laserOK_  = true;
  bool hltResOK_ = true;

  // disable attempt to grab trigger/sim TP digis if trigger primitive monitor not enabled
  if (tpMon_==NULL) tpdOK_=false;

  // try to get raw data and unpacker report
  edm::Handle<FEDRawDataCollection> rawraw;  

  // Trying new getByLabel
  if (!(e.getByLabel(FEDRawDataCollection_,rawraw)))
    {
      rawOK_=false;
      LogWarning("HcalMonitorModule")<<" source not available";
    }
  if (rawOK_&&!rawraw.isValid()) {
    rawOK_=false;
  }


  edm::Handle<HcalUnpackerReport> report;  
  if (!(e.getByLabel(inputLabelDigi_,report)))
    {
      rawOK_=false;
      LogWarning("HcalMonitorModule")<<" Digi Collection "<<inputLabelDigi_<<" not available";
    }
  if (rawOK_ && !report.isValid()) {
    rawOK_=false;
  }
  if (rawOK_)
    {
      if(!fedsListed_){
	const std::vector<int> feds =  (*report).getFedsUnpacked();    
	for(unsigned int f=0; f<feds.size(); f++){
	  meFEDS_->Fill(feds[f]);    
	}
	fedsListed_ = true;
	fedss = feds; //Assign to a non-const holder
      }
    }
  if (rawOK_==true) ++ievt_rawdata_;

  // copy of Bryan Dahmes' calibration filter
  int calibType=-1;
  int dccBCN=-1;
  if (rawOK_==true)
  {
    // checking FEDs for calibration information
    int numEmptyFEDs = 0 ;
    std::vector<int> calibTypeCounter(8,0) ;
    for( int i = FEDNumbering::MINHCALFEDID; i <= FEDNumbering::MAXHCALFEDID; i++) {
      const FEDRawData& fedData = rawraw->FEDData(i) ;

      if ( fedData.size() < 24 ) numEmptyFEDs++ ;
      if ( fedData.size() < 24 ) continue;
      int value = ((const HcalDCCHeader*)(fedData.data()))->getCalibType() ;
      calibTypeCounter.at(value)++ ; // increment the counter for this calib type
      
      // Temporary for Pawel -- get BCN #101
      const HcalDCCHeader* dccHeader=(const HcalDCCHeader*)(fedData.data());
      dccBCN = dccHeader->getBunchId();
      if (dccBCN!=101)
	{
	  //if (i-FEDNumbering::MINHCALFEDID==0 && debug_>-1) std::cout <<"\t\tHUZZAH! ievent = "<< ievent_<<" BCN = "<<dccBCN<<endl;
	}
      else
	{
	  // std::cout <<"BCN = "<<dccBCN<<endl;
	  //return;
	}

    }
    
    int maxCount = 0;
    int numberOfFEDIds = FEDNumbering::MAXHCALFEDID  - FEDNumbering::MINHCALFEDID + 1 ;
    for (unsigned int i=0; i<calibTypeCounter.size(); i++) {
      if ( calibTypeCounter.at(i) > maxCount )
	{ calibType = i ; maxCount = calibTypeCounter.at(i) ; }
      if ( maxCount == numberOfFEDIds ) break ;
    }
    
    if ( maxCount != (numberOfFEDIds-numEmptyFEDs) )
      edm::LogWarning("HcalCalibTypeFilter") << "Conflicting calibration types found.  Assigning type "
					     << calibType ;
    LogDebug("HcalCalibTypeFilter") << "Calibration type is: " << calibType ;
  } // calibType assignment loop

  // skip this event if we're prescaling...
  ++ievt_;
  if(prescaleEvt_>0 && prescale()) return;
  if (rawOK_==true) ++ievt_rawdata_;


  // try to get digis
  edm::Handle<HBHEDigiCollection> hbhe_digi;
  edm::Handle<HODigiCollection> ho_digi;
  edm::Handle<HFDigiCollection> hf_digi;
  edm::Handle<ZDCDigiCollection> zdc_digi;
  edm::Handle<HcalTrigPrimDigiCollection> tp_digi;
  edm::Handle<HcalTrigPrimDigiCollection> emultp_digi;
  edm::Handle<HcalLaserDigi> laser_digi;

  if (!(e.getByLabel(inputLabelDigi_,hbhe_digi)))
    digiOK_=false;

  if (digiOK_&&!hbhe_digi.isValid()) {
    digiOK_=false;
    LogWarning("HcalMonitorModule")<< inputLabelDigi_<<" hbhe_digi not available";
  }

  if (!(e.getByLabel(inputLabelDigi_,hf_digi)))
    {
      digiOK_=false;
      LogWarning("HcalMonitorModule")<< inputLabelDigi_<<" hf_digi not available";
    }

  if (digiOK_&&!hf_digi.isValid()) {
    digiOK_=false;
  }

  if (!(e.getByLabel(inputLabelDigi_,ho_digi)))
    {
      digiOK_=false;
      LogWarning("HcalMonitorModule")<< inputLabelDigi_<<" ho_digi not available";
    }
  if (digiOK_&&!ho_digi.isValid()) {
    digiOK_=false;
  }
  
  if (!(e.getByLabel(inputLabelDigi_,zdc_digi)))
    {
      digiOK_=false;
      if (debug_>1) std::cout <<"<HcalMonitorModule> COULDN'T GET ZDC DIGI"<<endl;
      //LogWarning("HcalMonitorModule")<< inputLabelDigi_<<" zdc_digi not available";
    }
  if (digiOK_&&!zdc_digi.isValid()) {
    digiOK_=false;
    if (debug_>1) std::cout <<"<HcalMonitorModule> DIGI OK FAILED FOR ZDC"<<endl;
  }
  
  if (digiOK_) ++ievt_digi_;

  // check which Subdetectors are on by seeing which are reading out FED data
  // Assume subdetectors aren't present, unless we explicitly find otherwise

  if (digiOK_ && rawOK_)
    { 
      if ((checkHB_ && HBpresent_==0) ||
	  (checkHE_ && HEpresent_==0) ||
	  (checkHO_ && HOpresent_==0) ||
	  (checkHF_ && HFpresent_==0) )
	  
	
	CheckSubdetectorStatus(*rawraw,*report,*readoutMap_,*hbhe_digi, *ho_digi, *hf_digi);
    }
  else
    {
      // Is this the behavior we want?
      if (debug_>1)
	cout <<"<HcalMonitorModule::analyze>  digiOK or rawOK error.  Assuming all subdetectors present."<<endl;
      HBpresent_=1;
      HEpresent_=1;
      HOpresent_=1;
      HFpresent_=1;
    }

  // Case where all subdetectors have no raw data -- skip event
  if ((checkHB_ && HBpresent_==0) &&
      (checkHE_ && HEpresent_==0) &&
      (checkHO_ && HOpresent_==0) &&
      (checkHF_ && HFpresent_==0))
    {
      if (debug_>1) std::cout <<"<HcalMonitorModule::analyze>  No HCAL raw data found for event "<<ievt_<<endl;
      return;
    }

  // Only try to get trigger info if tpmon enabled
  if (tpdOK_)
    {
      if (!(e.getByLabel(inputLabelDigi_,tp_digi)))
	{
	  tpdOK_=false;
	  LogWarning("HcalMonitorModule")<< inputLabelDigi_<<" tp_digi not available"; 
	}

      if (tpdOK_ && !tp_digi.isValid()) {
	tpdOK_=false;
      }
      //Emulator
      
      if (!(e.getByLabel(inputLabelEmulDigi_,emultp_digi)))
	{
	  tpdOK_=false;
	  LogWarning("HcalMonitorModule")<< inputLabelEmulDigi_<<" emultp_digi not available";
	}
      
      if (tpdOK_ && !emultp_digi.isValid()) {
	tpdOK_=false;
      }
    }

  if (!(e.getByLabel(inputLabelLaser_,laser_digi)))
    {laserOK_=false;}
  if (laserOK_&&!laser_digi.isValid()) {
    laserOK_=false;
  }

  // try to get rechits
  edm::Handle<HBHERecHitCollection> hb_hits;
  edm::Handle<HORecHitCollection> ho_hits;
  edm::Handle<HFRecHitCollection> hf_hits;
  edm::Handle<ZDCRecHitCollection> zdc_hits;
  edm::Handle<CaloTowerCollection> calotowers;

  if (!(e.getByLabel(inputLabelRecHitHBHE_,hb_hits)))
    {
      rechitOK_=false;
      //if (debug_>0)
	LogWarning("HcalMonitorModule")<< inputLabelRecHitHBHE_<<" not available"; 
    }
 
  if (rechitOK_&&!hb_hits.isValid()) {
    rechitOK_ = false;
  }
  if (!(e.getByLabel(inputLabelRecHitHO_,ho_hits)))
    {
      rechitOK_=false;
      //if (debug_>0) 
	LogWarning("HcalMonitorModule")<< inputLabelRecHitHO_<<" not available"; 
    }
  if (rechitOK_&&!ho_hits.isValid()) {
    rechitOK_ = false;
  }
  if (!(e.getByLabel(inputLabelRecHitHF_,hf_hits)))
    {
      rechitOK_=false;
      //if (debug_>0) 
	LogWarning("HcalMonitorModule")<< inputLabelRecHitHF_<<" not available"; 
    }
  if (rechitOK_&&!hf_hits.isValid()) {
    rechitOK_ = false;
  }
  
  if (rechitOK_) ++ievt_rechit_;

  if (!(e.getByLabel(inputLabelRecHitZDC_,zdc_hits)))
    {
      zdchitOK_=false;
      // ZDC Warnings should be suppressed unless debugging is on (since we don't yet normally run zdcreco)
      if (debug_>0) 
	LogWarning("HcalMonitorModule")<< inputLabelRecHitZDC_<<" not available"; 
    }
  if (zdchitOK_&&!zdc_hits.isValid()) 
    {
      zdchitOK_ = false;
    }
  
  // try to get calotowers 
  if (ctMon_!=NULL)
    {
      if (!(e.getByLabel(inputLabelCaloTower_,calotowers)))
	{
	  calotowerOK_=false;
	  if (debug_>0) LogWarning("HcalMonitorModule")<< inputLabelCaloTower_<<" not available"; 
	}
      if(calotowerOK_&&!calotowers.isValid()){
	calotowerOK_=false;
      }
    }
  else
    calotowerOK_=false;
  
  //try to get HLT output if nzs mon is enabled
  edm::Handle<edm::TriggerResults> hltRes;
  if (nzsMon_!=NULL)
    {
      if (!(e.getByLabel(inputHLTresultsTag_,hltRes)))
	{
	  hltResOK_=false;
	  if (debug_>0) LogWarning("HcalMonitorModule")<< inputHLTresultsTag_<<" not available"; 
	}
      if(hltResOK_&&!hltRes.isValid()){
	hltResOK_=false;
      }
    }
  else
    hltResOK_=false;
  /////

  // Run the configured tasks, protect against missing products

  meIEVTALL_->Fill(ievt_);
  meIEVTRAW_->Fill(ievt_rawdata_);
  meIEVTDIGI_->Fill(ievt_digi_);
  meIEVTRECHIT_->Fill(ievt_rechit_);

  // Data Format monitor task
  
  if (showTiming_)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  if (zdchitOK_ && digiOK_) // make a separate boolean just for ZDC digis?
    {
      if (zdcMon_ !=NULL) zdcMon_->processEvent(*zdc_digi,*zdc_hits);
    }
  if (showTiming_)
    {
      cpu_timer.stop();
      if (zdcMon_ !=NULL) std::cout <<"TIMER:: ZDC MONITOR ->"<<cpu_timer.cpuTime()<<endl;
      cpu_timer.reset(); cpu_timer.start();
    }

  if((dfMon_ != NULL) && rawOK_) 
    {
      dfMon_->processEvent(*rawraw,*report,*readoutMap_,calibType);
    }
  if (showTiming_)
    {
      cpu_timer.stop();
      if (dfMon_ !=NULL) std::cout <<"TIMER:: DATAFORMAT MONITOR ->"<<cpu_timer.cpuTime()<<endl;
      cpu_timer.reset(); cpu_timer.start();
    }

  if ((diTask_ != NULL) && rawOK_)
    {
      diTask_->processEvent(*rawraw,*report,*readoutMap_);
    }
  if (showTiming_)
    {
      cpu_timer.stop();
      if (diTask_ !=NULL) std::cout <<"TIMER:: DATA INTEGRITY TASK ->"<<cpu_timer.cpuTime()<<endl;
      cpu_timer.reset(); cpu_timer.start();
    }

  // Digi monitor task
  if((digiMon_!=NULL)  && digiOK_ && report.isValid())
    {
      digiMon_->setSubDetectors(HBpresent_, HEpresent_, HOpresent_, HFpresent_ );
      digiMon_->processEvent(*hbhe_digi,*ho_digi,*hf_digi,
			     calibType,*conditions_,*report);
    }
  if (showTiming_)
    {
      cpu_timer.stop();
      if (digiMon_ != NULL) std::cout <<"TIMER:: DIGI MONITOR ->"<<cpu_timer.cpuTime()<<endl;
      cpu_timer.reset(); cpu_timer.start();
    }
  // Pedestal monitor task
  if((pedMon_!=NULL) && digiOK_) 
    {
      pedMon_->processEvent(*hbhe_digi,*ho_digi,*hf_digi,calibType,*conditions_);
    }
  if (showTiming_)
    {
      cpu_timer.stop();
      if (pedMon_!=NULL) std::cout <<"TIMER:: PEDESTAL MONITOR ->"<<cpu_timer.cpuTime()<<endl;
      cpu_timer.reset(); cpu_timer.start();
    }

  // LED monitor task
  if(ledMon_!=NULL && digiOK_)
    {
      ledMon_->processEvent(*hbhe_digi,*ho_digi,*hf_digi,*conditions_);
    }
  if (showTiming_)
    {
      cpu_timer.stop();
      if (ledMon_!=NULL) std::cout <<"TIMER:: LED MONITOR ->"<<cpu_timer.cpuTime()<<endl;
      cpu_timer.reset(); cpu_timer.start();
    }

  // Laser monitor task
  if((laserMon_!=NULL)  && digiOK_ && laserOK_)
    {
      laserMon_->processEvent(*hbhe_digi,*ho_digi,*hf_digi,*laser_digi,*conditions_,calibType);
    }
  if (showTiming_)
    {
      cpu_timer.stop();
      if (laserMon_!=NULL) std::cout <<"TIMER:: LASER MONITOR ->"<<cpu_timer.cpuTime()<<endl;
      cpu_timer.reset(); cpu_timer.start();
    }

  // Rec Hit monitor task
  if((rhMon_ != NULL)  && rechitOK_) 
    {
      rhMon_->processEvent(*hb_hits,*ho_hits,*hf_hits,calibType, dccBCN, e);
    }
  if (showTiming_)
    {
      cpu_timer.stop();
      if (rhMon_!=NULL) std::cout <<"TIMER:: RECHIT MONITOR ->"<<cpu_timer.cpuTime()<<endl;
      cpu_timer.reset(); cpu_timer.start();
    }
  
  // Beam Monitor task
  if ((beamMon_ != NULL)  && rechitOK_)
    {
      beamMon_->processEvent(*hb_hits,*ho_hits,*hf_hits,*hf_digi,calibType,e.bunchCrossing());
    }
  if (showTiming_)
    {
      cpu_timer.stop();
      if (beamMon_!=NULL) std::cout <<"TIMER:: BEAM MONITOR ->"<<cpu_timer.cpuTime( )<<endl;
      cpu_timer.reset(); cpu_timer.start();
    }

  // Hot Cell monitor task
  if((hotMon_ != NULL)  && rechitOK_) 
    {
      hotMon_->processEvent(*hb_hits,*ho_hits,*hf_hits, 
			    *conditions_, calibType);
      //hotMon_->setSubDetectors(HBpresent_,HEpresent_, HOpresent_, HFpresent_);
    }
  if (showTiming_)
    {
      cpu_timer.stop();
      if (hotMon_!=NULL) std::cout <<"TIMER:: HOTCELL MONITOR ->"<<cpu_timer.cpuTime()<<endl;
      cpu_timer.reset(); cpu_timer.start();
    }
  // Dead Cell monitor task -- may end up using both rec hits and digis?
  if((deadMon_ != NULL) && rechitOK_ && digiOK_) 
    {
      //deadMon_->setSubDetectors(HBpresent_,HEpresent_, HOpresent_, HFpresent_);
      deadMon_->processEvent(*hb_hits,*ho_hits,*hf_hits,
			     *hbhe_digi,*ho_digi,*hf_digi,calibType);
		
    }
  if (showTiming_)
    {
      cpu_timer.stop();
      if (deadMon_!=NULL) std::cout <<"TIMER:: DEADCELL MONITOR ->"<<cpu_timer.cpuTime()<<endl;
      cpu_timer.reset(); cpu_timer.start();
    }

  // CalotowerMonitor
  if ((ctMon_ !=NULL) )
    {
      ctMon_->processEvent(*calotowers);
    }

  if (showTiming_)
    {
      cpu_timer.stop();
      if (ctMon_ !=NULL) std::cout <<"TIMER:: CALOTOWER MONITOR ->"<<cpu_timer.cpuTime()<<endl;
      cpu_timer.reset(); cpu_timer.start();
    }


  // Trigger Primitive monitor task -- may end up using both rec hits and digis?
  if((tpMon_ != NULL) && rawOK_ && rechitOK_ && digiOK_ && tpdOK_) 
    {
      tpMon_->processEvent(*hb_hits,*ho_hits,*hf_hits,
			   *hbhe_digi,*ho_digi,*hf_digi,
			   *tp_digi, *emultp_digi, *rawraw, *readoutMap_,
			   calibType);			     
    }

  if (showTiming_)
    {
      cpu_timer.stop();
      if (tpMon_!=NULL) std::cout <<"TIMER:: TRIGGERPRIMITIVE MONITOR ->"<<cpu_timer.cpuTime()<<endl;
    }

  // Expert monitor plots
  if (expertMon_ != NULL) 
    {
      expertMon_->processEvent(*hb_hits,*ho_hits,*hf_hits,
			       *hbhe_digi,*ho_digi,*hf_digi,
			       *tp_digi,
			       *rawraw,*report,*readoutMap_);
    }
  if (showTiming_)
    {
      cpu_timer.stop();
      if (expertMon_!=NULL) std::cout <<"TIMER:: EXPERT MONITOR ->"<<cpu_timer.cpuTime()<<endl;
      cpu_timer.reset(); cpu_timer.start();
    }

  // Empty Event/Unsuppressed monitor plots
  if (eeusMon_ != NULL) 
    {
      eeusMon_->processEvent( *rawraw,*report,*readoutMap_);
    }
  if (showTiming_)
    {
      cpu_timer.stop();
      if (eeusMon_!=NULL) std::cout <<"TIMER:: EE/US MONITOR ->"<<cpu_timer.cpuTime()<<endl;
      cpu_timer.reset(); cpu_timer.start();
    }
  
  if (nzsMon_ != NULL && hltResOK_) 
    {
      nzsMon_->processEvent( *rawraw,*hltRes,e.bunchCrossing());
    }
  if (showTiming_)
    {
      cpu_timer.stop();
      if (nzsMon_!=NULL) std::cout <<"TIMER:: NZS MONITOR ->"<<cpu_timer.cpuTime()<<endl;
      cpu_timer.reset(); cpu_timer.start();
    }


  if(debug_>0 && ievt_%1000 == 0)
    std::cout << "HcalMonitorModule: processed " << ievt_ << " events" << std::endl;

  if(debug_>1)
    {
      std::cout << "HcalMonitorModule: processed " << ievt_ << " events" << std::endl;
      std::cout << "    RAW Data   ==> " << rawOK_<< std::endl;
      std::cout << "    Digis      ==> " << digiOK_<< std::endl;
      std::cout << "    RecHits    ==> " << rechitOK_<< std::endl;
      std::cout << "    TrigRec    ==> " << trigOK_<< std::endl;
      std::cout << "    TPdigis    ==> " << tpdOK_<< std::endl;    
      std::cout << "    CaloTower  ==> " << calotowerOK_ <<endl;
      std::cout << "    LaserDigis ==> " << laserOK_ << std::endl;
      std::cout << "    HLTresults ==> " << hltResOK_ << std::endl;
    }
  

  // Allow for periodic resets of histograms
  if (periodicReset_>0 && ievt_%periodicReset_==0 && ievt_>0)
    {
      if(digiMon_!=0)   {  digiMon_->periodicReset();}
      if(dfMon_!=0)     {  dfMon_->periodicReset();}
      if(diTask_!=0)    {  diTask_->periodicReset();}
      if(pedMon_!=0)    {  pedMon_->periodicReset();}
      if(ledMon_!=0)    {  ledMon_->periodicReset();}
      if(laserMon_!=0)  {  laserMon_->periodicReset();}
      if(hotMon_!=0)    {  hotMon_->periodicReset();}
      if(deadMon_!=0)   {  deadMon_->periodicReset();}
      if(mtccMon_!=0)   {  mtccMon_->periodicReset();}
      if(rhMon_!=0)     {  rhMon_->periodicReset();}
      if (zdcMon_!=0)   {  zdcMon_->periodicReset();}
      if (beamMon_!=0) { beamMon_->periodicReset();}
      if (nzsMon_!=0) { nzsMon_->periodicReset();}
      
      //////////////////////////////////////////////
      if(detDiagPed_!=0){  detDiagPed_->periodicReset();}
      if(detDiagLed_!=0){  detDiagLed_->periodicReset();}
      if(detDiagLas_!=0){  detDiagLas_->periodicReset();}
      if(detDiagNoise_!=0){  detDiagNoise_->periodicReset();}
      if(detDiagTiming_!=0){  detDiagTiming_->periodicReset();}
      /////////////////////////////////////////////
      
    }
  return;
}

//--------------------------------------------------------
bool HcalMonitorModule::prescale()
{
  ///Return true if this event should be skipped according to the prescale condition...
  ///    Accommodate a logical "OR" of the possible tests
  if (debug_>1) std::cout <<"HcalMonitorModule::prescale:  ievt = "<<ievt_<<endl;
  // If no prescales are set, return 'false'.  (This means that we should process the event.)
  if(prescaleEvt_<=0 && prescaleLS_<=0) return false;

  // Now check whether event should be kept.  Assume that it should not by default

  bool keepEvent=false;
  
  // Keep event if prescaleLS test is met or if prescaleEvt test is met
  if(prescaleLS_>0 && (ilumisec%prescaleLS_)==0) keepEvent = true; // check on ls prescale; 
  if (prescaleEvt_>0 && (ievt_%prescaleEvt_)==0) keepEvent = true; // 
  
  // if any criteria wants to keep the event, do so
  if (keepEvent) return false;  // event should be kept; don't apply prescale
  return true; // apply prescale by default

} // HcalMonitorModule::prescale(...)



// -------------------------------------------------
void HcalMonitorModule::CheckSubdetectorStatus(const FEDRawDataCollection& rawraw, 
					       const HcalUnpackerReport& report, 
					       const HcalElectronicsMap& emap,
					       const HBHEDigiCollection& hbhedigi,
					       const HODigiCollection& hodigi,
					       const HFDigiCollection& hfdigi
					       )
{
  vector<int> fedUnpackList;
  for (int i=FEDNumbering::MINHCALFEDID; 
       i<=FEDNumbering::MAXHCALFEDID; 
       i++) 
    {
      fedUnpackList.push_back(i);
    }

  for (vector<int>::const_iterator i=fedUnpackList.begin();
       i!=fedUnpackList.end(); 
       ++i) 
    {
      const FEDRawData& fed = rawraw.FEDData(*i);
      if (fed.size()<12) continue; // Was 16. How do such tiny events even get here?
      
      // get the DCC header
      const HcalDCCHeader* dccHeader=(const HcalDCCHeader*)(fed.data());
      if (!dccHeader) return;
      int dccid=dccHeader->getSourceId();
      // check for HF
      if (dccid>717 && dccid<724)
	{
	  if (HFpresent_==0 && hfdigi.size()>0)
	    {
	      HFpresent_ = 1;
	      meHF_->Fill(HFpresent_);
	    }
	  continue;
	}

      // check for HO
      if (dccid>723)
	{
	  if (HOpresent_==0 && hodigi.size()>0)
	    {
	      HOpresent_ = 1;
	      meHO_->Fill(HOpresent_);
	    }
	  continue;
	}
      
      // Looking at HB and HE is more complicated, since they're combined into HBHE
      // walk through the HTR data...
      HcalHTRData htr;  
      for (int spigot=0; spigot<HcalDCCHeader::SPIGOT_COUNT; spigot++) {    
	if (!dccHeader->getSpigotPresent(spigot)) continue;
	
	// Load the given decoder with the pointer and length from this spigot.
	dccHeader->getSpigotData(spigot,htr, fed.size()); 
	
	// check min length, correct wordcount, empty event, or total length if histo event.
	if (!htr.check()) continue;
	if (htr.isHistogramEvent()) continue;
	
	int firstFED =  FEDNumbering::MINHCALFEDID;
	
	// Tease out HB and HE, which share HTRs in HBHE
	for(int fchan=0; fchan<3; ++fchan) //0,1,2 are valid
	  {
	    for(int fib=1; fib<9; ++fib) //1...8 are valid
	      {
		HcalElectronicsId eid(fchan,fib,spigot,dccid-firstFED);
		eid.setHTR(htr.readoutVMECrateId(),
			   htr.htrSlot(),htr.htrTopBottom());
		
		DetId did=emap.lookup(eid);
		if (!did.null()) 
		  {
		    
		    switch (((HcalSubdetector)did.subdetId()))
		      {
		      case (HcalBarrel): 
			{
			  if (HBpresent_==0)
			    {
			      HBpresent_ = 1;
			      meHB_->Fill(HBpresent_);
			    }
			} break; // case (HcalBarrel)
		      case (HcalEndcap): 
			{
			  if (HEpresent_==0)
			    {
			      HEpresent_ = 1;
			      meHE_->Fill(HEpresent_);
			    }
			} break; // case (HcalEndcap)
		      case (HcalOuter): 
			{ // shouldn't reach these last two cases
			  if (HOpresent_==0)
			    {
			      {
				HOpresent_ = 1;
				meHO_->Fill(HOpresent_);
				return;
			      }
			    } 
			} break; // case (HcalOuter)
		      case (HcalForward): 
			{
			  if (HFpresent_==0)
			    {
			      meHF_->Fill(HFpresent_);
			      HFpresent_ = 1;
			    }
			} break; //case (HcalForward)
		      default: break;
		      } // switch ((HcalSubdetector...)
		  } // if (!did.null())
	      } // for (int fib=0;...)
	  } // for (int fchan = 0;...)
	
      } // for (int spigot=0;...)
    } //  for (vector<int>::const_iterator i=fedUnpackList.begin();
  return;
} // void HcalMonitorModule::CheckSubdetectorStatus(...)

#include "FWCore/Framework/interface/MakerMacros.h"
#include <DQM/HcalMonitorModule/src/HcalMonitorModule.h>
#include "DQMServices/Core/interface/DQMStore.h"

DEFINE_FWK_MODULE(HcalMonitorModule);
