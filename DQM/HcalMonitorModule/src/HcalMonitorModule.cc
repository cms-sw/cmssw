#include <DQM/HcalMonitorModule/src/HcalMonitorModule.h>

/*
 * \file HcalMonitorModule.cc
 * 
 * $Date: 2007/10/11 22:35:52 $
 * $Revision: 1.39 $
 * \author W Fisher
 *
*/

//--------------------------------------------------------
HcalMonitorModule::HcalMonitorModule(const edm::ParameterSet& ps)
  : DQMAnalyzer(ps){
  cout << endl;
  cout << " *** Hcal Monitor Module ***" << endl;
  cout << endl;

  inputLabelDigi_        = ps.getParameter<edm::InputTag>("digiLabel");
  inputLabelRecHitHBHE_  = ps.getParameter<edm::InputTag>("hbheRecHitLabel");
  inputLabelRecHitHF_    = ps.getParameter<edm::InputTag>("hfRecHitLabel");
  inputLabelRecHitHO_    = ps.getParameter<edm::InputTag>("hoRecHitLabel");

  meStatus_=0;  meRunType_=0;
  meEvtMask_=0; meFEDS_=0;
  fedsListed_ = false;

  evtSel_ = new HcalMonitorSelector(ps);
  digiMon_ = NULL;   dfMon_ = NULL; 
  rhMon_ = NULL;     pedMon_ = NULL; 
  ledMon_ = NULL;    mtccMon_ = NULL;
  hotMon_ = NULL;    tempAnalysis_ = NULL;
  deadMon_ = NULL;
  commisMon_ = NULL; tpMon_ = NULL;

  if ( ps.getUntrackedParameter<bool>("DataFormatMonitor", false) ) {
    if(debug_) cout << "HcalMonitorModule: DataFormat monitor flag is on...." << endl;
    dfMon_ = new HcalDataFormatMonitor();
    dfMon_->setup(ps, dbe_);
  }
  
  if ( ps.getUntrackedParameter<bool>("DigiMonitor", false) ) {
    if(debug_) cout << "HcalMonitorModule: Digi monitor flag is on...." << endl;
    digiMon_ = new HcalDigiMonitor();
    digiMon_->setup(ps, dbe_);
  }
  
  if ( ps.getUntrackedParameter<bool>("RecHitMonitor", false) ) {
    if(debug_) cout << "HcalMonitorModule: RecHit monitor flag is on...." << endl;
    rhMon_ = new HcalRecHitMonitor();
    rhMon_->setup(ps, dbe_);
  }
  
  if ( ps.getUntrackedParameter<bool>("TrigPrimMonitor", false) ) {
    if(debug_) cout << "HcalMonitorModule: TrigPrim monitor flag is on...." << endl;
    tpMon_ = new HcalTrigPrimMonitor();
    tpMon_->setup(ps, dbe_);
  }
  
  if ( ps.getUntrackedParameter<bool>("PedestalMonitor", false) ) {
    if(debug_) cout << "HcalMonitorModule: Pedestal monitor flag is on...." << endl;
    pedMon_ = new HcalPedestalMonitor();
    pedMon_->setup(ps, dbe_);
  }
  
  if ( ps.getUntrackedParameter<bool>("LEDMonitor", false) ) {
    if(debug_) cout << "HcalMonitorModule: LED monitor flag is on...." << endl;
    ledMon_ = new HcalLEDMonitor();
    ledMon_->setup(ps, dbe_);
  }
  
  if ( ps.getUntrackedParameter<bool>("MTCCMonitor", false) ) {
    if(debug_) cout << "HcalMonitorModule: MTCC monitor flag is on...." << endl;
    mtccMon_ = new HcalMTCCMonitor();
    mtccMon_->setup(ps, dbe_);
  }
  
  if ( ps.getUntrackedParameter<bool>("HotCellMonitor", false) ) {
    if(debug_) cout << "HcalMonitorModule: Hot Cell monitor flag is on...." << endl;
    hotMon_ = new HcalHotCellMonitor();
    hotMon_->setup(ps, dbe_);
  }
  
  if ( ps.getUntrackedParameter<bool>("DeadCellMonitor", false) ) {
    if(debug_ || 1>0) cout << "HcalMonitorModule: Dead Cell monitor flag is on...." << endl;
    deadMon_ = new HcalDeadCellMonitor();
    deadMon_->setup(ps, dbe_);
  }
  

  if ( ps.getUntrackedParameter<bool>("CommissioningMonitor", false) ) {
    if(debug_) cout << "HcalMonitorModule: Commissioning monitor flag is on...." << endl;
    commisMon_ = new HcalCommissioningMonitor();
    commisMon_->setup(ps, dbe_);
  }
  
  if ( ps.getUntrackedParameter<bool>("HcalAnalysis", false) ) {
    if(debug_) cout << "HcalMonitorModule: Hcal Analysis flag is on...." << endl;
    tempAnalysis_ = new HcalTemplateAnalysis();
    tempAnalysis_->setup(ps);
  }
  
}

//--------------------------------------------------------
HcalMonitorModule::~HcalMonitorModule(){
  
  if(debug_) printf("HcalMonitorModule: Destructor.....");
  
  if (dbe_){    
    if(digiMon_!=NULL) {  digiMon_->clearME();}
    if(dfMon_!=NULL)   {  dfMon_->clearME();}
    if(pedMon_!=NULL)  {  pedMon_->clearME();}
    if(ledMon_!=NULL)  {  ledMon_->clearME();}
    if(hotMon_!=NULL)  {  hotMon_->clearME();}
    if(deadMon_!=NULL) {  deadMon_->clearME();}
    if(commisMon_!=NULL) {  commisMon_->clearME();}
    if(mtccMon_!=NULL) {  mtccMon_->clearME();}
    if(rhMon_!=NULL)   {  rhMon_->clearME();}
    if(tpMon_!=NULL)   {  tpMon_->clearME();}
    
    dbe_->setCurrentFolder(rootFolder_);
    dbe_->removeContents();
  }
  
  if(digiMon_!=NULL) { delete digiMon_; digiMon_=NULL; }
  if(dfMon_!=NULL) { delete dfMon_; dfMon_=NULL; }
  if(pedMon_!=NULL) { delete pedMon_; pedMon_=NULL; }
  if(ledMon_!=NULL) { delete ledMon_; ledMon_=NULL; }
  if(hotMon_!=NULL) { delete hotMon_; hotMon_=NULL; }
  if(deadMon_!=NULL) { delete deadMon_; deadMon_=NULL; }
  if(commisMon_!=NULL) { delete commisMon_; commisMon_=NULL; }
  if(mtccMon_!=NULL) { delete mtccMon_; mtccMon_=NULL; }
  if(rhMon_!=NULL) { delete rhMon_; rhMon_=NULL; }
  if(tpMon_!=NULL) { delete tpMon_; tpMon_=NULL; }
  if(tempAnalysis_!=NULL) { delete tempAnalysis_; tempAnalysis_=NULL; }
  delete evtSel_; evtSel_ = NULL;

}

//--------------------------------------------------------
void HcalMonitorModule::beginJob(const edm::EventSetup& c){
  // call DQMAnalyzer in the beginning 
  DQMAnalyzer::beginJob(c);
  
  ievt_ = 0;
  
  if(debug_) cout << "HcalMonitorModule: begin job...." << endl;
  
  if ( dbe_ != NULL ){
    dbe_->setCurrentFolder(rootFolder_ );
    meStatus_  = dbe_->bookInt("STATUS");
    meRunType_ = dbe_->bookInt("RUN TYPE");
    meEvtMask_ = dbe_->bookInt("EVT MASK");
    meFEDS_    = dbe_->book1D("FEDs Unpacked","FEDs Unpacked",100,700,799);

    meStatus_->Fill(0);
    meRunType_->Fill(-1);
    meEvtMask_->Fill(-1);
  }

  // get the hcal mapping
  edm::ESHandle<HcalDbService> pSetup;
  c.get<HcalDbRecord>().get( pSetup );
  readoutMap_=pSetup->getHcalMapping();
  
  // get conditions
  c.get<HcalDbRecord>().get(conditions_);

  return;
}


//--------------------------------------------------------
void HcalMonitorModule::beginRun(const edm::Run& run, const edm::EventSetup& c) {
  // call DQMAnalyzer in the beginning 
  cout <<"HcalMonitorModule::beginRun"<<endl;
  DQMAnalyzer::beginRun(run, c);

  fedsListed_ = false;
  reset();
  cout <<"Finished beginRun"<<endl;
}

//--------------------------------------------------------
void HcalMonitorModule::beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
     const edm::EventSetup& context) {
  
  // call DQMAnalyzer in the beginning 
  DQMAnalyzer::beginLuminosityBlock(lumiSeg,context);
  // then do your thing
  if(actonLS_ && !prescale()){
    // do scheduled tasks...
  }
}


//--------------------------------------------------------
void HcalMonitorModule::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
					   const edm::EventSetup& context) {
  // do your thing here
  if(actonLS_ && !prescale()){
    // do scheduled tasks...
  }
  // call DQMAnalyzer at the end 
  DQMAnalyzer::endLuminosityBlock(lumiSeg,context);
}

//--------------------------------------------------------
void HcalMonitorModule::endRun(const edm::Run& r, const edm::EventSetup& context){
  // do your thing here
  
  cout <<"HcalMonitorModule::endRun"<<endl;
  // call DQMAnalyzer at the end
  DQMAnalyzer::endRun(r,context); 
  cout <<"Finished DQMAnalyzer::endRun"<<endl;
}


//--------------------------------------------------------
void HcalMonitorModule::endJob(void) {

  if(debug_) cout << "HcalMonitorModule: end job...." << endl;  
  cout << "HcalMonitorModule::endJob, analyzed " << ievt_ << " events" << endl;
  
  if ( meStatus_ ) meStatus_->Fill(2);

  if(rhMon_!=NULL) rhMon_->done();
  if(tpMon_!=NULL) tpMon_->done();
  if(digiMon_!=NULL) digiMon_->done();
  if(dfMon_!=NULL) dfMon_->done();
  if(pedMon_!=NULL) pedMon_->done();
  if(ledMon_!=NULL) ledMon_->done();
  if(hotMon_!=NULL) hotMon_->done();
  if(deadMon_!=NULL) deadMon_->done();
  if(commisMon_!=NULL) commisMon_->done();
  if(mtccMon_!=NULL) mtccMon_->done();
  if(tempAnalysis_!=NULL) tempAnalysis_->done();
  
  DQMAnalyzer::endJob();

  return;
}

//--------------------------------------------------------
void HcalMonitorModule::reset(){
  DQMAnalyzer::reset();

  if(debug_) cout << "HcalMonitorModule: reset...." << endl;

  if(rhMon_!=NULL)   rhMon_->reset();
  if(tpMon_!=NULL)   tpMon_->reset();
  if(digiMon_!=NULL) digiMon_->reset();
  if(dfMon_!=NULL)   dfMon_->reset();
  if(pedMon_!=NULL)  pedMon_->reset();
  if(ledMon_!=NULL)  ledMon_->reset();
  if(hotMon_!=NULL)  hotMon_->reset();
  if(deadMon_!=NULL)  deadMon_->reset();
  if(commisMon_!=NULL) commisMon_->reset();
  if(mtccMon_!=NULL)   mtccMon_->reset();
  if(tempAnalysis_!=NULL) tempAnalysis_->reset();
}

//--------------------------------------------------------
void HcalMonitorModule::analyze(const edm::Event& e, const edm::EventSetup& eventSetup){
  DQMAnalyzer::analyze(e,eventSetup);

  if(debug_) cout << "HcalMonitorModule: analyze...." << endl;

  // skip this event if we're prescaling...
  if(prescale()) return;

  // Do default setup...
  ievt_++;
  printf("Module Analyzing...\n");
  int evtMask=DO_HCAL_DIGIMON|DO_HCAL_DFMON|DO_HCAL_RECHITMON|DO_HCAL_PED_CALIBMON|DO_HCAL_LED_CALIBMON;

  //  int trigMask=0;
  if(mtccMon_==NULL){
    evtSel_->processEvent(e);
    evtMask = evtSel_->getEventMask();
    //    trigMask =  evtSel_->getTriggerMask();
  }
  if ( dbe_ ){ 
    meStatus_->Fill(1);
    meEvtMask_->Fill(evtMask);
  }
  
  ///See if our products are in the event...
  bool rawOK_    = true;
  bool digiOK_   = true;
  bool rechitOK_ = true;
  bool ltcOK_    = true;
  bool tpdOK_    = true;

  // try to get raw data and unpacker report
  edm::Handle<FEDRawDataCollection> rawraw;  
  try{e.getByType(rawraw);} catch(exception& ex){ rawOK_=false;};           
  edm::Handle<HcalUnpackerReport> report;  
  try{
    e.getByType(report);
    if(!fedsListed_){
      const std::vector<int> feds =  (*report).getFedsUnpacked();    
      for(unsigned int f=0; f<feds.size(); f++){
	meFEDS_->Fill(feds[f]);    
      }
      fedsListed_ = true;
    }
  } catch(exception& ex){ rawOK_=false;};
  
  // try to get digis
  edm::Handle<HBHEDigiCollection> hbhe_digi;
  edm::Handle<HODigiCollection> ho_digi;
  edm::Handle<HFDigiCollection> hf_digi;
  edm::Handle<HcalTrigPrimDigiCollection> tp_digi;
  try{e.getByLabel(inputLabelDigi_,hbhe_digi);} catch(exception& ex){digiOK_=false;};
  try{e.getByLabel(inputLabelDigi_,hf_digi);} catch(exception& ex){digiOK_=false;};
  try{e.getByLabel(inputLabelDigi_,ho_digi);} catch(exception& ex){digiOK_=false;};
  try{e.getByLabel(inputLabelDigi_,tp_digi);} catch(exception& ex){tpdOK_=false;};



  // try to get rechits
  edm::Handle<HBHERecHitCollection> hb_hits;
  edm::Handle<HORecHitCollection> ho_hits;
  edm::Handle<HFRecHitCollection> hf_hits;
  try{e.getByLabel(inputLabelRecHitHBHE_,hb_hits);} catch(exception& ex){rechitOK_ = false;}; 
  try{e.getByLabel(inputLabelRecHitHO_,ho_hits);} catch(exception& ex){rechitOK_ = false;}; 
  try{e.getByLabel(inputLabelRecHitHF_,hf_hits);} catch(exception& ex){rechitOK_ = false;}; 

  // try to get trigger info
  edm::Handle<LTCDigiCollection> ltc;
  try{ e.getByType(ltc); } catch(exception& ex){ltcOK_=false;};         


  /// Run the configured tasks, protect against missing products

  // Data Format monitor task
  if((dfMon_ != NULL) && (evtMask&DO_HCAL_DFMON) && rawOK_) 
    dfMon_->processEvent(*rawraw,*report,*readoutMap_);

  // Digi monitor task
  if((digiMon_!=NULL) && (evtMask&DO_HCAL_DIGIMON) && digiOK_) 
    digiMon_->processEvent(*hbhe_digi,*ho_digi,*hf_digi,*conditions_);

  // Pedestal monitor task
  if((pedMon_!=NULL) && (evtMask&DO_HCAL_PED_CALIBMON) && digiOK_) 
    pedMon_->processEvent(*hbhe_digi,*ho_digi,*hf_digi,*conditions_);

  // LED monitor task
  if((ledMon_!=NULL) && (evtMask&DO_HCAL_LED_CALIBMON) && digiOK_)
    ledMon_->processEvent(*hbhe_digi,*ho_digi,*hf_digi,*conditions_);
  
  // Rec Hit monitor task
  if((rhMon_ != NULL) && (evtMask&DO_HCAL_RECHITMON) && rechitOK_) 
    rhMon_->processEvent(*hb_hits,*ho_hits,*hf_hits);

  // Rec Hit monitor task
  if((tpMon_ != NULL) && (evtMask&DO_HCAL_RECHITMON) && rechitOK_ && digiOK_) 
    tpMon_->processEvent(*hb_hits,*ho_hits,*hf_hits,*tp_digi);

  // Hot Cell monitor task
  if((hotMon_ != NULL) && (evtMask&DO_HCAL_RECHITMON) && rechitOK_) 
    hotMon_->processEvent(*hb_hits,*ho_hits,*hf_hits);

  // Dead Cell monitor task -- may end up using both rec hits and digis?
  if((deadMon_ != NULL) && (evtMask&DO_HCAL_RECHITMON) && rechitOK_ && digiOK_) 
    {
      deadMon_->processEvent(*hb_hits,*ho_hits,*hf_hits,
			     *hbhe_digi,*ho_digi,*hf_digi,*conditions_);
			     
      //deadMon_->processEvent_digi(*hbhe_digi,*ho_digi,*hf_digi,*conditions_);
      //deadMon_->processEvent_hits(*hb_hits,*ho_hits,*hf_hits);
    }

  // Old MTCC monitor task
  if(mtccMon_ != NULL && digiOK_ && ltcOK_) mtccMon_->processEvent(*hbhe_digi,*ho_digi, *ltc,*conditions_);


  // Temporary or development tasks...
  if(commisMon_ != NULL && digiOK_ && ltcOK_ && rechitOK_) 
    commisMon_->processEvent(*hbhe_digi,*ho_digi, *hf_digi,
			      *hb_hits,*ho_hits,*hf_hits,
			      *ltc,*conditions_);
  
  if(tempAnalysis_ != NULL && digiOK_ && ltcOK_ && rechitOK_) 
    tempAnalysis_->processEvent(*hbhe_digi,*ho_digi, *hf_digi,
				 *hb_hits,*ho_hits,*hf_hits,
				 *ltc,*conditions_);



  if(ievt_%1000 == 0)
    cout << "HcalMonitorModule: processed " << ievt_ << " events" << endl;

  if(debug_){
    cout << "HcalMonitorModule: processed " << ievt_ << " events" << endl;
    cout << "    RAW Data==> " << rawOK_<< endl;
    cout << "    Digis   ==> " << digiOK_<< endl;
    cout << "    RecHits ==> " << rechitOK_<< endl;
    cout << "    LTCdigi ==> " << ltcOK_<< endl;
    cout << "    TPdigis ==> " << tpdOK_<< endl;    
  }

  return;
}

#include "FWCore/Framework/interface/MakerMacros.h"
#include <DQM/HcalMonitorModule/src/HcalMonitorModule.h>

DEFINE_FWK_MODULE(HcalMonitorModule);
