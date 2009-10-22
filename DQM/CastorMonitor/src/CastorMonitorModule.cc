#include <DQM/CastorMonitor/interface/CastorMonitorModule.h>
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"


//**************************************************************//
//***************** CastorMonitorModule       ******************//
//***************** Author: Dmytro Volyanskyy ******************//
//***************** Date  : 22.11.2008 (first version) *********// 
//**************************************************************//
///// Simple event filter which directs events to monitoring tasks: 
///// Access unpacked data from each event and pass them to monitoring tasks 


//==================================================================//
//======================= Constructor ==============================//
//==================================================================//
CastorMonitorModule::CastorMonitorModule(const edm::ParameterSet& ps){
 
   ////---- get steerable variables
  inputLabelDigi_        = ps.getParameter<edm::InputTag>("digiLabel");
  inputLabelRecHitCASTOR_  = ps.getParameter<edm::InputTag>("CastorRecHitLabel"); 
  fVerbosity = ps.getUntrackedParameter<int>("debug", 0);                        //-- show debug 
  showTiming_ = ps.getUntrackedParameter<bool>("showTiming", false);         //-- show CPU time 
  dump2database_   = ps.getUntrackedParameter<bool>("dump2database",false);  //-- dumps output to database file

  rootFile_ = ps.getUntrackedParameter<string>("rootFile","");
  cout << "Root File to display = " << rootFile_ << endl;
 
 if(fVerbosity>0) cout << "CastorMonitorModule Constructor (start)" << endl;

  ////---- initialize Run, LS, Event number and other parameters
  irun_=0; 
  ilumisec_=0; 
  ievent_=0; 
  itime_=0;
  actonLS_=false;

  meStatus_=0;  meRunType_=0;
  meEvtMask_=0; meFEDS_=0;
  meLatency_=0; meQuality_=0;
  fedsListed_ = false;

  RecHitMon_ = NULL;  RecHitMonValid_ = NULL;  
  PedMon_ = NULL; 
  LedMon_ = NULL;    


 ////---- get DQMStore service  
  dbe_ = Service<DQMStore>().operator->();

  ////---- initialise CastorMonitorSelector
  evtSel_ = new CastorMonitorSelector(ps);

 
 //---------------------- PedestalMonitor ----------------------// 
  if ( ps.getUntrackedParameter<bool>("PedestalMonitor", false) ) {
    if(fVerbosity>0) cout << "CastorMonitorModule: Pedestal monitor flag is on...." << endl;
    PedMon_ = new CastorPedestalMonitor();
    PedMon_->setup(ps, dbe_);
  }
 //------------------------------------------------------------//

 ////-------------------- RecHitMonitor ------------------------// 
  if ( ps.getUntrackedParameter<bool>("RecHitMonitor", false) ) {
    if(fVerbosity>0) cout << "CastorMonitorModule: RecHit monitor flag is on...." << endl;
    RecHitMon_ = new CastorRecHitMonitor();
    RecHitMon_->setup(ps, dbe_);
  }
 //-------------------------------------------------------------//
 
////-------------------- RecHitMonitorValid ------------------------// 
  if ( ps.getUntrackedParameter<bool>("RecHitMonitorValid", false) ) {
    if(fVerbosity>0) cout << "CastorMonitorModule: RecHitValid monitor flag is on...." << endl;
    RecHitMonValid_ = new CastorRecHitsValidation();
    RecHitMonValid_->setup(ps, dbe_);
  }
 //-------------------------------------------------------------//



  ////-------------------- LEDMonitor ------------------------// 
  if ( ps.getUntrackedParameter<bool>("LEDMonitor", false) ) {
    if(fVerbosity>0) cout << "CastorMonitorModule: LED monitor flag is on...." << endl;
    LedMon_ = new CastorLEDMonitor();
    LedMon_->setup(ps, dbe_);
  }
 //-------------------------------------------------------------//

   ////---- ADD OTHER MONITORS HERE !!!!!!!!!!!!
  
  ////---- get steerable variables
  prescaleEvt_ = ps.getUntrackedParameter<int>("diagnosticPrescaleEvt", -1);
  if(fVerbosity>0) cout << "===>CastorMonitor event prescale = " << prescaleEvt_ << " event(s)"<< endl;

  prescaleLS_ = ps.getUntrackedParameter<int>("diagnosticPrescaleLS", -1);
  if(fVerbosity>0) cout << "===>CastorMonitor lumi section prescale = " << prescaleLS_ << " lumi section(s)"<< endl;
  if (prescaleLS_>0) actonLS_=true;

  prescaleUpdate_ = ps.getUntrackedParameter<int>("diagnosticPrescaleUpdate", -1);
  if(fVerbosity>0) cout << "===>CastorMonitor update prescale = " << prescaleUpdate_ << " update(s)"<< endl;

  prescaleTime_ = ps.getUntrackedParameter<int>("diagnosticPrescaleTime", -1);
  if(fVerbosity>1) cout << "===>CastorMonitor time prescale = " << prescaleTime_ << " minute(s)"<< endl;

  ////---- base folder for the contents of this job
  string subsystemname = ps.getUntrackedParameter<string>("subSystemFolder", "Castor") ;
  if(fVerbosity>0) cout << "===>CastorMonitor name = " << subsystemname << endl;
  rootFolder_ = subsystemname + "/";
  
 if ( dbe_ != NULL ){
  dbe_->setCurrentFolder(rootFolder_);
   ////---- open the root file to display histograms 
  dbe_->open(rootFile_);
  }



  gettimeofday(&psTime_.updateTV,NULL);
  ////---- get time in milliseconds, convert to minutes
  psTime_.updateTime = (psTime_.updateTV.tv_sec*1000.0+psTime_.updateTV.tv_usec/1000.0);
  psTime_.updateTime /= 1000.0;
  psTime_.elapsedTime=0;
  psTime_.vetoTime=psTime_.updateTime;

 if(fVerbosity>0) cout << "CastorMonitorModule Constructor (end)" << endl;

}


//==================================================================//
//======================= Destructor ===============================//
//==================================================================//
CastorMonitorModule::~CastorMonitorModule(){
  
// if (dbe_){    
//   if(PedMon_!=NULL)     {  PedMon_->clearME();}
//   if(RecHitMon_!=NULL)  {  RecHitMon_->clearME();}
//   if(LedMon_!=NULL)     {  LedMon_->clearME();}
//   dbe_->setCurrentFolder(rootFolder_);
//   dbe_->removeContents();
// }
//
//  if(PedMon_!=NULL)    { delete PedMon_;   PedMon_=NULL;     }
//  if(RecHitMon_!=NULL) { delete RecHitMon_; RecHitMon_=NULL; }
//  if(LedMon_!=NULL)    { delete LedMon_;   LedMon_=NULL;     }
//  delete evtSel_; evtSel_ = NULL;

} 


//=================================================================//
//========================== beginJob =============================//
//================================================================//
void CastorMonitorModule::beginJob(const edm::EventSetup& c){
  ievt_ = 0;
  ievt_pre_=0;

  if(fVerbosity>0) cout << "CastorMonitorModule::beginJob (start)" << endl;

  if ( dbe_ != NULL ){
    dbe_->setCurrentFolder(rootFolder_+"DQM Job Status" );
    meStatus_  = dbe_->bookInt("STATUS");
    meRunType_ = dbe_->bookInt("RUN TYPE");
    meEvtMask_ = dbe_->bookInt("EVT MASK");
    meFEDS_    = dbe_->book1D("FEDs Unpacked","FEDs Unpacked",100,660,759);
    meCASTOR_ = dbe_->bookInt("CASTORpresent");
    ////---- process latency 
    meLatency_ = dbe_->book1D("Process Latency","Process Latency",2000,0,10);
    meQuality_ = dbe_->book1D("Quality Status","Quality Status",100,0,1);
    meStatus_->Fill(0);
    meRunType_->Fill(-1);
    meEvtMask_->Fill(-1);
    ////---- should fill with 0 to start
    meCASTOR_->Fill(0);
    
    }
  else{
  if(fVerbosity>0) cout << "CastorMonitorModule::beginJob - NO DQMStore service" << endl; 
 }
  /*
  ////  (NO "CastorDbRecord" record found in the EventSetup)
  ////---- get Castor Readout Map from the DB 
  edm::ESHandle<CastorDbService> pSetup;
  c.get<CastorDbRecord>().get( pSetup );
  readoutMap_=pSetup->getCastorMapping();
  DetId detid_;
  HcalCastorDetId CastorDetID_; 

  if(fVerbosity>0) cout << "CastorMonitorModule::beginJob 6" << endl; 
 
  ////---- build a map of readout hardware unit to calorimeter channel
  std::vector <CastorElectronicsId> AllElIds = readoutMap_->allElectronicsIdPrecision();
  int dccid;
  pair <int,int> dcc_spgt;
  ////---- by looping over all precision (non-trigger) items
  for (std::vector <CastorElectronicsId>::iterator eid = AllElIds.begin();
       eid != AllElIds.end(); eid++) 
  {
    ////---- get the HcalCastorDetId from the CastorElectronicsId
    detid_ = readoutMap_->lookup(*eid); 

  if(fVerbosity>0) cout << "CastorMonitorModule::beginJob (within loop)" << endl; 

    ////---- NULL if illegal; ignore
    if (!detid_.null()) {
      try {
	CastorDetID_ = HcalCastorDetId(detid_);

	dccid = eid->dccid();
	dcc_spgt = pair <int,int> (dccid, eid->spigot());
      
	thisDCC = DCCtoCell.find(dccid);
	thisHTR = HTRtoCell.find(dcc_spgt);
      
	////---- if this DCC has no entries, make this its first one
	if (thisDCC == DCCtoCell.end()) {
	  std::vector <HcalCastorDetId> tempv;
	  tempv.push_back(CastorDetID_);
	  pair <int, std::vector<HcalCastorDetId> > thispair;
	  thispair = pair <int, std::vector<HcalCastorDetId> > (dccid,tempv);
	  DCCtoCell.insert(thispair); 
	}
	else {
	  thisDCC->second.push_back(CastorDetID_);
	}
      
	////---- if this HTR has no entries, make this its first one
	if (thisHTR == HTRtoCell.end()) {
	  std::vector <HcalCastorDetId> tempv;
	  tempv.push_back(CastorDetID_);
	  pair < pair <int,int>, std::vector<HcalCastorDetId> > thispair;
	  thispair = pair <pair <int,int>, std::vector<HcalCastorDetId> > (dcc_spgt,tempv);
	  HTRtoCell.insert(thispair); 
	}
	else {
	  thisHTR->second.push_back(CastorDetID_);	
	}

      } 
      catch (...) { }
    }
  } 

  ////---- get conditions form the Castor DB
  c.get<CastorDbRecord>().get(conditions_);

  ////---- fill reference pedestals with database values
   if (PedMon_!=NULL)
  //////////////  PedMon_->fillDBValues(*conditions_); //FIX 

  
  edm::ESHandle<CastorChannelQuality> p;
  //// c.get<CastorChannelQualityRcd>().get(p);     //FIX
  //// chanquality_= new CastorChannelQuality(*p.product());
 
   */
 if(fVerbosity>0) cout << "CastorMonitorModule::beginJob (end)" << endl;

  return;
} 


//=================================================================//
//========================== beginRun =============================//
//================================================================//
void CastorMonitorModule::beginRun(const edm::Run& run, const edm::EventSetup& c) {
  fedsListed_ = false;
  reset();
}

//=================================================================//
//========================== beginLuminosityBlock ================//
//================================================================//
void CastorMonitorModule::beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
					       const edm::EventSetup& context) {
  
  if(actonLS_ && !prescale()){
    ////---- do scheduled tasks...
  }
}


//================================================================//
//========================== endLuminosityBlock ==================//
//================================================================//
void CastorMonitorModule::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
					     const edm::EventSetup& context) {
  if(actonLS_ && !prescale()){
    ////--- do scheduled tasks...
  }
}

//=================================================================//
//========================== endRun ===============================//
//=================================================================//
void CastorMonitorModule::endRun(const edm::Run& r, const edm::EventSetup& context)
{
  if (fVerbosity>0)  cout <<"CastorMonitorModule::endRun(...) "<<endl;
  ////--- do final pedestal histogram filling
  if (PedMon_!=NULL) //************ PedMon_->fillPedestalHistos(); //FIX
  return;
}

//=================================================================//
//========================== endJob ===============================//
//=================================================================//
void CastorMonitorModule::endJob(void) {
  if ( meStatus_ ) meStatus_->Fill(2);

  if(RecHitMon_!=NULL) RecHitMon_->done();
  if(PedMon_!=NULL) PedMon_->done();
  if(LedMon_!=NULL) LedMon_->done();
  if(RecHitMonValid_!=NULL) RecHitMonValid_->done();
  
  
  /* FIX
  // TO DUMP THE OUTPUT TO DATABASE FILE
  if (dump2database_)
    {
      std::vector<DetId> mydetids = chanquality_->getAllChannels();
      CastorChannelQuality* newChanQual = new CastorChannelQuality();
      for (unsigned int i=0;i<mydetids.size();++i)
	{
	  if (mydetids[i].det()!=4) continue; // not hcal
	  //HcalCastorDetId id(mydetids[i]);
	  HcalCastorDetId id=mydetids[i];
	  // get original channel status item
	  const CastorChannelStatus* origstatus=chanquality_->getValues(mydetids[i]);
	  // make copy of status
	  CastorChannelStatus* mystatus=new CastorChannelStatus(origstatus->rawId(),origstatus->getValue());
	  if (myquality_.find(id)!=myquality_.end())
	    {

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
	    }
	  newChanQual->addValues(*mystatus);
	} // for (unsigned int i=0;...)
      // Now dump out to text file
      std::ostringstream file;
      file <<"CastorDQMstatus_"<<irun_<<".txt";
      std::ofstream outStream(file.str().c_str());
      CastorDbASCIIIO::dumpObject (outStream, (*newChanQual));
      
      //std::ofstream dumb("orig.txt");
      //CastorDbASCIIIO::dumpObject (dumb,(*chanquality_));
      
    } 
  */
  return;
}


//=================================================================//
//========================== reset  ===============================//
//=================================================================//
void CastorMonitorModule::reset(){

  if(PedMon_!=NULL)       PedMon_->reset();
  if(RecHitMon_!=NULL)    RecHitMon_->reset();
  if(LedMon_!=NULL)     LedMon_->reset();
  if(RecHitMonValid_!=NULL) RecHitMonValid_->reset();
}


//=================================================================//
//========================== analyze  ===============================//
//=================================================================//
void CastorMonitorModule::analyze(const edm::Event& e, const edm::EventSetup& eventSetup){
cout << "==>CastorMonitorModule::analyze START !!!" << endl;

  // environment datamembers
  irun_     = e.id().run();
  ilumisec_ = e.luminosityBlock();
  ievent_   = e.id().event();
  itime_    = e.time().value();

  if (fVerbosity>0) { 
  cout << "==> CastorMonitorModule: evts: "<< nevt_ << ", run: " << irun_ << ", LS: " << ilumisec_ << endl;
  cout << " evt: " << ievent_ << ", time: " << itime_  <<"\t counter = "<< ievt_pre_<< "\t total count = "<<ievt_<<endl; 
  }

  // skip this event if we're prescaling...
  ievt_pre_++; // need to increment counter before calling prescale
  if(prescale()) return;

  meLatency_->Fill(psTime_.elapsedTime);

  // Do default setup...
  ievt_++;

  int evtMask=DO_CASTOR_RECHITMON|DO_CASTOR_PED_CALIBMON; 
 // add in DO_HCAL_TPMON, DO_HCAL_CTMON ?(in CastorMonitorSelector.h)
  /* FIX
  //  int trigMask=0;
  if(mtccMon_==NULL){
    evtSel_->processEvent(e);
    evtMask = evtSel_->getEventMask();
    //    trigMask =  evtSel_->getTriggerMask();
  }
  */
  if ( dbe_ ){ 
    meStatus_->Fill(1);
    meEvtMask_->Fill(evtMask);
  }
  
  /////---- See if our products are in the event...
  bool rawOK_    = true;
  bool digiOK_   = true;
  bool rechitOK_ = true;


 cout << "==>CastorMonitorModule::analyze 1 !!!" << endl;
 /*
  ////---- try to get raw data and unpacker report
  edm::Handle<FEDRawDataCollection> RawData;  
 
  try{
    e.getByType(RawData);
  }
  catch(...)
    {
      rawOK_=false;
    }
  if (rawOK_&&!RawData.isValid()) {
    rawOK_=false;
  }
 */
  //--- MY_DEBUG put it for now
  HcalUnpackerReport * report = new HcalUnpackerReport();
  /*
  edm::Handle<HcalUnpackerReport> report;  
  try{
    e.getByType(report);
  }
  catch(...)
    {
      rawOK_=false;
    }
  
  if (rawOK_&&!report.isValid()) {
    rawOK_=false;
  }
  else 
    {
      if(!fedsListed_){
	const std::vector<int> feds =  (*report).getFedsUnpacked();    
	for(unsigned int f=0; f<feds.size(); f++){
	  meFEDS_->Fill(feds[f]);    
	}
	fedsListed_ = true;
      }
    }
  */

  //---------------------------------------------------------------//
  //-------------------  try to get digis ------------------------//
  //---------------------------------------------------------------//

  edm::Handle<CastorDigiCollection> CastorDigi;

  try{
      e.getByLabel(inputLabelDigi_,CastorDigi);
    }
  catch(...) {
      digiOK_=false;
    }

  if (digiOK_ && !CastorDigi.isValid()) {
    digiOK_=false;
  }
  /* MY_DEBUG
  ////---- check that Castor is on by seeing which are reading out FED data
  if ( checkCASTOR_ )
    CheckCastorStatus(*RawData,*report,*readoutMap_,*CastorDigi);
  */ 
 CastorElectronicsMap* readoutMap_ = new CastorElectronicsMap();
 //  CheckCastorStatus(*RawData,*report,*readoutMap_,*CastorDigi);

  //---------------------------------------------------------------//
  //------------------- try to get RecHits ------------------------//
  //---------------------------------------------------------------//
  edm::Handle<CastorRecHitCollection> CastorHits;

  try{
  e.getByLabel(inputLabelRecHitCASTOR_,CastorHits);
  }
  catch(...) {
  rechitOK_=false;
  }
  
  if (rechitOK_&&!CastorHits.isValid()) {
    rechitOK_ = false;
  }

  //------------------------------------------------------------//
  //---------------- Run the configured tasks ------------------//
  //-------------- protect against missing products -----------//
  //-----------------------------------------------------------//

  if (showTiming_)
    {
      cpu_timer.reset(); cpu_timer.start();
    }



  //----------------- Pedestal monitor task ------------------//
  // MY_DEBUG
  // if((PedMon_!=NULL) && (evtMask&DO_CASTOR_PED_CALIBMON) && digiOK_) 
    PedMon_->processEvent(*CastorDigi,*conditions_);
  if (showTiming_)
    {
      cpu_timer.stop();
      if (PedMon_!=NULL) cout <<"TIMER:: PEDESTAL MONITOR ->"<<cpu_timer.cpuTime()<<endl;
      cpu_timer.reset(); cpu_timer.start();
    }


 //----------------- Rec Hit monitor task -------------------------//
  //  if((RecHitMon_ != NULL) && (evtMask&DO_CASTOR_RECHITMON) && rechitOK_) 
    RecHitMon_->processEvent(*CastorHits);
  if (showTiming_)
    {
      cpu_timer.stop();
      if (RecHitMon_!=NULL) cout <<"TIMER:: RECHIT MONITOR ->"<<cpu_timer.cpuTime()<<endl;
      cpu_timer.reset(); cpu_timer.start();
    }

    RecHitMonValid_->processEvent(*CastorHits);


  //---------------- LED monitor task ------------------------//
  //  if((LedMon_!=NULL) && (evtMask&DO_HCAL_LED_CALIBMON) && digiOK_)
     LedMon_->processEvent(*CastorDigi,*conditions_);
   if (showTiming_)
     {
       cpu_timer.stop();
       if (LedMon_!=NULL) cout <<"TIMER:: LED MONITOR ->"<<cpu_timer.cpuTime()<<endl;
       cpu_timer.reset(); cpu_timer.start();
     }
  


  if(fVerbosity>0 && ievt_%1000 == 0)
    cout << "CastorMonitorModule: processed " << ievt_ << " events" << endl;

  if(fVerbosity>0)
    {
      cout << "CastorMonitorModule: processed " << ievt_ << " events" << endl;
      // cout << "    RAW Data   ==> " << rawOK_<< endl;
      // cout << "    Digis      ==> " << digiOK_<< endl;
      // cout << "    RecHits    ==> " << rechitOK_<< endl;
    }
  
  return;
}
//=====================================================================//
//========================== prescale  ===============================//
//===================================================================//
////// It returns true if this event should be skipped according to 
////// the prescale condition.
bool CastorMonitorModule::prescale()
{
  if (fVerbosity>0) cout <<"CastorMonitorModule::prescale"<<endl;
  
  gettimeofday(&psTime_.updateTV,NULL);
  double time = (psTime_.updateTV.tv_sec*1000.0+psTime_.updateTV.tv_usec/1000.0);
  time/= (1000.0); ///in seconds
  psTime_.elapsedTime = time - psTime_.updateTime;
  psTime_.updateTime = time;
  ////---- determine if we care...
  bool evtPS =    prescaleEvt_>0;
  bool lsPS =     prescaleLS_>0;
  bool timePS =   prescaleTime_>0;
  bool updatePS = prescaleUpdate_>0;

  ////---- if no prescales are set, keep the event
  if(!evtPS && !lsPS && !timePS && !updatePS)
    {
      return false;
    }
  ////---- check each instance
  if(lsPS && (ilumisec_%prescaleLS_)!=0) lsPS = false; //-- LS veto
  //if(evtPS && (ievent_%prescaleEvt_)!=0) evtPS = false; //evt # veto
  if (evtPS && (ievt_pre_%prescaleEvt_)!=0) evtPS = false;
  if(timePS)
    {
      double elapsed = (psTime_.updateTime - psTime_.vetoTime)/60.0;
      if(elapsed<prescaleTime_){
	timePS = false;  //-- timestamp veto
	psTime_.vetoTime = psTime_.updateTime;
      }
    } 

  //  if(prescaleUpdate_>0 && (nupdates_%prescaleUpdate_)==0) updatePS=false; ///need to define what "updates" means
  
  if (fVerbosity>0) 
    {
      cout<<"CastorMonitorModule::prescale  evt: "<<ievent_<<"/"<<evtPS<<", ";
      cout <<"ls: "<<ilumisec_<<"/"<<lsPS<<",";
      cout <<"time: "<<psTime_.updateTime - psTime_.vetoTime<<"/"<<timePS<<endl;
    }  
  ////---- if any criteria wants to keep the event, do so
  if(evtPS || lsPS || timePS) return false;
  return true;
} 



//====================================================================//
//=================== CheckCastorStatus  =============================//
//====================================================================//
////---- This function provides a check whether the Castor is on 
////---- by seeing which are reading out FED data  

void CastorMonitorModule::CheckCastorStatus(const FEDRawDataCollection& RawData, 
					       const HcalUnpackerReport& report, 
					       const CastorElectronicsMap& emap,
					       const CastorDigiCollection& CastorDigi
		         		     )
{
  vector<int> fedUnpackList;

   ////---- NO getCastorFEDIds() at the moment
  for (int i=FEDNumbering::getHcalFEDIds().first; i<=FEDNumbering::getHcalFEDIds().second; i++) 
    {
      fedUnpackList.push_back(i);
    }
  
  for (vector<int>::const_iterator i=fedUnpackList.begin(); i!=fedUnpackList.end();++i) 
    {
      const FEDRawData& fed = RawData.FEDData(*i);
      if (fed.size()<12) continue; //-- Was 16 !
      
      ////---- get the DCC header - NO CastorDCCHeader at the moment
      const HcalDCCHeader* dccHeader=(const HcalDCCHeader*)(fed.data());
      if (!dccHeader) return;
      int dccid=dccHeader->getSourceId();
    
      ////---- check for CASTOR
      ////---- Castor FED numbering of DCCs= [690 -693]  
      if (dccid >= 690 && dccid <=693){
	if ( CastorDigi.size()>0){
	  meCASTOR_->Fill(1); 
	 }
	else {meCASTOR_->Fill(0);  }  
      }
      else{ meCASTOR_->Fill(-1); }
    }
  return;
}

#include "FWCore/Framework/interface/MakerMacros.h"
#include <DQM/CastorMonitor/interface/CastorMonitorModule.h>
#include "DQMServices/Core/interface/DQMStore.h"

DEFINE_FWK_MODULE(CastorMonitorModule);
