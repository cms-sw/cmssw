#include "DQM/HcalMonitorModule/interface/ZDCMonitorModule.h"
#include "DQM/HcalMonitorTasks/interface/HcalZDCMonitor.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Utilities/interface/CPUTimer.h"

#include "DataFormats/Provenance/interface/EventID.h"  
#include "DataFormats/HcalDigi/interface/HcalUnpackerReport.h"
#include "DataFormats/HcalDigi/interface/HcalCalibrationEventTypes.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DQM/HcalMonitorTasks/interface/HcalZDCMonitor.h"

#include "CondFormats/HcalObjects/interface/HcalChannelStatus.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"

#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sys/time.h>

//--------------------------------------------------------
ZDCMonitorModule::ZDCMonitorModule(const edm::ParameterSet& ps){

  irun_=0; ilumisec=0; ievent_=0; itime_=0;

  meStatus_=0;  
  meFEDS_=0;
  meLatency_=0; meQuality_=0;
  fedsListed_ = false;
  zdcMon_ = 0;
  
  // Assumed ZDC is out of the run by default
  ZDCpresent_=0;

  inputLabelDigi_        = ps.getParameter<edm::InputTag>("digiLabel");
  inputLabelRecHitZDC_   = ps.getParameter<edm::InputTag>("zdcRecHitLabel");
  showTiming_ = ps.getUntrackedParameter<bool>("showTiming", false);         //-- show CPU time 
  dump2database_   = ps.getUntrackedParameter<bool>("dump2database",false);  //-- dumps output to database file
  // Check Online running
  Online_                = ps.getUntrackedParameter<bool>("Online",false);
  checkZDC_=ps.getUntrackedParameter<bool>("checkZDC", true); 
  dbe_ = edm::Service<DQMStore>().operator->();
  debug_ = ps.getUntrackedParameter<int>("debug", 0);
  //FEDRawDataCollection_ = ps.getUntrackedParameter<edm::InputTag>("FEDRawDataCollection",edm::InputTag("source",""));

  if (checkZDC_)
    {
      zdcMon_ = new HcalZDCMonitor();
      zdcMon_->setup(ps, dbe_);
    }

  // set parameters   
  prescaleEvt_ = ps.getUntrackedParameter<int>("diagnosticPrescaleEvt", -1);
  if(debug_>1) std::cout << "===>ZDCMonitor event prescale = " << prescaleEvt_ << " event(s)"<< std::endl;

  prescaleLS_ = ps.getUntrackedParameter<int>("diagnosticPrescaleLS", -1);
  if(debug_>1) std::cout << "===>ZDCMonitor lumi section prescale = " << prescaleLS_ << " lumi section(s)"<< std::endl;
  
  // Base folder for the contents of this job
  std::string subsystemname = ps.getUntrackedParameter<std::string>("subSystemFolder", "ZDC") ;
  if(debug_>0) std::cout << "===>ZDCMonitor name = " << subsystemname << std::endl;
  rootFolder_ = subsystemname + "/";
  
  gettimeofday(&psTime_.updateTV,NULL);
  /// get time in milliseconds, convert to minutes
  psTime_.updateTime = (psTime_.updateTV.tv_sec*1000.0+psTime_.updateTV.tv_usec/1000.0);
  psTime_.updateTime /= 1000.0;
  psTime_.elapsedTime=0;
  psTime_.vetoTime=psTime_.updateTime;
}

//--------------------------------------------------------
ZDCMonitorModule::~ZDCMonitorModule()
{
  if (!checkZDC_) return;
  if (dbe_!=0)
    {    
      if (zdcMon_!=0)   zdcMon_->clearME();
      dbe_->setCurrentFolder(rootFolder_);
      dbe_->removeContents();
    }

if (zdcMon_!=0)
    {
      delete zdcMon_; zdcMon_=0;
    }
}

//--------------------------------------------------------
// beginJob no longer needed; trying setup within beginJob won't work !! -- IOV's not loaded
void ZDCMonitorModule::beginJob()
{
  if (!checkZDC_) return;
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
void ZDCMonitorModule::beginRun(const edm::Run& run, const edm::EventSetup& c) 
{
  if (!checkZDC_) return;
  fedsListed_ = false;
  ZDCpresent_ = 0;

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
   
    meFEDS_    = dbe_->book1D("FEDs Unpacked","FEDs Unpacked",1+(FEDNumbering::MAXHCALFEDID-FEDNumbering::MINHCALFEDID),FEDNumbering::MINHCALFEDID-0.5,FEDNumbering::MAXHCALFEDID+0.5);
    // process latency was (200,0,1), but that gave overflows
    meLatency_ = dbe_->book1D("Process Latency","Process Latency",200,0,10);
    meQuality_ = dbe_->book1D("Quality Status","Quality Status",100,0,1);
    // Store whether or not subdetectors are present
    meZDC_ = dbe_->bookInt("ZDCpresent");

    meStatus_->Fill(0);
    // Should fill with 0 to start
    meZDC_->Fill(ZDCpresent_);

  }
  // Create histograms for individual Tasks
  if (zdcMon_)    zdcMon_->beginRun();

  edm::ESHandle<HcalDbService> pSetup;
  c.get<HcalDbRecord>().get( pSetup );

  // Not checking ZDC raw data?  In that case, no readoutMap, hcaldetid_, etc. info needed

  
  //get conditions
  c.get<HcalDbRecord>().get(conditions_);

  // get channel quality -- not yet used for ZDC
  /*
  edm::ESHandle<HcalChannelQuality> p;
  c.get<HcalChannelQualityRcd>().get(p);
  chanquality_= new HcalChannelQuality(*p.product());
  */
  return;
}

//--------------------------------------------------------
void ZDCMonitorModule::beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
     const edm::EventSetup& context) 
{
  /* Don't start a new luminosity block if it is less than the current value
     when running online.  This avoids the problem of getting events
     from mis-ordered lumi blocks, which screws up our lumi block 
     monitoring.
  */
  if (!checkZDC_) return;
  if (Online_ && lumiSeg.luminosityBlock()<ilumisec)
    return;

  // Otherwise, run normal startups
  ilumisec = lumiSeg.luminosityBlock();
  if (zdcMon_!=0)   {  zdcMon_->beginLuminosityBlock(ilumisec);}
}


//--------------------------------------------------------
void ZDCMonitorModule::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
					   const edm::EventSetup& context) 
{
  if (!checkZDC_) return;
  // In online running, don't process events that occur before current luminosity block
  if (Online_ && lumiSeg.luminosityBlock()<ilumisec)
    return; 

  // Call these every luminosity block
  if (zdcMon_!=0)   {  zdcMon_->endLuminosityBlock();}
  // Call these only if prescale set
  if (prescaleLS_>-1 && !prescale())
    {
    }
  return;
}

//--------------------------------------------------------
void ZDCMonitorModule::endRun(const edm::Run& r, const edm::EventSetup& context)
{
  if (!checkZDC_) return;
  if (debug_>0)  
    std::cout <<"ZDCMonitorModule::endRun(...) ievt = "<<ievt_<<std::endl;

  // These should be unnecessary; call them just in case, so that
  // we're sure we get at least one fill per run
  if (zdcMon_!=0)   {  zdcMon_->endLuminosityBlock();}

  return;
}


//--------------------------------------------------------
void ZDCMonitorModule::endJob(void) 
{
  if (!checkZDC_) return;
  if ( dbe_ != NULL ){
    meStatus_  = dbe_->get(rootFolder_+"DQM Job Status/STATUS");
  }
  
  if ( meStatus_ ) meStatus_->Fill(2);

  return; // All of the rest of the endjob stuff (filling db, etc.) should be done in the client, right?

  if (zdcMon_!=NULL)       zdcMon_->done();

  return;
}

//--------------------------------------------------------
void ZDCMonitorModule::reset(){
  if (!checkZDC_) return;
  if (zdcMon_!=NULL) zdcMon_->reset();

}

//--------------------------------------------------------
void ZDCMonitorModule::analyze(const edm::Event& e, const edm::EventSetup& eventSetup)
{
  if (!checkZDC_) return;
  // environment datamembers
  irun_     = e.id().run();
  ievent_   = e.id().event();
  itime_    = e.time().value();
  
  if (Online_ && e.luminosityBlock()<ilumisec)
    return;

  if (debug_>1) std::cout << "ZDCMonitorModule: evts: "<< nevt_ << ", run: " << irun_ << ", LS: " << e.luminosityBlock() << ", evt: " << ievent_ << ", time: " << itime_ << std::endl <<"\t counter = "<<ievt_pre_<<"\t total count = "<<ievt_<<std::endl; 

  if ( meStatus_ ) meStatus_->Fill(1);
  meLatency_->Fill(psTime_.elapsedTime);

  
  ///See if our products are in the event...
  bool rawOK_    = true;
  bool digiOK_   = true;
  bool zdchitOK_ = true;

  edm::Handle<HcalUnpackerReport> report; 
  e.getByLabel(inputLabelDigi_,report);
  if (!report.isValid())
    {
      rawOK_=false;
      edm::LogWarning("ZDCMonitorModule")<<" Unpacker Report Digi Collection "<<inputLabelDigi_<<" not available";
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
  /*
    // need to get raw data first before running filter!
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
    } // if (rawOK_==true) // calibration loop
  */

  // skip this event if we're prescaling...
  ++ievt_;
  if(prescaleEvt_>0 && prescale()) return;
  
  ///////////////////////////////////////////////////////////////////////////////////////////
  // try to get digis
  edm::Handle<ZDCDigiCollection> zdc_digi;
  e.getByLabel(inputLabelDigi_,zdc_digi);
  if (!zdc_digi.isValid())
    {
      digiOK_=false;
      if (debug_>1) std::cout <<"<ZDCMonitorModule> COULDN'T GET ZDC DIGI"<<std::endl;
      //edm::LogWarning("ZDCMonitorModule")<< inputLabelDigi_<<" zdc_digi not available";
    }
  if (digiOK_) ++ievt_digi_;

  ///////////////////////////////////////////////////////////////////////////////////////////

  // try to get rechits
  edm::Handle<ZDCRecHitCollection> zdc_hits;
  e.getByLabel(inputLabelRecHitZDC_,zdc_hits);
  if (!zdc_hits.isValid())
    {
      zdchitOK_=false;
      // ZDC Warnings should be suppressed unless debugging is on (since we don't yet normally run zdcreco)
      if (debug_>0) 
	edm::LogWarning("ZDCMonitorModule")<< inputLabelRecHitZDC_<<" not available"; 
    }
  if (zdchitOK_) ++ievt_rechit_;


///////////////////////////////////////////////////////////////////////////////////////////

  // Run the configured tasks, protect against missing products
  meIEVTALL_->Fill(ievt_);
  meIEVTRAW_->Fill(ievt_rawdata_);
  meIEVTDIGI_->Fill(ievt_digi_);
  meIEVTRECHIT_->Fill(ievt_rechit_);

  if (ZDCpresent_==0 && (digiOK_ || zdchitOK_))
    {
      ZDCpresent_=1;
      meZDC_->Fill(ZDCpresent_);
    }

  // Data Format monitor task
  if (showTiming_)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  if (zdcMon_!=NULL && zdchitOK_ && digiOK_) 
    zdcMon_->processEvent(*zdc_digi,*zdc_hits);
 
  if (showTiming_)
    {
      cpu_timer.stop();
      if (zdcMon_ !=NULL) std::cout <<"TIMER:: ZDC MONITOR ->"<<cpu_timer.cpuTime()<<std::endl;
      cpu_timer.reset(); cpu_timer.start();
    }

 // Empty Event/Unsuppressed monitor plots

  if(debug_>0 && ievt_%1000 == 0)
    std::cout << "ZDCMonitorModule: processed " << ievt_ << " events" << std::endl;

  if(debug_>1)
    {
      std::cout << "ZDCMonitorModule: processed " << ievt_ << " events" << std::endl;
      std::cout << "    ZDC RAW Data   ==> " << rawOK_<< std::endl;
      std::cout << "    ZDC Digis      ==> " << digiOK_<< std::endl;
      std::cout << "    ZDC RecHits    ==> " << zdchitOK_<< std::endl;
    }
  
  return;
}

//--------------------------------------------------------
bool ZDCMonitorModule::prescale()
{
  if (!checkZDC_) return true;

  ///Return true if this event should be skipped according to the prescale condition...
  ///    Accommodate a logical "OR" of the possible tests
  if (debug_>1) std::cout <<"ZDCMonitorModule::prescale:  ievt = "<<ievt_<<std::endl;
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

} // ZDCMonitorModule::prescale(...)


// -------------------------------------------------

DEFINE_FWK_MODULE(ZDCMonitorModule);
