#include <DQM/HcalMonitorModule/interface/HcalMonitorModule.h>

#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDigi/interface/HcalCalibrationEventTypes.h"
#include "DataFormats/HcalDigi/interface/HcalUnpackerReport.h"
#include "DataFormats/Provenance/interface/EventID.h"  
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

#include "CondFormats/HcalObjects/interface/HcalChannelStatus.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"

#include "EventFilter/HcalRawToDigi/interface/HcalDCCHeader.h"
#include "EventFilter/HcalRawToDigi/interface/HcalHTRData.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"

#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

/*
 * \file HcalMonitorModule.cc
 *
 * $Date: 2012/05/27 11:31:22 $
 * $Revision: 1.166 $
 * \author J Temple
 *
 * New version of HcalMonitorModule stores only a few necessary variables that other tasks need to grab
 * (Online_ boolean, subsystem folder name, subdetector presence check, etc.)
 * Modeled heavily on EcalBarrelMonitorModule code.

 * Its only function during its analyze function is to determine the event type (normal, pedestal, laser, etc.) and to perform checks to see which subdetectors are present.  Heavy lifting will be done by individual tasks.

*/


//Constructor

HcalMonitorModule::HcalMonitorModule(const edm::ParameterSet& ps)
{  // Set initial values
  init_=false; // first event sets up Monitor Elements and sets init_ to true

  // get ps objects
  Online_                = ps.getUntrackedParameter<bool>("online",false);
  mergeRuns_             = ps.getUntrackedParameter<bool>("mergeRuns",false);
  enableCleanup_         = ps.getUntrackedParameter<bool>("enableCleanup",false);
  debug_                 = ps.getUntrackedParameter<int>("debug",0);
  
  FEDRawDataCollection_  = ps.getUntrackedParameter<edm::InputTag>("FEDRawDataCollection");
  inputLabelReport_      = ps.getUntrackedParameter<edm::InputTag>("UnpackerReport");
  
  prefixME_              = ps.getUntrackedParameter<std::string>("subSystemFolder","Hcal/");
  if (prefixME_.substr(prefixME_.size()-1,prefixME_.size())!="/")
    prefixME_.append("/");
  
  NLumiBlocks_           = ps.getUntrackedParameter<int>("NLumiBlocks",4000);

} // HcalMonitorModule::HcalMonitorModule


//Destructor
HcalMonitorModule::~HcalMonitorModule()
{

} //HcalMonitorModule::~HcalMonitorModule()



void HcalMonitorModule::beginJob(void)
{
  if (debug_>0) std::cout <<"HcalMonitorModule::beginJob()"<<std::endl;
  // Get DQM service
  dbe_ = edm::Service<DQMStore>().operator->();
  // set default values
  ievt_=0;
  fedsListed_=false;
  HBpresent_=0;
  HEpresent_=0;
  HOpresent_=0;
  HFpresent_=0;
  // Set pointers to null
  meCalibType_=0;
  meFEDS_=0;
  meIevt_=0;
  meIevtHist_=0;
  meEvtsVsLS_=0;
  meProcessedEndLumi_=0;
  meHB_=0;
  meHE_=0;
  meHO_=0;
  meHF_=0;
  eMap_=0;
}

void HcalMonitorModule::beginRun(const edm::Run& r, const edm::EventSetup& c) 
{
  if ( debug_>0 ) std::cout << "HcalMonitorModule: beginRun" << std::endl;
  // reset histograms & counters on a new run, unless merging allowed

  if (eMap_==0) //eMap_ not created yet
    {
      if (debug_>1) std::cout <<"\t<HcalMonitorModule::beginRun> Getting Emap!"<<std::endl;
      edm::ESHandle<HcalDbService> pSetup;
      c.get<HcalDbRecord>().get( pSetup );
      eMap_=pSetup->getHcalMapping(); 
    }
  if (mergeRuns_) return;
  this->setup();
  this->reset();

} //HcalMonitorModule::beginRun(....)


void HcalMonitorModule::endRun(const edm::Run& r, const edm::EventSetup& c) {

  if ( debug_>0 ) std::cout << "HcalMonitorModule: endRun" << std::endl;

  // end-of-run
  if ( meStatus_ ) meStatus_->Fill(2);

  if ( meRun_ ) meRun_->Fill(runNumber_);
  if ( meEvt_ ) meEvt_->Fill(evtNumber_);
}

void HcalMonitorModule::reset(void)
{
  if (debug_>0) std::cout <<"HcalMonitorModule::reset"<<std::endl;
  // Call Reset() on all MonitorElement histograms
  if (meCalibType_) meCalibType_->Reset();
  if (meFEDS_) meFEDS_->Reset();
  if (meIevt_) meIevt_->Fill(0);
  if (meIevtHist_) meIevtHist_->Reset();
  if (meEvtsVsLS_) meEvtsVsLS_->Reset();
  ievt_=0;
  if (meProcessedEndLumi_) meProcessedEndLumi_->Fill(-1);
  if (meHB_) meHB_->Fill(-1);
  if (meHE_) meHE_->Fill(-1);
  if (meHO_) meHO_->Fill(-1);
  if (meHF_) meHF_->Fill(-1);
  HBpresent_=0;
  HEpresent_=0;
  HOpresent_=0;
  HFpresent_=0;
  fedsListed_=false;
} // void HcalMonitorModule::reset(void)

void HcalMonitorModule::setup(void)
{
  // Run this on first event in run; set up all necessary monitor elements
  if (debug_>0) std::cout <<"HcalMonitorModule::setup"<<std::endl;
  init_=true;
  if (dbe_)
    {
      dbe_->setCurrentFolder(prefixME_+"HcalInfo");
      meStatus_ = dbe_->bookInt("STATUS");
      if (meStatus_) meStatus_->Fill(-1);
      meRun_ = dbe_->bookInt("RUN");
      if (meRun_) meRun_->Fill(-1);
      meEvt_ = dbe_->bookInt("EVT");
      if (meEvt_) meEvt_->Fill(-1);
      meIevt_ = dbe_->bookInt("EventsProcessed");
      if (meIevt_) meIevt_->Fill(-1);
      meIevtHist_ = dbe_->book1D("EventsInHcalMonitorModule","Events Seen by HcalMonitorModule",1,0.5,1.5);
      meIevtHist_->setBinLabel(1,"Nevents",1);
      meEvtsVsLS_ = dbe_->book1D("EventsVsLS","Events vs. Luminosity Section;LS;# events",NLumiBlocks_,0.5,NLumiBlocks_+0.5);
      meOnline_ = dbe_->bookInt("Online");
      meOnline_->Fill((int)Online_);
      meProcessedEndLumi_ = dbe_->bookInt("EndLumiBlock_MonitorModule");
      if (meProcessedEndLumi_) meProcessedEndLumi_->Fill(-1);
      meCurrentCalibType_= dbe_->bookInt("CURRENT_EVENT_TYPE");
      if (meCurrentCalibType_) meCurrentCalibType_->Fill(-1);
      
      meHB_ = dbe_->bookInt("HBpresent");
      meHE_ = dbe_->bookInt("HEpresent");
      meHO_ = dbe_->bookInt("HOpresent");
      meHF_ = dbe_->bookInt("HFpresent");
      if (meHB_) meHB_->Fill(-1);
      if (meHE_) meHE_->Fill(-1);
      if (meHO_) meHO_->Fill(-1);
      if (meHF_) meHF_->Fill(-1);

      meFEDS_    = dbe_->book1D("FEDs Unpacked","FEDs Unpacked; Hcal FEDs 700-731",1+(FEDNumbering::MAXHCALFEDID-FEDNumbering::MINHCALFEDID),FEDNumbering::MINHCALFEDID-0.5,FEDNumbering::MAXHCALFEDID+0.5);

      meCalibType_ = dbe_->book1D("CalibrationType","Calibration Type",9,-0.5,8.5);
      meCalibType_->setBinLabel(1,"Normal",1);
      meCalibType_->setBinLabel(2,"Ped",1);
      meCalibType_->setBinLabel(3,"RADDAM",1);
      meCalibType_->setBinLabel(4,"HBHEHPD",1);
      meCalibType_->setBinLabel(5,"HOHPD",1);
      meCalibType_->setBinLabel(6,"HFPMT",1);
      meCalibType_->setBinLabel(7,"ZDC",1);
      meCalibType_->setBinLabel(8,"CASTOR",1);

    } // if (dbe_)
  return;
} // void HcalMonitorModule::setup(void)


void HcalMonitorModule::cleanup(void)
{
  if (debug_>0) std::cout <<"HcalMonitorModule::cleanup"<<std::endl;
  if (!enableCleanup_) return;
  if (dbe_)
    {
      dbe_->setCurrentFolder(prefixME_+"HcalInfo");
      if ( meStatus_ ) 
	dbe_->removeElement( meStatus_->getName() );
      meStatus_ = 0;
      if ( meRun_ ) 
	dbe_->removeElement( meRun_->getName() );
      meRun_ = 0;
      if ( meEvt_ ) 
	dbe_->removeElement( meEvt_->getName() );
      meEvt_ = 0;
      if (meIevt_) 
	dbe_->removeElement(meIevt_->getName());
      meIevt_=0;
      if (meIevtHist_)
	dbe_->removeElement(meIevtHist_->getName());
      meIevtHist_=0;
      if (meFEDS_) 
	dbe_->removeElement(meFEDS_->getName());
      meFEDS_ = 0;
      if (meCalibType_) 
	dbe_->removeElement(meCalibType_->getName());
      meCalibType_ = 0;
      if (meCurrentCalibType_) 
	dbe_->removeElement(meCurrentCalibType_->getName());
      meCurrentCalibType_=0;
      if (meProcessedEndLumi_) 
	dbe_->removeElement(meProcessedEndLumi_->getName());
      meProcessedEndLumi_ = 0;
      if (meHB_) 
	dbe_->removeElement(meHB_->getName());
      meHB_=0;  

      if (meHE_) 
	dbe_->removeElement(meHE_->getName());
      meHE_=0;
      if (meHO_) 
	dbe_->removeElement(meHO_->getName());
      meHO_=0;
      if (meHF_) 
	dbe_->removeElement(meHF_->getName());
      meHF_=0;
    } // if (dbe_)

  fedsListed_=false;
  HBpresent_=0;
  HEpresent_=0;
  HOpresent_=0;
  HFpresent_=0;
  init_=false;

} // void HcalMonitorModule::cleanup(void)



void HcalMonitorModule::beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg,
						const edm::EventSetup& c) 
{
  if (debug_>0) std::cout <<"HcalMonitorModule::beginLuminosityBlock"<<std::endl;
}// void HcalMonitorModule::beginLuminosityBlock(...)




void HcalMonitorModule::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg,
					      const edm::EventSetup& c) 
{
  if (debug_>0) std::cout <<"HcalMonitorModule::endLuminosityBlock"<<std::endl;
  meProcessedEndLumi_->Fill(lumiSeg.luminosityBlock());
}// void HcalMonitorModule::endLuminosityBlock(...)



void HcalMonitorModule::endJob(void)
{
  if (debug_>0) std::cout <<"HcalMonitorModule::endJob()"<<std::endl;
  if (dbe_)
    {
      meStatus_ = dbe_->get(prefixME_ + "/EventInfo/STATUS");
      meRun_ = dbe_->get(prefixME_ + "/EventInfo/RUN");
      meEvt_ = dbe_->get(prefixME_ + "/EventInfo/EVT");
    }
  if (meStatus_) meStatus_->Fill(2);
  if (meRun_) meRun_->Fill(runNumber_);
  if (meEvt_) meEvt_->Fill(evtNumber_);
  if (init_) this->cleanup();
} // void HcalMonitorModule::endJob(void)



void HcalMonitorModule::analyze(const edm::Event& e, const edm::EventSetup& c)
{
  if (!init_) this->setup();
  
  LogDebug("HcalMonitorModule")<<"processing event "<<ievt_;
  
  // Fill Monitor Elements with run, evt, processed event info
  ++ievt_;
  runNumber_=e.id().run();
  evtNumber_=e.id().event();
  if (meRun_) meRun_->Fill(runNumber_);
  if (meEvt_) meEvt_->Fill(evtNumber_);
  if (meIevt_) meIevt_->Fill(ievt_);
  if (meIevtHist_) meIevtHist_->Fill(1);
  if (meEvtsVsLS_) meEvtsVsLS_->Fill(e.luminosityBlock(),1);
  if (ievt_==1)
    {
      LogDebug("HcalMonitorModule") << "processing run " << runNumber_;
      // begin-of-run
      if ( meStatus_ ) meStatus_->Fill(0);
    } 
  else 
    {
      // running
      if ( meStatus_ ) meStatus_->Fill(1);
    }
  
  // Try to get raw data
  edm::Handle<FEDRawDataCollection> rawraw;  
  if (!(e.getByLabel(FEDRawDataCollection_,rawraw)))
    {
      edm::LogWarning("HcalMonitorModule")<<" raw data with label "<<FEDRawDataCollection_ <<" not available";
      return;
    }
  
  // Get Event Calibration Type -- copy of Bryan Dahmes' filter
  int calibType=-1;
  int numEmptyFEDs = 0 ;
  std::vector<int> calibTypeCounter(8,0) ;
  for( int i = FEDNumbering::MINHCALFEDID; i <= FEDNumbering::MAXHCALFEDID; i++) 
    {
      const FEDRawData& fedData = rawraw->FEDData(i) ;
      
      if ( fedData.size() < 24 ) numEmptyFEDs++ ;
      if ( fedData.size() < 24 ) continue;

      int value = (int)((const HcalDCCHeader*)(fedData.data()))->getCalibType() ;
      if(value>7) 
	{
	  edm::LogWarning("HcalMonitorModule::CalibTypeFilter") << "Unexpected Calibration type: "<< value << " in FED: "<<i<<" (should be 0-7). I am bailing out...";
	  return;
	}

      calibTypeCounter.at(value)++ ; // increment the counter for this calib type
    } // for (int i = FEDNumbering::MINHCALFEDID; ...)

  int maxCount = 0;
  int numberOfFEDIds = FEDNumbering::MAXHCALFEDID  - FEDNumbering::MINHCALFEDID + 1 ;
  for (unsigned int i=0; i<calibTypeCounter.size(); i++) {
    if ( calibTypeCounter.at(i) > maxCount )
      { calibType = i ; maxCount = calibTypeCounter.at(i) ; }
    if ( maxCount == numberOfFEDIds ) break ;
  }
  
  if ( maxCount != (numberOfFEDIds-numEmptyFEDs) )
    edm::LogWarning("HcalMonitorModule::CalibTypeFilter") << "Conflicting calibration types found.  Assigning type "
					   << calibType ;
  LogDebug("HcalMonitorModule::CalibTypeFilter") << "Calibration type is: " << calibType ;
  // Fill histogram of calibration types, as well as integer to keep track of current value
  if (meCalibType_) meCalibType_->Fill(calibType);
  if (meCurrentCalibType_) meCurrentCalibType_->Fill(calibType);
  ////if (meCurrentCalibType_) meCurrentCalibType_->Fill(ievt_); // use for debugging check ONLY!

  if (debug_>2) std::cout <<"\t<HcalMonitorModule>  ievt = "<<ievt_<<"  calibration type = "<<calibType<<std::endl;

  // Check to see which subdetectors are present.
  // May only need to do this on first event?   Subdets don't appear during a run?
  if (HBpresent_==0)
    CheckSubdetectorStatus(rawraw, HcalBarrel, *eMap_);
  if (HEpresent_==0)
    CheckSubdetectorStatus(rawraw, HcalEndcap, *eMap_);
  if (HOpresent_==0)
    CheckSubdetectorStatus(rawraw, HcalOuter, *eMap_);
  if (HFpresent_==0)
    CheckSubdetectorStatus(rawraw, HcalForward, *eMap_);
  
  //  Here, we do need this information each event
  edm::Handle<HcalUnpackerReport> report;  
  if (!(e.getByLabel(inputLabelReport_,report)))
    {
      edm::LogWarning("HcalMonitorModule")<<" Unpacker Report "<<inputLabelReport_<<" not available";
      return;
    }

  if (!fedsListed_)
    {
      const std::vector<int> feds =  (*report).getFedsUnpacked();    
      for(unsigned int f=0; f<feds.size(); ++f)
	meFEDS_->Fill(feds[f]);    
      fedsListed_ = true;
    } // if (!fedsListed_)

} // void HcalMonitorModule::analyze(...)



void HcalMonitorModule::CheckSubdetectorStatus(const edm::Handle<FEDRawDataCollection>& rawraw,
						  HcalSubdetector subdet,
						  const HcalElectronicsMap& emap)
{

  std::vector<int> fedUnpackList;
  for (int i=FEDNumbering::MINHCALFEDID; 
       i<=FEDNumbering::MAXHCALFEDID; 
       i++) 
    fedUnpackList.push_back(i);

  if (debug_>1) std::cout <<"<HcalMonitorModule::CheckSubdetectorStatus>  Checking subdetector "<<subdet<<std::endl;
  for (std::vector<int>::const_iterator i=fedUnpackList.begin();
       i!=fedUnpackList.end(); 
       ++i) 
    {
      if (debug_>2) std::cout <<"\t<HcalMonitorModule::CheckSubdetectorStatus>  FED = "<<*i<<std::endl;
      const FEDRawData& fed =(*rawraw).FEDData(*i);
      if (fed.size()<12) continue; // Was 16. How do such tiny events even get here?
      
      // get the DCC header
      const HcalDCCHeader* dccHeader=(const HcalDCCHeader*)(fed.data());
      if (!dccHeader) return;
      int dccid=dccHeader->getSourceId();
      // check for HF
      if (subdet == HcalForward && dccid>717 && dccid<724)
	{
	  HFpresent_=1;
	  meHF_->Fill(HFpresent_);
	  return;
	}
      else if (subdet==HcalOuter && dccid>723)
	{
	  HOpresent_=1;
	  meHO_->Fill(HOpresent_);
	  return;
	}
      else if (dccid<718 && (subdet==HcalBarrel || subdet==HcalEndcap))
	{
	  HcalHTRData htr;  
	  for (int spigot=0; spigot<HcalDCCHeader::SPIGOT_COUNT; spigot++) 
	    {    
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
			  
			  if ((HcalSubdetector)did.subdetId()==subdet)
			    {
			      if (subdet==HcalBarrel)
				{
				  HBpresent_=1;
				  meHB_->Fill(HBpresent_);
				  return;
				}
			      else if (subdet==HcalEndcap)
			      {
				HEpresent_=1;
				meHE_->Fill(HEpresent_);
				return;
			      }
			    } // if ((HcalSubdetector)did.subdetId()==subdet)
			} // if (!did.null())
		    } // for (int fib=1;fib<9;...)
		} // for (int fchan=0; fchan<3;...)
	    } // for (int spigot=0;spigot<HcalDCCHeader::SPIGOT_COUNT; spigot++) 
	} //else if (dcc<718 && (subdet...))
  } // loop over fedUnpackList
  

} // void HcalMonitorModule::CheckSubdetectorStatus(...)

DEFINE_FWK_MODULE(HcalMonitorModule);
