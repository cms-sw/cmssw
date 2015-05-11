#include <DQM/HcalMonitorTasks/interface/HcalBaseDQMonitor.h>
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "CondFormats/HcalObjects/interface/HcalLogicalMap.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalLogicalMapGenerator.h"
#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "EventFilter/HcalRawToDigi/interface/HcalDCCHeader.h"
#include "EventFilter/HcalRawToDigi/interface/HcalHTRData.h"

#include <iostream>
#include <vector>

/*
 * \file HcalBaseDQMonitor.cc
 *
 * \author J Temple
 *
 * Base class for all Hcal DQM analyzers
 *

*/

// constructor

HcalBaseDQMonitor::HcalBaseDQMonitor(const edm::ParameterSet& ps)
{
  Online_                = ps.getUntrackedParameter<bool>("online",false);
  mergeRuns_             = ps.getUntrackedParameter<bool>("mergeRuns",false);
  enableCleanup_         = ps.getUntrackedParameter<bool>("enableCleanup",false);
  debug_                 = ps.getUntrackedParameter<int>("debug",0);
  prefixME_              = ps.getUntrackedParameter<std::string>("subSystemFolder","Hcal/"); 
  if (prefixME_.substr(prefixME_.size()-1,prefixME_.size())!="/")
    prefixME_.append("/");
  subdir_                = ps.getUntrackedParameter<std::string>("TaskFolder","Test/"); 
  if (subdir_.size()>0 && subdir_.substr(subdir_.size()-1,subdir_.size())!="/")
    subdir_.append("/");
  subdir_=prefixME_+subdir_;
  AllowedCalibTypes_     = ps.getUntrackedParameter<std::vector<int> > ("AllowedCalibTypes");
  skipOutOfOrderLS_      = ps.getUntrackedParameter<bool>("skipOutOfOrderLS",false);
  NLumiBlocks_           = ps.getUntrackedParameter<int>("NLumiBlocks",4000);
  makeDiagnostics_       = ps.getUntrackedParameter<bool>("makeDiagnostics",false);


  FEDRawDataCollection_  = ps.getUntrackedParameter<edm::InputTag>("FEDRawDataCollection");
  tok_braw_ = consumes<FEDRawDataCollection>(FEDRawDataCollection_);
  
  setupDone_ = false;
  logicalMap_= 0;
  needLogicalMap_=false;
  meIevt_=0;
  meLevt_=0;
  meTevtHist_=0;
  ProblemsVsLB=0;
  ProblemsVsLB_HB=0;
  ProblemsVsLB_HE=0;
  ProblemsVsLB_HF=0;
  ProblemsVsLB_HBHEHF=0;
  ProblemsVsLB_HO=0;
  ProblemsCurrentLB=0;

  eMap_ = 0;

  ievt_=0;
  levt_=0;
  tevt_=0;
  currenttype_=-1;
  HBpresent_=false;
  HEpresent_=false;
  HOpresent_=false;
  HFpresent_=false;


} //HcalBaseDQMonitor::HcalBaseDQMonitor(const ParameterSet& ps)




// destructor

HcalBaseDQMonitor::~HcalBaseDQMonitor()
{
  if (logicalMap_) delete logicalMap_;
}


//dqmBeginRun
void HcalBaseDQMonitor::dqmBeginRun(const edm::Run &run, const edm::EventSetup &es)
{

  if (eMap_==0) //eMap_ not created yet
    {
      if (debug_>1) std::cout <<"\t<HcalBaseDQMonitor::dqmBeginRun> Getting Emap!"<<std::endl;
      edm::ESHandle<HcalDbService> pSetup;
      es.get<HcalDbRecord>().get( pSetup );
      eMap_=pSetup->getHcalMapping(); 
    }
  if (mergeRuns_) return;
  if( setupDone_ ) this->reset();

}



void HcalBaseDQMonitor::bookHistograms(DQMStore::IBooker &ib, const edm::Run& run, const edm::EventSetup& c)
{
  if (debug_>0) std::cout <<"HcalBaseDQMonitor::bookHistograms():  task =  '"<<subdir_<<"'"<<std::endl;
  if (! mergeRuns_)
    {
      this->setup(ib);
      this->reset();
    }
  else if (tevt_ == 0)
    {
      this->setup(ib);
      this->reset();
    }
} // beginRun(const edm::Run& run, const edm::EventSetup& c)

void HcalBaseDQMonitor::endRun(const edm::Run& run, const edm::EventSetup& c)
{
  if (debug_>0) std::cout <<"HcalBaseDQMonitor::endRun:  task = "<<subdir_<<std::endl;
} //endRun(...)

void HcalBaseDQMonitor::reset(void)
{
  if (debug_>0) std::cout <<"HcalBaseDQMonitor::reset():  task = "<<subdir_<<std::endl;
  if (meIevt_) meIevt_->Fill(-1);
  ievt_=0;
  if (meLevt_) meLevt_->Fill(-1);
  levt_=0;
  if (meTevt_) meTevt_->Fill(-1);
  tevt_=0;
  if (meTevtHist_) meTevtHist_->Reset();
  if (ProblemsCurrentLB) ProblemsCurrentLB->Reset();
  HBpresent_=false;
  HEpresent_=false;
  HOpresent_=false;
  HFpresent_=false;
  currentLS=0;
  currenttype_=-1;
} //reset()

void HcalBaseDQMonitor::cleanup(void)
{

} //cleanup()

void HcalBaseDQMonitor::setup(DQMStore::IBooker &ib)
{
  if (setupDone_)
    return;
  setupDone_ = true;
  if (debug_>3) std::cout <<"<HcalBaseDQMonitor> setup in progress"<<std::endl;
  ib.setCurrentFolder(subdir_);
  meIevt_ = ib.bookInt("EventsProcessed");
  if (meIevt_) meIevt_->Fill(-1);
  meLevt_ = ib.bookInt("EventsProcessed_currentLS");
  if (meLevt_) meLevt_->Fill(-1);
  meTevt_ = ib.bookInt("EventsProcessed_Total");
  if (meTevt_) meTevt_->Fill(-1);
  meTevtHist_=ib.book1D("Events_Processed_Task_Histogram","Counter of Events Processed By This Task",1,0.5,1.5);
  if (meTevtHist_) meTevtHist_->Reset();
  ib.setCurrentFolder(subdir_+"LSvalues");
  ProblemsCurrentLB=ib.book2D("ProblemsThisLS","Problem Channels in current Lumi Section",
				 7,0,7,1,0,1);
  if (ProblemsCurrentLB)
    {
      (ProblemsCurrentLB->getTH2F())->GetXaxis()->SetBinLabel(1,"HB");
      (ProblemsCurrentLB->getTH2F())->GetXaxis()->SetBinLabel(2,"HE");
      (ProblemsCurrentLB->getTH2F())->GetXaxis()->SetBinLabel(3,"HO");
      (ProblemsCurrentLB->getTH2F())->GetXaxis()->SetBinLabel(4,"HF");
      (ProblemsCurrentLB->getTH2F())->GetXaxis()->SetBinLabel(5,"HO0");
      (ProblemsCurrentLB->getTH2F())->GetXaxis()->SetBinLabel(6,"HO12");
      (ProblemsCurrentLB->getTH2F())->GetXaxis()->SetBinLabel(7,"HFlumi");
      (ProblemsCurrentLB->getTH2F())->GetYaxis()->SetBinLabel(1,"Status");
      ProblemsCurrentLB->Reset();
    }
} // setup()


void HcalBaseDQMonitor::beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg,
			  const edm::EventSetup& c)
{
  if (this->LumiInOrder(lumiSeg.luminosityBlock())==false) return;
  currentLS=lumiSeg.luminosityBlock();
  levt_=0;
  if (meLevt_!=0) meLevt_->Fill(-1);
  if (ProblemsCurrentLB)
    ProblemsCurrentLB->Reset();
}

void HcalBaseDQMonitor::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg,
			const edm::EventSetup& c)
{
  if (this->LumiInOrder(lumiSeg.luminosityBlock())==false) return;
  // Inherited classes do end-of-lumi functions here
}


bool HcalBaseDQMonitor::LumiInOrder(int lumisec)
{
  if (skipOutOfOrderLS_==false) return true; // don't skip out-of-order lumi sections
  // check that latest lumi section is >= last processed
  if (Online_ && lumisec<currentLS)
    return false;
  return true;
}

bool HcalBaseDQMonitor::IsAllowedCalibType()
{
  if (debug_>9) std::cout <<"<HcalBaseDQMonitor::IsAllowedCalibType>"<<std::endl;
  if (AllowedCalibTypes_.size()==0)
    {
      if (debug_>9) std::cout <<"\tNo calib types specified by user; All events allowed"<<std::endl;
      return true;
    }

  if (debug_>9) std::cout <<"\tHcalBaseDQMonitor::IsAllowedCalibType  checking if calibration type = "<<currenttype_<<" is allowed...";
  for (std::vector<int>::size_type i=0;i<AllowedCalibTypes_.size();++i)
    {
      if (AllowedCalibTypes_[i]==currenttype_)
	{
	  if (debug_>9) std::cout <<"\t Type allowed!"<<std::endl;
	  return true;
	}
    }
  if (debug_>9) std::cout <<"\t Type not allowed!"<<std::endl;
  return false;

} // bool HcalBaseDQMonitor::IsAllowedCalibType()

void HcalBaseDQMonitor::getLogicalMap(const edm::EventSetup& c) {
  if (needLogicalMap_ && logicalMap_==0) {
    edm::ESHandle<HcalTopology> pT;
    c.get<IdealGeometryRecord>().get(pT);   
    HcalLogicalMapGenerator gen;
    logicalMap_=new HcalLogicalMap(gen.createMap(&(*pT)));
  }
}
 
void HcalBaseDQMonitor::analyze(const edm::Event& e, const edm::EventSetup& c)
{
  getLogicalMap(c);

  if (debug_>5) std::cout <<"\t<HcalBaseDQMonitor::analyze>  event = "<<ievt_<<std::endl;
  eventAllowed_=true; // assume event is allowed


  // Try to get raw data
  edm::Handle<FEDRawDataCollection> rawraw;  
  if (!(e.getByToken(tok_braw_,rawraw)))
    {
      edm::LogWarning("HcalMonitorModule")<<" raw data with label "<<FEDRawDataCollection_ <<" not available";
      return;
    }

  // fill with total events seen (this differs from ievent, which is total # of good events)
  ++tevt_;
  if (meTevt_) meTevt_->Fill(tevt_);
  if (meTevtHist_) meTevtHist_->Fill(1);
  // skip out of order lumi events
  if (this->LumiInOrder(e.luminosityBlock())==false)
    {
      eventAllowed_=false;
      return;
    }

  this->CheckCalibType(rawraw);
  // skip events of wrong calibration type
  eventAllowed_&=(this->IsAllowedCalibType());
  if (!eventAllowed_) return;

  // Event is good; count it
  ++ievt_;
  ++levt_;
  if (meIevt_) meIevt_->Fill(ievt_);
  if (meLevt_) meLevt_->Fill(levt_);


  if (HBpresent_==false)
    {
      CheckSubdetectorStatus(rawraw,HcalBarrel,*eMap_);
    }
  if (HEpresent_==false)
    {
      CheckSubdetectorStatus(rawraw,HcalEndcap,*eMap_);
    }
  if (HOpresent_==false)
    {
      CheckSubdetectorStatus(rawraw,HcalOuter,*eMap_);
    }
  if (HFpresent_==false)
    {
      CheckSubdetectorStatus(rawraw,HcalForward,*eMap_);
    }


} // void HcalBaseDQMonitor::analyze(const edm::Event& e, const edm::EventSetup& c)

//
//  CheckSubdetectorStatus
void HcalBaseDQMonitor::CheckSubdetectorStatus(const edm::Handle<FEDRawDataCollection>& rawraw, HcalSubdetector subdet, const HcalElectronicsMap &emap)
{

  std::vector<int> fedUnpackList;
  for (int i=FEDNumbering::MINHCALFEDID; 
       i<=FEDNumbering::MAXHCALuTCAFEDID; 
       i++) 
  {
	if (i>FEDNumbering::MAXHCALFEDID && i<FEDNumbering::MINHCALuTCAFEDID)
		continue;
	fedUnpackList.push_back(i);
  }

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
      if (subdet == HcalForward && ((dccid>=1118 && dccid<=1122) || 
				  (dccid>=718 && dccid<=723)))
	{
	  HFpresent_=true;
	  return;
	}
      else if (subdet==HcalOuter && dccid>723)
	{
	  HOpresent_=true;
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
				  HBpresent_=true;
				  return;
				}
			      else if (subdet==HcalEndcap)
			      {
				HEpresent_=true;
				return;
			      }
			    } // if ((HcalSubdetector)did.subdetId()==subdet)
			} // if (!did.null())
		    } // for (int fib=1;fib<9;...)
		} // for (int fchan=0; fchan<3;...)
	    } // for (int spigot=0;spigot<HcalDCCHeader::SPIGOT_COUNT; spigot++) 
	} //else if (dcc<718 && (subdet...))
  } // loop over fedUnpackList
  

} // void CheckSubdetectorStatus


// Check Calib type
void HcalBaseDQMonitor::CheckCalibType(const edm::Handle<FEDRawDataCollection> &rawraw)
{
  // Get Event Calibration Type -- copy of Bryan Dahmes' filter
  int calibType=-1;
  int numEmptyFEDs = 0 ;
  std::vector<int> calibTypeCounter(8,0) ;
  for( int i = FEDNumbering::MINHCALFEDID; 
		  i <= FEDNumbering::MAXHCALuTCAFEDID; i++) 
    {
		if (i>FEDNumbering::MAXHCALFEDID && i<FEDNumbering::MINHCALuTCAFEDID)
			continue;

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
  int numberOfFEDIds = (FEDNumbering::MAXHCALFEDID-FEDNumbering::MINHCALFEDID+1) +
	  (FEDNumbering::MAXHCALuTCAFEDID-FEDNumbering::MINHCALuTCAFEDID+1);
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
  currenttype_ = calibType;
  //if (meCalibType_) meCalibType_->Fill(calibType);
  //if (meCurrentCalibType_) meCurrentCalibType_->Fill(calibType);
  ////if (meCurrentCalibType_) meCurrentCalibType_->Fill(ievt_); // use for debugging check ONLY!

  if (debug_>2) std::cout <<"\t<HcalMonitorModule>  ievt = "<<ievt_<<"  calibration type = "<<calibType<<std::endl;

} // check calib type

DEFINE_FWK_MODULE(HcalBaseDQMonitor);
