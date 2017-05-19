#include "EventFilter/CastorRawToDigi/plugins/CastorRawToDigi.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Run.h"
#include <fstream>
#include "CalibFormats/CastorObjects/interface/CastorDbService.h"
#include "CalibFormats/CastorObjects/interface/CastorDbRecord.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include "EventFilter/CastorRawToDigi/interface/CastorRawCollections.h"
#include "CalibCalorimetry/HcalPlugins/src/HcalTextCalibrations.h"
#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
using namespace std;


CastorRawToDigi::CastorRawToDigi(edm::ParameterSet const& conf):
  dataTag_(conf.getParameter<edm::InputTag>("InputLabel")),
  unpacker_(conf.getParameter<int>("CastorFirstFED"),conf.getParameter<int>("firstSample"),conf.getParameter<int>("lastSample")),
  zdcunpacker_(conf.getParameter<int>("CastorFirstFED"),conf.getParameter<int>("firstSample"),conf.getParameter<int>("lastSample")),
  ctdcunpacker_(conf.getParameter<int>("CastorFirstFED"),conf.getParameter<int>("firstSample"),conf.getParameter<int>("lastSample")),
  filter_(conf.getParameter<bool>("FilterDataQuality"),conf.getParameter<bool>("FilterDataQuality"),false,0,0,-1),
  fedUnpackList_(conf.getUntrackedParameter<std::vector<int> >("FEDs",std::vector<int>())),
  firstFED_(conf.getParameter<int>("CastorFirstFED")),
  complainEmptyData_(conf.getUntrackedParameter<bool>("ComplainEmptyData",false)),
  usingctdc_(conf.getParameter<bool>("CastorCtdc")),
  unpackTTP_(conf.getParameter<bool>("UnpackTTP")),
  unpackZDC_(conf.getParameter<bool>("UnpackZDC")),
  silent_(conf.getUntrackedParameter<bool>("silent",true)),
  usenominalOrbitMessageTime_(conf.getParameter<bool>("UseNominalOrbitMessageTime")),
  expectedOrbitMessageTime_(conf.getParameter<int>("ExpectedOrbitMessageTime"))

{
  if (fedUnpackList_.empty()) {
    for (int i=FEDNumbering::MINCASTORFEDID; i<=FEDNumbering::MAXCASTORFEDID; i++)
     fedUnpackList_.push_back(i);
  } 

  unpacker_.setExpectedOrbitMessageTime(expectedOrbitMessageTime_);  
  std::ostringstream ss;
  for (unsigned int i=0; i<fedUnpackList_.size(); i++) 
    ss << fedUnpackList_[i] << " ";
  edm::LogInfo("CASTOR") << "CastorRawToDigi will unpack FEDs ( " << ss.str() << ")";
    
  // products produced...
  produces<CastorDigiCollection>();
  produces<ZDCDigiCollection>();
  produces<CastorTrigPrimDigiCollection>();
  produces<HcalUnpackerReport>();
  if (unpackTTP_)
    produces<HcalTTPDigiCollection>();

  tok_input_ = consumes<FEDRawDataCollection>(dataTag_);

}

// Virtual destructor needed.
CastorRawToDigi::~CastorRawToDigi() { }  

// Functions that gets called by framework every event
void CastorRawToDigi::produce(edm::Event& e, const edm::EventSetup& es)
{
  // Step A: Get Inputs 
  edm::Handle<FEDRawDataCollection> rawraw;  
  e.getByToken(tok_input_,rawraw);
  // get the mapping
  edm::ESHandle<CastorDbService> pSetup;
  es.get<CastorDbRecord>().get( pSetup );
  const CastorElectronicsMap* readoutMap=pSetup->getCastorMapping();
  
  // Step B: Create empty output  : three vectors for three classes...
  std::vector<CastorDataFrame> castor;
  std::vector<ZDCDataFrame> zdc;
  std::vector<HcalTTPDigi> ttp;
  std::vector<CastorTriggerPrimitiveDigi> htp;

  auto report = std::make_unique<HcalUnpackerReport>();

  CastorRawCollections colls;
  colls.castorCont=&castor;
  colls.zdcCont=&zdc;
  if (unpackTTP_) colls.ttp=&ttp;
  colls.tpCont=&htp;
 
  // Step C: unpack all requested FEDs
  const FEDRawData& fed722 = rawraw->FEDData(722);
  const int fed722size = fed722.size();
  const FEDRawData& fed693 = rawraw->FEDData(693);
  const int fed693size = fed693.size();
  for (std::vector<int>::const_iterator i=fedUnpackList_.begin(); i!=fedUnpackList_.end(); i++) {
    const FEDRawData& fed = rawraw->FEDData(*i);
    //std::cout<<"Fed number "<<*i<<"is being worked on"<<std::endl;
    if (*i == 693 && fed693size == 0 && fed722size != 0)
      continue;
    if (*i == 722 && fed722size == 0 && fed693size != 0)
      continue;      
      
    if (*i!=693 && *i!=722)
      {
	if (fed.size()==0) 
	  {
	    if (complainEmptyData_) 
	      {
		edm::LogWarning("EmptyData") << "No data for FED " << *i;
		report->addError(*i);
	      }
	  } 
	else if (fed.size()<8*3) 
	  {
	    edm::LogWarning("EmptyData") << "Tiny data " << fed.size() << " for FED " << *i;
	    report->addError(*i);
	  } 
	else 
	  {
	    try 
	      {
		if ( usingctdc_ ) 
		  { 
		    ctdcunpacker_.unpack(fed,*readoutMap,colls, *report);
		  } 
		else 
		  {
		    unpacker_.unpack(fed,*readoutMap,colls, *report);
		  }
		report->addUnpacked(*i);
	      } 
	    catch (cms::Exception& e) 
	      {
		edm::LogWarning("Unpacking error") << e.what();
		report->addError(*i);
	      } catch (...) 
	      {
		edm::LogWarning("Unpacking exception");
		report->addError(*i);
	      }
	  }
      }

    if (*i==693 && unpackZDC_)
      {
	if (fed.size()==0)
	  {
	    if (complainEmptyData_) 
	      {
			edm::LogWarning("EmptyData") << "No data for FED " << *i;
			report->addError(*i);
	      }
	  }
	if (fed.size()!=0)
	  {
	    zdcunpacker_.unpack(fed,*readoutMap,colls,*report);
	    report->addUnpacked(*i); 
	  }
      }
      
    if (*i==722 && unpackZDC_)
      {
	if (fed.size()==0)
	  {
	    if (complainEmptyData_) 
	      {
			edm::LogWarning("EmptyData") << "No data for FED " << *i;
			report->addError(*i);
	      }
	  }
	if (fed.size()!=0)
	  {
	    zdcunpacker_.unpack(fed,*readoutMap,colls,*report);
	    report->addUnpacked(*i);
	  }
      }      

  }//end of loop over feds

  // Step B: encapsulate vectors in actual collections
  auto castor_prod = std::make_unique<CastorDigiCollection>();
  auto htp_prod = std::make_unique<CastorTrigPrimDigiCollection>();

  castor_prod->swap_contents(castor);
  htp_prod->swap_contents(htp);

  // Step C2: filter FEDs, if required
  if (filter_.active()) {
    CastorDigiCollection filtered_castor=filter_.filter(*castor_prod,*report);
    
    castor_prod->swap(filtered_castor);
   }
   
   // Step D: Put outputs into event
   // just until the sorting is proven
   castor_prod->sort();
   htp_prod->sort();
   
   if(unpackZDC_)
     {
       auto zdc_prod = std::make_unique<ZDCDigiCollection>();
       zdc_prod->swap_contents(zdc);
       
       zdc_prod->sort();
       e.put(std::move(zdc_prod));
     }
   
   e.put(std::move(castor_prod));
   e.put(std::move(htp_prod));
   
  if (unpackTTP_) {
    auto prod = std::make_unique<HcalTTPDigiCollection>();
    prod->swap_contents(ttp);
    
    prod->sort();
    e.put(std::move(prod));
  }
  e.put(std::move(report));
}
void CastorRawToDigi::beginRun(edm::Run const& irun, edm::EventSetup const& es){
	if ( usenominalOrbitMessageTime_ ) {
		if ( irun.run() > 132640 ) { 
			expectedOrbitMessageTime_ = 3560;
		} else if ( irun.run() > 132174 ) {
			expectedOrbitMessageTime_ = 3559;
		} else if ( irun.run() > 124371 ) {
			expectedOrbitMessageTime_ = 3557;
		} else if ( irun.run() > 123984 ) {
			expectedOrbitMessageTime_ = 3559;
		} else if ( irun.run() > 123584 ) {
			expectedOrbitMessageTime_ = 2;
		} else {
			expectedOrbitMessageTime_ = 3562;
		}
		unpacker_.setExpectedOrbitMessageTime(expectedOrbitMessageTime_);
	}
}

