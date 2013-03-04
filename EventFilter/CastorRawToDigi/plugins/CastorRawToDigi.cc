using namespace std;
#include "EventFilter/CastorRawToDigi/plugins/CastorRawToDigi.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Run.h"

#include "CalibFormats/CastorObjects/interface/CastorDbService.h"
#include "CalibFormats/CastorObjects/interface/CastorDbRecord.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include "EventFilter/CastorRawToDigi/interface/CastorRawCollections.h"

CastorRawToDigi::CastorRawToDigi(edm::ParameterSet const& conf):
  dataTag_(conf.getParameter<edm::InputTag>("InputLabel")),
  unpacker_(conf.getUntrackedParameter<int>("CastorFirstFED",FEDNumbering::MINCASTORFEDID),conf.getParameter<int>("firstSample"),conf.getParameter<int>("lastSample")),
  ctdcunpacker_(conf.getUntrackedParameter<int>("CastorFirstFED",FEDNumbering::MINCASTORFEDID),conf.getParameter<int>("firstSample"),conf.getParameter<int>("lastSample")),
  filter_(conf.getParameter<bool>("FilterDataQuality"),conf.getParameter<bool>("FilterDataQuality"),
	  false,
	  0, 0, 
	  -1),
  fedUnpackList_(conf.getUntrackedParameter<std::vector<int> >("FEDs", std::vector<int>())),
  firstFED_(conf.getUntrackedParameter<int>("CastorFirstFED",FEDNumbering::MINCASTORFEDID)),
//  unpackCalib_(conf.getUntrackedParameter<bool>("UnpackCalib",false)),
  complainEmptyData_(conf.getUntrackedParameter<bool>("ComplainEmptyData",false)),
  usingctdc_(conf.getUntrackedParameter<bool>("CastorCtdc",false)),
  unpackTTP_(conf.getUntrackedParameter<bool>("UnpackTTP",false)),
  silent_(conf.getUntrackedParameter<bool>("silent",true)),
  usenominalOrbitMessageTime_(conf.getUntrackedParameter<bool>("UseNominalOrbitMessageTime",true)),
  expectedOrbitMessageTime_(conf.getUntrackedParameter<int>("ExpectedOrbitMessageTime",-1))

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
  produces<CastorTrigPrimDigiCollection>();
  produces<HcalUnpackerReport>();
  if (unpackTTP_)
    produces<HcalTTPDigiCollection>();

//  if (unpackCalib_)
//    produces<HcalCalibDigiCollection>();

}

// Virtual destructor needed.
CastorRawToDigi::~CastorRawToDigi() { }  

// Functions that gets called by framework every event
void CastorRawToDigi::produce(edm::Event& e, const edm::EventSetup& es)
{
  // Step A: Get Inputs 
  edm::Handle<FEDRawDataCollection> rawraw;  
  e.getByLabel(dataTag_,rawraw);
  // get the mapping
  edm::ESHandle<CastorDbService> pSetup;
  es.get<CastorDbRecord>().get( pSetup );
  const CastorElectronicsMap* readoutMap=pSetup->getCastorMapping();
  
  // Step B: Create empty output  : three vectors for three classes...
  std::vector<CastorDataFrame> castor;
  std::vector<HcalTTPDigi> ttp;
  std::vector<CastorTriggerPrimitiveDigi> htp;

  std::auto_ptr<HcalUnpackerReport> report(new HcalUnpackerReport);

  CastorRawCollections colls;
  colls.castorCont=&castor;
  if (unpackTTP_) colls.ttp=&ttp;
  colls.tpCont=&htp;
  //colls.calibCont=&hc;
 
  // Step C: unpack all requested FEDs
  for (std::vector<int>::const_iterator i=fedUnpackList_.begin(); i!=fedUnpackList_.end(); i++) {
    const FEDRawData& fed = rawraw->FEDData(*i);
    if (fed.size()==0) {
      if (complainEmptyData_) {
	edm::LogWarning("EmptyData") << "No data for FED " << *i;
	report->addError(*i);
      }
    } else if (fed.size()<8*3) {
      edm::LogWarning("EmptyData") << "Tiny data " << fed.size() << " for FED " << *i;
      report->addError(*i);
    } else {
      try {
		  if ( usingctdc_ ) { 
			  ctdcunpacker_.unpack(fed,*readoutMap,colls, *report);
		  } else {
			  unpacker_.unpack(fed,*readoutMap,colls, *report);
	      }
	      report->addUnpacked(*i);
      } catch (cms::Exception& e) {
	edm::LogWarning("Unpacking error") << e.what();
	report->addError(*i);
      } catch (...) {
	edm::LogWarning("Unpacking exception");
	report->addError(*i);
      }
    }
  }

  // Step B: encapsulate vectors in actual collections
  std::auto_ptr<CastorDigiCollection> castor_prod(new CastorDigiCollection()); 
  std::auto_ptr<CastorTrigPrimDigiCollection> htp_prod(new CastorTrigPrimDigiCollection());  

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

  e.put(castor_prod);
  e.put(htp_prod);

  /// calib
//  if (unpackCalib_) {
//    std::auto_ptr<CastorCalibDigiCollection> hc_prod(new CastorCalibDigiCollection());
//    hc_prod->swap_contents(hc);
//    hc_prod->sort();
//    e.put(hc_prod);
//  }
  if (unpackTTP_) {
    std::auto_ptr<HcalTTPDigiCollection> prod(new HcalTTPDigiCollection());
    prod->swap_contents(ttp);
    
    prod->sort();
    e.put(prod);
  }
  e.put(report);
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

