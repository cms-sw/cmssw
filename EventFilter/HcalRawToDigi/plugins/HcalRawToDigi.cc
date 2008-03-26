using namespace std;
#include "EventFilter/HcalRawToDigi/plugins/HcalRawToDigi.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

HcalRawToDigi::HcalRawToDigi(edm::ParameterSet const& conf):
  dataTag_(conf.getParameter<edm::InputTag>("InputLabel")),
  unpacker_(conf.getUntrackedParameter<int>("HcalFirstFED",FEDNumbering::getHcalFEDIds().first),conf.getParameter<int>("firstSample"),conf.getParameter<int>("lastSample")),
  filter_(conf.getParameter<bool>("FilterDataQuality"),conf.getParameter<bool>("FilterDataQuality"),
	  false,
	  0, 0, 
	  -1),
  fedUnpackList_(conf.getUntrackedParameter<std::vector<int> >("FEDs", std::vector<int>())),
  firstFED_(conf.getUntrackedParameter<int>("HcalFirstFED",FEDNumbering::getHcalFEDIds().first)),
  unpackCalib_(conf.getUntrackedParameter<bool>("UnpackCalib",false)),
  unpackZDC_(conf.getUntrackedParameter<bool>("UnpackZDC",false)),
  complainEmptyData_(conf.getUntrackedParameter<bool>("ComplainEmptyData",false))
{
  if (fedUnpackList_.empty()) {
    for (int i=FEDNumbering::getHcalFEDIds().first; i<=FEDNumbering::getHcalFEDIds().second; i++)
      fedUnpackList_.push_back(i);
  } 
  
  std::ostringstream ss;
  for (unsigned int i=0; i<fedUnpackList_.size(); i++) 
    ss << fedUnpackList_[i] << " ";
  edm::LogInfo("HCAL") << "HcalRawToDigi will unpack FEDs ( " << ss.str() << ")";
    
  // products produced...
  produces<HBHEDigiCollection>();
  produces<HFDigiCollection>();
  produces<HODigiCollection>();
  produces<HcalTrigPrimDigiCollection>();
  produces<HcalUnpackerReport>();
  if (unpackCalib_)
    produces<HcalCalibDigiCollection>();
  if (unpackZDC_)
    produces<ZDCDigiCollection>();
}

// Virtual destructor needed.
HcalRawToDigi::~HcalRawToDigi() { }  

// Functions that gets called by framework every event
void HcalRawToDigi::produce(edm::Event& e, const edm::EventSetup& es)
{
  // Step A: Get Inputs 
  edm::Handle<FEDRawDataCollection> rawraw;  
  e.getByLabel(dataTag_,rawraw);
  // get the mapping
  edm::ESHandle<HcalDbService> pSetup;
  es.get<HcalDbRecord>().get( pSetup );
  const HcalElectronicsMap* readoutMap=pSetup->getHcalMapping();
  
  // Step B: Create empty output  : three vectors for three classes...
  std::vector<HBHEDataFrame> hbhe;
  std::vector<HODataFrame> ho;
  std::vector<HFDataFrame> hf;
  std::vector<HcalTriggerPrimitiveDigi> htp;
  std::vector<HcalCalibDataFrame> hc;
  std::vector<ZDCDataFrame> zdc;
  std::auto_ptr<HcalUnpackerReport> report(new HcalUnpackerReport);

  HcalUnpacker::Collections colls;
  colls.hbheCont=&hbhe;
  colls.hoCont=&ho;
  colls.hfCont=&hf;
  colls.tpCont=&htp;
  colls.calibCont=&hc;
  colls.zdcCont=&zdc;
 
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
	unpacker_.unpack(fed,*readoutMap,colls, *report);
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
  std::auto_ptr<HBHEDigiCollection> hbhe_prod(new HBHEDigiCollection()); 
  std::auto_ptr<HFDigiCollection> hf_prod(new HFDigiCollection());
  std::auto_ptr<HODigiCollection> ho_prod(new HODigiCollection());
  std::auto_ptr<HcalTrigPrimDigiCollection> htp_prod(new HcalTrigPrimDigiCollection());  

  hbhe_prod->swap_contents(hbhe);
  hf_prod->swap_contents(hf);
  ho_prod->swap_contents(ho);
  htp_prod->swap_contents(htp);

  // Step C2: filter FEDs, if required
  if (filter_.active()) {
    HBHEDigiCollection filtered_hbhe=filter_.filter(*hbhe_prod,*report);
    HODigiCollection filtered_ho=filter_.filter(*ho_prod,*report);
    HFDigiCollection filtered_hf=filter_.filter(*hf_prod,*report);
    
    hbhe_prod->swap(filtered_hbhe);
    ho_prod->swap(filtered_ho);
    hf_prod->swap(filtered_hf);    
  }


  // Step D: Put outputs into event
  // just until the sorting is proven
  hbhe_prod->sort();
  ho_prod->sort();
  hf_prod->sort();
  htp_prod->sort();

  e.put(hbhe_prod);
  e.put(ho_prod);
  e.put(hf_prod);
  e.put(htp_prod);

  /// calib
  if (unpackCalib_) {
    std::auto_ptr<HcalCalibDigiCollection> hc_prod(new HcalCalibDigiCollection());
    hc_prod->swap_contents(hc);
    hc_prod->sort();
    e.put(hc_prod);
  }

  /// zdc
  if (unpackZDC_) {
    std::auto_ptr<ZDCDigiCollection> prod(new ZDCDigiCollection());
    prod->swap_contents(zdc);
    prod->sort();
    e.put(prod);
  }
  e.put(report);
}


