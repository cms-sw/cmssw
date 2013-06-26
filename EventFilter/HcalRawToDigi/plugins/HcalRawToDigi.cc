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
  unpacker_(conf.getUntrackedParameter<int>("HcalFirstFED",int(FEDNumbering::MINHCALFEDID)),conf.getParameter<int>("firstSample"),conf.getParameter<int>("lastSample")),
  filter_(conf.getParameter<bool>("FilterDataQuality"),conf.getParameter<bool>("FilterDataQuality"),
	  false,
	  0, 0, 
	  -1),
  fedUnpackList_(conf.getUntrackedParameter<std::vector<int> >("FEDs", std::vector<int>())),
  firstFED_(conf.getUntrackedParameter<int>("HcalFirstFED",FEDNumbering::MINHCALFEDID)),
  unpackCalib_(conf.getUntrackedParameter<bool>("UnpackCalib",false)),
  unpackZDC_(conf.getUntrackedParameter<bool>("UnpackZDC",false)),
  unpackTTP_(conf.getUntrackedParameter<bool>("UnpackTTP",false)),
  silent_(conf.getUntrackedParameter<bool>("silent",true)),
  complainEmptyData_(conf.getUntrackedParameter<bool>("ComplainEmptyData",false)),
  unpackerMode_(conf.getUntrackedParameter<int>("UnpackerMode",0)),
  expectedOrbitMessageTime_(conf.getUntrackedParameter<int>("ExpectedOrbitMessageTime",-1))
{
  if (fedUnpackList_.empty()) {
    for (int i=FEDNumbering::MINHCALFEDID; i<=FEDNumbering::MAXHCALFEDID; i++)
      fedUnpackList_.push_back(i);
  } 
  
  unpacker_.setExpectedOrbitMessageTime(expectedOrbitMessageTime_);
  unpacker_.setMode(unpackerMode_);
  std::ostringstream ss;
  for (unsigned int i=0; i<fedUnpackList_.size(); i++) 
    ss << fedUnpackList_[i] << " ";
  edm::LogInfo("HCAL") << "HcalRawToDigi will unpack FEDs ( " << ss.str() << ")";
    
  // products produced...
  produces<HBHEDigiCollection>();
  produces<HFDigiCollection>();
  produces<HODigiCollection>();
  produces<HcalTrigPrimDigiCollection>();
  produces<HOTrigPrimDigiCollection>();
  produces<HcalUnpackerReport>();
  if (unpackCalib_)
    produces<HcalCalibDigiCollection>();
  if (unpackZDC_)
    produces<ZDCDigiCollection>();
  if (unpackTTP_)
    produces<HcalTTPDigiCollection>();

  memset(&stats_,0,sizeof(stats_));

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
  std::vector<HcalTTPDigi> ttp;
  std::vector<HOTriggerPrimitiveDigi> hotp;
  std::auto_ptr<HcalUnpackerReport> report(new HcalUnpackerReport);

  // Heuristics: use ave+(max-ave)/8
  if (stats_.max_hbhe>0) hbhe.reserve(stats_.ave_hbhe+(stats_.max_hbhe-stats_.ave_hbhe)/8);
  if (stats_.max_ho>0) ho.reserve(stats_.ave_ho+(stats_.max_ho-stats_.ave_ho)/8);
  if (stats_.max_hf>0) hf.reserve(stats_.ave_hf+(stats_.max_hf-stats_.ave_hf)/8);
  if (stats_.max_calib>0) hc.reserve(stats_.ave_calib+(stats_.max_calib-stats_.ave_calib)/8);
  if (stats_.max_tp>0) htp.reserve(stats_.ave_tp+(stats_.max_tp-stats_.ave_tp)/8);
  if (stats_.max_tpho>0) hotp.reserve(stats_.ave_tpho+(stats_.max_tpho-stats_.ave_tpho)/8);

  if (unpackZDC_) zdc.reserve(24);


  HcalUnpacker::Collections colls;
  colls.hbheCont=&hbhe;
  colls.hoCont=&ho;
  colls.hfCont=&hf;
  colls.tpCont=&htp;
  colls.tphoCont=&hotp;
  colls.calibCont=&hc;
  colls.zdcCont=&zdc;
  if (unpackTTP_) colls.ttp=&ttp;
 
  // Step C: unpack all requested FEDs
  for (std::vector<int>::const_iterator i=fedUnpackList_.begin(); i!=fedUnpackList_.end(); i++) {
    const FEDRawData& fed = rawraw->FEDData(*i);
    if (fed.size()==0) {
      if (complainEmptyData_) {
	if (!silent_) edm::LogWarning("EmptyData") << "No data for FED " << *i;
	report->addError(*i);
      }
    } else if (fed.size()<8*3) {
      if (!silent_) edm::LogWarning("EmptyData") << "Tiny data " << fed.size() << " for FED " << *i;
      report->addError(*i);
    } else {
      try {
	unpacker_.unpack(fed,*readoutMap,colls, *report,silent_);
	report->addUnpacked(*i);
      } catch (cms::Exception& e) {
	if (!silent_) edm::LogWarning("Unpacking error") << e.what();
	report->addError(*i);
      } catch (...) {
	if (!silent_) edm::LogWarning("Unpacking exception");
	report->addError(*i);
      }
    }
  }


  // gather statistics
  stats_.max_hbhe=std::max(stats_.max_hbhe,(int)hbhe.size());
  stats_.ave_hbhe=(stats_.ave_hbhe*stats_.n+hbhe.size())/(stats_.n+1);
  stats_.max_ho=std::max(stats_.max_ho,(int)ho.size());
  stats_.ave_ho=(stats_.ave_ho*stats_.n+ho.size())/(stats_.n+1);
  stats_.max_hf=std::max(stats_.max_hf,(int)hf.size());
  stats_.ave_hf=(stats_.ave_hf*stats_.n+hf.size())/(stats_.n+1);
  stats_.max_tp=std::max(stats_.max_tp,(int)htp.size());
  stats_.ave_tp=(stats_.ave_tp*stats_.n+htp.size())/(stats_.n+1);
  stats_.max_tpho=std::max(stats_.max_tpho,(int)hotp.size());
  stats_.ave_tpho=(stats_.ave_tpho*stats_.n+hotp.size())/(stats_.n+1);
  stats_.max_calib=std::max(stats_.max_calib,(int)hc.size());
  stats_.ave_calib=(stats_.ave_calib*stats_.n+hc.size())/(stats_.n+1);


  stats_.n++;

  // Step B: encapsulate vectors in actual collections
  std::auto_ptr<HBHEDigiCollection> hbhe_prod(new HBHEDigiCollection()); 
  std::auto_ptr<HFDigiCollection> hf_prod(new HFDigiCollection());
  std::auto_ptr<HODigiCollection> ho_prod(new HODigiCollection());
  std::auto_ptr<HcalTrigPrimDigiCollection> htp_prod(new HcalTrigPrimDigiCollection());  
  std::auto_ptr<HOTrigPrimDigiCollection> hotp_prod(new HOTrigPrimDigiCollection());  

  hbhe_prod->swap_contents(hbhe);
  hf_prod->swap_contents(hf);
  ho_prod->swap_contents(ho);
  htp_prod->swap_contents(htp);
  hotp_prod->swap_contents(hotp);

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
  hotp_prod->sort();

  e.put(hbhe_prod);
  e.put(ho_prod);
  e.put(hf_prod);
  e.put(htp_prod);
  e.put(hotp_prod);

  /// calib
  if (unpackCalib_) {
    std::auto_ptr<HcalCalibDigiCollection> hc_prod(new HcalCalibDigiCollection());
    hc_prod->swap_contents(hc);

    if (filter_.active()) {
      HcalCalibDigiCollection filtered_calib=filter_.filter(*hc_prod,*report);
      hc_prod->swap(filtered_calib);
    }

    hc_prod->sort();
    e.put(hc_prod);
  }

  /// zdc
  if (unpackZDC_) {
    std::auto_ptr<ZDCDigiCollection> prod(new ZDCDigiCollection());
    prod->swap_contents(zdc);
    
    if (filter_.active()) {
      ZDCDigiCollection filtered_zdc=filter_.filter(*prod,*report);
      prod->swap(filtered_zdc);
    }

    prod->sort();
    e.put(prod);
  }

  if (unpackTTP_) {
    std::auto_ptr<HcalTTPDigiCollection> prod(new HcalTTPDigiCollection());
    prod->swap_contents(ttp);
    
    prod->sort();
    e.put(prod);
  }
  e.put(report);


}


