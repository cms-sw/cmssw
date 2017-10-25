#include "EventFilter/HcalRawToDigi/plugins/HcalRawToDigi.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalUMNioDigi.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <unordered_set>

HcalRawToDigi::HcalRawToDigi(edm::ParameterSet const& conf):
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
  unpackUMNio_(conf.getUntrackedParameter<bool>("UnpackUMNio",false)),
  silent_(conf.getUntrackedParameter<bool>("silent",true)),
  complainEmptyData_(conf.getUntrackedParameter<bool>("ComplainEmptyData",false)),
  unpackerMode_(conf.getUntrackedParameter<int>("UnpackerMode",0)),
  expectedOrbitMessageTime_(conf.getUntrackedParameter<int>("ExpectedOrbitMessageTime",-1))
{
  electronicsMapLabel_ = conf.getParameter<std::string>("ElectronicsMap");
  tok_data_ = consumes<FEDRawDataCollection>(conf.getParameter<edm::InputTag>("InputLabel"));

  if (fedUnpackList_.empty()) {
    // VME range for back-compatibility
    for (int i=FEDNumbering::MINHCALFEDID; i<=FEDNumbering::MAXHCALFEDID; i++)
      fedUnpackList_.push_back(i);

    // uTCA range 
    for (int i=FEDNumbering::MINHCALuTCAFEDID; i<=FEDNumbering::MAXHCALuTCAFEDID; i++)
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
  if (unpackUMNio_)
    produces<HcalUMNioDigi>();
  produces<QIE10DigiCollection>();
  produces<QIE11DigiCollection>();
  produces<QIE10DigiCollection>("ZDC");
  
  memset(&stats_,0,sizeof(stats_));

}

// Virtual destructor needed.
HcalRawToDigi::~HcalRawToDigi() { }  

void HcalRawToDigi::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<int>("HcalFirstFED",int(FEDNumbering::MINHCALFEDID));
  desc.add<int>("firstSample",0);
  desc.add<int>("lastSample",9);
  desc.add<bool>("FilterDataQuality",true);
  desc.addUntracked<std::vector<int>>("FEDs", std::vector<int>());
  desc.addUntracked<bool>("UnpackZDC",true);
  desc.addUntracked<bool>("UnpackCalib",true);
  desc.addUntracked<bool>("UnpackUMNio",true);
  desc.addUntracked<bool>("UnpackTTP",true);
  desc.addUntracked<bool>("silent",true);
  desc.addUntracked<bool>("ComplainEmptyData",false);
  desc.addUntracked<int>("UnpackerMode",0);
  desc.addUntracked<int>("ExpectedOrbitMessageTime",-1);
  desc.add<edm::InputTag>("InputLabel",edm::InputTag("rawDataCollector"));
  desc.add<std::string>("ElectronicsMap","");
  descriptions.add("hcalRawToDigi",desc);
}


// Functions that gets called by framework every event
void HcalRawToDigi::produce(edm::Event& e, const edm::EventSetup& es)
{
  // Step A: Get Inputs 
  edm::Handle<FEDRawDataCollection> rawraw;  
  e.getByToken(tok_data_,rawraw);
  // get the mapping
  edm::ESHandle<HcalDbService> pSetup;
  es.get<HcalDbRecord>().get( pSetup );
  edm::ESHandle<HcalElectronicsMap> item;
  es.get<HcalElectronicsMapRcd>().get(electronicsMapLabel_, item);
  const HcalElectronicsMap* readoutMap = item.product();
  filter_.setConditions(pSetup.product());
  
  // Step B: Create empty output  : three vectors for three classes...
  std::vector<HBHEDataFrame> hbhe;
  std::vector<HODataFrame> ho;
  std::vector<HFDataFrame> hf;
  std::vector<HcalTriggerPrimitiveDigi> htp;
  std::vector<HcalCalibDataFrame> hc;
  std::vector<ZDCDataFrame> zdc;
  std::vector<HcalTTPDigi> ttp;
  std::vector<HOTriggerPrimitiveDigi> hotp;
  HcalUMNioDigi umnio;
  auto report = std::make_unique<HcalUnpackerReport>();

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
  colls.umnio=&umnio;
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

  // check HF duplication
  std::unordered_set<uint32_t> cacheForHFdup;
  unsigned int cntHFdup = 0;
  for( auto & hf_digi : hf ){
     if( ! cacheForHFdup.insert(hf_digi.id().rawId()).second ) cntHFdup++;
  }
  if( cntHFdup ) edm::LogError("HcalRawToDigi") << "Duplicated HF digis found for "<<cntHFdup<<" times"<<std::endl;

  // Step B: encapsulate vectors in actual collections
  auto hbhe_prod = std::make_unique<HBHEDigiCollection>();
  auto hf_prod = std::make_unique<HFDigiCollection>();
  auto ho_prod = std::make_unique<HODigiCollection>();
  auto htp_prod = std::make_unique<HcalTrigPrimDigiCollection>();
  auto hotp_prod = std::make_unique<HOTrigPrimDigiCollection>();
  if (colls.qie10 == nullptr) {
    colls.qie10 = new QIE10DigiCollection(); 
  }
  std::unique_ptr<QIE10DigiCollection> qie10_prod(colls.qie10);
  if (colls.qie10ZDC == nullptr) {
    colls.qie10ZDC = new QIE10DigiCollection(); 
  }
  std::unique_ptr<QIE10DigiCollection> qie10ZDC_prod(colls.qie10ZDC);
  if (colls.qie11 == nullptr) {
    colls.qie11 = new QIE11DigiCollection(); 
  }
  std::unique_ptr<QIE11DigiCollection> qie11_prod(colls.qie11);

  hbhe_prod->swap_contents(hbhe);
  if( !cntHFdup ) hf_prod->swap_contents(hf);
  ho_prod->swap_contents(ho);
  htp_prod->swap_contents(htp);
  hotp_prod->swap_contents(hotp);

  // Step C2: filter FEDs, if required
  if (filter_.active()) {
    HBHEDigiCollection filtered_hbhe=filter_.filter(*hbhe_prod,*report);
    HODigiCollection filtered_ho=filter_.filter(*ho_prod,*report);
    HFDigiCollection filtered_hf=filter_.filter(*hf_prod,*report);
    QIE10DigiCollection filtered_qie10=filter_.filter(*qie10_prod,*report);
    QIE11DigiCollection filtered_qie11=filter_.filter(*qie11_prod,*report);
    
    hbhe_prod->swap(filtered_hbhe);
    ho_prod->swap(filtered_ho);
    hf_prod->swap(filtered_hf);    
    qie10_prod->swap(filtered_qie10);
    qie11_prod->swap(filtered_qie11);
  }


  // Step D: Put outputs into event
  // just until the sorting is proven
  hbhe_prod->sort();
  ho_prod->sort();
  hf_prod->sort();
  htp_prod->sort();
  hotp_prod->sort();
  qie10_prod->sort();
  qie10ZDC_prod->sort();
  qie11_prod->sort();

  e.put(std::move(hbhe_prod));
  e.put(std::move(ho_prod));
  e.put(std::move(hf_prod));
  e.put(std::move(htp_prod));
  e.put(std::move(hotp_prod));
  e.put(std::move(qie10_prod));
  e.put(std::move(qie10ZDC_prod),"ZDC");
  e.put(std::move(qie11_prod));

  /// calib
  if (unpackCalib_) {
    auto hc_prod = std::make_unique<HcalCalibDigiCollection>();
    hc_prod->swap_contents(hc);

    if (filter_.active()) {
      HcalCalibDigiCollection filtered_calib=filter_.filter(*hc_prod,*report);
      hc_prod->swap(filtered_calib);
    }

    hc_prod->sort();
    e.put(std::move(hc_prod));
  }

  /// zdc
  if (unpackZDC_) {
    auto prod = std::make_unique<ZDCDigiCollection>();
    prod->swap_contents(zdc);
    
    if (filter_.active()) {
      ZDCDigiCollection filtered_zdc=filter_.filter(*prod,*report);
      prod->swap(filtered_zdc);
    }

    prod->sort();
    e.put(std::move(prod));
  }

  if (unpackTTP_) {
    auto prod = std::make_unique<HcalTTPDigiCollection>();
    prod->swap_contents(ttp);
    
    prod->sort();
    e.put(std::move(prod));
  }
  e.put(std::move(report));
  /// umnio
  if (unpackUMNio_) {
    if(colls.umnio != nullptr) {
      e.put(std::make_unique<HcalUMNioDigi>(umnio));
    }

  }
}


