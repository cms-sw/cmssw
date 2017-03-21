#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/L1THGCal/interface/HGCFETriggerDigi.h"
#include "DataFormats/L1THGCal/interface/HGCFETriggerDigiFwd.h"
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"

#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerFECodecBase.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerBackendProcessor.h"

#include <sstream>
#include <memory>

class HGCalTriggerDigiProducer : public edm::EDProducer {  
 public:    
  HGCalTriggerDigiProducer(const edm::ParameterSet&);
  ~HGCalTriggerDigiProducer() { }
  
  virtual void beginRun(const edm::Run&, 
                        const edm::EventSetup&);
  virtual void produce(edm::Event&, const edm::EventSetup&);
  
 private:
  // inputs
  edm::EDGetToken inputee_, inputfh_, inputbh_;
  edm::ESHandle<HGCalTriggerGeometryBase> triggerGeometry_;
  // algorithm containers
  std::unique_ptr<HGCalTriggerFECodecBase> codec_;
  std::unique_ptr<HGCalTriggerBackendProcessor> backEndProcessor_;
};

DEFINE_FWK_MODULE(HGCalTriggerDigiProducer);

HGCalTriggerDigiProducer::
HGCalTriggerDigiProducer(const edm::ParameterSet& conf):
  inputee_(consumes<HGCEEDigiCollection>(conf.getParameter<edm::InputTag>("eeDigis"))),
  inputfh_(consumes<HGCHEDigiCollection>(conf.getParameter<edm::InputTag>("fhDigis"))) 
  //inputbh_(consumes<HGCHEDigiCollection>(conf.getParameter<edm::InputTag>("bhDigis"))) 
{
  
  
  //setup FE codec
  const edm::ParameterSet& feCodecConfig = conf.getParameterSet("FECodec");
  const std::string& feCodecName = feCodecConfig.getParameter<std::string>("CodecName");
  HGCalTriggerFECodecBase* codec = HGCalTriggerFECodecFactory::get()->create(feCodecName,feCodecConfig);
  codec_.reset(codec);
  codec_->unSetDataPayload();
  
  produces<l1t::HGCFETriggerDigiCollection>();
  //setup BE processor
  backEndProcessor_ = std::make_unique<HGCalTriggerBackendProcessor>(conf.getParameterSet("BEConfiguration"));
  // register backend processor products
  backEndProcessor_->setProduces(*this);
}

void HGCalTriggerDigiProducer::beginRun(const edm::Run& /*run*/, 
                                          const edm::EventSetup& es) {
  es.get<IdealGeometryRecord>().get(triggerGeometry_);
  codec_->setGeometry(triggerGeometry_.product());
  backEndProcessor_->setGeometry(triggerGeometry_.product());

}

void HGCalTriggerDigiProducer::produce(edm::Event& e, const edm::EventSetup& es) {
  std::unique_ptr<l1t::HGCFETriggerDigiCollection> 
    fe_output( new l1t::HGCFETriggerDigiCollection );
  
  edm::Handle<HGCEEDigiCollection> ee_digis_h;
  edm::Handle<HGCHEDigiCollection> fh_digis_h, bh_digis_h;

  e.getByToken(inputee_,ee_digis_h);
  e.getByToken(inputfh_,fh_digis_h);
  //e.getByToken(inputbh_,bh_digis_h);

  const HGCEEDigiCollection& ee_digis = *ee_digis_h;
  const HGCHEDigiCollection& fh_digis = *fh_digis_h;
  //const HGCHEDigiCollection& bh_digis = *bh_digis_h;

  // First find modules containing hits and prepare list of hits for each module
  std::unordered_map<uint32_t, HGCEEDigiCollection> hit_modules_ee;
  for(const auto& eedata : ee_digis) {
    uint32_t module = triggerGeometry_->getModuleFromCell(eedata.id());
    auto itr_insert = hit_modules_ee.emplace(module,HGCEEDigiCollection());
    itr_insert.first->second.push_back(eedata);
  }
  std::unordered_map<uint32_t,HGCHEDigiCollection> hit_modules_fh;
  for(const auto& fhdata : fh_digis) {
    uint32_t module = triggerGeometry_->getModuleFromCell(fhdata.id());
    auto itr_insert = hit_modules_fh.emplace(module, HGCHEDigiCollection());
    itr_insert.first->second.push_back(fhdata);
  }
  // loop on modules containing hits and call front-end processing
  // we produce one output trigger digi per module in the FE
  fe_output->reserve(hit_modules_ee.size() + hit_modules_fh.size());
  std::stringstream output;
  for( const auto& module_hits : hit_modules_ee ) {        
    fe_output->push_back(l1t::HGCFETriggerDigi());
    l1t::HGCFETriggerDigi& digi = fe_output->back();
    codec_->setDataPayload(module_hits.second,HGCHEDigiCollection(),HGCHEDigiCollection());
    codec_->encode(digi);
    digi.setDetId( DetId(module_hits.first) );
    codec_->print(digi,output);
    edm::LogInfo("HGCalTriggerDigiProducer")
      << output.str();
    codec_->unSetDataPayload(); 
    output.str(std::string());
    output.clear();
  } //end loop on EE modules
  for( const auto& module_hits : hit_modules_fh ) {        
    fe_output->push_back(l1t::HGCFETriggerDigi());
    l1t::HGCFETriggerDigi& digi = fe_output->back();
    codec_->setDataPayload(HGCEEDigiCollection(),module_hits.second,HGCHEDigiCollection());
    codec_->encode(digi);
    digi.setDetId( DetId(module_hits.first) );
    codec_->print(digi,output);
    edm::LogInfo("HGCalTriggerDigiProducer")
      << output.str();
    codec_->unSetDataPayload();
    output.str(std::string());
    output.clear();
  } //end loop on FH modules


  // get the orphan handle and fe digi collection
  auto fe_digis_handle = e.put(std::move(fe_output));
  auto fe_digis_coll = *fe_digis_handle;
  
  //now we run the emulation of the back-end processor
  backEndProcessor_->run(fe_digis_coll, es);
  backEndProcessor_->putInEvent(e);
  backEndProcessor_->reset();  
}
