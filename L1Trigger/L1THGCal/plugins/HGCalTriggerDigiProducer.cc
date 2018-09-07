#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "DataFormats/L1THGCal/interface/HGCFETriggerDigi.h"
#include "DataFormats/L1THGCal/interface/HGCFETriggerDigiDefs.h"
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerFECodecBase.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerBackendProcessor.h"

#include <memory>

class HGCalTriggerDigiProducer : public edm::stream::EDProducer<> {  
 public:    
  HGCalTriggerDigiProducer(const edm::ParameterSet&);
  ~HGCalTriggerDigiProducer() override { }
  
  void beginRun(const edm::Run&, 
                        const edm::EventSetup&) override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  
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
  inputee_(consumes<HGCalDigiCollection>(conf.getParameter<edm::InputTag>("eeDigis"))),
  inputfh_(consumes<HGCalDigiCollection>(conf.getParameter<edm::InputTag>("fhDigis"))), 
  inputbh_(consumes<HGCalDigiCollection>(conf.getParameter<edm::InputTag>("bhDigis"))), 
  backEndProcessor_(new HGCalTriggerBackendProcessor(conf.getParameterSet("BEConfiguration"),consumesCollector()) )
{
  //setup FE codec
  const edm::ParameterSet& feCodecConfig = conf.getParameterSet("FECodec");
  const std::string& feCodecName = feCodecConfig.getParameter<std::string>("CodecName");
  HGCalTriggerFECodecBase* codec = HGCalTriggerFECodecFactory::get()->create(feCodecName,feCodecConfig);
  codec_.reset(codec);
  codec_->unSetDataPayload();
  
  produces<l1t::HGCFETriggerDigiCollection>();
  // register backend processor products
  backEndProcessor_->setProduces(*this);
}

void HGCalTriggerDigiProducer::beginRun(const edm::Run& /*run*/, 
                                          const edm::EventSetup& es) {
  es.get<CaloGeometryRecord>().get(triggerGeometry_);
  codec_->setGeometry(triggerGeometry_.product());
  backEndProcessor_->setGeometry(triggerGeometry_.product());

}

void HGCalTriggerDigiProducer::produce(edm::Event& e, const edm::EventSetup& es) {
  std::unique_ptr<l1t::HGCFETriggerDigiCollection> 
    fe_output( new l1t::HGCFETriggerDigiCollection );
  
  edm::Handle<HGCalDigiCollection> ee_digis_h;
  edm::Handle<HGCalDigiCollection> fh_digis_h;
  edm::Handle<HGCalDigiCollection> bh_digis_h;

  e.getByToken(inputee_,ee_digis_h);
  e.getByToken(inputfh_,fh_digis_h);
  e.getByToken(inputbh_,bh_digis_h);

  const HGCalDigiCollection& ee_digis = *ee_digis_h;
  const HGCalDigiCollection& fh_digis = *fh_digis_h;
  const HGCalDigiCollection& bh_digis = *bh_digis_h;

  // First find modules containing hits and prepare list of hits for each module
  std::unordered_map<uint32_t, HGCalDigiCollection> hit_modules_ee;
  for(const auto& eedata : ee_digis) {
    uint32_t module = triggerGeometry_->getModuleFromCell(eedata.id());
    if(triggerGeometry_->disconnectedModule(module)) continue;
    auto itr_insert = hit_modules_ee.emplace(module,HGCalDigiCollection());
    itr_insert.first->second.push_back(eedata);
  }
  std::unordered_map<uint32_t,HGCalDigiCollection> hit_modules_fh;
  for(const auto& fhdata : fh_digis) {
    uint32_t module = triggerGeometry_->getModuleFromCell(fhdata.id());
    if(triggerGeometry_->disconnectedModule(module)) continue;
    auto itr_insert = hit_modules_fh.emplace(module, HGCalDigiCollection());
    itr_insert.first->second.push_back(fhdata);
  }
  std::unordered_map<uint32_t,HGCalDigiCollection> hit_modules_bh;
  for(const auto& bhdata : bh_digis) {
    if(DetId(bhdata.id()).det()!=DetId::HGCalHSc && HcalDetId(bhdata.id()).subdetId()!=HcalEndcap) continue;
    uint32_t module = triggerGeometry_->getModuleFromCell(bhdata.id());
    if(triggerGeometry_->disconnectedModule(module)) continue;
    auto itr_insert = hit_modules_bh.emplace(module, HGCalDigiCollection());
    itr_insert.first->second.push_back(bhdata);
  }
  // loop on modules containing hits and call front-end processing
  // we produce one output trigger digi per module in the FE
  fe_output->reserve(hit_modules_ee.size() + hit_modules_fh.size() + hit_modules_bh.size());
  for( const auto& module_hits : hit_modules_ee ) {        
    fe_output->push_back(l1t::HGCFETriggerDigi());
    l1t::HGCFETriggerDigi& digi = fe_output->back();
    codec_->setDataPayload(module_hits.second,HGCalDigiCollection(),HGCalDigiCollection());
    codec_->encode(digi);
    digi.setDetId( DetId(module_hits.first) );
    codec_->unSetDataPayload(); 
  } //end loop on EE modules
  for( const auto& module_hits : hit_modules_fh ) {        
    fe_output->push_back(l1t::HGCFETriggerDigi());
    l1t::HGCFETriggerDigi& digi = fe_output->back();
    codec_->setDataPayload(HGCalDigiCollection(),module_hits.second,HGCalDigiCollection());
    codec_->encode(digi);
    digi.setDetId( DetId(module_hits.first) );
    codec_->unSetDataPayload();
  } //end loop on FH modules
  for( const auto& module_hits : hit_modules_bh ) {
    fe_output->push_back(l1t::HGCFETriggerDigi());
    l1t::HGCFETriggerDigi& digi = fe_output->back();
    codec_->setDataPayload(HGCalDigiCollection(),HGCalDigiCollection(),module_hits.second);
    codec_->encode(digi);
    digi.setDetId( DetId(module_hits.first) );
    codec_->unSetDataPayload();
  } //end loop on BH modules


  // get the orphan handle and fe digi collection
  auto fe_digis_handle = e.put(std::move(fe_output));
  auto fe_digis_coll = *fe_digis_handle;
  
  //now we run the emulation of the back-end processor
  backEndProcessor_->reset();
  backEndProcessor_->run(fe_digis_coll,es,e);
  backEndProcessor_->putInEvent(e);
}
