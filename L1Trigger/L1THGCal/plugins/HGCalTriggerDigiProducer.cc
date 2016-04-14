#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/L1THGCal/interface/HGCFETriggerDigi.h"
#include "DataFormats/L1THGCal/interface/HGCFETriggerDigiFwd.h"
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"
#include "DataFormats/ForwardDetId/interface/HGCTriggerDetId.h"

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
  // algorithm containers
  std::unique_ptr<HGCalTriggerGeometryBase> triggerGeometry_;
  std::unique_ptr<HGCalTriggerFECodecBase> codec_;
  HGCalTriggerBackendProcessor backEndProcessor_;
};

DEFINE_FWK_MODULE(HGCalTriggerDigiProducer);

HGCalTriggerDigiProducer::
HGCalTriggerDigiProducer(const edm::ParameterSet& conf):
  inputee_(consumes<HGCEEDigiCollection>(conf.getParameter<edm::InputTag>("eeDigis"))),
  inputfh_(consumes<HGCHEDigiCollection>(conf.getParameter<edm::InputTag>("fhDigis"))), 
  inputbh_(consumes<HGCHEDigiCollection>(conf.getParameter<edm::InputTag>("bhDigis"))), 
  backEndProcessor_(conf.getParameterSet("BEConfiguration")) {
  
  //setup geometry configuration
  const edm::ParameterSet& geometryConfig = 
    conf.getParameterSet("TriggerGeometry");
  const std::string& trigGeomName = 
    geometryConfig.getParameter<std::string>("TriggerGeometryName");
  HGCalTriggerGeometryBase* geometry = 
    HGCalTriggerGeometryFactory::get()->create(trigGeomName,geometryConfig);
  triggerGeometry_.reset(geometry);
  
  //setup FE codec
  const edm::ParameterSet& feCodecConfig = 
    conf.getParameterSet("FECodec");
  const std::string& feCodecName = 
    feCodecConfig.getParameter<std::string>("CodecName");
  HGCalTriggerFECodecBase* codec =
    HGCalTriggerFECodecFactory::get()->create(feCodecName,feCodecConfig);
  codec_.reset(codec);
  codec_->unSetDataPayload();
  
  produces<l1t::HGCFETriggerDigiCollection>();
  // register backend processor products
  backEndProcessor_.setProduces(*this);
}

void HGCalTriggerDigiProducer::beginRun(const edm::Run& /*run*/, 
                                          const edm::EventSetup& es) {
  triggerGeometry_->reset();
  HGCalTriggerGeometryBase::es_info info;
  const std::string& ee_sd_name = triggerGeometry_->eeSDName();
  const std::string& fh_sd_name = triggerGeometry_->fhSDName();
  const std::string& bh_sd_name = triggerGeometry_->bhSDName();
  es.get<IdealGeometryRecord>().get(ee_sd_name,info.geom_ee);
  es.get<IdealGeometryRecord>().get(fh_sd_name,info.geom_fh);
  es.get<IdealGeometryRecord>().get(bh_sd_name,info.geom_bh);
  es.get<IdealGeometryRecord>().get(ee_sd_name,info.topo_ee);
  es.get<IdealGeometryRecord>().get(fh_sd_name,info.topo_fh);
  es.get<IdealGeometryRecord>().get(bh_sd_name,info.topo_bh);
  triggerGeometry_->initialize(info);
}

void HGCalTriggerDigiProducer::produce(edm::Event& e, const edm::EventSetup& es) {
  std::auto_ptr<l1t::HGCFETriggerDigiCollection> 
    fe_output( new l1t::HGCFETriggerDigiCollection );
  
  edm::Handle<HGCEEDigiCollection> ee_digis_h;
  edm::Handle<HGCHEDigiCollection> fh_digis_h, bh_digis_h;

  e.getByToken(inputee_,ee_digis_h);
  e.getByToken(inputfh_,fh_digis_h);
  e.getByToken(inputbh_,bh_digis_h);

  const HGCEEDigiCollection& ee_digis = *ee_digis_h;
  const HGCHEDigiCollection& fh_digis = *fh_digis_h;
  const HGCHEDigiCollection& bh_digis = *bh_digis_h;

  //we produce one output trigger digi per module in the FE
  //so we use the geometry to tell us what to loop over
  for( const auto& module : triggerGeometry_->modules() ) {    
    fe_output->push_back(l1t::HGCFETriggerDigi());
    l1t::HGCFETriggerDigi& digi = fe_output->back();
    codec_->setDataPayload(*(module.second),ee_digis,fh_digis,bh_digis);
    codec_->encode(digi);
    digi.setDetId( HGCTriggerDetId(module.first) );
    std::stringstream output;
    codec_->print(digi,output);
    edm::LogInfo("HGCalTriggerDigiProducer")
      << output.str();
    codec_->unSetDataPayload();
  }

  // get the orphan handle and fe digi collection
  auto fe_digis_handle = e.put(fe_output);
  auto fe_digis_coll = *fe_digis_handle;
  
  //now we run the emulation of the back-end processor
  backEndProcessor_.run(fe_digis_coll,triggerGeometry_);
  backEndProcessor_.putInEvent(e);
  backEndProcessor_.reset();  
}
