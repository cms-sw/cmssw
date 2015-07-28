#include "HGCalFETriggerDigiProducer.h"

#include "DataFormats/L1THGCal/interface/HGCFETriggerDigi.h"
#include "DataFormats/L1THGCal/interface/HGCFETriggerDigiFwd.h"
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"

#include <sstream>

HGCalFETriggerDigiProducer::
HGCalFETriggerDigiProducer(const edm::ParameterSet& conf) {

  inputee_ = consumes<HGCEEDigiCollection>(conf.getParameter<edm::InputTag>("eeDigis")); 
  inputfh_ = consumes<HGCHEDigiCollection>(conf.getParameter<edm::InputTag>("fhDigis")); 
  inputbh_ = consumes<HGCHEDigiCollection>(conf.getParameter<edm::InputTag>("bhDigis")); 
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

  produces<l1t::HGCFETriggerDigiCollection>();
}

void HGCalFETriggerDigiProducer::beginRun(const edm::Run& /*run*/, 
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

void HGCalFETriggerDigiProducer::produce(edm::Event& e, const edm::EventSetup& es) {
  std::auto_ptr<l1t::HGCFETriggerDigiCollection> 
    output( new l1t::HGCFETriggerDigiCollection );
  
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
    codec_->unSetDataPayload();
    output->push_back(l1t::HGCFETriggerDigi());
    l1t::HGCFETriggerDigi& digi = output->back();
    codec_->setDataPayload(*(module.second),ee_digis,fh_digis,bh_digis);
    codec_->encode(digi);
    std::stringstream output;
    codec_->print(digi,output);
    edm::LogInfo("HGCalFETriggerDigiProducer|EncodedDigiInfo")
      << output.str();
  }

  e.put(output);
}
