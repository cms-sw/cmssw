#include "HGCalFETriggerDigiProducer.h"

#include "DataFormats/L1THGCal/interface/HGCFETriggerDigi.h"
#include "DataFormats/L1THGCal/interface/HGCFETriggerDigiFwd.h"
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"

HGCalFETriggerDigiProducer::
HGCalFETriggerDigiProducer(const edm::ParameterSet& conf) {

  inputee_ = consumes<HGCEEDigiCollection>(conf.getParameter<edm::InputTag>("eeDigis")); 
  inputfh_ = consumes<HGCHEDigiCollection>(conf.getParameter<edm::InputTag>("fhDigis")); 
  inputbh_ = consumes<HGCHEDigiCollection>(conf.getParameter<edm::InputTag>("bhDigis")); 
  //setup geometry configuration
  const edm::ParameterSet& geometryConfig = 
    conf.getParameterSet("TriggerGeometry");
  const std::string& trigGeomName = 
    conf.getParameter<std::string>("TriggerGeometryName");
  HGCalTriggerGeometryBase* geometry = 
    HGCalTriggerGeometryFactory::get()->create(trigGeomName,geometryConfig);
  triggerGeometry_.reset(geometry);
  
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

  //we produce one output trigger digi per module in the FE
  //so we use the geometry to tell us what to loop over
  for( const auto& module : triggerGeometry_->modules() ) {
    std::cout << module.second->neighbours().size() << std::endl;
    output->push_back(l1t::HGCFETriggerDigi());
  }

  e.put(output);
}
