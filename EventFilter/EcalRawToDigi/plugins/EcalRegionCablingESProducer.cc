#include "EventFilter/EcalRawToDigi/plugins/EcalRegionCablingESProducer.h"

EcalRegionCablingESProducer::EcalRegionCablingESProducer(const edm::ParameterSet& iConfig) : conf_(iConfig) {
  auto cc = setWhatProduced(this);
  esEcalElectronicsMappingToken_ = cc.consumesFrom<EcalElectronicsMapping, EcalMappingRcd>();
}

EcalRegionCablingESProducer::~EcalRegionCablingESProducer() {}

EcalRegionCablingESProducer::ReturnType EcalRegionCablingESProducer::produce(const EcalRegionCablingRecord& iRecord) {
  edm::ESHandle<EcalElectronicsMapping> mapping = iRecord.getHandle(esEcalElectronicsMappingToken_);

  return std::make_unique<EcalRegionCabling>(conf_, mapping.product());
}
