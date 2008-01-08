#include "EventFilter/EcalRawToDigi/plugins/EcalRegionCablingESProducer.h"

EcalRegionCablingESProducer::EcalRegionCablingESProducer(const edm::ParameterSet& iConfig)
{
  conf_=iConfig;
  setWhatProduced(this);
}


EcalRegionCablingESProducer::~EcalRegionCablingESProducer(){}

EcalRegionCablingESProducer::ReturnType
EcalRegionCablingESProducer::produce(const EcalRegionCablingRecord & iRecord)
{
   using namespace edm::es;
   edm::ESHandle<EcalElectronicsMapping> mapping;
   iRecord.getRecord<EcalMappingRcd>().get(mapping);

   ReturnType erc( new EcalRegionCabling(conf_,
					 mapping.product()));

   return erc;
}
