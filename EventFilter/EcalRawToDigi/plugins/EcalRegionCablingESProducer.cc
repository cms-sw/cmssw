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

   ReturnType erc( new EcalRegionCabling(conf_));

   return erc;
}
