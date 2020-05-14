#include "EcalRawESProducerGPU.h"

#include "CondFormats/DataRecord/interface/EcalMappingElectronicsRcd.h"

#include "EventFilter/EcalRawToDigi/interface/ElectronicsMappingGPU.h"

#include <iostream>

using EcalElectronicsMappingGPUESProducer =
    EcalRawESProducerGPU<ecal::raw::ElectronicsMappingGPU, EcalMappingElectronics, EcalMappingElectronicsRcd>;

DEFINE_FWK_EVENTSETUP_MODULE(EcalElectronicsMappingGPUESProducer);
