#include "HcalRawESProducerGPU.h"

#include "CondFormats/DataRecord/interface/HcalElectronicsMapRcd.h"

#include "EventFilter/HcalRawToDigi/plugins/ElectronicsMappingGPU.h"

#include <iostream>

using HcalElectronicsMappingGPUESProducer =
    HcalRawESProducerGPU<hcal::raw::ElectronicsMappingGPU, HcalElectronicsMap, HcalElectronicsMapRcd>;

DEFINE_FWK_EVENTSETUP_MODULE(HcalElectronicsMappingGPUESProducer);
