#include <iostream>

#include "CondFormats/DataRecord/interface/HcalElectronicsMapRcd.h"

#include "ElectronicsMappingGPU.h"
#include "HcalRawESProducerGPU.h"

using HcalElectronicsMappingGPUESProducer =
    HcalRawESProducerGPU<hcal::raw::ElectronicsMappingGPU, HcalElectronicsMap, HcalElectronicsMapRcd>;

DEFINE_FWK_EVENTSETUP_MODULE(HcalElectronicsMappingGPUESProducer);
