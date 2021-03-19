#include "CondFormats/DataRecord/interface/HcalElectronicsMapRcd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "HeterogeneousCore/CUDACore/interface/ConvertingESProducerT.h"

#include "ElectronicsMappingGPU.h"

using HcalElectronicsMappingGPUESProducer =
    ConvertingESProducerT<HcalElectronicsMapRcd, hcal::raw::ElectronicsMappingGPU, HcalElectronicsMap>;

DEFINE_FWK_EVENTSETUP_MODULE(HcalElectronicsMappingGPUESProducer);
