#include "CondFormats/DataRecord/interface/EcalMappingElectronicsRcd.h"
#include "EventFilter/EcalRawToDigi/interface/ElectronicsMappingGPU.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "HeterogeneousCore/CUDACore/interface/ConvertingESProducerT.h"

using EcalElectronicsMappingGPUESProducer =
    ConvertingESProducerT<EcalMappingElectronicsRcd, ecal::raw::ElectronicsMappingGPU, EcalMappingElectronics>;

DEFINE_FWK_EVENTSETUP_MODULE(EcalElectronicsMappingGPUESProducer);
