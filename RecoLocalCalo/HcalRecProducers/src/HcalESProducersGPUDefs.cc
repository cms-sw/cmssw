#include "CondFormats/DataRecord/interface/HcalCombinedRecordsGPU.h"
#include "CondFormats/DataRecord/interface/HcalGainWidthsRcd.h"
#include "CondFormats/DataRecord/interface/HcalGainsRcd.h"
#include "CondFormats/DataRecord/interface/HcalLUTCorrsRcd.h"
#include "CondFormats/DataRecord/interface/HcalPedestalWidthsRcd.h"
#include "CondFormats/DataRecord/interface/HcalPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/HcalQIEDataRcd.h"
#include "CondFormats/DataRecord/interface/HcalQIETypesRcd.h"
#include "CondFormats/DataRecord/interface/HcalRecoParamsRcd.h"
#include "CondFormats/DataRecord/interface/HcalRespCorrsRcd.h"
#include "CondFormats/DataRecord/interface/HcalSiPMCharacteristicsRcd.h"
#include "CondFormats/DataRecord/interface/HcalSiPMParametersRcd.h"
#include "CondFormats/DataRecord/interface/HcalTimeCorrsRcd.h"
#include "CondFormats/HcalObjects/interface/HcalConvertedEffectivePedestalWidthsGPU.h"
#include "CondFormats/HcalObjects/interface/HcalConvertedEffectivePedestalsGPU.h"
#include "CondFormats/HcalObjects/interface/HcalConvertedPedestalWidthsGPU.h"
#include "CondFormats/HcalObjects/interface/HcalConvertedPedestalsGPU.h"
#include "CondFormats/HcalObjects/interface/HcalGainWidths.h"
#include "CondFormats/HcalObjects/interface/HcalGainWidthsGPU.h"
#include "CondFormats/HcalObjects/interface/HcalGains.h"
#include "CondFormats/HcalObjects/interface/HcalGainsGPU.h"
#include "CondFormats/HcalObjects/interface/HcalLUTCorrs.h"
#include "CondFormats/HcalObjects/interface/HcalLUTCorrsGPU.h"
#include "CondFormats/HcalObjects/interface/HcalPedestalWidths.h"
#include "CondFormats/HcalObjects/interface/HcalPedestalWidthsGPU.h"
#include "CondFormats/HcalObjects/interface/HcalPedestals.h"
#include "CondFormats/HcalObjects/interface/HcalPedestalsGPU.h"
#include "CondFormats/HcalObjects/interface/HcalQIECodersGPU.h"
#include "CondFormats/HcalObjects/interface/HcalQIEData.h"
#include "CondFormats/HcalObjects/interface/HcalQIETypes.h"
#include "CondFormats/HcalObjects/interface/HcalQIETypesGPU.h"
#include "CondFormats/HcalObjects/interface/HcalRecoParams.h"
#include "CondFormats/HcalObjects/interface/HcalRecoParamsGPU.h"
#include "CondFormats/HcalObjects/interface/HcalRespCorrs.h"
#include "CondFormats/HcalObjects/interface/HcalRespCorrsGPU.h"
#include "CondFormats/HcalObjects/interface/HcalSiPMCharacteristics.h"
#include "CondFormats/HcalObjects/interface/HcalSiPMCharacteristicsGPU.h"
#include "CondFormats/HcalObjects/interface/HcalSiPMParameters.h"
#include "CondFormats/HcalObjects/interface/HcalSiPMParametersGPU.h"
#include "CondFormats/HcalObjects/interface/HcalTimeCorrs.h"
#include "CondFormats/HcalObjects/interface/HcalTimeCorrsGPU.h"
#include "HeterogeneousCore/CUDACore/interface/ConvertingESProducerT.h"
#include "HeterogeneousCore/CUDACore/interface/ConvertingESProducerWithDependenciesT.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalRecoParamsWithPulseShapesGPU.h"

using HcalRecoParamsGPUESProducer = ConvertingESProducerT<HcalRecoParamsRcd, HcalRecoParamsGPU, HcalRecoParams>;

using HcalRecoParamsWithPulseShapesGPUESProducer =
    ConvertingESProducerT<HcalRecoParamsRcd, HcalRecoParamsWithPulseShapesGPU, HcalRecoParams>;

using HcalPedestalsGPUESProducer = ConvertingESProducerT<HcalPedestalsRcd, HcalPedestalsGPU, HcalPedestals>;

using HcalGainsGPUESProducer = ConvertingESProducerT<HcalGainsRcd, HcalGainsGPU, HcalGains>;

using HcalLUTCorrsGPUESProducer = ConvertingESProducerT<HcalLUTCorrsRcd, HcalLUTCorrsGPU, HcalLUTCorrs>;

using HcalRespCorrsGPUESProducer = ConvertingESProducerT<HcalRespCorrsRcd, HcalRespCorrsGPU, HcalRespCorrs>;

using HcalTimeCorrsGPUESProducer = ConvertingESProducerT<HcalTimeCorrsRcd, HcalTimeCorrsGPU, HcalTimeCorrs>;

using HcalPedestalWidthsGPUESProducer =
    ConvertingESProducerT<HcalPedestalWidthsRcd, HcalPedestalWidthsGPU, HcalPedestalWidths>;

using HcalGainWidthsGPUESProducer = ConvertingESProducerT<HcalGainWidthsRcd, HcalGainWidthsGPU, HcalGainWidths>;

using HcalQIECodersGPUESProducer = ConvertingESProducerT<HcalQIEDataRcd, HcalQIECodersGPU, HcalQIEData>;

using HcalQIETypesGPUESProducer = ConvertingESProducerT<HcalQIETypesRcd, HcalQIETypesGPU, HcalQIETypes>;

using HcalSiPMParametersGPUESProducer =
    ConvertingESProducerT<HcalSiPMParametersRcd, HcalSiPMParametersGPU, HcalSiPMParameters>;

using HcalSiPMCharacteristicsGPUESProducer =
    ConvertingESProducerT<HcalSiPMCharacteristicsRcd, HcalSiPMCharacteristicsGPU, HcalSiPMCharacteristics>;

using HcalConvertedPedestalsGPUESProducer = ConvertingESProducerWithDependenciesT<HcalConvertedPedestalsRcd,
                                                                                  HcalConvertedPedestalsGPU,
                                                                                  HcalPedestals,
                                                                                  HcalQIEData,
                                                                                  HcalQIETypes>;

using HcalConvertedEffectivePedestalsGPUESProducer =
    ConvertingESProducerWithDependenciesT<HcalConvertedPedestalsRcd,
                                          HcalConvertedEffectivePedestalsGPU,
                                          HcalPedestals,
                                          HcalQIEData,
                                          HcalQIETypes>;

using HcalConvertedPedestalWidthsGPUESProducer = ConvertingESProducerWithDependenciesT<HcalConvertedPedestalWidthsRcd,
                                                                                       HcalConvertedPedestalWidthsGPU,
                                                                                       HcalPedestals,
                                                                                       HcalPedestalWidths,
                                                                                       HcalQIEData,
                                                                                       HcalQIETypes>;

using HcalConvertedEffectivePedestalWidthsGPUESProducer =
    ConvertingESProducerWithDependenciesT<HcalConvertedPedestalWidthsRcd,
                                          HcalConvertedEffectivePedestalWidthsGPU,
                                          HcalPedestals,
                                          HcalPedestalWidths,
                                          HcalQIEData,
                                          HcalQIETypes>;

DEFINE_FWK_EVENTSETUP_MODULE(HcalRecoParamsGPUESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(HcalRecoParamsWithPulseShapesGPUESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(HcalPedestalsGPUESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(HcalGainsGPUESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(HcalLUTCorrsGPUESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(HcalRespCorrsGPUESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(HcalTimeCorrsGPUESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(HcalPedestalWidthsGPUESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(HcalGainWidthsGPUESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(HcalQIECodersGPUESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(HcalQIETypesGPUESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(HcalSiPMParametersGPUESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(HcalSiPMCharacteristicsGPUESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(HcalConvertedPedestalsGPUESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(HcalConvertedEffectivePedestalsGPUESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(HcalConvertedPedestalWidthsGPUESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(HcalConvertedEffectivePedestalWidthsGPUESProducer);
