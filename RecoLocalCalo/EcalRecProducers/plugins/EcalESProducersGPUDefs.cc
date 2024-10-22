#include "CondFormats/DataRecord/interface/EcalADCToGeVConstantRcd.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "CondFormats/DataRecord/interface/EcalGainRatiosRcd.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"
#include "CondFormats/DataRecord/interface/EcalLaserAPDPNRatiosRcd.h"
#include "CondFormats/DataRecord/interface/EcalLaserAPDPNRatiosRefRcd.h"
#include "CondFormats/DataRecord/interface/EcalLaserAlphasRcd.h"
#include "CondFormats/DataRecord/interface/EcalLinearCorrectionsRcd.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/EcalPulseCovariancesRcd.h"
#include "CondFormats/DataRecord/interface/EcalPulseShapesRcd.h"
#include "CondFormats/DataRecord/interface/EcalSamplesCorrelationRcd.h"
#include "CondFormats/DataRecord/interface/EcalTimeBiasCorrectionsRcd.h"
#include "CondFormats/DataRecord/interface/EcalTimeCalibConstantsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalGainRatiosGPU.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstantsGPU.h"
#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatiosGPU.h"
#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatiosRefGPU.h"
#include "CondFormats/EcalObjects/interface/EcalLaserAlphasGPU.h"
#include "CondFormats/EcalObjects/interface/EcalLinearCorrectionsGPU.h"
#include "CondFormats/EcalObjects/interface/EcalPedestalsGPU.h"
#include "CondFormats/EcalObjects/interface/EcalPulseCovariancesGPU.h"
#include "CondFormats/EcalObjects/interface/EcalPulseShapesGPU.h"
#include "CondFormats/EcalObjects/interface/EcalRechitADCToGeVConstantGPU.h"
#include "CondFormats/EcalObjects/interface/EcalRechitChannelStatusGPU.h"
#include "CondFormats/EcalObjects/interface/EcalSamplesCorrelationGPU.h"
#include "CondFormats/EcalObjects/interface/EcalTimeBiasCorrectionsGPU.h"
#include "CondFormats/EcalObjects/interface/EcalTimeCalibConstantsGPU.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "HeterogeneousCore/CUDACore/interface/ConvertingESProducerT.h"

using EcalPedestalsGPUESProducer = ConvertingESProducerT<EcalPedestalsRcd, EcalPedestalsGPU, EcalPedestals>;

using EcalGainRatiosGPUESProducer = ConvertingESProducerT<EcalGainRatiosRcd, EcalGainRatiosGPU, EcalGainRatios>;

using EcalPulseShapesGPUESProducer = ConvertingESProducerT<EcalPulseShapesRcd, EcalPulseShapesGPU, EcalPulseShapes>;

using EcalPulseCovariancesGPUESProducer =
    ConvertingESProducerT<EcalPulseCovariancesRcd, EcalPulseCovariancesGPU, EcalPulseCovariances>;

using EcalSamplesCorrelationGPUESProducer =
    ConvertingESProducerT<EcalSamplesCorrelationRcd, EcalSamplesCorrelationGPU, EcalSamplesCorrelation>;

using EcalTimeBiasCorrectionsGPUESProducer =
    ConvertingESProducerT<EcalTimeBiasCorrectionsRcd, EcalTimeBiasCorrectionsGPU, EcalTimeBiasCorrections>;

using EcalTimeCalibConstantsGPUESProducer =
    ConvertingESProducerT<EcalTimeCalibConstantsRcd, EcalTimeCalibConstantsGPU, EcalTimeCalibConstants>;

using EcalRechitADCToGeVConstantGPUESProducer =
    ConvertingESProducerT<EcalADCToGeVConstantRcd, EcalRechitADCToGeVConstantGPU, EcalADCToGeVConstant>;

using EcalIntercalibConstantsGPUESProducer =
    ConvertingESProducerT<EcalIntercalibConstantsRcd, EcalIntercalibConstantsGPU, EcalIntercalibConstants>;

using EcalRechitChannelStatusGPUESProducer =
    ConvertingESProducerT<EcalChannelStatusRcd, EcalRechitChannelStatusGPU, EcalChannelStatus>;

using EcalLaserAPDPNRatiosGPUESProducer =
    ConvertingESProducerT<EcalLaserAPDPNRatiosRcd, EcalLaserAPDPNRatiosGPU, EcalLaserAPDPNRatios>;

using EcalLaserAPDPNRatiosRefGPUESProducer =
    ConvertingESProducerT<EcalLaserAPDPNRatiosRefRcd, EcalLaserAPDPNRatiosRefGPU, EcalLaserAPDPNRatiosRef>;

using EcalLaserAlphasGPUESProducer = ConvertingESProducerT<EcalLaserAlphasRcd, EcalLaserAlphasGPU, EcalLaserAlphas>;

using EcalLinearCorrectionsGPUESProducer =
    ConvertingESProducerT<EcalLinearCorrectionsRcd, EcalLinearCorrectionsGPU, EcalLinearCorrections>;

//
// This below also creates the .py config files, as described in HeterogeneousCore/CUDACore/interface/ConvertingESProducerT.h
//

DEFINE_FWK_EVENTSETUP_MODULE(EcalPedestalsGPUESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(EcalGainRatiosGPUESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(EcalPulseShapesGPUESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(EcalPulseCovariancesGPUESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(EcalSamplesCorrelationGPUESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(EcalTimeBiasCorrectionsGPUESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(EcalTimeCalibConstantsGPUESProducer);

DEFINE_FWK_EVENTSETUP_MODULE(EcalRechitADCToGeVConstantGPUESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(EcalIntercalibConstantsGPUESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(EcalRechitChannelStatusGPUESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(EcalLaserAPDPNRatiosGPUESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(EcalLaserAPDPNRatiosRefGPUESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(EcalLaserAlphasGPUESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(EcalLinearCorrectionsGPUESProducer);
