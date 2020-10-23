#include "EcalESProducerGPU.h"

// for uncalibrechit
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/EcalGainRatiosRcd.h"
#include "CondFormats/DataRecord/interface/EcalPulseShapesRcd.h"
#include "CondFormats/DataRecord/interface/EcalPulseCovariancesRcd.h"
#include "CondFormats/DataRecord/interface/EcalSamplesCorrelationRcd.h"
#include "CondFormats/DataRecord/interface/EcalTimeBiasCorrectionsRcd.h"
#include "CondFormats/DataRecord/interface/EcalTimeCalibConstantsRcd.h"
// for rechit
#include "CondFormats/DataRecord/interface/EcalADCToGeVConstantRcd.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "CondFormats/DataRecord/interface/EcalLaserAPDPNRatiosRcd.h"
#include "CondFormats/DataRecord/interface/EcalLaserAPDPNRatiosRefRcd.h"
#include "CondFormats/DataRecord/interface/EcalLaserAlphasRcd.h"
#include "CondFormats/DataRecord/interface/EcalLinearCorrectionsRcd.h"

// for uncalibrechit
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalPedestalsGPU.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalGainRatiosGPU.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalPulseShapesGPU.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalPulseCovariancesGPU.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSamplesCorrelationGPU.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalTimeBiasCorrectionsGPU.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalTimeCalibConstantsGPU.h"
// for rechit
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalRechitADCToGeVConstantGPU.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalIntercalibConstantsGPU.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalRechitChannelStatusGPU.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalLaserAPDPNRatiosGPU.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalLaserAPDPNRatiosRefGPU.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalLaserAlphasGPU.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalLinearCorrectionsGPU.h"

#include <iostream>

using EcalPedestalsGPUESProducer = EcalESProducerGPU<EcalPedestalsGPU, EcalPedestals, EcalPedestalsRcd>;

using EcalGainRatiosGPUESProducer = EcalESProducerGPU<EcalGainRatiosGPU, EcalGainRatios, EcalGainRatiosRcd>;

using EcalPulseShapesGPUESProducer = EcalESProducerGPU<EcalPulseShapesGPU, EcalPulseShapes, EcalPulseShapesRcd>;

using EcalPulseCovariancesGPUESProducer =
    EcalESProducerGPU<EcalPulseCovariancesGPU, EcalPulseCovariances, EcalPulseCovariancesRcd>;

using EcalSamplesCorrelationGPUESProducer =
    EcalESProducerGPU<EcalSamplesCorrelationGPU, EcalSamplesCorrelation, EcalSamplesCorrelationRcd>;

using EcalTimeBiasCorrectionsGPUESProducer =
    EcalESProducerGPU<EcalTimeBiasCorrectionsGPU, EcalTimeBiasCorrections, EcalTimeBiasCorrectionsRcd>;

using EcalTimeCalibConstantsGPUESProducer =
    EcalESProducerGPU<EcalTimeCalibConstantsGPU, EcalTimeCalibConstants, EcalTimeCalibConstantsRcd>;

using EcalRechitADCToGeVConstantGPUESProducer =
    EcalESProducerGPU<EcalRechitADCToGeVConstantGPU, EcalADCToGeVConstant, EcalADCToGeVConstantRcd>;

using EcalIntercalibConstantsGPUESProducer =
    EcalESProducerGPU<EcalIntercalibConstantsGPU, EcalIntercalibConstants, EcalIntercalibConstantsRcd>;

using EcalRechitChannelStatusGPUESProducer =
    EcalESProducerGPU<EcalRechitChannelStatusGPU, EcalChannelStatus, EcalChannelStatusRcd>;

using EcalLaserAPDPNRatiosGPUESProducer =
    EcalESProducerGPU<EcalLaserAPDPNRatiosGPU, EcalLaserAPDPNRatios, EcalLaserAPDPNRatiosRcd>;

using EcalLaserAPDPNRatiosRefGPUESProducer =
    EcalESProducerGPU<EcalLaserAPDPNRatiosRefGPU, EcalLaserAPDPNRatiosRef, EcalLaserAPDPNRatiosRefRcd>;

using EcalLaserAlphasGPUESProducer = EcalESProducerGPU<EcalLaserAlphasGPU, EcalLaserAlphas, EcalLaserAlphasRcd>;

using EcalLinearCorrectionsGPUESProducer =
    EcalESProducerGPU<EcalLinearCorrectionsGPU, EcalLinearCorrections, EcalLinearCorrectionsRcd>;

//
// This below also creates the .py config files, as described in "EcalESProducerGPU.h"
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
