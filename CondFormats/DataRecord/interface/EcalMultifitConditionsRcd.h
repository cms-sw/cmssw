#ifndef CondFormats_DataRecord_EcalMultifitConditionsRcd_h
#define CondFormats_DataRecord_EcalMultifitConditionsRcd_h

#include "FWCore/Framework/interface/DependentRecordImplementation.h"

#include "CondFormats/DataRecord/interface/EcalGainRatiosRcd.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/EcalPulseCovariancesRcd.h"
#include "CondFormats/DataRecord/interface/EcalPulseShapesRcd.h"
#include "CondFormats/DataRecord/interface/EcalSampleMaskRcd.h"
#include "CondFormats/DataRecord/interface/EcalSamplesCorrelationRcd.h"
#include "CondFormats/DataRecord/interface/EcalTimeBiasCorrectionsRcd.h"
#include "CondFormats/DataRecord/interface/EcalTimeCalibConstantsRcd.h"
#include "CondFormats/DataRecord/interface/EcalTimeOffsetConstantRcd.h"

class EcalMultifitConditionsRcd
    : public edm::eventsetup::DependentRecordImplementation<EcalMultifitConditionsRcd,
                                                            edm::mpl::Vector<EcalGainRatiosRcd,
                                                                             EcalPedestalsRcd,
                                                                             EcalPulseCovariancesRcd,
                                                                             EcalPulseShapesRcd,
                                                                             EcalSampleMaskRcd,
                                                                             EcalSamplesCorrelationRcd,
                                                                             EcalTimeBiasCorrectionsRcd,
                                                                             EcalTimeCalibConstantsRcd,
                                                                             EcalTimeOffsetConstantRcd>> {};
#endif
