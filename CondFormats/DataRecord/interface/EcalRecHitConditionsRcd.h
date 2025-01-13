#ifndef CondFormats_DataRecord_EcalRecHitConditionsRcd_h
#define CondFormats_DataRecord_EcalRecHitConditionsRcd_h

#include "FWCore/Framework/interface/DependentRecordImplementation.h"

#include "CondFormats/DataRecord/interface/EcalADCToGeVConstantRcd.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"
#include "CondFormats/DataRecord/interface/EcalLaserAPDPNRatiosRcd.h"
#include "CondFormats/DataRecord/interface/EcalLaserAPDPNRatiosRefRcd.h"
#include "CondFormats/DataRecord/interface/EcalLaserAlphasRcd.h"
#include "CondFormats/DataRecord/interface/EcalLinearCorrectionsRcd.h"
#include "CondFormats/DataRecord/interface/EcalTimeCalibConstantsRcd.h"
#include "CondFormats/DataRecord/interface/EcalTimeOffsetConstantRcd.h"

class EcalRecHitConditionsRcd
    : public edm::eventsetup::DependentRecordImplementation<EcalRecHitConditionsRcd,
                                                            edm::mpl::Vector<EcalADCToGeVConstantRcd,
                                                                             EcalChannelStatusRcd,
                                                                             EcalIntercalibConstantsRcd,
                                                                             EcalLaserAPDPNRatiosRcd,
                                                                             EcalLaserAPDPNRatiosRefRcd,
                                                                             EcalLaserAlphasRcd,
                                                                             EcalLinearCorrectionsRcd,
                                                                             EcalTimeCalibConstantsRcd,
                                                                             EcalTimeOffsetConstantRcd>> {};
#endif
