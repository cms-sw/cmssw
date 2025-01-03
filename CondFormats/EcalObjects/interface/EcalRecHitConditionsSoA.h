#ifndef CondFormats_EcalObjects_EcalRecHitConditionsSoA_h
#define CondFormats_EcalObjects_EcalRecHitConditionsSoA_h

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"
#include "DataFormats/Provenance/interface/Timestamp.h"

GENERATE_SOA_LAYOUT(EcalRecHitConditionsSoALayout,
                    SOA_COLUMN(uint32_t, rawid),
                    // energy intercalibrations
                    SOA_COLUMN(float, intercalibConstants),
                    // time intercalibrations
                    SOA_COLUMN(float, timeCalibConstants),
                    // channel status
                    SOA_COLUMN(uint16_t, channelStatus),
                    // laser APDPN ratios
                    SOA_COLUMN(float, laserAPDPNRatios_p1),
                    SOA_COLUMN(float, laserAPDPNRatios_p2),
                    SOA_COLUMN(float, laserAPDPNRatios_p3),
                    SOA_COLUMN(edm::TimeValue_t, laserAPDPNRatios_t1),
                    SOA_COLUMN(edm::TimeValue_t, laserAPDPNRatios_t2),
                    SOA_COLUMN(edm::TimeValue_t, laserAPDPNRatios_t3),
                    // laser APDPN reference
                    SOA_COLUMN(float, laserAPDPNref),
                    // laser alphas
                    SOA_COLUMN(float, laserAlpha),
                    // linear corrections
                    SOA_COLUMN(float, linearCorrections_p1),
                    SOA_COLUMN(float, linearCorrections_p2),
                    SOA_COLUMN(float, linearCorrections_p3),
                    SOA_COLUMN(edm::TimeValue_t, linearCorrections_t1),
                    SOA_COLUMN(edm::TimeValue_t, linearCorrections_t2),
                    SOA_COLUMN(edm::TimeValue_t, linearCorrections_t3),
                    // ADC to GeV constants
                    SOA_SCALAR(float, adcToGeVConstantEB),
                    SOA_SCALAR(float, adcToGeVConstantEE),
                    // time offsets constants
                    SOA_SCALAR(float, timeOffsetConstantEB),
                    SOA_SCALAR(float, timeOffsetConstantEE),
                    // offset for hashed ID access to EE items of columns
                    SOA_SCALAR(uint32_t, offsetEE))

using EcalRecHitConditionsSoA = EcalRecHitConditionsSoALayout<>;

#endif
