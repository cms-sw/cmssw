#ifndef CondFormats_EcalObjects_EcalMultifitConditionsSoA_h
#define CondFormats_EcalObjects_EcalMultifitConditionsSoA_h

#include <array>
#include <Eigen/Dense>
#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"
#include "DataFormats/EcalDigi/interface/EcalConstants.h"
#include "CondFormats/EcalObjects/interface/EcalPulseShapes.h"

using PulseShapeArray = std::array<float, EcalPulseShape::TEMPLATESAMPLES>;
using SampleCorrelationArray = std::array<double, ecalPh1::sampleSize>;

using CovarianceMatrix = Eigen::Matrix<float, EcalPulseShape::TEMPLATESAMPLES, EcalPulseShape::TEMPLATESAMPLES>;

constexpr size_t kMaxTimeBiasCorrectionBinsEB = 71;
constexpr size_t kMaxTimeBiasCorrectionBinsEE = 58;
using TimeBiasCorrArrayEB = std::array<float, kMaxTimeBiasCorrectionBinsEB>;
using TimeBiasCorrArrayEE = std::array<float, kMaxTimeBiasCorrectionBinsEE>;

GENERATE_SOA_LAYOUT(EcalMultifitConditionsSoALayout,
                    SOA_COLUMN(uint32_t, rawid),
                    SOA_COLUMN(float, pedestals_mean_x12),
                    SOA_COLUMN(float, pedestals_mean_x6),
                    SOA_COLUMN(float, pedestals_mean_x1),
                    SOA_COLUMN(float, pedestals_rms_x12),
                    SOA_COLUMN(float, pedestals_rms_x6),
                    SOA_COLUMN(float, pedestals_rms_x1),
                    SOA_COLUMN(PulseShapeArray, pulseShapes),
                    // NxN  N=templatesamples  for each xtal
                    SOA_EIGEN_COLUMN(CovarianceMatrix, pulseCovariance),
                    SOA_COLUMN(float, gain12Over6),
                    SOA_COLUMN(float, gain6Over1),
                    SOA_COLUMN(float, timeCalibConstants),
                    // timeBiasCorrections (fixed since 2011)
                    SOA_SCALAR(TimeBiasCorrArrayEB, timeBiasCorrections_amplitude_EB),
                    SOA_SCALAR(TimeBiasCorrArrayEB, timeBiasCorrections_shift_EB),
                    SOA_SCALAR(TimeBiasCorrArrayEE, timeBiasCorrections_amplitude_EE),
                    SOA_SCALAR(TimeBiasCorrArrayEE, timeBiasCorrections_shift_EE),
                    SOA_SCALAR(size_t, timeBiasCorrectionSizeEB),
                    SOA_SCALAR(size_t, timeBiasCorrectionSizeEE),
                    // Sample correlation scalar: array of 10 values for each gain in EB and EE
                    SOA_SCALAR(SampleCorrelationArray, sampleCorrelation_EB_G12),
                    SOA_SCALAR(SampleCorrelationArray, sampleCorrelation_EB_G6),
                    SOA_SCALAR(SampleCorrelationArray, sampleCorrelation_EB_G1),
                    SOA_SCALAR(SampleCorrelationArray, sampleCorrelation_EE_G12),
                    SOA_SCALAR(SampleCorrelationArray, sampleCorrelation_EE_G6),
                    SOA_SCALAR(SampleCorrelationArray, sampleCorrelation_EE_G1),
                    // Samples Masks
                    SOA_SCALAR(unsigned int, sampleMask_EB),
                    SOA_SCALAR(unsigned int, sampleMask_EE),
                    SOA_SCALAR(float, timeOffset_EB),
                    SOA_SCALAR(float, timeOffset_EE),
                    // offset for hashed ID access to EE items of columns
                    SOA_SCALAR(uint32_t, offsetEE))

using EcalMultifitConditionsSoA = EcalMultifitConditionsSoALayout<>;

#endif
