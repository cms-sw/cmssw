#ifndef DataFormats_EcalRecHit_EcalUncalibratedRecHitSoA_h
#define DataFormats_EcalRecHit_EcalUncalibratedRecHitSoA_h

#include "DataFormats/Common/interface/StdArray.h"
#include "DataFormats/EcalDigi/interface/EcalConstants.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

// due to a ROOT limitation the std::array needs to be wrapped
// https://github.com/root-project/root/issues/12007
using EcalOotAmpArray =
    edm::StdArray<float, ecalPh1::sampleSize>;  //number of OOT amplitudes currently=number of samples, to be revised

GENERATE_SOA_LAYOUT(EcalUncalibratedRecHitSoALayout,
                    SOA_COLUMN(uint32_t, id),
                    SOA_SCALAR(uint32_t, size),
                    SOA_COLUMN(float, amplitude),
                    SOA_COLUMN(float, amplitudeError),
                    SOA_COLUMN(float, pedestal),
                    SOA_COLUMN(float, jitter),
                    SOA_COLUMN(float, jitterError),
                    SOA_COLUMN(float, chi2),
                    SOA_COLUMN(float, OOTchi2),
                    SOA_COLUMN(uint32_t, flags),
                    SOA_COLUMN(uint32_t, aux),
                    SOA_COLUMN(EcalOotAmpArray, outOfTimeAmplitudes))

using EcalUncalibratedRecHitSoA = EcalUncalibratedRecHitSoALayout<>;

#endif
