#ifndef RecoLocalCalo_HcalRecProducers_plugins_alpaka_Mahi_h
#define RecoLocalCalo_HcalRecProducers_plugins_alpaka_Mahi_h

#include <vector>

#include "DataFormats/HcalDigi/interface/alpaka/HcalDigiDeviceCollection.h"
#include "DataFormats/HcalRecHit/interface/alpaka/HcalRecHitDeviceCollection.h"
#include "CondFormats/HcalObjects/interface/alpaka/HcalMahiConditionsDevice.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/traits.h"
#include "CondFormats/HcalObjects/interface/alpaka/HcalMahiConditionsDevice.h"
#include "CondFormats/HcalObjects/interface/alpaka/HcalSiPMCharacteristicsDevice.h"
#include "CondFormats/HcalObjects/interface/alpaka/HcalRecoParamWithPulseShapeDevice.h"

#include "HcalMahiPulseOffsetsSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::hcal::reconstruction {

  struct ConfigParameters {
    uint32_t maxTimeSamples;
    uint32_t kprep1dChannelsPerBlock;
    int sipmQTSShift;
    int sipmQNTStoSum;
    int firstSampleShift;
    bool useEffectivePedestals;

    float meanTime;
    float timeSigmaSiPM, timeSigmaHPD;
    float ts4Thresh;

    std::array<uint32_t, 3> kernelMinimizeThreads;

    // FIXME:
    //   - add "getters" to HcalTimeSlew calib formats
    //   - add ES Producer to consume what is produced above not to replicate.
    //   which ones to use is hardcoded, therefore no need to send those to the device
    bool applyTimeSlew;
    float tzeroTimeSlew, slopeTimeSlew, tmaxTimeSlew;
  };

  using IProductTypef01 = hcal::Phase1DigiDeviceCollection;
  using IProductTypef5 = hcal::Phase0DigiDeviceCollection;
  using IProductTypef3 = hcal::Phase1DigiDeviceCollection;
  using OProductType = hcal::RecHitDeviceCollection;

  void runMahiAsync(Queue& queue,
                    IProductTypef01::ConstView const& f01HEDigis,
                    IProductTypef5::ConstView const& f5HBDigis,
                    IProductTypef3::ConstView const& f3HBDigis,
                    OProductType::View outputGPU,
                    HcalMahiConditionsPortableDevice::ConstView const& mahi,
                    HcalSiPMCharacteristicsPortableDevice::ConstView const& sipmCharacteristics,
                    HcalRecoParamWithPulseShapeDevice::ConstView const& recoParams,
                    HcalMahiPulseOffsetsSoA::ConstView const& mahiPulseOffsets,
                    ConfigParameters const& configParameters);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::hcal::reconstruction
#endif
