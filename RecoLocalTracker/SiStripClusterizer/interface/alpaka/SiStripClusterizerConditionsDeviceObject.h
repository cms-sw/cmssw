#ifndef RecoLocalTracker_SiStripClusterizer_interface_alpaka_SiStripClusterizerConditionsDeviceObject_h
#define RecoLocalTracker_SiStripClusterizer_interface_alpaka_SiStripClusterizerConditionsDeviceObject_h

#include "DataFormats/Portable/interface/alpaka/PortableObject.h"

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripClusterizerConditionsHostObject.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripClusterizerConditionsStruct.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace sistrip {
    // make the names from the top-level sistrip namespace visible for unqualified lookup
    using namespace ::sistrip;
    using SiStripClusterizerConditionsDetToFedsDeviceObject = PortableObject<DetToFeds>;
    using SiStripClusterizerConditionsGainNoiseCalsDeviceObject = PortableObject<GainNoiseCals>;
  }  // namespace sistrip

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

// check that the portable device collection for the host device is the same as the portable host collection
ASSERT_DEVICE_MATCHES_HOST_COLLECTION(sistrip::SiStripClusterizerConditionsDetToFedsDeviceObject,
                                      sistrip::SiStripClusterizerConditionsDetToFedsHostObject);
ASSERT_DEVICE_MATCHES_HOST_COLLECTION(sistrip::SiStripClusterizerConditionsGainNoiseCalsDeviceObject,
                                      sistrip::SiStripClusterizerConditionsGainNoiseCalsHostObject);

#endif
