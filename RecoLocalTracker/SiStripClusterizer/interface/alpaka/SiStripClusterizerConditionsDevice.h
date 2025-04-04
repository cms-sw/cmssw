#ifndef RecoLocalTracker_SiStripClusterizer_interface_alpaka_SiStripClusterizerConditionsDevice_h
#define RecoLocalTracker_SiStripClusterizer_interface_alpaka_SiStripClusterizerConditionsDevice_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
// #include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripClusterizerConditionsHost.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripClusterizerConditionsSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip {
  using namespace ::sistrip;
  using SiStripClusterizerConditionsDetToFedsDevice = PortableCollection<SiStripClusterizerConditionsDetToFedsSoA>;
  using SiStripClusterizerConditionsDataDevice = PortableCollection3<SiStripClusterizerConditionsData_fedchSoA,
                                                                     SiStripClusterizerConditionsData_stripSoA,
                                                                     SiStripClusterizerConditionsData_apvSoA>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip

// check that the sistrip device collection for the host device is the same as the sistrip host collection
ASSERT_DEVICE_MATCHES_HOST_COLLECTION(sistrip::SiStripClusterizerConditionsDetToFedsDevice,
                                      sistrip::SiStripClusterizerConditionsDetToFedsHost);
ASSERT_DEVICE_MATCHES_HOST_COLLECTION(sistrip::SiStripClusterizerConditionsDataDevice,
                                      sistrip::SiStripClusterizerConditionsDataHost);

#endif  // RecoLocalTracker_SiStripClusterizer_interface_alpaka_SiStripClusterizerConditionsDevice_h
