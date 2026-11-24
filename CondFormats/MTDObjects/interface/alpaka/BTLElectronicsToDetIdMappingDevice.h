#ifndef CondFormats_MTDObjects_alpaka_BTLElectronicsToDetIdMappingDevice_h
#define CondFormats_MTDObjects_alpaka_BTLElectronicsToDetIdMappingDevice_h

#include "CondFormats/MTDObjects/interface/BTLElectronicsToDetIdMappingHost.h"
#include "CondFormats/MTDObjects/interface/BTLElectronicsToDetIdMappingSoA.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using ::BTLElectronicsToDetIdMappingHost;
  using BTLElectronicsToDetIdMappingDevice = PortableCollection<BTLElectronicsToDetIdMappingSoA>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

// check that the btl device collection is the same as the host collection
ASSERT_DEVICE_MATCHES_HOST_COLLECTION(BTLElectronicsToDetIdMappingDevice, BTLElectronicsToDetIdMappingHost);
#endif
