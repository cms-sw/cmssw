#ifndef CondFormats_SiStripObjects_interface_alpaka_SiStripMappingDevice_h
#define CondFormats_SiStripObjects_interface_alpaka_SiStripMappingDevice_h

#include "CondFormats/SiStripObjects/interface/SiStripMappingHost.h"
#include "CondFormats/SiStripObjects/interface/SiStripMappingSoA.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  // PortableCollection-based model
  using SiStripMappingHost = ::SiStripMappingHost;
  using SiStripMappingDevice = PortableCollection<SiStripMappingSoA>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

// check that the sistrip device collection for the host device is the same as the sistrip host collection
ASSERT_DEVICE_MATCHES_HOST_COLLECTION(SiStripMappingDevice, SiStripMappingHost);

#endif  // CondFormats_SiStripObjects_interface_alpaka_SiStripMappingDevice_h