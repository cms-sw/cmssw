#ifndef CondFormats_SiStripObjects_interface_alpaka_SiStripClusterizerConditionsDevice_h
#define CondFormats_SiStripObjects_interface_alpaka_SiStripClusterizerConditionsDevice_h

#include "CondFormats/SiStripObjects/interface/SiStripClusterizerConditionsHost.h"
#include "CondFormats/SiStripObjects/interface/SiStripClusterizerConditionsSoA.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  // PortableCollection-based model
  using SiStripClusterizerConditionsHost = ::SiStripClusterizerConditionsHost;
  using SiStripClusterizerConditionsDevice = PortableMultiCollection<Device,
                                                                     SiStripClusterizerConditionsDetToFedsSoA,
                                                                     SiStripClusterizerConditionsData_fedchSoA,
                                                                     SiStripClusterizerConditionsData_stripSoA,
                                                                     SiStripClusterizerConditionsData_apvSoA>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

// check that the sistrip device collection for the host device is the same as the sistrip host collection
ASSERT_DEVICE_MATCHES_HOST_COLLECTION(SiStripClusterizerConditionsDevice, SiStripClusterizerConditionsHost);

#endif  // CondFormats_SiStripObjects_interface_alpaka_SiStripClusterizerConditionsDevice_h