#ifndef HeterogeneousCore_Tutorial_interface_alpaka_JetsSelectionDeviceCollection_h
#define HeterogeneousCore_Tutorial_interface_alpaka_JetsSelectionDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/Tutorial/interface/JetsSelectionHostCollection.h"
#include "HeterogeneousCore/Tutorial/interface/JetsSelectionSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::tutorial {

  // Make the names from the top-level tutorial namespace visible for unqualified lookup
  // inside the ALPAKA_ACCELERATOR_NAMESPACE::tutorial namespace.
  using namespace ::tutorial;

  using JetsSelectionDeviceCollection = PortableCollection<JetsSelectionSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::tutorial

// Check that the portable device collection for the host device is the same as the portable host collection.
ASSERT_DEVICE_MATCHES_HOST_COLLECTION(tutorial::JetsSelectionDeviceCollection, tutorial::JetsSelectionHostCollection);

#endif  // HeterogeneousCore_Tutorial_interface_alpaka_JetsSelectionDeviceCollection_h
