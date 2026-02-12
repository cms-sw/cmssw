#ifndef DataFormats_FTLDigiSoA_interface_alpaka_BTLDigiDeviceCollection_h
#define DataFormats_FTLDigiSoA_interface_alpaka_BTLDigiDeviceCollection_h

#include "DataFormats/FTLDigiSoA/interface/BTLDigiHostCollection.h"
#include "DataFormats/FTLDigiSoA/interface/BTLDigiSoA.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::btldigi {

  // Make the names from the top-level btldigi namespace visible for unqualified lookup
  // inside the ALPAKA_ACCELERATOR_NAMESPACE::btldigi namespace.
  using namespace ::btldigi;

  using BTLDigiDeviceCollection = PortableCollection<BTLDigiSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::btldigi

// Check that the portable device collection for the host device is the same as the portable host collection.
ASSERT_DEVICE_MATCHES_HOST_COLLECTION(btldigi::BTLDigiDeviceCollection, btldigi::BTLDigiHostCollection);

#endif  // DataFormats_FTLDigi_interface_alpaka_BTLDigiDeviceCollection_h
