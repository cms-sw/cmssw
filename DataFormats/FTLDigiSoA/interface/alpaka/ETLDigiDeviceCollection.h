#ifndef DataFormats_FTLDigiSoA_interface_alpaka_ETLDigiDeviceCollection_h
#define DataFormats_FTLDigiSoA_interface_alpaka_ETLDigiDeviceCollection_h

#include "DataFormats/FTLDigiSoA/interface/ETLDigiHostCollection.h"
#include "DataFormats/FTLDigiSoA/interface/ETLDigiSoA.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::etldigi {

  // Make the names from the top-level etldigi namespace visible for unqualified lookup
  // inside the ALPAKA_ACCELERATOR_NAMESPACE::etldigi namespace.
  using ETLDigiDeviceCollection = PortableCollection<::etldigi::ETLDigiSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::etldigi

// Check that the portable device collection for the host device is the same as the portable host collection.
ASSERT_DEVICE_MATCHES_HOST_COLLECTION(etldigi::ETLDigiDeviceCollection, etldigi::ETLDigiHostCollection);

#endif  // DataFormats_FTLDigiSoA_interface_alpaka_ETLDigiDeviceCollection_h
