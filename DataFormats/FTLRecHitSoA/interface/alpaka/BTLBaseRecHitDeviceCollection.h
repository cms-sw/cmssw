#ifndef DataFormats_FTLRecHitSoA_interface_alpaka_BTLBaseRecHitDeviceCollection_h
#define DataFormats_FTLRecHitSoA_interface_alpaka_BTLBaseRecHitDeviceCollection_h

#include "DataFormats/FTLRecHitSoA/interface/BTLBaseRecHitHostCollection.h"
#include "DataFormats/FTLRecHitSoA/interface/BTLBaseRecHitSoA.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::btlrechit {

  // Make the names from the top-level btlrechit namespace visible for unqualified lookup
  // inside the ALPAKA_ACCELERATOR_NAMESPACE::btlrechit namespace.
  using namespace ::btlrechit;

  using BTLBaseRecHitDeviceCollection = PortableCollection<BTLBaseRecHitSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::btlrechit

// Check that the portable device collection for the host device is the same as the portable host collection.
ASSERT_DEVICE_MATCHES_HOST_COLLECTION(btlrechit::BTLBaseRecHitDeviceCollection, btlrechit::BTLBaseRecHitHostCollection);

#endif  // DataFormats_FTLRecHitSoA_interface_alpaka_BTLBaseRecHitDeviceCollection_h
