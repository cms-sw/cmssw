#ifndef DataFormats_FTLRecHitSoA_interface_alpaka_ETLRecHitDeviceCollection_h
#define DataFormats_FTLRecHitSoA_interface_alpaka_ETLRecHitDeviceCollection_h

#include "DataFormats/FTLRecHitSoA/interface/ETLRecHitHostCollection.h"
#include "DataFormats/FTLRecHitSoA/interface/ETLRecHitSoA.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::etlrechit {

  using ETLRecHitDeviceCollection = PortableCollection<::etlrechit::ETLRecHitSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::etlrechit

// Check that the portable device collection for the host device is the same as the portable host collection.
ASSERT_DEVICE_MATCHES_HOST_COLLECTION(etlrechit::ETLRecHitDeviceCollection, etlrechit::ETLRecHitHostCollection);

#endif  // DataFormats_FTLRecHitSoA_interface_alpaka_ETLRecHitDeviceCollection_h
