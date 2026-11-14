#ifndef DataFormats_FTLRecHitSoA_interface_alpaka_ETLBaseRecHitDeviceCollection_h
#define DataFormats_FTLRecHitSoA_interface_alpaka_ETLBaseRecHitDeviceCollection_h

#include "DataFormats/FTLRecHitSoA/interface/ETLBaseRecHitHostCollection.h"
#include "DataFormats/FTLRecHitSoA/interface/ETLBaseRecHitSoA.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::etlrechit {

  using ETLBaseRecHitDeviceCollection = PortableCollection<::etlrechit::ETLBaseRecHitSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::etlrechit

// Check that the portable device collection for the host device is the same as the portable host collection.
ASSERT_DEVICE_MATCHES_HOST_COLLECTION(etlrechit::ETLBaseRecHitDeviceCollection, etlrechit::ETLBaseRecHitHostCollection);

#endif  // DataFormats_FTLRecHitSoA_interface_alpaka_ETLBaseRecHitDeviceCollection_h
