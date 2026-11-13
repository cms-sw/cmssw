#ifndef DataFormats_CaloRecHit_CaloClusterDeviceCollection_h
#define DataFormats_CaloRecHit_CaloClusterDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterSoA.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterHostCollection.h"
#include <alpaka/alpaka.hpp>

namespace ALPAKA_ACCELERATOR_NAMESPACE::reco {

  using CaloClusterDeviceCollection = PortableCollection<::reco::CaloClusterSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::reco

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(reco::CaloClusterDeviceCollection, reco::CaloClusterHostCollection);

#endif
