
#ifndef DataFormats_CaloRecHit_CaloClusterDeviceCollection_H
#define DataFormats_CaloRecHit_CaloClusterDeviceCollection_H

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterSoA.h"
#include <alpaka/alpaka.hpp>

namespace ALPAKA_ACCELERATOR_NAMESPACE::reco {

  using CaloClusterDeviceCollection = PortableCollection<::reco::CaloClusterSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::reco

#endif
