
#pragma once

#include "DataFormats/CaloRecHit/interface/CaloClusterSoA.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include <alpaka/alpaka.hpp>

namespace reco {

  using CaloClusterHostCollection = PortableHostCollection<CaloClusterSoA>;

}  // namespace reco
