
#ifndef DataFormats_CaloRecHit_CaloClusterHostCollection_H
#define DataFormats_CaloRecHit_CaloClusterHostCollection_H

#include "DataFormats/CaloRecHit/interface/CaloClusterSoA.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include <alpaka/alpaka.hpp>

namespace reco {

  using CaloClusterHostCollection = PortableHostCollection<CaloClusterSoA>;

}  // namespace reco

#endif
