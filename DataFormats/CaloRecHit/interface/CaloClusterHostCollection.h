#ifndef DataFormats_CaloRecHit_CaloClusterHostCollection_h
#define DataFormats_CaloRecHit_CaloClusterHostCollection_h

#include "DataFormats/CaloRecHit/interface/CaloClusterSoA.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

namespace reco {

  using CaloClusterHostCollection = PortableHostCollection<CaloClusterSoA>;

}  // namespace reco

#endif
