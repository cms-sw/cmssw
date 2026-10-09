#ifndef DataFormats_EgammaReco_interface_SuperClusterHostCollection_h
#define DataFormats_EgammaReco_interface_SuperClusterHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/EgammaReco/interface/SuperClusterSoA.h"

namespace reco {

  // SoA with x, y, z, id fields in host memory
  using SuperClusterHostCollection = PortableHostCollection<SuperClusterSoA>;

}  // namespace reco

#endif  // DataFormats_EgammaReco_interface_SuperClusterHostCollection_h
