#ifndef DataFormats_EgammaReco_interface_ElectronSeedHostCollection_h
#define DataFormats_EgammaReco_interface_ElectronSeedHostCollection_h

#include <Eigen/Core>
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/TestSoA.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedSoA.h"

namespace reco {
  using ElectronSeedHostCollection = PortableHostCollection<ElectronSeedSoA>;
}  // namespace reco

#endif  // DataFormats_EgammaReco_interface_ElectronSeedHostCollection_h
