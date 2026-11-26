#ifndef DataFormats_EgammaReco_interface_ElectronSeedHostCollection_h
#define DataFormats_EgammaReco_interface_ElectronSeedHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedSoA.h"

namespace reco {
  using ElectronSeedHostCollection = PortableHostCollection<ElectronSeedSoA>;
}  // namespace reco

#endif  // DataFormats_EgammaReco_interface_ElectronSeedHostCollection_h
