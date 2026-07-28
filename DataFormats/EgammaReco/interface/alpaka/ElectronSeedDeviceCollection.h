#ifndef DataFormats_EgammaReco_interface_alpaka_ElectronSeedDeviceCollection_h
#define DataFormats_EgammaReco_interface_alpaka_ElectronSeedDeviceCollection_h

#include <Eigen/Core>
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/PortableTestObjects/interface/TestSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "DataFormats/EgammaReco/interface/ElectronSeedSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::reco {
  using namespace ::reco;
  using ElectronSeedDeviceCollection = PortableCollection<ElectronSeedSoA>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::reco

#endif  // DataFormats_EgammaReco_interface_alpaka_ElectronSeedDeviceCollection_h
