#ifndef DataFormats_EgammaReco_interface_alpaka_ElectronSeedDeviceCollection_h
#define DataFormats_EgammaReco_interface_alpaka_ElectronSeedDeviceCollection_h

#include "DataFormats/EgammaReco/interface/ElectronSeedHostCollection.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedSoA.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::reco {
  using namespace ::reco;
  using ElectronSeedDeviceCollection = PortableCollection<ElectronSeedSoA>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::reco

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(reco::ElectronSeedDeviceCollection, reco::ElectronSeedHostCollection);

#endif  // DataFormats_EgammaReco_interface_alpaka_ElectronSeedDeviceCollection_h
