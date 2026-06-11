#ifndef DataFormats_EgammaReco_interface_alpaka_SuperClusterDeviceCollection_h
#define DataFormats_EgammaReco_interface_alpaka_SuperClusterDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "DataFormats/EgammaReco/interface/SuperClusterSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace reco {

    using namespace ::reco;
    using SuperClusterDeviceCollection = PortableCollection<SuperClusterSoA>;

  }  // namespace reco

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // DataFormats_EgammaReco_interface_alpaka_SuperClusterDeviceCollection_h
