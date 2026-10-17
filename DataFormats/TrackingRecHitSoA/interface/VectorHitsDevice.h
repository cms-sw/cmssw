#ifndef DataFormats_TrackingRecHitSoA_interface_VectorHitsDevice_h
#define DataFormats_TrackingRecHitSoA_interface_VectorHitsDevice_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Common/interface/Uninitialized.h"
#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/VectorHitsHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/VectorHitsSoA.h"

// TODO: The class is created via inheritance of the PortableCollection.
// This is generally discouraged, and should be done via composition.
// See: https://github.com/cms-sw/cmssw/pull/40465#discussion_r1067364306

namespace reco {

  template <typename TDev>
  using VectorHitPortableCollectionDevice = PortableDeviceCollection<TDev, reco::VectorHitBlocksSoA>;

  template <typename TDev>
  class VectorHitsDevice : public VectorHitPortableCollectionDevice<TDev> {
  public:
    VectorHitsDevice() = default;

    VectorHitsDevice(edm::Uninitialized) : VectorHitPortableCollectionDevice<TDev>{edm::kUninitialized} {}

    // The module block is allocated with nModules+1 elements and carries no payload: its extent is
    // the only record of the module count.
    template <typename TQueue>
    explicit VectorHitsDevice(TQueue queue, uint32_t nVectorHits, uint32_t nModules)
        : VectorHitPortableCollectionDevice<TDev>(queue, nVectorHits, nModules + 1) {}

    // Number of vector hits in the collection
    uint32_t nVectorHits() const { return this->view().vectorHits().metadata().size(); }

    // Number of stacked modules
    uint32_t nModules() const { return this->view().vectorHitModules().metadata().size() - 1; }
  };

}  // namespace reco

#endif  // DataFormats_TrackingRecHitSoA_interface_VectorHitsDevice_h
