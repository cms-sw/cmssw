#ifndef DataFormats_TrackingRecHitSoA_interface_OTRecHitsDevice_h
#define DataFormats_TrackingRecHitSoA_interface_OTRecHitsDevice_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Common/interface/Uninitialized.h"
#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/OTRecHitsHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/OTRecHitsSoA.h"

namespace reco {

  template <typename TDev>
  using OTRecHitPortableCollectionDevice = PortableDeviceCollection<TDev, reco::OTRecHitBlocksSoA>;

  template <typename TDev>
  class OTRecHitsDevice : public OTRecHitPortableCollectionDevice<TDev> {
  public:
    OTRecHitsDevice() = default;

    OTRecHitsDevice(edm::Uninitialized) : OTRecHitPortableCollectionDevice<TDev>{edm::kUninitialized} {}

    // nHits, nModules: SoA sizes (nModules+1 elements for the cumulative-sum array).
    template <typename TQueue>
    explicit OTRecHitsDevice(TQueue queue, uint32_t nHits, uint32_t nModules)
        : OTRecHitPortableCollectionDevice<TDev>(queue, nHits, nModules + 1) {}

    uint32_t nHits() const { return this->view().otRecHits().metadata().size(); }

    uint32_t nModules() const { return this->view().otHitModules().metadata().size() - 1; }

    // No-op: the data is already on the device.
    template <typename TQueue>
    void updateFromDevice(TQueue) {}
  };

}  // namespace reco

#endif  // DataFormats_TrackingRecHitSoA_interface_OTRecHitsDevice_h
