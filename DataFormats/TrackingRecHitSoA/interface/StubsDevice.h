#ifndef DataFormats_TrackingRecHitSoA_interface_StubsDevice_h
#define DataFormats_TrackingRecHitSoA_interface_StubsDevice_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Common/interface/Uninitialized.h"
#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/StubsHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/StubsSoA.h"

// TODO: The class is created via inheritance of the PortableCollection.
// This is generally discouraged, and should be done via composition.
// See: https://github.com/cms-sw/cmssw/pull/40465#discussion_r1067364306

namespace reco {

  template <typename TDev>
  using StubPortableCollectionDevice = PortableDeviceCollection<TDev, reco::StubBlocksSoA>;

  template <typename TDev>
  class StubsDevice : public StubPortableCollectionDevice<TDev> {
  public:
    StubsDevice() = default;

    StubsDevice(edm::Uninitialized) : StubPortableCollectionDevice<TDev>{edm::kUninitialized} {}

    // The module block is allocated with nModules+1 elements and carries no payload: its extent is
    // the only record of the module count.
    template <typename TQueue>
    explicit StubsDevice(TQueue queue, uint32_t nStubs, uint32_t nModules)
        : StubPortableCollectionDevice<TDev>(queue, nStubs, nModules + 1) {}

    // Number of stubs in the collection
    uint32_t nStubs() const { return this->view().stubs().metadata().size(); }

    // Number of stacked modules
    uint32_t nModules() const { return this->view().stubModules().metadata().size() - 1; }
  };

}  // namespace reco

#endif  // DataFormats_TrackingRecHitSoA_interface_StubsDevice_h
