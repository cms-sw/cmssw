#ifndef DataFormats_Track_interface_alpaka_TrackSoACollection_h
#define DataFormats_Track_interface_alpaka_TrackSoACollection_h

#include <cstdint>
#include <alpaka/alpaka.hpp>
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "DataFormats/TrackSoA/interface/TrackLayout.h"
#include "DataFormats/TrackSoA/interface/TrackDefinitions.h"
#include "DataFormats/TrackSoA/interface/TrackSoAHost.h"
#include "DataFormats/TrackSoA/interface/TrackSoADevice.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"

// TODO: The class is created via inheritance of the PortableCollection.
// This is generally discouraged, and should be done via composition.
// See: https://github.com/cms-sw/cmssw/pull/40465#discussion_r1067364306
namespace ALPAKA_ACCELERATOR_NAMESPACE {
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
  template <typename TrackerTraits>
  using TrackSoACollection = TrackSoAHost<TrackerTraits>;
#else
  template <typename TrackerTraits>
  using TrackSoACollection = TrackSoADevice<TrackerTraits, Device>;
#endif
  //Classes definition for Phase1/Phase2/HIonPhase1, to make the classes_def lighter. Not actually used in the code.
  namespace pixelTrack {
    using TrackSoACollectionPhase1 = TrackSoACollection<pixelTopology::Phase1>;
    using TrackSoACollectionPhase2 = TrackSoACollection<pixelTopology::Phase2>;
    using TrackSoACollectionHIonPhase1 = TrackSoACollection<pixelTopology::HIonPhase1>;
  }  // namespace pixelTrack
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

namespace cms::alpakatools {
  template <typename TrackerTraits>
  struct CopyToHost<ALPAKA_ACCELERATOR_NAMESPACE::TrackSoACollection<TrackerTraits>> {
    template <typename TQueue>
    static auto copyAsync(TQueue& queue,
                          ALPAKA_ACCELERATOR_NAMESPACE::TrackSoACollection<TrackerTraits> const& deviceData) {
      ::TrackSoAHost<TrackerTraits> hostData(queue);
      alpaka::memcpy(queue, hostData.buffer(), deviceData.buffer());
      return hostData;
    }
  };
}  // namespace cms::alpakatools

#endif  // DataFormats_Track_interface_alpaka_TrackSoACollection_h
