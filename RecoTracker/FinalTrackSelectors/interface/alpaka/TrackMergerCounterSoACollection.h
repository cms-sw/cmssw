#ifndef RecoTracker_FinalTrackSelectors_interface_TrackMergerCounterSoACollection_h
#define RecoTracker_FinalTrackSelectors_interface_TrackMergerCounterSoACollection_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "RecoTracker/FinalTrackSelectors/interface/TrackMergerCounterDevice.h"
#include "RecoTracker/FinalTrackSelectors/interface/TrackMergerCounterHost.h"
#include "RecoTracker/FinalTrackSelectors/interface/TrackMergerCounterSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::reco {

  using ::reco::TrackMergerCounterDevice;
  using ::reco::TrackMergerCounterHost;
  using TrackMergerCounterSoACollection = std::
      conditional_t<std::is_same_v<Device, alpaka::DevCpu>, TrackMergerCounterHost, TrackMergerCounterDevice<Device>>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::reco

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(reco::TrackMergerCounterSoACollection, ::reco::TrackMergerCounterHost);

#endif  // RecoTracker_FinalTrackSelectors_interface_TrackMergerCounterSoACollection_h
