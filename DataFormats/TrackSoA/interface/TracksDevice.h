#ifndef DataFormats_TrackSoA_interface_TracksDevice_h
#define DataFormats_TrackSoA_interface_TracksDevice_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Common/interface/Uninitialized.h"
#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "DataFormats/TrackSoA/interface/TrackDefinitions.h"
#include "DataFormats/TrackSoA/interface/TracksSoA.h"

namespace reco {
  template <typename TDev>
  using TracksDevice = PortableDeviceCollection<TDev, TrackBlocks>;
}

#endif  // DataFormats_Track_TracksDevice_H
