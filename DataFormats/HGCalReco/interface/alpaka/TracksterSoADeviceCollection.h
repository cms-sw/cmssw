#ifndef DataFormats_PortableTestObjects_interface_alpaka_TracksterSoADeviceCollection_h
#define DataFormats_PortableTestObjects_interface_alpaka_TracksterSoADeviceCollection_h

#include <Eigen/Core>

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "DataFormats/HGCalReco/interface/TracksterSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using TracksterSoADeviceCollection = PortableCollection<TracksterSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // DataFormats_PortableTestObjects_interface_alpaka_TracksterSoADeviceCollection_h