#ifndef DataFormats_OfflineVertexSoA_interface_alpaka_TrackDeviceCollection_h
#define DataFormats_OfflineVertexSoA_interface_alpaka_TrackDeviceCollection_h

#include "DataFormats/OfflineVertexSoA/interface/TrackHostCollection.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/OfflineVertexSoA/interface/TrackSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using ::TrackHostCollection;

  using TrackDeviceCollection = PortableCollection<::TrackSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // DataFormats_OfflineVertexSoA_interface_alpaka_TrackDeviceCollection_h
