#ifndef DataFormats_VertexSoA_interface_alpaka_TrackDeviceCollection_h
#define DataFormats_VertexSoA_interface_alpaka_TrackDeviceCollection_h

#include "DataFormats/VertexSoA/interface/TrackHostCollection.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/VertexSoA/interface/TrackSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using ::TrackHostCollection;

  using TrackDeviceCollection = PortableCollection<::TrackSoA>;

}

#endif  // DataFormats_VertexSoA_interface_alpaka_TrackDeviceCollection_h
