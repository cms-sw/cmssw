#ifndef DataFormats_VertexSoA_interface_alpaka_VertexGNNDeviceCollection_h
#define DataFormats_VertexSoA_interface_alpaka_VertexGNNDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "DataFormats/VertexSoA/interface/VertexGNNSoA.h"
#include "DataFormats/VertexSoA/interface/VertexGNNHostCollection.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::vertexgnn {

  using TrackFeaturesDeviceCollection = PortableCollection<::vertexgnn::TrackFeaturesSoA>;

  using GNNOutputDeviceCollection = PortableCollection<::vertexgnn::GNNOutputSoA>;

  using GNNSlotDeviceCollection = PortableCollection<::vertexgnn::GNNSlotSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::vertexgnn

#endif
