#ifndef DataFormats_VertexGNNReco_interface_alpaka_VertexGNNDeviceCollection_h
#define DataFormats_VertexGNNReco_interface_alpaka_VertexGNNDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "DataFormats/VertexGNNReco/interface/VertexGNNSoA.h"
#include "DataFormats/VertexGNNReco/interface/VertexGNNHostCollection.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::vertexgnn {

  using TrackFeaturesDeviceCollection = PortableCollection<::vertexgnn::TrackFeaturesSoA>;

  using GNNOutputDeviceCollection = PortableCollection<::vertexgnn::GNNOutputSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::vertexgnn

#endif
