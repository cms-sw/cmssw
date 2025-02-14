#ifndef DataFormats_PortableVertex_interface_alpaka_VertexDeviceCollection_h
#define DataFormats_PortableVertex_interface_alpaka_VertexDeviceCollection_h

#include "DataFormats/PortableVertex/interface/VertexHostCollection.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/PortableVertex/interface/VertexSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::portablevertex {

  // make the names from the top-level portablevertex namespace visible for unqualified lookup
  // inside the ALPAKA_ACCELERATOR_NAMESPACE::portablevertex namespace
  using ::portablevertex::ClusterParamsHostCollection;
  using ::portablevertex::TrackHostCollection;
  using ::portablevertex::VertexHostCollection;

  using VertexDeviceCollection = PortableCollection<::portablevertex::VertexSoA>;
  using TrackDeviceCollection = PortableCollection<::portablevertex::TrackSoA>;
  using ClusterParamsDeviceCollection = PortableCollection<::portablevertex::ClusterParamsSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::portablevertex

#endif  // DataFormats_PortableVertex_interface_alpaka_VertexDeviceCollection_h
