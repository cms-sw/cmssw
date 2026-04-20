#ifndef DataFormats_VertexSoA_interface_alpaka_VertexDeviceCollection_h
#define DataFormats_VertexSoA_interface_alpaka_VertexDeviceCollection_h

#include "DataFormats/VertexSoA/interface/VertexHostCollection.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/VertexSoA/interface/VertexSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using ::VertexHostCollection;

  using VertexDeviceCollection = PortableCollection<::VertexSoA>;

}

#endif  // DataFormats_VertexSoA_interface_alpaka_VertexDeviceCollection_h
