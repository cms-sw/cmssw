#ifndef DataFormats_VertexSoA_interface_alpaka_VertexDeviceCollection_h
#define DataFormats_VertexSoA_interface_alpaka_VertexDeviceCollection_h

#include "DataFormats/VertexSoA/interface/VertexHostCollection.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/VertexSoA/interface/VertexSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::reco {
  using ::reco::VertexHostCollection;

  using VertexDeviceCollection = PortableCollection<::reco::VertexSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // DataFormats_VertexSoA_interface_alpaka_VertexDeviceCollection_h
