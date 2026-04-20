#ifndef DataFormats_OfflineVertexSoA_interface_alpaka_VertexDeviceCollection_h
#define DataFormats_OfflineVertexSoA_interface_alpaka_VertexDeviceCollection_h

#include "DataFormats/OfflineVertexSoA/interface/VertexHostCollection.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/OfflineVertexSoA/interface/VertexSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using ::VertexHostCollection;

  using VertexDeviceCollection = PortableCollection<::VertexSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // DataFormats_OfflineVertexSoA_interface_alpaka_VertexDeviceCollection_h
