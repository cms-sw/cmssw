#ifndef DataFormats_VertexSoA_interface_alpaka_OfflineVertexDeviceCollection_h
#define DataFormats_VertexSoA_interface_alpaka_OfflineVertexDeviceCollection_h

#include "DataFormats/VertexSoA/interface/OfflineVertexHostCollection.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/VertexSoA/interface/OfflineVertexSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using ::OfflineVertexHostCollection;

  using OfflineVertexDeviceCollection = PortableCollection<::OfflineVertexSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // DataFormats_VertexSoA_interface_alpaka_OfflineVertexDeviceCollection_h
