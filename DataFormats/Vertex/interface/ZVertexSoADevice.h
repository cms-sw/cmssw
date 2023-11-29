#ifndef DataFormats_Vertex_interface_ZVertexSoADevice_h
#define DataFormats_Vertex_interface_ZVertexSoADevice_h

#include <cstdint>

#include <alpaka/alpaka.hpp>
#include "DataFormats/Vertex/interface/ZVertexLayout.h"
#include "DataFormats/Vertex/interface/ZVertexDefinitions.h"
#include "DataFormats/Vertex/interface/alpaka/ZVertexUtilities.h"
#include "DataFormats/Vertex/interface/ZVertexSoAHost.h"
#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"

template <int32_t S, typename TDev>
class ZVertexSoADevice : public PortableDeviceCollection<ZVertexSoAHeterogeneousLayout<>, TDev> {
public:
  ZVertexSoADevice() = default;  // necessary for ROOT dictionaries

  // Constructor which specifies the SoA size
  template <typename TQueue>
  explicit ZVertexSoADevice(TQueue queue) : PortableDeviceCollection<ZVertexSoAHeterogeneousLayout<>, TDev>(S, queue) {}
};

using namespace ::zVertex;
template <typename TDev>
using ZVertexDevice = ZVertexSoADevice<MAXTRACKS, TDev>;

#endif  // DataFormats_Vertex_interface_ZVertexSoADevice_h
