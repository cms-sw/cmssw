#ifndef DataFormats_Vertex_interface_ZVertexDevice_h
#define DataFormats_Vertex_interface_ZVertexDevice_h

#include <cstdint>

#include <alpaka/alpaka.hpp>
#include "DataFormats/VertexSoA/interface/ZVertexSoA.h"
#include "DataFormats/VertexSoA/interface/ZVertexDefinitions.h"
#include "DataFormats/VertexSoA/interface/alpaka/ZVertexUtilities.h"
#include "DataFormats/VertexSoA/interface/ZVertexHost.h"
#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"

template <int32_t S, typename TDev>
class ZVertexDeviceSoA : public PortableDeviceCollection<ZVertexLayout<>, TDev> {
public:
  ZVertexDeviceSoA() = default;  // necessary for ROOT dictionaries

  // Constructor which specifies the SoA size
  template <typename TQueue>
  explicit ZVertexDeviceSoA(TQueue queue) : PortableDeviceCollection<ZVertexLayout<>, TDev>(S, queue) {}
};

using namespace ::zVertex;
template <typename TDev>
using ZVertexDevice = ZVertexDeviceSoA<MAXTRACKS, TDev>;

#endif  // DataFormats_Vertex_interface_ZVertexDevice_h
