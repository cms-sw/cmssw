#ifndef DataFormats_VertexSoA_interface_ZVertexDevice_h
#define DataFormats_VertexSoA_interface_ZVertexDevice_h

#include <cstdint>

#include <alpaka/alpaka.hpp>
#include "DataFormats/VertexSoA/interface/ZVertexSoA.h"
#include "DataFormats/VertexSoA/interface/ZVertexDefinitions.h"
#include "DataFormats/VertexSoA/interface/ZVertexHost.h"
#include "DataFormats/Portable/interface/PortableDeviceCollection.h"

template <int32_t S, typename TDev>
class ZVertexDeviceSoA : public PortableDeviceCollection<reco::ZVertexLayout<>, TDev> {
public:
  ZVertexDeviceSoA() = default;  // necessary for ROOT dictionaries

  // Constructor which specifies the SoA size
  template <typename TQueue>
  explicit ZVertexDeviceSoA(TQueue queue) : PortableDeviceCollection<reco::ZVertexLayout<>, TDev>(S, queue) {}
};

using namespace ::zVertex;
template <typename TDev>
using ZVertexDevice = ZVertexDeviceSoA<MAXTRACKS, TDev>;

#endif  // DataFormats_VertexSoA_interface_ZVertexDevice_h
