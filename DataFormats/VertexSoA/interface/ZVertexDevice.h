#ifndef DataFormats_VertexSoA_interface_ZVertexDevice_h
#define DataFormats_VertexSoA_interface_ZVertexDevice_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/VertexSoA/interface/ZVertexSoA.h"
#include "DataFormats/VertexSoA/interface/ZVertexHost.h"
#include "DataFormats/Portable/interface/PortableDeviceCollection.h"

template <typename TDev>
using ZVertexDevice = PortableDeviceCollection<reco::ZVertexBlocks, TDev>;

#endif  // DataFormats_VertexSoA_interface_ZVertexDevice_h
