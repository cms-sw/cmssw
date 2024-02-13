#ifndef DataFormats_VertexSoA_interface_ZVertexDevice_h
#define DataFormats_VertexSoA_interface_ZVertexDevice_h

#include <cstdint>

#include <alpaka/alpaka.hpp>
#include "DataFormats/VertexSoA/interface/ZVertexSoA.h"
#include "DataFormats/VertexSoA/interface/ZVertexDefinitions.h"
#include "DataFormats/VertexSoA/interface/ZVertexHost.h"
#include "DataFormats/Portable/interface/PortableDeviceCollection.h"

template <int32_t NVTX, int32_t NTRK, typename TDev>
class ZVertexDeviceSoA : public PortableDeviceMultiCollection<TDev, reco::ZVertexSoA, reco::ZVertexTracksSoA> {
public:
  ZVertexDeviceSoA() = default;  // necessary for ROOT dictionaries

  // Constructor which specifies the queue
  template <typename TQueue>
  explicit ZVertexDeviceSoA(TQueue queue)
      : PortableDeviceMultiCollection<TDev, reco::ZVertexSoA, reco::ZVertexTracksSoA>({{NVTX, NTRK}}, queue) {}
};

template <typename TDev>
using ZVertexDevice = ZVertexDeviceSoA<zVertex::MAXVTX, zVertex::MAXTRACKS, TDev>;

#endif  // DataFormats_VertexSoA_interface_ZVertexDevice_h
