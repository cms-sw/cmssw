#ifndef RecoTracker_LSTCore_interface_LSTESData_h
#define RecoTracker_LSTCore_interface_LSTESData_h

#include "RecoTracker/LSTCore/interface/Constants.h"
#include "RecoTracker/LSTCore/interface/EndcapGeometryBuffer.h"
#include "RecoTracker/LSTCore/interface/Module.h"
#include "RecoTracker/LSTCore/interface/PixelMap.h"

#include "HeterogeneousCore/AlpakaInterface/interface/CopyToDevice.h"

#include <filesystem>
#include <memory>

namespace lst {

  template <typename TDev>
  struct LSTESData {
    uint16_t nModules;
    uint16_t nLowerModules;
    unsigned int nPixels;
    unsigned int nEndCapMap;
    ModulesBuffer<TDev> modulesBuffers;
    EndcapGeometryBuffer<TDev> endcapGeometryBuffers;
    std::shared_ptr<const PixelMap> pixelMapping;

    LSTESData(uint16_t const& nModulesIn,
              uint16_t const& nLowerModulesIn,
              unsigned int const& nPixelsIn,
              unsigned int const& nEndCapMapIn,
              ModulesBuffer<TDev> const& modulesBuffersIn,
              EndcapGeometryBuffer<TDev> const& endcapGeometryBuffersIn,
              std::shared_ptr<const PixelMap> const& pixelMappingIn)
        : nModules(nModulesIn),
          nLowerModules(nLowerModulesIn),
          nPixels(nPixelsIn),
          nEndCapMap(nEndCapMapIn),
          modulesBuffers(modulesBuffersIn),
          endcapGeometryBuffers(endcapGeometryBuffersIn),
          pixelMapping(pixelMappingIn) {}
  };

  std::unique_ptr<LSTESData<alpaka_common::DevHost>> loadAndFillESHost();

}  // namespace lst

namespace cms::alpakatools {
  template <>
  struct CopyToDevice<lst::LSTESData<alpaka_common::DevHost>> {
    template <typename TQueue>
    static lst::LSTESData<alpaka::Dev<TQueue>> copyAsync(TQueue& queue,
                                                         lst::LSTESData<alpaka_common::DevHost> const& srcData) {
      auto deviceModulesBuffers =
          lst::ModulesBuffer<alpaka::Dev<TQueue>>(alpaka::getDev(queue), srcData.nModules, srcData.nPixels);
      deviceModulesBuffers.copyFromSrc(queue, srcData.modulesBuffers);
      auto deviceEndcapGeometryBuffers =
          lst::EndcapGeometryBuffer<alpaka::Dev<TQueue>>(alpaka::getDev(queue), srcData.nEndCapMap);
      deviceEndcapGeometryBuffers.copyFromSrc(queue, srcData.endcapGeometryBuffers);

      return lst::LSTESData<alpaka::Dev<TQueue>>(srcData.nModules,
                                                 srcData.nLowerModules,
                                                 srcData.nPixels,
                                                 srcData.nEndCapMap,
                                                 std::move(deviceModulesBuffers),
                                                 std::move(deviceEndcapGeometryBuffers),
                                                 srcData.pixelMapping);
    }
  };
}  // namespace cms::alpakatools

#endif
