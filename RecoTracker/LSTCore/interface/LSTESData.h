#ifndef RecoTracker_LSTCore_interface_LSTESData_h
#define RecoTracker_LSTCore_interface_LSTESData_h

#include "RecoTracker/LSTCore/interface/Common.h"
#include "RecoTracker/LSTCore/interface/EndcapGeometryDevHostCollection.h"
#include "RecoTracker/LSTCore/interface/ModulesHostCollection.h"
#include "RecoTracker/LSTCore/interface/PixelMap.h"

#include "HeterogeneousCore/AlpakaInterface/interface/CopyToDevice.h"

#include <memory>

namespace lst {

  template <typename TDev>
  struct LSTESData {
    uint16_t nModules;
    uint16_t nLowerModules;
    unsigned int nPixels;
    unsigned int nEndCapMap;
    // Using shared_ptr so that for the serial backend all streams can use the same data
    std::shared_ptr<const PortableMultiCollection<TDev, ModulesSoA, ModulesPixelSoA>> modules;
    std::shared_ptr<const PortableCollection<EndcapGeometryDevSoA, TDev>> endcapGeometry;
    // Host-side object that is shared between the LSTESData<TDev> objects for different devices
    std::shared_ptr<const PixelMap> pixelMapping;

    LSTESData(uint16_t const& nModulesIn,
              uint16_t const& nLowerModulesIn,
              unsigned int const& nPixelsIn,
              unsigned int const& nEndCapMapIn,
              std::shared_ptr<const PortableMultiCollection<TDev, ModulesSoA, ModulesPixelSoA>> modulesIn,
              std::shared_ptr<const PortableCollection<EndcapGeometryDevSoA, TDev>> endcapGeometryIn,
              std::shared_ptr<const PixelMap> const& pixelMappingIn)
        : nModules(nModulesIn),
          nLowerModules(nLowerModulesIn),
          nPixels(nPixelsIn),
          nEndCapMap(nEndCapMapIn),
          modules(std::move(modulesIn)),
          endcapGeometry(std::move(endcapGeometryIn)),
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
      using TDev = alpaka::Dev<TQueue>;
      std::shared_ptr<const PortableMultiCollection<TDev, lst::ModulesSoA, lst::ModulesPixelSoA>> deviceModules;
      std::shared_ptr<const PortableCollection<lst::EndcapGeometryDevSoA, TDev>> deviceEndcapGeometry;

      if constexpr (std::is_same_v<TDev, alpaka_common::DevHost>) {
        deviceModules = srcData.modules;
        deviceEndcapGeometry = srcData.endcapGeometry;
      } else {
        deviceModules = std::make_shared<PortableMultiCollection<TDev, lst::ModulesSoA, lst::ModulesPixelSoA>>(
            CopyToDevice<PortableHostMultiCollection<lst::ModulesSoA, lst::ModulesPixelSoA>>::copyAsync(
                queue, *srcData.modules));
        deviceEndcapGeometry = std::make_shared<PortableCollection<lst::EndcapGeometryDevSoA, TDev>>(
            CopyToDevice<PortableHostCollection<lst::EndcapGeometryDevSoA>>::copyAsync(queue, *srcData.endcapGeometry));
      }

      return lst::LSTESData<alpaka::Dev<TQueue>>(srcData.nModules,
                                                 srcData.nLowerModules,
                                                 srcData.nPixels,
                                                 srcData.nEndCapMap,
                                                 std::move(deviceModules),
                                                 std::move(deviceEndcapGeometry),
                                                 srcData.pixelMapping);
    }
  };
}  // namespace cms::alpakatools

#endif
