#include "RecoTracker/LSTCore/interface/LSTESData.h"
#include "RecoTracker/LSTCore/interface/EndcapGeometry.h"
#include "RecoTracker/LSTCore/interface/ModuleConnectionMap.h"
#include "RecoTracker/LSTCore/interface/TiltedGeometry.h"
#include "RecoTracker/LSTCore/interface/PixelMap.h"

#include "ModuleMethods.h"

#include <filesystem>

namespace {
  std::string geometryDataDir() {
    const char* path_lst_base = std::getenv("LST_BASE");
    const char* path_tracklooperdir = std::getenv("TRACKLOOPERDIR");
    std::string path_str;
    if (path_lst_base != nullptr) {
      path_str = path_lst_base;
    } else if (path_tracklooperdir != nullptr) {
      path_str = path_tracklooperdir;
      path_str += "/../";
    } else {
      std::stringstream search_path(std::getenv("CMSSW_SEARCH_PATH"));
      std::string path;
      while (std::getline(search_path, path, ':')) {
        if (std::filesystem::exists(path + "/RecoTracker/LSTCore/data")) {
          path_str = path;
          break;
        }
      }
      path_str += "/RecoTracker/LSTCore";
    }
    return path_str;
  }

  std::string get_absolute_path_after_check_file_exists(std::string const& name) {
    std::filesystem::path fullpath = std::filesystem::absolute(name);
    if (not std::filesystem::exists(fullpath)) {
      throw std::runtime_error("Could not find the file = " + fullpath.string());
    }
    return fullpath.string();
  }

  void loadMapsHost(lst::MapPLStoLayer& pLStoLayer,
                    lst::EndcapGeometry& endcapGeometry,
                    lst::TiltedGeometry& tiltedGeometry,
                    lst::ModuleConnectionMap& moduleConnectionMap,
                    std::string& ptCutLabel) {
    // Module orientation information (DrDz or phi angles)
    auto endcap_geom = get_absolute_path_after_check_file_exists(geometryDataDir() + "/data/OT800_IT615_pt" +
                                                                 ptCutLabel + "/endcap_orientation.bin");
    auto tilted_geom = get_absolute_path_after_check_file_exists(geometryDataDir() + "/data/OT800_IT615_pt" +
                                                                 ptCutLabel + "/tilted_barrel_orientation.bin");
    // Module connection map (for line segment building)
    auto mappath = get_absolute_path_after_check_file_exists(geometryDataDir() + "/data/OT800_IT615_pt" + ptCutLabel +
                                                             "/module_connection_tracing_merged.bin");

    endcapGeometry.load(endcap_geom);
    tiltedGeometry.load(tilted_geom);
    moduleConnectionMap.load(mappath);

    auto pLSMapDir = geometryDataDir() + "/data/OT800_IT615_pt" + ptCutLabel + "/pixelmap/pLS_map";
    const std::array<std::string, 4> connects{
        {"_layer1_subdet5", "_layer2_subdet5", "_layer1_subdet4", "_layer2_subdet4"}};
    std::string path;

    static_assert(connects.size() == std::tuple_size<std::decay_t<decltype(pLStoLayer[0])>>{});
    for (unsigned int i = 0; i < connects.size(); i++) {
      auto connectData = connects[i].data();

      path = pLSMapDir + connectData + ".bin";
      pLStoLayer[0][i] = lst::ModuleConnectionMap(get_absolute_path_after_check_file_exists(path));

      path = pLSMapDir + "_pos" + connectData + ".bin";
      pLStoLayer[1][i] = lst::ModuleConnectionMap(get_absolute_path_after_check_file_exists(path));

      path = pLSMapDir + "_neg" + connectData + ".bin";
      pLStoLayer[2][i] = lst::ModuleConnectionMap(get_absolute_path_after_check_file_exists(path));
    }
  }
}  // namespace

std::unique_ptr<lst::LSTESData<alpaka_common::DevHost>> lst::loadAndFillESHost(std::string& ptCutLabel) {
  uint16_t nModules;
  uint16_t nLowerModules;
  unsigned int nPixels;
  MapPLStoLayer pLStoLayer;
  EndcapGeometry endcapGeometry;
  TiltedGeometry tiltedGeometry;
  PixelMap pixelMapping;
  ModuleConnectionMap moduleConnectionMap;
  ::loadMapsHost(pLStoLayer, endcapGeometry, tiltedGeometry, moduleConnectionMap, ptCutLabel);

  auto endcapGeometryDev =
      std::make_shared<EndcapGeometryDevHostCollection>(endcapGeometry.nEndCapMap, cms::alpakatools::host());
  std::memcpy(endcapGeometryDev->view().geoMapDetId(),
              endcapGeometry.geoMapDetId_buf.data(),
              endcapGeometry.nEndCapMap * sizeof(unsigned int));
  std::memcpy(endcapGeometryDev->view().geoMapPhi(),
              endcapGeometry.geoMapPhi_buf.data(),
              endcapGeometry.nEndCapMap * sizeof(float));

  auto path = get_absolute_path_after_check_file_exists(geometryDataDir() + "/data/OT800_IT615_pt" + ptCutLabel +
                                                        "/sensor_centroids.bin");
  auto modulesBuffers = lst::loadModulesFromFile(pLStoLayer,
                                                 path.c_str(),
                                                 nModules,
                                                 nLowerModules,
                                                 nPixels,
                                                 pixelMapping,
                                                 endcapGeometry,
                                                 tiltedGeometry,
                                                 moduleConnectionMap);
  auto pixelMappingPtr = std::make_shared<PixelMap>(std::move(pixelMapping));
  return std::make_unique<LSTESData<alpaka_common::DevHost>>(nModules,
                                                             nLowerModules,
                                                             nPixels,
                                                             endcapGeometry.nEndCapMap,
                                                             std::move(modulesBuffers),
                                                             std::move(endcapGeometryDev),
                                                             pixelMappingPtr);
}
