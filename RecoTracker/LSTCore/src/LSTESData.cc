#include <filesystem>

#include "RecoTracker/LSTCore/interface/LSTESData.h"
#include "RecoTracker/LSTCore/interface/EndcapGeometry.h"
#include "RecoTracker/LSTCore/interface/ModuleConnectionMap.h"
#include "RecoTracker/LSTCore/interface/TiltedGeometry.h"
#include "RecoTracker/LSTCore/interface/PixelMap.h"

#include "ModuleMethods.h"

namespace {
  std::string geometryDataDir() {
    std::string path_str, path;
    const char* path_tracklooperdir = std::getenv("TRACKLOOPERDIR");
    std::stringstream search_path;
    search_path << std::getenv("CMSSW_SEARCH_PATH");

    while (std::getline(search_path, path, ':')) {
      if (std::filesystem::exists(path + "/RecoTracker/LSTCore/data")) {
        path_str = path;
        break;
      }
    }

    if (path_str.empty()) {
      path_str = path_tracklooperdir;
      path_str += "/..";
    } else {
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

std::unique_ptr<lst::LSTESData<alpaka_common::DevHost>> lst::loadAndFillESDataHost(std::string& ptCutLabel) {
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
      std::make_shared<EndcapGeometryDevHostCollection>(cms::alpakatools::host(), endcapGeometry.nEndCapMap);
  std::memcpy(endcapGeometryDev->view().geoMapDetId().data(),
              endcapGeometry.geoMapDetId_buf.data(),
              endcapGeometry.nEndCapMap * sizeof(unsigned int));
  std::memcpy(endcapGeometryDev->view().geoMapPhi().data(),
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

std::unique_ptr<lst::LSTESData<alpaka_common::DevHost>> lst::fillESDataHost(lstgeometry::Geometry const& lstg) {
  uint16_t nModules;
  uint16_t nLowerModules;
  unsigned int nPixels;
  MapPLStoLayer pLStoLayer;
  EndcapGeometry endcapGeometry;
  TiltedGeometry tiltedGeometry;
  PixelMap pixelMapping;
  ModuleConnectionMap moduleConnectionMap;

  endcapGeometry.load(lstg.endcap_slopes, lstg.sensors);
  auto endcapGeometryDev =
      std::make_shared<EndcapGeometryDevHostCollection>(cms::alpakatools::host(), endcapGeometry.nEndCapMap);
  std::memcpy(endcapGeometryDev->view().geoMapDetId().data(),
              endcapGeometry.geoMapDetId_buf.data(),
              endcapGeometry.nEndCapMap * sizeof(unsigned int));
  std::memcpy(endcapGeometryDev->view().geoMapPhi().data(),
              endcapGeometry.geoMapPhi_buf.data(),
              endcapGeometry.nEndCapMap * sizeof(float));

  tiltedGeometry.load(lstg.barrel_slopes);

  std::map<unsigned int, std::vector<unsigned int>> final_modulemap;
  for (auto const& [detId, connections] : lstg.module_map) {
    final_modulemap[detId] = std::vector<unsigned int>(connections.begin(), connections.end());
  }
  moduleConnectionMap.load(final_modulemap);

  for (auto& [layersubdetcharge, map] : lstg.pixel_map) {
    auto& [layer, subdet, charge] = layersubdetcharge;

    std::map<unsigned int, std::vector<unsigned int>> final_pixelmap;
    for (unsigned int isuperbin = 0; isuperbin < map.size(); isuperbin++) {
      auto const& set = map.at(isuperbin);
      final_pixelmap[isuperbin] = std::vector<unsigned int>(set.begin(), set.end());
    }

    if (charge == 0) {
      pLStoLayer[0][layer - 1 + (subdet == Endcap ? 2 : 0)] = lst::ModuleConnectionMap(final_pixelmap);
    } else if (charge > 0) {
      pLStoLayer[1][layer - 1 + (subdet == Endcap ? 2 : 0)] = lst::ModuleConnectionMap(final_pixelmap);
    } else {
      pLStoLayer[2][layer - 1 + (subdet == Endcap ? 2 : 0)] = lst::ModuleConnectionMap(final_pixelmap);
    }
  }

  ModuleMetaData mmd;
  unsigned int counter = 0;
  for (auto const& [detId, sensor] : lstg.sensors) {
    mmd.detIdToIndex[detId] = counter;
    mmd.module_x[detId] = sensor.centerX;
    mmd.module_y[detId] = sensor.centerY;
    mmd.module_z[detId] = sensor.centerZ;
    mmd.module_type[detId] = static_cast<unsigned int>(sensor.moduleType);
    counter++;
  }
  mmd.detIdToIndex[kPixelModuleId] = counter;  //pixel module is the last module in the module list
  counter++;
  nModules = counter;

  auto modulesBuffers = constructModuleCollection(pLStoLayer,
                                                  mmd,
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
