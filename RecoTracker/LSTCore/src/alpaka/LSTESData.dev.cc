#ifdef LST_IS_CMSSW_PACKAGE
#include "RecoTracker/LSTCore/interface/alpaka/LSTESData.h"
#else
#include "LSTESData.h"
#endif

#include "EndcapGeometry.h"
#include "ModuleConnectionMap.h"
#include "TiltedGeometry.h"
#include "PixelMap.h"
#include "ModuleMethods.h"

namespace {
  std::string trackLooperDir() {
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

  std::string get_absolute_path_after_check_file_exists(const std::string name) {
    std::filesystem::path fullpath = std::filesystem::absolute(name.c_str());
    if (not std::filesystem::exists(fullpath)) {
      std::cout << "ERROR: Could not find the file = " << fullpath << std::endl;
      exit(2);
    }
    return fullpath.string();
  }

  void loadMapsHost(SDL::MapPLStoLayer& pLStoLayer,
                    std::shared_ptr<SDL::EndcapGeometryHost<SDL::Dev>> endcapGeometry,
                    std::shared_ptr<SDL::TiltedGeometry<SDL::Dev>> tiltedGeometry,
                    std::shared_ptr<SDL::ModuleConnectionMap<SDL::Dev>> moduleConnectionMap) {
    // Module orientation information (DrDz or phi angles)
    auto endcap_geom =
        get_absolute_path_after_check_file_exists(trackLooperDir() + "/data/OT800_IT615_pt0.8/endcap_orientation.bin");
    auto tilted_geom = get_absolute_path_after_check_file_exists(
        trackLooperDir() + "/data/OT800_IT615_pt0.8/tilted_barrel_orientation.bin");
    // Module connection map (for line segment building)
    auto mappath = get_absolute_path_after_check_file_exists(
        trackLooperDir() + "/data/OT800_IT615_pt0.8/module_connection_tracing_merged.bin");

    endcapGeometry->load(endcap_geom);
    tiltedGeometry->load(tilted_geom);
    moduleConnectionMap->load(mappath);

    auto pLSMapDir = trackLooperDir() + "/data/OT800_IT615_pt0.8/pixelmap/pLS_map";
    const std::array<std::string, 4> connects{
        {"_layer1_subdet5", "_layer2_subdet5", "_layer1_subdet4", "_layer2_subdet4"}};
    std::string path;

    static_assert(connects.size() == std::tuple_size<std::decay_t<decltype(pLStoLayer[0])>>{});
    for (unsigned int i = 0; i < connects.size(); i++) {
      auto connectData = connects[i].data();

      path = pLSMapDir + connectData + ".bin";
      pLStoLayer[0][i] = SDL::ModuleConnectionMap<SDL::Dev>(get_absolute_path_after_check_file_exists(path));

      path = pLSMapDir + "_pos" + connectData + ".bin";
      pLStoLayer[1][i] = SDL::ModuleConnectionMap<SDL::Dev>(get_absolute_path_after_check_file_exists(path));

      path = pLSMapDir + "_neg" + connectData + ".bin";
      pLStoLayer[2][i] = SDL::ModuleConnectionMap<SDL::Dev>(get_absolute_path_after_check_file_exists(path));
    }
  }
}  // namespace

std::unique_ptr<SDL::LSTESHostData<SDL::Dev>> SDL::loadAndFillESHost() {
  auto pLStoLayer = std::make_shared<SDL::MapPLStoLayer>();
  auto endcapGeometry = std::make_shared<SDL::EndcapGeometryHost<SDL::Dev>>();
  auto tiltedGeometry = std::make_shared<SDL::TiltedGeometry<SDL::Dev>>();
  auto moduleConnectionMap = std::make_shared<SDL::ModuleConnectionMap<SDL::Dev>>();
  ::loadMapsHost(*pLStoLayer, endcapGeometry, tiltedGeometry, moduleConnectionMap);
  return std::make_unique<LSTESHostData<SDL::Dev>>(pLStoLayer, endcapGeometry, tiltedGeometry, moduleConnectionMap);
}

std::unique_ptr<SDL::LSTESDeviceData<SDL::Dev>> SDL::loadAndFillESDevice(SDL::QueueAcc& queue,
                                                                         const LSTESHostData<SDL::Dev>* hostData) {
  SDL::Dev const& devAccIn = alpaka::getDev(queue);
  uint16_t nModules;
  uint16_t nLowerModules;
  unsigned int nPixels;
  std::shared_ptr<SDL::modulesBuffer<SDL::Dev>> modulesBuffers = nullptr;
  auto endcapGeometry = std::make_shared<SDL::EndcapGeometry<SDL::Dev>>(devAccIn, queue, *hostData->endcapGeometry);
  auto pixelMapping = std::make_shared<SDL::pixelMap>();
  auto moduleConnectionMap = hostData->moduleConnectionMap;

  auto path =
      get_absolute_path_after_check_file_exists(trackLooperDir() + "/data/OT800_IT615_pt0.8/sensor_centroids.bin");
  SDL::loadModulesFromFile(queue,
                           hostData->mapPLStoLayer.get(),
                           path.c_str(),
                           nModules,
                           nLowerModules,
                           nPixels,
                           modulesBuffers,
                           pixelMapping.get(),
                           endcapGeometry.get(),
                           hostData->tiltedGeometry.get(),
                           moduleConnectionMap.get());
  return std::make_unique<LSTESDeviceData<SDL::Dev>>(
      nModules, nLowerModules, nPixels, modulesBuffers, endcapGeometry, pixelMapping);
}
