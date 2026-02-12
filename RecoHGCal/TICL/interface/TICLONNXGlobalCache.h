#ifndef RecoHGCal_TICL_TICLONNXGlobalCache_h
#define RecoHGCal_TICL_TICLONNXGlobalCache_h

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/FileInPath.h"
#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"

namespace ticl {

  struct TICLONNXGlobalCache {
    // Sessions are indexed by fully resolved path (FileInPath::fullPath()).
    std::unordered_map<std::string, std::unique_ptr<cms::Ort::ONNXRuntime>> sessionsByFullPath;

    // Returns nullptr if the session is not in the cache.
    cms::Ort::ONNXRuntime const* getByFullPath(std::string const& fullPath) const {
      auto it = sessionsByFullPath.find(fullPath);
      if (it == sessionsByFullPath.end()) {
        return nullptr;
      }
      return it->second.get();
    }

    // Convenience: resolve a FileInPath-like string and return the cached session (or nullptr).
    cms::Ort::ONNXRuntime const* getByModelPathString(std::string const& modelPath) const {
      if (modelPath.empty()) {
        return nullptr;
      }
      return getByFullPath(edm::FileInPath(modelPath).fullPath());
    }

    // Initializes the cache by scanning the full module ParameterSet recursively.
    // It loads all unique non-empty model paths found in string parameters matching common naming patterns.
    static std::unique_ptr<TICLONNXGlobalCache> initialize(edm::ParameterSet const& modulePSet) {
      auto cache = std::make_unique<TICLONNXGlobalCache>();
      std::unordered_set<std::string> uniqueFullPaths;
      collectModelPathsRecursively(modulePSet, uniqueFullPaths);

      for (auto const& fullPath : uniqueFullPaths) {
        cache->sessionsByFullPath.emplace(fullPath, std::make_unique<cms::Ort::ONNXRuntime>(fullPath));
      }
      return cache;
    }

  private:
    // Returns true if a parameter name looks like it may contain an ONNX model path.
    static bool looksLikeModelPathParam(std::string const& name) {
      // Keep this intentionally generic: no dependence on specific module PSet structure.
      // Matches "onnxModelPath", "onnxPIDModelPath", "onnxEnergyModelPath", etc.
      return (name.find("onnx") != std::string::npos) && (name.find("ModelPath") != std::string::npos);
    }

    static void collectModelPathsRecursively(edm::ParameterSet const& pset,
                                             std::unordered_set<std::string>& outFullPaths) {
      // Scan string parameters in this PSet.
      for (auto const& name : pset.getParameterNames()) {
        if (!looksLikeModelPathParam(name)) {
          continue;
        }
        if (!pset.existsAs<std::string>(name, true)) {
          continue;
        }

        std::string model = pset.getParameter<std::string>(name);
        if (model.empty()) {
          continue;
        }

        outFullPaths.emplace(edm::FileInPath(model).fullPath());
      }

      // Recurse into nested PSets.
      std::vector<std::string> nestedNames;
      pset.getParameterSetNames(nestedNames);
      for (auto const& nestedName : nestedNames) {
        auto const& nested = pset.getParameter<edm::ParameterSet>(nestedName);
        collectModelPathsRecursively(nested, outFullPaths);
      }

      // Recurse into nested VParametersets.
      std::vector<std::string> vpsetNames;
      pset.getParameterSetVectorNames(vpsetNames);
      for (auto const& vpsetName : vpsetNames) {
        auto const& vpsets = pset.getParameter<std::vector<edm::ParameterSet>>(vpsetName);
        for (auto const& elem : vpsets) {
          collectModelPathsRecursively(elem, outFullPaths);
        }
      }
    }
  };

}  // namespace ticl

#endif
