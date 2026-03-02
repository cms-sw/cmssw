#ifndef RecoHGCal_TICL_TICLONNXGlobalCache_h
#define RecoHGCal_TICL_TICLONNXGlobalCache_h

#include <memory>
#include <string>
#include <unordered_map>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/FileInPath.h"
#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"

namespace ticl {

  struct TICLONNXGlobalCache {
    // Sessions indexed by fully resolved path (FileInPath::fullPath()).
    std::unordered_map<std::string, std::unique_ptr<cms::Ort::ONNXRuntime>> sessionsByFullPath;

    cms::Ort::ONNXRuntime const* getByFullPath(std::string const& fullPath) const noexcept {
      const auto it = sessionsByFullPath.find(fullPath);
      return (it == sessionsByFullPath.end()) ? nullptr : it->second.get();
    }

    cms::Ort::ONNXRuntime const* getByModelPathString(std::string const& modelPath) const {
      if (modelPath.empty()) {
        return nullptr;
      }
      return getByFullPath(edm::FileInPath(modelPath).fullPath());
    }

    //  - Consider inference plugin only if inferenceAlgo is non-empty.
    static std::unique_ptr<TICLONNXGlobalCache> initialize(edm::ParameterSet const& modulePSet) {
      Ort::SessionOptions sess_opts;
      sess_opts.SetIntraOpNumThreads(1);
      sess_opts.SetInterOpNumThreads(1);
      sess_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
      // Enable profiling if "profileOutputFile" is set in the module PSet. This will create a file with detailed timing info for each ONNX Runtime session run, which can be visualized with tools like
      if (modulePSet.existsAs<std::string>("profileOutputFile", /*trackPar=*/true)) {
        const auto profileFile = modulePSet.getParameter<std::string>("profileOutputFile");
        if (!profileFile.empty()) {
          sess_opts.EnableProfiling(profileFile.c_str());
        }
      }
      sess_opts.EnableCpuMemArena();
      sess_opts.EnableMemPattern();
      auto cache = std::make_unique<TICLONNXGlobalCache>();

      // 1) Linking model (TracksterLinksProducer / TracksterLinksProducer-like modules)
      // Load only if present and non-empty.
      if (modulePSet.existsAs<edm::ParameterSet>("linkingPSet", /*trackPar=*/true)) {
        const auto linkingPSet = modulePSet.getParameter<edm::ParameterSet>("linkingPSet");
        cache->tryLoadSessionFromKey(linkingPSet, "onnxModelPath", sess_opts);
      }

      // 2) Inference models (TrackstersProducer / TracksterLinksProducer when regressionAndPid)
      const std::string inferenceAlgo = modulePSet.getParameter<std::string>("inferenceAlgo");
      if (inferenceAlgo.empty()) {
        // Inference disabled => do not scan anything else.
        return cache;
      }

      const std::string infPSetName = std::string{"pluginInferenceAlgo"} + inferenceAlgo;
      if (!modulePSet.existsAs<edm::ParameterSet>(infPSetName, /*trackPar=*/true)) {
        // Misconfigured: inferenceAlgo set but corresponding PSet missing.
        // Keep cache as-is; inference plugin construction will effectively disable inference.
        return cache;
      }

      const auto infPSet = modulePSet.getParameter<edm::ParameterSet>(infPSetName);

      cache->tryLoadSessionFromKey(infPSet, "onnxModelPath", sess_opts);
      cache->tryLoadSessionFromKey(infPSet, "onnxPIDModelPath", sess_opts);
      cache->tryLoadSessionFromKey(infPSet, "onnxEnergyModelPath", sess_opts);

      return cache;
    }

  private:
    void tryLoadSessionFromKey(edm::ParameterSet const& pset, char const* key, Ort::SessionOptions const& sess_opts) {
      if (!pset.existsAs<std::string>(key, /*trackPar=*/true)) {
        return;
      }
      const std::string model = pset.getParameter<std::string>(key);
      if (model.empty()) {
        return;
      }

      const std::string fullPath = edm::FileInPath(model).fullPath();

      sessionsByFullPath.try_emplace(fullPath, std::make_unique<cms::Ort::ONNXRuntime>(fullPath, &sess_opts));
    }
  };

}  // namespace ticl

#endif
