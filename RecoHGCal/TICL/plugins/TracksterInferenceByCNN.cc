#include "RecoHGCal/TICL/interface/TracksterInferenceByCNN.h"
#include "RecoHGCal/TICL/interface/TracksterInferenceAlgoFactory.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"

namespace ticl {

  TracksterInferenceByCNN::TracksterInferenceByCNN(const edm::ParameterSet& conf, TICLONNXGlobalCache const* cache)
      : TracksterInferenceAlgoBase(conf, cache),
        inputNames_(conf.getParameter<std::vector<std::string>>("inputNames")),
        outputNames_(conf.getParameter<std::vector<std::string>>("outputNames")),
        eidMinClusterEnergy_(conf.getParameter<double>("eid_min_cluster_energy")),
        eidNLayers_(conf.getParameter<int>("eid_n_layers")),
        eidNClusters_(conf.getParameter<int>("eid_n_clusters")),
        doPID_(conf.getParameter<int>("doPID")),
        miniBatchSize_(conf.getUntrackedParameter<int>("miniBatchSize", 64)) {
    const std::string modelPath = conf.getParameter<std::string>("onnxModelPath");

    if (cache_ != nullptr && !modelPath.empty()) {
      onnxSession_ = cache_->getByModelPathString(modelPath);
    }

    enabled_ = (doPID_ != 0 && onnxSession_ != nullptr);

    // Single input tensor for the CNN model
    ortScratch_.inputs.resize(1);
    ortScratch_.input_shapes.resize(1);
  }

  void TracksterInferenceByCNN::runInference(const std::vector<reco::CaloCluster>& layerClusters,
                                             std::vector<Trackster>& tracksters,
                                             const hgcal::RecHitTools& rhtools) const {
    if (!enabled_ || tracksters.empty()) {
      return;
    }

    std::vector<int> indices;
    indices.reserve(tracksters.size());

    for (int i = 0; i < static_cast<int>(tracksters.size()); ++i) {
      float sumClusterEnergy = 0.f;
      for (const unsigned int& vertex : tracksters[i].vertices()) {
        sumClusterEnergy += static_cast<float>(layerClusters[vertex].energy());
        if (sumClusterEnergy >= eidMinClusterEnergy_) {
          tracksters[i].zeroProbabilities();
          indices.push_back(i);
          break;
        }
      }
    }

    const int total = static_cast<int>(indices.size());
    if (total == 0) {
      return;
    }

    const int mb = std::max(1, miniBatchSize_);

    // Reuse buffers across events
    ortScratch_.clearPerEvent();

    std::vector<int> seenClusters;
    seenClusters.resize(eidNLayers_);

    std::vector<int> clusterIndices;

    for (int start = 0; start < total; start += mb) {
      const int n = std::min(mb, total - start);

      ortScratch_.input_shapes[0] = {n, eidNLayers_, eidNClusters_, eidNFeatures_};

      const size_t nInput = static_cast<size_t>(n) * eidNLayers_ * eidNClusters_ * eidNFeatures_;

      auto& inputTensor = ortScratch_.inputs[0];
      if (inputTensor.size() != nInput) {
        inputTensor.resize(nInput);
      }
      std::fill(inputTensor.begin(), inputTensor.end(), 0.f);

      for (int bi = 0; bi < n; ++bi) {
        const int tsIdx = indices[start + bi];
        Trackster const& ts = tracksters[tsIdx];

        const int vtxCount = static_cast<int>(ts.vertices().size());
        clusterIndices.resize(vtxCount);
        std::iota(clusterIndices.begin(), clusterIndices.end(), 0);

        std::sort(clusterIndices.begin(), clusterIndices.end(), [&layerClusters, &ts](int a, int b) {
          return layerClusters[ts.vertices(a)].energy() > layerClusters[ts.vertices(b)].energy();
        });

        std::fill(seenClusters.begin(), seenClusters.end(), 0);

        for (int k : clusterIndices) {
          const unsigned int v = ts.vertices(k);
          auto const& cl = layerClusters[v];

          const int j = rhtools.getLayerWithOffset(cl.hitsAndFractions()[0].first) - 1;
          if (j < 0 || j >= eidNLayers_) {
            continue;
          }
          if (seenClusters[j] >= eidNClusters_) {
            continue;
          }

          const int base = (bi * eidNLayers_ + j) * (eidNClusters_ * eidNFeatures_) + seenClusters[j] * eidNFeatures_;

          inputTensor[base + 0] = static_cast<float>(cl.energy() / static_cast<float>(ts.vertex_multiplicity(k)));
          inputTensor[base + 1] = static_cast<float>(std::abs(cl.eta()));
          inputTensor[base + 2] = static_cast<float>(cl.phi());

          ++seenClusters[j];
        }
      }

      ortScratch_.outputs.clear();

      onnxSession_->runInto(
          inputNames_, ortScratch_.inputs, ortScratch_.input_shapes, outputNames_, ortScratch_.outputs, {}, n);

      if (!ortScratch_.outputs.empty() && !outputNames_.empty()) {
        float* probs = ortScratch_.outputs[0].data();
        for (int bi = 0; bi < n; ++bi) {
          auto& ts = tracksters[indices[start + bi]];
          ts.setProbabilities(probs);
          probs += ts.id_probabilities().size();
        }
      }
    }
  }

  void TracksterInferenceByCNN::fillPSetDescription(edm::ParameterSetDescription& iDesc) {
    TracksterInferenceAlgoBase::fillPSetDescription(iDesc);

    iDesc.add<std::string>("onnxModelPath", "RecoHGCal/TICL/data/ticlv5/onnx_models/CNN/patternrecognition/id_v0.onnx")
        ->setComment("Path to ONNX PID model. If empty, PID is skipped.");

    iDesc.add<std::vector<std::string>>("inputNames", {"input"});
    iDesc.add<std::vector<std::string>>("outputNames", {"pid_output"});

    iDesc.add<double>("eid_min_cluster_energy", 1.0);
    iDesc.add<int>("eid_n_layers", 50);
    iDesc.add<int>("eid_n_clusters", 10);
    iDesc.add<int>("doPID", 1);

    iDesc.addUntracked<int>("miniBatchSize", 64)
        ->setComment("Mini-batch size for inference to limit peak memory usage.");
  }

}  // namespace ticl
