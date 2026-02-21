#include "RecoHGCal/TICL/interface/TracksterInferenceByDNN.h"
#include "RecoHGCal/TICL/interface/TracksterInferenceAlgoFactory.h"

#include <algorithm>
#include <cmath>
#include <numeric>

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

namespace ticl {

  TracksterInferenceByDNN::TracksterInferenceByDNN(const edm::ParameterSet& conf, TICLONNXGlobalCache const* cache)
      : TracksterInferenceAlgoBase(conf, cache),
        inputNames_(conf.getParameter<std::vector<std::string>>("inputNames")),
        output_en_(conf.getParameter<std::vector<std::string>>("output_en")),
        output_id_(conf.getParameter<std::vector<std::string>>("output_id")),
        eidMinClusterEnergy_(conf.getParameter<double>("eid_min_cluster_energy")),
        eidNLayers_(conf.getParameter<int>("eid_n_layers")),
        eidNClusters_(conf.getParameter<int>("eid_n_clusters")),
        doPID_(conf.getParameter<int>("doPID")),
        doRegression_(conf.getParameter<int>("doRegression")),
        miniBatchSize_(conf.getUntrackedParameter<int>("miniBatchSize", 256)) {
    const std::string pidModel = conf.getParameter<std::string>("onnxPIDModelPath");
    const std::string energyModel = conf.getParameter<std::string>("onnxEnergyModelPath");

    if (cache_ != nullptr) {
      if (!pidModel.empty()) {
        onnxPIDSession_ = cache_->getByModelPathString(pidModel);
      }
      if (!energyModel.empty()) {
        onnxEnergySession_ = cache_->getByModelPathString(energyModel);
      }
    }

    enabled_ = ((doPID_ != 0 && onnxPIDSession_ != nullptr) || (doRegression_ != 0 && onnxEnergySession_ != nullptr));

    ortScratch_.inputs.resize(1);
    ortScratch_.input_shapes.resize(1);
  }

  void TracksterInferenceByDNN::runInference(const std::vector<reco::CaloCluster>& layerClusters,
                                             std::vector<Trackster>& tracksters,
                                             const hgcal::RecHitTools& rhtools) const {
    if (!enabled_ || tracksters.empty()) {
      return;
    }

    // ---- select tracksters (same physics logic), reset outputs once
    std::vector<int> indices;
    indices.reserve(tracksters.size());

    for (int i = 0; i < static_cast<int>(tracksters.size()); ++i) {
      float sumClusterEnergy = 0.f;

      // Note: keep the same semantics you had (skip barrel clusters, sum endcap energy)
      for (const unsigned int& v : tracksters[i].vertices()) {
        if (rhtools.isBarrel(layerClusters[v].seed())) {
          continue;
        }
        sumClusterEnergy += static_cast<float>(layerClusters[v].energy());
        if (sumClusterEnergy >= eidMinClusterEnergy_) {
          tracksters[i].setRegressedEnergy(0.f);
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

    // Per-minibatch reusable temporaries to avoid churn
    std::vector<int> seenClusters;
    seenClusters.resize(eidNLayers_);

    std::vector<int> clusterIndices;

    // Alias for input tensor
    auto& in = ortScratch_.inputs[0];

    for (int start = 0; start < total; start += mb) {
      const int n = std::min(mb, total - start);

      // shape: [B, L, C, F]
      ortScratch_.input_shapes[0] = {n, eidNLayers_, eidNClusters_, eidNFeatures_};

      const size_t nFloats = static_cast<size_t>(n) * eidNLayers_ * eidNClusters_ * eidNFeatures_;
      in.assign(nFloats, 0.f);  // sparse fill -> must zero

      // ---- build sparse tensor for this minibatch
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

          const size_t base =
              (static_cast<size_t>(bi) * eidNLayers_ + static_cast<size_t>(j)) * (eidNClusters_ * eidNFeatures_) +
              static_cast<size_t>(seenClusters[j]) * eidNFeatures_;

          in[base + 0] = static_cast<float>(cl.energy() / static_cast<float>(ts.vertex_multiplicity(k)));
          in[base + 1] = static_cast<float>(std::abs(cl.eta()));
          in[base + 2] = static_cast<float>(cl.phi());

          ++seenClusters[j];
        }
      }

      // ---- regression
      if (doRegression_ != 0 && onnxEnergySession_ != nullptr) {
        ortScratch_.outputs.clear();

        onnxEnergySession_->runInto(
            inputNames_, ortScratch_.inputs, ortScratch_.input_shapes, output_en_, ortScratch_.outputs, {}, n);

        if (!ortScratch_.outputs.empty() && !output_en_.empty()) {
          auto const& energy = ortScratch_.outputs[0];
          for (int bi = 0; bi < n; ++bi) {
            tracksters[indices[start + bi]].setRegressedEnergy(energy[bi]);
          }
        }
      }

      // ---- PID
      if (doPID_ != 0 && onnxPIDSession_ != nullptr) {
        ortScratch_.outputs.clear();

        onnxPIDSession_->runInto(
            inputNames_, ortScratch_.inputs, ortScratch_.input_shapes, output_id_, ortScratch_.outputs, {}, n);

        if (!ortScratch_.outputs.empty() && !output_id_.empty()) {
          float* probs = ortScratch_.outputs[0].data();
          for (int bi = 0; bi < n; ++bi) {
            auto& ts = tracksters[indices[start + bi]];
            ts.setProbabilities(probs);
            probs += ts.id_probabilities().size();
          }
        }
      }
    }
  }

  void TracksterInferenceByDNN::fillPSetDescription(edm::ParameterSetDescription& iDesc) {
    TracksterInferenceAlgoBase::fillPSetDescription(iDesc);

    iDesc.add<std::string>("onnxPIDModelPath", "")
        ->setComment("Path to ONNX PID model. If empty, PID inference is skipped.");
    iDesc.add<std::string>("onnxEnergyModelPath", "")
        ->setComment("Path to ONNX energy model. If empty, energy regression is skipped.");

    iDesc.add<std::vector<std::string>>("inputNames", {"input"});
    iDesc.add<std::vector<std::string>>("output_en", {"enreg_output"});
    iDesc.add<std::vector<std::string>>("output_id", {"pid_output"});

    iDesc.add<double>("eid_min_cluster_energy", 1.0);
    iDesc.add<int>("eid_n_layers", 50);
    iDesc.add<int>("eid_n_clusters", 10);
    iDesc.add<int>("doPID", 1);
    iDesc.add<int>("doRegression", 1);

    iDesc.addUntracked<int>("miniBatchSize", 256)
        ->setComment("Mini-batch size for inference to limit peak memory usage.");
  }

}  // namespace ticl
