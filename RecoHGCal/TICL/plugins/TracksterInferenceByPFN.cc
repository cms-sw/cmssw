#include "RecoHGCal/TICL/interface/TracksterInferenceByPFN.h"
#include "RecoHGCal/TICL/interface/TracksterInferenceAlgoFactory.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <span>
#include <vector>

namespace ticl {

  TracksterInferenceByPFN::TracksterInferenceByPFN(const edm::ParameterSet& conf, TICLONNXGlobalCache const* cache)
      : TracksterInferenceAlgoBase(conf, cache),
        inputNames_(conf.getParameter<std::vector<std::string>>("inputNames")),
        output_en_(conf.getParameter<std::vector<std::string>>("output_en")),
        output_id_(conf.getParameter<std::vector<std::string>>("output_id")),
        eidMinClusterEnergy_(conf.getParameter<double>("eid_min_cluster_energy")),
        eidNLayers_(conf.getParameter<int>("eid_n_layers")),
        eidNClusters_(conf.getParameter<int>("eid_n_clusters")),
        doPID_(conf.getParameter<int>("doPID")),
        doRegression_(conf.getParameter<int>("doRegression")),
        miniBatchSize_(conf.getUntrackedParameter<int>("miniBatchSize", 64)) {
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
  }

  void TracksterInferenceByPFN::runInference(const std::vector<reco::CaloCluster>& layerClusters,
                                             std::vector<Trackster>& tracksters,
                                             const hgcal::RecHitTools& rhtools) const {
    if (!enabled_ || tracksters.empty()) {
      return;
    }

    std::vector<int> indices;
    indices.reserve(tracksters.size());

    for (int i = 0; i < static_cast<int>(tracksters.size()); ++i) {
      bool anyBarrel = false;
      for (const unsigned int& v : tracksters[i].vertices()) {
        if (rhtools.isBarrel(layerClusters[v].seed())) {
          anyBarrel = true;
          break;
        }
      }
      if (anyBarrel) {
        continue;
      }

      tracksters[i].setRegressedEnergy(0.f);
      tracksters[i].zeroProbabilities();
      indices.push_back(i);
    }

    const int total = static_cast<int>(indices.size());
    if (total == 0) {
      return;
    }

    const int mb = std::max(1, miniBatchSize_);

    // Scratch buffers are local to this event.
    TracksterInferenceAlgoBase::OrtScratch ortScratch;
    ortScratch.inputs.resize(2);
    ortScratch.input_shapes.resize(2);
    ortScratch.clearPerEvent();

    // Reused within the event to avoid minibatch-level churn.
    std::vector<int> seenClusters(eidNLayers_);
    std::vector<int> clusterIndices;

    for (int start = 0; start < total; start += mb) {
      const int n = std::min(mb, total - start);

      // Input shapes:
      //  - [B, L, C, F_lc] for layer-cluster tensor
      //  - [B, F_tr] for trackster-level features
      ortScratch.input_shapes[0] = {n, eidNLayers_, eidNClusters_, eidNFeatures_};
      ortScratch.input_shapes[1] = {n, eidNFeatures_};

      const size_t nLC = static_cast<size_t>(n) * eidNLayers_ * eidNClusters_ * eidNFeatures_;
      const size_t nTR = static_cast<size_t>(n) * eidNFeatures_;

      auto& lcTensor = ortScratch.inputs[0];
      auto& trTensor = ortScratch.inputs[1];

      lcTensor.assign(nLC, 0.f);  // sparse fill, must zero
      trTensor.resize(nTR);       // fully overwritten below

      for (int bi = 0; bi < n; ++bi) {
        const int tsIdx = indices[start + bi];
        Trackster const& ts = tracksters[tsIdx];

        const size_t base_tr = static_cast<size_t>(bi) * eidNFeatures_;
        trTensor[base_tr + 0] = static_cast<float>(ts.raw_energy());
        trTensor[base_tr + 1] = static_cast<float>(ts.raw_em_energy());
        trTensor[base_tr + 2] = static_cast<float>(ts.barycenter().x());
        trTensor[base_tr + 3] = static_cast<float>(ts.barycenter().y());
        trTensor[base_tr + 4] = static_cast<float>(std::abs(ts.barycenter().z()));
        trTensor[base_tr + 5] = static_cast<float>(std::abs(ts.barycenter().eta()));
        trTensor[base_tr + 6] = static_cast<float>(ts.barycenter().phi());

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

          const size_t base_lc =
              (static_cast<size_t>(bi) * eidNLayers_ + static_cast<size_t>(j)) * (eidNClusters_ * eidNFeatures_) +
              static_cast<size_t>(seenClusters[j]) * eidNFeatures_;

          lcTensor[base_lc + 0] = static_cast<float>(cl.energy() / static_cast<float>(ts.vertex_multiplicity(k)));
          lcTensor[base_lc + 1] = static_cast<float>(std::abs(cl.eta()));
          lcTensor[base_lc + 2] = static_cast<float>(cl.phi());
          lcTensor[base_lc + 3] = static_cast<float>(cl.x());
          lcTensor[base_lc + 4] = static_cast<float>(cl.y());
          lcTensor[base_lc + 5] = static_cast<float>(std::abs(cl.z()));
          lcTensor[base_lc + 6] = static_cast<float>(cl.hitsAndFractions().size());

          ++seenClusters[j];
        }
      }

      if (doRegression_ != 0 && onnxEnergySession_ != nullptr) {
        ortScratch.outputs.clear();

        onnxEnergySession_->runInto(
            inputNames_, ortScratch.inputs, ortScratch.input_shapes, output_en_, ortScratch.outputs, {}, n);

        if (!ortScratch.outputs.empty() && !output_en_.empty()) {
          auto const& energy = ortScratch.outputs[0];
          for (int bi = 0; bi < n; ++bi) {
            auto& ts = tracksters[indices[start + bi]];
            const float regE = energy[bi];
            const float finalE = (ts.raw_energy() > eidMinClusterEnergy_) ? regE : static_cast<float>(ts.raw_energy());
            ts.setRegressedEnergy(finalE);
          }
        }
      }

      if (doPID_ != 0 && onnxPIDSession_ != nullptr) {
        ortScratch.outputs.clear();

        onnxPIDSession_->runInto(
            inputNames_, ortScratch.inputs, ortScratch.input_shapes, output_id_, ortScratch.outputs, {}, n);

        if (!ortScratch.outputs.empty() && !output_id_.empty()) {
          float* probs = ortScratch.outputs[0].data();
          for (int bi = 0; bi < n; ++bi) {
            auto& ts = tracksters[indices[start + bi]];
            ts.setProbabilities(probs);
            probs += ts.id_probabilities().size();
          }
        }
      }
    }
  }

  void TracksterInferenceByPFN::fillPSetDescription(edm::ParameterSetDescription& iDesc) {
    TracksterInferenceAlgoBase::fillPSetDescription(iDesc);

    iDesc.add<std::string>("onnxPIDModelPath", "")->setComment("Path to ONNX PID model. If empty, PID is skipped.");
    iDesc.add<std::string>("onnxEnergyModelPath", "")
        ->setComment("Path to ONNX energy model. If empty, regression is skipped.");

    iDesc.add<std::vector<std::string>>("inputNames", {"input", "input_tr_features"});
    iDesc.add<std::vector<std::string>>("output_en", {"enreg_output"});
    iDesc.add<std::vector<std::string>>("output_id", {"pid_output"});

    iDesc.add<double>("eid_min_cluster_energy", 1.0);
    iDesc.add<int>("eid_n_layers", 50);
    iDesc.add<int>("eid_n_clusters", 10);
    iDesc.add<int>("doPID", 1);
    iDesc.add<int>("doRegression", 1);

    iDesc.addUntracked<int>("miniBatchSize", 64)
        ->setComment("Mini-batch size for inference to limit peak memory usage.");
  }

}  // namespace ticl
