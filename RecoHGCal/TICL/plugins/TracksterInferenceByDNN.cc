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
  }

  void TracksterInferenceByDNN::runInference(const std::vector<reco::CaloCluster>& layerClusters,
                                             std::vector<Trackster>& tracksters,
                                             const hgcal::RecHitTools& rhtools) const {
    if (!enabled_ || tracksters.empty()) {
      return;
    }

    std::vector<int> indices;
    indices.reserve(tracksters.size());

    for (int i = 0; i < static_cast<int>(tracksters.size()); ++i) {
      float sumClusterEnergy = 0.f;

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

    // Scratch buffers are local to this event.
    OrtScratch ortScratch;
    ortScratch.inputs.resize(1);
    ortScratch.input_shapes.resize(1);
    ortScratch.clearPerEvent();

    // Reused within the event to avoid minibatch-level churn.
    std::vector<int> seenClusters(eidNLayers_);
    std::vector<int> clusterIndices;

    auto& in = ortScratch.inputs[0];

    for (int start = 0; start < total; start += mb) {
      const int n = std::min(mb, total - start);

      ortScratch.input_shapes[0] = {n, eidNLayers_, eidNClusters_, eidNFeatures_};

      const size_t nFloats = static_cast<size_t>(n) * eidNLayers_ * eidNClusters_ * eidNFeatures_;
      in.assign(nFloats, 0.f);

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

      if (doRegression_ != 0 && onnxEnergySession_ != nullptr) {
        ortScratch.outputs.clear();

        onnxEnergySession_->runInto(
            inputNames_, ortScratch.inputs, ortScratch.input_shapes, output_en_, ortScratch.outputs, {}, n);

        if (!ortScratch.outputs.empty() && !output_en_.empty()) {
          auto const& energy = ortScratch.outputs[0];
          for (int bi = 0; bi < n; ++bi) {
            tracksters[indices[start + bi]].setRegressedEnergy(energy[bi]);
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

    iDesc.addUntracked<int>("miniBatchSize", 64)
        ->setComment("Mini-batch size for inference to limit peak memory usage.");
  }

}  // namespace ticl
