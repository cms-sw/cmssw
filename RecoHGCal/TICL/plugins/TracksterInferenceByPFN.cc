#include "RecoHGCal/TICL/interface/TracksterInferenceByPFN.h"
#include "RecoHGCal/TICL/interface/TracksterInferenceAlgoFactory.h"

#include <algorithm>
#include <cmath>
#include <numeric>

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

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
        doRegression_(conf.getParameter<int>("doRegression")) {
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

    // Ensure vectors have stable "outer" size; inner buffers are resized per event.
    inputs_.resize(2);
    outputs_.clear();
  }

  void TracksterInferenceByPFN::inputData(const std::vector<reco::CaloCluster>& layerClusters,
                                          std::vector<Trackster>& tracksters,
                                          const hgcal::RecHitTools& rhtools) {
    if (!enabled_) {
      batchSize_ = 0;
      return;
    }

    tracksterIndices_.clear();

    // Keep logic consistent with your current code (no physics-side behavior changes here).
    for (int i = 0; i < static_cast<int>(tracksters.size()); ++i) {
      for (const unsigned int& vertex : tracksters[i].vertices()) {
        if (rhtools.isBarrel(layerClusters[vertex].seed())) {
          continue;
        }
      }
      tracksters[i].setRegressedEnergy(0.f);
      tracksters[i].zeroProbabilities();
      tracksterIndices_.push_back(i);
    }

    batchSize_ = static_cast<int>(tracksterIndices_.size());
    if (batchSize_ == 0) {
      return;
    }

    // Shapes
    input_shapes_.clear();
    input_shapes_.push_back({batchSize_, eidNLayers_, eidNClusters_, eidNFeatures_});  // LC tensor
    input_shapes_.push_back({batchSize_, eidNFeatures_});                              // trackster tensor

    const size_t nLC = static_cast<size_t>(batchSize_) * eidNLayers_ * eidNClusters_ * eidNFeatures_;
    const size_t nTR = static_cast<size_t>(batchSize_) * eidNFeatures_;

    // Resize buffers. Only LC needs to be zeroed because filling is sparse.
    inputs_[0].resize(nLC);
    std::fill(inputs_[0].begin(), inputs_[0].end(), 0.f);

    inputs_[1].resize(nTR);
    // No std::fill for inputs_[1]: we overwrite every element deterministically.

    for (int i = 0; i < batchSize_; ++i) {
      const Trackster& trackster = tracksters[tracksterIndices_[i]];

      const int base_tr = i * eidNFeatures_;
      inputs_[1][base_tr + 0] = static_cast<float>(trackster.raw_energy());
      inputs_[1][base_tr + 1] = static_cast<float>(trackster.raw_em_energy());
      inputs_[1][base_tr + 2] = static_cast<float>(trackster.barycenter().x());
      inputs_[1][base_tr + 3] = static_cast<float>(trackster.barycenter().y());
      inputs_[1][base_tr + 4] = static_cast<float>(std::abs(trackster.barycenter().z()));
      inputs_[1][base_tr + 5] = static_cast<float>(std::abs(trackster.barycenter().eta()));
      inputs_[1][base_tr + 6] = static_cast<float>(trackster.barycenter().phi());

      // Sort clusters by energy (descending)
      std::vector<int> clusterIndices(trackster.vertices().size());
      std::iota(clusterIndices.begin(), clusterIndices.end(), 0);

      std::sort(clusterIndices.begin(), clusterIndices.end(), [&layerClusters, &trackster](int a, int b) {
        return layerClusters[trackster.vertices(a)].energy() > layerClusters[trackster.vertices(b)].energy();
      });

      std::vector<int> seenClusters(eidNLayers_, 0);

      for (int k : clusterIndices) {
        const reco::CaloCluster& cluster = layerClusters[trackster.vertices(k)];
        const int j = rhtools.getLayerWithOffset(cluster.hitsAndFractions()[0].first) - 1;

        if (j < 0 || j >= eidNLayers_) {
          continue;
        }
        if (seenClusters[j] >= eidNClusters_) {
          continue;
        }

        const int base_lc = (i * eidNLayers_ + j) * (eidNClusters_ * eidNFeatures_) + seenClusters[j] * eidNFeatures_;

        inputs_[0][base_lc + 0] =
            static_cast<float>(cluster.energy() / static_cast<float>(trackster.vertex_multiplicity(k)));
        inputs_[0][base_lc + 1] = static_cast<float>(std::abs(cluster.eta()));
        inputs_[0][base_lc + 2] = static_cast<float>(cluster.phi());
        inputs_[0][base_lc + 3] = static_cast<float>(cluster.x());
        inputs_[0][base_lc + 4] = static_cast<float>(cluster.y());
        inputs_[0][base_lc + 5] = static_cast<float>(std::abs(cluster.z()));
        inputs_[0][base_lc + 6] = static_cast<float>(cluster.hitsAndFractions().size());

        ++seenClusters[j];
      }
    }
  }

  void TracksterInferenceByPFN::runInference(std::vector<Trackster>& tracksters) {
    if (!enabled_ || batchSize_ == 0) {
      return;
    }

    // Regression (energy)
    if (doRegression_ != 0 && onnxEnergySession_ != nullptr) {
      // outputs_ reused; runInto will resize as needed.
      onnxEnergySession_->runInto(inputNames_, inputs_, input_shapes_, output_en_, outputs_, {}, batchSize_);

      if (!outputs_.empty() && !output_en_.empty()) {
        auto const& energyOutput = outputs_[0];
        for (int i = 0; i < batchSize_; ++i) {
          auto& ts = tracksters[tracksterIndices_[i]];
          const float regE = energyOutput[i];
          const float finalE = (ts.raw_energy() > eidMinClusterEnergy_) ? regE : static_cast<float>(ts.raw_energy());
          ts.setRegressedEnergy(finalE);
        }
      }
    }

    // PID
    if (doPID_ != 0 && onnxPIDSession_ != nullptr) {
      onnxPIDSession_->runInto(inputNames_, inputs_, input_shapes_, output_id_, outputs_, {}, batchSize_);

      if (!outputs_.empty() && !output_id_.empty()) {
        float* probs = outputs_[0].data();
        for (int i = 0; i < batchSize_; ++i) {
          auto& ts = tracksters[tracksterIndices_[i]];
          ts.setProbabilities(probs);
          probs += ts.id_probabilities().size();
        }
      }
    }
  }

  void TracksterInferenceByPFN::fillPSetDescription(edm::ParameterSetDescription& iDesc) {
    TracksterInferenceAlgoBase::fillPSetDescription(iDesc);

    iDesc.add<std::string>("onnxPIDModelPath", "")
        ->setComment("Path to ONNX PID model. If empty, PID inference is skipped.");
    iDesc.add<std::string>("onnxEnergyModelPath", "")
        ->setComment("Path to ONNX energy model. If empty, energy regression is skipped.");

    iDesc.add<std::vector<std::string>>("inputNames", {"input", "input_tr_features"});
    iDesc.add<std::vector<std::string>>("output_en", {"enreg_output"});
    iDesc.add<std::vector<std::string>>("output_id", {"pid_output"});

    iDesc.add<double>("eid_min_cluster_energy", 1.0);
    iDesc.add<int>("eid_n_layers", 50);
    iDesc.add<int>("eid_n_clusters", 10);
    iDesc.add<int>("doPID", 1);
    iDesc.add<int>("doRegression", 1);
  }

}  // namespace ticl
