#include "RecoHGCal/TICL/interface/TracksterInferenceByDNN.h"
#include "RecoHGCal/TICL/interface/TracksterInferenceAlgoFactory.h"

#include <algorithm>
#include <cmath>

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
        doRegression_(conf.getParameter<int>("doRegression")) {
    // Empty string means "disabled": the cache will not load anything and we keep nullptr sessions.
    const std::string pidModel = conf.getParameter<std::string>("onnxPIDModelPath");
    const std::string energyModel = conf.getParameter<std::string>("onnxEnergyModelPath");

    if (cache != nullptr) {
      if (!pidModel.empty()) {
        onnxPIDSession_ = cache->getByModelPathString(pidModel);
      }
      if (!energyModel.empty()) {
        onnxEnergySession_ = cache->getByModelPathString(energyModel);
      }
    }

    // Enable only if the requested tasks have sessions.
    enabled_ = ((doPID_ != 0 && onnxPIDSession_ != nullptr) || (doRegression_ != 0 && onnxEnergySession_ != nullptr));
  }

  void TracksterInferenceByDNN::inputData(const std::vector<reco::CaloCluster>& layerClusters,
                                          std::vector<Trackster>& tracksters,
                                          const hgcal::RecHitTools& rhtools) {
    if (!enabled_) {
      batchSize_ = 0;
      return;
    }

    tracksterIndices_.clear();
    for (int i = 0; i < static_cast<int>(tracksters.size()); i++) {
      float sumClusterEnergy = 0.f;
      for (const unsigned int& vertex : tracksters[i].vertices()) {
        if (rhtools.isBarrel(layerClusters[vertex].seed())) {
          continue;
        }
        sumClusterEnergy += static_cast<float>(layerClusters[vertex].energy());
        if (sumClusterEnergy >= eidMinClusterEnergy_) {
          tracksters[i].setRegressedEnergy(0.f);
          tracksters[i].zeroProbabilities();
          tracksterIndices_.push_back(i);
          break;
        }
      }
    }

    batchSize_ = static_cast<int>(tracksterIndices_.size());
    if (batchSize_ == 0) {
      return;
    }

    std::vector<int64_t> inputShape = {batchSize_, eidNLayers_, eidNClusters_, eidNFeatures_};
    input_shapes_.clear();
    input_shapes_.push_back(std::move(inputShape));

    input_Data_.clear();
    input_Data_.emplace_back(batchSize_ * eidNLayers_ * eidNClusters_ * eidNFeatures_, 0.f);

    for (int i = 0; i < batchSize_; i++) {
      const Trackster& trackster = tracksters[tracksterIndices_[i]];

      std::vector<int> clusterIndices(trackster.vertices().size());
      for (int k = 0; k < static_cast<int>(trackster.vertices().size()); k++) {
        clusterIndices[k] = k;
      }

      std::sort(clusterIndices.begin(), clusterIndices.end(), [&layerClusters, &trackster](int a, int b) {
        return layerClusters[trackster.vertices(a)].energy() > layerClusters[trackster.vertices(b)].energy();
      });

      std::vector<int> seenClusters(eidNLayers_, 0);

      for (int k : clusterIndices) {
        const reco::CaloCluster& cluster = layerClusters[trackster.vertices(k)];
        int j = rhtools.getLayerWithOffset(cluster.hitsAndFractions()[0].first) - 1;
        if (j < eidNLayers_ && seenClusters[j] < eidNClusters_) {
          const int index = (i * eidNLayers_ + j) * eidNFeatures_ * eidNClusters_ + seenClusters[j] * eidNFeatures_;
          input_Data_[0][index] =
              static_cast<float>(cluster.energy() / static_cast<float>(trackster.vertex_multiplicity(k)));
          input_Data_[0][index + 1] = static_cast<float>(std::abs(cluster.eta()));
          input_Data_[0][index + 2] = static_cast<float>(cluster.phi());
          seenClusters[j]++;
        }
      }
    }
  }

  void TracksterInferenceByDNN::runInference(std::vector<Trackster>& tracksters) {
    if (!enabled_ || batchSize_ == 0) {
      return;
    }

    if (doRegression_ != 0 && onnxEnergySession_ != nullptr) {
      auto result = onnxEnergySession_->run(inputNames_, input_Data_, input_shapes_, output_en_, batchSize_);
      auto const& energyOutputTensor = result[0];
      if (!output_en_.empty()) {
        for (int i = 0; i < batchSize_; i++) {
          const float energy = energyOutputTensor[i];
          tracksters[tracksterIndices_[i]].setRegressedEnergy(energy);
        }
      }
    }

    if (doPID_ != 0 && onnxPIDSession_ != nullptr) {
      auto pidOutput = onnxPIDSession_->run(inputNames_, input_Data_, input_shapes_, output_id_, batchSize_);
      auto pidOutputTensor = pidOutput[0];
      float* probs = pidOutputTensor.data();
      if (!output_id_.empty()) {
        for (int i = 0; i < batchSize_; i++) {
          tracksters[tracksterIndices_[i]].setProbabilities(probs);
          probs += tracksters[tracksterIndices_[i]].id_probabilities().size();
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
  }

}  // namespace ticl
