#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"
#include "RecoHGCal/TICL/interface/TracksterInferenceByPFN.h"
#include "RecoHGCal/TICL/interface/TracksterInferenceAlgoFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoHGCal/TICL/interface/PatternRecognitionAlgoBase.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"
#include "TrackstersPCA.h"

namespace ticl {

  TracksterInferenceByPFN::TracksterInferenceByPFN(const edm::ParameterSet& conf,
                                                   ticl::TICLONNXGlobalCache const* cache)
      : TracksterInferenceAlgoBase(conf, cache),
        inputNames_(conf.getParameter<std::vector<std::string>>("inputNames")),
        output_en_(conf.getParameter<std::vector<std::string>>("output_en")),
        output_id_(conf.getParameter<std::vector<std::string>>("output_id")),
        eidMinClusterEnergy_(conf.getParameter<double>("eid_min_cluster_energy")),
        eidNLayers_(conf.getParameter<int>("eid_n_layers")),
        eidNClusters_(conf.getParameter<int>("eid_n_clusters")),
        doPID_(conf.getParameter<int>("doPID")),
        doRegression_(conf.getParameter<int>("doRegression")) {
    // Resolve sessions only if model paths are provided.
    std::string pidModel = conf.getParameter<std::string>("onnxPIDModelPath");
    std::string energyModel = conf.getParameter<std::string>("onnxEnergyModelPath");

    if (cache_ != nullptr) {
      if (!pidModel.empty()) {
        onnxPIDSession_ = cache_->getByModelPathString(pidModel);
      }
      if (!energyModel.empty()) {
        onnxEnergySession_ = cache_->getByModelPathString(energyModel);
      }
    }

    // Enable only if at least one requested task has an available model.
    enabled_ = ((doPID_ && onnxPIDSession_ != nullptr) || (doRegression_ && onnxEnergySession_ != nullptr));
  }

  // Method to process input data and prepare it for inference
  void TracksterInferenceByPFN::inputData(const std::vector<reco::CaloCluster>& layerClusters,
                                          std::vector<Trackster>& tracksters,
                                          const hgcal::RecHitTools& rhtools) {
    if (!enabled_) {
      batchSize_ = 0;
      return;
    }
    tracksterIndices_.clear();  // Clear previous indices
    for (int i = 0; i < static_cast<int>(tracksters.size()); i++) {
      for (const unsigned int& vertex : tracksters[i].vertices()) {
        if (rhtools.isBarrel(layerClusters[vertex].seed()))
          continue;
      }
      tracksters[i].setRegressedEnergy(0.f);  // Initialize regressed energy to 0
      tracksters[i].zeroProbabilities();      // Zero out probabilities
      tracksterIndices_.push_back(i);         // Only tracksters above threshold go to inference
    }

    // Prepare input shapes and data for inference
    batchSize_ = static_cast<int>(tracksterIndices_.size());
    if (batchSize_ == 0)
      return;  // Exit if no tracksters

    std::vector<int64_t> inputShape_lc = {batchSize_, eidNLayers_, eidNClusters_, eidNFeatures_};
    std::vector<int64_t> inputShape_tr = {batchSize_, eidNFeatures_};
    input_shapes_ = {inputShape_lc, inputShape_tr};

    input_Data_.clear();
    input_Data_.emplace_back(batchSize_ * eidNLayers_ * eidNClusters_ * eidNFeatures_, 0);
    input_Data_.emplace_back(batchSize_ * eidNFeatures_, 0);

    for (int i = 0; i < batchSize_; i++) {
      const Trackster& trackster = tracksters[tracksterIndices_[i]];
      auto index_tr = i * eidNFeatures_;
      input_Data_[1][index_tr] = static_cast<float>(trackster.raw_energy());
      input_Data_[1][index_tr + 1] = static_cast<float>(trackster.raw_em_energy());
      input_Data_[1][index_tr + 2] = static_cast<float>(trackster.barycenter().x());
      input_Data_[1][index_tr + 3] = static_cast<float>(trackster.barycenter().y());
      input_Data_[1][index_tr + 4] = static_cast<float>(std::abs(trackster.barycenter().z()));
      input_Data_[1][index_tr + 5] = static_cast<float>(std::abs(trackster.barycenter().eta()));
      input_Data_[1][index_tr + 6] = static_cast<float>(trackster.barycenter().phi());

      // Prepare indices and sort clusters based on energy
      std::vector<int> clusterIndices(trackster.vertices().size());
      for (int k = 0; k < static_cast<int>(trackster.vertices().size()); k++) {
        clusterIndices[k] = k;
      }

      std::sort(clusterIndices.begin(), clusterIndices.end(), [&layerClusters, &trackster](const int& a, const int& b) {
        return layerClusters[trackster.vertices(a)].energy() > layerClusters[trackster.vertices(b)].energy();
      });

      std::vector<int> seenClusters(eidNLayers_, 0);

      // Fill input data with cluster information
      for (const int& k : clusterIndices) {
        const reco::CaloCluster& cluster = layerClusters[trackster.vertices(k)];
        int j = rhtools.getLayerWithOffset(cluster.hitsAndFractions()[0].first) - 1;
        if (j < eidNLayers_ && seenClusters[j] < eidNClusters_) {
          auto index_lc = (i * eidNLayers_ + j) * eidNFeatures_ * eidNClusters_ + seenClusters[j] * eidNFeatures_;
          // Adding more features regarding LC, such as E, eta, phi, x, y, z, and nhits.
          input_Data_[0][index_lc] =
              static_cast<float>(cluster.energy() / static_cast<float>(trackster.vertex_multiplicity(k)));
          input_Data_[0][index_lc + 1] = static_cast<float>(std::abs(cluster.eta()));
          input_Data_[0][index_lc + 2] = static_cast<float>(cluster.phi());
          input_Data_[0][index_lc + 3] = static_cast<float>(cluster.x());
          input_Data_[0][index_lc + 4] = static_cast<float>(cluster.y());
          input_Data_[0][index_lc + 5] = static_cast<float>(std::abs(cluster.z()));
          input_Data_[0][index_lc + 6] = static_cast<float>(cluster.hitsAndFractions().size());
          seenClusters[j]++;
        }
      }
    }
  }

  // Method to run inference and update tracksters
  void TracksterInferenceByPFN::runInference(std::vector<Trackster>& tracksters) {
    if (batchSize_ == 0)
      return;  // Exit if no batch

    if (doRegression_ && onnxEnergySession_ != nullptr) {
      // Run energy model inference
      auto result = onnxEnergySession_->run(inputNames_, input_Data_, input_shapes_, output_en_, batchSize_);
      auto& energyOutput = result[0];
      if (!output_en_.empty()) {
        for (int i = 0; i < static_cast<int>(batchSize_); i++) {
          auto& ts = tracksters[tracksterIndices_[i]];
          float energy = ts.raw_energy() > eidMinClusterEnergy_ ? energyOutput[i] : ts.raw_energy();
          ts.setRegressedEnergy(energy);
        }
      }
    }
    if (doPID_ && onnxPIDSession_ != nullptr) {
      // Run PID model inference
      auto pidOutput = onnxPIDSession_->run(inputNames_, input_Data_, input_shapes_, output_id_, batchSize_);
      auto pidOutputTensor = pidOutput[0];
      float* probs = pidOutputTensor.data();
      if (!output_id_.empty()) {
        for (int i = 0; i < batchSize_; i++) {
          tracksters[tracksterIndices_[i]].setProbabilities(probs);             // Update probabilities
          probs += tracksters[tracksterIndices_[i]].id_probabilities().size();  // Move to next set of probabilities
        }
      }
    }
  }
  // Method to fill parameter set description for configuration
  void TracksterInferenceByPFN::fillPSetDescription(edm::ParameterSetDescription& iDesc) {
    iDesc.add<int>("algo_verbosity", 0);

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
