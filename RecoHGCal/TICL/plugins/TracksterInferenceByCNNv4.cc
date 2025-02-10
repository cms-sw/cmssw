#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"
#include "RecoHGCal/TICL/interface/TracksterInferenceByCNNv4.h"
#include "RecoHGCal/TICL/interface/TracksterInferenceAlgoFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoHGCal/TICL/interface/PatternRecognitionAlgoBase.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"
#include "TrackstersPCA.h"

namespace ticl {
  using namespace cms::Ort;  // Use ONNXRuntime namespace

  // Constructor for TracksterInferenceByCNNv4
  TracksterInferenceByCNNv4::TracksterInferenceByCNNv4(const edm::ParameterSet& conf)
      : TracksterInferenceAlgoBase(conf),
        modelPath_(conf.getParameter<edm::FileInPath>("onnxModelPath").fullPath()),  // Path to the PID model CLU3D
        inputNames_(conf.getParameter<std::vector<std::string>>("inputNames")),      // Define input names for inference
        outputNames_(conf.getParameter<std::vector<std::string>>("outputNames")),   // Define output names for inference
        eidMinClusterEnergy_(conf.getParameter<double>("eid_min_cluster_energy")),  // Minimum cluster energy
        eidNLayers_(conf.getParameter<int>("eid_n_layers")),                        // Number of layers
        eidNClusters_(conf.getParameter<int>("eid_n_clusters")),                    // Number of clusters
        doPID_(conf.getParameter<int>("doPID")),                                    // Number of clusters
        doRegression_(conf.getParameter<int>("doRegression"))                       // Number of clusters
  {
    // Initialize ONNX Runtime sessions for PID and Energy models
    static std::unique_ptr<cms::Ort::ONNXRuntime> onnxRuntimeInstance =
        std::make_unique<cms::Ort::ONNXRuntime>(modelPath_.c_str());
    onnxSession_ = onnxRuntimeInstance.get();
  }

  // Method to process input data and prepare it for inference
  void TracksterInferenceByCNNv4::inputData(const std::vector<reco::CaloCluster>& layerClusters,
                                            std::vector<Trackster>& tracksters) {
    tracksterIndices_.clear();  // Clear previous indices
    for (int i = 0; i < static_cast<int>(tracksters.size()); i++) {
      float sumClusterEnergy = 0.;
      for (const unsigned int& vertex : tracksters[i].vertices()) {
        sumClusterEnergy += static_cast<float>(layerClusters[vertex].energy());
        if (sumClusterEnergy >= eidMinClusterEnergy_) {
          tracksters[i].setRegressedEnergy(0.f);  // Set regressed energy to 0
          tracksters[i].zeroProbabilities();      // Zero out probabilities
          tracksterIndices_.push_back(i);         // Add index to the list
          break;
        }
      }
    }

    // Prepare input shapes and data for inference
    batchSize_ = static_cast<int>(tracksterIndices_.size());
    if (batchSize_ == 0)
      return;  // Exit if no tracksters

    std::vector<int64_t> inputShape = {batchSize_, eidNLayers_, eidNClusters_, eidNFeatures_};
    input_shapes_ = {inputShape};

    input_Data_.clear();
    input_Data_.emplace_back(batchSize_ * eidNLayers_ * eidNClusters_ * eidNFeatures_, 0);

    for (int i = 0; i < batchSize_; i++) {
      const Trackster& trackster = tracksters[tracksterIndices_[i]];

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
        int j = rhtools_.getLayerWithOffset(cluster.hitsAndFractions()[0].first) - 1;
        if (j < eidNLayers_ && seenClusters[j] < eidNClusters_) {
          auto index = (i * eidNLayers_ + j) * eidNFeatures_ * eidNClusters_ + seenClusters[j] * eidNFeatures_;
          input_Data_[0][index] =
              static_cast<float>(cluster.energy() / static_cast<float>(trackster.vertex_multiplicity(k)));
          input_Data_[0][index + 1] = static_cast<float>(std::abs(cluster.eta()));
          input_Data_[0][index + 2] = static_cast<float>(cluster.phi());
          seenClusters[j]++;
        }
      }
    }
  }

  // Method to run inference and update tracksters
  void TracksterInferenceByCNNv4::runInference(std::vector<Trackster>& tracksters) {
    if (batchSize_ == 0)
      return;  // Exit if no batch

    std::vector<std::vector<float>> outputTensors;
    outputTensors = onnxSession_->run(inputNames_, input_Data_, input_shapes_, outputNames_, batchSize_);
    if (doPID_ and doRegression_) {
      // Run energy model inference
      if (!outputNames_.empty()) {
        for (int i = 0; i < static_cast<int>(batchSize_); i++) {
          const float energy = outputTensors[0][i];
          tracksters[tracksterIndices_[i]].setRegressedEnergy(energy);  // Update energy
        }
      }
    }

    if (doPID_) {
      // Run PID model inference
      if (!outputNames_.empty()) {
        int probsIdx = outputNames_.empty() ? 0 : 1;
        std::vector<float> vec = outputTensors[probsIdx];
        float* probs = vec.data();
        for (int i = 0; i < batchSize_; i++) {
          tracksters[tracksterIndices_[i]].setProbabilities(probs);             // Update probabilities
          probs += tracksters[tracksterIndices_[i]].id_probabilities().size();  // Move to next set of probabilities
        }
      }
    }
  }

  // Method to fill parameter set description for configuration
  void TracksterInferenceByCNNv4::fillPSetDescription(edm::ParameterSetDescription& iDesc) {
    iDesc.add<int>("algo_verbosity", 0);
    iDesc
        .add<edm::FileInPath>("onnxModelPath",
                              edm::FileInPath("RecoHGCal/TICL/data/ticlv4/onnx_models/energy_id_v0.onnx"))
        ->setComment("Path to ONNX PID model CLU3D");
    iDesc.add<std::vector<std::string>>("inputNames", {"input:0"});
    iDesc.add<std::vector<std::string>>("outputNames", {"output/regressed_energy:0", "output/id_probabilities:0"});
    iDesc.add<double>("eid_min_cluster_energy", 1.0);
    iDesc.add<int>("eid_n_layers", 50);
    iDesc.add<int>("eid_n_clusters", 10);
    iDesc.add<int>("doPID", 1);
    iDesc.add<int>("doRegression", 0);
  }
}  // namespace ticl
