#ifndef RecoEcal_EgammaCoreTools_DeepSCGraphEvaluation_h
#define RecoEcal_EgammaCoreTools_DeepSCGraphEvaluation_h

#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include <vector>
#include <array>
#include <memory>
#include <string>
#include <functional>

//author: Davide Valsecchi
//description:
// Handles Tensorflow DNN graphs and variables scaler configuration.
// To be used for DeepSC.

namespace reco {

  struct DeepSCConfiguration {
    std::string modelFile;
    std::string scalerFileClusterFeatures;
    std::string scalerFileWindowFeatures;
    uint nClusterFeatures;
    uint nWindowFeatures;
    static constexpr uint nRechitsFeatures = 4;
    uint maxNClusters;
    uint maxNRechits;
    uint batchSize;
    std::string collectionStrategy;
  };

  struct DeepSCInputs {
    std::vector<std::vector<std::vector<double>>> clustersX;
    std::vector<std::vector<std::vector<std::vector<double>>>> hitsX;
    std::vector<std::vector<double>> windowX;
    std::vector<std::vector<bool>> isSeed;
  };

  class DeepSCGraphEvaluation {
  public:
    DeepSCGraphEvaluation(const DeepSCConfiguration&);
    ~DeepSCGraphEvaluation();

    std::vector<double> scaleClusterFeatures(const std::vector<double>& input) const;
    std::vector<double> scaleWindowFeatures(const std::vector<double>& inputs) const;

    std::vector<std::vector<float>> evaluate(const DeepSCInputs& inputs) const;

  private:
    void initTensorFlowGraphAndSession();
    uint readScalerConfig(std::string file, std::vector<std::pair<float, float>>& scalingParams);

    void prepareTensorflowInput(const DeepSCInputs& inputs) const;

    const DeepSCConfiguration cfg_;
    std::unique_ptr<tensorflow::GraphDef> graphDef_;
    tensorflow::Session* session_;

    std::vector<std::pair<float, float>> scalerParamsClusters_;
    std::vector<std::pair<float, float>> scalerParamsWindows_;
  };

};  // namespace reco

#endif
