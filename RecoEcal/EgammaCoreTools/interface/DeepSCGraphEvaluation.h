#ifndef RecoEcal_EgammaCoreTools_DeepSCGraphEvaluation_h
#define RecoEcal_EgammaCoreTools_DeepSCGraphEvaluation_h

#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include <vector>
#include <array>
#include <memory>
#include <string>
#include <functional>
#include <cmath>

//author: Davide Valsecchi
//description:
// Handles Tensorflow DNN graphs and variables scaler configuration.
// To be used for DeepSC.

namespace reco {

  struct DeepSCConfiguration {
    std::string modelFile;
    std::string configFileClusterFeatures;
    std::string configFileWindowFeatures;
    std::string configFileHitsFeatures;
    uint nClusterFeatures;
    uint nWindowFeatures;
    uint nHitsFeatures;
    uint maxNClusters;
    uint maxNRechits;
    uint batchSize;
    std::string collectionStrategy;
  };

  /*
   * Structure representing the detector windows of a single events, to be evaluated with the DeepSC model.
   * The index structure is described in the following
   */

  namespace DeepSCInputs {
    enum ScalerType {
      MeanRms,  // scale as (var - mean)/rms
      MinMax,   // scale as (var - min) (max-min)
      None      // do nothing
    };
    struct InputConfig {
      // Each input variable is represented by the tuple <varname, standardization_type, par1, par2>
      std::string varName;
      ScalerType type;
      float par1;
      float par2;
    };
    typedef std::vector<InputConfig> InputConfigs;
    typedef std::map<std::string, double> FeaturesMap;

    struct Inputs {
      std::vector<std::vector<std::vector<float>>> clustersX;
      std::vector<std::vector<std::vector<std::vector<float>>>> hitsX;
      std::vector<std::vector<float>> windowX;
      std::vector<std::vector<bool>> isSeed;
    };

  };  // namespace DeepSCInputs

  class DeepSCGraphEvaluation {
  public:
    DeepSCGraphEvaluation(const DeepSCConfiguration&);
    ~DeepSCGraphEvaluation();

    std::vector<float> getScaledInputs(const DeepSCInputs::FeaturesMap& variables,
                                       const DeepSCInputs::InputConfigs& config) const;

    std::vector<std::vector<float>> evaluate(const DeepSCInputs::Inputs& inputs) const;

    // List of input variables names used to check the variables request as
    // inputs in a dynamic way from configuration file.
    // If an input variables is not found at construction time an expection is thrown.
    static const std::vector<std::string> availableClusterInputs;
    static const std::vector<std::string> availableWindowInputs;
    static const std::vector<std::string> availableHitsInputs;

    // Configuration of the input variables including the scaling parameters.
    // The list is used to define the vector of input features passed to the tensorflow model.
    DeepSCInputs::InputConfigs inputFeaturesClusters;
    DeepSCInputs::InputConfigs inputFeaturesWindows;
    DeepSCInputs::InputConfigs inputFeaturesHits;

  private:
    void initTensorFlowGraphAndSession();
    DeepSCInputs::InputConfigs readInputFeaturesConfig(std::string file,
                                                       const std::vector<std::string>& availableInputs) const;

    const DeepSCConfiguration cfg_;
    std::unique_ptr<tensorflow::GraphDef> graphDef_;
    tensorflow::Session* session_;
  };

};  // namespace reco

#endif
