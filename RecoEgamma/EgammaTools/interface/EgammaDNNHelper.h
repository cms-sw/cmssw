#ifndef RecoEgamma_ElectronTools_EgammaDNNHelper_h
#define RecoEgamma_ElectronTools_EgammaDNNHelper_h

#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include <vector>
#include <memory>
#include <string>
#include <functional>

//author: Davide Valsecchi
//description:
// Handles Tensorflow DNN graphs and variables scaler configuration.
// To be used for PFID egamma DNNs

namespace egammaTools {

  struct DNNConfiguration {
    std::string inputTensorName;
    std::string outputTensorName;
    std::vector<std::string> modelsFiles;
    std::vector<std::string> scalersFiles;
    std::vector<unsigned int> outputDim;
  };

  struct ScalerConfiguration {
    /* Each input variable is represented by the tuple <varname, standardization_type, par1, par2>
    * The standardization_type can be:
    * 0 = Do not scale the variable
    * 1 = standard norm. par1=mean, par2=std
    * 2 = MinMax. par1=min, par2=max */
    std::string varName;
    uint type;
    float par1;
    float par2;
  };

  // Model for function to be used on the specific candidate to get the model
  // index to be used for the evaluation.
  typedef std::function<uint(const std::map<std::string, float>&)> ModelSelector;

  class EgammaDNNHelper {
  public:
    EgammaDNNHelper(const DNNConfiguration&, const ModelSelector& sel, const std::vector<std::string>& availableVars);

    std::vector<tensorflow::Session*> getSessions() const;
    // Function getting the input vector for a specific electron, already scaled
    // together with the model index it has to be used.
    // The model index is determined by the ModelSelector functor passed in the constructor
    // which has access to all the variables.
    std::pair<uint, std::vector<float>> getScaledInputs(const std::map<std::string, float>& variables) const;

    std::vector<std::pair<uint, std::vector<float>>> evaluate(
        const std::vector<std::map<std::string, float>>& candidates,
        const std::vector<tensorflow::Session*>& sessions) const;

  private:
    void initTensorFlowGraphs();
    void initScalerFiles(const std::vector<std::string>& availableVars);

    const DNNConfiguration cfg_;
    const ModelSelector modelSelector_;
    // Number of models handled by the object
    uint nModels_;
    // Number of inputs for each loaded model
    std::vector<uint> nInputs_;

    std::vector<std::unique_ptr<const tensorflow::GraphDef>> graphDefs_;

    // List of input variables for each of the model;
    std::vector<std::vector<ScalerConfiguration>> featuresMap_;
  };

};  // namespace egammaTools

#endif
