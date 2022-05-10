
#include "RecoEgamma/EgammaTools/interface/EgammaDNNHelper.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/FileInPath.h"
#include <iostream>
#include <fstream>
using namespace egammaTools;

EgammaDNNHelper::EgammaDNNHelper(const DNNConfiguration& cfg,
                                 const ModelSelector& modelSelector,
                                 const std::vector<std::string>& availableVars)
    : cfg_(cfg), modelSelector_(modelSelector), nModels_(cfg_.modelsFiles.size()), graphDefs_(cfg_.modelsFiles.size()) {
  initTensorFlowGraphs();
  initScalerFiles(availableVars);
}

void EgammaDNNHelper::initTensorFlowGraphs() {
  // load the graph definition
  LogDebug("EgammaDNNHelper") << "Loading " << nModels_ << " graphs";
  size_t i = 0;
  for (const auto& model_file : cfg_.modelsFiles) {
    graphDefs_[i] =
        std::unique_ptr<tensorflow::GraphDef>(tensorflow::loadGraphDef(edm::FileInPath(model_file).fullPath()));
    i++;
  }
}

std::vector<tensorflow::Session*> EgammaDNNHelper::getSessions() const {
  std::vector<tensorflow::Session*> sessions;
  LogDebug("EgammaDNNHelper") << "Starting " << nModels_ << " TF sessions";
  for (const auto& graphDef : graphDefs_) {
    sessions.push_back(tensorflow::createSession(graphDef.get()));
  }
  LogDebug("EgammaDNNHelper") << "TF sessions started";
  return sessions;
}

void EgammaDNNHelper::initScalerFiles(const std::vector<std::string>& availableVars) {
  for (const auto& scaler_file : cfg_.scalersFiles) {
    // Parse scaler configuration
    std::vector<ScalerConfiguration> features;
    std::ifstream inputfile_scaler{edm::FileInPath(scaler_file).fullPath()};
    int ninputs = 0;
    if (inputfile_scaler.fail()) {
      throw cms::Exception("MissingFile") << "Scaler file for PFid DNN not found";
    } else {
      // Now read mean, scale factors for each variable
      float par1, par2;
      std::string varName, type_str;
      uint type;
      while (inputfile_scaler >> varName >> type_str >> par1 >> par2) {
        if (type_str == "stdscale")
          type = 1;
        else if (type_str == "minmax")
          type = 2;
        else if (type_str == "custom1")  // 2*((X_train - minValues)/(MaxMinusMin)) -1.0
          type = 3;
        else
          type = 0;
        features.push_back(ScalerConfiguration{.varName = varName, .type = type, .par1 = par1, .par2 = par2});
        // Protection for mismatch between requested variables and the available ones
        auto match = std::find(availableVars.begin(), availableVars.end(), varName);
        if (match == std::end(availableVars)) {
          throw cms::Exception("MissingVariable")
              << "Requested variable (" << varName << ") not available between DNN inputs";
        }
        ninputs += 1;
      }
    }
    inputfile_scaler.close();
    featuresMap_.push_back(features);
    nInputs_.push_back(ninputs);
  }
}

std::pair<uint, std::vector<float>> EgammaDNNHelper::getScaledInputs(
    const std::map<std::string, float>& variables) const {
  // Call the modelSelector function passing the variables map to return
  // the modelIndex to be used for the current candidate
  const auto modelIndex = modelSelector_(variables);
  std::vector<float> inputs;
  // Loop on the list of requested variables and scaling values for the specific modelIndex
  // Different type of scaling are available: 0=no scaling, 1=standard scaler, 2=minmax
  for (auto& [varName, type, par1, par2] : featuresMap_[modelIndex]) {
    if (type == 1)  // Standard scaling
      inputs.push_back((variables.at(varName) - par1) / par2);
    else if (type == 2)  // MinMax
      inputs.push_back((variables.at(varName) - par1) / (par2 - par1));
    else if (type == 3)  //2*((X_train - minValues)/(MaxMinusMin)) -1.0
      inputs.push_back(2 * (variables.at(varName) - par1) / (par2 - par1) - 1.);
    else {
      inputs.push_back(variables.at(varName));  // Do nothing on the variable
    }
    //Protection for mismatch between requested variables and the available ones
    // have been added when the scaler config are loaded --> here we know that the variables are available
  }
  return std::make_pair(modelIndex, inputs);
}

std::vector<std::vector<float>> EgammaDNNHelper::evaluate(const std::vector<std::map<std::string, float>>& candidates,
                                                          const std::vector<tensorflow::Session*>& sessions) const {
  /*
    Evaluate the PFID DNN for all the electrons/photons. 
    nModels_ are defined depending on modelIndex  --> we need to build N input tensors to evaluate
    the DNNs with batching.
    
    1) Get all the variable for each candidate  vector<map<string:float>>
    2) Scale the input and select the variables for each model
    2) Prepare the input tensors for the  models
    3) Run the models and get the output for each candidate
    4) Sort the output by candidate index
    5) Return the DNN outputs 

    */
  size_t nCandidates = candidates.size();
  std::vector<std::vector<int>> indexMap(nModels_);  // for each model; the list of candidate index is saved
  std::vector<std::vector<float>> inputsVectors(nCandidates);
  std::vector<uint> counts(nModels_);

  LogDebug("EgammaDNNHelper") << "Working on " << nCandidates << " candidates";

  int icand = 0;
  for (auto& candidate : candidates) {
    LogDebug("EgammaDNNHelper") << "Working on candidate: " << icand;
    const auto& [model_index, inputs] = getScaledInputs(candidate);
    counts[model_index] += 1;
    indexMap[model_index].push_back(icand);
    inputsVectors[icand] = inputs;
    icand++;
  }

  // Prepare one input tensors for each model
  std::vector<tensorflow::Tensor> input_tensors(nModels_);
  // Pointers for filling efficiently the input tensors
  std::vector<float*> input_tensors_pointer(nModels_);
  for (size_t i = 0; i < nModels_; i++) {
    LogDebug("EgammaDNNHelper") << "Initializing TF input " << i << " with rows:" << counts[i]
                                << " and cols:" << nInputs_[i];
    input_tensors[i] = tensorflow::Tensor{tensorflow::DT_FLOAT, {counts[i], nInputs_[i]}};
    input_tensors_pointer[i] = input_tensors[i].flat<float>().data();
  }

  // Filling the input tensors
  for (size_t m = 0; m < nModels_; m++) {
    LogDebug("EgammaDNNHelper") << "Loading TF input tensor for model: " << m;
    float* T = input_tensors_pointer[m];
    for (size_t cand_index : indexMap[m]) {
      for (size_t k = 0; k < nInputs_[m]; k++, T++) {  //Note the input tensor pointer incremented
        *T = inputsVectors[cand_index][k];
      }
    }
  }

  // Define the output and run
  // Define the output and run
  std::vector<std::pair<int, std::vector<float>>> outputs;
  // Run all the models
  for (size_t m = 0; m < nModels_; m++) {
    if (counts[m] == 0)
      continue;  //Skip model witout inputs
    std::vector<tensorflow::Tensor> output;
    LogDebug("EgammaDNNHelper") << "Run model: " << m << " with " << counts[m] << "objects";
    tensorflow::run(sessions[m], {{cfg_.inputTensorName, input_tensors[m]}}, {cfg_.outputTensorName}, &output);
    // Get the output and save the ElectronDNNEstimator::outputDim numbers along with the ele index
    const auto& r = output[0].tensor<float, 2>();
    // Iterate on the list of elements in the batch --> many electrons
    LogDebug("EgammaDNNHelper") << "Model " << m << " has " << cfg_.outputDim[m] << " nodes!";
    for (uint b = 0; b < counts[m]; b++) {
      //auto outputDim=cfg_.outputDim;
      std::vector<float> result(cfg_.outputDim[m]);
      for (size_t k = 0; k < cfg_.outputDim[m]; k++) {
        result[k] = r(b, k);
        LogDebug("EgammaDNNHelper") << "For Object " << b + 1 << " : Node " << k + 1 << " score = " << r(b, k);
      }
      // Get the original index of the electorn in the original order
      const auto cand_index = indexMap[m][b];
      outputs.push_back(std::make_pair(cand_index, result));
    }
  }
  // Now we have just to re-order the outputs
  std::sort(outputs.begin(), outputs.end());
  std::vector<std::vector<float>> final_outputs(outputs.size());
  std::transform(outputs.begin(), outputs.end(), final_outputs.begin(), [](auto a) { return a.second; });

  return final_outputs;
}
