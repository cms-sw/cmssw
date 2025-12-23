
#include "RecoEgamma/EgammaTools/interface/EgammaDNNHelper.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/FileInPath.h"
#include <iostream>
#include <fstream>
using namespace egammaTools;

EgammaDNNHelper::EgammaDNNHelper(const DNNConfiguration& cfg,
                                 const ModelSelector& modelSelector,
                                 const std::vector<std::string>& availableVars)
    : cfg_(cfg),
      modelSelector_(modelSelector),
      nModels_(cfg_.modelsFiles.size()),
      onnx_sessions_(cfg_.modelsFiles.size()) {
  initONNXRuntimeSessions();
  initScalerFiles(availableVars);
}

void EgammaDNNHelper::initONNXRuntimeSessions() {
  // load the graph definition
  LogDebug("EgammaDNNHelper") << "Loading " << nModels_ << " ONNX models";
  size_t i = 0;
  for (auto& model_file : cfg_.modelsFiles) {
    onnx_sessions_[i] = std::make_unique<cms::Ort::ONNXRuntime>(edm::FileInPath(model_file).fullPath());
    i++;
  }
  LogDebug("EgammaDNNHelper") << "ONNX sessions initialized";
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

std::vector<std::pair<uint, std::vector<float>>> EgammaDNNHelper::evaluate(
    const std::vector<std::map<std::string, float>>& candidates) const {
  /*
    Evaluate the PFID DNN for all the electrons/photons. 
    nModels_ are defined depending on modelIndex  --> we need to build N input tensors to evaluate
    the DNNs with batching.
    
    1) Get all the variable for each candidate  vector<map<string:float>>
    2) Scale the input and select the variables for each model
    2) Prepare the input tensors for the  models
    3) Run the models and get the output for each candidate
    4) Sort the output by candidate index
    5) Return the DNN outputs along with the model index used on it

    */
  size_t nCandidates = candidates.size();
  std::vector<std::vector<uint>> indexMap(nModels_);  // for each model; the list of candidate index is saved
  std::vector<std::vector<float>> inputsVectors(nCandidates);
  std::vector<uint> counts(nModels_);

  LogDebug("EgammaDNNHelper") << "Working on " << nCandidates << " candidates";

  uint icand = 0;
  for (auto& candidate : candidates) {
    LogDebug("EgammaDNNHelper") << "Working on candidate: " << icand;
    const auto& [model_index, inputs] = getScaledInputs(candidate);
    counts[model_index] += 1;
    indexMap[model_index].push_back(icand);
    inputsVectors[icand] = inputs;
    icand++;
  }

  // Define the output and run
  // The initial output is [(cand_index,(model_index, outputs)),.. ]
  std::vector<std::pair<uint, std::pair<uint, std::vector<float>>>> outputs;
  // Run all the models
  for (size_t m = 0; m < nModels_; m++) {
    if (counts[m] == 0)
      continue;  //Skip model witout inputs

    // Prepare input data
    std::vector<float> input_data;
    input_data.reserve(counts[m] * nInputs_[m]);

    for (size_t cand_index : indexMap[m]) {
      input_data.insert(input_data.end(), inputsVectors[cand_index].begin(), inputsVectors[cand_index].end());
    }

    cms::Ort::FloatArrays input_values;
    input_values.push_back(std::move(input_data));

    std::vector<std::string> input_names = {cfg_.inputTensorName};
    std::vector<std::string> output_names = {cfg_.outputTensorName};
    // Input shape: [batch_size, nInputs]
    std::vector<std::vector<int64_t>> input_shapes = {
        {static_cast<int64_t>(counts[m]), static_cast<int64_t>(nInputs_[m])}};

    LogDebug("EgammaDNNHelper") << "Run model: " << m << " with " << counts[m] << "objects";

    auto output_values = onnx_sessions_[m]->run(input_names, input_values, input_shapes, output_names, counts[m]);

    // output_values[0] contains the results
    const auto& result_flat = output_values[0];

    // Iterate on the list of elements in the batch --> many electrons
    LogDebug("EgammaDNNHelper") << "Model " << m << " has " << cfg_.outputDim[m] << " nodes!";
    for (uint b = 0; b < counts[m]; b++) {
      //auto outputDim=cfg_.outputDim;
      std::vector<float> result(cfg_.outputDim[m]);
      for (size_t k = 0; k < cfg_.outputDim[m]; k++) {
        result[k] = result_flat[b * cfg_.outputDim[m] + k];
        LogDebug("EgammaDNNHelper") << "For Object " << b + 1 << " : Node " << k + 1 << " score = " << result[k];
      }
      // Get the original index of the electorn in the original order
      const auto cand_index = indexMap[m][b];
      outputs.push_back(std::make_pair(cand_index, std::make_pair(m, result)));
    }
  }
  // Now we have just to re-order the outputs
  std::sort(outputs.begin(), outputs.end());
  std::vector<std::pair<uint, std::vector<float>>> final_outputs(outputs.size());
  std::transform(outputs.begin(), outputs.end(), final_outputs.begin(), [](auto a) { return a.second; });

  return final_outputs;
}
