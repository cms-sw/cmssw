#include "RecoEgamma/PhotonIdentification/interface/PhotonDNNEstimator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <fstream>
#include <memory>

PhotonDNNEstimator::PhotonDNNEstimator() : cfg_{} {}

PhotonDNNEstimator::PhotonDNNEstimator(std::vector<std::string>& models_files,
                                       std::vector<std::string>& scalers_files,
                                       std::string inputTensorName,
                                       std::string outputTensorName)
    : cfg_{.inputTensorName = inputTensorName,
           .outputTensorName = outputTensorName,
           .models_files = models_files,
           .scalers_files = scalers_files,
           .log_level = "2"} {
  nModels_ = cfg_.models_files.size();
  debug_ = cfg_.log_level == "0";
  initTensorFlowGraphs();
  initScalerFiles();
  LogDebug("PhotonDNNPFid") << "Photon PFID DNN evaluation with " << nModels_ << " models and " << nInputs_[0]
                            << " variables --> LOADED";
}

PhotonDNNEstimator::PhotonDNNEstimator(const Configuration& cfg) : cfg_(cfg) {
  // Init tensorflow sessions
  nModels_ = cfg_.models_files.size();
  debug_ = cfg_.log_level == "0";
  initTensorFlowGraphs();
  initScalerFiles();
  LogDebug("PhotonDNNPFid") << "Photon PFID DNN evaluation with " << nModels_ << " models and " << nInputs_[0]
                            << " variables --> LOADED";
}

void PhotonDNNEstimator::initTensorFlowGraphs() {
  // configure logging to show warnings (see table below)
  tensorflow::setLogging(cfg_.log_level);
  // load the graph definition
  LogDebug("PhotonDNNPFid") << "Loading " << nModels_ << " graphs";
  for (const auto& model_file : cfg_.models_files) {
    tensorflow::GraphDef* graphDef = tensorflow::loadGraphDef(model_file);  //-->should be atomic but does not compile
    graphDefs_.push_back(graphDef);
  }
  LogDebug("PhotonDNNPFid") << "Graphs loaded";
}

void PhotonDNNEstimator::initScalerFiles() {
  for (const auto& scaler_file : cfg_.scalers_files) {
    // Parse scaler configuration
    std::vector<std::tuple<std::string, uint, float, float>> features;
    std::ifstream inputfile_scaler{scaler_file};
    int ninputs = 0;
    if (inputfile_scaler.fail()) {
      throw cms::Exception("MissingFile") << "Scaler file for Electron PFid DNN not found";
    } else {
      // Now read mean, scale factors for each variable
      float par1, par2;
      std::string varname, type_str;
      uint type;
      while (inputfile_scaler >> varname >> type_str >> par1 >> par2) {
        if (type_str == "stdscale")
          type = 1;
        else if (type_str == "minmax")
          type = 2;
        else if (type_str == "custom1")  // 2*((X_train - minValues)/(MaxMinusMin)) -1.0
          type = 3;
        else
          type = 0;
        features.push_back(std::make_tuple(varname, type, par1, par2));
        auto match = std::find(
            PhotonDNNEstimator::dnnAvaibleInputs.begin(), PhotonDNNEstimator::dnnAvaibleInputs.end(), varname);
        if (match == std::end(PhotonDNNEstimator::dnnAvaibleInputs)) {
          throw cms::Exception("MissingVariable")
              << "Requested variable (" << varname << ") not available between Photon PFid DNN inputs";
        }
        ninputs += 1;
      }
    }
    inputfile_scaler.close();
    featuresMap_.push_back(features);
    nInputs_.push_back(ninputs);
  }
}

std::vector<tensorflow::Session*> PhotonDNNEstimator::getSessions() const {
  LogDebug("PhotonDNNPFid") << "Starting " << nModels_ << "TF sessions";
  std::vector<tensorflow::Session*> sessions;
  for (auto& graphDef : graphDefs_) {
    sessions.push_back(tensorflow::createSession(graphDef));
  }
  LogDebug("PhotonDNNPFid") << "TF sessions started";
  return sessions;
}

const std::array<std::string, PhotonDNNEstimator::nAvailableVars> PhotonDNNEstimator::dnnAvaibleInputs = {
    {"hadTowOverEm",
     "TrkSumPtHollow",
     "EcalRecHit",
     "SigmaIetaIeta",
     "SigmaIEtaIEtaFull5x5",
     "SigmaIEtaIPhiFull5x5",
     "EcalPFClusterIso",
     "HcalPFClusterIso",
     "HasPixelSeed",
     "R9Full5x5",
     "hcalTower"}};

std::map<std::string, float> PhotonDNNEstimator::getInputsVars(const reco::Photon& photon) const {
  // Prepare a map with all the defined variables
  std::map<std::string, float> variables;
  variables["hadTowOverEm"] = photon.hadTowOverEmValid() ? photon.hadTowOverEm() : 0;
  variables["TrkSumPtHollow"] = photon.trkSumPtHollowConeDR03();
  variables["EcalRecHit"] = photon.ecalRecHitSumEtConeDR03();
  variables["SigmaIetaIeta"] = photon.sigmaIetaIeta();
  variables["SigmaIEtaIEtaFull5x5"] = photon.full5x5_sigmaIetaIeta();
  variables["SigmaIEtaIPhiFull5x5"] = photon.full5x5_showerShapeVariables().sigmaIetaIphi;
  variables["EcalPFClusterIso"] = photon.ecalPFClusterIso();
  variables["HcalPFClusterIso"] = photon.hcalPFClusterIso();
  variables["HasPixelSeed"] = (Int_t)photon.hasPixelSeed();
  variables["R9Full5x5"] = photon.full5x5_r9();
  variables["hcalTower"] = photon.hcalTowerSumEtConeDR03();
  variables["R9Full5x5"] = photon.full5x5_r9();
  // Define more variables here and use them directly in the model config!
  return variables;
}

uint PhotonDNNEstimator::getModelIndex(const reco::Photon& photon) const {
  /* 
  Selection of the model to be applied on the Photon based on pt/eta cuts or whatever selection
  */
  uint modelIndex;
  if (std::abs(photon.eta()) <= 1.466) {
    modelIndex = 0;
  } else {
    modelIndex = 1;
  }
  return modelIndex;
}

std::pair<uint, std::vector<float>> PhotonDNNEstimator::getScaledInputs(const reco::Photon& photon) const {
  uint modelIndex = getModelIndex(photon);
  auto allInputs = getInputsVars(photon);
  std::vector<float> inputs;
  // Loop on the list of requested variables and scaling values for the specific modelIndex
  // Different type of scaling are available: 0=noscaling, 1=standard scaler, 2=minmax
  for (auto& [varName, type, par1, par2] : featuresMap_[modelIndex]) {
    if (type == 1)  // Standard scaling
      inputs.push_back((allInputs[varName] - par1) / par2);
    else if (type == 2)  // MinMax
      inputs.push_back((allInputs[varName] - par1) / (par2 - par1));
    else if (type == 3)  //2*((X_train - minValues)/(MaxMinusMin)) -1.0
      inputs.push_back(2 * (allInputs[varName] - par1) / (par2 - par1) - 1.);
    else {
      inputs.push_back(allInputs[varName]);  // Do nothing on the variable
    }
    //Protection for mismatch between requested variables and the available ones
    // have been added when the scaler config is loaded --> here we know that the variables are available
  }
  return std::make_pair(modelIndex, inputs);
}

std::vector<std::array<float, PhotonDNNEstimator::nOutputs>> PhotonDNNEstimator::evaluate(
    const reco::PhotonCollection& photons, const std::vector<tensorflow::Session*> sessions) const {
  /*
      Evaluate the Photon PFID DNN for all the Photons. 
      2 models are defined depending on eta --> we need to build 2 input tensors to evaluate
      the DNNs with batching.
      
      1) Get the inputs vector, already scaled correctly for each Photon and the modelIndex
      2) Prepare 2 input tensors for the 2 models
      3) Run the model and get the output for each Photon
      4) Sort the output by Photon index
      5) Return the DNN output 

    */
  std::vector<std::vector<int>> photonIndexMap(nModels_);  // for each model; the list of ele index is saved
  std::vector<std::vector<float>> inputsVectors;
  std::vector<uint> counts(nModels_);

  LogDebug("PhotonDNNPFid") << "Working on " << photons.size() << " photons";

  int iphoton = -1;
  for (auto& photon : photons) {
    iphoton++;
    LogDebug("PhotonDNNPFid") << "Working on photon: " << iphoton;
    auto [model_index, inputs] = getScaledInputs(photon);
    counts[model_index] += 1;
    photonIndexMap[model_index].push_back(iphoton);
    inputsVectors.push_back(inputs);
  }

  // Prepare one input tensors for each model
  std::vector<tensorflow::Tensor> input_tensors(nModels_);
  // Pointers for filling efficiently the input tensors
  std::vector<float*> input_tensors_pointer(nModels_);
  for (uint i = 0; i < nModels_; i++) {
    LogDebug("PhotonDNNPFid") << "Initializing TF input " << i << " with rows:" << counts[i]
                              << " and cols:" << nInputs_[i];
    input_tensors[i] = tensorflow::Tensor{tensorflow::DT_FLOAT, {counts[i], nInputs_[i]}};
    input_tensors_pointer[i] = input_tensors[i].flat<float>().data();
  }

  // Filling the input tensors
  for (uint m = 0; m < nModels_; m++) {
    LogDebug("PhotonDNNPFid") << "Loading TF input tensor for model: " << m;
    float* T = input_tensors_pointer[m];
    for (int photon_index : photonIndexMap[m]) {
      for (int k = 0; k < nInputs_[m]; k++, T++) {  //Note the input tensor pointer incremented
        *T = inputsVectors[photon_index][k];
      }
    }
  }

  // Define the output and run
  std::vector<std::pair<int, std::array<float, PhotonDNNEstimator::nOutputs>>> outputs;
  // Run all the models
  for (uint m = 0; m < nModels_; m++) {
    if (counts[m] == 0)
      continue;  //Skip model without inputs
    std::vector<tensorflow::Tensor> output;
    LogDebug("PhotonDNNPFid") << "Run model: " << m << " with " << counts[m] << " photons";
    tensorflow::run(sessions[m], {{cfg_.inputTensorName, input_tensors[m]}}, {cfg_.outputTensorName}, &output);
    // Get the output and save the PhotonDNNEstimator::nOutputs numbers along with the photon index
    auto r = output[0].tensor<float, 2>();
    // Iterate on the list of elements in the batch --> many Photons
    for (uint b = 0; b < counts[m]; b++) {
      std::array<float, PhotonDNNEstimator::nOutputs> result;
      for (uint k = 0; k < PhotonDNNEstimator::nOutputs; k++)
        result[k] = r(b, k);
      // Get the original index of the electorn in the original order
      int photon_index = photonIndexMap[m][b];
      LogDebug("PhotonDNNPFid") << "DNN output, model " << m << " photon " << photon_index << " : " << result[0];
      outputs.push_back(std::make_pair(photon_index, result));
    }
  }

  // Now we have just to re-order the outputs
  std::sort(outputs.begin(), outputs.end());
  std::vector<std::array<float, PhotonDNNEstimator::nOutputs>> final_outputs(outputs.size());
  std::transform(outputs.begin(), outputs.end(), final_outputs.begin(), [](auto a) { return a.second; });

  return final_outputs;
}
