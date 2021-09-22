#include "RecoEgamma/ElectronIdentification/interface/ElectronDNNEstimator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <fstream>
#include <memory>

ElectronDNNEstimator::ElectronDNNEstimator() : cfg_{} {}

ElectronDNNEstimator::ElectronDNNEstimator(std::vector<std::string>& models_files,
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
  LogDebug("EleDNNPFid") << "Ele PFID DNN evaluation with " << nModels_ << " models and " << nInputs_[0]
                         << " variables --> LOADED";
}

ElectronDNNEstimator::ElectronDNNEstimator(const Configuration& cfg) : cfg_(cfg) {
  // Init tensorflow sessions
  nModels_ = cfg_.models_files.size();
  debug_ = cfg_.log_level == "0";
  initTensorFlowGraphs();
  initScalerFiles();
  LogDebug("EleDNNPFid") << "Ele PFID DNN evaluation with " << nModels_ << " models and " << nInputs_[0]
                         << " variables --> LOADED";
}

void ElectronDNNEstimator::initTensorFlowGraphs() {
  // configure logging to show warnings (see table below)
  tensorflow::setLogging(cfg_.log_level);
  // load the graph definition
  LogDebug("EleDNNPFid") << "Loading " << nModels_ << " graphs";
  for (const auto& model_file : cfg_.models_files) {
    tensorflow::GraphDef* graphDef = tensorflow::loadGraphDef(model_file);  //-->should be atomic but does not compile
    graphDefs_.push_back(graphDef);
  }
  LogDebug("EleDNNPFid") << "Graphs loaded";
}

void ElectronDNNEstimator::initScalerFiles() {
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
        else
          type = 0;
        features.push_back(std::make_tuple(varname, type, par1, par2));
        // Protection for mismatch between requested variables and the available ones
        auto match = std::find(
            ElectronDNNEstimator::dnnAvaibleInputs.begin(), ElectronDNNEstimator::dnnAvaibleInputs.end(), varname);
        if (match == std::end(ElectronDNNEstimator::dnnAvaibleInputs)) {
          throw cms::Exception("MissingVariable")
              << "Requested variable (" << varname << ") not available between Electron PFid DNN inputs";
        }
        ninputs += 1;
      }
    }
    inputfile_scaler.close();
    featuresMap_.push_back(features);
    nInputs_.push_back(ninputs);
  }
}

std::vector<tensorflow::Session*> ElectronDNNEstimator::getSessions() const {
  LogDebug("EleDNNPFid") << "Starting " << nModels_ << "TF sessions";
  std::vector<tensorflow::Session*> sessions;
  for (auto& graphDef : graphDefs_) {
    sessions.push_back(tensorflow::createSession(graphDef));
  }
  LogDebug("EleDNNPFid") << "TF sessions started";
  return sessions;
}

const std::array<std::string, ElectronDNNEstimator::nAvailableVars> ElectronDNNEstimator::dnnAvaibleInputs = {
    {"fbrem",
     "abs(deltaEtaSuperClusterTrackAtVtx)",
     "abs(deltaPhiSuperClusterTrackAtVtx)",
     "full5x5_sigmaIetaIeta",
     "full5x5_hcalOverEcal",
     "eSuperClusterOverP",
     "full5x5_e1x5",
     "eEleClusterOverPout",
     "closestCtfTrackNormChi2",
     "closestCtfTrackNLayers",
     "gsfTrack.missing_inner_hits",
     "dr03TkSumPt",
     "dr03EcalRecHitSumEt",
     "dr03HcalTowerSumEt",
     "gsfTrack.normalizedChi2",
     "superCluster.eta",
     "pt",
     "ecalPFClusterIso",
     "hcalPFClusterIso",
     "numberOfBrems",
     "abs(deltaEtaSeedClusterTrackAtCalo)",
     "hadronicOverEm",
     "full5x5_e2x5Max",
     "full5x5_e5x5"}};

std::map<std::string, float> ElectronDNNEstimator::getInputsVars(const reco::GsfElectron& ele) const {
  // Prepare a map with all the defined variables
  std::map<std::string, float> variables;
  reco::TrackRef myTrackRef = ele.closestCtfTrackRef();
  bool validKF = (myTrackRef.isNonnull() && myTrackRef.isAvailable());
  variables["fbrem"] = ele.fbrem();
  variables["abs(deltaEtaSuperClusterTrackAtVtx)"] = std::abs(ele.deltaEtaSuperClusterTrackAtVtx());
  variables["abs(deltaPhiSuperClusterTrackAtVtx)"] = std::abs(ele.deltaPhiSuperClusterTrackAtVtx());
  variables["full5x5_sigmaIetaIeta"] = ele.full5x5_sigmaIetaIeta();
  variables["full5x5_hcalOverEcal"] = ele.full5x5_hcalOverEcalValid() ? ele.full5x5_hcalOverEcal() : 0;
  variables["eSuperClusterOverP"] = ele.eSuperClusterOverP();
  variables["full5x5_e1x5"] = ele.full5x5_e1x5();
  variables["eEleClusterOverPout"] = ele.eEleClusterOverPout();
  variables["closestCtfTrackNormChi2"] = ele.closestCtfTrackNormChi2();
  variables["closestCtfTrackNLayers"] = ele.closestCtfTrackNLayers();
  variables["gsfTrack.missing_inner_hits"] =
      (validKF) ? myTrackRef->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS) : -1.;
  variables["dr03TkSumPt"] = ele.dr03TkSumPt();
  variables["dr03EcalRecHitSumEt"] = ele.dr03EcalRecHitSumEt();
  variables["dr03HcalTowerSumEt"] = ele.dr03HcalTowerSumEt();
  variables["gsfTrack.normalizedChi2"] = (validKF) ? myTrackRef->normalizedChi2() : 0;
  variables["superCluster.eta"] = ele.superCluster()->eta();
  variables["pt"] = ele.pt();
  variables["ecalPFClusterIso"] = ele.ecalPFClusterIso();
  variables["hcalPFClusterIso"] = ele.hcalPFClusterIso();
  variables["numberOfBrems"] = ele.numberOfBrems();
  variables["abs(deltaEtaSeedClusterTrackAtCalo)"] = std::abs(ele.deltaEtaSeedClusterTrackAtCalo());
  variables["hadronicOverEm"] = ele.hcalOverEcalValid() ? ele.hadronicOverEm() : 0;
  variables["full5x5_e2x5Max"] = ele.full5x5_e2x5Max();
  variables["full5x5_e5x5"] = ele.full5x5_e5x5();
  // Define more variables here and use them directly in the model config!
  return variables;
}

uint ElectronDNNEstimator::getModelIndex(const reco::GsfElectron& ele) const {
  /* 
  Selection of the model to be applied on the electron based on pt/eta cuts or whatever selection
  */
  uint modelIndex;
  if (ele.pt() < 10)
    modelIndex = 0;
  if (ele.pt() >= 10) {
    if (std::abs(ele.eta()) <= 1.466) {
      modelIndex = 1;
    } else {
      modelIndex = 2;
    }
  }
  return modelIndex;
}

std::pair<uint, std::vector<float>> ElectronDNNEstimator::getScaledInputs(const reco::GsfElectron& ele) const {
  uint modelIndex = getModelIndex(ele);
  auto allInputs = getInputsVars(ele);
  std::vector<float> inputs;
  // Loop on the list of requested variables and scaling values for the specific modelIndex
  // Different type of scaling are available: 0=no scaling, 1=standard scaler, 2=minmax
  for (auto& [varName, type, par1, par2] : featuresMap_[modelIndex]) {
    if (type == 1)  // Standard scaling
      inputs.push_back((allInputs[varName] - par1) / par2);
    else if (type == 2)  // MinMax
      inputs.push_back((allInputs[varName] - par1) / (par2 - par1));
    else {
      inputs.push_back(allInputs[varName]);  // Do nothing on the variable
    }
    //Protection for mismatch between requested variables and the available ones
    // have been added when the scaler config are loaded --> here we know that the variables are available
  }
  return std::make_pair(modelIndex, inputs);
}

std::vector<std::array<float, ElectronDNNEstimator::nOutputs>> ElectronDNNEstimator::evaluate(
    const reco::GsfElectronCollection& electrons, const std::vector<tensorflow::Session*> sessions) const {
  /*
      Evaluate the Electron PFID DNN for all the electrons. 
      3 models are defined depending on the pt and eta --> we need to build 3 input tensors to evaluate
      the DNNs with batching.
      
      1) Get the inputs vector, already scaled correctly for each electron and the modelIndex
      2) Prepare 3 input tensors for the 3 models
      3) Run the model and get the output for each electron
      4) Sort the output by electron index
      5) Return the DNN output 

    */
  std::vector<std::vector<int>> eleIndexMap(nModels_);  // for each model; the list of ele index is saved
  std::vector<std::vector<float>> inputsVectors;
  std::vector<uint> counts (nModels_);

  LogDebug("EleDNNPFid") << "Working on " << electrons.size() << " electrons";

  int iele = -1;
  for (auto& ele : electrons) {
    iele++;
    LogDebug("EleDNNPFid") << "Working on ele: " << iele;
    auto [model_index, inputs] = getScaledInputs(ele);
    counts[model_index] += 1;
    eleIndexMap[model_index].push_back(iele);
    inputsVectors.push_back(inputs);
  }

  // Prepare one input tensors for each model
  std::vector<tensorflow::Tensor> input_tensors(nModels_);
  // Pointers for filling efficiently the input tensors
  std::vector<float*> input_tensors_pointer(nModels_);
  for (uint i = 0; i < nModels_; i++) {
    LogDebug("EleDNNPFid") << "Initializing TF input " << i << " with rows:" << counts[i]
                           << " and cols:" << nInputs_[i];
    input_tensors[i] = tensorflow::Tensor{tensorflow::DT_FLOAT, {counts[i], nInputs_[i]}};
    input_tensors_pointer[i] = input_tensors[i].flat<float>().data();
  }

  // Filling the input tensors
  for (uint m = 0; m < nModels_; m++) {
    LogDebug("EleDNNPFid") << "Loading TF input tensor for model: " << m;
    float* T = input_tensors_pointer[m];
    for (int ele_index : eleIndexMap[m]) {
      for (int k = 0; k < nInputs_[m]; k++, T++) {  //Note the input tensor pointer incremented
        *T = inputsVectors[ele_index][k];
      }
    }
  }

  // Define the output and run
  std::vector<std::pair<int, std::array<float, ElectronDNNEstimator::nOutputs>>> outputs;
  // Run all the models
  for (uint m = 0; m < nModels_; m++) {
    if (counts[m] == 0)
      continue;  //Skip model witout inputs
    std::vector<tensorflow::Tensor> output;
    LogDebug("EleDNNPFid") << "Run model: " << m << " with " << counts[m] << " electrons";
    tensorflow::run(sessions[m], {{cfg_.inputTensorName, input_tensors[m]}}, {cfg_.outputTensorName}, &output);
    // Get the output and save the ElectronDNNEstimator::nOutputs numbers along with the ele index
    auto r = output[0].tensor<float, 2>();
    // Iterate on the list of elements in the batch --> many electrons
    for (uint b = 0; b < counts[m]; b++) {
      std::array<float, ElectronDNNEstimator::nOutputs> result;
      for (uint k = 0; k < ElectronDNNEstimator::nOutputs; k++)
        result[k] = r(b, k);
      // Get the original index of the electorn in the original order
      int ele_index = eleIndexMap[m][b];
      LogDebug("EleDNNPFid") << "DNN output, model " << m << " ele " << ele_index << " : " << result[0] << " "
                             << result[1] << " " << result[2] << " " << result[3] << " " << result[4];
      outputs.push_back(std::make_pair(ele_index, result));
    }
  }

  // Now we have just to re-order the outputs
  std::sort(outputs.begin(), outputs.end());
  std::vector<std::array<float, ElectronDNNEstimator::nOutputs>> final_outputs(outputs.size());
  std::transform(outputs.begin(), outputs.end(), final_outputs.begin(), [](auto a) { return a.second; });

  return final_outputs;
}
