#include "RecoEcal/EgammaCoreTools/interface/DeepSCGraphEvaluation.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "TMath.h"
#include <iostream>
#include <fstream>
using namespace reco;

const std::vector<std::string> DeepSCGraphEvaluation::availableClusterInputs = {"cl_energy",
                                                                                "cl_et",
                                                                                "cl_eta",
                                                                                "cl_phi",
                                                                                "cl_ieta",
                                                                                "cl_iphi",
                                                                                "cl_iz",
                                                                                "cl_seed_dEta",
                                                                                "cl_seed_dPhi",
                                                                                "cl_seed_dEnergy",
                                                                                "cl_seed_dEt",
                                                                                "cl_etaWidth",
                                                                                "cl_phiWidth",
                                                                                "cl_nxtals",
                                                                                "cl_swissCross",
                                                                                "cl_r9",
                                                                                "cl_sigmaIetaIeta",
                                                                                "cl_sigmaIetaIphi",
                                                                                "cl_sigmaIphiIphi",
                                                                                "cl_f5_swissCross",
                                                                                "cl_f5_r9",
                                                                                "cl_f5_sigmaIetaIeta",
                                                                                "cl_f5_sigmaIetaIphi",
                                                                                "cl_f5_sigmaIphiIphi"};
const std::vector<std::string> DeepSCGraphEvaluation::availableWindowInputs = {
    "max_cl_energy",
    "max_cl_et",
    "max_cl_eta",
    "max_cl_phi",
    "max_cl_ieta",
    "max_cl_iphi",
    "max_cl_iz",
    "max_cl_seed_dEta",
    "max_cl_seed_dPhi",
    "max_cl_seed_dEnergy",
    "max_cl_seed_dEt",
    "max_cl_nxtals",
    "max_cl_etaWidth",
    "max_cl_phiWidth",
    "max_cl_swissCross",
    "max_cl_r9",
    "max_cl_sigmaIetaIeta",
    "max_cl_sigmaIetaIphi",
    "max_cl_sigmaIphiIphi",
    "max_cl_f5_swissCross",
    "max_cl_f5_r9",
    "max_cl_f5_sigmaIetaIeta",
    "max_cl_f5_sigmaIetaIphi",
    "max_cl_f5_sigmaIphiIphi",
    "min_cl_energy",
    "min_cl_et",
    "min_cl_eta",
    "min_cl_phi",
    "min_cl_ieta",
    "min_cl_iphi",
    "min_cl_iz",
    "min_cl_seed_dEta",
    "min_cl_seed_dPhi",
    "min_cl_seed_dEnergy",
    "min_cl_seed_dEt",
    "min_cl_nxtals",
    "min_cl_etaWidth",
    "min_cl_phiWidth",
    "min_cl_swissCross",
    "min_cl_r9",
    "min_cl_sigmaIetaIeta",
    "min_cl_sigmaIetaIphi",
    "min_cl_sigmaIphiIphi",
    "min_cl_f5_swissCross",
    "min_cl_f5_r9",
    "min_cl_f5_sigmaIetaIeta",
    "min_cl_f5_sigmaIetaIphi",
    "min_cl_f5_sigmaIphiIphi",
    "avg_cl_energy",
    "avg_cl_et",
    "avg_cl_eta",
    "avg_cl_phi",
    "avg_cl_ieta",
    "avg_cl_iphi",
    "avg_cl_iz",
    "avg_cl_seed_dEta",
    "avg_cl_seed_dPhi",
    "avg_cl_seed_dEnergy",
    "avg_cl_seed_dEt",
    "avg_cl_nxtals",
    "avg_cl_etaWidth",
    "avg_cl_phiWidth",
    "avg_cl_swissCross",
    "avg_cl_r9",
    "avg_cl_sigmaIetaIeta",
    "avg_cl_sigmaIetaIphi",
    "avg_cl_sigmaIphiIphi",
    "avg_cl_f5_swissCross",
    "avg_cl_f5_r9",
    "avg_cl_f5_sigmaIetaIeta",
    "avg_cl_f5_sigmaIetaIphi",
    "avg_cl_f5_sigmaIphiIphi",
};
const std::vector<std::string> DeepSCGraphEvaluation::availableHitsInputs = {"ieta", "iphi", "iz", "en_withfrac"};

DeepSCGraphEvaluation::DeepSCGraphEvaluation(const DeepSCConfiguration& cfg) : cfg_(cfg) {
  tensorflow::setLogging("0");
  // Init TF graph and session objects
  initTensorFlowGraphAndSession();
  // Init scaler configs
  inputFeaturesClusters =
      readInputFeaturesConfig(cfg_.configFileClusterFeatures, DeepSCGraphEvaluation::availableClusterInputs);
  if (inputFeaturesClusters.size() != cfg_.nClusterFeatures) {
    throw cms::Exception("WrongConfiguration") << "Mismatch between number of input features for Clusters and "
                                               << "parameters in the scaler file.";
  }
  inputFeaturesWindows =
      readInputFeaturesConfig(cfg_.configFileWindowFeatures, DeepSCGraphEvaluation::availableWindowInputs);
  if (inputFeaturesWindows.size() != cfg_.nWindowFeatures) {
    throw cms::Exception("WrongConfiguration") << "Mismatch between number of input features for Clusters and "
                                               << "parameters in the scaler file.";
  }
  inputFeaturesHits = readInputFeaturesConfig(cfg_.configFileHitsFeatures, DeepSCGraphEvaluation::availableHitsInputs);
  if (inputFeaturesHits.size() != cfg_.nHitsFeatures) {
    throw cms::Exception("WrongConfiguration") << "Mismatch between number of input features for Clusters and "
                                               << "parameters in the scaler file.";
  }
}

DeepSCGraphEvaluation::~DeepSCGraphEvaluation() {
  for (auto& [i, s] : sessions_) {
    if (s != nullptr)
      LogDebug("DeepSCGraphEvaluation") << "Closing sessions: " << i;
    tensorflow::closeSession(s);
  }
}

void DeepSCGraphEvaluation::initTensorFlowGraphAndSession() {
  // load the graph definition
  uint imodel = 0;
  for (const auto& modelFile : cfg_.modelFiles) {
    LogDebug("DeepSCGraphEvaluation") << "Loading graph: " << modelFile << " and starting the TF session";
    sessions_[imodel] = tensorflow::createSession(tensorflow::loadGraphDef(edm::FileInPath(modelFile).fullPath()));
    imodel++;
  }
  LogDebug("DeepSCGraphEvaluation") << "TF ready";
}

DeepSCInputs::InputConfigs DeepSCGraphEvaluation::readInputFeaturesConfig(
    std::string file, const std::vector<std::string>& availableInputs) const {
  DeepSCInputs::InputConfigs features;
  LogDebug("DeepSCGraphEvaluation") << "Reading scaler file: " << edm::FileInPath(file).fullPath();
  std::ifstream inputfile{edm::FileInPath(file).fullPath()};
  if (inputfile.fail()) {
    throw cms::Exception("MissingFile") << "Input features config file not found: " << file;
  } else {
    // Now read mean, scale factors for each variable
    float par1, par2;
    std::string varName, type_str;
    DeepSCInputs::ScalerType type;
    while (inputfile >> varName >> type_str >> par1 >> par2) {
      if (type_str == "MeanRms")
        type = DeepSCInputs::ScalerType::MeanRms;
      else if (type_str == "MinMax")
        type = DeepSCInputs::ScalerType::MinMax;
      else
        type = DeepSCInputs::ScalerType::None;  //do nothing
      features.push_back(DeepSCInputs::InputConfig{.varName = varName, .type = type, .par1 = par1, .par2 = par2});
      // Protection for mismatch between requested variables and the available ones
      auto match = std::find(availableInputs.begin(), availableInputs.end(), varName);
      if (match == std::end(availableInputs)) {
        throw cms::Exception("MissingInput") << "Requested input (" << varName << ") not available between DNN inputs";
      }
      LogDebug("DeepSCGraphEvalutation") << "Registered input feature: " << varName << ", scaler=" << type_str;
    }
  }
  return features;
}

std::vector<float> DeepSCGraphEvaluation::getScaledInputs(const DeepSCInputs::FeaturesMap& variables,
                                                          const DeepSCInputs::InputConfigs& config) const {
  std::vector<float> inputs;
  inputs.reserve(config.size());
  // Loop on the list of requested variables and scaling values
  // Different type of scaling are available: 0=no scaling, 1=standard scaler, 2=minmax
  for (auto& [varName, type, par1, par2] : config) {
    if (type == DeepSCInputs::ScalerType::MeanRms)
      inputs.push_back((variables.at(varName) - par1) / par2);
    else if (type == DeepSCInputs::ScalerType::MinMax)
      inputs.push_back((variables.at(varName) - par1) / (par2 - par1));
    else if (type == DeepSCInputs::ScalerType::None) {
      inputs.push_back(variables.at(varName));  // Do nothing on the variable
    }
    //Protection for mismatch between requested variables and the available ones
    // have been added when the scaler config are loaded --> here we know that the variables are available
  }
  return inputs;
}

std::vector<std::vector<float>> DeepSCGraphEvaluation::evaluate(const DeepSCInputs::Inputs& inputs) const {
  LogDebug("DeepSCGraphEvaluation") << "Starting evaluation";

  // We need to split the total inputs in N batches of size batchSize (configured in the producer)
  // being careful with the last batch which will have less than batchSize elements
  size_t nInputs = inputs.clustersX.size();
  // Final output
  std::vector<std::vector<float>> outputs_clustering(nInputs);

  // Preparing the two sets of inputs for the small and large model
  std::map<uint, std::vector<size_t>> windowModelMapping = {{0, {}}, {1, {}}};  // Map between window -> Model
  // Quickly checking the dimension of each window to assig it to the correct model
  LogDebug("DeepSCGraphEvaluation") << "Assigning windows to models. Max Ncls " << cfg_.maxNClusters[0]
                                    << " max Nrechit:" << cfg_.maxNRechits[0];
  for (size_t i = 0; i < nInputs; i++) {
    if ((inputs.clustersX[i].size() > cfg_.maxNClusters[0]) || (inputs.maxNRechits[i] > cfg_.maxNRechits[0])) {
      LogDebug("DeepSCGraphEvaluation") << "wind: " << i << " to model: 1. Nrechits:  " << inputs.maxNRechits[i]
                                        << " nclusters: " << inputs.clustersX[i].size();
      windowModelMapping[1].push_back(i);
    } else {
      LogDebug("DeepSCGraphEvaluation") << "wind: " << i << " to model: 0. Nrechits:  " << inputs.maxNRechits[i]
                                        << " nclusters: " << inputs.clustersX[i].size();
      windowModelMapping[0].push_back(i);
    }
  }

  auto nsamples0 = static_cast<long int>(windowModelMapping[0].size());
  std::map<uint, std::map<std::string, tensorflow::Tensor>> tfInputs = {
      {0,
       {{"clsX", {tensorflow::DT_FLOAT, {nsamples0, cfg_.maxNClusters[0], cfg_.nClusterFeatures}}},
        {"windX", {tensorflow::DT_FLOAT, {nsamples0, cfg_.nWindowFeatures}}},
        {"hitsX", {tensorflow::DT_FLOAT, {nsamples0, cfg_.maxNClusters[0], cfg_.maxNRechits[0], cfg_.nHitsFeatures}}},
        {"isSeedX", {tensorflow::DT_FLOAT, {nsamples0, cfg_.maxNClusters[0]}}},
        {"maskCls", {tensorflow::DT_FLOAT, {nsamples0, cfg_.maxNClusters[0]}}},
        {"maskRechits", {tensorflow::DT_FLOAT, {nsamples0, cfg_.maxNClusters[0], cfg_.maxNRechits[0]}}}}}};

  auto nsamples1 = static_cast<long int>(windowModelMapping[1].size());
  if (nsamples1 > 0) {
    tfInputs[1] = {
        {"clsX", {tensorflow::DT_FLOAT, {nsamples1, cfg_.maxNClusters[1], cfg_.nClusterFeatures}}},
        {"windX", {tensorflow::DT_FLOAT, {nsamples1, cfg_.nWindowFeatures}}},
        {"hitsX", {tensorflow::DT_FLOAT, {nsamples1, cfg_.maxNClusters[1], cfg_.maxNRechits[1], cfg_.nHitsFeatures}}},
        {"isSeedX", {tensorflow::DT_FLOAT, {nsamples1, cfg_.maxNClusters[1]}}},
        {"maskCls", {tensorflow::DT_FLOAT, {nsamples1, cfg_.maxNClusters[1]}}},
        {"maskRechits", {tensorflow::DT_FLOAT, {nsamples1, cfg_.maxNClusters[1], cfg_.maxNRechits[1]}}}};
  }
  // The input tensors are now ready to be filled
  // Now we can loop on all the elements and fill the correct tensor
  for (const auto& [modelIdx, windIdx] : windowModelMapping) {
    auto tf = tfInputs[modelIdx];
    for (size_t b = 0; b < windIdx.size(); b++) {
      auto iwindow = windIdx[b];
      LogTrace("DeepSC") << "Filling window " << iwindow << " row: " << b << " model: " << modelIdx;

      auto tf_clsX = tf["clsX"];
      auto tf_maskCls = tf["maskCls"];

      const auto& cls_data = inputs.clustersX[iwindow];
      // Loop on clusters
      for (size_t k = 0; k < cfg_.maxNClusters[modelIdx]; k++) {
        // Loop on features
        for (size_t z = 0; z < cfg_.nClusterFeatures; z++) {
          if (k < cls_data.size()) {
            tf_clsX.tensor<float, 3>()(b, k, z) = float(cls_data[k][z]);
            // Clusters mask
            tf_maskCls.matrix<float>()(b, k) = 1.0;
          } else {
            tf_clsX.tensor<float, 3>()(b, k, z) = 0.;
            tf_maskCls.matrix<float>()(b, k) = 0.;
          }
        }
      }

      auto tf_windX = tf["windX"];
      const auto& wind_features = inputs.windowX[iwindow];
      // Loop on features
      for (size_t k = 0; k < cfg_.nWindowFeatures; k++) {
        tf_windX.matrix<float>()(b, k) = float(wind_features[k]);
      }

      auto tf_hitsX = tf["hitsX"];
      auto tf_maskRechits = tf["maskRechits"];
      const auto& hits_data = inputs.hitsX[iwindow];
      size_t ncls_in_window = hits_data.size();
      // Loop on clusters
      for (size_t k = 0; k < cfg_.maxNClusters[modelIdx]; k++) {
        // Check padding
        size_t nhits_in_cluster;
        if (k < ncls_in_window)
          nhits_in_cluster = hits_data[k].size();
        else
          nhits_in_cluster = 0;
        // Loop on hits
        for (size_t j = 0; j < cfg_.maxNRechits[modelIdx]; j++) {
          // Check the number of clusters and hits for padding
          bool ok = j < nhits_in_cluster;
          // Rechits masc
          if (ok) {
            tf_maskRechits.tensor<float, 3>()(b, k, j) = 1.;
          } else {
            tf_maskRechits.tensor<float, 3>()(b, k, j) = 0.;
          }
          // Loop on rechits features
          for (size_t z = 0; z < cfg_.nHitsFeatures; z++) {
            if (ok) {
              tf_hitsX.tensor<float, 4>()(b, k, j, z) = float(hits_data[k][j][z]);
            } else {
              tf_hitsX.tensor<float, 4>()(b, k, j, z) = 0.;
            }
          }
        }
      }

      auto tf_isSeedX = tf["isSeedX"];
      const auto& isSeed_data = inputs.isSeed[iwindow];
      // Loop on clusters
      for (size_t k = 0; k < cfg_.maxNClusters[modelIdx]; k++) {
        if (k < isSeed_data.size()) {
          tf_isSeedX.matrix<float>()(b, k) = float(isSeed_data[k]);
        } else {
          tf_isSeedX.matrix<float>()(b, k) = 0.;
        }
      }
    }  //---> loop on rows for each model
  }    // --> loop on models

  // prepare tensorflow outputs
  for (auto& [modelIdx, tfInputs] : tfInputs) {
    // Skip the evaluation if we don't have rows for this model
    if (windowModelMapping[modelIdx].empty())
      continue;
    // Run the models
    LogDebug("DeepSCGraphEvaluation") << "Run model: " << modelIdx;

    std::vector<std::pair<std::string, tensorflow::Tensor>> feed_dict = {{"input_1", tfInputs["clsX"]},
                                                                         {"input_2", tfInputs["windX"]},
                                                                         {"input_3", tfInputs["hitsX"]},
                                                                         {"input_4", tfInputs["isSeedX"]},
                                                                         {"input_5", tfInputs["maskCls"]},
                                                                         {"input_6", tfInputs["maskRechits"]}};

    // Define the output and run
    std::vector<tensorflow::Tensor> outputs_tf;
    tensorflow::run(sessions_.at(modelIdx), feed_dict, {"cl_class", "wind_class"}, &outputs_tf);
    // Reading the 1st output: clustering probability
    const auto& y_cl = outputs_tf[0].tensor<float, 3>();
    // Iterate on the clusters for each window
    for (size_t b = 0; b < windowModelMapping[modelIdx].size(); b++) {
      auto originalWindIdx = windowModelMapping[modelIdx][b];
      auto ncls = inputs.clustersX[originalWindIdx].size();
      std::vector<float> cl_output(ncls);
      for (size_t iC = 0; iC < ncls; iC++) {
        if (iC < cfg_.maxNClusters[modelIdx]) {
          float y = y_cl(b, iC, 0);
          // Applying sigmoid to logit
          cl_output[iC] = 1 / (1 + std::exp(-y));
        } else {
          // The number of clusters is over the padding max dim
          cl_output[iC] = 0;
        }
      }
      // Assign the output to the correct window in the initial ordering
      // using the model mapping
      outputs_clustering[originalWindIdx] = cl_output;
    }
  }
  return outputs_clustering;
}
