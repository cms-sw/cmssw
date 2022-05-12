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
                                                                                "cl_nxtals"};
const std::vector<std::string> DeepSCGraphEvaluation::availableWindowInputs = {
    "max_cl_energy", "max_cl_et",        "max_cl_eta",       "max_cl_phi",          "max_cl_ieta",     "max_cl_iphi",
    "max_cl_iz",     "max_cl_seed_dEta", "max_cl_seed_dPhi", "max_cl_seed_dEnergy", "max_cl_seed_dEt", "max_cl_nxtals",
    "min_cl_energy", "min_cl_et",        "min_cl_eta",       "min_cl_phi",          "min_cl_ieta",     "min_cl_iphi",
    "min_cl_iz",     "min_cl_seed_dEta", "min_cl_seed_dPhi", "min_cl_seed_dEnergy", "min_cl_seed_dEt", "min_cl_nxtals",
    "avg_cl_energy", "avg_cl_et",        "avg_cl_eta",       "avg_cl_phi",          "avg_cl_ieta",     "avg_cl_iphi",
    "avg_cl_iz",     "avg_cl_seed_dEta", "avg_cl_seed_dPhi", "avg_cl_seed_dEnergy", "avg_cl_seed_dEt", "avg_cl_nxtals"};
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
  if (session_ != nullptr)
    tensorflow::closeSession(session_);
}

void DeepSCGraphEvaluation::initTensorFlowGraphAndSession() {
  // load the graph definition
  LogDebug("DeepSCGraphEvaluation") << "Loading graph";
  graphDef_ =
      std::unique_ptr<tensorflow::GraphDef>(tensorflow::loadGraphDef(edm::FileInPath(cfg_.modelFile).fullPath()));
  LogDebug("DeepSCGraphEvaluation") << "Starting TF sessions";
  session_ = tensorflow::createSession(graphDef_.get());
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

  // Final output
  std::vector<std::vector<float>> outputs_clustering;

  // We need to split the total inputs in N batches of size batchSize (configured in the producer)
  // being careful with the last batch which will have less than batchSize elements
  size_t nInputs = inputs.clustersX.size();
  uint iB = -1;  // batch index
  while (nInputs > 0) {
    iB++;  // go to next batch
    size_t nItems;
    if (nInputs >= cfg_.batchSize) {
      nItems = cfg_.batchSize;
      nInputs -= cfg_.batchSize;
    } else {
      nItems = nInputs;
      nInputs = 0;
    }

    // Inputs
    tensorflow::Tensor clsX_{tensorflow::DT_FLOAT,
                             {static_cast<long int>(nItems), cfg_.maxNClusters, cfg_.nClusterFeatures}};
    tensorflow::Tensor windX_{tensorflow::DT_FLOAT, {static_cast<long int>(nItems), cfg_.nWindowFeatures}};
    tensorflow::Tensor hitsX_{tensorflow::DT_FLOAT,
                              {static_cast<long int>(nItems), cfg_.maxNClusters, cfg_.maxNRechits, cfg_.nHitsFeatures}};
    tensorflow::Tensor isSeedX_{tensorflow::DT_FLOAT, {static_cast<long int>(nItems), cfg_.maxNClusters, 1}};
    tensorflow::Tensor nClsSize_{tensorflow::DT_FLOAT, {static_cast<long int>(nItems)}};

    // Look on batch dim
    for (size_t b = 0; b < nItems; b++) {
      const auto& cls_data = inputs.clustersX[iB * cfg_.batchSize + b];
      // Loop on clusters
      for (size_t k = 0; k < cfg_.maxNClusters; k++) {
        // Loop on features
        for (size_t z = 0; z < cfg_.nClusterFeatures; z++) {
          if (k < cls_data.size()) {
            clsX_.tensor<float, 3>()(b, k, z) = float(cls_data[k][z]);
          } else {
            clsX_.tensor<float, 3>()(b, k, z) = 0.;
          }
        }
      }
    }

    // Look on batch dim
    for (size_t b = 0; b < nItems; b++) {
      const auto& wind_features = inputs.windowX[iB * cfg_.batchSize + b];
      // Loop on features
      for (size_t k = 0; k < cfg_.nWindowFeatures; k++) {
        windX_.matrix<float>()(b, k) = float(wind_features[k]);
      }
    }

    // Look on batch dim
    for (size_t b = 0; b < nItems; b++) {
      const auto& hits_data = inputs.hitsX[iB * cfg_.batchSize + b];
      size_t ncls_in_window = hits_data.size();
      // Loop on clusters
      for (size_t k = 0; k < cfg_.maxNClusters; k++) {
        // Check padding
        size_t nhits_in_cluster;
        if (k < ncls_in_window)
          nhits_in_cluster = hits_data[k].size();
        else
          nhits_in_cluster = 0;

        // Loop on hits
        for (size_t j = 0; j < cfg_.maxNRechits; j++) {
          // Check the number of clusters and hits for padding
          bool ok = j < nhits_in_cluster;
          // Loop on rechits features
          for (size_t z = 0; z < cfg_.nHitsFeatures; z++) {
            if (ok)
              hitsX_.tensor<float, 4>()(b, k, j, z) = float(hits_data[k][j][z]);
            else
              hitsX_.tensor<float, 4>()(b, k, j, z) = 0.;
          }
        }
      }
    }

    // Look on batch dim
    for (size_t b = 0; b < nItems; b++) {
      const auto& isSeed_data = inputs.isSeed[iB * cfg_.batchSize + b];
      // Loop on clusters
      for (size_t k = 0; k < cfg_.maxNClusters; k++) {
        if (k < isSeed_data.size()) {
          isSeedX_.tensor<float, 3>()(b, k, 0) = float(isSeed_data[k]);
        } else {
          isSeedX_.tensor<float, 3>()(b, k, 0) = 0.;
        }
      }
    }

    for (size_t b = 0; b < nItems; b++) {
      nClsSize_.vec<float>()(b) = float(inputs.clustersX[iB * cfg_.batchSize + b].size());
    }

    std::vector<std::pair<std::string, tensorflow::Tensor>> feed_dict = {
        {"input_1", clsX_}, {"input_2", windX_}, {"input_3", hitsX_}, {"input_4", isSeedX_}, {"input_5", nClsSize_}};

    // prepare tensorflow outputs
    std::vector<tensorflow::Tensor> outputs_tf;
    // // Define the output and run
    // // Run the models
    LogDebug("DeepSCGraphEvaluation") << "Run model";
    tensorflow::run(session_, feed_dict, {"cl_class", "wind_class"}, &outputs_tf);
    // Reading the 1st output: clustering probability
    const auto& y_cl = outputs_tf[0].tensor<float, 3>();
    // Iterate on the clusters for each window
    for (size_t b = 0; b < nItems; b++) {
      uint ncls = inputs.clustersX[iB * cfg_.batchSize + b].size();
      std::vector<float> cl_output(ncls);
      for (size_t iC = 0; iC < ncls; iC++) {
        if (iC < cfg_.maxNClusters) {
          float y = y_cl(b, iC, 0);
          // Applying sigmoid to logit
          cl_output[iC] = 1 / (1 + std::exp(-y));
        } else {
          // The number of clusters is over the padding max dim
          cl_output[iC] = 0;
        }
      }
      outputs_clustering.push_back(cl_output);
    }
  }

  return outputs_clustering;
}
