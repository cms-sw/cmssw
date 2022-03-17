#include "RecoEcal/EgammaCoreTools/interface/DeepSCGraphEvaluation.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "TMath.h"
#include <iostream>
#include <fstream>
using namespace reco;

DeepSCGraphEvaluation::DeepSCGraphEvaluation(const DeepSCConfiguration& cfg) : cfg_(cfg) {
  tensorflow::setLogging("0");
  // Init TF graph and session objects
  initTensorFlowGraphAndSession();
  // Init scaler configs
  uint nClFeat = readScalerConfig(cfg_.scalerFileClusterFeatures, scalerParamsClusters_);
  if (nClFeat != cfg_.nClusterFeatures) {
    throw cms::Exception("WrongConfiguration") << "Mismatch between number of input features for Clusters and "
                                               << "parameters in the scaler file.";
  }
  uint nClWind = readScalerConfig(cfg_.scalerFileWindowFeatures, scalerParamsWindows_);
  if (nClWind != cfg_.nWindowFeatures) {
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

uint DeepSCGraphEvaluation::readScalerConfig(std::string file, std::vector<std::pair<float, float>>& scalingParams) {
  LogDebug("DeepSCGraphEvaluation") << "Reading scaler file: " << edm::FileInPath(file).fullPath();
  std::ifstream inputfile{edm::FileInPath(file).fullPath()};
  int ninputs = 0;
  if (inputfile.fail()) {
    throw cms::Exception("MissingFile") << "Scaler file not found: " << file;
  } else {
    // Now read mean, scale factors for each variable
    float par1, par2;
    while (inputfile >> par1 >> par2) {
      scalingParams.push_back(std::make_pair(par1, par2));
      ninputs += 1;
    }
  }
  return ninputs;
}

std::vector<double> DeepSCGraphEvaluation::scaleClusterFeatures(const std::vector<double>& input) const {
  std::vector<double> out(input.size());
  for (size_t i = 0; i < input.size(); i++) {
    const auto& [par1, par2] = scalerParamsClusters_[i];
    out[i] = (input[i] - par1) / par2;
  }
  return out;
}

std::vector<double> DeepSCGraphEvaluation::scaleWindowFeatures(const std::vector<double>& input) const {
  std::vector<double> out(input.size());
  for (size_t i = 0; i < input.size(); i++) {
    const auto& [par1, par2] = scalerParamsWindows_[i];
    out[i] = (input[i] - par1) / par2;
  }
  return out;
}

std::vector<std::vector<float>> DeepSCGraphEvaluation::evaluate(const DeepSCInputs& inputs) const {
  /*
   Evaluate the DeepSC model
  */
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
    // Input tensors initialization

    // Inputs
    tensorflow::Tensor clsX_{tensorflow::DT_FLOAT,
                             {static_cast<long int>(nItems), cfg_.maxNClusters, cfg_.nClusterFeatures}};
    tensorflow::Tensor windX_{tensorflow::DT_FLOAT, {static_cast<long int>(nItems), cfg_.nWindowFeatures}};
    tensorflow::Tensor hitsX_{
        tensorflow::DT_FLOAT,
        {static_cast<long int>(nItems), cfg_.maxNClusters, cfg_.maxNRechits, cfg_.nRechitsFeatures}};
    tensorflow::Tensor isSeedX_{tensorflow::DT_FLOAT, {static_cast<long int>(nItems), cfg_.maxNClusters, 1}};
    tensorflow::Tensor nClsSize_{tensorflow::DT_FLOAT, {static_cast<long int>(nItems)}};

    float* C = clsX_.flat<float>().data();
    // Look on batch dim
    for (size_t b = 0; b < nItems; b++) {
      const auto& cls_data = inputs.clustersX[iB * cfg_.batchSize + b];
      // Loop on clusters
      for (size_t k = 0; k < cfg_.maxNClusters; k++) {
        // Loop on features
        for (size_t z = 0; z < cfg_.nClusterFeatures; z++, C++) {
          if (k < cls_data.size()) {
            *C = float(cls_data[k][z]);
          } else {
            *C = 0.;
          }
        }
      }
    }

    float* W = windX_.flat<float>().data();
    // Look on batch dim
    for (size_t b = 0; b < nItems; b++) {
      const auto& wind_features = inputs.windowX[iB * cfg_.batchSize + b];
      // Loop on features
      for (size_t k = 0; k < cfg_.nWindowFeatures; k++, W++) {
        *W = float(wind_features[k]);
      }
    }

    float* H = hitsX_.flat<float>().data();
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
          for (size_t z = 0; z < cfg_.nRechitsFeatures; z++, H++) {
            if (ok)
              *H = float(hits_data[k][j][z]);
            else
              *H = 0.;
          }
        }
      }
    }

    float* S = isSeedX_.flat<float>().data();
    // Look on batch dim
    for (size_t b = 0; b < nItems; b++) {
      const auto& isSeed_data = inputs.isSeed[iB * cfg_.batchSize + b];
      // Loop on clusters
      for (size_t k = 0; k < cfg_.maxNClusters; k++, S++) {
        if (k < isSeed_data.size()) {
          *S = float(isSeed_data[k]);
        } else {
          *S = 0.;
        }
      }
    }

    float* M = nClsSize_.flat<float>().data();
    for (size_t b = 0; b < nItems; b++, M++) {
      *M = float(inputs.clustersX[iB * cfg_.batchSize + b].size());
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
    float* y_cl = outputs_tf[0].flat<float>().data();
    // Iterate on the clusters for each window
    for (size_t b = 0; b < nItems; b++) {
      uint ncls = inputs.clustersX[iB * cfg_.batchSize + b].size();
      std::vector<float> cl_output(ncls);
      for (size_t c = 0; c < ncls; c++) {
        float y = y_cl[b * cfg_.maxNClusters + c];
        // Applying sigmoid to logit
        cl_output[c] = 1 / (1 + std::exp(-y));
      }
      outputs_clustering.push_back(cl_output);
    }
  }

  return outputs_clustering;
}
