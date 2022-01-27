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

DeepSCGraphEvaluation::~DeepSCGraphEvaluation(){
  if(session_ != nullptr) tensorflow::closeSession(session_);
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
  LogDebug("DeepSCGraphEvaluation") << "Reading scaler file: "<< edm::FileInPath(file).fullPath();
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

std::vector<double> DeepSCGraphEvaluation::scaleClusterFeatures(
    const std::vector<double>& input) const {
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
  // Input tensors initialization
  tensorflow::Tensor clsX {tensorflow::DT_FLOAT, {inputs.batchSize, cfg_.maxNClusters, cfg_.nClusterFeatures}};
  tensorflow::Tensor windX {tensorflow::DT_FLOAT, {inputs.batchSize, cfg_.nWindowFeatures}};
  tensorflow::Tensor hitsX {tensorflow::DT_FLOAT, {inputs.batchSize, cfg_.maxNClusters, cfg_.maxNRechits, cfg_.nRechitsFeatures }};
  tensorflow::Tensor isSeedX {tensorflow::DT_FLOAT, {inputs.batchSize, cfg_.maxNClusters, 1}};
  tensorflow::Tensor nClsSize {tensorflow::DT_FLOAT, {inputs.batchSize}};

  float * C = clsX.flat<float>().data();
  // Look on batch dim
  for (const auto & cls_data : inputs.clustersX ){
    // Loop on clusters
    for (size_t k = 0; k < cfg_.maxNClusters; k++){
      // Loop on features
      for (size_t z=0; z < cfg_.nClusterFeatures; z++, C++){//--> note the double loop on the tensor pointer
        if (k < cls_data.size()){
          *C = float(cls_data[k][z]);
        }else{
          *C = 0.;
        }
      }
    }
  }

  float * W = windX.flat<float>().data();
  // Look on batch dim
  for (const auto & wind_features : inputs.windowX ){
    // Loop on features
    for (size_t k = 0; k < cfg_.nWindowFeatures; k++, W++){ //--> note the double loop on the tensor pointer
        *W =  float(wind_features[k]);
    }
  }

  float * H = hitsX.flat<float>().data();
  size_t iW = -1;
  // Look on batch dim
  for (const auto & hits_data : inputs.hitsX ){
    iW++;
    size_t ncls_in_window = hits_data.size();
    // Loop on clusters
    for (size_t k = 0; k < cfg_.maxNClusters; k++){ //--> note the triple loop on the tensor pointer
      // Check padding
      size_t nhits_in_cluster;
      if (k < ncls_in_window)  nhits_in_cluster = hits_data[k].size();
      else                     nhits_in_cluster = 0;

      // Loop on hits
      for (size_t j=0; j < cfg_.maxNRechits; j++){//--> note the triple loop on the tensor pointer
        // Check the number of clusters and hits for padding
        bool ok = j < nhits_in_cluster;
        // Loop on rechits features
        for (size_t z=0; z< cfg_.nRechitsFeatures; z++, H++){//--> note the triple loop on the tensor pointe
          if (ok)  *H = float( hits_data[k][j][z]);
          else    *H = 0.;
        }
      }
    }
  }

  float * S = isSeedX.flat<float>().data();
  // Look on batch dim
  for (const auto & isSeed_data : inputs.isSeed ){
    // Loop on clusters
    for (size_t k = 0; k < cfg_.maxNClusters; k++, S++){ //--> note the double loop on the tensor pointer
      if (k < isSeed_data.size()){
          *S =  float(isSeed_data[k]);
      }else{
        *S = 0.;
      }
    }
  }

  float * M = nClsSize.flat<float>().data();
  for (size_t k = 0; k < inputs.batchSize; k++, M++){
      *M =  float(inputs.clustersX[k].size());
  }

  std::vector<std::pair<std::string, tensorflow::Tensor>> feed_dict = {
    { "input_1", clsX },
    { "input_2", windX},
    { "input_3", hitsX},
    { "input_4", isSeedX},
    { "input_5", nClsSize}
  };

  // prepare tensorflow outputs
  std::vector<tensorflow::Tensor> outputs_tf;
  // // Define the output and run
  std::vector<std::vector<float>> outputs_clustering;
  // // Run the models
  LogDebug("DeepSCGraphEvaluation") << "Run model";
  tensorflow::run(session_, feed_dict, {"Identity", "Identity_1","Identity_2","Identity_3"}, &outputs_tf);

  // Reading the 1st output: clustering probability
  // const auto& r = outputs_tf[0].tensor<float, 3>();

  float * y_cl = outputs_tf[0].flat<float>().data();
  // Iterate on the clusters for each window
  for (size_t b = 0; b< inputs.batchSize; b++) {
    uint ncls = inputs.clustersX[b].size();
    std::vector<float> cl_output(ncls);
    for (size_t c = 0; c < ncls; c++){
      float y = y_cl[b*cfg_.maxNClusters + c];
      cl_output[c] = 1 / (1 + TMath::Exp(- y));
    }
    std::cout << b << ") ";
    std::for_each(cl_output.begin(), cl_output.end(), [](float x){std::cout <<x << " ";});
    std::cout << std::endl;
    outputs_clustering.push_back(cl_output);
  }

  return outputs_clustering;
}



// Cache for SuperCluster Producer containing Tensorflow objects
SCProducerCache::SCProducerCache(const edm::ParameterSet& conf) {
    // Here we will have to load the DNN PFID if present in the config
    reco::DeepSCConfiguration config;
    auto clustering_type = conf.getParameter<std::string>("ClusteringType");
    const auto& pset_dnn = conf.getParameter<edm::ParameterSet>("deepSuperClusterGraphConfig");

    if (clustering_type == "DeepSC") {
      config.modelFile = pset_dnn.getParameter<std::string>("modelFile");
      config.scalerFileClusterFeatures = pset_dnn.getParameter<std::string>("scalerFileClusterFeatures");
      config.scalerFileWindowFeatures = pset_dnn.getParameter<std::string>("scalerFileWindowFeatures");
      config.nClusterFeatures = pset_dnn.getParameter<uint>("nClusterFeatures");
      config.nWindowFeatures = pset_dnn.getParameter<uint>("nWindowFeatures");
      config.maxNClusters = pset_dnn.getParameter<uint>("maxNClusters");
      config.maxNRechits = pset_dnn.getParameter<uint>("maxNRechits");
      deepSCEvaluator = std::make_unique<DeepSCGraphEvaluation>(config);
    }
}
