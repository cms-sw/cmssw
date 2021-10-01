#include "RecoEgamma/PhotonIdentification/interface/PhotonDNNEstimator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <fstream>
#include <memory>


PhotonDNNEstimator::PhotonDNNEstimator() : cfg_{} {}

PhotonDNNEstimator::PhotonDNNEstimator( std::vector<std::string>& models_files,  std::vector<std::string>& scalers_files,
                                            std::string inputTensorName,  std::string outputTensorName)
 : cfg_{.inputTensorName=inputTensorName,.outputTensorName=outputTensorName,
        .models_files=models_files,.scalers_files=scalers_files,.log_level="2"} {
  N_models_ = cfg_.models_files.size();
  debug_ = cfg_.log_level == "0";
  initTensorFlowGraphs();
  initScalerFiles();
  LogDebug("PhotonDNNPFid") << "Photon PFID DNN evaluation with " << N_models_ << " models and "<< n_inputs_[0] << " variables --> LOADED";  
}


PhotonDNNEstimator::PhotonDNNEstimator(const Configuration& cfg) : cfg_(cfg) {
   // Init tensorflow sessions
   N_models_ = cfg_.models_files.size();
   debug_ = cfg_.log_level == "0";
   initTensorFlowGraphs();
   initScalerFiles();
   LogDebug("PhotonDNNPFid") << "Photon PFID DNN evaluation with " << N_models_ << " models and "<< n_inputs_[0] << " variables --> LOADED";
}

void PhotonDNNEstimator::initTensorFlowGraphs(){
    // configure logging to show warnings (see table below)
    tensorflow::setLogging(cfg_.log_level);
    // load the graph definition  
    LogDebug("PhotonDNNPFid") << "Loading " << N_models_ << " graphs"; 
    for(auto model_file : cfg_.models_files){
      tensorflow::GraphDef* graphDef = tensorflow::loadGraphDef(model_file); //-->should be atomic but does not compile
      graphDefs_.push_back(graphDef);
    }
    LogDebug("PhotonDNNPFid") << "Graphs loaded"; 
}

void PhotonDNNEstimator::initScalerFiles(){
  for (auto scaler_file : cfg_.scalers_files){
    // Parse scaler configuration
    std::vector<std::tuple<std::string,float,float>> features;
    std::ifstream inputfile_scaler{scaler_file};
    int ninputs = 0;
    if(inputfile_scaler.fail())  
    { 
        throw cms::Exception("MissingFile") << "Scaler file for Electron PFid DNN not found";
    }else{ 
        // Now read mean, scale factors for each variable
        float m,s;
        std::string varname{};
        while (inputfile_scaler >> varname >> m >> s){
            features.push_back(std::make_tuple(varname, m,s));
            auto match = std::find(PhotonDNNEstimator::dnnAvaibleInputs.begin(),PhotonDNNEstimator::dnnAvaibleInputs.end(), varname);
            if (match == std::end(PhotonDNNEstimator::dnnAvaibleInputs)) {
              throw cms::Exception("MissingVariable") << "Requested variable (" << varname << ") not available between Photon PFid DNN inputs";
            }
            ninputs += 1;
        }  
    }   
    inputfile_scaler.close();
    features_maps_.push_back(features);
    n_inputs_.push_back(ninputs);
  }
}


std::vector<tensorflow::Session*> PhotonDNNEstimator::getSessions() const{
  LogDebug("PhotonDNNPFid") << "Starting " << N_models_ << "TF sessions"; 
   std::vector<tensorflow::Session*> sessions;
   for (auto & graphDef : graphDefs_){
     sessions.push_back(tensorflow::createSession(graphDef));
   }
   LogDebug("PhotonDNNPFid") << "TF sessions started"; 
   return sessions;
}

const std::array<std::string, PhotonDNNEstimator::nInputs > PhotonDNNEstimator::dnnAvaibleInputs = {{
    "hadTowOverEm","phoTrkSumPtHollow","phoEcalRecHit",
    "phoSigmaIetaIeta","phoSigmaIEtaIEtaFull5x5","phoSigmaIEtaIPhiFull5x5",
    "phoEcalPFClusterIso","phoHcalPFClusterIso","phoHasPixelSeed",
    "phoR9Full5x5","phohcalTower"
  }};

std::map<std::string, float> PhotonDNNEstimator::getInputsVars(const reco::Photon& photon) const{
    // Prepare a map with all the defined variables
    std::map<std::string, float> variables;
    variables["hadTowOverEm"] = photon.hadTowOverEmValid();
    variables["phoTrkSumPtHollow"] = photon.trkSumPtHollowConeDR03();
    variables["phoEcalRecHit"] = photon.ecalRecHitSumEtConeDR03();
    variables["phoSigmaIetaIeta"] = photon.sigmaIetaIeta();
    variables["phoSigmaIEtaIEtaFull5x5"] = photon.full5x5_sigmaIetaIeta();
    variables["phoSigmaIEtaIPhiFull5x5"] = photon.full5x5_showerShapeVariables().sigmaIetaIphi;
    variables["phoEcalPFClusterIso"] = photon.ecalPFClusterIso();
    variables["phoHcalPFClusterIso"] = photon.hcalPFClusterIso();
    variables["phoHasPixelSeed"] = (Int_t)photon.hasPixelSeed();
    variables["phoR9Full5x5"] = photon.full5x5_r9();
    variables["phohcalTower"] = photon.hcalTowerSumEtConeDR03();
    // Define more variables here and use them directly in the model config!
    return variables;
}

uint PhotonDNNEstimator::getModelIndex(const reco::Photon& photon) const{
  /* 
  Selection of the model to be applied on the Photon based on pt/eta cuts or whatever selection
  */
    uint modelIndex;
    if(std::abs(photon.eta()) <= 1.466){
      modelIndex = 0;
    }else{
      modelIndex = 1;
    }
    return modelIndex;
}

std::pair<uint, std::vector<float>> PhotonDNNEstimator::getScaledInputs(const reco::Photon& photon) const{
    uint modelIndex = getModelIndex(photon);
    auto allInputs = getInputsVars(photon);
    std::vector<float> inputs;
    // Loop on the list of requested variables and scaling values for the specific modelIndex
    for( auto& [varName, mean, scale] : features_maps_[modelIndex]){
      inputs.push_back( (allInputs[varName]-mean)/scale );
      //TODO Add protection for mismatch between requested variables and the available ones
    } 
    return std::make_pair(modelIndex, inputs);
}

std::vector<std::array<float,PhotonDNNEstimator::nOutputs>> PhotonDNNEstimator::evaluate(const reco::PhotonCollection& photons, const std::vector<tensorflow::Session*> sessions) const {
    /*
      Evaluate the Photon PFID DNN for all the Photons. 
      3 models are defined depending on the pt and eta --> we need to build 3 input tensors to evaluate
      the DNNs with batching.
      
      1) Get the inputs vector, already scaled correctly for each Photon and the modelIndex
      2) Prepare 3 input tensors for the 3 models
      3) Run the model and get the output for each Photon
      4) Sort the output by Photon index
      5) Return the DNN output 

    */
    std::vector<int> model_index_map; // for each photon the model index is saved
    std::vector<std::vector<int>> photon_index_map (N_models_); // for each model; the list of ele index is saved 
    std::vector<std::vector<float>> inputs_vectors;
    int counts[3] = {0,0,0}; 

    LogDebug("PhotonDNNPFid") << "Working on "<< photons.size() << " photons"; 

    int iphoton = -1;
    for (auto & photon : photons){
      iphoton++;
      LogDebug("PhotonDNNPFid") << "Working on photon: "<< iphoton; 
      auto [model_index, inputs] = getScaledInputs(photon);
      counts[model_index] += 1;
      model_index_map.push_back(model_index);
      photon_index_map[model_index].push_back(iphoton);
      inputs_vectors.push_back(inputs);
    }

    // Prepare one input tensors for each model
    std::vector<tensorflow::Tensor> input_tensors (N_models_);
     // Pointers for filling efficiently the input tensors
    std::vector<float*> input_tensors_pointer (N_models_);
    for (uint i=0; i< N_models_; i++) {
      LogDebug("PhotonDNNPFid") << "Initializing TF input " << i << " with rows:" << counts[i] << " and cols:" << n_inputs_[i]; 
      input_tensors[i] = tensorflow::Tensor{tensorflow::DT_FLOAT, { counts[i], n_inputs_[i] }};
      input_tensors_pointer[i] = input_tensors[i].flat<float>().data();
    }

    // Filling the input tensors 
    for (uint m=0; m< N_models_; m++) {
      LogDebug("PhotonDNNPFid") << "Loading TF input tensor for model: "<< m;
      float * T = input_tensors_pointer[m];
      for (int photon_index : photon_index_map[m]){
        for (int k = 0; k < n_inputs_[m]; k++, T++ ){ //Note the input tensor pointer incremented
          *T =  inputs_vectors[photon_index][k];
        }
      }
    }

    // Define the output and run
    std::vector< std::pair< int, std::array<float,PhotonDNNEstimator::nOutputs>>> outputs;
    // Run all the models
    for (uint i=0; i< N_models_; i++) {
      if (counts[i] ==0) continue; //Skip model witout inputs
      std::vector<tensorflow::Tensor> output;
      LogDebug("PhotonDNNPFid") << "Run model: " << i << " with " << counts[i] << " photons";
      tensorflow::run(sessions[i], {{cfg_.inputTensorName, input_tensors[i]}}, {cfg_.outputTensorName}, &output);
      // Get the output and save the PhotonDNNEstimator::nOutputs numbers along with the photon index
      auto r = output[0].tensor<float,2>();
      // Iterate on the list of elements in the batch --> many Photons
      for (int b =0; b< counts[i]; b++){
          std::array<float,PhotonDNNEstimator::nOutputs> result;
          for (uint k=0; k< PhotonDNNEstimator::nOutputs ; k++) 
              result[k] = r(b, k);
          // Get the original index of the electorn in the original order
          int photon_index = photon_index_map[i][b];
          LogDebug("PhotonDNNPFid") << "DNN output, model "<<i << " photon "<<photon_index << " : " 
                        << result[0] << " "<<result[1] << " "<<result[2] << " "<<result[3] << " "<<result[4];
          outputs.push_back(std::make_pair(photon_index, result));
      }
    }

    // Now we have just to re-order the outputs
    std::sort(outputs.begin(), outputs.end());
    std::vector<std::array<float,PhotonDNNEstimator::nOutputs>> final_outputs (outputs.size());
    std::transform(outputs.begin(), outputs.end(), final_outputs.begin(), [](auto a){return a.second;});

    return final_outputs;
}


    
    