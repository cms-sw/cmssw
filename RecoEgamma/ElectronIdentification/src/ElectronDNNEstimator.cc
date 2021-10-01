#include "RecoEgamma/ElectronIdentification/interface/ElectronDNNEstimator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <fstream>
#include <memory>


ElectronDNNEstimator::ElectronDNNEstimator() : cfg_{} {}

ElectronDNNEstimator::ElectronDNNEstimator( std::vector<std::string>& models_files,  std::vector<std::string>& scalers_files,
                                            std::string inputTensorName,  std::string outputTensorName)
 : cfg_{.inputTensorName=inputTensorName,.outputTensorName=outputTensorName,
        .models_files=models_files,.scalers_files=scalers_files,.log_level="2"} {
  N_models_ = cfg_.models_files.size();
  debug_ = cfg_.log_level == "0";
  initTensorFlowGraphs();
  initScalerFiles();
  LogDebug("EleDNNPFid") << "Ele PFID DNN evaluation with " << N_models_ << " models and "<< n_inputs_[0] << " variables --> LOADED";  
}

ElectronDNNEstimator::ElectronDNNEstimator(const Configuration& cfg) : cfg_(cfg) {
   // Init tensorflow sessions
   N_models_ = cfg_.models_files.size();
   debug_ = cfg_.log_level == "0";
   initTensorFlowGraphs();
   initScalerFiles();
   LogDebug("EleDNNPFid") << "Ele PFID DNN evaluation with " << N_models_ << " models and "<< n_inputs_[0] << " variables --> LOADED";
}


void ElectronDNNEstimator::initTensorFlowGraphs(){
    // configure logging to show warnings (see table below)
    tensorflow::setLogging(cfg_.log_level);
    // load the graph definition  
    LogDebug("EleDNNPFid") << "Loading " << N_models_ << " graphs"; 
    for(auto model_file : cfg_.models_files){
      tensorflow::GraphDef* graphDef = tensorflow::loadGraphDef(model_file); //-->should be atomic but does not compile
      graphDefs_.push_back(graphDef);
    }
    LogDebug("EleDNNPFid") << "Graphs loaded"; 
}

void ElectronDNNEstimator::initScalerFiles(){
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
            // Protection for mismatch between requested variables and the available ones
            auto match = std::find(ElectronDNNEstimator::dnnAvaibleInputs.begin(),ElectronDNNEstimator::dnnAvaibleInputs.end(), varname);
            if (match == std::end(ElectronDNNEstimator::dnnAvaibleInputs)) {
              throw cms::Exception("MissingVariable") << "Requested variable (" << varname << ") not available between Electron PFid DNN inputs";
            }
            ninputs += 1;
        }  
    }   
    inputfile_scaler.close();
    features_maps_.push_back(features);
    n_inputs_.push_back(ninputs);
  }
}


std::vector<tensorflow::Session*> ElectronDNNEstimator::getSessions() const{
  LogDebug("EleDNNPFid") << "Starting " << N_models_ << "TF sessions"; 
   std::vector<tensorflow::Session*> sessions;
   for (auto & graphDef : graphDefs_){
     sessions.push_back(tensorflow::createSession(graphDef));
   }
   LogDebug("EleDNNPFid") << "TF sessions started"; 
   return sessions;
}

const std::array<std::string, ElectronDNNEstimator::nInputs>  ElectronDNNEstimator::dnnAvaibleInputs = {{
    "fbrem", "abs(deltaEtaSuperClusterTrackAtVtx)", "abs(deltaPhiSuperClusterTrackAtVtx)",
    "full5x5_sigmaIetaIeta","full5x5_hcalOverEcal", "eSuperClusterOverP",
    "full5x5_e1x5","eEleClusterOverPout","closestCtfTrackNormChi2", 
    "closestCtfTrackNLayers", "gsfTrack.missing_inner_hits", "dr03TkSumPt", 
    "dr03EcalRecHitSumEt", "dr03HcalTowerSumEt", "gsfTrack.normalizedChi2",
     "superCluster.eta", "pt",  "ecalPFClusterIso", 
     "hcalPFClusterIso", "numberOfBrems", "abs(deltaEtaSeedClusterTrackAtCalo)",
    "hadronicOverEm", "full5x5_e2x5Max", "full5x5_e5x5"
  }};


std::map<std::string, float> ElectronDNNEstimator::getInputsVars(const reco::GsfElectron& ele) const{
    // Prepare a map with all the defined variables
    std::map<std::string, float> variables;
    reco::TrackRef myTrackRef = ele.closestCtfTrackRef();
    bool validKF = (myTrackRef.isNonnull() && myTrackRef.isAvailable());
    variables["fbrem"] = ele.fbrem();
    variables["abs(deltaEtaSuperClusterTrackAtVtx)"] = std::abs(ele.deltaEtaSuperClusterTrackAtVtx());
    variables["abs(deltaPhiSuperClusterTrackAtVtx)"] = std::abs(ele.deltaPhiSuperClusterTrackAtVtx());
    variables["full5x5_sigmaIetaIeta"] = ele.full5x5_sigmaIetaIeta();
    variables["full5x5_hcalOverEcal"] = ele.full5x5_hcalOverEcal();
    variables["eSuperClusterOverP"] = ele.eSuperClusterOverP();
    variables["full5x5_e1x5"] = ele.full5x5_e1x5();
    variables["eEleClusterOverPout"] = ele.eEleClusterOverPout();
    variables["closestCtfTrackNormChi2"] = ele.closestCtfTrackNormChi2();
    variables["closestCtfTrackNLayers"] = ele.closestCtfTrackNLayers();
    variables["gsfTrack.missing_inner_hits"] = (validKF) ? myTrackRef->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS) : -1.;
    variables["dr03TkSumPt"] = ele.dr03TkSumPt();
    variables["dr03EcalRecHitSumEt"] = ele.dr03EcalRecHitSumEt();
    variables["dr03HcalTowerSumEt"] = ele.dr03HcalTowerSumEt();
    variables["gsfTrack.normalizedChi2"] =  (validKF) ? myTrackRef->normalizedChi2() : 0;
    variables["superCluster.eta"] = ele.superCluster()->eta();
    variables["pt"] = ele.pt();
    variables["ecalPFClusterIso"] = ele.ecalPFClusterIso();
    variables["hcalPFClusterIso"] = ele.hcalPFClusterIso();
    variables["numberOfBrems"] = ele.numberOfBrems();
    variables["abs(deltaEtaSeedClusterTrackAtCalo)"] = std::abs(ele.deltaEtaSeedClusterTrackAtCalo());
    variables["hadronicOverEm"] = ele.hadronicOverEm();
    variables["full5x5_e2x5Max"] = ele.full5x5_e2x5Max();
    variables["full5x5_e5x5 "] = ele.full5x5_e5x5();
    // Define more variables here and use them directly in the model config!
    return variables;
}

uint ElectronDNNEstimator::getModelIndex(const reco::GsfElectron& ele) const{
  /* 
  Selection of the model to be applied on the electron based on pt/eta cuts or whatever selection
  */
    uint modelIndex;
    if (ele.pt() < 10) modelIndex = 0;
    if (ele.pt() >= 10){
      if(std::abs(ele.eta()) <= 1.466){
        modelIndex = 1;
      }else{
        modelIndex = 2;
      }
    }
    return modelIndex;
}

std::pair<uint, std::vector<float>> ElectronDNNEstimator::getScaledInputs(const reco::GsfElectron& ele) const{
    uint modelIndex = getModelIndex(ele);
    auto allInputs = getInputsVars(ele);
    std::vector<float> inputs;
    // Loop on the list of requested variables and scaling values for the specific modelIndex
    for( auto& [varName, mean, scale] : features_maps_[modelIndex]){
      inputs.push_back( (allInputs[varName]-mean)/scale );
      //TODO Add protection for mismatch between requested variables and the available ones
    } 
    return std::make_pair(modelIndex, inputs);
}

std::vector<std::array<float,ElectronDNNEstimator::nOutputs>> ElectronDNNEstimator::evaluate(const reco::GsfElectronCollection& electrons, const std::vector<tensorflow::Session*> sessions) const {
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
    std::vector<int> model_index_map; // for each ele the model index is saved
    std::vector<std::vector<int>> ele_index_map (N_models_); // for each model; the list of ele index is saved 
    std::vector<std::vector<float>> inputs_vectors;
    int counts[3] = {0,0,0}; 

    LogDebug("EleDNNPFid") << "Working on "<< electrons.size() << " electrons"; 

    int iele = -1;
    for (auto & ele : electrons){
      iele++;
      LogDebug("EleDNNPFid") << "Working on ele: "<< iele; 
      auto [model_index, inputs] = getScaledInputs(ele);
      counts[model_index] += 1;
      model_index_map.push_back(model_index);
      ele_index_map[model_index].push_back(iele);
      inputs_vectors.push_back(inputs);
    }

    // Prepare one input tensors for each model
    std::vector<tensorflow::Tensor> input_tensors (N_models_);
     // Pointers for filling efficiently the input tensors
    std::vector<float*> input_tensors_pointer (N_models_);
    for (uint i=0; i< N_models_; i++) {
      LogDebug("EleDNNPFid") << "Initializing TF input " << i << " with rows:" << counts[i] << " and cols:" << n_inputs_[i]; 
      input_tensors[i] = tensorflow::Tensor{tensorflow::DT_FLOAT, { counts[i], n_inputs_[i] }};
      input_tensors_pointer[i] = input_tensors[i].flat<float>().data();
    }

    // Filling the input tensors 
    for (uint m=0; m< N_models_; m++) {
      LogDebug("EleDNNPFid") << "Loading TF input tensor for model: "<< m;
      float * T = input_tensors_pointer[m];
      for (int ele_index : ele_index_map[m]){
        for (int k = 0; k < n_inputs_[m]; k++, T++ ){ //Note the input tensor pointer incremented
          *T =  inputs_vectors[ele_index][k];
        }
      }
    }

    // Define the output and run
    std::vector< std::pair< int, std::array<float,ElectronDNNEstimator::nOutputs>>> outputs;
    // Run all the models
    for (uint i=0; i< N_models_; i++) {
      if (counts[i] ==0) continue; //Skip model witout inputs
      std::vector<tensorflow::Tensor> output;
      LogDebug("EleDNNPFid") << "Run model: " << i << " with " << counts[i] << " electrons";
      tensorflow::run(sessions[i], {{cfg_.inputTensorName, input_tensors[i]}}, {cfg_.outputTensorName}, &output);
      // Get the output and save the ElectronDNNEstimator::nOutputs numbers along with the ele index
      auto r = output[0].tensor<float,2>();
      // Iterate on the list of elements in the batch --> many electrons
      for (int b =0; b< counts[i]; b++){
          std::array<float,ElectronDNNEstimator::nOutputs> result;
          for (uint k=0; k<ElectronDNNEstimator::nOutputs; k++) 
              result[k] = r(b, k);
          // Get the original index of the electorn in the original order
          int ele_index = ele_index_map[i][b];
          LogDebug("EleDNNPFid") << "DNN output, model "<<i << " ele "<<ele_index << " : " 
                        << result[0] << " "<<result[1] << " "<<result[2] << " "<<result[3] << " "<<result[4];
          outputs.push_back(std::make_pair(ele_index, result));
      }
    }

    // Now we have just to re-order the outputs
    std::sort(outputs.begin(), outputs.end());
    std::vector<std::array<float,ElectronDNNEstimator::nOutputs>> final_outputs (outputs.size());
    std::transform(outputs.begin(), outputs.end(), final_outputs.begin(), [](auto a){return a.second;});

    return final_outputs;
}


    
    