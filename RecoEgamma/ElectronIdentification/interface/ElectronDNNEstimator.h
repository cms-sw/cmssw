#ifndef __RecoEgamma_ElectronIdentification_ElectronDNNEstimator_H__
#define __RecoEgamma_ElectronIdentification_ElectronDNNEstimator_H__

#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

#include <vector>
#include <memory>
#include <string>


class ElectronDNNEstimator {
public:
  struct Configuration {
    std::string inputTensorName;
    std::string outputTensorName;
    std::vector<std::string> models_files;
    std::vector<std::string> scalers_files;
    std::string log_level="2";
  };
  static constexpr uint  nInputs = 24;
  static constexpr uint  nOutputs = 5;
  
  ElectronDNNEstimator();
  ElectronDNNEstimator(std::vector<std::string>& models_files, std::vector<std::string>& scalers_files,
                       std::string inputTensorName, std::string outputTensorName);
  ElectronDNNEstimator(const Configuration&);
  ~ElectronDNNEstimator(){ for(auto graph : graphDefs_) if(graph!=nullptr) delete graph;};

  std::vector<tensorflow::Session*> getSessions() const;

  // Function returning a map with all the possible variables and their name
  std::map<std::string, float> getInputsVars(const reco::GsfElectron& ele) const;
  // Function getting the input vector for a specific electron, already scaled 
  // together with the model index it has to be used (depending on pt/eta)
  std::pair<uint, std::vector<float>> getScaledInputs(const reco::GsfElectron& ele) const;

  uint getModelIndex(const reco::GsfElectron& ele) const;

  // Evaluate the DNN on all the electrons with the correct model
  std::vector<std::array<float,ElectronDNNEstimator::nOutputs>> evaluate(const reco::GsfElectronCollection& ele, const std::vector<tensorflow::Session*> session) const;

  // List of input variables names used to check the variables request as inputs in a dynamic way from configuration file.
  // If an input variables is not found at construction time an expection is thrown.
  static const std::array<std::string, nInputs> dnnAvaibleInputs;

private:

  void initTensorFlowGraphs() ;
  void initScalerFiles();

  const Configuration cfg_;

  std::vector< tensorflow::GraphDef* > graphDefs_; // --> Should be std::atomic but does not compile
  uint N_models_;
  std::vector<int> n_inputs_;
  /* List of input variables for each of the model; 
  * Each input variable is represented by the tuple <varname, standardization_type, par1, par2>
  * The standardization_type can be:
  * 0 = Do not scale the variable
  * 1 = standard norm. par1=mean, par2=std
  * 2 = MinMax. par1=min, par2=max */
  std::vector< std::vector< std::tuple< std::string, uint, float,float >> > features_maps_;
  
  bool debug_;
  
};

#endif
