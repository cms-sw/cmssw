#ifndef __RecoEgamma_PhotonIdentification_PhotonDNNEstimator_H__
#define __RecoEgamma_PhotonIdentification_PhotonDNNEstimator_H__

#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"

#include <vector>
#include <memory>
#include <string>


class PhotonDNNEstimator {
public:
  struct Configuration {
    std::string inputTensorName;
    std::string outputTensorName;
    std::vector<std::string> models_files;
    std::vector<std::string> scalers_files;
    std::string log_level="2";
  };
  static constexpr uint nInputs = 13;
  static constexpr uint nOutputs = 1;

  PhotonDNNEstimator();
  PhotonDNNEstimator(std::vector<std::string>& models_files, std::vector<std::string>& scalers_files,
                       std::string inputTensorName, std::string outputTensorName);
  PhotonDNNEstimator(const Configuration&);
  ~PhotonDNNEstimator(){ for(auto graph : graphDefs_) if(graph!=nullptr) delete graph;};

  std::vector<tensorflow::Session*> getSessions() const;

  // Function returning a map with all the possible variables and their name
  std::map<std::string, float> getInputsVars(const reco::Photon& photon) const;
  // Function getting the input vector for a specific Photon, already scaled 
  // together with the model index it has to be used (depending on pt/eta)
  std::pair<uint, std::vector<float>> getScaledInputs(const reco::Photon& photon) const;

  uint getModelIndex(const reco::Photon& photon) const;

  // Evaluate the DNN on all the Photons with the correct model
  std::vector<std::array<float,PhotonDNNEstimator::nOutputs>> evaluate(const reco::PhotonCollection& photons, const std::vector<tensorflow::Session*> session) const;

  static const std::array<std::string, nInputs > dnnAvaibleInputs;

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
