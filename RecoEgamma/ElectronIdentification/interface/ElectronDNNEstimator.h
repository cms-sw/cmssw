#ifndef RecoEgamma_ElectronIdentification_ElectronDNNEstimator_h
#define RecoEgamma_ElectronIdentification_ElectronDNNEstimator_h

#include "RecoEgamma/EgammaTools/interface/EgammaDNNHelper.h"
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

#include <vector>
#include <memory>
#include <string>

class ElectronDNNEstimator {
public:
  ElectronDNNEstimator(const egammaTools::DNNConfiguration&, const bool useEBModelInGap);

  std::vector<tensorflow::Session*> getSessions() const;
  ;

  // Function returning a map with all the possible variables and their name
  std::map<std::string, float> getInputsVars(const reco::GsfElectron& ele) const;

  // Evaluate the DNN on all the electrons with the correct model
  std::vector<std::pair<uint, std::vector<float>>> evaluate(const reco::GsfElectronCollection& ele,
                                                            const std::vector<tensorflow::Session*>& sessions) const;

  // List of input variables names used to check the variables request as
  // inputs in a dynamic way from configuration file.
  // If an input variables is not found at construction time an expection is thrown.
  static const std::vector<std::string> dnnAvaibleInputs;

  static constexpr float ptThreshold = 10.;
  static constexpr float ecalBarrelMaxEtaWithGap = 1.566;
  static constexpr float ecalBarrelMaxEtaNoGap = 1.485;
  static constexpr float endcapBoundary = 2.5;
  static constexpr float extEtaBoundary = 2.65;

private:
  const egammaTools::EgammaDNNHelper dnnHelper_;

  const bool useEBModelInGap_;
};

#endif
