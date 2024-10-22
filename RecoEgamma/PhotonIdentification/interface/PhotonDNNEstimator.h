#ifndef RecoEgamma_PhotonIdentification_PhotonDNNEstimator_h
#define RecoEgamma_PhotonIdentification_PhotonDNNEstimator_h

#include "RecoEgamma/EgammaTools/interface/EgammaDNNHelper.h"
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"

#include <vector>
#include <memory>
#include <string>

class PhotonDNNEstimator {
public:
  PhotonDNNEstimator(const egammaTools::DNNConfiguration&, const bool useEBModelInGap);

  std::vector<tensorflow::Session*> getSessions() const;
  ;

  // Function returning a map with all the possible variables and their name
  std::map<std::string, float> getInputsVars(const reco::Photon& ele) const;

  // Evaluate the DNN on all the electrons with the correct model
  std::vector<std::pair<uint, std::vector<float>>> evaluate(const reco::PhotonCollection& ele,
                                                            const std::vector<tensorflow::Session*>& sessions) const;

  // List of input variables names used to check the variables request as
  // inputs in a dynamic way from configuration file.
  // If an input variables is not found at construction time an expection is thrown.
  static const std::vector<std::string> dnnAvaibleInputs;

  static constexpr float ecalBarrelMaxEtaWithGap = 1.566;
  static constexpr float ecalBarrelMaxEtaNoGap = 1.485;

private:
  const egammaTools::EgammaDNNHelper dnnHelper_;

  const bool useEBModelInGap_;
};

#endif
