#ifndef RecoEgamma_EgammaTools_EpCombinationTool_h
#define RecoEgamma_EgammaTools_EpCombinationTool_h

#include "RecoEgamma/EgammaTools/interface/EgammaRegressionContainer.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"

#include <string>
#include <vector>
#include <utility>

namespace edm {
  class ParameterSet;
  class ParameterSetDescription;
  class EventSetup;
  class ConsumesCollector;
}  // namespace edm

class EpCombinationTool {
public:
  EpCombinationTool(const edm::ParameterSet& iConfig, edm::ConsumesCollector&& cc);
  ~EpCombinationTool() {}

  static edm::ParameterSetDescription makePSetDescription();

  void setEventContent(const edm::EventSetup& iSetup);
  std::pair<float, float> combine(const reco::GsfElectron& electron) const;
  std::pair<float, float> combine(const reco::GsfElectron& electron, float corrEcalEnergyErr) const;

private:
  EgammaRegressionContainer ecalTrkEnergyRegress_;
  EgammaRegressionContainer ecalTrkEnergyRegressUncert_;
  float maxEcalEnergyForComb_;
  float minEOverPForComb_;
  float maxEPDiffInSigmaForComb_;
  float maxRelTrkMomErrForComb_;
};

#endif
