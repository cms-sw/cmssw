#ifndef RecoEgamma_EgammaTools_MVAVariableHelper_H
#define RecoEgamma_EgammaTools_MVAVariableHelper_H

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/Event.h"

#include <unordered_map>
#include <vector>
#include <string>

class MVAVariableHelper {
public:
  MVAVariableHelper(edm::ConsumesCollector&& cc)
      : tokens_({cc.consumes<double>(edm::InputTag("fixedGridRhoFastjetAll")),
                 cc.consumes<double>(edm::InputTag("fixedGridRhoAll"))}) {}

  const std::vector<float> getAuxVariables(const edm::Event& iEvent) const {
    return std::vector<float>{static_cast<float>(iEvent.get(tokens_[0])), static_cast<float>(iEvent.get(tokens_[1]))};
  }

  static std::unordered_map<std::string, int> indexMap() {
    return {{"fixedGridRhoFastjetAll", 0}, {"fixedGridRhoAll", 1}};
  }

private:
  const std::vector<edm::EDGetTokenT<double>> tokens_;
};

#endif
