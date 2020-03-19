#include "RecoEgamma/EgammaTools/interface/MVAVariableHelper.h"

MVAVariableHelper::MVAVariableHelper(edm::ConsumesCollector&& cc)
    : tokens_({cc.consumes<double>(edm::InputTag("fixedGridRhoFastjetAll")),
               cc.consumes<double>(edm::InputTag("fixedGridRhoAll"))}) {}

const std::vector<float> MVAVariableHelper::getAuxVariables(const edm::Event& iEvent) const {
  return std::vector<float>{getVariableFromDoubleToken(tokens_[0], iEvent),
                            getVariableFromDoubleToken(tokens_[1], iEvent)};
}

MVAVariableIndexMap::MVAVariableIndexMap() : indexMap_({{"fixedGridRhoFastjetAll", 0}, {"fixedGridRhoAll", 1}}) {}
