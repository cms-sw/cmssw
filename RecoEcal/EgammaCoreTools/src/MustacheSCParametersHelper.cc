// Implementation of the mustache parameters interface

#include "RecoEcal/EgammaCoreTools/interface/MustacheSCParametersHelper.h"

#include <algorithm>
#include <utility>

using namespace reco;

MustacheSCParametersHelper::MustacheSCParametersHelper(const EcalMustacheSCParameters &params)
    : EcalMustacheSCParameters(params) {}

MustacheSCParametersHelper::MustacheSCParametersHelper(const edm::ParameterSet &iConfig) {
  sqrtLogClustETuning_ = iConfig.getParameter<double>("sqrtLogClustETuning");  //1.1

  // parabola parameters
  const auto parabolaPSets = iConfig.getParameter<std::vector<edm::ParameterSet>>("parabolaParameterSets");
  for (const auto &pSet : parabolaPSets) {
    EcalMustacheSCParameters::ParabolaParameters params = {pSet.getParameter<double>("log10EMin"),
                                                           pSet.getParameter<double>("etaMin"),
                                                           pSet.getParameter<std::vector<double>>("pUp"),
                                                           pSet.getParameter<std::vector<double>>("pLow"),
                                                           pSet.getParameter<std::vector<double>>("w0Up"),
                                                           pSet.getParameter<std::vector<double>>("w1Up"),
                                                           pSet.getParameter<std::vector<double>>("w0Low"),
                                                           pSet.getParameter<std::vector<double>>("w1Low")};
    addParabolaParameters(params);
    sortParabolaParametersCollection();
  }
}

void MustacheSCParametersHelper::setSqrtLogClustETuning(const float sqrtLogClustETuning) {
  sqrtLogClustETuning_ = sqrtLogClustETuning;
}

void MustacheSCParametersHelper::addParabolaParameters(const EcalMustacheSCParameters::ParabolaParameters &params) {
  parabolaParametersCollection_.emplace_back(params);
}

void MustacheSCParametersHelper::sortParabolaParametersCollection() {
  std::sort(parabolaParametersCollection_.begin(),
            parabolaParametersCollection_.end(),
            [](const EcalMustacheSCParameters::ParabolaParameters &p1,
               const EcalMustacheSCParameters::ParabolaParameters &p2) {
              const auto p1Mins = std::make_pair(p1.log10EMin, p1.etaMin);
              const auto p2Mins = std::make_pair(p2.log10EMin, p2.etaMin);
              return p1Mins > p2Mins;
            });
}
