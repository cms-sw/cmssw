// Implementation of the mustache parameters interface

#include "RecoEcal/EgammaCoreTools/interface/MustacheSCParametersHelper.h"

#include <algorithm>
#include <utility>

using namespace reco;

MustacheSCParametersHelper::MustacheSCParametersHelper(EcalMustacheSCParameters &params) : parameters_(params) {}

MustacheSCParametersHelper::MustacheSCParametersHelper(EcalMustacheSCParameters &params,
                                                       const edm::ParameterSet &iConfig)
    : parameters_(params) {
  setSqrtLogClustETuning(iConfig.getParameter<double>("sqrtLogClustETuning"));

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
  parameters_.sqrtLogClustETuning_ = sqrtLogClustETuning;
}

void MustacheSCParametersHelper::addParabolaParameters(const EcalMustacheSCParameters::ParabolaParameters &params) {
  parameters_.parabolaParametersCollection_.emplace_back(params);
}

void MustacheSCParametersHelper::sortParabolaParametersCollection() {
  std::sort(parameters_.parabolaParametersCollection_.begin(),
            parameters_.parabolaParametersCollection_.end(),
            [](const EcalMustacheSCParameters::ParabolaParameters &p1,
               const EcalMustacheSCParameters::ParabolaParameters &p2) {
              const auto p1Mins = std::make_pair(p1.log10EMin, p1.etaMin);
              const auto p2Mins = std::make_pair(p2.log10EMin, p2.etaMin);
              return p1Mins < p2Mins;
            });
}
