// Implementation of the mustache parameters interface

#include "RecoEcal/EgammaCoreTools/interface/MustacheSCParametersHelper.h"

using namespace reco;

MustacheSCParametersHelper::MustacheSCParametersHelper(const EcalMustacheSCParameters &params) : EcalMustacheSCParameters(params)
{
}

MustacheSCParametersHelper::MustacheSCParametersHelper(const edm::ParameterSet& iConfig)
{
  sqrtLogClustETuning_ = iConfig.getParameter<double>("sqrtLogClustETuning"); //1.1

  // parabola parameters
  const auto parabolaPSets = iConfig.getParameter<std::vector<edm::ParameterSet>>("parabolaParameterSets");
  for (const auto &pSet : parabolaPSets) {
    EcalMustacheSCParameters::ParabolaParameters params = {
      pSet.getParameter<double>("log10EMin"),
      pSet.getParameter<double>("etaMin"),
      pSet.getParameter<std::vector<double>>("pUp"),
      pSet.getParameter<std::vector<double>>("pLow"),
      pSet.getParameter<std::vector<double>>("w0Up"),
      pSet.getParameter<std::vector<double>>("w1Up"),
      pSet.getParameter<std::vector<double>>("w0Low"),
      pSet.getParameter<std::vector<double>>("w1Low")};
    addParabolaParameters(params);
  }
}

float MustacheSCParametersHelper::sqrtLogClustETuning() const
{
  return sqrtLogClustETuning_;
}

void MustacheSCParametersHelper::setSqrtLogClustETuning(const float sqrtLogClustETuning)
{
  sqrtLogClustETuning_ = sqrtLogClustETuning;
}

EcalMustacheSCParameters::ParabolaParameters MustacheSCParametersHelper::parabolaParameters(float log10ClustE, float absSeedEta) const
{
  // assume the collection is sorted in descending ParabolaParameters.etaMin and ascending ParabolaParameters.minEt
  for (const auto &parabolaParams : parabolaParametersCollection_) {
    if (log10ClustE < parabolaParams.log10EMin || absSeedEta < parabolaParams.etaMin) {
      continue;
    } else {
      return parabolaParams;
    }
  }
  return EcalMustacheSCParameters::ParabolaParameters();
}

void MustacheSCParametersHelper::addParabolaParameters(const EcalMustacheSCParameters::ParabolaParameters &params)
{
  parabolaParametersCollection_.emplace_back(params);
}

