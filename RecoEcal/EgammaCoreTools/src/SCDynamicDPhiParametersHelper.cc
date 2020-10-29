// Implementation of the SC dynamic dPhi parameters interface

#include "RecoEcal/EgammaCoreTools/interface/SCDynamicDPhiParametersHelper.h"

using namespace reco;

SCDynamicDPhiParametersHelper::SCDynamicDPhiParametersHelper(const EcalSCDynamicDPhiParameters &params) : EcalSCDynamicDPhiParameters(params)
{
}

SCDynamicDPhiParametersHelper::SCDynamicDPhiParametersHelper(const edm::ParameterSet& iConfig)
{
  // dynamic dPhi parameters
  const auto dynamicDPhiPSets = iConfig.getParameter<std::vector<edm::ParameterSet>>("dynamicDPhiParameterSets");
  for (const auto &pSet : dynamicDPhiPSets) {
    EcalSCDynamicDPhiParameters::DynamicDPhiParameters params({
      pSet.getParameter<double>("eMin"),
      pSet.getParameter<double>("etaMin"),
      pSet.getParameter<double>("yoffset"),
      pSet.getParameter<double>("scale"),
      pSet.getParameter<double>("xoffset"),
      pSet.getParameter<double>("width"),
      pSet.getParameter<double>("saturation"),
      pSet.getParameter<double>("cutoff")});
    addDynamicDPhiParameters(params);
  }
}

EcalSCDynamicDPhiParameters::DynamicDPhiParameters SCDynamicDPhiParametersHelper::dynamicDPhiParameters(double clustE, double absSeedEta) const
{
  // assume the collection is sorted in descending DynamicDPhiParams.etaMin and ascending DynamicDPhiParams.eMin
  for (const auto &dynamicDPhiParams : dynamicDPhiParametersCollection_) { 
    if (clustE < dynamicDPhiParams.eMin || absSeedEta < dynamicDPhiParams.etaMin) {
      continue;
    } else {
      return dynamicDPhiParams;
    }
  }
  return EcalSCDynamicDPhiParameters::DynamicDPhiParameters();
}

void SCDynamicDPhiParametersHelper::addDynamicDPhiParameters(const EcalSCDynamicDPhiParameters::DynamicDPhiParameters &params)
{
  dynamicDPhiParametersCollection_.emplace_back(params);
}

