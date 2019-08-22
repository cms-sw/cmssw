#include "FastSimulation/ShowerDevelopment/interface/HSParameters.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

HSParameters::HSParameters(const edm::ParameterSet& param) {
  lossesOpt_ = param.getParameter<int>("lossesOpt");
  nDepthSteps_ = param.getParameter<int>("nDepthSteps");
  nTRsteps_ = param.getParameter<int>("nTRsteps");
  transParam_ = param.getParameter<double>("transRparam");
  eSpotSize_ = param.getParameter<double>("eSpotSize");
  depthStep_ = param.getParameter<double>("depthStep");
  criticalEnergy_ = param.getParameter<double>("criticalHDEnergy");
  maxTRfactor_ = param.getParameter<double>("maxTRfactor");
  balanceEH_ = param.getParameter<double>("balanceEH");
  hcalDepthFactor_ = param.getParameter<double>("hcalDepthFactor");
}
