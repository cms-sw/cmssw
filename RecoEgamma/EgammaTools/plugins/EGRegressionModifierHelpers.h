#ifndef RecoEgamma_EgammaTools_EGRegressionModifier_H
#define RecoEgamma_EgammaTools_EGRegressionModifier_H

#include "CondFormats/DataRecord/interface/GBRDWrapperRcd.h"
#include "CondFormats/EgammaObjects/interface/GBRForestD.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <vector>

struct EGRegressionModifierCondTokens {
  EGRegressionModifierCondTokens(edm::ParameterSet const& config,
                                 std::string const& regressionKey,
                                 std::string const& uncertaintyKey,
                                 edm::ConsumesCollector& cc);
  std::vector<edm::ESGetToken<GBRForestD, GBRDWrapperRcd>> mean;
  std::vector<edm::ESGetToken<GBRForestD, GBRDWrapperRcd>> sigma;
};

std::vector<const GBRForestD*> retrieveGBRForests(
    edm::EventSetup const& evs, std::vector<edm::ESGetToken<GBRForestD, GBRDWrapperRcd>> const& tokens);

#endif
