#include "RecoEgamma/EgammaTools/plugins/EGRegressionModifierHelpers.h"

EGRegressionModifierCondTokens::EGRegressionModifierCondTokens(edm::ParameterSet const& config,
                                                               std::string const& regressionKey,
                                                               std::string const& uncertaintyKey,
                                                               edm::ConsumesCollector& cc) {
  for (auto const& name : config.getParameter<std::vector<std::string>>(regressionKey)) {
    mean.push_back(cc.esConsumes<GBRForestD, GBRDWrapperRcd>(edm::ESInputTag("", name)));
  }
  for (auto const& name : config.getParameter<std::vector<std::string>>(uncertaintyKey)) {
    sigma.push_back(cc.esConsumes<GBRForestD, GBRDWrapperRcd>(edm::ESInputTag("", name)));
  }
}

std::vector<const GBRForestD*> retrieveGBRForests(
    edm::EventSetup const& evs, std::vector<edm::ESGetToken<GBRForestD, GBRDWrapperRcd>> const& tokens) {
  std::vector<const GBRForestD*> items;

  items.reserve(tokens.size());
  for (auto const& token : tokens) {
    items.push_back(&evs.getData(token));
  }

  return items;
}
