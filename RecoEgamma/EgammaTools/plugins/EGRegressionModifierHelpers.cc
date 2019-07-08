#include "RecoEgamma/EgammaTools/plugins/EGRegressionModifierHelpers.h"

std::vector<const GBRForestD*> retrieveGBRForests(edm::EventSetup const& evs, std::vector<std::string> const& names) {
  std::vector<const GBRForestD*> items;
  edm::ESHandle<GBRForestD> handle;

  for (auto const& name : names) {
    evs.get<GBRDWrapperRcd>().get(name, handle);
    items.push_back(handle.product());
  }

  return items;
}
