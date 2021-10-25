///////////////////////////////////////
//
// data catalogs are filled in "parse"
//
///////////////////////////////////////

//<<<<<< INCLUDES                                                       >>>>>>

#include "FWCore/Services/src/SiteLocalConfigService.h"
#include "FWCore/Services/interface/setupSiteLocalConfig.h"

#include <memory>

namespace edm {
  ServiceRegistry::Operate setupSiteLocalConfig() {
    std::unique_ptr<edm::SiteLocalConfig> slcptr =
        std::make_unique<edm::service::SiteLocalConfigService>(edm::ParameterSet());
    auto slc = std::make_shared<edm::serviceregistry::ServiceWrapper<edm::SiteLocalConfig> >(std::move(slcptr));
    edm::ServiceToken slcToken = edm::ServiceRegistry::createContaining(slc);
    return edm::ServiceRegistry::Operate(slcToken);
  }
}  // namespace edm
