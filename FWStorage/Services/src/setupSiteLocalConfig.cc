///////////////////////////////////////
//
// data catalogs are filled in "parse"
//
///////////////////////////////////////

//<<<<<< INCLUDES                                                       >>>>>>

#include "FWStorage/Services/src/SiteLocalConfigService.h"
#include "FWStorage/Services/interface/setupSiteLocalConfig.h"

#include <memory>

namespace edm {
  ServiceRegistry::LockedOperate setupSiteLocalConfig() {
    std::unique_ptr<edm::SiteLocalConfig> slcptr =
        std::make_unique<edm::service::SiteLocalConfigService>(edm::ParameterSet());
    auto slc = std::make_shared<edm::serviceregistry::ServiceWrapper<edm::SiteLocalConfig> >(std::move(slcptr));
    return ServiceRegistry::LockedOperate(edm::ServiceRegistry::createContaining(slc));
  }
}  // namespace edm
