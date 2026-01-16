#include "FWStorage/Services/src/SiteLocalConfigService.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

using edm::service::SiteLocalConfigService;

typedef edm::serviceregistry::ParameterSetMaker<edm::SiteLocalConfig, SiteLocalConfigService> SiteLocalConfigMaker;
DEFINE_FWK_SERVICE_MAKER(SiteLocalConfigService, SiteLocalConfigMaker);
