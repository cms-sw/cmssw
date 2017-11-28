#include "FWCore/Services/src/SiteLocalConfigService.h"
#include "FWCore/Services/src/JobReportService.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

using edm::service::JobReportService;
using edm::service::SiteLocalConfigService;

typedef edm::serviceregistry::ParameterSetMaker<edm::SiteLocalConfig, SiteLocalConfigService> SiteLocalConfigMaker;
DEFINE_FWK_SERVICE_MAKER(SiteLocalConfigService, SiteLocalConfigMaker);
typedef edm::serviceregistry::AllArgsMaker<edm::JobReport, JobReportService> JobReportMaker;
DEFINE_FWK_SERVICE_MAKER(JobReportService, JobReportMaker);
