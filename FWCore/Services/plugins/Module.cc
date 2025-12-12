#include "FWCore/Services/src/JobReportService.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

using edm::service::JobReportService;

typedef edm::serviceregistry::AllArgsMaker<edm::JobReport, JobReportService> JobReportMaker;
DEFINE_FWK_SERVICE_MAKER(JobReportService, JobReportMaker);
