#include "FWCore/Services/src/SiteLocalConfigService.h"
#include "FWCore/Services/src/Tracer.h"
#include "FWCore/Services/src/InitRootHandlers.h"
#include "FWCore/Services/src/UnixSignalService.h"

#include "FWCore/Services/src/JobReportService.h"
#include "FWCore/Services/interface/Timing.h"
#include "FWCore/Services/src/Memory.h"
#include "FWCore/Services/src/Profiling.h"
#include "FWCore/Services/src/LoadAllDictionaries.h"
#include "FWCore/Services/src/EnableFloatingPointExceptions.h"
#include "FWCore/Services/src/LockService.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

using edm::service::JobReportService;
using edm::service::Tracer;
using edm::service::Timing;
using edm::service::SimpleMemoryCheck;
using edm::service::SimpleProfiling;
using edm::service::LoadAllDictionaries;
using edm::service::SiteLocalConfigService;
using edm::service::EnableFloatingPointExceptions;
using edm::service::InitRootHandlers;
using edm::service::UnixSignalService;
using edm::rootfix::LockService;

DEFINE_FWK_SERVICE(Tracer);
DEFINE_ANOTHER_FWK_SERVICE(Timing);
typedef edm::serviceregistry::AllArgsMaker<edm::SiteLocalConfig,SiteLocalConfigService> SiteLocalConfigMaker;
DEFINE_ANOTHER_FWK_SERVICE_MAKER(SiteLocalConfigService,SiteLocalConfigMaker);
#if defined(__linux__)
DEFINE_ANOTHER_FWK_SERVICE(SimpleMemoryCheck);
DEFINE_ANOTHER_FWK_SERVICE(SimpleProfiling);
DEFINE_ANOTHER_FWK_SERVICE(InitRootHandlers);
DEFINE_ANOTHER_FWK_SERVICE(UnixSignalService);
DEFINE_ANOTHER_FWK_SERVICE_MAKER(EnableFloatingPointExceptions,edm::serviceregistry::AllArgsMaker<EnableFloatingPointExceptions>);
#endif
DEFINE_ANOTHER_FWK_SERVICE_MAKER(LoadAllDictionaries,edm::serviceregistry::ParameterSetMaker<LoadAllDictionaries>);
typedef edm::serviceregistry::AllArgsMaker<edm::JobReport,JobReportService> JobReportMaker;
DEFINE_ANOTHER_FWK_SERVICE_MAKER(JobReportService, JobReportMaker);
typedef edm::serviceregistry::AllArgsMaker<LockService> LockServiceMaker;
DEFINE_ANOTHER_FWK_SERVICE_MAKER(LockService, LockServiceMaker);
