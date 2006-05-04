#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
using cond::service::PoolDBOutputService;
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_SERVICE(PoolDBOutputService)

