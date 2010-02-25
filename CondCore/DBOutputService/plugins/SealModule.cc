#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
using cond::service::PoolDBOutputService;

DEFINE_FWK_SERVICE(PoolDBOutputService);
