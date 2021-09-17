#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondCore/DBOutputService/interface/OnlineDBOutputService.h"
using cond::service::OnlineDBOutputService;
using cond::service::PoolDBOutputService;

DEFINE_FWK_SERVICE(PoolDBOutputService);
DEFINE_FWK_SERVICE(OnlineDBOutputService);
