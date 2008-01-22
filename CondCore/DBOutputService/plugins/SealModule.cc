#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondCore/DBOutputService/interface/PopConDBOutputService.h"
using cond::service::PoolDBOutputService;
using popcon::service::PopConDBOutputService;
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_SERVICE(PoolDBOutputService);
DEFINE_ANOTHER_FWK_SERVICE(PopConDBOutputService);
