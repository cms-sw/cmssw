#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "RecoLuminosity/LumiProducer/interface/DBService.h"
using lumi::service::DBService;
DEFINE_FWK_SERVICE(DBService);
