#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "RecoLuminosity/LumiProducer/interface/DBService.h"
using lumi::service::DBService;
typedef edm::serviceregistry::ParameterSetMaker<DBService> DBServiceMaker;
DEFINE_FWK_SERVICE_MAKER(DBService,DBServiceMaker);
