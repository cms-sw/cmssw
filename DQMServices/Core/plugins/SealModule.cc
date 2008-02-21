#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Core/src/DQMService.h"

DEFINE_ANOTHER_FWK_SERVICE_MAKER(DQM,edm::serviceregistry::AllArgsMaker<DQMService>);
DEFINE_ANOTHER_FWK_SERVICE_MAKER(DaqMonitorBEInterface,edm::serviceregistry::ParameterSetMaker<DaqMonitorBEInterface>);
