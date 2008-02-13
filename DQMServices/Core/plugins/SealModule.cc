#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Core/interface/DaqMonitorROOTBackEnd.h"
#include "DQMServices/Core/src/DQMService.h"

#define COMMA ,
DEFINE_ANOTHER_FWK_SERVICE_MAKER(DQM,edm::serviceregistry::AllArgsMaker<DQMService>);
DEFINE_ANOTHER_FWK_SERVICE_MAKER(DaqMonitorROOTBackEnd,edm::serviceregistry::ParameterSetMaker<DaqMonitorBEInterface COMMA DaqMonitorROOTBackEnd>);
