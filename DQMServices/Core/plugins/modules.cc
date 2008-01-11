#include "DQMServices/Core/interface/DaqMonitorROOTBackEnd.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

using namespace edm::serviceregistry;

typedef ParameterSetMaker<DaqMonitorBEInterface,DaqMonitorROOTBackEnd> maker;

DEFINE_FWK_SERVICE_MAKER(DaqMonitorROOTBackEnd,maker);
