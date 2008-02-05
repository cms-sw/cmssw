#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

#include "DQMServices/Core/src/MonitorDaemon.h"
#include "DQMServices/Core/src/DQMShipMonitoring.h"

typedef edm::serviceregistry::ParameterSetMaker<MonitorDaemon> maker_md;
typedef edm::serviceregistry::AllArgsMaker<DQMShipMonitoring> maker_sm;
DEFINE_ANOTHER_FWK_SERVICE_MAKER(MonitorDaemon,maker_md);
DEFINE_ANOTHER_FWK_SERVICE_MAKER(DQMShipMonitoring,maker_sm);

