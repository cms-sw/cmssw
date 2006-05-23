#include "PluginManager/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include <DQM/DTMonitorModule/interface/DTDigiTask.h>
DEFINE_FWK_MODULE(DTDigiTask)

#include <DQM/DTMonitorModule/interface/DTDataIntegrityTask.h>
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

using namespace edm::serviceregistry;

typedef ParameterSetMaker<DTDataMonitorInterface,DTDataIntegrityTask> maker;

DEFINE_ANOTHER_FWK_SERVICE_MAKER(DTDataIntegrityTask,maker)
