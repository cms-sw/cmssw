#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "EventFilter/Utilities/interface/MicroStateService.h"
#include "EventFilter/Utilities/interface/ModuleWebRegistry.h"

using namespace edm::serviceregistry;
using namespace evf;

typedef edm::serviceregistry::AllArgsMaker<MicroStateService> MicroStateServiceMaker;
typedef ParameterSetMaker<ModuleWebRegistry> maker;

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_SERVICE_MAKER(MicroStateService, MicroStateServiceMaker)
DEFINE_ANOTHER_FWK_SERVICE_MAKER(ModuleWebRegistry,maker);
