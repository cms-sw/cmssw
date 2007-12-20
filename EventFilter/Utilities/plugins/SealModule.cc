#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "EventFilter/Utilities/interface/MicroStateService.h"
#include "EventFilter/Utilities/interface/ModuleWebRegistry.h"
#include "EventFilter/Utilities/interface/TimeProfilerService.h"
#include "EventFilter/Utilities/plugins/ExceptionGenerator.h"

using namespace edm::serviceregistry;
using namespace evf;

typedef edm::serviceregistry::AllArgsMaker<MicroStateService> MicroStateServiceMaker;
typedef ParameterSetMaker<ModuleWebRegistry> maker;

DEFINE_ANOTHER_FWK_SERVICE_MAKER(MicroStateService, MicroStateServiceMaker);
DEFINE_ANOTHER_FWK_SERVICE_MAKER(ModuleWebRegistry,maker);
DEFINE_FWK_SERVICE(TimeProfilerService);
DEFINE_FWK_MODULE(ExceptionGenerator);
