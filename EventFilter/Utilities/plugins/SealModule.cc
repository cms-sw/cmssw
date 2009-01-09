#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "EventFilter/Utilities/interface/MicroStateService.h"
#include "EventFilter/Utilities/interface/ModuleWebRegistry.h"
#include "EventFilter/Utilities/interface/ServiceWebRegistry.h"
#include "EventFilter/Utilities/interface/TimeProfilerService.h"
#include "EventFilter/Utilities/interface/Stepper.h"
#include "EventFilter/Utilities/plugins/ExceptionGenerator.h"

using namespace edm::serviceregistry;
using namespace evf;

typedef edm::serviceregistry::AllArgsMaker<MicroStateService> MicroStateServiceMaker;
typedef ParameterSetMaker<ModuleWebRegistry> maker1;
typedef ParameterSetMaker<ServiceWebRegistry> maker2;

DEFINE_ANOTHER_FWK_SERVICE_MAKER(MicroStateService, MicroStateServiceMaker);
DEFINE_ANOTHER_FWK_SERVICE_MAKER(ModuleWebRegistry,maker1);
DEFINE_ANOTHER_FWK_SERVICE_MAKER(ServiceWebRegistry,maker2);
DEFINE_FWK_SERVICE(TimeProfilerService);
DEFINE_ANOTHER_FWK_SERVICE(Stepper);
DEFINE_FWK_MODULE(ExceptionGenerator);
