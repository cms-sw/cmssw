#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "EventFilter/Utilities/interface/MicroStateService.h"
#include "EventFilter/Utilities/interface/ModuleWebRegistry.h"
#include "EventFilter/Utilities/interface/ServiceWebRegistry.h"
#include "EventFilter/Utilities/interface/TimeProfilerService.h"
#include "EventFilter/Utilities/interface/Stepper.h"
#include "EventFilter/Utilities/plugins/ExceptionGenerator.h"
#include "EventFilter/Utilities/plugins/EvFRecordInserter.h"
#include "EventFilter/Utilities/plugins/EvFRecordUnpacker.h"

using namespace edm::serviceregistry;
using namespace evf;

typedef edm::serviceregistry::AllArgsMaker<MicroStateService> MicroStateServiceMaker;
typedef ParameterSetMaker<ModuleWebRegistry> maker1;
typedef ParameterSetMaker<ServiceWebRegistry> maker2;

DEFINE_FWK_SERVICE_MAKER(MicroStateService, MicroStateServiceMaker);
DEFINE_FWK_SERVICE_MAKER(ModuleWebRegistry,maker1);
DEFINE_FWK_SERVICE_MAKER(ServiceWebRegistry,maker2);
DEFINE_FWK_SERVICE(TimeProfilerService);
DEFINE_FWK_SERVICE(Stepper);
DEFINE_FWK_MODULE(ExceptionGenerator);
DEFINE_FWK_MODULE(EvFRecordInserter);
DEFINE_FWK_MODULE(EvFRecordUnpacker);
