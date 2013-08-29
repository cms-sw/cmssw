#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
//#include "EventFilter/Utilities/interface/MicroStateService.h"
#include "EventFilter/Utilities/interface/ModuleWebRegistry.h"
#include "EventFilter/Utilities/interface/ServiceWebRegistry.h"
#include "EventFilter/Utilities/interface/TimeProfilerService.h"
#include "EventFilter/Utilities/interface/Stepper.h"
#include "EventFilter/Utilities/plugins/ExceptionGenerator.h"
#include "EventFilter/Utilities/plugins/EvFRecordInserter.h"
#include "EventFilter/Utilities/plugins/EvFRecordUnpacker.h"
#include "EventFilter/Utilities/plugins/EvFBuildingThrottle.h"
#include "EventFilter/Utilities/plugins/EvFDaqDirector.h"
#include "EventFilter/Utilities/plugins/RawEventFileWriterForBU.h"
#include "EventFilter/Utilities/plugins/MTRawEventFileWriterForBU.h"
#include "EventFilter/Utilities/plugins/RecoEventWriterForFU.h"
#include "EventFilter/Utilities/plugins/RecoEventOutputModuleForFU.h"
#include "EventFilter/Utilities/plugins/RawEventOutputModuleForBU.h"
#include "FastMonitoringService.h"

using namespace edm::serviceregistry;
using namespace evf;

//typedef edm::serviceregistry::AllArgsMaker<MicroStateService> MicroStateServiceMaker;
typedef edm::serviceregistry::AllArgsMaker<MicroStateService, FastMonitoringService> FastMonitoringServiceMaker;
typedef ParameterSetMaker<ModuleWebRegistry> maker1;
typedef ParameterSetMaker<ServiceWebRegistry> maker2;

typedef RawEventOutputModuleForBU<RawEventFileWriterForBU> RawStreamFileWriterForBU;
typedef RawEventOutputModuleForBU<MTRawEventFileWriterForBU> MTRawStreamFileWriterForBU;
typedef RecoEventOutputModuleForFU<RecoEventWriterForFU> Stream;

//DEFINE_FWK_SERVICE_MAKER(MicroStateService, MicroStateServiceMaker);
DEFINE_FWK_SERVICE_MAKER(ModuleWebRegistry,maker1);
DEFINE_FWK_SERVICE_MAKER(ServiceWebRegistry,maker2);
DEFINE_FWK_SERVICE_MAKER(FastMonitoringService, FastMonitoringServiceMaker);
DEFINE_FWK_SERVICE(TimeProfilerService);
DEFINE_FWK_SERVICE(Stepper);
DEFINE_FWK_SERVICE(EvFBuildingThrottle);
DEFINE_FWK_SERVICE(EvFDaqDirector);
DEFINE_FWK_MODULE(ExceptionGenerator);
DEFINE_FWK_MODULE(EvFRecordInserter);
DEFINE_FWK_MODULE(EvFRecordUnpacker);
DEFINE_FWK_MODULE(RawStreamFileWriterForBU);
DEFINE_FWK_MODULE(MTRawStreamFileWriterForBU);
DEFINE_FWK_MODULE(Stream);
