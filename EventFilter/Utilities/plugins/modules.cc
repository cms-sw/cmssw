#include "EventFilter/Utilities/interface/EvFDaqDirector.h"
#include "EventFilter/Utilities/interface/EvFOutputModule.h"
#include "EventFilter/Utilities/interface/FastMonitoringService.h"
#include "EventFilter/Utilities/interface/FedRawDataInputSource.h"
#include "EventFilter/Utilities/plugins/DaqFakeReader.h"
#include "EventFilter/Utilities/plugins/EvFBuildingThrottle.h"
#include "EventFilter/Utilities/plugins/EvFFEDSelector.h"
#include "EventFilter/Utilities/plugins/ExceptionGenerator.h"
#include "EventFilter/Utilities/plugins/RawEventFileWriterForBU.h"
#include "EventFilter/Utilities/plugins/RawEventOutputModuleForBU.h"
#include "EventFilter/Utilities/plugins/RecoEventOutputModuleForFU.h"
#include "EventFilter/Utilities/plugins/RecoEventWriterForFU.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

using namespace edm::serviceregistry;
using namespace evf;

typedef edm::serviceregistry::AllArgsMaker<MicroStateService, FastMonitoringService> FastMonitoringServiceMaker;
DEFINE_FWK_SERVICE_MAKER(FastMonitoringService, FastMonitoringServiceMaker);

typedef RawEventOutputModuleForBU<RawEventFileWriterForBU> RawStreamFileWriterForBU;
DEFINE_FWK_MODULE(RawStreamFileWriterForBU);

// legacy name for ConfDB compatibility
typedef RecoEventOutputModuleForFU<RecoEventWriterForFU> ShmStreamConsumer;
DEFINE_FWK_MODULE(ShmStreamConsumer);

DEFINE_FWK_SERVICE(EvFBuildingThrottle);
DEFINE_FWK_SERVICE(EvFDaqDirector);
DEFINE_FWK_MODULE(ExceptionGenerator);
DEFINE_FWK_MODULE(EvFFEDSelector);
DEFINE_FWK_MODULE(EvFOutputModule);
DEFINE_FWK_MODULE(DaqFakeReader);
DEFINE_FWK_INPUT_SOURCE(FedRawDataInputSource);
