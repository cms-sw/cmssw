#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "EventFilter/Utilities/interface/EvFDaqDirector.h"
#include "EventFilter/Utilities/interface/FastMonitoringService.h"
#include "EventFilter/Utilities/plugins/ExceptionGenerator.h"
#include "EventFilter/Utilities/plugins/EvFRecordInserter.h"
#include "EventFilter/Utilities/plugins/EvFRecordUnpacker.h"
#include "EventFilter/Utilities/plugins/EvFBuildingThrottle.h"
#include "EventFilter/Utilities/plugins/RawEventFileWriterForBU.h"
#include "EventFilter/Utilities/plugins/RecoEventWriterForFU.h"
#include "EventFilter/Utilities/plugins/RecoEventOutputModuleForFU.h"
#include "EventFilter/Utilities/plugins/RawEventOutputModuleForBU.h"
#include "EventFilter/Utilities/plugins/DaqFakeReader.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "EventFilter/Utilities/interface/FedRawDataInputSource.h"

using namespace edm::serviceregistry;
using namespace evf;

//typedef edm::serviceregistry::AllArgsMaker<MicroStateService> MicroStateServiceMaker;
typedef edm::serviceregistry::AllArgsMaker<MicroStateService, FastMonitoringService> FastMonitoringServiceMaker;

typedef RawEventOutputModuleForBU<RawEventFileWriterForBU> RawStreamFileWriterForBU;
typedef RecoEventOutputModuleForFU<RecoEventWriterForFU> EvFOutputModule;

//legacy name for ConfDB compatibility
typedef EvFOutputModule ShmStreamConsumer;

//DEFINE_FWK_SERVICE_MAKER(MicroStateService, MicroStateServiceMaker);

DEFINE_FWK_SERVICE_MAKER(FastMonitoringService, FastMonitoringServiceMaker);
DEFINE_FWK_SERVICE(EvFBuildingThrottle);
DEFINE_FWK_SERVICE(EvFDaqDirector);
DEFINE_FWK_MODULE(ExceptionGenerator);
DEFINE_FWK_MODULE(EvFRecordInserter);
DEFINE_FWK_MODULE(EvFRecordUnpacker);
DEFINE_FWK_MODULE(RawStreamFileWriterForBU);
DEFINE_FWK_MODULE(EvFOutputModule);
DEFINE_FWK_MODULE(ShmStreamConsumer);
DEFINE_FWK_MODULE(DaqFakeReader);
DEFINE_FWK_INPUT_SOURCE(FedRawDataInputSource);
