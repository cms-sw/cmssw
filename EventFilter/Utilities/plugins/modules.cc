#include "EventFilter/Utilities/interface/EvFDaqDirector.h"
#include "EventFilter/Utilities/interface/FastMonitoringService.h"
#include "EventFilter/Utilities/interface/FedRawDataInputSource.h"
#include "EventFilter/Utilities/interface/DAQSource.h"
#include "EventFilter/Utilities/plugins/DaqFakeReader.h"
#include "EventFilter/Utilities/plugins/DTHFakeReader.h"
#include "EventFilter/Utilities/plugins/EvFBuildingThrottle.h"
#include "EventFilter/Utilities/plugins/EvFFEDSelector.h"
#include "EventFilter/Utilities/plugins/EvFFEDExcluder.h"
#include "EventFilter/Utilities/plugins/ExceptionGenerator.h"
#include "EventFilter/Utilities/plugins/RawEventFileWriterForBU.h"
#include "EventFilter/Utilities/plugins/RawEventOutputModuleForBU.h"
#include "EventFilter/Utilities/plugins/FRDOutputModule.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

using namespace edm::serviceregistry;
using namespace evf;

typedef RawEventOutputModuleForBU<RawEventFileWriterForBU> RawStreamFileWriterForBU;
DEFINE_FWK_MODULE(RawStreamFileWriterForBU);
DEFINE_FWK_MODULE(FRDOutputModule);
DEFINE_FWK_SERVICE(EvFBuildingThrottle);
DEFINE_FWK_SERVICE(EvFDaqDirector);
DEFINE_FWK_SERVICE(FastMonitoringService);
DEFINE_FWK_MODULE(ExceptionGenerator);
DEFINE_FWK_MODULE(EvFFEDSelector);
DEFINE_FWK_MODULE(EvFFEDExcluder);
DEFINE_FWK_MODULE(DaqFakeReader);
DEFINE_FWK_MODULE(DTHFakeReader);
DEFINE_FWK_INPUT_SOURCE(FedRawDataInputSource);
DEFINE_FWK_INPUT_SOURCE(DAQSource);
