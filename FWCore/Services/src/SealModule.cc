#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/InputServiceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Services/src/EmptyInputService.h"
#include "FWCore/Services/src/AsciiOutputModule.h"
#include "FWCore/Services/src/EmptyESSource.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Services/src/EventSetupRecordDataGetter.h"

using edm::EmptyInputService;
using edm::AsciiOutputModule;
using edm::EventSetupRecordDataGetter;
using edm::EmptyESSource;
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_INPUT_SERVICE(EmptyInputService)
DEFINE_ANOTHER_FWK_MODULE(AsciiOutputModule)
DEFINE_ANOTHER_FWK_MODULE(EventSetupRecordDataGetter)
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(EmptyESSource)
