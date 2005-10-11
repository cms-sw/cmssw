#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Modules/src/AsciiOutputModule.h"
#include "FWCore/Modules/src/EmptyESSource.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Modules/src/EventSetupRecordDataGetter.h"
#include "FWCore/Modules/src/EventContentAnalyzer.h"

using edm::AsciiOutputModule;
using edm::EventSetupRecordDataGetter;
using edm::EmptyESSource;
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(AsciiOutputModule)
DEFINE_ANOTHER_FWK_MODULE(EventSetupRecordDataGetter)
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(EmptyESSource)
DEFINE_ANOTHER_FWK_MODULE(EventContentAnalyzer)
