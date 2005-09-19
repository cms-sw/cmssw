#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Services/src/AsciiOutputModule.h"
#include "FWCore/Services/src/EmptyESSource.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Services/src/EventSetupRecordDataGetter.h"
#include "FWCore/Services/src/Tracer.h"
#include "FWCore/Services/src/LoadAllDictionaries.h"
#include "FWCore/Services/src/EventContentAnalyzer.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

using edm::AsciiOutputModule;
using edm::EventSetupRecordDataGetter;
using edm::EmptyESSource;
using edm::service::Tracer;
using edm::service::LoadAllDictionaries;
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(AsciiOutputModule)
DEFINE_ANOTHER_FWK_MODULE(EventSetupRecordDataGetter)
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(EmptyESSource)
DEFINE_ANOTHER_FWK_SERVICE(Tracer)
DEFINE_ANOTHER_FWK_SERVICE_MAKER(LoadAllDictionaries,edm::serviceregistry::NoArgsMaker<LoadAllDictionaries>)
DEFINE_ANOTHER_FWK_MODULE(EventContentAnalyzer)
