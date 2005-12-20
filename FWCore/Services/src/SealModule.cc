#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Services/src/Tracer.h"
#include "FWCore/Services/src/Timing.h"
#include "FWCore/Services/src/LoadAllDictionaries.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

using edm::service::Tracer;
using edm::service::Timing;
using edm::service::LoadAllDictionaries;

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_SERVICE(Tracer)
DEFINE_ANOTHER_FWK_SERVICE(Timing)
DEFINE_ANOTHER_FWK_SERVICE_MAKER(LoadAllDictionaries,edm::serviceregistry::ParameterSetMaker<LoadAllDictionaries>)
