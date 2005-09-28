#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/src/EmptySource.h"
#include "FWCore/Framework/interface/SourceFactory.h"

using edm::EmptySource;
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_INPUT_SOURCE(EmptySource)
