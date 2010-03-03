#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

DEFINE_SEAL_MODULE();

#include "Geometry/CommonTopologies/test/ValidateRadial.h"
DEFINE_ANOTHER_FWK_MODULE(ValidateRadial);

