#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "Geometry/EcalMapping/interface/EcalElectronicsMappingBuilder.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(EcalElectronicsMappingBuilder);
// DEFINE_ANOTHER_FWK_MODULE(L1ScalesTester);
