#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/InputServiceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "IORawData/HcalTBInputService/interface/HcalTBInputService.h"

using namespace cms::hcal;

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_INPUT_SERVICE(HcalTBInputService)
