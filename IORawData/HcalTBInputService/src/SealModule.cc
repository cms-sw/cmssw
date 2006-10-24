#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "IORawData/HcalTBInputService/interface/HcalTBSource.h"
#include "IORawData/HcalTBInputService/src/HcalTBWriter.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_INPUT_SOURCE(HcalTBSource);
DEFINE_ANOTHER_FWK_MODULE(HcalTBWriter);
