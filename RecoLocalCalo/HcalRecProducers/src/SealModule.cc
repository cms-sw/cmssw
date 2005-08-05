#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoLocalCalo/HcalRecProducers/interface/HcalSimpleReconstructor.h"

using namespace cms::hcal;

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(HcalSimpleReconstructor)
