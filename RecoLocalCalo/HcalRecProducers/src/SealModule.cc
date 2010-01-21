#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "HcalSimpleReconstructor.h"
#include "ZdcSimpleReconstructor.h"
#include "HcalHitReconstructor.h"
#include "ZdcHitReconstructor.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"


DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(HcalSimpleReconstructor);
DEFINE_ANOTHER_FWK_MODULE(HcalHitReconstructor);
DEFINE_ANOTHER_FWK_MODULE(ZdcSimpleReconstructor);
DEFINE_ANOTHER_FWK_MODULE(ZdcHitReconstructor);
