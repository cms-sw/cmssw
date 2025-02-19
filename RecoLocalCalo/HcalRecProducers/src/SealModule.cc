#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "HcalSimpleReconstructor.h"
#include "ZdcSimpleReconstructor.h"
#include "HcalHitReconstructor.h"
#include "ZdcHitReconstructor.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"



DEFINE_FWK_MODULE(HcalSimpleReconstructor);
DEFINE_FWK_MODULE(HcalHitReconstructor);
DEFINE_FWK_MODULE(ZdcSimpleReconstructor);
DEFINE_FWK_MODULE(ZdcHitReconstructor);
