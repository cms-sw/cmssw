#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoTBCalo/EcalTBRecProducers/interface/EcalTBWeightUncalibRecHitProducer.h"
#include "RecoTBCalo/EcalTBRecProducers/interface/IsTBH4Type.h"


DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE( EcalTBWeightUncalibRecHitProducer );
DEFINE_ANOTHER_FWK_MODULE( IsTBH4Type );
