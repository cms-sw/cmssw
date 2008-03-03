#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoLocalCalo/EcalRecProducers/interface/EcalWeightUncalibRecHitProducer.h"
#include "RecoLocalCalo/EcalRecProducers/interface/EcalAnalFitUncalibRecHitProducer.h"
#include "RecoLocalCalo/EcalRecProducers/interface/EcalFixedAlphaBetaFitUncalibRecHitProducer.h"
#include "RecoLocalCalo/EcalRecProducers/interface/EcalMaxSampleUncalibRecHitProducer.h"
#include "RecoLocalCalo/EcalRecProducers/interface/EcalRecHitProducer.h"
#include "RecoLocalCalo/EcalRecProducers/interface/ESRecHitProducer.h"
#include "RecoLocalCalo/EcalRecProducers/interface/EcalRecalibRecHitProducer.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE( EcalWeightUncalibRecHitProducer );
DEFINE_ANOTHER_FWK_MODULE( EcalAnalFitUncalibRecHitProducer );
DEFINE_ANOTHER_FWK_MODULE( EcalFixedAlphaBetaFitUncalibRecHitProducer );
DEFINE_ANOTHER_FWK_MODULE( EcalMaxSampleUncalibRecHitProducer );
DEFINE_ANOTHER_FWK_MODULE( EcalRecHitProducer );
DEFINE_ANOTHER_FWK_MODULE( ESRecHitProducer );
DEFINE_ANOTHER_FWK_MODULE( EcalRecalibRecHitProducer );
