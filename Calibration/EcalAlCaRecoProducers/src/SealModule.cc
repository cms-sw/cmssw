#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "Calibration/EcalAlCaRecoProducers/interface/AlCaElectronsProducer.h"
#include "Calibration/EcalAlCaRecoProducers/interface/AlCaPhiSymRecHitsProducer.h"
#include "Calibration/EcalAlCaRecoProducers/interface/AlCaPi0BasicClusterRecHitsProducer.h"

DEFINE_SEAL_MODULE();

DEFINE_ANOTHER_FWK_MODULE(AlCaElectronsProducer)
DEFINE_ANOTHER_FWK_MODULE(AlCaPhiSymRecHitsProducer)
DEFINE_ANOTHER_FWK_MODULE(AlCaPi0BasicClusterRecHitsProducer)
