#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "Calibration/EcalAlCaRecoProducers/interface/AlCaElectronsProducer.h"
#include "Calibration/EcalAlCaRecoProducers/interface/AlCaPhiSymRecHitsProducer.h"

DEFINE_SEAL_MODULE();

DEFINE_ANOTHER_FWK_MODULE(AlCaElectronsProducer);
DEFINE_ANOTHER_FWK_MODULE(AlCaPhiSymRecHitsProducer);
