#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "Calibration/HcalAlCaRecoProducers/interface/AlCaEcalHcalReadoutsProducer.h"
#include "Calibration/HcalAlCaRecoProducers/interface/AlCaIsoTracksProducer.h"
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(AlCaIsoTracksProducer)
DEFINE_ANOTHER_FWK_MODULE(AlCaEcalHcalReadoutsProducer)
