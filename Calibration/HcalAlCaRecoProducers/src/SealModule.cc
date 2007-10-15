#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "Calibration/HcalAlCaRecoProducers/interface/AlCaIsoTracksProducer.h"
#include "Calibration/HcalAlCaRecoProducers/interface/AlCaEcalHcalReadoutsProducer.h"
#include "Calibration/HcalAlCaRecoProducers/interface/AlCaDiJetsProducer.h"
#include "Calibration/HcalAlCaRecoProducers/interface/AlCaGammaJetProducer.h"
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(AlCaIsoTracksProducer);
DEFINE_ANOTHER_FWK_MODULE(AlCaEcalHcalReadoutsProducer);
DEFINE_ANOTHER_FWK_MODULE(AlCaDiJetsProducer);
DEFINE_ANOTHER_FWK_MODULE(AlCaGammaJetProducer);
