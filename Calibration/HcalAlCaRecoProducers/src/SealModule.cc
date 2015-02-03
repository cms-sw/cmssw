#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "Calibration/HcalAlCaRecoProducers/interface/AlCaEcalHcalReadoutsProducer.h"
#include "Calibration/HcalAlCaRecoProducers/src/ProducerAnalyzer.h"
#include "Calibration/HcalAlCaRecoProducers/interface/AlCaIsoTracksProducer.h"

using cms::ProducerAnalyzer;

DEFINE_FWK_MODULE(AlCaEcalHcalReadoutsProducer);
DEFINE_FWK_MODULE(AlCaIsoTracksProducer);
DEFINE_FWK_MODULE(ProducerAnalyzer);
