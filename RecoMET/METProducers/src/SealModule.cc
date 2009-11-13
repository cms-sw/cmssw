#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoMET/METProducers/interface/METProducer.h"
#include "RecoMET/METProducers/interface/BeamHaloSummaryProducer.h"
#include "RecoMET/METProducers/interface/CSCHaloDataProducer.h" 
#include "RecoMET/METProducers/interface/HcalHaloDataProducer.h" 
#include "RecoMET/METProducers/interface/EcalHaloDataProducer.h" 
#include "RecoMET/METProducers/interface/GlobalHaloDataProducer.h" 
using cms::METProducer;


using cms::METProducer;

DEFINE_SEAL_MODULE();

DEFINE_ANOTHER_FWK_MODULE(METProducer);
DEFINE_ANOTHER_FWK_MODULE(BeamHaloSummaryProducer);
DEFINE_ANOTHER_FWK_MODULE(CSCHaloDataProducer);
DEFINE_ANOTHER_FWK_MODULE(HcalHaloDataProducer);
DEFINE_ANOTHER_FWK_MODULE(EcalHaloDataProducer);
DEFINE_ANOTHER_FWK_MODULE(GlobalHaloDataProducer);
