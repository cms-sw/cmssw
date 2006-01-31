#include "PluginManager/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

//#include "RecoMET/METProducers/interface/UncorrTowersMETProducer.h"
#include "RecoMET/METProducers/interface/TestMETProducer.h"
#include "RecoMET/METProducers/interface/CorrMETProducer.h"

//using cms::UncorrTowersMETProducer;
using cms::TestMETProducer;
using cms::CorrMETProducer;

DEFINE_SEAL_MODULE();

//DEFINE_ANOTHER_FWK_MODULE(UncorrTowersMETProducer)
DEFINE_ANOTHER_FWK_MODULE(TestMETProducer)
DEFINE_ANOTHER_FWK_MODULE(CorrMETProducer)
