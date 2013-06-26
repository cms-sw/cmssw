#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "GeneratorInterface/CosmicMuonGenerator/interface/CosMuoGenSource.h"
#include "GeneratorInterface/CosmicMuonGenerator/interface/CosMuoGenProducer.h"

using edm::CosMuoGenSource;
using edm::CosMuoGenProducer;

DEFINE_FWK_INPUT_SOURCE(CosMuoGenSource);
DEFINE_FWK_MODULE(CosMuoGenProducer);
