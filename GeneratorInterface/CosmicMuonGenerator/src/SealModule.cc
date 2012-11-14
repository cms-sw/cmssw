#include "FWCore/Framework/interface/MakerMacros.h"
#include "GeneratorInterface/CosmicMuonGenerator/interface/CosMuoGenProducer.h"
#include "GeneratorInterface/CosmicMuonGenerator/interface/CosMuoGenEDFilter.h"

using edm::CosMuoGenProducer;
using edm::CosMuoGenEDFilter;

DEFINE_FWK_MODULE(CosMuoGenProducer);
DEFINE_FWK_MODULE(CosMuoGenEDFilter);
