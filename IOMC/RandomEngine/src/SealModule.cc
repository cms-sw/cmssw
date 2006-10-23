
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

#include "IOMC/RandomEngine/src/RandomNumberGeneratorService.h"
#include "IOMC/RandomEngine/src/RandomEngineStateProducer.h"

using edm::service::RandomNumberGeneratorService;

DEFINE_SEAL_MODULE();

typedef edm::serviceregistry::AllArgsMaker<edm::RandomNumberGenerator,RandomNumberGeneratorService> RandomMaker;
DEFINE_ANOTHER_FWK_SERVICE_MAKER(RandomNumberGeneratorService, RandomMaker);

DEFINE_ANOTHER_FWK_MODULE(RandomEngineStateProducer);
