
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

#include "IOMC/RandomEngine/src/RandomNumberGeneratorService.h"
#include "IOMC/RandomEngine/src/RandomEngineStateProducer.h"
#include "IOMC/RandomEngine/src/RandomFilter.h"

using edm::service::RandomNumberGeneratorService;
using edm::RandomFilter;

typedef edm::serviceregistry::AllArgsMaker<edm::RandomNumberGenerator,RandomNumberGeneratorService> RandomMaker;
DEFINE_FWK_SERVICE_MAKER(RandomNumberGeneratorService, RandomMaker);

DEFINE_FWK_MODULE(RandomEngineStateProducer);
DEFINE_FWK_MODULE(RandomFilter);
