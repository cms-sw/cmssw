
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

#include "RandomNumberGeneratorService.h"
#include "RandomEngineStateProducer.h"
#include "RandomFilter.h"

using edm::RandomFilter;
using edm::service::RandomNumberGeneratorService;

typedef edm::serviceregistry::AllArgsMaker<edm::RandomNumberGenerator, RandomNumberGeneratorService> RandomMaker;
DEFINE_FWK_SERVICE_MAKER(RandomNumberGeneratorService, RandomMaker);

DEFINE_FWK_MODULE(RandomEngineStateProducer);
DEFINE_FWK_MODULE(RandomFilter);
