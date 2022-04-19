#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "HeterogeneousCore/MPIServices/interface/MPIService.h"
DEFINE_FWK_SERVICE_MAKER(MPIService, edm::serviceregistry::ParameterSetMaker<MPIService>);
