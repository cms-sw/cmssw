#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"

DEFINE_FWK_SERVICE_MAKER(CUDAService, edm::serviceregistry::ParameterSetMaker<CUDAService>);
