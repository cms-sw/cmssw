#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

DEFINE_FWK_SERVICE_MAKER(CUDAService, edm::serviceregistry::ParameterSetMaker<CUDAService>);
