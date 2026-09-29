#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "HeterogeneousCore/MPIServices/interface/MPIConsistencyChecker.h"
DEFINE_FWK_SERVICE_MAKER(MPIConsistencyChecker, edm::serviceregistry::ParameterSetMaker<MPIConsistencyChecker>);
