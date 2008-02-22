#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoLocalTracker/SubCollectionProducers/interface/RemainingClusterProducer.h"
#include "RecoLocalTracker/SubCollectionProducers/interface/ClusterMultiplicityFilter.h"


DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(RemainingClusterProducer);
DEFINE_ANOTHER_FWK_MODULE(ClusterMultiplicityFilter);

