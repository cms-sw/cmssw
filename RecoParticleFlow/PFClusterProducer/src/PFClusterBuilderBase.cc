#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterBuilderBase.h"

std::ostream& operator<<(std::ostream& o, const PFClusterBuilderBase& a) { return a << o; }

EDM_REGISTER_PLUGINFACTORY(PFClusterBuilderFactory, "PFClusterBuilderFactory");
