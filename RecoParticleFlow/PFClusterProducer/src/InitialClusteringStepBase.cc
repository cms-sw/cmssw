#include "RecoParticleFlow/PFClusterProducer/interface/InitialClusteringStepBase.h"

std::ostream& operator<<(std::ostream& o, const InitialClusteringStepBase& a) { return a << o; }

EDM_REGISTER_PLUGINFACTORY(InitialClusteringStepFactory, "InitialClusteringStepFactory");
