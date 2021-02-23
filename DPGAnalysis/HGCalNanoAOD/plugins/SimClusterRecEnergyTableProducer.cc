#include "PhysicsTools/NanoAOD/interface/ObjectPropertyFromIndexMapTableProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef ObjectPropertyFromIndexMapTableProducer<SimClusterCollection, float> SimClusterRecEnergyTableProducer;
DEFINE_FWK_MODULE(SimClusterRecEnergyTableProducer);
