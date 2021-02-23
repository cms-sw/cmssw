#include "PhysicsTools/NanoAOD/interface/ObjectPropertyFromIndexMapTableProducer.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"
#include "SimDataFormats/CaloAnalysis/interface/SimClusterFwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef ObjectPropertyFromIndexMapTableProducer<SimClusterCollection, float> SimClusterRecEnergyTableProducer;
DEFINE_FWK_MODULE(SimClusterRecEnergyTableProducer);
