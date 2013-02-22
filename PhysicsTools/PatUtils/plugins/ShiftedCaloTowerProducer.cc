#include "PhysicsTools/PatUtils/interface/ShiftedParticleProducerT.h"

#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/CaloTowers/interface/CaloTowerFwd.h"

typedef ShiftedParticleProducerT<CaloTower, CaloTowerCollection> ShiftedCaloTowerProducer;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(ShiftedCaloTowerProducer);

