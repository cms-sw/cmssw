#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/RecoAlgos/plugins/CaloRecHitCandidateProducer.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

typedef reco::modules::CaloRecHitCandidateProducer<HORecHitCollection> HORecHitCandidateProducer;

DEFINE_FWK_MODULE( HORecHitCandidateProducer );
