#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/RecoAlgos/plugins/CaloRecHitCandidateProducer.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

typedef reco::modules::CaloRecHitCandidateProducer<HFRecHitCollection> HFRecHitCandidateProducer;

DEFINE_FWK_MODULE( HFRecHitCandidateProducer );
