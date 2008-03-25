#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/RecoAlgos/plugins/CaloRecHitCandidateProducer.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

typedef reco::modules::CaloRecHitCandidateProducer<ZDCRecHitCollection> ZDCRecHitCandidateProducer;

DEFINE_FWK_MODULE( ZDCRecHitCandidateProducer );
