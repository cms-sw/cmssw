#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "HcalSimpleReconstructor.h"
#include "CaloRecHitCandidateProducer.h"

#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

typedef CaloRecHitCandidateProducer<HBHERecHitCollection> HBHERecHitCandidateProducer;
typedef CaloRecHitCandidateProducer<HORecHitCollection> HORecHitCandidateProducer;
typedef CaloRecHitCandidateProducer<HFRecHitCollection> HFRecHitCandidateProducer;
typedef CaloRecHitCandidateProducer<ZDCRecHitCollection> ZDCRecHitCandidateProducer;


DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(HcalSimpleReconstructor);
DEFINE_ANOTHER_FWK_MODULE(HBHERecHitCandidateProducer);
DEFINE_ANOTHER_FWK_MODULE(HORecHitCandidateProducer);
DEFINE_ANOTHER_FWK_MODULE(HFRecHitCandidateProducer);
DEFINE_ANOTHER_FWK_MODULE(ZDCRecHitCandidateProducer);
