#include "PhysicsTools/NanoAOD/interface/SimpleFlatTableProducer.h"

#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
typedef SimpleFlatTableProducer<HBHERecHit> HBHERecHitFlatTableProducer;

#include "DataFormats/HcalRecHit/interface/HFRecHit.h"
typedef SimpleFlatTableProducer<HFRecHit> HFRecHitFlatTableProducer;

#include "DataFormats/HcalRecHit/interface/HORecHit.h"
typedef SimpleFlatTableProducer<HORecHit> HORecHitFlatTableProducer;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HBHERecHitFlatTableProducer);
DEFINE_FWK_MODULE(HFRecHitFlatTableProducer);
DEFINE_FWK_MODULE(HORecHitFlatTableProducer);
