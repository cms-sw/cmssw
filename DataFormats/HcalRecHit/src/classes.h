#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/HcalRecHit/interface/HFRecHit.h"
#include "DataFormats/HcalRecHit/interface/HORecHit.h"
#include "DataFormats/HcalRecHit/interface/HcalTriggerPrimitiveRecHit.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "FWCore/EDProduct/interface/Wrapper.h"

namespace {
  namespace {
    std::vector<cms::HBHERecHit> vHBHE_;
    std::vector<cms::HORecHit> vHO_;
    std::vector<cms::HFRecHit> vHF_;
    std::vector<cms::HcalTriggerPrimitiveRecHit> vHTP_;

    cms::HBHERecHitCollection theHBHE_;
    cms::HORecHitCollection theHO_;
    cms::HFRecHitCollection theHF_;
    cms::HcalTrigPrimRecHitCollection theHTP_;

    edm::Wrapper<cms::HBHERecHitCollection> theHBHEw_;
    edm::Wrapper<cms::HORecHitCollection> theHOw_;
    edm::Wrapper<cms::HFRecHitCollection> theHFw_;
    edm::Wrapper<cms::HcalTrigPrimRecHitCollection> theHTPw_; 
 }
}

