#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/HcalRecHit/interface/HFRecHit.h"
#include "DataFormats/HcalRecHit/interface/HORecHit.h"
#include "DataFormats/HcalRecHit/interface/HcalTriggerPrimitiveRecHit.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalSourcePositionData.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace {
  namespace {
    std::vector<HBHERecHit> vHBHE_;
    std::vector<HORecHit> vHO_;
    std::vector<HFRecHit> vHF_;
    std::vector<HcalTriggerPrimitiveRecHit> vHTP_;

    HBHERecHitCollection theHBHE_;
    HORecHitCollection theHO_;
    HFRecHitCollection theHF_;
    HcalTrigPrimRecHitCollection theHTP_;
    HcalSourcePositionData theSPD_;

    edm::Wrapper<HBHERecHitCollection> theHBHEw_;
    edm::Wrapper<HORecHitCollection> theHOw_;
    edm::Wrapper<HFRecHitCollection> theHFw_;
    edm::Wrapper<HcalTrigPrimRecHitCollection> theHTPw_;
    edm::Wrapper<HcalSourcePositionData> theSPDw_;
 }
}

