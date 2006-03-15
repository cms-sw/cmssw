#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include <vector>

namespace {
  namespace {
    EcalUncalibratedRecHit _ahit;
    std::vector<EcalUncalibratedRecHit> _hitVect;
    edm::SortedCollection<EcalUncalibratedRecHit> _theEURsc;
    EcalUncalibratedRecHitCollection _theEURHitCollection;
    edm::Wrapper< EcalUncalibratedRecHitCollection > _EURHitProd;
    EcalUncalibratedRecHitRef _EURHitRef;
    EcalUncalibratedRecHitRefs _EURHitRefs;
    EcalUncalibratedRecHitsRef _EURHitsRef;

    EcalRecHit _aRecHit;
    std::vector<EcalRecHit> _ERHitVect;
    edm::SortedCollection<EcalRecHit> _theERsc;
    EcalRecHitCollection _theERHitCollection;
    edm::Wrapper< EcalRecHitCollection > _ERHitProd;
    EcalRecHitRef _ERHitRef;
    EcalRecHitRefs _ERHitRefs;
    EcalRecHitsRef _ERHitsRef;
  }
}
