#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/Common/interface/RefProd.h" 
#include "DataFormats/Common/interface/Wrapper.h"
#include <vector>

namespace {
  namespace {
    EcalUncalibratedRecHit _ahit;
    std::vector<EcalUncalibratedRecHit> _hitVect;
    edm::SortedCollection<EcalUncalibratedRecHit> _theEURsc;
    EcalUncalibratedRecHitCollection _theEURHitCollection;
    EBUncalibratedRecHitCollection _theEBURHitCollection;
    EEUncalibratedRecHitCollection _theEEURHitCollection;
    edm::Wrapper< EcalUncalibratedRecHitCollection > _EURHitProd;
    edm::Wrapper< EBUncalibratedRecHitCollection > _EBURHitProd;
    edm::Wrapper< EEUncalibratedRecHitCollection > _EEURHitProd;

    EcalUncalibratedRecHitRef _EURHitRef;
    EcalUncalibratedRecHitRefs _EURHitRefs;
    EcalUncalibratedRecHitsRef _EURHitsRef;

    EcalRecHit _aRecHit;
    std::vector<EcalRecHit> _ERHitVect;
    edm::SortedCollection<EcalRecHit> _theERsc;
    EcalRecHitCollection _theERHitCollection;
    EBRecHitCollection _theEBRHitCollection;
    EERecHitCollection _theEERHitCollection;
    edm::Wrapper< EcalRecHitCollection > _ERHitProd;
    edm::Wrapper< EBRecHitCollection > _EBRHitProd;
    edm::Wrapper< EERecHitCollection > _EERHitProd;
    EcalRecHitRef _ERHitRef;
    EcalRecHitRefs _ERHitRefs;
    EcalRecHitsRef _ERHitsRef;
  }
}
