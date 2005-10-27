#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "FWCore/EDProduct/interface/Wrapper.h"
#include <vector>

namespace {
  namespace {
    EcalUncalibratedRecHit _ahit;
    std::vector<EcalUncalibratedRecHit> _hitVect;

    EcalRecHit _aRecHit;
    std::vector<EcalRecHit> _ERHitVect;
    edm::SortedCollection<EcalRecHit> _theERsc;
    EcalRecHitCollection _theERHitCollection;

    edm::Wrapper< std::vector<EcalUncalibratedRecHit> > _hitProd;
    edm::Wrapper< EcalRecHitCollection > _ERHitProd;
    edm::Wrapper< edm::SortedCollection<EcalRecHit> > _theERHitProd;
  }
}
