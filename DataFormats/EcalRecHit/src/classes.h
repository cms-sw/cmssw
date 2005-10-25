#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "FWCore/EDProduct/interface/Wrapper.h"
#include <vector>

namespace {
  namespace {
    EcalUncalibratedRecHit _ahit;
    std::vector<EcalUncalibratedRecHit> _hitVect;
    edm::Wrapper< std::vector<EcalUncalibratedRecHit> > _hitProd;
  }
}
