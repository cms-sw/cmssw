#include "Calibration/IsolatedParticles/interface/FindCaloHit.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include <iostream>

namespace spr {

  void find(edm::Handle<EcalRecHitCollection>& hits, DetId thisDet, std::vector<EcalRecHitCollection::const_iterator>& hit, bool ) {

    if (hits->find(thisDet) != hits->end())
      hit.push_back( hits->find(thisDet) );
  }

  void find(edm::Handle<HBHERecHitCollection>& hits, DetId thisDet, std::vector<HBHERecHitCollection::const_iterator>& hit, bool ) {

    if (hits->find(thisDet) != hits->end())
      hit.push_back( hits->find(thisDet) );
  }

  void find(edm::Handle<edm::PCaloHitContainer>& hits, DetId thisDet, std::vector<edm::PCaloHitContainer::const_iterator>& hit, bool ) {

    edm::PCaloHitContainer::const_iterator ihit;
    for (ihit=hits->begin(); ihit!=hits->end(); ihit++) {
      DetId detId(ihit->id());
      if (detId == thisDet) {
        hit.push_back(ihit);
      }
    }
  }
}
