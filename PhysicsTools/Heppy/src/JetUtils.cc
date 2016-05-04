#include "PhysicsTools/Heppy/interface/JetUtils.h"

namespace heppy{

  const pat::Jet
  JetUtils::copyJet(const edm::ProductID& id, const pat::Jet& ijet, unsigned long key) {

    edm::Ptr<pat::Jet> ptrJet(id, &ijet, key);
    pat::Jet jet(ptrJet);
    
    return jet;
  }

}
