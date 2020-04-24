#include "PhysicsTools/Heppy/interface/JetUtils.h"

namespace heppy{

  const pat::Jet
  JetUtils::copyJet(const pat::Jet& ijet) {

    edm::Ptr<pat::Jet> ptrJet(edm::ProductID(), &ijet, 0,0);
    pat::Jet jet(ptrJet);
    
    return jet;
  }

}
