#include "PhysicsTools/NanoAOD/plugins/FilterValueMapWrapper.h"
#include "PhysicsTools/SelectorUtils/interface/PFJetIDSelectionFunctor.h"

typedef edm::FilterValueMapWrapper<PFJetIDSelectionFunctor, std::vector<pat::Jet> > PatJetIDValueMapProducer;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PatJetIDValueMapProducer);
