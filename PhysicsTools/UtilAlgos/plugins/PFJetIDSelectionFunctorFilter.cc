#include "PhysicsTools/UtilAlgos/interface/EDFilterObjectWrapper.h"
#include "PhysicsTools/SelectorUtils/interface/PFJetIDSelectionFunctor.h"

typedef edm::FilterObjectWrapper<PFJetIDSelectionFunctor, std::vector<pat::Jet> > PFJetIDSelectionFunctorFilter;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFJetIDSelectionFunctorFilter);
