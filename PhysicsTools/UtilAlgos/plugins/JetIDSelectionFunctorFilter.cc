#include "PhysicsTools/UtilAlgos/interface/EDFilterObjectWrapper.h"
#include "PhysicsTools/SelectorUtils/interface/JetIDSelectionFunctor.h"

typedef edm::FilterObjectWrapper<JetIDSelectionFunctor, std::vector<pat::Jet> > JetIDSelectionFunctorFilter;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(JetIDSelectionFunctorFilter);

