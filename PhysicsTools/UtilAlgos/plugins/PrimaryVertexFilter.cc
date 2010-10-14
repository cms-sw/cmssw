#include "PhysicsTools/UtilAlgos/interface/FWLiteFilterWrapper.h"
#include "PhysicsTools/SelectorUtils/interface/PVSelector.h"

typedef edm::FWLiteFilterWrapper<PVSelector> PrimaryVertexFilter;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PrimaryVertexFilter);
