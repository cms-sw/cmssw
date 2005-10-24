#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/CandAlgos/src/TwoBodyProducer.h"
#include "PhysicsTools/CandAlgos/src/PSetSelectors.h"
#include "PhysicsTools/CandAlgos/src/CandSelector.h"
#include "PhysicsTools/CandAlgos/src/CandCombiner.h"

DEFINE_SEAL_MODULE();

DEFINE_ANOTHER_FWK_MODULE( CandSelector );
DEFINE_ANOTHER_FWK_MODULE( CandCombiner );
DEFINE_ANOTHER_FWK_MODULE( PtMinCandSelector );
DEFINE_ANOTHER_FWK_MODULE( MassWindowCandSelector );
DEFINE_ANOTHER_FWK_MODULE( TwoBodyProducer );
