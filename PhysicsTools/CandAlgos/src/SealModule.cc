#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/CandAlgos/src/TwoSameBodyProducer.h"
#include "PhysicsTools/CandAlgos/src/TwoDifferentBodyProducer.h"
#include "PhysicsTools/CandAlgos/interface/SelectorProducer.h"
#include "PhysicsTools/CandAlgos/src/PSetSelectors.h"
#include "PhysicsTools/CandAlgos/src/CandSelector.h"

DEFINE_SEAL_MODULE();

DEFINE_ANOTHER_FWK_MODULE( TwoSameBodyProducer );
DEFINE_ANOTHER_FWK_MODULE( TwoDifferentBodyProducer );
DEFINE_ANOTHER_FWK_MODULE( PtMinCandSelector );
DEFINE_ANOTHER_FWK_MODULE( MassWindowCandSelector );
DEFINE_ANOTHER_FWK_MODULE( CandSelector );
