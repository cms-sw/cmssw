#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/CandAlgos/src/TwoSameBodyProducer.h"
#include "PhysicsTools/CandAlgos/src/TwoDifferentBodyProducer.h"
#include "PhysicsTools/CandAlgos/interface/SelectorProducer.h"
#include "PhysicsTools/CandUtils/interface/PtMinSelector.h"
#include "PhysicsTools/CandUtils/interface/MassWindowSelector.h"

DEFINE_SEAL_MODULE();

typedef SelectorProducer<PtMinSelector> PtMinSelectorProducer;
typedef SelectorProducer<MassWindowSelector> MassWindowSelectorProducer;

DEFINE_ANOTHER_FWK_MODULE( TwoSameBodyProducer );
DEFINE_ANOTHER_FWK_MODULE( TwoDifferentBodyProducer );
DEFINE_ANOTHER_FWK_MODULE( PtMinSelectorProducer );
DEFINE_ANOTHER_FWK_MODULE( MassWindowSelectorProducer );
