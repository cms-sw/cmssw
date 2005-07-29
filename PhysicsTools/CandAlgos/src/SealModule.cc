#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/CandAlgos/src/DSTPreOutput.h"
#include "PhysicsTools/CandAlgos/src/TrackCandidateProducer.h"
#include "PhysicsTools/CandAlgos/src/TwoSameBodyCombiner.h"
#include "PhysicsTools/CandAlgos/src/TwoDifferentBodyCombiner.h"
#include "PhysicsTools/CandAlgos/interface/SelectorProducer.h"
#include "PhysicsTools/CandUtils/interface/PtMinSelector.h"
#include "PhysicsTools/CandUtils/interface/MassWindowSelector.h"

DEFINE_SEAL_MODULE();

namespace phystools {
  typedef SelectorProducer<PtMinSelector> PtMinSelectorProducer;
  typedef SelectorProducer<MassWindowSelector> MassWindowSelectorProducer;
  
  DEFINE_ANOTHER_FWK_MODULE( DSTPreOutput );
  DEFINE_ANOTHER_FWK_MODULE( TrackCandidateProducer );
  DEFINE_ANOTHER_FWK_MODULE( TwoSameBodyCombiner );
  DEFINE_ANOTHER_FWK_MODULE( TwoDifferentBodyCombiner );
  DEFINE_ANOTHER_FWK_MODULE( PtMinSelectorProducer );
  DEFINE_ANOTHER_FWK_MODULE( MassWindowSelectorProducer );
}
