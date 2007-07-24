#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/JetMCAlgos/interface/CandOneToOneDeltaRMatcher.h"
#include "PhysicsTools/JetMCAlgos/interface/CandOneToManyDeltaRMatcher.h"
#include "PhysicsTools/JetMCAlgos/src/jetTest.h"
#include "PhysicsTools/JetMCAlgos/interface/printGenEvent.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE( CandOneToOneDeltaRMatcher );
DEFINE_ANOTHER_FWK_MODULE( CandOneToManyDeltaRMatcher );
DEFINE_ANOTHER_FWK_MODULE( printGenEvent );
DEFINE_ANOTHER_FWK_MODULE( jetTest );
