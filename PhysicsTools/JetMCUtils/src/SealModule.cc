#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/JetMCUtils/interface/GenJetRecoJetMatcher.h"
#include "PhysicsTools/JetMCUtils/src/jetTest.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE( GenJetRecoJetMatcher );
DEFINE_ANOTHER_FWK_MODULE( jetTest );
