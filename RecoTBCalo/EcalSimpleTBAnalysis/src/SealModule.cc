#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoTBCalo/EcalSimpleTBAnalysis/interface/EcalSimpleTBAnalyzer.h"
#include "RecoTBCalo/EcalSimpleTBAnalysis/interface/EcalSimple2007H4TBAnalyzer.h"

DEFINE_SEAL_MODULE();

DEFINE_ANOTHER_FWK_MODULE( EcalSimpleTBAnalyzer );
DEFINE_ANOTHER_FWK_MODULE( EcalSimple2007H4TBAnalyzer );
