#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoTBCalo/EcalHVScan/src/EcalHVScanAnalyzer.h"
#include "RecoTBCalo/EcalHVScan/src/EcalEventFilter.h"

DEFINE_SEAL_MODULE();

DEFINE_ANOTHER_FWK_MODULE( EcalHVScanAnalyzer );
DEFINE_ANOTHER_FWK_MODULE( EcalEventFilter );
