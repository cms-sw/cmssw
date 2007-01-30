#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
#include "PhysicsTools/UtilAlgos/src/StopAfterNEvents.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE( StopAfterNEvents );
DEFINE_ANOTHER_FWK_SERVICE( TFileService );
