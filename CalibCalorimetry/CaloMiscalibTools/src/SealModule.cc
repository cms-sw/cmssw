#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/EcalRecHitMiscalib.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/CaloMiscalibTools.h"


DEFINE_SEAL_MODULE();

DEFINE_ANOTHER_FWK_MODULE(EcalRecHitMiscalib);
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(CaloMiscalibTools);


