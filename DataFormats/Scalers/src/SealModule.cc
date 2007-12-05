#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Scalers/interface/L1TriggerScalers.h"

DEFINE_SEAL_MODULE();

#include "DataFormats/Scalers/interface/ScalersProducer.h" 
DEFINE_ANOTHER_FWK_MODULE(ScalersProducer);
