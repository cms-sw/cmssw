
#include "PluginManager/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripZeroSuppression.h"

using cms::SiStripZeroSuppression;
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(SiStripZeroSuppression);

