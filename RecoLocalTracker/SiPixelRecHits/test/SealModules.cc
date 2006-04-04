
#include "PluginManager/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoLocalTracker/SiPixelRecHits/test/ReadPixelRecHit.h"
#include "RecoLocalTracker/SiPixelRecHits/test/PixelNtuplizer.h"


DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(ReadPixelRecHit)
DEFINE_ANOTHER_FWK_MODULE(PixelNtuplizer)

