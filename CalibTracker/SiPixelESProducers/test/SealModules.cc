#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CalibTracker/SiPixelESProducers/test/SiPixelFakeGainReader.h"

using cms::SiPixelFakeGainReader;

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(SiPixelFakeGainReader);
