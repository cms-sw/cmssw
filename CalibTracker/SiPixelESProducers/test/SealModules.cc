#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CalibTracker/SiPixelESProducers/test/SiPixelFakeGainReader.h"
#include "CalibTracker/SiPixelESProducers/test/SiPixelFakeGainForHLTReader.h"
#include "CalibTracker/SiPixelESProducers/test/SiPixelFakeGainOfflineReader.h"

using cms::SiPixelFakeGainReader;
using cms::SiPixelFakeGainForHLTReader;
using cms::SiPixelFakeGainOfflineReader;


DEFINE_FWK_MODULE(SiPixelFakeGainReader);
DEFINE_FWK_MODULE(SiPixelFakeGainForHLTReader);
DEFINE_FWK_MODULE(SiPixelFakeGainOfflineReader);
