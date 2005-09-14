
#include "PluginManager/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripClusterizer.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/SiStrip1DLocalMeasurementConverter.h"

using cms::SiStripClusterizer;
using cms::SiStrip1DLocalMeasurementConverter;
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(SiStripClusterizer)
DEFINE_ANOTHER_FWK_MODULE(SiStrip1DLocalMeasurementConverter)

