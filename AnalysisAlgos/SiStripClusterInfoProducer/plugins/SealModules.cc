
#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "AnalysisAlgos/SiStripClusterInfoProducer/plugins/SiStripClusterInfoProducer.h"
#include "AnalysisAlgos/SiStripClusterInfoProducer/plugins/SiStripFakeRawDigiModule.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(SiStripClusterInfoProducer);
DEFINE_ANOTHER_FWK_MODULE(SiStripFakeRawDigiModule);
