#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "AnalysisExamples/SiStripDetectorPerformance/test/SiStripInfoAnalysis/ClusterInfoAnalyzerExample.h"


DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(ClusterInfoAnalyzerExample);
