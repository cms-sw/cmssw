#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "AnalysisExamples/SiStripDetectorPerformance/interface/AnalyzeTracksClusters.h"
#include "AnalysisExamples/SiStripDetectorPerformance/interface/ClusterAnalysis.h"

using cms::ClusterAnalysis;

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(AnalyzeTracksClusters);
DEFINE_ANOTHER_FWK_MODULE(ClusterAnalysis);
