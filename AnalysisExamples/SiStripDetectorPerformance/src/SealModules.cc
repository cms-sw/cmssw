#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "AnalysisExamples/SiStripDetectorPerformance/interface/AnalyzeTracksClusters.h"
#include "AnalysisExamples/SiStripDetectorPerformance/interface/ClusterAnalysis.h"
#include "AnalysisExamples/SiStripDetectorPerformance/interface/ClusterAnalysisFilter.h"
#include "AnalysisExamples/SiStripDetectorPerformance/interface/MTCCNtupleMaker.h"

using cms::ClusterAnalysis;
using cms::ClusterAnalysisFilter;

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(AnalyzeTracksClusters);
DEFINE_ANOTHER_FWK_MODULE(ClusterAnalysis);
DEFINE_ANOTHER_FWK_MODULE(ClusterAnalysisFilter);
DEFINE_ANOTHER_FWK_MODULE(MTCCNtupleMaker);
