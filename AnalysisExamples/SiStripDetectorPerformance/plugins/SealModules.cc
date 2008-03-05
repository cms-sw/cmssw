
#include "FWCore/Framework/interface/MakerMacros.h"
#include "AnalysisExamples/SiStripDetectorPerformance/plugins/ClusterAnalysis.h"
#include "AnalysisExamples/SiStripDetectorPerformance/plugins/ClusterAnalysisFilter.h"

using cms::ClusterAnalysis;
using cms::ClusterAnalysisFilter;

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(ClusterAnalysis);
DEFINE_ANOTHER_FWK_MODULE(ClusterAnalysisFilter);
